#!/usr/bin/env python3
"""Compile a registered, isolated FA2 PPU1.5 experiment; never run a kernel."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from native import inspect_native

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shipping_flags():
    # One compiler-flag authority: the current package, not a copied flag list.
    tree = ast.parse((ROOT / "setup.py").read_text())
    lists = [node.value for node in ast.walk(tree)
             if isinstance(node, ast.Assign) and isinstance(node.value, ast.List)
             and any(isinstance(t, ast.Name) and t.id == "hgcc_flags" for t in node.targets)]
    if len(lists) != 1:
        raise RuntimeError("cannot identify the shipping hgcc_flags literal exactly once")
    return ast.literal_eval(lists[0])


def command(args, log, cwd):
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)  # do not inject Torch's CUDA libs into HGCC
    with log.open("w") as handle:
        handle.write(shlex.join(map(str, args)) + "\n")
        handle.flush()
        result = subprocess.run(list(map(str, args)), cwd=cwd, env=env,
                                stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}\n"
                           + log.read_text(errors="replace")[-5000:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, default=os.environ.get("PPU_SDK"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--compile-only", action="store_true",
                        help="save/disassemble device-bearing objects; linking and device execution are NOT_RUN")
    parser.add_argument("--contract", type=Path, default=HERE / "contract.json")
    args = parser.parse_args()
    if args.sdk is None:
        parser.error("set PPU_SDK or --sdk")
    sdk = args.sdk.resolve()
    compiler, dumper = sdk / "bin/hgcc", sdk / "bin/hgobjdump"
    if not os.access(compiler, os.X_OK) or not os.access(dumper, os.X_OK):
        print(f"[FA15 BN128 compile] SKIP: HGCC/native dumper unavailable at {sdk}")
        return 77
    contract_path = args.contract.resolve()
    contract_path.relative_to(ROOT)  # contracts are versioned, not ad hoc external thresholds
    contract = json.loads(contract_path.read_text())
    dependency = ROOT / "csrc/actlize"
    actual = subprocess.check_output(["git", "-C", dependency, "rev-parse", "HEAD"], text=True).strip()
    if actual != contract["actlize"]:
        raise RuntimeError(f"actlize is {actual}, expected {contract['actlize']}; run git submodule update --init csrc/actlize")
    if subprocess.check_output(["git", "-C", dependency, "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise RuntimeError("actlize has tracked edits; refusing an unbound dependency")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"build directory must be empty: {output}")
    flags = shipping_flags() + ["-arch=ppu_15", "-lineinfo"]
    includes = [ROOT / "csrc/flash_attn/src", dependency / "include",
                sdk / "include", sdk / "targets/x86_64-linux/include"]
    sources = [ROOT / "setup.py", *HERE.glob("*.cu"), *sorted(HERE.glob("*.py")),
               contract_path, ROOT / contract.get("runner", "tools/run_ppu15_bn128_box.sh"),
               *sorted(contract_path.parent.glob("*.py")),
               *sorted(contract_path.parent.glob("*.cpp")),
               *sorted((ROOT / "csrc/flash_attn/src").glob("*.h")),
               *sorted((ROOT / "csrc/flash_attn/src").glob("*.hpp"))]
    receipt = {"schema": contract["schema"], "target": contract["target"],
               "contract_source": str(contract_path.relative_to(ROOT)),
               "source_sha": subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip(),
               "source_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "-C", ROOT, "diff", "HEAD"])).hexdigest(),
               "source_files": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
               "actlize": actual, "sdk": str(sdk), "compiler_sha256": sha256(compiler),
               "flags": flags, "device_correctness": "NOT_RUN", "device_timing": "NOT_RUN"}
    receipt["link_status"] = "NOT_RUN_COMPILE_ONLY" if args.compile_only else "PENDING"
    command([compiler, "--version"], output / "compiler.log", output)

    def build(config):
        begin = time.monotonic()
        name = config["name"]
        library = output / f"{name}.so"
        obj = output / f"{name}.o"
        compile_cmd = [compiler, *flags, *(f"-I{p}" for p in includes),
                       f"-DFA15_CONFIG={config.get('geometry_id', config['id'])}",
                       *(f"-D{definition}" for definition in config.get("defines", [])),
                       "-c", HERE / "kernel.cu", "-o", obj]
        command(compile_cmd, output / f"{name}.build.log", output)
        if not args.compile_only:
            command([compiler, "-shared", "-Xlinker", "--no-undefined", obj, "-o", library],
                    output / f"{name}.link.log", output)
        artifact = obj if args.compile_only else library
        command([dumper, "--arch=ppu1.5", "--dump-resource-usage=all", artifact],
                output / f"{name}.resources.txt", output)
        command([dumper, "--arch=ppu1.5", "--dump-isa", artifact],
                output / f"{name}.isa.txt", output)
        native = inspect_native(config, (output / f"{name}.resources.txt").read_text(),
                                (output / f"{name}.isa.txt").read_text())
        for dtype, info in native.items():
            print(f"[FA15 experiment native] {name}/{dtype} vregs={info['vector_registers']} "
                  f"stack={info['stack_bytes']} dynamic_shared={info['dynamic_shared_bytes']} "
                  f"static_sites={info['static_instruction_sites']} PASS", flush=True)
        print(f"[FA15 experiment compile] {name} built in {time.monotonic()-begin:.1f}s; {artifact}", flush=True)
        return {**config, "library": None if args.compile_only else library.name,
                "artifact": artifact.name, "sha256": sha256(artifact),
                "build_seconds": time.monotonic()-begin, "native": native}

    with ThreadPoolExecutor(max_workers=max(1, min(args.jobs, 3))) as pool:
        receipt["configs"] = list(pool.map(build, contract["configs"]))
    if contract["schema"] == "fa15-softmax-row-interleave-v1":
        sys.path.insert(0, str(contract_path.parent))
        from native_schedule import check_build
        check_build(output, contract_path)
        receipt["native_schedule"] = {"status": "PASS", "file": "row-schedule.json",
                                      "sha256": sha256(output / "row-schedule.json")}
    if not args.compile_only:
        receipt["link_status"] = "PASS"
    (output / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"[FA15 experiment compile] built {len(receipt['configs'])}/{len(contract['configs'])} arms, FP16+BF16; "
          f"link={receipt['link_status']} device=NOT_RUN; {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"[FA15 BN128 compile] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
