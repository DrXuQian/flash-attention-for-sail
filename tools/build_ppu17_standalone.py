#!/usr/bin/env python3
"""Build a Torch/pip-free executable using the shipping FP16/D128 Hopper TU.

Python standard library is used only to build. The ELF itself needs no Python.
This produces CUDA SM90a simulator input, NOT a native PPU binary certificate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hopper"))
import ppu17_build as build
from check_ppu17_source import inspect_ptx, source_receipt

SHIPPING = ROOT / "hopper/instantiations/flash_fwd_hdim128_fp16_sm90.cu"
APPLICATION = ROOT / "dev/ppu17/standalone_fwd.cu"


def receipt(backend):
    result = source_receipt(backend)
    # The existing source checker already binds all production headers and
    # generated units. Include our CPU oracle/build tool as well.
    for path in (APPLICATION, APPLICATION.with_name("standalone_reference.hpp"), Path(__file__)):
        result["source_manifest"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    result["source_manifest_sha256"] = hashlib.sha256(
        json.dumps(result["source_manifest"], sort_keys=True).encode()).hexdigest()
    return result


def inspect_dependencies(text):
    if any(name in text.lower() for name in ("libtorch", "libc10", "libpython", "libhggc")):
        raise ValueError("unexpected Torch/Python/legacy-PPU dependency in standalone ELF")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutlass", default=os.environ.get("CUTLASS_PPU17_ROOT"))
    parser.add_argument("--cuda-home", default=os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    parser.add_argument("--out", type=Path, required=True, help="new artifact directory; existing directories are refused")
    parser.add_argument("--inspect-codegen", action="store_true",
                        help="optional local gate: also emit/check PTX and disassemble the ELF")
    parser.add_argument("--softmax-overlap", action="store_true",
                        help="opt-in PPU1.7 S1024 softmax/PV overlap experiment; default is unchanged control")
    parser.add_argument("--kv-tile128", action="store_true",
                        help="opt-in C07 KV tile128; original softmax and pipeline, no forced overlap")
    parser.add_argument("--q-tail-mode", choices=("m64",),
                        help="separate Q-tail experiment; requires --kv-tile128, unchanged exp")
    args = parser.parse_args()
    if args.kv_tile128 and args.softmax_overlap:
        parser.error("KV tile experiment must not be composed with forced softmax overlap")
    if args.q_tail_mode and not args.kv_tile128:
        parser.error("Q-tail experiment requires --kv-tile128")
    print(f"[FA17 standalone build] validating CUTLASS root: {args.cutlass}", flush=True)
    backend = build.cutlass_root(args.cutlass)
    cuda = Path(args.cuda_home).resolve()
    nvcc = cuda / "bin/nvcc"
    if not nvcc.is_file():
        parser.error(f"CUDA compiler not found: {nvcc}")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    temp = out / "compiler-tmp"
    temp.mkdir()
    env = {**os.environ, "TMPDIR": str(temp)}
    env.pop("PPU_SDK", None)
    build.check_environment(env)
    print(f"[FA17 standalone build] checking compiler target: {nvcc}; logs: {out / 'target'}", flush=True)
    build.check_compiler(nvcc, out / "target", source_check=True)
    before = receipt(backend)
    flags = [*build.compile_flags(backend), f"-I{ROOT / 'hopper'}", f"-I{backend / 'include'}"]
    if args.softmax_overlap:
        flags += ["-DFLASHATTN_PPU17_SOFTMAX_OVERLAP=1"]
    if args.kv_tile128:
        flags += ["-DFLASHATTN_PPU17_KV_TILE128=1"]
    if args.q_tail_mode:
        flags += ["-DFLASHATTN_PPU17_Q_TAIL_MODE=1"]
    exe = out / "flash_attn_ppu17_s1024_fp16"
    generated, app = out / "shipping_fp16_d128.o", out / "standalone.o"
    ptx = out / "shipping_fp16_d128.ptx"
    commands = []

    def run(command, log):
        command = [str(item) for item in command]
        commands.append(command)
        print("[FA17 standalone build] " + shlex.join(command), flush=True)
        return build.run_logged(command, out / log, env=env)

    code = {"verdict": "SKIP", "reason": "--inspect-codegen not selected; compile/link only"}
    if args.inspect_codegen:
        run([nvcc, *flags, "--ptx", SHIPPING, "-o", ptx], "shipping-ptx.log")
        # Preserve the PTX file unchanged. Non-UTF8 comments/paths may be
        # displayed escaped; a damaged opcode still fails the live-body checks.
        try:
            code = {"verdict": "PASS", **inspect_ptx(build.diagnostic_text(ptx.read_bytes()))}
        except ValueError as error:
            raise RuntimeError(f"PTX inspection failed: {ptx}: {error}") from error
    run([nvcc, *flags, "-Xptxas=-v", "-c", SHIPPING, "-o", generated], "shipping-object.log")
    run([nvcc, *flags, "-Xcompiler=-fopenmp",
         f'-DFA17_BUILD_SOURCE_SHA256="{before["source_manifest_sha256"]}"',
         "-c", APPLICATION, "-o", app], "application.log")
    # libcuda is a link dependency of the real TMA descriptor encoder. The
    # SDK link-only stub is legal here; NEVER put its directory in runtime
    # RPATH/LD_LIBRARY_PATH. The simulator/device supplies the real driver.
    driver = cuda / "lib64/stubs"
    link = [nvcc, "--cudart=shared", "-Xcompiler=-fopenmp", generated, app,
            f"-L{driver}", "-lcuda", f"-Xlinker=-rpath,{cuda / 'lib64'}", "-o", exe]
    run(link, "link.log")
    if args.inspect_codegen:
        run([cuda / "bin/cuobjdump", "--dump-sass", exe], "executable.sass")
    dependencies = run(["readelf", "-d", exe], "dependencies.log")
    inspect_dependencies(build.diagnostic_text(dependencies))
    compiler_version = run([nvcc, "--version"], "compiler-version.log")
    if receipt(backend) != before:
        raise RuntimeError("source/backend changed during compilation; mixed build refused")
    record = {
        "scope": "CUDA-SM90a simulation input; native PPU1.7 NOT VERIFIED",
        "cutlass": str(backend), "cutlass_version": build.cutlass_version(backend),
        "compiler": build.diagnostic_text(compiler_version),
        "compiler_version_log": "compiler-version.log",
        "compiler_version_log_sha256": hashlib.sha256(compiler_version).hexdigest(),
        "commands": commands, **before, "shipping_device_code": code,
        "executable": str(exe), "sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
        "build_and_link": "PASS", "torch_dependency": "NONE",
        "device_numerics": "NOT_RUN", "performance": "NOT_RUN",
        "softmax_overlap": "row-sum-token" if args.softmax_overlap else "control",
        "kv_tile": 128 if args.kv_tile128 else 176,
        "q_tail_mode": args.q_tail_mode or "control",
    }
    (out / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"[FA17 standalone build] PASS: {exe}\nsha256={record['sha256']}\n"
          "device correctness/performance NOT_RUN; execute once through your simulator", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"[FA17 standalone build] FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
