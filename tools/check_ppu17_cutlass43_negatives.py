#!/usr/bin/env python3
"""Plant real CUTLASS4.3 API/barrier/include defects in isolated header copies.

Compile only. Shipping sources and the selected backend are never modified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_ppu17_source as check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutlass", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
    args = parser.parse_args()
    backend = check.build.cutlass_root(args.cutlass)
    if check.build.cutlass_version(backend) != (4, 3, 0):
        parser.error("these mutation controls require the PPU CUTLASS4.3 backend")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "compiler-tmp"
    tmp.mkdir(exist_ok=True)
    env = {**os.environ, "TMPDIR": str(tmp)}
    flags = [*check.build.compile_flags(backend), f"-I{check.ROOT / 'hopper'}", f"-I{backend / 'include'}"]
    positive = check.check_cutlass_compatibility(args.nvcc, flags, env, out)
    print("[CUTLASS4.3 control] unmodified PASS", flush=True)
    cases = (
        ("old-two-argument-selector", "epilogue_fwd.hpp",
         "sm90_get_smem_store_op_for_accumulator<StrideO, Element, EpilogueTile_MN>()",
         "sm90_get_smem_store_op_for_accumulator<StrideO, Element>()",
         "sm90_get_smem_store_op_for_accumulator"),
        ("per-thread-pipeline", "sm90_pipeline_no_cluster.hpp",
         "#if defined(FLASHATTN_PPU17) && CUTLASS_VERSION == 360",
         "#if defined(FLASHATTN_PPU17)",
         "4.3 must retain warpgroup, not per-thread arrivals"),
        ("wrong-arrival-stride", "sm90_pipeline_no_cluster.hpp",
         "return thread % NumThreadsPerWarpGroup == 0;",
         "return thread % (NumThreadsPerWarpGroup / 2) == 0;",
         "barrier expected count differs from actual consumer arrivals"),
    )
    results = []
    for name, filename, before, after, expected in cases:
        staged = out / name
        # Explicit, per-case copy: no mutations in the source worktree.
        shutil.copytree(check.ROOT / "hopper", staged,
                        ignore=shutil.ignore_patterns("__pycache__", "build", "*.egg-info"))
        target = staged / filename
        text = target.read_text()
        if text.count(before) != 1:
            raise RuntimeError(f"{name}: mutation no longer has exactly one target")
        target.write_text(text.replace(before, after, 1))
        mutant_flags = [*check.build.compile_flags(backend), f"-I{staged}", f"-I{backend / 'include'}"]
        cmd = [args.nvcc, *mutant_flags, "-c", str(check.ROOT / "dev/ppu17/cutlass_compatibility.cu"),
               "-o", str(out / f"{name}.o")]
        result = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out / f"{name}.log").write_text(json.dumps(cmd) + "\n" + result.stdout)
        if result.returncode == 0 or expected not in result.stdout:
            raise AssertionError(f"{name}: missing intended compile rejection; see log")
        results.append({"plant": name, "verdict": "EXPECTED_RED", "changed_header": filename,
                        "mutant_sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        print(f"[CUTLASS4.3 negative] {name} EXPECTED_RED/PASS", flush=True)

    # Deliberately bind the build to 3.6 while the compiler reads 4.3.
    mismatch_flags = [flag.replace("EXPECTED_CUTLASS_VERSION=430", "EXPECTED_CUTLASS_VERSION=360")
                      for flag in flags]
    cmd = [args.nvcc, *mismatch_flags, "-x", "cu", "-c",
           str(check.ROOT / "hopper/ppu17_cutlass_compat.h"), "-o", str(out / "mixed-include.o")]
    result = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / "mixed-include.log").write_text(json.dumps(cmd) + "\n" + result.stdout)
    if result.returncode == 0 or "compiler headers differ from selected backend" not in result.stdout:
        raise AssertionError("mixed-include: version mismatch was not rejected")
    results.append({"plant": "mixed-include-version", "verdict": "EXPECTED_RED"})
    print("[CUTLASS4.3 negative] mixed-include-version EXPECTED_RED/PASS", flush=True)
    (out / "negative-builds.json").write_text(json.dumps({"positive": positive, "plants": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
