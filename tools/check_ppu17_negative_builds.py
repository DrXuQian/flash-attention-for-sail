#!/usr/bin/env python3
"""Constructive target/old-macro/cross-TU controls; no GPU is executed."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_ppu17_source as check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", type=Path, required=True, help="completed source check directory with host link")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
    parser.add_argument("--legacy-nvcc", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "TMPDIR": str(args.out.resolve())}
    results = []

    # Genuine CUDA compile is a positive source control, but cannot pass the
    # native-PPU provenance gate just by spelling ACOMPUTE_VERSION=10700.
    for name, extra, expected in (
        ("cuda-is-not-native", [], "native PPU1.7 build requires"),
        ("sm80-is-not-sm90a", ["-DFLASHATTN_PPU17_SOURCE_CHECK=1", "-gencode=arch=compute_80,code=sm_80"],
         "compiler did not select SM90a"),
    ):
        flags = [x for x in check.build.compile_flags() if not x.startswith("-gencode")]
        if name == "cuda-is-not-native":
            flags += ["-gencode=arch=compute_90a,code=sm_90a"]
        cmd = [args.nvcc, *flags, *extra, "-c", str(check.ROOT / "dev/ppu17/compiler_target.cu"),
               "-o", str(args.out / f"{name}.o")]
        p = subprocess.run(cmd, text=True, capture_output=True, env=env)
        (args.out / f"{name}.log").write_text(p.stdout + p.stderr)
        if p.returncode == 0 or expected not in p.stdout + p.stderr:
            raise AssertionError(f"{name}: expected target rejection missing")
        results.append({"plant": name, "verdict": "EXPECTED_RED"})

    # Same real boundary header, only the old macro changes. No fake copy.
    header = check.ROOT / "hopper/backend_runtime.h"
    for name, extra in (("boundary-positive", []), ("legacy-macro", ["-DUSE_PPU=1"])):
        cmd = [args.nvcc, *check.build.compile_flags(), *extra, "-x", "cu", "-c", str(header),
               "-o", str(args.out / f"{name}.o")]
        p = subprocess.run(cmd, text=True, capture_output=True, env=env)
        (args.out / f"{name}.log").write_text(p.stdout + p.stderr)
        if name == "boundary-positive":
            if p.returncode:
                raise AssertionError("unplanted boundary did not compile")
            results.append({"control": name, "verdict": "PASS"})
        else:
            if p.returncode == 0 or "must not select the legacy" not in p.stdout + p.stderr:
                raise AssertionError("legacy USE_PPU macro was not rejected")
            results.append({"plant": name, "verdict": "EXPECTED_RED"})

    # ld -r intentionally permits undefined symbols. Our closure checker must
    # reject the same real host dispatcher with exactly one unit withheld.
    missing = "flash_fwd_hdim256_bf16_sm90"
    objects = [args.objects / "flash_api.o", args.objects / "flash_prepare_scheduler.o",
               *(args.objects / (src.stem + ".o") for src in check.build.source_files() if src.stem != missing)]
    if any(not path.is_file() for path in objects):
        raise ValueError("run the --torch-root source check before this control")
    broken = args.out / "missing-hdim256-bf16.o"
    subprocess.run(["ld", "-r", *(str(p) for p in objects), "-o", str(broken)], check=True)
    symbols = subprocess.check_output(["nm", "-C", "--undefined-only", str(broken)], text=True)
    (args.out / "missing-definition.log").write_text(symbols)
    try:
        check.check_undefined_symbols(symbols)
    except ValueError as error:
        if "bfloat16_t, 256, 256" not in str(error):
            raise AssertionError("cross-TU control failed for the wrong definition") from error
    else:
        raise AssertionError("missing generated definition falsely passed")
    results.append({"plant": "missing-hdim256-bf16", "verdict": "EXPECTED_RED"})

    if args.legacy_nvcc:
        # This is a specifically requested old SDK control, not a rule that
        # future compilers with the same executable name must fail.
        try:
            check.build.check_compiler(args.legacy_nvcc, args.out / "legacy-sdk", source_check=True)
        except RuntimeError as error:
            if "compiler did not select SM90a" not in str(error):
                raise AssertionError("old SDK failed for a reason other than the target guard") from error
            results.append({"plant": "legacy-sdk-target-fallback", "verdict": "EXPECTED_RED"})
        else:
            raise AssertionError("old SDK now passes SM90a: re-audit its native target instead of assuming fallback")
    else:
        results.append({"control": "legacy-sdk-target-fallback", "verdict": "SKIP", "reason": "--legacy-nvcc not provided"})
    (args.out / "negative-builds.json").write_text(json.dumps(results, indent=2) + "\n")
    for row in results:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
