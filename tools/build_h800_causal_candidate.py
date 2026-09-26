#!/usr/bin/env python3
"""Narrow experimental build, not a replacement for the six-unit PPU1.7 package."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hopper"))
import ppu17_build as build
from check_ppu17_source import source_receipt
from bench_ppu17_hopper_control import require_idle


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, choices=(64, 80), required=True)
    ap.add_argument("--scheduler", choices=("LPT", "single"), required=True)
    ap.add_argument("--backend", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    require_idle()
    backend = build.cutlass_root(args.backend)
    args.out.mkdir(parents=True, exist_ok=False)
    tmp = args.out / "tmp"
    tmp.mkdir()
    os.environ["TMPDIR"] = str(tmp)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    cuda = Path(os.environ["CUDA_HOME"])
    build.check_compiler(cuda / "bin/nvcc", args.out / "probe", source_check=True)
    flags = [*build.compile_flags(), "-DFLASHATTENTION_DISABLE_FP16",
             "-DFLASHATTENTION_DISABLE_HDIM64", "-DFLASHATTENTION_DISABLE_HDIM128",
             "-DFLASHATTN_PPU17_CAUSAL_EXPERIMENT=1", f"-DFLASHATTN_PPU17_CAUSAL_N={args.n}",
             f"-DFLASHATTN_PPU17_CAUSAL_SINGLE={int(args.scheduler == 'single')}"]
    objects = args.out / "objects"
    objects.mkdir()
    from torch.utils.cpp_extension import load
    extension = load(name="_C", sources=[str(ROOT / "hopper" / f) for f in
        ("flash_api.cpp", "flash_prepare_scheduler.cu", "instantiations/flash_fwd_hdim256_bf16_sm90.cu")],
        extra_cflags=["-O3", "-std=c++17", *(f for f in flags if f.startswith("-D"))],
        extra_cuda_cflags=[*flags, "--ptxas-options=-v"],
        extra_include_paths=[str(ROOT / "hopper"), str(backend / "include")],
        extra_ldflags=["-lcuda"], build_directory=str(objects), verbose=True)
    if extension.ppu17_backend != "cutlass36-sm90-forward-v1":
        raise RuntimeError("wrong extension identity")
    package = args.out / "python/flash_attn_3"
    package.mkdir(parents=True)
    target = package / "_C.so"
    shutil.copy2(extension.__file__, target)
    record = {"scope": "experimental BF16 D256 causal only; NOT six-unit shipping package",
              "tile": [128, args.n], "scheduler": args.scheduler,
              "flags": flags, "source": source_receipt(backend),
              "compiler": subprocess.check_output([str(cuda / "bin/nvcc"), "--version"], text=True),
              "extension_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
              "extension": str(target), "extension_dir": str(package.parent)}
    (args.out / "identity.json").write_text(json.dumps(record, indent=2)+"\n")
    print(f"[causal candidate build] PASS: {args.out / 'identity.json'}", flush=True)


if __name__ == "__main__":
    main()
