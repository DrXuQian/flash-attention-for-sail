#!/usr/bin/env python3
"""Compile the actual Hopper units; never call this a native PPU verdict."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hopper"))
import ppu17_build as build


def inspect_ptx(text):
    counts = {
        "entries": len(re.findall(r"\.entry\s+", text)),
        "wgmma": len(re.findall(r"\bwgmma\.mma_async\.", text)),
        "tma_load": len(re.findall(r"cp\.async\.bulk\.tensor[^;\n]*\.shared::cluster\.global", text)),
        "tma_store": len(re.findall(r"cp\.async\.bulk\.tensor[^;\n]*\.global\.shared::cta", text)),
    }
    # Every generated unit includes causal and noncausal. Checking only a
    # file-wide mnemonic would let one valid body conceal another empty one.
    entries = re.split(r"(?m)(?=^(?:(?:\.visible|\.weak)\s+)?\.entry\s)", text)[1:]
    if len(entries) != 2 or counts["entries"] != 2:
        raise ValueError(f"expected two forward kernel bodies, got {counts['entries']}")
    for body in entries:
        for opcode in ("wgmma.mma_async.", ".shared::cluster.global", ".global.shared::cta"):
            if opcode not in body:
                raise ValueError(f"empty/incomplete Hopper entry: missing {opcode}")
    if re.search(r"\bppu\.[a-z]", text):
        raise ValueError("legacy PPU instructions leaked into SM90 source target")
    return counts


def source_receipt(backend):
    # Include shared headers and build/runner code, not just the almost-empty
    # generated instantiation TU. This binds the body that actually compiled.
    paths = sorted({*ROOT.glob("hopper/*.h"), *ROOT.glob("hopper/*.hpp"),
                    *ROOT.glob("hopper/*.cpp"), *ROOT.glob("hopper/*.cu"),
                    *ROOT.glob("hopper/*.py"), *ROOT.glob("dev/ppu17/*.cu"),
                    ROOT / "tools/check_ppu17_source.py", ROOT / "tools/run_ppu17_forward.py"})
    sources = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    headers = {str(path.relative_to(backend)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted((backend / "include").rglob("*")) if path.is_file()}
    def digest(values):
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
    return {"source_manifest": sources, "source_manifest_sha256": digest(sources),
            "backend_include_sha256": digest(headers)}


def check_undefined_symbols(symbols):
    missing = [line for line in symbols.splitlines()
               if re.search(r"\b(run_mha_fwd_|prepare_varlen_num_blocks)\b", line)]
    if missing:
        raise ValueError(f"generated definition missing at internal link: {missing}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutlass", required=True)
    parser.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
    parser.add_argument("--out", required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--torch-root", type=Path,
                        help="optional installed CUDA Torch tree for host API + cross-TU closure")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    backend = build.cutlass_root(args.cutlass)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "compiler-tmp"
    tmp.mkdir(exist_ok=True)
    build.check_compiler(args.nvcc, out / "target", source_check=True)
    env = {**os.environ, "TMPDIR": str(tmp)}
    flags = [*build.compile_flags(), f"-I{ROOT / 'hopper'}", f"-I{backend / 'include'}"]
    receipt = source_receipt(backend)

    def compile_one(source):
        ptx = out / (source.stem + ".ptx")
        obj = out / (source.stem + ".o")
        commands = [
            [args.nvcc, *flags, "--ptx", str(source), "-o", str(ptx)],
            [args.nvcc, *flags, "-Xcompiler=-fPIC", "-c", str(source), "-o", str(obj)],
        ]
        log = out / (source.stem + ".log")
        with log.open("w") as handle:
            for cmd in commands:
                handle.write(json.dumps(cmd) + "\n")
                handle.flush()
                subprocess.run(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        counts = inspect_ptx(ptx.read_text())
        row = {"source": str(source.relative_to(ROOT)), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "ptx_sha256": hashlib.sha256(ptx.read_bytes()).hexdigest(), "object_sha256": hashlib.sha256(obj.read_bytes()).hexdigest(),
               **counts, "verdict": "SOURCE_COMPILE_PASS"}
        print(json.dumps(row), flush=True)
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(compile_one, build.source_files()))
    host = {"verdict": "SKIP", "reason": "--torch-root not supplied; device units alone do not prove API linkage"}
    if args.torch_root:
        host = check_host_link(args.nvcc, flags, env, args.torch_root.resolve(), out)
    version = subprocess.check_output([args.nvcc, "--version"], text=True)
    if receipt != source_receipt(backend):
        raise RuntimeError("source/backend changed during compilation; mixed-revision evidence is invalid")
    result = {"scope": "CUDA-SM90a source compatibility, NOT native PPU1.7", "compiler": version,
              "cutlass": str(backend), **receipt, "cells": rows,
              "host_api_and_internal_link": host,
              "native_ppu17": "SKIP: capable SDK unavailable; not inferred from CUDA compilation",
              "device_numerics": "NOT_RUN", "performance": "NOT_RUN"}
    (out / "source-compile.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[PPU1.7 source] PASS {len(rows)}/{len(build.source_files())}; native PPU compile/device NOT VERIFIED")


def check_host_link(nvcc, flags, env, torch_root, out):
    """Real host TU + relocatable link; no fake Torch symbols or GPU execution."""
    include = torch_root / "include"
    if not (include / "c10/cuda/impl/cuda_cmake_macros.h").is_file():
        raise ValueError(f"CUDA Torch development headers unavailable: {include}")
    cuda_include = Path(nvcc).resolve().parents[1] / "include"
    api_obj = out / "flash_api.o"
    prep_obj = out / "flash_prepare_scheduler.o"
    linked = out / "forward-linked.o"
    macros_includes = [flag for flag in flags if flag.startswith(("-D", "-I"))]
    commands = [
        ["g++", "-std=c++17", "-O1", "-fPIC", "-D_GLIBCXX_USE_CXX11_ABI=1",
         *macros_includes, f"-I{include}", f"-I{include / 'torch/csrc/api/include'}",
         f"-I{sysconfig.get_path('include')}", f"-I{cuda_include}",
         "-c", str(ROOT / "hopper/flash_api.cpp"), "-o", str(api_obj)],
        [nvcc, *flags, "-Xcompiler=-fPIC", "-c", str(ROOT / "hopper/flash_prepare_scheduler.cu"),
         "-o", str(prep_obj)],
        ["ld", "-r", str(api_obj), str(prep_obj),
         *(str(out / (source.stem + ".o")) for source in build.source_files()), "-o", str(linked)],
    ]
    log = out / "host-and-internal-link.log"
    with log.open("w") as handle:
        for cmd in commands:
            handle.write(json.dumps(cmd) + "\n")
            handle.flush()
            subprocess.run(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    symbols = subprocess.check_output(["nm", "-C", "--undefined-only", str(linked)], text=True)
    check_undefined_symbols(symbols)
    return {"verdict": "PASS", "scope": "real Torch API object + all generated definitions; relocatable link",
            "linked_sha256": hashlib.sha256(linked.read_bytes()).hexdigest(),
            "torch_headers": str(include), "python_headers": sysconfig.get_path("include"),
            "torch_load_and_runtime": "NOT_RUN"}


if __name__ == "__main__":
    main()
