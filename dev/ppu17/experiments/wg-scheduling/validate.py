#!/usr/bin/env python3
"""W01 actual-type, source and CUDA codegen gates. Never launch a GPU kernel."""
import argparse
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tests"), str(ROOT / "hopper")]
import check_ppu17_softmax_overlap as code
from build_ppu17_standalone import receipt
import ppu17_build as build

spec = importlib.util.spec_from_file_location("overlap_evidence", HERE.parent / "softmax-overlap/validate.py")
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)
require = code.require
PARENT = "d19d27a"
MAINLOOP = "hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_source():
    current = (ROOT / MAINLOOP).read_text()
    parent = subprocess.check_output(["git", "show", f"{PARENT}:{MAINLOOP}"], cwd=ROOT, text=True)
    start = "    static constexpr bool RescaleOBeforeGemm"
    # Everything below the policy is unchanged except the TWO identical
    # K/V-readiness predicates, now exposed to the actual-type host probe.
    require(current.count("if (requires_kv_wait(warp_group_idx))") == 2, "both K/V predicates must use policy")
    normalized = current[current.index(start):].replace(
        "if (requires_kv_wait(warp_group_idx))", "if (!UseSchedulerBarrier || warp_group_idx == 0)")
    require(normalized == parent[parent.index(start):], "math or lifetime operation changed")
    fixed = ("hopper/softmax.h", "hopper/epilogue_fwd.hpp", "hopper/flash_fwd_kernel_sm90.h",
             "hopper/flash_fwd_launch_template.h", "hopper/tile_scheduler.hpp",
             "hopper/tile_size.h", "hopper/sm90_pipeline_no_cluster.hpp")
    for path in fixed:
        original = subprocess.check_output(["git", "show", f"{PARENT}:{path}"], cwd=ROOT)
        require((ROOT / path).read_bytes() == original, f"unrelated axis changed: {path}")
    return list(fixed)


def barriers(text):
    return [op for _, op in code.sass_body(text) if re.search(r"\bBAR\.", op)]


def inspect_candidate(sass, ptx):
    native = code.inspect_sass(sass, kv_tile=128)
    intermediate = code.inspect_ptx(ptx, kv_tile=128)
    # Independent lane/participant anchor for the remaining data barriers.
    # 0: initial CTA; 8: QueryEmpty; 1: epilogue publication.
    wanted = collections.Counter({
        "BAR.SYNC.DEFER_BLOCKING 0x0 ;": 1,
        "BAR.ARV 0x8, 0x120 ;": 2,
        "BAR.SYNC.DEFER_BLOCKING 0x8, 0x120 ;": 1,
        "BAR.SYNC.DEFER_BLOCKING 0x1, 0x100 ;": 1,
        "BAR.ARV 0x1, 0x120 ;": 1,
        "BAR.SYNC.DEFER_BLOCKING 0x1, 0x120 ;": 1,
    })
    require(collections.Counter(barriers(sass)) == wanted, "candidate data-barrier inventory differs")
    return {"native_cuda": native, "ptx": intermediate, "data_barrier_sites": dict(wanted)}


def codegen_negatives(candidate, control):
    sass = (candidate / "executable.sass").read_text()
    ptx = (candidate / "shipping_fp16_d128.ptx").read_text()
    # Mutate inside the exact selected body, not an unrelated causal body.
    head, body = sass.split("StaticPersistentTileScheduler", 1)
    plants = (
        ("missing-matrix", head + "StaticPersistentTileScheduler" + body.replace("HGMMA.64x128x16", "REMOVED_MMA", 1), "matrix denominator"),
        ("missing-wait", head + "StaticPersistentTileScheduler" + body.replace("WARPGROUP.DEPBAR.LE", "REMOVED_WAIT", 1), "waits"),
        ("missing-query-barrier", head + "StaticPersistentTileScheduler" + body.replace("BAR.ARV 0x8, 0x120", "REMOVED_QUERY_BARRIER", 1), "barrier inventory"),
        ("pingpong-restored", (control / "executable.sass").read_text(), "barrier inventory"),
    )
    for name, bad, diagnostic in plants:
        try:
            inspect_candidate(bad, ptx)
        except ValueError as error:
            require(diagnostic in str(error), f"wrong rejection for {name}: {error}")
        else:
            raise ValueError(f"negative unexpectedly green: {name}")
    return [name for name, _, _ in plants]


def actual_policy(manifest, out):
    backend = Path(manifest["cutlass"])
    nvcc = Path(manifest["commands"][0][0])
    flags = [*build.compile_flags(backend), "-I" + str(ROOT / "hopper"), "-I" + str(backend / "include"),
             "-DFLASHATTN_PPU17_KV_TILE128=1"]
    results = {}
    for variant, extra in (("control", []), ("independent", ["-DFLASHATTN_PPU17_INDEPENDENT_WG=1"])):
        binary = out / f"policy-{variant}"
        command = [str(nvcc), *flags, *extra, str(HERE / "policy.cu"), "-o", str(binary)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        (out / f"policy-{variant}.log").write_text(result.stdout + result.stderr)
        require(result.returncode == 0, f"actual {variant} policy compile failed; see {out}")
        results[variant] = subprocess.check_output([str(binary)], text=True, timeout=30).strip()
    # A generated negative overlay, never a shipping include/source authority.
    bad = out / "leader-only-negative"
    bad.mkdir()
    original = (ROOT / MAINLOOP).read_text()
    correct = "return !UseSchedulerBarrier || warp_group_idx == 0;"
    require(original.count(correct) == 1, "missing policy mutation seam")
    (bad / Path(MAINLOOP).name).write_text(original.replace(correct, "return warp_group_idx == 0;"))
    command = [str(nvcc), "-I" + str(bad), *flags, "-DFLASHATTN_PPU17_INDEPENDENT_WG=1",
               "-c", str(HERE / "policy.cu"), "-o", str(bad / "must-not-build.o")]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    (out / "leader-only-negative.log").write_text(result.stdout + result.stderr)
    require(result.returncode != 0 and "WG1 must wait for K/V" in result.stderr,
            "leader-only negative did not fail the production policy assertion")
    results["leader-only-negative"] = "EXPECTED_RED: actual WG1 readiness assertion"
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True, help="immutable C07 build directory")
    parser.add_argument("--out", type=Path, required=True, help="new validation directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    dirs = (args.control, args.candidate)
    meta = [json.loads((path / "build.json").read_text()) for path in dirs]
    require([m["wg_scheduling"] for m in meta] == ["pingpong", "independent"], "wrong policy pair")
    require(all(m["kv_tile"] == 128 and m["softmax_overlap"] == "control" for m in meta), "mixed axis")
    for key in ("source_manifest_sha256", "backend_include_sha256", "compiler"):
        require(meta[0][key] == meta[1][key], f"mixed authority: {key}")
    current = receipt(Path(meta[0]["cutlass"]))
    for key in ("source_manifest_sha256", "backend_include_sha256"):
        require(current[key] == meta[0][key], f"stale build: {key}")
    for directory, m in zip(dirs, meta):
        require(sha(directory / "flash_attn_ppu17_s1024_fp16") == m["sha256"], "ELF hash differs")
    original_meta = json.loads((args.parent / "build.json").read_text())
    for key in ("backend_include_sha256", "compiler"):
        require(original_meta[key] == meta[0][key], f"parent {key} differs")
    sass = [(path / "executable.sass").read_text() for path in (args.parent, *dirs)]
    encoded = [evidence.words(text) for text in sass]
    require(len(encoded[0]) == 2 and encoded[0] == encoded[1], "C07 default encoded bodies changed")
    require(encoded[1].keys() == encoded[2].keys(), "symbol/type inventory changed")
    causal = next(key for key in encoded[0] if "DynamicPersistent" in key)
    require(encoded[0][causal] == encoded[2][causal], "causal body changed")
    native = inspect_candidate(sass[2], (args.candidate / "shipping_fp16_d128.ptx").read_text())
    control_native = code.inspect_sass(sass[1], kv_tile=128)
    allocations = [evidence.resources((path / "shipping-object.log").read_text()) for path in dirs]
    require(allocations[0] == allocations[1], "register/stack/spill resources changed")
    require(control_native["local_sites"] == native["native_cuda"]["local_sites"], "local-memory sites changed")
    # Exactly three removed pingpong sites; all seven data sites are identical.
    removed = collections.Counter(barriers(sass[1])) - collections.Counter(barriers(sass[2]))
    require(sum(removed.values()) == 3 and len(barriers(sass[1])) == 10, "unexpected removed barrier sites")
    fixed = inspect_source()
    policies = actual_policy(meta[1], args.out)
    negatives = codegen_negatives(args.candidate, args.control)
    os.environ.update(FA17_STANDALONE_EXE=meta[1]["executable"],
                      FA17_KV128_EXE=meta[0]["executable"],
                      FA17_WG_CANDIDATE=str(args.candidate), FA17_WG_CONTROL=str(args.control))
    loader, suite = unittest.TestLoader(), unittest.TestSuite()
    for path in sorted((ROOT / "tests").glob("test_ppu17*.py")):
        if path.stem == "test_ppu17_cpu_reference" and importlib.util.find_spec("torch") is None:
            for name in re.findall(r"^    def (test_\w+)\(", path.read_text(), re.M):
                test = unittest.FunctionTestCase(lambda: None, description=f"{path.stem}.{name}")
                test.setUp = lambda: (_ for _ in ()).throw(unittest.SkipTest(
                    "Torch unavailable; independent standalone C++ CPU oracle is tested separately"))
                suite.addTest(test)
        else:
            suite.addTests(loader.loadTestsFromName(path.stem))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    require(result.wasSuccessful(), "local suite failed")
    record = {
        "scope": "local CUDA12.8 SM90a control + CPU; native PPU numerics/performance NOT_RUN",
        "parent": PARENT, "source_manifest_sha256": meta[0]["source_manifest_sha256"],
        "backend_include_sha256": meta[0]["backend_include_sha256"], "compiler": meta[0]["compiler"],
        "executable_sha256": {m["wg_scheduling"]: m["sha256"] for m in meta},
        "artifact_sha256": {p.name: {name: sha(p / name) for name in
                             ("build.json", "shipping_fp16_d128.ptx", "executable.sass", "shipping-object.log")}
                            for p in dirs},
        "control_native": control_native, "candidate_codegen": native,
        "resources": allocations[1], "removed_pingpong_sites": dict(removed),
        "default_encoded_bodies_unchanged": "2/2", "causal_body_unchanged": True,
        "fixed_source": fixed, "actual_policy": policies, "codegen_negatives": negatives,
        "validation_sources_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                                      (Path(__file__), HERE / "policy.cu", ROOT / "tests/test_ppu17_wg_scheduling.py")},
        "local_tests": {"pass": result.testsRun - len(result.skipped), "skip": len(result.skipped), "fail": 0,
                        "skip_reasons": [reason for _, reason in result.skipped]},
        "device_numerics": "NOT_RUN", "performance": "NOT_RUN", "default_promotion": False,
    }
    (args.out / "results.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
