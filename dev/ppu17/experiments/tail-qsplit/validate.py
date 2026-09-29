#!/usr/bin/env python3
"""Bind final Q64 compile + CPU coverage gates; never execute a GPU kernel."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tests")]
import check_ppu17_softmax_overlap as gate
from build_ppu17_standalone import receipt

spec = importlib.util.spec_from_file_location("old_evidence", HERE.parent / "softmax-overlap/validate.py")
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True, help="immutable C07 build directory")
    parser.add_argument("--parent-default", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--transition-negatives", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new local validation directory")
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    dirs = [a.control, a.candidate, a.parent, a.rejected]
    records = [json.loads((d / "build.json").read_text()) for d in dirs]
    control, candidate, parent, rejected = records
    gate.require([control["q_tail_mode"], candidate["q_tail_mode"], rejected["q_tail_mode"]] ==
                 ["control", "m64", "hybrid"], "wrong candidate identity")
    gate.require(all(m["softmax_overlap"] == "control" for m in records), "exp experiment mixed in")
    for field in ("source_manifest_sha256", "backend_include_sha256", "compiler"):
        gate.require(control[field] == candidate[field], "mixed final authorities: " + field)
    current = receipt(Path(candidate["cutlass"]))
    for field in ("source_manifest_sha256", "backend_include_sha256"):
        gate.require(current[field] == candidate[field], "stale build: " + field)
    unchanged = ["hopper/softmax.h", "hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp",
                 "hopper/epilogue_fwd.hpp", "hopper/flash_fwd_kernel_sm90.h",
                 "hopper/flash_fwd_launch_template.h"]
    for path in unchanged:
        gate.require(candidate["source_manifest"][path] == parent["source_manifest"][path],
                     "math/phase source changed: " + path)
    for d, m in zip(dirs, records):
        gate.require(sha(d / "flash_attn_ppu17_s1024_fp16") == m["sha256"], "ELF hash changed")
        gate.require(m["backend_include_sha256"] == parent["backend_include_sha256"], "backend drift")
    bodies = [evidence.words((d / "executable.sass").read_text()) for d in
              (a.parent_default, a.control, a.candidate)]
    gate.require(len(bodies[0]) == 2 and bodies[0] == bodies[1], "default encoded bodies changed")
    causal = next(k for k in bodies[0] if "DynamicPersistent" in k)
    gate.require(bodies[0][causal] == bodies[2][causal], "causal body changed")
    native = gate.inspect_sass((a.candidate / "executable.sass").read_text(), kv_tile=128)
    ptx = gate.inspect_ptx((a.candidate / "shipping_fp16_d128.ptx").read_text(), kv_tile=128)
    allocation = [evidence.resources((d / "shipping-object.log").read_text()) for d in dirs]
    role_resources = [{"causal" if "DynamicPersistent" in k else "noncausal": v
                       for k, v in allocation_one.items()} for allocation_one in allocation]
    new = role_resources[1]["noncausal"]
    gate.require(new["spill_store_bytes"] == 0 and new["spill_load_bytes"] == 0 and
                 new["stack_bytes"] == 0, "M64 introduced stack/spill")
    bad = role_resources[3]["noncausal"]
    budget = role_resources[2]["noncausal"]
    gate.require(bad["spill_store_bytes"] > budget["spill_store_bytes"] and
                 bad["spill_load_bytes"] > budget["spill_load_bytes"],
                 "hybrid unexpectedly passed resource gate; review, do not relabel it")

    command = next(c for c in candidate["commands"] if "--ptx" in c)
    owners = a.out / "owners"
    compile_cmd = [*command[:command.index("--ptx")], str(HERE / "ownership.cu"), "-o", str(owners)]
    run = subprocess.run(compile_cmd, capture_output=True,
                         env={**os.environ, "TMPDIR": str(a.candidate / "compiler-tmp")})
    (a.out / "owners-build.log").write_bytes(run.stdout + run.stderr)
    gate.require(run.returncode == 0, "actual ownership compile failed: " + run.stderr.decode(errors="backslashreplace"))
    negatives = json.loads(a.transition_negatives.read_text())
    gate.require(len(negatives) == 4 and all(n["contract_pass"] for n in negatives), "transition negative missing")
    gate.require({n["name"]: n["rc"] for n in negatives} ==
                 {"init-good": 0, "init-bad": 1, "storage-good": 0, "storage-bad": 1},
                 "changed transition negative verdict")
    os.environ.update(FA17_TAIL_OWNERS=str(owners), FA17_TAIL_EXE=candidate["executable"],
                      FA17_STANDALONE_EXE=candidate["executable"], FA17_KV128_EXE=parent["executable"])
    loader, suite = unittest.TestLoader(), unittest.TestSuite()
    for path in sorted((ROOT / "tests").glob("test_ppu17*.py")):
        if path.stem == "test_ppu17_cpu_reference" and importlib.util.find_spec("torch") is None:
            for name in re.findall(r"^    def (test_\w+)\(", path.read_text(), re.M):
                test = unittest.FunctionTestCase(lambda: None, description=f"{path.stem}.{name}")
                test.setUp = lambda: (_ for _ in ()).throw(unittest.SkipTest(
                    "Torch unavailable; independent standalone C++ CPU oracle tested separately"))
                suite.addTest(test)
        else:
            suite.addTests(loader.loadTestsFromName(path.stem))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    gate.require(result.wasSuccessful(), "local test failure")
    record = {
        "scope": "CUDA12.8 SM90a build + CPU only; no device invocation",
        "source_manifest_sha256": candidate["source_manifest_sha256"],
        "backend_include_sha256": candidate["backend_include_sha256"],
        "compiler": candidate["compiler"],
        "binaries": {m.get("q_tail_mode", "C07") + "-kv" + str(m["kv_tile"]): m["sha256"] for m in records},
        "candidate_native": native, "candidate_ptx": ptx,
        "resources": dict(zip(("default", "M64", "C07", "rejected-hybrid"), role_resources)),
        "default_encoded_bodies_unchanged": "2/2", "causal_encoded_body_unchanged": True,
        "unchanged_source": unchanged,
        "ownership": "uniform896 and prefix440+half16:7340032cells each;5plants each EXPECTED_RED",
        "ownership_source_sha256": sha(HERE / "ownership.cu"), "ownership_exe_sha256": sha(owners),
        "ownership_compile": compile_cmd,
        "transition_negatives": negatives,
        "rejected_patch_sha256": sha(HERE / "rejected-hybrid.patch"),
        "local_tests": {"pass": result.testsRun - len(result.skipped), "skip": len(result.skipped), "fail": 0,
                        "skip_reasons": [reason for _, reason in result.skipped]},
        "M64_decision": "READY_FOR_SINGLE_SIMULATION; NOT A PERFORMANCE WIN",
        "hybrid_decision": "REJECTED_BEFORE_SIMULATION;32/32B spill exceeds12/12B",
        "native_ppu_codegen": "NOT_AVAILABLE", "candidate_numerics": "NOT_RUN",
        "candidate_performance": "NOT_RUN", "default_promotion": False,
    }
    (a.out / "results.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
