#!/usr/bin/env python3
"""Reproduce local overlap admission from existing builds. No CUDA invocation."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tests")]
import check_ppu17_softmax_overlap as gate


def words(text):
    parts = re.split(r"(?m)(?=^[ \t]*Function[ \t]*:)", text)
    return {part.splitlines()[0].split(":", 1)[1].strip():
            re.findall(r"/\* (0x[0-9a-f]{16}) \*/", part)
            for part in parts if part.strip() and "Function :" in part.splitlines()[0]}


def resources(log):
    bodies = re.split(r"ptxas info\s*: Function properties for ", log)[1:]
    result = {}
    for body in bodies:
        name = body.splitlines()[0]
        frame = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", body)
        regs = re.search(r"Used (\d+) registers", body)
        gate.require(frame is not None and regs is not None, "incomplete ptxas resource evidence")
        result[name] = dict(zip(("stack_bytes", "spill_store_bytes", "spill_load_bytes", "registers"),
                               [*map(int, frame.groups()), int(regs[1])]))
    gate.require(len(result) == 2, "expected both generated bodies' resource records")
    gate.require("C751" not in log, "ptxas asynchronous serialization warning")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-sass", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifests = [json.loads((directory / "build.json").read_text()) for directory in (args.control, args.candidate)]
    gate.require([r["softmax_overlap"] for r in manifests] == ["control", "row-sum-token"], "wrong variant pair")
    for key in ("source_manifest_sha256", "backend_include_sha256", "compiler"):
        gate.require(manifests[0][key] == manifests[1][key], f"mixed control/candidate authority: {key}")
    texts = [args.parent_sass.read_text(), (args.control / "executable.sass").read_text(),
             (args.candidate / "executable.sass").read_text()]
    encoded = [words(text) for text in texts]
    gate.require(len(encoded[0]) == 2 and encoded[0] == encoded[1], "default bodies differ from admitted parent")
    gate.require(encoded[1].keys() == encoded[2].keys(), "generated symbol inventory changed")
    causal = next(key for key in encoded[1] if "DynamicPersistent" in key)
    gate.require(encoded[1][causal] == encoded[2][causal], "ineligible causal body changed")
    native = [gate.inspect_sass(text) for text in texts[1:]]
    gate.require([r["exp2_before_wait0"] for r in native] == [19, 90], "old overlap retained")
    gate.require(native[0]["local_sites"] == native[1]["local_sites"], "new local-memory sites")
    allocation = [resources((directory / "shipping-object.log").read_text()) for directory in (args.control, args.candidate)]
    gate.require(allocation[0] == allocation[1], "new register/stack/spill requirement")
    for directory in (args.control, args.candidate):
        gate.require(gate.inspect_ptx((directory / "shipping_fp16_d128.ptx").read_text())["exp2_before_wait0"] == 90,
                     "source PTX lost intended overlap")
    os.environ.update(FA17_STANDALONE_EXE=manifests[1]["executable"],
                      FA17_OVERLAP_SASS=str(args.candidate / "executable.sass"),
                      FA17_OVERLAP_REPORT=str(args.report))
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for path in sorted((ROOT / "tests").glob("test_ppu17*.py")):
        if path.stem == "test_ppu17_cpu_reference" and importlib.util.find_spec("torch") is None:
            # Preserve the exact omitted denominator; do not import a broken
            # unrelated Python ABI or relabel an unavailable oracle as PASS.
            names = re.findall(r"^    def (test_\w+)\(", path.read_text(), re.M)
            for name in names:
                test = unittest.FunctionTestCase(lambda: None, description=f"{path.stem}.{name}")
                test.setUp = lambda: (_ for _ in ()).throw(unittest.SkipTest("Torch unavailable in this Python; standalone C++ CPU oracle is tested separately"))
                suite.addTest(test)
        else:
            suite.addTests(loader.loadTestsFromName(path.stem))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    gate.require(result.wasSuccessful(), "local test suite failed")
    report = gate.inspect_report(json.loads(args.report.read_bytes()))
    record = {
        "scope": "local CUDA12.8 SM90a compile/control only; candidate PPU model NOT_RUN",
        "source_manifest_sha256": manifests[0]["source_manifest_sha256"],
        "backend_include_sha256": manifests[0]["backend_include_sha256"],
        "executable_sha256": {r["softmax_overlap"]: r["sha256"] for r in manifests},
        "uploaded_report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "control_report": report, "native_control": native[0], "native_candidate": native[1],
        "resources": allocation[0], "default_encoded_bodies_unchanged": "2/2",
        "ineligible_causal_body_unchanged": True,
        "local_tests": {"pass": result.testsRun - len(result.skipped), "skip": len(result.skipped),
                        "fail": 0, "skip_reasons": [reason for _, reason in result.skipped]},
        "candidate_numerics": "NOT_RUN; CPU oracle tests do not execute the kernel",
        "candidate_performance": "NOT_RUN", "default_promotion": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
