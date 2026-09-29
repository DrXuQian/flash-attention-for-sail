#!/usr/bin/env python3
"""Replay C07 compile and host gates; never invoke a GPU kernel."""
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
from build_ppu17_standalone import receipt

spec = importlib.util.spec_from_file_location(
    "overlap_validation", Path(__file__).parent.parent / "softmax-overlap/validate.py")
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-sass", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dirs = [args.control, args.candidate]
    manifests = [json.loads((d / "build.json").read_text()) for d in dirs]
    gate.require([m["kv_tile"] for m in manifests] == [176, 128], "wrong geometry pair")
    gate.require(all(m["softmax_overlap"] == "control" for m in manifests), "forced overlap present")
    for key in ("source_manifest_sha256", "backend_include_sha256", "compiler"):
        gate.require(manifests[0][key] == manifests[1][key], "mixed authority: " + key)
    current = receipt(Path(manifests[0]["cutlass"]))
    for key in ("source_manifest_sha256", "backend_include_sha256"):
        gate.require(current[key] == manifests[0][key], "build is stale: " + key)
    for manifest in manifests:
        gate.require(digest(Path(manifest["executable"])) == manifest["sha256"], "ELF hash mismatch")
    texts = [args.parent_sass.read_text(), *[(d / "executable.sass").read_text() for d in dirs]]
    encoded = [evidence.words(t) for t in texts]
    gate.require(len(encoded[0]) == 2 and encoded[0] == encoded[1], "default bodies changed")
    causal = next(key for key in encoded[0] if "DynamicPersistent" in key)
    gate.require(encoded[0][causal] == encoded[2][causal], "opt-in changed causal body")
    native = [gate.inspect_sass(t, kv_tile=n) for t, n in zip(texts[1:], (176, 128))]
    ptx = [gate.inspect_ptx((d / "shipping_fp16_d128.ptx").read_text(), kv_tile=n)
           for d, n in zip(dirs, (176, 128))]
    allocation = [evidence.resources((d / "shipping-object.log").read_text()) for d in dirs]
    # Geometry is encoded in the noncausal symbol. Compare its properties by
    # scheduler role, not by assuming mangled symbol names stay the same.
    role_resources = [{"causal" if "DynamicPersistent" in k else "noncausal": v
                       for k, v in a.items()} for a in allocation]
    gate.require(role_resources[0] == role_resources[1], "new register/stack/spill requirement")
    gate.require(native[0]["local_sites"] == native[1]["local_sites"], "new local-memory sites")
    os.environ.update(FA17_STANDALONE_EXE=manifests[1]["executable"],
                      FA17_KV128_EXE=manifests[1]["executable"])
    loader, suite = unittest.TestLoader(), unittest.TestSuite()
    for path in sorted((ROOT / "tests").glob("test_ppu17*.py")):
        if path.stem == "test_ppu17_cpu_reference" and importlib.util.find_spec("torch") is None:
            for name in re.findall(r"^    def (test_\w+)\(", path.read_text(), re.M):
                test = unittest.FunctionTestCase(lambda: None, description=f"{path.stem}.{name}")
                test.setUp = lambda: (_ for _ in ()).throw(unittest.SkipTest(
                    "Torch unavailable; independent standalone C++ CPU oracle runs separately"))
                suite.addTest(test)
        else:
            suite.addTests(loader.loadTestsFromName(path.stem))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    gate.require(result.wasSuccessful(), "local suite failed")
    record = {
        "scope": "CUDA12.8 SM90a compile and host checks only; no device invocation",
        "source_manifest_sha256": manifests[0]["source_manifest_sha256"],
        "backend_include_sha256": manifests[0]["backend_include_sha256"],
        "compiler": manifests[0]["compiler"],
        "executable_sha256": {str(m["kv_tile"]): m["sha256"] for m in manifests},
        "artifacts_sha256": {str(m["kv_tile"]): {name: digest(d / name) for name in
                              ("build.json", "executable.sass", "shipping_fp16_d128.ptx", "shipping-object.log")}
                             for d, m in zip(dirs, manifests)},
        "parent_sass_sha256": digest(args.parent_sass),
        "native_control": native[0], "native_candidate": native[1],
        "ptx_control": ptx[0], "ptx_candidate": ptx[1],
        "resources": role_resources[0],
        "default_encoded_bodies_unchanged": "2/2", "opt_in_causal_body_unchanged": True,
        "local_tests": {"pass": result.testsRun - len(result.skipped), "skip": len(result.skipped),
                        "fail": 0, "skip_reasons": [reason for _, reason in result.skipped]},
        "candidate_numerics": "NOT_RUN; host oracle tests do not execute the GPU body",
        "candidate_performance": "NOT_RUN", "native_ppu_lowering": "NOT_RUN",
        "default_promotion": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
