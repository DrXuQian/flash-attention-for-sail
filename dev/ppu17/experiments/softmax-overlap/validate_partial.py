#!/usr/bin/env python3
"""Re-evaluate the rejected C04/C05/C06 compile experiment, without CUDA calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import validate as original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifacts.resolve()
    read = lambda folder, name: (root / folder / name).read_text()
    parent_words = original.words(read("control", "executable.sass"))
    control_words = original.words(read("control-partial", "executable.sass"))
    original.gate.require(len(parent_words) == 2 and parent_words == control_words,
                          "disabled candidate changed default machine code")
    control_resources = original.resources(read("control", "shipping-object.log"))
    noncausal = next(k for k in control_resources if "StaticPersistent" in k)
    causal = next(k for k in control_resources if "DynamicPersistent" in k)
    results = []
    for name, folder in (("C04", "c04-row0"), ("C05", "c05-rowwise"), ("C06", "c06-dependency")):
        manifest = json.loads(read(folder, "build.json"))
        sass = read(folder, "executable.sass")
        code = original.gate.inspect_sass(sass)
        ptx = original.gate.inspect_ptx(read(folder, "shipping_fp16_d128.ptx"))
        allocation = original.resources(read(folder, "shipping-object.log"))
        original.gate.require(original.words(sass)[causal] == parent_words[causal],
                              "ineligible causal body changed")
        schedule_ok = code["exp2_before_wait0"] == code["exp2_after_wait0"] == 45
        increased = {k: [control_resources[noncausal][k], allocation[noncausal][k]]
                     for k in control_resources[noncausal]
                     if allocation[noncausal][k] > control_resources[noncausal][k]}
        reasons = ([] if schedule_ok else ["SCHEDULE"]) + (["ADDED-RESOURCE/SPILL"] if increased else [])
        # A FAIL in a candidate is the expected result of this campaign, NOT
        # a passed kernel or environment SKIP. A changed outcome needs review.
        original.gate.require(bool(reasons), f"{name} unexpectedly admitted; re-evaluate evidence")
        results.append({"candidate": name, "decision": "REJECTED", "reasons": reasons,
                        "ptx_before_after": [ptx["exp2_before_wait0"], ptx["exp2_after_wait0"]],
                        "sass_before_after": [code["exp2_before_wait0"], code["exp2_after_wait0"]],
                        "sass": code, "resource_increase": increased,
                        "resources": allocation[noncausal], "source_sha256": manifest["source_manifest_sha256"],
                        "backend_include_sha256": manifest["backend_include_sha256"],
                        "executable_sha256": manifest["sha256"], "model_numerics_performance": "NOT_RUN"})

    host = root / "partial-host/partial-layout-dependency"
    checks = []
    for plant in (None, "row", "token", "coverage"):
        cmd = [str(host)] + ([plant] if plant else [])
        p = subprocess.run(cmd, text=True, capture_output=True)
        original.gate.require(p.returncode == (1 if plant else 0), "host layout/negative verdict changed")
        checks.append({"plant": plant, "returncode": p.returncode, "output": (p.stdout + p.stderr).strip()})
    original.gate.require("512 private word owners" in checks[0]["output"], "not the C06 layout executable")
    result = {"scope": "CUDA12.8 compile/control and CPU layout only; no candidate simulated",
              "default_encoded_bodies_unchanged": "2/2", "candidate_count": len(results),
              "admitted_candidates": 0, "candidates": results, "actual_layout_checks": checks,
              "rejected_source_patch_sha256": hashlib.sha256((HERE / "rejected-partial.patch").read_bytes()).hexdigest(),
              "layout_source_sha256": hashlib.sha256((HERE / "partial_layout.cpp").read_bytes()).hexdigest(),
              "retained_default": "original control; no new flags or softmax implementation left active"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
