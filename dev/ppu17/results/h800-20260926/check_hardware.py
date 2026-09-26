"""Bounded H800 control: fresh process per case, no GPU reference or timing."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PLAN = ROOT / "cases.json"


def idle_evidence():
    gpu = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader"], text=True).strip()
    processes = subprocess.check_output(["ps", "-e", "-o", "pid=,comm="], text=True)
    competitors = []
    for line in processes.splitlines():
        pid, name = line.strip().split(maxsplit=1)
        if int(pid) in (os.getpid(), os.getppid()):
            continue
        if name.startswith("python") or name in {"nvcc", "cicc", "cc1plus", "single-launch"}:
            competitors.append({"pid": int(pid), "name": name})
    evidence = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "gpu_processes": gpu, "other_validation_or_build_processes": competitors}
    if gpu or competitors:
        raise RuntimeError("BUSY: " + json.dumps(evidence))
    return evidence


def main():
    plan = json.loads(PLAN.read_text())
    cases = plan["cases"]
    if len(cases) != 22 or plan["expected_cases"] != 22 or len({c["id"] for c in cases}) != 22:
        raise ValueError("case denominator changed or duplicated")
    summary = {"plan_sha256": hashlib.sha256(PLAN.read_bytes()).hexdigest(),
               "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "source_commit": plan["source_commit"], "backend_commit": plan["backend_commit"],
               "expected_cases": 22, "passed": 0, "complete": False, "cases": []}
    result_dir = ROOT / "results"
    result_dir.mkdir(exist_ok=True)
    common_binary = None
    try:
        for case in cases:
            name = case["id"]
            before = idle_evidence()
            out = result_dir / f"{name}.json"
            if out.exists():
                raise ValueError(f"refusing to reuse a result: {out}")
            cmd = [sys.executable, str(ROOT / "source/tools/run_ppu17_forward.py"),
                   "--extension-dir", str(ROOT / "python"), "--hardware-validation",
                   "--expected-sms", "114", "--verify", "--output", str(out)]
            for key in ("batch", "seqlen", "heads", "kv_heads", "head_dim", "dtype"):
                cmd += ["--" + key.replace("_", "-"), str(case[key])]
            if not case["causal"]:
                cmd.append("--noncausal")
            row = {"id": name, "command": cmd, "idle_before": before, "status": "PENDING"}
            summary["cases"].append(row)
            print(f"[H800 control] START {name}", flush=True)
            log = result_dir / f"{name}.log"
            with log.open("x") as stream:
                completed = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT,
                    env={**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}, timeout=180)
            row["returncode"] = completed.returncode
            if completed.returncode:
                row["status"] = "FAIL"
                print(log.read_text()[-6000:], flush=True)
                raise RuntimeError(f"case failed: {name}")
            data = json.loads(out.read_text())
            if data["numerics"] != "CPU-FLOAT64-REFERENCE/PASS" or data["role"] != "Hopper-hardware-validation":
                raise RuntimeError("result did not establish hardware numerical correctness")
            if common_binary is None:
                common_binary = data["extension_sha256"]
            if data["extension_sha256"] != common_binary or data["sm_count"] != 114:
                raise RuntimeError("binary/device changed within the control")
            row.update(status="PASS", max_abs=data["max_abs"],
                       extension_sha256=data["extension_sha256"], idle_after=idle_evidence())
            summary["passed"] += 1
            print(f"[H800 control] PASS {name} max_abs={data['max_abs']:.8g}", flush=True)
        summary["complete"] = summary["passed"] == summary["expected_cases"]
    except Exception as exc:
        summary["stop_reason"] = str(exc)
        raise
    finally:
        (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[H800 control] passed={summary['passed']}/22 complete={summary['complete']}", flush=True)


if __name__ == "__main__":
    main()
