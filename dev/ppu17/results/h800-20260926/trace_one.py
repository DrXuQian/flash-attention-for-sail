"""One traced forward; trace times are not a performance result."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

from check_hardware import idle_evidence


root = Path(__file__).resolve().parent
before = idle_evidence()
import torch

sys.argv = [str(root / "source/tools/run_ppu17_forward.py"),
            "--extension-dir", str(root / "python"), "--hardware-validation",
            "--expected-sms", "114", "--verify", "--seqlen", "2048",
            "--heads", "32", "--kv-heads", "2", "--head-dim", "256", "--dtype", "bf16",
            "--output", str(root / "trace-forward.json")]
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
    runpy.run_path(sys.argv[0], run_name="__main__")
trace_path = root / "single-forward-trace.json"
prof.export_chrome_trace(str(trace_path))
trace = json.loads(trace_path.read_text())
kernels = [event for event in trace["traceEvents"] if event.get("cat") == "kernel"]
if len(kernels) != 1:
    raise AssertionError(f"expected one GPU target, found {len(kernels)}: {[e.get('name') for e in kernels]}")
current = json.loads((root / "trace-forward.json").read_text())
prior = json.loads((root / "results/s2048-bf16-d256-causal.json").read_text())
if current["output_sha256"] != prior["output_sha256"]:
    raise AssertionError("fresh-process output fingerprint changed")
receipt = {"role": "Hopper-hardware-launch-count", "kernel_count": len(kernels),
           "kernel_name": kernels[0]["name"], "kernel_arguments": kernels[0].get("args", {}),
           "idle_before": before, "output_sha256": current["output_sha256"],
           "fresh_process_replay": "RAW-BIT/STABLE", "extension_sha256": current["extension_sha256"],
           "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
           "timing": "NOT_A_PERFORMANCE_RESULT"}
(root / "trace-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print("[H800 trace] one GPU target; CPU-only reference; fresh-process replay STABLE", flush=True)
