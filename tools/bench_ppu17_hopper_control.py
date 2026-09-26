#!/usr/bin/env python3
"""Physical H800 timing only; never use this repeat loop in the simulator."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from run_ppu17_forward import cpu_reference, invoke_once, logical_flops, validate_extension


ROOT = Path(__file__).resolve().parents[1]
ADMISSION = ROOT / "dev/ppu17/results/h800-20260926/s2048-bf16-d256-causal.json"
PEAK_BF16_DENSE_TFLOPS = 1513 / 2
PEAK_SOURCE = "https://lenovopress.lenovo.com/lp1814-thinksystem-nvidia-h800-pcie-gen5-gpu"
WARMUP, SAMPLES, CALLS = 200, 9, 50


def require_idle():
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                      "--format=csv,noheader,nounits"], text=True)
    others = [line.strip() for line in active.splitlines() if line.strip() != str(os.getpid())]
    processes = subprocess.check_output(["ps", "-e", "-o", "pid=,comm="], text=True)
    for line in processes.splitlines():
        pid, name = line.strip().split(maxsplit=1)
        if int(pid) in (os.getpid(), os.getppid()):
            continue
        if name.startswith("python") or name in {"nvcc", "cicc", "cc1plus", "single-launch"}:
            others.append(f"{pid}:{name}")
    if others:
        raise RuntimeError(f"BUSY: other compute/validation tasks: {others}")


def device_snapshot():
    fields = "name,uuid,driver_version,clocks.current.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu"
    return subprocess.check_output(["nvidia-smi", "--query-gpu=" + fields,
                                    "--format=csv"], text=True).strip()


def summarize(samples, flops):
    if not samples or any(t <= 0 for t in samples):
        raise ValueError("positive timing samples required")
    median = statistics.median(samples)
    target = flops / (PEAK_BF16_DENSE_TFLOPS * 1e6 * 0.70)
    verdict = ("AT_LEAST_70" if max(samples) <= target else
               "BELOW_70" if min(samples) > target else "UNRESOLVED_AT_70")
    return {"samples_us": samples, "median_us": median, "min_us": min(samples), "max_us": max(samples),
            "useful_tflops": flops / median / 1e6,
            "useful_mfu_percent": flops / median / 1e6 / PEAK_BF16_DENSE_TFLOPS * 100,
            "target_70_latency_us": target, "verdict": verdict}


def kernel_samples(trace, expected):
    kernels = [e for e in trace["traceEvents"] if e.get("cat") == "kernel"]
    if len(kernels) != expected or any("FlashAttnFwdSm90" not in e.get("name", "") for e in kernels):
        raise ValueError(f"wrong GPU kernel inventory: {len(kernels)}, expected {expected}")
    kernels.sort(key=lambda e: e["ts"])
    return [float(e["dur"]) for e in kernels], kernels[0]


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(__import__("torch").uint8).numpy().tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extension-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    require_idle()
    record = {"role": "H800-physical-performance-NOT-PPU17-simulation", "device_before": device_snapshot(),
              "warmup": WARMUP, "samples": SAMPLES, "calls_per_sample": CALLS,
              "peak_bf16_dense_tflops": PEAK_BF16_DENSE_TFLOPS, "peak_source": PEAK_SOURCE,
              "cache": "repeated-prepared-inputs-no-flush", "kernel_changed": False,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    import torch
    sys.path.insert(0, str(args.extension_dir.resolve()))
    extension = importlib.import_module("flash_attn_3._C")
    validate_extension(extension)
    interface = importlib.import_module("flash_attn_3.flash_attn_interface")
    admitted = json.loads(ADMISSION.read_text())
    binary_hash = hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest()
    if binary_hash != admitted["extension_sha256"]:
        raise ValueError("binary differs from the CPU-reference-admitted control")
    prop = torch.cuda.get_device_properties(0)
    if prop.name != "NVIDIA H800 PCIe" or prop.multi_processor_count != 114:
        raise ValueError("this peak/threshold is registered only for the inspected H800 PCIe")
    record.update(extension_sha256=binary_hash, shape=[1, 2048, 32, 256], kv_heads=2,
                  causal=True, dtype="bf16", sm_count=prop.multi_processor_count,
                  l2_cache_bytes=getattr(prop, "L2_cache_size", None))
    gen = torch.Generator(device="cpu").manual_seed(170020)
    host = [torch.randn(shape, generator=gen, device="cpu").to(torch.bfloat16)
            for shape in ((1, 2048, 32, 256), (1, 2048, 2, 256), (1, 2048, 2, 256))]
    if [digest(t) for t in host] != admitted["input_sha256"]:
        raise ValueError("fixture drift")
    q, k, v = [t.to("cuda") for t in host]
    expected, expected_lse = cpu_reference(host, causal=True)
    flops = logical_flops(1, 2048, 32, 256, True)
    record["useful_causal_flops"] = flops

    def validate(result):
        out, lse = result[0].cpu(), result[1].cpu()
        torch.testing.assert_close(out.double(), expected, atol=0.02, rtol=0.02)
        torch.testing.assert_close(lse.double(), expected_lse, atol=0.002, rtol=0.002)
        if digest(out) != admitted["output_sha256"]:
            raise AssertionError("admitted output fingerprint changed")
        return digest(out)

    with torch.inference_mode():
        result = invoke_once(interface, q, k, v, causal=True)
        validate(result)
        require_idle()
        for _ in range(WARMUP):
            result = invoke_once(interface, q, k, v, causal=True)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        call_samples = []
        for _ in range(SAMPLES):
            start.record()
            for _ in range(CALLS):
                result = invoke_once(interface, q, k, v, causal=True)
            end.record()
            end.synchronize()
            call_samples.append(start.elapsed_time(end) * 1000 / CALLS)
        record["full_call_event_span"] = summarize(call_samples, flops)
        record["device_after_events"] = device_snapshot()
        validate(result)
        require_idle()
        for _ in range(WARMUP):
            result = invoke_once(interface, q, k, v, causal=True)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(SAMPLES * CALLS):
                result = invoke_once(interface, q, k, v, causal=True)
            torch.cuda.synchronize()
        record["device_after_trace"] = device_snapshot()
        record["output_sha256"] = validate(result)
        require_idle()
    trace_path = args.out / "kernel-timing-trace.json"
    prof.export_chrome_trace(str(trace_path))
    durations, first = kernel_samples(json.loads(trace_path.read_text()), SAMPLES * CALLS)
    batches = [statistics.mean(durations[i:i + CALLS]) for i in range(0, len(durations), CALLS)]
    record["instrumented_kernel_only"] = summarize(batches, flops)
    record["kernel_durations_us"] = durations
    record["kernel_name"] = first["name"]
    record["kernel_arguments"] = first.get("args", {})
    record["trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    record["numerics"] = "CPU-FP64-O/LSE+ADMITTED-FINGERPRINT/PASS"
    (args.out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    for role in ("full_call_event_span", "instrumented_kernel_only"):
        print("[H800 causal perf] " + json.dumps({"role": role, **record[role]}), flush=True)
    print("[H800 causal perf] correctness PASS; artifacts=" + str(args.out), flush=True)


if __name__ == "__main__":
    main()
