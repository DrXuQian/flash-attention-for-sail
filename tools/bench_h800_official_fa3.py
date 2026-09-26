#!/usr/bin/env python3
"""Physical H800 official FA3 A/B; never use this repeat loop in simulation."""
import argparse
from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import sys

from bench_ppu17_hopper_control import (
    ADMISSION, CALLS, PEAK_BF16_DENSE_TFLOPS, PEAK_SOURCE, SAMPLES, WARMUP,
    device_snapshot, digest, require_idle, summarize,
)
from run_ppu17_forward import cpu_reference, logical_flops, validate_extension


UPSTREAM = "a8aa52b1ab3e9ca574c8a33b3f35afc017ffa2e2"
UPSTREAM_CUTLASS = "dc4817921edda44a549197ff3a9dcf5df0636e7b"
SEQUENCE_LENGTHS = (2048, 4096, 8192, 16384)


def check_identity(arm, actual_hash, expected_hash, admitted_hash):
    if actual_hash != expected_hash:
        raise ValueError("binary identity mismatch")
    if arm == "control" and actual_hash != admitted_hash:
        raise ValueError("control is not the immutable admitted binary")
    if arm == "official" and actual_hash == admitted_hash:
        raise ValueError("official arm aliases the admitted control")


def check_inputs(actual, admitted):
    if len(actual) != 3 or actual != admitted:
        raise ValueError("same-input admission failed")


def check_sequence_inputs(seqlen, hashes, reference):
    if (reference.get("shape") != [1, seqlen, 32, 256] or
            reference.get("kv_heads") != 2 or reference.get("dtype") != "bf16" or
            reference.get("causal") is not True):
        raise ValueError("same-input reference shape/type mismatch")
    check_inputs(hashes, reference["input_sha256"])


def parse_trace(trace, expected, arm):
    kernels = sorted((e for e in trace["traceEvents"] if e.get("cat") == "kernel"),
                     key=lambda e: e["ts"])
    attention = [e for e in kernels if "FlashAttnFwdSm90" in e.get("name", "")]
    auxiliary = [e for e in kernels if "FlashAttnFwdSm90" not in e.get("name", "")]
    if len(attention) != expected:
        raise ValueError(f"attention inventory {len(attention)} != {expected}")
    # Upstream's integer scheduling semaphore is initialized by Tensor.zero_().
    # Admit only this specific auxiliary kind, never an arbitrary GPU reference.
    if auxiliary and (arm != "official" or len(auxiliary) != expected or
                      any("FillFunctor<int>" not in e.get("name", "") for e in auxiliary)):
        raise ValueError("unregistered auxiliary GPU kernel inventory")
    if any(float(e["dur"]) <= 0 for e in kernels):
        raise ValueError("nonpositive kernel duration")
    other_gpu = [e for e in trace["traceEvents"] if e.get("cat") in ("gpu_memcpy", "gpu_memset")]
    return {
        "attention_durations_us": [float(e["dur"]) for e in attention],
        "attention_name": attention[0]["name"],
        "attention_arguments": attention[0].get("args", {}),
        "auxiliary_kernel_counts": dict(Counter(e["name"] for e in auxiliary)),
        "auxiliary_kernel_total_us": sum(float(e["dur"]) for e in auxiliary),
        "gpu_copy_memset_counts": dict(Counter(e["name"] for e in other_gpu)),
        "gpu_copy_memset_total_us": sum(float(e.get("dur", 0)) for e in other_gpu),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("control", "official"), required=True)
    ap.add_argument("--extension-dir", type=Path, required=True)
    ap.add_argument("--expected-sha256", required=True)
    ap.add_argument("--seqlen", type=int, choices=SEQUENCE_LENGTHS, default=2048)
    ap.add_argument("--same-input-as", type=Path,
                    help="prior completed result for this length; required for new official lengths")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.seqlen != 2048 and args.arm == "official" and args.same_input_as is None:
        ap.error("larger official shapes require --same-input-as from the control")
    args.out.mkdir(parents=True, exist_ok=False)
    require_idle()
    record = {
        "role": "H800-physical-official-FA3-AB-NOT-PPU17", "arm": args.arm,
        "device_before": device_snapshot(), "warmup": WARMUP,
        "samples": SAMPLES, "calls_per_sample": CALLS,
        "peak_bf16_dense_tflops": PEAK_BF16_DENSE_TFLOPS, "peak_source": PEAK_SOURCE,
        "cache": "repeated-prepared-inputs-no-flush", "kernel_changed": False,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_commit": UPSTREAM if args.arm == "official" else "e7ee864",
        "backend_commit": UPSTREAM_CUTLASS if args.arm == "official" else "023e82d",
        "invocation": "torch.ops.flash_attn_3.fwd: identical kwargs; num_splits=1,pack_gqa=False",
    }
    import torch
    sys.path.insert(0, str(args.extension_dir.resolve()))
    extension = importlib.import_module("flash_attn_3._C")
    if args.arm == "control":
        validate_extension(extension)
    elif hasattr(extension, "ppu17_backend"):
        raise ValueError("official arm loaded a ported extension")
    admitted = json.loads(ADMISSION.read_text())
    binary_hash = hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest()
    check_identity(args.arm, binary_hash, args.expected_sha256, admitted["extension_sha256"])
    prop = torch.cuda.get_device_properties(0)
    if prop.name != "NVIDIA H800 PCIe" or prop.multi_processor_count != 114:
        raise ValueError("peak and experiment registered only for inspected H800 PCIe")
    record.update(extension_path=extension.__file__, extension_sha256=binary_hash,
                  shape=[1, args.seqlen, 32, 256], kv_heads=2, dtype="bf16", causal=True,
                  sm_count=prop.multi_processor_count,
                  l2_cache_bytes=getattr(prop, "L2_cache_size", None),
                  torch_version=torch.__version__, torch_cuda=torch.version.cuda)
    gen = torch.Generator(device="cpu").manual_seed(170020)
    host = [torch.randn(shape, generator=gen, device="cpu").to(torch.bfloat16)
            for shape in ((1, args.seqlen, 32, 256), (1, args.seqlen, 2, 256),
                          (1, args.seqlen, 2, 256))]
    record["input_sha256"] = [digest(t) for t in host]
    record["input_seed"] = 170020
    if args.seqlen == 2048:
        check_inputs(record["input_sha256"], admitted["input_sha256"])
    if args.same_input_as:
        check_sequence_inputs(args.seqlen, record["input_sha256"],
                              json.loads(args.same_input_as.read_text()))
        record["same_input_reference_sha256"] = hashlib.sha256(args.same_input_as.read_bytes()).hexdigest()
    print(f"[H800 official A/B] arm={args.arm} S={args.seqlen} full CPU-FP64 reference starting", flush=True)
    expected, expected_lse = cpu_reference(host, causal=True)
    require_idle()
    q, k, v = [t.to("cuda") for t in host]
    flops = logical_flops(1, args.seqlen, 32, 256, True)
    record["useful_causal_flops"] = flops
    anchor = None

    def invoke():
        return torch.ops.flash_attn_3.fwd(q, k, v, softmax_scale=1 / 16,
                                         is_causal=True, num_splits=1, pack_gqa=False)

    def validate(result):
        nonlocal anchor
        out, lse = result[0].cpu(), result[1].cpu()
        if not torch.isfinite(out).all() or not torch.isfinite(lse).all():
            raise AssertionError("nonfinite output/LSE")
        torch.testing.assert_close(out.double(), expected, atol=.02, rtol=.02)
        torch.testing.assert_close(lse.double(), expected_lse, atol=.002, rtol=.002)
        hashes = [digest(out), digest(lse)]
        if anchor is not None and hashes != anchor:
            raise AssertionError("within-arm output/LSE replay changed")
        if args.arm == "control" and args.seqlen == 2048 and hashes[0] != admitted["output_sha256"]:
            raise AssertionError("admitted control output changed")
        anchor = hashes
        record.update(output_sha256=hashes[0], lse_sha256=hashes[1],
                      max_output_abs=(out.double() - expected).abs().max().item(),
                      max_lse_abs=(lse.double() - expected_lse).abs().max().item())

    with torch.inference_mode():
        validate(invoke())
        print("[H800 official A/B] " + args.arm + " CPU-FP64 admission PASS", flush=True)
        require_idle()
        for _ in range(WARMUP):
            result = invoke()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(SAMPLES):
            start.record()
            for _ in range(CALLS):
                result = invoke()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / CALLS)
        record["full_call_event_span"] = summarize(samples, flops)
        record["device_after_events"] = device_snapshot()
        validate(result)
        # Save completed uninstrumented data before the independent trace phase.
        (args.out / "events.json").write_text(json.dumps(record, indent=2) + "\n")
        require_idle()
        for _ in range(WARMUP):
            result = invoke()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(SAMPLES * CALLS):
                result = invoke()
            torch.cuda.synchronize()
        record["device_after_trace"] = device_snapshot()
        require_idle()
        validate(result)
    trace_path = args.out / "kernel-timing-trace.json"
    prof.export_chrome_trace(str(trace_path))
    inventory = parse_trace(json.loads(trace_path.read_text()), SAMPLES * CALLS, args.arm)
    durations = inventory["attention_durations_us"]
    batches = [statistics.mean(durations[i:i + CALLS]) for i in range(0, len(durations), CALLS)]
    record.update(inventory)
    record["instrumented_kernel_only"] = summarize(batches, flops)
    record["trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    record["numerics"] = "CPU-FP64-O/LSE+WITHIN-ARM-RAW-REPLAY/PASS"
    (args.out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    for role in ("full_call_event_span", "instrumented_kernel_only"):
        print("[H800 official A/B] " + json.dumps({"arm": args.arm, "role": role, **record[role]}), flush=True)
    print("[H800 official A/B] PASS; artifacts=" + str(args.out), flush=True)


if __name__ == "__main__":
    main()
