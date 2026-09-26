#!/usr/bin/env python3
"""Bounded physical-H800 causal comparison; never run loops in the simulator."""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

from bench_h800_official_fa3 import check_inputs, wait_before_first_launch
from bench_ppu17_hopper_control import (
    ADMISSION, CALLS, SAMPLES, WARMUP, PEAK_BF16_DENSE_TFLOPS, PEAK_SOURCE,
    device_snapshot, digest, require_idle, summarize,
)
from run_ppu17_forward import cpu_reference, logical_flops, validate_extension


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(trace, arm, calls):
    kernels = sorted((e for e in trace["traceEvents"] if e.get("cat") == "kernel"),
                     key=lambda e: e["ts"])
    names = Counter(e["name"] for e in kernels)
    if arm in ("control", "candidate"):
        admitted = lambda n: "FlashAttnFwdSm90" in n
    elif arm == "flashinfer":
        # SinglePrefill... is the host dispatcher. The Hopper device entry is
        # PrefillWithKVCacheKernel; bind its TMA traits and single-tile scheduler
        # too, so the generic name cannot accidentally admit an FA2 fallback.
        admitted = lambda n: ("flashinfer::PrefillWithKVCacheKernel<" in n and
                              "flashinfer::AttentionKernelTraits<true, 256, 256, 128, 64, 2," in n and
                              "flashinfer::SingleTileScheduler" in n)
    else:
        # The direct aten cuDNN op has no SDPA fallback. Also require its actual
        # attention symbol, not an arbitrary reference/transpose/expand kernel.
        admitted = lambda n: any(x in n.lower() for x in ("cudnn", "fmha"))
    if not kernels or any(not admitted(n) for n in names):
        raise ValueError(f"wrong explicit-backend kernel inventory: {dict(names)}")
    if len(names) != 1 or len(kernels) != calls:
        raise ValueError(f"expected one attention kernel/call, got {dict(names)}, calls={calls}")
    if any(float(e["dur"]) <= 0 for e in kernels):
        raise ValueError("nonpositive kernel duration")
    extras = [e for e in trace["traceEvents"] if e.get("cat") in ("gpu_memcpy", "gpu_memset")]
    return {
        "kernel_counts": dict(names), "kernel_name": kernels[0]["name"],
        "kernel_arguments": kernels[0].get("args", {}),
        "kernel_durations_us": [float(e["dur"]) for e in kernels],
        "gpu_copy_memset_counts": dict(Counter(e["name"] for e in extras)),
        "gpu_copy_memset_total_us": sum(float(e.get("dur", 0)) for e in extras),
    }


def oracle(host, cache):
    import torch
    identity = {"input_sha256": [digest(t) for t in host],
                "oracle_source_sha256": sha(Path(__file__).with_name("run_ppu17_forward.py"))}
    if cache.exists():
        data = torch.load(cache, map_location="cpu", weights_only=True)
        if data["identity"] != identity:
            raise ValueError("CPU oracle cache identity mismatch")
    else:
        print("[causal tuning] full CPU FP64 O/LSE oracle starting", flush=True)
        out, lse = cpu_reference(host, causal=True)
        data = {"identity": identity, "out": out, "lse": lse,
                "reference_sha256": [digest(out), digest(lse)]}
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, cache)
    if data["reference_sha256"] != [digest(data["out"]), digest(data["lse"])]:
        raise ValueError("CPU oracle tensor hash mismatch")
    if any(t.device.type != "cpu" or t.dtype != torch.float64 for t in (data["out"], data["lse"])):
        raise ValueError("oracle must remain CPU FP64")
    return data["out"], data["lse"], {**identity, "reference_sha256": data["reference_sha256"],
                                      "cache_sha256": sha(cache)}


def check_candidate_receipt(record):
    """Bind requested axes to the actual instantiated body AND host behavior."""
    if record["arm"] not in ("control", "candidate"):
        return
    config = {"tile": [128,80], "scheduler": "LPT"} if record["arm"] == "control" else record["source_identity"]
    n = config["tile"][1]
    compact = re.sub(r"\s+", "", record["kernel_name"])
    shape = f"cute::tuple<cute::C<128>,cute::C<{n}>,cute::C<256>>"
    if shape not in compact:
        raise ValueError("generated kernel tile differs from requested candidate")
    single = config["scheduler"] == "single"
    if ("SingleTileScheduler" if single else "DynamicPersistentTileScheduler") not in compact:
        raise ValueError("generated scheduler differs from requested candidate")
    grid = [(record["shape"][1]+127)//128, 32, 1] if single else [114,1,1]
    if record["kernel_arguments"].get("grid") != grid:
        raise ValueError("actual grid differs from scheduler contract")
    if single and record["gpu_copy_memset_counts"]:
        raise ValueError("single-tile retained unused host counter initialization")


def loaded_libraries():
    paths = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
             if "/" in line and ".so" in line}
    return {p: sha(p) for p in sorted(paths) if Path(p).is_file() and
            any(x in p for x in ("flashinfer", "cudnn", "flash_attn_3"))}


@contextmanager
def power_samples(out, record):
    """Read-only NVML sampling; terminate only the child we created."""
    path = out / "power.csv"
    cmd = ["nvidia-smi", "--query-gpu=timestamp,uuid,power.draw,power.limit,clocks.current.sm,clocks_event_reasons.sw_power_cap",
           "--format=csv", "--loop-ms=200"]
    record["power_monitor"] = {"command": cmd, "start_utc": datetime.now(timezone.utc).isoformat(),
                               "host_timezone": datetime.now().astimezone().isoformat(),
                               "read_only": True, "interval_ms": 200}
    with path.open("x") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            yield
        finally:
            failed = proc.poll()
            if failed is None:
                proc.terminate()
            proc.wait(timeout=10)
            record["power_monitor"].update(end_utc=datetime.now(timezone.utc).isoformat(),
                                            sampler_exit_before_stop=failed)
    record["power_monitor"]["sha256"] = sha(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("control", "candidate", "flashinfer", "cudnn"), required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--extension-dir", type=Path)
    ap.add_argument("--flashinfer-source-root", type=Path,
                    help="explicit uninstalled source tree; JIT reads it without editing its package")
    ap.add_argument("--expected-sha256")
    ap.add_argument("--seqlen", type=int, choices=(2048, 8192), required=True)
    ap.add_argument("--oracle-cache", type=Path, required=True)
    ap.add_argument("--source-identity", type=Path, required=True)
    ap.add_argument("--prelaunch-idle-wait", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.arm in ("control", "candidate") and (not args.extension_dir or not args.expected_sha256):
        ap.error("extension and hash required")
    args.out.mkdir(parents=True, exist_ok=False)
    require_idle()
    import torch
    torch.set_num_threads(8)
    admitted = json.loads(ADMISSION.read_text())
    record = {
        "role": "H800-physical-causal-tuning-NOT-PPU17", "arm": args.arm, "label": args.label,
        "device_before": device_snapshot(), "warmup": WARMUP, "samples": SAMPLES,
        "calls_per_sample": CALLS, "cache": "repeated-prepared-inputs-no-flush",
        "peak_bf16_dense_tflops": PEAK_BF16_DENSE_TFLOPS, "peak_source": PEAK_SOURCE,
        "script_sha256": sha(__file__), "source_identity": json.loads(args.source_identity.read_text()),
        "source_identity_sha256": sha(args.source_identity),
        "shape": [1, args.seqlen, 32, 256], "kv_heads": 2, "dtype": "bf16", "causal": True,
        "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if args.arm in ("control", "candidate"):
        sys.path.insert(0, str(args.extension_dir.resolve()))
        extension = importlib.import_module("flash_attn_3._C")
        validate_extension(extension)
        if sha(extension.__file__) != args.expected_sha256:
            raise ValueError("extension identity mismatch")
        if args.arm == "control" and args.expected_sha256 != admitted["extension_sha256"]:
            raise ValueError("control is not immutable admitted binary")
        record.update(extension_path=extension.__file__, extension_sha256=args.expected_sha256)
    elif args.arm == "flashinfer":
        import flashinfer
        record.update(flashinfer_version=flashinfer.__version__, flashinfer_file=flashinfer.__file__)
        if args.flashinfer_source_root:
            # FlashInfer's JIT env supports path overrides (also used by its
            # AOT scripts). The raw checkout lacks wheel data/ symlinks. Do NOT
            # install/edit the other task's package or substitute a kernel.
            from flashinfer.jit import env as jit_env
            root = args.flashinfer_source_root.resolve()
            if Path(flashinfer.__file__).resolve().parent.parent != root:
                raise ValueError("FlashInfer Python and C++ source roots differ")
            jit_env.FLASHINFER_INCLUDE_DIR = root / "include"
            jit_env.FLASHINFER_CSRC_DIR = root / "csrc"
            jit_env.CUTLASS_INCLUDE_DIRS = [root / "3rdparty/cutlass/include",
                                           root / "3rdparty/cutlass/tools/util/include"]
            jit_env.SPDLOG_INCLUDE_DIR = root / "3rdparty/spdlog/include"
            jit_env.CCCL_INCLUDE_DIRS = [root / "3rdparty/cccl" / p
                                        for p in ("cub", "libcudacxx/include", "thrust")]
            required = [root / "csrc/single_prefill_sm90_customize_config.jinja",
                        root / "include/flashinfer/attention/hopper/prefill_sm90.cuh",
                        root / "3rdparty/cutlass/include/cutlass/cutlass.h"]
            if any(not p.is_file() for p in required):
                raise ValueError("incomplete explicit FlashInfer source/dependencies")
            paths = [*root.glob("include/flashinfer/attention/hopper/*.cuh"),
                     *root.glob("csrc/*single_prefill*"), root / "flashinfer/prefill.py"]
            record["flashinfer_source_sha256"] = {str(p.relative_to(root)): sha(p) for p in paths}
    prop = torch.cuda.get_device_properties(0)
    if prop.name != "NVIDIA H800 PCIe" or prop.multi_processor_count != 114:
        raise ValueError("registered H800 device mismatch")
    record.update(sm_count=prop.multi_processor_count, l2_cache_bytes=getattr(prop, "L2_cache_size", None))
    gen = torch.Generator(device="cpu").manual_seed(170020)
    host = [torch.randn(shape, generator=gen, device="cpu").to(torch.bfloat16)
            for shape in ((1, args.seqlen, 32, 256), (1, args.seqlen, 2, 256),
                          (1, args.seqlen, 2, 256))]
    record["input_sha256"] = [digest(t) for t in host]
    if args.seqlen == 2048:
        check_inputs(record["input_sha256"], admitted["input_sha256"])
    expected, expected_lse, record["oracle"] = oracle(host, args.oracle_cache)
    wait_before_first_launch(args.prelaunch_idle_wait)
    q, k, v = [t.to("cuda") for t in host]
    if args.arm == "flashinfer":
        q, k, v = q[0], k[0], v[0]  # views only, same BSHD storage
        def invoke():
            return flashinfer.single_prefill_with_kv_cache(
                q, k, v, causal=True, kv_layout="NHD", sm_scale=1/16,
                backend="fa3", return_lse=True)
        record["invocation"] = "single_prefill_with_kv_cache: explicit fa3,NHD,causal,no reduction in precision"
        record["lse_convention"] = "log2; converted to ln on CPU only for validation"
    elif args.arm == "cudnn":
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        def invoke():
            return torch.ops.aten._scaled_dot_product_cudnn_attention.default(
                q, k, v, None, True, 0., True, False, scale=1/16)
        record["invocation"] = "aten::_scaled_dot_product_cudnn_attention: direct backend,no fallback,BHSD views"
    else:
        def invoke():
            return torch.ops.flash_attn_3.fwd(q, k, v, softmax_scale=1/16,
                                             is_causal=True, num_splits=1, pack_gqa=False)
        record["invocation"] = "flash_attn_3.fwd: num_splits=1,pack_gqa=False"
    record["device_input_strides"] = [list(t.stride()) for t in (q, k, v)]
    anchor = None

    def validate(result):
        nonlocal anchor
        out, lse = result[0].cpu(), result[1].cpu()
        if args.arm == "flashinfer":
            out, lse = out.unsqueeze(0), lse.T.unsqueeze(0).double() * math.log(2)
        elif args.arm == "cudnn":
            out = out.transpose(1, 2)
        if not torch.isfinite(out).all() or not torch.isfinite(lse).all():
            raise AssertionError("nonfinite O/LSE")
        torch.testing.assert_close(out.double(), expected, atol=.02, rtol=.02)
        torch.testing.assert_close(lse.double(), expected_lse, atol=.002, rtol=.002)
        hashes = [digest(out), digest(lse)]
        if anchor is not None and hashes != anchor:
            raise AssertionError("within-arm raw replay changed")
        if args.arm == "control" and args.seqlen == 2048 and hashes[0] != admitted["output_sha256"]:
            raise AssertionError("immutable control changed")
        anchor = hashes
        record.update(output_sha256=hashes[0], lse_sha256=hashes[1],
                      max_output_abs=(out.double()-expected).abs().max().item(),
                      max_lse_abs=(lse.double()-expected_lse).abs().max().item())

    flops = logical_flops(1, args.seqlen, 32, 256, True)
    record["useful_causal_flops"] = flops
    with power_samples(args.out, record), torch.inference_mode():
        validate(invoke())
        require_idle()
        print(f"[causal tuning] {args.label}/S{args.seqlen} CPU-FP64 O/LSE PASS", flush=True)
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
            samples.append(start.elapsed_time(end)*1000/CALLS)
        validate(result)
        require_idle()
        record["full_call_event_span"] = summarize(samples, flops)
        record["device_after_events"] = device_snapshot()
        (args.out / "events.json").write_text(json.dumps(record, indent=2)+"\n")
        for _ in range(WARMUP):
            result = invoke()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(SAMPLES*CALLS):
                result = invoke()
            torch.cuda.synchronize()
        validate(result)
        require_idle()
    record["device_after_trace"] = device_snapshot()
    record["loaded_library_sha256"] = loaded_libraries()
    trace = args.out / "kernel-timing-trace.json"
    prof.export_chrome_trace(str(trace))
    record.update(inventory(json.loads(trace.read_text()), args.arm, SAMPLES*CALLS))
    check_candidate_receipt(record)
    durations = record["kernel_durations_us"]
    batches = [statistics.mean(durations[i:i+CALLS]) for i in range(0, len(durations), CALLS)]
    record["instrumented_kernel_only"] = summarize(batches, flops)
    record["trace_sha256"] = sha(trace)
    record["numerics"] = "CPU-FP64-O/LSE+WITHIN-ARM-RAW-REPLAY/PASS"
    (args.out / "result.json").write_text(json.dumps(record, indent=2)+"\n")
    for role in ("full_call_event_span", "instrumented_kernel_only"):
        print("[causal tuning] "+json.dumps({"label": args.label, "seqlen": args.seqlen,
                                            "role": role, **record[role]}), flush=True)
    print("[causal tuning] PASS; artifacts="+str(args.out), flush=True)


if __name__ == "__main__":
    main()
