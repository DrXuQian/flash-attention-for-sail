#!/usr/bin/env python3
"""Same-input, CPU-oracle, interleaved FA2 geometry comparison (no install)."""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
CONTRACT = json.loads((HERE / "contract.json").read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_digest(tensor):
    import torch
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def require_numeric(got, expected, dtype):
    import torch
    tol = CONTRACT["numeric"][dtype]
    if not torch.isfinite(got).all() or not torch.isfinite(expected).all():
        raise AssertionError("non-finite output/reference")
    error = (got.double() - expected).abs()
    allowance = tol["atol"] + tol["rtol"] * expected.abs()
    if not (error <= allowance).all():
        raise AssertionError(f"CPU FP64 mismatch: max_abs={error.max().item():.9g}, "
                             f"max_error/limit={(error/allowance).max().item():.9g}")
    return {"max_abs": error.max().item(), "max_error_over_limit": (error/allowance).max().item()}


def envelope(control, candidate):
    if max(candidate) < min(control):
        return "CANDIDATE-FASTER"
    if min(candidate) > max(control):
        return "CONTROL-FASTER"
    return "UNRESOLVED"


class Variant:
    def __init__(self, entry, build, dtype):
        self.entry, self.dtype = entry, dtype
        path = build / entry["library"]
        if digest(path) != entry["sha256"]:
            raise RuntimeError(f"binary hash mismatch: {path}")
        self.lib = ctypes.CDLL(str(path))
        self.lib.fa15_error_string.argtypes = [ctypes.c_int]
        self.lib.fa15_error_string.restype = ctypes.c_char_p
        self.lib.fa15_runtime_versions.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.lib.fa15_runtime_versions.restype = ctypes.c_int
        self.lib.fa15_resources.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.lib.fa15_resources.restype = ctypes.c_int
        self.lib.fa15_create.argtypes = [ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p]*5
        self.lib.fa15_create.restype = ctypes.c_void_p
        self.lib.fa15_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.fa15_launch.restype = ctypes.c_int
        self.lib.fa15_destroy.argtypes = [ctypes.c_void_p]
        self.lib.fa15_destroy.restype = None
        values = (ctypes.c_int*7)()
        self.dtype_id = int(dtype == "bf16")
        rc = self.lib.fa15_resources(self.dtype_id, values)
        if rc:
            raise RuntimeError(f"resource query/admission failed: {entry['name']} "
                               f"rc={rc} {self.lib.fa15_error_string(rc).decode()}")
        self.resources = dict(zip(("bm", "bn", "warps", "shared_bytes", "registers",
                                   "local_bytes", "maximum_active_blocks"), values))
        for field in ("bm", "bn", "warps", "shared_bytes"):
            if self.resources[field] != entry[field]:
                raise RuntimeError(f"compiled geometry disagrees with manifest: {field}")
        if self.resources["local_bytes"] != 0 or self.resources["maximum_active_blocks"] < 1:
            raise RuntimeError(f"resource gate rejected {entry['name']}: {self.resources}")
        versions = (ctypes.c_int*2)()
        rc = self.lib.fa15_runtime_versions(versions)
        if rc:
            raise RuntimeError(f"runtime/driver version query failed: rc={rc}")
        self.versions = {"runtime": versions[0], "driver": versions[1]}

    def prepare(self, q, k, v):
        import torch
        self.o = torch.empty_like(q)
        self.lse = torch.empty((1, 56, q.shape[1]), device=q.device, dtype=torch.float32)
        self.handle = self.lib.fa15_create(self.dtype_id, q.shape[1], q.data_ptr(),
                                          k.data_ptr(), v.data_ptr(), self.o.data_ptr(),
                                          self.lse.data_ptr())
        if not self.handle:
            raise RuntimeError("C ABI rejected input geometry/pointers")
        self.stream = torch.cuda.current_stream().cuda_stream

    def launch(self):
        rc = self.lib.fa15_launch(self.handle, self.stream)
        if rc:
            raise RuntimeError(f"launch failed {self.entry['name']} rc={rc} "
                               + self.lib.fa15_error_string(rc).decode())

    def release(self):
        self.lib.fa15_destroy(self.handle)
        self.handle = None
        self.o = self.lse = None


def fixture(seq, dtype):
    import torch
    generator = torch.Generator().manual_seed(20261014 + seq)
    dt = getattr(torch, "float16" if dtype == "fp16" else "bfloat16")
    cpu = [torch.randn((1, seq, 56, 128), generator=generator).to(dt) for _ in range(3)]
    device = [x.cuda() for x in cpu]
    return cpu, device


def oracle(cpu, seq):
    import torch
    # Entire small cases; all 56 heads, every K and boundary Q rows on target.
    rows = list(range(seq)) if seq <= 129 else sorted(set(
        [0, 1, 63, 64, 65, 127, 128, 129, seq//2, 8191, 8192, 8319, 8320, seq-1]))
    q, k, v = cpu
    qh = q[0, rows].permute(1, 0, 2).double()
    kh, vh = (x[0].permute(1, 0, 2).double() for x in (k, v))
    scores = (qh @ kh.transpose(-1, -2)) / math.sqrt(128)
    expected = (scores.softmax(-1) @ vh).permute(1, 0, 2)
    lse = scores.logsumexp(-1)
    return rows, expected, lse


def validate(variants, cpu, device, dtype, seq):
    import torch
    rows, expected, lse_ref = oracle(cpu, seq)
    control = None
    for variant in variants:
        variant.prepare(*device)
        variant.o.fill_(float("nan")); variant.lse.fill_(float("nan"))
        variant.launch()
        output, lse = variant.o.cpu(), variant.lse.cpu()
        if not torch.isfinite(output).all() or not torch.isfinite(lse).all():
            raise AssertionError("unwritten/non-finite output or LSE")
        result = require_numeric(output[0, rows], expected, dtype)
        lse_error = (lse[0, :, rows].double() - lse_ref).abs().max().item()
        if lse_error > CONTRACT["numeric"]["lse_atol"]:
            raise AssertionError(f"LSE mismatch {lse_error}")
        full_control_max_abs = 0.0
        if variant.entry["id"] == 0:
            control = output
        else:
            if control is None:
                raise AssertionError("the full-output control must be admitted first")
            # Independent CPU samples anchor the control. Compare every output
            # against it as well; chunk host temporaries, not device computation.
            for begin in range(0, seq, 128):
                pair = require_numeric(output[:, begin:begin+128], control[:, begin:begin+128], dtype)
                full_control_max_abs = max(full_control_max_abs, pair["max_abs"])
        for _ in range(CONTRACT["numeric"]["repeat_launches"] - 1):
            variant.launch()
            if not torch.equal(output.view(torch.int16), variant.o.cpu().view(torch.int16)):
                raise AssertionError("output replay is not RAW-BIT stable")
            if not torch.equal(lse.view(torch.int32), variant.lse.cpu().view(torch.int32)):
                raise AssertionError("LSE replay is not RAW-BIT stable")
        variant.output_hash, variant.lse_hash = tensor_digest(output), tensor_digest(lse)
        print("[FA15 BN128 numeric] " + json.dumps({"config": variant.entry["name"],
              "dtype": dtype, "sequence": seq, "sampled_outputs": expected.numel(),
              "full_output_finite": True, "replay": "RAW-BIT/STABLE", "lse_max_abs": lse_error,
              "full_control_max_abs": full_control_max_abs, "output_sha256": variant.output_hash,
              **result}), flush=True)
        variant.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "both"), default="both")
    parser.add_argument("--profile", choices=[c["name"] for c in CONTRACT["configs"]])
    parser.add_argument("--warmup", type=int, default=CONTRACT["timing"]["warmup"])
    parser.add_argument("--samples", type=int, default=CONTRACT["timing"]["samples"])
    parser.add_argument("--launches", type=int, default=CONTRACT["timing"]["launches"])
    parser.add_argument("--peak-tflops", type=float)
    args = parser.parse_args()
    if min(args.samples, args.launches) < 1 or args.warmup < 0:
        parser.error("invalid timing counts")
    if args.profile and args.dtype == "both":
        parser.error("profile one exact dtype at a time")
    import torch
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    build = args.build.resolve()
    receipt = json.loads((build / "build.json").read_text())
    if receipt["link_status"] != "PASS":
        raise RuntimeError("compile-only receipt is not a runnable build; link on a complete PPU SDK")
    if receipt["source_files"]["dev/ppu15_bn128/contract.json"] != digest(HERE / "contract.json"):
        raise RuntimeError("contract changed since compilation")
    if receipt["target"] != CONTRACT["target"] or receipt["actlize"] != CONTRACT["actlize"]:
        raise RuntimeError("compiled target/dependency does not match contract")
    if [(c["id"], c["name"]) for c in receipt["configs"]] != [(c["id"], c["name"]) for c in CONTRACT["configs"]]:
        raise RuntimeError("build candidate denominator differs from registered three configs")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    output = {"build": receipt, "contract_sha256": digest(HERE / "contract.json"),
              "torch": torch.__version__, "device": str(props), "samples": [],
              "timing_scope": "warm-prepared-forward device events, no profiler", "default_changed": False,
              "dtype_scope": ["fp16", "bf16"] if args.dtype == "both" else [args.dtype], "status": "INCOMPLETE"}
    if not args.profile:
        if args.output.exists():
            raise RuntimeError(f"result already exists: {args.output}")
        print("[FA15 BN128 device] " + str(props), flush=True)
    for dtype in ("fp16", "bf16") if args.dtype == "both" else (args.dtype,):
        entries = [c for c in receipt["configs"] if not args.profile or c["name"] == args.profile]
        variants = [Variant(entry, build, dtype) for entry in entries]
        output["runtime_versions"] = variants[0].versions
        for variant in variants:
            print("[FA15 BN128 resources] " + json.dumps({"config": variant.entry["name"],
                  "dtype": dtype, **variant.resources}), flush=True)
        seq = CONTRACT["shape"]["sequence"]
        if args.profile:
            _, device = fixture(seq, dtype)
            variants[0].prepare(*device)
            torch.cuda.synchronize()
            variants[0].launch()  # exactly one attention launch; no timing/oracle launches
            torch.cuda.synchronize()
            variants[0].release()
            print("[FA15 BN128 profile] attention_launches=1; not a correctness/performance verdict")
            return 0
        for small_seq in (1, 65, 129):
            cpu, device = fixture(small_seq, dtype)
            validate(variants, cpu, device, dtype, small_seq)
        cpu, device = fixture(seq, dtype)
        validate(variants, cpu, device, dtype, seq)
        del cpu
        # All correctness gates close before the first timed call.
        for variant in variants:
            variant.prepare(*device)
            for _ in range(args.warmup):
                variant.launch()
        torch.cuda.synchronize()
        times = {v.entry["name"]: [] for v in variants}
        for sample in range(args.samples):
            order = variants[sample % len(variants):] + variants[:sample % len(variants)]
            if sample % 2:
                order = list(reversed(order))
            for variant in order:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.launches):
                    variant.launch()
                end.record(); end.synchronize()
                times[variant.entry["name"]].append(start.elapsed_time(end)*1000/args.launches)
        for variant in variants:
            samples = times[variant.entry["name"]]
            if tensor_digest(variant.o.cpu()) != variant.output_hash or tensor_digest(variant.lse.cpu()) != variant.lse_hash:
                raise AssertionError("post-timing output/LSE no longer matches admitted preflight")
            median = statistics.median(samples)
            tflops = 4*56*seq*seq*128 / (median*1e6)
            row = {"config": variant.entry["name"], "dtype": dtype, "median_us": median,
                   "samples_us": samples, "logical_tflops": tflops, "resources": variant.resources,
                   "output_sha256": variant.output_hash, "post_timing_raw": "PASS",
                   "speedup_vs_control": statistics.median(times["control"])/median,
                   "verdict": "CONTROL" if variant.entry["id"] == 0 else envelope(times["control"], samples)}
            if args.peak_tflops:
                row["mfu_percent"] = 100*tflops/args.peak_tflops
                row["peak_denominator_tflops"] = args.peak_tflops
            output["samples"].append(row)
            print("[FA15 BN128 result] " + json.dumps(row), flush=True)
            variant.release()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if len(output["samples"]) == len(output["dtype_scope"])*len(CONTRACT["configs"]):
            output["status"] = "PASS"
        args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"[FA15 BN128] PASS: {args.output}; shipping selector unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
