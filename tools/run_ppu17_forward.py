#!/usr/bin/env python3
"""One FA3 forward launch for the PPU1.7 performance model (not ACU).

Creates inputs on CPU and copies them to the target. No extra warmup or
reference GPU kernels are launched. Optional correctness uses a row-blocked
CPU float64 reference, including for the full performance shape.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys


def logical_flops(batch, seqlen, heads, dim, causal):
    pairs = seqlen * (seqlen + 1) // 2 if causal else seqlen * seqlen
    return 4 * batch * heads * dim * pairs


def validate_extension(extension, expected_cutlass=None):
    versions = {"cutlass36-sm90-forward-v1": "3.6.0", "cutlass43-sm90-forward-v1": "4.3.0"}
    identity = getattr(extension, "ppu17_backend", None)
    if identity not in versions:
        raise RuntimeError("wrong/stale FA extension: missing PPU1.7 forward build identity")
    if expected_cutlass is not None and versions[identity] != expected_cutlass:
        raise RuntimeError(f"wrong CUTLASS backend: expected {expected_cutlass}, loaded {versions[identity]}")


def invoke_once(interface, q, k, v, *, causal):
    # One invocation only, no retry, timing loop, autograd or reference here.
    return interface._flash_attn_forward(q, k, v, causal=causal, num_splits=1, pack_gqa=False)


def device_evidence(prop, *, hardware_validation):
    if hardware_validation:
        cache_bytes = getattr(prop, "L2_cache_size", None)
        return {
            "role": "Hopper-hardware-validation",
            "target_cache_bytes": cache_bytes,
            "cache_source": "cuda-device-properties" if cache_bytes else "UNAVAILABLE",
            "evidence_scope": "shared-SM90-forward; NOT native-PPU1.7 or model-performance",
        }
    return {
        "role": "PPU1.7-simulation-input",
        "target_cache_bytes": 32 * 1024 * 1024,
        "cache_source": "user-specified-simulation-model",
    }


def cpu_reference(host, *, causal, query_block=128):
    import torch
    if query_block <= 0 or len(host) != 3 or any(t.device.type != "cpu" for t in host):
        raise ValueError("reference requires three CPU tensors and a positive query block")
    q, k, v = [t.double() for t in host]
    batch, seqlen, heads, dim = q.shape
    if k.shape != v.shape or k.shape[:2] != q.shape[:2] or k.shape[3] != dim or heads % k.shape[2]:
        raise ValueError("reference scope is equal-length, equal-dimension GQA")
    ratio = heads // k.shape[2]
    out = torch.empty(q.shape, dtype=torch.float64, device="cpu")
    lse = torch.empty((batch, heads, seqlen), dtype=torch.float64, device="cpu")
    cols = torch.arange(seqlen, device="cpu")
    for b in range(batch):
        for h in range(heads):
            kh, vh = k[b, :, h // ratio], v[b, :, h // ratio]
            for start in range(0, seqlen, query_block):
                end = min(start + query_block, seqlen)
                score = (q[b, start:end, h] @ kh.T) / (dim ** 0.5)
                if causal:
                    rows = torch.arange(start, end, device="cpu")
                    score.masked_fill_(cols[None, :] > rows[:, None], float("-inf"))
                out[b, start:end, h] = score.softmax(-1) @ vh
                lse[b, h, start:end] = score.logsumexp(-1)
    return out, lse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension-dir", type=Path)
    parser.add_argument("--expected-cutlass", choices=("3.6.0", "4.3.0"),
                        help="reject an installed extension built against the other admitted backend")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, choices=(64, 128, 256), default=256)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--noncausal", action="store_true")
    parser.add_argument("--expected-sms", type=int, default=20,
                        help="0 disables the device-count assertion; does not restrict hardware")
    parser.add_argument("--hardware-validation", action="store_true",
                        help="label physical Hopper correctness separately from PPU1.7 simulation")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch, args.seqlen, args.heads, args.kv_heads) <= 0 or args.heads % args.kv_heads:
        parser.error("positive dimensions and Hq divisible by Hkv required")
    if args.extension_dir:
        sys.path.insert(0, str(args.extension_dir.resolve()))
    import torch
    extension = importlib.import_module("flash_attn_3._C")
    validate_extension(extension, args.expected_cutlass)
    # The package interface is kept separate from FA2; never silently fall back.
    interface = importlib.import_module("flash_attn_3.flash_attn_interface")
    prop = torch.cuda.get_device_properties(0)
    if (prop.major, prop.minor) != (9, 0):
        raise RuntimeError(f"expected SM90-compatible runtime, got {prop.major}.{prop.minor}")
    if args.expected_sms and prop.multi_processor_count != args.expected_sms:
        raise RuntimeError(f"expected {args.expected_sms} SMs, got {prop.multi_processor_count}")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    generator = torch.Generator(device="cpu").manual_seed(170020)
    shape_q = (args.batch, args.seqlen, args.heads, args.head_dim)
    shape_kv = (args.batch, args.seqlen, args.kv_heads, args.head_dim)
    host = [torch.randn(shape, generator=generator, device="cpu").to(dtype) for shape in (shape_q, shape_kv, shape_kv)]
    q, k, v = [tensor.to("cuda") for tensor in host]
    torch.cuda.synchronize()
    record = {
        **device_evidence(prop, hardware_validation=args.hardware_validation),
        "shape": list(shape_q), "kv_heads": args.kv_heads,
        "dtype": args.dtype, "causal": not args.noncausal, "sm_count": prop.multi_processor_count,
        "device_name": prop.name, "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
        "input_seed": 170020, "input_strides": [list(t.stride()) for t in host],
        "input_sha256": [hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                         for t in host],
        "logical_flops": logical_flops(args.batch, args.seqlen, args.heads, args.head_dim, not args.noncausal),
        "attention_launches": 1, "python_elapsed_time": "NOT_USED_AS_SIMULATION_TIME",
        "extension": extension.__file__,
        "backend_identity": extension.ppu17_backend,
        "extension_sha256": hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
        "numerics": "NOT_CHECKED" if not args.verify else "PENDING",
    }
    print("[PPU1.7 forward config] " + json.dumps(record), flush=True)
    with torch.inference_mode():
        out, lse, _, _ = invoke_once(interface, q, k, v, causal=not args.noncausal)
    torch.cuda.synchronize()
    out_host, lse_host = out.cpu(), lse.cpu()
    if not torch.isfinite(out_host).all() or not torch.isfinite(lse_host).all():
        raise AssertionError("nonfinite forward output/LSE")
    record["output_sha256"] = hashlib.sha256(out_host.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    if args.verify:
        expected, expected_lse = cpu_reference(host, causal=not args.noncausal)
        # Fixed before device results: matches the BF16/FP16 rounding contract,
        # not a RAW-BIT claim about real-valued softmax.
        atol = 2e-2 if args.dtype == "bf16" else 2e-3
        torch.testing.assert_close(out_host.double(), expected, atol=atol, rtol=atol)
        torch.testing.assert_close(lse_host.double(), expected_lse, atol=2e-3, rtol=2e-3)
        record.update(numerics="CPU-FLOAT64-REFERENCE/PASS", max_abs=(out_host.double() - expected).abs().max().item())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print("[PPU1.7 forward] " + json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
