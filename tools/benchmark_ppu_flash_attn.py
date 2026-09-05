#!/usr/bin/env python3
"""Time one fixed-shape FlashAttention forward invocation on a PPU."""

from __future__ import annotations

import argparse
import hashlib
import os
import statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=73774)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--peak-tflops", type=float, default=500.0)
    parser.add_argument("--causal", action="store_true")
    return parser.parse_args()


def sampled_fingerprint(tensor) -> str:
    # This is a replay-stability witness, not a full numerical oracle.  Keep the
    # transfer small even for the 1 GiB output of the registered shape.
    seq_step = max(1, tensor.shape[1] // 17)
    head_step = max(1, tensor.shape[2] // 7)
    sample = tensor[:, ::seq_step, ::head_step, : min(8, tensor.shape[3])]
    payload = sample.float().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def main() -> int:
    args = parse_args()

    import torch
    import flash_attn_2_cuda
    from flash_attn import flash_attn_func

    if not torch.cuda.is_available():
        raise RuntimeError("PPU CUDA-compatible device is not visible")
    if args.warmup < 1 or args.samples < 1:
        raise ValueError("warmup and samples must both be positive")

    torch.cuda.set_device(0)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    shape = (args.batch, args.seqlen, args.heads, args.head_dim)
    element_bytes = torch.empty((), dtype=torch.bfloat16).element_size()
    tensor_bytes = args.batch * args.seqlen * args.heads * args.head_dim * element_bytes
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)

    print(
        "[PPU FA config] "
        f"shape=B{args.batch},Sq{args.seqlen},Sk{args.seqlen},"
        f"Hq{args.heads},Hkv{args.heads},D{args.head_dim} "
        f"dtype=bf16 causal={int(args.causal)} custom_mask=0 dropout=0 "
        f"warmup={args.warmup} samples={args.samples}"
    )
    print(
        "[PPU FA device] "
        f"name={torch.cuda.get_device_name(0)!r} torch={torch.__version__} "
        f"runtime={torch.version.cuda!r} free_bytes={free_bytes} "
        f"total_bytes={total_bytes} one_tensor_bytes={tensor_bytes} "
        f"extension={flash_attn_2_cuda.__file__}"
    )

    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        for _ in range(args.warmup):
            out = flash_attn_func(q, k, v, dropout_p=0.0, causal=args.causal)
        torch.cuda.synchronize()

        times_ms: list[float] = []
        fingerprints: list[str] = []
        for _ in range(args.samples):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = flash_attn_func(q, k, v, dropout_p=0.0, causal=args.causal)
            end.record()
            end.synchronize()
            times_ms.append(float(start.elapsed_time(end)))
            fingerprints.append(sampled_fingerprint(out))

        output_finite = bool(torch.isfinite(out).all().item())

    median_ms = statistics.median(times_ms)
    mean_ms = statistics.fmean(times_ms)
    if args.causal:
        attended_pairs = args.seqlen * (args.seqlen + 1) // 2
    else:
        attended_pairs = args.seqlen * args.seqlen
    logical_flops = 4.0 * args.batch * args.heads * attended_pairs * args.head_dim
    logical_tflops = logical_flops / (median_ms * 1.0e9)
    logical_mfu = logical_tflops / args.peak_tflops * 100.0
    fingerprint_state = "STABLE" if len(set(fingerprints)) == 1 else "UNSTABLE"

    print("[PPU FA samples_ms] " + ",".join(f"{value:.3f}" for value in times_ms))
    print(
        "[PPU FA result] "
        f"median_ms={median_ms:.3f} mean_ms={mean_ms:.3f} "
        f"range=[{min(times_ms):.3f},{max(times_ms):.3f}]_ms "
        f"logical_tflops={logical_tflops:.3f} "
        f"logical_mfu={logical_mfu:.2f}% peak_denominator={args.peak_tflops:.3f}_TFLOPS "
        f"output_finite={int(output_finite)} "
        f"sampled_replay={fingerprint_state} fingerprint={fingerprints[-1]}"
    )

    if not output_finite or fingerprint_state != "STABLE":
        raise RuntimeError("output finite/replay-stability admission failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
