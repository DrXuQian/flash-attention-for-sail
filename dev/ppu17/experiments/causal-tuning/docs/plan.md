# H800 causal tuning preregistration — 2026-09-26

This is an isolated experiment, not a default selector change. Parent ed150c9;
port backend 023e82d (CUTLASS3.6); branch ppu17-causal-tuning. Work is bounded
to the table below, then correctness, confirmation and handoff. Target deadline
is 2026-09-26 10:00 UTC; report missing cells rather than inventing results.

## Workload and reference contract

Priority: B1/S2048/Hq32/Hkv2/D256 BF16 causal. Generalization: S8192 only.
CPU seed170020, BSHD inputs, softmax scale1/16, no split, no packed GQA,
no quantization, no physical replication of KV heads. Full CPU FP64 O/LSE
oracle: O atol=rtol=.02; LSE atol=rtol=.002. Within-arm raw replay must be
stable. Tile changes need not preserve cross-arm raw bits, but all candidates
must pass the unchanged oracle. Cache of the CPU oracle is allowed only with
input, oracle-source and reference-tensor hashes bound in every result.

Use the immutable port extension a78aff6c443238cd888f53dc692bf1e017b9191cff137427677ff1da84640320
as incumbent. Official FA3 c0611358efe2a1511843ef2638befce6932b1d5855225c7d7d89ebf8583ef585
is the already measured same-input reference. Add explicit FlashInfer fa3
single-prefill and cuDNN SDPA references. Explicit backend only; a capability
failure is SKIP with reason, not a fallback timing. Bind the actual trace
kernel, library/source hashes, runtime and device. List auxiliary work.

## Bounded candidate inventory

| ID | Tile | Scheduler | Hypothesis |
|---|---|---|---|
| N80-LPT |128x80|existing dynamic LPT|immutable incumbent|
| N64-LPT |128x64|existing dynamic LPT|less diagonal padding / shared state, more K iterations|
| N80-single |128x80|existing SingleTileScheduler|no task counter, but different ordering and more prologues|
| N64-single |128x64|existing SingleTileScheduler|combined geometry; matches FlashInfer's high-level axes|

Keep M128, stages2, two math warpgroups, producer warpgroup, BF16/FP32 math
and causal masking unchanged. Single-tile must remove unused host counter
initialization too; host/kernel use one shared eligibility predicate. Otherwise
complete-call overhead would not represent the scheduler being tested. This
axis also changes order/LPT, so it is NOT an isolated atomic-cost measurement.
No new cluster, split, mixed precision or launch-resource redesign.

## Measurement and decision

Physical H800 PCIe only, measured114SM/50MiB L2, default350W envelope.
Do not change clocks/power/MIG or run with foreign tasks. Native PPU1.7 and
20SM/32MiB simulator performance remain unmeasured. Repeated prepared inputs,
not a cold-cache claim. Fixed nominal dense BF16 peak756.5TF/s. Useful causal
FLOPs = 4*B*Hq*D*S*(S+1)/2, never rehalve the denominator or clock-adjust it.

Each arm:200 warmup,9x50 calls. Complete-call CUDA events without profiler;
separate CUPTI kernel durations grouped9x50. Parse exact inventory and keep
helper kernels/copies visible. Passive power samples may explain variability,
but do not alter the denominator. Correctness and idle checks before/after.
Screen each bounded cell, then confirm incumbent/finalist in reverse order.
Only disjoint sample envelopes establish a relative winner; overlap is
UNRESOLVED. Report kernel-only AND complete-call conclusions independently.
70% MFU remains the target, not guaranteed by a smaller tile.

## Local contracts / negatives

Compile the shared constexpr tile/scheduler policy in all four configurations;
verify default and non-target shapes unchanged, host counter and kernel policy
agree, and per-head output coverage is exact-once. Plant wrong tile, a missing
head and a host-counter mismatch: each must fail. Compile/link actual D256
BF16 generated units with real SM90a ptxas; report spills/resources separately
from measured speed. Narrow experimental binaries reject unsupported scope;
the shipping six-unit build manifest is not weakened.

Validation: tests/test_h800_causal_tuning.py plus existing PPU17 source,
hardware timing, official A/B and CPU-reference tests. Never remove losing
cells, loosen oracle/idle gates or promote an experimental winner silently.
