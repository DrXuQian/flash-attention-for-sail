# H800 causal performance control: preregistered before timing

2026-09-26 UTC. User-authorized physical H800 timing, not the PPU1.7 simulator.
No kernel/configuration edits or sweep. Reuse the 22/22-admitted binary
`a78aff6c443238cd888f53dc692bf1e017b9191cff137427677ff1da84640320`,
source `e7ee864`, backend `023e82d`.

Workload: BF16 fixed BSHD, B1/S2048/Hq32/Hkv2/D256, causal; seed170020,
same CPU-prepared fixture and CPU FP64 O/LSE oracle as the correctness control.
Cache protocol: repeated prepared inputs, no flush. Only run when other GPU
and validation/build task parents are absent; recheck between phases.
No frequency, power, partition or system-package changes.

Numerator counts QK and PV only, two FLOPs per MAC, only valid causal pairs:
`4 * B * Hq * D * S*(S+1)/2 = 68,753,031,168 FLOP`.
Do not count masked/padded upper-triangle work or halve the numerator twice.

The H800 80GB PCIe BF16 rating is1513 TFLOPS **with structural sparsity**:
[OEM product specification, table2 and footnote](https://lenovopress.lenovo.com/lp1814-thinksystem-nvidia-h800-pcie-gen5-gpu).
Use1513/2 = **756.5 TFLOPS dense** as the fixed nameplate denominator, not
FP8 throughput, the PPU peak, or the idle clock. Runtime SM count and clock
snapshots are measured separately, not borrowed from another row of that table.
The 70% target is therefore **129.832935829 us** or less.

Two sequential measurements, never concurrent:

1. Uninstrumented CUDA-event aggregate around complete forward calls:
   200 warmups, 9 samples of50 calls. Includes public API counter setup/copy
   and any device-idle gaps while the CPU submits work. Excludes QKV creation,
   initial H2D and CPU reference. This is a call-span bound, not kernel-only time.
2. Separate warmed Torch/CUPTI activity trace: 200 warmups, 450 calls;
   group consecutive kernel durations into9 samples of50. Require exactly450
   real `FlashAttnFwdSm90` kernel events and no extra GPU kernels. This isolates
   device execution but remains explicitly instrumented. Never relabel it
   uninstrumented latency. Export the trace, including host/copy gaps.

Record both raw sample sets, medians/ranges, useful TFLOPS and MFU. A phase is
`AT_LEAST_70` only if its entire sample range is within129.832935829us;
`BELOW_70` if its entire range is above; otherwise `UNRESOLVED_AT_70`.
Do not move this threshold after seeing results. Recheck O/LSE and the admitted
output fingerprint after timing. No GPU reference. Correctness failures void
timing interpretation. Keep unsuccessful results.

This establishes an H800 control only (114SM/50MiB L2), not70% on the
20SM/32MiB PPU1.7 model. Trace and raw timing artifacts stay under the explicit
`/workspace/flash-attn-ppu17-perf-20260926` artifact path; summarize and hash-bind
them in the source checkout.
