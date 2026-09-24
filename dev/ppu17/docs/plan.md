# PPU1.7 Hopper source integration

Parent: `f600591`; backend: `/root/cutlass3-3.6.0`, commit
`023e82d03e80b4d5982f664925e15499033df314` (PPU CUTLASS 3.6.0).

## Contract

Reuse the existing Hopper forward mainloop, WGMMA QK/PV, online softmax,
TMA operands and output. Do not change the attention mathematics or the
admitted PPU1.0/1.5 AIU implementation. Select the new backend explicitly;
do not change the default Python FA2 route or publish a replacement binary.

First integration scope: FP16/BF16 forward, equal Q/K/V head dimensions
64/128/256, fixed-length batches, causal/noncausal, tails and GQA (without
packed-GQA optimization). Cluster launch, backward, FP8, paged/append KV,
varlen, split, softcap, local attention and auxiliary sinks are not admitted.
Unsupported calls must fail before launch, never silently ignore an argument.

## Validation before handoff

1. Pure-host build-plan/dispatch tests, including wrong SDK, wrong dependency,
   forbidden legacy macros, missing generated units and unsupported features.
2. Compile the actual generated forward units against the selected PPU
   CUTLASS headers with local CUDA `sm_90a`; require nonempty WGMMA and TMA
   load/store bodies. This is **source compatibility**, not PPU native proof.
3. Compile/link the host dispatcher when local Torch headers are available;
   ensure every referenced specialization has a generated definition.
4. Verify the old PPU branch remains selected without the new opt-in and
   that code excluded from the new backend remains unchanged.
5. Native PPU1.7 compile and device numerics/performance remain NOT RUN until
   a capable toolchain/device is available. SDK2.1.1 only exposes PPU1.0/1.5;
   its CUDA driver can silently map `sm_90a` to PPU1.0. Reject that behavior.

No latency/MFU claim is made. Scope is wiring plus bounded local verification,
not tuning. Work stops at a reproducible source handoff, with unverified
hardware boundaries explicitly listed rather than held behind a box run.
