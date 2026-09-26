# H800 causal length scaling: higher at8K, not monotonically higher

Completed2026-09-26 UTC. Fixed **B1/Hq32/Hkv2/D256, BF16, causal**;
num_splits=1, pack_gqa=False. Physical H800 PCIe114SM/50MiB L2,
UUID `GPU-1d5fdef3-4899-79d9-19e6-c9c815b2a59c`. This is not native
PPU1.7 or the20-SM/32MiB simulator result.

## Results

Each new length ran control/official/official/control, separate processes,
200 warmups and9x50 calls per run. Pool all18 aggregate samples per arm,
not just the fastest run. Reused prepared inputs, not cold-cache timing.
Both immutable binaries and C++ op kwargs are unchanged from the
[official control](h800-official-ab.md).

| Sequence | Our kernel median | Our causal MFU | Official FA3 kernel median | Official MFU |
|---|---:|---:|---:|---:|
| 2048, previous control |161.030us|56.44%|164.234us|55.34%|
| 4096 |601.988us|60.37%|614.033us|59.19%|
| 8192 |2262.173us|64.26%|2227.810us|65.25%|
| 16384 |9300.694us|62.51%|9353.025us|62.16%|

These are explicitly **CUPTI-instrumented attention-kernel-only** durations.
Useful FLOPs=4*B*Hq*D*S*(S+1)/2; diagonal included, upper triangle excluded.
Dense BF16 peak stays756.5TF/s. No masked work or sparse peak substitution.

| Sequence | Our full-call event median | Official full-call event median |
|---|---:|---:|
| 2048, previous control |177.806us|173.629us|
| 4096 |625.862us|610.805us|
| 8192 |2408.788us|2403.074us|
| 16384 |9489.452us|9455.866us|

Full-call events include API counter setup and submission gaps. Kernel-only
timing uses a separate trace phase; subtracting these medians does not measure
API cost. Port/reference sample envelopes overlap at **every length and in
both roles: all relative verdicts UNRESOLVED**. No resolved port penalty is
not proof of exact speed equivalence or an optimized performance ceiling.

Median MFU increases through8K, then falls at16K in both implementations.
No length establishes stable70%. The4K port has one aggregate sample518.733us
just below the519.205us threshold, so its registered range verdict is
UNRESOLVED_AT_70; all other cells are BELOW_70. Do not hide that exception.
Criteria: [h800-sequence-scaling-plan.md](h800-sequence-scaling-plan.md).

## Work structure and operating conditions

Every actual launch retains tile128x80x256,2 stages, grid114,384 threads,
168 registers/thread,232448B shared memory. No kernel/tile/build/clock/power/
partition change. The2K row is previous-run context, not concurrently retimed.
New CPU references use8 threads outside GPU timing. Power/thermal state is
not artificially fixed.

The actual bound in `hopper/block.h` gives these static work facts. The final
column is masked/padded rectangle work, **not a fraction of wall time**:

| S | Query work tiles | Mean KV iterations/tile | Masked/padded rectangle fraction |
|---|---:|---:|---:|
| 2048 |512|14|8.53%|
| 4096 |1024|26.8125|4.50%|
| 8192 |2048|52.40625|2.29%|
| 16384 |4096|103.6015625|1.15%|

Longer loops can amortize setup/drain and reduce triangular tile waste, but
this test does not isolate either term's latency. The scheduler is already
dynamic and LPT-ordered;512/114 is not a naive equal-work tail-wave model.

## Direct observation of power limiting

Passive200ms NVML samples were joined to the attention trace's wall-clock
window, not taken only after kernels ended. Trace absolute time is
`baseTimeNanoseconds + ts*1000`. NVML timestamps are remote-host CST/UTC+08:00,
verified with `date`, and converted to UTC before joining. Samples outside
these windows, including other jobs and CPU-reference phases, are excluded.

| Attention trace | Samples in window | SW Power Cap active | Clock while cap active |
|---|---:|---:|---:|
| 8K official B2 |5|5|990–1140MHz|
| 8K port A2 |5|4|1110–1155MHz|
| 16K official B1 |21|20|975–1110MHz|
| 16K official B2 |21|20|975–1125MHz|

Default power limit=350W. The16K window includes samples at approximately
350W,1020–1050MHz with `SW Power Cap=Active`. This directly establishes power
limiting during the benchmark, but is **not** a cycle-weighted frequency,
percentage of runtime throttled, or proof that power alone explains all
missing nominal MFU. Do not renormalize the fixed denominator. No setting was
changed; a different power/clock experiment needs separate authorization.

Conclusion: larger S helps to a point, not automatically to70%. Both code
paths share the measured trend. Further diagnosis must distinguish kernel
delivery/scheduling from this operating envelope, rather than treating all
missing nominal MFU as a porting defect.

## Correctness, interruptions and identities

All12 new runs pass full CPU FP64 O/LSE before/after timing, finite checks and
within-arm raw replay. No sampled or GPU reference. At each length, cross-arm
output AND LSE hashes also match (observed, not required). Max output errors
at4K/8K/16K:0.008851649/0.008259462/0.008222517; max LSE error2.891649e-6.

Two8K attempts and the first16K attempt were refused before first launch
because new foreign tasks arrived during CPU reference. No timing retained.
A bounded prelaunch wait then retained the CPU oracle;16K official B1 waited
before launching. During/after timing BUSY still fails immediately. Other
tasks were not killed/reconfigured; GPU empty after the final arm.

Harness19005e9 for4K/8K; b7d5192 for16K adds only prelaunch waiting, no timing
or accuracy changes. Port binary SHA256:
`a78aff6c443238cd888f53dc692bf1e017b9191cff137427677ff1da84640320`.
Official FA3 binary SHA256:
`c0611358efe2a1511843ef2638befce6932b1d5855225c7d7d89ebf8583ef585`.
Source/backend identities remain in the original official-control document.

[Machine-readable results](../results/h800-sequence-scaling-20260926/summary.json)
contain all aggregate samples, input/output/LSE hashes, original result/trace
hashes, resources and time-joined clock evidence. Raw artifacts on both hosts:
`/workspace/flash-attn-h800-sequence-scaling-20260926`; original2K artifacts:
`/workspace/flash-attn-official-h800-ab-20260926`.

Local checks:21 source +4 timing +8 A/B/admission +3 CPU-reference =36 PASS.
New negative controls reject wrong sequence/type, same-shape input drift and
busy prelaunch admission; bounded waiting must recheck, not ignore BUSY.
