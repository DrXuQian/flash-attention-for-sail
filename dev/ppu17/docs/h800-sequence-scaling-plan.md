# H800 causal sequence-length scaling (registered before measurement)

User request, 2026-09-26: determine whether larger shapes reach higher useful
MFU. Change only S from2048 to4096/8192/16384; B1/Hq32/Hkv2/D256, BF16,
causal, num_splits=1 and pack_gqa=False remain fixed. No kernel edits or new
builds. This is physical H800 evidence, not the20-SM PPU1.7 model.

Use the exact two binaries, source/backend identities, CPU seed170020,
unchanged full CPU FP64 O/LSE oracle and tolerances, and idle-only rules in
[the official A/B plan](h800-official-ab-plan.md). For each new length run
control/official/official/control in separate serial processes. Bind the
latter three to the first control's input hashes and exact shape. Validate
all output elements and LSE, before and after timing, with within-arm replay.
No sampled numerical reference or GPU reference. The admitted S2048 result
is retained as a previous-run context, not a fresh simultaneous measurement.

Timing stays200 warmups,9 samples x50 calls. Report uninstrumented full-call
event spans and separately instrumented CUPTI attention-kernel durations.
Inventory every auxiliary GPU operation. Preserve raw samples, traces,
resource metadata and snapshots. No clocks/power/partition changes. Other
task parents count as busy even when utilization momentarily reads zero.

| S | Useful causal FLOPs | Latency at70% of756.5 dense BF16 TF/s |
|---|---:|---:|
| 2048 (previous run) | 68,753,031,168 | 129.832936 us |
| 4096 | 274,945,015,808 | 519.205015 us |
| 8192 | 1,099,645,845,504 | 2076.566605 us |
| 16384 | 4,398,314,946,560 | 8305.759506 us |

Formula:4*B*Hq*D*S*(S+1)/2; GQA does not reduce the Hq arithmetic. Repeated
prepared inputs are warm/reused, not a cold-cache claim. Distinct BF16 K+V
sizes8/16/32MiB are recorded context, not a cache residency guarantee.

Per-arm pooled sample range wholly at/below the fixed70% latency means
AT_LEAST_70; wholly above means BELOW_70; crossing means UNRESOLVED_AT_70.
Compare port/reference only with disjoint pooled ranges, otherwise UNRESOLVED.
The shape trend is reported regardless of whether it increases or decreases;
no dropping losing lengths, redefining the causal denominator or changing the
peak. Any source-selected tile/resource change must be printed, not mistaken
for constant geometry. Higher MFU at larger S would establish workload
dependence, not prove which startup/masking/scheduling term caused it or that
the smaller shape has no headroom.

Evidence root: /workspace/flash-attn-h800-sequence-scaling-20260926 (both hosts).
