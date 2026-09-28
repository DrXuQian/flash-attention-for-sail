# S1024 softmax / PV overlap

Registered before kernel edits, 2026-09-28 22:10 UTC. Parent: `4b26264`.
One bounded lowering/lifetime experiment; no tile, stages, scheduler, dtype,
rounding, mask, shared layout or public API changes. Default remains control.

## Authority and baseline

- B1/Sq1024/Sk1024/H56/Hkv56/D128, FP16, noncausal, contiguous BSHD.
- User's current model reports **40 SM**, not the historical 20-SM model.
  Same model/cache/frequency configuration for all comparisons; no clock guess.
- Uploaded `/root/perfstatistics.json`, SHA256
  `3cdd7babe0b51542f497b081350f3fab3b5899412fb31ad369fe537ea3254557`.
  Kernel cycles 247267; actual input binary/model hashes were not uploaded.
  Retain that provenance limitation; local parent builds are compile controls.
- Effective QK+PV work 30064771072 FLOPs; executed padded work 31004295168.
  User-supplied tensor ceiling 55.556%; corrected tensor utilization 76.53%,
  useful MFU 74.21%. No report occupancy/peak percentage is accepted blindly.
- Local CUDA12.8 + PPU CUTLASS4.3 parent: 90 PTX EX2 between steady wait1/wait0,
  19 NVIDIA SASS EX2 in that interval. Uploaded PPU code: zero there, all 90
  after wait0; that wait averages 205.40 stall cycles per warp execution.
  These are separate codegen layers, not a bound on recoverable kernel time.

## Bounded candidates and gates

Try at most three small source dependency/lifetime variants, one at a time.
Start with CuTe operand fences on completed softmax values before wait0 and
old P liveness across its completion boundary. If inert, record and reject it;
do not ship empty assembly as a proven optimization. Any later mechanism must
be recorded before compiling it. Keep old P unmodified while asynchronous PV
may read it, O unmodified until PV completion, and V unreleased until wait0.

Local acceptance: exact generated FP16/D128 body links; baseline body stays
bit-identical when disabled; selected geometry/math/barrier obligations fixed;
PTX/native postconditions locate the recurring wait pair and verify improved
softmax overlap, not aggregate instruction counts. No new stack/spills or
premature per-MMA serialization. Actual PPU schedule remains unverified until
the candidate simulator report arrives. Native NVIDIA code is only a control.

Negatives: retained old schedule fails the improved-overlap gate; an omitted
completion wait fails the lifetime gate; an early old-P overwrite fails the
operand-lifetime gate. Test the entire denominator, not a selected first PC.

Numerics: unchanged existing full CPU FP64 O/LSE contract (atol=rtol=0.002),
nonfinite rejection; same fixture hashes. Delivery-only candidate additionally
compares parent O/LSE fingerprints where available. One target invocation per
fresh simulation process, zero warmup/replay/auxiliary GPU references.

Performance: compare complete kernel modeled cycles at the same model inputs;
lower stalls alone cannot pass. First target 229376 cycles (80% useful MFU),
not a prediction. Candidate has no measured performance until model execution.
Retain failed compiler candidates; stop this cycle after a verified local
candidate and a directly runnable one-invocation handoff, or a concrete blocker.

## Deliverables

Opt-in source variant + unchanged default, same standalone build/run workflow,
local codegen/resources/negative-control evidence, report schedule checker,
and explicit simulator numerical/performance NOT_RUN status. No GPU job here.

## Compile checkpoint / C02 registration

C01 is inert: local native overlap remains 19/90 EX2. The existing parent
has 8B stack / 12B spill stores / 12B spill loads; the original no-NEW-spill
criterion applies, not an invented zero-spill baseline. C02 adds a private
2KiB shared row-sum buffer, with two float volatile stores before wait0 and
two matching loads after it per consumer lane/steady step. This is real added
traffic, not an empty compiler barrier: 4,587,520B written and as many read
for this shape. Each lane owns its two cells; no CTA publication barrier or
new numerical rounding. Default and other specializations keep old storage.
If the compiler still moves the math/wait, or this adds spill, reject it.

C02 produced all90 EX2 before wait0 and no premature P/O writes, but increased
stack8->24B and spill stores/loads12->36B, so it is rejected. Last candidate
C03 publishes one XOR of the two row-sum bit patterns to a thread-private
1KiB buffer. Both completed row sums are dependencies, without changing either
value or performing an extra floating-point operation. The stored token is
intentionally not consumed: the volatile publication is a real compiler-order
anchor, not a correctness flag. Cost: one integer XOR + one shared store per
lane/steady iteration, 2,293,760B shared writes for the shape, no reads, no new
barriers. Compare same native, resource and numerical gates; no fourth variant
in this cycle. Actual PPU lowering and performance require the user's model.
