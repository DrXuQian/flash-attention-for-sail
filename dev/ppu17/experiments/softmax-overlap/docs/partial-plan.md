# C04 partial-overlap follow-up

Registered before edits, 2026-09-29 02:42:42 UTC. Source parent `fc4947d`;
performance incumbent remains the unchanged `4b26264` control (247267 cycles),
not the slower C03 (266011 cycles). Same FP16 noncausal B1/S1024/H56/D128,
40-SM simulation, 128x176 tile, two stages, one target invocation and CPU oracle.

## One change and bounded scope

C04 changes the volatile token dependency from XOR of both completed row sums
to just row0. It leaves the existing max/exp2/FP32 accumulation/conversion
expressions untouched. Row1 is free to be sunk past the existing PV wait0;
the expected recurring split is45 EX2 before /45 after (44 scores and one
rescale exponent per row). No new algorithm, tiling, precision or scheduler.
Keep C03 opt-in for the counterexample; C04 is a separate mutually exclusive
build option. Neither option becomes default.

The1KiB thread-private shared payload and one store/steady step remain real
costs. C04 removes the token XOR. It does not reuse in-flight P/O or release V
early. No barrier is added or removed. It may still lose or spill on PPU.

At most two narrowly scoped compile attempts in this follow-up. Start C04;
if its source order is not realized or local allocation regresses, reject it
and explicitly register the second attempt before making that edit. Stop after
one locally validated candidate handoff; do not request a sweep from the model.

## Fixed local and simulator gates

- Compile/link the actual generated FP16/D128 shipping TU and standalone.
  Default-off encoded bodies must match the old control2/2; ineligible causal
  must also stay unchanged. Existing fixed CPU FP64 tolerances are unchanged.
- Native CUDA control: exactly90 steady EX2 total, with45 before/45 after;
  matrix sites16QK+22PV fixed, no early scalar old-P/O writes. Reject added
  stack/spill resources against the original control, not against C03.
- PTX math inventory and repeated operation ordering per row unchanged.
  The token must depend only on row0 and no numerical result may read it.
- Independent host mapping/lifetime checks cover all256 math threads,2rows
  and44scores per row, not selected lanes. Plant an early P overwrite, missing
  exponent and full-row token to show the corresponding checks turn red.
- Actual PPU report: verify the intended partial schedule, complete math work,
  private-memory reads/writes against control0, and full cycles below247267.
  The report gate must refuse C03 despite its full overlap. Native NVIDIA
  resources do not establish this PPU condition; mark it pending locally.
- Simulator application must pass the existing full O/LSE CPU FP64 reference;
  retain source/binary/model/input bindings and parent fingerprints when
  available. One invocation/process, no warmup, repeats or GPU reference.

First performance target remains229376 cycles (80% useful MFU), not a predicted
result. Record failure even if EX2 ordering or one stall counter improves.

## C04 rejected; C05 registered before editing

C04's local native split is75/15, not the registered45/45. It is rejected for
this experiment despite unchanged168regs/8Bstack/12Bspill. The compiler can
still advance much of row1 because the source computes both rows together.

C05 explicitly computes one row's complete online update (max/rescale/exp/sum),
publishes the existing private token, waits for PV and releases V, then
computes row1. Reuse the existing `Softmax<1>` methods over a view of the real
accumulator, copying the two FP32 state scalars into/out of that row object.
Do not round/cast or re-associate any within-row sum. Cross-row order is not
an arithmetic dependency. Default/prologue/drain remain unchanged. There is
still only one existing wait0; old P/O are not modified early.

This is the second and final compile attempt in this cycle. Keep the45/45
native, unchanged matrix/math, no-new-spill and simulator gates above. If it
fails them, record the failure instead of widening the gate to fit the result.

## C05 rejected; separate dependency experiment C06

C05 passes the45/45 PTX condition but fails native CUDA:76/14. Math opcode
inventory, matrix count and12Bspill stay unchanged. This closes both planned
source-order attempts as rejected, not as simulation candidates. Splitting
source functions is insufficient to stop native scheduling across wait0.

Before requesting another simulation, register one additional, different
mechanism, C06: a real read-after-wait dependency. Save row1's previous FP32
maximum bit pattern in a second private shared word, then read it back AFTER
PV wait0 and use it as row1's original maximum. The existing row0 token still
forces row0 completion before wait0. Neither row's floating-point formula or
value changes; the bitcast roundtrip is identity. The new load, unlike mere
source order, supplies a dependency for row1's final maximum and exponents.

Explicit cost:2KiB private shared payload/CTA, two32-bit stores +one32-bit load
per math thread/steady step:4,587,520B writes +2,293,760B reads for this shape.
No new synchronization or global traffic is intended. All23C03private global
slots remain forbidden by the original control0budget. Require the native
load AFTER wait0,45/45 EX2, no early P/O writes, no added compiler spill,
default2/2 unchanged, identical math and complete model cycles below247267.
If this fails, end this follow-up with no runnable performance candidate.
The45/45 gate and numeric tolerances are not changed to admit76/14.
