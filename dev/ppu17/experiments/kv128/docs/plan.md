# C07: original pipeline, KV tile128

Registered before implementation, 2026-09-29 03:10:27 UTC. Parent `454298e`.
The partial-overlap follow-up is closed; none of its source changes is active.
One geometry candidate, not a sweep and not a composition with forced overlap.

Workload: FP16 noncausal B1/Sq1024/Sk1024/H56/Hkv56/D128, contiguous BSHD,
40-SM model, one target call, CPU-only reference. Keep TM128,2stages,384threads,
static persistent grid40, QK/PV types and the original softmax/pipeline.
Only the noncausal D128 two-byte-element KV width changes176→128 under an
explicit experimental build flag. The standalone still admits only FP16.
Disable the flag: original generated machine code must match2/2. Causal must
remain unchanged. No default routing or wider-workload claim.

## Hypothesis and real costs

QK/P lane storage falls88+44→64+32words (36fewer live words before other
temporaries). Padded keys1056→1024, removing3.03% executed matrix FLOPs.
But KV steps6→8 and steady updates5→7 increase softmax/control iterations.
Do not claim occupancy increases from smaller shared storage alone. These
competing effects require full modeled cycles, not a static instruction win.

## Fixed gates

- Exact shipping generated FP16/D128 body, TMA reads/output and real ELF link.
  Fresh disabled control has unchanged two encoded bodies; opt-in causal same.
- Native/PTX noncausal body32m64n128k16 sites (16QK+16PV),66steady EX2 total.
  Existing completion waits present, no early writes to32old-P/64O words.
  No increased stack/spill vs control8/12/12B; PPU private traffic budget0.
- Actual host type/selector checks: only admitted geometry changes; planted
  wrong-width/missing-MMA/missing-exponent inputs must fail accounting.
- Single-run PPU:1kernel/40CTAs; waits3584,25088,25088,3584executions;
  total458752m64n128k16 warp-executions, i.e.30064771072matrix FLOPs.
  CPU FP64 O/LSE reference must pass unchanged atol/rtol0.002. KV grouping
  changes floating-point order, so cross-geometry RAW equality is diagnostic,
  not a substitute for the reference or a newly invented tolerance.
- Full compute cycles must beat247267 at identical model settings. Useful
  work/peak remain30064771072FLOPs/(40*4096FLOP/cycle);80%target229376cycles
  is aspirational. Record a loss as a loss; do not retune thresholds afterward.

No local PPU1.7 simulator/native toolchain is available. CUDA compilation and
host tests cannot certify PPU lowering, numerics or performance. No device jobs
or remote builds. Deliver one opt-in binary build/application command only if
local gates pass, retaining the original default and immutable control image.
