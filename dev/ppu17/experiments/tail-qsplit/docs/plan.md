# Tail Q split: separate from software exp

Registered 2026-09-29 04:49:34 UTC, parent `d19d27a`, branch
`ppu17-tail-qsplit`. Do not compose with exp emulation or C03 forced overlap.
Keep `hopper/softmax.h`, all exponent math, K order, dtype, input, tolerances
and KV128 fixed. No default promotion and no wider-shape admission.

## Workload and baseline

B1/Sq1024/Sk1024/H56/Hkv56/D128, FP16/noncausal/BSHD, 40-SM model,
TM128/TN128/two stages, one invocation and CPU FP64 O/LSE reference only.
C07 user-reported cycles240412 (76.327638% useful MFU). Its full report,
numerical stdout and binary/model binding are NOT_PROVIDED; do not borrow
the old C03 report under `/root/perfstatistics.json` (266011cycles).
Keep the original C07 ELF under `/workspace/fa17-kv128-20260929/candidate`.

## Bounded candidates

T01: compile the real generated FP16/D128 entrypoint with uniform TM64.
It is a geometry/resource/ownership preflight, not a presumed winner.
T02: keep440full TM128 tasks (11per CTA); replace the final8full tasks with
16disjoint TM64 tasks assigned to16CTAs. Reuse real existing collectives.
No second launch, workspace reducer, atomics, global barrier or waiting for
other CTAs. One local phase boundary must drain original TMA/GMMA/store work
before shared storage reuse and initialize the half-tile pipeline correctly.
Register redistribution, active-thread initialization and barrier lifetimes
must be explicit. Never implement TM64 by masking half a TM128 computation.

Zero-overhead equal-cost model:12t->11.5t (-4.1667% time), not a prediction.
K/V tile-fill count448->456 (+1.7857%); this is not measured HBM traffic.
Uniform TM64 is896tasks/23waves and doubles K/V task-level fills; do not
promote it just because tail fill improves.

## Admission fixed before results

1. Full `(head, q_row, output_col)` exact-once; tail half ownership independent
   of exp and K partitioning. Enumerate actual CuTe TM128/TM64 accumulator
   coordinates against an independent SM90 lane/warp formula. Denominator is
   57344rows/7340032output elements. No missing/duplicate tail outputs.
2. Real generated unit compiles, assembles and links. TMA/GMMA/output bodies
   present. T02 main-loop resource footprint must not introduce CUDA spills
   above C07's12/12B, or C751 asynchronous-serialization warnings. All
   resources/static-body deltas recorded; native PPU still requires report.
3. Default and causal encoded bodies remain unchanged. Softmax source unchanged.
   One invocation, no GPU reference/helper/initialization body. Output CPU
   poison detects omissions; full FP64 O/LSE atol/rtol0.002 unchanged. Parent
   RAW O/LSE comparison is desired for this delivery-only change when both
   outputs are available, not invented from host coverage.
4. Model gate: same40SM settings, full compute cycles<240412, no new private
   spill traffic. Preserve numerical/binary/model evidence. Same math work,
   no phantom padded rows. Report tail duration and occupied CTAs as secondary
   observations, not replacements for full cycles.

## Required negative controls

- Duplicate one half or omit the final half: exact output denominator red.
- Give a half the wrong head/row origin: independent coordinate anchor red.
- Wrong init/consumer arrival count or overlapping live pipeline storage:
  actual type/lifetime contract red.
- Compose software/forced exp change, wrong geometry/SM count, or lose the
  half device body/output path: fail before simulation, never SKIP.

Native PPU1.7 toolchain/model unavailable locally. CUDA12.8 SM90a compilation
is simulator-input evidence only, not a numerical or performance PASS.

## T02 compile checkpoint (2026-09-29 05:05:18 UTC)

First full-body lowering exceeded the unchanged spill gate:32/32B vs12/12B,
16B stack vs8B. Reject that image (application compile also caught an invalid
string-array initializer). Review additionally found the full collective's
final QueryEmpty arrival is pending: load_tail drains TMA/O but not that named
barrier. It must receive the final32producer arrivals before the half's160
participant phase. Allow one bounded corrected compile with this explicit
drain; do not relax the spill gate if it still fails. If rejected, preserve
the source as a replay patch and hand off only admitted uniform-M64 preflight.
