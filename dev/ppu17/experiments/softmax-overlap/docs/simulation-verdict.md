# C03: real overlap, slower kernel — rejected

Reviewed 2026-09-28 23:41:51 UTC, before any further kernel changes.
This closes the registered three-variant experiment as **no performance win**.
Keep the original default, which was never switched to C03. Retain the opt-in
as a recorded counterexample, not as a recommended build.

## Same work, different cost

Scope remains FP16/noncausal B1/S1024/H56/D128, 40 CTAs / 480 warps,
tile128x176/two stages. One target invocation in the uploaded report.

| Metric | Uploaded control | Uploaded C03 |
|---|---:|---:|
| Compute cycles | 247,267 | 266,011 (+7.58047%) |
| Useful MFU, 40 × 4096 FLOP/cycle | 74.2116% | 68.9824% |
| Tensor utilization / user's 55.556% ceiling | 76.53% | 71.14% |
| Steady EX2 sites before PV wait0 | 0 / 90 | 90 / 90 |
| Steady PV wait0 sync warp-cycles | 3,680,826 | 0 |
| Executed instructions | 13,726,904 | 14,700,954 (+7.09592%) |
| QK / PV matrix warp-executions | 172,032 / 236,544 | identical |
| EX2 / FMA / ADD / MUL FP32 executions | 1,928,192 / 1,935,360 / 1,899,520 / 1,462,592 | identical |
| TMA shared writes | 256,901,120 B | identical |
| Scalar vector-memory reads | 0 B | 35,323,904 B |
| Scalar vector-memory writes, including LSE | 229,376 B | 19,193,856 B |
| LSU shared writes | 14,905,344 B | 17,199,104 B |

The extra shared writes are exactly the declared token cost, 2,293,760 B.
The **54,288,384 B of private spill traffic were not declared source work**.
Neither statistic is additional HBM traffic: caches service most of these
private requests. HBM reads rise only44,194,816→44,258,880 B; writes
14,913,088→16,182,208 B. This is an operand/dependency cost, not Q/K/V growth.

## Native proof and closed byte ledger

The PPU instructions use a lane-private address pattern:

```
vmem.st.b32 value, [slot + (vreg2 + %tid) * 0x4] @sreg[108:109]
vmem.ld.b32 value, [slot + (vreg2 + %tid) * 0x4] @sreg[108:109]
```

The prologue constructs that private base; these are not TMA Q/K/V reads.
`slot` below is the encoded operand, **not a byte offset**. The source has
no such global array. Value stores and their reloads expose register spilling:

- `0x38bf870`: store an FMA score intermediate in slot`0x805`;
  `0x38bf938`: load it for the exponent path.
- `0x38bf4f0/0x38bf508`: save the two previous row sums in`0x804/0x803`;
  `0x38bfab0/0x38bfc88`: reload them for the online update.
- `0x38bf880/0x38bf888`: save row maxima in`0x801/0x802`;
  each is loaded for softmax and again after wait0 for the next iteration.
- `0x38bdd48/0x38bdd58`: save the scheduler barrier identifiers in`0x80a/0x80b`;
  `0x38bf2b8/0x38bf620` load them, wait for vector memory, transfer to scalar
  registers, then execute the original barriers. Metadata is spilling too.
- `0x38bdb10` and following sites save epilogue addresses; later loads directly
  feed `tsm.st.mat.b32x4`. The cost extends beyond the changed softmax block.

All private accesses have32active math lanes and4B per lane. There are17,920
steady warp-iterations,3,584 output-tile warp visits and320 math-warp setup
visits. The full denominator, not a selected example:

| Private values | Read warp-instructions | Write warp-instructions | Read bytes | Write bytes |
|---|---:|---:|---:|---:|
| Eight hot slots`0x801..0x808`: max/sum/score intermediates | 10×17,920 | 8×17,920 | 22,937,600 | 18,350,080 |
| Three invariant slots`0x809..0x80b`: token pointer/barrier IDs | 3×17,920 | 3×320 | 6,881,280 | 122,880 |
| Twelve tile/epilogue metadata slots`0x80c..0x817` | 12×3,584 | 12×320 | 5,505,024 | 491,520 |
| **Total private traffic** | **275,968** | **148,160** | **35,323,904** | **18,964,480** |

Read bytes equal `vmem_inst_read_bytes` exactly. Private writes plus the
unchanged229,376B LSE stores equal `vmem_inst_write_bytes` exactly.
The report checker reproduces this accounting and refuses unclassified reads.

The intended schedule did materialize: the final row sum feeds XOR at
`0x38c01d8`, the volatile shared store at`0x38c01e8`, and wait0 at`0x38c01f0`.
Thus the observed failure is not an ignored source annotation. Making all
softmax results live while old P/O remain in flight raised native allocation
pressure; values now spill. NVIDIA's equal168regs/12Bspill compile result was
only a control and failed to predict the PPU allocator's behavior.

## What the stalls do and do not prove

Aggregate sync warp-cycles fall13,784,057→11,512,138. At the same time,
memory-dependency stalls rise1,379,437→3,411,390, compute-RAW
1,975,726→3,501,214, and SFU-busy149,338→906,187. `s.wait` executions grow
302,387→582,336. These are real dependency/issue costs accompanying the spills.

They are **not additive wall-clock components**. Sampling/timeline data is
empty; we cannot assign all18,744 extra kernel cycles to spilling alone.
The new softmax schedule also changes SFU pressure, WG issue timing and V-stage
release timing. Zero wait0 means PV is already complete when reached, not
that the kernel saved every former wait cycle. No general claim that useful
softmax/PV overlap is impossible follows from this failed implementation.

## Evidence, limitations and replay

- Baseline report hash:
  `3cdd7babe0b51542f497b081350f3fab3b5899412fb31ad369fe537ea3254557`.
  Its extracted evidence is in `results.json`; the user's replacement upload
  overwrote `/root/perfstatistics.json`. The original full raw report is no
  longer present locally. Do not run baseline-only tests on the replacement.
- C03 report hash:
  `3158e2fad86e4c176e2f606dbe832890d3a0779e499768787906973327a09f49`.
  Archived without changing the upload at
  `/workspace/fa17-softmax-overlap-20260929/c03-simulation-3158e2fa/perfstatistics.json`.
- Candidate is consistent with the C03 native instruction signature, full
  matrix/exponent work and fixed launch. Application output, input/binary hash
  and simulator configuration hash were not uploaded. Therefore CPU O/LSE
  correctness and complete run provenance remain **NOT PROVIDED**. The losing
  result is sufficient to reject promotion, not to claim numerical admission.
- Model period units and host `real_time` are not a physical device frequency.
  These are single-run simulated cycles, not physical-device microseconds.

```bash
python tools/check_ppu17_softmax_overlap.py report \
  /workspace/fa17-softmax-overlap-20260929/c03-simulation-3158e2fa/perfstatistics.json \
  --require-before 90
```

That passes the schedule/work/byte accounting. Adding `--max-private-bytes 0`
must fail: zero is this uploaded control's private-traffic baseline, not a
post-hoc performance threshold. Tests also remove one private read while
updating both instruction totals: the independent byte counter must turn red.

Local full suite after the checker change: **58 PASS /4 SKIP /0 FAIL**.
Three skips are unavailable Torch Python oracles, one is the overwritten raw
baseline report. The actual C++ CPU-oracle self-tests, linked executable host
paths, new report checks and negative controls run. These do not substitute
for the absent application numerical log. Default encoded bodies remain2/2
identical to the admitted parent; the opt-in causal body is also unchanged.

Next experiment, if continued: first reduce simultaneous softmax temporary
lifetimes/allow partial overlap, preserving FP32 arithmetic and P/O lifetimes.
Do not repeat this whole-softmax publication mechanism or alter the default.
Native PPU private-memory traffic must be part of admission, separately from
the NVIDIA compile controls. No new kernel variant is introduced by this review.
