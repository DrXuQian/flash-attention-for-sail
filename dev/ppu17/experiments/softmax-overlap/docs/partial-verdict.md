# Partial overlap: three compile rejections, no simulation requested

2026-09-29 03:06:05 UTC. Source parent `fc4947d`; performance incumbent is
still the uploaded original control,247267 cycles, not C03's266011 cycles.
No numerical/performance result is claimed for C04–C06.

| Candidate | Mechanism | PTX EX2 before/after | CUDA native before/after | Stack / spill store / spill load | Verdict |
|---|---|---:|---:|---:|---|
| C04 | Only row0 contributes to token | 90/0 | 75/15 | 8/12/12 B | Reject: not45/45 |
| C05 | Separate whole-row update functions | 45/45 | 76/14 | 8/12/12 B | Reject: native scheduler hoists row1 |
| C06 | Save row1 old max, volatile shared readback after wait0 | 45/45 | 45/45 | 24/40/40 B | Reject: added spill |

C04/C05 close the first bounded two-attempt source-order experiment. C06 was
separately registered before editing as a real data-dependency experiment.
Its schedule works but allocation fails. The45/45/no-new-spill criteria were
not widened to admit any result. PPU lowering was not run for these losers;
NVIDIA evidence is not a prediction of their PPU performance.

All retain16QK+22PV static matrix sites and safe old-P44/O64 live ranges.
C05's FP32 PTX math-op inventory is identical to control. C06's static body
returns to2624instructions, yet spill sites double3→6 per direction: total
instruction count is not a resource/lifetime admission gate. Newly built
default-off machine words match the original control2/2; all candidate causal
bodies also match. The local suite after cleanup is58 PASS /4 SKIP /0 FAIL:
three missing-Torch Python oracles and one overwritten original raw report.

The real CuTe layout was enumerated on host for256threads×2rows×44scores:
22528QK cells exact-once, with an independent SM90 lane/warp coordinate anchor.
512C06 private words have one owner each. Wrong row, aliased token and missing
last element plants all reject. The pure layout helper originally had a
device-only annotation; the archived patch makes it host/device constexpr for
this check. It does not fake an architecture macro or replace an atom.

## Cleanup and replay

The five edited live source/build files were restored exactly to `fc4947d`.
No new overlap option or softmax helper remains active. The original C03
opt-in stays as before; it is already rejected by its PPU result.
`rejected-partial.patch` preserves the final experimental source, applicable
to that parent in a separate disposable worktree. `partial_layout.cpp` requires
that patch and the exact recorded CUTLASS4.3 backend; it is not a shipping TU.

Local artifact root: `/workspace/fa17-softmax-overlap-20260929`.
Per-build `build.json`, actual ELFs, PTX/SASS and resource logs remain in
`c04-row0`, `c05-rowwise`, `c06-dependency`, and `control-partial`.
`partial-results.json` records their identities and computed rejection reasons.
The patch is an exact C06 source replay; earlier native variants' evidence is
their saved artifacts, not a claim that the later patch reproduces their hashes.

```bash
python dev/ppu17/experiments/softmax-overlap/validate_partial.py \
  --artifacts /workspace/fa17-softmax-overlap-20260929 \
  --out /workspace/fa17-softmax-overlap-20260929/partial-validation-replay.json
```

This inspects compiled files and executes only the host layout enumerator and
its negatives. A successful replay confirms that all three are **rejected**;
it does not relabel them numerical or performance PASS.

Next distinct axis: an opt-in KV tile128 instead of176 with the original
softmax/pipeline. It lowers simultaneous QK/P storage and removes the1056→1024
KV padding, but increases loop/softmax steps6→8. Register a separate geometry
test and measure full modeled cycles; do not assume the smaller tile is faster.
