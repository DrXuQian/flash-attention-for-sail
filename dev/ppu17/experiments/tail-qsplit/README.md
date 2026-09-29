# Q-tail experiment — separate from software exp

Branch `ppu17-tail-qsplit`, parent `d19d27a`. No exp, softmax, mainloop or
epilogue arithmetic changes. **No performance winner/default promotion yet.**

The mixed440-full +16-half implementation was compiled, not merely modeled,
and rejected by the registered resource gate: CUDA spill12->32B in each
direction, stack8->16B. Its live kernel/launcher changes have been removed;
`--q-tail-mode hybrid` and macro value2 fail closed. The complete rejected
implementation remains in [rejected-hybrid.patch](rejected-hybrid.patch),
applicable to a fresh worktree at `d19d27a` (`git apply --check` passes).
It is a replay artifact, **not a candidate to simulate**. See the actual
[transition seam audit](docs/transition.md), including the final QueryEmpty
arrival that must be drained before changing the cohort count.

## What can be measured now

T01 is a **uniform Q64 control**, not the rejected tail-only implementation.
It reuses the existing one-math-warpgroup collective for every task. It tests
whether64-row work is sufficiently cheap before retrying mixed granularity.

| Fixed FP16 B1/S1024/H56/D128, noncausal,40SM | C07 Q128/KV128 | T01 Q64/KV128 |
|---|---:|---:|
| Total Q tasks |448|896|
| Most tasks assigned to one CTA |12|23|
| CTAs in final wave |8/40|16/40|
| CTA threads / math WGs |384/2|256/1|
| Shared allocation, bytes |166912|150528|
| CUDA registers / stack / spill stores / loads |168/8/12/12|255/0/0/0|
| Static CUDA instruction sites |2344|2264|
| Matrix instruction shape / math |m64n128k16|unchanged|
| PPU cycles |240412, user-reported only|NOT_RUN|

255regs/thread is not a reduction from168. The CTA has fewer threads, and
the role budgets differ. Do not infer extra resident CTAs from these numbers.
Q/K/V bytes per logical attention problem do not change, but K/V *task fills*
double. Cache-served bytes are not measured HBM bytes. Math/exp source is
identical; machine scheduling can still change with geometry (this is not an
exp algorithm experiment).

Equal-cost task model: baseline12*t128 versus23*t64. T01 wins only if
`t64/t128 < 12/23 ≈ 0.52174`, including setup/delivery/epilogue. This is a
break-even condition, **not a predicted speedup**. It is not a clean
tail-only speed attribution: every task changes geometry. The ideal tail-only
12->11.5 model would offer4.17% time reduction before its own added overhead;
that implementation did not pass local admission.

## Build / one simulation invocation

Use `DrXuQian/flash-attention-for-sail`, branch `ppu17-tail-qsplit`. Keep the
same CUDA/CUTLASS backend and simulator/model settings as C07. Paths below
match the earlier handoff; substitute only your already-used backend path.
The output directory must be new. No Torch installation or GPU work in build.

```bash
python tools/build_ppu17_standalone.py \
  --cutlass /workspace/flash-attention-for-sail/csrc/cutlass3 \
  --cuda-home /usr/local/cuda-12.8 \
  --out /workspace/fa17-tail-m64-t01 \
  --kv-tile128 --q-tail-mode m64
```

Give this application command to the **same40-SM simulator** as C07:

```bash
/workspace/fa17-tail-m64-t01/flash_attn_ppu17_s1024_fp16 --expected-sms 40 --verify
```

One target invocation; no warmup/repeat or helper GPU kernel. CPU FP64 O/LSE
reference keeps atol/rtol0.002. Preserve stdout, build.json and perfstatistics
under this candidate's directory. The model's runtime settings/hash and C07's
raw report/numerical stdout are still needed for a fully bound comparison;
the message `240412cycles` alone does not provide them.

Gate: numerics pass, no new native PPU private spill traffic, and complete
compute cycles below240412 on the same model. Tail duration/active-warps are
secondary. A lower tail with a slower complete call is a loss. Useful MFU is
`30064771072 / (cycles * 163840)` for this40-SM model, not the raw report's
72-SM denominator. No percentage adjustment to unrelated counters.

## Local proof and limits

Final complete tier: **68 PASS /4 SKIP /0 FAIL**. Three skipped old Python
oracles need unavailable Torch; the fourth needs the overwritten original
PPU report. The independent standalone C++ CPU oracle ran and passed.

- Real generated shipping TU assembled and linked, both causal/noncausal
  bodies live; no replacement miniature GPU model.
- Default encoded bodies unchanged2/2; candidate causal unchanged. Original
  softmax, mainloop, epilogue, kernel and launcher source hashes unchanged.
- Actual CuTe accumulator and actual scheduler coordinate method enumerate
  all7340032output cells per layout, independently anchored by lane/warp
  coordinates. Omitted/duplicated half, wrong head/origin and wrong denominator
  each fail. Coverage is not simulated numerical correctness.
- Actual wrong initialization count128instead of256 fails compilation. An
  enlarged consumer that overlaps old pipeline storage fails its type guard.
  Both corresponding unmutated controls compile. These are archived T02
  transition contracts, not proof that its rejected body should be timed.
- CUDA is source/native-control evidence only. Native PPU1.7 allocator and
  model execution are not available locally. A CUDA zero-spill result cannot
  promise PPU zero spills (C03 already demonstrated that boundary).

Local artifact root: `/workspace/fa17-tail-qsplit-20260929`. The validated
executable is `t01-final/flash_attn_ppu17_s1024_fp16`. Local replay is
`validate.py --help`; [results.json](results.json) binds source/backend/ELF and
the complete suite's PASS/SKIP/FAIL counts. Rejected source is preserved, not
left in an active collective. Native PPU numerics/performance remain NOT_RUN.
