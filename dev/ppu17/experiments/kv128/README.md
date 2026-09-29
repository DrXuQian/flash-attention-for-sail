# C07: KV tile128, original pipeline

An opt-in test, **not a selected performance winner**. The current default
remains KV176. C03's forced overlap and C04–C06's rejected partial variants
are not active here. See the fixed [plan](docs/plan.md).

Fixed case: B1/Sq1024/Sk1024/H56/Hkv56/D128, FP16, noncausal, contiguous BSHD,
40-SM model. Keep query tile128, two stages,384threads and persistent grid40.

| Local evidence | KV176 control | C07 KV128 |
|---|---:|---:|
| Encoded static CUDA instructions | 2624 | 2344 |
| CUDA registers / stack / spill stores / loads | 168 /8 /12 /12 B | unchanged |
| Old P / O words protected before completion | 44 /64 | 32 /64 |
| KV iterations / padded key positions | 6 /1056 | 8 /1024 |
| Default encoded bodies | 2/2 unchanged | opt-in causal unchanged |
| PPU simulation cycles | 247267 (previous upload) | NOT_RUN |

The smaller fragment does **not** reduce the reported register allocation in
this CUDA build. Fewer static instructions are not fewer dynamic instructions:
the loop count rises. The smaller padded matrix workload alone removes3.03%
of matrix FLOPs. Measure complete modeled cycles, preserving useful-FLOP MFU.

Full local suite: **63 PASS /4 SKIP /0 FAIL**. Three Python-reference tests
need unavailable Torch; the fourth needs the original full baseline report,
which was overwritten by the user's later upload. The independent C++ CPU
oracle, actual ELF host paths, real missing-generated-unit linker negative,
1536 selector combinations, geometry/accounting negatives all run and pass.
These are host/compile checks, not the candidate GPU's numerical result.

## Build and single simulation invocation

Use branch `ppu17-softmax-overlap` from `DrXuQian/flash-attention-for-sail`.
Run from its root. Keep the CUTLASS4.3 and CUDA paths from the last successful
standalone build if they differ from these examples. The build has no Torch,
pip installation or mandatory PTX/disassembly step. Output directory must be new.

```bash
python tools/build_ppu17_standalone.py \
  --cutlass /workspace/flash-attention-for-sail/csrc/cutlass3 \
  --cuda-home /usr/local/cuda-12.8 \
  --out /workspace/fa17-kv128-c07 \
  --kv-tile128
```

Give the following application command to the **same simulator and model
configuration as the original KV176 run**:

```bash
/workspace/fa17-kv128-c07/flash_attn_ppu17_s1024_fp16 --expected-sms 40 --verify
```

Exactly one attention invocation; no warmup/repeat or GPU reference. Verification
uses full CPU FP64 O/LSE, unchanged atol/rtol0.002. Keep its stdout, `build.json`
and new `perfstatistics.json` together. Do not overwrite the original report.
An SM assertion does not partition hardware. This executable is CUDA SM90a
simulator input, not a native PPU binary certificate.

## Mechanical evidence and decision

The registered decision remains: correctness passes, no new PPU private spill
traffic, and compute cycles <247267 with the same model settings. A loss stays
a loss. KV grouping changes rounding order; cross-geometry bit equality is not
required and does not replace the unchanged CPU reference. The report alone
does not contain that numerical verdict or prove binary/model identity.

```bash
python tools/check_ppu17_softmax_overlap.py report /path/to/new/perfstatistics.json \
  --kv-tile 128 --max-private-bytes 0
```

The checker derives 66 steady EX2 sites, wait visit counts3584/25088/25088/3584,
and458752 executed m64n128k16 warp-instructions from the registered geometry.
It does not require the forced-overlap placement used by the rejected C03.
Useful work stays30064771072FLOPs; model peak40*4096FLOP/cycle.

## Local replay (no GPU execution)

The exact artifacts are in `/workspace/fa17-kv128-20260929/{control,candidate}`.
[results.json](results.json) records source/backend/compiler and ELF hashes.

```bash
FA17_OVERLAP_SASS=/workspace/fa17-softmax-overlap-20260929/candidate/executable.sass \
FA17_OVERLAP_CANDIDATE_REPORT=/workspace/fa17-softmax-overlap-20260929/c03-simulation-3158e2fa/perfstatistics.json \
python dev/ppu17/experiments/kv128/validate.py \
  --parent-sass /workspace/fa17-softmax-overlap-20260929/control/executable.sass \
  --control /workspace/fa17-kv128-20260929/control \
  --candidate /workspace/fa17-kv128-20260929/candidate \
  --out /workspace/fa17-kv128-20260929/validation-replay.json
```

The old C03 inputs above exercise historical **negative** controls, not C07
performance. If artifacts are unavailable those tests say SKIP; never borrow
C03 numerics or performance for this candidate.
