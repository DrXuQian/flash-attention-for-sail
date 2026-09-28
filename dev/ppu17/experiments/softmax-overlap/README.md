# PPU1.7 S1024 softmax / PV overlap experiment

Fixed workload: B1/Sq1024/Sk1024/H56/Hkv56/D128, FP16 noncausal BSHD,
40-SM model, tile128x176, two stages, 384 CTA threads. Default is unchanged.
**C03 is rejected: the uploaded PPU simulation is 7.58% slower and introduces
54.29 MB of lane-private traffic.** See [the verdict](docs/simulation-verdict.md).
Its 90/90 EX2 overlap is real, but not beneficial in this implementation.
Do not enable it for performance. The original default is unchanged.
The local native control is
CUDA12.8 SM90a code, not an HGGC/native-PPU admission certificate.

## Why this change

The user report (SHA256 in `docs/plan.md`) executes zero of the recurring90 EX2
before PV's wait0, although the source/PTX places the softmax there. The old P
registers are reused immediately after wait0. A raw source-line reorder or
deleting the wait is not safe and would not establish native overlap.

C01's empty CuTe operand fences leave native code unchanged (19/90 overlap
on the NVIDIA control). C02's two volatile row-sum stores/loads achieve90/90,
but spill stores/loads grow12->36B. Both are rejected and absent from the final
kernel. C03 uses one thread-private volatile store of the XOR of the TWO
completed row-sum bit patterns. No numerical state is replaced or rounded.
The token is intentionally unused: it is a real compiler-order anchor, with
an explicit cost, not a correctness flag or an inter-thread message.

- One additional 32-bit word per math thread:1KiB per CTA, before outer alignment.
- One XOR and one shared store per thread/steady iteration; no readback.
- 2,293,760 shared bytes written over this fixed shape; no extra global traffic.
- No added hardware barrier, no removed TMA/GMMA completion wait, no V early reuse.
- Existing prologue, drain, tile/stages, scheduler and precision are unchanged.
- Default and ineligible types select the original storage TYPE. The opt-in
  is compiled only for ordinary FP16, noncausal128x176/D128/two-stage mainloop.
  The standalone harness admits only the fixed S1024 case; no routing promotion.

## Reproducing the rejected candidate (not a performance recommendation)

From this branch of flash-attention-for-sail, using the same CUTLASS4.3 backend
and CUDA compiler as the baseline (replace only paths if installed elsewhere):

```bash
python tools/build_ppu17_standalone.py \
  --cutlass /workspace/flash-attention-for-sail/csrc/cutlass3 \
  --cuda-home /usr/local/cuda-12.8 \
  --out /workspace/fa17-softmax-overlap-c03 \
  --softmax-overlap
```

Output directory must be new. No package installation, identity probe, PTX dump
or profiler replay is required. `--inspect-codegen` is an OPTIONAL local build
inspection mode. The build uses the existing generated shipping TU, not a copy.
Its manifest binds compiler flags and `softmax_overlap=row-sum-token`; `--describe`
on the resulting ELF prints that variant without invoking CUDA.

Use this application command as the simulator input:

```bash
/workspace/fa17-softmax-overlap-c03/flash_attn_ppu17_s1024_fp16 \
  --expected-sms 40 --verify
```

Exactly one target invocation; no warmup, repeat, GPU reference or initialization
kernel. Full O/LSE verification runs on CPU after D2H. Expected SM count is an
assertion, not a partition. Use the simulator's unchanged launcher/model config;
this repository does not invent a simulator wrapper command.

For a fresh control build omit `--softmax-overlap` and use a different output
directory. Each arm runs in its OWN simulator process. Do not run both within
one process or infer the PPU timing from a CUDA control machine.

## Mechanical postconditions

For an inspected local build:

```bash
python tools/check_ppu17_softmax_overlap.py ptx \
  /workspace/fa17-softmax-overlap-c03/shipping_fp16_d128.ptx --require-before 90
python tools/check_ppu17_softmax_overlap.py sass \
  /workspace/fa17-softmax-overlap-c03/executable.sass --require-before 90
```

After simulating, inspect the new report (does not invoke a GPU):

```bash
python tools/check_ppu17_softmax_overlap.py report \
  /path/to/new/perfstatistics.json --require-before 90
```

Keep the entire report and application output. The checker verifies one
kernel/40 CTAs, the complete matrix/exponent/wait execution denominators,
and the exponent placement. It does NOT derive a numerical PASS from counters.
Require the application's CPU-FP64/PASS and compare parent O/LSE fingerprints
when available. Missing parent binary/model hashes remain a provenance gap.

Native old-P/O writes before the completion wait are rejected. Negative tests
also remove a wait, one EX2, one MMA, move the old schedule back, use the wrong
symbol and corrupt one executed-PC count. A compile PASS is not a timing win:
require unchanged model settings and complete kernel cycles below247267; first
target229376 cycles corresponds to80% useful MFU. No expected speedup is claimed.

## Local result and replay

`results.json`:56 PASS /3 SKIP /0 FAIL. The three skips are the existing Torch
Python oracle tests (Torch unavailable in the active interpreter); standalone
C++ CPU oracle, host ELF paths and the real missing-generated-unit link negative
pass. Neither oracle self-tests nor codegen gates execute the candidate kernel.
NVIDIA native control/candidate:168reported registers,8B stack,12B spill stores/loads;
EX2 overlap19->90, static instructions2624->2632. Both default body encodings
match parent4b26264; ineligible causal encoding also matches with the flag on.

Replay against the local artifacts without a GPU:

```bash
python dev/ppu17/experiments/softmax-overlap/validate.py \
  --parent-sass /workspace/flash-attn-ppu17-causal-tune-20260926/standalone-encoding-20260928/inspected/executable.sass \
  --control /workspace/fa17-softmax-overlap-20260929/control \
  --candidate /workspace/fa17-softmax-overlap-20260929/candidate \
  --report /path/to/archived-control-perfstatistics.json \
  --out /workspace/fa17-softmax-overlap-20260929/validation.json
```

The old raw control upload was overwritten by C03; only the baseline extracted
evidence remains locally. The preceding historical full replay now needs that
original control report. Do not substitute the C03 report to label it control.
Current report/negative tests use `FA17_OVERLAP_CANDIDATE_REPORT`; one missing
baseline-report test is explicitly SKIP, not PASS.
