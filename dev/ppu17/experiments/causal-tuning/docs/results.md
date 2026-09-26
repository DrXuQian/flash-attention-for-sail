# H800 causal comparison and 2x2 experiment

2026-09-26. **Keep the existing 128x80 dynamic-LPT default.** The narrow
tile did not establish a win; removing persistence/LPT made S2048 slower.
Neither the candidates nor the two new references reached70% useful MFU.
This is a physical **H800** result, not native PPU1.7 or the20-SM/32-MiB model.

## Registered measurements

[Contract](plan.md), [all samples and hash-bound receipts](../summary.json),
[flat table](../benchmark.csv). Same B1/Hq32/Hkv2/D256 BF16 causal input,
scale1/16, no KV-head expansion or quantization.200 warmups then9x50 calls,
two independent runs in reverse candidate order. Each table median pools18
samples. No losing samples were trimmed. Nominal dense BF16 peak756.5TF/s;
FLOPs=4*B*Hq*D*S*(S+1)/2. Kernel timings below are separately instrumented
CUPTI durations; full-call events were measured without instrumentation.

| Implementation | S2048 kernel us / MFU | S2048 full us | S8192 kernel us / MFU | S8192 full us |
|---|---:|---:|---:|---:|
| Port, N80 + dynamic LPT (unchanged) |159.409 /57.01%|176.705|2267.696 /64.10%|2413.047|
| Port, N64 + dynamic LPT |159.789 /56.88%|173.591|2318.987 /62.68%|2461.690|
| Port, N80 + single tile |204.844 /44.37%|207.671|2498.237 /58.18%|2558.964|
| Port, N64 + single tile |207.695 /43.76%|205.763|2553.013 /56.94%|2624.291|
| FlashInfer explicit Hopper FA3 |196.231 /46.31%|209.178|2578.593 /56.37%|2609.442|
| cuDNN SDPA direct backend |201.253 /45.16%|206.590|2657.545 /54.70%|2660.938|

Disjoint-envelope verdicts versus the unchanged port:

- **N64+LPT: UNRESOLVED**, both lengths and both timing scopes. Its small
  S2048 full-call median advantage is not an admitted improvement.
- **Single-tile: S2048 SLOWER**, both geometries, both timing scopes. At8K
  all kernel comparisons overlap; N64-single's full call is SLOWER, while
  N80-single's full-call comparison is UNRESOLVED. Do not promote either.
- **FlashInfer and cuDNN: SLOWER**, both lengths, both timing scopes, for
  these exact entrypoints, inputs, precision and operating conditions. This
  does not rank all of their backends/configurations or other workloads.
- Every primary cell is **BELOW_70** under the unchanged criterion, including
  the whole observed range. The priority shape's70% threshold is129.833us;
  S8192's is2076.567us. No goalpost or frequency-adjusted denominator change.

The earlier [official FA3 A/B](../../../docs/h800-official-ab.md) found no
resolved difference from this port. Its S2048 kernel median was164.234us;
the later sequence study measured2227.810us at8K. Those are **earlier sessions**,
not extra samples inserted into this table or a new simultaneous comparison.

## What the experiment actually changed

Only an opt-in shared policy was added. The selected mainloop, TMA/WGMMA,
softmax, precision and epilogue remain the existing implementation. No shipping
selector changed. Host and kernel both use `ppu17_causal::single_tile` so the
single-tile API does not allocate/initialize an unused global task counter.
Trace admission verifies the real template tile, scheduler and grid; single
tile has **zero GPU copy/memset events** in the profiled calls.

| Axis | N80 | N64 |
|---|---:|---:|
| Port CTA threads / static registers per thread |384 /168|384 /168|
| Port dynamic shared bytes |232448|199680|
| Causal body spill bytes (load/store) |0/0|0/0|
| S2048 KV iterations per head, all query tiles |224|272 (+21.43%)|
| S2048 executed rectangular QK/PV pairs per head |2293760|2228224 (-2.86%)|
| S8192 KV iterations per head |3354|4160 (+24.03%)|
| S8192 executed rectangular pairs per head |34344960|34078720 (-0.78%)|

The integer accounting is `sum(ceil(128*i/N), i=1..S/128)` iterations and
`iterations*128*N` rectangular pairs. It counts padded/masked matrix work,
not measured cycles. N64 removes little matrix work while increasing the
number of online-softmax iterations. This explains the tradeoff being tested;
it is not a measured cycle attribution of the entire difference.

The measured H800 has233472 shared bytes/SM and65536 registers/SM. Neither
199680 nor232448 shared bytes permits two such CTAs to reside simultaneously.
So reduced shared allocation is **not** an occupancy doubling. CUPTI's
`blocks per SM=4.491...` for a512-CTA grid is grid/114, not resident occupancy.

Dynamic LPT launches114 persistent CTAs. Single tile launches512 atS2048 and
2048 atS8192. It removes task-counter operations but also changes query-tile
ordering, load balance and the number of CTA/pipeline initializations. The
measured loss is a verdict against that whole substitution, **not** an isolated
measurement of atomic cost. Both mechanisms would need controlled follow-up
to separate them.

FlashInfer's actual kernel is `flashinfer::PrefillWithKVCacheKernel`, with
TMA traits128x64, stages2 and SingleTileScheduler (384threads,168regs,
196688shared bytes). Its Python/host dispatcher has a different name.
cuDNN emitted `cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_...`
(the name is not an assertion that the BF16 inputs were converted to FP16).
It used114x384 and232448shared bytes, and one separate **device memset/call**.
These helper operations remain in full-call timing and are listed separately
from attention in the trace. Never subtract the medians of the independently
timed full-call and CUPTI phases to claim exact API overhead.

## Correctness and code identity

- **24/24 primary runs PASS** full CPU FP64 O and LSE, unchanged tolerances,
  finite checks, within-arm raw replay, input/oracle/trace/binary identity.
  FlashInfer returns log2 LSE on this API; validation converts it to ln on CPU
  only. No reference-side GPU conversion is hidden in the timing.
- Same tile across LPT/single has identical raw O and LSE on both lengths.
  N64 and N80 differ in raw rounding but both pass the independent oracle.
- **42 host + 3 CPU tests PASS**. Wrong requested tile, counter policy, actual
  scheduler, launch grid, stale H2D initialization, fallback kernel, missing
  sample/shape and missing generated definition have failing controls.
- All6 default generated TUs/12 bodies compile and assemble. Their instruction
  text **and machine words** match the pre-experiment source build. The real
  host/internal relocatable link passes. Narrow experimental Torch2.8
  extensions also build, load and run; they reject noncausal/non-BF16/D!=256
  at the API. They are not replacements for the six-unit shipping package.
- The three candidate libraries also pass **9/9 actual API negative controls**:
  noncausal, FP16 and D64 each produce the intended scope error. Traces contain
  **zero device kernels** for the rejected calls, not a post-launch failure.

Immutable control SHA256:
`a78aff6c443238cd888f53dc692bf1e017b9191cff137427677ff1da84640320`.
Candidate hashes:

| Candidate | Extension SHA256 |
|---|---|
|N64-LPT|`164df8ade342d8dd5b007ea11eaf284bad9f3be7eb6b448b1ab25f6ca3257a73`|
|N80-single|`532014bfcde96dc39b5538ad7b837a18bec3dbfbe43449560f13b2739909cbce`|
|N64-single|`a1b27f3d2b54e8306f9b8b7522bdfe5b7e7f00549153b0f908f0c7d957f5ac4c`|

Compute source950cf98, backend023e82d, CUDA12.8.93 SM90a, Torch2.8.0+cu128,
Python3.12.3; harness evolution is bound per result. Reference pins are in
[reference-identity.json](../reference-identity.json). FlashInfer's missing
submodules were restored in an **independent snapshot**, without editing the
other task's source or altering the reference algorithm. cuDNN is91002.
The actual source manifests, loaded library hashes, device UUID and driver
are retained in receipts.

## Operating envelope and failed attempts

No clocks,350W power limit, partitions or other tasks were changed. Checks
reject occupied GPU/foreign validation processes before and after phases.
Several attempts were refused before any target launch; these have no timings.
Setup attempts also caught missing Ninja on PATH and missing FlashInfer
package-data/submodules. The first successful FlashInfer numeric run was
rejected by a checker expecting the host dispatcher name as the device symbol;
the corrected checker is stricter about real Hopper TMA traits/scheduler, and
new runs were measured. None of these failed attempts entered the primary pool.

A preliminary valid cuDNN S2048 run (205.586us kernel) remains in raw artifacts
but is not added as an unmatched third run to the balanced primary comparison.
The first S8192 reference pass was deferred after prelaunch BUSY; recovered
later as `references-*-r3`. The actual ordering is recorded, not called a
perfectly interleaved all-backend A/B/B/A.

Read-only200ms power samples were joined to **actual attention intervals**,
not the whole Python process. At8K the incumbent has9/10 hits with SW Power Cap
active (sampled clocks1005–1215MHz); all other variants/references also have
in-kernel cap hits. At2K the trace is too short for reliable attribution
(0–1 hits/run). These are point samples, not cycle-weighted clocks or a time
fraction. Do not convert this observation into "all remaining loss is power"
or adjust MFU upward. The baseline at8K still faces the same operating limit.

## Reproduce / next step

Scripts committed on `ppu17-causal-tuning`:

- `tools/build_h800_causal_candidate.py`: real narrow candidate extension.
- `tools/run_h800_causal_experiment.py`: serial bounded build/reference/
  screen/reverse-confirm phases, no default promotion.
- `tools/summarize_h800_causal_tuning.py`: re-read traces and hashes, require
  all24 primary runs, pool samples, apply the fixed envelope verdicts.

Raw evidence and candidate binaries are retained locally and on H800 at
`/workspace/flash-attn-ppu17-causal-tune-20260926`; no binary/large trace is
committed into source. Re-adjudicate without GPU work:

```bash
python tools/summarize_h800_causal_tuning.py \
  --root /workspace/flash-attn-ppu17-causal-tune-20260926 --format table
```

Re-run the registered matrix on an **idle physical H800**, using the retained
dependencies and immutable baseline. From this branch's checkout:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/workspace/gdn-sm90-library-compare-20260926/fi-venv/bin:$CUDA_HOME/bin:$PATH
export MAX_JOBS=3
export TMPDIR=/workspace/flash-attn-ppu17-causal-tune-20260926/tmp
# Use a fresh directory; existing result/log files are refused, never replaced.
OUT=/workspace/h800-causal-repeat
for PHASE in build measure; do
  /root/miniconda3/bin/python tools/run_h800_causal_experiment.py \
    --phase "$PHASE" --suffix=-r2 --out "$OUT" \
    --backend /workspace/flash-attn-ppu17-control-e7ee864/backend \
    --control-dir /workspace/flash-attn-ppu17-control-e7ee864/python \
    --flashinfer-root /workspace/flash-attn-ppu17-causal-tune-20260926/flashinfer \
    --flashinfer-python /workspace/gdn-sm90-library-compare-20260926/fi-venv/bin/python || break
done
python tools/summarize_h800_causal_tuning.py --root "$OUT" \
  --suffix=-r2 --large-reference-suffix=-r2 --format table
```

This repeat loop is **not** a PPU simulation command. Missing/failed cells make
the summary fail; they do not silently reduce the denominator.

Keep dynamic LPT. A useful next experiment should target the mainloop's
softmax/rescale/GMMA overlap or test a different math-warpgroup/resource
partition, with a separate bounded registration. If specifically isolating
the single-tile loss, first reverse its query order without changing CTA
initialization count. Do not repeat "remove persistence" or "smaller N is
free" as untested hypotheses, and do not transfer these H800 rankings to
the20-SM PPU1.7 model without its single-launch measurements.
