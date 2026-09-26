# Same-H800 official FA3 control: no resolved port penalty

Completed 2026-09-26 UTC. Shape: **B1/S2048/Hq32/Hkv2/D256, BF16, causal**.
Physical H800 PCIe,114 SM,50MiB L2, CUDA12.8.93, Torch2.8.0+cu128,
driver595.71.05. This does not validate native PPU1.7 or the20-SM model.

## Result

Each arm ran in two fresh processes, order A/B/B/A. Each run had200 warmups
and9 samples of50 calls. Reused prepared inputs, no cache flush. Complete-call
event spans and separately instrumented CUPTI kernel durations are different
measurements; do not subtract their medians as an API-overhead measurement.
The table pools the18 aggregate samples per arm, not just the fastest run.

| Measurement | Our admitted port A | Unmodified official FA3 B | Registered verdict |
|---|---:|---:|---|
| Attention kernel only, CUPTI |161.030us|164.234us|UNRESOLVED: ranges overlap|
| Useful causal kernel MFU |56.44%|55.34%|Both BELOW_70|
| Kernel sample range |146.528–174.850us|149.426–181.119us|Not a confidence interval|
| Complete registered-op CUDA-event span |177.806us|173.629us|UNRESOLVED: ranges overlap|
| Complete-call sample range |169.082–181.387us|158.383–177.823us|Both BELOW_70|

The independent kernel medians were A1=163.93666us, A2=158.12324us;
B1=162.11588us, B2=166.35270us. Their useful MFU values were
55.44/57.48% and56.06/54.63%, respectively. Thus **official FA3 also does
not reach70% on this exact shape/device/protocol**. There is no resolved
performance loss from our port in this comparison. This is not proof of
exact speed equivalence, an optimized upper bound, or a universal H800 limit.

Useful work is68,753,031,168 FLOPs: QK+PV, MAC=2, valid causal pairs including
the diagonal. Fixed dense BF16 peak=756.5TF/s;70% requires129.832936us.
Every aggregate sample is slower than that fixed threshold. Neither FP8 nor
masked upper-triangle work was used to inflate the numerator/denominator.
Protocol was committed before timing in
[h800-official-ab-plan.md](h800-official-ab-plan.md), harness `a87a28e`.

## Numerical and execution binding

All four runs passed the unchanged CPU FP64 output/LSE oracle before/after
timing and raw within-arm replay. In addition (observed, not required), both
arms' output and LSE hashes are identical on this fixture. Maximum output
absolute error=0.00915942203100295; maximum LSE error=1.624670279198881e-6.
The three CPU input hashes match the original admitted fixture exactly.
No GPU reference ran. The output SHA256 is
`65dad8c0e3574f5adad956142fd2ffd22cdf87d2c801238e5204f6d1aa7e5f1f`.

Both actual attention launches use tile128x80x256,2 stages, grid114,
384 threads/CTA,168 registers/thread and232448 shared bytes. Both actual
causal/unpacked-GQA bodies have zero stack/spill in compiled resources.
Do not mistake the trace's estimated-occupancy field for measured occupancy.

Per450 traced invocations, A has450 attention kernels +450 CPU-to-device
scheduling-counter copies, no auxiliary GPU kernel. B has450 attention
kernels +450 `FillFunctor<int>` counter-zero kernels, no H2D copy. Complete-call
timing includes those respective costs. Kernel-only timing excludes them but
is explicitly profiler-instrumented. The benchmark invokes the same registered
C++ operator kwargs, not two different Python wrappers.

Default clocks were not locked or changed. Post-phase snapshots span1155–1545MHz
after event timing and1635–1725MHz after tracing. These are not averages inside
the measured kernels. They neither establish a clock bottleneck nor justify
renormalizing MFU. The broad observed ranges are why a2% median difference is
not reported as a win.

## Source and artifact identities

- A: immutable22/22-admitted extension, packaged source `e7ee864`, compute
  source `a3d25ff`, PPU CUTLASS3.6.0 backend `023e82d`.
- B: official `v2.8.3.post1`, source
  `a8aa52b1ab3e9ca574c8a33b3f35afc017ffa2e2`, official CUTLASS4.0.0
  `dc4817921edda44a549197ff3a9dcf5df0636e7b`. Its setup.py and all source
  files stayed unmodified. Only upstream build switches narrowed unused
  dtype/head-dimension/features; PackGQA and cluster support remain compiled.
- Both calls explicitly use num_splits=1, pack_gqa=False. The official
  fixed-length heuristic also selects unpacked GQA for this exact2048 case.
- The mainloop source files are byte-identical (SHA256
  `4c39acafad4acc8931a873037c223dfc6ecfd1ede3af38c75ef3608acac8afdb`).
  Exact causal/unpacked generated bodies contain3640/3672 static SASS
  instructions, respectively. Counts are diagnostic, not executed instruction
  counts or a substitute for timing; backend/codegen are not byte-identical.

| Artifact | SHA256 |
|---|---|
| A extension |`a78aff6c443238cd888f53dc692bf1e017b9191cff137427677ff1da84640320`|
| B extension |`c0611358efe2a1511843ef2638befce6932b1d5855225c7d7d89ebf8583ef585`|
| Official source archive |`fd4e938c96edb4bca2d3dc5814f985dd43547e383aee5685d9bbbd4bac59e653`|
| Harness source archive |`2010bd45ad574b9853f95d333683711c2d9468d8a723905e72190528987f61eb`|
| Official build log |`55cb69cf4d26493369d3113b17559d6b5661ae1fd31922544e62b5269dc9137c`|
| `a1/result.json` |`d5bbe0e9b738e3f1669d6ffe846679478330e973c3922579ad61487c4a940836`|
| `b1-r2/result.json` |`2d8d46b0ad69cdaee67af39d502918501dbe27eebb523ddff848e0e886371618`|
| `b2/result.json` |`95bfc8da26b8df2c6eb3e7a81ad8b2b8cd6c97077ae380160dc4d87bfba000f9`|
| `a2/result.json` |`3e877dd4b20d2ce15010f93ed427c3b4f72d0557f8ba2cea78d6fa7cca209655`|

Raw samples, traces (hash-bound in each result), logs and binaries are retained
under `/workspace/flash-attn-official-h800-ab-20260926` locally and on H800;
they are not source commits. A first build attempt and a first B1 attempt
were refused by idle admission before work. B1-r2 began only after the foreign
extended-validation parent exited. No other task was killed or reconfigured.
The GPU was empty after the final arm.

Host checks:21 existing source contracts +4 existing timing negatives +4 new
A/B tests PASS. Wrong/aliased binary, changed fixture, missing/extra target,
substituted GPU reference, wrong fill datatype and a fill kernel on A all fail.

## Repeat

Only on the otherwise idle inspected H800; choose new output directories.
The fresh-process benchmark takes `--arm control|official`, `--extension-dir`,
`--expected-sha256`, and `--out`. For example, using the retained artifacts:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /root/miniconda3/bin/python \
  /workspace/flash-attn-official-h800-ab-20260926/harness/tools/bench_h800_official_fa3.py \
  --arm official \
  --extension-dir /workspace/flash-attn-official-h800-ab-20260926/build-official-r2/python \
  --expected-sha256 c0611358efe2a1511843ef2638befce6932b1d5855225c7d7d89ebf8583ef585 \
  --out /workspace/flash-attn-official-h800-ab-20260926/manual-official-confirmation
```

These repeated physical-Hopper tools must not replace the single-invocation
PPU1.7 simulator runner. Next optimization, if requested, must address this
specific shape's execution and measurement behavior; a long-sequence H100
headline is not an established70% result for B1/S2048 GQA.
