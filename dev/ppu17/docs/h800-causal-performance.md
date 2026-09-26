# H800 causal BF16 baseline: 70% not reached

Measured 2026-09-26 UTC, after numerical admission, on the otherwise idle
H800 PCIe. No kernel, configuration, clocks, power limit or driver changes.
This is a physical Hopper control, **not** a PPU1.7 simulator result.

Shape: **B1/S2048/Hq32/Hkv2/D256, BF16, causal**. Same prepared inputs,
seed170020, CPU FP64 O/LSE reference and admitted output fingerprint.
Binary `a78aff6c443238cd888f53dc692bf1e017b9191cff137427677ff1da84640320`
is unchanged from the 22/22 correctness control. Benchmark source: `e7a2157`.

## Causal-effective denominator and result

Useful QK+PV work, counting each MAC as2 FLOPs:

`4 * B * Hq * D * S*(S+1)/2 = 68,753,031,168 FLOP`.

This excludes masked upper-triangle/padded work and includes the diagonal.
Use the fixed **756.5 TFLOPS dense BF16** nameplate denominator: the
[H800 PCIe specification](https://lenovopress.lenovo.com/lp1814-thinksystem-nvidia-h800-pcie-gen5-gpu)
states1513 with structural sparsity, so divide that rating by2.
The preregistered 70% latency target is **129.832936 us**.

| Timing scope | Run1 median | Run2 median | Causal-effective MFU, run1 / run2 |
|---|---:|---:|---:|
| Instrumented device kernel only (CUPTI) | 168.810 us | 163.354 us | 53.84% / 55.64% |
| Uninstrumented complete-call CUDA-event span | 174.563 us | 182.928 us | 52.06% / 49.68% |

Both scopes in both completed runs are **BELOW_70** under the unchanged
criterion: every aggregate sample is slower than129.832936us. Kernel-only
sample ranges are152.918–175.809us and150.778–172.691us; median useful
throughput is407.28 and420.88 TFLOPS. Relative to those kernel medians,
reaching70% requires approximately **20.5–23.1% less latency**.

Each phase used200 warmups and9 samples of50 calls. The CUPTI phase contains
exactly450 real attention kernels and no extra GPU kernel. Complete-call
timing includes public API counter preparation/copy and submission gaps;
kernel timing excludes those gaps but is explicitly instrumented. These are
separate phases: **do not subtract their medians to estimate API overhead**.
Cache protocol is repeated prepared inputs, no flush, not cold attention.

Both completed runs passed pre/post CPU FP64 output and LSE checks and
retained the admitted raw output fingerprint. No GPU reference was used.
The second attempted process (`baseline-v2`) was refused before CUDA import
because another Python task was present. It contributed no timing samples.
After that parent exited, the independent confirmation ran as `baseline-v3`.

## What this does and does not establish

- Current default-clock H800 baseline is about54–56% useful causal kernel MFU,
  not70%; the achieved number is not inflated by counting the full square.
- This is not an optimized upper limit. Frequencies were not locked: snapshots
  after event timing were1245/1530MHz; after profiling1725/1740MHz. These are
  point observations, **not time-averaged clocks inside the kernels**, and do
  not establish clock drift as the bottleneck or justify rescaling the result.
- Actual specialization: tile128x80x256,2 stages, grid114,384 threads/CTA,
  168 registers/thread,232448 shared bytes. Resource metadata is not a
  measurement of stall causes or achieved occupancy.
- H800 has114SM /50MiB L2. The user's20SM /32MiB PPU1.7 model still needs its
  own single-invocation measurement and model peak. No70% claim is transferred.

## Evidence

Protocol and fixed verdicts: [h800-perf-plan.md](h800-perf-plan.md).
Raw results/traces/logs are retained on both local and remote hosts under
`/workspace/flash-attn-ppu17-perf-20260926` (not copied into the source tree).

| Artifact | SHA256 |
|---|---|
| `baseline-v1/result.json` | `ca137cdad671910ec5c7c7121be068cdde2d0ba5bbf72e5b12d09c64f84eaadf` |
| `baseline-v1/kernel-timing-trace.json` | `53fd207052fae83c638952b504a8a4d044069d5abf45fced7f4fe2f6f5765399` |
| `baseline-v3/result.json` | `b1ee1e8948f51bc07847c0ba27f3d6e06527771e67a72306b251298e15210ea5` |
| `baseline-v3/kernel-timing-trace.json` | `e5d635872f779d871f703d747ed720c9ac0f97af80913b3d039b8b822a032db5` |

Local host tests:4 timing/denominator/inventory checks plus21 source contracts.
Extra/missing GPU kernels and a substituted reference kernel fail the timing
inventory check. The numerical oracle's3 CPU tests remain unchanged.

To repeat only when idle, using a **new** output directory:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /root/miniconda3/bin/python \
  /workspace/flash-attn-ppu17-perf-20260926/source/tools/bench_ppu17_hopper_control.py \
  --extension-dir /workspace/flash-attn-ppu17-control-e7ee864/python \
  --out /workspace/flash-attn-ppu17-perf-20260926/manual-confirmation
```

This repeated-launch tool is for physical Hopper only. The simulator must
continue to use `run_ppu17_forward.py` and its one-target-invocation contract.
