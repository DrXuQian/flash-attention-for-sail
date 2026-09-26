# H800 correctness control, 2026-09-26 UTC

**22/22 declared cases PASS**, plus one separately traced, fresh-process
forward. This closes the matched Python/Torch extension-load and physical
Hopper numerical-control gaps. It does **not** establish native PPU1.7
execution, simulator correctness, latency or MFU. Compute code was unchanged.

## Bound source and environment

- Source: `e7ee8643ef77fbb389d67679ba319a4293633b97`; kernel source is the
  validated rebased implementation from `a3d25ff`.
- PPU CUTLASS: `023e82d03e80b4d5982f664925e15499033df314`, not the unrelated
  newer working-tree revision. Both repositories were transferred as exact
  committed snapshots; archive hashes matched on both hosts.
- NVIDIA H800 PCIe, SM90, **114 SM / 50 MiB L2**, driver 595.71.05;
  CUDA12.8.93, Python3.12.3, Torch2.8.0+cu128, CXX11 ABI enabled.
- CUDA source/simulation-input build mode, real SM90a compiler target probe;
  all six generated units assembled and linked into the loadable extension.
- Extension SHA256:
  `a78aff6c443238cd888f53dc692bf1e017b9191cff137427677ff1da84640320`.

Other remote tasks were observed and left untouched. The build began only
after they exited. The numerical controller checked GPU process occupancy
and other Python/compiler task parents before and after every case; all
22 case receipts are empty at both boundaries. No clocks, MIG, driver or
system packages were changed. No performance run was made.

## Declared numerical suite

Each case runs in a fresh process, with **one forward invocation**, CPU-created
inputs, and a row-blocked CPU FP64 reference. The unchanged tolerances are
BF16 `atol=rtol=0.02`, FP16 `atol=rtol=0.002`, and LSE
`atol=rtol=0.002`, with nonfinite checks. This is not RAW-BIT equality to FP64.

| Cases | Geometry | Coverage |
|---:|---|---|
| 12 | B1/S65/H4/Hkv2, D64/128/256 | BF16/FP16 × causal/noncausal; every admitted generated specialization, GQA and tail |
| 4 | B1/S2048/H32/Hkv2/D256 | BF16/FP16 × causal/noncausal; priority shape and repeated pipeline-stage reuse |
| 2 | B2/S129/H6/Hkv2/D128 | BF16 causal/noncausal; batch, GQA ratio3 and tail |
| 2 | B1/S127/H4/Hkv4/D128 | FP16 causal/noncausal; ordinary MHA and tail |
| 2 | B1/S1/H2/Hkv1/D64 | BF16 causal/noncausal; single-token boundary, exact output equality |

Priority-shape maximum output absolute errors:

| Dtype | Causal | Noncausal |
|---|---:|---:|
| BF16 | 0.009159422 | 0.0009205404 |
| FP16 | 0.001267713 | 0.0001206065 |

Output and LSE pass in every case. These are checks of this declared finite
suite, not an exhaustive proof over all inputs/shapes.

## Actual launch trace

A separate process ran the priority BF16 causal case once under Torch/CUPTI
tracing, with the same CPU reference. The trace contains exactly **one** GPU
kernel: the real `FlashAttnFwdSm90` / `DynamicPersistentTileScheduler` entry.
No GPU reference, initialization, prepare or combine kernel appeared.
The output fingerprint matches the earlier untraced process exactly.

Recorded launch: grid `(114,1,1)`, block `(384,1,1)`; kernel metadata reports
168 registers/thread and 232448 shared bytes. These are the H800 specialization's
resource/launch metadata, not PPU1.7 capacity claims or achieved occupancy/MFU.
Trace durations and its estimated occupancy field are **not performance evidence**.

The user model remains **20 SM / 32 MiB LLC**. Neither passing this H800
control nor setting its grid to 20 substitutes for that model.

## Evidence and rerun

Committed raw case reports, manifest, controller, build log and trace receipt:
[`../results/h800-20260926`](../results/h800-20260926).
The receipt audit checks all 22 IDs/shapes against the prewritten manifest,
the controller/manifest hashes, one common binary hash, and idle boundaries.
Both planted admission failures (another GPU job / another Python task parent)
were rejected; the empty-host control passed. Local runner tests: 21 host + 3 CPU.

Remote directory: `/workspace/flash-attn-ppu17-control-e7ee864`.
Local binary/full trace: `/workspace/flash-attn-ppu17-h800-20260926/artifacts`.
Full trace SHA256:
`81e94db3ba70bdbcb121d80442520edb4865cf7fdd67db889bce8c65149bf4e3`.
Only run the following after other tasks have finished; it performs one
forward and CPU validation, not a benchmark:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /root/miniconda3/bin/python \
  /workspace/flash-attn-ppu17-control-e7ee864/source/tools/run_ppu17_forward.py \
  --extension-dir /workspace/flash-attn-ppu17-control-e7ee864/python \
  --hardware-validation --expected-sms 114 --verify \
  --batch 1 --seqlen 2048 --heads 32 --kv-heads 2 --head-dim 256 --dtype bf16 \
  --output /workspace/flash-attn-ppu17-control-e7ee864/priority-rerun.json
```
