# Official FA3 / admitted port: same-device control

Registered 2026-09-26 UTC before this A/B. User authorized the physical H800
comparison. This is not a PPU1.7 simulator run or a kernel optimization.

## Frozen identities and work

- A: admitted extension SHA256
  `a78aff6c443238cd888f53dc692bf1e017b9191cff137427677ff1da84640320`;
  compute source `a3d25ff`, packaged source `e7ee864`, PPU CUTLASS3.6.0
  `023e82d03e80b4d5982f664925e15499033df314`. Do not rebuild or replace it.
- B: unmodified Dao-AILab/flash-attention tag `v2.8.3.post1`, commit
  `a8aa52b1ab3e9ca574c8a33b3f35afc017ffa2e2`; its official CUTLASS submodule
  `dc4817921edda44a549197ff3a9dcf5df0636e7b`. Build from its own setup.py.
  Narrow the build using upstream environment switches to BF16/D256 fixed
  forward on SM90, excluding backward, split, paged/append KV, local/softcap,
  FP16/FP8, varlen and other head dimensions. Keep upstream PackGQA/cluster
  compilation enabled; the measured call uses num_splits=1, pack_gqa=False,
  as does A. The official fixed-S2048 heuristic also selects unpacked GQA:
  both packed and unpacked lengths are exact multiples of BlockM=128.
- Both: real CUDA12.8.93, Torch2.8.0+cu128, same otherwise-idle H800 PCIe
  UUID. No clocks/power/partition/system-package changes. Respect other task
  parents, not just instantaneous zero GPU utilization.
- Same B1/S2048/Hq32/Hkv2/D256 BF16 BSHD causal inputs, CPU generator seed
  170020. Require all three input hashes to match the admitted fixture.
- Same CPU FP64 output and LSE reference, unchanged atol/rtol .02/.002.
  Require finite values and within-arm output replay stability, not raw-bit
  equality between independent implementations. Preserve A's admitted hash.

Useful causal work stays 68,753,031,168 FLOPs; dense BF16 denominator stays
756.5TF/s. The 70-percent threshold remains129.832935829us. Repeated prepared
inputs are warm/reused, not cold. No GPU reference.

## Measurement and interpretation

Use separate processes because both extensions register the same Torch
operator namespace. Order A/B/B/A, serial, with idle checks between phases.
Each process uses200 warmups and9 samples of50 calls, as in the original
H800 control. Invoke the same registered C++ forward operator kwargs in both
arms to avoid different Python wrapper overhead. Keep two timing roles:

1. Uninstrumented complete-call CUDA-event span (includes API setup and any
   auxiliary operation/submission gaps, excludes input upload and CPU oracle).
2. Separate warmed CUPTI trace:450 attention kernels, grouped9x50. Report
   attention-only time AND all auxiliary kernel/memset/copy inventory. The
   official unmodified API zeros its scheduling counter on GPU; A initializes
   on CPU+H2D for the single-launch simulation contract. Do not erase this
   difference or relabel full-call duration as attention-only duration.

No simultaneous timed workloads. A foreign GPU/task process, identity drift,
numerical failure, missing attention invocation or unrecognized auxiliary
kernel invalidates that run. Traces and all raw samples stay in
`/workspace/flash-attn-official-h800-ab-20260926`; commit only the scripts,
plan and hash-bound summary.

For each timing role compare both independent runs of each arm. Declare an
arm faster only when the combined per-arm sample ranges are disjoint;
otherwise UNRESOLVED. Apply the unchanged70-percent threshold independently.
If B clearly wins, the gap is between these source/backend/build/API bundles,
not yet proof of a particular cause. If they overlap, do not claim migration
loss or a hardware upper bound. A next causal experiment would isolate one
identified difference, not silently tune this reference.

Source inspection before timing: the two mainloop_fwd_sm90_tma_gmma_ws.hpp
files are byte-identical; both select tile128x80 for this shape. Record the
actual kernel/resource metadata rather than assuming generated code is equal.
