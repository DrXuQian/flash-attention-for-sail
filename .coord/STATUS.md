# PPU1.7 Hopper source integration

    updated-at: 2026-09-26 05:30:36 UTC
    working-on: S4096 four arms complete; S8192 a1-r3 after two prelaunch BUSY refusals
    blocked-on: no current foreign parent; no measurements retained from BUSY attempts
    last-commit: 19005e9 (sequence-scaling preregistration; 34 local tests PASS)
    branch: ppu17-hopper-source
    workspace: /workspace/flash-attn-ppu17-source
    scope: FP16/BF16 fixed forward, D64/128/256, causal/noncausal, GQA
    model-target: 20 SM / 32 MiB LLC; PPU1.7 model performance remains unmeasured

Completed task: official FA3 A/B on the same H800 and input, no kernel edits.
Order A1/B1-r2/B2/A2, each200 warmups+9x50 calls. Pooled kernel-only medians:
admitted port161.02995us/56.4386%, official164.23429us/55.3374%; ranges
overlap, so relative speed UNRESOLVED. Both fail the unchanged70% threshold.
Full-call event spans177.80608/173.62881us also overlap. All four runs pass
the unchanged CPU FP64 O/LSE oracle and within-arm replay; cross-arm output
AND LSE hashes are also identical on this fixture. H800 only, not PPU1.7.
Official extension SHA256=c0611358efe2a1511843ef2638befce6932b1d5855225c7d7d89ebf8583ef585.
Source a8aa52b1, official CUTLASS dc481792 (4.0.0); build-r2 completed with
unmodified sources. First build attempt and official b1 attempt were refused
by idle admission before compilation/GPU work respectively; B1-r2 began after
the foreign parent exited. Target kernels both168 registers, zero stack/spill,
grid114/block384/shared232448B. Static SASS counts3640/3672 are not a dynamic
verdict. Official API uses one counter-fill kernel per call; port uses H2D.
No clock/partition/power changes. GPU empty after final arm. Raw artifacts:
/workspace/flash-attn-official-h800-ab-20260926 (local and remote).
Summary and hashes: dev/ppu17/docs/h800-official-ab.md.

Previous performance task: H800 only, priority BF16 causal shape unchanged.
Numerator=68,753,031,168 useful causal FLOPs; fixed dense BF16 peak=756.5TFLOPS
(OEM1513 rating includes structural sparsity); 70-percent threshold=129.833us.
Keep uninstrumented complete-call event span separate from warmed CUPTI
kernel-only duration. Preserve the admitted binary; no kernel/config edits.
Contract: dev/ppu17/docs/h800-perf-plan.md. Completed runs baseline-v1/v3:
kernel-only warmed CUPTI medians168.810/163.354us (53.84/55.64% useful MFU);
uninstrumented full-call spans174.563/182.928us (52.06/49.68%). All sample
ranges miss the fixed70% target. CPU FP64 O/LSE + admitted fingerprint PASS
before/after timing. One intervening attempt was blocked by another task,
before any GPU work; it contributed no samples. No clocks/kernel/config
changed. Raw evidence: /workspace/flash-attn-ppu17-perf-20260926.
Interpretation/limits: dev/ppu17/docs/h800-causal-performance.md.

H800 preflight (2026-09-26, physical device, not PPU1.7 simulation):
NVIDIA H800 PCIe, SM90, 114 SM, 50 MiB L2; driver 595.71.05,
CUDA toolkit 12.8.93, Python 3.12.3, Torch 2.8.0+cu128, CXX11 ABI enabled.
Two initial samples showed no GPU process and zero memory/utilization. A later
sample caught another task's `single-launch` GPU process, so no FlashAttention
build, forward validation or timing was started. A gap between that task's
launches is not evidence that the task has finished. Respect idle-only admission
for the entire validation; do not terminate other processes or change MIG,
clocks, drivers, or system packages. H800 can validate the shared Hopper path,
not native PPU1.7-specific behavior or the 20-SM / 32-MiB model's performance.
No credentials are recorded in this checkout. Native PPU1.7 remains unverified.
Local preparation after resume: runner hardware mode records measured L2 and
runtime/input identities, without changing its single forward or fixed
CPU-reference tolerances. 21 host contracts + 3 CPU-reference tests PASS.
The exact backend snapshot for this control remains 023e82d, matching the
source/rebase validation; unrelated newer CUTLASS work is not substituted.
At 03:36:08 UTC both remote task parents had exited and the GPU was empty.
Build-only step started after a fresh guard checked GPU processes and other
Python/compiler parents. Remote build directory:
`/workspace/flash-attn-ppu17-control-e7ee864`. Source/backend archive hashes
match locally and remotely. The matched Python3.12/Torch2.8 extension built,
loaded and passed all 22 declared numerical cases with the unchanged FP64 CPU
oracle/tolerances. All per-case before/after occupancy and parent-task checks
were idle. No compute source was changed. The priority BF16 causal case
B1/S2048/H32/Hkv2/D256 has output max_abs=0.009159422 and LSE PASS.
One fresh-process trace contains exactly one actual SM90 attention kernel,
no GPU reference/init/prepare/combine, and the same output fingerprint as
the untraced process. Grid114/block384, 168 registers/thread, 232448 shared
bytes are H800 launch metadata, NOT PPU1.7 capacities or performance claims.
Evidence: dev/ppu17/results/h800-20260926; explanation:
dev/ppu17/docs/h800-validation.md. Binary SHA256:
a78aff6c443238cd888f53dc692bf1e017b9191cff137427677ff1da84640320.
H800 timing is now recorded separately above; native PPU1.7 and simulator
remain unverified. Do not repurpose the earlier single-call trace as timing.

Rebase parent: f056429 (latest DrXuQian/v2.8.2, including upstream 664597d).
Recovery branch: backup/ppu17-before-rebase-20260925 at a9e497a.
Initial rebased compile caught two HGGC-only declarations in the new shared
QSA header; they now use the existing runtime alias. Object compilation then
caught unguarded __ppu_read_firstlane/__ld_smem in its direct-index consumer.
Those legacy intrinsics are now outside the PPU1.7 parser path. QSA remains
unsupported on PPU1.7 and fails closed, without changing upstream legacy QSA.
Fresh validation: 19 host + 3 CPU tests PASS; 6/6 SM90 generated units / 12
bodies PASS; real host/internal link PASS; 5 real compile/link plants
EXPECTED_RED. Legacy PPU1.0 D128 BF16 compile PASS using upstream's updated
actlize submodule 9771215. Evidence: /workspace/flash-attn-ppu17-rebase-20260925.
That rebase checkpoint did not verify a matched extension load; the H800
control above now closes it. Native PPU1.7 / simulator remain NOT VERIFIED.

Historical, before rebase: local host contracts: 18 passed; CPU reference
tests: 3 passed. All six real
generated units / twelve bodies compiled and assembled with WGMMA + TMA
read/write. Shared-source hashes and real host/internal-link closure PASS.
Five real target/link negatives rejected the intended planted defects. Legacy
PPU1.0 D128 BF16 generated unit compiled on HGCC2.1.1. Simulation counter init
uses CPU zero + H2D rather than a GPU fill kernel. Native PPU1.7,
simulation numerics and performance are NOT VERIFIED.
The independent skill lives at /root/.codex/skills/ppu17-hopper-porting and
passed skill validation. No changes
to CUTLASS tree, admitted PPU1.0/1.5 compute bodies, or original worktree.
