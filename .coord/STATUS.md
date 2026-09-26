# PPU1.7 Hopper source integration

    updated-at: 2026-09-26 03:47:35 UTC
    working-on: H800 control complete: 22/22 CPU-reference cases + one-kernel trace PASS; publishing evidence
    blocked-on: none for H800 control; native PPU1.7 SDK/model still unavailable
    last-commit: e7ee864 (hardware evidence label and input identity; compute source unchanged)
    branch: ppu17-hopper-source
    workspace: /workspace/flash-attn-ppu17-source
    scope: FP16/BF16 fixed forward, D64/128/256, causal/noncausal, GQA
    model-target: 20 SM / 32 MiB LLC; no hardware performance result

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
H800 timing/MFU NOT_MEASURED; native PPU1.7 and simulator remain unverified.

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
