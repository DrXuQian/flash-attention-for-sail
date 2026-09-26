# PPU1.7 Hopper source integration

    updated-at: 2026-09-26 03:24:58 UTC
    working-on: H800 read-only preflight complete; FlashAttention validation NOT STARTED
    blocked-on: another H800 validation task is active; user requires no concurrent tasks
    last-commit: 3d7c07c (unchanged validated source; this checkpoint is documentation-only)
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
Native PPU1.7 / simulator / matched extension load remain NOT VERIFIED.

Historical, before rebase: local host contracts: 18 passed; CPU reference
tests: 3 passed. All six real
generated units / twelve bodies compiled and assembled with WGMMA + TMA
read/write. Shared-source hashes and real host/internal-link closure PASS.
Five real target/link negatives rejected the intended planted defects. Legacy
PPU1.0 D128 BF16 generated unit compiled on HGCC2.1.1. Simulation counter init
uses CPU zero + H2D rather than a GPU fill kernel. Native PPU1.7, extension
runtime import, simulation numerics and performance are NOT VERIFIED.
The independent skill lives at /root/.codex/skills/ppu17-hopper-porting and
passed skill validation. No changes
to CUTLASS tree, admitted PPU1.0/1.5 compute bodies, or original worktree.
