# PPU1.7 Hopper source integration

    updated-at: 2026-09-24 09:30:27 UTC
    working-on: source wiring and independent Hopper-to-PPU1.7 skill complete; source handoff
    blocked-on: native PPU1.7 SDK/device only; simulation input build is separate
    last-commit: f600591 (parent; integration not committed yet)
    branch: ppu17-hopper-source
    workspace: /workspace/flash-attn-ppu17-source
    scope: FP16/BF16 fixed forward, D64/128/256, causal/noncausal, GQA
    model-target: 20 SM / 32 MiB LLC; no hardware performance result

Local host contracts: 18 passed; CPU reference tests: 3 passed. All six real
generated units / twelve bodies compiled and assembled with WGMMA + TMA
read/write. Shared-source hashes and real host/internal-link closure PASS.
Five real target/link negatives rejected the intended planted defects. Legacy
PPU1.0 D128 BF16 generated unit compiled on HGCC2.1.1. Simulation counter init
uses CPU zero + H2D rather than a GPU fill kernel. Native PPU1.7, extension
runtime import, simulation numerics and performance are NOT VERIFIED.
The independent skill lives at /root/.codex/skills/ppu17-hopper-porting and
passed skill validation. No changes
to CUTLASS tree, admitted PPU1.0/1.5 compute bodies, or original worktree.
