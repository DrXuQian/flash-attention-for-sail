# Resume

- parent: f056429 (rebased from f600591 on 2026-09-25)
- source-commit: a3d25ff (validated rebased source and QSA compatibility closure)
- worktree: /workspace/flash-attn-ppu17-source
- branch: ppu17-hopper-source
- working-on: source handoff; native/simulator execution is the next evidence tier
- blocked-on: native PPU1.7 verification only (SDK2.1.1 cannot target it)
- device-results: NOT RUN
- validation-results: six generated units / twelve live forward bodies compiled
  and assembled on CUDA12.8 SM90a; real Torch host TU and generated definitions
  passed relocatable link; 19 host contracts + 3 CPU-reference tests passed;
  5 genuine build/link negative controls rejected the planted defects
- evidence: /workspace/flash-attn-ppu17-rebase-20260925/final/source-compile.json
- negative-evidence: /workspace/flash-attn-ppu17-rebase-20260925/negatives/negative-builds.json
- recovery: backup/ppu17-before-rebase-20260925 at a9e497a
- skill: /root/.codex/skills/ppu17-hopper-porting/SKILL.md

The original worktree's untracked files are user-owned and untouched.
The simulator runner uses one forward invocation and an optional CPU-only
oracle. No native PPU1.7 runtime result or latency/MFU is claimed. A matched
Python/Torch runtime extension import remains to be verified on the simulator.
