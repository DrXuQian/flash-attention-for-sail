# Resume

- current-task: standalone encoding hardening and default direct build completed locally on 2026-09-28
- migration-parent: ed150c9
- migration-implementation: d228fa3e72783c7063e0729f07d78a52d0f90afa
- migration-evidence: dev/ppu17/docs/cutlass43-migration.md and results/cutlass43-20260928/validation.json
- migration-results: both backends 6/6 units + 12 bodies + host/internal link;
  33 local tests PASS; 4 new compile mutations EXPECTED_RED; full 4.3 package
  link/import PASS on Python3.12/Torch2.9; 3.6 encoded kernel words unchanged12/12
- migration-boundary: no new device/simulator numeric or timing result;
  native PPU1.7 capable SDK/model remains unavailable

Prior 3.6 hardware baseline (not reattributed to the new 4.3 binary):

- parent: f056429 (rebased from f600591 on 2026-09-25)
- source-commit: e7ee864 (hardware runner provenance; compute source remains a3d25ff)
- worktree: /workspace/flash-attn-ppu17-source
- branch: ppu17-hopper-source
- working-on: prior sequence-scaling baseline complete; median64–65% at8K, about62% at16K; power cap observed
- blocked-on: native PPU1.7 verification only (SDK2.1.1 cannot target it)
- device-results: H800 22/22 numerical cases PASS; one traced target; fresh-process fingerprint STABLE
- validation-results: six generated units / twelve live forward bodies compiled
  and assembled on CUDA12.8 SM90a; real Torch host TU and generated definitions
  passed relocatable link; 19 host contracts + 3 CPU-reference tests passed;
  5 genuine build/link negative controls rejected the planted defects
- evidence: /workspace/flash-attn-ppu17-rebase-20260925/final/source-compile.json
- hardware-evidence: dev/ppu17/results/h800-20260926 (CPU FP64 reference, no timing)
- hardware-performance: dev/ppu17/docs/h800-causal-performance.md (two independent runs; no kernel edit)
- official-comparison: dev/ppu17/docs/h800-official-ab.md (A/B/B/A; same CPU fixture/oracle; both BELOW_70)
- sequence-scaling: dev/ppu17/docs/h800-sequence-scaling.md (12 new runs PASS; no kernel edit; relative speed UNRESOLVED)
- negative-evidence: /workspace/flash-attn-ppu17-rebase-20260925/negatives/negative-builds.json
- recovery: backup/ppu17-before-rebase-20260925 at a9e497a
- skill: /root/.codex/skills/ppu17-hopper-porting/SKILL.md

The original worktree's untracked files are user-owned and untouched.
The simulator runner uses one forward invocation and an optional CPU-only
oracle. The matched Python3.12/Torch2.8 extension built, loaded and ran on H800.
H800 warmed kernel-only medians are168.810/163.354us, useful causal MFU53.84/55.64%,
versus a129.833us threshold for70%. Full-call spans are separate. No native
PPU1.7 runtime result or model latency/MFU is claimed. That hardware has
114 SM / 50 MiB L2, not the user's 20 SM / 32 MiB model. Runtime admission on
the simulator remains unverified. Local runner checks now 21 host + 3 CPU PASS.
# Standalone handoff — 2026-09-28

Follow-up to user UTF8 error: ASCII version-macro parsing, binary tool logs,
escaped diagnostic display, and explicit failing-step paths. Actual error
exit remains a failure. PTX/disassembly now require `--inspect-codegen`; the
default only compiles/links. Both modes rebuilt locally,49unique contracts
PASS, direct-build kernel machine words unchanged2/2. The user's exact bad
input is not known from the one-line report. Evidence: standalone results
`build-encoding.json`. Device/model NOT_RUN.

User requests no wheel installation for B1/S1024/H56/Hkv56/D128 FP16
noncausal. `tools/build_ppu17_standalone.py` links the shipping generated
FP16/D128 unit with a plain C++ host application. No production kernel edits.
Build command and single-invocation simulation command: README, "No
installation: standalone FP16 S1024 executable".

Local CUTLASS4.3/CUDA12.8 real ELF link PASS, two native encoded kernel bodies
identical to release43 (including control words), seven new contracts PASS.
CPU-only self-test/describe succeed on the actual executable. Real missing
generated unit fails at link with `run_mha_fwd_` undefined; not an environment
SKIP. Actual simulator/device O/LSE and performance remain NOT_RUN.
Evidence: `results/standalone-20260928/validation.json`.
Artifacts: `/workspace/flash-attn-ppu17-causal-tune-20260926/standalone-20260928-v1`.
