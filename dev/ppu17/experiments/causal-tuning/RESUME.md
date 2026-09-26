# Causal tuning resume

- updated-at: 2026-09-26 08:14:53 UTC
- result-commit: 9d1d5f4
- parent: ed150c9
- worktree: /workspace/flash-attn-ppu17-causal-tune-source
- branch: ppu17-causal-tuning
- working-on: complete; keep existing N80+LPT default
- completed: 24/24 primary CPU FP64+replay; 9/9 real API negatives;42 host+3 CPU tests;
  6/6 default TUs (12 identical machine-code bodies); 3/3 candidate Torch2.8 link/load/run
- builds: /workspace/flash-attn-ppu17-causal-tune-20260926/build-r2
- verdict: N64+LPT UNRESOLVED; single-tile loses S2048; no70% result; no promotion
- report: docs/results.md; all sample/identity receipts in summary.json
- preparation-failures: prelaunch BUSY; missing raw FlashInfer package data/submodules;
  ninja missing from PATH. No failed attempt contributes a timing sample.
- blocked-on: none on H800; native PPU1.7 SDK/model unavailable
- artifacts: /workspace/flash-attn-ppu17-causal-tune-20260926
- no PPU1.7 native/model performance claim; H800 only
