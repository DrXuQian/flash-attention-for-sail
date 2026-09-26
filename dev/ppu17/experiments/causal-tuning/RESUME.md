# Causal tuning resume

- updated-at: 2026-09-26 07:35:42 UTC
- parent: ed150c9
- worktree: /workspace/flash-attn-ppu17-causal-tune-source
- branch: ppu17-causal-tuning
- working-on: FlashInfer JIT, then same-input reference / 2x2 measurements
- completed: host/CPU contracts; 6/6 default TUs; 3/3 candidate Torch2.8 link/load
- builds: /workspace/flash-attn-ppu17-causal-tune-20260926/build-r2
- first-result: cuDNN S2048 CPU FP64 O/LSE PASS;205.586us kernel, no confirmation yet
- preparation-failures: prelaunch BUSY; missing raw FlashInfer package data/submodules;
  ninja missing from PATH. No failed attempt contributes a timing sample.
- blocked-on: none on H800; native PPU1.7 SDK/model unavailable
- artifacts: /workspace/flash-attn-ppu17-causal-tune-20260926
- no candidate performance or promotion yet
