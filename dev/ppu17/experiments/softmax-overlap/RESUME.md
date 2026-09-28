# Resume

- updated-at: 2026-09-28 22:33:04 UTC
- parent: 4b26264
- branch: ppu17-softmax-overlap
- working-on: local C03 admission complete; opt-in user simulator handoff
- blocked-on: PPU1.7 simulator/native toolchain not available locally
- input-report: /root/perfstatistics.json
- plan: docs/plan.md
- default-route: unchanged
- local-results: results.json (56 PASS / 3 missing-Torch SKIP / 0 FAIL)
- actual-candidate-elf: /workspace/fa17-softmax-overlap-20260929/candidate/flash_attn_ppu17_s1024_fp16
- control: /workspace/fa17-softmax-overlap-20260929/control
- next: user executes one target with --expected-sms 40 --verify; compare report schedule and complete cycles
- candidate-device-numerics: NOT_RUN
- candidate-model-performance: NOT_RUN
