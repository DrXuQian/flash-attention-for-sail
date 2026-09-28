# Resume

- updated-at: 2026-09-28 23:48:27 UTC
- parent: 4b26264
- branch: ppu17-softmax-overlap
- working-on: C03 simulation rejected: +7.58047% cycles, 23 spilled private slots
- last-commit: 9733223 (before this report/checker checkpoint; kernel accdefc)
- blocked-on: application numerical log and binary/model identity not uploaded
- input-report: /workspace/fa17-softmax-overlap-20260929/c03-simulation-3158e2fa/perfstatistics.json
- plan: docs/plan.md
- default-route: unchanged
- local-results: simulation-result.json (58 PASS / 4 SKIP / 0 FAIL); historical compile checkpoint in results.json
- actual-candidate-elf: /workspace/fa17-softmax-overlap-20260929/candidate/flash_attn_ppu17_s1024_fp16
- control: /workspace/fa17-softmax-overlap-20260929/control
- next: retain original default; reduce softmax temporary live range before any new overlap candidate
- candidate-device-numerics: NOT_PROVIDED in uploaded report
- candidate-model-performance: 266011 cycles / 68.9824% useful MFU; REJECTED
- evidence: docs/simulation-verdict.md; simulation-result.json
- raw-control-report: overwritten by new upload; saved extracted baseline retained, full replay needs original
