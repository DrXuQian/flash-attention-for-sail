# C07 resume

- updated-at: 2026-09-29 03:21:01 UTC
- branch: ppu17-softmax-overlap
- parent: 454298e
- worktree: /workspace/flash-attn-ppu17-softmax-overlap-source
- plan: docs/plan.md (registered before implementation)
- candidate: original softmax/pipeline, opt-in KV176->128 only
- artifact-root: /workspace/fa17-kv128-20260929
- local-status: 63 PASS /4 SKIP /0 FAIL; results.json
- default: encoded 2/2 unchanged; causal opt-in unchanged
- resources: CUDA registers168/stack8B/spill stores12B/loads12B, unchanged
- native-PPU/model-numerics/model-performance: NOT_RUN; tools unavailable locally
- next: one user simulation invocation, CPU O/LSE verification and report readback
- incumbent: 247267 cycles,74.211601% useful MFU, original pipeline/KV176
- previous-hypothesis: partial overlap closed; C04/C05 schedule rejects, C06 spill reject
- promotion: none; candidate is experimental, not a routing rule
