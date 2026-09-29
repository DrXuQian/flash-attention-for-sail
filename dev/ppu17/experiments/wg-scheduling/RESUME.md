# WG scheduling resume

- updated-at: 2026-09-29 08:22:41 UTC
- last-commit: b1ee1c6 (preceding handoff; tested implementation4ee8db7)
- parent: d19d27a
- branch: ppu17-wg-scheduling
- working-on: W01 user reports240325cycles vs C07 240412; tiny decrease, not promoted; no new kernel edits
- blocked-on: current raw reports/binary/model binding and W01 numerical verdict not supplied
- artifacts: /workspace/fa17-wg-scheduling-20260929
- baseline: C07 user-reported240412cycles; raw report identities not supplied
- scope: Q128/KV128 original math, two math WGs, grid40; no exp/cache/tail changes
- next: inspect existing W01/C07 reports if supplied; distinguish optional sync from shifted TMA/GMMA waits; no rerun requested
- executable: /workspace/fa17-wg-scheduling-20260929/candidate/flash_attn_ppu17_s1024_fp16
- executable-sha256: f47d49f3827e7e61c9b43c5c3daa017e879049e09f9efa405695ce40bf6fb746
- evidence: results.json; raw logs under /workspace/fa17-wg-scheduling-20260929/validation-complete
- native-PPU: lowering/spill unverified; simulator-numerics: NOT_PROVIDED; performance: USER-REPORTED240325cycles
