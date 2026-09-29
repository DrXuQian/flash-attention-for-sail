# Tail Q split resume

- updated-at: 2026-09-29 05:19:03 UTC
- branch: ppu17-tail-qsplit
- parent: d19d27a
- working-on: local complete; T01 ready for one simulator invocation; T02 rejected and removed from live source
- blocked-on: native PPU simulation unavailable locally
- artifacts: /workspace/fa17-tail-qsplit-20260929
- baseline: C07 user-reported240412cycles; raw report/numerics not uploaded
- math: original exp/softmax; no changes permitted in this branch
- next: compare T01 full CPU O/LSE and full cycles to the identity-bound C07 report; do not compose with exp
- local: 68 PASS /4 SKIP /0 FAIL;7340032actual CuTe output owners each inventory;5plants red each
- T01: /workspace/fa17-tail-qsplit-20260929/t01-final/flash_attn_ppu17_s1024_fp16
- T01-sha256: 24dc2a0e181b7fc9aa69a250a408a7b6466f582462bde9abb418ce62782af122
- T02: corrected one-kernel wrapper32/32B CUDA spill versus12/12B budget; replay patch retained at parent d19d27a
- scope-note: T01 uniform M64 changes all Q tasks and doubles KV task fills; not a tail-only attribution or promised win
