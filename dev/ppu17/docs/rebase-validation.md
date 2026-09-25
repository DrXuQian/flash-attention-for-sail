# Rebase validation, 2026-09-25 UTC

Requested rebase of `ppu17-hopper-source` onto `DrXuQian/v2.8.2` at
`f056429cc84cbecbae91f39c672f80b568c2c26d`, which includes upstream
`664597d`. Original tip `a9e497a` is retained locally as
`backup/ppu17-before-rebase-20260925`. Neither remote base branch nor the
original working directory's user files were changed.

## Resolution and compatibility changes

- Retained upstream's `Is_QSA=false` template parameter and legacy QSA
  implementation; retained PPU1.7 `FlashStream` in declarations/definitions.
- PPU1.7 explicitly rejects QSA instantiations, `qsa_allow_aiu` and requests
  to enable QSA at build time. Its fixed dense forward scope did not expand.
- Initial real compile failed on the new shared `qsa/dispatch.h` HGGC types.
  Declarations now use the existing runtime aliases (still HGGC on legacy).
- Object compilation then rejected `__ppu_read_firstlane` / `__ld_smem` in
  the new QSA direct-index consumer even though QSA was not selected. Those
  bodies remain unchanged on legacy targets; PPU1.7 has an explicit dependent
  rejection instead of parsing an unsupported intrinsic or making a no-op.
- Source receipts now hash nested shared headers, including `hopper/qsa/`.
- Kept the new base's actlize gitlink `9771215` unchanged. The explicitly
  selected PPU1.7 CUTLASS3.6 tree remains `023e82d`; these are different build
  authorities, not interchangeable headers with the same version label.

## Fresh checks (not copied from the old commit)

| Check | Result |
|---|---|
| Host contracts, including explicit QSA switch negative | 19 PASS |
| Independent CPU reference checks | 3 PASS |
| Actual SM90 FP16/BF16 D64/128/256 units, PTX + object assembly | 6/6 PASS, 12 live causal/noncausal bodies |
| Real Torch host dispatcher + generated-definition relocatable link | PASS |
| Wrong target, native mislabel, old macro, missing TU, old SDK fallback | 5 EXPECTED_RED; unplanted boundary PASS |
| Legacy PPU1.0 D128 BF16 unit with updated submodule, HGCC2.1.1 | PASS (compile only) |
| Native PPU1.7 SDK compile | SKIP: local capable SDK unavailable |
| Matched Torch extension load, simulator correctness/performance | NOT_RUN |

Each dtype retains static WGMMA/TMA-load/TMA-store sites per two-entry unit:
D64 = 68/18/2, D128 = 86/36/4, D256 = 105/72/8. This is a source-body
sanity check, not a dynamic instruction, numeric or performance verdict.
The one-forward-call / CPU-only-reference simulator contract remains unchanged.

Artifacts under `/workspace/flash-attn-ppu17-rebase-20260925/`:

- `final/source-compile.json`:
  `0f6e368a77627bbb827e24c62cdc304e79c156c6d0edae52b558069cd3745258`.
- `negatives/negative-builds.json`:
  `d750addd25d406b869ccdc30be857ce7cf130ba347394f092592279ee77ade4a`.
- `legacy-control/hdim128.o`:
  `9c7807c0934386ff7cf8b022393e98d7fccbb55be25baf3a41c0e2129c0c7089`.

Build logs from the two failed intermediate states are retained in `initial/`
and `verified/`; only `final/` is the successful source receipt.
