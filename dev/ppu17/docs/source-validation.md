# Source admission, 2026-09-24 UTC

This is the **pre-rebase historical receipt**, not validation of a later
source tree. The 2026-09-25 rebase and fresh checks are recorded separately
in `rebase-validation.md`.

Parent FA commit: `f600591`. PPU CUTLASS3.6 backend:
`023e82d03e80b4d5982f664925e15499033df314`. No backend/submodule update.

| Check | Result | Boundary |
|---|---|---|
| Build/manifest/body/one-call contracts | 18 PASS | Host tests, no device execution |
| CPU FP64 oracle | 3 PASS | Closed-form uniform/prefix GQA, literal nonuniform scalar reference, non-CPU negative |
| Six actual generated FP16/BF16 D64/128/256 units | 6/6 PASS | CUDA12.8 SM90a PTX **and** assembly; each has causal/noncausal WGMMA/TMA read+write |
| Real host dispatcher + all generated definitions | PASS | Relocatable link; not a loadable Torch extension certificate |
| Compile/link plants | 5 EXPECTED_RED | CUDA mislabeled native, SM80 mislabeled SM90a, legacy USE_PPU, missing D256 BF16 TU, SDK2.1.1 silent target fallback |
| Same runtime boundary without planted legacy macro | PASS | Positive counterpart of the macro plant |
| Legacy PPU1.0 D128 BF16 generated TU | PASS | HGCC2.1.1; compile control only, not a full legacy numerical regression |
| Native PPU1.7 compile | SKIP | Local SDK2.1.1-a5c56e cannot select the required target |
| Matched Python/Torch extension load | NOT_RUN | Local host check used Torch2.9 CUDA headers and Python3.10 headers; no ABI-matched wheel delivered |
| Simulator correctness / performance | NOT_RUN | No simulator available in this local run |

The source check binds the shared Hopper headers, build/runner code and backend
include-tree hashes, not just a generated TU containing one template line.
Checks live in `tools/check_ppu17_source.py`,
`tools/check_ppu17_negative_builds.py`, and the two `tests/test_ppu17_*` files.
Commands are in `dev/ppu17/README.md`.

Local artifacts (not a shipped runtime):

- `/workspace/flash-attn-ppu17-evidence/verified-with-host/source-compile.json`
  SHA256 `8b83dd07b9d65960c0a7972ba94f2603229bb07d422ceb4d8f42cbd837da6ad2`.
- `/workspace/flash-attn-ppu17-evidence/negative-builds/negative-builds.json`
  SHA256 `d750addd25d406b869ccdc30be857ce7cf130ba347394f092592279ee77ade4a`.
- Legacy control object
  `/workspace/flash-attn-ppu17-evidence/legacy-control/hdim128.o`, SHA256
  `84759efeb7163a745b28a18b11f848cd7021cfbd08cd2ff4f71e3af75794e572`.

## Single-launch simulator input

The runner performs one forward invocation with `num_splits=1`,
`pack_gqa=False` and inference-only execution. Inputs and scheduler counter
initialization are CPU-side followed by copies. Optional verification is a
CPU FP64 row-blocked attention reference, including for S2048. No warmup,
GPU reference, retry or timing loop. The simulator must still confirm the
actual runtime kernel symbol/count; a host source test cannot certify that
trace. `--expected-sms 20` checks runtime identity, not physical partitioning.

No time/MFU or physical PPU correctness claim follows from this document.

## Separate skill

Installed locally at `/root/.codex/skills/ppu17-hopper-porting/SKILL.md`:
mapping, debug/simulation and performance references, with automatic discovery
metadata and explicit `$ppu17-hopper-porting` invocation. Skill validation
passed. It does not change the library's default route or performance policy.
