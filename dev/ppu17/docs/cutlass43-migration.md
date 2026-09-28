# PPU1.7 forward: CUTLASS3.6 / 4.3 compatibility

Source parent: `ed150c9`, branch `ppu17-hopper-source`. This is a dependency
migration, not an attention algorithm or performance-tuning change.

## Authorities and scope

- PPU CUTLASS3.6: immutable `023e82d03e80b4d5982f664925e15499033df314` snapshot.
- PPU CUTLASS4.3: `v4.3.0_ppu_dev-0eb6a3b218977892e2b6cd30d59880c440e54fd6` archive.
- CUDA source/assembly checks: CUDA12.8.93, real `compute_90a/sm_90a` target.
- Same six generated units: BF16/FP16, D64/128/256; causal and noncausal
  bodies must each contain WGMMA and TMA read/write. Fixed forward/GQA only.
- Native PPU1.7 SDK/model and new device numeric/performance results are not
  implied by these checks. Earlier H800 numbers belong to the 3.6 binary.

## Changes

| Boundary | PPU3.6 | PPU4.3 |
|---|---|---|
| Epilogue selector | `StrideO, Element` | Adds actual `EpilogueTile_MN` |
| Selected output store | `SM90_U32x4_STSM_N` | Same for all six admitted types |
| Cluster1 pipeline | Native backend already signals per warpgroup | FA no-cluster wrapper overrides the per-thread default |
| EMPTY count, 256 consumers | 2 | 2, with matching 128-thread release stride |
| Runtime identity | `cutlass36-sm90-forward-v1` | `cutlass43-sm90-forward-v1` |

The 4.3 base constructor receives `init_barriers=false, init_masks=false`;
the wrapper owns both initialization and consumer release. Never change only
one half. The selected path admits cluster1 and complete consumer warpgroups.
QK/PV, online softmax, TMA descriptors, work scheduler and output layout are
unchanged by the adapter. No claim is made that different CUTLASS versions
must produce identical code or performance.

The Python build admits precisely 3.6.0 and 4.3.0 PPU roots, and emits an
expected-version macro checked against the compiler's real version header.
This catches a build validating 3.6 but actually including 4.3 (or vice versa).
It is a version check, not a proof that arbitrary forks sharing a version are
identical; evidence additionally hashes the whole selected backend include tree.

## Constructive checks

`cutlass_compatibility.cu` instantiates the production epilogue, selects the
real pipeline, enumerates signalling threads for all 1..7 complete consumer
cohorts behind a loader warpgroup, and compiles a real two-stage constructor
and release receipt. The checker resolves the emitted `mbarrier.init` count
operands to `[1,1,2,2]`; it does not just search for an incidental literal 2.
The receipt is assembled, never launched.

The CUTLASS4.3 negative checker copies the actual headers into isolated build
directories, applies exactly one mutation per copy, and requires failure:

1. Reintroduce the two-argument selector (the user's original compile error).
2. Substitute the default per-thread pipeline for the admitted wrapper.
3. Change release stride from 128 to 64 while retaining the expected count.
4. Select expected version360 but compile real version430 headers.

Additional existing target/legacy-macro/cross-TU controls still apply. A
missing generated unit is a failure, not a reduced denominator. Runtime
`--expected-cutlass` rejects loading the other admitted backend by accident.

## Results

Completed 2026-09-28. Machine-readable hashes and outcomes:
[validation.json](../results/cutlass43-20260928/validation.json).

- Local contracts: **26 source/build/identity + 3 CPU-reference + 4 timing =
  33 PASS, 0 FAIL**. CPU-reference tests are tests of the oracle, not a claim
  that the new kernel was executed.
- Each backend: **6/6 actual generated units, 12/12 live forward bodies**,
  PTX inspection + assembled objects + real Torch host API/internal link PASS.
- Each backend: six output-store type assertions, 1..7 consumer-cohort
  enumeration, two-stage compiled FULL/EMPTY barrier count receipt PASS.
- CUTLASS4.3: **4/4 new actual-header/include mutations EXPECTED_RED**.
- Existing CUDA-is-not-native / SM80 / legacy-macro / missing-generated-unit
  controls: **4/4 EXPECTED_RED on each backend**. The optional legacy SDK
  target-fallback control is SKIP (`--legacy-nvcc` not supplied), not PASS.
- CUTLASS3.6 backward compatibility: **12/12 kernel encoded instruction streams,
  including control words, match the admitted pre-migration 3.6 build**.
- Full 4.3 extension: all eight package objects compiled, shared-library link
  and real Python import PASS on **CPython3.12.14 / Torch2.9.0+cu128**.
  Both `_C` and the public Python interface imported; actual extension identity
  is `cutlass43-sm90-forward-v1`. Requiring 3.6 on that extension is EXPECTED_RED.
- Native PPU1.7 compile/model: **SKIP**, capable SDK/model unavailable.
  Device numerical tests and performance: **NOT_RUN**. No H800 or simulator
  run was started, and the old 3.6 performance numbers are not 4.3 results.

Library SHA256:
`bc5276a912ee30a78d82aadd7babb81054d8269e4635d6c87350837c3c79f48d`.
The initial package link failed because this container had no default
`-lcuda` search path. The exact setup.py linker command was replayed with
the installed `/usr/local/cuda-12.8/compat` directory, with all eight object
hashes checked before/after linking. Import used the real CUDA compat
user-space library, **not a stub runtime**; no GPU device nodes are present.
The first exploratory source check was deliberately not admitted: it correctly
rejected a source change during compilation. `release36` and `release43`
are fresh, hash-stable complete runs after the final source/checker edits.

Artifact root:
`/workspace/flash-attn-ppu17-causal-tune-20260926/cutlass43-20260928` (data disk).
Reproduce the source and mutation gates using the commands in the README.
Do not infer native-PPU correctness from the successful CUDA SM90a build.
