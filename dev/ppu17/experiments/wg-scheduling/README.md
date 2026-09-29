# W01: independent math-WG issue, Q128/KV128

Branch `ppu17-wg-scheduling`, C07 parent `d19d27a`. One optional scheduling
policy change, **not** a Q64/tail/cache/exp experiment. No default promotion.
Native PPU1.7 performance and numerical results are pending.

Local final validation: **67 PASS /4 SKIP /0 FAIL**. Three skipped Python
oracle tests need unavailable Torch; the fourth needs the overwritten old
baseline raw report. The standalone C++ CPU oracle and actual host-only ELF
paths pass. See [results.json](results.json) for source/backend/ELF hashes,
real type and codegen negatives. These are not GPU numerical results.

## What changes

`--independent-wg` selects only the admitted FP16/noncausal Q128/KV128,
two-stage, cluster1 specialization. Both math WGs independently check K/V
readiness instead of WG1 relying on WG0's pingpong signal. All actual data
barriers, stage-release counts, GMMA completion waits, math and output
ownership remain intact. Keep384threads, two math WGs and grid40.

| Local CUDA12.8 control (not native PPU measurement) | C07 | W01 |
|---|---:|---:|
| Static instruction sites |2344|2336|
| Optional pingpong named-barrier sites |3|0|
| Data/initialization named-barrier sites |7|7|
| GMMA completion-wait sites |4|4|
| Matrix instruction sites (m64n128k16) |32|32|
| Registers / stack / spill stores / spill loads |168 /8 /12 /12|168 /8 /12 /12|

The compiler also moves EX2 sites:13/53 before/after steady PV wait0 becomes
23/43, with the same total66. The source exp/softmax is unchanged, but do not
claim identical machine instruction scheduling outside the removed barriers.
This CUDA movement is not proof of the PPU compiler's behavior or a speedup.

The flag-disabled encoded C07 bodies match2/2, and the candidate causal body
is unchanged. The actual-type test checks both WG readiness predicates,
QueryEmpty288, consumer-release arrivals2, shared166912B and role register
budgets24/240. A leader-only readiness mutation must fail its WG1 assertion.

## Build and give ONE application invocation to the simulator

Use the same CUDA/CUTLASS/model inputs as the240412cycle C07 run. This script
does not install Python packages, invoke a GPU, or require Torch. Use a NEW
output directory; substitute your already-used CUTLASS/CUDA paths if needed.

```bash
python tools/build_ppu17_standalone.py \
  --cutlass /workspace/flash-attention-for-sail/csrc/cutlass3 \
  --cuda-home /usr/local/cuda-12.8 \
  --out /workspace/fa17-wg-independent-w01 \
  --kv-tile128 --independent-wg
```

Application command for the same40-SM model:

```bash
/workspace/fa17-wg-independent-w01/flash_attn_ppu17_s1024_fp16 \
  --expected-sms 40 --verify
```

The initial description must print `tile_m:128`, `tile_n:128` and
`wg_scheduling:"independent"`. It must NOT be the Q64 binary. Exactly one
target invocation, no warmup/replay/helpers. Full O/LSE reference runs on CPU
with unchanged atol/rtol0.002, after the target finishes. Keep stdout,
build.json and the simulator report together; never overwrite C07's report.

The same builder without `--independent-wg` is the C07 control. There is no
need to resimulate it merely to change a label; bind the existing report to
its actual binary/model instead. User-reported C07 cycles240412 remain the
provisional comparator until that report binding is supplied.

## Fixed verdict

Correct CPU O/LSE, no new native PPU spills and fewer complete cycles than
the same-model C07 admit a win. Equal/slower cycles reject this candidate even
if sync stalls fall.229376cycles/80% useful MFU is only an aspiration. Model
denominator remains40*4096FLOP/cycle; useful FLOPs30064771072. Report TMA
waits, tensor/SFU issue and removed sync as explanations, not additive wall
times. This experiment cannot establish a cache or tail improvement.

## Local replay

Build control/candidate with `--inspect-codegen` into separate new directories,
then use `validate.py --help`. Its `--parent` is an immutable C07 build;
it checks actual compiled policy/types, protected source, exact machine
words, codegen/resources, planted omissions and the full local test suite.
The validator never runs a GPU kernel. Unavailable native PPU/model or Python
capability remains explicitly unverified/SKIP, not a numerical PASS.

Local final candidate:
```
/workspace/fa17-wg-scheduling-20260929/candidate/flash_attn_ppu17_s1024_fp16
sha256=f47d49f3827e7e61c9b43c5c3daa017e879049e09f9efa405695ce40bf6fb746
```
