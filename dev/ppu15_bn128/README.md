# PPU1.5 FA2: reduce KV-update frequency with BN128

This is a bounded, opt-in geometry experiment on v2.8.2 base
`3b78b2f0522d32aebbfbf76129c0e3db99f4b876`, actlize
`9771215b2119d19464c0663299885699d74985a8`.
**The installed package, shipping selector and all production kernel sources
are unchanged.** The small C ABI only launches existing `flash_fwd_kernel`
specializations; it does not reimplement attention.

## Scope and hypothesis

Exact target: **B1/Sq=Sk8398/Hq=Hkv56/Dq=Dv128**, contiguous BSHD,
non-causal, no dropout, scale `1/sqrt(128)`. Compile both FP16 and BF16,
keeping the same FP32 accumulation/softmax, AIU/SWZL and exp implementation.
The control is the production selection for this dense grid, with
`IsEvenMN=false`, `IsEvenK=true`, `Is_Q_in_regs=false`, `Share_Q_K_smem=false`.
No rescale-skipping or other softmax edit is combined with this experiment.

BN64 needs 132 KV blocks; BN128 needs 66. In the source-level work account,
output-rescale MUL falls 247,898,112 -> 123,002,880, and counted scalar
FP32 FMA/ADD/MUL falls about 26.1%. That is **not** a prediction that latency,
total dynamic instructions or sync-stall time falls 26.1%.
128x128/W8 preserves Q tiling but increases shared memory/warps. 64x128/W4
doubles Q tiles and may pay extra K/V delivery. This is a geometry comparison,
not isolation of BN from every other geometry axis.

## Local compiler result (not device performance)

HGCC `2.1.1-a5c56e`, `-arch=ppu_15`; compiler options are read directly from
the literal `hgcc_flags` in `setup.py`, not maintained in a second flag list.
Both dtypes produced these native allocations:

| BM x BN / warps | Grid CTAs | Dynamic shared | Native vregs/thread | Stack | Static instruction sites/body |
|---|---:|---:|---:|---:|---:|
| 128x64 / 4 (control) | 3696 | 65,536 B | 256 | 0 | 2943 |
| 128x128 / 8 | 3696 | 98,304 B | 192 | 0 | 2850 |
| 64x128 / 4 | 7392 | 81,920 B | 184 | 0 | 2822 |

All six bodies have the exact registered type and native
`v.mma.f32.f16.m16n16k16` or `v.mma.f32.bf16.m16n16k16`, zero stack and no
enabled private memory. Dynamic shared is passed by the launcher; a native
`shared_memory_size:0` field is NOT zero total shared usage.
Static sites are not comparable to ACU dynamic counts without loop/warp work.
Runtime occupancy is queried on the user's device, not inferred here.

The local compiler installation has missing `libhggc`/`libhggcrt1` runtime
libraries: **local final linking SKIP (missing SDK runtime); PPU correctness
and timing NOT_RUN**. Objects and native disassembly were checked locally.
The normal box script performs a real `--no-undefined` shared-library link;
it cannot accept a compile-only receipt as runnable evidence.

## Run on box

From this branch's repository root, with its pinned submodule:

```bash
git submodule update --init csrc/actlize
PPU_SDK=/usr/local/PPU_SDK bash tools/run_ppu15_bn128_box.sh
```

Default: both dtypes, three configs, CPU-oracle admission, then interleaved
7 x 20-launch event samples. Set `DTYPE=fp16` or `DTYPE=bf16` to match the
original profile exactly. No installation/replacement of `flash_attn` occurs.
Set `PPU_PEAK_TFLOPS` only if the device's applicable dense FP16/BF16 peak is
known; otherwise the script reports TFLOPS without guessing an MFU denominator.

For the same comparison plus three full ACU reports for one dtype:

```bash
DTYPE=bf16 PROFILE=1 PPU_SDK=/usr/local/PPU_SDK \
  bash tools/run_ppu15_bn128_box.sh
```

Use `ACU=/path/to/acu` if the site's profiler is separate from the compiler SDK.
Each report invocation contains exactly one attention launch. Reports use
`acu -o ... --set full`, not CSV. Profile durations do not replace uninstrumented
timing. `OUT` defaults to a new directory under `/workspace`, created with
`mkdir`; no `mktemp`, no exit command is pasted into the interactive shell.

The last line prints `result.json`; raw event samples, source/submodule/SDK/
binary hashes, device properties, runtime/driver versions, resource queries,
native dumps and logs are retained alongside it. Overlapping timing envelopes
are **UNRESOLVED**. No result automatically changes the production selector.

## Correctness and local negatives

`contract.json` fixes the numeric tolerances before any box data. For S1/65/129,
CPU FP64 checks every output using already-quantized Q/K/V. On S8398 it checks
all K for sampled Q boundary rows in every head, plus full-output finite checks
and every candidate output against the CPU-anchored control. LSE is checked
independently. Three repeats per config require bit-stable output and LSE;
post-timing hashes must match preflight. Different BN values are **not**
required to be mutually bit-identical, because online-softmax association changes.

```bash
python dev/ppu15_bn128/test_contract.py -v
```

Host gates include a wrong output, unwritten NaN, wrong native target/geometry,
positive stack, missing native MMA, and an overlapping/noisy timing winner.
The CPU stand-in test validates harness indexing only, not GPU numerics.

Compile-only access (no device required):

```bash
python dev/ppu15_bn128/build.py --sdk /path/to/PPU_SDK \
  --output /workspace/fa15-bn128-compile --compile-only
```

The build output directory must be new/empty. Missing compiler/dumper returns
SKIP/77; compile errors, changed dependency, wrong native type, or spills FAIL.
The untested 128x128/W4 geometry is deliberately absent: its two accumulator
fragments alone require 256 FP32 values/thread before other live registers.
