# PPU1.5: interleave softmax query rows (option 1 only)

An opt-in experiment, **not a default change**. Source parent: `b9fc4ca`;
shipping FA2 source `3b78b2f`, actlize `9771215`. The versioned
[contract](contract.json) fixes the workload and admission before device data.

## One change

`B=1, S=8398, H=Hkv=56, D=Dv=128`, noncausal, no dropout,
scale `1/sqrt(128)`, contiguous BSHD. Both arms use **BM128/BN64/4 warps**.
FP16 and BF16 are independent comparisons; never combine their timings.

Original: process a row's score EXPs, then move to the next row.
Candidate: traverse columns, interleaving the four independent query rows
owned by a thread. Sum traversal also interleaves rows, preserving each row's
left-to-right order. The actual device iterator is host-tested too.

Hardware EXP2, FMA expression/rounding, FP32 sums, row max, O rescaling,
AIU/SWZL loads, MMA, P conversion, masks, geometry and layouts are unchanged.
There is no full extra score fragment. This does **not** include software
EXP, packed arithmetic, BN128, early-P conversion, or register cap tuning.

`FLASHATTN_PPU15_SOFTMAX_ROW_INTERLEAVE` is defined only for the experimental
arm. Its `flash_row_interleave` namespace gives it a distinct device symbol,
so loading two same-geometry DSOs cannot accidentally alias their kernels.
The control retains its original symbol and default implementation.

## Local evidence (not device performance)

HGCC 2.1.1-a5c56e, `-arch=ppu_15`, exact full FP16/BF16 bodies and the shipping
compiler flags. Both precisions have the same findings:

| Native evidence | Control | Cross-row |
|---|---:|---:|
| Vregs/thread | 256 | 256 |
| Stack/private spill | 0 | 0 |
| Dynamic shared bytes | 65536 | 65536 |
| Longest same-row EXP run, first tile | 16 | 3 |
| Longest same-row EXP run, recurrent tile | 16 | 2 |
| Recurrent row switches among 64 score EXPs | 15 | 59 |
| EXP immediately followed by FMA overwriting its input, recurrent sites | 26 | 14 |
| Static instruction sites | 2943 | 2954 |

The extra static sites are **5 `s.wait` + 6 `v.reg.dchk`**. MMA, shared-load,
EXP, FMA, MUL, ADD and packed conversion inventories are unchanged. The native
gate uses scaled-maximum *definitions*, not just register names: the old
kernel reuses one register for different rows. It covers exactly 128 score
EXPs (two 4x16 bodies); missing an element cannot shrink the denominator.

These are static observations, not a dynamic instruction or stall prediction.
The input-register reuse pattern is **not proof of a hardware WAR hazard**.
The control's encoded instructions match the untouched parent with this SDK.

Host checks: 8192 finite/masked, zero/nonzero-initial-sum order fixtures; actual
iterator coverage/order. Negatives: omit one element, swap within-row order,
restore the original native schedule under the candidate's name, remove one
native EXP, introduce stack spill, and perturb O or LSE by one ULP. CPU order
tests do not pretend that CPU exp2 is PPU exp2.

Local final linking and device execution are **NOT_RUN**: the local SDK has
compiler/disassembler but is missing runtime libraries. Box links with its
complete SDK; no runtime compatibility aliases or prebuilt binary are shipped.

## Box: one command, no installation/probe

From this branch in the FlashAttention repository:

```bash
PPU_SDK=/usr/local/PPU_SDK PROFILE=1 DTYPE=both \
  bash tools/run_ppu15_row_interleave_box.sh
```

Use `DTYPE=bf16` or `DTYPE=fp16` to match the original report. `PROFILE=0` runs
correctness + ordinary timing only. Outputs go to a new `/workspace/...`
directory, with `result.json`, `performance.log`, hashed build/native evidence,
and `control-<dtype>.acurep` / `row-interleave-<dtype>.acurep` (ACU may add
`.report` before `.acurep`). No CSV export is needed.

The existing benchmark is reused, not copied. It runs full small tail cases
and CPU-FP64 sampled-query/all-key/all-head reference checks at S8398.
**Every O and LSE bit must equal control**, and repeats/post-timing fingerprints
must be stable. A failure stops before profiling; the tolerance is not relaxed.

Normal timing rotates A/B order, 5 warmups, 7 samples x 20 launches. ACU is a
separate single-attention-launch capture per arm; its absolute duration is
not the benchmark timing. Compare compute-dependency at the FMA/EXP sites,
overall normal latency, register/local-memory use and memory traffic. Report
both increases and decreases. Overlapping timing envelopes are `UNRESOLVED`.
Lower stall readings without lower latency are **not a performance win**.

## Reproduce local checks

```bash
python dev/ppu15_bn128/build.py --sdk "$PPU_SDK" \
  --compile-only --output /workspace/fa15-row-native \
  --contract dev/ppu15_row_interleave/contract.json --jobs 2
g++ -std=c++17 -O2 -ffp-contract=off \
  -Icsrc/actlize/include -Icsrc/flash_attn/src \
  -I"$PPU_SDK/include" -I"$PPU_SDK/targets/x86_64-linux/include" \
  dev/ppu15_row_interleave/check_order.cpp -o /workspace/fa15-row-order
/workspace/fa15-row-order
FA15_ROW_BUILD=/workspace/fa15-row-native \
  python dev/ppu15_row_interleave/test_contract.py -v
```

Set `FA15_PARENT_NATIVE` to the untouched parent control's dump directory to
exercise native default-byte equality too; without it that test reports SKIP.
