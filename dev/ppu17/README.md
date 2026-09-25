# PPU1.7: reuse the Hopper forward implementation

This is an explicit backend of `flash_attn_3`, not a new attention algorithm.
The default `flash_attn` (FA2) import and PPU1.0/1.5 AIU path do not change.
`USE_PPU` in the original fork means the **legacy AIU algorithm**; it must not
be defined for this SM90 backend. `FLASHATTN_PPU17` selects the CUDA-compatible
runtime and PPU CUTLASS3.6 headers, with `ACOMPUTE_VERSION=10700`.

## Scope

- FP16/BF16, forward only, fixed-length BSHD, Dq=Dk=Dv in {64,128,256}.
- Causal/noncausal, sequence tails and ordinary GQA (Hq divisible by Hkv).
- Reuses WGMMA QK/PV, FP32 accumulators, online softmax, TMA and the Hopper
  persistent schedulers. Q/K/V storage and the attention formula do not change.
- Backward, FP8, D96/192, unequal head dimensions, cluster launch, varlen, QSA,
  split, paged/append KV, packed-GQA optimization, softcap, local attention,
  rotary fusion and sink bias are **not admitted**, and are rejected.
- PPU1.7 native compilation and device correctness are NOT implied by the
  local CUDA/SM90 source checks. No performance result is claimed.

The user's simulation target is **20 SM, 32 MiB LLC/L2**. SM count comes from
the runtime (`params.num_sm`), not a fixed 72/132. The existing causal Hopper
LPT scheduler already uses a 32 MiB K/V section budget. Its dynamic work queue
assigns long causal tiles first. This port does not retune it yet.

For B1/S2048/Hq32/Hkv2/D256/BF16, K+V is 4 MiB and causal useful work is
68,753,031,168 FLOP. GQA reduces KV storage, not the query-head arithmetic.
An MFU from a performance simulator must use that model's peak and simulated
duration, not the PPU1.0 500-TFLOPS denominator or Python wall time.

## Build with a capable native toolchain

```bash
cd hopper
FLASH_ATTENTION_PPU_ARCH=10700 \
CUTLASS_PPU17_ROOT=/path/to/cutlass3-3.6.0 \
PPU_SDK=/path/to/ppu17-sdk \
MAX_JOBS=6 python setup.py build_ext --build-temp /workspace/fa17-build/temp \
    --build-lib /workspace/fa17-build/lib
python setup.py --help  # legacy entry remains unchanged without the opt-in
```

Use `build` or `pip wheel . --no-build-isolation` under the same environment
when a complete Python package is needed; `build_ext` alone produces `_C`
without copying the Python package files.

SDK2.1.1-a5c56e is **not** a PPU1.7 compiler. In particular, its CUDA wrapper
can accept `sm_90a` and still select `ppu001 / __CUDA_ARCH__=800`. The target
probe rejects this before compiling FA; it does not fall back to `vm_15`.

## Produce SM90 input for a performance simulator

If the simulation tool accepts CUDA SM90a binaries/PyTorch, this explicit mode
uses its selected CUDA-compatible toolchain. This is simulation input, **not
a native PPU binary certificate**. No need for a fake PPU SDK environment.

```bash
cd hopper
FLASH_ATTENTION_PPU_ARCH=10700 \
FLASH_ATTENTION_PPU17_COMPILE_MODE=simulation \
CUTLASS_PPU17_ROOT=/root/cutlass3-3.6.0 \
CUDA_HOME=/usr/local/cuda MAX_JOBS=6 \
    python setup.py build --build-base /workspace/fa17-simulation
```

Add the produced `lib.*` directory to `PYTHONPATH` (or install the wheel), then
run the following Python command **through your simulation tool**:

```bash
python tools/run_ppu17_forward.py --seqlen 2048 --heads 32 --kv-heads 2 \
    --head-dim 256 --dtype bf16 --expected-sms 20 \
    --output /workspace/fa17-simulation/causal-2048-h32-hkv2-d256.json
```

Causal is the default; `--noncausal` selects the other arm. This is designed
for one attention launch, without GPU warmup/reference kernels. Inputs and the
causal scheduler counter are initialized on CPU and copied to the target;
the API no longer needs a GPU fill kernel for that counter. Confirm the actual
single kernel symbol/count in the simulation trace. The runner rejects an
extension without the new PPU1.7 build identity. Runtime identity and binary
hash are printed. `--expected-sms` is an assertion, **not a hardware
partition**. A Hopper with only `grid=20` is not a 20-SM hardware simulation.

For correctness first use `--seqlen 65 --heads 4 --kv-heads 2 --verify` and
run each causal/noncausal, dtype and dimension case in a separate process.
This uses a CPU float64 softmax-attention reference with predeclared tolerances.
`--verify` also works for the complete S2048 workload: the reference blocks
query rows on CPU to avoid allocating all heads' score matrices at once.
It launches no GPU reference. Reference time is not simulated target time.

## Local checks

```bash
python tests/test_ppu17_source.py
python tests/test_ppu17_cpu_reference.py  # requires Torch; CPU-only is enough
python tools/check_ppu17_source.py --cutlass /root/cutlass3-3.6.0 \
    --nvcc /usr/local/cuda/bin/nvcc --jobs 3 \
    --torch-root /path/to/installed/cuda-torch \
    --out /workspace/fa17-source-check
python tools/check_ppu17_negative_builds.py \
    --objects /workspace/fa17-source-check --out /workspace/fa17-negative-check
```

The source-check command instantiates all six generated source units, separately
checks causal/noncausal bodies for WGMMA and TMA load/store, and assembles them.
It rejects empty bodies, missing output stores and legacy PPU PTX leakage.
`--torch-root` enables a real host API compile plus generated-definition
relocatable link; omitting it records that check as SKIP, not PASS. This is
not a complete Python/Torch extension-load test. The negative command removes
one real generated unit and requires the link-closure checker to reject it.
Optional `--legacy-nvcc /path/to/old-sdk/CUDA_SDK/bin/nvcc` also reproduces the
old SDK's false target selection. CUDA assembler success is still not PPU
numerical or performance evidence.

## Deliberate compatibility changes

- Select existing SM90 generated units; select `Sm90`, not `PPU0010/0015`.
- Keep the real WGMMA helper and TMA output store enabled (both were compiled
  out under the legacy umbrella `USE_PPU`). Use native SM90 named barriers.
- CUTLASS3.6 epilogue store selector has two arguments, not three.
- Its original TMA pipeline already signals once per consumer warpgroup for
  a 1x1x1 cluster, so use it rather than FA's workaround for a newer API.
- Use `KernelHardwareInfo::sm_count`, without changing the legacy `cu_count`.
- Keep host/device stream types coherent and reject unsupported API options
  before any launch. No blanket architecture macro redefinition or header stubs.
- Preserve upstream's new `Is_QSA` template axis with its default `false`;
  PPU1.7 rejects QSA instantiations, the QSA build switch and its AIU option.
  The shared QSA header uses the runtime aliases. Its legacy direct-index
  intrinsics stay outside the PPU1.7 parser path, without changing their
  PPU1.0/1.5 implementation.
