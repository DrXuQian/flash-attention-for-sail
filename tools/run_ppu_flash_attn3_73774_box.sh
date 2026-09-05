#!/usr/bin/env bash
# Build the narrow PPU FA3 D128/BF16 forward target, then time the registered shape.

set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source_sha=$(git -C "$repo_root" rev-parse --short=8 HEAD)
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
out_dir=${OUT:-/workspace/flash-attn3-ppu-fa73774-${source_sha}-${timestamp}}
runtime_dir="$out_dir/runtime"
build_dir="$out_dir/build"
python_bin=${PYTHON:-python}
pinned_ppu_sdk=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK
ppu_sdk=${PPU_SDK:-$pinned_ppu_sdk}

mkdir -p -- "$out_dir" "$runtime_dir" "$build_dir"

if [[ ! -x "$ppu_sdk/bin/hgcc" || ! -x "$ppu_sdk/CUDA_SDK/bin/nvcc" ]]; then
  echo "[PPU FA3 runner] FAIL: required PPU SDK 2.1.1-a5c56e not found at $ppu_sdk" >&2
  false
fi
if [[ ! -f "$repo_root/csrc/actlize/include/cute/tensor.hpp" ]]; then
  git -C "$repo_root" submodule update --init --recursive csrc/actlize
fi

export PATH="$ppu_sdk/CUDA_SDK/bin:$ppu_sdk/bin:$PATH"
export LD_LIBRARY_PATH="$ppu_sdk/lib:$ppu_sdk/CUDA_SDK/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="$ppu_sdk/CUDA_SDK"

echo "[PPU FA3 runner] sha=$(git -C "$repo_root" rev-parse HEAD) out=$out_dir ppu_sdk=$ppu_sdk"
env -u LD_LIBRARY_PATH "$ppu_sdk/bin/hgcc" --version >"$out_dir/hgcc-version.log" 2>&1
sed -n '1,2p' "$out_dir/hgcc-version.log"
"$python_bin" - <<'PY' | tee "$out_dir/python-runtime.log"
import sys
import torch
print(f"python={sys.version.split()[0]} torch={torch.__version__} cxx11abi={int(torch.compiled_with_cxx11_abi())}")
PY

echo "[PPU FA3 build] scope=BF16-D128-fixed-forward-only jobs=${JOBS:-16} nvcc_threads=${NVCC_THREADS:-2}"
(
  cd "$repo_root/hopper"
  PPU_SDK="$ppu_sdk" \
  MAX_JOBS="${JOBS:-16}" \
  NVCC_THREADS="${NVCC_THREADS:-2}" \
  FLASH_ATTENTION_FORCE_BUILD=TRUE \
  FLASH_ATTENTION_DISABLE_BACKWARD=TRUE \
  FLASH_ATTENTION_DISABLE_SPLIT=TRUE \
  FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE \
  FLASH_ATTENTION_DISABLE_APPENDKV=TRUE \
  FLASH_ATTENTION_DISABLE_LOCAL=TRUE \
  FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE \
  FLASH_ATTENTION_DISABLE_PACKGQA=TRUE \
  FLASH_ATTENTION_DISABLE_FP16=TRUE \
  FLASH_ATTENTION_DISABLE_FP8=TRUE \
  FLASH_ATTENTION_DISABLE_VARLEN=TRUE \
  FLASH_ATTENTION_DISABLE_HDIM64=TRUE \
  FLASH_ATTENTION_DISABLE_HDIM96=TRUE \
  FLASH_ATTENTION_DISABLE_HDIM192=TRUE \
  FLASH_ATTENTION_DISABLE_HDIM256=TRUE \
  FLASH_ATTENTION_DISABLE_SM90=TRUE \
    "$python_bin" setup.py build_ext \
      --build-temp "$build_dir/temp" \
      --build-lib "$runtime_dir"
) 2>&1 | tee "$out_dir/build.log"

shopt -s nullglob
artifacts=("$runtime_dir"/flash_attn_3/_C*.so)
shopt -u nullglob
if [[ ${#artifacts[@]} -ne 1 ]]; then
  echo "[PPU FA3 runner] FAIL: expected exactly one FA3 extension, found ${#artifacts[@]}" >&2
  false
fi
artifact=${artifacts[0]}
sha256sum "$artifact" | tee "$out_dir/binary.sha256"
git -C "$repo_root" status --short --branch >"$out_dir/git-status.txt"

export PYTHONPATH="$runtime_dir:$repo_root${PYTHONPATH:+:$PYTHONPATH}"

"$python_bin" "$repo_root/tools/benchmark_ppu_flash_attn.py" \
  --backend fa3 \
  --batch 1 \
  --seqlen 73774 \
  --heads 56 \
  --head-dim 128 \
  --warmup "${WARMUP:-2}" \
  --samples "${SAMPLES:-7}" \
  --peak-tflops "${PPU_PEAK_TFLOPS:-500.0}" \
  2>&1 | tee "$out_dir/performance.log"

echo "[PPU FA3 comparison] historical_fa2_median_ms=580.539 historical_fa2_logical_mfu=53.76%"
echo "[PPU FA3 runner] PASS: artifacts=$out_dir"
