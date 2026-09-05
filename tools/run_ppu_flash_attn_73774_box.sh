#!/usr/bin/env bash
# Reproduce the registered long-sequence FlashAttention forward shape on PPU.

set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source_sha=$(git -C "$repo_root" rev-parse --short=8 HEAD)
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
out_dir=${OUT:-/workspace/flash-attn-ppu-fa73774-${source_sha}-${timestamp}}
runtime_dir="$out_dir/runtime"
build_dir="$out_dir/build"
python_bin=${PYTHON:-python}
ppu_sdk=${PPU_SDK:-/usr/local/PPU_SDK}

mkdir -p -- "$out_dir" "$runtime_dir" "$build_dir"

if [[ ! -d "$ppu_sdk/lib" ]]; then
  echo "[PPU FA runner] FAIL: PPU SDK runtime directory not found: $ppu_sdk/lib" >&2
  false
fi

export LD_LIBRARY_PATH="$ppu_sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[PPU FA runner] sha=$(git -C "$repo_root" rev-parse HEAD) out=$out_dir"
set +e
"$python_bin" "$repo_root/tools/materialize_ppu_prebuilt.py" \
  --destination "$runtime_dir" 2>&1 | tee "$out_dir/materialize.log"
materialize_rc=${PIPESTATUS[0]}
set -e

case "$materialize_rc" in
  0)
    binary_provider=checked-in-prebuilt
    ;;
  3)
    binary_provider=box-source-build
    if [[ ! -f "$repo_root/csrc/actlize/include/cute/tensor.hpp" ]]; then
      git -C "$repo_root" submodule update --init --recursive csrc/actlize
    fi
    echo "[PPU FA build] provider=$binary_provider torch=$($python_bin -c 'import torch; print(torch.__version__)') jobs=${JOBS:-16}"
    (
      cd "$repo_root"
      PPU_SDK="$ppu_sdk" MAX_JOBS="${JOBS:-16}" \
        "$python_bin" setup.py build_ext \
          --build-temp "$build_dir/temp" \
          --build-lib "$runtime_dir"
    ) 2>&1 | tee "$out_dir/build.log"
    ;;
  *)
    echo "[PPU FA runner] FAIL: checked-in prebuilt exists but failed validation" >&2
    false
    ;;
esac

shopt -s nullglob
artifacts=("$runtime_dir"/flash_attn_2_cuda*.so)
shopt -u nullglob
if [[ ${#artifacts[@]} -ne 1 ]]; then
  echo "[PPU FA runner] FAIL: expected exactly one extension, found ${#artifacts[@]}" >&2
  false
fi
artifact=${artifacts[0]}
echo "[PPU FA binary] provider=$binary_provider artifact=$artifact"
sha256sum "$artifact" | tee "$out_dir/binary.sha256"
git -C "$repo_root" status --short --branch >"$out_dir/git-status.txt"

export PYTHONPATH="$runtime_dir:$repo_root${PYTHONPATH:+:$PYTHONPATH}"

"$python_bin" "$repo_root/tools/benchmark_ppu_flash_attn.py" \
  --batch 1 \
  --seqlen 73774 \
  --heads 56 \
  --head-dim 128 \
  --warmup "${WARMUP:-2}" \
  --samples "${SAMPLES:-7}" \
  --peak-tflops "${PPU_PEAK_TFLOPS:-500.0}" \
  2>&1 | tee "$out_dir/performance.log"

echo "[PPU FA runner] PASS: artifacts=$out_dir"
