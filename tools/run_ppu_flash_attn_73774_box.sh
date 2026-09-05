#!/usr/bin/env bash
# Reproduce the registered long-sequence FlashAttention forward shape on PPU.

set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source_sha=$(git -C "$repo_root" rev-parse --short=8 HEAD)
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
out_dir=${OUT:-/workspace/flash-attn-ppu-fa73774-${source_sha}-${timestamp}}
runtime_dir="$out_dir/runtime"
python_bin=${PYTHON:-python}
ppu_sdk=${PPU_SDK:-/usr/local/PPU_SDK}

mkdir -p -- "$out_dir" "$runtime_dir"

if [[ ! -d "$ppu_sdk/lib" ]]; then
  echo "[PPU FA runner] FAIL: PPU SDK runtime directory not found: $ppu_sdk/lib" >&2
  false
fi

export LD_LIBRARY_PATH="$ppu_sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[PPU FA runner] sha=$(git -C "$repo_root" rev-parse HEAD) out=$out_dir"
"$python_bin" "$repo_root/tools/materialize_ppu_prebuilt.py" \
  --destination "$runtime_dir" | tee "$out_dir/materialize.log"

artifact="$runtime_dir/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so"
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
