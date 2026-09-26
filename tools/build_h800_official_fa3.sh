#!/usr/bin/env bash
# Build upstream's unchanged FA3 sources, not a replacement production library.
set -euo pipefail
source_dir=${1:?usage: build_h800_official_fa3.sh SOURCE_DIR OUTPUT_DIR}
output_dir=${2:?usage: build_h800_official_fa3.sh SOURCE_DIR OUTPUT_DIR}
source_dir=$(realpath -e "$source_dir")
case "$output_dir" in /workspace/*) ;; *) echo 'output must be under /workspace' >&2; exit 2;; esac
test ! -e "$output_dir"
test "$(git -C "$source_dir" rev-parse HEAD)" = a8aa52b1ab3e9ca574c8a33b3f35afc017ffa2e2
test "$(git -C "$source_dir/csrc/cutlass" rev-parse HEAD)" = dc4817921edda44a549197ff3a9dcf5df0636e7b
test -z "$(git -C "$source_dir" status --porcelain --untracked-files=no)"
test -z "$(git -C "$source_dir/csrc/cutlass" status --porcelain --untracked-files=no)"
mkdir -p "$output_dir/scratch"
output_dir=$(realpath -e "$output_dir")
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON:-/root/miniconda3/bin/python}
# No task launch on a machine borrowed by another validation/build process.
"$python_bin" -c 'import sys; sys.path.insert(0, sys.argv[1]); from bench_ppu17_hopper_control import require_idle; require_idle()' "$script_dir"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TMPDIR="$output_dir/scratch"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_OFFLINE_BUILD=TRUE
export FLASH_ATTENTION_DISABLE_BACKWARD=TRUE
export FLASH_ATTENTION_DISABLE_SPLIT=TRUE
export FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE
export FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
export FLASH_ATTENTION_DISABLE_LOCAL=TRUE
export FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
export FLASH_ATTENTION_DISABLE_FP16=TRUE
export FLASH_ATTENTION_DISABLE_FP8=TRUE
export FLASH_ATTENTION_DISABLE_VARLEN=TRUE
export FLASH_ATTENTION_DISABLE_HDIM64=TRUE
export FLASH_ATTENTION_DISABLE_HDIM96=TRUE
export FLASH_ATTENTION_DISABLE_HDIM128=TRUE
export FLASH_ATTENTION_DISABLE_HDIM192=TRUE
export FLASH_ATTENTION_DISABLE_SM80=TRUE
export FLASH_ATTENTION_DISABLE_HDIM256=FALSE
export FLASH_ATTENTION_DISABLE_PACKGQA=FALSE
export FLASH_ATTENTION_DISABLE_CLUSTER=FALSE
export TORCH_CUDA_ARCH_LIST=9.0a
export MAX_JOBS=2
export NVCC_THREADS=2
unset PPU_SDK FLASH_ATTENTION_PPU_ARCH PYTORCH_NVCC
cd "$source_dir/hopper"
"$python_bin" setup.py build --build-base "$output_dir/build" --build-lib "$output_dir/python"
test -z "$(git -C "$source_dir" status --porcelain --untracked-files=no)"
test -z "$(git -C "$source_dir/csrc/cutlass" status --porcelain --untracked-files=no)"
sha256sum "$output_dir"/python/flash_attn_3/_C*.so
