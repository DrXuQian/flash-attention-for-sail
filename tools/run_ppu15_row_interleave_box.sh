#!/usr/bin/env bash
# One variable: cross-row softmax scheduling. No installation/default change.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source_sha=$(git -C "$repo_root" rev-parse --short=8 HEAD)
stamp=$(date -u +%Y%m%dT%H%M%SZ)
out_dir=${OUT:-/workspace/flash-attn-ppu15-row-interleave-${source_sha}-${stamp}-$$}
ppu_sdk=${PPU_SDK:-/usr/local/PPU_SDK}
python_bin=${PYTHON:-python}
dtype=${DTYPE:-both}
mkdir -p -- "$out_dir"

if [[ ! -f "$repo_root/csrc/actlize/include/cute/tensor.hpp" ]]; then
    git -C "$repo_root" submodule update --init csrc/actlize
fi
echo "[FA15 row-interleave runner] sha=$(git -C "$repo_root" rev-parse HEAD) sdk=$ppu_sdk dtype=$dtype out=$out_dir"
"$python_bin" "$repo_root/dev/ppu15_bn128/build.py" \
    --sdk "$ppu_sdk" --output "$out_dir/build" --jobs "${JOBS:-2}" \
    --contract "$repo_root/dev/ppu15_row_interleave/contract.json" \
    2>&1 | tee "$out_dir/build.log"

export LD_LIBRARY_PATH="$ppu_sdk/lib:$ppu_sdk/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
extra=()
if [[ -n ${PPU_PEAK_TFLOPS:-} ]]; then extra+=(--peak-tflops "$PPU_PEAK_TFLOPS"); fi
"$python_bin" "$repo_root/dev/ppu15_bn128/bench.py" \
    --build "$out_dir/build" --output "$out_dir/result.json" --dtype "$dtype" \
    --warmup "${WARMUP:-5}" --samples "${SAMPLES:-7}" --launches "${LAUNCHES:-20}" \
    "${extra[@]}" 2>&1 | tee "$out_dir/performance.log"

# ACU is separate from normal timing; one attention launch per report.
# All numerical/replay gates above must pass before profiling.
if [[ ${PROFILE:-1} == 1 ]]; then
    acu=${ACU:-$ppu_sdk/asight/bin/acu}
    dtypes=("$dtype")
    if [[ $dtype == both ]]; then dtypes=(fp16 bf16); fi
    for dt in "${dtypes[@]}"; do
        for config in control row-interleave; do
            "$acu" -f -o "$out_dir/$config-$dt" --set full \
                --kernel-name 'regex:flash_fwd_kernel' \
                "$python_bin" "$repo_root/dev/ppu15_bn128/bench.py" \
                --build "$out_dir/build" --output "$out_dir/profile-unused.json" \
                --dtype "$dt" --profile "$config" \
                2>&1 | tee "$out_dir/$config-$dt.acu.log"
            if [[ ! -s "$out_dir/$config-$dt.acurep" && ! -s "$out_dir/$config-$dt.report.acurep" ]]; then
                echo "[FA15 row-interleave runner] FAIL: ACU returned without report: $config/$dt" >&2
                false
            fi
        done
    done
fi
echo "[FA15 row-interleave runner] PASS: $out_dir; default unchanged"
