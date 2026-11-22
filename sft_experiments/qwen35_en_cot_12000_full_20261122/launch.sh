#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here/../.."
root="$(cd .. && pwd)"
mkdir -p "$here/runtime"
exec 9>"$here/runtime/launch.lock"
flock -n 9
gpu="${1:?GPU index required}"
mem=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
test "$mem" -eq 0
export CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONPATH=src TMPDIR="$root/tmp_q35" TRITON_CACHE_DIR="$root/tmp_q35/triton_cot12"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
printf '%s\n' "$$" > "$here/runtime/pid"
exec /opt/anaconda3/envs/training/bin/python -u "$here/train.py"
