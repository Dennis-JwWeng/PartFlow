#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
DATA_ROOT="${DATA_ROOT:-data/Pxform_v1/training}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/training_smoke}"
extra=()
if [[ -n "${PRETRAINED_ROOT:-}" ]]; then extra+=(--pretrained_dir "$PRETRAINED_ROOT"); fi
if [[ -n "${MASK_ROOT:-}" ]]; then extra+=(--mask_sidecar_root "$MASK_ROOT"); fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
mkdir -p "$OUTPUT_ROOT"
for stage in stage1_ss stage2_slat; do
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u train.py \
        --config "configs/train_${stage}.json" --data_dir "$DATA_ROOT" \
        "${extra[@]}" --output_dir "$OUTPUT_ROOT/$stage" --num_gpus 1 --smoke_steps 2 \
        --ckpt none --auto_retry 0 2>&1 | tee "$OUTPUT_ROOT/${stage}.log"
done
