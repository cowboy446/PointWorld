#!/usr/bin/env bash
set -euo pipefail

# Example: evaluate a DROID WDS subset.
#
# This script has two phases:
# 1. Optionally generate expert confidence annotations for the subset test split.
# 2. Run normal evaluation, using the generated confidence file for filtered metrics.
#
# Important:
# - DATA_DIR should be an absolute path to a WDS root containing test/*.tar.
# - The confidence file is written to:
#     ${DATA_DIR}/test/expert_confidence-seed=42.h5
# - GRID_SIZE and CONFIDENCE_THRES must match between annotation and evaluation.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DOMAIN="${DOMAIN:-droid}"
DATA_DIR="${DATA_DIR:-/absolute/path/to/droid/wds_subset}"
NORM_STATS_PATH="${NORM_STATS_PATH:-stats/droid}"

# Use the released confidence/filtering checkpoint for annotation if available.
CONFIDENCE_MODEL_PATH="${CONFIDENCE_MODEL_PATH:-pretrained_checkpoints/filter_droid_test_split/model-last.pt}"

# Use a scene-flow checkpoint for the actual evaluation.
EVAL_MODEL_PATH="${EVAL_MODEL_PATH:-pretrained_checkpoints/large-droid/model-best.pt}"

BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-5}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-5}"
EVAL_NUM_BATCHES="${EVAL_NUM_BATCHES:--1}"
CONFIDENCE_THRES="${CONFIDENCE_THRES:-0.8}"
GRID_SIZE="${GRID_SIZE:-0.015}"
PTV3_SIZE="${PTV3_SIZE:-large}"
PREDICTOR_DIM="${PREDICTOR_DIM:-256}"
VIEWER_PORT="${VIEWER_PORT:-8080}"

# Set to 0 if you already have ${DATA_DIR}/test/expert_confidence-seed=42.h5.
RUN_CONFIDENCE="${RUN_CONFIDENCE:-1}"

if [[ "${DATA_DIR}" != /* ]]; then
  echo "DATA_DIR must be an absolute path. Got: ${DATA_DIR}" >&2
  exit 1
fi

if [[ ! -d "${DATA_DIR}/test" ]]; then
  echo "Missing test split directory: ${DATA_DIR}/test" >&2
  exit 1
fi

if [[ "${RUN_CONFIDENCE}" == "1" ]]; then
  echo "Generating expert confidence annotations for ${DATA_DIR}/test ..."
  python eval.py \
    --model_path "${CONFIDENCE_MODEL_PATH}" \
    --domains="${DOMAIN}" \
    --data_dirs="${DATA_DIR}" \
    --norm_stats_path="${NORM_STATS_PATH}" \
    --ptv3_size="${PTV3_SIZE}" \
    --predictor_dim="${PREDICTOR_DIM}" \
    --grid_size="${GRID_SIZE}" \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --eval_num_workers="${EVAL_NUM_WORKERS}" \
    --eval_num_batches="${EVAL_NUM_BATCHES}" \
    --confidence_thres="${CONFIDENCE_THRES}" \
    --run_confidence_annotation=true \
    --eval_skip_viz=true \
    --viewer_port="${VIEWER_PORT}"
fi

echo "Running filtered evaluation for ${DATA_DIR}/test ..."
python eval.py \
  --model_path "${EVAL_MODEL_PATH}" \
  --domains="${DOMAIN}" \
  --data_dirs="${DATA_DIR}" \
  --norm_stats_path="${NORM_STATS_PATH}" \
  --ptv3_size="${PTV3_SIZE}" \
  --predictor_dim="${PREDICTOR_DIM}" \
  --grid_size="${GRID_SIZE}" \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --eval_num_workers="${EVAL_NUM_WORKERS}" \
  --eval_num_batches="${EVAL_NUM_BATCHES}" \
  --confidence_thres="${CONFIDENCE_THRES}" \
  --eval_skip_viz=true \
  --viewer_port="${VIEWER_PORT}"

