#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${POINTWORLD_PYTHON:-python}"
: "${DATA_DIR:?请设置 DATA_DIR 为 LIBERO PointWorld 数据集（WDS 或 indexed H5）的绝对路径}"
STATS_DIR="${STATS_DIR:-${REPO_DIR}/stats/libero_scene3_20260802}"
EXP_NAME="${EXP_NAME:-libero-scene3-uniform-cap12k}"
MODEL_PATH="${MODEL_PATH:-${REPO_DIR}/train_logs/${EXP_NAME}/model-last.pt}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
EVAL_NUM_BATCHES="${EVAL_NUM_BATCHES:-0}"
NUM_CAMERAS="${NUM_CAMERAS:-2}"
MAX_SCENE_POINTS="${MAX_SCENE_POINTS:-12000}"
MAX_ROBOT_POINTS="${MAX_ROBOT_POINTS:-1024}"
SEED="${SEED:-512026}"
EVAL_VIZ_NUM="${EVAL_VIZ_NUM:--1}"
EVAL_SKIP_VIZ="${EVAL_SKIP_VIZ:-true}"
VIEWER_PORT="${VIEWER_PORT:-8080}"

if [[ "${DATA_DIR}" != /* ]]; then
  echo "DATA_DIR 必须是绝对路径: ${DATA_DIR}" >&2
  exit 2
fi
for required in "${DATA_DIR}/metadata_rank0.json" "${STATS_DIR}/norm_stats.json" "${MODEL_PATH}"; do
  if [[ ! -f "${required}" ]]; then
    echo "找不到必要文件: ${required}" >&2
    exit 2
  fi
done

cd "${REPO_DIR}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
exec "${PYTHON_BIN}" eval.py \
  --domains libero \
  --data_dirs "${DATA_DIR}" \
  --norm_stats_path "${STATS_DIR}" \
  --model_path "${MODEL_PATH}" \
  --device cuda \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${EVAL_NUM_WORKERS}" \
  --eval_num_workers "${EVAL_NUM_WORKERS}" \
  --eval_num_batches "${EVAL_NUM_BATCHES}" \
  --seed "${SEED}" \
  --allow_missing_confidence_mask true \
  --eval_viz_num "${EVAL_VIZ_NUM}" \
  --eval_skip_viz "${EVAL_SKIP_VIZ}" \
  --viewer_port "${VIEWER_PORT}" \
  --disable_compile true \
  --max_scene_points "${MAX_SCENE_POINTS}" \
  --max_robot_points "${MAX_ROBOT_POINTS}" \
  --eval_min_num_cameras "${NUM_CAMERAS}" \
  --eval_max_num_cameras "${NUM_CAMERAS}"
