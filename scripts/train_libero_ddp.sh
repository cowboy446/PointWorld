#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${POINTWORLD_PYTHON:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"
: "${DATA_DIR:?请设置 DATA_DIR 为 LIBERO PointWorld WebDataset 的绝对路径}"
STATS_DIR="${STATS_DIR:-${REPO_DIR}/stats/libero_scene3_20260802}"
EXP_NAME="${EXP_NAME:-libero-scene3-uniform-cap12k-ddp4}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/train_logs}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"
NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-22}"
NUM_WORKERS="${NUM_WORKERS:-16}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-5}"
NUM_EPOCHS="${NUM_EPOCHS:-200}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:--1}"
EVAL_FREQ="${EVAL_FREQ:--1}"
SAVE_FREQ="${SAVE_FREQ:-300}"
NUM_EVAL_BATCHES="${NUM_EVAL_BATCHES:-10}"
MAX_SCENE_POINTS="${MAX_SCENE_POINTS:-12000}"
MAX_ROBOT_POINTS="${MAX_ROBOT_POINTS:-1024}"
NUM_CAMERAS="${NUM_CAMERAS:-2}"
SEED="${SEED:-512026}"
PTV3_SIZE="${PTV3_SIZE:-small}"
PTV3_PATCH_SIZE="${PTV3_PATCH_SIZE:-128}"
PREDICTOR_DIM="${PREDICTOR_DIM:-128}"

if [[ "${DATA_DIR}" != /* ]]; then
  echo "DATA_DIR 必须是绝对路径: ${DATA_DIR}" >&2
  exit 2
fi
for required in "${DATA_DIR}/metadata_rank0.json" "${STATS_DIR}/norm_stats.json" "${TORCHRUN_BIN}"; do
  if [[ ! -e "${required}" ]]; then
    echo "找不到必要路径: ${required}" >&2
    exit 2
  fi
done

cd "${REPO_DIR}"
export CUDA_VISIBLE_DEVICES
export WANDB_MODE="${WANDB_MODE:-disabled}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "${TORCHRUN_BIN}" \
  --standalone \
  --nproc_per_node "${NUM_GPUS}" \
  train.py \
  --distributed true \
  --domains libero \
  --data_dirs "${DATA_DIR}" \
  --norm_stats_path "${STATS_DIR}" \
  --exp_name "${EXP_NAME}" \
  --log_dir "${LOG_DIR}" \
  --device cuda \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --eval_num_workers "${EVAL_NUM_WORKERS}" \
  --num_epochs "${NUM_EPOCHS}" \
  --max_train_steps "${MAX_TRAIN_STEPS}" \
  --eval_freq "${EVAL_FREQ}" \
  --save_freq "${SAVE_FREQ}" \
  --num_eval_batches "${NUM_EVAL_BATCHES}" \
  --deterministic_data true \
  --deterministic_train true \
  --disable_compile true \
  --max_scene_points "${MAX_SCENE_POINTS}" \
  --max_robot_points "${MAX_ROBOT_POINTS}" \
  --seed "${SEED}" \
  --train_min_num_cameras "${NUM_CAMERAS}" \
  --train_max_num_cameras "${NUM_CAMERAS}" \
  --eval_min_num_cameras "${NUM_CAMERAS}" \
  --eval_max_num_cameras "${NUM_CAMERAS}" \
  --ptv3_size "${PTV3_SIZE}" \
  --ptv3_patch_size "${PTV3_PATCH_SIZE}" \
  --predictor_dim "${PREDICTOR_DIM}"
