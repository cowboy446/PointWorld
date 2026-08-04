#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${POINTWORLD_PYTHON:-python}"
: "${DATA_DIR:?请设置 DATA_DIR 为 LIBERO PointWorld 数据集（WDS 或 indexed H5）的绝对路径}"
STATS_DIR="${STATS_DIR:-${REPO_DIR}/stats/libero_scene3_20260802}"
MAX_SAMPLES="${MAX_SAMPLES:-256}"
MAX_SCENE_POINTS="${MAX_SCENE_POINTS:-12000}"
MAX_ROBOT_POINTS="${MAX_ROBOT_POINTS:-1024}"
NUM_CAMERAS="${NUM_CAMERAS:-2}"
SEED="${SEED:-512026}"

if [[ "${DATA_DIR}" != /* ]]; then
  echo "DATA_DIR 必须是绝对路径: ${DATA_DIR}" >&2
  exit 2
fi
if [[ ! -f "${DATA_DIR}/metadata_rank0.json" ]]; then
  echo "找不到 ${DATA_DIR}/metadata_rank0.json" >&2
  exit 2
fi

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" scripts/compute_libero_norm_stats.py \
  "${DATA_DIR}" "${STATS_DIR}" \
  --max-samples "${MAX_SAMPLES}" \
  --seed "${SEED}" \
  --max-scene-points "${MAX_SCENE_POINTS}" \
  --max-robot-points "${MAX_ROBOT_POINTS}" \
  --num-cameras "${NUM_CAMERAS}"
