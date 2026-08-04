#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATA_DIR="${DATA_DIR:-/backup/zhangrong/data/workspace/robot-wm/point-wm/PointWorld/restore_data/pointworld_libero_wds_scene3_compact_20260804}"
STATS_DIR="${STATS_DIR:-stats/libero_scene3_compact_20260804}"
EXP_NAME="${EXP_NAME:-pointworld_libero_scene3_small_cap12k_ddp4gpu}"

[[ "${DATA_DIR}" = /* ]] || { echo "DATA_DIR 必须是绝对路径" >&2; exit 2; }
[[ -f "${DATA_DIR}/metadata_rank0.json" ]] || { echo "找不到 LIBERO 数据: ${DATA_DIR}" >&2; exit 2; }
[[ -f "${STATS_DIR}/norm_stats.json" ]] || { echo "找不到统计量: ${STATS_DIR}/norm_stats.json" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES=1,2,3,4
export OMP_NUM_THREADS=1
export WANDB_MODE="${WANDB_MODE:-disabled}"

exec torchrun --standalone --nproc_per_node=4 train.py \
  --distributed=True \
  --domains=libero \
  --ptv3_size=small \
  --ptv3_patch_size=128 \
  --predictor_dim=128 \
  --data_dirs="${DATA_DIR}" \
  --norm_stats_path="${STATS_DIR}" \
  --batch_size=22 \
  --eval_freq=-1 \
  --save_freq=300 \
  --num_workers=16 \
  --eval_num_workers=5 \
  --num_epochs=200 \
  --max_scene_points=12000 \
  --max_robot_points=1024 \
  --train_min_num_cameras=2 \
  --train_max_num_cameras=2 \
  --eval_min_num_cameras=2 \
  --eval_max_num_cameras=2 \
  --deterministic_data=True \
  --deterministic_train=True \
  --disable_compile=True \
  --seed=512026 \
  --exp_name="${EXP_NAME}"
