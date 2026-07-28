#!/usr/bin/env bash
set -euo pipefail

# Continue the BEHAVIOR uniform-subset extraction locally without Docker.
# Existing flow outputs are skipped by run_behavior_flows_local.sh.

POINTWORLD_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export POINTWORLD_REPO
export BEHAVIOR_1K_REPO="$POINTWORLD_REPO/../BEHAVIOR-1K"
export OG_DATA_ROOT="$BEHAVIOR_1K_REPO/datasets"
export POINTWORLD_CACHE_DIR="$POINTWORLD_REPO/.cache/pointworld"
export BEHAVIOR_ROOT="$POINTWORLD_REPO/restore_data/pointworld_behavior_subset_restored/behavior_unisamp"
export INPUT_LIST="$POINTWORLD_REPO/simulation/behavior_uniform_subset_paths.txt"
export RANK=0
export WORLD_SIZE=1

# Domestic mirror by default. Override before running to use another endpoint.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export POINTWORLD_HF_DOWNLOAD_RETRIES="${POINTWORLD_HF_DOWNLOAD_RETRIES:-8}"
if [[ "$HF_ENDPOINT" == https://hf-mirror.com* ]]; then
  export NO_PROXY="${NO_PROXY:+$NO_PROXY,}hf-mirror.com"
  export no_proxy="${no_proxy:+$no_proxy,}hf-mirror.com"
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "behavior" ]]; then
  echo "Please activate the conda environment first: conda activate behavior" >&2
  exit 1
fi

exec bash "$POINTWORLD_REPO/run_behavior_flows_local.sh"
