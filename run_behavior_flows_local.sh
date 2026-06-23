#!/usr/bin/env bash
set -euo pipefail

# Local, non-Docker equivalent of the official BEHAVIOR flow extraction command.
# Run from any directory, ideally inside the conda environment that has OmniGibson/Isaac Sim deps.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export POINTWORLD_REPO="${POINTWORLD_REPO:-$SCRIPT_DIR}"
export BEHAVIOR_1K_REPO="${BEHAVIOR_1K_REPO:-$POINTWORLD_REPO/../BEHAVIOR-1K}"
export OG_DATA_ROOT="${OG_DATA_ROOT:-$BEHAVIOR_1K_REPO/datasets}"
export POINTWORLD_CACHE_DIR="${POINTWORLD_CACHE_DIR:-$POINTWORLD_REPO/.cache/pointworld}"
export BEHAVIOR_ROOT="${BEHAVIOR_ROOT:-$POINTWORLD_REPO/restore_data/pointworld_behavior_subset_restored/behavior_unisamp}"

INPUT_LIST="${INPUT_LIST:-$POINTWORLD_REPO/simulation/behavior_uniform_subset_paths.txt}"
RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"
MAX_EPISODES="${MAX_EPISODES:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$POINTWORLD_CACHE_DIR" "$BEHAVIOR_ROOT/flows"

required_paths=(
  "$POINTWORLD_REPO/simulation/behavior_3d_flows.py"
  "$INPUT_LIST"
  "$BEHAVIOR_1K_REPO/OmniGibson/omnigibson"
  "$OG_DATA_ROOT/behavior-1k-assets"
  "$OG_DATA_ROOT/omnigibson-robot-assets"
  "$OG_DATA_ROOT/omnigibson.key"
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
done

export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"
export OMNIGIBSON_HEADLESS="${OMNIGIBSON_HEADLESS:-1}"
export OMNIGIBSON_DATA_PATH="$OG_DATA_ROOT"
export OMNIGIBSON_DATASET_PATH="$OG_DATA_ROOT/behavior-1k-assets"
export OMNIGIBSON_ASSET_PATH="$OG_DATA_ROOT/omnigibson-robot-assets"
export OMNIGIBSON_KEY_PATH="$OG_DATA_ROOT/omnigibson.key"
export OMNIGIBSON_APPDATA_PATH="${OMNIGIBSON_APPDATA_PATH:-$BEHAVIOR_1K_REPO/OmniGibson/appdata}"
export PYTHONPATH="$POINTWORLD_REPO:$BEHAVIOR_1K_REPO/OmniGibson${PYTHONPATH:+:$PYTHONPATH}"

echo "POINTWORLD_REPO=$POINTWORLD_REPO"
echo "BEHAVIOR_1K_REPO=$BEHAVIOR_1K_REPO"
echo "OG_DATA_ROOT=$OG_DATA_ROOT"
echo "POINTWORLD_CACHE_DIR=$POINTWORLD_CACHE_DIR"
echo "BEHAVIOR_ROOT=$BEHAVIOR_ROOT"
echo "INPUT_LIST=$INPUT_LIST"
echo "RANK=$RANK WORLD_SIZE=$WORLD_SIZE"
if [[ -n "$MAX_EPISODES" ]]; then
  echo "MAX_EPISODES=$MAX_EPISODES"
fi
if [[ -n "$EXTRA_ARGS" ]]; then
  echo "EXTRA_ARGS=$EXTRA_ARGS"
fi

cd "$POINTWORLD_REPO"

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

mapfile -t input_paths < <(grep -v '^[[:space:]]*#' "$INPUT_LIST" | grep -v '^[[:space:]]*$')
total_inputs="${#input_paths[@]}"
processed=0
assigned=0

for idx in "${!input_paths[@]}"; do
  if (( idx % WORLD_SIZE != RANK )); then
    continue
  fi

  assigned=$((assigned + 1))
  if [[ -n "$MAX_EPISODES" && "$processed" -ge "$MAX_EPISODES" ]]; then
    break
  fi

  one_input_list="$tmp_dir/behavior_input_${idx}.txt"
  printf '%s\n' "${input_paths[$idx]}" > "$one_input_list"
  output_relpath="$(python - "$one_input_list" <<'PY'
import sys
from simulation.behavior.raw_input import derive_behavior_output_relpath

with open(sys.argv[1], "r", encoding="utf-8") as f:
    input_path = f.read().strip()
print(derive_behavior_output_relpath(input_path, local_input_root=None))
PY
)"
  output_path="$BEHAVIOR_ROOT/flows/$output_relpath"
  if [[ -f "$output_path" ]]; then
    echo "Skipping existing output: $output_path"
    continue
  fi

  echo "[$((processed + 1))/$assigned assigned, source index $idx/$total_inputs] ${input_paths[$idx]}"
  # shellcheck disable=SC2206
  extra_args_array=($EXTRA_ARGS)
  python simulation/behavior_3d_flows.py \
    --headless \
    --input_list "$one_input_list" \
    --output_root "$BEHAVIOR_ROOT/flows" \
    --rank 0 \
    --world_size 2 \
    "${extra_args_array[@]}"

  processed=$((processed + 1))
done

echo "Processed $processed episode(s) on rank $RANK/$WORLD_SIZE."
