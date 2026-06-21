#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${1:-google/siglip2-base-patch16-256}"
LOCAL_DIR="${2:-third_party/siglip/checkpoints/${MODEL_ID//\//-}}"

# Xet can be slow or unavailable on some research servers; ordinary HTTP is
# usually easier to debug and resumes through the Hugging Face cache.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

huggingface-cli download "${MODEL_ID}" --local-dir "${LOCAL_DIR}"

echo "Downloaded ${MODEL_ID} to ${LOCAL_DIR}"
