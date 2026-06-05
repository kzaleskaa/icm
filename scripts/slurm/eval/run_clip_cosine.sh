#!/bin/bash
# CLIP cosine-similarity evaluation between generated images and reference attribute prompts.
set -euo pipefail

BASE_DIR="${BASE_DIR:?directory containing the *.pkl experiment outputs}"
OUT_FILE="${OUT_FILE:-${BASE_DIR}/clip_cosine.csv}"
GROUPS="${GROUPS:-gender,age,race}"

export PYTHONPATH="${PYTHONPATH:-$PWD}"

python scripts/eval/sae_paper_clip_pkl.py \
    --base_dir "$BASE_DIR" \
    --output_file "$OUT_FILE" \
    --groups "$GROUPS"
