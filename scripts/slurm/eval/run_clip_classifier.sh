#!/bin/bash
# CLIP-based attribute classification over pickles produced by the
# layer-replacement experiments (counts per attribute).
set -euo pipefail

BASE_DIR="${BASE_DIR:?directory containing the *.pkl experiment outputs}"
OUT_FILE="${OUT_FILE:-${BASE_DIR}/clip_classifier_counts.csv}"
GROUPS="${GROUPS:-gender,age,race}"

export PYTHONPATH="${PYTHONPATH:-$PWD}"

python scripts/eval/clip_as_classifier_pkl.py \
    --base_dir "$BASE_DIR" \
    --output_file "$OUT_FILE" \
    --groups "$GROUPS"
