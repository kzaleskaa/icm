#!/bin/bash
# Train per-timestep logistic-regression probes across every cross-attention
# layer in the activation packs. For each timestep the script auto-discovers
# the layers available in the packs, splits 80/20 with stratification, fits
# a logistic regression per (layer, timestep), and writes:
#
#   $OUT_DIR/results_timestep_<NNN>.csv         # train + test accuracy
#   $OUT_DIR/vectors/timestep_<NNN>_vectors.npz # theta per layer
#   $OUT_DIR/vectors/timestep_<NNN>_metadata.joblib
set -euo pipefail

DATASET="${DATASET:-woman_man_prompts_neutral_v1}"
ACTIVATIONS_ROOT="${ACTIVATIONS_ROOT:-activations/sd_attn2}"
TRAIN_DIR="${ACTIVATIONS_ROOT}/${DATASET}/packs"
OUT_DIR="${OUT_DIR:-logreg_runs/${DATASET}}"
POS_LABEL="${POS_LABEL:-man}"
NUM_STEPS="${NUM_STEPS:-50}"

export PYTHONPATH="${PYTHONPATH:-$PWD}"

for step in $(seq 0 $((NUM_STEPS - 1))); do
    python scripts/probing/train_logreg_layer_timestep.py \
        --timestep "$step" \
        --train_dir "$TRAIN_DIR" \
        --out_dir "$OUT_DIR" \
        --pos_label "$POS_LABEL"
done
