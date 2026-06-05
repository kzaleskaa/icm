#!/bin/bash
# Collect cross-attention activations from a diffusion model on a probing dataset.
set -euo pipefail

MODEL="${MODEL:-sd}"                       # one of: sd, sdxl, sana
ATTN_LAYER="${ATTN_LAYER:-attn2}"          # one of: attn1, attn2
DATASET="${DATASET:-woman_man_prompts_neutral_v1}"
OUT_DIR="${OUT_DIR:-activations/${MODEL}_${ATTN_LAYER}}"
JOB_NAME="${JOB_NAME:-${DATASET}}"
NUM_IMAGES="${NUM_IMAGES:-5}"
NUM_STEPS="${NUM_STEPS:-50}"

export PYTHONPATH="${PYTHONPATH:-$PWD}"

python scripts/probing/collect_attn_unified.py \
    --model "$MODEL" \
    --attention_layer "$ATTN_LAYER" \
    --dataset "$DATASET" \
    --output_dir "$OUT_DIR" \
    --job_name "$JOB_NAME" \
    --num_images "$NUM_IMAGES" \
    --num_inference_steps "$NUM_STEPS"
