#!/bin/bash
# Run cross-attention embedding replacement on the Sana transformer.
set -euo pipefail

PROMPTS_FILE="${PROMPTS_FILE:-data/decision_prompts_longer/age_prompts.json}"
OUT_DIR="${OUT_DIR:-results/layer_emb_replace/sana/$(basename "$PROMPTS_FILE" .json)}"
mkdir -p "$OUT_DIR"

# Edit this list to target specific cross-attention modules in the Sana transformer.
LAYERS=(
    "transformer_blocks.0.attn2"
    "transformer_blocks.5.attn2"
    "transformer_blocks.10.attn2"
    "transformer_blocks.15.attn2"
    "None"
)

export PYTHONPATH="${PYTHONPATH:-$PWD}"

for layer in "${LAYERS[@]}"; do
    safe="${layer//./_}"
    out_file="$OUT_DIR/layer_${safe}.pkl"
    echo "==> Replacing layer: $layer"
    if [ "$layer" = "None" ]; then
        python scripts/layers/replace_emb_hidden_states_sana.py \
            --prompts_file "$PROMPTS_FILE" \
            --output_file "$out_file"
    else
        python scripts/layers/replace_emb_hidden_states_sana.py \
            --prompts_file "$PROMPTS_FILE" \
            --output_file "$out_file" \
            --layer_name "$layer"
    fi
done
