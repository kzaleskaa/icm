#!/bin/bash
# Run cross-attention embedding replacement on Stable Diffusion XL.
# Set PROMPTS_FILE / OUT_DIR / LAYERS via environment if you want to override defaults.
set -euo pipefail

PROMPTS_FILE="${PROMPTS_FILE:-data/decision_prompts_longer/gender_prompts.json}"
OUT_DIR="${OUT_DIR:-results/layer_emb_replace/sdxl/$(basename "$PROMPTS_FILE" .json)}"
mkdir -p "$OUT_DIR"

# Edit this list to target specific cross-attention layers in the SDXL UNet.
LAYERS=(
    "down_blocks.2.attentions.0.transformer_blocks.0.attn2"
    "down_blocks.2.attentions.1.transformer_blocks.0.attn2"
    "mid_block.attentions.0.transformer_blocks.0.attn2"
    "up_blocks.0.attentions.0.transformer_blocks.0.attn2"
    "up_blocks.0.attentions.1.transformer_blocks.0.attn2"
    "None"
)

export PYTHONPATH="${PYTHONPATH:-$PWD}"

for layer in "${LAYERS[@]}"; do
    safe="${layer//./_}"
    out_file="$OUT_DIR/layer_${safe}.pkl"
    echo "==> Replacing layer: $layer"
    if [ "$layer" = "None" ]; then
        python scripts/layers/replace_emb_hidden_states_sdxl.py \
            --prompts_file "$PROMPTS_FILE" \
            --output_file "$out_file"
    else
        python scripts/layers/replace_emb_hidden_states_sdxl.py \
            --prompts_file "$PROMPTS_FILE" \
            --output_file "$out_file" \
            --layer_name "$layer"
    fi
done
