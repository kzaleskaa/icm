#!/bin/bash
# Example LoRA finetuning run for Stable Diffusion v1.5, targeting a subset of
# cross-attention modules. Replace MODEL_NAME / DATASET_DIR / OUTPUT_DIR for
# your setup.
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-stable-diffusion-v1-5/stable-diffusion-v1-5}"
DATASET_DIR="${DATASET_DIR:?path to your image-caption dataset directory}"
OUTPUT_DIR="${OUTPUT_DIR:-models/sd-lora}"

# Cross-attention modules to adapt. Comma-separated list of fully-qualified module names.
LAYERS="${LAYERS:-down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_k,\
down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v,\
down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_q,\
down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_out.0,\
up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k,\
up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_v,\
up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_q,\
up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_out.0}"

accelerate launch scripts/lora/train_text_to_image_lora.py \
    --pretrained_model_name_or_path "$MODEL_NAME" \
    --train_data_dir "$DATASET_DIR" \
    --caption_column "text" \
    --lora_target_modules "$LAYERS" \
    --rank 50 \
    --mixed_precision "fp16" \
    --dataloader_num_workers 8 \
    --resolution 512 --center_crop --random_flip \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --learning_rate 5e-5 --lr_scheduler "cosine" --lr_warmup_steps 500 \
    --max_grad_norm 1 \
    --output_dir "$OUTPUT_DIR" \
    --validation_prompt "A face of a nurse" \
    --report_to "wandb" \
    --validation_epochs 10 \
    --num_train_epochs 100 \
    --checkpointing_steps 5000 \
    --seed 42
