import argparse
import torch
from typing import Dict, List
from diffusers import SanaPipeline
from PIL import Image
import random
import numpy as np
import os
import torch.nn as nn

from src.utils import read_json, save_to_pickle

SEEDS: List[int] = [0, 42, 123]
NUM_IMAGES_PER_PROMPT = 3

def set_generate_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class CollectInputHook(nn.Module):
    def __init__(self, buffer):
        self.buffer = buffer

    @torch.no_grad()
    def __call__(self, module, args, kwargs, output):
        encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
        attention_mask = kwargs.get("attention_mask", None)
        _, prompt_embeds = encoder_hidden_states.chunk(2)
        _, prompt_attention_mask = attention_mask.chunk(2)
        if len(self.buffer) == 0:
            self.buffer.append(prompt_embeds)
            self.buffer.append(prompt_attention_mask)
        return output

class ModifyInputHook:
    def __init__(self, buffer):
        self.buffer = buffer

    @torch.no_grad()
    def __call__(self, module, args, kwargs):
        encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
        attention_mask = kwargs.get("attention_mask", None)
        negative_prompt_embeds, _ = encoder_hidden_states.chunk(2, dim=0)
        negative_prompt_attention_mask, _ = attention_mask.chunk(2, dim=0)
        kwargs["encoder_hidden_states"] = torch.cat([negative_prompt_embeds, self.buffer[0]], dim=0)
        kwargs["attention_mask"] = torch.cat([negative_prompt_attention_mask, self.buffer[1]], dim=0)
        return args, kwargs

def generate_images(pipeline, prompt: str, seed: int, num_inference_steps: int = 20) -> List[Image.Image]:
    output_images = []
    images = pipeline(
        prompt,
        num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator(device=pipeline.device).manual_seed(seed),
    ).images
    output_images.extend(images)
    return output_images

def process_prompt_pair(model, prompt1, prompt2s, seed, layer_name):
    if layer_name is None:
        results = generate_images(model, prompt1, seed)
    else:
        results = {"images": {}}
        for name, module in model.transformer.named_modules():
            if name != layer_name:
                continue
            print(f"Processing layer: {name}")

            for prompt2 in prompt2s:
                buffer = []
                input_hook = module.register_forward_hook(CollectInputHook(buffer), with_kwargs=True)
                _ = generate_images(model, prompt2, seed, num_inference_steps=1)
                input_hook.remove()

                input_hook = module.register_forward_pre_hook(ModifyInputHook(buffer), with_kwargs=True)
                results["images"][prompt2] = generate_images(model, prompt1, seed, num_inference_steps=20)
                input_hook.remove()

    return results

def main(args):
    set_generate_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SanaPipeline.from_pretrained("Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers", torch_dtype=torch.float32).to(device)
    if hasattr(model, "text_encoder"): model.text_encoder.to(torch.bfloat16)
    if hasattr(model, "transformer"): model.transformer = model.transformer.to(torch.bfloat16)

    prompts = read_json(args.prompts_file)

    results = {}
    for seed in SEEDS:
        print(f"[MAIN] Processing seed: {seed} with layer: {args.layer_name}")
        results[seed] = {}
        for base_prompt, variations in prompts.items():
            results[seed][base_prompt] = process_prompt_pair(
                model, base_prompt, variations, seed, args.layer_name
            )

    save_to_pickle(results, args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_file", type=str, required=True, help="Path to prompts JSON file")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--layer_name", type=str, default=None)
    args = parser.parse_args()
    main(args)
