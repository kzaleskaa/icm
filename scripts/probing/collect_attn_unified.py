"""
Unified script for collecting cross-attention outputs from different diffusion models.

This script consolidates all the collect_cross_attention_output_* scripts into a single
configurable script that supports:
- Multiple models: SD v1.5, SDXL, Sana
- Multiple attention layers: attn1, attn2
"""

import argparse
import json
import re
from pathlib import Path
import torch
import torch.nn as nn
import clip
import numpy as np
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DDIMScheduler, SanaPipeline
from scripts.probing import prompts as P
from scripts.probing.constants import DATASET_CHOICES, label_config


def parse_args():
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(description="Collect cross-attention outputs from diffusion models")
    
    # Model configuration
    ap.add_argument("--model", type=str, required=True, 
                    choices=["sd", "sdxl", "sana"],
                    help="Model type: sd (Stable Diffusion v1.5), sdxl (Stable Diffusion XL), sana (Sana)")
    ap.add_argument("--attention_layer", type=str, required=True,
                    choices=["attn1", "attn2"],
                    help="Attention layer to collect from: attn1 or attn2")
    
    # Dataset configuration
    ap.add_argument("--dataset", type=str, required=True, choices=DATASET_CHOICES,
                    help="Dataset name for prompts and labels")
    
    # Output configuration
    ap.add_argument("--output_dir", type=str, required=True,
                    help="Base output directory")
    ap.add_argument("--job_name", type=str, required=True,
                    help="Job name for organizing outputs")
    
    # Generation parameters
    ap.add_argument("--num_images", type=int, default=5,
                    help="Number of images to generate per prompt")
    ap.add_argument("--num_inference_steps", type=int, default=50,
                    help="Number of inference steps")
    ap.add_argument("--seed_start", type=int, default=0,
                    help="Starting seed for random generation")
    ap.add_argument("--batch_size", type=int, default=1,
                    help="Number of images to generate at once (default: 1)")
    
    # System configuration
    ap.add_argument("--device", type=str, default="cuda",
                    help="Device to run on (cuda or cpu)")
    ap.add_argument("--shard_size", type=int, default=1,
                    help="Number of samples per shard file")
    
    return ap.parse_args()


def slugify(s):
    """Convert string to slug format."""
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def setup_output(root):
    """Create output directory structure."""
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "packs").mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)


def load_pipeline(model_type, device, steps):
    """
    Load the appropriate diffusion pipeline based on model type.
    
    Args:
        model_type: One of "sd", "sdxl", "sana"
        device: Device to load model on
        steps: Number of inference steps
        
    Returns:
        Loaded pipeline
    """
    if model_type == "sd":
        pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(device)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.set_timesteps(steps)
        
    elif model_type == "sdxl":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(device)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.set_timesteps(steps)
        
    elif model_type == "sana":
        pipe = SanaPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
            torch_dtype=torch.float32
        ).to(device)
        # Convert specific components to bfloat16 for efficiency
        if hasattr(pipe, "text_encoder"):
            pipe.text_encoder.to(torch.bfloat16)
        if hasattr(pipe, "transformer"):
            pipe.transformer = pipe.transformer.to(torch.bfloat16)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return pipe


def find_attention_modules(model, model_type, attention_layer):
    """
    Find attention modules in the model.
    
    Args:
        model: The diffusion model pipeline
        model_type: One of "sd", "sdxl", "sana"
        attention_layer: One of "attn1", "attn2"
        
    Returns:
        List of (name, module) tuples for attention layers
    """
    # Get the appropriate module to search
    if model_type in ["sd", "sdxl"]:
        search_module = model.unet
    elif model_type == "sana":
        search_module = model.transformer
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Find matching attention modules
    mods = []
    for n, m in search_module.named_modules():
        if n.endswith(f".{attention_layer}"):
            mods.append((n, m))
    
    return mods


class CollectResults(nn.Module):
    """Hook to collect attention outputs during forward pass."""
    
    def __init__(self, store_list, layer_name, max_steps=50, batch_size=1):
        super().__init__()
        self.store_list = store_list  # List of dicts, one per image in batch
        self.layer_name = layer_name
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.step_idx = 0
    
    @torch.no_grad()
    def __call__(self, module, args, kwargs, output):
        """Collect and store attention output with mean pooling for batched generation."""
        t = output.detach()
        
        # Handle CFG: split into unconditional and conditional
        chunks = t.chunk(2, dim=0)
        t_conditional = chunks[1]  # [batch_size, seq_len, hidden_dim]
        
        # Process each image in the batch separately
        for batch_idx in range(self.batch_size):
            t_sample = t_conditional[batch_idx].cpu()  # [seq_len, hidden_dim]
            t_pooled = t_sample.mean(dim=0)  # [hidden_dim]
            
            # Store in the corresponding dict for this batch item
            self.store_list[batch_idx].setdefault(self.layer_name, {})[self.step_idx] = t_pooled
        
        self.step_idx = (self.step_idx + 1) % self.max_steps
        return output


def normalize_group_spec(text_spec):
    """
    Normalize text specification to group format.
    
    Handles both simple pairs and complex group specifications.
    """
    if isinstance(text_spec, (list, tuple)) and len(text_spec) == 2 and \
       (isinstance(text_spec[0], (list, tuple)) or isinstance(text_spec[1], (list, tuple))):
        g0 = list(text_spec[0])
        g1 = list(text_spec[1])
    else:
        g0 = [text_spec[0]]
        g1 = [text_spec[1]]
    return [g0, g1]


def load_clip_groups(text_groups, device):
    """
    Load CLIP model and encode text groups for classification.
    
    Args:
        text_groups: Text specifications for two groups
        device: Device to load on
        
    Returns:
        Tuple of (clip_model, preprocess, group_features)
    """
    model_clip, preprocess = clip.load("ViT-L/14", device=device)
    model_clip.eval()
    groups = normalize_group_spec(text_groups)
    feats = []
    with torch.no_grad():
        for group in groups:
            toks = clip.tokenize(group).to(device)
            tf = model_clip.encode_text(toks)
            tf = tf / tf.norm(dim=-1, keepdim=True)
            feats.append(tf)
    return model_clip, preprocess, feats


def classify_batch_groups(pil_imgs, preprocess, model_clip, group_text_features, device, label0, label1, text_groups=None):
    """
    Classify images using CLIP into two groups.
    
    Args:
        pil_imgs: List of PIL images
        preprocess: CLIP preprocessing function
        model_clip: CLIP model
        group_text_features: Encoded text features for each group
        device: Device to run on
        label0: Label for group 0
        label1: Label for group 1
        text_groups: Original text groups (for determining best label)
        
    Returns:
        Tuple of (labels, scores_0, scores_1, best_labels)
    """
    xs = torch.stack([preprocess(im.convert("RGB")) for im in pil_imgs]).to(device)
    with torch.no_grad():
        img_feats = model_clip.encode_image(xs)
    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
    s0_all = img_feats @ group_text_features[0].T  # [batch_size, num_texts_in_group0]
    s1_all = img_feats @ group_text_features[1].T  # [batch_size, num_texts_in_group1]
    s0, s0_idx = s0_all.max(dim=1)
    s1, s1_idx = s1_all.max(dim=1)
    idxs = (s1 > s0).long()
    labs = [label0 if int(i) == 0 else label1 for i in idxs]
    
    # Determine best matching text for each image
    best_labels = []
    if text_groups is not None:
        groups = normalize_group_spec(text_groups)
        for batch_idx in range(len(pil_imgs)):
            if idxs[batch_idx] == 0:
                # Group 0 won
                best_text_idx = int(s0_idx[batch_idx])
                best_text = groups[0][best_text_idx]
            else:
                # Group 1 won
                best_text_idx = int(s1_idx[batch_idx])
                best_text = groups[1][best_text_idx]
            best_labels.append(best_text)
    else:
        best_labels = [None] * len(pil_imgs)
    
    return labs, s0.tolist(), s1.tolist(), best_labels


def pack_acts(acts):
    """
    Pack activation dictionaries into tensors.
    
    Args:
        acts: Dictionary of layer_name -> step -> tensors
        
    Returns:
        Packed dictionary with stacked tensors
    """
    packed = {}
    for layer_name, steps_dict in acts.items():
        inner = {}
        for s, tensors in steps_dict.items():
            if isinstance(tensors, list):
                inner[int(s)] = torch.stack([t.detach().cpu() for t in tensors], dim=0)
            else:
                inner[int(s)] = tensors.detach().cpu()
        packed[layer_name] = inner
    return packed


def save_pack_if_needed(pack, shard_size, out_root, pack_idx):
    """
    Save pack to disk if it reaches shard size.
    
    Args:
        pack: List of data items
        shard_size: Maximum items per shard
        out_root: Output directory root
        pack_idx: Current pack index
        
    Returns:
        Tuple of (updated_pack, updated_pack_idx)
    """
    if len(pack) >= shard_size:
        torch.save({"items": pack}, out_root / "packs" / f"pack_{pack_idx:04d}.pt")
        return [], pack_idx + 1
    return pack, pack_idx


def save_results_json(results, out_root):
    """Save results metadata to JSON file."""
    with open(out_root / "meta" / "results.json", "w") as f:
        json.dump(results, f, indent=2)


def run_generation(pipe, attn_list, prompts, device, steps, seed_start, out_root, shard_size, classify_fn, num_images=5, batch_size=1):
    """
    Run generation loop and collect attention outputs with mean pooling.
    Uses num_images_per_prompt to generate multiple images efficiently in one forward pass.
    
    Args:
        pipe: Diffusion pipeline
        attn_list: List of attention modules to hook
        prompts: List of text prompts
        device: Device to run on
        steps: Number of inference steps
        seed_start: Starting seed
        out_root: Output directory
        shard_size: Items per shard file
        classify_fn: Function to classify generated images
        num_images: Number of images to generate per prompt
        batch_size: Number of images to generate at once (num_images_per_prompt)
        
    Returns:
        List of results metadata
    """
    results = []
    pack = []
    pack_idx = 0
    
    # Calculate how many batches we need per prompt
    num_batches_per_prompt = (num_images + batch_size - 1) // batch_size
    
    # Iterate over prompts
    for prompt_idx, ptxt in enumerate(tqdm(prompts, desc="Processing prompts")):
        prompt_slug = slugify(ptxt)
        img_dir = out_root / "images" / prompt_slug
        img_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate images in batches for this prompt
        for batch_idx in range(num_batches_per_prompt):
            # Calculate how many images to generate in this batch
            start_img_idx = batch_idx * batch_size
            end_img_idx = min(start_img_idx + batch_size, num_images)
            current_batch_size = end_img_idx - start_img_idx
            
            # Create seeds for each image in the batch
            batch_seeds = [
                seed_start + (start_img_idx + i) * len(prompts) + prompt_idx 
                for i in range(current_batch_size)
            ]
            
            # Create activation stores for each image in batch
            acts_list = [{} for _ in range(current_batch_size)]
            
            # Set up hooks for batch collection
            hooks = [m.register_forward_hook(
                CollectResults(acts_list, n, max_steps=steps, batch_size=current_batch_size), 
                with_kwargs=True
            ) for n, m in attn_list]
            
            # Generate batch using num_images_per_prompt with multiple generators
            # Create a list of generators, one for each image
            generators = [torch.Generator(device=device).manual_seed(seed) for seed in batch_seeds]
            out = pipe(
                ptxt,
                num_inference_steps=steps,
                num_images_per_prompt=current_batch_size,
                generator=generators
            )
            
            # Remove hooks
            for h in hooks:
                h.remove()
            
            # Get generated images
            batch_images = out.images
            
            # Classify batch
            labels, s0s, s1s, best_labels = classify_fn(batch_images)
            
            # Process and save results
            for img_idx_in_batch, (img, label, s0, s1, best_label, seed) in enumerate(
                zip(batch_images, labels, s0s, s1s, best_labels, batch_seeds)
            ):
                
                final_path = img_dir / f"seed-{seed}.png"
                img.save(final_path)
                
                item = {
                    "seed": seed,
                    "prompt": ptxt,
                    "prompt_slug": prompt_slug,
                    "label": label,
                    "best_label": best_label,
                    "image_path": str(final_path),
                    "score_0": float(s0),
                    "score_1": float(s1),
                }
                
                results.append(item.copy())
                pack.append(item | {"acts": pack_acts(acts_list[img_idx_in_batch])})
                
                pack, pack_idx = save_pack_if_needed(pack, shard_size, out_root, pack_idx)
    
    # Save remaining items
    if len(pack) > 0:
        torch.save({"items": pack}, out_root / "packs" / f"pack_{pack_idx:04d}.pt")
    
    return results


def main():
    """Main execution function."""
    args = parse_args()
    
    # Load prompts and labels
    prompts = getattr(P, args.dataset)
    text_pair, labels_two = label_config(args.dataset)
    label0, label1 = labels_two
    
    # Set up parameters
    device = args.device
    num_images = args.num_images
    steps = args.num_inference_steps
    seed_start = args.seed_start
    out_root = Path(args.output_dir) / args.job_name
    
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Attention layer: {args.attention_layer}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Output: {out_root}")
    print(f"  Device: {device}")
    
    # Set up output directories
    setup_output(out_root)
    
    # Load pipeline
    print(f"\nLoading {args.model} pipeline...")
    pipe = load_pipeline(args.model, device, steps)
    
    # Find attention modules
    print(f"Finding {args.attention_layer} modules...")
    attn_list = find_attention_modules(pipe, args.model, args.attention_layer)
    print(f"Found {len(attn_list)} attention modules")
    
    # Load CLIP for classification
    print("Loading CLIP model...")
    model_clip, preprocess, group_text_features = load_clip_groups(text_pair, device)
    
    def classify_fn(imgs):
        labs, s0, s1, best_labels = classify_batch_groups(
            imgs, preprocess, model_clip, group_text_features, device, label0, label1, text_pair
        )
        return labs, s0, s1, best_labels
    
    # Run generation
    print(f"\nGenerating {num_images} images for {len(prompts)} prompts...")
    print(f"Using mean pooling for attention outputs")
    print(f"Batch size: {args.batch_size}")
    results = run_generation(
        pipe, attn_list, prompts, device, steps, seed_start, 
        out_root, args.shard_size, classify_fn, num_images, args.batch_size
    )
    
    # Save results
    save_results_json(results, out_root)
    print(f"\nComplete! Generated {len(results)} total images")
    print(f"Results saved to {out_root}")


if __name__ == "__main__":
    main()

