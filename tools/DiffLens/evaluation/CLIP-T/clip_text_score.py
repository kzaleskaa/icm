import torch
import clip
from PIL import Image
import os
from tqdm import tqdm
import glob

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

def calculate_clip_t_score(image_folder, text_prompt):
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.png")) +
                         glob.glob(os.path.join(image_folder, "*.jpg")) +
                         glob.glob(os.path.join(image_folder, "*.jpeg")))

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_folder}")

    text = clip.tokenize([text_prompt]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)

    cos = torch.nn.CosineSimilarity(dim=1)
    total_score = 0.0

    for image_path in tqdm(image_paths, desc=f"Processing {text_prompt}", leave=False):
        try:
            image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
            similarity = cos(image_features, text_features).item()
            similarity = (similarity + 1) / 2  # scale to [0,1]
            total_score += similarity
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    return total_score / len(image_paths)

if __name__ == "__main__":
    print("=" * 60)
    print("Calculating CLIP-T scores for generated images")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("parent_dir", type=str, help="Parent directory containing prompt folders")
    args = parser.parse_args()

    parent_dir = args.parent_dir

    print(parent_dir)

    prompt_folders = sorted([
        f for f in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, f))
    ])

    all_scores = []

    for folder in prompt_folders:
        image_dir = os.path.join(parent_dir, folder)
        text_prompt = folder.replace("-", " ")

        try:
            score = calculate_clip_t_score(image_dir, text_prompt)
            all_scores.append(score)
            print(f"{folder}: CLIP-T = {score:.4f}")
        except Exception as e:
            print(f"Error processing {folder}: {e}")

    if all_scores:
        mean_clip_t = sum(all_scores) / len(all_scores)
        print(f"\nMean CLIP-T across all prompts: {mean_clip_t:.4f}")
    else:
        print("No valid scores computed.")