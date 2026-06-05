import argparse
import torch
import clip
from PIL import Image
import os
from tqdm import tqdm
import glob

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

def calculate_clip_similarity(folder1, folder2):
    files1 = sorted(glob.glob(os.path.join(folder1, "*.png")) + 
                    glob.glob(os.path.join(folder1, "*.jpg")) + 
                    glob.glob(os.path.join(folder1, "*.jpeg")))
    files2 = sorted(glob.glob(os.path.join(folder2, "*.png")) + 
                    glob.glob(os.path.join(folder2, "*.jpg")) + 
                    glob.glob(os.path.join(folder2, "*.jpeg")))
    
    min_length = min(len(files1), len(files2))

    files1 = files1[:min_length]
    files2 = files2[:min_length]

    if len(files1) != len(files2):
        raise ValueError("The number of images in both folders should be the same.")

    cos = torch.nn.CosineSimilarity(dim=1)
    total_similarity = 0

    for file1, file2 in tqdm(zip(files1, files2), total=len(files1), desc="Processing images"):
        try:
            # print(file1, "             ", file2)
            img1 = preprocess(Image.open(file1)).unsqueeze(0).to(device)
            img2 = preprocess(Image.open(file2)).unsqueeze(0).to(device)

            with torch.no_grad():
                features1 = model.encode_image(img1)
                features2 = model.encode_image(img2)

            similarity = cos(features1, features2).item()
            similarity = (similarity + 1) / 2

            # print(similarity)
            total_similarity += similarity
        except Exception as e:
            print(f"Error processing images {file1} and {file2}: {e}")

    average_similarity = total_similarity / len(files1)
    return average_similarity

if __name__ == "__main__":
    print("=" * 60)
    print("Calculating CLIP-I scores for generated images")
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path", type=str, help="Base path to model results (general folder)")
    args = parser.parse_args()

    folder1 = [
        "/net/scratch/hscra/plgrid/plgkzaleska/projects/decision-in-diffusion-models/samples/original_images/splitted_images/A-face-of-a-doctor",
        "/net/scratch/hscra/plgrid/plgkzaleska/projects/decision-in-diffusion-models/samples/original_images/splitted_images/A-face-of-a-firefighter",
        "/net/scratch/hscra/plgrid/plgkzaleska/projects/decision-in-diffusion-models/samples/original_images/splitted_images/A-face-of-a-nurse",
        "/net/scratch/hscra/plgrid/plgkzaleska/projects/decision-in-diffusion-models/samples/original_images/splitted_images/A-face-of-a-receptionist",
    ]

    prompts = [
        "A-face-of-a-doctor",
        "A-face-of-a-firefighter",
        "A-face-of-a-nurse",
        "A-face-of-a-receptionist",
    ]

    paths_list = [
        os.path.join(args.results_path, f"{p}") for p in prompts
    ]

    print(paths_list)

    all_scores = []
    for _path, _folder in zip(paths_list, folder1):
        try:
            avg_similarity = calculate_clip_similarity(_folder, _path)
            print(_path)
            print(f"Average CLIP similarity score: {avg_similarity:.4f}")
            all_scores.append(avg_similarity)
        except Exception as e:
            print(f"An error occurred: {e}")

    if all_scores:
        mean_score = sum(all_scores) / len(all_scores)
        print("=" * 60)
        print(f"Final mean similarity across all prompts: {mean_score:.4f}")