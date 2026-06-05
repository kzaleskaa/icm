from cleanfid import fid
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_paths", nargs="+", type=str, help="One or more test directories")
    args = parser.parse_args()

    base_path = "/net/scratch/hscra/plgrid/plgkzaleska/projects/decision-in-diffusion-models/data/ffhq/all_images"
    output_file = "./fid_results.txt"

    fid_scores = []
    for path in args.test_paths:
        score = fid.compute_fid(base_path, path)
        fid_scores.append((path, score))
        print(f"FID score for {path}: {score}")

    with open(output_file, 'w') as file:
        for path, score in fid_scores:
            file.write(f"{path}: {score}\n")

    print(f"Results saved to {output_file}")
