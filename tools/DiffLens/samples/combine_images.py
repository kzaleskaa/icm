import os
import shutil
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", type=str, help="Base directory containing subfolders with cropped images")
    parser.add_argument("output_dir", type=str, help="Output directory to collect all images")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for subdir in os.listdir(args.base_dir):
        subpath = os.path.join(args.base_dir, subdir)
        if os.path.isdir(subpath):
            for fname in os.listdir(subpath):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    src = os.path.join(subpath, fname)

                    prompt = subdir.replace(" ", "_")
                    new_name = f"{prompt}_{fname}"
                    dst = os.path.join(args.output_dir, new_name)

                    shutil.copy2(src, dst)

    print(f"All images collected into: {args.output_dir}")
