#!/bin/bash
# Crop faces from generated images using DiffLens' face-crop utility.
# Requires DiffLens cloned into tools/DiffLens (see tools/DiffLens/README.md).
set -euo pipefail

DIFFLENS_DIR="${DIFFLENS_DIR:-tools/DiffLens}"
IN_BASE="${IN_BASE:?base directory containing the per-prompt split image folders}"
OUT_BASE="${OUT_BASE:-${IN_BASE%/}/cropped_images}"
# space-separated list of per-prompt subdirectories to crop
SUBDIRS="${SUBDIRS:-A-face-of-a-doctor A-face-of-a-firefighter A-face-of-a-nurse A-face-of-a-receptionist}"

for sub in $SUBDIRS; do
    in_path="$IN_BASE/$sub"
    out_path="$OUT_BASE/$sub"
    mkdir -p "$out_path"
    python3 "$DIFFLENS_DIR/evaluation/crop_face/crop.py" \
        --image_paths "$in_path/*.png" \
        --output_dir "$out_path"
done
