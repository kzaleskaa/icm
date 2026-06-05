#!/bin/bash
# Collect SAE-paper evaluation metrics (Fairface, CLIP-I, CLIP-T, FID) for a
# results directory, using DiffLens' evaluation scripts. The protected
# attribute is inferred from the directory name (age / gender / race).
# Requires DiffLens cloned into tools/DiffLens (see tools/DiffLens/README.md).
set -euo pipefail

DIFFLENS_DIR="${DIFFLENS_DIR:-tools/DiffLens}"
BASE="${BASE:?results directory (must contain splitted_images/ and cropped_images/)}"

# Resolve to absolute paths before changing into the DiffLens directory.
BASE="$(realpath "$BASE")"
cd "$DIFFLENS_DIR"
export PYTHONPATH="$PWD"

mkdir -p "${BASE}/all_images"
python samples/combine_images.py "${BASE}/splitted_images" "${BASE}/all_images"

# Single file collecting all metrics for this run.
metrics_file="${BASE}/all_metrics.txt"
{
    echo "=================================="
    echo "All Metrics for: ${BASE}"
    echo "Generated on: $(date)"
    echo "=================================="
    echo ""
} > "$metrics_file"

echo "Running Fairface evaluation for ${BASE}"
echo "--- Fairface Evaluation ---" >> "$metrics_file"
case "${BASE}" in
    *age*)    python evaluation/Fairface/age.py    "${BASE}/cropped_images" | tee -a "$metrics_file" ;;
    *gender*) python evaluation/Fairface/gender.py "${BASE}/cropped_images" | tee -a "$metrics_file" ;;
    *)        python evaluation/Fairface/race.py   "${BASE}/cropped_images" | tee -a "$metrics_file" ;;
esac
echo "" >> "$metrics_file"

echo "--- CLIP Image Score ---" >> "$metrics_file"
python evaluation/CLIP/clip_image_score.py "${BASE}/splitted_images" | tee -a "$metrics_file"
echo "" >> "$metrics_file"

echo "--- CLIP Text Score ---" >> "$metrics_file"
python evaluation/CLIP-T/clip_text_score.py "${BASE}/splitted_images" | tee -a "$metrics_file"
echo "" >> "$metrics_file"

echo "--- FID Score ---" >> "$metrics_file"
python evaluation/FID/fid.py "${BASE}/all_images" | tee -a "$metrics_file"
echo "" >> "$metrics_file"

echo "==================================" >> "$metrics_file"
echo "All metrics saved to: ${metrics_file}"

rm -rf "${BASE}/all_images"
