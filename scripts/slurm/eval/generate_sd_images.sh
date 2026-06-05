#!/bin/bash
# Generate Stable Diffusion images with DiffLens (logreg-vector steering) from
# one or more entry config .yaml files. Run from the repo root.
# Requires DiffLens under tools/DiffLens (see tools/DiffLens/README.md) and the
# trained logreg models referenced by the config's lr_models_path.
set -euo pipefail

DIFFLENS_DIR="${DIFFLENS_DIR:-tools/DiffLens}"
# space-separated list of entry config .yaml files (paths relative to repo root)
CONFIGS="${CONFIGS:-tools/DiffLens/config/SD/config_gender_lr_alfa_10.yaml}"

# Paths inside the configs (sample_dir, lr_models_path) are relative to the repo
# root; difflens must be importable for `python -m difflens`.
export PYTHONPATH="${PYTHONPATH:-$DIFFLENS_DIR}"

for cfg in $CONFIGS; do
    python -m difflens "$cfg"
done
