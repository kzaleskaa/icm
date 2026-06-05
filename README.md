<h1 align="center">Attention, May I Have Your Decision?<br>Localizing Generative Choices in Diffusion Models</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2604.06052"><b>Paper (arXiv 2604.06052)</b></a>
</p>


## Environment setup

```bash
conda create -n icm python=3.11
conda activate icm
pip install -r requirements.txt
```

The code targets PyTorch ≥ 2.4 + diffusers ≥ 0.32. All commands below assume
the repository root is the current working directory and `PYTHONPATH=$PWD`.

---

## Probing self-attention activations

Two steps: generate images while saving **self-attention (`attn1`)**
activations, then train per-`(layer, timestep)` logistic-regression probes.

```bash
# 1. Generate images + save attn1 activations.
#    Writes activations/sd_attn1/<dataset>/{packs,images,meta}
MODEL=sd ATTN_LAYER=attn1 DATASET=woman_man_prompts_neutral_v1 \
    bash scripts/slurm/probing/collect_activations.sh

# 2. Train + evaluate probes (80/20 stratified split, test accuracy reported).
#    Writes logreg_runs/<dataset>/results_timestep_<NNN>.csv and theta vectors.
DATASET=woman_man_prompts_neutral_v1 POS_LABEL=man \
ACTIVATIONS_ROOT=activations/sd_attn1 \
    bash scripts/slurm/probing/train_logreg.sh
```

Set `MODEL=sdxl|sana` for other backbones. `scripts/probing/constants.py`
lists all supported datasets and the CLIP text labels each maps to.

---

## LoRA finetuning

Two user-supplied inputs:

1. **A dataset of images + captions**, in the standard `diffusers`
   image-folder layout — an `images/` directory plus a `metadata.jsonl` with
   one `{"file_name": "0001.jpg", "text": "example caption"}` per line.
2. **The attention modules to adapt**, via the `LAYERS` env var (or
   `--lora_target_modules`) — the paper targets a small subset identified by
   the probing analysis.

```bash
DATASET_DIR=/path/to/your/dataset \
OUTPUT_DIR=models/sd-gender-lora \
    bash scripts/lora/run_text_to_image_lora.sh

# Target specific modules:
LAYERS="up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_q,..." \
DATASET_DIR=/path/to/your/dataset OUTPUT_DIR=models/sd-gender-targeted \
    bash scripts/lora/run_text_to_image_lora.sh
```

---

## DiffLens (steering-based generation & evaluation)

We use a copy of [**DiffLens**](https://github.com/foundation-model-research/DiffLens)
under [`tools/DiffLens/`](tools/DiffLens/), **modified to run our steering /
LoRA evaluations** (logistic-probe steering generation and the bias-mitigation
metrics). See its README for setup; the large FairFace `.pt` / dlib `.dat`
checkpoints are **not** committed — download them as documented there.

```bash
# Generate steered images from the probe vectors (config's lr_models_path
# points at your logreg_runs/ models).
bash scripts/slurm/eval/generate_sd_images.sh

# Crop faces, then collect Fairface / CLIP / FID metrics.
IN_BASE=<results>/splitted_images bash scripts/slurm/eval/crop_sd_images.sh
BASE=<results>                    bash scripts/slurm/eval/collect_sd_sae_paper_results.sh
```

---

## Citation

```bibtex
@misc{zaleska2026attentionidecisionlocalizing,
      title={Attention, May I Have Your Decision? Localizing Generative Choices in Diffusion Models}, 
      author={Katarzyna Zaleska and Łukasz Popek and Monika Wysoczańska and Kamil Deja},
      year={2026},
      eprint={2604.06052},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.06052}, 
}
```

## Credits

- LoRA finetuning scripts are adapted from the
  [diffusers text-to-image example](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/README.md).
- Steering generation / evaluation use
  [**DiffLens**](https://github.com/foundation-model-research/DiffLens).
