import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Any

import torch
import clip
import pandas as pd
from PIL import Image
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, PREPROCESS = clip.load("ViT-L/14", device=DEVICE)
MODEL.eval()

GROUP_DEFS = {
    "gender": ["image of woman", "image of man"],
    "age": ["image of young person", "image of adult person", "image of old person"],
    "race": ["image of asian person", "image of black person", "image of white person", "image of indian person"],
}

def extract_layer_name(path: str) -> str:
    return Path(path).stem.replace("layer_", "")

def _to_pil(img: Any) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, str):
        return Image.open(img).convert("RGB")
    raise ValueError("unsupported image type")

def _norm_text(t: str) -> str:
    return " ".join(t.strip().lower().split())

@torch.no_grad()
def _encode_texts(prompts: List[str]) -> torch.Tensor:
    tokens = clip.tokenize(prompts, truncate=True).to(DEVICE)
    feats = MODEL.encode_text(tokens)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats

@torch.no_grad()
def _encode_images(images: List[Image.Image]) -> torch.Tensor:
    imgs = [PREPROCESS(im).unsqueeze(0) for im in images]
    batch = torch.cat(imgs, dim=0).to(DEVICE)
    feats = MODEL.encode_image(batch)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats

@torch.no_grad()
def _classify_counts(images: List[Image.Image], txt_feats: torch.Tensor, ref_texts: List[str]) -> Dict[str, int]:
    if len(images) == 0 or txt_feats.numel() == 0:
        return {f"count::{ref}": 0 for ref in ref_texts}
    img_feats = _encode_images(images)
    logits = img_feats @ txt_feats.t()
    preds = logits.argmax(dim=1)
    counts = torch.bincount(preds, minlength=txt_feats.size(0)).tolist()
    return {f"count::{ref}": int(c) for ref, c in zip(ref_texts, counts)}

def _iter_records_nested(d: Dict) -> List[Tuple[str, str, str, List[Image.Image]]]:
    recs = []
    for _, layer_data in d.items():
        for general_prompt, payload in layer_data.items():
            if isinstance(payload, dict) and "images" in payload:
                for specific_prompt, images in payload["images"].items():
                    recs.append((general_prompt, "images", specific_prompt, images))
            elif isinstance(payload, dict):
                for layer_name, specific in payload.items():
                    specific_dict = specific["images"] if isinstance(specific, dict) and "images" in specific else specific
                    if isinstance(specific_dict, dict):
                        for specific_prompt, images in specific_dict.items():
                            recs.append((general_prompt, layer_name, specific_prompt, images))
    return recs

def _iter_records_flat(d: Dict) -> List[Tuple[str, str, str, List[Image.Image]]]:
    recs = []
    inner = d.get(0, {})
    for general_prompt, images in inner.items():
        recs.append((general_prompt, "None", "", images))
    return recs

def _load_records(file_path: Path) -> List[Tuple[str, str, str, List[Image.Image]]]:
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    try:
        recs = _iter_records_nested(data)
        if recs:
            return recs
    except Exception:
        pass
    return _iter_records_flat(data)

def process_file(file_path: Path, ref_texts: List[str]) -> List[Dict]:
    layer_file_name = extract_layer_name(str(file_path))
    records = _load_records(file_path)
    txt_cache: Dict[str, torch.Tensor] = {}
    refs_feats = _encode_texts(ref_texts) if ref_texts else torch.empty(0)
    out = []
    for general_prompt, layer_name, specific_prompt, images in tqdm(records, desc=f"Processing {file_path.name}", leave=False):
        pil_images = []
        for im in images:
            try:
                pil_images.append(_to_pil(im))
            except Exception:
                continue
        if not pil_images:
            continue
        gp_key = _norm_text(general_prompt)
        if gp_key not in txt_cache:
            txt_cache[gp_key] = _encode_texts([general_prompt])
        if specific_prompt:
            sp_key = _norm_text(specific_prompt)
            if sp_key not in txt_cache:
                txt_cache[sp_key] = _encode_texts([specific_prompt])
        row = {
            "file_layer": layer_file_name,
            "layer_name": layer_name,
            "general_prompt": general_prompt,
            "specific_prompt": specific_prompt,
            "num_images": len(pil_images),
        }
        for ref in ref_texts:
            row[f"count::{ref}"] = 0
        if ref_texts:
            counts = _classify_counts(pil_images, refs_feats, ref_texts)
            row.update(counts)
        out.append(row)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--groups", type=str, default="gender,age,race")
    args = parser.parse_args()
    selected = [g.strip() for g in args.groups.split(",") if g.strip() in GROUP_DEFS]
    refs: List[str] = []
    for g in selected:
        refs.extend(GROUP_DEFS[g])
    base = Path(args.base_dir)
    rows: List[Dict] = []
    pkl_files = list(base.rglob("*.pkl"))
    for pkl in tqdm(pkl_files, desc="Evaluating .pkl files"):
        rows.extend(process_file(pkl, refs))
    df = pd.DataFrame(rows)
    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".json":
        df.to_json(out, orient="records", indent=2)
    elif out.suffix == ".csv":
        df.to_csv(out, index=False)
    else:
        df.to_csv(str(out) + ".csv", index=False)

if __name__ == "__main__":
    main()
