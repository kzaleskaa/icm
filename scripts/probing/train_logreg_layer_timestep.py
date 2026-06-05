#!/usr/bin/env python3
import argparse
import torch, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

def parse_args():
    p = argparse.ArgumentParser(description="Train logistic regression on train set and report train/test accuracy.")
    p.add_argument("--layer", type=str, default=None, help="Specific layer name to train on (if None, train on all layers)")
    p.add_argument("--timestep", type=int, required=True, help="Timestep to train on")
    p.add_argument("--train_dir", type=str, required=True, help="Directory containing training data packs")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for models and results")
    p.add_argument("--pos_label", type=str, default=None, help="Positive label name (default: alphabetically last)")
    p.add_argument("--use_scaler", action="store_true", help="Use StandardScaler for feature normalization")
    p.add_argument("--test_size", type=float, default=0.2, help="Test set size (0-1) when splitting train data (default: 0.2)")
    p.add_argument("--random_state", type=int, default=42, help="Random state for train/test split (default: 42)")
    p.add_argument("--no_test_split", action="store_true", help="Skip test split and use all data for training only")
    p.add_argument("--save_models", action="store_true", help="Save full trained models (default: only save theta vectors)")
    return p.parse_args()

def iter_labels(packs_dir):
    labs = []
    for pack_path in Path(packs_dir).glob("pack_*.pt"):
        data = torch.load(pack_path, map_location="cpu", weights_only=False).get("items", [])
        for it in data:
            if "label" in it:
                labs.append(it["label"])
    return labs

def build_label_map(train_dir, test_dir, pos_label=None):
    labels = set(iter_labels(train_dir)) | set(iter_labels(test_dir))

    if len(labels) != 2:
        raise ValueError(f"Expected exactly 2 labels, got {sorted(labels)}")

    a, b = sorted(labels)

    if pos_label is None:
        pos = b
        neg = a
    else:
        if pos_label not in labels:
            raise ValueError(f"--pos_label {pos_label} not found in {sorted(labels)}")
        pos = pos_label
        neg = next(x for x in labels if x != pos)

    return {neg: 0, pos: 1}, neg, pos


def get_available_layers(packs_dir, timestep):
    """
    Scan pack files to find all layers that have data for the given timestep.
    
    Args:
        packs_dir: Directory containing pack files
        timestep: Timestep to check for
        
    Returns:
        Set of layer names that have data at this timestep
    """
    layers = set()
    for pack_path in Path(packs_dir).glob("pack_*.pt"):
        data = torch.load(pack_path, map_location="cpu", weights_only=False).get("items", [])
        for it in data:
            acts = it.get("acts", {})
            for layer_name, steps_dict in acts.items():
                if timestep in steps_dict:
                    layers.add(layer_name)
        if layers:  # Found some layers, no need to scan all files
            break
    return sorted(layers)


def collect_xy(packs_dir, layer, step, lbl_map):
    X, y = [], []
    original_shape = None
    
    for pack_path in sorted(Path(packs_dir).glob("pack_*.pt")):
        data = torch.load(pack_path, map_location="cpu", weights_only=False).get("items", [])
        for it in data:
            acts = it.get("acts", {}).get(layer)
            if acts is None or step not in acts:
                continue
            
            # Get the activation tensor
            t = torch.as_tensor(acts[step])
            if t.dtype == torch.bfloat16:
                t = t.to(torch.float32)
            
            # Store original shape for reshaping later
            if original_shape is None:
                original_shape = t.shape
            
            # Vectors are already mean-reduced to 1D [hidden_dim], use as-is
            # (flatten on 1D vector does nothing anyway)
            X.append(t.detach().cpu().numpy())

            lab = it.get("label")
            
            if lab in lbl_map:
                y.append(lbl_map[lab])
            else:
                # Skip samples with labels not in the label map
                X.pop()  # Remove the last added X since we're skipping this sample
    
    if not X:
        return None, None, None
    
    return np.stack(X), np.array(y, dtype=np.int64), original_shape


def train_single_layer(layer, step, Xtr, ytr, Xte, yte, lbl_map, neg_label, pos_label, 
                       original_shape, args, models_dir):
    """
    Train logistic regression for a single layer and timestep.
    
    Returns:
        Dictionary with results and theta vector(s) for this layer
    """
    print(f"\n{'='*60}")
    print(f"Training layer: {layer}")
    print(f"Training data: {len(Xtr)} samples")
    print(f"Testing data: {0 if Xte is None else len(Xte)} samples")
    print(f"Original shape: {original_shape}")

    # Build classifier pipeline
    if args.use_scaler:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, solver="lbfgs")
        )
    else:
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    
    # Train
    clf.fit(Xtr, ytr)
    acc_tr = clf.score(Xtr, ytr)
    acc_te = float("nan") if Xte is None else clf.score(Xte, yte)

    # Extract and normalize coefficient vector (theta)
    if args.use_scaler:
        logreg = clf.named_steps['logisticregression']
    else:
        logreg = clf
    
    theta = np.asarray(logreg.coef_, dtype=np.float32).squeeze()
    
    # Normalize theta
    theta_norm = np.linalg.norm(theta)
    if theta_norm > 0:
        theta_normalized = theta / theta_norm
    else:
        theta_normalized = theta
        print("Warning: theta has zero norm!")
    
    # Store theta (already 1D from mean-reduced vectors)
    theta_dict = {"theta_1d": theta_normalized}
    
    # Save model (optional)
    if args.save_models:
        model_path = models_dir / f"{layer.replace('.', '_')}__t{step:03d}.joblib"
        joblib.dump({
            "model": clf, 
            "label_map": lbl_map, 
            "layer": layer, 
            "timestep": step,
            "neg_label": neg_label,
            "pos_label": pos_label,
            "original_shape": original_shape
        }, model_path)
        print(f"Saved model → {model_path}")
    
    # Print results
    print(f"Train accuracy: {acc_tr:.4f}")
    if not np.isnan(acc_te):
        print(f"Test accuracy: {acc_te:.4f}")
    print(f"Theta norm: {theta_norm:.6f}")
    print(f"Features: {len(theta)}")
    
    return {
        "results": {
            "layer": layer,
            "timestep": step,
            "n_train": len(Xtr),
            "n_test": 0 if Xte is None else len(Xte),
            "shape_x": Xtr.shape,
            "original_shape": str(original_shape),
            "accuracy_train": acc_tr,
            "accuracy_test": acc_te,
            "neg_label": neg_label,
            "pos_label": pos_label,
            "use_scaler": args.use_scaler,
            "theta_norm": float(theta_norm),
            "n_features": len(theta)
        },
        "theta": theta_dict
    }


def main():
    args = parse_args()
    train_dir = Path(args.train_dir)
    out_dir = Path(args.out_dir)
    models_dir = out_dir / "models"
    vectors_dir = out_dir / "vectors"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_models:
        models_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir.mkdir(parents=True, exist_ok=True)

    step = args.timestep
    
    # Determine which layers to train on
    if args.layer is not None:
        # Single layer specified
        layers_to_train = [args.layer]
        print(f"Training on specified layer: {args.layer}")
    else:
        # Find all available layers for this timestep
        print(f"Finding all layers with data at timestep {step}...")
        layers_to_train = get_available_layers(train_dir, step)
        if not layers_to_train:
            print(f"No layers found with data at timestep {step}")
            return
        print(f"Found {len(layers_to_train)} layers to train on")
    
    # Build label map (always use train_dir for both since we split internally)
    lbl_map, neg_label, pos_label = build_label_map(train_dir, train_dir, args.pos_label)
    
    print(f"\nLabel mapping: {neg_label}->0, {pos_label}->1")
    print(f"Using StandardScaler: {args.use_scaler}")
    print(f"Saving full models: {args.save_models}")
    
    # Train on each layer
    all_results = []
    all_vectors = {}  # Dictionary to store all theta vectors
    all_shapes = {}   # Dictionary to store original shapes
    successful = 0
    failed = 0
    
    for layer in layers_to_train:
        try:
            # Collect data for this layer
            X_all, y_all, original_shape = collect_xy(train_dir, layer, step, lbl_map)
            
            if X_all is None:
                print(f"\nSkipping {layer}: No data at timestep {step}")
                failed += 1
                continue
            
            # Split into train and test with stratification (or use all data for training)
            if args.no_test_split:
                # Use all data for training only
                Xtr, ytr = X_all, y_all
                Xte, yte = None, None
            else:
                # Split into train and test
                Xtr, Xte, ytr, yte = train_test_split(
                    X_all, y_all, 
                    test_size=args.test_size, 
                    random_state=args.random_state,
                    stratify=y_all
                )
            
            # Train on this layer
            output = train_single_layer(
                layer, step, Xtr, ytr, Xte, yte, 
                lbl_map, neg_label, pos_label,
                original_shape,
                args, models_dir
            )
            all_results.append(output["results"])
            all_vectors[layer] = output["theta"]
            all_shapes[layer] = original_shape
            successful += 1
            
        except Exception as e:
            print(f"\nError training on layer {layer}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue
    
    # Save all vectors in a single file
    if all_vectors:
        # Prepare vectors for saving
        vectors_to_save = {}
        for layer_name, theta_dict in all_vectors.items():
            # Save 1D version
            vectors_to_save[f"{layer_name}__1d"] = theta_dict["theta_1d"]
            # Save reshaped version if available
            if "theta_reshaped" in theta_dict:
                vectors_to_save[f"{layer_name}__reshaped"] = theta_dict["theta_reshaped"]
        
        vectors_path = vectors_dir / f"timestep_{step:03d}_vectors.npz"
        np.savez(vectors_path, **vectors_to_save)
        print(f"\n{'='*60}")
        print(f"Saved all theta vectors → {vectors_path}")
        print(f"  Contains {len(all_vectors)} layer vectors")
        
        # Also save metadata about the vectors
        metadata = {
            "timestep": step,
            "layers": list(all_vectors.keys()),
            "shapes": all_shapes,
            "neg_label": neg_label,
            "pos_label": pos_label,
            "use_scaler": args.use_scaler
        }
        metadata_path = vectors_dir / f"timestep_{step:03d}_metadata.joblib"
        joblib.dump(metadata, metadata_path)
        print(f"Saved metadata → {metadata_path}")
    
    # Save combined results
    if all_results:
        df = pd.DataFrame(all_results)
        
        # If training a single layer, include layer name in CSV filename to avoid conflicts
        if args.layer is not None:
            safe_layer = args.layer.replace(".", "_")
            csv_path = out_dir / f"results_{safe_layer}__t{step:03d}.csv"
        else:
            # Training all layers, use timestep-only filename
            csv_path = out_dir / f"results_timestep_{step:03d}.csv"
        
        df.to_csv(csv_path, index=False)
        print(f"Saved combined results → {csv_path}")
        print(f"Successfully trained: {successful} layers")
        print(f"Failed: {failed} layers")
        print(f"Total layers: {len(layers_to_train)}")
    else:
        print("\nNo successful training runs.")

if __name__ == "__main__":
    main()
