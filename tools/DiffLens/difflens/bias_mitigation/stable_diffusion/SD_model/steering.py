import os
import re
import glob
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class ApplySteering(nn.Module):
    def __init__(self, vectors, alpha):
        super().__init__()
        self.step_idx = 0
        self.vectors = vectors
        self.alpha = alpha

    @torch.no_grad()
    def __call__(self, module, args, kwargs, output):
        if self.step_idx < 15:
            self.step_idx = (self.step_idx + 1) % len(self.vectors)
            return output
        # extract prompt output
        out_neg, out_pos = output.chunk(2, dim=0)
        # save norm value
        # norm = out_pos.norm(dim=-1, keepdim=True)
        # extract steering vector
        steering_vector = torch.from_numpy(self.vectors[self.step_idx]).to(out_pos.device, dtype=out_pos.dtype)
        steering_vector = steering_vector.reshape(1, 1, -1)
        # apply steering vector
        temp_tensor = out_pos + self.alpha * steering_vector
        # re-normalize to original norm
        # temp_tensor = temp_tensor / torch.norm(temp_tensor, dim=2, keepdim=True)
        # temp_tensor = temp_tensor * norm
        # combine back
        out_combined = torch.cat([out_neg, temp_tensor], dim=0).to(output.device)
        # update step index
        self.step_idx = (self.step_idx + 1) % len(self.vectors)
        return out_combined

class ApplySteeringSteps(nn.Module):
    def __init__(self, vectors, alpha):
        super().__init__()
        self.step_idx = 0
        self.vectors = vectors
        self.alpha = alpha

    @torch.no_grad()
    def __call__(self, module, args, kwargs, output):
        # if self.step_idx < 5:
        #     self.step_idx = (self.step_idx + 1) % len(self.vectors)
        #     return output
        # extract prompt output
        if self.step_idx < 15:
            self.step_idx = (self.step_idx + 1) % len(self.vectors)
            return output
        out_neg, out_pos = output.chunk(2, dim=0)
        # save norm value
        # norm = out_pos.norm(dim=-1, keepdim=True)
        # extract steering vector
        steering_vector = torch.from_numpy(self.vectors[self.step_idx]).to(out_pos.device, dtype=out_pos.dtype)
        steering_vector = steering_vector.reshape(1, 1, -1)
        # apply steering vector
        temp_tensor = out_pos + self.alpha * steering_vector
        # re-normalize to original norm
        # temp_tensor = temp_tensor / torch.norm(temp_tensor, dim=2, keepdim=True)
        # temp_tensor = temp_tensor * norm
        # combine back
        out_combined = torch.cat([out_neg, temp_tensor], dim=0).to(output.device)
        # update step index
        self.step_idx = (self.step_idx + 1) % len(self.vectors)
        return out_combined

class ApplySteeringStepsNorm(nn.Module):
    def __init__(self, vectors, alpha):
        super().__init__()
        self.step_idx = 0
        self.vectors = vectors
        self.alpha = alpha

    @torch.no_grad()
    def __call__(self, module, args, kwargs, output):
        if self.step_idx < 5:
            self.step_idx = (self.step_idx + 1) % len(self.vectors)
            return output
        # extract prompt output
        # if self.step_idx < 20:
            # self.step_idx = (self.step_idx + 1) % len(self.vectors)
            # return output
        out_neg, out_pos = output.chunk(2, dim=0)
        # save norm value
        norm = out_pos.norm(dim=-1, keepdim=True)
        # extract steering vector
        steering_vector = torch.from_numpy(self.vectors[self.step_idx]).to(out_pos.device, dtype=out_pos.dtype)
        steering_vector = steering_vector.reshape(1, 1, -1)
        # apply steering vector
        temp_tensor = out_pos + self.alpha * steering_vector
        # re-normalize to original norm
        temp_tensor = temp_tensor / torch.norm(temp_tensor, dim=2, keepdim=True)
        temp_tensor = temp_tensor * norm
        # combine back
        out_combined = torch.cat([out_neg, temp_tensor], dim=0).to(output.device)
        # update step index
        self.step_idx = (self.step_idx + 1) % len(self.vectors)
        return out_combined


def load_logreg_vectors(models_dir, pos_label=1):
    def coef_unit_pos(model, pos_label=1):
        clf = None
        if isinstance(model, Pipeline):
            for _, step in model.steps:
                if isinstance(step, LogisticRegression):
                    clf = step
                    break
        elif isinstance(model, LogisticRegression):
            clf = model
        else:
            clf = model
        theta = clf.coef_.ravel().astype(np.float32)
        return theta / (np.linalg.norm(theta) + 1e-12)

    pat = re.compile(r"^(?P<layer>.+)__t(?P<t>\d+)\.joblib$")
    files = glob.glob(os.path.join(models_dir, "*.joblib"))
    vectors = {}
    for f in files:
        fname = os.path.basename(f)
        m = pat.match(fname)
        if not m:
            continue
        layer = m.group("layer")
        t = int(m.group("t"))
        loaded = joblib.load(f)
        model = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
        v = coef_unit_pos(model, pos_label).astype(np.float32)
        vectors.setdefault(layer, {})[t] = v
    return vectors

def _coef_unit_pos_original_space(model, pos_label=1):
    scaler, clf = None, None
    if isinstance(model, Pipeline):
        for _, step in model.steps:
            if isinstance(step, StandardScaler):
                scaler = step
            if isinstance(step, LogisticRegression):
                clf = step
    elif isinstance(model, LogisticRegression):
        clf = model
    else:
        clf = model
    theta = np.asarray(clf.coef_, dtype=np.float32)
    if theta.ndim == 2 and theta.shape[0] > 1 and hasattr(clf, "classes_"):
        classes = clf.classes_
        idx = int(np.where(classes == pos_label)[0][0]) if pos_label in classes else 0
        theta = theta[idx]
    else:
        theta = theta.ravel()
        if hasattr(clf, "classes_") and len(clf.classes_) == 2:
            classes = clf.classes_
            idx = int(np.where(classes == pos_label)[0][0]) if pos_label in classes else 1
            if idx == 0:
                theta = -theta
    if scaler is not None and hasattr(scaler, "scale_"):
        print("Transforming to original space using scaler.")
        scale = scaler.scale_.astype(np.float32)
        scale = np.where(scale == 0, 1.0, scale)
        v = theta / scale
    else:
        v = theta
    v /= (np.linalg.norm(v) + 1e-12)
    return v.astype(np.float32)

def load_logreg_vectors_original_space(models_dir, pos_label=1):
    pat = re.compile(r"^(?P<layer>.+)__t(?P<t>\d+)\.joblib$")
    files = glob.glob(os.path.join(models_dir, "*.joblib"))
    vectors = {}
    for f in files:
        m = pat.match(os.path.basename(f))
        if not m:
            continue
        layer = m.group("layer")
        t = int(m.group("t"))
        loaded = joblib.load(f)
        model = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
        v = _coef_unit_pos_original_space(model, pos_label)
        vectors.setdefault(layer, {})[t] = v
    return vectors