"""Held-out metrics (macro ROC-AUC) + per-class F1-maximising thresholds."""
from __future__ import annotations
import json
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    aucs = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[:, c], y_score[:, c]))
    return float(np.mean(aucs)) if aucs else 0.0

def calibrate_thresholds(y_true, y_score, roots) -> dict[str, float]:
    out = {}
    grid = np.linspace(0.05, 0.95, 19)
    for c, name in enumerate(roots):
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            f1 = f1_score(y_true[:, c], (y_score[:, c] >= t).astype(int),
                          zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        out[name] = best_t
    return out

def evaluate(ckpt: str, manifest: str, out_thresholds: str) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from .dataset import GenreChunkDataset
    from .model import ShortChunkCNN
    from .train import default_device
    blob = torch.load(ckpt, map_location="cpu", weights_only=True)  # safe unpickle
    roots = blob["roots"]
    device = default_device()
    model = ShortChunkCNN(num_classes=len(roots))
    model.load_state_dict(blob["state_dict"]); model.eval(); model.to(device)
    ds = GenreChunkDataset(manifest, roots)
    loader = DataLoader(ds, batch_size=32)
    ys, ss = [], []
    with torch.no_grad():
        for mel, target in loader:
            probs = torch.sigmoid(model(mel.to(device))).cpu()
            ss.append(probs.numpy()); ys.append(target.numpy())
    y_true, y_score = np.concatenate(ys), np.concatenate(ss)
    thresholds = calibrate_thresholds(y_true, y_score, roots)
    with open(out_thresholds, "w") as f:
        json.dump(thresholds, f, indent=2)
    return {"macro_auc": macro_auc(y_true, y_score), "thresholds": thresholds}

def chunk_averaged_metrics(ckpt: str, manifest: str, device: str | None = None) -> dict:
    """Track-level metrics matching infer.py: average sigmoid over all chunks."""
    import torch
    import pandas as pd
    from .model import ShortChunkCNN
    from .dataset import _load_mono_16k, multihot
    from .mel import log_mel, CHUNK_SAMPLES
    from .train import default_device
    dev = device or default_device()
    blob = torch.load(ckpt, map_location="cpu", weights_only=True)
    roots = blob["roots"]
    model = ShortChunkCNN(num_classes=len(roots))
    model.load_state_dict(blob["state_dict"]); model.eval(); model.to(dev)
    df = pd.read_csv(manifest)
    ys, ss = [], []
    with torch.no_grad():
        for _, row in df.iterrows():
            try:
                wav = _load_mono_16k(row["filepath"], None)
            except Exception:
                continue
            n = wav.shape[-1]
            if n < CHUNK_SAMPLES:
                chunks = [torch.nn.functional.pad(wav, (0, CHUNK_SAMPLES - n))]
            else:
                chunks = [wav[:, s:s + CHUNK_SAMPLES]
                          for s in range(0, n - CHUNK_SAMPLES + 1, CHUNK_SAMPLES)]
            mels = torch.stack([log_mel(c) for c in chunks]).to(dev)
            probs = torch.sigmoid(model(mels)).mean(0).cpu().numpy()
            ss.append(probs)
            raw = row["root_labels"]
            labs = [] if pd.isna(raw) else str(raw).split("|")
            ys.append(multihot(labs, roots).numpy())
    y, s = np.array(ys), np.array(ss)
    per_class, aucs = {}, []
    for c, name in enumerate(roots):
        sup = int(y[:, c].sum())
        if len(np.unique(y[:, c])) < 2:
            per_class[name] = {"auc": None, "support": sup}
        else:
            a = float(roc_auc_score(y[:, c], s[:, c])); aucs.append(a)
            per_class[name] = {"auc": a, "support": sup}
    return {"macro_auc": float(np.mean(aucs)) if aucs else 0.0, "per_class": per_class}
