"""Held-out metrics (macro ROC-AUC) + per-class F1-maximising thresholds.

All track-level scoring is chunk-averaged via track_scores(), the same
behaviour as production inference (infer.tag_track) — thresholds calibrated
here are applied to chunk-averaged probabilities in the app, so they must be
tuned on chunk-averaged probabilities too.
"""
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

def track_scores(ckpt: str, manifest: str, device: str | None = None):
    """Chunk-averaged sigmoid scores per track, matching infer.py exactly:
    non-overlapping CHUNK_SAMPLES windows (tail dropped), mean over chunks.
    Returns (y_true, y_score, roots, skipped) — skipped counts undecodable
    tracks so a mass decode failure can't silently inflate metrics."""
    import torch
    import pandas as pd
    from .model import ShortChunkCNN
    from .dataset import _load_mono_16k, multihot
    from .mel import log_mel, CHUNK_SAMPLES
    from .train import default_device
    dev = device or default_device()
    blob = torch.load(ckpt, map_location="cpu", weights_only=True)  # safe unpickle
    roots = blob["roots"]
    model = ShortChunkCNN(num_classes=len(roots))
    model.load_state_dict(blob["state_dict"]); model.eval(); model.to(dev)
    df = pd.read_csv(manifest)
    ys, ss, skipped = [], [], 0
    with torch.no_grad():
        for _, row in df.iterrows():
            try:
                wav = _load_mono_16k(row["filepath"], None)
            except Exception:
                skipped += 1
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
    return np.array(ys), np.array(ss), roots, skipped

def _per_class(y: np.ndarray, s: np.ndarray, roots) -> dict:
    per_class = {}
    for c, name in enumerate(roots):
        sup = int(y[:, c].sum())
        if len(np.unique(y[:, c])) < 2:
            per_class[name] = {"auc": None, "support": sup}
        else:
            per_class[name] = {"auc": float(roc_auc_score(y[:, c], s[:, c])),
                               "support": sup}
    return per_class

def evaluate(ckpt: str, manifest: str, out_thresholds: str,
             device: str | None = None) -> dict:
    """Chunk-averaged validation metrics + thresholds calibrated on those same
    chunk-averaged scores (the distribution production inference thresholds)."""
    y_true, y_score, roots, skipped = track_scores(ckpt, manifest, device)
    thresholds = calibrate_thresholds(y_true, y_score, roots)
    with open(out_thresholds, "w") as f:
        json.dump(thresholds, f, indent=2)
    return {"macro_auc": macro_auc(y_true, y_score),
            "per_class": _per_class(y_true, y_score, roots),
            "thresholds": thresholds, "skipped": skipped}

def chunk_averaged_metrics(ckpt: str, manifest: str, device: str | None = None) -> dict:
    """Track-level metrics matching infer.py: average sigmoid over all chunks."""
    y, s, roots, skipped = track_scores(ckpt, manifest, device)
    return {"macro_auc": macro_auc(y, s), "per_class": _per_class(y, s, roots),
            "skipped": skipped}
