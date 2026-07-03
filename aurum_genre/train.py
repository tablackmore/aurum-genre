"""Multi-label training (BCEWithLogits). fit() writes {state_dict, roots}."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .dataset import GenreChunkDataset, class_pos_weights
from .model import ShortChunkCNN
from .seed import seed_everything, worker_init_fn
from .taxonomy import load_taxonomy, output_labels

# train.py lives at aurum_genre/train.py → parent.parent is the repo root
_DEFAULT_TAXONOMY = Path(__file__).resolve().parent.parent / "taxonomy.json"

def default_device() -> str:
    """Pick the best available accelerator: CUDA, then Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def train_one_epoch(model, loader, optim, device, crit=None, mixup_alpha: float = 0.0) -> float:
    model.train()
    crit = crit or nn.BCEWithLogitsLoss()
    total, n = 0.0, 0
    for mel, target in loader:
        mel, target = mel.to(device), target.to(device)
        if mixup_alpha > 0:
            # Mixup: convex-combine a batch with a shuffled copy; soft multi-label
            # targets are valid for BCE and regularise a small, imbalanced set.
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            perm = torch.randperm(mel.size(0), device=device)
            mel = lam * mel + (1 - lam) * mel[perm]
            target = lam * target + (1 - lam) * target[perm]
        optim.zero_grad()
        loss = crit(model(mel), target)
        loss.backward()
        optim.step()
        total += float(loss); n += 1
    return total / max(n, 1)

def _val_macro_auc(model, val_manifest, roots, device, cache_dir=None) -> float:
    from .eval import macro_auc
    ds = GenreChunkDataset(val_manifest, roots, cache_dir=cache_dir)
    loader = DataLoader(ds, batch_size=64)
    model.eval()
    ys, ss = [], []
    with torch.no_grad():
        for mel, target in loader:
            ss.append(torch.sigmoid(model(mel.to(device))).cpu().numpy())
            ys.append(target.numpy())
    model.train()
    return macro_auc(np.concatenate(ys), np.concatenate(ss))

def fit(manifest: str, epochs: int, out_ckpt: str,
        batch_size: int = 32, lr: float = 1e-3, device: str | None = None,
        taxonomy_path: str | Path | None = None, val_manifest: str | None = None,
        cache_dir: str | None = None, chunks_per_track: int = 4, augment: bool = True,
        mixup_alpha: float = 0.2, weight_decay: float = 1e-4,
        use_pos_weight: bool = True, patience: int = 8,
        seed: int = 1337, num_workers: int = 4) -> None:
    device = device or default_device()
    seed_everything(seed)
    # "roots" holds the full output vocabulary (roots + namespaced subgenres).
    roots = output_labels(load_taxonomy(taxonomy_path or _DEFAULT_TAXONOMY))
    ds = GenreChunkDataset(manifest, roots, cache_dir=cache_dir,
                           chunks_per_track=chunks_per_track, augment=augment)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, drop_last=True,
                        worker_init_fn=worker_init_fn)
    model = ShortChunkCNN(num_classes=len(roots)).to(device)
    crit = nn.BCEWithLogitsLoss(
        pos_weight=class_pos_weights(manifest, roots).to(device)
    ) if use_pos_weight else nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(epochs, 1))
    best_auc, best_state, bad = -1.0, None, 0
    for e in range(epochs):
        loss = train_one_epoch(model, loader, optim, device, crit, mixup_alpha)
        sched.step()
        msg = f"epoch {e+1}/{epochs} loss={loss:.4f}"
        if val_manifest:
            auc = _val_macro_auc(model, val_manifest, roots, device, cache_dir)
            msg += f" val_auc={auc:.4f}"
            if auc > best_auc:
                best_auc, bad = auc, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        print(msg)
        if val_manifest and bad >= patience:
            print(f"early stop @ epoch {e+1} (best val_auc={best_auc:.4f})")
            break
    if best_state is not None:  # restore best-on-validation weights
        model.load_state_dict(best_state)
    config = {"seed": seed, "epochs": epochs, "batch_size": batch_size, "lr": lr,
              "weight_decay": weight_decay, "chunks_per_track": chunks_per_track,
              "augment": augment, "mixup_alpha": mixup_alpha,
              "use_pos_weight": use_pos_weight, "patience": patience,
              "best_val_auc": (best_auc if best_state is not None else None)}
    torch.save({"state_dict": model.state_dict(), "roots": roots, "config": config}, out_ckpt)
    tag = f" (best val_auc={best_auc:.4f})" if best_state is not None else ""
    print(f"wrote {out_ckpt}{tag}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--out", default="genre.pt")
    a = ap.parse_args()
    fit(a.manifest, a.epochs, a.out)
