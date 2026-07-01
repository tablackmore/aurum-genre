"""Multi-label training (BCEWithLogits). fit() writes {state_dict, roots}."""
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .dataset import GenreChunkDataset
from .model import ShortChunkCNN
from .taxonomy import load_taxonomy, root_labels

# train.py lives at aurum_genre/train.py → parent.parent is the repo root
_DEFAULT_TAXONOMY = Path(__file__).resolve().parent.parent / "taxonomy.json"

def train_one_epoch(model, loader, optim, device) -> float:
    model.train()
    crit = nn.BCEWithLogitsLoss()
    total, n = 0.0, 0
    for mel, target in loader:
        mel, target = mel.to(device), target.to(device)
        optim.zero_grad()
        loss = crit(model(mel), target)
        loss.backward()
        optim.step()
        total += float(loss); n += 1
    return total / max(n, 1)

def fit(manifest: str, epochs: int, out_ckpt: str,
        batch_size: int = 32, lr: float = 1e-3, device: str | None = None,
        taxonomy_path: str | Path | None = None) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    roots = root_labels(load_taxonomy(taxonomy_path or _DEFAULT_TAXONOMY))
    ds = GenreChunkDataset(manifest, roots)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    model = ShortChunkCNN(num_classes=len(roots)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for e in range(epochs):
        loss = train_one_epoch(model, loader, optim, device)
        print(f"epoch {e+1}/{epochs} loss={loss:.4f}")
    torch.save({"state_dict": model.state_dict(), "roots": roots}, out_ckpt)
    print(f"wrote {out_ckpt}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--out", default="genre.pt")
    a = ap.parse_args()
    fit(a.manifest, a.epochs, a.out)
