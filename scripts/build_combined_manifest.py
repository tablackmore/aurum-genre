"""Combine FMA and Jamendo CC-BY manifests into train/val for genre-v2.

The Jamendo manifest has no split, so we split it deterministically (seeded,
per-track hash) and merge with the FMA train/val manifests. Output rows are
(filepath, root_labels) — identical schema, ready for training.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import pandas as pd


def _val_bucket(filepath: str, seed: int, val_frac: float) -> bool:
    """Deterministic per-track val assignment (stable across runs)."""
    h = hashlib.sha1(f"{seed}:{filepath}".encode()).hexdigest()
    return (int(h[:8], 16) % 1000) < int(val_frac * 1000)


def split_jamendo(jamendo: pd.DataFrame, seed: int, val_frac: float):
    is_val = jamendo["filepath"].apply(lambda fp: _val_bucket(fp, seed, val_frac))
    return jamendo[~is_val].copy(), jamendo[is_val].copy()


def build_combined(fma_train_csv, fma_val_csv, jamendo_csv,
                   out_train, out_val, seed: int = 1337, val_frac: float = 0.12):
    """Write combined train/val manifests; returns (n_train, n_val)."""
    cols = ["filepath", "root_labels"]
    fma_tr = pd.read_csv(fma_train_csv)[cols]
    fma_va = pd.read_csv(fma_val_csv)[cols]
    jam = pd.read_csv(jamendo_csv)[cols]
    jam_tr, jam_va = split_jamendo(jam, seed, val_frac)
    train = pd.concat([fma_tr, jam_tr], ignore_index=True)
    val = pd.concat([fma_va, jam_va], ignore_index=True)
    Path(out_train).parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_train, index=False)
    val.to_csv(out_val, index=False)
    print(f"wrote {out_train} ({len(train)} = {len(fma_tr)} FMA + {len(jam_tr)} Jamendo)")
    print(f"wrote {out_val} ({len(val)} = {len(fma_va)} FMA + {len(jam_va)} Jamendo)")
    return len(train), len(val)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fma-train", default="data/train.csv")
    ap.add_argument("--fma-val", default="data/val.csv")
    ap.add_argument("--jamendo", default="data/train_jamendo.csv")
    ap.add_argument("--out-train", default="data/train_combined.csv")
    ap.add_argument("--out-val", default="data/val_combined.csv")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val-frac", type=float, default=0.12)
    a = ap.parse_args()
    build_combined(a.fma_train, a.fma_val, a.jamendo, a.out_train, a.out_val,
                   a.seed, a.val_frac)


if __name__ == "__main__":
    main()
