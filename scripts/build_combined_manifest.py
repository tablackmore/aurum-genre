"""Combine FMA and Jamendo CC-BY manifests into train/val for genre-v2.

The Jamendo manifest has no split, so we split it deterministically (seeded,
per-track hash) and merge with the FMA train/val manifests. Output rows are
(filepath, root_labels) — identical schema, ready for training.
"""
from __future__ import annotations
import argparse
import hashlib
import random
from collections import defaultdict
from pathlib import Path
import pandas as pd


def stratified_split(df: pd.DataFrame, seed: int = 1337,
                     val_frac: float = 0.12, min_val: int = 4):
    """Label-aware train/val split so even rare subgenres get val examples.

    Assign rarest labels first: each label gets ~val_frac of its tracks in val,
    but at least min_val when it has enough tracks. Tracks are multi-label, so a
    val track counts for all its labels (overlap keeps the val set small).
    NOTE: this is a track-level split — for very small subgenres drawn from few
    artists it can leak an artist across train/val, mildly inflating their AUC.
    """
    rows = df.to_dict("records")
    random.Random(seed).shuffle(rows)
    label_rows: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        for lab in str(r["root_labels"]).split("|"):
            label_rows[lab].append(i)
    val: set[int] = set()
    for lab in sorted(label_rows, key=lambda l: len(label_rows[l])):
        cnt = len(label_rows[lab])
        want = max(min_val if cnt >= 2 * min_val else max(1, cnt // 3),
                   round(val_frac * cnt))
        have = sum(1 for i in label_rows[lab] if i in val)
        for i in label_rows[lab]:
            if have >= want:
                break
            if i not in val:
                val.add(i); have += 1
    target = int(val_frac * len(rows))
    for i in range(len(rows)):
        if len(val) >= target:
            break
        val.add(i)
    train = pd.DataFrame([rows[i] for i in range(len(rows)) if i not in val])
    valr = pd.DataFrame([rows[i] for i in sorted(val)])
    return train, valr


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


def build_combined_stratified(fma_train_csv, fma_val_csv, jamendo_csv,
                              out_train, out_val, seed: int = 1337,
                              val_frac: float = 0.12, min_val: int = 4):
    """Pool FMA (train+val) + Jamendo and re-split with stratified_split so rare
    subgenres get validation examples. Returns (n_train, n_val)."""
    cols = ["filepath", "root_labels"]
    pool = pd.concat([pd.read_csv(fma_train_csv)[cols], pd.read_csv(fma_val_csv)[cols],
                      pd.read_csv(jamendo_csv)[cols]], ignore_index=True)
    train, val = stratified_split(pool, seed, val_frac, min_val)
    Path(out_train).parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_train, index=False)
    val.to_csv(out_val, index=False)
    print(f"wrote {out_train} ({len(train)}) + {out_val} ({len(val)}) [stratified, pool={len(pool)}]")
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
    ap.add_argument("--stratified", action="store_true",
                    help="pool all data and split label-aware (rare subgenres get val examples)")
    a = ap.parse_args()
    if a.stratified:
        build_combined_stratified(a.fma_train, a.fma_val, a.jamendo,
                                  a.out_train, a.out_val, a.seed, a.val_frac)
    else:
        build_combined(a.fma_train, a.fma_val, a.jamendo, a.out_train, a.out_val,
                       a.seed, a.val_frac)


if __name__ == "__main__":
    main()
