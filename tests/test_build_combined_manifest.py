"""Tests for the FMA+Jamendo combined manifest builder."""
from __future__ import annotations
import pandas as pd
from scripts.build_combined_manifest import build_combined, split_jamendo


def _csv(tmp_path, name, n, label):
    fp = tmp_path / name
    pd.DataFrame({"filepath": [f"{name}/{i}.mp3" for i in range(n)],
                  "root_labels": [label] * n}).to_csv(fp, index=False)
    return fp


def test_split_is_deterministic_and_partitions(tmp_path):
    jam = pd.read_csv(_csv(tmp_path, "jam", 200, "electronic"))
    tr1, va1 = split_jamendo(jam, seed=1337, val_frac=0.2)
    tr2, va2 = split_jamendo(jam, seed=1337, val_frac=0.2)
    assert len(tr1) + len(va1) == 200               # partition, no loss/overlap
    assert set(tr1["filepath"]).isdisjoint(set(va1["filepath"]))
    assert list(va1["filepath"]) == list(va2["filepath"])   # deterministic
    assert 0.1 < len(va1) / 200 < 0.3               # roughly val_frac


def test_build_combined_merges_fma_and_jamendo(tmp_path):
    fma_tr = _csv(tmp_path, "fma_tr", 10, "rock")
    fma_va = _csv(tmp_path, "fma_va", 4, "rock")
    jam = _csv(tmp_path, "jam", 100, "electronic|electronic:techno")
    ntr, nva = build_combined(fma_tr, fma_va, jam,
                              tmp_path / "ctr.csv", tmp_path / "cva.csv",
                              seed=1337, val_frac=0.2)
    assert ntr + nva == 114                          # 10+4 FMA + 100 Jamendo
    ctr = pd.read_csv(tmp_path / "ctr.csv")
    assert (ctr["root_labels"] == "rock").sum() == 10          # all FMA train present
    assert (ctr["root_labels"] == "electronic|electronic:techno").sum() > 0
