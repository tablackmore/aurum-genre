"""Tests for the FMA+Jamendo combined manifest builder."""
from __future__ import annotations
import pandas as pd
from scripts.build_combined_manifest import build_combined, split_jamendo, stratified_split


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


def test_stratified_split_gives_rare_labels_val_examples():
    import pandas as pd
    # 200 common 'rock' tracks + 20 rare 'rock|rock:garage' tracks.
    rows = [{"filepath": f"c{i}.mp3", "root_labels": "rock"} for i in range(200)]
    rows += [{"filepath": f"g{i}.mp3", "root_labels": "rock|rock:garage"} for i in range(20)]
    df = pd.DataFrame(rows)
    train, val = stratified_split(df, seed=1337, val_frac=0.12, min_val=4)
    assert len(train) + len(val) == 220                        # partition
    val_garage = val["root_labels"].str.contains("rock:garage").sum()
    assert val_garage >= 4                                      # rare label got val examples
    # deterministic
    t2, v2 = stratified_split(df, seed=1337, val_frac=0.12, min_val=4)
    assert list(v2["filepath"]) == list(val["filepath"])


def test_stratified_split_never_strips_a_label_from_train():
    """Every label must keep >=1 training example: a singleton label sent
    entirely to val can never be learned (and its val AUC is undefined)."""
    rows = [{"filepath": f"c{i}.mp3", "root_labels": "rock"} for i in range(200)]
    rows.append({"filepath": "solo.mp3", "root_labels": "jazz"})       # 1 track
    rows += [{"filepath": f"b{i}.mp3", "root_labels": "blues"} for i in range(2)]
    df = pd.DataFrame(rows)
    for seed in (1, 7, 1337):
        train, val = stratified_split(df, seed=seed, val_frac=0.12, min_val=4)
        assert len(train) + len(val) == len(rows)                      # partition
        train_labels = set()
        for labs in train["root_labels"]:
            train_labels.update(str(labs).split("|"))
        assert {"rock", "jazz", "blues"} <= train_labels               # none orphaned
