import hashlib
import json
from aurum_genre.provenance import (sha256_file, manifest_track_hash,
                                     build_run_manifest, write_run_manifest)

def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "x.bin"; p.write_bytes(b"aurum")
    assert sha256_file(p) == hashlib.sha256(b"aurum").hexdigest()

def test_manifest_track_hash_is_order_independent(tmp_path):
    import pandas as pd
    a = tmp_path / "a.csv"; b = tmp_path / "b.csv"
    pd.DataFrame({"filepath": ["z.mp3", "a.mp3"], "root_labels": ["x", "y"]}).to_csv(a, index=False)
    pd.DataFrame({"filepath": ["a.mp3", "z.mp3"], "root_labels": ["y", "x"]}).to_csv(b, index=False)
    assert manifest_track_hash(a) == manifest_track_hash(b)

def test_build_and_write_manifest_has_required_keys(tmp_path):
    m = build_run_manifest(repo_dir=tmp_path, seed=1337, hyperparameters={"epochs": 1},
                           dataset={"train_rows": 10}, device="cpu",
                           metrics={"macro_auc": 0.8}, mel_recipe={"SR": 16000},
                           timestamps={"start": "t0", "end": "t1"})
    for k in ("schema", "git", "seed", "hyperparameters", "dataset",
              "environment", "metrics", "mel_recipe", "timestamps"):
        assert k in m
    out = tmp_path / "run_manifest.json"; write_run_manifest(out, m)
    assert json.loads(out.read_text())["seed"] == 1337
