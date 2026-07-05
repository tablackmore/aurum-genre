import numpy as np
from aurum_genre.eval import macro_auc, calibrate_thresholds

def test_macro_auc_perfect_separation_is_one():
    y_true = np.array([[1,0],[0,1],[1,0],[0,1]])
    y_score = np.array([[0.9,0.1],[0.1,0.9],[0.8,0.2],[0.2,0.8]])
    assert macro_auc(y_true, y_score) == 1.0

def test_calibrate_picks_separating_threshold():
    y_true = np.array([[1],[1],[0],[0]])
    y_score = np.array([[0.8],[0.7],[0.2],[0.1]])
    th = calibrate_thresholds(y_true, y_score, ["x"])
    assert 0.2 < th["x"] <= 0.7

def test_chunk_averaged_metrics_shape(tmp_path):
    import torch, torchaudio, pandas as pd
    from aurum_genre.train import fit
    from aurum_genre.eval import chunk_averaged_metrics
    sr = 16000
    rows = []
    for i in range(40):
        f = 220 if i % 2 == 0 else 440
        wav = (0.2 * torch.sin(2 * torch.pi * f * torch.arange(sr * 4) / sr)).unsqueeze(0)
        fp = tmp_path / f"t{i}.wav"; torchaudio.save(str(fp), wav, sr)
        rows.append({"filepath": str(fp), "root_labels": "electronic" if i % 2 == 0 else "rock"})
    man = tmp_path / "m.csv"; pd.DataFrame(rows).to_csv(man, index=False)
    ckpt = tmp_path / "g.pt"
    fit(str(man), epochs=1, out_ckpt=str(ckpt), device="cpu", num_workers=0)
    m = chunk_averaged_metrics(str(ckpt), str(man), device="cpu")
    assert 0.0 <= m["macro_auc"] <= 1.0
    assert "electronic" in m["per_class"] and "support" in m["per_class"]["electronic"]
    assert m["skipped"] == 0                  # every track decoded


def _tiny_ckpt_and_manifest(tmp_path, n_tracks=12, secs=8):
    """Train a 1-epoch model on n_tracks sine tracks; returns (ckpt, manifest)."""
    import torch, torchaudio, pandas as pd
    from aurum_genre.train import fit
    sr = 16000
    rows = []
    for i in range(n_tracks):
        f = 220 if i % 2 == 0 else 440
        wav = (0.2 * torch.sin(2 * torch.pi * f * torch.arange(sr * secs) / sr)).unsqueeze(0)
        fp = tmp_path / f"t{i}.wav"; torchaudio.save(str(fp), wav, sr)
        rows.append({"filepath": str(fp), "root_labels": "electronic" if i % 2 == 0 else "rock"})
    man = tmp_path / "m.csv"; pd.DataFrame(rows).to_csv(man, index=False)
    ckpt = tmp_path / "g.pt"
    fit(str(man), epochs=1, out_ckpt=str(ckpt), device="cpu", num_workers=0)
    return ckpt, man


def test_track_scores_reports_skipped_tracks(tmp_path):
    import pandas as pd
    from aurum_genre.eval import track_scores
    ckpt, man = _tiny_ckpt_and_manifest(tmp_path, n_tracks=8, secs=4)
    bad = tmp_path / "bad.mp3"; bad.write_bytes(b"not audio")
    df = pd.read_csv(man)
    df.loc[len(df)] = [str(bad), "rock"]
    df.to_csv(man, index=False)
    y, s, roots, skipped = track_scores(str(ckpt), str(man), device="cpu")
    assert skipped == 1                       # corrupt track counted, not silent
    assert y.shape == (8, len(roots)) and s.shape == y.shape


def test_evaluate_calibrates_on_chunk_averaged_scores(tmp_path):
    """Thresholds must be tuned on the same chunk-averaged scores that
    production inference (infer.tag_track) applies them to."""
    import json
    from aurum_genre.eval import (evaluate, track_scores, calibrate_thresholds,
                                  macro_auc)
    ckpt, man = _tiny_ckpt_and_manifest(tmp_path)
    out = tmp_path / "thresholds.json"
    m = evaluate(str(ckpt), str(man), str(out), device="cpu")
    y, s, roots, skipped = track_scores(str(ckpt), str(man), device="cpu")
    assert m["thresholds"] == calibrate_thresholds(y, s, roots)
    assert m["macro_auc"] == macro_auc(y, s)  # headline metric = chunk-averaged
    assert m["skipped"] == skipped == 0
    assert json.loads(out.read_text()) == m["thresholds"]
