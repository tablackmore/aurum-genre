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
