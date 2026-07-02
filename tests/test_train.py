import torch
from aurum_genre.model import ShortChunkCNN
from aurum_genre.train import train_one_epoch

def test_loss_decreases_on_repeated_tiny_batch():
    torch.manual_seed(0)
    mel = torch.randn(6, 1, 128, 188)
    target = torch.zeros(6, 3); target[:, 0] = 1.0
    loader = [(mel, target)] * 8
    model = ShortChunkCNN(num_classes=3)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = train_one_epoch(model, loader, optim, "cpu")
    last = None
    for _ in range(5):
        last = train_one_epoch(model, loader, optim, "cpu")
    assert last < first

def test_train_one_epoch_with_mixup_runs_and_is_finite():
    torch.manual_seed(0)
    mel = torch.randn(6, 1, 128, 188)
    target = torch.zeros(6, 3); target[:, 0] = 1.0
    loader = [(mel, target)] * 4
    model = ShortChunkCNN(num_classes=3)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = train_one_epoch(model, loader, optim, "cpu", mixup_alpha=0.2)
    assert loss == loss and loss >= 0.0        # not NaN, valid loss

def test_fit_records_config_and_seed(tmp_path):
    import torch, torchaudio, pandas as pd
    from aurum_genre.train import fit
    sr = 16000
    rows = []
    for i in range(40):
        f = 220 if i % 2 == 0 else 440
        wav = (0.2 * torch.sin(2 * torch.pi * f * torch.arange(sr * 4) / sr)).unsqueeze(0)
        fp = tmp_path / f"t{i}.wav"; torchaudio.save(str(fp), wav, sr)
        rows.append({"filepath": str(fp), "root_labels": "electronic" if i % 2 == 0 else "rock"})
    man = tmp_path / "m.csv"; pd.DataFrame(rows).to_csv(man, index=False)
    ckpt = tmp_path / "g.pt"
    fit(str(man), epochs=1, out_ckpt=str(ckpt), device="cpu", num_workers=0, seed=99)
    blob = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    cfg = blob["config"]
    assert cfg["seed"] == 99 and cfg["epochs"] == 1
    assert set(cfg) >= {"seed", "epochs", "batch_size", "lr", "weight_decay",
                        "chunks_per_track", "augment", "mixup_alpha",
                        "use_pos_weight", "patience", "best_val_auc"}
