import torch
from aurum_genre.dataset import multihot, GenreChunkDataset

ROOTS = ["electronic", "rock", "pop"]

def test_multihot_sets_correct_indices():
    v = multihot(["electronic", "pop"], ROOTS)
    assert torch.equal(v, torch.tensor([1.0, 0.0, 1.0]))
    assert torch.equal(multihot([], ROOTS), torch.zeros(3))

def test_dataset_yields_mel_and_target(tmp_path):
    # a 1-second 16k sine as a fake track
    import torchaudio
    sr = 16000
    wav = (0.2 * torch.sin(2 * torch.pi * 220 *
           torch.arange(sr * 5) / sr)).unsqueeze(0)
    fp = tmp_path / "track.wav"
    torchaudio.save(str(fp), wav, sr)
    import pandas as pd
    man = tmp_path / "m.csv"
    pd.DataFrame({"filepath": [str(fp)], "root_labels": ["electronic|pop"]}).to_csv(man, index=False)

    ds = GenreChunkDataset(str(man), ROOTS)
    mel, target = ds[0]
    assert mel.shape[0] == 1 and mel.shape[1] == 128
    assert torch.equal(target, torch.tensor([1.0, 0.0, 1.0]))
