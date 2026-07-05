import torch
from aurum_genre.dataset import multihot, GenreChunkDataset, class_pos_weights

ROOTS = ["electronic", "rock", "pop"]


def test_chunks_per_track_scales_length_and_maps_rows(tmp_path):
    import torchaudio, pandas as pd
    sr = 16000
    wav = (0.2 * torch.sin(2 * torch.pi * 220 * torch.arange(sr * 5) / sr)).unsqueeze(0)
    fp = tmp_path / "t.wav"
    torchaudio.save(str(fp), wav, sr)
    man = tmp_path / "m.csv"
    pd.DataFrame({"filepath": [str(fp)], "root_labels": ["rock"]}).to_csv(man, index=False)
    ds = GenreChunkDataset(str(man), ROOTS, chunks_per_track=8)
    assert len(ds) == 8                       # one track → 8 virtual items
    m0, t0 = ds[0]; m7, _ = ds[7]             # both map to the single row, no IndexError
    assert m0.shape == m7.shape
    assert torch.equal(t0, torch.tensor([0.0, 1.0, 0.0]))


def test_corrupt_file_is_skipped_for_readable_neighbor(tmp_path):
    import torchaudio, pandas as pd
    sr = 16000
    good = tmp_path / "good.wav"
    torchaudio.save(str(good),
                    (0.2 * torch.sin(2 * torch.pi * 220 * torch.arange(sr * 5) / sr)).unsqueeze(0),
                    sr)
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"not really audio")           # unreadable
    man = tmp_path / "m.csv"
    # bad file first; dataset should fall through to the good neighbor.
    pd.DataFrame({"filepath": [str(bad), str(good)],
                  "root_labels": ["rock", "pop"]}).to_csv(man, index=False)
    ds = GenreChunkDataset(str(man), ROOTS)
    mel, target = ds[0]                              # index 0 is the corrupt file
    assert mel.shape[0] == 1 and mel.shape[1] == 128
    assert torch.equal(target, torch.tensor([0.0, 0.0, 1.0]))  # got the 'pop' neighbor


def test_class_pos_weights_rare_class_gets_higher_clamped_weight(tmp_path):
    import pandas as pd
    # 10 rock tracks, 1 pop track → pop is rare and should get a bigger pos_weight.
    rows = [{"filepath": f"x{i}.mp3", "root_labels": "rock"} for i in range(10)]
    rows.append({"filepath": "p.mp3", "root_labels": "pop"})
    man = tmp_path / "m.csv"
    pd.DataFrame(rows).to_csv(man, index=False)
    w = class_pos_weights(str(man), ROOTS, max_weight=10.0)
    assert w[2] > w[1]                        # pop weight > rock weight
    assert torch.all(w <= 10.0) and torch.all(w >= 1.0)

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

def test_cache_writes_once_and_reuses(tmp_path):
    import torchaudio
    import pandas as pd
    from aurum_genre.dataset import _cache_path
    from pathlib import Path
    sr = 16000
    wav = (0.2 * torch.sin(2 * torch.pi * 220 *
           torch.arange(sr * 5) / sr)).unsqueeze(0)
    fp = tmp_path / "track.wav"
    torchaudio.save(str(fp), wav, sr)
    man = tmp_path / "m.csv"
    pd.DataFrame({"filepath": [str(fp)], "root_labels": ["electronic"]}).to_csv(man, index=False)

    cache = tmp_path / "cache"
    ds = GenreChunkDataset(str(man), ROOTS, cache_dir=str(cache))
    cp = _cache_path(Path(cache), str(fp))
    assert not cp.exists()
    mel1, _ = ds[0]                       # first access decodes + writes cache
    assert cp.exists()
    # Corrupt the source; a second read must come from cache, proving reuse.
    fp.unlink()
    mel2, _ = ds[0]
    assert mel2.shape == mel1.shape       # served from cache despite missing source


def test_deterministic_dataset_returns_center_chunk_repeatably(tmp_path):
    """deterministic=True must always yield the same (center) chunk, so val
    metrics are comparable across epochs (model selection isn't chunk-lottery)."""
    import torchaudio, pandas as pd
    from aurum_genre.mel import log_mel, CHUNK_SAMPLES
    sr = 16000
    torch.manual_seed(0)
    wav = (torch.rand(1, sr * 5) * 2 - 1) * 0.5        # noise → chunks differ
    fp = tmp_path / "t.wav"
    torchaudio.save(str(fp), wav, sr)
    man = tmp_path / "m.csv"
    pd.DataFrame({"filepath": [str(fp)], "root_labels": ["rock"]}).to_csv(man, index=False)
    ds = GenreChunkDataset(str(man), ROOTS, deterministic=True)
    m1, _ = ds[0]
    m2, _ = ds[0]
    assert torch.equal(m1, m2)                          # repeatable
    decoded, _ = torchaudio.load(str(fp))               # 16-bit quantised copy
    start = (decoded.shape[1] - CHUNK_SAMPLES) // 2
    expected = log_mel(decoded[:, start:start + CHUNK_SAMPLES])
    assert torch.equal(m1, expected)                    # and it is the center chunk
