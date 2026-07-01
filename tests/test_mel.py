import torch
from aurum_genre.mel import log_mel, mel_recipe, SR, N_MELS, CHUNK_SAMPLES


def test_log_mel_shape_and_finiteness():
    wav = torch.zeros(1, CHUNK_SAMPLES)
    wav[0, ::100] = 0.5  # sparse clicks
    mel = log_mel(wav)
    assert mel.shape[0] == 1 and mel.shape[1] == N_MELS
    assert torch.isfinite(mel).all()


def test_log_mel_is_deterministic():
    torch.manual_seed(0)
    wav = torch.rand(1, CHUNK_SAMPLES) * 2 - 1
    a = log_mel(wav)
    b = log_mel(wav)
    assert torch.equal(a, b)


def test_recipe_lists_all_params():
    r = mel_recipe()
    for key in ["sample_rate=16000", "n_fft=512", "hop=256", "n_mels=128"]:
        assert key in r
