"""Log-mel spectrogram — the single source of truth the Rust port must mirror.
Keep these params fixed; any change is a new model version + new golden vectors."""
from __future__ import annotations
import torch
import torchaudio

SR = 16000
N_FFT = 512
HOP = 256
N_MELS = 128
CHUNK_SECS = 3.69
CHUNK_SAMPLES = 59049  # ~3.69 s @ 16 kHz (matches Short-chunk CNN receptive field)

_MEL = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
    power=2.0, center=True, norm=None, mel_scale="htk",
)


def log_mel(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    mel = _MEL(waveform)                       # [1, N_MELS, frames], power
    return torch.log10(torch.clamp(mel, min=1e-7))


def mel_recipe() -> str:
    return (
        "AURUM genre mel recipe (Rust must match exactly):\n"
        "sample_rate=16000\nn_fft=512\nhop=256\nn_mels=128\n"
        "power=2.0 center=True norm=None mel_scale=htk\n"
        "log=log10(clamp(mel, min=1e-7))\n"
        "chunk_samples=59049\n"
    )
