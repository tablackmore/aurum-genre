"""Dataset of mel chunks with multi-hot root labels, read from a manifest CSV."""
from __future__ import annotations
import random
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from .mel import log_mel, SR, CHUNK_SAMPLES

def multihot(labels: list[str], roots: list[str]) -> torch.Tensor:
    idx = {r: i for i, r in enumerate(roots)}
    v = torch.zeros(len(roots))
    for label in labels:
        if label in idx:
            v[idx[label]] = 1.0
    return v

def _load_mono_16k(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)            # [C, N]
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav

def _random_chunk(wav: torch.Tensor) -> torch.Tensor:
    n = wav.shape[1]
    if n < CHUNK_SAMPLES:
        pad = CHUNK_SAMPLES - n
        return torch.nn.functional.pad(wav, (0, pad))
    start = random.randint(0, n - CHUNK_SAMPLES)
    return wav[:, start:start + CHUNK_SAMPLES]

class GenreChunkDataset(Dataset):
    def __init__(self, manifest_csv: str, roots: list[str]):
        self.df = pd.read_csv(manifest_csv)
        self.roots = roots

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        wav = _load_mono_16k(row["filepath"])
        chunk = _random_chunk(wav)
        mel = log_mel(chunk)
        raw = row["root_labels"]
        labels = [] if pd.isna(raw) else str(raw).split("|")
        return mel, multihot(labels, self.roots)
