"""Dataset of mel chunks with multi-hot root labels, read from a manifest CSV."""
from __future__ import annotations
import hashlib
import os
import random
from pathlib import Path
import numpy as np
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

def _decode_mono_16k(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)            # [C, N]
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav

def _cache_path(cache_dir: Path, src: str) -> Path:
    # Key on the absolute source path so identical files share one cache entry.
    key = hashlib.sha1(os.path.abspath(src).encode()).hexdigest()
    return cache_dir / f"{key}.npy"

def _load_mono_16k(path: str, cache_dir: Path | None = None) -> torch.Tensor:
    """Decode to mono 16 kHz. With cache_dir, decode once and reuse a float16 .npy."""
    if cache_dir is None:
        return _decode_mono_16k(path)
    cp = _cache_path(cache_dir, path)
    if cp.exists():
        arr = np.load(cp)                                   # [N] float16
        return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)
    wav = _decode_mono_16k(path)                            # [1, N] float32
    # Atomic write: unique temp name per process, then rename (safe across workers).
    tmp = cp.with_suffix(f".{os.getpid()}.tmp.npy")
    np.save(tmp, wav.squeeze(0).numpy().astype(np.float16))
    os.replace(tmp, cp)
    return wav

def _random_chunk(wav: torch.Tensor) -> torch.Tensor:
    n = wav.shape[1]
    if n < CHUNK_SAMPLES:
        pad = CHUNK_SAMPLES - n
        return torch.nn.functional.pad(wav, (0, pad))
    start = random.randint(0, n - CHUNK_SAMPLES)
    return wav[:, start:start + CHUNK_SAMPLES]

def class_pos_weights(manifest_csv: str, roots: list[str],
                      max_weight: float = 10.0) -> torch.Tensor:
    """Per-class BCE pos_weight = (#neg / #pos), clamped so ultra-rare classes
    (e.g. country=4) don't destabilise training. Counteracts label imbalance."""
    df = pd.read_csv(manifest_csv)
    idx = {r: i for i, r in enumerate(roots)}
    counts = torch.zeros(len(roots))
    for raw in df["root_labels"]:
        if pd.isna(raw):
            continue
        for lab in str(raw).split("|"):
            if lab in idx:
                counts[idx[lab]] += 1
    n = float(len(df))
    pos = counts.clamp(min=1.0)
    neg = (n - counts).clamp(min=0.0)
    return (neg / pos).clamp(min=1.0, max=max_weight)


class GenreChunkDataset(Dataset):
    def __init__(self, manifest_csv: str, roots: list[str], cache_dir: str | None = None,
                 chunks_per_track: int = 1, augment: bool = False):
        self.df = pd.read_csv(manifest_csv)
        self.roots = roots
        self.chunks_per_track = max(1, int(chunks_per_track))
        # Decoded-audio cache: explicit arg wins, else AURUM_CACHE_DIR, else disabled.
        cache_dir = cache_dir or os.environ.get("AURUM_CACHE_DIR") or None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        # SpecAugment (train-time only): mask mel freq/time bands for regularisation.
        self._spec_aug = None
        if augment:
            self._spec_aug = torch.nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=16),
                torchaudio.transforms.TimeMasking(time_mask_param=24),
            )

    def __len__(self) -> int:
        # Draw chunks_per_track random chunks per track per epoch (each access picks
        # a fresh random chunk), multiplying signal from a small track set (~8 chunks
        # fit in a 30 s track, matching how infer.py averages over all chunks).
        return len(self.df) * self.chunks_per_track

    def __getitem__(self, i: int):
        n = len(self.df)
        base = i % n
        # FMA-large ships a handful of corrupt mp3s (e.g. 148786–148795). Skip to
        # the next readable track so one bad file can't kill a DataLoader worker;
        # the (mel, label) pair always comes from a single consistent row.
        last_err = None
        for off in range(min(16, n)):
            row = self.df.iloc[(base + off) % n]
            try:
                wav = _load_mono_16k(row["filepath"], self.cache_dir)
            except Exception as e:  # decode errors vary by backend/file
                last_err = e
                continue
            chunk = _random_chunk(wav)
            mel = log_mel(chunk)
            if self._spec_aug is not None:
                mel = self._spec_aug(mel)
            raw = row["root_labels"]
            labels = [] if pd.isna(raw) else str(raw).split("|")
            return mel, multihot(labels, self.roots)
        raise RuntimeError(f"no readable audio within 16 rows of index {i}: {last_err}")
