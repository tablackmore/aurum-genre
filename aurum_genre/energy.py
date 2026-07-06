"""Perceptual energy of a track as a 0..1 number.

Energy is a physical/acoustic property, not a subjective tag, so we COMPUTE it
directly rather than train a model — no ground-truth energy labels exist in the
permissive data anyway. Heuristic blend of loudness (RMS), brightness
(zero-crossing rate), and activity (frame-to-frame RMS variation / dynamics).
Deterministic and explainable; works on any 16 kHz mono waveform.
"""
from __future__ import annotations
import numpy as np


def energy(wav_16k_mono, sr: int = 16000) -> float:
    x = np.asarray(wav_16k_mono, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return 0.0
    # loudness: RMS, mapped so ~0.3 RMS reads as full loudness
    rms = float(np.sqrt(np.mean(x ** 2)))
    loud = min(1.0, rms / 0.3)
    # brightness: zero-crossing rate (broadband/percussive → high; pure tone → low)
    zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))
    bright = min(1.0, zcr / 0.3)
    # activity: variation of short-frame RMS (dynamics / onsets)
    frame, hop = 1024, 512
    fr_rms = np.array([np.sqrt(np.mean(x[i:i + frame] ** 2) + 1e-9)
                       for i in range(0, max(1, x.size - frame), hop)])
    flux = float(np.mean(np.abs(np.diff(fr_rms)))) if fr_rms.size > 1 else 0.0
    act = min(1.0, flux / 0.05)
    e = 0.5 * loud + 0.3 * bright + 0.2 * act
    return float(max(0.0, min(1.0, e)))
