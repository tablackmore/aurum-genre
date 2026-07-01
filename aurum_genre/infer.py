"""Reference track-level inference: chunk the track, run onnx per chunk, average
sigmoid probs, threshold. This defines the behaviour the Rust port must mirror."""
from __future__ import annotations
import json
import numpy as np
import onnxruntime as ort
from .mel import CHUNK_SAMPLES


def average_chunk_probs(probs: np.ndarray) -> np.ndarray:
    return probs.mean(axis=0)


def _chunks(wav: np.ndarray, hop: int = CHUNK_SAMPLES):
    n = wav.shape[-1]
    if n < CHUNK_SAMPLES:
        yield np.pad(wav, ((0, 0), (0, CHUNK_SAMPLES - n)))
        return
    for start in range(0, n - CHUNK_SAMPLES + 1, hop):
        yield wav[:, start:start + CHUNK_SAMPLES]


def tag_track(onnx_path, thresholds_path, roots, wav_16k_mono, log_mel_fn):
    thresholds = json.loads(open(thresholds_path).read())
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    probs = []
    for ch in _chunks(wav_16k_mono):
        mel = log_mel_fn(np.ascontiguousarray(ch))[None]   # [1,1,128,T]
        logits = sess.run(None, {"mel": mel.astype(np.float32)})[0][0]
        probs.append(1.0 / (1.0 + np.exp(-logits)))
    avg = average_chunk_probs(np.stack(probs))
    return [(r, float(avg[i])) for i, r in enumerate(roots)
            if avg[i] >= thresholds.get(r, 0.5)]
