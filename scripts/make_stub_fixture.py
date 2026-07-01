"""Generate a STUB genre-v1 asset + Rust golden fixtures for the app integration.

The weights are random (untrained) — this exists ONLY to give the Rust
`analysis::genre` integration a real, contract-matching artifact to build and
test against before the real model is trained. It produces:

  out/genre.onnx          - opset-17, mel input, 11-root output (random weights)
  out/taxonomy.json       - copy of the repo taxonomy
  out/thresholds.json     - all 0.5 (stub)
  out/mel_recipe.txt      - the mel param recipe
  out/genre_mel_golden.f32 - flat f32: [waveform(59049)][mel(128*T)]
  out/genre_mel_golden.json - shapes + roots + a few onnx logits, for the Rust test

Run: python scripts/make_stub_fixture.py
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from aurum_genre.model import ShortChunkCNN
from aurum_genre.taxonomy import load_taxonomy, root_labels
from aurum_genre.export import export_onnx
from aurum_genre.mel import log_mel, CHUNK_SAMPLES

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

tax = load_taxonomy(ROOT / "taxonomy.json")
roots = root_labels(tax)

# 1. Random (untrained) checkpoint with the real root count.
torch.manual_seed(0)
model = ShortChunkCNN(num_classes=len(roots)).eval()
ckpt = OUT / "genre_stub.pt"
torch.save({"state_dict": model.state_dict(), "roots": roots}, ckpt)

# 2. Export ONNX + golden npz + recipe via the real exporter.
export_onnx(str(ckpt), str(OUT / "genre.onnx"),
            str(OUT / "mel_golden.npz"), str(OUT / "mel_recipe.txt"))

# 3. Stub taxonomy + thresholds (0.5 each) alongside the model.
shutil.copy(ROOT / "taxonomy.json", OUT / "taxonomy.json")
(OUT / "thresholds.json").write_text(
    json.dumps({r: 0.5 for r in roots}, indent=2))

# 4. Rust golden fixture: a fixed waveform + its 16 kHz log-mel, as flat f32 LE,
#    plus a JSON sidecar with shapes/roots so the Rust mel-parity test can pin
#    its implementation against torchaudio exactly.
torch.manual_seed(4242)
wav = (torch.rand(1, CHUNK_SAMPLES) * 2 - 1).float()
mel = log_mel(wav)                      # [1, 128, T]
mel_np = mel.squeeze(0).numpy().astype("<f4")   # [128, T]
wav_np = wav.squeeze(0).numpy().astype("<f4")   # [59049]

with open(OUT / "genre_mel_golden.f32", "wb") as f:
    f.write(wav_np.tobytes())
    f.write(mel_np.tobytes())

sidecar = {
    "waveform_len": int(wav_np.shape[0]),
    "n_mels": int(mel_np.shape[0]),
    "n_frames": int(mel_np.shape[1]),
    "roots": roots,
    "layout": "f32 LE: waveform[waveform_len] then mel[n_mels*n_frames] row-major",
    "note": "STUB (random weights). Regenerate with make_stub_fixture.py.",
}
(OUT / "genre_mel_golden.json").write_text(json.dumps(sidecar, indent=2))

print("stub fixture written to", OUT)
for p in sorted(OUT.iterdir()):
    print(f"  {p.name}: {p.stat().st_size} bytes")
