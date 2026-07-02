"""One-off: reconstruct run_manifest.json for the shipped genre-v1 from artifacts
we still have (data/genre.pt config + data/train.csv/val.csv + training log).
Flagged code_state=uncommitted-reconstructed; superseded by the next clean run."""
from __future__ import annotations
import datetime as dt
from pathlib import Path
import torch
from aurum_genre import provenance, eval as geval, train
from aurum_genre.mel import SR, N_FFT, HOP, N_MELS, CHUNK_SAMPLES

ROOT = Path(__file__).resolve().parent.parent
ckpt = ROOT / "data/genre.pt"
train_csv, val_csv = ROOT / "data/train.csv", ROOT / "data/val.csv"
cfg = torch.load(str(ckpt), map_location="cpu", weights_only=True).get("config", {})
chunked = geval.chunk_averaged_metrics(str(ckpt), str(val_csv))
git = provenance.git_info(ROOT)
git["code_state"] = "uncommitted-reconstructed"  # honesty: trained before SP1
m = provenance.build_run_manifest(
    repo_dir=ROOT, seed=cfg.get("seed"), hyperparameters={**cfg, "subset": "large"},
    dataset={"source": {"kind": "fma", "subset": "large"},
             "train_rows": sum(1 for _ in open(train_csv)) - 1,
             "val_rows": sum(1 for _ in open(val_csv)) - 1,
             "train_manifest_sha256": provenance.sha256_file(train_csv),
             "val_manifest_sha256": provenance.sha256_file(val_csv),
             "train_track_list_sha256": provenance.manifest_track_hash(train_csv)},
    device="mps", metrics={"macro_auc_chunk_avg": chunked["macro_auc"],
                           "per_class": chunked["per_class"],
                           "note": "single-chunk pipeline verdict was 0.7627"},
    mel_recipe={"SR": SR, "N_FFT": N_FFT, "HOP": HOP, "N_MELS": N_MELS,
                "CHUNK_SAMPLES": CHUNK_SAMPLES},
    timestamps={"start": None, "end": None, "note": "reconstructed 2026-07-02"})
m["git"] = git
provenance.write_run_manifest(ROOT / "release/run_manifest.json", m)
print("wrote release/run_manifest.json (reconstructed) — macro_avg",
      round(chunked["macro_auc"], 4))
