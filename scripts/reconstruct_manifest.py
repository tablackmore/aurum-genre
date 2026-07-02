"""One-off: reconstruct run_manifest.json for the shipped genre-v1 from artifacts
we still have (data/genre.pt + data/train.csv/val.csv + data/train_run.log).
Flagged code_state=uncommitted-reconstructed; superseded by the next clean run.

Provenance note: genre-v1 predates the checkpoint-config feature (SP1), so
data/genre.pt carries no embedded config. Hyperparameters are reconstructed from
data/train_run.log and the baseline commit 22905ff that trained the model.
The git.commit is pinned to that baseline, NOT live HEAD."""
from __future__ import annotations
from pathlib import Path
from aurum_genre import provenance, eval as geval
from aurum_genre.mel import SR, N_FFT, HOP, N_MELS, CHUNK_SAMPLES

# Baseline commit whose working tree trained genre-v1 (before SP1 / seeding).
GENRE_V1_COMMIT = "22905ff89ee1dbb91b71442ffa21d4b12820a82c"

ROOT = Path(__file__).resolve().parent.parent
ckpt = ROOT / "data/genre.pt"
train_csv, val_csv = ROOT / "data/train.csv", ROOT / "data/val.csv"

# Metrics are computed live from the actual weights — do not hardcode.
chunked = geval.chunk_averaged_metrics(str(ckpt), str(val_csv))

# Hyperparameters reconstructed from data/train_run.log and baseline fit() defaults.
# genre-v1 was trained: run_pipeline.py --subset large --epochs 60, early-stopped
# at epoch 18, best_val_auc=0.8274.  No seed was used (predates seeding feature).
hyperparameters = {
    "subset": "large",
    "epochs_requested": 60,
    "early_stopped_epoch": 18,
    "best_val_auc": 0.8274,
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "chunks_per_track": 4,
    "augment": True,
    "mixup_alpha": 0.2,
    "use_pos_weight": True,
    "patience": 8,
    "note": (
        "Reconstructed from data/train_run.log and baseline commit "
        f"{GENRE_V1_COMMIT[:7]}; genre-v1 predates checkpoint-config feature."
    ),
}

# Determine current branch for the git block (keeping branch honest).
git = provenance.git_info(ROOT)
current_branch = git.get("branch", "feat/reproducibility-provenance")

m = provenance.build_run_manifest(
    repo_dir=ROOT,
    seed=None,  # genre-v1 was unseeded (predates seeding feature)
    hyperparameters=hyperparameters,
    dataset={"source": {"kind": "fma", "subset": "large"},
             "train_rows": sum(1 for _ in open(train_csv)) - 1,
             "val_rows": sum(1 for _ in open(val_csv)) - 1,
             "train_manifest_sha256": provenance.sha256_file(train_csv),
             "val_manifest_sha256": provenance.sha256_file(val_csv),
             "train_track_list_sha256": provenance.manifest_track_hash(train_csv)},
    device="mps",
    metrics={"macro_auc_chunk_avg": chunked["macro_auc"],
             "per_class": chunked["per_class"],
             "note": "single-chunk pipeline verdict was 0.7627"},
    mel_recipe={"SR": SR, "N_FFT": N_FFT, "HOP": HOP, "N_MELS": N_MELS,
                "CHUNK_SAMPLES": CHUNK_SAMPLES},
    timestamps={"start": None, "end": None, "note": "reconstructed 2026-07-02"})

# Override git block: pin commit to the baseline that actually trained genre-v1.
# Live HEAD includes SP1 features (seeding, config-recording) that genre-v1 never
# used, so reporting HEAD would be misleading.
m["git"] = {
    "commit": GENRE_V1_COMMIT,
    "branch": current_branch,
    "dirty": True,
    "code_state": "uncommitted-reconstructed",
}
m["unseeded"] = True  # top-level flag: genre-v1 predates seeding feature

provenance.write_run_manifest(ROOT / "release/run_manifest.json", m)
print("wrote release/run_manifest.json (reconstructed) — macro_avg",
      round(chunked["macro_auc"], 4))
