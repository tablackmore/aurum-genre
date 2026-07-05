# Reproducibility

Every `release/` ships `run_manifest.json` — the source of truth for how that
`genre.onnx` was produced: git commit + dirty flag, seed, hyperparameters,
dataset hashes (manifest SHA256 + track-list hash), environment (python/torch/
torchaudio/OS/device), and metrics (chunk-averaged macro + per-class AUC —
the same chunk-averaged scoring production inference uses, which is also what
the shipped thresholds are calibrated on).

## Rebuild from scratch

    python3.12 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.lock          # exact pinned versions
    pip install -e ".[dev]"
    bash scripts/download_fma.sh <subset> data # SHA1-verified audio
    export AURUM_CACHE_DIR=data/cache
    python scripts/run_pipeline.py --subset <subset> --epochs 60 --seed 1337

Verify against a prior `run_manifest.json`:
- `dataset.train_track_list_sha256` must match → identical training tracks.
- `metrics.macro_auc_chunk_avg` should match within noise (Tier 1: near-, not
  bit-deterministic; MPS is nondeterministic, so weights may differ).
- `mel_golden.npz` gives byte-level feature/inference parity for the exported model.

## What is and isn't guaranteed

- **Guaranteed:** exact data lineage (which tracks, which licenses), exact code
  (git SHA), pinned env, and metric reproducibility within noise.
- **Not guaranteed:** bit-identical weights (deliberate Tier-1 choice; would
  require deterministic algorithms and fixed non-MPS hardware).
