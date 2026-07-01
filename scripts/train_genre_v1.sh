#!/usr/bin/env bash
# One-command local training: download FMA (if missing) → run the full pipeline.
#
# Usage: scripts/train_genre_v1.sh <small|medium> [extra run_pipeline args...]
#   e.g. scripts/train_genre_v1.sh small --epochs 30
#        scripts/train_genre_v1.sh medium --epochs 60 --min-auc 0.80
#
# Requires the venv active (source .venv/bin/activate) and ~8 GiB (small) /
# ~25 GiB (medium) free disk. On a Mac this trains on CPU/MPS — slow but works;
# for a GPU use notebooks/train_genre_colab.ipynb instead.
set -euo pipefail

SUBSET="${1:-medium}"; shift || true

if [ ! -d "data/fma_${SUBSET}" ] || [ ! -d "data/fma_metadata" ]; then
  echo "FMA ${SUBSET} not found — downloading ..."
  scripts/download_fma.sh "$SUBSET"
fi

python scripts/run_pipeline.py --subset "$SUBSET" "$@"
