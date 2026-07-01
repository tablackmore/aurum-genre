#!/usr/bin/env bash
set -euo pipefail
python scripts/build_manifest.py --fma-meta data/fma_metadata \
  --fma-audio data/fma_medium --out data/manifest.csv --notice release/NOTICE
python -m aurum_genre.train --manifest data/manifest.csv --epochs 50 --out genre.pt
