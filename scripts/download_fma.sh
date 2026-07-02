#!/usr/bin/env bash
# Download + verify + unzip the FMA dataset (metadata + one audio subset).
#
# Usage: scripts/download_fma.sh <small|medium> [dest_dir]
#   small  = 8 genres, 8k tracks, 7.2 GiB  (fast pipeline validation; ~6 of our roots)
#   medium = 16 genres, 25k tracks, 22 GiB (full taxonomy; the real genre-v1)
#
# Idempotent: skips downloads/unzips that already exist. Resumable (curl -C -).
set -euo pipefail

SUBSET="${1:-}"
DEST="${2:-data}"
BASE="https://os.unil.cloud.switch.ch/fma"

case "$SUBSET" in
  small)  AUDIO_SHA1="ade154f733639d52e35e32f5593efe5be76c6d70" ;;
  medium) AUDIO_SHA1="c67b69ea232021025fca9231fc1c7c1a063ab50b" ;;
  *) echo "usage: $0 <small|medium> [dest_dir]" >&2; exit 2 ;;
esac
META_SHA1="f0df49ffe5f2a6008d7dc83c6915b31835dfe733"

mkdir -p "$DEST"

dl_verify_unzip() {
  local url="$1" zip="$2" sha1="$3" outdir="$4"
  if [ -d "$outdir" ]; then echo "✓ $outdir already present — skipping"; return; fi
  if [ ! -f "$zip" ]; then
    echo "↓ downloading $(basename "$zip") ..."
    curl -fSL -C - "$url" -o "$zip"
  fi
  echo "· verifying SHA1 of $(basename "$zip") ..."
  local got
  got="$(shasum -a 1 "$zip" | awk '{print $1}')"
  if [ "$got" != "$sha1" ]; then
    echo "SHA1 MISMATCH for $zip: got $got expected $sha1" >&2
    echo "(delete the file and re-run to re-download)" >&2
    exit 1
  fi
  # FMA zips use bzip2 compression (method 12); macOS Info-ZIP `unzip` can't
  # handle it (exit 81). Python's zipfile does, and it's portable (macOS/Linux/Colab).
  echo "· extracting $(basename "$zip") (python zipfile — FMA uses bzip2) ..."
  python3 -c "import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$zip" "$DEST"
}

dl_verify_unzip "$BASE/fma_metadata.zip" "$DEST/fma_metadata.zip" "$META_SHA1" "$DEST/fma_metadata"
dl_verify_unzip "$BASE/fma_${SUBSET}.zip" "$DEST/fma_${SUBSET}.zip" "$AUDIO_SHA1" "$DEST/fma_${SUBSET}"

echo
echo "FMA $SUBSET ready:"
echo "  metadata: $DEST/fma_metadata"
echo "  audio:    $DEST/fma_${SUBSET}"
echo "Next: scripts/train_genre_v1.sh $SUBSET"
