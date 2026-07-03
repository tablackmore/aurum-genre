"""Consolidate the permissively-licensed training audio into a self-contained,
fully-attributed local library — so the tracks (and their licenses) survive
independently of the working data/ dir, ready for future retraining or reuse.

Output (default ~/Documents/aurum/free-music-library/):
  audio/fma/<track_id>.mp3
  audio/jamendo/<track_id>.low.mp3
  library_manifest.csv   # source, track_id, artist, title, license, genre_labels, path, sha256
  NOTICE                 # attribution
  README.md              # what this is + how to reuse

All audio is CC-BY / CC0 / public-domain (FMA-large permissive subset + MTG-Jamendo CC-BY).
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import os
import shutil
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
HOME_MTG = Path.home() / "Documents/aurum/mtg-jamendo-dataset"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fma_meta() -> dict:
    """track_id -> (artist, title, license) from FMA tracks.csv (covers all splits)."""
    tr = pd.read_csv(REPO / "data/fma_metadata/tracks.csv", index_col=0, header=[0, 1])
    return {str(int(tid)): (tr.loc[tid, ("artist", "name")],
                            tr.loc[tid, ("track", "title")],
                            tr.loc[tid, ("track", "license")]) for tid in tr.index}


def _jamendo_meta() -> dict:
    """track_id -> (artist, title) from MTG raw.meta.tsv (tolerant parse)."""
    m = pd.read_csv(HOME_MTG / "data/raw.meta.tsv", sep="\t", engine="python",
                    quoting=csv.QUOTE_NONE, on_bad_lines="skip")
    m["tid"] = m["TRACK_ID"].astype(str).str.replace("track_", "").apply(lambda s: str(int(s)))
    return {r.tid: (r.ARTIST_NAME, r.TRACK_NAME) for r in m.itertuples()}


def build(out_dir: Path) -> None:
    audio_fma = out_dir / "audio/fma"; audio_jam = out_dir / "audio/jamendo"
    audio_fma.mkdir(parents=True, exist_ok=True)
    audio_jam.mkdir(parents=True, exist_ok=True)

    # union of FMA train+val rows (filepath, root_labels), deduped
    fma_rows, seen = {}, set()
    for f in ["data/train.csv", "data/val.csv"]:
        for r in pd.read_csv(REPO / f).itertuples():
            if "fma_large" in r.filepath and r.filepath not in seen:
                seen.add(r.filepath); fma_rows[r.filepath] = r.root_labels
    jam_rows = {r.filepath: r.root_labels
                for r in pd.read_csv(REPO / "data/train_jamendo.csv").itertuples()}

    fma_meta, jam_meta = _fma_meta(), _jamendo_meta()
    manifest = []

    print(f"[library] copying {len(fma_rows)} FMA + {len(jam_rows)} Jamendo tracks ...", flush=True)
    for i, (fp, labels) in enumerate(fma_rows.items(), 1):
        tid = os.path.basename(fp).replace(".mp3", "")
        dst = audio_fma / f"{tid}.mp3"
        if not dst.exists():
            shutil.copy2(fp, dst)
        art, tit, lic = fma_meta.get(str(int(tid)), ("", "", "CC-BY"))  # manifest ids are unpadded
        manifest.append(["fma", tid, art, tit, lic, labels, f"audio/fma/{tid}.mp3", sha256(dst)])
        if i % 500 == 0: print(f"[library]   FMA {i}/{len(fma_rows)}", flush=True)

    for i, (fp, labels) in enumerate(jam_rows.items(), 1):
        tid = os.path.basename(fp).replace(".low.mp3", "")
        dst = audio_jam / f"{tid}.low.mp3"
        if not dst.exists():
            shutil.copy2(fp, dst)
        art, tit = jam_meta.get(tid, ("", ""))
        manifest.append(["jamendo", tid, art, tit, "CC-BY", labels, f"audio/jamendo/{tid}.low.mp3", sha256(dst)])
        if i % 500 == 0: print(f"[library]   Jamendo {i}/{len(jam_rows)}", flush=True)

    with open(out_dir / "library_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "track_id", "artist", "title", "license", "genre_labels", "path", "sha256"])
        w.writerows(manifest)

    n_fma, n_jam = len(fma_rows), len(jam_rows)
    (out_dir / "NOTICE").write_text(
        "AURUM free-to-use music library — attribution\n\n"
        "All tracks are permissively licensed (CC-BY / CC0 / public domain).\n"
        f"  - FMA-large permissive subset: {n_fma} tracks (https://freemusicarchive.org)\n"
        f"  - MTG-Jamendo CC-BY subset:    {n_jam} tracks (https://www.jamendo.com)\n\n"
        "Per-track artist, title, and license are in library_manifest.csv.\n"
        "CC-BY requires attribution to the listed artist on use.\n")
    (out_dir / "README.md").write_text(
        "# AURUM free-to-use music library\n\n"
        f"{n_fma + n_jam} permissively-licensed tracks (CC-BY / CC0 / public domain), "
        "consolidated for reuse and future model retraining.\n\n"
        "## Layout\n"
        "- `audio/fma/<id>.mp3` — FMA-large permissive subset\n"
        "- `audio/jamendo/<id>.low.mp3` — MTG-Jamendo CC-BY subset (mono, low bitrate)\n"
        "- `library_manifest.csv` — source, track_id, artist, title, license, genre_labels, path, sha256\n"
        "- `NOTICE` — attribution summary\n\n"
        "## Reuse / retraining\n"
        "`genre_labels` uses the aurum taxonomy (`electronic`, `rock:metal`, `electronic:techno`, ...). "
        "To retrain: point a manifest's `filepath` at these files with `root_labels` = `genre_labels`, "
        "then run `scripts/run_pipeline.py --train-csv ... --val-csv ...`.\n\n"
        "## Licensing\n"
        "Every track is CC-BY, CC0, or public domain — free to use commercially **with attribution** "
        "(see `library_manifest.csv` / `NOTICE`). No NonCommercial / NoDerivatives / ShareAlike tracks are included.\n")
    print(f"[library] DONE: {len(manifest)} tracks + manifest + NOTICE + README at {out_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "Documents/aurum/free-music-library"))
    a = ap.parse_args()
    build(Path(a.out))


if __name__ == "__main__":
    main()
