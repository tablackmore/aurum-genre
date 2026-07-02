"""MTG-Jamendo CC-BY subset → manifest (filepath, root_labels) for genre-v2.

Reads the MTG metadata we already have (autotagging_genre.tsv for genre tags,
audio_licenses.txt for per-track license), keeps CC-BY / CC0 / public-domain
tracks, maps Jamendo genre tags to aurum roots + `electronic:<sub>` labels via
scripts/jamendo_genre_map.json, and points filepaths at the downloaded audio.

The resulting manifest has the SAME schema as scripts/build_manifest.py output,
so it can be concatenated with the FMA manifest and fed straight to training.

Usage:
  python scripts/build_jamendo_manifest.py --mtg-meta <dir> --audio-base data/mtg_jamendo \
      --out data/train_jamendo.csv [--audio-ext .low.mp3]
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MAP = _REPO_ROOT / "scripts" / "jamendo_genre_map.json"

_PERMISSIVE = {"by", "zero", "mark", "publicdomain"}


def parse_licenses(audio_licenses_txt: str | Path) -> dict[str, str]:
    """path (e.g. '14/214.mp3') -> CC license segment (e.g. 'by', 'by-nc-sa')."""
    lic: dict[str, str] = {}
    cur = None
    for ln in Path(audio_licenses_txt).read_text(errors="ignore").splitlines():
        ln = ln.strip()
        if re.match(r"^\d+/\d+\.mp3$", ln):
            cur = ln
        elif cur and "creativecommons.org" in ln:
            m = re.search(r"creativecommons\.org/(?:licenses|publicdomain)/([a-z0-9-]+)", ln)
            if m:
                lic[cur] = m.group(1)
            cur = None
    return lic


def parse_genres(autotagging_genre_tsv: str | Path) -> dict[str, list[str]]:
    """path -> list of genre tags (from 'genre---<tag>' columns)."""
    out: dict[str, list[str]] = {}
    with open(autotagging_genre_tsv) as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 6:
                continue
            out[p[3]] = [t.split("---")[1] for t in p[5:] if t.startswith("genre---")]
    return out


def labels_for_tags(tags: list[str], gmap: dict) -> list[str]:
    """Jamendo genre tags -> ordered, deduped [roots..., electronic:<sub>...]."""
    root_map, sub_map = gmap["root_map"], gmap["sub_map"]
    labels: list[str] = []
    for t in tags:
        r = root_map.get(t)
        if r and r not in labels:
            labels.append(r)
    for t in tags:
        s = sub_map.get(t)
        if s:
            lab = f"electronic:{s}"
            if "electronic" not in labels:
                labels.append("electronic")
            if lab not in labels:
                labels.append(lab)
    return labels


def _find(mtg_meta: str | Path, name: str) -> Path:
    """Locate a metadata file under mtg_meta or mtg_meta/data (MTG repo layout)."""
    root = Path(mtg_meta)
    for cand in (root / name, root / "data" / name):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"{name} not found under {root} or {root}/data")


def build(mtg_meta: str | Path, audio_base: str | Path, out: str | Path,
          genre_map: str | Path | None = None, audio_ext: str = ".mp3") -> int:
    """Write the CC-BY Jamendo manifest; returns the number of tracks written."""
    gmap = json.loads(Path(genre_map or _DEFAULT_MAP).read_text())
    lic = parse_licenses(_find(mtg_meta, "audio_licenses.txt"))
    genres = parse_genres(_find(mtg_meta, "autotagging_genre.tsv"))
    rows = []
    for path, seg in sorted(lic.items()):
        if seg not in _PERMISSIVE:
            continue
        labels = labels_for_tags(genres.get(path, []), gmap)
        if not labels:
            continue
        fp = str(Path(audio_base) / path.replace(".mp3", audio_ext))
        rows.append((fp, "|".join(labels)))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "root_labels"])
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} CC-BY Jamendo tracks)")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtg-meta", required=True, help="dir with audio_licenses.txt + autotagging_genre.tsv")
    ap.add_argument("--audio-base", required=True, help="dir holding the unpacked Jamendo audio")
    ap.add_argument("--out", required=True)
    ap.add_argument("--map", default=None)
    ap.add_argument("--audio-ext", default=".mp3", help="extension of downloaded files (e.g. .low.mp3)")
    a = ap.parse_args()
    build(a.mtg_meta, a.audio_base, a.out, a.map, a.audio_ext)


if __name__ == "__main__":
    main()
