"""FMA metadata → filtered permissive manifest CSV (filepath, root_labels).

Usage: python scripts/build_manifest.py --fma-meta data/fma_metadata \
         --fma-audio data/fma_medium --out data/manifest.csv --notice release/NOTICE
Reads fma tracks.csv (multi-index header), keeps permissive licenses, maps the
track's top genre(s) to root labels, writes the manifest + NOTICE."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from aurum_genre.licenses import filter_permissive, build_notice
from aurum_genre.taxonomy import load_taxonomy, map_fma_root

_REPO_ROOT = Path(__file__).resolve().parent.parent

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fma-meta", required=True)
    ap.add_argument("--fma-audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--notice", required=True)
    a = ap.parse_args()

    tracks = pd.read_csv(Path(a.fma_meta) / "tracks.csv", index_col=0, header=[0, 1])
    df = pd.DataFrame({
        "track_id": tracks.index,
        "license": tracks[("track", "license")].values,
        "genre_top": tracks[("track", "genre_top")].values,
        "artist_name": tracks[("artist", "name")].values,
        "track_title": tracks[("track", "title")].values,
    }).dropna(subset=["genre_top"])

    df = filter_permissive(df, "license")
    tax = load_taxonomy(_REPO_ROOT / "taxonomy.json")
    df["root"] = df["genre_top"].apply(lambda g: map_fma_root(g, tax))
    df = df.dropna(subset=["root"])

    def fp(tid: int) -> str:
        s = f"{int(tid):06d}"
        return str(Path(a.fma_audio) / s[:3] / f"{s}.mp3")
    df["filepath"] = df["track_id"].apply(fp)
    df["root_labels"] = df["root"]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df[["filepath", "root_labels"]].to_csv(a.out, index=False)
    Path(a.notice).parent.mkdir(parents=True, exist_ok=True)
    Path(a.notice).write_text(build_notice(df))
    print(f"wrote {a.out} ({len(df)} tracks) + {a.notice}")

if __name__ == "__main__":
    main()
