"""FMA metadata → filtered permissive manifest CSV (filepath, root_labels).

Usage: python scripts/build_manifest.py --fma-meta data/fma_metadata \
         --fma-audio data/fma_medium --out data/manifest.csv --notice release/NOTICE \
         --license-manifest release/license_manifest.csv \
         --subset medium --split training
Reads fma tracks.csv (multi-index header), optionally restricts to an FMA
`subset` (small|medium|large; hierarchical — medium includes small) and a
`split` (training|validation|test), keeps permissive licenses, maps the track's
top genre to a root label, and writes the manifest + NOTICE + license_manifest."""
from __future__ import annotations
import argparse
import ast
from pathlib import Path
import pandas as pd
from aurum_genre.licenses import filter_permissive, build_notice
from aurum_genre.taxonomy import load_taxonomy, map_fma_root, map_fma_subgenres


def _load_genre_id_titles(fma_meta: str | Path) -> dict[int, str]:
    """FMA genre_id → title from genres.csv, if present (else empty → no subgenres)."""
    p = Path(fma_meta) / "genres.csv"
    if not p.exists():
        return {}
    g = pd.read_csv(p)
    return dict(zip(g["genre_id"].astype(int), g["title"].astype(str)))


def _fine_titles(raw, id2title: dict[int, str]) -> list[str]:
    """Parse a track.genres cell ('[15, 25]') to fine-genre titles."""
    if not id2title or pd.isna(raw):
        return []
    try:
        ids = ast.literal_eval(raw) if isinstance(raw, str) else list(raw)
    except (ValueError, SyntaxError):
        return []
    return [id2title[i] for i in ids if i in id2title]

_REPO_ROOT = Path(__file__).resolve().parent.parent

# FMA subsets are hierarchical: small ⊂ medium ⊂ large. A track's `subset` value
# is the SMALLEST subset it belongs to, so "medium" = rank <= 1.
_SUBSET_RANK = {"small": 0, "medium": 1, "large": 2}


def build(fma_meta: str | Path, fma_audio: str | Path,
          out: str | Path, notice: str | Path,
          license_manifest: str | Path | None = None,
          subset: str | None = None, split: str | None = None) -> None:
    """Core manifest-build logic; importable for testing."""
    tracks = pd.read_csv(Path(fma_meta) / "tracks.csv", index_col=0, header=[0, 1])
    # Fine-genre IDs (for subgenre labels) are optional — absent in minimal fixtures.
    has_genres = ("track", "genres") in tracks.columns
    df = pd.DataFrame({
        "track_id": tracks.index,
        "license": tracks[("track", "license")].values,
        "genre_top": tracks[("track", "genre_top")].values,
        "genres": tracks[("track", "genres")].values if has_genres else "",
        "artist_name": tracks[("artist", "name")].values,
        "track_title": tracks[("track", "title")].values,
        "subset": tracks[("set", "subset")].values,
        "split": tracks[("set", "split")].values,
    }).dropna(subset=["genre_top"])

    if subset is not None:
        max_rank = _SUBSET_RANK[subset]
        df = df[df["subset"].map(lambda s: _SUBSET_RANK.get(s, 99) <= max_rank)]
    if split is not None:
        df = df[df["split"] == split]

    df = filter_permissive(df, "license")
    tax = load_taxonomy(_REPO_ROOT / "taxonomy.json")
    df["root"] = df["genre_top"].apply(lambda g: map_fma_root(g, tax))
    df = df.dropna(subset=["root"])

    def fp(tid: int) -> str:
        s = f"{int(tid):06d}"
        return str(Path(fma_audio) / s[:3] / f"{s}.mp3")
    df["filepath"] = df["track_id"].apply(fp)

    # Multi-label output: root plus (for electronic tracks) namespaced subgenres
    # derived from the track's fine FMA genres. Degrades to root-only when the
    # genres column / genres.csv are absent.
    id2title = _load_genre_id_titles(fma_meta)

    subbed_roots = set(tax.get("sub", {}))

    def labels_for(row) -> str:
        labels = [row["root"]]
        if row["root"] in subbed_roots:
            subs = map_fma_subgenres(_fine_titles(row["genres"], id2title), tax)
            labels += [s for s in subs if s.split(":", 1)[0] == row["root"]]
        return "|".join(labels)
    df["root_labels"] = df.apply(labels_for, axis=1)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df[["filepath", "root_labels"]].to_csv(out, index=False)
    Path(notice).parent.mkdir(parents=True, exist_ok=True)
    Path(notice).write_text(build_notice(df))
    tag = f" [subset={subset or 'all'} split={split or 'all'}]"
    print(f"wrote {out} ({len(df)} tracks){tag} + {notice}")

    if license_manifest is not None:
        Path(license_manifest).parent.mkdir(parents=True, exist_ok=True)
        df[["track_id", "artist_name", "track_title", "license", "root"]].to_csv(
            license_manifest, index=False
        )
        print(f"wrote {license_manifest} ({len(df)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fma-meta", required=True)
    ap.add_argument("--fma-audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--notice", required=True)
    ap.add_argument("--license-manifest", default=None,
                    help="Path to write per-track provenance CSV")
    ap.add_argument("--subset", default=None, choices=["small", "medium", "large"],
                    help="Restrict to an FMA subset (hierarchical)")
    ap.add_argument("--split", default=None,
                    choices=["training", "validation", "test"],
                    help="Restrict to an FMA split")
    a = ap.parse_args()
    build(a.fma_meta, a.fma_audio, a.out, a.notice,
          a.license_manifest, a.subset, a.split)


if __name__ == "__main__":
    main()
