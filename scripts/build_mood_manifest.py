"""MTG-Jamendo CC-BY mood tags → train/val manifests for a separate mood model.

Mood labels exist for only ~1,095 of our tracks, so a mood model is trained on
those alone (adding mood to the genre model would treat un-annotated tracks as
mood-negative — the partial-label trap). Same manifest schema as the genre
pipeline, so training reuses run_pipeline / train.fit with --taxonomy mood_taxonomy.json.
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `scripts` import
from scripts.build_combined_manifest import stratified_split  # noqa: E402

MOODS = ["happy", "sad", "dark", "uplifting", "energetic",
         "relaxing", "emotional", "meditative"]


def _ccby_paths(mtg_meta: Path) -> set[str]:
    lic, cur = set(), None
    for ln in (mtg_meta / "audio_licenses.txt").read_text(errors="ignore").splitlines():
        ln = ln.strip()
        if re.match(r"^\d+/\d+\.mp3$", ln):
            cur = ln
        elif cur and "creativecommons.org/licenses/by/" in ln:
            lic.add(cur); cur = None
    return lic


def build(mtg_meta, audio_base, out_train, out_val, moods=MOODS,
          seed: int = 1337, val_frac: float = 0.15, audio_ext: str = ".low.mp3"):
    mtg_meta, audio_base = Path(mtg_meta), Path(audio_base)
    ccby = _ccby_paths(mtg_meta)
    keep = set(moods)
    rows = []
    with open(mtg_meta / "data" / "autotagging_moodtheme.tsv") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 6 or p[3] not in ccby:
                continue
            tags = [t.split("---")[1] for t in p[5:] if t.startswith("mood/theme---")]
            labels = [m for m in tags if m in keep]
            if not labels:
                continue
            prefix, name = p[3].split("/")
            fp = audio_base / prefix / (name.replace(".mp3", audio_ext))
            if not fp.exists():
                continue
            rows.append({"filepath": str(fp), "root_labels": "|".join(sorted(labels))})
    df = pd.DataFrame(rows).drop_duplicates(subset=["filepath"])
    train, val = stratified_split(df, seed, val_frac, min_val=4)
    Path(out_train).parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_train, index=False)
    val.to_csv(out_val, index=False)
    print(f"wrote {out_train} ({len(train)}) + {out_val} ({len(val)}) — {len(df)} mood-tagged tracks")
    return len(train), len(val)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtg-meta", required=True)
    ap.add_argument("--audio-base", default="data/mtg_jamendo")
    ap.add_argument("--out-train", default="data/mood_train.csv")
    ap.add_argument("--out-val", default="data/mood_val.csv")
    a = ap.parse_args()
    build(a.mtg_meta, a.audio_base, a.out_train, a.out_val)


if __name__ == "__main__":
    main()
