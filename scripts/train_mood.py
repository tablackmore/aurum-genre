"""Train the separate mood model (8 moods) on the CC-BY mood-tagged tracks.

Reuses the genre pipeline via mood_taxonomy.json. Produces data/mood.pt and
release/mood.onnx + mood metrics. Small dataset (~560 tracks) — mood is inherently
harder and noisier than genre, so treat AUCs as a proof-of-concept.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from aurum_genre import train, eval as geval, export  # noqa: E402
from aurum_genre.seed import seed_everything  # noqa: E402


def main() -> None:
    seed_everything(1337)
    tax = str(ROOT / "mood_taxonomy.json")
    ckpt = str(ROOT / "data/mood.pt")
    train.fit(str(ROOT / "data/mood_train.csv"), 60, ckpt,
              val_manifest=str(ROOT / "data/mood_val.csv"),
              taxonomy_path=tax, seed=1337)
    m = geval.chunk_averaged_metrics(ckpt, str(ROOT / "data/mood_val.csv"))
    print(f"\nMOOD macro-AUC (chunk-avg) = {round(m['macro_auc'], 4)}")
    for k in sorted(m["per_class"], key=lambda k: -(m["per_class"][k]["auc"] or -1)):
        v = m["per_class"][k]
        print(f"   {k:<12} sup={v['support']:<4} auc={'n/a' if v['auc'] is None else round(v['auc'], 3)}")
    export.export_onnx(ckpt, str(ROOT / "release/mood.onnx"),
                       str(ROOT / "data/mood_golden.npz"), str(ROOT / "data/mood_recipe.txt"))
    print("exported release/mood.onnx")


if __name__ == "__main__":
    main()
