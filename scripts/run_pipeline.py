"""End-to-end genre-v1 pipeline: manifests → train → eval → export → package.

Assumes FMA is already downloaded (see scripts/download_fma.sh). Produces a
complete `release/` dir ready to publish as the `genre-v1` GitHub release asset.

Usage:
  python scripts/run_pipeline.py --subset small  --epochs 30
  python scripts/run_pipeline.py --subset medium --epochs 60 --min-auc 0.80

Everything is best-effort resumable at the file level: manifests/checkpoint are
written under data/, final assets under release/.
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Allow `python scripts/run_pipeline.py` (script mode) to import the `scripts`
# package + `aurum_genre` by putting the repo root on the path.
sys.path.insert(0, str(ROOT))

from aurum_genre import train, eval as geval, export  # noqa: E402
from scripts.build_manifest import build as build_manifest  # noqa: E402
from scripts import package_release  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, choices=["small", "medium", "large"])
    ap.add_argument("--fma-meta", default="data/fma_metadata")
    ap.add_argument("--fma-audio", default=None, help="default data/fma_<subset>")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--min-auc", type=float, default=0.80,
                    help="advisory release bar for validation macro-AUC")
    a = ap.parse_args()
    audio = a.fma_audio or f"data/fma_{a.subset}"

    data = ROOT / "data"
    rel = ROOT / "release"
    data.mkdir(exist_ok=True)
    rel.mkdir(exist_ok=True)
    train_csv, val_csv = data / "train.csv", data / "val.csv"
    ckpt = data / "genre.pt"

    print(f"[1/5] building manifests (subset={a.subset}) ...")
    # NOTICE + license_manifest reflect the TRAINING data (the attribution obligation).
    build_manifest(a.fma_meta, audio, train_csv, rel / "NOTICE",
                   rel / "license_manifest.csv", subset=a.subset, split="training")
    build_manifest(a.fma_meta, audio, val_csv, data / "_val_notice",
                   None, subset=a.subset, split="validation")

    print(f"[2/5] training ({a.epochs} epochs) ...")
    train.fit(str(train_csv), a.epochs, str(ckpt))

    print("[3/5] evaluating on the validation split ...")
    metrics = geval.evaluate(str(ckpt), str(val_csv), str(rel / "thresholds.json"))
    auc = metrics["macro_auc"]

    print("[4/5] exporting genre.onnx + golden vectors ...")
    export.export_onnx(str(ckpt), str(rel / "genre.onnx"),
                       str(rel / "mel_golden.npz"), str(rel / "mel_recipe.txt"))
    shutil.copy(ROOT / "taxonomy.json", rel / "taxonomy.json")

    print("[5/5] packaging release/ ...")
    ok, missing = package_release.verify_release(rel)
    if not ok:
        raise SystemExit(f"release incomplete, missing: {missing}")
    package_release.write_manifest(rel)

    verdict = "PASS" if auc >= a.min_auc else "BELOW BAR — review before publishing"
    print("\n" + "=" * 60)
    print(f"validation macro-AUC = {auc:.4f}   (advisory bar {a.min_auc:.2f}) → {verdict}")
    print(f"release assets ready in: {rel}")
    print("Publish:  gh release create genre-v1 -R tablackmore/aurum-genre "
          f"{rel}/genre.onnx {rel}/taxonomy.json {rel}/thresholds.json "
          f"{rel}/mel_recipe.txt {rel}/NOTICE {rel}/license_manifest.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
