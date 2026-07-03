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

import datetime as _dt
from aurum_genre import train, eval as geval, export  # noqa: E402
from aurum_genre import provenance  # noqa: E402
from aurum_genre.seed import seed_everything  # noqa: E402
from aurum_genre.mel import SR, N_FFT, HOP, N_MELS, CHUNK_SAMPLES  # noqa: E402
from scripts.build_manifest import build as build_manifest  # noqa: E402
from scripts import package_release  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["small", "medium", "large"],
                    help="FMA subset to build manifests from (omit when using --train-csv/--val-csv)")
    ap.add_argument("--fma-meta", default="data/fma_metadata")
    ap.add_argument("--fma-audio", default=None, help="default data/fma_<subset>")
    ap.add_argument("--train-csv", default=None,
                    help="pre-built training manifest; skips FMA manifest build (e.g. combined FMA+Jamendo)")
    ap.add_argument("--val-csv", default=None, help="pre-built validation manifest")
    ap.add_argument("--source", default=None, help="dataset source descriptor for the run manifest")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--min-auc", type=float, default=0.80,
                    help="advisory release bar for validation macro-AUC")
    a = ap.parse_args()
    seed_everything(a.seed)
    _t_start = _dt.datetime.now(_dt.timezone.utc).isoformat()

    data = ROOT / "data"
    rel = ROOT / "release"
    data.mkdir(exist_ok=True)
    rel.mkdir(exist_ok=True)
    ckpt = data / "genre.pt"

    if a.train_csv and a.val_csv:
        # Pre-built manifests (e.g. combined FMA+Jamendo). NOTICE + license_manifest
        # are expected to already be in release/ (written by the combined builder).
        train_csv, val_csv = Path(a.train_csv), Path(a.val_csv)
        print(f"[1/5] using pre-built manifests: {train_csv.name} / {val_csv.name}")
        if not (rel / "NOTICE").exists():
            (rel / "NOTICE").write_text("Attribution: see license_manifest.csv\n")
    else:
        if not a.subset:
            raise SystemExit("provide either --subset or both --train-csv and --val-csv")
        audio = a.fma_audio or f"data/fma_{a.subset}"
        train_csv, val_csv = data / "train.csv", data / "val.csv"
        print(f"[1/5] building manifests (subset={a.subset}) ...")
        # NOTICE + license_manifest reflect the TRAINING data (the attribution obligation).
        build_manifest(a.fma_meta, audio, train_csv, rel / "NOTICE",
                       rel / "license_manifest.csv", subset=a.subset, split="training")
        build_manifest(a.fma_meta, audio, val_csv, data / "_val_notice",
                       None, subset=a.subset, split="validation")

    print(f"[2/5] training ({a.epochs} epochs) ...")
    # Pass the val split so training keeps the best-on-validation checkpoint and
    # can early-stop (see aurum_genre.train.fit). Cache dir from AURUM_CACHE_DIR.
    train.fit(str(train_csv), a.epochs, str(ckpt), val_manifest=str(val_csv), seed=a.seed)

    print("[3/5] evaluating on the validation split ...")
    metrics = geval.evaluate(str(ckpt), str(val_csv), str(rel / "thresholds.json"))
    auc = metrics["macro_auc"]

    print("[4/5] exporting genre.onnx + golden vectors ...")
    export.export_onnx(str(ckpt), str(rel / "genre.onnx"),
                       str(rel / "mel_golden.npz"), str(rel / "mel_recipe.txt"))
    shutil.copy(ROOT / "taxonomy.json", rel / "taxonomy.json")

    print("[5/5] writing run manifest + packaging release/ ...")
    import torch as _torch
    cfg = _torch.load(str(ckpt), map_location="cpu", weights_only=True).get("config", {})
    chunked = geval.chunk_averaged_metrics(str(ckpt), str(val_csv))
    manifest = provenance.build_run_manifest(
        repo_dir=ROOT, seed=a.seed, hyperparameters={**cfg, "subset": a.subset},
        dataset={
            "source": ({"kind": a.source} if a.source else {"kind": "fma", "subset": a.subset}),
            "train_rows": sum(1 for _ in open(train_csv)) - 1,
            "val_rows": sum(1 for _ in open(val_csv)) - 1,
            "train_manifest_sha256": provenance.sha256_file(train_csv),
            "val_manifest_sha256": provenance.sha256_file(val_csv),
            "train_track_list_sha256": provenance.manifest_track_hash(train_csv),
        },
        device=train.default_device(),
        metrics={"macro_auc_single_chunk": auc,
                 "macro_auc_chunk_avg": chunked["macro_auc"],
                 "per_class": chunked["per_class"]},
        mel_recipe={"SR": SR, "N_FFT": N_FFT, "HOP": HOP, "N_MELS": N_MELS,
                    "CHUNK_SAMPLES": CHUNK_SAMPLES},
        timestamps={"start": _t_start,
                    "end": _dt.datetime.now(_dt.timezone.utc).isoformat()})
    provenance.write_run_manifest(rel / "run_manifest.json", manifest)
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
