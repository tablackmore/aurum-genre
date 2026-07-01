# Training `genre-v1`

This produces the `genre.onnx` + config that AURUM ships. The whole pipeline is
automated — you pick **where** to run it and **which FMA subset** to train on.

## 1. Pick a subset

| Subset | Size | Genres | Covers our roots | Use for |
|---|---|---|---|---|
| `small` | 7.2 GiB | 8 | ~6 of 11 (no jazz/classical/blues/country/soul) | **Validate the pipeline fast** |
| `medium` | 22 GiB | 16 | **all 11** | **The real `genre-v1`** |

Do a `small` run first end-to-end (proves the plumbing + gives a real macro-AUC),
then `medium` for the model you actually ship.

> Note: only **permissively-licensed** tracks (CC-BY/CC0/PD) survive the filter, so
> the trained set is a subset of the subset. The run prints the retained count.

## 2. Pick where to run — recommendation: **Colab (free GPU)**

| Where | GPU | Cost | Time (small→medium) | Setup |
|---|---|---|---|---|
| **Google Colab** ⭐ | free T4 | $0 | ~20 min → ~1–2 h | open notebook, Run all |
| Local Mac | MPS/CPU | $0 | ~1–2 h → overnight | `source .venv/bin/activate` |
| Cloud GPU (Lambda/RunPod/Vast) | A10/4090 | ~$0.30–0.50/h → a few $ | ~10 min → ~30 min | rent + `git clone` |

**Colab is the recommended path** — free GPU, nothing to install, reproducible.

### A) Colab (recommended)
1. Open **`notebooks/train_genre_colab.ipynb`** in Colab (github.com/tablackmore/aurum-genre → the notebook → "Open in Colab", or upload it).
2. **Runtime → Change runtime type → T4 GPU.**
3. Because the repo is **private**, add a Colab secret **`GH_TOKEN`** (🔑 sidebar) = a fine-grained PAT with `Contents:read` on `tablackmore/aurum-genre`. *(Or make the repo public and skip this — see §4.)*
4. Set `SUBSET`/`EPOCHS` in the third cell, then **Runtime → Run all**.
5. The last cell downloads `genre-v1.zip` (all release assets).

### B) Local Mac (no setup, slower)
```bash
cd ~/aurum/aurum-genre
source .venv/bin/activate
scripts/train_genre_v1.sh small --epochs 30      # validate
scripts/train_genre_v1.sh medium --epochs 60     # real genre-v1
```
Downloads FMA if missing, then trains → evals → exports → packages `release/`.
(Dataloading decodes mp3s on CPU, so this is the slow part on a Mac.)

### C) Cloud GPU (fastest)
Rent an instance, then run exactly the Local steps (the pipeline is identical).

## 3. What you get + the gate

Every path ends by writing **`release/`** and printing:
```
validation macro-AUC = 0.8xxx   (advisory bar 0.80) → PASS
```
`release/` contains: `genre.onnx`, `taxonomy.json`, `thresholds.json`,
`mel_recipe.txt`, `mel_golden.npz`, `NOTICE`, `license_manifest.csv`.

**Macro-AUC** is the standard multi-label auto-tagging metric (mean per-class
ROC-AUC on the held-out validation split). ~0.80+ at root level is a reasonable
first bar; if it's low, train more epochs or move from `small`→`medium`.

## 4. Publish + wire into the app

```bash
# from the run's output — publish the release asset:
gh release create genre-v1 -R tablackmore/aurum-genre \
  release/genre.onnx release/taxonomy.json release/thresholds.json \
  release/mel_recipe.txt release/NOTICE release/license_manifest.csv
```
Then in the **aurum** repo, `scripts/fetch-models.sh` already knows how to pull
`genre.onnx` + config from this release. Re-run it, rebuild, and the app tags on
import + via "Re-tag library".

When you're happy with a shipped model, **flip `aurum-genre` public** (the design
intends it to be open) so the Colab clone needs no token and the release is fetchable
without auth:
```bash
gh repo edit tablackmore/aurum-genre --visibility public
```

## Pipeline internals (for reference)
`scripts/run_pipeline.py` = build manifests (FMA `training`/`validation` split,
permissive-filtered) → `train.fit` → `eval.evaluate` (macro-AUC + per-class
F1-max thresholds) → `export.export_onnx` (opset-17, mel input, golden vectors) →
`package_release.verify_release`. Each stage is a tested module in `aurum_genre/`.
