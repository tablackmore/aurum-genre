# SP1: Reproducibility & Provenance Foundation

**Date:** 2026-07-02
**Status:** Approved (design) — pending spec review
**Reproducibility tier:** Tier 1 — *provable recipe + metrics* (not bit-exact weights)

## Context

`aurum-genre` trains an on-device genre tagger (`genre.onnx`) on permissively-licensed
audio. We have just trained a `genre-v1` model on the FMA-large permissive subset (2,206
tracks) with a set of training improvements. However, that model is **not currently
provable or reproducible**:

- The code that produced it is **uncommitted** (7 modified files); no commit/tag pins it.
- **No random seed** is set in training (`train.py`) — runs are nondeterministic and the
  early-stop "best epoch" depended on random chunk draws.
- The environment is **not pinned** (`pyproject.toml` uses version ranges).
- There is **no run record** capturing hyperparameters, dataset identity, hardware, or
  metrics for a specific model.

We are about to add more data (Jamendo CC-BY) and a subgenre model, so we need every future
run — and, as far as possible, the current one — to be documented and reproducible. This
is the foundation the later sub-projects (SP2 Jamendo integration, SP3 subgenre classifier)
build on.

**Goal:** any `genre.onnx` ships with a machine-readable record proving exactly what
produced it, such that someone can re-run the same recipe on the same data and obtain an
equivalent model (same metrics within noise), with a full audit trail. We explicitly do
**not** require bit-identical weights (MPS is nondeterministic; that was a deliberate
Tier-1 choice).

## What we already have (keep)

- `release/license_manifest.csv` — per-track provenance (id, artist, title, license, genre)
- `release/NOTICE` — CC-BY attribution
- `release/mel_recipe.txt` + `release/mel_golden.npz` — feature params + golden I/O vectors
- `scripts/download_fma.sh` — SHA1-verified FMA download

## Design

### 1. Commit discipline
- **First implementation step:** commit the current working tree (the 7 modified files that
  trained `genre-v1`) as the provenance baseline, *before* any SP1 code changes, so the
  recorded SHA maps to the shipped model.
- The run manifest records the **git commit SHA** and a **dirty flag** (`git status
  --porcelain` non-empty → `code_state: "dirty"`). Training does not refuse to run when
  dirty, but the dirty state is recorded honestly.

### 2. Pinned environment
- Generate `requirements.lock` via `pip freeze` (exact versions, incl. torch/torchaudio),
  committed alongside the ranged `pyproject.toml`.
- Record Python version, platform, and torch/torchaudio versions in the run manifest.

### 3. Seeding (`aurum_genre/seed.py`)
- New `seed_everything(seed: int)` sets `random`, `numpy`, and `torch` (CPU + MPS/CUDA)
  seeds and a DataLoader `worker_init_fn`.
- `fit()` and `run_pipeline.py` accept `--seed` (default `1337`). Makes runs *near*-
  deterministic; not bit-exact on MPS (accepted per Tier 1).

### 4. Run manifest writer (`aurum_genre/provenance.py`)
- `write_run_manifest(path, **fields)` writes `release/run_manifest.json` capturing:
  - `git`: commit SHA, dirty flag, branch
  - `seed`
  - `hyperparameters`: epochs, batch_size, lr, weight_decay, chunks_per_track, augment,
    mixup_alpha, pos_weight, patience, subset
  - `dataset`: train/val row counts, **SHA256 of each sorted manifest CSV**, and a
    `track_list_sha256` (hash of the sorted filepath list) so the exact track set is pinned;
    source descriptors (FMA subset + zip SHA1s; later Jamendo track list + `sha256_tracks`)
  - `environment`: python, platform, device, torch, torchaudio versions
  - `metrics`: final macro-AUC (single-chunk **and** chunk-averaged) + per-class AUC + support
  - `mel_recipe`: SR/N_FFT/HOP/N_MELS/CHUNK_SAMPLES (mirrors mel.py)
  - `timestamps`: start/end (passed in; `Date`/clock not used inside library code)
- Wired into `run_pipeline.py` as part of step 5 (packaging). Added to
  `package_release.verify_release` required-assets list.

### 5. Data checksums
- Manifest CSVs hashed (SHA256) into the run manifest (see §4).
- A helper `sha256_file(path)` reused for manifests and (in SP2) Jamendo tracks.

### 6. Retroactive record for the current model
- Reconstruct `release/run_manifest.json` for the shipped `genre-v1` from the training log
  we still have (`data/train_run.log`: 60→18 epochs early-stop, best val_auc=0.8274,
  final metrics) + the per-class numbers already computed, flagged
  `code_state: "uncommitted-reconstructed"` for honesty. Superseded by a clean manifest on
  the next full run.

### 7. Human-readable doc (`docs/REPRODUCIBILITY.md`)
- "How this model was made" + "How to rebuild it from scratch" (download → manifests →
  train with seed → verify metrics + golden vectors). Points at `run_manifest.json` as the
  source of truth.

## Components & boundaries

| Unit | Responsibility | Depends on |
|---|---|---|
| `aurum_genre/seed.py` | deterministic seeding + worker_init_fn | torch, numpy |
| `aurum_genre/provenance.py` | build/write run manifest; file + manifest hashing; git introspection | stdlib, subprocess(git) |
| `train.py` (edit) | accept/seed `--seed`; expose hyperparameters for the manifest | seed.py |
| `run_pipeline.py` (edit) | gather fields, call `write_run_manifest`, seed | provenance.py, seed.py |
| `package_release.py` (edit) | require `run_manifest.json` | — |

Each is independently testable and small.

## Testing

- `provenance.py`: `sha256_file` deterministic on known bytes; `write_run_manifest` produces
  JSON with all required top-level keys; manifest-hash stable across row reorderings that
  sort identically.
- `seed.py`: two seeded CPU runs of a tiny `fit` give identical first-epoch loss.
- `package_release`: fails when `run_manifest.json` is absent, passes when present.
- Full suite stays green (currently 33 tests).

## Out of scope

- Bit-exact deterministic weights (Tier 2).
- ShareAlike-licensed data (excluded by user decision).
- SP2/SP3 work (separate specs), though the manifest schema is designed to accommodate the
  Jamendo source descriptor and subgenre metrics.

## Verification

1. `pytest` green.
2. A fresh `run_pipeline.py --subset small --epochs 2 --seed 1337` produces
   `release/run_manifest.json` with all fields populated and `package_release` passing.
3. Re-running the same command reproduces the recorded **metrics within tolerance** and an
   identical **dataset SHA256** (proving data identity), even if weights differ.
4. `docs/REPRODUCIBILITY.md` steps, followed manually, rebuild an equivalent model.
