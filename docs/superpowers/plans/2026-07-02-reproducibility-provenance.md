# Reproducibility & Provenance Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every `genre.onnx` come with a machine-readable `run_manifest.json` proving exactly what produced it (code SHA, seed, hyperparameters, dataset hashes, environment, metrics), and make runs near-deterministic via seeding.

**Architecture:** Two new small library modules — `aurum_genre/seed.py` (seeding) and `aurum_genre/provenance.py` (manifest build/write, hashing, git introspection). `train.fit` records its effective config into the checkpoint; `run_pipeline.py` seeds, gathers metrics (single-chunk + chunk-averaged per-class), and writes `release/run_manifest.json`, which `package_release` now requires.

**Tech Stack:** Python 3.12, PyTorch/torchaudio (MPS), pandas, scikit-learn, pytest. Git for provenance.

## Global Constraints

- `requires-python = ">=3.12"`; deps stay within `pyproject.toml` ranges (`torch>=2.5,<2.7`, `torchaudio>=2.5,<2.7`, `numpy>=1.26,<2.2`). **Do not add new runtime dependencies** — provenance uses only stdlib + `subprocess` git.
- Tier 1 reproducibility: near-deterministic, **not** bit-exact. Do not add `torch.use_deterministic_algorithms` or force CPU.
- Checkpoint dicts must stay loadable with `torch.load(..., weights_only=True)` — only tensors + basic Python types (str/int/float/bool/list/dict) in the blob.
- Default seed is `1337`. Golden-vector seeds elsewhere (`export.py` 1234) are unrelated — do not touch them.
- Keep the full test suite green (currently 33 tests). Run `pytest -q` from repo root inside `.venv`.

---

### Task 1: Provenance baseline commit

Commit the current training code (already in the working tree) so its SHA maps to the shipped `genre-v1` model, **before** any SP1 changes.

**Files:**
- Modify (commit as-is): `aurum_genre/dataset.py`, `aurum_genre/eval.py`, `aurum_genre/train.py`, `scripts/download_fma.sh`, `scripts/run_pipeline.py`, `tests/test_dataset.py`, `tests/test_train.py`

- [ ] **Step 1: Verify the suite passes on the current tree**

Run: `cd ~/Documents/aurum/aurum-genre && source .venv/bin/activate && pytest -q`
Expected: `33 passed`

- [ ] **Step 2: Commit the training code as the provenance baseline**

```bash
git add aurum_genre/dataset.py aurum_genre/eval.py aurum_genre/train.py \
        scripts/download_fma.sh scripts/run_pipeline.py \
        tests/test_dataset.py tests/test_train.py
git commit -m "feat: genre-v1 training baseline (MPS, cache, A/B/C/D, corrupt-file skip)

Source of record for the FMA-large genre-v1 genre.onnx.
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 3: Record the baseline SHA for Task 8**

Run: `git rev-parse HEAD`
Expected: prints a 40-char SHA. Note it — Task 8's retroactive manifest references it.

---

### Task 2: Seeding module

**Files:**
- Create: `aurum_genre/seed.py`
- Test: `tests/test_seed.py`

**Interfaces:**
- Produces: `seed_everything(seed: int) -> None`; `worker_init_fn(worker_id: int) -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_seed.py
import random
import numpy as np
import torch
from aurum_genre.seed import seed_everything

def test_seed_makes_draws_reproducible():
    seed_everything(123)
    a = (torch.rand(4).tolist(), np.random.rand(4).tolist(), random.random())
    seed_everything(123)
    b = (torch.rand(4).tolist(), np.random.rand(4).tolist(), random.random())
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_seed.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aurum_genre.seed'`

- [ ] **Step 3: Write minimal implementation**

```python
# aurum_genre/seed.py
"""Deterministic seeding for near-reproducible training (Tier 1; not bit-exact)."""
from __future__ import annotations
import os
import random
import numpy as np
import torch

def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id: int) -> None:
    # torch seeds each worker's torch RNG; mirror that into random + numpy so
    # the dataset's random chunk selection is deterministic per run.
    base = torch.initial_seed() % (2 ** 31)
    random.seed(base + worker_id)
    np.random.seed((base + worker_id) % (2 ** 31))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_seed.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aurum_genre/seed.py tests/test_seed.py
git commit -m "feat: add seed_everything + worker_init_fn for reproducible runs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Seed training + record config in checkpoint

Add `seed` and `num_workers` params to `fit`, seed at the top, pass `worker_init_fn` to the DataLoader, and record the effective config (incl. `best_val_auc`) into the checkpoint blob.

**Files:**
- Modify: `aurum_genre/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `aurum_genre.seed.seed_everything`, `aurum_genre.seed.worker_init_fn`
- Produces: `fit(..., seed: int = 1337, num_workers: int = 4)`; checkpoint blob gains `blob["config"]: dict` with keys `seed, epochs, batch_size, lr, weight_decay, chunks_per_track, augment, mixup_alpha, use_pos_weight, patience, best_val_auc`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_train.py
def test_fit_records_config_and_seed(tmp_path):
    import torch, torchaudio, pandas as pd
    from aurum_genre.train import fit
    sr = 16000
    rows = []
    for i in range(40):
        f = 220 if i % 2 == 0 else 440
        wav = (0.2 * torch.sin(2 * torch.pi * f * torch.arange(sr * 4) / sr)).unsqueeze(0)
        fp = tmp_path / f"t{i}.wav"; torchaudio.save(str(fp), wav, sr)
        rows.append({"filepath": str(fp), "root_labels": "electronic" if i % 2 == 0 else "rock"})
    man = tmp_path / "m.csv"; pd.DataFrame(rows).to_csv(man, index=False)
    ckpt = tmp_path / "g.pt"
    fit(str(man), epochs=1, out_ckpt=str(ckpt), device="cpu", num_workers=0, seed=99)
    blob = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    cfg = blob["config"]
    assert cfg["seed"] == 99 and cfg["epochs"] == 1
    assert set(cfg) >= {"seed", "epochs", "batch_size", "lr", "weight_decay",
                        "chunks_per_track", "augment", "mixup_alpha",
                        "use_pos_weight", "patience", "best_val_auc"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train.py::test_fit_records_config_and_seed -v`
Expected: FAIL — `TypeError` (unexpected `num_workers`/`seed`) or `KeyError: 'config'`

- [ ] **Step 3: Implement**

In `aurum_genre/train.py`, add to imports near the top:

```python
from .seed import seed_everything, worker_init_fn
```

Change the `fit` signature to add `seed` and `num_workers`:

```python
def fit(manifest: str, epochs: int, out_ckpt: str,
        batch_size: int = 32, lr: float = 1e-3, device: str | None = None,
        taxonomy_path: str | Path | None = None, val_manifest: str | None = None,
        cache_dir: str | None = None, chunks_per_track: int = 4, augment: bool = True,
        mixup_alpha: float = 0.2, weight_decay: float = 1e-4,
        use_pos_weight: bool = True, patience: int = 8,
        seed: int = 1337, num_workers: int = 4) -> None:
```

Immediately after the `device = ...` line, add:

```python
    seed_everything(seed)
```

Change the `DataLoader(...)` line to pass the worker seeder:

```python
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, drop_last=True,
                        worker_init_fn=worker_init_fn)
```

Replace the final `torch.save({...})` call with a config-carrying blob:

```python
    config = {"seed": seed, "epochs": epochs, "batch_size": batch_size, "lr": lr,
              "weight_decay": weight_decay, "chunks_per_track": chunks_per_track,
              "augment": augment, "mixup_alpha": mixup_alpha,
              "use_pos_weight": use_pos_weight, "patience": patience,
              "best_val_auc": (best_auc if best_state is not None else None)}
    torch.save({"state_dict": model.state_dict(), "roots": roots, "config": config}, out_ckpt)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_train.py -v && pytest tests/test_export.py tests/test_eval.py -q`
Expected: PASS (export/eval still load the checkpoint fine — they read `roots`/`state_dict` and ignore `config`).

- [ ] **Step 5: Commit**

```bash
git add aurum_genre/train.py tests/test_train.py
git commit -m "feat: seed training and record effective config in checkpoint

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Chunk-averaged per-class metrics

Add a function computing track-level metrics the way `infer.py` runs (average sigmoid over all chunks), returning per-class AUC + support.

**Files:**
- Modify: `aurum_genre/eval.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Produces: `chunk_averaged_metrics(ckpt: str, manifest: str, device: str | None = None) -> dict` returning `{"macro_auc": float, "per_class": {root: {"auc": float | None, "support": int}}}`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_eval.py
def test_chunk_averaged_metrics_shape(tmp_path):
    import torch, torchaudio, pandas as pd
    from aurum_genre.train import fit
    from aurum_genre.eval import chunk_averaged_metrics
    sr = 16000
    rows = []
    for i in range(40):
        f = 220 if i % 2 == 0 else 440
        wav = (0.2 * torch.sin(2 * torch.pi * f * torch.arange(sr * 4) / sr)).unsqueeze(0)
        fp = tmp_path / f"t{i}.wav"; torchaudio.save(str(fp), wav, sr)
        rows.append({"filepath": str(fp), "root_labels": "electronic" if i % 2 == 0 else "rock"})
    man = tmp_path / "m.csv"; pd.DataFrame(rows).to_csv(man, index=False)
    ckpt = tmp_path / "g.pt"
    fit(str(man), epochs=1, out_ckpt=str(ckpt), device="cpu", num_workers=0)
    m = chunk_averaged_metrics(str(ckpt), str(man), device="cpu")
    assert 0.0 <= m["macro_auc"] <= 1.0
    assert "electronic" in m["per_class"] and "support" in m["per_class"]["electronic"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval.py::test_chunk_averaged_metrics_shape -v`
Expected: FAIL — `ImportError: cannot import name 'chunk_averaged_metrics'`

- [ ] **Step 3: Implement** (append to `aurum_genre/eval.py`; `roc_auc_score` is already imported at top)

```python
def chunk_averaged_metrics(ckpt: str, manifest: str, device: str | None = None) -> dict:
    """Track-level metrics matching infer.py: average sigmoid over all chunks."""
    import torch
    import pandas as pd
    from .model import ShortChunkCNN
    from .dataset import _load_mono_16k, multihot
    from .mel import log_mel, CHUNK_SAMPLES
    from .train import default_device
    dev = device or default_device()
    blob = torch.load(ckpt, map_location="cpu", weights_only=True)
    roots = blob["roots"]
    model = ShortChunkCNN(num_classes=len(roots))
    model.load_state_dict(blob["state_dict"]); model.eval(); model.to(dev)
    df = pd.read_csv(manifest)
    ys, ss = [], []
    with torch.no_grad():
        for _, row in df.iterrows():
            try:
                wav = _load_mono_16k(row["filepath"], None)
            except Exception:
                continue
            n = wav.shape[-1]
            if n < CHUNK_SAMPLES:
                chunks = [torch.nn.functional.pad(wav, (0, CHUNK_SAMPLES - n))]
            else:
                chunks = [wav[:, s:s + CHUNK_SAMPLES]
                          for s in range(0, n - CHUNK_SAMPLES + 1, CHUNK_SAMPLES)]
            mels = torch.stack([log_mel(c) for c in chunks]).to(dev)
            probs = torch.sigmoid(model(mels)).mean(0).cpu().numpy()
            ss.append(probs)
            raw = row["root_labels"]
            labs = [] if pd.isna(raw) else str(raw).split("|")
            ys.append(multihot(labs, roots).numpy())
    y, s = np.array(ys), np.array(ss)
    per_class, aucs = {}, []
    for c, name in enumerate(roots):
        sup = int(y[:, c].sum())
        if len(np.unique(y[:, c])) < 2:
            per_class[name] = {"auc": None, "support": sup}
        else:
            a = float(roc_auc_score(y[:, c], s[:, c])); aucs.append(a)
            per_class[name] = {"auc": a, "support": sup}
    return {"macro_auc": float(np.mean(aucs)) if aucs else 0.0, "per_class": per_class}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aurum_genre/eval.py tests/test_eval.py
git commit -m "feat: chunk-averaged per-class metrics (matches infer.py)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Provenance module

**Files:**
- Create: `aurum_genre/provenance.py`
- Test: `tests/test_provenance.py`

**Interfaces:**
- Produces:
  - `sha256_file(path) -> str`
  - `manifest_track_hash(manifest_csv) -> str` (SHA256 of sorted `filepath` column)
  - `git_info(repo_dir) -> dict` keys `commit, dirty, branch, code_state`
  - `env_info(device) -> dict` keys `python, platform, device, torch, torchaudio`
  - `build_run_manifest(*, repo_dir, seed, hyperparameters, dataset, device, metrics, mel_recipe, timestamps) -> dict`
  - `write_run_manifest(path, manifest) -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_provenance.py
import hashlib
import json
from aurum_genre.provenance import (sha256_file, manifest_track_hash,
                                     build_run_manifest, write_run_manifest)

def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "x.bin"; p.write_bytes(b"aurum")
    assert sha256_file(p) == hashlib.sha256(b"aurum").hexdigest()

def test_manifest_track_hash_is_order_independent(tmp_path):
    import pandas as pd
    a = tmp_path / "a.csv"; b = tmp_path / "b.csv"
    pd.DataFrame({"filepath": ["z.mp3", "a.mp3"], "root_labels": ["x", "y"]}).to_csv(a, index=False)
    pd.DataFrame({"filepath": ["a.mp3", "z.mp3"], "root_labels": ["y", "x"]}).to_csv(b, index=False)
    assert manifest_track_hash(a) == manifest_track_hash(b)

def test_build_and_write_manifest_has_required_keys(tmp_path):
    m = build_run_manifest(repo_dir=tmp_path, seed=1337, hyperparameters={"epochs": 1},
                           dataset={"train_rows": 10}, device="cpu",
                           metrics={"macro_auc": 0.8}, mel_recipe={"SR": 16000},
                           timestamps={"start": "t0", "end": "t1"})
    for k in ("schema", "git", "seed", "hyperparameters", "dataset",
              "environment", "metrics", "mel_recipe", "timestamps"):
        assert k in m
    out = tmp_path / "run_manifest.json"; write_run_manifest(out, m)
    assert json.loads(out.read_text())["seed"] == 1337
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_provenance.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aurum_genre.provenance'`

- [ ] **Step 3: Implement**

```python
# aurum_genre/provenance.py
"""Build/write the release run manifest: git state, env, dataset hashes, metrics."""
from __future__ import annotations
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def manifest_track_hash(manifest_csv) -> str:
    import pandas as pd
    df = pd.read_csv(manifest_csv)
    joined = "\n".join(sorted(df["filepath"].astype(str)))
    return hashlib.sha256(joined.encode()).hexdigest()

def git_info(repo_dir) -> dict:
    def run(*args):
        return subprocess.run(["git", "-C", str(repo_dir), *args],
                              capture_output=True, text=True).stdout.strip()
    dirty = bool(run("status", "--porcelain"))
    return {"commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": dirty, "code_state": "dirty" if dirty else "clean"}

def env_info(device) -> dict:
    import torch
    import torchaudio
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "device": str(device), "torch": torch.__version__,
            "torchaudio": torchaudio.__version__}

def build_run_manifest(*, repo_dir, seed, hyperparameters, dataset, device,
                       metrics, mel_recipe, timestamps) -> dict:
    return {"schema": "aurum-genre/run-manifest/1",
            "git": git_info(repo_dir), "seed": seed,
            "hyperparameters": hyperparameters, "dataset": dataset,
            "environment": env_info(device), "metrics": metrics,
            "mel_recipe": mel_recipe, "timestamps": timestamps}

def write_run_manifest(path, manifest) -> None:
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_provenance.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aurum_genre/provenance.py tests/test_provenance.py
git commit -m "feat: provenance module (hashing, git/env info, run manifest)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Wire manifest + seed into the pipeline; require it in release

**Files:**
- Modify: `scripts/run_pipeline.py`
- Modify: `scripts/package_release.py:5-6` (add `run_manifest.json` to `REQUIRED`)
- Test: `tests/test_package_release.py` (create)

**Interfaces:**
- Consumes: `aurum_genre.provenance.{build_run_manifest,write_run_manifest,sha256_file,manifest_track_hash}`, `aurum_genre.eval.chunk_averaged_metrics`, `aurum_genre.seed.seed_everything`, checkpoint `blob["config"]`.

- [ ] **Step 1: Write the failing test for the release requirement**

```python
# tests/test_package_release.py
from scripts.package_release import verify_release, REQUIRED

def test_run_manifest_is_required(tmp_path):
    assert "run_manifest.json" in REQUIRED
    for name in REQUIRED:
        if name != "run_manifest.json":
            (tmp_path / name).write_text("x")
    ok, missing = verify_release(tmp_path)
    assert not ok and "run_manifest.json" in missing
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_package_release.py -v`
Expected: FAIL — `run_manifest.json` not in `REQUIRED`.

- [ ] **Step 3: Add to REQUIRED** in `scripts/package_release.py`:

```python
REQUIRED = ["genre.onnx", "taxonomy.json", "thresholds.json",
            "mel_recipe.txt", "mel_golden.npz", "NOTICE", "license_manifest.csv",
            "run_manifest.json"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_package_release.py -v`
Expected: PASS

- [ ] **Step 5: Seed + write the manifest in `run_pipeline.py`**

Add to imports (after existing `from aurum_genre import ...` line):

```python
import datetime as _dt
from aurum_genre import provenance
from aurum_genre.seed import seed_everything
from aurum_genre.mel import SR, N_FFT, HOP, N_MELS, CHUNK_SAMPLES
```

Add `--seed` to the argparser (next to `--epochs`):

```python
    ap.add_argument("--seed", type=int, default=1337)
```

Right after `a = ap.parse_args()`, seed and stamp the start time:

```python
    seed_everything(a.seed)
    _t_start = _dt.datetime.now(_dt.timezone.utc).isoformat()
```

Pass the seed into training — change the `train.fit(...)` call:

```python
    train.fit(str(train_csv), a.epochs, str(ckpt), val_manifest=str(val_csv), seed=a.seed)
```

After the export step and `shutil.copy(... taxonomy ...)`, before `[5/5] packaging`, build and write the manifest:

```python
    print("[5/5] writing run manifest + packaging release/ ...")
    import torch as _torch
    cfg = _torch.load(str(ckpt), map_location="cpu", weights_only=True).get("config", {})
    chunked = geval.chunk_averaged_metrics(str(ckpt), str(val_csv))
    manifest = provenance.build_run_manifest(
        repo_dir=ROOT, seed=a.seed, hyperparameters={**cfg, "subset": a.subset},
        dataset={
            "source": {"kind": "fma", "subset": a.subset},
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
```

(The existing `package_release.verify_release` / `write_manifest` block stays as-is, now after the write above.)

- [ ] **Step 6: Integration test — a real small run produces a valid manifest**

Run:
```bash
source .venv/bin/activate && export AURUM_CACHE_DIR=data/cache
python scripts/run_pipeline.py --subset small --epochs 2 --seed 1337
python -c "import json;m=json.load(open('release/run_manifest.json'));\
print('git',m['git']['commit'][:8],m['git']['code_state']);\
print('seed',m['seed']);print('macro_avg',round(m['metrics']['macro_auc_chunk_avg'],3));\
assert m['dataset']['train_track_list_sha256'] and m['environment']['torch']"
```
Expected: prints git/seed/macro and the assert passes; run ends `... → PASS`/`BELOW BAR` with `release/run_manifest.json` present.
(Requires `data/fma_small` — if absent, run `bash scripts/download_fma.sh small data` first.)

- [ ] **Step 7: Run full suite**

Run: `pytest -q`
Expected: PASS (all prior + new tests).

- [ ] **Step 8: Commit**

```bash
git add scripts/run_pipeline.py scripts/package_release.py tests/test_package_release.py
git commit -m "feat: write release/run_manifest.json and require it in packaging

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Pin the environment

**Files:**
- Create: `requirements.lock`

- [ ] **Step 1: Freeze exact versions**

Run: `source .venv/bin/activate && pip freeze --exclude-editable > requirements.lock`
Expected: `requirements.lock` lists exact pins incl. `torch==2.6.0`, `torchaudio==2.6.0`, `numpy==2.1.3`, `scikit-learn==...`.

- [ ] **Step 2: Sanity-check it's non-empty and has torch**

Run: `grep -E "^torch==|^torchaudio==" requirements.lock`
Expected: two lines printed.

- [ ] **Step 3: Commit**

```bash
git add requirements.lock
git commit -m "chore: pin exact dependency versions (requirements.lock)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: Retroactive manifest for the shipped genre-v1

Reconstruct a manifest for the existing `release/genre.onnx` from the training log, flagged honestly. Uses the already-present `data/genre.pt`, `data/train.csv`, `data/val.csv`.

**Files:**
- Create: `scripts/reconstruct_manifest.py`
- Output: `release/run_manifest.json` (retroactive)

- [ ] **Step 1: Write the reconstruction script**

```python
# scripts/reconstruct_manifest.py
"""One-off: reconstruct run_manifest.json for the shipped genre-v1 from artifacts
we still have (data/genre.pt config + data/train.csv/val.csv + training log).
Flagged code_state=uncommitted-reconstructed; superseded by the next clean run."""
from __future__ import annotations
import datetime as dt
from pathlib import Path
import torch
from aurum_genre import provenance, eval as geval, train
from aurum_genre.mel import SR, N_FFT, HOP, N_MELS, CHUNK_SAMPLES

ROOT = Path(__file__).resolve().parent.parent
ckpt = ROOT / "data/genre.pt"
train_csv, val_csv = ROOT / "data/train.csv", ROOT / "data/val.csv"
cfg = torch.load(str(ckpt), map_location="cpu", weights_only=True).get("config", {})
chunked = geval.chunk_averaged_metrics(str(ckpt), str(val_csv))
git = provenance.git_info(ROOT)
git["code_state"] = "uncommitted-reconstructed"  # honesty: trained before SP1
m = provenance.build_run_manifest(
    repo_dir=ROOT, seed=cfg.get("seed"), hyperparameters={**cfg, "subset": "large"},
    dataset={"source": {"kind": "fma", "subset": "large"},
             "train_rows": sum(1 for _ in open(train_csv)) - 1,
             "val_rows": sum(1 for _ in open(val_csv)) - 1,
             "train_manifest_sha256": provenance.sha256_file(train_csv),
             "val_manifest_sha256": provenance.sha256_file(val_csv),
             "train_track_list_sha256": provenance.manifest_track_hash(train_csv)},
    device="mps", metrics={"macro_auc_chunk_avg": chunked["macro_auc"],
                           "per_class": chunked["per_class"],
                           "note": "single-chunk pipeline verdict was 0.7627"},
    mel_recipe={"SR": SR, "N_FFT": N_FFT, "HOP": HOP, "N_MELS": N_MELS,
                "CHUNK_SAMPLES": CHUNK_SAMPLES},
    timestamps={"start": None, "end": None, "note": "reconstructed 2026-07-02"})
m["git"] = git
provenance.write_run_manifest(ROOT / "release/run_manifest.json", m)
print("wrote release/run_manifest.json (reconstructed) — macro_avg",
      round(chunked["macro_auc"], 4))
```

- [ ] **Step 2: Run it**

Run: `source .venv/bin/activate && python scripts/reconstruct_manifest.py`
Expected: prints `wrote release/run_manifest.json (reconstructed) — macro_avg 0.83xx`.

- [ ] **Step 3: Verify the shipped release now passes packaging**

Run: `python scripts/package_release.py release && echo OK`
Expected: `release OK` then `OK` (now that `run_manifest.json` exists).

- [ ] **Step 4: Commit**

```bash
git add scripts/reconstruct_manifest.py
git commit -m "chore: reconstruct run manifest for shipped genre-v1 (flagged)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: Human-readable reproducibility doc

**Files:**
- Create: `docs/REPRODUCIBILITY.md`

- [ ] **Step 1: Write the doc**

```markdown
# Reproducibility

Every `release/` ships `run_manifest.json` — the source of truth for how that
`genre.onnx` was produced: git commit + dirty flag, seed, hyperparameters,
dataset hashes (manifest SHA256 + track-list hash), environment (python/torch/
torchaudio/OS/device), and metrics (single-chunk and chunk-averaged per-class AUC).

## Rebuild from scratch

    python3.12 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.lock          # exact pinned versions
    pip install -e ".[dev]"
    bash scripts/download_fma.sh <subset> data # SHA1-verified audio
    export AURUM_CACHE_DIR=data/cache
    python scripts/run_pipeline.py --subset <subset> --epochs 60 --seed 1337

Verify against a prior `run_manifest.json`:
- `dataset.train_track_list_sha256` must match → identical training tracks.
- `metrics.macro_auc_chunk_avg` should match within noise (Tier 1: near-, not
  bit-deterministic; MPS is nondeterministic, so weights may differ).
- `mel_golden.npz` gives byte-level feature/inference parity for the exported model.

## What is and isn't guaranteed

- **Guaranteed:** exact data lineage (which tracks, which licenses), exact code
  (git SHA), pinned env, and metric reproducibility within noise.
- **Not guaranteed:** bit-identical weights (deliberate Tier-1 choice; would
  require deterministic algorithms and fixed non-MPS hardware).
```

- [ ] **Step 2: Commit**

```bash
git add docs/REPRODUCIBILITY.md
git commit -m "docs: how the model is made and how to rebuild it

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** §1 commit discipline → Task 1 + `git_info` dirty flag (Task 5). §2 pinned env → Task 7 + `env_info`. §3 seeding → Tasks 2–3. §4 run manifest → Tasks 5–6 (all listed fields present: git/seed/hyperparameters/dataset+SHA256/environment/metrics single+chunk-avg+per-class/mel_recipe/timestamps). §5 data checksums → `sha256_file`/`manifest_track_hash` (Task 5, used Task 6). §6 retroactive → Task 8. §7 doc → Task 9. Testing bullets → tests in Tasks 2,3,4,5,6. All covered.

**Placeholder scan:** No TBD/TODO; every code step shows full code; commands have expected output.

**Type consistency:** `chunk_averaged_metrics` returns `{"macro_auc", "per_class"}` — consumed as `chunked["macro_auc"]`/`chunked["per_class"]` in Tasks 6 & 8. `build_run_manifest` keyword args match call sites. Checkpoint `blob["config"]` written in Task 3, read in Tasks 6 & 8. `REQUIRED` includes `run_manifest.json` (Task 6) and is satisfied by Task 6/8 writes.
