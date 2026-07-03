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
        result = subprocess.run(["git", "-C", str(repo_dir), *args],
                                capture_output=True, text=True)
        return result.returncode, result.stdout.strip()

    rc, commit = run("rev-parse", "HEAD")
    if rc != 0 or not commit:
        return {"commit": "unknown", "branch": "unknown",
                "dirty": False, "code_state": "unknown"}

    _, branch = run("rev-parse", "--abbrev-ref", "HEAD")
    _, status = run("status", "--porcelain")
    dirty = bool(status)
    return {"commit": commit, "branch": branch,
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
