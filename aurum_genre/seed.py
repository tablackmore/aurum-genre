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
