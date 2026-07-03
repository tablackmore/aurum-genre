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
