import numpy as np
from aurum_genre.infer import average_chunk_probs


def test_average_chunk_probs_is_mean_over_chunks():
    probs = np.array([[0.2, 0.8], [0.4, 0.6], [0.6, 0.4]])
    out = average_chunk_probs(probs)
    assert np.allclose(out, [0.4, 0.6])
