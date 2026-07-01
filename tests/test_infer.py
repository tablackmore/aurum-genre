import numpy as np
from aurum_genre.infer import average_chunk_probs, _chunks
from aurum_genre.mel import CHUNK_SAMPLES


def test_average_chunk_probs_is_mean_over_chunks():
    probs = np.array([[0.2, 0.8], [0.4, 0.6], [0.6, 0.4]])
    out = average_chunk_probs(probs)
    assert np.allclose(out, [0.4, 0.6])


def test_chunks_short_yields_one_padded_chunk():
    """Array shorter than CHUNK_SAMPLES → one zero-padded chunk of shape [1, CHUNK_SAMPLES]."""
    short = np.ones((1, CHUNK_SAMPLES // 2), dtype=np.float32)
    chunks = list(_chunks(short))
    assert len(chunks) == 1
    assert chunks[0].shape == (1, CHUNK_SAMPLES)
    # trailing half must be zeros (padded)
    assert np.all(chunks[0][0, CHUNK_SAMPLES // 2:] == 0.0)


def test_chunks_exact_length_yields_one_chunk():
    """Array of exactly CHUNK_SAMPLES → one chunk, no padding."""
    exact = np.ones((1, CHUNK_SAMPLES), dtype=np.float32)
    chunks = list(_chunks(exact))
    assert len(chunks) == 1
    assert chunks[0].shape == (1, CHUNK_SAMPLES)


def test_chunks_longer_yields_multiple_chunks():
    """Array of 2*CHUNK_SAMPLES → at least two chunks."""
    long = np.ones((1, 2 * CHUNK_SAMPLES), dtype=np.float32)
    chunks = list(_chunks(long))
    assert len(chunks) >= 2
    for ch in chunks:
        assert ch.shape == (1, CHUNK_SAMPLES)


def test_chunks_2d_input_shape_preserved():
    """2-D [1, N] input: chunks retain the leading batch dim."""
    wav = np.ones((1, CHUNK_SAMPLES + 100), dtype=np.float32)
    for ch in _chunks(wav):
        assert ch.ndim == 2
        assert ch.shape[0] == 1
