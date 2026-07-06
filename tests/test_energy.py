import numpy as np
from aurum_genre.energy import energy


def test_range_and_silence():
    assert energy(np.zeros(16000)) < 0.05                 # silence ~ 0
    v = energy(0.3 * np.random.RandomState(0).randn(16000))
    assert 0.0 <= v <= 1.0
    assert energy(np.zeros(0)) == 0.0                     # empty is safe


def test_louder_is_higher():
    base = np.random.RandomState(0).randn(16000)
    assert energy(0.05 * base) < energy(0.5 * base)


def test_busy_broadband_higher_than_steady_tone():
    t = np.arange(16000) / 16000
    tone = 0.3 * np.sin(2 * np.pi * 220 * t)              # calm, steady
    noise = 0.3 * np.random.RandomState(1).randn(16000)   # busy, broadband
    assert energy(noise) > energy(tone)
