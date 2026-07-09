import numpy as np
import pytest

from waveforge import dft_naive, fft, ifft, sine


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 64, 256, 1024])
def test_fft_matches_naive_dft(n):
    real = np.random.default_rng(n).standard_normal(n)
    imag = np.random.default_rng(n + 1).standard_normal(n)
    x = real + 1j * imag
    assert np.allclose(fft(x), dft_naive(x), atol=1e-9)


@pytest.mark.parametrize("n", [3, 5, 7, 12, 100, 101, 250])
def test_bluestein_arbitrary_length_matches_naive(n):
    """Non-power-of-two lengths (Bluestein path) must match the naive DFT."""
    x = np.random.default_rng(n).standard_normal(n)
    assert np.allclose(fft(x), dft_naive(x), atol=1e-8)


@pytest.mark.parametrize("n", [8, 64, 256, 100, 333])
def test_matches_numpy_fft(n):
    x = np.random.default_rng(n).standard_normal(n)
    assert np.allclose(fft(x), np.fft.fft(x), atol=1e-8)


@pytest.mark.parametrize("n", [4, 16, 128, 30, 127])
def test_inverse_roundtrip(n):
    x = np.random.default_rng(n).standard_normal(n)
    assert np.allclose(ifft(fft(x)), x, atol=1e-9)


def test_parseval_theorem():
    """Energy is conserved: sum|x|^2 == (1/N) sum|X|^2."""
    x = np.random.default_rng(0).standard_normal(512)
    X = fft(x)
    assert np.sum(np.abs(x) ** 2) == pytest.approx(np.sum(np.abs(X) ** 2) / len(x), rel=1e-9)


def test_pure_tone_has_single_peak():
    """A sine at a bin-centered frequency shows one sharp spectral peak."""
    sr = 8000
    n = 1024
    freq = sr * 64 / n  # exactly bin 64
    x = sine(freq, n, sr)
    mag = np.abs(fft(x))[: n // 2]
    peak = int(np.argmax(mag))
    assert peak == 64
    # energy concentrated: the peak bin dominates its neighbours
    assert mag[64] > 100 * mag[100]


def test_linearity():
    a = np.random.default_rng(1).standard_normal(128)
    b = np.random.default_rng(2).standard_normal(128)
    assert np.allclose(fft(3 * a + 2 * b), 3 * fft(a) + 2 * fft(b), atol=1e-9)


def test_empty_rejected():
    with pytest.raises(ValueError):
        fft(np.array([]))
