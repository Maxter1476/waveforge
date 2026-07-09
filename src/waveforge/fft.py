"""Radix-2 Cooley-Tukey FFT and inverse, implemented from scratch.

The transform is the discrete Fourier transform

    X[k] = sum_{n=0}^{N-1} x[n] exp(-2 pi i k n / N).

A direct evaluation is O(N^2); the Cooley-Tukey factorization recursively
splits even- and odd-indexed samples to reach O(N log N). This module keeps a
readable recursive radix-2 core (power-of-two lengths) plus a Bluestein
fallback so *any* length works, and the test suite pins both against the naive
DFT and against NumPy.
"""

from __future__ import annotations

import numpy as np

__all__ = ["fft", "ifft", "dft_naive", "rfft_freqs"]


def dft_naive(x: np.ndarray) -> np.ndarray:
    """Direct O(N^2) DFT — the ground-truth reference for the fast paths."""
    x = np.asarray(x, dtype=complex)
    n = len(x)
    k = np.arange(n)
    matrix = np.exp(-2j * np.pi * np.outer(k, k) / n)
    return matrix @ x


def _fft_radix2(x: np.ndarray) -> np.ndarray:
    n = len(x)
    if n == 1:
        return x
    even = _fft_radix2(x[0::2])
    odd = _fft_radix2(x[1::2])
    twiddle = np.exp(-2j * np.pi * np.arange(n // 2) / n)
    t = twiddle * odd
    return np.concatenate([even + t, even - t])


def _bluestein(x: np.ndarray) -> np.ndarray:
    """Bluestein's chirp-z algorithm for arbitrary (non-power-of-two) N."""
    n = len(x)
    m = 1 << (2 * n - 1).bit_length()  # next power of two >= 2N-1
    k = np.arange(n)
    chirp = np.exp(-1j * np.pi * k * k / n)
    a = np.zeros(m, dtype=complex)
    a[:n] = x * chirp
    b = np.zeros(m, dtype=complex)
    b[:n] = np.conj(chirp)
    b[-(n - 1):] = np.conj(chirp[1:][::-1])
    conv = _ifft_pow2(_fft_radix2(a) * _fft_radix2(b))
    return conv[:n] * chirp


def _ifft_pow2(x: np.ndarray) -> np.ndarray:
    n = len(x)
    return np.conj(_fft_radix2(np.conj(x))) / n


def fft(x: np.ndarray) -> np.ndarray:
    """Fast Fourier transform of a 1-D signal of any length."""
    x = np.asarray(x, dtype=complex)
    n = len(x)
    if n == 0:
        raise ValueError("input must be non-empty")
    if n & (n - 1) == 0:  # power of two
        return _fft_radix2(x)
    return _bluestein(x)


def ifft(x: np.ndarray) -> np.ndarray:
    """Inverse FFT; ``ifft(fft(x)) == x`` up to floating-point error."""
    x = np.asarray(x, dtype=complex)
    n = len(x)
    if n == 0:
        raise ValueError("input must be non-empty")
    return np.conj(fft(np.conj(x))) / n


def rfft_freqs(n: int, sample_rate: float) -> np.ndarray:
    """Frequencies (Hz) of the first half of an N-point spectrum."""
    return np.arange(n // 2 + 1) * sample_rate / n
