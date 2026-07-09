"""Oscillators, windows, and basic signal utilities."""

from __future__ import annotations

import numpy as np

__all__ = [
    "sine",
    "square",
    "sawtooth",
    "chirp",
    "white_noise",
    "hann_window",
    "hamming_window",
]


def _t(n: int, sample_rate: float) -> np.ndarray:
    return np.arange(n) / sample_rate


def sine(freq: float, n: int, sample_rate: float, *, amplitude: float = 1.0, phase: float = 0.0):
    """A pure sine tone."""
    return amplitude * np.sin(2 * np.pi * freq * _t(n, sample_rate) + phase)


def square(freq: float, n: int, sample_rate: float, *, amplitude: float = 1.0):
    """A square wave (band-unlimited; will alias if freq is high).

    Strictly bilevel: the first half of each cycle is +amplitude, the second
    -amplitude, with the zero-crossing sample assigned to +amplitude (so the
    output is never 0, unlike ``sign(sin(...))``).
    """
    phase = (freq * _t(n, sample_rate)) % 1.0
    return amplitude * np.where(phase < 0.5, 1.0, -1.0)


def sawtooth(freq: float, n: int, sample_rate: float, *, amplitude: float = 1.0):
    """A rising sawtooth in [-amplitude, amplitude]."""
    phase = (freq * _t(n, sample_rate)) % 1.0
    return amplitude * (2.0 * phase - 1.0)


def chirp(f0: float, f1: float, n: int, sample_rate: float, *, amplitude: float = 1.0):
    """A linear frequency sweep from f0 to f1 over the buffer."""
    t = _t(n, sample_rate)
    duration = n / sample_rate
    inst_phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / duration * t**2)
    return amplitude * np.sin(inst_phase)


def white_noise(n: int, *, amplitude: float = 1.0, seed: int | None = None):
    """Uniform white noise in [-amplitude, amplitude]."""
    rng = np.random.default_rng(seed)
    return amplitude * (2.0 * rng.random(n) - 1.0)


def hann_window(n: int) -> np.ndarray:
    """Periodic Hann window (good general-purpose spectral window)."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def hamming_window(n: int) -> np.ndarray:
    """Periodic Hamming window."""
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / n)
