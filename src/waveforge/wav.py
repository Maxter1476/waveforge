"""Minimal WAV (PCM) reader and writer using only the stdlib ``wave`` module.

Signals are float arrays in [-1, 1]; writing quantizes to 16-bit PCM and
reading dequantizes back. The round-trip is exact to within one quantization
step, which the test suite verifies.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

__all__ = ["write_wav", "read_wav"]

_INT16_MAX = 32767


def write_wav(path: str | Path, signal: np.ndarray, sample_rate: int) -> None:
    """Write a mono float signal (clipped to [-1, 1]) as 16-bit PCM."""
    signal = np.clip(np.asarray(signal, dtype=float), -1.0, 1.0)
    pcm = np.round(signal * _INT16_MAX).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a mono 16-bit PCM WAV into a float array and its sample rate."""
    with wave.open(str(path), "rb") as r:
        if r.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM is supported")
        n_frames = r.getnframes()
        sample_rate = r.getframerate()
        raw = r.readframes(n_frames)
    pcm = np.frombuffer(raw, dtype="<i2")
    if r.getnchannels() == 2:  # pragma: no cover - downmix if stereo
        pcm = pcm.reshape(-1, 2).mean(axis=1)
    return pcm.astype(float) / _INT16_MAX, sample_rate
