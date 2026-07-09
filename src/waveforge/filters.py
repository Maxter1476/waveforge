"""First- and second-order IIR filters with known analytic responses.

Each filter is a biquad (or one-pole) with coefficients from the standard
Audio-EQ-Cookbook / RBJ formulas. Because the transfer function is known in
closed form, the test suite compares the *measured* frequency response (from
FFT of the impulse response) against the analytic H(e^{jw}) — a filter that is
even slightly miscoefficiented fails immediately.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Biquad", "one_pole_lowpass", "biquad_lowpass", "biquad_highpass"]


class Biquad:
    """A direct-form-I biquad: y = b0 x + b1 x1 + b2 x2 - a1 y1 - a2 y2.

    Coefficients are normalized so a0 = 1.
    """

    def __init__(self, b: tuple[float, float, float], a: tuple[float, float, float]) -> None:
        a0 = a[0]
        self.b = np.array(b) / a0
        self.a = np.array(a) / a0

    def process(self, x: np.ndarray) -> np.ndarray:
        """Filter a signal (transient starts from rest)."""
        x = np.asarray(x, dtype=float)
        y = np.zeros_like(x)
        x1 = x2 = y1 = y2 = 0.0
        b0, b1, b2 = self.b
        _, a1, a2 = self.a
        for i, xi in enumerate(x):
            yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            y[i] = yi
            x2, x1 = x1, xi
            y2, y1 = y1, yi
        return y

    def frequency_response(self, freqs: np.ndarray, sample_rate: float) -> np.ndarray:
        """Analytic H(e^{jw}) at the given frequencies (Hz)."""
        w = 2 * np.pi * np.asarray(freqs) / sample_rate
        z1 = np.exp(-1j * w)
        z2 = z1 * z1
        num = self.b[0] + self.b[1] * z1 + self.b[2] * z2
        den = self.a[0] + self.a[1] * z1 + self.a[2] * z2
        return num / den

    def impulse_response(self, n: int) -> np.ndarray:
        impulse = np.zeros(n)
        impulse[0] = 1.0
        return self.process(impulse)


def one_pole_lowpass(cutoff: float, sample_rate: float) -> Biquad:
    """One-pole lowpass as a degenerate biquad (b2 = a2 = 0)."""
    x = np.exp(-2.0 * np.pi * cutoff / sample_rate)
    return Biquad(b=(1.0 - x, 0.0, 0.0), a=(1.0, -x, 0.0))


def biquad_lowpass(cutoff: float, sample_rate: float, q: float = 0.7071) -> Biquad:
    """RBJ cookbook lowpass biquad."""
    w0 = 2 * np.pi * cutoff / sample_rate
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    b0 = (1 - cos_w0) / 2
    b1 = 1 - cos_w0
    b2 = (1 - cos_w0) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    return Biquad(b=(b0, b1, b2), a=(a0, a1, a2))


def biquad_highpass(cutoff: float, sample_rate: float, q: float = 0.7071) -> Biquad:
    """RBJ cookbook highpass biquad."""
    w0 = 2 * np.pi * cutoff / sample_rate
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    b0 = (1 + cos_w0) / 2
    b1 = -(1 + cos_w0)
    b2 = (1 + cos_w0) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    return Biquad(b=(b0, b1, b2), a=(a0, a1, a2))
