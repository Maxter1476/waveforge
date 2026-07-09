"""waveforge — a from-scratch DSP toolkit: FFT, oscillators, filters, WAV I/O."""

from .fft import dft_naive, fft, ifft, rfft_freqs
from .filters import Biquad, biquad_highpass, biquad_lowpass, one_pole_lowpass
from .signals import (
    chirp,
    hamming_window,
    hann_window,
    sawtooth,
    sine,
    square,
    white_noise,
)
from .wav import read_wav, write_wav

__all__ = [
    "Biquad",
    "biquad_highpass",
    "biquad_lowpass",
    "chirp",
    "dft_naive",
    "fft",
    "hamming_window",
    "hann_window",
    "ifft",
    "one_pole_lowpass",
    "read_wav",
    "rfft_freqs",
    "sawtooth",
    "sine",
    "square",
    "white_noise",
    "write_wav",
]

__version__ = "0.1.0"
