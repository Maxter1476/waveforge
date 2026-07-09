import numpy as np
import pytest

from waveforge import (
    biquad_highpass,
    biquad_lowpass,
    fft,
    one_pole_lowpass,
    rfft_freqs,
    sine,
)


def measured_response(biquad, sample_rate, n=8192):
    """Frequency response from the FFT of the impulse response."""
    h = biquad.impulse_response(n)
    spectrum = fft(h)[: n // 2 + 1]
    freqs = rfft_freqs(n, sample_rate)
    return freqs, spectrum


@pytest.mark.parametrize("cutoff", [500, 1000, 4000])
def test_lowpass_measured_matches_analytic(cutoff):
    """The impulse-response FFT must match the closed-form transfer function."""
    sr = 44100
    bq = biquad_lowpass(cutoff, sr)
    freqs, measured = measured_response(bq, sr)
    analytic = bq.frequency_response(freqs, sr)
    assert np.allclose(measured, analytic, atol=1e-3)


def test_lowpass_minus3db_at_cutoff():
    """A Butterworth (Q=1/sqrt2) lowpass is -3 dB at its cutoff frequency."""
    sr = 44100
    cutoff = 2000
    bq = biquad_lowpass(cutoff, sr)
    h = np.abs(bq.frequency_response(np.array([cutoff]), sr))[0]
    db = 20 * np.log10(h)
    assert db == pytest.approx(-3.0, abs=0.2)


def test_lowpass_passes_low_blocks_high():
    sr = 44100
    bq = biquad_lowpass(1000, sr)
    low = sine(100, 8192, sr)
    high = sine(10000, 8192, sr)
    low_out = bq.process(low)
    high_out = bq.process(high)
    # compare steady-state RMS (skip the transient)
    low_rms = np.sqrt(np.mean(low_out[2000:] ** 2))
    high_rms = np.sqrt(np.mean(high_out[2000:] ** 2))
    assert low_rms > 0.6
    assert high_rms < 0.1


def test_highpass_blocks_low_passes_high():
    sr = 44100
    bq = biquad_highpass(1000, sr)
    low_out = bq.process(sine(100, 8192, sr))
    high_out = bq.process(sine(10000, 8192, sr))
    assert np.sqrt(np.mean(low_out[2000:] ** 2)) < 0.1
    assert np.sqrt(np.mean(high_out[2000:] ** 2)) > 0.6


def test_dc_gain_of_lowpass_is_unity():
    """At DC (0 Hz) a lowpass should pass with gain 1."""
    sr = 44100
    for cutoff in (500, 2000, 8000):
        bq = biquad_lowpass(cutoff, sr)
        dc = np.abs(bq.frequency_response(np.array([0.0]), sr))[0]
        assert dc == pytest.approx(1.0, abs=1e-6)


def test_one_pole_lowpass_dc_unity():
    sr = 44100
    bq = one_pole_lowpass(1000, sr)
    dc = np.abs(bq.frequency_response(np.array([0.0]), sr))[0]
    assert dc == pytest.approx(1.0, abs=1e-6)


def test_highpass_blocks_dc():
    sr = 44100
    bq = biquad_highpass(1000, sr)
    dc = np.abs(bq.frequency_response(np.array([0.0]), sr))[0]
    assert dc == pytest.approx(0.0, abs=1e-6)
