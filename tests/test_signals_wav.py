import numpy as np
import pytest

from waveforge import (
    chirp,
    fft,
    hann_window,
    read_wav,
    sawtooth,
    sine,
    square,
    white_noise,
    write_wav,
)


def test_sine_frequency_and_amplitude():
    sr = 8000
    x = sine(440, sr, sr, amplitude=0.5)  # one second
    assert np.max(np.abs(x)) == pytest.approx(0.5, abs=1e-3)
    # dominant spectral bin corresponds to 440 Hz
    mag = np.abs(fft(x))[: sr // 2]
    assert int(np.argmax(mag)) == 440


def test_sawtooth_range_and_shape():
    x = sawtooth(100, 1000, 8000)
    assert x.min() >= -1.0 and x.max() <= 1.0
    # sawtooth is monotonically rising within a period
    period = 80  # 8000/100
    seg = x[1 : period - 1]
    assert np.all(np.diff(seg) > 0)


def test_square_is_strictly_bilevel():
    """Every sample is exactly +1 or -1 (never 0, even at zero crossings)."""
    x = square(50, 800, 8000)
    assert set(np.unique(x)).issubset({-1.0, 1.0})
    assert np.all(np.abs(np.abs(x) - 1.0) < 1e-12)


def test_chirp_sweeps_upward():
    """A chirp's instantaneous frequency should rise: the spectral centroid of
    the second half exceeds that of the first half."""
    sr = 16000
    n = 16000
    x = chirp(200, 4000, n, sr)

    def centroid(seg):
        mag = np.abs(fft(seg))[: len(seg) // 2]
        freqs = np.arange(len(mag))
        return np.sum(freqs * mag) / np.sum(mag)

    assert centroid(x[n // 2 :]) > centroid(x[: n // 2])


def test_hann_window_endpoints():
    w = hann_window(1024)
    assert w[0] == pytest.approx(0.0, abs=1e-12)
    assert w.max() == pytest.approx(1.0, abs=1e-3)


def test_white_noise_reproducible_and_bounded():
    a = white_noise(1000, seed=1)
    b = white_noise(1000, seed=1)
    assert np.array_equal(a, b)
    assert np.max(np.abs(a)) <= 1.0


def test_wav_roundtrip(tmp_path):
    """Write then read must recover the signal to within one 16-bit step."""
    sr = 22050
    x = 0.8 * sine(330, sr, sr)
    path = tmp_path / "tone.wav"
    write_wav(path, x, sr)
    y, sr_read = read_wav(path)
    assert sr_read == sr
    assert len(y) == len(x)
    assert np.max(np.abs(y - x)) < 1.0 / 32767 + 1e-9


def test_wav_clips_out_of_range(tmp_path):
    sr = 8000
    loud = 2.0 * sine(200, sr, sr)  # exceeds [-1, 1]
    path = tmp_path / "loud.wav"
    write_wav(path, loud, sr)
    y, _ = read_wav(path)
    assert np.max(np.abs(y)) <= 1.0
