"""README figure: a chirp spectrogram + biquad filter responses, all from the
waveforge FFT (no scipy.signal)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from waveforge import biquad_highpass, biquad_lowpass, chirp, fft, hann_window, rfft_freqs


def spectrogram(signal, sample_rate, win=512, hop=128):
    window = hann_window(win)
    frames = []
    for start in range(0, len(signal) - win, hop):
        seg = signal[start : start + win] * window
        frames.append(np.abs(fft(seg))[: win // 2 + 1])
    spec = np.array(frames).T
    times = np.arange(len(frames)) * hop / sample_rate
    freqs = rfft_freqs(win, sample_rate)
    return times, freqs, 20 * np.log10(spec + 1e-6)


def main() -> None:
    sr = 16000
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    # left: spectrogram of an ascending chirp
    sig = chirp(200, 6000, sr * 2, sr)
    t, f, s = spectrogram(sig, sr)
    ax1.pcolormesh(t, f, s, shading="auto", cmap="magma")
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("frequency [Hz]")
    ax1.set_title("Chirp spectrogram (own FFT + Hann STFT)")

    # right: measured vs analytic filter responses
    freqs = np.logspace(1, np.log10(sr / 2), 500)
    for name, bq, color in [
        ("lowpass 1 kHz", biquad_lowpass(1000, sr), "#2166ac"),
        ("highpass 2 kHz", biquad_highpass(2000, sr), "#b2182b"),
    ]:
        analytic = 20 * np.log10(np.abs(bq.frequency_response(freqs, sr)) + 1e-9)
        h = bq.impulse_response(8192)
        meas_f = rfft_freqs(8192, sr)
        meas = 20 * np.log10(np.abs(fft(h)[: 8192 // 2 + 1]) + 1e-9)
        ax2.semilogx(freqs, analytic, color=color, lw=2, label=f"{name} (analytic)")
        ax2.semilogx(meas_f, meas, color=color, lw=0.8, ls="--", alpha=0.8)
    ax2.axhline(-3, color="gray", ls=":", lw=0.8, label="-3 dB")
    ax2.set_ylim(-48, 6)
    ax2.set_xlabel("frequency [Hz]")
    ax2.set_ylabel("gain [dB]")
    ax2.set_title("Biquad responses: measured (dashed) = analytic (solid)")
    ax2.legend(fontsize=8)

    fig.suptitle("waveforge — DSP from scratch, validated against analytic DSP")
    fig.tight_layout()
    fig.savefig("docs/figures/showcase.png", dpi=150)
    print("wrote docs/figures/showcase.png")


if __name__ == "__main__":
    main()
