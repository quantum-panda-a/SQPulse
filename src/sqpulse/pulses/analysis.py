"""Spectral and time-domain analysis tools for pulses in SI units."""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from .base import Pulse


def spectral_leakage(
    pulse: Pulse,
    cutoff_freq: float,
    dt: Optional[float] = None,
) -> float:
    r"""Calculate the ratio of spectral energy lying outside a given cutoff frequency.

    .. math::
        \eta = \frac{\int_{|f| > f_{\text{cutoff}}} |S(f)|^2 df}{\int_{-\infty}^{+\infty} |S(f)|^2 df}

    Args:
        pulse: Pulse object to analyze.
        cutoff_freq: Cutoff frequency in Hz (positive).
        dt: Sampling time in seconds (s). If None, defaults to pulse's adaptive step.

    Returns:
        Fraction of spectral energy outside [-cutoff_freq, +cutoff_freq] (in [0, 1]).
    """
    cutoff_freq = float(cutoff_freq)
    if cutoff_freq <= 0:
        raise ValueError(f"cutoff_freq must be positive, got {cutoff_freq} Hz")
    if cutoff_freq < 1e3 and pulse.duration < 1e-3:
        import warnings
        warnings.warn(
            f"cutoff_freq={cutoff_freq} Hz is extremely low for pulse duration {pulse.duration:.2e} s. "
            f"Note that all frequency parameters in SQPulse are in Hz (e.g. 100e6 for 100 MHz).",
            UserWarning,
            stacklevel=2,
        )

    freqs, spec, _ = pulse.fft(dt=dt, pad_factor=8)
    psd = np.abs(spec) ** 2
    total_energy = np.sum(psd)
    if total_energy == 0:
        return 0.0

    outside_mask = np.abs(freqs) > cutoff_freq
    outside_energy = np.sum(psd[outside_mask])
    return float(outside_energy / total_energy)


def compare_pulses(
    pulses: List[Pulse],
    domain: str = "both",
    dt: Optional[float] = None,
    freq_range: Optional[Tuple[float, float]] = None,
    log_scale: bool = True,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """Plot multiple pulses together to compare their time and frequency features.

    Args:
        pulses: List of Pulse objects.
        domain: 'time', 'freq', or 'both'.
        dt: Sampling time in seconds (s). If None, auto-selected based on shortest pulse.
        freq_range: Frequency range tuple (f_min, f_max) in Hz.
        log_scale: Whether to plot frequency spectrum in dB.
        figsize: Figure size tuple.

    Returns:
        matplotlib Figure.
    """
    if not pulses:
        raise ValueError("pulses list cannot be empty")

    min_dur = min(p.duration for p in pulses)
    if dt is not None:
        dt = float(dt)
        if dt <= 0:
            raise ValueError(f"Sampling step dt must be positive, got {dt} s")
        if dt >= min_dur:
            hint = ""
            if dt > 1e-3 and min_dur < 1e-3:
                hint = f" Did you mean dt={dt}e-9 s ({dt} ns)? All time parameters in SQPulse are in SI units (seconds)."
            raise ValueError(
                f"Sampling interval dt={dt} s cannot be greater than or equal to shortest pulse duration={min_dur} s.{hint}"
            )
    else:
        dt = min(min_dur / 200, 1e-10)

    if domain == "time":
        fig, ax = plt.subplots(figsize=figsize or (8, 5))
        for p in pulses:
            t, wave = p.sample(dt=dt)
            ax.plot(t, wave.real, lw=2, label=f"{p.name} (I)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Pulse Time-Domain Comparison")
        ax.grid(True, alpha=0.3)
        ax.legend()
        return fig

    elif domain == "freq":
        fig, ax = plt.subplots(figsize=figsize or (8, 5))
        for p in pulses:
            freqs, spec, psd_dB = p.fft(dt=dt)
            if log_scale:
                ax.plot(freqs, psd_dB, lw=2, label=p.name)
            else:
                mag = np.abs(spec)
                norm_mag = mag / np.max(mag) if np.max(mag) > 0 else mag
                ax.plot(freqs, norm_mag, lw=2, label=p.name)
        if log_scale:
            ax.set_ylabel("Power Spectral Density (dB)")
            ax.set_ylim(-80, 5)
        else:
            ax.set_ylabel("Normalized Magnitude")
            ax.set_ylim(0, 1.05)
        if freq_range:
            ax.set_xlim(freq_range)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title("Pulse Frequency-Domain Comparison")
        ax.grid(True, alpha=0.3)
        ax.legend()
        return fig

    elif domain == "both":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize or (14, 5))
        for p in pulses:
            t, wave = p.sample(dt=dt)
            ax1.plot(t, wave.real, lw=2, label=f"{p.name}")
            freqs, spec, psd_dB = p.fft(dt=dt)
            if log_scale:
                ax2.plot(freqs, psd_dB, lw=2, label=p.name)
            else:
                mag = np.abs(spec)
                norm_mag = mag / np.max(mag) if np.max(mag) > 0 else mag
                ax2.plot(freqs, norm_mag, lw=2, label=p.name)

        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("In-Phase Amplitude (I)")
        ax1.set_title("Time-Domain Envelopes")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        if log_scale:
            ax2.set_ylabel("Power Spectral Density (dB)")
            ax2.set_ylim(-80, 5)
        else:
            ax2.set_ylabel("Normalized Magnitude")
            ax2.set_ylim(0, 1.05)
        if freq_range:
            ax2.set_xlim(freq_range)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_title("Frequency Spectra (Leakage Comparison)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        plt.tight_layout()
        return fig
    else:
        raise ValueError(f"Unknown domain: {domain}")
