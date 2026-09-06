"""Spectral and time-domain analysis tools for pulses."""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from .base import Pulse


def spectral_leakage(
    pulse: Pulse,
    cutoff_freq: float,
    dt: float = 0.05,
) -> float:
    r"""Calculate the ratio of spectral energy lying outside a given cutoff frequency.

    .. math::
        \eta = \frac{\int_{|f| > f_{\text{cutoff}}} |S(f)|^2 df}{\int_{-\infty}^{+\infty} |S(f)|^2 df}

    Args:
        pulse: Pulse object to analyze.
        cutoff_freq: Cutoff frequency in GHz (positive).
        dt: Sampling time in ns.

    Returns:
        Fraction of spectral energy outside [-cutoff_freq, +cutoff_freq] (in [0, 1]).
    """
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
    dt: float = 0.1,
    freq_range: Optional[Tuple[float, float]] = None,
    log_scale: bool = True,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """Plot multiple pulses together to compare their time and frequency features.

    Args:
        pulses: List of Pulse objects.
        domain: 'time', 'freq', or 'both'.
        dt: Sampling time in ns.
        freq_range: Frequency range tuple (f_min, f_max) in GHz.
        log_scale: Whether to plot frequency spectrum in dB.
        figsize: Figure size tuple.

    Returns:
        matplotlib Figure.
    """
    if domain == "time":
        fig, ax = plt.subplots(figsize=figsize or (8, 5))
        for p in pulses:
            t, wave = p.sample(dt=dt)
            ax.plot(t, wave.real, lw=2, label=f"{p.name} (I)")
        ax.set_xlabel("Time (ns)")
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
        ax.set_xlabel("Frequency (GHz)")
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

        ax1.set_xlabel("Time (ns)")
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
        ax2.set_xlabel("Frequency (GHz)")
        ax2.set_title("Frequency Spectra (Leakage Comparison)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        plt.tight_layout()
        return fig
    else:
        raise ValueError(f"Unknown domain: {domain}")
