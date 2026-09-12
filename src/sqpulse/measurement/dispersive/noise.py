"""Noise injection models for dispersive readout in SQPulse."""

from __future__ import annotations
from typing import Optional
import numpy as np


def add_readout_noise(
    iq_points: np.ndarray,
    snr_db: float = 12.0,
    signal_separation: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    r"""Add realistic Gaussian amplifier and quantum shot noise to IQ data points.

    In circuit QED homodyne/heterodyne detection:
    .. math::
        \text{SNR} = \frac{|\Delta S|^2}{2 \sigma^2} = 10^{\text{SNR}_{\text{dB}} / 10}
        \sigma = \frac{|\Delta S|}{\sqrt{2 \cdot 10^{\text{SNR}_{\text{dB}} / 10}}}

    Args:
        iq_points: 1D or ND complex array of noiseless IQ values.
        snr_db: Signal-to-noise ratio in decibels (dB). Default 12 dB.
        signal_separation: Distance |S_1 - S_0| in the IQ plane between state |0> and |1>.
        seed: Optional random seed for reproducible sampling.

    Returns:
        Complex array of same shape with added Gaussian noise.
    """
    iq_points = np.asarray(iq_points, dtype=complex)
    linear_snr = 10.0 ** (float(snr_db) / 10.0)
    sigma = signal_separation / np.sqrt(2.0 * max(1e-6, linear_snr))

    rng = np.random.default_rng(seed)
    noise_i = rng.normal(0.0, sigma, size=iq_points.shape)
    noise_q = rng.normal(0.0, sigma, size=iq_points.shape)

    return iq_points + (noise_i + 1j * noise_q)
