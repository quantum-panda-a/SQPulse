"""Digital down-conversion and IQ integration for dispersive readout in SI units."""

from __future__ import annotations
from typing import Optional
import numpy as np


def demodulate_and_integrate(
    times: np.ndarray,
    alpha_t: np.ndarray,
    kappa_ext: float,
    weights: Optional[np.ndarray] = None,
) -> complex:
    r"""Demodulate and integrate cavity output field into a complex IQ point.

    The transmitted signal emitted into the readout line is:
    .. math::
        s_{\text{out}}(t) = \sqrt{\kappa_{\text{ext}}} \alpha(t)

    The integrated baseband IQ signal is:
    .. math::
        S = I + i Q = \frac{1}{T_{\text{meas}}} \int_0^{T_{\text{meas}}} s_{\text{out}}(t) w^*(t) dt

    Args:
        times: 1D array of timestamps in seconds.
        alpha_t: 1D complex array of cavity field amplitude alpha(t).
        kappa_ext: External coupling rate in rad/s.
        weights: Optional complex temporal weight function w(t). Defaults to uniform boxcar weighting.

    Returns:
        Complex baseband scalar S = I + i Q.
    """
    times = np.asarray(times)
    alpha_t = np.asarray(alpha_t, dtype=complex)
    s_out = np.sqrt(kappa_ext) * alpha_t

    t_total = times[-1] - times[0]
    if t_total <= 0:
        return 0.0 + 0.0j

    if weights is None:
        weights = np.ones_like(times, dtype=complex)
    else:
        weights = np.asarray(weights, dtype=complex)

    integrand = s_out * np.conj(weights)
    # Numerical trapezoidal integration
    integrated = np.trapezoid(integrand, times) if hasattr(np, "trapezoid") else np.trapz(integrand, times)
    return complex(integrated / t_total)
