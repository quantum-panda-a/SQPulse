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


def extract_dynamic_s21(
    times: np.ndarray,
    v_in: np.ndarray,
    v_out: np.ndarray,
    weights: Optional[np.ndarray] = None,
    integration_window: Optional[Tuple[float, float]] = None,
) -> complex:
    r"""Extract complex S_21 transmission coefficient from time-domain input and output microwave signals.

    The dynamic transmission scattering parameter is evaluated as the normalized ratio of
    integrated transmitted field to integrated reference drive:
    .. math::
        S_{21}^{\text{dyn}} = \frac{\int_{W} V_{\text{out}}(t) w^*(t) dt}{\int_{W} V_{\text{in}}(t) w^*(t) dt}

    Args:
        times: 1D array of timestamps in seconds.
        v_in: 1D complex array of input drive signal.
        v_out: 1D complex array of output transmitted signal.
        weights: Optional complex temporal weight function w(t). Defaults to uniform boxcar weighting.
        integration_window: Optional (t_start, t_end) window in seconds.
            If None, automatically selects the steady-state post-ringup region where |v_in| >= 90% peak.

    Returns:
        Complex S_21 scalar representing the dynamic transmission scattering parameter.
    """
    times = np.asarray(times, dtype=float)
    v_in = np.asarray(v_in, dtype=complex)
    v_out = np.asarray(v_out, dtype=complex)

    if len(times) == 0:
        return 0.0 + 0.0j

    if integration_window is not None:
        t_start, t_end = integration_window
        mask = (times >= t_start) & (times <= t_end)
    else:
        # Auto-detect steady/active flat region:
        # Where |v_in(t)| is >= 90% of peak amplitude (the flat top region)
        peak_amp = float(np.max(np.abs(v_in)))
        if peak_amp > 1e-12:
            flat_mask = np.abs(v_in) >= 0.90 * peak_amp
            flat_indices = np.where(flat_mask)[0]
            if len(flat_indices) > 4:
                # Use the second half of the flat region (after cavity ring-up has settled)
                mid_idx = flat_indices[len(flat_indices) // 2]
                mask = np.zeros_like(times, dtype=bool)
                mask[mid_idx : flat_indices[-1] + 1] = True
            else:
                mask = flat_mask
        else:
            mask = np.ones_like(times, dtype=bool)

    if not np.any(mask):
        mask = np.ones_like(times, dtype=bool)

    t_sub = times[mask]
    vin_sub = v_in[mask]
    vout_sub = v_out[mask]

    if weights is not None:
        w_sub = np.asarray(weights, dtype=complex)[mask]
    else:
        w_sub = np.ones_like(t_sub, dtype=complex)

    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    int_in = trapz_fn(vin_sub * np.conj(w_sub), t_sub)
    int_out = trapz_fn(vout_sub * np.conj(w_sub), t_sub)

    if abs(int_in) < 1e-15:
        return 0.0 + 0.0j

    return complex(int_out / int_in)

