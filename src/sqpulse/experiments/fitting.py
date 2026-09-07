"""Fitting routines for experimental calibrations in SQPulse."""

from __future__ import annotations
from typing import Tuple, Dict, Any, Callable
import numpy as np
from scipy.optimize import curve_fit


def fit_sine(
    xs: np.ndarray,
    ys: np.ndarray,
) -> Dict[str, Any]:
    """Fit data to sinusoidal oscillation: y = y0 - amp * cos(2*pi*f0*x + phi).

    Returns:
        dict containing 'f0', 'amp', 'y0', 'phi', 'fit_fn', 'y_fit', 'period'.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    # Initial parameter guesses using FFT
    n = len(xs)
    dx = np.mean(np.diff(xs))
    freqs = np.fft.rfftfreq(n, d=dx)
    fft_vals = np.abs(np.fft.rfft(ys - np.mean(ys)))
    # Exclude DC component
    if len(freqs) > 1:
        peak_idx = np.argmax(fft_vals[1:]) + 1
        f_guess = freqs[peak_idx]
    else:
        f_guess = 1.0 / (xs[-1] - xs[0])

    amp_guess = float(np.ptp(ys) / 2.0)
    y0_guess = float(np.mean(ys))

    def model(x, f0, amp, y0, phi):
        return y0 - amp * np.cos(2.0 * np.pi * f0 * x + phi)

    p0 = [f_guess, amp_guess, y0_guess, 0.0]
    try:
        popt, _ = curve_fit(model, xs, ys, p0=p0, maxfev=10000)
    except Exception:
        popt = p0

    f0_fit, amp_fit, y0_fit, phi_fit = popt
    f0_fit = abs(f0_fit)
    period = 1.0 / f0_fit if f0_fit > 0 else np.inf

    return {
        "f0": float(f0_fit),
        "amp": float(abs(amp_fit)),
        "y0": float(y0_fit),
        "phi": float(phi_fit),
        "period": float(period),
        "fit_fn": lambda x: model(x, *popt),
        "y_fit": model(xs, *popt),
    }


def fit_decay(
    xs: np.ndarray,
    ys: np.ndarray,
) -> Dict[str, Any]:
    """Fit data to single exponential relaxation: y = y_inf + amp * exp(-x / T).

    Returns:
        dict containing 'T', 'amp', 'y_inf', 'fit_fn', 'y_fit'.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    y_inf_guess = float(ys[-1])
    amp_guess = float(ys[0] - ys[-1])
    t_guess = float((xs[-1] - xs[0]) / 2.0)

    def model(x, T, amp, y_inf):
        return y_inf + amp * np.exp(-x / T)

    p0 = [max(t_guess, 1.0), amp_guess, y_inf_guess]
    bounds = ([1e-6, -10.0, -10.0], [np.inf, 10.0, 10.0])

    try:
        popt, _ = curve_fit(model, xs, ys, p0=p0, bounds=bounds, maxfev=10000)
    except Exception:
        popt = p0

    t_fit, amp_fit, y_inf_fit = popt

    return {
        "T": float(t_fit),
        "amp": float(amp_fit),
        "y_inf": float(y_inf_fit),
        "fit_fn": lambda x: model(x, *popt),
        "y_fit": model(xs, *popt),
    }


def fit_decaying_sine(
    xs: np.ndarray,
    ys: np.ndarray,
) -> Dict[str, Any]:
    """Fit Ramsey fringes to decaying sine: y = y0 + amp * exp(-x / T2) * cos(2*pi*f0*x + phi).

    Returns:
        dict containing 'T2', 'f0', 'amp', 'y0', 'phi', 'fit_fn', 'y_fit'.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    n = len(xs)
    dx = np.mean(np.diff(xs))
    freqs = np.fft.rfftfreq(n, d=dx)
    fft_vals = np.abs(np.fft.rfft(ys - np.mean(ys)))
    if len(freqs) > 1:
        peak_idx = np.argmax(fft_vals[1:]) + 1
        f_guess = freqs[peak_idx]
    else:
        f_guess = 1.0 / (xs[-1] - xs[0])

    amp_guess = float(np.ptp(ys) / 2.0)
    y0_guess = float(np.mean(ys))
    t2_guess = float((xs[-1] - xs[0]) / 2.0)

    def model(x, T2, f0, amp, y0, phi):
        return y0 + amp * np.exp(-x / T2) * np.cos(2.0 * np.pi * f0 * x + phi)

    p0 = [max(t2_guess, 1.0), f_guess, amp_guess, y0_guess, 0.0]
    bounds = ([1e-6, 0.0, 0.0, -10.0, -2 * np.pi], [np.inf, np.inf, 10.0, 10.0, 2 * np.pi])

    try:
        popt, _ = curve_fit(model, xs, ys, p0=p0, bounds=bounds, maxfev=10000)
    except Exception:
        popt = p0

    t2_fit, f0_fit, amp_fit, y0_fit, phi_fit = popt

    return {
        "T2": float(t2_fit),
        "f0": float(f0_fit),
        "amp": float(amp_fit),
        "y0": float(y0_fit),
        "phi": float(phi_fit),
        "fit_fn": lambda x: model(x, *popt),
        "y_fit": model(xs, *popt),
    }


def fit_lorentzian(
    xs: np.ndarray,
    ys: np.ndarray,
    center_guess: Optional[float] = None,
    fwhm_guess: Optional[float] = None,
) -> Dict[str, Any]:
    """Fit resonance spectrum data to a Lorentzian peak: y = y0 + amp / (1 + ((x - f0) / (fwhm / 2))^2).

    Args:
        xs: 1D array of frequencies in Hz.
        ys: 1D array of population or signal values.
        center_guess: Optional initial estimate for resonance frequency f0 in Hz.
        fwhm_guess: Optional initial estimate for full width at half maximum (FWHM) in Hz.

    Returns:
        dict containing 'f0', 'fwhm', 'amp', 'y0', 'fit_fn', 'y_fit', 'success'.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    f0_guess = float(center_guess if center_guess is not None else xs[np.argmax(ys)])
    y0_guess = float(np.percentile(ys, 10))
    amp_guess = float(np.max(ys) - y0_guess)
    if amp_guess <= 0:
        amp_guess = float(np.ptp(ys))

    if fwhm_guess is not None:
        gamma_guess = float(abs(fwhm_guess))
    else:
        half_max = y0_guess + 0.5 * amp_guess
        mask = ys >= half_max
        if np.sum(mask) >= 2:
            gamma_guess = float(xs[mask].max() - xs[mask].min())
        else:
            gamma_guess = float(abs(xs[-1] - xs[0]) / 20.0)
    gamma_guess = max(gamma_guess, abs(xs[1] - xs[0]) if len(xs) > 1 else 1e3)

    def model(x, f0, gamma, amp, y0):
        hwhm = gamma / 2.0
        return y0 + amp / (1.0 + ((x - f0) / hwhm) ** 2)

    p0 = [f0_guess, gamma_guess, amp_guess, y0_guess]
    x_min, x_max = min(xs[0], xs[-1]), max(xs[0], xs[-1])
    bounds = (
        [x_min, 1e-3, 0.0, -10.0],
        [x_max, x_max - x_min if x_max > x_min else 1e12, 10.0, 10.0],
    )

    success = True
    try:
        popt, _ = curve_fit(model, xs, ys, p0=p0, bounds=bounds, maxfev=10000)
    except Exception:
        popt = p0
        success = False

    f0_fit, gamma_fit, amp_fit, y0_fit = popt

    return {
        "f0": float(f0_fit),
        "fwhm": float(abs(gamma_fit)),
        "amp": float(amp_fit),
        "y0": float(y0_fit),
        "fit_fn": lambda x: model(x, *popt),
        "y_fit": model(xs, *popt),
        "success": success,
    }


def fit_spectroscopy_peaks(
    xs: np.ndarray,
    ys: np.ndarray,
    f01_guess: Optional[float] = None,
    f02_half_guess: Optional[float] = None,
    fwhm_guess: Optional[float] = None,
    p2_vals: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Fit qubit spectroscopy data for 0->1 peak and optional two-photon 0->2 peak.

    Uses p2_vals if available to cleanly isolate the two-photon transition without
    confusing or swapping the fundamental 0->1 resonance.

    Returns:
        dict containing:
            'f01': fitted 0->1 frequency in Hz
            'f02_half': fitted two-photon frequency (0->2)/2 in Hz (or None)
            'alpha': measured anharmonicity in Hz: 2 * (f02_half - f01) (or None)
            'fwhm_01': FWHM of 0->1 peak in Hz
            'fwhm_02_half': FWHM of two-photon peak in Hz (or None)
            'amp_01': amplitude of 0->1 peak
            'amp_02_half': amplitude of two-photon peak (or 0.0)
            'y0': baseline offset
            'fit_fn': callable function
            'y_fit': fitted values array
            'has_two_photon': bool whether two-photon peak was detected
            'success': bool
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    # 1. Fit fundamental 01 resonance
    single_fit = fit_lorentzian(xs, ys, center_guess=f01_guess, fwhm_guess=fwhm_guess)
    f01_fit = single_fit["f0"]
    fwhm_fit = single_fit["fwhm"]
    amp_fit = single_fit["amp"]
    y0_fit = single_fit["y0"]

    # 2. Check for two-photon peak
    has_two_photon = False
    f02_half_fit = None
    fwhm_02_fit = None
    amp_02_fit = 0.0

    if p2_vals is not None:
        p2_arr = np.asarray(p2_vals, dtype=float)
        # Check if P2 has a significant peak (> 0.04)
        if np.max(p2_arr) > 0.04:
            fit_p2 = fit_lorentzian(xs, p2_arr, center_guess=f02_half_guess)
            if fit_p2["success"] and fit_p2["amp"] > 0.03:
                # Ensure the P2 peak is distinct from f01
                if abs(fit_p2["f0"] - f01_fit) > 0.3 * fwhm_fit:
                    has_two_photon = True
                    f02_half_fit = fit_p2["f0"]
                    fwhm_02_fit = fit_p2["fwhm"]
                    amp_02_fit = fit_p2["amp"]
    elif f02_half_guess is not None and abs(f02_half_guess - f01_fit) > 0.5 * fwhm_fit:
        # Check if there is an actual local peak in ys near f02_half_guess
        idx_02 = np.argmin(np.abs(xs - f02_half_guess))
        if ys[idx_02] - y0_fit > 0.05:
            # Fit dual Lorentzian
            def dual_model(x, f1, g1, a1, f2, g2, a2, y0):
                return (
                    y0
                    + a1 / (1.0 + ((x - f1) / (g1 / 2.0)) ** 2)
                    + a2 / (1.0 + ((x - f2) / (g2 / 2.0)) ** 2)
                )

            p0 = [f01_fit, fwhm_fit, amp_fit, f02_half_guess, fwhm_fit * 0.6, 0.3 * amp_fit, y0_fit]
            x_min, x_max = min(xs[0], xs[-1]), max(xs[0], xs[-1])
            bounds = (
                [x_min, 1e-3, 0.0, x_min, 1e-3, 0.0, -10.0],
                [x_max, x_max - x_min if x_max > x_min else 1e12, 10.0, x_max, x_max - x_min if x_max > x_min else 1e12, 10.0, 10.0],
            )
            try:
                popt, _ = curve_fit(dual_model, xs, ys, p0=p0, bounds=bounds, maxfev=15000)
                f1_f, g1_f, a1_f, f2_f, g2_f, a2_f, y0_f = popt
                # Preserve identity: f1 is closest to f01_guess, f2 is closest to f02_half_guess
                if abs(f1_f - f01_guess) > abs(f2_f - f01_guess):
                    f1_f, f2_f = f2_f, f1_f
                    g1_f, g2_f = g2_f, g1_f
                    a1_f, a2_f = a2_f, a1_f
                if a2_f > 0.03 and abs(f2_f - f1_f) > 0.3 * fwhm_fit:
                    has_two_photon = True
                    f01_fit = f1_f
                    fwhm_fit = abs(g1_f)
                    amp_fit = a1_f
                    f02_half_fit = f2_f
                    fwhm_02_fit = abs(g2_f)
                    amp_02_fit = a2_f
                    y0_fit = y0_f
            except Exception:
                pass

    if has_two_photon and f02_half_fit is not None:
        alpha_measured = 2.0 * (f02_half_fit - f01_fit)
        def fit_fn(x):
            h1 = fwhm_fit / 2.0
            h2 = (fwhm_02_fit or (fwhm_fit * 0.8)) / 2.0
            return y0_fit + amp_fit / (1.0 + ((x - f01_fit) / h1) ** 2) + amp_02_fit / (1.0 + ((x - f02_half_fit) / h2) ** 2)

        return {
            "f01": float(f01_fit),
            "f02_half": float(f02_half_fit),
            "alpha": float(alpha_measured),
            "fwhm_01": float(fwhm_fit),
            "fwhm_02_half": float(fwhm_02_fit) if fwhm_02_fit else None,
            "amp_01": float(amp_fit),
            "amp_02_half": float(amp_02_fit),
            "y0": float(y0_fit),
            "fit_fn": fit_fn,
            "y_fit": fit_fn(xs),
            "has_two_photon": True,
            "success": True,
        }

    return {
        "f01": float(f01_fit),
        "f02_half": None,
        "alpha": None,
        "fwhm_01": float(fwhm_fit),
        "fwhm_02_half": None,
        "amp_01": float(amp_fit),
        "amp_02_half": 0.0,
        "y0": float(y0_fit),
        "fit_fn": single_fit["fit_fn"],
        "y_fit": single_fit["y_fit"],
        "has_two_photon": False,
        "success": single_fit["success"],
    }

