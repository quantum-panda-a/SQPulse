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
