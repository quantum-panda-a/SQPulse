"""Concrete pulse shapes for SQPulse."""

from __future__ import annotations
from typing import Optional, Callable, Union
import numpy as np
from .base import Pulse


class GaussianPulse(Pulse):
    r"""Gaussian pulse with truncated zero-offset edges.

    .. math::
        f(t) = \frac{e^{-(t - \tau/2)^2 / (2\sigma^2)} - e^{-(\tau/2)^2 / (2\sigma^2)}}
                    {1 - e^{-(\tau/2)^2 / (2\sigma^2)}}

    Args:
        duration (float): Total pulse length in ns.
        amp (float): Pulse amplitude.
        sigma (Optional[float]): Standard deviation in ns. If None, set to duration / chop.
        chop (float): Ratio of duration / sigma (default: 4.0).
        drag (float): DRAG coefficient for Q quadrature.
        phase (float): Phase in radians.
        detune (float): Detuning in GHz.
        name (Optional[str]): Pulse name.
    """

    def __init__(
        self,
        duration: float,
        amp: float = 1.0,
        sigma: Optional[float] = None,
        chop: float = 4.0,
        drag: float = 0.0,
        phase: float = 0.0,
        detune: float = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(duration=duration, amp=amp, phase=phase, detune=detune, drag=drag, name=name)
        self.chop = float(chop)
        self.sigma = float(sigma) if sigma is not None else float(duration) / self.chop

    def envelope(self, t: np.ndarray) -> np.ndarray:
        tc = self.duration / 2.0
        p = np.exp(-((t - tc) ** 2) / (2.0 * self.sigma**2))
        p0 = np.exp(-(tc**2) / (2.0 * self.sigma**2))
        return (p - p0) / (1.0 - p0)

    def envelope_derivative(self, t: np.ndarray, dt: float = 0.01) -> np.ndarray:
        tc = self.duration / 2.0
        p = np.exp(-((t - tc) ** 2) / (2.0 * self.sigma**2))
        p0 = np.exp(-(tc**2) / (2.0 * self.sigma**2))
        return -((t - tc) / (self.sigma**2)) * p / (1.0 - p0)


class CosinePulse(Pulse):
    r"""Raised-cosine / Hann window pulse.

    Provides extremely low spectral leakage and zero boundary slope.

    .. math::
        f(t) = \frac{1}{2} \left[1 - \cos\left(\frac{2\pi t}{\tau}\right)\right]
             = \sin^2\left(\frac{\pi t}{\tau}\right)

    Args:
        duration (float): Pulse duration in ns.
        amp (float): Pulse amplitude.
        drag (float): DRAG coefficient.
        phase (float): Phase in radians.
        detune (float): Detuning in GHz.
        name (Optional[str]): Pulse name.
    """

    def __init__(
        self,
        duration: float,
        amp: float = 1.0,
        drag: float = 0.0,
        phase: float = 0.0,
        detune: float = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(duration=duration, amp=amp, phase=phase, detune=detune, drag=drag, name=name)

    def envelope(self, t: np.ndarray) -> np.ndarray:
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * t / self.duration))

    def envelope_derivative(self, t: np.ndarray, dt: float = 0.01) -> np.ndarray:
        return (np.pi / self.duration) * np.sin(2.0 * np.pi * t / self.duration)


class LorentzianPulse(Pulse):
    r"""Cauchy-Lorentzian pulse with zero-offset boundaries.

    .. math::
        f(t) = \frac{1}{1 + \left(\frac{t - \tau/2}{\Gamma / 2}\right)^2}

    Args:
        duration (float): Pulse duration in ns.
        amp (float): Pulse amplitude.
        gamma (Optional[float]): Full width at half maximum (FWHM) in ns. Defaults to duration / 4.
        drag (float): DRAG coefficient.
        phase (float): Phase in radians.
        detune (float): Detuning in GHz.
        name (Optional[str]): Pulse name.
    """

    def __init__(
        self,
        duration: float,
        amp: float = 1.0,
        gamma: Optional[float] = None,
        drag: float = 0.0,
        phase: float = 0.0,
        detune: float = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(duration=duration, amp=amp, phase=phase, detune=detune, drag=drag, name=name)
        self.gamma = float(gamma) if gamma is not None else float(duration) / 4.0

    def envelope(self, t: np.ndarray) -> np.ndarray:
        tc = self.duration / 2.0
        hwhm = self.gamma / 2.0
        l_raw = 1.0 / (1.0 + ((t - tc) / hwhm) ** 2)
        l0 = 1.0 / (1.0 + (tc / hwhm) ** 2)
        return (l_raw - l0) / (1.0 - l0)

    def envelope_derivative(self, t: np.ndarray, dt: float = 0.01) -> np.ndarray:
        tc = self.duration / 2.0
        hwhm = self.gamma / 2.0
        denom = 1.0 + ((t - tc) / hwhm) ** 2
        l0 = 1.0 / (1.0 + (tc / hwhm) ** 2)
        return -2.0 * (t - tc) / (hwhm**2 * (denom**2) * (1.0 - l0))


class SquarePulse(Pulse):
    """Ideal rectangular (square) pulse.

    In the frequency domain, it exhibits the characteristic sinc function
    with high spectral sidelobes (-13 dB first sidelobe).

    Args:
        duration (float): Pulse duration in ns.
        amp (float): Pulse amplitude.
        phase (float): Phase in radians.
        detune (float): Detuning in GHz.
        name (Optional[str]): Pulse name.
    """

    def __init__(
        self,
        duration: float,
        amp: float = 1.0,
        phase: float = 0.0,
        detune: float = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(duration=duration, amp=amp, phase=phase, detune=detune, drag=0.0, name=name)

    def envelope(self, t: np.ndarray) -> np.ndarray:
        return np.ones_like(t, dtype=float)

    def envelope_derivative(self, t: np.ndarray, dt: float = 0.01) -> np.ndarray:
        return np.zeros_like(t, dtype=float)


class FlatTopPulse(Pulse):
    """Flat-top pulse with smooth ramp-up and ramp-down edges.

    Useful for long resonant drives, parametric gates, or readout pulses.

    Args:
        duration (float): Total pulse duration in ns.
        amp (float): Pulse amplitude during flat region.
        ramp_time (Optional[float]): Duration of ramp-up and ramp-down in ns. Defaults to duration / 8.
        ramp_type (str): 'cosine' (Hann edge) or 'gaussian'.
        drag (float): DRAG coefficient.
        phase (float): Phase in radians.
        detune (float): Detuning in GHz.
        name (Optional[str]): Pulse name.
    """

    def __init__(
        self,
        duration: float,
        amp: float = 1.0,
        ramp_time: Optional[float] = None,
        ramp_type: str = "cosine",
        drag: float = 0.0,
        phase: float = 0.0,
        detune: float = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(duration=duration, amp=amp, phase=phase, detune=detune, drag=drag, name=name)
        self.ramp_type = ramp_type.lower()
        if ramp_time is None:
            self.ramp_time = min(10.0, duration / 4.0)
        else:
            if 2 * ramp_time > duration:
                raise ValueError(f"2 * ramp_time ({2*ramp_time}) cannot exceed duration ({duration})")
            self.ramp_time = float(ramp_time)

    def envelope(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t)
        env = np.ones_like(t, dtype=float)
        t_ramp = self.ramp_time

        if t_ramp <= 0:
            return env

        # Ramp up: t < t_ramp
        mask_up = t < t_ramp
        if self.ramp_type == "cosine":
            env[mask_up] = 0.5 * (1.0 - np.cos(np.pi * t[mask_up] / t_ramp))
        elif self.ramp_type == "gaussian":
            sigma = t_ramp / 2.0
            env[mask_up] = np.exp(-((t[mask_up] - t_ramp) ** 2) / (2.0 * sigma**2))

        # Ramp down: t > duration - t_ramp
        t_down_start = self.duration - t_ramp
        mask_down = t > t_down_start
        t_rel = t[mask_down] - t_down_start
        if self.ramp_type == "cosine":
            env[mask_down] = 0.5 * (1.0 + np.cos(np.pi * t_rel / t_ramp))
        elif self.ramp_type == "gaussian":
            sigma = t_ramp / 2.0
            env[mask_down] = np.exp(-(t_rel**2) / (2.0 * sigma**2))

        return env


class SechPulse(Pulse):
    r"""Hyperbolic secant (Sech) pulse.

    .. math::
        f(t) = \text{sech}\left(\frac{t - \tau/2}{\sigma}\right)

    Args:
        duration (float): Pulse duration in ns.
        amp (float): Pulse amplitude.
        sigma (Optional[float]): Characteristic width in ns.
        chop (float): Ratio of duration / sigma.
        drag (float): DRAG coefficient.
        phase (float): Phase in radians.
        detune (float): Detuning in GHz.
        name (Optional[str]): Pulse name.
    """

    def __init__(
        self,
        duration: float,
        amp: float = 1.0,
        sigma: Optional[float] = None,
        chop: float = 4.0,
        drag: float = 0.0,
        phase: float = 0.0,
        detune: float = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(duration=duration, amp=amp, phase=phase, detune=detune, drag=drag, name=name)
        self.chop = float(chop)
        self.sigma = float(sigma) if sigma is not None else float(duration) / self.chop

    def envelope(self, t: np.ndarray) -> np.ndarray:
        tc = self.duration / 2.0
        rho = 1.0 / self.sigma
        s = 1.0 / np.cosh(rho * (t - tc))
        s0 = 1.0 / np.cosh(rho * tc)
        return (s - s0) / (1.0 - s0)

    def envelope_derivative(self, t: np.ndarray, dt: float = 0.01) -> np.ndarray:
        tc = self.duration / 2.0
        rho = 1.0 / self.sigma
        arg = rho * (t - tc)
        s0 = 1.0 / np.cosh(rho * tc)
        ds = -rho * np.sinh(arg) / (np.cosh(arg) ** 2)
        return ds / (1.0 - s0)


class CustomPulse(Pulse):
    """Pulse defined by an arbitrary user-supplied envelope function f(t).

    Args:
        duration (float): Pulse duration in ns.
        envelope_fn (Callable[[np.ndarray], np.ndarray]): Function taking time array and returning [0, 1] envelope.
        amp (float): Amplitude.
        drag (float): DRAG coefficient.
        phase (float): Phase in radians.
        detune (float): Detuning in GHz.
        name (Optional[str]): Pulse name.
    """

    def __init__(
        self,
        duration: float,
        envelope_fn: Callable[[np.ndarray], np.ndarray],
        amp: float = 1.0,
        drag: float = 0.0,
        phase: float = 0.0,
        detune: float = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(duration=duration, amp=amp, phase=phase, detune=detune, drag=drag, name=name)
        self._fn = envelope_fn

    def envelope(self, t: np.ndarray) -> np.ndarray:
        return np.asarray(self._fn(t), dtype=float)


class DRAGPulse(GaussianPulse):
    """Derivative Removal by Adiabatic Gate (DRAG) pulse.

    Convenience subclass of GaussianPulse with DRAG coefficient specifically set,
    typically beta = -1 / anharmonicity.
    """

    def __init__(
        self,
        duration: float,
        amp: float = 1.0,
        sigma: Optional[float] = None,
        drag: float = 0.5,
        phase: float = 0.0,
        detune: float = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(
            duration=duration,
            amp=amp,
            sigma=sigma,
            drag=drag,
            phase=phase,
            detune=detune,
            name=name or "DRAGPulse",
        )


class ScaledPulse(Pulse):
    """Internal wrapper for scaling a pulse amplitude."""

    def __init__(self, base_pulse: Pulse, scalar: float):
        super().__init__(
            duration=base_pulse.duration,
            amp=base_pulse.amp * scalar,
            phase=base_pulse.phase,
            detune=base_pulse.detune,
            drag=base_pulse.drag,
            name=f"{base_pulse.name}*{scalar:.2f}",
        )
        self.base_pulse = base_pulse

    def envelope(self, t: np.ndarray) -> np.ndarray:
        return self.base_pulse.envelope(t)

    def envelope_derivative(self, t: np.ndarray, dt: float = 0.01) -> np.ndarray:
        return self.base_pulse.envelope_derivative(t, dt=dt)
