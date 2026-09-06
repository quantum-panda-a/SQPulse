"""Concrete pulse shapes for SQPulse in SI units."""

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
        duration (float): Total pulse length in seconds (s).
        amp (float): Pulse amplitude (rad/s or normalized).
        sigma (Optional[float]): Standard deviation in seconds (s). If None, set to duration / chop.
        chop (float): Ratio of duration / sigma (default: 4.0).
        drag (float): Dimensionless DRAG coefficient beta (default: 0.0).
        alpha (Optional[float]): Anharmonicity in Hz for DRAG quadrature scaling.
        phase (float): Phase in radians.
        detune (float): Detuning in Hz.
        name (Optional[str]): Pulse name.
        length (Optional[float]): Alias for duration in seconds (s).
    """

    def __init__(
        self,
        duration: Optional[float] = None,
        amp: float = 1.0,
        sigma: Optional[float] = None,
        chop: float = 4.0,
        drag: float = 0.0,
        alpha: Optional[float] = None,
        phase: float = 0.0,
        detune: float = 0.0,
        name: Optional[str] = None,
        length: Optional[float] = None,
    ):
        super().__init__(
            duration=duration,
            amp=amp,
            phase=phase,
            detune=detune,
            drag=drag,
            alpha=alpha,
            name=name,
            length=length,
        )
        self.chop = float(chop)
        if sigma is not None:
            sigma = float(sigma)
            if sigma <= 0:
                raise ValueError(f"sigma must be positive, got {sigma} s")
            if sigma > self.duration:
                hint = ""
                if sigma > 1e-3 and self.duration < 1e-3:
                    hint = f" Did you mean sigma={sigma}e-9 s ({sigma} ns)? All time parameters in SQPulse are in SI units (seconds)."
                raise ValueError(
                    f"Gaussian standard deviation sigma ({sigma} s) cannot exceed pulse duration ({self.duration} s).{hint}"
                )
            self.sigma = sigma
        else:
            self.sigma = self.duration / self.chop

    def envelope(self, t: np.ndarray) -> np.ndarray:
        tc = self.duration / 2.0
        p = np.exp(-((t - tc) ** 2) / (2.0 * self.sigma**2))
        p0 = np.exp(-(tc**2) / (2.0 * self.sigma**2))
        return (p - p0) / (1.0 - p0)

    def envelope_derivative(self, t: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
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
        duration (float): Pulse duration in seconds (s).
        amp (float): Pulse amplitude.
        drag (float): DRAG coefficient.
        phase (float): Phase in radians.
        detune (float): Detuning in Hz.
        name (Optional[str]): Pulse name.
        length (Optional[float]): Alias for duration in seconds (s).
    """

    def __init__(
        self,
        duration: Optional[float] = None,
        amp: float = 1.0,
        drag: float = 0.0,
        alpha: Optional[float] = None,
        phase: float = 0.0,
        detune: float = 0.0,
        name: Optional[str] = None,
        length: Optional[float] = None,
    ):
        super().__init__(
            duration=duration,
            amp=amp,
            phase=phase,
            detune=detune,
            drag=drag,
            alpha=alpha,
            name=name,
            length=length,
        )

    def envelope(self, t: np.ndarray) -> np.ndarray:
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * t / self.duration))

    def envelope_derivative(self, t: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        return (np.pi / self.duration) * np.sin(2.0 * np.pi * t / self.duration)


class LorentzianPulse(Pulse):
    r"""Cauchy-Lorentzian pulse with zero-offset boundaries.

    .. math::
        f(t) = \frac{1}{1 + \left(\frac{t - \tau/2}{\Gamma / 2}\right)^2}

    Args:
        duration (float): Pulse duration in seconds (s).
        amp (float): Pulse amplitude.
        gamma (Optional[float]): Full width at half maximum (FWHM) in seconds (s). Defaults to duration / 4.
        drag (float): Dimensionless DRAG coefficient beta.
        alpha (Optional[float]): Anharmonicity in Hz for DRAG quadrature scaling.
        phase (float): Phase in radians.
        detune (float): Detuning in Hz.
        name (Optional[str]): Pulse name.
        length (Optional[float]): Alias for duration in seconds (s).
    """

    def __init__(
        self,
        duration: Optional[float] = None,
        amp: float = 1.0,
        gamma: Optional[float] = None,
        drag: float = 0.0,
        alpha: Optional[float] = None,
        phase: float = 0.0,
        detune: float = 0.0,
        name: Optional[str] = None,
        length: Optional[float] = None,
    ):
        super().__init__(
            duration=duration,
            amp=amp,
            phase=phase,
            detune=detune,
            drag=drag,
            alpha=alpha,
            name=name,
            length=length,
        )
        if gamma is not None:
            gamma = float(gamma)
            if gamma <= 0:
                raise ValueError(f"gamma must be positive, got {gamma} s")
            if gamma > 2.0 * self.duration:
                hint = ""
                if gamma > 1e-3 and self.duration < 1e-3:
                    hint = f" Did you mean gamma={gamma}e-9 s ({gamma} ns)? All time parameters in SQPulse are in SI units (seconds)."
                raise ValueError(
                    f"Lorentzian linewidth gamma ({gamma} s) is unusually large compared to pulse duration ({self.duration} s).{hint}"
                )
            self.gamma = gamma
        else:
            self.gamma = self.duration / 4.0

    def envelope(self, t: np.ndarray) -> np.ndarray:
        tc = self.duration / 2.0
        hwhm = self.gamma / 2.0
        l_raw = 1.0 / (1.0 + ((t - tc) / hwhm) ** 2)
        l0 = 1.0 / (1.0 + (tc / hwhm) ** 2)
        return (l_raw - l0) / (1.0 - l0)

    def envelope_derivative(self, t: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
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
        duration (float): Pulse duration in seconds (s).
        amp (float): Pulse amplitude.
        phase (float): Phase in radians.
        detune (float): Detuning in Hz.
        name (Optional[str]): Pulse name.
        length (Optional[float]): Alias for duration in seconds (s).
    """

    def __init__(
        self,
        duration: Optional[float] = None,
        amp: float = 1.0,
        phase: float = 0.0,
        detune: float = 0.0,
        name: Optional[str] = None,
        length: Optional[float] = None,
    ):
        super().__init__(
            duration=duration,
            amp=amp,
            phase=phase,
            detune=detune,
            drag=0.0,
            alpha=None,
            name=name,
            length=length,
        )

    def envelope(self, t: np.ndarray) -> np.ndarray:
        return np.ones_like(t, dtype=float)

    def envelope_derivative(self, t: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        return np.zeros_like(t, dtype=float)


class FlatTopPulse(Pulse):
    """Flat-top pulse with smooth ramp-up and ramp-down edges.

    Useful for long resonant drives, parametric gates, or readout pulses.

    Args:
        duration (float): Total pulse duration in seconds (s).
        amp (float): Pulse amplitude during flat region.
        ramp_time (Optional[float]): Duration of ramp-up and ramp-down in seconds (s). Defaults to duration / 4.
        ramp_type (str): 'cosine' (Hann edge) or 'gaussian'.
        drag (float): Dimensionless DRAG coefficient beta.
        alpha (Optional[float]): Anharmonicity in Hz for DRAG quadrature scaling.
        phase (float): Phase in radians.
        detune (float): Detuning in Hz.
        name (Optional[str]): Pulse name.
        length (Optional[float]): Alias for duration in seconds (s).
    """

    def __init__(
        self,
        duration: Optional[float] = None,
        amp: float = 1.0,
        ramp_time: Optional[float] = None,
        ramp_type: str = "cosine",
        drag: float = 0.0,
        alpha: Optional[float] = None,
        phase: float = 0.0,
        detune: float = 0.0,
        name: Optional[str] = None,
        length: Optional[float] = None,
    ):
        super().__init__(
            duration=duration,
            amp=amp,
            phase=phase,
            detune=detune,
            drag=drag,
            alpha=alpha,
            name=name,
            length=length,
        )
        self.ramp_type = ramp_type.lower()
        if ramp_time is None:
            self.ramp_time = self.duration / 4.0
        else:
            ramp_time = float(ramp_time)
            if ramp_time < 0:
                raise ValueError(f"ramp_time must be non-negative, got {ramp_time} s")
            if 2 * ramp_time > self.duration:
                hint = ""
                if ramp_time > 1e-3 and self.duration < 1e-3:
                    hint = f" Note: ramp_time={ramp_time} s appears to be in nanoseconds. Did you mean ramp_time={ramp_time}e-9 s ({ramp_time} ns)?"
                raise ValueError(
                    f"2 * ramp_time ({2 * ramp_time} s) cannot exceed duration ({self.duration} s). "
                    f"All time parameters in SQPulse are in SI units (seconds).{hint}"
                )
            self.ramp_time = ramp_time

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
        duration (float): Pulse duration in seconds (s).
        amp (float): Pulse amplitude.
        sigma (Optional[float]): Characteristic width in seconds (s).
        chop (float): Ratio of duration / sigma.
        drag (float): DRAG coefficient.
        phase (float): Phase in radians.
        detune (float): Detuning in Hz.
        name (Optional[str]): Pulse name.
        length (Optional[float]): Alias for duration in seconds (s).
    """

    def __init__(
        self,
        duration: Optional[float] = None,
        amp: float = 1.0,
        sigma: Optional[float] = None,
        chop: float = 4.0,
        drag: float = 0.0,
        alpha: Optional[float] = None,
        phase: float = 0.0,
        detune: float = 0.0,
        name: Optional[str] = None,
        length: Optional[float] = None,
    ):
        super().__init__(
            duration=duration,
            amp=amp,
            phase=phase,
            detune=detune,
            drag=drag,
            alpha=alpha,
            name=name,
            length=length,
        )
        self.chop = float(chop)
        if sigma is not None:
            sigma = float(sigma)
            if sigma <= 0:
                raise ValueError(f"sigma must be positive, got {sigma} s")
            if sigma > self.duration:
                hint = ""
                if sigma > 1e-3 and self.duration < 1e-3:
                    hint = f" Did you mean sigma={sigma}e-9 s ({sigma} ns)? All time parameters in SQPulse are in SI units (seconds)."
                raise ValueError(
                    f"Sech characteristic width sigma ({sigma} s) cannot exceed pulse duration ({self.duration} s).{hint}"
                )
            self.sigma = sigma
        else:
            self.sigma = self.duration / self.chop

    def envelope(self, t: np.ndarray) -> np.ndarray:
        tc = self.duration / 2.0
        rho = 1.0 / self.sigma
        s = 1.0 / np.cosh(rho * (t - tc))
        s0 = 1.0 / np.cosh(rho * tc)
        return (s - s0) / (1.0 - s0)

    def envelope_derivative(self, t: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        tc = self.duration / 2.0
        rho = 1.0 / self.sigma
        arg = rho * (t - tc)
        s0 = 1.0 / np.cosh(rho * tc)
        ds = -rho * np.sinh(arg) / (np.cosh(arg) ** 2)
        return ds / (1.0 - s0)


class CustomPulse(Pulse):
    """Pulse defined by an arbitrary user-supplied envelope function f(t).

    Args:
        duration (float): Pulse duration in seconds (s).
        envelope_fn (Callable[[np.ndarray], np.ndarray]): Function taking time array (s) and returning [0, 1] envelope.
        amp (float): Amplitude.
        drag (float): Dimensionless DRAG coefficient beta.
        alpha (Optional[float]): Anharmonicity in Hz for DRAG quadrature scaling.
        phase (float): Phase in radians.
        detune (float): Detuning in Hz.
        name (Optional[str]): Pulse name.
        length (Optional[float]): Alias for duration in seconds (s).
    """

    def __init__(
        self,
        duration: Optional[float] = None,
        envelope_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        amp: float = 1.0,
        drag: float = 0.0,
        alpha: Optional[float] = None,
        phase: float = 0.0,
        detune: float = 0.0,
        name: Optional[str] = None,
        length: Optional[float] = None,
    ):
        super().__init__(
            duration=duration,
            amp=amp,
            phase=phase,
            detune=detune,
            drag=drag,
            alpha=alpha,
            name=name,
            length=length,
        )
        if envelope_fn is None:
            raise ValueError("envelope_fn must be provided for CustomPulse")
        self._fn = envelope_fn

    def envelope(self, t: np.ndarray) -> np.ndarray:
        return np.asarray(self._fn(t), dtype=float)


class DRAGPulse(GaussianPulse):
    """Derivative Removal by Adiabatic Gate (DRAG) pulse.

    Convenience subclass of GaussianPulse with dimensionless DRAG coefficient beta.
    A value of drag=1.0 corresponds to the ideal first-order theoretical DRAG correction:
        Q(t) = - (drag / (2 * pi * alpha)) * dI/dt
    """

    def __init__(
        self,
        duration: Optional[float] = None,
        amp: float = 1.0,
        sigma: Optional[float] = None,
        drag: float = 1.0,
        alpha: Optional[float] = None,
        phase: float = 0.0,
        detune: float = 0.0,
        name: Optional[str] = None,
        length: Optional[float] = None,
    ):
        super().__init__(
            duration=duration,
            amp=amp,
            sigma=sigma,
            drag=drag,
            alpha=alpha,
            phase=phase,
            detune=detune,
            name=name or "DRAGPulse",
            length=length,
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
            alpha=base_pulse.alpha,
            name=f"{base_pulse.name}*{scalar:.2f}",
        )
        self.base_pulse = base_pulse

    def envelope(self, t: np.ndarray) -> np.ndarray:
        return self.base_pulse.envelope(t)

    def envelope_derivative(self, t: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        return self.base_pulse.envelope_derivative(t, dt=dt)


# Functional pulse factory wrappers
def gaussian_pulse(
    duration: Optional[float] = None,
    amp: float = 1.0,
    sigma: Optional[float] = None,
    chop: float = 4.0,
    drag: float = 0.0,
    alpha: Optional[float] = None,
    phase: float = 0.0,
    detune: float = 0.0,
    name: Optional[str] = None,
    length: Optional[float] = None,
) -> GaussianPulse:
    """Create a Gaussian pulse."""
    return GaussianPulse(
        duration=duration,
        amp=amp,
        sigma=sigma,
        chop=chop,
        drag=drag,
        alpha=alpha,
        phase=phase,
        detune=detune,
        name=name,
        length=length,
    )


def cosine_pulse(
    duration: Optional[float] = None,
    amp: float = 1.0,
    drag: float = 0.0,
    alpha: Optional[float] = None,
    phase: float = 0.0,
    detune: float = 0.0,
    name: Optional[str] = None,
    length: Optional[float] = None,
) -> CosinePulse:
    """Create a raised-cosine / Hann pulse."""
    return CosinePulse(
        duration=duration,
        amp=amp,
        drag=drag,
        alpha=alpha,
        phase=phase,
        detune=detune,
        name=name,
        length=length,
    )


def lorentzian_pulse(
    duration: Optional[float] = None,
    amp: float = 1.0,
    gamma: Optional[float] = None,
    drag: float = 0.0,
    alpha: Optional[float] = None,
    phase: float = 0.0,
    detune: float = 0.0,
    name: Optional[str] = None,
    length: Optional[float] = None,
) -> LorentzianPulse:
    """Create a Cauchy-Lorentzian pulse."""
    return LorentzianPulse(
        duration=duration,
        amp=amp,
        gamma=gamma,
        drag=drag,
        alpha=alpha,
        phase=phase,
        detune=detune,
        name=name,
        length=length,
    )


def square_pulse(
    duration: Optional[float] = None,
    amp: float = 1.0,
    phase: float = 0.0,
    detune: float = 0.0,
    name: Optional[str] = None,
    length: Optional[float] = None,
) -> SquarePulse:
    """Create an ideal rectangular / square pulse."""
    return SquarePulse(
        duration=duration,
        amp=amp,
        phase=phase,
        detune=detune,
        name=name,
        length=length,
    )


def flattop_pulse(
    duration: Optional[float] = None,
    amp: float = 1.0,
    ramp_time: Optional[float] = None,
    ramp_type: str = "cosine",
    drag: float = 0.0,
    alpha: Optional[float] = None,
    phase: float = 0.0,
    detune: float = 0.0,
    name: Optional[str] = None,
    length: Optional[float] = None,
) -> FlatTopPulse:
    """Create a flat-top pulse with smooth ramp-up and ramp-down edges."""
    return FlatTopPulse(
        duration=duration,
        amp=amp,
        ramp_time=ramp_time,
        ramp_type=ramp_type,
        drag=drag,
        alpha=alpha,
        phase=phase,
        detune=detune,
        name=name,
        length=length,
    )


def sech_pulse(
    duration: Optional[float] = None,
    amp: float = 1.0,
    sigma: Optional[float] = None,
    chop: float = 4.0,
    drag: float = 0.0,
    alpha: Optional[float] = None,
    phase: float = 0.0,
    detune: float = 0.0,
    name: Optional[str] = None,
    length: Optional[float] = None,
) -> SechPulse:
    """Create a hyperbolic secant (Sech) pulse."""
    return SechPulse(
        duration=duration,
        amp=amp,
        sigma=sigma,
        chop=chop,
        drag=drag,
        alpha=alpha,
        phase=phase,
        detune=detune,
        name=name,
        length=length,
    )


def drag_pulse(
    duration: Optional[float] = None,
    amp: float = 1.0,
    sigma: Optional[float] = None,
    drag: float = 1.0,
    alpha: Optional[float] = None,
    phase: float = 0.0,
    detune: float = 0.0,
    name: Optional[str] = None,
    length: Optional[float] = None,
) -> DRAGPulse:
    """Create a DRAG pulse."""
    return DRAGPulse(
        duration=duration,
        amp=amp,
        sigma=sigma,
        drag=drag,
        alpha=alpha,
        phase=phase,
        detune=detune,
        name=name,
        length=length,
    )


def custom_pulse(
    duration: Optional[float] = None,
    envelope_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    amp: float = 1.0,
    drag: float = 0.0,
    alpha: Optional[float] = None,
    phase: float = 0.0,
    detune: float = 0.0,
    name: Optional[str] = None,
    length: Optional[float] = None,
) -> CustomPulse:
    """Create a custom pulse from an arbitrary envelope function."""
    return CustomPulse(
        duration=duration,
        envelope_fn=envelope_fn,
        amp=amp,
        drag=drag,
        alpha=alpha,
        phase=phase,
        detune=detune,
        name=name,
        length=length,
    )
