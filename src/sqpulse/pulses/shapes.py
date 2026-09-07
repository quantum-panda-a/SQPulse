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
        noise_sigma: float = 0.0,
        noise_alpha: float = 0.0,
        scale_noise: bool = False,
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
            noise_sigma=noise_sigma,
            noise_alpha=noise_alpha,
            scale_noise=scale_noise,
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
        noise_sigma: float = 0.0,
        noise_alpha: float = 0.0,
        scale_noise: bool = False,
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
            noise_sigma=noise_sigma,
            noise_alpha=noise_alpha,
            scale_noise=scale_noise,
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
        noise_sigma: float = 0.0,
        noise_alpha: float = 0.0,
        scale_noise: bool = False,
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
            noise_sigma=noise_sigma,
            noise_alpha=noise_alpha,
            scale_noise=scale_noise,
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
        noise_sigma: float = 0.0,
        noise_alpha: float = 0.0,
        scale_noise: bool = False,
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
            noise_sigma=noise_sigma,
            noise_alpha=noise_alpha,
            scale_noise=scale_noise,
            name=name,
            length=length,
        )

    def envelope(self, t: np.ndarray) -> np.ndarray:
        return np.ones_like(t, dtype=float)

    def envelope_derivative(self, t: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        return np.zeros_like(t, dtype=float)


class IdlePulse(Pulse):
    """Idle (delay / wait) pulse where waveform amplitude is always zero.

    Args:
        duration (float): Pulse duration in seconds (s).
        phase (float): Carrier phase in radians (default: 0.0).
        detune (float): Detuning in Hz (default: 0.0).
        name (Optional[str]): Pulse name (default: "IdlePulse").
        length (Optional[float]): Alias for duration in seconds (s).
    """

    def __init__(
        self,
        duration: Optional[float] = None,
        phase: float = 0.0,
        detune: float = 0.0,
        name: Optional[str] = None,
        length: Optional[float] = None,
    ):
        super().__init__(
            duration=duration,
            amp=0.0,
            phase=phase,
            detune=detune,
            drag=0.0,
            alpha=None,
            name=name or "IdlePulse",
            length=length,
        )

    def envelope(self, t: np.ndarray) -> np.ndarray:
        return np.zeros_like(t, dtype=float)

    def envelope_derivative(self, t: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        return np.zeros_like(t, dtype=float)


class FlatTopPulse(Pulse):
    """Flat-top pulse with smooth ramp-up and ramp-down edges, and optional pre-distortion overshoot.

    Useful for long resonant drives, parametric gates, readout pulses, or fast flux drives.

    Args:
        duration (float): Total pulse duration in seconds (s).
        amp (float): Pulse amplitude during flat region.
        ramp_time (Optional[float]): Duration of ramp-up and ramp-down in seconds (s). Defaults to duration / 4.
        ramp_type (str): 'cosine' (Hann edge), 'gaussian', or 'tanh'.
        overshoot_amp (float): Additional pre-distortion overshoot height (default: 0.0).
        overshoot_len (Optional[float]): Duration of overshoot spike at rising/falling edges (default: ramp_time).
        sigma (Optional[float]): Standard deviation for gaussian ramp in seconds (s). Defaults to ramp_time / 2.
        drag (float): Dimensionless DRAG coefficient beta.
        alpha (Optional[float]): Anharmonicity in Hz for DRAG quadrature scaling.
        phase (float): Phase in radians.
        detune (float): Detuning in Hz.
        noise_sigma (float): Standard deviation of additive noise. Default: 0.0.
        noise_alpha (float): Exponent for noise PSD S(f) ~ (1/f)^alpha. Default: 0.0.
        scale_noise (bool): Whether to scale additive noise by amplitude. Default: False.
        name (Optional[str]): Pulse name.
        length (Optional[float]): Alias for duration in seconds (s).
    """

    def __init__(
        self,
        duration: Optional[float] = None,
        amp: float = 1.0,
        ramp_time: Optional[float] = None,
        ramp_type: str = "cosine",
        overshoot_amp: float = 0.0,
        overshoot_len: Optional[float] = None,
        sigma: Optional[float] = None,
        drag: float = 0.0,
        alpha: Optional[float] = None,
        phase: float = 0.0,
        detune: float = 0.0,
        noise_sigma: float = 0.0,
        noise_alpha: float = 0.0,
        scale_noise: bool = False,
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
            noise_sigma=noise_sigma,
            noise_alpha=noise_alpha,
            scale_noise=scale_noise,
            name=name,
            length=length,
        )
        self.ramp_type = ramp_type.lower()
        if self.ramp_type not in ("cosine", "gaussian", "tanh"):
            raise ValueError(
                f"ramp_type must be 'cosine', 'gaussian', or 'tanh', got '{ramp_type}'"
            )
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

        self.overshoot_amp = float(overshoot_amp)
        if overshoot_len is None:
            self.overshoot_len = self.ramp_time if self.ramp_time > 0 else 0.1 * self.duration
        else:
            overshoot_len = float(overshoot_len)
            if overshoot_len < 0:
                raise ValueError(f"overshoot_len must be non-negative, got {overshoot_len} s")
            if 2 * overshoot_len > self.duration:
                raise ValueError(
                    f"2 * overshoot_len ({2 * overshoot_len} s) cannot exceed duration ({self.duration} s)."
                )
            self.overshoot_len = overshoot_len

        if sigma is not None:
            self.sigma = float(sigma)
            if self.sigma <= 0:
                raise ValueError(f"sigma must be positive, got {self.sigma} s")
        else:
            self.sigma = self.ramp_time / 2.0 if self.ramp_time > 0 else 0.0

    def envelope(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t)
        env = np.ones_like(t, dtype=float)
        t_ramp = self.ramp_time

        if t_ramp > 0:
            # Ramp up: t < t_ramp
            mask_up = t < t_ramp
            if self.ramp_type == "cosine":
                env[mask_up] = 0.5 * (1.0 - np.cos(np.pi * t[mask_up] / t_ramp))
            elif self.ramp_type == "gaussian":
                sigma = self.sigma if self.sigma > 0 else t_ramp / 2.0
                g = np.exp(-((t[mask_up] - t_ramp) ** 2) / (2.0 * sigma**2))
                g0 = np.exp(-(t_ramp**2) / (2.0 * sigma**2))
                env[mask_up] = (g - g0) / (1.0 - g0)
            elif self.ramp_type == "tanh":
                k = 2.0
                u_up = k * (2.0 * t[mask_up] / t_ramp - 1.0)
                env[mask_up] = (np.tanh(u_up) + np.tanh(k)) / (2.0 * np.tanh(k))

            # Ramp down: t > duration - t_ramp
            t_down_start = self.duration - t_ramp
            mask_down = t > t_down_start
            t_rel = t[mask_down] - t_down_start
            if self.ramp_type == "cosine":
                env[mask_down] = 0.5 * (1.0 + np.cos(np.pi * t_rel / t_ramp))
            elif self.ramp_type == "gaussian":
                sigma = self.sigma if self.sigma > 0 else t_ramp / 2.0
                g = np.exp(-(t_rel**2) / (2.0 * sigma**2))
                g0 = np.exp(-(t_ramp**2) / (2.0 * sigma**2))
                env[mask_down] = (g - g0) / (1.0 - g0)
            elif self.ramp_type == "tanh":
                k = 2.0
                u_down = k * (2.0 * t_rel / t_ramp - 1.0)
                env[mask_down] = (np.tanh(k) - np.tanh(u_down)) / (2.0 * np.tanh(k))

        # Add pre-distortion overshoot if configured
        if self.overshoot_amp != 0.0 and self.overshoot_len > 0:
            t_os = self.overshoot_len
            t_ramp = self.ramp_time
            # Rising edge overshoot bump centered at t_ramp
            t_s_up = max(0.0, t_ramp - t_os / 2.0)
            t_e_up = min(self.duration / 2.0, t_ramp + t_os / 2.0)
            w_up = t_e_up - t_s_up
            if w_up > 0:
                m_os_up = (t >= t_s_up) & (t <= t_e_up)
                env[m_os_up] += self.overshoot_amp * 0.5 * (1.0 - np.cos(2.0 * np.pi * (t[m_os_up] - t_s_up) / w_up))

            # Falling edge overshoot bump centered at duration - t_ramp
            t_down = self.duration - t_ramp
            t_s_down = max(self.duration / 2.0, t_down - t_os / 2.0)
            t_e_down = min(self.duration, t_down + t_os / 2.0)
            w_down = t_e_down - t_s_down
            if w_down > 0:
                m_os_down = (t >= t_s_down) & (t <= t_e_down)
                env[m_os_down] += self.overshoot_amp * 0.5 * (1.0 - np.cos(2.0 * np.pi * (t[m_os_down] - t_s_down) / w_down))

        return env

    def envelope_derivative(self, t: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        t = np.asarray(t)
        d_env = np.zeros_like(t, dtype=float)
        t_ramp = self.ramp_time

        if t_ramp > 0:
            # Ramp up: t < t_ramp
            mask_up = t < t_ramp
            if self.ramp_type == "cosine":
                d_env[mask_up] = (np.pi / (2.0 * t_ramp)) * np.sin(np.pi * t[mask_up] / t_ramp)
            elif self.ramp_type == "gaussian":
                sigma = self.sigma if self.sigma > 0 else t_ramp / 2.0
                g0 = np.exp(-(t_ramp**2) / (2.0 * sigma**2))
                d_env[mask_up] = -((t[mask_up] - t_ramp) / (sigma**2 * (1.0 - g0))) * np.exp(
                    -((t[mask_up] - t_ramp) ** 2) / (2.0 * sigma**2)
                )
            elif self.ramp_type == "tanh":
                k = 2.0
                u_up = k * (2.0 * t[mask_up] / t_ramp - 1.0)
                d_env[mask_up] = (k / (t_ramp * np.tanh(k))) / (np.cosh(u_up) ** 2)

            # Ramp down: t > duration - t_ramp
            t_down_start = self.duration - t_ramp
            mask_down = t > t_down_start
            t_rel = t[mask_down] - t_down_start
            if self.ramp_type == "cosine":
                d_env[mask_down] = -(np.pi / (2.0 * t_ramp)) * np.sin(np.pi * t_rel / t_ramp)
            elif self.ramp_type == "gaussian":
                sigma = self.sigma if self.sigma > 0 else t_ramp / 2.0
                g0 = np.exp(-(t_ramp**2) / (2.0 * sigma**2))
                d_env[mask_down] = -(t_rel / (sigma**2 * (1.0 - g0))) * np.exp(
                    -(t_rel**2) / (2.0 * sigma**2)
                )
            elif self.ramp_type == "tanh":
                k = 2.0
                u_down = k * (2.0 * t_rel / t_ramp - 1.0)
                d_env[mask_down] = -(k / (t_ramp * np.tanh(k))) / (np.cosh(u_down) ** 2)

        # Add derivative of overshoot bump
        if self.overshoot_amp != 0.0 and self.overshoot_len > 0:
            t_os = self.overshoot_len
            t_ramp = self.ramp_time
            t_s_up = max(0.0, t_ramp - t_os / 2.0)
            t_e_up = min(self.duration / 2.0, t_ramp + t_os / 2.0)
            w_up = t_e_up - t_s_up
            if w_up > 0:
                m_os_up = (t >= t_s_up) & (t <= t_e_up)
                d_env[m_os_up] += self.overshoot_amp * (np.pi / w_up) * np.sin(2.0 * np.pi * (t[m_os_up] - t_s_up) / w_up)

            t_down = self.duration - t_ramp
            t_s_down = max(self.duration / 2.0, t_down - t_os / 2.0)
            t_e_down = min(self.duration, t_down + t_os / 2.0)
            w_down = t_e_down - t_s_down
            if w_down > 0:
                m_os_down = (t >= t_s_down) & (t <= t_e_down)
                d_env[m_os_down] += self.overshoot_amp * (np.pi / w_down) * np.sin(2.0 * np.pi * (t[m_os_down] - t_s_down) / w_down)

        return d_env


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
        noise_sigma: float = 0.0,
        noise_alpha: float = 0.0,
        scale_noise: bool = False,
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
            noise_sigma=noise_sigma,
            noise_alpha=noise_alpha,
            scale_noise=scale_noise,
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


class SlepianPulse(Pulse):
    r"""Discrete Prolate Spheroidal Sequences (DPSS / Slepian) window pulse.

    Slepian pulses maximize energy concentration within a specified frequency bandwidth [-W, W],
    offering optimal suppression of spectral leakage into adjacent qubits or resonator channels.

    Args:
        duration (float): Pulse duration in seconds (s).
        amp (float): Pulse amplitude (normalized AWG amplitude V_0 in [-1.0, 1.0]).
        nw (float): Time-half-bandwidth product NW (default: 3.0). Higher NW widens the
            frequency mainlobe while providing deeper sidelobe suppression.
        drag (float): Dimensionless DRAG coefficient beta (default: 0.0).
        alpha (Optional[float]): Anharmonicity in Hz for DRAG quadrature scaling.
        phase (float): Phase in radians.
        detune (float): Detuning in Hz.
        zero_offset (bool): Whether to subtract boundary baseline offset to ensure f(0) = f(tau) = 0.
            Default: True.
        noise_sigma (float): Standard deviation of additive noise. Default: 0.0.
        noise_alpha (float): Exponent for noise PSD S(f) ~ (1/f)^alpha. Default: 0.0.
        scale_noise (bool): Whether to scale additive noise by amplitude. Default: False.
        name (Optional[str]): Pulse name.
        length (Optional[float]): Alias for duration in seconds (s).
    """

    def __init__(
        self,
        duration: Optional[float] = None,
        amp: float = 1.0,
        nw: float = 3.0,
        drag: float = 0.0,
        alpha: Optional[float] = None,
        phase: float = 0.0,
        detune: float = 0.0,
        zero_offset: bool = True,
        noise_sigma: float = 0.0,
        noise_alpha: float = 0.0,
        scale_noise: bool = False,
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
            noise_sigma=noise_sigma,
            noise_alpha=noise_alpha,
            scale_noise=scale_noise,
            name=name,
            length=length,
        )
        if nw <= 0:
            raise ValueError(f"Time-half-bandwidth product nw must be positive, got {nw}")
        self.nw = float(nw)
        self.zero_offset = bool(zero_offset)

        # Precompute high-resolution reference DPSS window and derivative for continuous interpolation
        from scipy.signal.windows import dpss

        self._n_ref = 2001
        self._t_ref = np.linspace(0, self.duration, self._n_ref)
        w = dpss(self._n_ref, NW=self.nw)
        if self.zero_offset:
            w0 = w[0]
            w = (w - w0) / (np.max(w) - w0)
        else:
            w = w / np.max(w)
        self._w_ref = w
        self._dw_ref = np.gradient(self._w_ref, self._t_ref)

    def envelope(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t)
        return np.interp(t, self._t_ref, self._w_ref, left=0.0, right=0.0)

    def envelope_derivative(self, t: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        t = np.asarray(t)
        return np.interp(t, self._t_ref, self._dw_ref, left=0.0, right=0.0)


class NetZeroPulse(Pulse):
    r"""Bipolar Net-Zero flux pulse for two-qubit gates (e.g. CZ, iSWAP).

    Consists of a positive pulse lobe in the first half [0, tau/2] and an
    antisymmetric negative pulse lobe in the second half [tau/2, tau], ensuring
    exact zero net flux integral:

    .. math::
        \int_0^\tau V(t) \, dt = 0

    This avoids long-lived magnetic flux vortex pinning, dielectric relaxation,
    and quasi-DC baseline drift in superconducting flux bias lines.

    Args:
        duration (float): Total pulse duration in seconds (s).
        amp (float): Peak pulse amplitude in [-1.0, 1.0].
        sigma (Optional[float]): Rise/fall edge smoothing parameter in seconds (s).
            Defaults to duration / 8.
        ramp_type (str): 'erf' or 'cosine' (default: 'erf').
        overshoot_amp (float): Pre-distortion overshoot height (default: 0.0).
        overshoot_len (Optional[float]): Duration of overshoot spike (default: sigma).
        phase (float): Carrier phase in radians.
        detune (float): Detuning in Hz.
        noise_sigma (float): Standard deviation of additive noise. Default: 0.0.
        noise_alpha (float): Exponent for noise PSD S(f) ~ (1/f)^alpha. Default: 0.0.
        scale_noise (bool): Whether to scale additive noise by amplitude. Default: False.
        name (Optional[str]): Pulse name.
        length (Optional[float]): Alias for duration in seconds (s).
    """

    def __init__(
        self,
        duration: Optional[float] = None,
        amp: float = 1.0,
        sigma: Optional[float] = None,
        ramp_type: str = "erf",
        overshoot_amp: float = 0.0,
        overshoot_len: Optional[float] = None,
        phase: float = 0.0,
        detune: float = 0.0,
        noise_sigma: float = 0.0,
        noise_alpha: float = 0.0,
        scale_noise: bool = False,
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
            noise_sigma=noise_sigma,
            noise_alpha=noise_alpha,
            scale_noise=scale_noise,
            name=name,
            length=length,
        )
        self.ramp_type = ramp_type.lower()
        if self.ramp_type not in ("erf", "cosine"):
            raise ValueError(f"ramp_type must be 'erf' or 'cosine', got '{ramp_type}'")

        if sigma is None:
            self.sigma = min(0.5e-9, self.duration / 16.0)
        else:
            sigma = float(sigma)
            if sigma <= 0:
                raise ValueError(f"sigma must be positive, got {sigma} s")
            if 4 * sigma > self.duration:
                raise ValueError(f"4 * sigma ({4 * sigma} s) cannot exceed duration ({self.duration} s)")
            self.sigma = sigma

        self.overshoot_amp = float(overshoot_amp)
        if overshoot_len is None:
            self.overshoot_len = self.sigma
        else:
            overshoot_len = float(overshoot_len)
            if overshoot_len < 0:
                raise ValueError(f"overshoot_len must be non-negative, got {overshoot_len} s")
            if 4 * overshoot_len > self.duration:
                raise ValueError(f"4 * overshoot_len ({4 * overshoot_len} s) cannot exceed duration ({self.duration} s)")
            self.overshoot_len = overshoot_len

    def envelope(self, t: np.ndarray) -> np.ndarray:
        from scipy.special import erf

        t = np.asarray(t, dtype=float)
        tau = self.duration
        t_mid = tau / 2.0

        if self.ramp_type == "erf":
            s = self.sigma * np.sqrt(2)
            raw = 0.5 * (
                erf(t / s - 2.0 * np.sqrt(2))
                - 2.0 * erf((t - t_mid) / s)
                + erf((t - tau) / s + 2.0 * np.sqrt(2))
            )
            f0 = 0.5 * (
                erf(-2.0 * np.sqrt(2))
                - 2.0 * erf(-tau / (2.0 * s))
                + erf(-tau / s + 2.0 * np.sqrt(2))
            )
            env = raw - f0 * (1.0 - 2.0 * t / tau)
        else:  # "cosine"
            t_ramp = self.sigma
            env = np.zeros_like(t)
            # First half: positive lobe
            m1_up = (t < t_ramp)
            env[m1_up] = 0.5 * (1.0 - np.cos(np.pi * t[m1_up] / t_ramp))
            m1_plat = (t >= t_ramp) & (t <= t_mid - t_ramp)
            env[m1_plat] = 1.0
            m1_down = (t > t_mid - t_ramp) & (t <= t_mid)
            env[m1_down] = 0.5 * (1.0 + np.cos(np.pi * (t[m1_down] - (t_mid - t_ramp)) / t_ramp))

            # Second half: antisymmetric negative lobe env(t) = -env(tau - t)
            t_rev = tau - t
            m2 = (t > t_mid)
            t2 = t_rev[m2]
            val2 = np.zeros_like(t2)
            v_up = (t2 < t_ramp)
            val2[v_up] = 0.5 * (1.0 - np.cos(np.pi * t2[v_up] / t_ramp))
            v_plat = (t2 >= t_ramp) & (t2 <= t_mid - t_ramp)
            val2[v_plat] = 1.0
            v_down = (t2 > t_mid - t_ramp) & (t2 <= t_mid)
            val2[v_down] = 0.5 * (1.0 + np.cos(np.pi * (t2[v_down] - (t_mid - t_ramp)) / t_ramp))
            env[m2] = -val2

        # Add antisymmetric predistortion overshoot if configured
        if self.overshoot_amp != 0.0 and self.overshoot_len > 0:
            t_os = self.overshoot_len
            m_os_up = t < t_os
            env[m_os_up] += self.overshoot_amp * 0.5 * (1.0 - np.cos(2.0 * np.pi * t[m_os_up] / t_os))
            m_os_down = t > (tau - t_os)
            t_rel = tau - t[m_os_down]
            env[m_os_down] -= self.overshoot_amp * 0.5 * (1.0 - np.cos(2.0 * np.pi * t_rel / t_os))

        return env


class CosineHD2DRAGPulse(Pulse):
    r"""High-Derivative (HD) DRAG pulse with analytical control envelopes.

    Based on Eric Hyyppä et al., "Reducing leakage of single-qubit gates for
    superconducting quantum processors using analytical control pulse envelopes",
    PRX Quantum 5, 030353 (2024).

    Synthesizes a 4-term cosine sum and its first through third analytical
    derivatives to simultaneously suppress leakage from the qubit subspace
    into both |2> and |3> transmon states.

    .. math::
        I(t) = D_0(t) + \beta_2 D_2(t) \\
        Q(t) = \frac{1}{\tau |\alpha|} (D_1(t) + \beta_2 D_3(t))

    where :math:`\beta_2 = 1 / (f_{\text{suppress}} \tau)^2`.

    Args:
        duration (float): Pulse duration in seconds (s).
        amp (float): Pulse amplitude (peak AWG amplitude V_0 in [-1.0, 1.0]).
        alpha (float): Transmon anharmonicity parameter in Hz (default: -250.0e6 Hz).
        f_suppress (float): Frequency parameter in Hz for higher-level suppressed transition
            (default: 90.0e6 Hz).
        phase (float): Carrier phase offset in radians.
        detune (float): Detuning in Hz.
        noise_sigma (float): Standard deviation of additive noise. Default: 0.0.
        noise_alpha (float): Exponent for noise PSD S(f) ~ (1/f)^alpha. Default: 0.0.
        scale_noise (bool): Whether to scale additive noise by amplitude. Default: False.
        name (Optional[str]): Pulse name.
        length (Optional[float]): Alias for duration in seconds (s).
    """

    def __init__(
        self,
        duration: Optional[float] = None,
        amp: float = 1.0,
        alpha: float = -250.0e6,
        f_suppress: float = 90.0e6,
        phase: float = 0.0,
        detune: float = 0.0,
        noise_sigma: float = 0.0,
        noise_alpha: float = 0.0,
        scale_noise: bool = False,
        name: Optional[str] = None,
        length: Optional[float] = None,
    ):
        super().__init__(
            duration=duration,
            amp=amp,
            phase=phase,
            detune=detune,
            drag=1.0,
            alpha=alpha,
            noise_sigma=noise_sigma,
            noise_alpha=noise_alpha,
            scale_noise=scale_noise,
            name=name or "CosineHD2DRAGPulse",
            length=length,
        )
        if f_suppress <= 0:
            raise ValueError(f"f_suppress must be positive, got {f_suppress} Hz")
        self.f_suppress = float(f_suppress)
        self.beta2 = 1.0 / ((self.f_suppress * self.duration) ** 2)

    def _cos_sum_Dn(self, t: np.ndarray, n: int) -> np.ndarray:
        tau = self.duration
        cos_t = np.cos(2.0 * np.pi * t / tau)
        cos_2t = np.cos(4.0 * np.pi * t / tau)
        sin_t = np.sin(2.0 * np.pi * t / tau)
        sin_2t = np.sin(4.0 * np.pi * t / tau)

        if n == 0:
            return (3.0 / 8.0) * (1.0 - (4.0 / 3.0) * cos_t + (1.0 / 3.0) * cos_2t)
        elif n == 1:
            return (3.0 / 8.0) * ((4.0 / 3.0) * sin_t - (2.0 / 3.0) * sin_2t)
        elif n == 2:
            return (3.0 / 8.0) * ((4.0 / 3.0) * cos_t - (4.0 / 3.0) * cos_2t)
        elif n == 3:
            return (3.0 / 8.0) * (-(4.0 / 3.0) * sin_t + (8.0 / 3.0) * sin_2t)
        else:
            raise ValueError(f"Derivative order n must be 0, 1, 2, or 3, got {n}")

    def envelope(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        d0 = self._cos_sum_Dn(t, 0)
        d2 = self._cos_sum_Dn(t, 2)
        return d0 + self.beta2 * d2

    def sample(
        self,
        dt: Optional[float] = None,
        alpha: Optional[float] = None,
        noise_sigma: Optional[float] = None,
        noise_alpha: Optional[float] = None,
        scale_noise: Optional[bool] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        eff_dt = self._resolve_dt(dt, default_points=200, max_dt=1e-9)
        n_samples = max(int(np.round(self.duration / eff_dt)) + 1, 2)
        t = np.linspace(0, self.duration, n_samples, endpoint=True)

        i_env = self.envelope(t)
        i_wave = self.amp * i_env

        eff_alpha = self.alpha if self.alpha is not None else (alpha if alpha is not None else -250.0e6)
        d1 = self._cos_sum_Dn(t, 1)
        d3 = self._cos_sum_Dn(t, 3)
        q_env = d1 + self.beta2 * d3
        drag_mult = 1.0 / (self.duration * abs(eff_alpha))
        q_wave = self.amp * drag_mult * q_env

        sigma = self.noise_sigma if noise_sigma is None else float(noise_sigma)
        if sigma > 0.0:
            from .base import generate_powerlaw_noise
            n_alpha = self.noise_alpha if noise_alpha is None else float(noise_alpha)
            do_scale = self.scale_noise if scale_noise is None else bool(scale_noise)
            rng = seed if isinstance(seed, np.random.Generator) else (np.random.default_rng(seed) if seed is not None else np.random.default_rng())
            i_noise = sigma * generate_powerlaw_noise(len(i_wave), alpha=n_alpha, rng=rng)
            q_noise = sigma * generate_powerlaw_noise(len(q_wave), alpha=n_alpha, rng=rng)
            scale = self.amp if do_scale else 1.0
            i_wave += scale * i_noise
            q_wave += scale * q_noise

        c_wave = i_wave + 1j * q_wave
        if self.phase != 0.0:
            c_wave = c_wave * np.exp(1j * self.phase)
        if self.detune != 0.0:
            c_wave = c_wave * np.exp(-2j * np.pi * self.detune * t)

        return t, c_wave


class PhaseModulatedSinPulse(Pulse):
    r"""Phase-modulated sine pulse for adiabatic / parametric state transitions.

    Applies a sine amplitude envelope with a continuous nonlinear time-dependent
    phase modulation:

    .. math::
        f(t) = \sin\left(\frac{\pi t}{\tau}\right) \\
        \phi(t) = -2 (2\pi f_{\text{mod}} \tau) \sin\left(\frac{\pi t}{\tau}\right)

    Args:
        duration (float): Pulse duration in seconds (s).
        amp (float): Pulse amplitude in [-1.0, 1.0].
        mod_freq (float): Phase modulation frequency depth in Hz (default: 50.0e6 Hz).
        phase (float): Static carrier phase offset in radians.
        detune (float): Detuning in Hz.
        noise_sigma (float): Standard deviation of additive noise. Default: 0.0.
        noise_alpha (float): Exponent for noise PSD S(f) ~ (1/f)^alpha. Default: 0.0.
        scale_noise (bool): Whether to scale additive noise by amplitude. Default: False.
        name (Optional[str]): Pulse name.
        length (Optional[float]): Alias for duration in seconds (s).
    """

    def __init__(
        self,
        duration: Optional[float] = None,
        amp: float = 1.0,
        mod_freq: float = 50.0e6,
        phase: float = 0.0,
        detune: float = 0.0,
        noise_sigma: float = 0.0,
        noise_alpha: float = 0.0,
        scale_noise: bool = False,
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
            noise_sigma=noise_sigma,
            noise_alpha=noise_alpha,
            scale_noise=scale_noise,
            name=name,
            length=length,
        )
        self.mod_freq = float(mod_freq)

    def envelope(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        theta = np.pi * t / self.duration
        return np.sin(theta)

    def sample(
        self,
        dt: Optional[float] = None,
        alpha: Optional[float] = None,
        noise_sigma: Optional[float] = None,
        noise_alpha: Optional[float] = None,
        scale_noise: Optional[bool] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        eff_dt = self._resolve_dt(dt, default_points=200, max_dt=1e-9)
        n_samples = max(int(np.round(self.duration / eff_dt)) + 1, 2)
        t = np.linspace(0, self.duration, n_samples, endpoint=True)

        theta = np.pi * t / self.duration
        env = np.sin(theta)
        mod_phase = -2.0 * (2.0 * np.pi * self.mod_freq * self.duration) * np.sin(theta)

        c_wave = self.amp * env * np.exp(1j * mod_phase)
        i_wave = c_wave.real
        q_wave = c_wave.imag

        sigma = self.noise_sigma if noise_sigma is None else float(noise_sigma)
        if sigma > 0.0:
            from .base import generate_powerlaw_noise
            n_alpha = self.noise_alpha if noise_alpha is None else float(noise_alpha)
            do_scale = self.scale_noise if scale_noise is None else bool(scale_noise)
            rng = seed if isinstance(seed, np.random.Generator) else (np.random.default_rng(seed) if seed is not None else np.random.default_rng())
            i_noise = sigma * generate_powerlaw_noise(len(i_wave), alpha=n_alpha, rng=rng)
            q_noise = sigma * generate_powerlaw_noise(len(q_wave), alpha=n_alpha, rng=rng)
            scale = self.amp if do_scale else 1.0
            i_wave += scale * i_noise
            q_wave += scale * q_noise

        c_wave = i_wave + 1j * q_wave
        if self.phase != 0.0:
            c_wave = c_wave * np.exp(1j * self.phase)
        if self.detune != 0.0:
            c_wave = c_wave * np.exp(-2j * np.pi * self.detune * t)

        return t, c_wave


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
        noise_sigma: float = 0.0,
        noise_alpha: float = 0.0,
        scale_noise: bool = False,
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
            noise_sigma=noise_sigma,
            noise_alpha=noise_alpha,
            scale_noise=scale_noise,
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
        noise_sigma: float = 0.0,
        noise_alpha: float = 0.0,
        scale_noise: bool = False,
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
            noise_sigma=noise_sigma,
            noise_alpha=noise_alpha,
            scale_noise=scale_noise,
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
            noise_sigma=base_pulse.noise_sigma,
            noise_alpha=base_pulse.noise_alpha,
            scale_noise=base_pulse.scale_noise,
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


def idle_pulse(
    duration: Optional[float] = None,
    phase: float = 0.0,
    detune: float = 0.0,
    name: Optional[str] = None,
    length: Optional[float] = None,
) -> IdlePulse:
    """Create an idle (delay / wait) pulse where amplitude is always zero."""
    return IdlePulse(
        duration=duration,
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
    overshoot_amp: float = 0.0,
    overshoot_len: Optional[float] = None,
    sigma: Optional[float] = None,
    drag: float = 0.0,
    alpha: Optional[float] = None,
    phase: float = 0.0,
    detune: float = 0.0,
    noise_sigma: float = 0.0,
    noise_alpha: float = 0.0,
    scale_noise: bool = False,
    name: Optional[str] = None,
    length: Optional[float] = None,
) -> FlatTopPulse:
    """Create a flat-top pulse with smooth ramp edges and optional pre-distortion overshoot."""
    return FlatTopPulse(
        duration=duration,
        amp=amp,
        ramp_time=ramp_time,
        ramp_type=ramp_type,
        overshoot_amp=overshoot_amp,
        overshoot_len=overshoot_len,
        sigma=sigma,
        drag=drag,
        alpha=alpha,
        phase=phase,
        detune=detune,
        noise_sigma=noise_sigma,
        noise_alpha=noise_alpha,
        scale_noise=scale_noise,
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
    noise_sigma: float = 0.0,
    noise_alpha: float = 0.0,
    scale_noise: bool = False,
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
        noise_sigma=noise_sigma,
        noise_alpha=noise_alpha,
        scale_noise=scale_noise,
        name=name,
        length=length,
    )


def slepian_pulse(
    duration: Optional[float] = None,
    amp: float = 1.0,
    nw: float = 3.0,
    drag: float = 0.0,
    alpha: Optional[float] = None,
    phase: float = 0.0,
    detune: float = 0.0,
    zero_offset: bool = True,
    noise_sigma: float = 0.0,
    noise_alpha: float = 0.0,
    scale_noise: bool = False,
    name: Optional[str] = None,
    length: Optional[float] = None,
) -> SlepianPulse:
    """Create a Slepian (DPSS) window pulse."""
    return SlepianPulse(
        duration=duration,
        amp=amp,
        nw=nw,
        drag=drag,
        alpha=alpha,
        phase=phase,
        detune=detune,
        zero_offset=zero_offset,
        noise_sigma=noise_sigma,
        noise_alpha=noise_alpha,
        scale_noise=scale_noise,
        name=name,
        length=length,
    )


def net_zero_pulse(
    duration: Optional[float] = None,
    amp: float = 1.0,
    sigma: Optional[float] = None,
    ramp_type: str = "erf",
    overshoot_amp: float = 0.0,
    overshoot_len: Optional[float] = None,
    phase: float = 0.0,
    detune: float = 0.0,
    noise_sigma: float = 0.0,
    noise_alpha: float = 0.0,
    scale_noise: bool = False,
    name: Optional[str] = None,
    length: Optional[float] = None,
) -> NetZeroPulse:
    """Create a bipolar Net-Zero flux pulse for two-qubit gates."""
    return NetZeroPulse(
        duration=duration,
        amp=amp,
        sigma=sigma,
        ramp_type=ramp_type,
        overshoot_amp=overshoot_amp,
        overshoot_len=overshoot_len,
        phase=phase,
        detune=detune,
        noise_sigma=noise_sigma,
        noise_alpha=noise_alpha,
        scale_noise=scale_noise,
        name=name,
        length=length,
    )


def cosine_hd2_drag_pulse(
    duration: Optional[float] = None,
    amp: float = 1.0,
    alpha: float = -250.0e6,
    f_suppress: float = 90.0e6,
    phase: float = 0.0,
    detune: float = 0.0,
    noise_sigma: float = 0.0,
    noise_alpha: float = 0.0,
    scale_noise: bool = False,
    name: Optional[str] = None,
    length: Optional[float] = None,
) -> CosineHD2DRAGPulse:
    """Create a High-Derivative (HD) DRAG pulse with multi-derivative leakage suppression."""
    return CosineHD2DRAGPulse(
        duration=duration,
        amp=amp,
        alpha=alpha,
        f_suppress=f_suppress,
        phase=phase,
        detune=detune,
        noise_sigma=noise_sigma,
        noise_alpha=noise_alpha,
        scale_noise=scale_noise,
        name=name,
        length=length,
    )


def phase_modulated_sin_pulse(
    duration: Optional[float] = None,
    amp: float = 1.0,
    mod_freq: float = 50.0e6,
    phase: float = 0.0,
    detune: float = 0.0,
    noise_sigma: float = 0.0,
    noise_alpha: float = 0.0,
    scale_noise: bool = False,
    name: Optional[str] = None,
    length: Optional[float] = None,
) -> PhaseModulatedSinPulse:
    """Create a phase-modulated sine pulse for adiabatic / parametric transitions."""
    return PhaseModulatedSinPulse(
        duration=duration,
        amp=amp,
        mod_freq=mod_freq,
        phase=phase,
        detune=detune,
        noise_sigma=noise_sigma,
        noise_alpha=noise_alpha,
        scale_noise=scale_noise,
        name=name,
        length=length,
    )


