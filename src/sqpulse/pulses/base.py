"""Base Pulse class and waveform primitives for SQPulse in SI units."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Callable, Optional, Union
import numpy as np
import matplotlib.pyplot as plt


class Pulse(ABC):
    """Abstract base class representing a physical microwave or RF pulse in SI units.

    Attributes:
        duration (float): Pulse duration in seconds (s).
        amp (float): Peak amplitude scaling (arbitrary units or rad/s).
        phase (float): Carrier phase offset in radians.
        detune (float): Frequency detuning offset in Hz (df in exp(-i * 2pi * df * t)).
        drag (float): DRAG coefficient for imaginary quadrature (Q = drag * dI/dt).
        name (str): Optional identifier for the pulse.
    """

    def __init__(
        self,
        duration: float,
        amp: float = 1.0,
        phase: float = 0.0,
        detune: float = 0.0,
        drag: float = 0.0,
        name: Optional[str] = None,
    ):
        if duration <= 0:
            raise ValueError(f"Pulse duration must be positive, got {duration} s")
        self.duration = float(duration)
        self.amp = float(amp)
        self.phase = float(phase)
        self.detune = float(detune)
        self.drag = float(drag)
        self.name = name or self.__class__.__name__

    @abstractmethod
    def envelope(self, t: np.ndarray) -> np.ndarray:
        """Calculate the real normalized envelope f(t) in [0, 1] for time array t in [0, duration].

        Args:
            t: 1D numpy array of time points relative to pulse start (0 <= t <= duration, in seconds).

        Returns:
            1D numpy array of real envelope values.
        """
        pass

    def envelope_derivative(self, t: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        """Calculate the numerical derivative df/dt of the envelope.

        Args:
            t: 1D numpy array of time points in seconds.
            dt: Differentiation step in seconds. If None, defaults to min(duration * 1e-3, 1e-12).
        """
        if dt is None:
            dt = min(self.duration * 1e-3, 1e-12)

        t_plus = np.clip(t + dt, 0, self.duration)
        t_minus = np.clip(t - dt, 0, self.duration)
        delta_t = t_plus - t_minus
        delta_t[delta_t == 0] = 1.0
        return (self.envelope(t_plus) - self.envelope(t_minus)) / delta_t

    def sample(self, dt: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
        """Sample the complex baseband waveform Omega(t) = I(t) + i*Q(t).

        Args:
            dt: Sampling interval in seconds (default 1e-9 s = 1 ns).

        Returns:
            t: 1D numpy array of sample times [0, dt, 2*dt, ..., duration] in seconds.
            c_wave: 1D complex numpy array of sampled pulse values.
        """
        n_samples = max(int(np.round(self.duration / dt)) + 1, 2)
        t = np.linspace(0, self.duration, n_samples, endpoint=True)

        env = self.envelope(t)
        i_wave = self.amp * env

        if self.drag != 0.0:
            d_env = self.envelope_derivative(t, dt=min(dt * 0.1, self.duration * 1e-3))
            q_wave = self.amp * self.drag * d_env
        else:
            q_wave = np.zeros_like(i_wave)

        c_wave = i_wave + 1j * q_wave

        # Apply carrier phase offset
        if self.phase != 0.0:
            c_wave = c_wave * np.exp(1j * self.phase)

        # Apply software detuning: exp(-i * 2pi * detune * t) (detune in Hz, t in s)
        if self.detune != 0.0:
            c_wave = c_wave * np.exp(-2j * np.pi * self.detune * t)

        return t, c_wave

    def fft(
        self,
        dt: float = 1e-10,
        pad_factor: int = 4,
        window: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the Fast Fourier Transform (FFT) and Power Spectral Density (PSD).

        Args:
            dt: Sampling time step in seconds (default 1e-10 s = 0.1 ns).
            pad_factor: Zero-padding multiplier to improve spectral frequency resolution.
            window: Whether to apply a Hann window prior to FFT (default False).

        Returns:
            freqs: Frequency axis in Hz, centered at 0 (from -f_Nyquist to +f_Nyquist).
            spec: Complex frequency spectrum.
            psd_dB: Normalized Power Spectral Density in dB (peak at 0 dB).
        """
        t, c_wave = self.sample(dt=dt)
        n = len(c_wave)
        n_fft = int(n * pad_factor)

        if window:
            c_wave = c_wave * np.hanning(n)

        # Zero-pad symmetrically to center time domain if desired
        padded_wave = np.zeros(n_fft, dtype=complex)
        start_idx = (n_fft - n) // 2
        padded_wave[start_idx : start_idx + n] = c_wave

        spec = np.fft.fftshift(np.fft.fft(padded_wave) * dt)
        freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=dt))  # in Hz (since dt is in s)

        # PSD in dB
        magnitude_sq = np.abs(spec) ** 2
        max_val = np.max(magnitude_sq)
        if max_val > 0:
            psd_norm = magnitude_sq / max_val
            psd_dB = 10 * np.log10(np.clip(psd_norm, 1e-10, 1.0))
        else:
            psd_dB = np.zeros_like(magnitude_sq)

        return freqs, spec, psd_dB

    def spectral_bandwidth(self, dt: float = 1e-10, threshold_dB: float = -3.0) -> float:
        """Calculate the frequency bandwidth (e.g. -3 dB FWHM) in Hz."""
        freqs, _, psd_dB = self.fft(dt=dt)
        mask = psd_dB >= threshold_dB
        if np.any(mask):
            return float(freqs[mask].max() - freqs[mask].min())
        return 0.0

    def plot_time(
        self,
        dt: float = 5e-10,
        ax: Optional[plt.Axes] = None,
        show_envelope: bool = True,
        title: Optional[str] = None,
    ) -> plt.Axes:
        """Plot the pulse waveform in the time domain."""
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))

        t, c_wave = self.sample(dt=dt)
        ax.plot(t, c_wave.real, label="I (In-phase / Real)", color="#1f77b4", lw=2)
        ax.plot(t, c_wave.imag, label="Q (Quadrature / Imag)", color="#ff7f0e", lw=1.8, ls="--")

        if show_envelope:
            ax.plot(t, np.abs(c_wave), label="|Envelope|", color="#2ca02c", lw=1.2, ls=":")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(title or f"{self.name} - Time Domain (T={self.duration:.2e} s)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        return ax

    def plot_freq(
        self,
        dt: float = 1e-10,
        freq_range: Optional[Tuple[float, float]] = None,
        ax: Optional[plt.Axes] = None,
        log_scale: bool = True,
        title: Optional[str] = None,
    ) -> plt.Axes:
        """Plot the pulse spectrum in the frequency domain.

        Args:
            dt: Sampling step in seconds.
            freq_range: Tuple of (min_freq, max_freq) in Hz. Defaults to auto.
            ax: Optional matplotlib axes.
            log_scale: If True, plot in dB; otherwise linear magnitude.
            title: Optional plot title.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))

        freqs, spec, psd_dB = self.fft(dt=dt)

        if log_scale:
            ax.plot(freqs, psd_dB, color="#9467bd", lw=2, label="PSD (dB)")
            ax.set_ylabel("Power Spectral Density (dB)")
            ax.set_ylim(-80, 5)
            ax.axhline(-3, color="gray", ls="--", alpha=0.6, label="-3 dB bandwidth")
        else:
            mag = np.abs(spec)
            norm_mag = mag / np.max(mag) if np.max(mag) > 0 else mag
            ax.plot(freqs, norm_mag, color="#9467bd", lw=2, label="|Spectrum|")
            ax.set_ylabel("Normalized Magnitude")
            ax.set_ylim(0, 1.05)

        if freq_range is not None:
            ax.set_xlim(freq_range)
        else:
            # Auto zoom to main lobe region
            fwhm = self.spectral_bandwidth(dt=dt, threshold_dB=-20)
            if fwhm > 0:
                ax.set_xlim(-3 * fwhm, 3 * fwhm)

        ax.set_xlabel("Frequency Detuning (Hz)")
        ax.set_title(title or f"{self.name} - Frequency Domain")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        return ax

    def plot(
        self,
        domain: str = "both",
        dt: float = 5e-10,
        freq_range: Optional[Tuple[float, float]] = None,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> plt.Figure:
        """Convenience method to plot time-domain, frequency-domain, or both side-by-side.

        Args:
            domain: 'time', 'freq', or 'both'.
            dt: Time sampling in seconds.
            freq_range: Optional (f_min, f_max) in Hz for frequency plot.
            figsize: Figure size.
        """
        if domain == "time":
            fig, ax = plt.subplots(figsize=figsize or (6, 4))
            self.plot_time(dt=dt, ax=ax)
            return fig
        elif domain == "freq":
            fig, ax = plt.subplots(figsize=figsize or (6, 4))
            self.plot_freq(dt=dt, freq_range=freq_range, ax=ax)
            return fig
        elif domain == "both":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize or (12, 4.5))
            self.plot_time(dt=dt, ax=ax1)
            self.plot_freq(dt=dt, freq_range=freq_range, ax=ax2)
            plt.tight_layout()
            return fig
        else:
            raise ValueError(f"Unknown domain: {domain}, expected 'time', 'freq', or 'both'")

    def __mul__(self, scalar: Union[float, int]) -> Pulse:
        """Scale amplitude by scalar: p2 = p1 * 2.0."""
        from .shapes import ScaledPulse
        return ScaledPulse(self, scalar=float(scalar))

    def __rmul__(self, scalar: Union[float, int]) -> Pulse:
        return self.__mul__(scalar)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(duration={self.duration:.2e}s, "
            f"amp={self.amp:.3e}, phase={self.phase:.2f}, detune={self.detune:.2e}Hz, "
            f"drag={self.drag:.2e})"
        )
