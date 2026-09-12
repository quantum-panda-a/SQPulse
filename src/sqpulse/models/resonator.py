"""Readout resonator model for circuit QED dispersive measurement in SI units."""

from __future__ import annotations
from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt


class ReadoutResonator:
    r"""Physical model of a microwave readout resonator (e.g. CPW or 3D cavity) coupled to a Transmon.

    In the dispersive regime (|Delta| = |omega_q - omega_r| >> g), the cavity frequency
    shifts depending on the state |n> of the transmon by the dispersive shift chi:
    .. math::
        f_r^{(0)} = f_r - \chi / (2\pi)
        f_r^{(1)} = f_r + \chi / (2\pi)

    The steady-state transmission scattering parameter S_21(f) is:
    .. math::
        S_{21}(f) = \frac{\kappa_{\text{ext}}}{i 2\pi (f - f_r^{(n)}) + \kappa / 2}

    Args:
        name (str): Identifier for this resonator (e.g. 'r0').
        f_r (float): Bare resonance frequency in Hz (e.g. 7.0e9 for 7 GHz).
        kappa (float): Total cavity decay rate / linewidth in rad/s (default 2*pi * 2.5 MHz).
        chi (Optional[float]): Dispersive frequency shift in rad/s (default 2*pi * 1.2 MHz).
            If None and g is provided, calculated from g and transmon detuning.
        kappa_ext (Optional[float]): External coupling decay rate to feedline in rad/s.
            Defaults to kappa / 2 (critically coupled / symmetrical port).
        g (Optional[float]): Transmon-resonator capacitive coupling strength in rad/s.
    """

    def __init__(
        self,
        name: str = "r0",
        f_r: float = 7.0e9,
        kappa: float = 2.0 * np.pi * 2.5e6,
        chi: Optional[float] = None,
        kappa_ext: Optional[float] = None,
        g: Optional[float] = None,
    ):
        self.name = name
        self.f_r = float(f_r)
        self.kappa = float(kappa)
        self.kappa_ext = float(kappa_ext) if kappa_ext is not None else 0.5 * self.kappa
        self.g = float(g) if g is not None else None

        if chi is not None:
            self.chi = float(chi)
        else:
            # Default dispersive shift: 2*pi * 1.2 MHz (rad/s)
            self.chi = 2.0 * np.pi * 1.2e6

    @property
    def kappa_hz(self) -> float:
        """Total linewidth in Hz: kappa / (2*pi)."""
        return self.kappa / (2.0 * np.pi)

    @property
    def chi_hz(self) -> float:
        """Dispersive shift in Hz: chi / (2*pi)."""
        return self.chi / (2.0 * np.pi)

    def effective_frequency(self, qubit_state: int = 0) -> float:
        """Return the effective cavity frequency (in Hz) when transmon is in Fock state |n>.

        For state |0>: f_r - chi_hz
        For state |1>: f_r + chi_hz
        Frequency separation is 2 * chi_hz.
        """
        if qubit_state == 0:
            return self.f_r - self.chi_hz
        elif qubit_state == 1:
            return self.f_r + self.chi_hz
        else:
            # Higher state approximation
            return self.f_r + (2 * qubit_state - 1) * self.chi_hz

    def s21(self, freqs: np.ndarray, qubit_state: int = 0) -> np.ndarray:
        """Compute complex transmission coefficient S_21(f) across probe frequencies.

        Args:
            freqs: 1D array of probe frequencies in Hz.
            qubit_state: Transmon level |n> (0 or 1).

        Returns:
            1D complex array of S_21 transmission values.
        """
        freqs = np.asarray(freqs, dtype=float)
        f_eff = self.effective_frequency(qubit_state)
        delta_omega = 2.0 * np.pi * (freqs - f_eff)
        denominator = 1j * delta_omega + 0.5 * self.kappa
        return self.kappa_ext / denominator

    def plot_s21(
        self,
        freq_sweep: np.ndarray,
        states: List[int] = [0, 1],
        figsize: Tuple[int, int] = (9, 4),
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot S_21 transmission magnitude (dB) and phase for different transmon states.

        Args:
            freq_sweep: 1D array of probe frequencies in Hz.
            states: Transmon states to compare (default [0, 1]).
            figsize: Matplotlib figure size tuple.
            title: Optional custom title.

        Returns:
            Matplotlib Figure.
        """
        fig, (ax_mag, ax_phase) = plt.subplots(1, 2, figsize=figsize)
        colors = {0: "#1f77b4", 1: "#d62728", 2: "#2ca02c"}

        freq_ghz = np.asarray(freq_sweep) / 1e9

        for st in states:
            s21_val = self.s21(freq_sweep, qubit_state=st)
            mag_db = 20.0 * np.log10(np.abs(s21_val) + 1e-15)
            phase_rad = np.angle(s21_val)

            c = colors.get(st, "#333333")
            lbl = f"Qubit |{st}⟩ (f_r = {self.effective_frequency(st)/1e9:.4f} GHz)"
            ax_mag.plot(freq_ghz, mag_db, label=lbl, color=c, lw=2)
            ax_phase.plot(freq_ghz, np.degrees(phase_rad), label=lbl, color=c, lw=2)

        ax_mag.set_xlabel("Frequency (GHz)")
        ax_mag.set_ylabel("|S₂₁| (dB)")
        ax_mag.set_title("Transmission Magnitude")
        ax_mag.grid(True, alpha=0.3)
        ax_mag.legend(fontsize=9)

        ax_phase.set_xlabel("Frequency (GHz)")
        ax_phase.set_ylabel("Phase (deg)")
        ax_phase.set_title("Transmission Phase")
        ax_phase.grid(True, alpha=0.3)

        fig.suptitle(title or f"{self.name} - Dispersive Shift 2χ = {2 * self.chi_hz / 1e6:.2f} MHz", fontsize=11)
        fig.tight_layout()
        return fig

    def __repr__(self) -> str:
        return (
            f"ReadoutResonator('{self.name}', f_r={self.f_r:.3e}Hz, "
            f"kappa/(2pi)={self.kappa_hz/1e6:.2f}MHz, chi/(2pi)={self.chi_hz/1e6:.2f}MHz)"
        )
