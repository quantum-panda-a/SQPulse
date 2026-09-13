"""Readout resonator model for circuit QED dispersive measurement in SI units."""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Tuple, Union
import json
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
        kappa (float): Total cavity decay rate / linewidth in Hz (default 2.5 MHz).
            Automatically multiplied by 2*pi internally.
        chi (Optional[float]): Dispersive frequency shift in Hz (default 1.2 MHz).
            Automatically multiplied by 2*pi internally.
        kappa_ext (Optional[float]): External coupling decay rate to feedline in Hz.
            Defaults to kappa / 2 (critically coupled / symmetrical port).
        g (Optional[float]): Transmon-resonator capacitive coupling strength in Hz.
    """

    def __init__(
        self,
        name: str = "r0",
        f_r: float = 7.0e9,
        kappa: float = 2.5e6,
        chi: Optional[float] = None,
        kappa_ext: Optional[float] = None,
        g: Optional[float] = None,
    ):
        from ..units import parse_quantity

        self.name = name
        self.f_r = float(parse_quantity(f_r))
        self.kappa = 2.0 * np.pi * float(parse_quantity(kappa))
        self.kappa_ext = (
            2.0 * np.pi * float(parse_quantity(kappa_ext))
            if kappa_ext is not None
            else 0.5 * self.kappa
        )
        self.g = 2.0 * np.pi * float(parse_quantity(g)) if g is not None else None

        if chi is not None:
            self.chi = 2.0 * np.pi * float(parse_quantity(chi))
        else:
            # Default dispersive shift: 1.2 MHz (rad/s: 2*pi * 1.2 MHz)
            self.chi = 2.0 * np.pi * 1.2e6

    @property
    def kappa_hz(self) -> float:
        """Total linewidth in Hz: kappa / (2*pi)."""
        return self.kappa / (2.0 * np.pi)

    @property
    def chi_hz(self) -> float:
        """Dispersive shift in Hz: chi / (2*pi)."""
        return self.chi / (2.0 * np.pi)

    @property
    def kappa_ext_hz(self) -> Optional[float]:
        """External coupling decay rate in Hz: kappa_ext / (2*pi)."""
        return self.kappa_ext / (2.0 * np.pi) if self.kappa_ext is not None else None

    @property
    def g_hz(self) -> Optional[float]:
        """Transmon-resonator capacitive coupling strength in Hz: g / (2*pi)."""
        return self.g / (2.0 * np.pi) if self.g is not None else None

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
            f"kappa={self.kappa_hz/1e6:.2f}MHz, chi={self.chi_hz/1e6:.2f}MHz)"
        )

    def to_dict(self, human_readable: bool = False) -> dict:
        """Serialize ReadoutResonator configuration to a dictionary."""
        return {
            "name": self.name,
            "f_r": f"{self.f_r / 1e9:.6g} GHz" if human_readable else self.f_r,
            "kappa": f"{self.kappa_hz / 1e6:.6g} MHz" if human_readable else self.kappa_hz,
            "chi": f"{self.chi_hz / 1e6:.6g} MHz" if human_readable else self.chi_hz,
            "kappa_ext": (
                f"{self.kappa_ext_hz / 1e6:.6g} MHz"
                if (human_readable and self.kappa_ext is not None)
                else self.kappa_ext_hz
            ),
            "g": (
                f"{self.g_hz / 1e6:.6g} MHz"
                if (human_readable and self.g is not None)
                else self.g_hz
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ReadoutResonator:
        """Construct a ReadoutResonator instance from a dictionary.

        Supports:
            - 'name': str (e.g. 'r0')
            - 'f_r': bare resonance frequency (number in Hz or string like '7.0 GHz')
            - 'kappa': cavity decay rate in Hz (number or string like '2.5 MHz')
            - 'kappa_hz': legacy key for cavity linewidth in Hz
            - 'chi': dispersive shift in Hz (number or string like '1.2 MHz')
            - 'chi_hz': legacy key for dispersive shift in Hz
            - 'kappa_ext': external coupling rate in Hz
            - 'g': coupling strength in Hz
        """
        from ..units import parse_quantity

        name = data.get("name", "r0")
        f_r = parse_quantity(data.get("f_r", 7.0e9))

        raw_kappa = data.get("kappa", data.get("kappa_hz", None))
        if raw_kappa is not None:
            if isinstance(raw_kappa, str) and "rad" in raw_kappa.lower():
                kappa = parse_quantity(raw_kappa) / (2.0 * np.pi)
            else:
                kappa = parse_quantity(raw_kappa)
        else:
            kappa = 2.5e6

        raw_chi = data.get("chi", data.get("chi_hz", None))
        if raw_chi is not None:
            if isinstance(raw_chi, str) and "rad" in raw_chi.lower():
                chi = parse_quantity(raw_chi) / (2.0 * np.pi)
            else:
                chi = parse_quantity(raw_chi)
        else:
            chi = None

        raw_kappa_ext = data.get("kappa_ext", None)
        if raw_kappa_ext is not None:
            if isinstance(raw_kappa_ext, str) and "rad" in raw_kappa_ext.lower():
                kappa_ext = parse_quantity(raw_kappa_ext) / (2.0 * np.pi)
            else:
                kappa_ext = parse_quantity(raw_kappa_ext)
        else:
            kappa_ext = None

        raw_g = data.get("g", None)
        if raw_g is not None:
            if isinstance(raw_g, str) and "rad" in raw_g.lower():
                g = parse_quantity(raw_g) / (2.0 * np.pi)
            else:
                g = parse_quantity(raw_g)
        else:
            g = None

        return cls(
            name=name,
            f_r=f_r,
            kappa=kappa,
            chi=chi,
            kappa_ext=kappa_ext,
            g=g,
        )

    @classmethod
    def from_json(cls, source: Union[str, Path]) -> ReadoutResonator:
        """Load a ReadoutResonator model from a JSON file path or a raw JSON string."""
        p = Path(source) if isinstance(source, (str, Path)) else None
        if p is not None and p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(str(source))

        return cls.from_dict(data)

    def to_json(
        self,
        filepath_or_buf: Optional[Union[str, Path]] = None,
        indent: int = 2,
        human_readable: bool = False,
    ) -> Optional[str]:
        """Serialize ReadoutResonator configuration to a JSON file or JSON string."""
        data = self.to_dict(human_readable=human_readable)
        if filepath_or_buf is not None:
            p = Path(filepath_or_buf)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent)
            return None
        return json.dumps(data, indent=indent)

