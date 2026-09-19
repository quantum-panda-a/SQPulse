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

    The steady-state transmission scattering parameter S_21(f) depends on the geometry:
    - For Hanger (Notch / Dip) geometry (default):
      .. math::
          S_{21}(f) = 1 - \frac{\kappa_{\text{ext}} / 2}{i 2\pi (f - f_r^{(n)}) + \kappa / 2}
    - For Transmission (Through / Peak) geometry:
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
        geometry (str): Resonator microwave coupling geometry / structure type.
            Options: 'hanger' (default, produces a dip / notch) or 'transmission' (produces a peak).
            Aliases 'dip', 'notch' -> 'hanger'; 'peak', 'through' -> 'transmission'.
        structure_type (Optional[str]): Alias for geometry.
    """

    def __init__(
        self,
        name: str = "r0",
        f_r: float = 7.0e9,
        kappa: float = 2.5e6,
        chi: Optional[float] = None,
        kappa_ext: Optional[float] = None,
        g: Optional[float] = None,
        geometry: str = "hanger",
        structure_type: Optional[str] = None,
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

        raw_geom = structure_type if structure_type is not None else geometry
        geom_str = str(raw_geom).lower().strip()
        if geom_str in ("hanger", "notch", "dip", "shunt"):
            self.geometry = "hanger"
        elif geom_str in ("transmission", "through", "peak", "inline"):
            self.geometry = "transmission"
        else:
            raise ValueError(
                f"Unknown resonator geometry '{raw_geom}'. "
                f"Supported options: 'hanger' ('dip') and 'transmission' ('peak')."
            )

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

    @property
    def structure_type(self) -> str:
        """Structure type of the resonator ('hanger' for dip, 'transmission' for peak)."""
        return self.geometry

    @structure_type.setter
    def structure_type(self, val: str) -> None:
        geom_str = str(val).lower().strip()
        if geom_str in ("hanger", "notch", "dip", "shunt"):
            self.geometry = "hanger"
        elif geom_str in ("transmission", "through", "peak", "inline"):
            self.geometry = "transmission"
        else:
            raise ValueError(
                f"Unknown resonator structure_type '{val}'. "
                f"Supported options: 'hanger' ('dip') and 'transmission' ('peak')."
            )

    @property
    def is_dip(self) -> bool:
        """True if resonator operates in hanger/notch (dip) mode."""
        return self.geometry == "hanger"

    @property
    def is_peak(self) -> bool:
        """True if resonator operates in transmission (peak) mode."""
        return self.geometry == "transmission"

    def _compute_s21(
        self,
        delta_omega: np.ndarray,
        geometry: Optional[str] = None,
    ) -> np.ndarray:
        """Internal helper to compute complex S21 given detuning delta_omega = 2*pi*(f - f_eff)."""
        denominator = 1j * delta_omega + 0.5 * self.kappa
        geom = (geometry or self.geometry).lower().strip()
        if geom in ("hanger", "notch", "dip", "shunt"):
            # Hanger / Notch mode: S21(f) = 1 - (kappa_ext / 2) / (i * delta_omega + kappa / 2) -> Dip
            return 1.0 - (0.5 * self.kappa_ext) / denominator
        elif geom in ("transmission", "through", "peak", "inline"):
            # Transmission mode: S21(f) = kappa_ext / (i * delta_omega + kappa / 2) -> Peak
            return self.kappa_ext / denominator
        else:
            raise ValueError(
                f"Unknown resonator geometry '{geometry or self.geometry}'. "
                f"Supported options: 'hanger' ('dip') and 'transmission' ('peak')."
            )

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

    def s21(
        self,
        freqs: np.ndarray,
        qubit_state: int = 0,
        geometry: Optional[str] = None,
    ) -> np.ndarray:
        """Compute complex transmission coefficient S_21(f) across probe frequencies.

        Args:
            freqs: 1D array of probe frequencies in Hz.
            qubit_state: Transmon level |n> (0 or 1).
            geometry: Optional override of resonator geometry ('hanger' for dip, 'transmission' for peak).

        Returns:
            1D complex array of S_21 transmission values.
        """
        freqs = np.asarray(freqs, dtype=float)
        f_eff = self.effective_frequency(qubit_state)
        delta_omega = 2.0 * np.pi * (freqs - f_eff)
        return self._compute_s21(delta_omega, geometry=geometry)

    def effective_frequency_at_flux(
        self,
        transmon: Any,
        flux: float,
        g: Optional[float] = None,
        qubit_state: int = 0,
    ) -> float:
        """Calculate dressed cavity frequency when coupled to a transmon at a given magnetic flux (in units of Phi_0).

        Args:
            transmon: Transmon qubit physical model.
            flux: Magnetic flux in units of Phi_0.
            g: Coupling strength in Hz. Defaults to self.g_hz (or 50 MHz if None).
            qubit_state: Transmon Fock state (0 for ground state, 1 for excited state).

        Returns:
            Dressed resonator frequency in Hz.
        """
        from ..units import parse_quantity
        f_q_phi = transmon.frequency_at_flux(flux)
        g_val = float(parse_quantity(g)) if g is not None else (self.g_hz if self.g_hz is not None else 50.0e6)

        # Detuning: Delta = f_q(Phi) - f_r
        delta = f_q_phi - self.f_r
        if abs(delta) < 1e6:
            delta = -1e6

        # Dispersive Lamb shift on ground state: delta_f0 = - g^2 / (f_q - f_r) = g^2 / (f_r - f_q)
        delta_f0 = - (g_val ** 2) / delta

        if qubit_state == 0:
            return self.f_r + delta_f0
        elif qubit_state == 1:
            alpha = transmon.alpha
            denom = delta * (delta + alpha)
            if abs(denom) < 1e6:
                denom = -1e6
            two_chi = -2.0 * (g_val ** 2) * alpha / denom
            return self.f_r + delta_f0 + two_chi
        else:
            return self.f_r + delta_f0

    def s21_at_flux(
        self,
        freqs: np.ndarray,
        transmon: Any,
        flux: float,
        g: Optional[float] = None,
        qubit_state: int = 0,
        geometry: Optional[str] = None,
    ) -> np.ndarray:
        """Compute complex S_21 transmission coefficient across probe frequencies at a given flux bias."""
        freqs = np.asarray(freqs, dtype=float)
        f_eff = self.effective_frequency_at_flux(transmon, flux=flux, g=g, qubit_state=qubit_state)
        delta_omega = 2.0 * np.pi * (freqs - f_eff)
        return self._compute_s21(delta_omega, geometry=geometry)

    def s21_vs_flux(
        self,
        freqs: np.ndarray,
        transmon: Any,
        fluxes: np.ndarray,
        g: Optional[float] = None,
        qubit_state: int = 0,
        geometry: Optional[str] = None,
    ) -> np.ndarray:
        """Compute 2D array of S_21 transmission coefficients across probe frequencies and flux biases.

        Returns:
            2D complex numpy array with shape (len(freqs), len(fluxes)).
        """
        freqs = np.asarray(freqs, dtype=float)
        fluxes = np.asarray(fluxes, dtype=float)
        grid = np.zeros((len(freqs), len(fluxes)), dtype=complex)
        for j, phi in enumerate(fluxes):
            grid[:, j] = self.s21_at_flux(
                freqs, transmon, flux=phi, g=g, qubit_state=qubit_state, geometry=geometry
            )
        return grid

    def plot_s21_vs_flux(
        self,
        freqs: np.ndarray,
        transmon: Any,
        fluxes: np.ndarray,
        g: Optional[float] = None,
        qubit_state: int = 0,
        geometry: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 4.5),
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot 2D colormap of S_21 transmission magnitude (dB) and phase vs probe frequency and flux bias."""
        s21_mat = self.s21_vs_flux(
            freqs, transmon, fluxes, g=g, qubit_state=qubit_state, geometry=geometry
        )
        mag_db = 20.0 * np.log10(np.abs(s21_mat) + 1e-15)
        phase_deg = np.degrees(np.angle(s21_mat))

        fig, (ax_mag, ax_phase) = plt.subplots(1, 2, figsize=figsize)
        extent = [fluxes[0], fluxes[-1], freqs[0] / 1e9, freqs[-1] / 1e9]

        im1 = ax_mag.imshow(
            mag_db,
            extent=extent,
            origin="lower",
            aspect="auto",
            cmap="viridis",
        )
        ax_mag.set_xlabel(r"Magnetic Flux $\Phi / \Phi_0$")
        ax_mag.set_ylabel("Probe Frequency (GHz)")
        ax_mag.set_title(r"Transmission $|S_{21}|$ (dB)")
        fig.colorbar(im1, ax=ax_mag, label="dB")

        im2 = ax_phase.imshow(
            phase_deg,
            extent=extent,
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
        )
        ax_phase.set_xlabel(r"Magnetic Flux $\Phi / \Phi_0$")
        ax_phase.set_ylabel("Probe Frequency (GHz)")
        ax_phase.set_title(r"Transmission Phase (deg)")
        fig.colorbar(im2, ax=ax_phase, label="deg")

        geom_label = "Dip (Hanger)" if (geometry or self.geometry) in ("hanger", "notch", "dip", "shunt") else "Peak (Transmission)"
        fig.suptitle(
            title or f"Resonator Flux Arc [{geom_label}]: {self.name} coupled to {transmon.name} (|{qubit_state}⟩)",
            fontsize=12,
        )
        fig.tight_layout()
        return fig

    def plot_s21(
        self,
        freq_sweep: np.ndarray,
        states: List[int] = [0, 1],
        geometry: Optional[str] = None,
        figsize: Tuple[int, int] = (9, 4),
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot S_21 transmission magnitude (dB) and phase for different transmon states.

        Args:
            freq_sweep: 1D array of probe frequencies in Hz.
            states: Transmon states to compare (default [0, 1]).
            geometry: Optional override of resonator geometry.
            figsize: Matplotlib figure size tuple.
            title: Optional custom title.

        Returns:
            Matplotlib Figure.
        """
        fig, (ax_mag, ax_phase) = plt.subplots(1, 2, figsize=figsize)
        colors = {0: "#1f77b4", 1: "#d62728", 2: "#2ca02c"}

        freq_ghz = np.asarray(freq_sweep) / 1e9

        for st in states:
            s21_val = self.s21(freq_sweep, qubit_state=st, geometry=geometry)
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
            f"ReadoutResonator('{self.name}', geometry='{self.geometry}', f_r={self.f_r:.3e}Hz, "
            f"kappa={self.kappa_hz/1e6:.2f}MHz, chi={self.chi_hz/1e6:.2f}MHz)"
        )

    def to_dict(self, human_readable: bool = False) -> dict:
        """Serialize ReadoutResonator configuration to a dictionary."""
        return {
            "name": self.name,
            "geometry": self.geometry,
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
            - 'geometry' / 'structure_type': str ('hanger' / 'dip' or 'transmission' / 'peak')
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
        geometry = data.get("geometry", data.get("structure_type", "hanger"))
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
            geometry=geometry,
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

