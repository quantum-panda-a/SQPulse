"""Qubit spectroscopy experiments (1D Frequency Sweep and 2D Multi-parameter Spectroscopy) in SI units."""

from __future__ import annotations
import warnings
from typing import Optional, Type, Dict, Any, Union, Tuple, List, Sequence
import numpy as np
import matplotlib.pyplot as plt

from ..models.transmon import Transmon
from ..pulses.base import Pulse
from ..pulses.shapes import SquarePulse, FlatTopPulse
from ..sequence.sequence import PulseSequence
from ..measurement import Measurement
from .fitting import fit_spectroscopy_peaks


def _is_multi_value(val: Any) -> bool:
    """Check if a value is a sequence or array with more than one element."""
    if val is None:
        return False
    if isinstance(val, (list, tuple, np.ndarray)):
        return len(val) > 1
    return False


def _to_single_value(val: Any, default: Optional[float] = None) -> Optional[float]:
    """Convert scalar or single-element sequence to float."""
    if val is None:
        return default
    if isinstance(val, (list, tuple, np.ndarray)):
        if len(val) == 0:
            return default
        return float(val[0])
    return float(val)


SWEEP_PARAM_REGISTRY: Dict[str, Dict[str, Any]] = {
    "amp": {"label": "Drive Amplitude V₀", "unit": "AWG V₀", "scale": 1.0, "type": "amplitude"},
    "amps": {"label": "Drive Amplitude V₀", "unit": "AWG V₀", "scale": 1.0, "type": "amplitude"},
    "flux": {"label": "External Flux Bias Φ", "unit": "Φ₀", "scale": 1.0, "type": "flux"},
    "duration": {"label": "Probe Duration", "unit": "ns", "scale": 1e-9, "type": "time"},
    "length": {"label": "Probe Duration", "unit": "ns", "scale": 1e-9, "type": "time"},
    "ramp_time": {"label": "FlatTop Ramp Time", "unit": "ns", "scale": 1e-9, "type": "time"},
    "sigma": {"label": "Gaussian Sigma (σ)", "unit": "ns", "scale": 1e-9, "type": "time"},
    "chop": {"label": "Chop Ratio", "unit": "a.u.", "scale": 1.0, "type": "shape"},
    "drag": {"label": "DRAG Beta (β)", "unit": "a.u.", "scale": 1.0, "type": "shape"},
    "alpha": {"label": "Anharmonicity α", "unit": "MHz", "scale": 1e6, "type": "frequency"},
    "detune": {"label": "Detuning Δ", "unit": "MHz", "scale": 1e6, "type": "frequency"},
    "phase": {"label": "Carrier Phase", "unit": "rad", "scale": 1.0, "type": "shape"},
    "mod_freq": {"label": "Modulation Frequency", "unit": "MHz", "scale": 1e6, "type": "frequency"},
    "omega_0": {"label": "Modulation Frequency (ω₀)", "unit": "MHz", "scale": 1e6, "type": "frequency"},
}


def _get_param_meta(param_name: str) -> Dict[str, Any]:
    """Retrieve metadata for a swept parameter, or generate fallback defaults."""
    if param_name in SWEEP_PARAM_REGISTRY:
        return SWEEP_PARAM_REGISTRY[param_name]
    p_lower = param_name.lower()
    if any(t in p_lower for t in ("time", "dur", "tau", "delay", "sigma", "width")):
        return {"label": param_name, "unit": "ns", "scale": 1e-9, "type": "time"}
    if any(f in p_lower for f in ("freq", "detune", "omega", "bandwidth")):
        return {"label": param_name, "unit": "MHz", "scale": 1e6, "type": "frequency"}
    if "amp" in p_lower or "volt" in p_lower:
        return {"label": param_name, "unit": "AWG V₀", "scale": 1.0, "type": "amplitude"}
    if "flux" in p_lower or "phi" in p_lower:
        return {"label": param_name, "unit": "Φ₀", "scale": 1.0, "type": "flux"}
    return {"label": param_name, "unit": "a.u.", "scale": 1.0, "type": "shape"}


class SpectroscopyResult:
    """Unified container for 1D and 2D Qubit Spectroscopy experiment results in SI units."""

    def __init__(
        self,
        freqs: np.ndarray,
        transmon: Transmon,
        is_2d: bool = False,
        # 1D fields
        p1_vals: Optional[np.ndarray] = None,
        p2_vals: Optional[np.ndarray] = None,
        p_exc_vals: Optional[np.ndarray] = None,
        populations: Optional[Dict[int, np.ndarray]] = None,
        fit_info: Optional[Dict[str, Any]] = None,
        # 2D fields
        sweep_param: Optional[str] = None,
        sweep_vals: Optional[np.ndarray] = None,
        p1_grid: Optional[np.ndarray] = None,
        p2_grid: Optional[np.ndarray] = None,
        p_exc_grid: Optional[np.ndarray] = None,
        populations_grid: Optional[Dict[int, np.ndarray]] = None,
        # Context parameters
        pulse_type: Type[Pulse] = SquarePulse,
        duration: float = 200e-9,
        amp: Optional[float] = None,
        flux: Optional[float] = None,
    ):
        self.freqs = np.asarray(freqs, dtype=float)
        self.transmon = transmon
        self.is_2d = is_2d
        self.pulse_type = pulse_type
        self.duration = duration
        self.amp = amp
        self.flux = flux

        # 1D attributes
        self.p1_vals = np.asarray(p1_vals, dtype=float) if p1_vals is not None else np.array([])
        self.p2_vals = np.asarray(p2_vals, dtype=float) if p2_vals is not None else np.array([])
        self.p_exc_vals = np.asarray(p_exc_vals, dtype=float) if p_exc_vals is not None else np.array([])
        self.populations = populations or {}
        self.fit_info = fit_info or {}

        # 2D attributes
        self.sweep_param = sweep_param
        self.sweep_vals = np.asarray(sweep_vals, dtype=float) if sweep_vals is not None else np.array([])
        self.p1_grid = p1_grid if p1_grid is not None else np.array([])
        self.p2_grid = p2_grid if p2_grid is not None else np.array([])
        self.p_exc_grid = p_exc_grid if p_exc_grid is not None else np.array([])
        self.populations_grid = populations_grid or {}

    @property
    def amps(self) -> np.ndarray:
        """Swept drive amplitudes if sweep_param is 'amp' or 'amps', or array with single amp."""
        if self.sweep_param in ("amp", "amps"):
            return self.sweep_vals
        return np.array([self.amp]) if self.amp is not None else np.array([])

    @property
    def fluxes(self) -> np.ndarray:
        """Swept flux biases if sweep_param is 'flux', or array with single flux."""
        if self.sweep_param == "flux":
            return self.sweep_vals
        return np.array([self.flux]) if self.flux is not None else np.array([])

    @property
    def durations(self) -> np.ndarray:
        """Swept pulse durations if sweep_param is 'duration' or 'length', or array with single duration."""
        if self.sweep_param in ("duration", "length"):
            return self.sweep_vals
        return np.array([self.duration]) if self.duration is not None else np.array([])

    @property
    def f01(self) -> float:
        """Measured / fitted qubit 0-1 transition frequency in Hz (1D mode)."""
        if self.is_2d:
            raise AttributeError("f01 property is only available for 1D spectroscopy sweeps.")
        if "f01" in self.fit_info and self.fit_info["f01"] is not None:
            return self.fit_info["f01"]
        return float(self.freqs[np.argmax(self.p1_vals)])

    @property
    def f02_half(self) -> Optional[float]:
        """Measured / fitted two-photon transition frequency (f_02 / 2) in Hz (1D mode)."""
        if self.is_2d:
            raise AttributeError("f02_half property is only available for 1D spectroscopy sweeps.")
        if "f02_half" in self.fit_info and self.fit_info["f02_half"] is not None:
            return self.fit_info["f02_half"]
        if self.transmon.levels >= 3 and len(self.p2_vals) > 0:
            eff_f01 = self.transmon.frequency_at_flux(self.flux)
            expected_f02_half = eff_f01 + self.transmon.alpha / 2.0
            idx = np.argmin(np.abs(self.freqs - expected_f02_half))
            if self.p2_vals[idx] > 0.05:
                window = int(max(1, len(self.freqs) // 10))
                start_w = max(0, idx - window)
                end_w = min(len(self.freqs), idx + window + 1)
                sub_peak = np.argmax(self.p2_vals[start_w:end_w])
                return float(self.freqs[start_w + sub_peak])
        return None

    @property
    def alpha_measured(self) -> Optional[float]:
        """Measured anharmonicity in Hz extracted from two-photon transition (1D mode)."""
        if self.f02_half is not None:
            return 2.0 * (self.f02_half - self.f01)
        return None

    @property
    def fwhm_01(self) -> Optional[float]:
        """Full width at half maximum (FWHM) linewidth of the 0-1 resonance in Hz (1D mode)."""
        return self.fit_info.get("fwhm_01", None)

    @property
    def has_two_photon(self) -> bool:
        """Whether a significant two-photon transition was detected (1D mode)."""
        if self.is_2d:
            return False
        return bool(self.fit_info.get("has_two_photon", False) or (self.f02_half is not None))

    def plot(
        self,
        observable: str = "both",
        ax: Optional[Union[plt.Axes, Sequence[plt.Axes]]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        cmap: str = "viridis",
        show_theory_lines: bool = True,
        show_fit: bool = True,
        title: Optional[str] = None,
        plot_levels: Sequence[int] = (1, 2),
        show_total_exc: bool = True,
    ) -> Union[plt.Axes, Tuple[plt.Axes, ...]]:
        """Plot the spectroscopy measurement data.

        Automatically detects 1D vs 2D sweep. For 2D sweeps, automatically adapts axes,
        labels, and theoretical reference overlays based on whether 'amps' or 'flux' was swept.

        Args:
            observable: In 2D mode: 'both' (side-by-side P_exc and P(|2>)),
                'p_exc' (total excited population), 'p1' (P(|1>)), or 'p2' (P(|2>)).
            ax: Optional Matplotlib Axes (or tuple of two axes for observable='both').
            figsize: Figure size tuple.
            cmap: Colormap name for 2D heatmaps (default 'viridis').
            show_theory_lines: Whether to draw theoretical resonance curves/lines (default True).
            show_fit: Whether to overlay fitted curves in 1D mode (default True).
            title: Custom title string.
            plot_levels: Fock level populations to scatter in 1D mode (default (1, 2)).
            show_total_exc: Whether to plot total excitation curve in 1D mode (default True).

        Returns:
            Matplotlib Axes or tuple of Axes.
        """
        use_ghz = np.max(self.freqs) > 1e6
        scale = 1e9 if use_ghz else 1.0
        unit = "GHz" if use_ghz else "Hz"
        freqs_scaled = self.freqs / scale

        # -------------------------------------------------------------
        # 1D Spectroscopy Plotting
        # -------------------------------------------------------------
        if not self.is_2d:
            if ax is None:
                _, ax = plt.subplots(figsize=figsize or (8, 4.8))

            colors = {1: "#1f77b4", 2: "#d62728", 3: "#9467bd", 0: "#7f7f7f"}
            markers = {1: "o", 2: "s", 3: "^"}

            for n in plot_levels:
                if n in self.populations and n < self.transmon.levels:
                    pop = self.populations[n]
                    color = colors.get(n, "#ff7f0e")
                    marker = markers.get(n, "o")
                    ax.scatter(
                        freqs_scaled,
                        pop,
                        label=f"P(|{n}⟩)",
                        color=color,
                        marker=marker,
                        s=24,
                        alpha=0.85,
                        zorder=3,
                    )

            if show_total_exc and self.transmon.levels > 2:
                ax.plot(
                    freqs_scaled,
                    self.p_exc_vals,
                    label="P_exc (Total)",
                    color="#2ca02c",
                    lw=1.8,
                    ls="--",
                    zorder=2,
                )

            if show_fit and self.fit_info.get("fit_fn") is not None:
                f_dense = np.linspace(self.freqs[0], self.freqs[-1], 400)
                y_fit = self.fit_info["fit_fn"](f_dense)
                ax.plot(
                    f_dense / scale,
                    y_fit,
                    color="#e377c2",
                    lw=2.0,
                    label="Lorentzian Fit",
                    zorder=2,
                )

            if show_theory_lines:
                eff_f01 = self.transmon.frequency_at_flux(self.flux)
                f01_theory = eff_f01 / scale
                flux_str = f" [flux={self.flux:.3f}Φ₀]" if self.flux is not None else ""
                ax.axvline(
                    f01_theory,
                    color="#1f77b4",
                    ls=":",
                    lw=1.5,
                    alpha=0.8,
                    label=f"f_01 theory ({f01_theory:.4f} {unit}{flux_str})",
                )
                if self.transmon.levels >= 3:
                    f02_half_theory = (eff_f01 + self.transmon.alpha / 2.0) / scale
                    ax.axvline(
                        f02_half_theory,
                        color="#d62728",
                        ls=":",
                        lw=1.5,
                        alpha=0.8,
                        label=f"f_02/2 theory ({f02_half_theory:.4f} {unit})",
                    )

            ax.set_xlabel(f"Drive Carrier Frequency ({unit})")
            ax.set_ylabel("Population")
            ax.set_ylim(-0.04, 1.05)
            ax.grid(True, alpha=0.3)

            pulse_name = self.pulse_type.__name__ if hasattr(self.pulse_type, "__name__") else str(self.pulse_type)
            amp_val = self.amp if self.amp is not None else 0.1
            pulse_desc = f"{pulse_name}(T={self.duration*1e9:.1f}ns, amp={amp_val:.2f})"
            if self.flux is not None:
                pulse_desc += f", flux={self.flux:.3f}Φ₀"
            default_title = f"Qubit Spectroscopy ({self.transmon.name}) - {pulse_desc}"
            if self.has_two_photon and self.alpha_measured is not None:
                default_title += f"\nDetected α: {self.alpha_measured/1e6:.1f} MHz (True: {self.transmon.alpha/1e6:.1f} MHz)"

            ax.set_title(title or default_title, fontsize=11)
            ax.legend(loc="best", fontsize=9)
            return ax

        # -------------------------------------------------------------
        # 2D Multi-parameter Spectroscopy Plotting
        # -------------------------------------------------------------
        meta = _get_param_meta(str(self.sweep_param or ""))
        scale_y = meta["scale"]
        unit_y = meta["unit"]
        label_y = meta["label"]
        y_vals = self.sweep_vals / scale_y
        y_label = f"{label_y} ({unit_y})" if unit_y and unit_y != "a.u." else label_y

        if self.sweep_param in ("amp", "amps"):
            title_prefix = "Power Spectroscopy"
        elif self.sweep_param == "flux":
            title_prefix = "Flux Spectroscopy"
        else:
            title_prefix = f"2D Spectroscopy ({label_y})"

        def _draw_theory_2d(target_ax):
            if not show_theory_lines:
                return
            if self.sweep_param == "flux":
                phi_dense = np.linspace(np.min(self.sweep_vals), np.max(self.sweep_vals), 200)
                f01_dense = np.array([self.transmon.frequency_at_flux(phi) for phi in phi_dense]) / scale
                target_ax.plot(
                    f01_dense,
                    phi_dense / scale_y,
                    color="white",
                    ls="--",
                    lw=1.8,
                    alpha=0.9,
                    label="f_01(Φ) Theory",
                )
                if self.transmon.levels >= 3:
                    f02_dense = (f01_dense * scale + self.transmon.alpha / 2.0) / scale
                    target_ax.plot(
                        f02_dense,
                        phi_dense / scale_y,
                        color="red",
                        ls="--",
                        lw=1.8,
                        alpha=0.9,
                        label="f_02(Φ)/2 Theory",
                    )
                target_ax.legend(loc="best", framealpha=0.8, fontsize=9)
            else:
                eff_f01 = self.transmon.frequency_at_flux(self.flux)
                f01_th = eff_f01 / scale
                target_ax.axvline(
                    f01_th,
                    color="white",
                    ls="--",
                    lw=1.5,
                    alpha=0.85,
                    label=f"f_01 ({f01_th:.4f} {unit})",
                )
                if self.transmon.levels >= 3:
                    f02_half_th = (eff_f01 + self.transmon.alpha / 2.0) / scale
                    target_ax.axvline(
                        f02_half_th,
                        color="red",
                        ls="--",
                        lw=1.5,
                        alpha=0.85,
                        label=f"f_02/2 ({f02_half_th:.4f} {unit})",
                    )
                target_ax.legend(loc="upper right", framealpha=0.8, fontsize=9)

        # Check subplots layout
        if observable in ("both", "all") and self.transmon.levels >= 3:
            if ax is None:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize or (15, 5))
            elif isinstance(ax, (list, tuple, np.ndarray)) and len(ax) >= 2:
                ax1, ax2 = ax[0], ax[1]
            else:
                ax1 = ax
                fig, ax2 = plt.subplots(figsize=(7, 5))

            # Left: Total Excitation
            mesh1 = ax1.pcolormesh(
                freqs_scaled, y_vals, self.p_exc_grid, shading="auto", cmap=cmap, vmin=0.0, vmax=1.0
            )
            cb1 = plt.colorbar(mesh1, ax=ax1)
            cb1.set_label("Total Excitation P_exc")
            _draw_theory_2d(ax1)
            ax1.set_xlim(float(np.min(freqs_scaled)), float(np.max(freqs_scaled)))
            ax1.set_ylim(float(np.min(y_vals)), float(np.max(y_vals)))
            ax1.set_xlabel(f"Drive Frequency ({unit})")
            ax1.set_ylabel(y_label)
            ax1.set_title("Total Excitation P_exc (Shows both f_01 & f_02/2)")

            # Right: P(|2>)
            mesh2 = ax2.pcolormesh(
                freqs_scaled, y_vals, self.p2_grid, shading="auto", cmap=cmap, vmin=0.0, vmax=1.0
            )
            cb2 = plt.colorbar(mesh2, ax=ax2)
            cb2.set_label("Population P(|2⟩)")
            _draw_theory_2d(ax2)
            ax2.set_xlim(float(np.min(freqs_scaled)), float(np.max(freqs_scaled)))
            ax2.set_ylim(float(np.min(y_vals)), float(np.max(y_vals)))
            ax2.set_xlabel(f"Drive Frequency ({unit})")
            ax2.set_ylabel(y_label)
            ax2.set_title("Population P(|2⟩) (Two-Photon |0⟩→|2⟩ Branch)")

            main_title = title or f"{title_prefix} ({self.transmon.name})"
            if hasattr(plt, "suptitle"):
                plt.suptitle(main_title, fontsize=13, y=1.02)
            return ax1, ax2

        # Single 2D plot
        data_map = {
            "both": (self.p_exc_grid, "Total Excitation P_exc"),
            "all": (self.p_exc_grid, "Total Excitation P_exc"),
            "p_exc": (self.p_exc_grid, "Total Excitation P_exc (Shows both f_01 & f_02/2)"),
            "p1": (self.p1_grid, "Population P(|1⟩) (Single-Photon 0→1)"),
            "p2": (self.p2_grid, "Population P(|2⟩) (Two-Photon 0→2 Branch)"),
        }
        if observable not in data_map:
            raise ValueError(f"Unknown observable '{observable}', expected one of {list(data_map.keys())}")

        grid, obs_name = data_map[observable]
        if ax is None:
            _, ax = plt.subplots(figsize=figsize or (8.5, 5))

        mesh = ax.pcolormesh(freqs_scaled, y_vals, grid, shading="auto", cmap=cmap, vmin=0.0, vmax=1.0)
        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label(obs_name)
        _draw_theory_2d(ax)

        ax.set_xlim(float(np.min(freqs_scaled)), float(np.max(freqs_scaled)))
        ax.set_ylim(float(np.min(y_vals)), float(np.max(y_vals)))
        ax.set_xlabel(f"Drive Frequency ({unit})")
        ax.set_ylabel(y_label)
        default_title = f"{title_prefix} ({self.transmon.name}) - {obs_name}"
        ax.set_title(title or default_title, fontsize=11)
        return ax


class _SpectroscopyRunMethod:
    """Descriptor to enforce deprecation of static Spectroscopy.run()."""

    def __get__(self, instance, owner=None):
        if instance is None:
            def deprecated_static_run(*args, **kwargs):
                raise RuntimeError(
                    "Spectroscopy.run() as a static method has been deprecated. "
                    "Please use the two-step workflow: "
                    "exp = Spectroscopy.set(...); exp.plot_sequence(); res = exp.run()"
                )
            deprecated_static_run.__doc__ = (
                "Deprecated static run method. Use Spectroscopy.set(...); exp.run() instead."
            )
            return deprecated_static_run
        return instance._run_instance


class Spectroscopy:
    """Configures and executes Qubit Spectroscopy experiments (1D Frequency Sweep and 2D Multi-parameter Spectroscopy).

    Supports 1D Frequency sweeps as well as arbitrary 2D parameter sweeps (e.g. freqs vs. amps, flux,
    duration, ramp_time, drag, mod_freq, sigma, or any valid pulse parameter).

    Workflow:
        exp = Spectroscopy.set(transmon, freqs, ...)
        exp.plot_sequence()  # Inspect pulse sequence before running
        res = exp.run()      # Run simulation and retrieve SpectroscopyResult
    """

    run = _SpectroscopyRunMethod()

    def __init__(
        self,
        transmon: Transmon,
        freqs: Union[Sequence[float], np.ndarray],
        amps: Union[float, Sequence[float], np.ndarray] = 0.1,
        flux: Optional[Union[float, Sequence[float], np.ndarray]] = None,
        pulse_type: Type[Pulse] = SquarePulse,
        duration: Union[float, Sequence[float], np.ndarray] = 200e-9,
        dt: float = 1e-9,
        fit: bool = True,
        sweep_param: Optional[str] = None,
        **pulse_kwargs,
    ):
        if transmon.levels < 3:
            warnings.warn(
                f"Transmon '{transmon.name}' has levels={transmon.levels} (< 3). "
                f"Two-photon transition (|0> -> |2>) requires at least 3 Hilbert levels. "
                f"Consider initializing Transmon with levels >= 3 (default is levels=4).",
                UserWarning,
                stacklevel=2,
            )

        freqs_arr = np.asarray(freqs, dtype=float)
        if freqs_arr.ndim != 1 or len(freqs_arr) == 0:
            raise ValueError("freqs must be a non-empty 1D array of frequencies in Hz.")

        # Standardize candidates
        candidates: Dict[str, Any] = {
            "amp": amps,
            "flux": flux,
            "duration": duration,
        }
        candidates.update(pulse_kwargs)

        multi_params = [k for k, v in candidates.items() if _is_multi_value(v)]

        if sweep_param is not None:
            norm_param = "amp" if sweep_param == "amps" else sweep_param
            if norm_param not in candidates:
                raise ValueError(
                    f"Explicit sweep_param '{sweep_param}' was specified, but no such parameter was provided."
                )
            selected_param = norm_param
            param_vals = np.asarray(candidates[selected_param], dtype=float)
            is_2d = len(param_vals) > 1
        elif len(multi_params) > 1:
            raise ValueError(
                f"Cannot sweep both or multiple secondary parameters simultaneously: {multi_params}. "
                f"Spectroscopy supports 1D frequency sweep or 2D (frequency vs. 1 parameter)."
            )
        elif len(multi_params) == 1:
            is_2d = True
            selected_param = multi_params[0]
            param_vals = np.asarray(candidates[selected_param], dtype=float)
        else:
            is_2d = False
            selected_param = None
            param_vals = np.array([])

        self.transmon = transmon
        self.freqs = freqs_arr
        self.pulse_type = pulse_type
        self.dt = float(dt)
        self.fit = bool(fit)
        self.is_2d = is_2d
        self.sweep_param = selected_param
        self.sweep_vals = param_vals

        # Base scalar values
        self.single_amp = _to_single_value(amps, default=0.1)
        self.single_flux = _to_single_value(flux, default=None)
        self.single_duration = _to_single_value(duration, default=200e-9)
        self.single_kwargs: Dict[str, Any] = {}
        for k, v in pulse_kwargs.items():
            if k == selected_param:
                self.single_kwargs[k] = v
            else:
                self.single_kwargs[k] = _to_single_value(v, default=v) if _is_multi_value(v) is False else v

        # Preserved attributes
        self.amps = amps
        self.flux = flux
        self.duration = duration
        self.pulse_kwargs = pulse_kwargs

    @classmethod
    def set(
        cls,
        transmon: Transmon,
        freqs: Union[Sequence[float], np.ndarray],
        amps: Union[float, Sequence[float], np.ndarray] = 0.1,
        flux: Optional[Union[float, Sequence[float], np.ndarray]] = None,
        pulse_type: Type[Pulse] = SquarePulse,
        duration: Union[float, Sequence[float], np.ndarray] = 200e-9,
        dt: float = 1e-9,
        fit: bool = True,
        sweep_param: Optional[str] = None,
        **pulse_kwargs,
    ) -> Spectroscopy:
        """Initialize and configure a Spectroscopy instance."""
        return cls(
            transmon=transmon,
            freqs=freqs,
            amps=amps,
            flux=flux,
            pulse_type=pulse_type,
            duration=duration,
            dt=dt,
            fit=fit,
            sweep_param=sweep_param,
            **pulse_kwargs,
        )

    def _get_step_params(self, val: float) -> Tuple[float, Optional[float], float, Dict[str, Any]]:
        """Return (amp, flux, duration, kwargs) for a given sweep value."""
        step_amp = float(val) if self.sweep_param in ("amps", "amp") else self.single_amp
        step_flux = float(val) if self.sweep_param == "flux" else self.single_flux
        step_duration = float(val) if self.sweep_param in ("duration", "length") else self.single_duration
        step_kwargs = dict(self.single_kwargs)
        if self.sweep_param in step_kwargs:
            step_kwargs[self.sweep_param] = float(val)
        return step_amp, step_flux, step_duration, step_kwargs

    @staticmethod
    def _build_sequence(
        transmon: Transmon,
        current_amp: float,
        current_flux: Optional[float],
        pulse_type: Type[Pulse],
        duration: float,
        **pulse_kwargs,
    ) -> PulseSequence:
        """Construct the probe pulse sequence according to fixed experimental physics.

        If current_flux is None, only the XY probe pulse is applied on transmon.xy.
        If current_flux is provided, a FlatTop Z pulse of duration + 100ns is applied on
        transmon.z, strictly center-aligned with the XY probe pulse.
        """
        xy_pulse = pulse_type(duration=duration, amp=current_amp, **pulse_kwargs)
        seq = PulseSequence(name="spectroscopy_probe")

        if current_flux is not None:
            z_duration = duration + 100e-9
            ramp_time = min(10e-9, z_duration / 4.0)
            z_pulse = FlatTopPulse(duration=z_duration, amp=current_flux, ramp_time=ramp_time)
            seq.align_center((transmon.z, z_pulse), (transmon.xy, xy_pulse))
        else:
            seq.add(transmon.xy, xy_pulse)

        return seq

    def plot_sequence(
        self,
        dt: Optional[float] = None,
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot the pulse schedule and waveform sequence in the time domain before running the experiment."""
        # Case 1: 1D Frequency Sweep
        if not self.is_2d:
            s_amp, s_flux, s_dur, s_kwargs = self._get_step_params(0.0)
            seq = self._build_sequence(
                transmon=self.transmon,
                current_amp=s_amp,
                current_flux=s_flux,
                pulse_type=self.pulse_type,
                duration=s_dur,
                **s_kwargs,
            )
            default_title = (
                f"Spectroscopy Pulse Sequence: 1D Frequency Sweep ({self.transmon.name})"
                + (f" [amp={s_amp:.2f}]" if s_amp is not None else "")
                + (f" [flux={s_flux:.3f}Φ₀]" if s_flux is not None else "")
            )
            return seq.plot(dt=dt, figsize=figsize, title=title or default_title)

        # Case 2: 2D Multi-parameter Sweep
        meta = _get_param_meta(self.sweep_param or "")
        p_type = meta["type"]
        scale_val = meta["scale"]
        unit_str = meta["unit"]
        label_str = meta["label"]

        v_min = float(np.min(self.sweep_vals))
        v_max = float(np.max(self.sweep_vals))
        v_min_scaled = v_min / scale_val
        v_max_scaled = v_max / scale_val

        if p_type == "time":
            v_rep = v_max
        elif p_type == "flux":
            v_rep = v_max if abs(v_max) >= abs(v_min) and v_max != 0 else (v_min if v_min != 0 else 0.25)
        elif p_type == "amplitude":
            v_rep = v_max if abs(v_max) >= abs(v_min) and v_max != 0 else (v_min if v_min != 0 else 0.5)
        else:
            v_rep = v_max if v_max != 0 else v_min

        s_amp_rep, s_flux_rep, s_dur_rep, s_kw_rep = self._get_step_params(v_rep)
        s_amp_min, s_flux_min, s_dur_min, s_kw_min = self._get_step_params(v_min)
        s_amp_max, s_flux_max, s_dur_max, s_kw_max = self._get_step_params(v_max)

        seq_rep = self._build_sequence(self.transmon, s_amp_rep, s_flux_rep, self.pulse_type, s_dur_rep, **s_kw_rep)
        seq_min = self._build_sequence(self.transmon, s_amp_min, s_flux_min, self.pulse_type, s_dur_min, **s_kw_min)
        seq_max = self._build_sequence(self.transmon, s_amp_max, s_flux_max, self.pulse_type, s_dur_max, **s_kw_max)

        max_dur = max(seq_rep.duration, seq_min.duration, seq_max.duration)
        eff_dt = dt if dt is not None else min(max_dur / 250, 5e-10)

        times, waves_rep = seq_rep.sample(dt=eff_dt)
        _, waves_min = seq_min.sample(dt=eff_dt)
        _, waves_max = seq_max.sample(dt=eff_dt)

        target_len = len(times)
        def _align_len(arr):
            if len(arr) < target_len:
                return np.pad(arr, (0, target_len - len(arr)))
            return arr[:target_len]

        has_z_channel = (self.sweep_param == "flux") or (self.single_flux is not None)
        xy_ch = self.transmon.xy
        z_ch = self.transmon.z

        n_ch = 2 if has_z_channel else 1
        fig, axes = plt.subplots(n_ch, 1, figsize=figsize or (10, 3.2 * n_ch), sharex=True, squeeze=False)

        ax_xy = axes[0, 0]
        w_xy_rep = _align_len(waves_rep.get(xy_ch, np.zeros(target_len, dtype=complex)).real)
        w_xy_min = _align_len(waves_min.get(xy_ch, np.zeros(target_len, dtype=complex)).real)
        w_xy_max = _align_len(waves_max.get(xy_ch, np.zeros(target_len, dtype=complex)).real)

        times_ns = times * 1e9

        if p_type == "amplitude":
            ax_xy.plot(times_ns, w_xy_rep, label=f"Representative Pulse ({label_str}={v_rep / scale_val:.2f})", color="#1f77b4", lw=2)
            if v_min != v_max:
                ax_xy.plot(times_ns, w_xy_min, ls="--", color="#ff7f0e", lw=1.2, alpha=0.8, label=f"Min ({v_min_scaled:.2f})")
                ax_xy.plot(times_ns, w_xy_max, ls="--", color="#d62728", lw=1.2, alpha=0.8, label=f"Max ({v_max_scaled:.2f})")
                ax_xy.fill_between(times_ns, w_xy_min, w_xy_max, color="#1f77b4", alpha=0.15, label="Swept Range")

                peak_idx = int(np.argmax(np.abs(w_xy_rep)))
                t_peak_ns = times_ns[peak_idx]
                val_min = w_xy_min[peak_idx]
                val_max = w_xy_max[peak_idx]

                ax_xy.annotate(
                    "",
                    xy=(t_peak_ns, val_min),
                    xytext=(t_peak_ns, val_max),
                    arrowprops=dict(arrowstyle="<->", color="#d62728", lw=2.2, mutation_scale=16),
                    zorder=5,
                )
                total_t_ns = times_ns[-1] - times_ns[0]
                text_x = min(t_peak_ns + 0.03 * total_t_ns, times_ns[-1] - 0.15 * total_t_ns)
                text_y = (val_min + val_max) / 2.0
                ax_xy.text(
                    text_x,
                    text_y,
                    f"  Sweep {self.sweep_param}:\n  [{v_min_scaled:.2f} → {v_max_scaled:.2f}] {unit_str}",
                    color="#d62728",
                    fontsize=10,
                    fontweight="bold",
                    va="center",
                    zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="#d62728"),
                )
        elif p_type == "time" and self.sweep_param in ("duration", "length"):
            ax_xy.plot(times_ns, w_xy_min, ls="--", color="#ff7f0e", lw=1.8, label=f"Min Duration ({v_min_scaled:.1f} ns)")
            ax_xy.plot(times_ns, w_xy_max, color="#1f77b4", lw=2.0, label=f"Max Duration ({v_max_scaled:.1f} ns)")
            ax_xy.fill_between(times_ns, w_xy_min, w_xy_max, color="#1f77b4", alpha=0.15, label="Duration Expansion Range")

            amp_peak = np.max(np.abs(w_xy_max))
            y_arrow = (amp_peak if amp_peak > 0 else 1.0) * 0.5
            ax_xy.annotate(
                "",
                xy=(v_min_scaled, y_arrow),
                xytext=(v_max_scaled, y_arrow),
                arrowprops=dict(arrowstyle="<->", color="#d62728", lw=2.2, mutation_scale=16),
                zorder=5,
            )
            ax_xy.text(
                (v_min_scaled + v_max_scaled) / 2.0,
                y_arrow + 0.08 * (amp_peak if amp_peak > 0 else 1.0),
                f"Sweep duration: [{v_min_scaled:.1f} → {v_max_scaled:.1f}] ns",
                color="#d62728",
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="bottom",
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="#d62728"),
            )
        elif p_type == "time":
            ax_xy.plot(times_ns, w_xy_min, ls="--", color="#ff7f0e", lw=1.5, label=f"Min {label_str} ({v_min_scaled:.1f} ns)")
            ax_xy.plot(times_ns, w_xy_max, color="#1f77b4", lw=2.0, label=f"Max {label_str} ({v_max_scaled:.1f} ns)")
            ax_xy.fill_between(times_ns, w_xy_min, w_xy_max, color="#1f77b4", alpha=0.15, label="Edge/Ramp Variation")

            amp_peak = np.max(np.abs(w_xy_max))
            ax_xy.text(
                times_ns[len(times_ns)//2],
                amp_peak * 0.85,
                f"Sweep {self.sweep_param}: [{v_min_scaled:.1f} → {v_max_scaled:.1f}] ns",
                color="#d62728",
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="top",
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="#d62728"),
            )
        else:
            if self.sweep_param == "flux":
                ax_xy.plot(times_ns, w_xy_rep, label=f"Probe Pulse (amp={self.single_amp:.2f})", color="#1f77b4", lw=2)
            else:
                ax_xy.plot(times_ns, w_xy_min, ls="--", color="#ff7f0e", lw=1.5, label=f"Min ({v_min_scaled:.2f} {unit_str})")
                ax_xy.plot(times_ns, w_xy_max, color="#1f77b4", lw=2.0, label=f"Max ({v_max_scaled:.2f} {unit_str})")
                ax_xy.fill_between(times_ns, w_xy_min, w_xy_max, color="#1f77b4", alpha=0.15, label="Swept Envelope Range")

                amp_peak = np.max(np.abs(w_xy_max))
                ax_xy.text(
                    times_ns[len(times_ns)//2],
                    amp_peak * 0.85,
                    f"Sweep {self.sweep_param}: [{v_min_scaled:.2f} → {v_max_scaled:.2f}] {unit_str}",
                    color="#d62728",
                    fontsize=10,
                    fontweight="bold",
                    ha="center",
                    va="top",
                    zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="#d62728"),
                )

        ax_xy.set_ylabel(f"{xy_ch}\nAmplitude (AWG V₀)", fontsize=10)
        ax_xy.grid(True, alpha=0.3)
        ax_xy.legend(loc="upper right")

        if has_z_channel:
            ax_z = axes[1, 0]
            w_z_rep = _align_len(waves_rep.get(z_ch, np.zeros(target_len, dtype=complex)).real)
            w_z_min = _align_len(waves_min.get(z_ch, np.zeros(target_len, dtype=complex)).real)
            w_z_max = _align_len(waves_max.get(z_ch, np.zeros(target_len, dtype=complex)).real)

            if self.sweep_param == "flux":
                ax_z.plot(times_ns, w_z_rep, label=f"Representative Z Pulse (Φ={v_rep:.2f}Φ₀)", color="#2ca02c", lw=2)
                if v_min != v_max:
                    ax_z.plot(times_ns, w_z_min, ls="--", color="#ff7f0e", lw=1.2, alpha=0.8, label=f"Min Flux ({v_min_scaled:.2f}Φ₀)")
                    ax_z.plot(times_ns, w_z_max, ls="--", color="#d62728", lw=1.2, alpha=0.8, label=f"Max Flux ({v_max_scaled:.2f}Φ₀)")
                    ax_z.fill_between(times_ns, w_z_min, w_z_max, color="#2ca02c", alpha=0.15, label="Swept Flux Range")

                    mid_idx = len(times) // 2
                    t_mid_ns = times_ns[mid_idx]
                    val_min = w_z_min[mid_idx]
                    val_max = w_z_max[mid_idx]

                    ax_z.annotate(
                        "",
                        xy=(t_mid_ns, val_min),
                        xytext=(t_mid_ns, val_max),
                        arrowprops=dict(arrowstyle="<->", color="#d62728", lw=2.2, mutation_scale=16),
                        zorder=5,
                    )
                    total_t_ns = times_ns[-1] - times_ns[0]
                    text_x = min(t_mid_ns + 0.03 * total_t_ns, times_ns[-1] - 0.15 * total_t_ns)
                    text_y = (val_min + val_max) / 2.0
                    ax_z.text(
                        text_x,
                        text_y,
                        f"  Sweep Φ:\n  [{v_min_scaled:.2f} → {v_max_scaled:.2f}] Φ₀",
                        color="#d62728",
                        fontsize=10,
                        fontweight="bold",
                        va="center",
                        zorder=6,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="#d62728"),
                    )
            else:
                if self.sweep_param in ("duration", "length"):
                    ax_z.plot(times_ns, w_z_min, ls="--", color="#ff7f0e", lw=1.5, label=f"Min Z Bias (Φ={self.single_flux:.3f}Φ₀)")
                    ax_z.plot(times_ns, w_z_max, color="#2ca02c", lw=2.0, label=f"Max Z Bias (Φ={self.single_flux:.3f}Φ₀)")
                    ax_z.fill_between(times_ns, w_z_min, w_z_max, color="#2ca02c", alpha=0.15, label="Z Pulse Dynamic Extension")
                else:
                    ax_z.plot(times_ns, w_z_rep, label=f"Constant Flux Pulse (Φ={self.single_flux:.3f}Φ₀)", color="#2ca02c", lw=2)

            ax_z.set_ylabel(f"{z_ch}\nFlux Bias (Φ₀)", fontsize=10)
            ax_z.grid(True, alpha=0.3)
            ax_z.legend(loc="upper right")

        axes[-1, 0].set_xlabel("Time (ns)")
        default_title = (
            f"Spectroscopy Pulse Sequence: 2D {label_str} Sweep ({self.transmon.name}) "
            f"[{self.sweep_param}: {v_min_scaled:.2f} → {v_max_scaled:.2f}{(' ' + unit_str) if unit_str and unit_str != 'a.u.' else ''}]"
        )
        fig.suptitle(title or default_title, fontsize=12, y=1.01)
        plt.tight_layout()
        return fig

    def _run_instance(self) -> SpectroscopyResult:
        """Execute the configured 1D or 2D Qubit Spectroscopy experiment."""
        transmon = self.transmon
        freqs_arr = self.freqs
        pulse_type = self.pulse_type
        dt = self.dt
        fit = self.fit

        # -------------------------------------------------------------
        # 2D Sweep: Multi-parameter Spectroscopy (Any swept parameter)
        # -------------------------------------------------------------
        if self.is_2d:
            n_sweep = len(self.sweep_vals)
            n_freqs = len(freqs_arr)
            p1_grid = np.zeros((n_sweep, n_freqs), dtype=float)
            p2_grid = np.zeros((n_sweep, n_freqs), dtype=float)
            p_exc_grid = np.zeros((n_sweep, n_freqs), dtype=float)
            populations_grid: Dict[int, np.ndarray] = {
                n: np.zeros((n_sweep, n_freqs), dtype=float) for n in range(transmon.levels)
            }

            for i, val in enumerate(self.sweep_vals):
                s_amp, s_flux, s_dur, s_kwargs = self._get_step_params(val)
                seq = self._build_sequence(
                    transmon=transmon,
                    current_amp=s_amp,
                    current_flux=s_flux,
                    pulse_type=pulse_type,
                    duration=s_dur,
                    **s_kwargs,
                )
                for j, f_d in enumerate(freqs_arr):
                    res = Measurement.run(transmon, seq, dt=dt, f_d=float(f_d))
                    p0 = res.final_population(0)
                    p1_grid[i, j] = res.final_population(1)
                    if transmon.levels >= 3:
                        p2_grid[i, j] = res.final_population(2)
                    p_exc_grid[i, j] = max(0.0, min(1.0, 1.0 - p0))
                    for n in range(transmon.levels):
                        populations_grid[n][i, j] = res.final_population(n)

            return SpectroscopyResult(
                freqs=freqs_arr,
                transmon=transmon,
                is_2d=True,
                sweep_param=self.sweep_param,
                sweep_vals=self.sweep_vals,
                p1_grid=p1_grid,
                p2_grid=p2_grid,
                p_exc_grid=p_exc_grid,
                populations_grid=populations_grid,
                pulse_type=pulse_type,
                duration=self.single_duration if self.sweep_param not in ("duration", "length") else None,
                amp=self.single_amp if self.sweep_param not in ("amps", "amp") else None,
                flux=self.single_flux if self.sweep_param != "flux" else None,
            )

        # -------------------------------------------------------------
        # 1D Sweep: Frequency Sweep
        # -------------------------------------------------------------
        s_amp, s_flux, s_dur, s_kwargs = self._get_step_params(0.0)
        seq = self._build_sequence(
            transmon=transmon,
            current_amp=s_amp,
            current_flux=s_flux,
            pulse_type=pulse_type,
            duration=s_dur,
            **s_kwargs,
        )

        p_all: Dict[int, List[float]] = {n: [] for n in range(transmon.levels)}
        p_exc_list: List[float] = []

        for f_d in freqs_arr:
            res = Measurement.run(transmon, seq, dt=dt, f_d=float(f_d))
            p0 = res.final_population(0)
            p_exc_list.append(max(0.0, min(1.0, 1.0 - p0)))
            for n in range(transmon.levels):
                p_all[n].append(res.final_population(n))

        p_all_arr = {n: np.array(p_all[n]) for n in p_all}
        p1_arr = p_all_arr[1]
        p2_arr = p_all_arr[2] if transmon.levels >= 3 else np.zeros_like(p1_arr)
        p_exc_arr = np.array(p_exc_list)

        fit_info: Dict[str, Any] = {}
        if fit and len(freqs_arr) >= 7:
            eff_f01 = transmon.frequency_at_flux(s_flux)
            f01_guess = eff_f01
            f02_half_guess = (eff_f01 + transmon.alpha / 2.0) if transmon.levels >= 3 else None
            fit_info = fit_spectroscopy_peaks(
                xs=freqs_arr,
                ys=p1_arr,
                f01_guess=f01_guess,
                f02_half_guess=f02_half_guess,
                p2_vals=p2_arr if transmon.levels >= 3 else None,
            )

        return SpectroscopyResult(
            freqs=freqs_arr,
            transmon=transmon,
            is_2d=False,
            p1_vals=p1_arr,
            p2_vals=p2_arr,
            p_exc_vals=p_exc_arr,
            populations=p_all_arr,
            fit_info=fit_info,
            pulse_type=pulse_type,
            duration=s_dur,
            amp=s_amp,
            flux=s_flux,
        )

    def __repr__(self) -> str:
        mode = f"2D ({self.sweep_param})" if self.is_2d else "1D"
        dur_str = f"{self.single_duration:.2e}s" if self.sweep_param not in ("duration", "length") else f"swept[{len(self.sweep_vals)}]"
        return (
            f"Spectroscopy(transmon='{self.transmon.name}', mode='{mode}', "
            f"freqs=[{self.freqs[0]:.3e}, {self.freqs[-1]:.3e}] Hz, "
            f"pulse={self.pulse_type.__name__}, duration={dur_str})"
        )

