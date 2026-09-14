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
from ..measurement.projective import Simulator
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
        """Swept drive amplitudes if sweep_param is 'amp', or array with single amp."""
        if self.sweep_param == "amp":
            return self.sweep_vals
        return np.array([self.amp]) if self.amp is not None else np.array([])

    @property
    def fluxes(self) -> np.ndarray:
        """Swept flux biases if sweep_param is 'flux', or array with single flux."""
        if self.sweep_param == "flux":
            return self.sweep_vals
        return np.array([self.flux]) if self.flux is not None else np.array([])

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
        if self.sweep_param == "amp":
            y_vals = self.sweep_vals
            y_label = "Drive Amplitude V_0 (Normalized AWG)"
            title_prefix = "Power Spectroscopy"

            def _draw_theory_2d(target_ax):
                if show_theory_lines:
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

        elif self.sweep_param == "flux":
            y_vals = self.sweep_vals
            y_label = "External Flux Bias Φ / Φ₀"
            title_prefix = "Flux Spectroscopy"

            def _draw_theory_2d(target_ax):
                if show_theory_lines:
                    phi_dense = np.linspace(np.min(self.sweep_vals), np.max(self.sweep_vals), 200)
                    f01_dense = np.array([self.transmon.frequency_at_flux(phi) for phi in phi_dense]) / scale
                    target_ax.plot(
                        f01_dense,
                        phi_dense,
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
                            phi_dense,
                            color="red",
                            ls="--",
                            lw=1.8,
                            alpha=0.9,
                            label="f_02(Φ)/2 Theory",
                        )
                    target_ax.legend(loc="best", framealpha=0.8, fontsize=9)
        else:
            y_vals = self.sweep_vals
            y_label = str(self.sweep_param)
            title_prefix = f"2D Spectroscopy ({self.sweep_param})"

            def _draw_theory_2d(target_ax):
                pass

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


class QubitSpectroscopyExperiment:
    """Executes Qubit Spectroscopy experiments (1D Frequency Sweep and 2D Multi-parameter Spectroscopy) in SI units."""

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

    @staticmethod
    def run(
        transmon: Transmon,
        freqs: Union[Sequence[float], np.ndarray],
        amps: Union[float, Sequence[float], np.ndarray] = 0.1,
        flux: Optional[Union[float, Sequence[float], np.ndarray]] = None,
        pulse_type: Type[Pulse] = SquarePulse,
        duration: float = 200e-9,
        dt: float = 1e-9,
        fit: bool = True,
        **pulse_kwargs,
    ) -> SpectroscopyResult:
        """Execute a 1D or 2D Qubit Spectroscopy experiment.

        Automatically determines whether to perform a 1D sweep or a 2D sweep:
        - If 'amps' is an array/sequence (len > 1): sweeps amplitude vs frequency (2D Power Spectroscopy).
        - If 'flux' is an array/sequence (len > 1): sweeps flux bias vs frequency (2D Flux Spectroscopy).
        - If neither is multi-valued: sweeps frequency (1D Frequency Sweep).

        When 'flux' is provided, a baseband Z flux pulse (100 ns longer than the probe pulse,
        center-aligned) is automatically scheduled on transmon.z.

        Args:
            transmon: Target Transmon model. Levels >= 3 recommended to observe two-photon effects.
            freqs: 1D array of drive carrier frequencies in Hz.
            amps: Drive amplitude (normalized AWG V_0 in [-1.0, 1.0]). Can be scalar or 1D array (default 0.1).
            flux: External flux bias (in units of Phi_0). Can be scalar or 1D array (default None).
            pulse_type: Pulse shape class for the XY probe pulse (default: SquarePulse).
            duration: XY probe pulse duration in seconds (default: 200e-9 s = 200 ns).
            dt: Simulation time step in seconds (default: 1e-9 s = 1 ns).
            fit: Whether to perform automatic Lorentzian/dual-peak fitting in 1D mode (default True).
            **pulse_kwargs: Additional keyword arguments passed to the pulse_type constructor.

        Returns:
            SpectroscopyResult containing measurement data, grids, and auto-adaptive .plot() method.
        """
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

        amps_multi = _is_multi_value(amps)
        flux_multi = _is_multi_value(flux)

        if amps_multi and flux_multi:
            raise ValueError(
                "Cannot sweep both 'amps' and 'flux' simultaneously. "
                "QubitSpectroscopyExperiment supports 1D (frequency) or 2D (frequency vs. amps or frequency vs. flux)."
            )

        # -------------------------------------------------------------
        # 2D Sweep: Power Spectroscopy (Amplitudes vs Frequencies)
        # -------------------------------------------------------------
        if amps_multi:
            sweep_param = "amp"
            sweep_vals = np.asarray(amps, dtype=float)
            single_flux = _to_single_value(flux, default=None)

            n_sweep = len(sweep_vals)
            n_freqs = len(freqs_arr)
            p1_grid = np.zeros((n_sweep, n_freqs), dtype=float)
            p2_grid = np.zeros((n_sweep, n_freqs), dtype=float)
            p_exc_grid = np.zeros((n_sweep, n_freqs), dtype=float)
            populations_grid: Dict[int, np.ndarray] = {
                n: np.zeros((n_sweep, n_freqs), dtype=float) for n in range(transmon.levels)
            }

            for i, a in enumerate(sweep_vals):
                seq = QubitSpectroscopyExperiment._build_sequence(
                    transmon=transmon,
                    current_amp=float(a),
                    current_flux=single_flux,
                    pulse_type=pulse_type,
                    duration=duration,
                    **pulse_kwargs,
                )
                for j, f_d in enumerate(freqs_arr):
                    res = Simulator.run(transmon, seq, dt=dt, f_d=float(f_d))
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
                sweep_param=sweep_param,
                sweep_vals=sweep_vals,
                p1_grid=p1_grid,
                p2_grid=p2_grid,
                p_exc_grid=p_exc_grid,
                populations_grid=populations_grid,
                pulse_type=pulse_type,
                duration=duration,
                amp=None,
                flux=single_flux,
            )

        # -------------------------------------------------------------
        # 2D Sweep: Flux Spectroscopy (Flux vs Frequencies)
        # -------------------------------------------------------------
        if flux_multi:
            sweep_param = "flux"
            sweep_vals = np.asarray(flux, dtype=float)
            single_amp = _to_single_value(amps, default=0.1)

            n_sweep = len(sweep_vals)
            n_freqs = len(freqs_arr)
            p1_grid = np.zeros((n_sweep, n_freqs), dtype=float)
            p2_grid = np.zeros((n_sweep, n_freqs), dtype=float)
            p_exc_grid = np.zeros((n_sweep, n_freqs), dtype=float)
            populations_grid = {
                n: np.zeros((n_sweep, n_freqs), dtype=float) for n in range(transmon.levels)
            }

            for i, phi in enumerate(sweep_vals):
                seq = QubitSpectroscopyExperiment._build_sequence(
                    transmon=transmon,
                    current_amp=single_amp,
                    current_flux=float(phi),
                    pulse_type=pulse_type,
                    duration=duration,
                    **pulse_kwargs,
                )
                for j, f_d in enumerate(freqs_arr):
                    res = Simulator.run(transmon, seq, dt=dt, f_d=float(f_d))
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
                sweep_param=sweep_param,
                sweep_vals=sweep_vals,
                p1_grid=p1_grid,
                p2_grid=p2_grid,
                p_exc_grid=p_exc_grid,
                populations_grid=populations_grid,
                pulse_type=pulse_type,
                duration=duration,
                amp=single_amp,
                flux=None,
            )

        # -------------------------------------------------------------
        # 1D Sweep: Frequency Sweep
        # -------------------------------------------------------------
        single_amp = _to_single_value(amps, default=0.1)
        single_flux = _to_single_value(flux, default=None)

        seq = QubitSpectroscopyExperiment._build_sequence(
            transmon=transmon,
            current_amp=single_amp,
            current_flux=single_flux,
            pulse_type=pulse_type,
            duration=duration,
            **pulse_kwargs,
        )

        p_all: Dict[int, List[float]] = {n: [] for n in range(transmon.levels)}
        p_exc_list: List[float] = []

        for f_d in freqs_arr:
            res = Simulator.run(transmon, seq, dt=dt, f_d=float(f_d))
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
            eff_f01 = transmon.frequency_at_flux(single_flux)
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
            duration=duration,
            amp=single_amp,
            flux=single_flux,
        )


# Alias
SpectroscopyExperiment = QubitSpectroscopyExperiment
