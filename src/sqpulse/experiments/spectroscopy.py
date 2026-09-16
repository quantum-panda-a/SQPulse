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


class _QubitSpectroscopyRunMethod:
    """Descriptor to enforce deprecation of static QubitSpectroscopyExperiment.run()."""

    def __get__(self, instance, owner=None):
        if instance is None:
            def deprecated_static_run(*args, **kwargs):
                raise RuntimeError(
                    "QubitSpectroscopyExperiment.run() as a static method has been deprecated. "
                    "Please use the two-step workflow: "
                    "exp = QubitSpectroscopyExperiment.set(...); exp.plot_sequence(); res = exp.run()"
                )
            deprecated_static_run.__doc__ = (
                "Deprecated static run method. Use QubitSpectroscopyExperiment.set(...); exp.run() instead."
            )
            return deprecated_static_run
        return instance._run_instance


class QubitSpectroscopyExperiment:
    """Configures and executes Qubit Spectroscopy experiments (1D Frequency Sweep and 2D Multi-parameter Spectroscopy).

    Workflow:
        exp = QubitSpectroscopyExperiment.set(transmon, freqs, ...)
        exp.plot_sequence()  # Inspect pulse sequence before running
        res = exp.run()      # Run simulation and retrieve SpectroscopyResult
    """

    run = _QubitSpectroscopyRunMethod()

    def __init__(
        self,
        transmon: Transmon,
        freqs: Union[Sequence[float], np.ndarray],
        amps: Union[float, Sequence[float], np.ndarray] = 0.1,
        flux: Optional[Union[float, Sequence[float], np.ndarray]] = None,
        pulse_type: Type[Pulse] = SquarePulse,
        duration: float = 200e-9,
        dt: float = 1e-9,
        fit: bool = True,
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

        amps_multi = _is_multi_value(amps)
        flux_multi = _is_multi_value(flux)

        if amps_multi and flux_multi:
            raise ValueError(
                "Cannot sweep both 'amps' and 'flux' simultaneously. "
                "QubitSpectroscopyExperiment supports 1D (frequency) or 2D (frequency vs. amps or frequency vs. flux)."
            )

        self.transmon = transmon
        self.freqs = freqs_arr
        self.amps = amps
        self.flux = flux
        self.pulse_type = pulse_type
        self.duration = float(duration)
        self.dt = float(dt)
        self.fit = bool(fit)
        self.pulse_kwargs = pulse_kwargs

        self.amps_multi = amps_multi
        self.flux_multi = flux_multi
        self.is_2d = amps_multi or flux_multi

        if amps_multi:
            self.sweep_param = "amp"
            self.sweep_vals = np.asarray(amps, dtype=float)
        elif flux_multi:
            self.sweep_param = "flux"
            self.sweep_vals = np.asarray(flux, dtype=float)
        else:
            self.sweep_param = None
            self.sweep_vals = np.array([])

    @classmethod
    def set(
        cls,
        transmon: Transmon,
        freqs: Union[Sequence[float], np.ndarray],
        amps: Union[float, Sequence[float], np.ndarray] = 0.1,
        flux: Optional[Union[float, Sequence[float], np.ndarray]] = None,
        pulse_type: Type[Pulse] = SquarePulse,
        duration: float = 200e-9,
        dt: float = 1e-9,
        fit: bool = True,
        **pulse_kwargs,
    ) -> QubitSpectroscopyExperiment:
        """Initialize and configure a QubitSpectroscopyExperiment instance."""
        return cls(
            transmon=transmon,
            freqs=freqs,
            amps=amps,
            flux=flux,
            pulse_type=pulse_type,
            duration=duration,
            dt=dt,
            fit=fit,
            **pulse_kwargs,
        )

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
        """Plot the pulse schedule and waveform sequence in the time domain before running the experiment.

        - 1D Frequency Sweep: Plots the fixed probe pulse sequence on XY (and Z if constant flux bias is set).
        - 2D Power Spectroscopy (swept amplitude): Plots the representative XY probe waveform with a shaded
          sweep envelope and a vertical double-headed range arrow indicating the sweep variable amp in [min, max].
        - 2D Flux Spectroscopy (swept flux bias): Plots the XY probe pulse and the center-aligned Z FlatTop
          flux pulse, with a double-headed range arrow on the Z channel indicating the sweep variable Phi in [min, max] Phi_0.

        Args:
            dt: Sampling time resolution in seconds (defaults to sequence-adaptive step).
            figsize: Figure size tuple (width, height).
            title: Custom title string.

        Returns:
            plt.Figure: The rendered matplotlib figure.
        """
        # Case 1: 1D Frequency Sweep
        if not self.is_2d:
            single_amp = _to_single_value(self.amps, default=0.1)
            single_flux = _to_single_value(self.flux, default=None)
            seq = self._build_sequence(
                transmon=self.transmon,
                current_amp=single_amp,
                current_flux=single_flux,
                pulse_type=self.pulse_type,
                duration=self.duration,
                **self.pulse_kwargs,
            )
            default_title = (
                f"Spectroscopy Pulse Sequence: 1D Frequency Sweep ({self.transmon.name})"
                + (f" [amp={single_amp:.2f}]" if single_amp is not None else "")
                + (f" [flux={single_flux:.3f}Φ₀]" if single_flux is not None else "")
            )
            return seq.plot(dt=dt, figsize=figsize, title=title or default_title)

        # Case 2: 2D Power Spectroscopy (Sweep Amplitude)
        if self.sweep_param == "amp":
            a_min = float(np.min(self.sweep_vals))
            a_max = float(np.max(self.sweep_vals))
            if abs(a_max) >= abs(a_min) and a_max != 0.0:
                a_rep = a_max
            elif a_min != 0.0:
                a_rep = a_min
            else:
                a_rep = 0.5

            single_flux = _to_single_value(self.flux, default=None)
            seq_rep = self._build_sequence(
                transmon=self.transmon,
                current_amp=a_rep,
                current_flux=single_flux,
                pulse_type=self.pulse_type,
                duration=self.duration,
                **self.pulse_kwargs,
            )
            seq_min = self._build_sequence(
                transmon=self.transmon,
                current_amp=a_min,
                current_flux=single_flux,
                pulse_type=self.pulse_type,
                duration=self.duration,
                **self.pulse_kwargs,
            )
            seq_max = self._build_sequence(
                transmon=self.transmon,
                current_amp=a_max,
                current_flux=single_flux,
                pulse_type=self.pulse_type,
                duration=self.duration,
                **self.pulse_kwargs,
            )

            eff_dt = dt if dt is not None else min(seq_rep.duration / 250, 5e-10)
            times, waves_rep = seq_rep.sample(dt=eff_dt)
            _, waves_min = seq_min.sample(dt=eff_dt)
            _, waves_max = seq_max.sample(dt=eff_dt)

            channels = seq_rep.channels
            n_ch = len(channels)
            fig, axes = plt.subplots(
                n_ch, 1, figsize=figsize or (10, 3.2 * n_ch), sharex=True, squeeze=False
            )

            xy_ch = self.transmon.xy
            z_ch = self.transmon.z if single_flux is not None else None

            # Subplot for XY channel
            ax_xy = axes[0, 0]
            w_rep = waves_rep[xy_ch].real
            w_min = waves_min[xy_ch].real
            w_max = waves_max[xy_ch].real

            ax_xy.plot(times * 1e9, w_rep, label=f"Representative Pulse (amp={a_rep:.2f})", color="#1f77b4", lw=2)
            if a_min != a_max:
                ax_xy.plot(times * 1e9, w_min, ls="--", color="#ff7f0e", lw=1.2, alpha=0.8, label=f"Min Amp ({a_min:.2f})")
                ax_xy.plot(times * 1e9, w_max, ls="--", color="#d62728", lw=1.2, alpha=0.8, label=f"Max Amp ({a_max:.2f})")
                ax_xy.fill_between(times * 1e9, w_min, w_max, color="#1f77b4", alpha=0.15, label="Swept Amp Range")

                peak_idx = int(np.argmax(np.abs(w_rep)))
                t_peak_ns = times[peak_idx] * 1e9
                val_min = w_min[peak_idx]
                val_max = w_max[peak_idx]

                ax_xy.annotate(
                    "",
                    xy=(t_peak_ns, val_min),
                    xytext=(t_peak_ns, val_max),
                    arrowprops=dict(arrowstyle="<->", color="#d62728", lw=2.2, mutation_scale=16),
                    zorder=5,
                )
                total_t_ns = (times[-1] - times[0]) * 1e9
                text_x = min(t_peak_ns + 0.03 * total_t_ns, times[-1] * 1e9 - 0.1 * total_t_ns)
                text_y = (val_min + val_max) / 2.0
                ax_xy.text(
                    text_x,
                    text_y,
                    f"  Sweep amp:\n  [{a_min:.2f} → {a_max:.2f}]",
                    color="#d62728",
                    fontsize=10,
                    fontweight="bold",
                    va="center",
                    zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="#d62728"),
                )

            ax_xy.set_ylabel(f"{xy_ch}\nAmplitude (AWG V₀)", fontsize=10)
            ax_xy.grid(True, alpha=0.3)
            ax_xy.legend(loc="upper right")

            if z_ch and z_ch in channels:
                ax_z = axes[1, 0]
                w_z = waves_rep[z_ch].real
                ax_z.plot(times * 1e9, w_z, label=f"Constant Flux Pulse (flux={single_flux:.3f}Φ₀)", color="#2ca02c", lw=2)
                ax_z.set_ylabel(f"{z_ch}\nFlux Bias (Φ₀)", fontsize=10)
                ax_z.grid(True, alpha=0.3)
                ax_z.legend(loc="upper right")

            axes[-1, 0].set_xlabel("Time (ns)")
            default_title = f"Spectroscopy Pulse Sequence: 2D Power Sweep ({self.transmon.name}) [amp: {a_min:.2f} → {a_max:.2f}]"
            fig.suptitle(title or default_title, fontsize=12, y=1.01)
            plt.tight_layout()
            return fig

        # Case 3: 2D Flux Spectroscopy (Sweep Flux)
        if self.sweep_param == "flux":
            phi_min = float(np.min(self.sweep_vals))
            phi_max = float(np.max(self.sweep_vals))
            if abs(phi_max) >= abs(phi_min) and phi_max != 0.0:
                phi_rep = phi_max
            elif phi_min != 0.0:
                phi_rep = phi_min
            else:
                phi_rep = 0.25

            single_amp = _to_single_value(self.amps, default=0.1)
            seq_rep = self._build_sequence(
                transmon=self.transmon,
                current_amp=single_amp,
                current_flux=phi_rep,
                pulse_type=self.pulse_type,
                duration=self.duration,
                **self.pulse_kwargs,
            )
            seq_min = self._build_sequence(
                transmon=self.transmon,
                current_amp=single_amp,
                current_flux=phi_min,
                pulse_type=self.pulse_type,
                duration=self.duration,
                **self.pulse_kwargs,
            )
            seq_max = self._build_sequence(
                transmon=self.transmon,
                current_amp=single_amp,
                current_flux=phi_max,
                pulse_type=self.pulse_type,
                duration=self.duration,
                **self.pulse_kwargs,
            )

            eff_dt = dt if dt is not None else min(seq_rep.duration / 250, 5e-10)
            times, waves_rep = seq_rep.sample(dt=eff_dt)
            _, waves_min = seq_min.sample(dt=eff_dt)
            _, waves_max = seq_max.sample(dt=eff_dt)

            xy_ch = self.transmon.xy
            z_ch = self.transmon.z

            fig, axes = plt.subplots(2, 1, figsize=figsize or (10, 6.2), sharex=True, squeeze=False)

            # Subplot 1: XY probe pulse
            ax_xy = axes[0, 0]
            w_xy = waves_rep[xy_ch].real
            ax_xy.plot(times * 1e9, w_xy, label=f"Probe Pulse (amp={single_amp:.2f}, dur={self.duration * 1e9:.0f}ns)", color="#1f77b4", lw=2)
            ax_xy.set_ylabel(f"{xy_ch}\nAmplitude (AWG V₀)", fontsize=10)
            ax_xy.grid(True, alpha=0.3)
            ax_xy.legend(loc="upper right")

            # Subplot 2: Z flux bias pulse
            ax_z = axes[1, 0]
            w_z_rep = waves_rep[z_ch].real
            w_z_min = waves_min[z_ch].real
            w_z_max = waves_max[z_ch].real

            ax_z.plot(times * 1e9, w_z_rep, label=f"Representative Z Pulse (Φ={phi_rep:.2f}Φ₀)", color="#2ca02c", lw=2)
            if phi_min != phi_max:
                ax_z.plot(times * 1e9, w_z_min, ls="--", color="#ff7f0e", lw=1.2, alpha=0.8, label=f"Min Flux ({phi_min:.2f}Φ₀)")
                ax_z.plot(times * 1e9, w_z_max, ls="--", color="#d62728", lw=1.2, alpha=0.8, label=f"Max Flux ({phi_max:.2f}Φ₀)")
                ax_z.fill_between(times * 1e9, w_z_min, w_z_max, color="#2ca02c", alpha=0.15, label="Swept Flux Range")

                mid_idx = len(times) // 2
                t_mid_ns = times[mid_idx] * 1e9
                val_min = w_z_min[mid_idx]
                val_max = w_z_max[mid_idx]

                ax_z.annotate(
                    "",
                    xy=(t_mid_ns, val_min),
                    xytext=(t_mid_ns, val_max),
                    arrowprops=dict(arrowstyle="<->", color="#d62728", lw=2.2, mutation_scale=16),
                    zorder=5,
                )
                total_t_ns = (times[-1] - times[0]) * 1e9
                text_x = min(t_mid_ns + 0.03 * total_t_ns, times[-1] * 1e9 - 0.1 * total_t_ns)
                text_y = (val_min + val_max) / 2.0
                ax_z.text(
                    text_x,
                    text_y,
                    f"  Sweep Φ:\n  [{phi_min:.2f} → {phi_max:.2f}] Φ₀",
                    color="#d62728",
                    fontsize=10,
                    fontweight="bold",
                    va="center",
                    zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="#d62728"),
                )

            ax_z.set_ylabel(f"{z_ch}\nFlux Bias (Φ₀)", fontsize=10)
            ax_z.grid(True, alpha=0.3)
            ax_z.legend(loc="upper right")

            axes[-1, 0].set_xlabel("Time (ns)")
            default_title = f"Spectroscopy Pulse Sequence: 2D Flux Sweep ({self.transmon.name}) [Φ: {phi_min:.2f} → {phi_max:.2f}]"
            fig.suptitle(title or default_title, fontsize=12, y=1.01)
            plt.tight_layout()
            return fig

        raise RuntimeError("Invalid experiment configuration.")

    def _run_instance(self) -> SpectroscopyResult:
        """Execute the configured 1D or 2D Qubit Spectroscopy experiment."""
        transmon = self.transmon
        freqs_arr = self.freqs
        amps = self.amps
        flux = self.flux
        pulse_type = self.pulse_type
        duration = self.duration
        dt = self.dt
        fit = self.fit
        pulse_kwargs = self.pulse_kwargs

        # -------------------------------------------------------------
        # 2D Sweep: Power Spectroscopy (Amplitudes vs Frequencies)
        # -------------------------------------------------------------
        if self.amps_multi:
            sweep_param = "amp"
            sweep_vals = self.sweep_vals
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
                seq = self._build_sequence(
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
        if self.flux_multi:
            sweep_param = "flux"
            sweep_vals = self.sweep_vals
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
                seq = self._build_sequence(
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

        seq = self._build_sequence(
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

    def __repr__(self) -> str:
        mode = f"2D ({self.sweep_param})" if self.is_2d else "1D"
        return (
            f"QubitSpectroscopyExperiment(transmon='{self.transmon.name}', mode='{mode}', "
            f"freqs=[{self.freqs[0]:.3e}, {self.freqs[-1]:.3e}] Hz, "
            f"pulse={self.pulse_type.__name__}, duration={self.duration:.2e}s)"
        )


# Alias
SpectroscopyExperiment = QubitSpectroscopyExperiment
