"""Qubit spectroscopy experiments (Frequency Sweep and Power Spectroscopy) in SI units."""

from __future__ import annotations
import warnings
from typing import Optional, Type, Dict, Any, Union, Tuple, List, Sequence
import numpy as np
import matplotlib.pyplot as plt

from ..models.transmon import Transmon
from ..pulses.base import Pulse
from ..pulses.shapes import SquarePulse, GaussianPulse, FlatTopPulse
from ..sequence.sequence import PulseSequence
from ..measurement.projective import Simulator
from .fitting import fit_spectroscopy_peaks, fit_lorentzian


class SpectroscopyResult:
    """Container for 1D Qubit Spectroscopy experiment measurements and fit results in SI units."""

    def __init__(
        self,
        freqs: np.ndarray,
        p1_vals: np.ndarray,
        p2_vals: np.ndarray,
        p_exc_vals: np.ndarray,
        populations: Dict[int, np.ndarray],
        pulse: Pulse,
        transmon: Transmon,
        fit_info: Optional[Dict[str, Any]] = None,
    ):
        self.freqs = np.asarray(freqs, dtype=float)
        self.p1_vals = np.asarray(p1_vals, dtype=float)
        self.p2_vals = np.asarray(p2_vals, dtype=float)
        self.p_exc_vals = np.asarray(p_exc_vals, dtype=float)
        self.populations = populations
        self.pulse = pulse
        self.transmon = transmon
        self.fit_info = fit_info or {}

    @property
    def f01(self) -> float:
        """Measured / fitted qubit 0-1 transition frequency in Hz."""
        if "f01" in self.fit_info and self.fit_info["f01"] is not None:
            return self.fit_info["f01"]
        # Fallback to maximum of P1 or P_exc
        return float(self.freqs[np.argmax(self.p1_vals)])

    @property
    def f02_half(self) -> Optional[float]:
        """Measured / fitted two-photon transition frequency (f_02 / 2) in Hz, or None if not detected."""
        if "f02_half" in self.fit_info and self.fit_info["f02_half"] is not None:
            return self.fit_info["f02_half"]
        # If P2 peak exists around expected f_q + alpha/2
        if self.transmon.levels >= 3 and len(self.p2_vals) > 0:
            expected_f02_half = self.transmon.f_q + self.transmon.alpha / 2.0
            idx = np.argmin(np.abs(self.freqs - expected_f02_half))
            if self.p2_vals[idx] > 0.05:
                # Local peak around expected
                window = int(max(1, len(self.freqs) // 10))
                start_w = max(0, idx - window)
                end_w = min(len(self.freqs), idx + window + 1)
                sub_peak = np.argmax(self.p2_vals[start_w:end_w])
                return float(self.freqs[start_w + sub_peak])
        return None

    @property
    def alpha_measured(self) -> Optional[float]:
        """Measured anharmonicity in Hz extracted from two-photon transition: 2 * (f02/2 - f01)."""
        if self.f02_half is not None:
            return 2.0 * (self.f02_half - self.f01)
        return None

    @property
    def fwhm_01(self) -> Optional[float]:
        """Full width at half maximum (FWHM) linewidth of the 0-1 resonance in Hz."""
        return self.fit_info.get("fwhm_01", None)

    @property
    def has_two_photon(self) -> bool:
        """Whether a significant two-photon transition was detected."""
        return bool(self.fit_info.get("has_two_photon", False) or (self.f02_half is not None))

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: Optional[Tuple[float, float]] = None,
        plot_levels: Sequence[int] = (1, 2),
        show_total_exc: bool = True,
        show_theory_lines: bool = True,
        show_fit: bool = True,
        title: Optional[str] = None,
    ) -> plt.Axes:
        """Plot the spectroscopy response across frequency in SI / GHz units.

        Args:
            ax: Optional Matplotlib Axes.
            figsize: Figure size tuple.
            plot_levels: Which individual Fock level populations to plot (default (1, 2)).
            show_total_exc: Whether to plot total excited population P_exc = 1 - P0 (default True).
            show_theory_lines: Whether to show vertical reference lines for f_01 and f_02/2 (default True).
            show_fit: Whether to overlay fitted curves (default True).
            title: Custom title string.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize or (8, 4.8))

        use_ghz = np.max(self.freqs) > 1e6
        scale = 1e9 if use_ghz else 1.0
        unit = "GHz" if use_ghz else "Hz"

        freqs_scaled = self.freqs / scale

        # Plot individual level populations
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

        # Plot fitted curve
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

        # Theoretical vertical lines
        if show_theory_lines:
            f01_theory = self.transmon.f_q / scale
            ax.axvline(
                f01_theory,
                color="#1f77b4",
                ls=":",
                lw=1.5,
                alpha=0.8,
                label=f"f_01 theory ({f01_theory:.4f} {unit})",
            )
            if self.transmon.levels >= 3:
                f02_half_theory = (self.transmon.f_q + self.transmon.alpha / 2.0) / scale
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

        pulse_desc = f"{self.pulse.__class__.__name__}(T={self.pulse.duration*1e9:.1f}ns, amp={self.pulse.amp:.2f})"
        default_title = f"Qubit Spectroscopy ({self.transmon.name}) - {pulse_desc}"
        if self.has_two_photon and self.alpha_measured is not None:
            default_title += f"\nDetected α: {self.alpha_measured/1e6:.1f} MHz (True: {self.transmon.alpha/1e6:.1f} MHz)"

        ax.set_title(title or default_title, fontsize=11)
        ax.legend(loc="best", fontsize=9)
        return ax


class PowerSpectroscopyResult:
    """Container for 2D Power Spectroscopy (Drive Amplitude vs. Frequency) measurements."""

    def __init__(
        self,
        freqs: np.ndarray,
        amps: np.ndarray,
        p1_grid: np.ndarray,
        p2_grid: np.ndarray,
        p_exc_grid: np.ndarray,
        pulse_template: Pulse,
        transmon: Transmon,
    ):
        self.freqs = np.asarray(freqs, dtype=float)
        self.amps = np.asarray(amps, dtype=float)
        self.p1_grid = np.asarray(p1_grid, dtype=float)
        self.p2_grid = np.asarray(p2_grid, dtype=float)
        self.p_exc_grid = np.asarray(p_exc_grid, dtype=float)
        self.pulse_template = pulse_template
        self.transmon = transmon

    def plot(
        self,
        observable: str = "both",
        ax: Optional[Union[plt.Axes, Sequence[plt.Axes]]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        cmap: str = "viridis",
        show_theory_lines: bool = True,
        title: Optional[str] = None,
    ) -> Union[plt.Axes, Tuple[plt.Axes, plt.Axes]]:
        """Plot the 2D power spectroscopy colormap (frequency vs. drive amplitude).

        Args:
            observable: 'both' (default, side-by-side P_exc and P(|2>)),
                'p_exc' (total excited population, showing both f_01 and f_02/2),
                'p1' (P(|1>), showing fundamental f_01),
                or 'p2' (P(|2>), selectively isolating two-photon transition).
            ax: Optional Matplotlib Axes (or tuple of two axes if observable='both').
            figsize: Figure size tuple.
            cmap: Colormap name (default 'viridis').
            show_theory_lines: Whether to draw theoretical f_01 and f_02/2 lines.
            title: Custom title string.
        """
        use_ghz = np.max(self.freqs) > 1e6
        scale = 1e9 if use_ghz else 1.0
        unit = "GHz" if use_ghz else "Hz"
        freqs_scaled = self.freqs / scale

        f01_theory = self.transmon.f_q / scale
        f02_half_theory = (self.transmon.f_q + self.transmon.alpha / 2.0) / scale if self.transmon.levels >= 3 else None

        def _draw_lines(target_ax):
            if show_theory_lines:
                target_ax.axvline(
                    f01_theory,
                    color="white",
                    ls="--",
                    lw=1.5,
                    alpha=0.85,
                    label=f"f_01 ({f01_theory:.4f} {unit})",
                )
                if f02_half_theory is not None:
                    target_ax.axvline(
                        f02_half_theory,
                        color="red",
                        ls="--",
                        lw=1.5,
                        alpha=0.85,
                        label=f"f_02/2 ({f02_half_theory:.4f} {unit})",
                    )
                target_ax.legend(loc="upper right", framealpha=0.8, fontsize=9)

        if observable in ("both", "all"):
            if ax is None:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize or (15, 5))
            elif isinstance(ax, (list, tuple, np.ndarray)) and len(ax) >= 2:
                ax1, ax2 = ax[0], ax[1]
            else:
                ax1 = ax
                fig, ax2 = plt.subplots(figsize=(7, 5))

            # Left: Total Excitation
            mesh1 = ax1.pcolormesh(freqs_scaled, self.amps, self.p_exc_grid, shading="auto", cmap=cmap, vmin=0.0, vmax=1.0)
            cb1 = plt.colorbar(mesh1, ax=ax1)
            cb1.set_label("Total Excitation P_exc")
            _draw_lines(ax1)
            ax1.set_xlabel(f"Drive Frequency ({unit})")
            ax1.set_ylabel("Drive Amplitude V_0 (Normalized AWG)")
            ax1.set_title(f"Total Excitation P_exc (Shows both f_01 & f_02/2)")

            # Right: Second excited state P(|2>)
            mesh2 = ax2.pcolormesh(freqs_scaled, self.amps, self.p2_grid, shading="auto", cmap=cmap, vmin=0.0, vmax=1.0)
            cb2 = plt.colorbar(mesh2, ax=ax2)
            cb2.set_label("Population P(|2⟩)")
            _draw_lines(ax2)
            ax2.set_xlabel(f"Drive Frequency ({unit})")
            ax2.set_ylabel("Drive Amplitude V_0 (Normalized AWG)")
            ax2.set_title("Population P(|2⟩) (Two-Photon |0⟩→|2⟩ Branch)")

            if title:
                plt.suptitle(title, fontsize=13, y=1.02)
            return ax1, ax2

        data_map = {
            "p_exc": (self.p_exc_grid, "Total Excitation P_exc (Shows both f_01 & f_02/2)"),
            "p1": (self.p1_grid, "Population P(|1⟩) (Single-Photon 0→1)"),
            "p2": (self.p2_grid, "Population P(|2⟩) (Two-Photon 0→2 Branch)"),
        }
        if observable not in data_map:
            raise ValueError(f"Unknown observable '{observable}', expected one of ['both', 'p_exc', 'p1', 'p2']")

        if ax is None:
            _, ax = plt.subplots(figsize=figsize or (8.5, 5))

        grid, obs_name = data_map[observable]
        mesh = ax.pcolormesh(freqs_scaled, self.amps, grid, shading="auto", cmap=cmap, vmin=0.0, vmax=1.0)
        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label(obs_name)
        _draw_lines(ax)

        ax.set_xlabel(f"Drive Frequency ({unit})")
        ax.set_ylabel("Drive Amplitude V_0 (Normalized AWG)")

        default_title = f"Power Spectroscopy ({self.transmon.name}) - {obs_name}"
        if observable == "p2":
            default_title += "\n(Note: Isolates |0⟩→|2⟩; f_01 excites |1⟩ so P(|2⟩)=0 at 5.0 GHz)"

        ax.set_title(title or default_title, fontsize=11)
        return ax


class QubitSpectroscopyExperiment:
    """Executes Qubit Spectroscopy (Frequency Sweep and Power Spectroscopy) in SI units."""

    @staticmethod
    def run(
        transmon: Transmon,
        pulse: Optional[Pulse] = None,
        pulse_type: Type[Pulse] = SquarePulse,
        duration: float = 200e-9,
        amp: float = 0.1,
        freqs: Optional[np.ndarray] = None,
        freq_range: Optional[Tuple[float, float]] = None,
        detunings: Optional[np.ndarray] = None,
        num_points: int = 101,
        dt: float = 1e-9,
        fit: bool = True,
        **pulse_kwargs,
    ) -> SpectroscopyResult:
        """Perform a 1D Qubit Frequency Spectroscopy sweep.

        Allows choosing the pulse (custom instance or pulse type) and setting the frequency range.
        When drive amplitude is large, detects two-photon transition (|0> -> |2>) at f_q + alpha/2.

        Args:
            transmon: Physical Transmon model. Levels >= 3 recommended to observe two-photon effects.
            pulse: Explicit Pulse instance to use. If provided, overrides pulse_type, duration, and amp
                (unless duration or amp is explicitly passed to override).
            pulse_type: Pulse class to instantiate if pulse is None (default: SquarePulse).
            duration: Pulse duration in seconds (default 200e-9 s = 200 ns).
            amp: Normalized AWG pulse amplitude V_0 in [-1.0, 1.0] (default 0.1).
            freqs: 1D array of drive carrier frequencies in Hz to sweep.
            freq_range: Tuple of (f_min, f_max) in Hz. Used if freqs is None.
            detunings: 1D array of frequency offsets relative to transmon.f_q in Hz.
            num_points: Number of frequency points to sweep if freqs is auto-generated (default 101).
            dt: Simulation time step in seconds (default 1e-9 s = 1 ns).
            fit: Whether to perform automatic Lorentzian/dual-peak fitting (default True).
            **pulse_kwargs: Additional keyword arguments passed to pulse_type constructor.

        Returns:
            SpectroscopyResult containing frequency grid, state populations, and fit details.
        """
        if transmon.levels < 3:
            warnings.warn(
                f"Transmon '{transmon.name}' has levels={transmon.levels} (< 3). "
                f"Two-photon transition (|0> -> |2>) requires at least 3 Hilbert levels. "
                f"Consider initializing Transmon with levels >= 3 (default is levels=4).",
                UserWarning,
                stacklevel=2,
            )

        # Resolve pulse
        if pulse is not None:
            probe_pulse = pulse
            if "amp" in pulse_kwargs:
                probe_pulse.amp = pulse_kwargs["amp"]
        else:
            probe_pulse = pulse_type(duration=duration, amp=amp, **pulse_kwargs)

        # Resolve frequency grid
        resolved_freqs = QubitSpectroscopyExperiment._resolve_freqs(
            transmon=transmon,
            freqs=freqs,
            freq_range=freq_range,
            detunings=detunings,
            num_points=num_points,
        )

        # Pre-build sequence template
        seq = PulseSequence(name="spectroscopy_probe").add(transmon.xy, probe_pulse)

        # Run sweep
        p_all: Dict[int, List[float]] = {n: [] for n in range(transmon.levels)}
        p_exc_list: List[float] = []

        for f_d in resolved_freqs:
            res = Simulator.run(transmon, seq, dt=dt, f_d=float(f_d))
            p0 = res.final_population(0)
            p_exc_list.append(max(0.0, min(1.0, 1.0 - p0)))
            for n in range(transmon.levels):
                p_all[n].append(res.final_population(n))

        p_all_arr = {n: np.array(p_all[n]) for n in p_all}
        p1_arr = p_all_arr[1]
        p2_arr = p_all_arr[2] if transmon.levels >= 3 else np.zeros_like(p1_arr)
        p_exc_arr = np.array(p_exc_list)

        # Fit resonance peaks if requested
        fit_info: Dict[str, Any] = {}
        if fit and len(resolved_freqs) >= 7:
            f01_guess = transmon.f_q
            f02_half_guess = (
                transmon.f_q + transmon.alpha / 2.0 if transmon.levels >= 3 else None
            )
            fit_info = fit_spectroscopy_peaks(
                xs=resolved_freqs,
                ys=p1_arr,
                f01_guess=f01_guess,
                f02_half_guess=f02_half_guess,
                p2_vals=p2_arr if transmon.levels >= 3 else None,
            )

        return SpectroscopyResult(
            freqs=resolved_freqs,
            p1_vals=p1_arr,
            p2_vals=p2_arr,
            p_exc_vals=p_exc_arr,
            populations=p_all_arr,
            pulse=probe_pulse,
            transmon=transmon,
            fit_info=fit_info,
        )

    # Alias for run()
    frequency_sweep = run

    @staticmethod
    def power_spectroscopy(
        transmon: Transmon,
        pulse: Optional[Pulse] = None,
        pulse_type: Type[Pulse] = SquarePulse,
        duration: float = 200e-9,
        amps: Optional[np.ndarray] = None,
        freqs: Optional[np.ndarray] = None,
        freq_range: Optional[Tuple[float, float]] = None,
        detunings: Optional[np.ndarray] = None,
        num_points_freq: int = 61,
        num_points_amp: int = 15,
        dt: float = 1e-9,
        **pulse_kwargs,
    ) -> PowerSpectroscopyResult:
        """Perform a 2D Power Spectroscopy sweep (drive amplitude vs. frequency).

        Shows the power broadening of the fundamental 0-1 transition and the emergence of
        the two-photon 0-2 transition branch at higher drive amplitudes.

        Args:
            transmon: Physical Transmon model.
            pulse: Optional template Pulse instance. If provided, its class, duration, and kwargs are used.
            pulse_type: Pulse shape class (default: SquarePulse, used if pulse is None).
            duration: Pulse duration in seconds (default 200e-9 s = 200 ns).
            amps: 1D array of AWG pulse amplitudes V_0 in [-1.0, 1.0].
            freqs: 1D array of frequencies in Hz.
            freq_range: (f_min, f_max) in Hz.
            detunings: 1D array of detunings in Hz relative to transmon.f_q.
            num_points_freq: Number of frequency points (default 61).
            num_points_amp: Number of amplitude points (default 15).
            dt: Simulation sampling step in seconds.
            **pulse_kwargs: Additional kwargs for pulse_type.
        """
        if amps is None:
            amps = np.linspace(0.05, 0.6, num_points_amp)
        else:
            amps = np.asarray(amps, dtype=float)

        resolved_freqs = QubitSpectroscopyExperiment._resolve_freqs(
            transmon=transmon,
            freqs=freqs,
            freq_range=freq_range,
            detunings=detunings,
            num_points=num_points_freq,
        )

        n_amps = len(amps)
        n_freqs = len(resolved_freqs)
        p1_grid = np.zeros((n_amps, n_freqs), dtype=float)
        p2_grid = np.zeros((n_amps, n_freqs), dtype=float)
        p_exc_grid = np.zeros((n_amps, n_freqs), dtype=float)

        actual_pulse_type = pulse.__class__ if pulse is not None else pulse_type
        actual_duration = pulse.duration if pulse is not None else duration

        template_pulse = actual_pulse_type(duration=actual_duration, amp=amps[0], **pulse_kwargs)

        for i, a in enumerate(amps):
            current_pulse = actual_pulse_type(duration=actual_duration, amp=a, **pulse_kwargs)
            seq = PulseSequence(name=f"pwr_spec_a_{a:.2f}").add(transmon.xy, current_pulse)
            for j, f_d in enumerate(resolved_freqs):
                res = Simulator.run(transmon, seq, dt=dt, f_d=float(f_d))
                p0 = res.final_population(0)
                p1_grid[i, j] = res.final_population(1)
                if transmon.levels >= 3:
                    p2_grid[i, j] = res.final_population(2)
                p_exc_grid[i, j] = max(0.0, min(1.0, 1.0 - p0))

        return PowerSpectroscopyResult(
            freqs=resolved_freqs,
            amps=amps,
            p1_grid=p1_grid,
            p2_grid=p2_grid,
            p_exc_grid=p_exc_grid,
            pulse_template=template_pulse,
            transmon=transmon,
        )

    @staticmethod
    def _resolve_freqs(
        transmon: Transmon,
        freqs: Optional[np.ndarray],
        freq_range: Optional[Tuple[float, float]],
        detunings: Optional[np.ndarray],
        num_points: int,
    ) -> np.ndarray:
        """Resolve frequency grid based on user specifications."""
        if freqs is not None:
            return np.asarray(freqs, dtype=float)

        if detunings is not None:
            return transmon.f_q + np.asarray(detunings, dtype=float)

        if freq_range is not None:
            f_min, f_max = freq_range
            return np.linspace(f_min, f_max, num_points)

        # Default intelligent window:
        # Cover f_q + 1.6*alpha to f_q + 0.5*|alpha|
        # This guarantees both f_01 and f_02/2 = f_q + alpha/2 are inside the window.
        alpha = transmon.alpha if transmon.alpha != 0 else -250e6
        if alpha < 0:
            f_min = transmon.f_q + 1.6 * alpha
            f_max = transmon.f_q - 0.5 * alpha
        else:
            f_min = transmon.f_q - 0.5 * alpha
            f_max = transmon.f_q + 1.6 * alpha
        return np.linspace(f_min, f_max, num_points)


# Alias
SpectroscopyExperiment = QubitSpectroscopyExperiment
