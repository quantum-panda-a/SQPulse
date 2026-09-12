"""Rabi oscillation experiments (Amplitude Rabi and Time Rabi) in SI units."""

from __future__ import annotations
from typing import Optional, Type, Dict, Any, Union
import numpy as np
import matplotlib.pyplot as plt

from ..models.transmon import Transmon
from ..pulses.base import Pulse
from ..pulses.shapes import GaussianPulse, SquarePulse
from ..sequence.sequence import PulseSequence
from ..measurement.projective import Simulator
from .fitting import fit_sine


class RabiResult:
    """Container for Rabi experiment measurements and fit results in SI units."""

    def __init__(
        self,
        sweep_type: str,
        sweep_vals: np.ndarray,
        p1_vals: np.ndarray,
        fit_info: Dict[str, Any],
        transmon: Transmon,
    ):
        self.sweep_type = sweep_type  # 'amplitude' or 'time'
        self.sweep_vals = np.asarray(sweep_vals)
        self.p1_vals = np.asarray(p1_vals)
        self.fit_info = fit_info
        self.transmon = transmon

    @property
    def rabi_frequency(self) -> float:
        """Fitted Rabi frequency in units of 1 / (sweep units)."""
        return self.fit_info["f0"]

    @property
    def amp_pi(self) -> float:
        """Calibrated AWG amplitude V_0 for a pi pulse (if amplitude Rabi)."""
        if self.sweep_type != "amplitude":
            raise ValueError("amp_pi is only valid for Amplitude Rabi")
        return self.fit_info["period"] / 2.0

    @property
    def amp_pi_half(self) -> float:
        """Calibrated AWG amplitude V_0 for a pi/2 pulse (if amplitude Rabi)."""
        return self.amp_pi / 2.0

    @property
    def time_pi(self) -> float:
        """Calibrated pulse duration for a pi pulse (if time Rabi) in seconds."""
        if self.sweep_type != "time":
            raise ValueError("time_pi is only valid for Time Rabi")
        return self.fit_info["period"] / 2.0

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: Optional[tuple] = None,
        title: Optional[str] = None,
    ) -> plt.Axes:
        """Plot the Rabi measurement data along with the fitted cosine curve."""
        if ax is None:
            _, ax = plt.subplots(figsize=figsize or (7, 4))

        ax.scatter(self.sweep_vals, self.p1_vals, color="#1f77b4", label="Data (P1)", zorder=3)
        if self.fit_info.get("success", False):
            x_fit = np.linspace(self.sweep_vals[0], self.sweep_vals[-1], 200)
            y_fit = self.fit_info["fit_func"](x_fit)
            ax.plot(x_fit, y_fit, color="#d62728", lw=2, label="Fit", zorder=2)

        if self.sweep_type == "amplitude":
            ax.set_xlabel("Pulse Amplitude V_0 (AWG / Normalized)")
            ax.set_title(title or f"Amplitude Rabi ({self.transmon.name}) - π-amp: {self.amp_pi:.3f}")
        else:
            ax.set_xlabel("Pulse Duration (s)")
            ax.set_title(title or f"Time Rabi ({self.transmon.name}) - π-time: {self.time_pi:.2e} s")

        ax.set_ylabel("P(|1⟩)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        return ax


class RabiExperiment:
    """Executes Amplitude Rabi or Time Rabi experiments in SI units."""

    @staticmethod
    def amplitude_rabi(
        transmon: Transmon,
        pulse_type: Type[Pulse] = GaussianPulse,
        duration: float = 40e-9,
        amps: Optional[np.ndarray] = None,
        dt: float = 5e-10,
        backend: Union[str, Any] = "projective",
        backend_kwargs: Optional[Dict[str, Any]] = None,
        **pulse_kwargs,
    ) -> RabiResult:
        """Perform an Amplitude Rabi sweep to calibrate pi and pi/2 pulse amplitudes.

        Args:
            transmon: Physical Transmon model.
            pulse_type: Pulse class to instantiate (default GaussianPulse).
            duration: Pulse length in seconds (default 40e-9 s = 40 ns).
            amps: 1D array of AWG pulse amplitudes V_0 in [-1.0, 1.0] (defaults to [0.0, 1.0]).
            dt: Simulation time step in seconds (default 5e-10 s = 0.5 ns).
            backend: Measurement backend ('projective' or 'dispersive', default 'projective').
            backend_kwargs: Additional kwargs passed to the measurement backend.
            pulse_kwargs: Additional kwargs passed to pulse_type.
        """
        from ..measurement import Measurement

        if amps is None:
            amps = np.linspace(0.0, 1.0, 41)

        b_opts = dict(dt=dt)
        if backend_kwargs:
            b_opts.update(backend_kwargs)

        p1_list = []
        for a in amps:
            p = pulse_type(duration=duration, amp=a, **pulse_kwargs)
            seq = PulseSequence(name=f"rabi_a_{a:.2e}").add(transmon.xy, p)
            res = Measurement.run(transmon, seq, backend=backend, **b_opts)
            if hasattr(res, "final_population"):
                p1_list.append(res.final_population(1))
            elif hasattr(res, "counts"):
                cts = res.counts()
                total = sum(cts.values())
                p1_list.append(cts.get(1, 0) / max(1, total))

        p1_arr = np.array(p1_list)
        fit = fit_sine(amps, p1_arr)

        return RabiResult(
            sweep_type="amplitude",
            sweep_vals=amps,
            p1_vals=p1_arr,
            fit_info=fit,
            transmon=transmon,
        )

    @staticmethod
    def time_rabi(
        transmon: Transmon,
        pulse_type: Type[Pulse] = SquarePulse,
        amp: float = 0.5,
        durations: Optional[np.ndarray] = None,
        dt: float = 5e-10,
        backend: Union[str, Any] = "projective",
        backend_kwargs: Optional[Dict[str, Any]] = None,
        **pulse_kwargs,
    ) -> RabiResult:
        """Perform a Time Rabi sweep (duration sweep with fixed AWG amplitude).

        Args:
            transmon: Physical Transmon model.
            pulse_type: Pulse class to instantiate (default SquarePulse).
            amp: Normalized AWG amplitude V_0 in [-1.0, 1.0] (default 0.5).
            durations: 1D array of pulse lengths in seconds (s).
            dt: Simulation sampling step in seconds (default 5e-10 s = 0.5 ns).
            backend: Measurement backend ('projective' or 'dispersive', default 'projective').
            backend_kwargs: Additional kwargs passed to the measurement backend.
            pulse_kwargs: Additional kwargs passed to pulse_type.
        """
        from ..measurement import Measurement

        if durations is None:
            durations = np.linspace(2e-9, 100e-9, 41)

        b_opts = dict(dt=dt)
        if backend_kwargs:
            b_opts.update(backend_kwargs)

        p1_list = []
        for d in durations:
            p = pulse_type(duration=d, amp=amp, **pulse_kwargs)
            seq = PulseSequence(name=f"rabi_t_{d:.2e}").add(transmon.xy, p)
            res = Measurement.run(transmon, seq, backend=backend, **b_opts)
            if hasattr(res, "final_population"):
                p1_list.append(res.final_population(1))
            elif hasattr(res, "counts"):
                cts = res.counts()
                total = sum(cts.values())
                p1_list.append(cts.get(1, 0) / max(1, total))

        p1_arr = np.array(p1_list)
        fit = fit_sine(durations, p1_arr)

        return RabiResult(
            sweep_type="time",
            sweep_vals=durations,
            p1_vals=p1_arr,
            fit_info=fit,
            transmon=transmon,
        )
