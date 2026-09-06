"""Rabi oscillation experiments (Amplitude Rabi and Time Rabi) in SI units."""

from __future__ import annotations
from typing import Optional, Type, Dict, Any, Union
import numpy as np
import matplotlib.pyplot as plt

from ..models.transmon import Transmon
from ..pulses.base import Pulse
from ..pulses.shapes import GaussianPulse, SquarePulse
from ..sequence.sequence import PulseSequence
from ..simulation.solver import Simulator
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
        """Calibrated amplitude for a pi pulse (if amplitude Rabi) in rad/s (or drive units)."""
        if self.sweep_type != "amplitude":
            raise ValueError("amp_pi is only valid for Amplitude Rabi")
        return self.fit_info["period"] / 2.0

    @property
    def amp_pi_half(self) -> float:
        """Calibrated amplitude for a pi/2 pulse (if amplitude Rabi)."""
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
    ) -> plt.Axes:
        """Plot Rabi data and sinusoidal fit."""
        if ax is None:
            _, ax = plt.subplots(figsize=figsize or (8, 4.5))

        ax.scatter(self.sweep_vals, self.p1_vals, color="#1f77b4", label="Simulation Data", zorder=3)
        ax.plot(self.sweep_vals, self.fit_info["y_fit"], color="#d62728", lw=2, label="Sine Fit", zorder=2)

        if self.sweep_type == "amplitude":
            ax.set_xlabel("Drive Amplitude (rad/s or arb. units)")
            ax.set_title(f"Amplitude Rabi ({self.transmon.name})")
            ax.axvline(self.amp_pi, color="green", ls="--", label=f"π-pulse: {self.amp_pi:.3e}")
            ax.axvline(self.amp_pi_half, color="orange", ls=":", label=f"π/2-pulse: {self.amp_pi_half:.3e}")
        else:
            ax.set_xlabel("Pulse Duration (s)")
            ax.set_title(f"Time Rabi ({self.transmon.name})")
            ax.axvline(self.time_pi, color="green", ls="--", label=f"π-time: {self.time_pi:.2e} s")

        ax.set_ylabel("Excited State Population P(|1⟩)")
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
        **pulse_kwargs,
    ) -> RabiResult:
        """Perform an Amplitude Rabi sweep to calibrate pi and pi/2 pulse amplitudes.

        Args:
            transmon: Physical Transmon model.
            pulse_type: Pulse class to instantiate (default GaussianPulse).
            duration: Pulse length in seconds (default 40e-9 s = 40 ns).
            amps: 1D array of amplitudes to sweep in rad/s.
            dt: Simulation time step in seconds (default 5e-10 s = 0.5 ns).
            pulse_kwargs: Additional kwargs passed to pulse_type.
        """
        if amps is None:
            amps = np.linspace(0.0, 2.0e8, 41)

        p1_list = []
        for a in amps:
            p = pulse_type(duration=duration, amp=a, **pulse_kwargs)
            seq = PulseSequence(name=f"rabi_a_{a:.2e}").add(transmon.drive, p)
            res = Simulator.run(transmon, seq, dt=dt)
            p1_list.append(res.final_population(1))

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
        amp: float = 5.0e7,
        durations: Optional[np.ndarray] = None,
        dt: float = 5e-10,
        **pulse_kwargs,
    ) -> RabiResult:
        """Perform a Time Rabi sweep (duration sweep with fixed amplitude).

        Args:
            transmon: Physical Transmon model.
            pulse_type: Pulse class to instantiate (default SquarePulse).
            amp: Drive amplitude in rad/s (default 5e7 rad/s).
            durations: 1D array of durations in seconds to sweep.
            dt: Simulation time step in seconds (default 5e-10 s = 0.5 ns).
            pulse_kwargs: Additional kwargs passed to pulse_type.
        """
        if durations is None:
            durations = np.linspace(2e-9, 60e-9, 40)

        p1_list = []
        for d in durations:
            p = pulse_type(duration=d, amp=amp, **pulse_kwargs)
            seq = PulseSequence(name=f"rabi_t_{d:.2e}").add(transmon.drive, p)
            res = Simulator.run(transmon, seq, dt=dt)
            p1_list.append(res.final_population(1))

        p1_arr = np.array(p1_list)
        fit = fit_sine(durations, p1_arr)

        return RabiResult(
            sweep_type="time",
            sweep_vals=durations,
            p1_vals=p1_arr,
            fit_info=fit,
            transmon=transmon,
        )
