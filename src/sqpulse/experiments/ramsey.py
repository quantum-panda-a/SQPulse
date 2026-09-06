"""Ramsey interferometry and dephasing measurement experiment."""

from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

from ..models.transmon import Transmon
from ..pulses.base import Pulse
from ..sequence.sequence import PulseSequence
from ..simulation.solver import Simulator
from .fitting import fit_decaying_sine


class RamseyResult:
    """Container for Ramsey experiment measurements and fit."""

    def __init__(
        self,
        delays: np.ndarray,
        p1_vals: np.ndarray,
        fit_info: Dict[str, Any],
        detuning: float,
        transmon: Transmon,
    ):
        self.delays = np.asarray(delays)
        self.p1_vals = np.asarray(p1_vals)
        self.fit_info = fit_info
        self.detuning = float(detuning)
        self.transmon = transmon

    @property
    def t2_star(self) -> float:
        """Measured T2* dephasing time in nanoseconds."""
        return self.fit_info["T2"]

    @property
    def fitted_detuning(self) -> float:
        """Measured precession frequency in GHz."""
        return self.fit_info["f0"]

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: Optional[tuple] = None,
    ) -> plt.Axes:
        """Plot Ramsey fringes and decaying sinusoidal fit."""
        if ax is None:
            _, ax = plt.subplots(figsize=figsize or (8, 4.5))

        use_us = np.max(self.delays) >= 5000.0
        scale = 1e-3 if use_us else 1.0
        unit = "us" if use_us else "ns"

        xs = self.delays * scale
        t2_scaled = self.t2_star * scale

        ax.scatter(xs, self.p1_vals, color="#1f77b4", label="Simulation Data", zorder=3)
        ax.plot(xs, self.fit_info["y_fit"], color="#d62728", lw=2, label="Decaying Sine Fit", zorder=2)

        # Plot decay envelope
        env_up = self.fit_info["y0"] + self.fit_info["amp"] * np.exp(-self.delays / self.t2_star)
        env_down = self.fit_info["y0"] - self.fit_info["amp"] * np.exp(-self.delays / self.t2_star)
        ax.plot(xs, env_up, color="black", ls="--", alpha=0.5, label="Envelope")
        ax.plot(xs, env_down, color="black", ls="--", alpha=0.5)

        ax.set_xlabel(f"Delay Time ({unit})")
        ax.set_ylabel("Excited State Population P(|1⟩)")
        ax.set_title(
            f"Ramsey Fringes ({self.transmon.name}): T2* = {t2_scaled:.2f} {unit}, "
            f"Δf = {self.fitted_detuning*1e3:.2f} MHz"
        )
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        return ax


class RamseyExperiment:
    """Measures T2* dephasing time and qubit frequency detuning via Ramsey interferometry."""

    @staticmethod
    def run(
        transmon: Transmon,
        pi_half_pulse: Pulse,
        detuning: float = 0.002,  # 2 MHz in GHz
        delays: Optional[np.ndarray] = None,
        dt: float = 0.5,
    ) -> RamseyResult:
        """Run Ramsey sequence: π/2 - delay(τ) - π/2 under reference detuning Δ.

        Args:
            transmon: Transmon model instance.
            pi_half_pulse: Pre-calibrated π/2 pulse.
            detuning: Artificial reference detuning in GHz (default 0.002 GHz = 2 MHz).
            delays: Array of delay durations in ns.
            dt: Simulation sampling step in ns.
        """
        if delays is None:
            max_delay = min(3.0 * transmon.t2 if not np.isinf(transmon.t2) else 3000.0, 5000.0)
            delays = np.linspace(0, max_delay, 80)

        # Drive frequency offset: f_d = f_q - detuning
        f_d = transmon.f_q - detuning

        p1_list = []
        for d in delays:
            seq = PulseSequence(name=f"ramsey_{d:.0f}ns")
            seq.add(transmon.drive, pi_half_pulse)
            if d > 0:
                seq.delay(transmon.drive, d)
            seq.add(transmon.drive, pi_half_pulse)

            res = Simulator.run(transmon, seq, dt=dt, f_d=f_d)
            p1_list.append(res.final_population(1))

        p1_arr = np.array(p1_list)
        fit = fit_decaying_sine(delays, p1_arr)

        return RamseyResult(
            delays=delays,
            p1_vals=p1_arr,
            fit_info=fit,
            detuning=detuning,
            transmon=transmon,
        )
