"""T1 relaxation time measurement experiment."""

from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

from ..models.transmon import Transmon
from ..pulses.base import Pulse
from ..sequence.sequence import PulseSequence
from ..simulation.solver import Simulator
from .fitting import fit_decay


class T1Result:
    """Container for T1 relaxation experiment measurements and fit."""

    def __init__(
        self,
        delays: np.ndarray,
        p1_vals: np.ndarray,
        fit_info: Dict[str, Any],
        transmon: Transmon,
    ):
        self.delays = np.asarray(delays)
        self.p1_vals = np.asarray(p1_vals)
        self.fit_info = fit_info
        self.transmon = transmon

    @property
    def t1_fit(self) -> float:
        """Measured T1 time in nanoseconds."""
        return self.fit_info["T"]

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: Optional[tuple] = None,
    ) -> plt.Axes:
        """Plot T1 relaxation decay data and exponential fit."""
        if ax is None:
            _, ax = plt.subplots(figsize=figsize or (8, 4.5))

        # Convert to microseconds if large
        use_us = np.max(self.delays) >= 5000.0
        scale = 1e-3 if use_us else 1.0
        unit = "us" if use_us else "ns"

        xs = self.delays * scale
        t1_scaled = self.t1_fit * scale

        ax.scatter(xs, self.p1_vals, color="#1f77b4", label="Simulation Data", zorder=3)
        ax.plot(xs, self.fit_info["y_fit"], color="#d62728", lw=2, label="Exponential Fit", zorder=2)

        ax.axvline(
            t1_scaled,
            color="green",
            ls="--",
            label=f"Fitted T1: {t1_scaled:.2f} {unit} (True: {self.transmon.t1*scale:.2f} {unit})",
        )

        ax.set_xlabel(f"Delay Time ({unit})")
        ax.set_ylabel("Excited State Population P(|1⟩)")
        ax.set_title(f"T1 Relaxation Measurement ({self.transmon.name})")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        return ax


class T1Experiment:
    """Measures qubit energy relaxation time T1."""

    @staticmethod
    def run(
        transmon: Transmon,
        pi_pulse: Pulse,
        delays: Optional[np.ndarray] = None,
        dt: float = 1.0,
    ) -> T1Result:
        """Run T1 experiment by applying a pi-pulse and varying the wait time before measurement.

        Args:
            transmon: Transmon model instance with finite T1.
            pi_pulse: Pre-calibrated pi pulse.
            delays: Array of delay durations in ns.
            dt: Simulation sampling step in ns.
        """
        if np.isinf(transmon.t1) or transmon.t1 <= 0:
            raise ValueError(f"Transmon T1 must be finite positive, got {transmon.t1}")

        if delays is None:
            delays = np.linspace(0, 3.5 * transmon.t1, 40)

        p1_list = []
        for d in delays:
            seq = PulseSequence(name=f"t1_delay_{d:.0f}ns")
            seq.add(transmon.drive, pi_pulse)
            if d > 0:
                seq.delay(transmon.drive, d)
            res = Simulator.run(transmon, seq, dt=dt)
            p1_list.append(res.final_population(1))

        p1_arr = np.array(p1_list)
        fit = fit_decay(delays, p1_arr)

        return T1Result(
            delays=delays,
            p1_vals=p1_arr,
            fit_info=fit,
            transmon=transmon,
        )
