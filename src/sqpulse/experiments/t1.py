"""T1 relaxation time measurement experiment in SI units."""

from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

from ..models.transmon import Transmon
from ..pulses.base import Pulse
from ..sequence.sequence import PulseSequence
from ..measurement.projective import Simulator
from .fitting import fit_decay


class T1Result:
    """Container for T1 relaxation experiment measurements and fit in SI units."""

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
        """Measured T1 time in seconds (s)."""
        return self.fit_info["T"]

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: Optional[tuple] = None,
    ) -> plt.Axes:
        """Plot T1 relaxation decay data and exponential fit."""
        if ax is None:
            _, ax = plt.subplots(figsize=figsize or (8, 4.5))

        ax.scatter(self.delays, self.p1_vals, color="#1f77b4", label="Simulation Data", zorder=3)
        ax.plot(self.delays, self.fit_info["y_fit"], color="#d62728", lw=2, label="Exponential Fit", zorder=2)

        ax.axvline(
            self.t1_fit,
            color="green",
            ls="--",
            label=f"Fitted T1: {self.t1_fit:.2e} s (True: {self.transmon.t1:.2e} s)",
        )

        ax.set_xlabel("Delay Time (s)")
        ax.set_ylabel("Excited State Population P(|1⟩)")
        ax.set_title(f"T1 Relaxation Measurement ({self.transmon.name})")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        return ax


class T1Experiment:
    """Measures qubit energy relaxation time T1 in SI units."""

    @staticmethod
    def run(
        transmon: Transmon,
        pi_pulse: Pulse,
        delays: Optional[np.ndarray] = None,
        dt: float = 1e-9,
        backend: Union[str, Any] = "projective",
        backend_kwargs: Optional[Dict[str, Any]] = None,
    ) -> T1Result:
        """Run T1 experiment by applying a pi-pulse and varying the wait time before measurement.

        Args:
            transmon: Transmon model instance with finite T1 (in seconds).
            pi_pulse: Pre-calibrated pi pulse.
            delays: Array of delay durations in seconds.
            dt: Simulation sampling step in seconds (default 1e-9 s = 1 ns).
            backend: Measurement backend ('projective' or 'dispersive', default 'projective').
            backend_kwargs: Additional kwargs passed to the measurement backend.
        """
        from ..measurement import Measurement

        if np.isinf(transmon.t1) or transmon.t1 <= 0:
            raise ValueError(f"Transmon T1 must be finite positive, got {transmon.t1} s")

        if delays is None:
            delays = np.linspace(0, 3.5 * transmon.t1, 40)

        b_opts = dict(dt=dt)
        if backend_kwargs:
            b_opts.update(backend_kwargs)

        p1_list = []
        for d in delays:
            seq = PulseSequence(name=f"t1_delay_{d:.2e}s")
            seq.add(transmon.drive, pi_pulse)
            if d > 0:
                seq.delay(transmon.drive, d)
            res = Measurement.run(transmon, seq, backend=backend, **b_opts)
            if hasattr(res, "final_population"):
                p1_list.append(res.final_population(1))
            elif hasattr(res, "counts"):
                cts = res.counts()
                total = sum(cts.values())
                p1_list.append(cts.get(1, 0) / max(1, total))

        p1_arr = np.array(p1_list)
        fit = fit_decay(delays, p1_arr)

        return T1Result(
            delays=delays,
            p1_vals=p1_arr,
            fit_info=fit,
            transmon=transmon,
        )
