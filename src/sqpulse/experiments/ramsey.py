"""Ramsey interferometry and dephasing measurement experiment in SI units."""

from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

from ..models.transmon import Transmon
from ..pulses.base import Pulse
from ..sequence.sequence import PulseSequence
from ..measurement.projective import Simulator
from .fitting import fit_decaying_sine


class RamseyResult:
    """Container for Ramsey experiment measurements and fit in SI units."""

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
        """Measured T2* dephasing time in seconds (s)."""
        return self.fit_info["T2"]

    @property
    def fitted_detuning(self) -> float:
        """Measured precession frequency in Hz."""
        return self.fit_info["f0"]

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: Optional[tuple] = None,
    ) -> plt.Axes:
        """Plot Ramsey fringes and decaying sinusoidal fit."""
        if ax is None:
            _, ax = plt.subplots(figsize=figsize or (8, 4.5))

        ax.scatter(self.delays, self.p1_vals, color="#1f77b4", label="Simulation Data", zorder=3)
        ax.plot(self.delays, self.fit_info["y_fit"], color="#d62728", lw=2, label="Decaying Sine Fit", zorder=2)

        # Plot decay envelope
        env_up = self.fit_info["y0"] + self.fit_info["amp"] * np.exp(-self.delays / self.t2_star)
        env_down = self.fit_info["y0"] - self.fit_info["amp"] * np.exp(-self.delays / self.t2_star)
        ax.plot(self.delays, env_up, color="black", ls="--", alpha=0.5, label="Envelope")
        ax.plot(self.delays, env_down, color="black", ls="--", alpha=0.5)

        ax.set_xlabel("Delay Time (s)")
        ax.set_ylabel("Excited State Population P(|1⟩)")
        ax.set_title(
            f"Ramsey Fringes ({self.transmon.name}): T2* = {self.t2_star:.2e} s, "
            f"Δf = {self.fitted_detuning:.2e} Hz"
        )
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        return ax


class RamseyExperiment:
    """Measures T2* dephasing time and qubit frequency detuning via Ramsey interferometry in SI units."""

    @staticmethod
    def run(
        transmon: Transmon,
        pi_half_pulse: Pulse,
        detuning: float = 2.0e6,  # 2 MHz in Hz
        delays: Optional[np.ndarray] = None,
        dt: float = 5e-10,
        backend: Union[str, Any] = "projective",
        backend_kwargs: Optional[Dict[str, Any]] = None,
    ) -> RamseyResult:
        """Run Ramsey sequence: π/2 - delay(τ) - π/2 under reference detuning Δ (in Hz).

        Args:
            transmon: Transmon model instance.
            pi_half_pulse: Pre-calibrated π/2 pulse.
            detuning: Artificial reference detuning in Hz (default 2.0e6 Hz = 2 MHz).
            delays: Array of delay durations in seconds (s).
            dt: Simulation sampling step in seconds (default 5e-10 s = 0.5 ns).
            backend: Measurement backend ('projective' or 'dispersive', default 'projective').
            backend_kwargs: Additional kwargs passed to the measurement backend.
        """
        from ..measurement import Measurement

        if delays is None:
            max_delay = min(3.0 * transmon.t2 if not np.isinf(transmon.t2) else 3.0e-6, 5.0e-6)
            delays = np.linspace(0, max_delay, 80)

        # Drive frequency offset: f_d = f_q - detuning (all in Hz)
        f_d = transmon.f_q - detuning

        b_opts = dict(dt=dt, f_d=f_d)
        if backend_kwargs:
            b_opts.update(backend_kwargs)

        p1_list = []
        for d in delays:
            seq = PulseSequence(name=f"ramsey_{d:.2e}s")
            seq.add(transmon.xy, pi_half_pulse)
            if d > 0:
                seq.delay(transmon.xy, d)
            seq.add(transmon.xy, pi_half_pulse)

            res = Measurement.run(transmon, seq, backend=backend, **b_opts)
            if hasattr(res, "final_population"):
                p1_list.append(res.final_population(1))
            elif hasattr(res, "counts"):
                cts = res.counts()
                total = sum(cts.values())
                p1_list.append(cts.get(1, 0) / max(1, total))

        p1_arr = np.array(p1_list)
        fit = fit_decaying_sine(delays, p1_arr)

        return RamseyResult(
            delays=delays,
            p1_vals=p1_arr,
            fit_info=fit,
            detuning=detuning,
            transmon=transmon,
        )
