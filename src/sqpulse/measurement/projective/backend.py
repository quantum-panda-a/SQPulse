"""Projective state evolution measurement backend for SQPulse in SI units."""

from __future__ import annotations
from typing import List, Optional, Dict, Any
import qutip

from ...models.transmon import Transmon
from ...sequence.sequence import PulseSequence
from ..base import MeasurementBackend
from .result import ProjectiveResult


class ProjectiveBackend(MeasurementBackend):
    """Ideal state evolution and projective measurement backend using QuTiP mesolve."""

    @property
    def name(self) -> str:
        return "projective"

    def run(
        self,
        transmon: Transmon,
        sequence: PulseSequence,
        init_state: Optional[qutip.Qobj] = None,
        dt: float = 5e-10,
        f_d: Optional[float] = None,
        c_ops: Optional[List[qutip.Qobj]] = None,
        solver_options: Optional[Dict[str, Any]] = None,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> ProjectiveResult:
        """Simulate the time evolution of a Transmon driven by a PulseSequence.

        Args:
            transmon: Physical Transmon model.
            sequence: PulseSequence containing the scheduled pulses.
            init_state: Initial state (Ket or density matrix). Defaults to ground state |0>.
            dt: Simulation time step in seconds (default 5e-10 s = 0.5 ns).
            f_d: Rotating frame reference frequency in Hz (defaults to transmon resonant frequency).
            c_ops: Additional custom Lindblad collapse operators.
            solver_options: Optional dict of options passed to qutip.mesolve.
            shots: Optional number of projective shots to sample.
            seed: Optional seed for random sampling.

        Returns:
            ProjectiveResult containing times, states, and analysis methods.
        """
        if init_state is None:
            init_state = transmon.ground_state()

        times, evo = sequence.to_qutip_evo(transmon, dt=dt, f_d=f_d)

        # Collect Lindblad collapse operators
        all_c_ops = list(transmon.c_ops())
        if c_ops:
            all_c_ops.extend(c_ops)

        mesolve_opts = solver_options or {}
        res = qutip.mesolve(
            evo,
            init_state,
            times,
            c_ops=all_c_ops,
            options=mesolve_opts,
        )

        return ProjectiveResult(
            times=times,
            states=res.states,
            transmon=transmon,
            sequence=sequence,
            shots=shots,
            seed=seed,
        )


__all__ = ["ProjectiveBackend"]
