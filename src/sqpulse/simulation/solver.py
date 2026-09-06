"""Dynamic evolution solver for SQPulse in SI units."""

from __future__ import annotations
from typing import List, Optional, Dict, Any
import numpy as np
import qutip

from ..models.transmon import Transmon
from ..sequence.sequence import PulseSequence
from .result import SimulationResult


class Simulator:
    """Master equation and Schrödinger equation solver for quantum pulse sequences in SI units."""

    @staticmethod
    def run(
        transmon: Transmon,
        sequence: PulseSequence,
        init_state: Optional[qutip.Qobj] = None,
        dt: float = 5e-10,
        f_d: Optional[float] = None,
        c_ops: Optional[List[qutip.Qobj]] = None,
        solver_options: Optional[Dict[str, Any]] = None,
    ) -> SimulationResult:
        """Simulate the time evolution of a Transmon driven by a PulseSequence.

        Args:
            transmon: Physical Transmon model.
            sequence: PulseSequence containing the scheduled pulses.
            init_state: Initial state (Ket or density matrix). Defaults to ground state |0>.
            dt: Simulation time step in seconds (default 5e-10 s = 0.5 ns).
            f_d: Rotating frame reference frequency in Hz (defaults to transmon.f_q).
            c_ops: Additional custom Lindblad collapse operators.
            solver_options: Optional dict of options passed to qutip.mesolve.

        Returns:
            SimulationResult containing times (in seconds) and states.
        """
        if init_state is None:
            init_state = transmon.ground_state()

        times, evo = sequence.to_qutip_evo(transmon, dt=dt, f_d=f_d)

        # Collect Lindblad collapse operators
        all_c_ops = list(transmon.c_ops())
        if c_ops:
            all_c_ops.extend(c_ops)

        # Solve system dynamics
        mesolve_opts = solver_options or {}
        res = qutip.mesolve(
            evo,
            init_state,
            times,
            c_ops=all_c_ops,
            options=mesolve_opts,
        )

        return SimulationResult(
            times=times,
            states=res.states,
            transmon=transmon,
        )
