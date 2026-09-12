"""Simulator proxy interface for projective state evolution in SI units."""

from __future__ import annotations
from typing import List, Optional, Dict, Any
import qutip

from ...models.transmon import Transmon
from ...sequence.sequence import PulseSequence
from .result import ProjectiveResult, SimulationResult
from .backend import ProjectiveBackend


class Simulator:
    """Master equation and Schrödinger equation solver for quantum pulse sequences in SI units.

    Convenience wrapper around ProjectiveBackend for direct dynamics simulation.
    """

    @staticmethod
    def run(
        transmon: Transmon,
        sequence: PulseSequence,
        init_state: Optional[qutip.Qobj] = None,
        dt: float = 5e-10,
        f_d: Optional[float] = None,
        c_ops: Optional[List[qutip.Qobj]] = None,
        solver_options: Optional[Dict[str, Any]] = None,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
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
            shots: Optional number of projective measurement shots to sample.
            seed: Optional random seed for shot sampling.

        Returns:
            ProjectiveResult containing times (in seconds), states, and analysis methods.
        """
        backend = ProjectiveBackend()
        return backend.run(
            transmon=transmon,
            sequence=sequence,
            init_state=init_state,
            dt=dt,
            f_d=f_d,
            c_ops=c_ops,
            solver_options=solver_options,
            shots=shots,
            seed=seed,
        )


__all__ = ["Simulator"]
