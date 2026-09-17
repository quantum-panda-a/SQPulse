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
        target: Any = None,
        sequence: Optional[PulseSequence] = None,
        init_state: Optional[qutip.Qobj] = None,
        dt: float = 5e-10,
        f_d: Optional[Union[float, Dict[str, float]]] = None,
        c_ops: Optional[List[qutip.Qobj]] = None,
        solver_options: Optional[Dict[str, Any]] = None,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
        transmon: Any = None,
        include_dissipation: bool = True,
        **kwargs,
    ) -> ProjectiveResult:
        """Simulate the time evolution of a Transmon or QuantumSystem driven by a PulseSequence.

        Args:
            target: Physical Transmon model or QuantumSystem composite system.
            sequence: PulseSequence containing the scheduled pulses.
            init_state: Initial state (Ket or density matrix). Defaults to ground state.
            dt: Simulation time step in seconds (default 5e-10 s = 0.5 ns).
            f_d: Rotating frame reference frequency in Hz.
            c_ops: Additional custom Lindblad collapse operators.
            solver_options: Optional dict of options passed to qutip.mesolve.
            shots: Optional number of projective measurement shots to sample.
            seed: Optional random seed for shot sampling.
            include_dissipation: Whether to include intrinsic T1/T2 collapse operators from target (default True).

        Returns:
            ProjectiveResult containing times (in seconds), states, and analysis methods.
        """
        backend = ProjectiveBackend()
        return backend.run(
            target=target,
            sequence=sequence,
            init_state=init_state,
            dt=dt,
            f_d=f_d,
            c_ops=c_ops,
            solver_options=solver_options,
            shots=shots,
            seed=seed,
            transmon=transmon,
            include_dissipation=include_dissipation,
            **kwargs,
        )


__all__ = ["Simulator"]
