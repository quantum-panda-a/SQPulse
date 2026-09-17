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
            shots: Optional number of projective shots to sample.
            seed: Optional seed for random sampling.
            include_dissipation: Whether to include intrinsic T1/T2 collapse operators from target (default True).

        Returns:
            ProjectiveResult containing times, states, and analysis methods.
        """
        resolved_target = target if target is not None else transmon
        if resolved_target is None:
            raise ValueError("Must provide either target or transmon to ProjectiveBackend.run")

        seq = sequence or PulseSequence()

        if init_state is None:
            init_state = resolved_target.ground_state()

        times, evo = seq.to_qutip_evo(resolved_target, dt=dt, f_d=f_d)

        # Collect Lindblad collapse operators
        all_c_ops = []
        if include_dissipation:
            all_c_ops.extend(resolved_target.c_ops())
        if c_ops:
            all_c_ops.extend(c_ops)

        mesolve_opts = dict(solver_options) if solver_options else {}
        if "max_step" not in mesolve_opts and len(times) > 1:
            mesolve_opts["max_step"] = float(times[1] - times[0])

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
            target=resolved_target,
            sequence=seq,
            shots=shots,
            seed=seed,
        )


__all__ = ["ProjectiveBackend"]
