"""Unified measurement and simulation framework for SQPulse."""

from __future__ import annotations
from typing import Union, Dict, Type, Optional, Any
from ..models.transmon import Transmon
from ..sequence.sequence import PulseSequence
from .base import MeasurementBackend, BaseMeasurementResult
from .projective import (
    ProjectiveBackend,
    ProjectiveResult,
    SimulationResult,
)
from .dispersive import DispersiveReadoutBackend, DispersiveResult, IQDiscriminator


class Measurement:
    """Unified entry point for quantum simulation and measurement in SQPulse.

    Supports multiple backends:
    - 'projective': Ideal master equation evolution, continuous population/Bloch tracking, and projective sampling.
    - 'dispersive': Physical readout resonator simulation with Langevin cavity photon dynamics, IQ demodulation, noise, and confusion matrix.
    """

    _backends: Dict[str, Type[MeasurementBackend]] = {
        "projective": ProjectiveBackend,
        "statevector": ProjectiveBackend,
        "ideal": ProjectiveBackend,
        "dispersive": DispersiveReadoutBackend,
        "readout": DispersiveReadoutBackend,
        "cavity": DispersiveReadoutBackend,
    }

    @classmethod
    def register_backend(cls, name: str, backend_cls: Type[MeasurementBackend]) -> None:
        """Register a new measurement backend."""
        cls._backends[name.lower()] = backend_cls

    @classmethod
    def run(
        cls,
        target: Optional[Any] = None,
        sequence: Optional[PulseSequence] = None,
        backend: Union[str, MeasurementBackend] = "projective",
        **kwargs,
    ) -> BaseMeasurementResult:
        """Execute a measurement or state evolution simulation with the selected backend.

        Args:
            target: Physical quantum model (e.g. Transmon) or composite system (QuantumSystem).
            sequence: PulseSequence to simulate.
            backend: Backend name (e.g. 'projective', 'dispersive') or a MeasurementBackend instance (default 'projective').
            **kwargs: Extra backend-specific parameters passed to the backend run() method:
                For 'projective' backend:
                    - dt (float): Simulation time step in seconds (default 5e-10 s = 0.5 ns).
                    - f_d (float or dict): Rotating frame reference frequency in Hz.
                    - init_state (qutip.Qobj): Initial state vector or density matrix (default ground state).
                    - shots (int): Number of projective shots to sample.
                    - seed (int): Random seed for shot sampling.
                    - c_ops (list[qutip.Qobj]): Custom Lindblad collapse operators.
                    - solver_options (dict): Options passed to qutip.mesolve.
                    - include_dissipation (bool): Whether to include intrinsic T1/T2 collapse operators (default True).
                For 'dispersive' backend:
                    - resonator (ReadoutResonator): Readout resonator cavity model.
                    - readout_pulse (Pulse): Micro-wave pulse applied to readout resonator.
                    - shots (int): Number of readout shots to sample (default 1000).
                    - snr_db (float): Readout signal-to-noise ratio in dB (default 12.0).
                    - f_ro (float): Readout carrier frequency in Hz.
                    - dt (float): Cavity ODE simulation time step in seconds (default 1e-9 s = 1 ns).
                    - seed (int): Random seed for noise and shot sampling.

        Returns:
            Result instance specific to the backend (ProjectiveResult or DispersiveResult).
        """
        if target is None:
            raise ValueError("Must provide target to Measurement.run")

        seq = sequence or PulseSequence()

        if isinstance(backend, MeasurementBackend):
            return backend.run(target, seq, **kwargs)

        backend_key = str(backend).lower()
        if backend_key not in cls._backends:
            available = list(cls._backends.keys())
            raise ValueError(
                f"Unknown measurement backend '{backend}'. Available backends: {available}"
            )

        backend_instance = cls._backends[backend_key]()
        return backend_instance.run(target, seq, **kwargs)


__all__ = [
    "Measurement",
    "MeasurementBackend",
    "BaseMeasurementResult",
    "ProjectiveBackend",
    "ProjectiveResult",
    "SimulationResult",
    "DispersiveReadoutBackend",
    "DispersiveResult",
    "IQDiscriminator",
]

