"""Unified measurement and simulation framework for SQPulse."""

from __future__ import annotations
from typing import Union, Dict, Type, Optional, Any
from ..models.transmon import Transmon
from ..sequence.sequence import PulseSequence
from .base import MeasurementBackend, BaseMeasurementResult
from .projective import (
    ProjectiveBackend,
    ProjectiveResult,
    Simulator,
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
        qubit: Transmon,
        sequence: PulseSequence,
        backend: Union[str, MeasurementBackend] = "projective",
        **kwargs,
    ) -> BaseMeasurementResult:
        """Execute a measurement/simulation with the selected backend.

        Args:
            qubit: Physical Transmon model.
            sequence: PulseSequence to simulate.
            backend: Backend name (e.g. 'projective', 'dispersive') or a MeasurementBackend instance.
            **kwargs: Extra parameters passed to the backend run() method.

        Returns:
            Result instance specific to the backend (e.g. ProjectiveResult, DispersiveResult).
        """
        if isinstance(backend, MeasurementBackend):
            return backend.run(qubit, sequence, **kwargs)

        backend_key = str(backend).lower()
        if backend_key not in cls._backends:
            available = list(cls._backends.keys())
            raise ValueError(
                f"Unknown measurement backend '{backend}'. Available backends: {available}"
            )

        backend_instance = cls._backends[backend_key]()
        return backend_instance.run(qubit, sequence, **kwargs)


__all__ = [
    "Measurement",
    "MeasurementBackend",
    "BaseMeasurementResult",
    "ProjectiveBackend",
    "ProjectiveResult",
    "Simulator",
    "SimulationResult",
    "DispersiveReadoutBackend",
    "DispersiveResult",
    "IQDiscriminator",
]
