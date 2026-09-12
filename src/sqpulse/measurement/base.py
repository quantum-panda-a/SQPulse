"""Base abstractions for the SQPulse measurement and simulation framework."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Any, Dict

if TYPE_CHECKING:
    from ..models.transmon import Transmon
    from ..sequence.sequence import PulseSequence


class BaseMeasurementResult:
    """Base container for measurement and simulation results in SI units.

    Args:
        transmon (Transmon): Physical model simulated.
        sequence (PulseSequence): Pulse sequence executed.
    """

    def __init__(
        self,
        transmon: Transmon,
        sequence: PulseSequence,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.transmon = transmon
        self.sequence = sequence
        self.metadata = metadata or {}


class MeasurementBackend(ABC):
    """Abstract base class for all calculation and measurement backends in SQPulse."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the backend."""
        pass

    @abstractmethod
    def run(
        self,
        transmon: Transmon,
        sequence: PulseSequence,
        **kwargs,
    ) -> BaseMeasurementResult:
        """Execute the measurement/simulation on the given qubit and pulse sequence.

        Args:
            transmon: Transmon qubit model.
            sequence: PulseSequence to simulate.
            **kwargs: Backend-specific arguments.

        Returns:
            BaseMeasurementResult instance.
        """
        pass
