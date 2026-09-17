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
        target (Union[Transmon, QuantumSystem, Any]): Physical model or system simulated.
        sequence (PulseSequence): Pulse sequence executed.
    """

    def __init__(
        self,
        target: Any = None,
        sequence: Optional[PulseSequence] = None,
        metadata: Optional[Dict[str, Any]] = None,
        transmon: Any = None,
    ):
        # Support both target and transmon parameter names for backward compatibility
        resolved_target = target if target is not None else transmon
        self.target = resolved_target
        self.transmon = resolved_target
        self.sequence = sequence or (None)
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
        target: Any,
        sequence: PulseSequence,
        **kwargs,
    ) -> BaseMeasurementResult:
        """Execute the measurement/simulation on the given qubit/system and pulse sequence.

        Args:
            target: Transmon qubit model or QuantumSystem composite system.
            sequence: PulseSequence to simulate.
            **kwargs: Backend-specific arguments.

        Returns:
            BaseMeasurementResult instance.
        """
        pass
