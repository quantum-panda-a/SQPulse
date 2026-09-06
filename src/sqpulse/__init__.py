"""SQPulse: A modern framework for quantum pulse engineering,
Transmon dynamics simulation, and experimental calibration.
"""

__version__ = "0.1.0"

from .pulses import (
    Pulse,
    GaussianPulse,
    CosinePulse,
    LorentzianPulse,
    SquarePulse,
    FlatTopPulse,
    SechPulse,
    DRAGPulse,
    CustomPulse,
    spectral_leakage,
    compare_pulses,
)
from .models import Transmon
from .sequence import PulseSequence, Channel
from .simulation import Simulator, SimulationResult
from .experiments import (
    RabiExperiment,
    RabiResult,
    T1Experiment,
    T1Result,
    RamseyExperiment,
    RamseyResult,
)

__all__ = [
    "Pulse",
    "GaussianPulse",
    "CosinePulse",
    "LorentzianPulse",
    "SquarePulse",
    "FlatTopPulse",
    "SechPulse",
    "DRAGPulse",
    "CustomPulse",
    "spectral_leakage",
    "compare_pulses",
    "Transmon",
    "PulseSequence",
    "Channel",
    "Simulator",
    "SimulationResult",
    "RabiExperiment",
    "RabiResult",
    "T1Experiment",
    "T1Result",
    "RamseyExperiment",
    "RamseyResult",
]
