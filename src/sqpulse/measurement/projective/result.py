"""Projective measurement and state evolution result container for SQPulse in SI units."""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import qutip

from ...models.transmon import Transmon
from ...sequence.sequence import PulseSequence
from ..base import BaseMeasurementResult


class ProjectiveResult(BaseMeasurementResult):
    """Encapsulates the exact state evolution and projective measurement data in SI units.

    Args:
        times (np.ndarray): 1D array of evolution timestamps in seconds (s).
        states (List[qutip.Qobj]): State kets or density matrices at each timestamp.
        transmon (Transmon): Physical model simulated.
        sequence (PulseSequence): The pulse sequence executed.
        shots (Optional[int]): Number of projective measurement shots sampled, if requested.
        seed (Optional[int]): Random seed for shot sampling.
    """

    def __init__(
        self,
        times: np.ndarray,
        states: List[qutip.Qobj],
        transmon: Transmon,
        sequence: Optional[PulseSequence] = None,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(transmon=transmon, sequence=sequence or PulseSequence())
        self.times = np.asarray(times)
        self.states = states
        self._shots_data: Optional[np.ndarray] = None
        if shots is not None and shots > 0:
            self.sample_shots(shots=shots, seed=seed)

    @property
    def final_state(self) -> qutip.Qobj:
        """Quantum state at the end of the simulation."""
        return self.states[-1]

    def population(self, n: int) -> np.ndarray:
        """Return the population P_n(t) = <n|rho(t)|n> for level n over time."""
        proj = self.transmon.proj(n)
        return np.array([qutip.expect(proj, state) for state in self.states])

    def final_population(self, n: int) -> float:
        """Return the final population in level n."""
        proj = self.transmon.proj(n)
        return float(qutip.expect(proj, self.final_state))

    def populations(self) -> Dict[int, np.ndarray]:
        """Return a dictionary mapping level index n to population array P_n(t)."""
        return {n: self.population(n) for n in range(self.transmon.levels)}

    def bloch_vector(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the Bloch vector trajectory (X(t), Y(t), Z(t)) in the {|0>, |1>} subspace.

        Returns:
            Tuple of 1D arrays (x, y, z).
        """
        sx = self.transmon.sx
        sy = self.transmon.sy
        sz = self.transmon.sz
        x = np.array([qutip.expect(sx, s) for s in self.states])
        y = np.array([qutip.expect(sy, s) for s in self.states])
        z = np.array([qutip.expect(sz, s) for s in self.states])
        return x, y, z

    def sample_shots(self, shots: int = 1024, seed: Optional[int] = None) -> np.ndarray:
        """Sample discrete projective measurement outcomes on the final quantum state.

        Args:
            shots: Number of independent measurement samples.
            seed: Optional random seed.

        Returns:
            1D numpy array of measured integer states (e.g. 0, 1, 2).
        """
        probs = np.array([max(0.0, self.final_population(n)) for n in range(self.transmon.levels)])
        total_p = probs.sum()
        if total_p <= 0:
            probs = np.ones(self.transmon.levels) / self.transmon.levels
        else:
            probs = probs / total_p

        rng = np.random.default_rng(seed)
        self._shots_data = rng.choice(np.arange(self.transmon.levels), size=shots, p=probs)
        return self._shots_data

    @property
    def shots(self) -> Optional[np.ndarray]:
        """Array of sampled single-shot outcomes, or None if sample_shots has not been called."""
        return self._shots_data

    def counts(self) -> Dict[int, int]:
        """Return histogram count dictionary of sampled measurement outcomes."""
        if self._shots_data is None:
            self.sample_shots(shots=1024)
        unique, counts = np.unique(self._shots_data, return_counts=True)
        return {int(u): int(c) for u, c in zip(unique, counts)}

    def plot_populations(
        self,
        levels: Optional[List[int]] = None,
        ax: Optional[plt.Axes] = None,
        figsize: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None,
    ) -> plt.Axes:
        """Plot the level populations P_n(t) as a function of time."""
        if ax is None:
            _, ax = plt.subplots(figsize=figsize or (8, 4.5))

        target_levels = levels if levels is not None else list(range(self.transmon.levels))
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

        for n in target_levels:
            pop = self.population(n)
            c = colors[n % len(colors)]
            ax.plot(self.times, pop, label=f"|{n}⟩ (P{n})", color=c, lw=2)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("State Population")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(title or f"{self.transmon.name} - Dynamics Evolution")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        return ax

    def plot_bloch_vector(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None,
    ) -> plt.Axes:
        """Plot the expectation values <X>(t), <Y>(t), <Z>(t) over time."""
        if ax is None:
            _, ax = plt.subplots(figsize=figsize or (8, 4.5))

        x, y, z = self.bloch_vector()
        ax.plot(self.times, x, label="⟨X⟩", color="#1f77b4", lw=2)
        ax.plot(self.times, y, label="⟨Y⟩", color="#2ca02c", lw=2)
        ax.plot(self.times, z, label="⟨Z⟩", color="#d62728", lw=2)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Bloch Coordinate")
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(title or f"{self.transmon.name} - Bloch Coordinates vs Time")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        return ax

    def plot_bloch_sphere(
        self,
        point_interval: int = 5,
    ) -> qutip.Bloch:
        """Render the state trajectory on the 3D Bloch sphere."""
        b = qutip.Bloch()
        x, y, z = self.bloch_vector()

        indices = np.arange(0, len(x), point_interval)
        b.add_points([x[indices], y[indices], z[indices]], meth="l")
        b.add_vectors([x[-1], y[-1], z[-1]])
        b.show()
        return b


# Alias SimulationResult to ProjectiveResult for full backward compatibility
SimulationResult = ProjectiveResult

__all__ = ["ProjectiveResult", "SimulationResult"]
