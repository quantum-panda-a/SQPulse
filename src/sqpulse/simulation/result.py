"""Simulation result container and analysis tools for SQPulse."""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import qutip

from ..models.transmon import Transmon


class SimulationResult:
    """Encapsulates the time evolution data of a simulated quantum system.

    Args:
        times (np.ndarray): 1D array of evolution timestamps in ns.
        states (List[qutip.Qobj]): State kets or density matrices at each timestamp.
        transmon (Transmon): Physical model simulated.
    """

    def __init__(
        self,
        times: np.ndarray,
        states: List[qutip.Qobj],
        transmon: Transmon,
    ):
        self.times = np.asarray(times)
        self.states = states
        self.transmon = transmon

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

    def plot_populations(
        self,
        levels: Optional[List[int]] = None,
        ax: Optional[plt.Axes] = None,
        figsize: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None,
    ) -> plt.Axes:
        """Plot the level populations P_n(t) as a function of time.

        Args:
            levels: Optional list of levels to plot (defaults to all).
            ax: Optional matplotlib axes.
            figsize: Figure size tuple.
            title: Title string.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize or (8, 4.5))

        target_levels = levels if levels is not None else list(range(self.transmon.levels))
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

        for n in target_levels:
            pop = self.population(n)
            c = colors[n % len(colors)]
            ax.plot(self.times, pop, label=f"|{n}⟩ (P{n})", color=c, lw=2)

        ax.set_xlabel("Time (ns)")
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

        ax.set_xlabel("Time (ns)")
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

        # Downsample trajectory points to keep plot clean
        indices = np.arange(0, len(x), point_interval)
        b.add_points([x[indices], y[indices], z[indices]], meth="l")
        # Add final point as a vector
        b.add_vectors([x[-1], y[-1], z[-1]])
        b.show()
        return b
