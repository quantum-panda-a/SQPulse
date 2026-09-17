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
        transmon: Any = None,
        target: Any = None,
        sequence: Optional[PulseSequence] = None,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        resolved_target = target if target is not None else transmon
        super().__init__(target=resolved_target, sequence=sequence or PulseSequence())
        self.times = np.asarray(times)
        self.states = states
        self._shots_data: Optional[np.ndarray] = None
        self.is_multi_qubit = hasattr(resolved_target, "modes")
        if shots is not None and shots > 0:
            self.sample_shots(shots=shots, seed=seed)

    @property
    def final_state(self) -> qutip.Qobj:
        """Quantum state at the end of the simulation."""
        return self.states[-1]

    def _parse_state_indices(self, state: Union[int, str, Tuple[int, ...], List[int]]) -> Tuple[int, ...]:
        """Convert a state specification into a tuple of level indices."""
        if not self.is_multi_qubit:
            return (int(state),)
        if isinstance(state, str):
            return tuple(int(c) for c in state)
        elif isinstance(state, (tuple, list)):
            return tuple(int(c) for c in state)
        elif isinstance(state, (int, np.integer)):
            # Unravel flat index
            return tuple(int(x) for x in np.unravel_index(int(state), self.target.levels))
        raise TypeError(f"Invalid state specification: {type(state)}")

    def population(self, state: Union[int, str, Tuple[int, ...], List[int]]) -> np.ndarray:
        """Return the population P_s(t) = <s|rho(t)|s> for level/state s over time.

        For single Transmon: integer level index n (e.g. 0, 1).
        For QuantumSystem: bitstring (e.g. '00', '11') or tuple of integers (e.g. (0, 1)).
        """
        if self.is_multi_qubit:
            indices = self._parse_state_indices(state)
            proj = self.target.proj(*indices)
        else:
            proj = self.transmon.proj(int(state))
        return np.array([qutip.expect(proj, s) for s in self.states])

    def final_population(self, state: Union[int, str, Tuple[int, ...], List[int]]) -> float:
        """Return the final population in level/state s."""
        pop = self.population(state)
        return float(pop[-1])

    def populations(self) -> Dict[Any, np.ndarray]:
        """Return a dictionary mapping state labels to population arrays over time."""
        if not self.is_multi_qubit:
            return {n: self.population(n) for n in range(self.transmon.levels)}

        # For QuantumSystem: build standard computational basis plus any leakage states
        import itertools
        pops = {}
        # Always include computational subspace {|0>, |1>}^N
        comp_states = list(itertools.product([0, 1], repeat=self.target.num_modes))
        for st in comp_states:
            st_str = "".join(str(x) for x in st)
            pops[st_str] = self.population(st)

        # Also check other product states for noticeable leakage (> 0.5% population)
        all_levels = [range(lvl) for lvl in self.target.levels]
        for st in itertools.product(*all_levels):
            if st in comp_states:
                continue
            p_arr = self.population(st)
            if np.max(p_arr) > 0.005:
                st_str = "".join(str(x) for x in st)
                pops[st_str] = p_arr

        return pops

    def fidelity(self, target_state: qutip.Qobj) -> float:
        r"""Compute quantum state fidelity between simulated final state and target state.

        .. math::
            \mathcal{F}(\rho, |\psi\rangle) = \langle\psi|\rho|\psi\rangle

        Returns:
            Fidelity in [0.0, 1.0].
        """
        return float(qutip.fidelity(self.final_state, target_state) ** 2)

    def bloch_vector(self, mode: Optional[Union[str, Any, int]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the Bloch vector trajectory (X(t), Y(t), Z(t)) in the {|0>, |1>} subspace.

        For multi-qubit systems, specify mode to inspect that qubit's reduced Bloch coordinates.
        """
        if self.is_multi_qubit:
            target_mode = self.target.get_mode(mode if mode is not None else 0)
            sx = self.target.sx(target_mode)
            sy = self.target.sy(target_mode)
            sz = self.target.sz(target_mode)
        else:
            sx = self.transmon.sx
            sy = self.transmon.sy
            sz = self.transmon.sz

        x = np.array([qutip.expect(sx, s) for s in self.states])
        y = np.array([qutip.expect(sy, s) for s in self.states])
        z = np.array([qutip.expect(sz, s) for s in self.states])
        return x, y, z

    def sample_shots(self, shots: int = 1024, seed: Optional[int] = None) -> np.ndarray:
        """Sample discrete projective measurement outcomes on the final quantum state.

        Returns:
            1D array of sampled outcomes (integers for single transmon, bitstrings for QuantumSystem).
        """
        rng = np.random.default_rng(seed)
        if not self.is_multi_qubit:
            probs = np.array([max(0.0, self.final_population(n)) for n in range(self.transmon.levels)])
            total_p = probs.sum()
            probs = probs / total_p if total_p > 0 else np.ones(self.transmon.levels) / self.transmon.levels
            self._shots_data = rng.choice(np.arange(self.transmon.levels), size=shots, p=probs)
            return self._shots_data

        import itertools
        all_states = list(itertools.product(*[range(lvl) for lvl in self.target.levels]))
        labels = ["".join(str(x) for x in st) for st in all_states]
        probs = np.array([max(0.0, self.final_population(st)) for st in all_states])
        total_p = probs.sum()
        probs = probs / total_p if total_p > 0 else np.ones(len(all_states)) / len(all_states)
        self._shots_data = rng.choice(labels, size=shots, p=probs)
        return self._shots_data

    @property
    def shots(self) -> Optional[np.ndarray]:
        """Array of sampled single-shot outcomes, or None if sample_shots has not been called."""
        return self._shots_data

    def counts(self) -> Dict[Any, int]:
        """Return histogram count dictionary of sampled measurement outcomes."""
        if self._shots_data is None:
            self.sample_shots(shots=1024)
        unique, counts = np.unique(self._shots_data, return_counts=True)
        return {u if isinstance(u, str) else int(u): int(c) for u, c in zip(unique, counts)}

    def plot_populations(
        self,
        states: Optional[List[Any]] = None,
        levels: Optional[List[int]] = None,
        ax: Optional[plt.Axes] = None,
        figsize: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None,
    ) -> plt.Axes:
        """Plot the state populations P(t) as a function of time."""
        if ax is None:
            _, ax = plt.subplots(figsize=figsize or (8, 4.5))

        colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#7f7f7f"]

        if not self.is_multi_qubit:
            target_levels = levels if levels is not None else (states if states is not None else list(range(self.transmon.levels)))
            for n in target_levels:
                pop = self.population(n)
                c = colors[int(n) % len(colors)]
                ax.plot(self.times, pop, label=f"|{n}⟩ (P{n})", color=c, lw=2)
            sys_name = getattr(self.target, "name", "Transmon")
        else:
            all_pops = self.populations()
            target_keys = states if states is not None else list(all_pops.keys())
            for i, st_key in enumerate(target_keys):
                pop = all_pops.get(st_key, self.population(st_key))
                c = colors[i % len(colors)]
                ax.plot(self.times, pop, label=f"|{st_key}⟩", color=c, lw=2)
            sys_name = getattr(self.target, "name", "QuantumSystem")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("State Population")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(title or f"{sys_name} - Dynamics Evolution")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        return ax

    def plot_bloch_vector(
        self,
        mode: Optional[Union[str, Any, int]] = None,
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
