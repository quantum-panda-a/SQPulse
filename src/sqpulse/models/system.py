"""Composite multi-mode quantum system and inter-mode coupling models for SQPulse in SI units."""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import qutip

from .transmon import Transmon
from ..units import parse_quantity


class CouplingTerm:
    r"""Represents an interaction Hamiltonian between modes in SI units.

    The Hamiltonian operator is defined as:
    .. math::
        H_c = \text{strength} \times \prod_k O_k

    If ``add_hc=True``, then:
    .. math::
        H_c = \text{strength} \times \left( \prod_k O_k + \prod_k O_k^\dagger \right)

    Args:
        terms: List of tuples ``(mode, op)``, where ``mode`` is a Transmon or mode name,
            and ``op`` is either a qutip.Qobj operator defined in the mode's local subspace,
            or a standard operator name string (e.g. 'a', 'ad', 'n', 'sx', 'sy', 'sz').
        strength: Coupling angular frequency in rad/s (i.e. 2 * pi * g, where g is in Hz).
        add_hc: Whether to add the Hermitian conjugate of the operator product.
        name: Optional label for this interaction term.
    """

    def __init__(
        self,
        terms: List[Tuple[Union[Transmon, str], Union[qutip.Qobj, str]]],
        strength: float = 1.0,
        add_hc: bool = False,
        name: Optional[str] = None,
    ):
        if len(terms) < 1:
            raise ValueError("CouplingTerm must contain at least one (mode, op) pair")
        self.terms = list(terms)
        self.strength = float(strength)
        self.add_hc = bool(add_hc)
        self.name = name or "coupling"

    @classmethod
    def capacitive(
        cls,
        mode1: Union[Transmon, str],
        mode2: Union[Transmon, str],
        g: float,
        name: Optional[str] = None,
    ) -> CouplingTerm:
        r"""Create a standard transverse capacitive exchange coupling term between two modes in SI units.

        Under the rotating wave approximation (RWA):
        .. math::
            H_{\text{cap}} = 2\pi g (a_1^\dagger a_2 + a_1 a_2^\dagger) \quad (\text{rad/s})

        Args:
            mode1: First mode.
            mode2: Second mode.
            g: Coupling strength in Hz (e.g. 10e6 for 10 MHz, or 10 * MHz).
            name: Optional descriptive name.

        Returns:
            CouplingTerm representing 2 * pi * g * (a1^dag * a2 + h.c.).
        """
        g_hz = float(parse_quantity(g))
        strength = 2.0 * np.pi * g_hz
        m1_name = mode1.name if hasattr(mode1, "name") else str(mode1)
        m2_name = mode2.name if hasattr(mode2, "name") else str(mode2)
        term_name = name or f"g_cap({m1_name},{m2_name})"
        return cls(
            terms=[(mode1, "ad"), (mode2, "a")],
            strength=strength,
            add_hc=True,
            name=term_name,
        )

    @classmethod
    def zz(
        cls,
        mode1: Union[Transmon, str],
        mode2: Union[Transmon, str],
        zeta: float,
        name: Optional[str] = None,
    ) -> CouplingTerm:
        r"""Create a static residual ZZ cross-Kerr coupling term between two modes in SI units.

        .. math::
            H_{ZZ} = 2\pi \zeta (a_1^\dagger a_1)(a_2^\dagger a_2) = 2\pi \zeta n_1 n_2 \quad (\text{rad/s})

        Args:
            mode1: First mode.
            mode2: Second mode.
            zeta: Cross-Kerr coupling strength in Hz (e.g. 100e3 for 100 kHz, or 100 * kHz).
            name: Optional descriptive name.

        Returns:
            CouplingTerm representing 2 * pi * zeta * (n1 * n2).
        """
        zeta_hz = float(parse_quantity(zeta))
        strength = 2.0 * np.pi * zeta_hz
        m1_name = mode1.name if hasattr(mode1, "name") else str(mode1)
        m2_name = mode2.name if hasattr(mode2, "name") else str(mode2)
        term_name = name or f"zeta_zz({m1_name},{m2_name})"
        return cls(
            terms=[(mode1, "n"), (mode2, "n")],
            strength=strength,
            add_hc=False,
            name=term_name,
        )

    def to_qobj(self, system: QuantumSystem) -> qutip.Qobj:
        """Convert this coupling term into a qutip.Qobj embedded in the full Hilbert space of system."""
        prod_op = None
        for mode_ref, op_item in self.terms:
            mode = system.get_mode(mode_ref)
            if isinstance(op_item, str):
                op_local = system._get_mode_operator_by_name(mode, op_item)
            elif isinstance(op_item, qutip.Qobj):
                op_local = op_item
            else:
                raise TypeError(f"Expected operator str or qutip.Qobj, got {type(op_item)}")

            op_full = system.tensor_with_I(mode, op_local)
            if prod_op is None:
                prod_op = op_full
            else:
                prod_op = prod_op * op_full

        if self.add_hc:
            prod_op = prod_op + prod_op.dag()

        return self.strength * prod_op

    def __repr__(self) -> str:
        names = []
        for m, op in self.terms:
            m_name = m.name if hasattr(m, "name") else str(m)
            op_name = op if isinstance(op, str) else "Qobj"
            names.append(f"({m_name}, {op_name})")
        return (
            f"CouplingTerm({', '.join(names)}, strength={self.strength / (2 * np.pi):.3e} Hz, "
            f"add_hc={self.add_hc}, name='{self.name}')"
        )


class QuantumSystem:
    r"""Composite multi-mode quantum system container for SQPulse in SI units.

    Orchestrates multiple quantum modes (e.g. Transmons), automatically manages the tensor product
    Hilbert space, mode embedding, static interactions (capacitive coupling, cross-Kerr ZZ),
    and aggregates multi-mode Lindblad master equation dissipation channels.

    Args:
        modes: List of quantum modes (e.g. Transmon objects).
        name: System identifier string.
    """

    def __init__(self, modes: List[Transmon], name: str = "system"):
        if not modes:
            raise ValueError("QuantumSystem must contain at least one mode")

        # Verify uniqueness of mode names
        self.name = str(name)
        self._modes: List[Transmon] = []
        self._mode_map: Dict[str, Transmon] = {}
        for m in modes:
            if not isinstance(m, Transmon):
                raise TypeError(f"Expected Transmon instance, got {type(m)}")
            if m.name in self._mode_map:
                raise ValueError(f"Duplicate mode name '{m.name}' in QuantumSystem")
            self._modes.append(m)
            self._mode_map[m.name] = m

        self._coupling_terms: List[CouplingTerm] = []

    @property
    def modes(self) -> List[Transmon]:
        """List of all modes in the system in their fixed tensor product order."""
        return list(self._modes)

    @property
    def num_modes(self) -> int:
        """Total number of modes in the system."""
        return len(self._modes)

    @property
    def mode_names(self) -> List[str]:
        """List of names of all modes in order."""
        return [m.name for m in self._modes]

    @property
    def levels(self) -> List[int]:
        """List of truncated energy level dimensions for each mode."""
        return [m.levels for m in self._modes]

    @property
    def hilbert_dimension(self) -> int:
        """Total composite Hilbert space dimension (product of all mode levels)."""
        return int(np.prod(self.levels))

    @property
    def coupling_terms(self) -> List[CouplingTerm]:
        """List of inter-mode coupling terms registered in this system."""
        return list(self._coupling_terms)

    def get_mode(self, mode: Union[Transmon, str, int]) -> Transmon:
        """Retrieve a mode by its instance, name, or integer index."""
        if isinstance(mode, Transmon):
            if mode.name not in self._mode_map:
                raise ValueError(f"Mode '{mode.name}' is not registered in this QuantumSystem")
            return mode
        elif isinstance(mode, str):
            if mode not in self._mode_map:
                raise KeyError(f"Mode '{mode}' not found in QuantumSystem. Available: {self.mode_names}")
            return self._mode_map[mode]
        elif isinstance(mode, (int, np.integer)):
            return self._modes[int(mode)]
        raise TypeError(f"Invalid mode reference type: {type(mode)}")

    def get_mode_index(self, mode: Union[Transmon, str, int]) -> int:
        """Retrieve the 0-indexed position of a mode in the composite tensor product."""
        target_mode = self.get_mode(mode)
        return self._modes.index(target_mode)

    def __getitem__(self, item: Union[str, int]) -> Transmon:
        return self.get_mode(item)

    def __getattr__(self, name: str) -> Any:
        # Allow attribute access like system.q0
        if "_mode_map" in self.__dict__ and name in self._mode_map:
            return self._mode_map[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __contains__(self, item: Union[Transmon, str]) -> bool:
        if isinstance(item, Transmon):
            return item.name in self._mode_map
        elif isinstance(item, str):
            return item in self._mode_map
        return False

    def add_coupling(self, term: CouplingTerm) -> QuantumSystem:
        """Add a CouplingTerm to the system.

        Returns:
            self (for method chaining).
        """
        if not isinstance(term, CouplingTerm):
            raise TypeError(f"Expected CouplingTerm, got {type(term)}")
        # Validate that participating modes are in this system
        for m, _ in term.terms:
            _ = self.get_mode(m)
        self._coupling_terms.append(term)
        return self

    def add_capacitive_coupling(
        self,
        mode1: Union[Transmon, str],
        mode2: Union[Transmon, str],
        g: float,
        name: Optional[str] = None,
    ) -> QuantumSystem:
        """Add transverse capacitive coupling 2*pi*g*(a1^dag*a2 + a1*a2^dag) between mode1 and mode2.

        Args:
            mode1: First mode.
            mode2: Second mode.
            g: Coupling rate in Hz (e.g. 10e6 or 10 * MHz).
            name: Optional label.
        """
        term = CouplingTerm.capacitive(mode1, mode2, g=g, name=name)
        return self.add_coupling(term)

    def add_zz_coupling(
        self,
        mode1: Union[Transmon, str],
        mode2: Union[Transmon, str],
        zeta: float,
        name: Optional[str] = None,
    ) -> QuantumSystem:
        """Add static residual ZZ cross-Kerr coupling 2*pi*zeta*(n1*n2) between mode1 and mode2.

        Args:
            mode1: First mode.
            mode2: Second mode.
            zeta: Cross-Kerr rate in Hz (e.g. 100e3 or 100 * kHz).
            name: Optional label.
        """
        term = CouplingTerm.zz(mode1, mode2, zeta=zeta, name=name)
        return self.add_coupling(term)

    def tensor_with_I(self, mode: Union[Transmon, str, int], op: qutip.Qobj) -> qutip.Qobj:
        r"""Embed a single-mode operator into the full Hilbert space by tensoring with Identity on all other modes.

        .. math::
            O_{\text{full}} = I_0 \otimes \dots \otimes I_{k-1} \otimes O_k \otimes I_{k+1} \otimes \dots \otimes I_{N-1}

        Args:
            mode: Target mode.
            op: Local operator acting on mode's subspace (dimension must match mode.levels).

        Returns:
            qutip.Qobj on the composite Hilbert space.
        """
        idx = self.get_mode_index(mode)
        target_mode = self._modes[idx]

        if op.dims[0][0] != target_mode.levels or op.dims[1][0] != target_mode.levels:
            raise ValueError(
                f"Operator dimension {op.dims} does not match mode '{target_mode.name}' levels={target_mode.levels}"
            )

        ops = [qutip.qeye(m.levels) for m in self._modes]
        ops[idx] = op
        return qutip.tensor(*ops)

    def _get_mode_operator_by_name(self, mode: Transmon, op_name: str) -> qutip.Qobj:
        """Internal helper to retrieve standard single-mode local operators by name."""
        op_clean = op_name.lower().strip()
        if op_clean in ("a", "destroy"):
            return mode.a
        elif op_clean in ("ad", "create", "a_dag", "dag"):
            return mode.ad
        elif op_clean in ("n", "num", "number"):
            return mode.n
        elif op_clean in ("i", "eye", "identity"):
            return mode.I
        elif op_clean in ("sx", "sigmax", "x"):
            return mode.sx
        elif op_clean in ("sy", "sigmay", "y"):
            return mode.sy
        elif op_clean in ("sz", "sigmaz", "z"):
            return mode.sz
        elif op_clean in ("hx", "h_drive_x"):
            return mode.H_drive_x
        elif op_clean in ("hy", "h_drive_y"):
            return mode.H_drive_y
        raise ValueError(f"Unknown operator name '{op_name}' for mode '{mode.name}'")

    # Convenience operator accessors embedded in full space
    def a(self, mode: Union[Transmon, str, int]) -> qutip.Qobj:
        """Annihilation operator for mode embedded in full space."""
        m = self.get_mode(mode)
        return self.tensor_with_I(m, m.a)

    def ad(self, mode: Union[Transmon, str, int]) -> qutip.Qobj:
        """Creation operator for mode embedded in full space."""
        m = self.get_mode(mode)
        return self.tensor_with_I(m, m.ad)

    def n(self, mode: Union[Transmon, str, int]) -> qutip.Qobj:
        """Number operator for mode embedded in full space."""
        m = self.get_mode(mode)
        return self.tensor_with_I(m, m.n)

    def sx(self, mode: Union[Transmon, str, int]) -> qutip.Qobj:
        """Pauli X operator in {|0>, |1>} subspace for mode embedded in full space."""
        m = self.get_mode(mode)
        return self.tensor_with_I(m, m.sx)

    def sy(self, mode: Union[Transmon, str, int]) -> qutip.Qobj:
        """Pauli Y operator in {|0>, |1>} subspace for mode embedded in full space."""
        m = self.get_mode(mode)
        return self.tensor_with_I(m, m.sy)

    def sz(self, mode: Union[Transmon, str, int]) -> qutip.Qobj:
        """Pauli Z operator in {|0>, |1>} subspace for mode embedded in full space."""
        m = self.get_mode(mode)
        return self.tensor_with_I(m, m.sz)

    def H_drive_x(self, mode: Union[Transmon, str, int]) -> qutip.Qobj:
        """Microwave X drive operator for mode embedded in full space."""
        m = self.get_mode(mode)
        return self.tensor_with_I(m, m.H_drive_x)

    def H_drive_y(self, mode: Union[Transmon, str, int]) -> qutip.Qobj:
        """Microwave Y drive operator for mode embedded in full space."""
        m = self.get_mode(mode)
        return self.tensor_with_I(m, m.H_drive_y)

    def fock(self, *args, **kwargs) -> qutip.Qobj:
        r"""Construct a product Fock state ket in the composite Hilbert space.

        States can be specified positionally or via keyword arguments:
        - Positionally: ``system.fock(0, 1)`` corresponds to :math:`|0\rangle \otimes |1\rangle`.
        - By keyword: ``system.fock(q0=0, q1=1)``.
        - Default for omitted modes is 0 (ground state).

        Returns:
            qutip.Qobj ket vector.
        """
        state_indices = [0] * self.num_modes
        if args:
            if kwargs:
                raise ValueError("Cannot mix positional and keyword arguments in fock()")
            if len(args) != self.num_modes:
                raise ValueError(
                    f"Expected {self.num_modes} positional arguments matching modes {self.mode_names}, got {len(args)}"
                )
            state_indices = [int(a) for a in args]
        elif kwargs:
            for k, val in kwargs.items():
                idx = self.get_mode_index(k)
                state_indices[idx] = int(val)

        # Validate bounds
        kets = []
        for i, (m, level_idx) in enumerate(zip(self._modes, state_indices)):
            if not (0 <= level_idx < m.levels):
                raise ValueError(
                    f"Level index {level_idx} for mode '{m.name}' out of range [0, {m.levels - 1}]"
                )
            kets.append(qutip.basis(m.levels, level_idx))

        return qutip.tensor(*kets)

    basis = fock

    def ground_state(self) -> qutip.Qobj:
        """Return the composite ground state |00...0>."""
        return self.fock(*([0] * self.num_modes))

    def proj(self, *args, **kwargs) -> qutip.Qobj:
        """Construct the projection operator |s><s| for a composite Fock state."""
        ket = self.fock(*args, **kwargs)
        return ket * ket.dag()

    def c_ops(self) -> List[qutip.Qobj]:
        """Aggregate Lindblad collapse operators from all constituent modes, embedded in the full Hilbert space.

        Returns:
            List of qutip.Qobj collapse operators (in s^-1/2).
        """
        all_c_ops = []
        for m in self._modes:
            m_c_ops = m.c_ops()
            for c in m_c_ops:
                all_c_ops.append(self.tensor_with_I(m, c))
        return all_c_ops

    def H0(
        self,
        f_d: Optional[Union[float, Dict[str, float]]] = None,
        fluxes: Optional[Dict[str, float]] = None,
    ) -> qutip.Qobj:
        r"""Compute the static multi-mode Hamiltonian in SI units (rad/s).

        Includes the sum of individual mode Kerr/detuning Hamiltonians and all inter-mode coupling terms:
        .. math::
            H_0 = \sum_{i} H_{0, i} + \sum_{k} H_{\text{coupling}, k}

        Args:
            f_d: Rotating frame drive reference frequency (in Hz). Can be:
                - None: Each mode uses its resonant frequency at its static flux offset (detunings are zero at sweet spots).
                - float: A single common carrier frequency for all modes.
                - dict: Mapping from mode name to reference frequency (e.g. {'q0': 5e9, 'q1': 4.6e9}).
            fluxes: Optional dictionary mapping mode name to static flux bias (in Phi_0).

        Returns:
            qutip.Qobj static Hamiltonian on the composite Hilbert space.
        """
        H_total = qutip.qzero(self.levels)
        flux_dict = fluxes or {}

        # 1. Sum individual mode static Hamiltonians
        for m in self._modes:
            phi = flux_dict.get(m.name, m.flux_offset)
            if f_d is None:
                # Default unified rotating frame: reference frequency of the first mode
                m_fd = self._modes[0].frequency_at_flux(self._modes[0].flux_offset)
            elif isinstance(f_d, (int, float, np.floating)):
                m_fd = float(f_d)
            elif isinstance(f_d, dict):
                m_fd = float(f_d.get(m.name, m.frequency_at_flux(phi)))
            else:
                raise TypeError(f"Invalid f_d specification: {type(f_d)}")

            h0_local = m.H0(f_d=m_fd, flux=phi)
            H_total += self.tensor_with_I(m, h0_local)

        # 2. Add static coupling terms
        for term in self._coupling_terms:
            H_total += term.to_qobj(self)

        return H_total

    def __repr__(self) -> str:
        mode_str = ", ".join(m.name for m in self._modes)
        n_couplings = len(self._coupling_terms)
        return (
            f"QuantumSystem(name='{self.name}', modes=[{mode_str}], "
            f"dim={self.hilbert_dimension}, couplings={n_couplings})"
        )


__all__ = ["CouplingTerm", "QuantumSystem"]
