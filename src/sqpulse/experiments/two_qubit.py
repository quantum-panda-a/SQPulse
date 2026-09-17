"""Pulse-level two-qubit quantum gate implementations and dynamics analysis for SQPulse in SI units."""

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple, Union, List
import numpy as np
import matplotlib.pyplot as plt
import qutip
from scipy.optimize import brentq

from ..models.transmon import Transmon
from ..models.system import QuantumSystem
from ..pulses.shapes import FlatTopPulse, SquarePulse, GaussianPulse
from ..sequence.sequence import PulseSequence
from ..measurement import Measurement
from ..measurement.projective.result import ProjectiveResult


class TwoQubitGateResult:
    """Encapsulates the simulation, process matrix, and fidelity analysis of a two-qubit gate.

    Args:
        gate_name: Name of the gate (e.g. 'FluxCZ', 'FluxISWAP').
        system: QuantumSystem on which the gate was executed.
        sequence: The PulseSequence that realized the gate.
        unitary: Reconstructed 4x4 unitary matrix in computational basis {|00>, |01>, |10>, |11>}.
        target_unitary: Ideal theoretical 4x4 target unitary matrix.
        state_results: Dict mapping input basis string ('00', '01', '10', '11') to ProjectiveResult.
    """

    def __init__(
        self,
        gate_name: str,
        system: QuantumSystem,
        sequence: PulseSequence,
        unitary: np.ndarray,
        target_unitary: Optional[np.ndarray] = None,
        state_results: Optional[Dict[str, ProjectiveResult]] = None,
    ):
        self.gate_name = gate_name
        self.system = system
        self.sequence = sequence
        self.unitary = np.asarray(unitary, dtype=complex)
        self.target_unitary = np.asarray(target_unitary, dtype=complex) if target_unitary is not None else None
        self.state_results = state_results or {}

    def calculate_fidelity(self, virtual_z_correction: bool = True) -> Tuple[Optional[float], Optional[float]]:
        r"""Compute entangling fidelity and average gate fidelity against target_unitary.

        Args:
            virtual_z_correction: If True (default), optimizes single-qubit Z frame phases
                diag(1, e^{i phi_2}, e^{i phi_1}, e^{i (phi_1 + phi_2)}), corresponding to
                standard Virtual-Z gate calibration on superconducting hardware.

        Returns:
            Tuple of (entangling_fidelity, average_gate_fidelity).
        """
        if self.target_unitary is None:
            return None, None
        d = 4
        if not virtual_z_correction:
            tr = np.trace(self.target_unitary.conj().T @ self.unitary)
            f_ent = float((np.abs(tr) ** 2) / (d ** 2))
            f_avg = float((d * f_ent + 1.0) / (d + 1.0))
            return float(np.clip(f_ent, 0.0, 1.0)), float(np.clip(f_avg, 0.0, 1.0))

        import scipy.optimize as opt

        def loss(p):
            # p = [phi_1, phi_2] single-qubit Z frame phases
            r_z = np.diag([1.0, np.exp(1j * p[1]), np.exp(1j * p[0]), np.exp(1j * (p[0] + p[1]))])
            target_adj = r_z @ self.target_unitary
            tr = np.trace(target_adj.conj().T @ self.unitary)
            return -float(np.abs(tr) ** 2 / (d ** 2))

        res = opt.minimize(loss, [0.0, 0.0], method="Nelder-Mead")
        f_ent = float(-res.fun)
        f_avg = float((d * f_ent + 1.0) / (d + 1.0))
        return float(np.clip(f_ent, 0.0, 1.0)), float(np.clip(f_avg, 0.0, 1.0))

    @property
    def average_gate_fidelity(self) -> Optional[float]:
        r"""Compute the average gate fidelity against target_unitary with Virtual-Z correction.

        .. math::
            F_{\text{avg}} = \frac{|\text{Tr}(U_{\text{target}}^\dagger U)|^2 + d}{d(d+1)} \quad (d=4)
        """
        _, f_avg = self.calculate_fidelity(virtual_z_correction=True)
        return f_avg

    @property
    def entangling_fidelity(self) -> Optional[float]:
        """Entangling gate fidelity against target_unitary with Virtual-Z correction."""
        f_ent, _ = self.calculate_fidelity(virtual_z_correction=True)
        return f_ent

    @property
    def leakage(self) -> float:
        """Measure of population leakage out of computational subspace {|00>, |01>, |10>, |11>}."""
        u = self.unitary
        # Frobenius norm squared of the 4x4 matrix divided by 4 (1.0 for unitary)
        subspace_retention = np.sum(np.abs(u) ** 2) / 4.0
        return float(max(0.0, 1.0 - subspace_retention))

    def plot_matrix(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Visualize the real and imaginary parts of the reconstructed 4x4 process matrix."""
        fig, axes = plt.subplots(1, 2, figsize=figsize or (10, 4.5))
        labels = ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]

        im0 = axes[0].imshow(self.unitary.real, cmap="Blues", vmin=-1, vmax=1)
        axes[0].set_title(f"Re[{self.gate_name}]")
        axes[0].set_xticks(range(4))
        axes[0].set_yticks(range(4))
        axes[0].set_xticklabels(labels)
        axes[0].set_yticklabels(labels)
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(self.unitary.imag, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[1].set_title(f"Im[{self.gate_name}]")
        axes[1].set_xticks(range(4))
        axes[1].set_yticks(range(4))
        axes[1].set_xticklabels(labels)
        axes[1].set_yticklabels(labels)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        fid_str = f" (F_avg = {self.average_gate_fidelity:.3%})" if self.average_gate_fidelity is not None else ""
        fig.suptitle(title or f"{self.gate_name} Reconstructed Matrix{fid_str}", y=1.02)
        fig.tight_layout()
        return fig


class FluxISWAP:
    """Pulse-level resonant flux-tunable iSWAP gate implementation and calibration in SI units.

    Brings a flux-tunable Transmon q1 into resonance with target Transmon q2 (f1(Phi) = f2)
    for interaction duration tau = 1 / (4 * g).
    """

    @staticmethod
    def calculate_resonance_flux(
        tunable_qubit: Transmon,
        target_frequency: float,
    ) -> float:
        """Find the magnetic flux (in units of Phi_0) that tunes tunable_qubit to target_frequency."""
        if tunable_qubit.d >= 1.0:
            raise ValueError(f"Transmon '{tunable_qubit.name}' has d=1.0 (fixed-frequency, cannot tune flux)")

        f_min = tunable_qubit.frequency_at_flux(0.5)
        f_max = tunable_qubit.frequency_at_flux(0.0)
        if not (f_min <= target_frequency <= f_max):
            raise ValueError(
                f"Target frequency {target_frequency/1e9:.3f} GHz is out of tunable range "
                f"[{f_min/1e9:.3f}, {f_max/1e9:.3f}] GHz for '{tunable_qubit.name}'"
            )

        def obj(phi):
            return tunable_qubit.frequency_at_flux(phi) - target_frequency

        return float(brentq(obj, 0.0, 0.5))

    @classmethod
    def create_sequence(
        cls,
        system: QuantumSystem,
        q_tunable: Union[Transmon, str] = 0,
        q_target: Union[Transmon, str] = 1,
        g: Optional[float] = None,
        duration: Optional[float] = None,
        ramp_time: float = 2.0e-9,
        pulse_shape: str = "flattop",
    ) -> PulseSequence:
        """Construct a PulseSequence for a flux-pulsed iSWAP gate.

        Args:
            system: QuantumSystem containing the coupled qubits.
            q_tunable: Tunable transmon (control line q.z will be pulsed).
            q_target: Target transmon.
            g: Coupling strength in Hz. If None, inferred from system coupling terms.
            duration: Gate interaction duration in seconds. If None, calculated as 1 / (4 * g).
            ramp_time: Ramp edge duration for smooth flat-top pulses in seconds (default 2 ns).
            pulse_shape: 'flattop' or 'square'.

        Returns:
            PulseSequence implementing the pulse-level iSWAP.
        """
        q1 = system.get_mode(q_tunable)
        q2 = system.get_mode(q_target)

        # Infer coupling g if not explicitly provided
        resolved_g = g
        if resolved_g is None:
            for term in system.coupling_terms:
                modes_in_term = [system.get_mode(m).name for m, _ in term.terms]
                if q1.name in modes_in_term and q2.name in modes_in_term:
                    resolved_g = term.strength / (2.0 * np.pi)
                    break
        if resolved_g is None:
            resolved_g = 10.0e6  # Default fallback 10 MHz

        # Resonance condition: tune q1 to q2 frequency
        target_f = q2.frequency_at_flux(q2.flux_offset)
        res_flux = cls.calculate_resonance_flux(q1, target_f)
        flux_amp = res_flux - q1.flux_offset

        # Interaction duration: 1 / (4 * g) for pi/2 angle in single-excitation subspace
        if duration is not None:
            gate_dur = float(duration)
        else:
            gate_dur = 1.0 / (4.0 * resolved_g)

        # Convert flux amplitude to voltage amplitude if v_phi0 is set
        v_amp = q1.flux_to_voltage(flux_amp)

        seq = PulseSequence(name=f"iswap_{q1.name}_{q2.name}")
        if pulse_shape == "flattop":
            eff_ramp = min(ramp_time, gate_dur / 3.0)
            p = FlatTopPulse(duration=gate_dur, ramp_time=eff_ramp, amp=v_amp)
        else:
            p = SquarePulse(duration=gate_dur, amp=v_amp)

        seq.add(q1.z, p)
        return seq


class FluxCZ:
    """Pulse-level Controlled-Z (CZ) gate implementation and calibration in SI units.

    Brings state |11> into avoided crossing with |02> by pulsing tunable Transmon q1
    to frequency f1(Phi) ~= f2 + alpha2 = f2 - |alpha2| for duration tau ~= 1 / (sqrt(2) * g).
    Accumulates a conditional pi phase on |11> while preserving single-qubit states.
    """

    @staticmethod
    def calculate_cz_flux(
        tunable_qubit: Transmon,
        target_qubit: Transmon,
    ) -> float:
        r"""Find the flux bias (in Phi_0) that brings |11> into resonance with |02>.

        Resonance condition:
        .. math::
            f_1(\Phi) = f_2 + \alpha_2 = f_2 - |\alpha_2|
        """
        f2 = target_qubit.frequency_at_flux(target_qubit.flux_offset)
        alpha2 = target_qubit.alpha
        target_f1 = f2 + alpha2  # Note: alpha2 is negative, so f2 + alpha2 < f2

        f_min = tunable_qubit.frequency_at_flux(0.5)
        f_max = tunable_qubit.frequency_at_flux(0.0)
        if not (f_min <= target_f1 <= f_max):
            raise ValueError(
                f"Required |11>-|02> resonance frequency {target_f1/1e9:.3f} GHz is out of tunable range "
                f"[{f_min/1e9:.3f}, {f_max/1e9:.3f}] GHz for '{tunable_qubit.name}'"
            )

        def obj(phi):
            return tunable_qubit.frequency_at_flux(phi) - target_f1

        return float(brentq(obj, 0.0, 0.5))

    @classmethod
    def create_sequence(
        cls,
        system: QuantumSystem,
        q_tunable: Union[Transmon, str] = 0,
        q_target: Union[Transmon, str] = 1,
        g: Optional[float] = None,
        duration: Optional[float] = None,
        ramp_time: float = 3.0e-9,
        pulse_shape: str = "flattop",
    ) -> PulseSequence:
        """Construct a PulseSequence for a flux-pulsed Controlled-Z (CZ) gate.

        Args:
            system: QuantumSystem containing the coupled qubits.
            q_tunable: Tunable transmon (control line q.z will be pulsed).
            q_target: Target transmon.
            g: Capacitive coupling strength in Hz. If None, inferred from system.
            duration: Gate interaction duration in seconds. If None, calculated as 1 / (sqrt(2) * g).
            ramp_time: Ramp edge duration for smooth flat-top pulses in seconds (default 3 ns).
            pulse_shape: 'flattop' or 'square'.

        Returns:
            PulseSequence implementing the pulse-level CZ gate.
        """
        q1 = system.get_mode(q_tunable)
        q2 = system.get_mode(q_target)

        # Infer coupling g if not explicitly provided
        resolved_g = g
        if resolved_g is None:
            for term in system.coupling_terms:
                modes_in_term = [system.get_mode(m).name for m, _ in term.terms]
                if q1.name in modes_in_term and q2.name in modes_in_term:
                    resolved_g = term.strength / (2.0 * np.pi)
                    break
        if resolved_g is None:
            resolved_g = 10.0e6  # Default fallback 10 MHz

        # CZ resonance condition: |11> and |02>
        res_flux = cls.calculate_cz_flux(q1, q2)
        flux_amp = res_flux - q1.flux_offset

        # Effective interaction strength in |11> <-> |02> manifold is sqrt(2) * g
        # Full 2*pi excursion period is tau = 1 / (sqrt(2) * g)
        if duration is not None:
            gate_dur = float(duration)
        else:
            gate_dur = 1.0 / (np.sqrt(2.0) * resolved_g)

        # Convert flux amplitude to voltage amplitude if v_phi0 is set
        v_amp = q1.flux_to_voltage(flux_amp)

        seq = PulseSequence(name=f"cz_{q1.name}_{q2.name}")
        if pulse_shape == "flattop":
            eff_ramp = min(ramp_time, gate_dur / 3.0)
            p = FlatTopPulse(duration=gate_dur, ramp_time=eff_ramp, amp=v_amp)
        else:
            p = SquarePulse(duration=gate_dur, amp=v_amp)

        seq.add(q1.z, p)
        return seq


def evaluate_two_qubit_gate(
    system: QuantumSystem,
    sequence: PulseSequence,
    gate_name: str = "TwoQubitGate",
    target_unitary: Optional[np.ndarray] = None,
    dt: float = 2e-10,
    f_d: Optional[Union[float, Dict[str, float]]] = None,
    include_dissipation: bool = False,
) -> TwoQubitGateResult:
    """Evaluate the action of a PulseSequence on the two-qubit computational subspace.

    Runs time-dependent simulation starting from each of the 4 computational basis states:
    |00>, |01>, |10>, |11>, and reconstructs the 4x4 unitary matrix in this basis.

    Args:
        system: QuantumSystem containing the qubits (first two modes are evaluated).
        sequence: The PulseSequence to evaluate.
        gate_name: Descriptive name for reporting.
        target_unitary: Optional 4x4 matrix for average fidelity comparison.
        dt: Simulation step in seconds (default 2e-10 s = 0.2 ns).
        f_d: Rotating frame reference frequency.
        include_dissipation: Whether to include T1/T2 decoherence (default False for coherent unitary reconstruction).

    Returns:
        TwoQubitGateResult containing reconstructed matrix and fidelity analysis.
    """
    basis_states = ["00", "01", "10", "11"]
    state_results = {}
    u_mat = np.zeros((4, 4), dtype=complex)

    kets_basis = [system.fock(int(s[0]), int(s[1])) for s in basis_states]

    for j, s_in in enumerate(basis_states):
        init_k = kets_basis[j]
        res = Measurement.run(
            target=system,
            sequence=sequence,
            backend="projective",
            init_state=init_k,
            dt=dt,
            f_d=f_d,
            include_dissipation=include_dissipation,
        )
        state_results[s_in] = res
        psi_final = res.final_state

        for i, s_out in enumerate(basis_states):
            bra_out = kets_basis[i].dag()
            if psi_final.isket:
                prod = bra_out * psi_final
                val = complex(prod.full()[0, 0]) if hasattr(prod, "full") else complex(prod)
            else:
                # Open-system density matrix: extract subspace transition element
                prod = bra_out * psi_final * bra_out.dag()
                val = complex(prod.full()[0, 0]) if hasattr(prod, "full") else complex(prod)
            u_mat[i, j] = val

    return TwoQubitGateResult(
        gate_name=gate_name,
        system=system,
        sequence=sequence,
        unitary=u_mat,
        target_unitary=target_unitary,
        state_results=state_results,
    )


# Convenience alias functions
flux_iswap_sequence = FluxISWAP.create_sequence
flux_cz_sequence = FluxCZ.create_sequence

__all__ = [
    "TwoQubitGateResult",
    "FluxISWAP",
    "FluxCZ",
    "flux_iswap_sequence",
    "flux_cz_sequence",
    "evaluate_two_qubit_gate",
]
