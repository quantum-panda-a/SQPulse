"""Transmon qubit physical model for SQPulse in SI units."""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import qutip


class Transmon:
    r"""Physical model of a multi-level Transmon qubit in SI units.

    The Hamiltonian in the rotating frame of reference (carrier frequency f_d in Hz) is:
    .. math::
        H_0 = 2\pi (f_q - f_d) a^\dagger a + \pi \alpha a^{\dagger 2} a^2 \quad (\text{rad/s})

    The time-dependent drive Hamiltonian (under RWA) is:
    .. math::
        H_d(t) = \frac{1}{2} \Omega_x(t) (a + a^\dagger) + \frac{1}{2} \Omega_y(t) i(a^\dagger - a) \quad (\text{rad/s})

    Args:
        name (str): Unique name of this transmon (e.g. 'q0').
        f_q (float): Qubit 0-1 transition frequency in Hz (e.g. 5.0e9 for 5 GHz).
        alpha (float): Anharmonicity in Hz (typically negative, e.g. -250e6 for -250 MHz).
        levels (int): Number of Hilbert space levels to model (default 3: |0>, |1>, |2>).
        t1 (float): Energy relaxation time T1 in seconds (default inf).
        t2 (float): Dephasing time T2 in seconds (default inf).
        thermal_population (float): Excited state thermal occupation n_th (default 0.0).
        drive_coupling (float): Relative drive coupling efficiency (default 1.0).
    """

    def __init__(
        self,
        name: str = "q0",
        f_q: float = 5.0e9,
        alpha: float = -250.0e6,
        levels: int = 3,
        t1: float = np.inf,
        t2: float = np.inf,
        thermal_population: float = 0.0,
        drive_coupling: float = 1.0,
    ):
        if levels < 2:
            raise ValueError(f"Transmon levels must be >= 2, got {levels}")
        self.name = name
        self.f_q = float(f_q)
        self.alpha = float(alpha)
        self.levels = int(levels)
        self.t1 = float(t1)
        self.t2 = float(t2)
        self.thermal_population = float(thermal_population)
        self.drive_coupling = float(drive_coupling)

        # Drive channel name
        self.drive = f"{self.name}.drive"

        # Precompute standard operators in this mode's Hilbert space
        self._a = qutip.destroy(self.levels)
        self._ad = self._a.dag()
        self._n = self._ad * self._a
        self._I = qutip.qeye(self.levels)

        # Drive quadrature operators
        self._H_drive_x = 0.5 * (self._a + self._ad) * self.drive_coupling
        self._H_drive_y = 0.5 * 1j * (self._ad - self._a) * self.drive_coupling

        # Pauli projections in {|0>, |1>} subspace
        zero = qutip.basis(self.levels, 0)
        one = qutip.basis(self.levels, 1)
        self._sx = zero * one.dag() + one * zero.dag()
        self._sy = -1j * (zero * one.dag() - one * zero.dag())
        self._sz = zero * zero.dag() - one * one.dag()

    @property
    def a(self) -> qutip.Qobj:
        """Annihilation operator a."""
        return self._a

    @property
    def ad(self) -> qutip.Qobj:
        """Creation operator a^dagger."""
        return self._ad

    @property
    def n(self) -> qutip.Qobj:
        """Number operator a^dagger a."""
        return self._n

    @property
    def I(self) -> qutip.Qobj:
        """Identity operator."""
        return self._I

    @property
    def sx(self) -> qutip.Qobj:
        """Pauli X operator in {|0>, |1>} subspace."""
        return self._sx

    @property
    def sy(self) -> qutip.Qobj:
        """Pauli Y operator in {|0>, |1>} subspace."""
        return self._sy

    @property
    def sz(self) -> qutip.Qobj:
        """Pauli Z operator in {|0>, |1>} subspace."""
        return self._sz

    @property
    def H_drive_x(self) -> qutip.Qobj:
        """Drive operator for In-phase (I) quadrature: 0.5 * (a + a^dagger)."""
        return self._H_drive_x

    @property
    def H_drive_y(self) -> qutip.Qobj:
        """Drive operator for Quadrature (Q) quadrature: 0.5 * i(a^dagger - a)."""
        return self._H_drive_y

    def fock(self, n: int = 0) -> qutip.Qobj:
        """Return the Fock state |n> as a Ket."""
        if n >= self.levels:
            raise ValueError(f"State index {n} exceeds truncated Hilbert dimension {self.levels}")
        return qutip.basis(self.levels, n)

    def ground_state(self) -> qutip.Qobj:
        """Return the ground state |0>."""
        return self.fock(0)

    def proj(self, n: int) -> qutip.Qobj:
        """Return projector |n><n|."""
        ket = self.fock(n)
        return ket * ket.dag()

    def H0(self, f_d: Optional[float] = None) -> qutip.Qobj:
        r"""Compute static Hamiltonian in the frame rotating at frequency f_d (Hz).

        If f_d is None, defaults to resonant frame (f_d = f_q), so detuning is 0.

        .. math::
            H_0 = 2\pi (f_q - f_d) a^\dagger a + \pi \alpha a^{\dagger 2} a^2 \quad (\text{rad/s})
        """
        if f_d is None:
            f_d = self.f_q

        detune = self.f_q - f_d
        H_detune = 2.0 * np.pi * detune * self.n

        # Non-linear Kerr term: pi * alpha * a^dag * a^dag * a * a
        # (in rad/s, corresponding to 2pi * (alpha/2))
        H_kerr = np.pi * self.alpha * (self.ad * self.ad * self.a * self.a)

        return H_detune + H_kerr

    def c_ops(self) -> List[qutip.Qobj]:
        """Compute Lindblad collapse operators corresponding to T1, T2, and thermal excitation in SI units.

        Returns:
            List of collapse operators (in s^-1/2) for qutip.mesolve.
        """
        ops = []

        # T1 Relaxation and thermal excitation
        if not np.isinf(self.t1) and self.t1 > 0:
            gamma_down = (1.0 + self.thermal_population) / self.t1
            if gamma_down > 0:
                ops.append(np.sqrt(gamma_down) * self.a)

            if self.thermal_population > 0:
                gamma_up = self.thermal_population / self.t1
                ops.append(np.sqrt(gamma_up) * self.ad)

        # Pure dephasing T_phi
        if not np.isinf(self.t2) and self.t2 > 0:
            rate_t2 = 1.0 / self.t2
            rate_t1_half = 1.0 / (2.0 * self.t1) if not np.isinf(self.t1) else 0.0
            rate_phi = rate_t2 - rate_t1_half

            if rate_phi < -1e-15:
                raise ValueError(
                    f"Invalid coherence parameters: T2 ({self.t2} s) cannot exceed 2 * T1 ({2 * self.t1} s)"
                )
            elif rate_phi > 1e-18:
                # Collapse operator sqrt(2 * Gamma_phi) * n
                ops.append(np.sqrt(2.0 * rate_phi) * self.n)

        return ops

    def __repr__(self) -> str:
        return (
            f"Transmon('{self.name}', f_q={self.f_q:.3e}Hz, alpha={self.alpha:.3e}Hz, "
            f"levels={self.levels}, T1={self.t1:.2e}s, T2={self.t2:.2e}s)"
        )
