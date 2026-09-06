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
        omega_d (Optional[float]): Physical drive coupling strength in rad/s (defaults to 2*pi*50 MHz = 3.14e8 rad/s).
        drive_coupling (Optional[float]): Alias / backward-compatible argument for omega_d.
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
        omega_d: Optional[float] = None,
        drive_coupling: Optional[float] = None,
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

        if omega_d is not None:
            self.omega_d = float(omega_d)
        elif drive_coupling is not None:
            self.omega_d = float(drive_coupling)
        else:
            # Default physical drive coupling: 2*pi * 50 MHz (rad/s)
            self.omega_d = 2.0 * np.pi * 50.0e6

        self.drive_coupling = self.omega_d

        # Drive channel name
        self.drive = f"{self.name}.drive"

        # Precompute standard operators in this mode's Hilbert space
        self._a = qutip.destroy(self.levels)
        self._ad = self._a.dag()
        self._n = self._ad * self._a
        self._I = qutip.qeye(self.levels)

        # Drive quadrature operators: 0.5 * omega_d * (a + ad)
        self._H_drive_x = 0.5 * self.omega_d * (self._a + self._ad)
        self._H_drive_y = 0.5 * self.omega_d * 1j * (self._ad - self._a)

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

    @classmethod
    def from_circuit(
        cls,
        name: str = "q0",
        c_d: float = 5.0e-17,
        c_g: float = 70.0e-15,
        f_q: float = 5.0e9,
        alpha: float = -250.0e6,
        attenuation_dB: float = -60.0,
        v_max: float = 1.0,
        levels: int = 3,
        t1: float = np.inf,
        t2: float = np.inf,
        thermal_population: float = 0.0,
        **kwargs,
    ) -> Transmon:
        r"""Construct a Transmon model with drive coupling omega_d derived from circuit parameters.

        According to circuit QED capacitive drive theory (Krantz et al., 2019):
        .. math::
            C_\Sigma = C_g + C_d
            \omega_q = 2\pi f_q
            Q_{\text{zpf}} = \sqrt{\frac{\hbar \omega_q C_\Sigma}{2}}
            \Omega_{\text{chip}} = \frac{C_d}{C_\Sigma} \frac{Q_{\text{zpf}}}{\hbar} \quad [\text{rad}/(\text{s}\cdot\text{V})]
            \alpha_{\text{line}} = 10^{\text{attenuation\_dB} / 20}
            \Omega_d = \Omega_{\text{chip}} \cdot \alpha_{\text{line}} \cdot V_{\text{max}} \quad [\text{rad/s}]

        Args:
            name: Transmon qubit name.
            c_d: Drive line coupling capacitance in Farads (default 5e-17 F = 0.05 fF).
            c_g: Shunt capacitance to ground in Farads (default 70e-15 F = 70 fF).
            f_q: Qubit 0-1 transition frequency in Hz (e.g. 5.0e9 Hz).
            alpha: Anharmonicity in Hz (e.g. -250e6 Hz).
            attenuation_dB: Total microwave line attenuation from AWG to chip in dB (default: -60.0 dB).
            v_max: Maximum output voltage of the AWG in Volts (default: 1.0 V).
            levels: Number of Hilbert space levels (default: 3).
            t1: T1 relaxation time in seconds.
            t2: T2 dephasing time in seconds.
            thermal_population: Thermal population.

        Returns:
            Transmon instance with physically calculated omega_d.
        """
        import scipy.constants as const
        hbar = const.hbar
        c_sigma = float(c_g) + float(c_d)
        omega_q = 2.0 * np.pi * float(f_q)
        q_zpf = np.sqrt(0.5 * hbar * omega_q * c_sigma)
        omega_chip = (float(c_d) / c_sigma) * (q_zpf / hbar)
        alpha_line = 10.0 ** (float(attenuation_dB) / 20.0)
        omega_d = omega_chip * alpha_line * float(v_max)

        instance = cls(
            name=name,
            f_q=f_q,
            alpha=alpha,
            levels=levels,
            t1=t1,
            t2=t2,
            thermal_population=thermal_population,
            omega_d=omega_d,
            **kwargs,
        )
        instance.c_d = float(c_d)
        instance.c_g = float(c_g)
        instance.c_sigma = c_sigma
        instance.q_zpf = q_zpf
        instance.omega_chip = omega_chip
        instance.attenuation_dB = float(attenuation_dB)
        instance.v_max = float(v_max)
        return instance

    def __repr__(self) -> str:
        return (
            f"Transmon('{self.name}', f_q={self.f_q:.3e}Hz, alpha={self.alpha:.3e}Hz, "
            f"levels={self.levels}, omega_d={self.omega_d:.3e}rad/s, T1={self.t1:.2e}s, T2={self.t2:.2e}s)"
        )
