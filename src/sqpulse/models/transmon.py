"""Transmon qubit physical model for SQPulse in SI units."""

from __future__ import annotations
from typing import List, Optional, Tuple, Union
import numpy as np
import qutip


from ..sequence.channel import Channel, normalize_channel


class Transmon:
    r"""Physical model of a multi-level Transmon qubit in SI units.

    Supports both fixed-frequency single-junction transmons and flux-tunable SQUID transmons.
    Single-junction transmons are a special case with junction asymmetry d = 1.0.

    The Hamiltonian in the rotating frame of reference (carrier frequency f_d in Hz) is:
    .. math::
        H_0 = 2\pi (f_q - f_d) a^\dagger a + \pi \alpha a^{\dagger 2} a^2 \quad (\text{rad/s})

    The time-dependent drive Hamiltonian (under RWA) is:
    .. math::
        H_d(t) = \frac{1}{2} \Omega_x(t) (a + a^\dagger) + \frac{1}{2} \Omega_y(t) i(a^\dagger - a) \quad (\text{rad/s})

    Args:
        name (str): Unique name of this transmon (e.g. 'q0').
        f_q (float): Maximum qubit 0-1 transition frequency at sweet spot in Hz (e.g. 5.0e9 for 5 GHz).
        alpha (float): Anharmonicity in Hz (typically negative, e.g. -250e6 for -250 MHz).
        d (float): SQUID junction asymmetry parameter d = |Ej1 - Ej2| / (Ej1 + Ej2) in [0.0, 1.0].
            d = 1.0 represents a single-junction transmon (frequency is fixed, flux-insensitive).
            d = 0.0 represents a fully symmetric SQUID. Default is 1.0.
        flux_offset (float): Static external magnetic flux bias in units of Phi_0 (default 0.0).
        levels (int): Number of Hilbert space levels to model (default 4: |0>, |1>, |2>, |3>).
        t1 (float): Energy relaxation time T1 in seconds (default inf).
        t2 (float): Dephasing time T2 in seconds (default inf).
        thermal_population (float): Excited state thermal occupation n_th (default 0.0).
        omega_d (Optional[float]): Physical drive coupling strength in rad/s (defaults to 2*pi*50 MHz = 3.14e8 rad/s).
        v_phi0 (Optional[float]): Voltage required on Z line to induce one flux quantum Phi_0 in V / Phi_0 (default None).
            When None, pulse amplitudes on the Z channel are treated directly as flux in units of Phi_0.
    """

    def __init__(
        self,
        name: str = "q0",
        f_q: float = 5.0e9,
        alpha: float = -250.0e6,
        d: float = 1.0,
        flux_offset: float = 0.0,
        levels: int = 4,
        t1: float = np.inf,
        t2: float = np.inf,
        thermal_population: float = 0.0,
        omega_d: Optional[float] = None,
        v_phi0: Optional[float] = None,
    ):
        if levels < 2:
            raise ValueError(f"Transmon levels must be >= 2, got {levels}")
        if not (0.0 <= d <= 1.0):
            raise ValueError(f"Junction asymmetry d must be within [0.0, 1.0], got {d}")

        self.name = name
        self.f_q = float(f_q)
        self.alpha = float(alpha)
        self.d = float(d)
        self.flux_offset = float(flux_offset)
        self.levels = int(levels)
        self.t1 = float(t1)
        self.t2 = float(t2)
        self.thermal_population = float(thermal_population)
        self.v_phi0 = float(v_phi0) if v_phi0 is not None else None

        if omega_d is not None:
            self.omega_d = float(omega_d)
        else:
            # Default physical drive coupling: 2*pi * 50 MHz (rad/s)
            self.omega_d = 2.0 * np.pi * 50.0e6

        # Physical control lines: xy, z, ro
        self.xy = Channel(f"{self.name}.xy", description=f"XY line for {self.name}")
        self.z = Channel(f"{self.name}.z", description=f"Z flux line for {self.name}")
        self.ro = Channel(f"{self.name}.ro", description=f"Readout resonator line for {self.name}")

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

    @property
    def g_flux(self) -> float:
        """Flux transfer gain in units of Phi_0 / V (equal to 1 / v_phi0 if v_phi0 is set, else 1.0)."""
        return (1.0 / self.v_phi0) if self.v_phi0 is not None else 1.0

    def voltage_to_flux(self, voltage: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert voltage signal on the Z line (in Volts) to magnetic flux in units of Phi_0.

        If v_phi0 is None, voltage is treated directly as flux in units of Phi_0.
        """
        if self.v_phi0 is None:
            return voltage
        return voltage / self.v_phi0

    def flux_to_voltage(self, flux: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert magnetic flux in units of Phi_0 to required voltage signal on the Z line (in Volts).

        If v_phi0 is None, returns flux directly.
        """
        if self.v_phi0 is None:
            return flux
        return flux * self.v_phi0


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

    def frequency_at_flux(self, flux: Optional[float] = None) -> float:
        """Calculate qubit 0-1 transition frequency at an external flux (in units of Phi_0).

        If flux is None, uses self.flux_offset.
        For single-junction transmon (d = 1.0), returns self.f_q identically.
        """
        phi = self.flux_offset if flux is None else float(flux)
        if self.d >= 1.0:
            return self.f_q

        cos_term = np.cos(np.pi * phi)
        sin_term = np.sin(np.pi * phi)
        g = cos_term**2 + (self.d**2) * (sin_term**2)
        ej_ratio = np.sqrt(max(0.0, g))

        ec = abs(self.alpha)
        f_at_phi = (self.f_q + ec) * (ej_ratio**0.5) - ec
        return float(max(0.0, f_at_phi))

    def flux_sensitivity(self, flux: Optional[float] = None) -> float:
        """Calculate first-order flux sensitivity df_q / dPhi (in Hz / Phi_0).

        Returns 0.0 for single-junction transmons (d = 1.0) and at sweet spots (Phi = 0 mod 1).
        """
        if self.d >= 1.0:
            return 0.0
        phi = self.flux_offset if flux is None else float(flux)
        cos_term = np.cos(np.pi * phi)
        sin_term = np.sin(np.pi * phi)
        g = cos_term**2 + (self.d**2) * (sin_term**2)
        if g <= 1e-14:
            return 0.0
        dg_dphi = np.pi * (self.d**2 - 1.0) * np.sin(2.0 * np.pi * phi)
        ec = abs(self.alpha)
        return float((self.f_q + ec) * 0.25 * (g**(-0.75)) * dg_dphi)

    def H0(self, f_d: Optional[float] = None, flux: Optional[float] = None) -> qutip.Qobj:
        r"""Compute static Hamiltonian in the frame rotating at frequency f_d (Hz).

        If f_d is None, defaults to resonant frame (f_d = self.frequency_at_flux(flux)).
        If flux is provided, the detuning is calculated relative to the flux-shifted qubit frequency.

        .. math::
            H_0 = 2\pi (f_q(\Phi) - f_d) a^\dagger a + \pi \alpha a^{\dagger 2} a^2 \quad (\text{rad/s})
        """
        eff_fq = self.frequency_at_flux(flux)
        if f_d is None:
            f_d = eff_fq

        detune = eff_fq - f_d
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
        levels: int = 4,
        t1: float = np.inf,
        t2: float = np.inf,
        thermal_population: float = 0.0,
        d: float = 1.0,
        flux_offset: float = 0.0,
        m_mutual: Optional[float] = None,
        attenuation_z_dB: float = -20.0,
        z0: float = 50.0,
        v_phi0: Optional[float] = None,
        **kwargs,
    ) -> Transmon:
        r"""Construct a Transmon model with drive coupling omega_d and flux period voltage v_phi0 derived from circuit parameters.

        According to circuit QED capacitive drive theory (Krantz et al., 2019):
        .. math::
            C_\Sigma = C_g + C_d
            \omega_q = 2\pi f_q
            Q_{\text{zpf}} = \sqrt{\frac{\hbar \omega_q C_\Sigma}{2}}
            \Omega_{\text{chip}} = \frac{C_d}{C_\Sigma} \frac{Q_{\text{zpf}}}{\hbar} \quad [\text{rad}/(\text{s}\cdot\text{V})]
            \alpha_{\text{line}} = 10^{\text{attenuation\_dB} / 20}
            \Omega_d = \Omega_{\text{chip}} \cdot \alpha_{\text{line}} \cdot V_{\text{max}} \quad [\text{rad/s}]

        For Z flux line mutual coupling to the SQUID loop:
        .. math::
            \alpha_{\text{line}, z} = 10^{\text{attenuation\_z\_dB} / 20}
            I_{\text{chip}} = \frac{V_{\text{AWG}} \cdot \alpha_{\text{line}, z}}{Z_0}
            \Phi = M \cdot I_{\text{chip}}
            V_{\Phi_0} = \frac{\Phi_0 Z_0}{M \cdot \alpha_{\text{line}, z}} \quad [\text{V}/\Phi_0]

        Args:
            name: Transmon qubit name.
            c_d: Drive line coupling capacitance in Farads (default 5e-17 F = 0.05 fF).
            c_g: Shunt capacitance to ground in Farads (default 70e-15 F = 70 fF).
            f_q: Qubit 0-1 transition frequency in Hz (e.g. 5.0e9 Hz).
            alpha: Anharmonicity in Hz (e.g. -250e6 Hz).
            attenuation_dB: Total microwave line attenuation from AWG to chip in dB (default: -60.0 dB).
            v_max: Maximum output voltage of the AWG in Volts (default: 1.0 V).
            levels: Number of Hilbert space levels (default: 4).
            t1: T1 relaxation time in seconds.
            t2: T2 dephasing time in seconds.
            thermal_population: Thermal population.
            d: SQUID junction asymmetry parameter in [0.0, 1.0] (default 1.0 for single junction).
            flux_offset: Static flux bias in units of Phi_0 (default 0.0).
            m_mutual: Mutual inductance between Z flux line and SQUID loop in Henry (e.g. 2.5e-12 H = 2.5 pH).
            attenuation_z_dB: Total attenuation on Z flux line from AWG/DAC to chip in dB (default: -20.0 dB).
            z0: Characteristic transmission line impedance in Ohms (default: 50.0 Ohm).
            v_phi0: Explicit flux period voltage in V/Phi_0. Overrides m_mutual if provided.

        Returns:
            Transmon instance with physically calculated omega_d and v_phi0.
        """
        import scipy.constants as const
        hbar = const.hbar
        c_sigma = float(c_g) + float(c_d)
        omega_q = 2.0 * np.pi * float(f_q)
        q_zpf = np.sqrt(0.5 * hbar * omega_q * c_sigma)
        omega_chip = (float(c_d) / c_sigma) * (q_zpf / hbar)
        alpha_line = 10.0 ** (float(attenuation_dB) / 20.0)
        omega_d = omega_chip * alpha_line * float(v_max)

        resolved_v_phi0 = v_phi0
        if resolved_v_phi0 is None and m_mutual is not None:
            phi_0 = const.h / (2.0 * const.e)
            alpha_line_z = 10.0 ** (float(attenuation_z_dB) / 20.0)
            resolved_v_phi0 = (phi_0 * float(z0)) / (float(m_mutual) * alpha_line_z)

        instance = cls(
            name=name,
            f_q=f_q,
            alpha=alpha,
            d=d,
            flux_offset=flux_offset,
            levels=levels,
            t1=t1,
            t2=t2,
            thermal_population=thermal_population,
            omega_d=omega_d,
            v_phi0=resolved_v_phi0,
            **kwargs,
        )
        instance.c_d = float(c_d)
        instance.c_g = float(c_g)
        instance.c_sigma = c_sigma
        instance.q_zpf = q_zpf
        instance.omega_chip = omega_chip
        instance.attenuation_dB = float(attenuation_dB)
        instance.v_max = float(v_max)
        instance.m_mutual = float(m_mutual) if m_mutual is not None else None
        instance.attenuation_z_dB = float(attenuation_z_dB)
        instance.z0 = float(z0)
        return instance

    def __repr__(self) -> str:
        d_str = f", d={self.d:.2f}" if self.d < 1.0 else ""
        v_phi0_str = f", v_phi0={self.v_phi0:.3f}V/Phi_0" if self.v_phi0 is not None else ""
        return (
            f"Transmon('{self.name}', f_q={self.f_q:.3e}Hz, alpha={self.alpha:.3e}Hz{d_str}{v_phi0_str}, "
            f"levels={self.levels}, omega_d={self.omega_d:.3e}rad/s, T1={self.t1:.2e}s, T2={self.t2:.2e}s)"
        )
