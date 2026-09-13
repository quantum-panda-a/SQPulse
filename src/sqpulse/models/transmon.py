"""Transmon qubit physical model for SQPulse in SI units."""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple, Union
import json
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
        omega_d (Optional[float]): Physical drive coupling strength in Hz (defaults to 50 MHz = 50e6 Hz).
            Automatically multiplied by 2*pi internally to obtain angular frequency.
        v_phi0 (Optional[float]): Voltage required on Z line to induce one flux quantum Phi_0 in V / Phi_0 (default None).
            When None, pulse amplitudes on the Z channel are treated directly as flux in units of Phi_0.
    """

    @staticmethod
    def thermal_population_from_temperature(f_q: float, temperature: float) -> float:
        """Calculate Bose-Einstein thermal occupation n_th from qubit frequency f_q and bath temperature T.

        .. math::
            n_{\\text{th}} = \\frac{1}{\\exp\\left(\\frac{h f_q}{k_B T}\\right) - 1}

        Args:
            f_q: Qubit transition frequency in Hz.
            temperature: Bath temperature in Kelvin (K).

        Returns:
            Thermal population n_th (float >= 0.0).
        """
        if temperature is None or temperature <= 0:
            return 0.0
        import scipy.constants as const
        x = (const.h * float(f_q)) / (const.k * float(temperature))
        if x > 500.0:
            # Underflow / negligible thermal population
            return 0.0
        if x < 1e-15:
            return float(1.0 / x)
        return float(1.0 / (np.exp(x) - 1.0))

    @staticmethod
    def temperature_from_thermal_population(f_q: float, n_th: float) -> float:
        """Calculate effective bath temperature in Kelvin from qubit frequency and thermal occupation n_th.

        .. math::
            T = \\frac{h f_q}{k_B \\ln(1 + 1 / n_{\\text{th}})}

        Args:
            f_q: Qubit transition frequency in Hz.
            n_th: Thermal population n_th.

        Returns:
            Equivalent temperature in Kelvin (K).
        """
        if n_th is None or n_th <= 0:
            return 0.0
        import scipy.constants as const
        x = np.log(1.0 + 1.0 / float(n_th))
        return float((const.h * float(f_q)) / (const.k * x))

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
        thermal_population: Optional[float] = None,
        omega_d: Optional[float] = None,
        v_phi0: Optional[float] = None,
        temperature: Optional[Union[float, str]] = None,
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
        self.v_phi0 = float(v_phi0) if v_phi0 is not None else None

        # Resolve temperature and thermal_population
        from ..units import parse_quantity

        if temperature is not None:
            t_val = parse_quantity(temperature)
            if t_val is not None and t_val < 0:
                raise ValueError(f"Temperature must be non-negative, got {t_val} K")
            self._temperature = float(t_val) if t_val is not None else 0.0
            self._thermal_population = self.thermal_population_from_temperature(self.f_q, self._temperature)
        elif thermal_population is not None and float(thermal_population) > 0:
            t_pop = float(thermal_population)
            if t_pop < 0:
                raise ValueError(f"Thermal population must be non-negative, got {t_pop}")
            self._thermal_population = t_pop
            self._temperature = self.temperature_from_thermal_population(self.f_q, self._thermal_population)
        else:
            self._temperature = 0.0
            self._thermal_population = 0.0

        if omega_d is not None:
            self.omega_d = 2.0 * np.pi * float(parse_quantity(omega_d))
        else:
            # Default physical drive coupling: 50 MHz (rad/s = 2*pi * 50 MHz)
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
    def temperature(self) -> float:
        """Effective qubit thermal bath temperature in Kelvin (K)."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: Union[float, str, None]):
        from ..units import parse_quantity
        val = parse_quantity(value) if value is not None else 0.0
        if val is not None and val < 0:
            raise ValueError(f"Temperature must be non-negative, got {val} K")
        self._temperature = float(val) if val is not None else 0.0
        self._thermal_population = self.thermal_population_from_temperature(self.f_q, self._temperature)

    @property
    def thermal_population(self) -> float:
        """Excited state thermal occupation n_th."""
        return self._thermal_population

    @thermal_population.setter
    def thermal_population(self, value: Union[float, int, None]):
        val = float(value) if value is not None else 0.0
        if val < 0:
            raise ValueError(f"Thermal population must be non-negative, got {val}")
        self._thermal_population = val
        self._temperature = self.temperature_from_thermal_population(self.f_q, self._thermal_population)

    @property
    def omega_d_hz(self) -> float:
        """Physical drive coupling strength in Hz: omega_d / (2*pi)."""
        return self.omega_d / (2.0 * np.pi)

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
        temperature: Optional[Union[float, str]] = None,
        thermal_population: Optional[float] = None,
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
        omega_d_rad = omega_chip * alpha_line * float(v_max)

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
            temperature=temperature,
            thermal_population=thermal_population,
            omega_d=omega_d_rad / (2.0 * np.pi),
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
        t_str = f", T={self.temperature * 1e3:.1f}mK" if self.temperature > 0 else ""
        return (
            f"Transmon('{self.name}', f_q={self.f_q:.3e}Hz, alpha={self.alpha:.3e}Hz{d_str}{v_phi0_str}{t_str}, "
            f"levels={self.levels}, omega_d={self.omega_d_hz/1e6:.2f}MHz, T1={self.t1:.2e}s, T2={self.t2:.2e}s)"
        )

    def to_dict(self, human_readable: bool = False) -> dict:
        """Serialize Transmon configuration to a dictionary.

        Args:
            human_readable: If True, formats numerical quantities with convenient SI unit strings (e.g. '5.0 GHz').
        """
        return {
            "name": self.name,
            "f_q": f"{self.f_q / 1e9:.6g} GHz" if human_readable else self.f_q,
            "alpha": f"{self.alpha / 1e6:.6g} MHz" if human_readable else self.alpha,
            "d": self.d,
            "flux_offset": self.flux_offset,
            "levels": self.levels,
            "t1": "inf" if np.isinf(self.t1) else (f"{self.t1 / 1e-6:.6g} us" if human_readable else self.t1),
            "t2": "inf" if np.isinf(self.t2) else (f"{self.t2 / 1e-6:.6g} us" if human_readable else self.t2),
            "temperature": f"{self.temperature * 1e3:.2f} mK" if (human_readable and self.temperature > 0) else self.temperature,
            "thermal_population": self.thermal_population,
            "omega_d": f"{self.omega_d_hz / 1e6:.6g} MHz" if human_readable else self.omega_d_hz,
            "v_phi0": self.v_phi0,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Transmon:
        """Construct a Transmon instance from a dictionary.

        Values can be pure numbers (in SI units) or strings with units (e.g. '5.0 GHz', '25 us', '35 mK').
        """
        from ..units import parse_quantity

        name = data.get("name", "q0")
        f_q = parse_quantity(data.get("f_q", 5.0e9))
        alpha = parse_quantity(data.get("alpha", -250.0e6))
        d = float(data.get("d", 1.0))
        flux_offset = parse_quantity(data.get("flux_offset", 0.0), default=0.0)
        levels = int(data.get("levels", 4))
        t1 = parse_quantity(data.get("t1", float("inf")), default=float("inf"))
        t2 = parse_quantity(data.get("t2", float("inf")), default=float("inf"))
        temperature = data.get("temperature", None)
        thermal_population = data.get("thermal_population", None)
        raw_omega_d = data.get("omega_d", None)
        if raw_omega_d is not None:
            if isinstance(raw_omega_d, str) and "rad" in raw_omega_d.lower():
                omega_d = parse_quantity(raw_omega_d) / (2.0 * np.pi)
            else:
                omega_d = parse_quantity(raw_omega_d)
        else:
            omega_d = None
        v_phi0 = parse_quantity(data.get("v_phi0", None))

        if "c_d" in data or "c_g" in data:
            c_d = parse_quantity(data.get("c_d", 5.0e-17))
            c_g = parse_quantity(data.get("c_g", 70.0e-15))
            attenuation_dB = float(data.get("attenuation_dB", -60.0))
            v_max = parse_quantity(data.get("v_max", 1.0))
            m_mutual = parse_quantity(data.get("m_mutual", None))
            attenuation_z_dB = float(data.get("attenuation_z_dB", -20.0))
            z0 = parse_quantity(data.get("z0", 50.0))

            return cls.from_circuit(
                name=name,
                c_d=c_d,
                c_g=c_g,
                f_q=f_q,
                alpha=alpha,
                attenuation_dB=attenuation_dB,
                v_max=v_max,
                levels=levels,
                t1=t1,
                t2=t2,
                temperature=temperature,
                thermal_population=thermal_population,
                d=d,
                flux_offset=flux_offset,
                m_mutual=m_mutual,
                attenuation_z_dB=attenuation_z_dB,
                z0=z0,
                v_phi0=v_phi0,
            )

        return cls(
            name=name,
            f_q=f_q,
            alpha=alpha,
            d=d,
            flux_offset=flux_offset,
            levels=levels,
            t1=t1,
            t2=t2,
            temperature=temperature,
            thermal_population=thermal_population,
            omega_d=omega_d,
            v_phi0=v_phi0,
        )

    @classmethod
    def from_json(cls, source: Union[str, Path]) -> Transmon:
        """Load a Transmon model from a JSON file path or a raw JSON string.

        Args:
            source: A file path (str or Path) or a valid JSON string.
        """
        p = Path(source) if isinstance(source, (str, Path)) else None
        if p is not None and p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(str(source))

        return cls.from_dict(data)

    def to_json(
        self,
        filepath_or_buf: Optional[Union[str, Path]] = None,
        indent: int = 2,
        human_readable: bool = False,
    ) -> Optional[str]:
        """Serialize Transmon configuration to a JSON file or JSON string.

        Args:
            filepath_or_buf: Optional destination file path. If None, returns the JSON string.
            indent: Indentation level for pretty-printing JSON.
            human_readable: If True, uses unit strings (e.g. '5.0 GHz').
        """
        data = self.to_dict(human_readable=human_readable)
        if filepath_or_buf is not None:
            p = Path(filepath_or_buf)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent)
            return None
        return json.dumps(data, indent=indent)

