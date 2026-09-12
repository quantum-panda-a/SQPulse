"""Cavity photon dynamics simulation for circuit QED dispersive readout in SI units."""

from __future__ import annotations
from typing import Tuple, Optional, Dict
import numpy as np

from ...models.resonator import ReadoutResonator
from ...models.transmon import Transmon
from ...pulses.base import Pulse
from ...pulses.shapes import FlatTopPulse


def simulate_cavity_dynamics(
    resonator: ReadoutResonator,
    qubit_state: int,
    readout_pulse: Optional[Pulse] = None,
    f_ro: Optional[float] = None,
    times: Optional[np.ndarray] = None,
    dt: float = 1e-9,
    drive_scale: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Simulate the time evolution of cavity field amplitude alpha(t) for a given qubit Fock state.

    The cavity field amplitude satisfies the classical Langevin equation:
    .. math::
        \frac{d\alpha_n(t)}{dt} = - \left[ i 2\pi (f_{\text{ro}} - f_r^{(n)}) + \frac{\kappa}{2} \right] \alpha_n(t)
                                  - i \epsilon_{\text{scale}} \epsilon(t)

    Args:
        resonator: Physical ReadoutResonator model.
        qubit_state: Transmon level (0, 1, etc.).
        readout_pulse: Optional Pulse object for readout drive. If None, defaults to a 1 us FlatTopPulse.
        f_ro: Readout drive frequency in Hz. Defaults to resonator bare frequency f_r.
        times: Optional custom time array.
        dt: Time step in seconds (default 1e-9 s = 1 ns).
        drive_scale: Drive coupling scale factor. If None, defaults to resonator.kappa,
            yielding a realistic steady-state intra-cavity photon number n_cav ~ 1-5 photons for amp=1.0.

    Returns:
        times: 1D array of timestamps in seconds.
        alpha: 1D complex array of cavity field amplitude alpha(t) over time.
    """
    if readout_pulse is None:
        # Default 1 us readout pulse with 20 ns ramp
        readout_pulse = FlatTopPulse(duration=1000e-9, ramp_time=20e-9, amp=1.0)

    if times is None:
        # Extend simulation slightly past pulse duration to observe ring-down
        t_end = readout_pulse.duration + 5.0 / (resonator.kappa_hz * 2.0 * np.pi)
        times = np.arange(0.0, max(readout_pulse.duration * 1.2, t_end), dt)

    if f_ro is None:
        f_ro = resonator.f_r

    f_eff = resonator.effective_frequency(qubit_state)
    detuning_rad = 2.0 * np.pi * (f_ro - f_eff)
    lambda_k = 1j * detuning_rad + 0.5 * resonator.kappa

    # Envelope drive field: scaled by drive coupling rate
    drive_env = np.array([readout_pulse(t) for t in times], dtype=complex)
    eff_scale = drive_scale if drive_scale is not None else resonator.kappa
    drive_field = -1j * eff_scale * drive_env

    alpha = np.zeros(len(times), dtype=complex)
    for i in range(len(times) - 1):
        dt_i = times[i + 1] - times[i]
        d_val = drive_field[i]
        decay = np.exp(-lambda_k * dt_i)
        if abs(lambda_k) > 1e-15:
            step_int = (1.0 - decay) / lambda_k
        else:
            step_int = dt_i
        alpha[i + 1] = alpha[i] * decay + d_val * step_int

    return times, alpha
