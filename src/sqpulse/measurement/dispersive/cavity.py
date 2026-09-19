"""Cavity photon dynamics simulation for circuit QED dispersive readout in SI units."""

from __future__ import annotations
from typing import Tuple, Optional, Dict, Union, Any
import numpy as np

from ...models.resonator import ReadoutResonator
from ...pulses.base import Pulse
from ...pulses.shapes import FlatTopPulse
from ...sequence.channel import normalize_channel


def simulate_cavity_dynamics(
    resonator: ReadoutResonator,
    qubit_state: int = 0,
    readout_pulse: Optional[Pulse] = None,
    f_ro: Optional[float] = None,
    times: Optional[np.ndarray] = None,
    dt: float = 1e-9,
    drive_scale: Optional[float] = None,
    sequence: Optional[Any] = None,
    transmon: Optional[Any] = None,
    g: Optional[float] = None,
    return_fields: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    r"""Simulate the time evolution of cavity field amplitude alpha(t) for a given qubit Fock state.

    Supports both static standalone readout pulses and full multi-channel PulseSequence
    with concurrent Z-flux pulses driving time-dependent cavity dressed frequency shifts.

    The cavity field amplitude satisfies the classical/semi-classical Langevin equation:
    .. math::
        \frac{d\alpha_n(t)}{dt} = - \left[ i 2\pi (f_{\text{ro}} - \tilde{f}_r(t)) + \frac{\kappa}{2} \right] \alpha_n(t)
                                  - i \epsilon_{\text{scale}} \epsilon(t)

    Args:
        resonator: Physical ReadoutResonator model.
        qubit_state: Transmon level (0, 1, etc.).
        readout_pulse: Optional Pulse object for readout drive. If None and sequence is None, defaults to a 1 us FlatTopPulse.
        f_ro: Readout drive frequency in Hz. Defaults to resonator bare frequency f_r.
        times: Optional custom time array.
        dt: Time step in seconds (default 1e-9 s = 1 ns).
        drive_scale: Drive coupling scale factor. If None, defaults to resonator.kappa.
        sequence: Optional PulseSequence containing concurrent readout and Z-flux pulses.
        transmon: Optional Transmon model associated with the resonator.
        g: Optional transmon-cavity coupling strength in Hz. Defaults to resonator.g_hz or 50 MHz.
        return_fields: If True, returns (times, alpha, v_in, v_out). If False, returns (times, alpha).

    Returns:
        If return_fields is False:
            times: 1D array of timestamps in seconds.
            alpha: 1D complex array of cavity field amplitude alpha(t).
        If return_fields is True:
            times, alpha, v_in, v_out
    """
    if f_ro is None:
        f_ro = resonator.f_r

    # 1. Determine time axis and channel waveforms
    if sequence is not None and getattr(sequence, "_channels", None):
        # Extend simulation slightly past sequence duration to capture ringdown
        t_ringdown = 5.0 / (resonator.kappa_hz * 2.0 * np.pi)
        t_end = sequence.duration + t_ringdown
        if times is None:
            n_pts = max(2, int(np.round(max(dt, t_end) / dt)) + 1)
            times = np.linspace(0.0, max(dt, t_end), n_pts, endpoint=True)
            times, waveforms = sequence.sample(dt=dt, total_time=t_end)
        else:
            _, waveforms = sequence.sample(dt=dt, total_time=times[-1])

        # Extract readout drive envelope
        drive_env = None
        if transmon is not None:
            ro_key = normalize_channel(transmon.ro)
            if ro_key in waveforms and np.any(np.abs(waveforms[ro_key]) > 1e-15):
                drive_env = waveforms[ro_key]

        if drive_env is None:
            # Search for any RO channel
            for ch_name, wave in waveforms.items():
                if "ro" in ch_name.lower() and np.any(np.abs(wave) > 1e-15):
                    drive_env = wave
                    break

        if drive_env is None:
            if readout_pulse is not None:
                drive_env = np.array([readout_pulse(t) for t in times], dtype=complex)
            else:
                default_p = FlatTopPulse(duration=min(1000e-9, max(100e-9, sequence.duration)), ramp_time=20e-9, amp=1.0)
                drive_env = np.array([default_p(t) for t in times], dtype=complex)

        # Extract flux waveform and compute time-dependent dressed frequency
        if transmon is not None:
            z_key = normalize_channel(transmon.z)
            if z_key in waveforms and np.any(np.abs(waveforms[z_key]) > 1e-15):
                raw_vz = waveforms[z_key].real
                phi_t = transmon.flux_offset + transmon.voltage_to_flux(raw_vz)
                f_eff_arr = np.array(
                    [
                        resonator.effective_frequency_at_flux(transmon, flux=phi, g=g, qubit_state=qubit_state)
                        for phi in phi_t
                    ],
                    dtype=float,
                )
            else:
                f_eff_static = resonator.effective_frequency_at_flux(
                    transmon, flux=transmon.flux_offset, g=g, qubit_state=qubit_state
                )
                f_eff_arr = np.full(len(times), f_eff_static, dtype=float)
        else:
            f_eff_arr = np.full(len(times), resonator.effective_frequency(qubit_state), dtype=float)

    else:
        # Standalone pulse mode
        if readout_pulse is None:
            readout_pulse = FlatTopPulse(duration=1000e-9, ramp_time=20e-9, amp=1.0)

        if times is None:
            t_end = readout_pulse.duration + 5.0 / (resonator.kappa_hz * 2.0 * np.pi)
            times = np.arange(0.0, max(readout_pulse.duration * 1.2, t_end), dt)

        drive_env = np.array([readout_pulse(t) for t in times], dtype=complex)

        if transmon is not None:
            f_eff_static = resonator.effective_frequency_at_flux(
                transmon, flux=transmon.flux_offset, g=g, qubit_state=qubit_state
            )
            f_eff_arr = np.full(len(times), f_eff_static, dtype=float)
        else:
            f_eff_arr = np.full(len(times), resonator.effective_frequency(qubit_state), dtype=float)

    # 2. Integrate Langevin equation
    detuning_rad = 2.0 * np.pi * (f_ro - f_eff_arr)
    lambda_k_arr = 1j * detuning_rad + 0.5 * resonator.kappa

    eff_scale = drive_scale if drive_scale is not None else resonator.kappa
    drive_field = -1j * eff_scale * drive_env

    alpha = np.zeros(len(times), dtype=complex)
    for i in range(len(times) - 1):
        dt_i = times[i + 1] - times[i]
        lam = lambda_k_arr[i]
        d_val = drive_field[i]
        decay = np.exp(-lam * dt_i)
        if abs(lam) > 1e-15:
            step_int = (1.0 - decay) / lam
        else:
            step_int = dt_i
        alpha[i + 1] = alpha[i] * decay + d_val * step_int

    # 3. Calculate transmitted output field V_out(t)
    if resonator.is_dip:
        # Hanger (Notch) mode: V_out = V_in - (kappa_ext / 2) * cavity_field
        v_out = drive_env - 1j * (0.5 * resonator.kappa_ext / eff_scale) * alpha
    else:
        # Transmission mode: V_out = kappa_ext * cavity_field
        v_out = 1j * (resonator.kappa_ext / eff_scale) * alpha

    if return_fields:
        return times, alpha, drive_env, v_out
    return times, alpha
