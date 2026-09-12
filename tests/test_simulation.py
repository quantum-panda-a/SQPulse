"""Tests for projective measurement simulation in SI units."""

import numpy as np
import pytest
from sqpulse.models import Transmon
from sqpulse.pulses import SquarePulse
from sqpulse.sequence import PulseSequence
from sqpulse import Simulator


def test_resonant_rabi_flip():
    """A square pi pulse with AWG amplitude V_0 = pi / (omega_d * duration) should flip |0> to |1>."""
    # Transmon with default omega_d = 2*pi * 50 MHz
    q = Transmon("q0", f_q=5.0e9, alpha=-300e6, levels=2)

    duration = 20e-9  # 20 ns
    # For pi rotation: omega_d * V_0 * duration = pi -> V_0 = pi / (omega_d * duration) = 0.5
    v0_pi = np.pi / (q.omega_d * duration)
    assert np.isclose(v0_pi, 0.5)

    p_pi = SquarePulse(duration=duration, amp=v0_pi)
    seq = PulseSequence().add(q.xy, p_pi)
    res = Simulator.run(q, seq, dt=2e-10)

    # Final population of |1> should be ~1.0
    p1 = res.final_population(1)
    assert np.isclose(p1, 1.0, atol=1e-3)

    # Check Bloch vector z coordinate goes from +1 to -1
    x, y, z = res.bloch_vector()
    assert np.isclose(z[0], 1.0, atol=1e-3)
    assert np.isclose(z[-1], -1.0, atol=1e-3)


def test_raw_omega_backward_compatibility():
    """Setting omega_d=1.0 allows using raw angular frequency in pulse amp."""
    q_raw = Transmon("q0", f_q=5.0e9, levels=2, omega_d=1.0)
    duration = 20e-9
    omega_raw = np.pi / duration  # raw rad/s

    with pytest.warns(UserWarning, match=r"exceeds the normalized range \[-1, 1\]"):
        p_pi = SquarePulse(duration=duration, amp=omega_raw)
    seq = PulseSequence().add(q_raw.xy, p_pi)
    res = Simulator.run(q_raw, seq, dt=2e-10)
    assert np.isclose(res.final_population(1), 1.0, atol=1e-3)


def test_drag_leakage_suppression():
    """Verify that a dimensionless DRAG pulse (drag=1.0) suppresses leakage to |2> by orders of magnitude."""
    from sqpulse.pulses import GaussianPulse, DRAGPulse

    q = Transmon("q0", f_q=5.0e9, alpha=-250.0e6, levels=3, omega_d=2.0 * np.pi * 100.0e6)
    duration = 10e-9
    amp_pi = 0.966  # Calibrated AWG amplitude for 10ns pi pulse

    # Pulse without DRAG
    p_nodrag = GaussianPulse(duration=duration, amp=amp_pi, drag=0.0)
    seq_nodrag = PulseSequence().add(q.xy, p_nodrag)
    res_nodrag = Simulator.run(q, seq_nodrag, dt=5e-11)
    leakage_nodrag = res_nodrag.final_population(2)

    # Pulse with dimensionless DRAG beta=1.0
    p_drag = DRAGPulse(duration=duration, amp=amp_pi, drag=1.0)
    seq_drag = PulseSequence().add(q.xy, p_drag)
    res_drag = Simulator.run(q, seq_drag, dt=5e-11)
    leakage_drag = res_drag.final_population(2)

    # DRAG should suppress leakage to |2> by at least a factor of 50
    assert leakage_nodrag > 1e-4
    assert leakage_drag < 1e-5
    assert (leakage_nodrag / leakage_drag) > 50.0


def test_dynamic_flux_pulse_phase_accumulation():
    """A flux pulse on q.z dynamically shifts the qubit frequency and accumulates a Z phase."""
    from sqpulse.pulses import SquarePulse
    from sqpulse import Simulator

    # Tunable qubit with asymmetry d=0.2, sweet spot at 5 GHz
    q = Transmon("q_flux", f_q=5.0e9, alpha=-250e6, d=0.2, levels=2, omega_d=2.0 * np.pi * 50e6)
    v0_pi = np.pi / (q.omega_d * 20e-9)
    p_pi2 = SquarePulse(duration=20e-9, amp=0.5 * v0_pi)

    # Flux pulse shifting flux to Phi = 0.1
    f_shift = q.frequency_at_flux(0.1)
    detuning = f_shift - 5.0e9  # Negative shift
    # Choose duration so that accumulated phase is pi: 2*pi * |detuning| * tau = pi -> tau = 1 / (2 * |detuning|)
    tau = 1.0 / (2.0 * abs(detuning))

    p_flux = SquarePulse(duration=tau, amp=0.1)

    # Sequence 1: pi/2 - delay(tau) without flux pulse - pi/2 (resonant, so accumulates 0 phase -> state flips to |1>)
    seq_resonant = PulseSequence()
    seq_resonant.add(q.xy, p_pi2)
    seq_resonant.delay(q.xy, tau)
    seq_resonant.add(q.xy, p_pi2)
    res_res = Simulator.run(q, seq_resonant, dt=5e-11)
    assert np.isclose(res_res.final_population(1), 1.0, atol=1e-2)

    # Sequence 2: pi/2 - flux pulse(tau, amp=0.1) - pi/2 (accumulates pi phase -> rotates back to |0>)
    seq_flux = PulseSequence()
    seq_flux.add(q.xy, p_pi2)
    seq_flux.sync()  # Advance z clock to end of first pulse
    seq_flux.add(q.z, p_flux)
    seq_flux.sync()  # Advance xy clock to end of flux pulse
    seq_flux.add(q.xy, p_pi2)
    res_flux = Simulator.run(q, seq_flux, dt=5e-11)
    # Because of the pi phase shift, the second pi/2 rotation returns state to |0>
    assert np.isclose(res_flux.final_population(0), 1.0, atol=1e-2)


def test_flux_pulsed_spectroscopy():
    """Verify that a Z flux pulse detunes a tunable Transmon and XY microwave probe detects resonance at shifted frequency."""
    from sqpulse.pulses import FlatTopPulse
    from sqpulse import Simulator

    # Tunable transmon (d=0.25)
    q = Transmon("q_spec", f_q=5.0e9, alpha=-250e6, d=0.25, levels=3)
    phi_bias = 0.15
    f_expected = q.frequency_at_flux(phi_bias)

    # Multi-channel pulse sequence: Z pulse + concurrent XY probe
    seq = PulseSequence(name="flux_qubit_spec")
    seq.add(q.z, FlatTopPulse(duration=120e-9, amp=phi_bias, ramp_time=10e-9))
    seq.delay(q.xy, 10e-9)
    seq.add(q.xy, FlatTopPulse(duration=100e-9, amp=0.04, ramp_time=5e-9))

    # Sweep XY carrier frequencies around shifted frequency
    freqs = np.linspace(f_expected - 40e6, f_expected + 40e6, 21)
    p1_vals = [Simulator.run(q, seq, dt=1e-9, f_d=float(fd)).final_population(1) for fd in freqs]

    f_peak = freqs[np.argmax(p1_vals)]
    assert np.isclose(f_peak, f_expected, atol=5e6)
    assert np.max(p1_vals) > 0.25


def test_v_phi0_voltage_pulse_simulation():
    """Verify that a Z voltage pulse scaled by v_phi0 yields identical dynamics to a direct flux pulse."""
    from sqpulse.pulses import FlatTopPulse
    from sqpulse import Simulator

    v_phi0 = 0.8  # 0.8 V per Phi_0
    v_target = 0.12  # 0.12 V -> 0.15 Phi_0
    phi_target = 0.15

    q_volt = Transmon("q_volt", f_q=5.0e9, alpha=-250e6, d=0.25, levels=3, v_phi0=v_phi0)
    q_flux = Transmon("q_flux", f_q=5.0e9, alpha=-250e6, d=0.25, levels=3, v_phi0=None)

    seq_volt = PulseSequence(name="seq_volt")
    seq_volt.add(q_volt.z, FlatTopPulse(duration=120e-9, amp=v_target, ramp_time=10e-9))
    seq_volt.delay(q_volt.xy, 10e-9)
    seq_volt.add(q_volt.xy, FlatTopPulse(duration=100e-9, amp=0.04, ramp_time=5e-9))

    seq_flux = PulseSequence(name="seq_flux")
    seq_flux.add(q_flux.z, FlatTopPulse(duration=120e-9, amp=phi_target, ramp_time=10e-9))
    seq_flux.delay(q_flux.xy, 10e-9)
    seq_flux.add(q_flux.xy, FlatTopPulse(duration=100e-9, amp=0.04, ramp_time=5e-9))

    f_target = q_volt.frequency_at_flux(phi_target)
    res_volt = Simulator.run(q_volt, seq_volt, dt=1e-9, f_d=float(f_target))
    res_flux = Simulator.run(q_flux, seq_flux, dt=1e-9, f_d=float(f_target))

    assert np.isclose(res_volt.final_population(0), res_flux.final_population(0), atol=1e-6)
    assert np.isclose(res_volt.final_population(1), res_flux.final_population(1), atol=1e-6)
    assert np.isclose(res_volt.final_population(2), res_flux.final_population(2), atol=1e-6)



