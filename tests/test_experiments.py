"""Tests for sqpulse.experiments in SI units."""

import numpy as np
import pytest
from sqpulse.models import Transmon
from sqpulse.pulses import GaussianPulse, SquarePulse
from sqpulse.sequence import PulseSequence
from sqpulse.simulation import Simulator
from sqpulse.experiments import RabiExperiment, T1Experiment, RamseyExperiment


def test_amplitude_rabi_calibration():
    q = Transmon("q0", f_q=5.0e9, alpha=-250e6, levels=3)
    duration = 30e-9  # 30 ns in s

    # Perform Amplitude Rabi sweep in rad/s
    amps = np.linspace(0.0, 2.5e8, 25)
    rabi_res = RabiExperiment.amplitude_rabi(
        q,
        pulse_type=GaussianPulse,
        duration=duration,
        amps=amps,
        dt=5e-10,
    )

    amp_pi = rabi_res.amp_pi
    assert 0.5e8 < amp_pi < 2.2e8

    # Verify that applying amp_pi achieves state inversion (P1 ~ 1.0)
    pi_pulse = GaussianPulse(duration=duration, amp=amp_pi)
    seq = PulseSequence().add(q.drive, pi_pulse)
    sim_res = Simulator.run(q, seq, dt=5e-10)

    assert np.isclose(sim_res.final_population(1), 1.0, atol=0.03)


def test_t1_experiment():
    target_t1 = 3.0e-6  # 3 us in s
    q = Transmon("q0", f_q=5.0e9, levels=2, t1=target_t1)

    # Use square pi pulse: duration 20ns, amp = pi / 20ns (in rad/s)
    duration = 20e-9
    pi_pulse = SquarePulse(duration=duration, amp=np.pi / duration)

    delays = np.linspace(0.0, 6.0e-6, 15)
    t1_res = T1Experiment.run(q, pi_pulse, delays=delays, dt=1e-9)

    # Fitted T1 should match target within 5%
    assert np.isclose(t1_res.t1_fit, target_t1, rtol=0.05)


def test_ramsey_experiment():
    target_t2 = 2.5e-6  # 2.5 us in s
    detuning = 2.0e6  # 2 MHz in Hz
    q = Transmon("q0", f_q=5.0e9, levels=2, t1=10.0e-6, t2=target_t2)

    # pi/2 pulse: duration 20ns, amp = (pi/2) / 20ns (in rad/s)
    duration = 20e-9
    pi_half_pulse = SquarePulse(duration=duration, amp=0.5 * np.pi / duration)

    delays = np.linspace(0.0, 2.5e-6, 35)
    ramsey_res = RamseyExperiment.run(
        q,
        pi_half_pulse=pi_half_pulse,
        detuning=detuning,
        delays=delays,
        dt=1e-9,
    )

    # Fitted detuning should match 2 MHz within 5%
    assert np.isclose(ramsey_res.fitted_detuning, detuning, rtol=0.05)
    # Fitted T2* should match target within 15%
    assert np.isclose(ramsey_res.t2_star, target_t2, rtol=0.15)
