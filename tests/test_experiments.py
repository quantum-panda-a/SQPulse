"""Tests for sqpulse.experiments."""

import numpy as np
import pytest
from sqpulse.models import Transmon
from sqpulse.pulses import GaussianPulse, SquarePulse
from sqpulse.sequence import PulseSequence
from sqpulse.simulation import Simulator
from sqpulse.experiments import RabiExperiment, T1Experiment, RamseyExperiment


def test_amplitude_rabi_calibration():
    q = Transmon("q0", f_q=5.0, alpha=-0.25, levels=3)
    duration = 30.0

    # Perform Amplitude Rabi sweep
    amps = np.linspace(0.0, 0.20, 25)
    rabi_res = RabiExperiment.amplitude_rabi(
        q,
        pulse_type=GaussianPulse,
        duration=duration,
        amps=amps,
        dt=0.5,
    )

    amp_pi = rabi_res.amp_pi
    assert 0.05 < amp_pi < 0.20

    # Verify that applying amp_pi achieves state inversion (P1 ~ 1.0)
    pi_pulse = GaussianPulse(duration=duration, amp=amp_pi)
    seq = PulseSequence().add(q.drive, pi_pulse)
    sim_res = Simulator.run(q, seq, dt=0.5)

    assert np.isclose(sim_res.final_population(1), 1.0, atol=0.03)


def test_t1_experiment():
    target_t1 = 3000.0  # ns
    q = Transmon("q0", f_q=5.0, levels=2, t1=target_t1)

    # Use square pi pulse: duration 20ns, amp = pi / 20
    pi_pulse = SquarePulse(duration=20.0, amp=np.pi / 20.0)

    delays = np.linspace(0.0, 6000.0, 15)
    t1_res = T1Experiment.run(q, pi_pulse, delays=delays, dt=1.0)

    # Fitted T1 should match target within 5%
    assert np.isclose(t1_res.t1_fit, target_t1, rtol=0.05)


def test_ramsey_experiment():
    target_t2 = 2500.0  # ns
    detuning = 0.002  # 2 MHz in GHz
    q = Transmon("q0", f_q=5.0, levels=2, t1=10000.0, t2=target_t2)

    # pi/2 pulse: duration 20ns, amp = (pi/2) / 20
    pi_half_pulse = SquarePulse(duration=20.0, amp=0.5 * np.pi / 20.0)

    delays = np.linspace(0.0, 2500.0, 35)
    ramsey_res = RamseyExperiment.run(
        q,
        pi_half_pulse=pi_half_pulse,
        detuning=detuning,
        delays=delays,
        dt=1.0,
    )

    # Fitted detuning should match 2 MHz within 5%
    assert np.isclose(ramsey_res.fitted_detuning, detuning, rtol=0.05)
    # Fitted T2* should match target within 15%
    assert np.isclose(ramsey_res.t2_star, target_t2, rtol=0.15)
