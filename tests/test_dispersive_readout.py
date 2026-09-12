"""Tests for dispersive cavity readout measurement backend and IQ discrimination."""

import numpy as np
import pytest
from sqpulse import (
    Transmon,
    ReadoutResonator,
    PulseSequence,
    SquarePulse,
    FlatTopPulse,
    Measurement,
    DispersiveReadoutBackend,
    DispersiveResult,
    IQDiscriminator,
)
from sqpulse.measurement.dispersive.cavity import simulate_cavity_dynamics
from sqpulse.measurement.dispersive.demodulation import demodulate_and_integrate


def test_cavity_ringup_and_ringdown():
    res = ReadoutResonator("r0", f_r=7.0e9, kappa=2.0 * np.pi * 5.0e6, chi=2.0 * np.pi * 1.5e6)
    p_ro = FlatTopPulse(duration=500e-9, ramp_time=20e-9, amp=1.0)

    times, alpha = simulate_cavity_dynamics(res, qubit_state=0, readout_pulse=p_ro, dt=1e-9)

    assert times[0] == 0.0
    assert abs(alpha[0]) == 0.0

    # Intra-cavity photon number should build up during pulse
    n_photons = np.abs(alpha) ** 2
    max_photons = np.max(n_photons)
    assert max_photons > 0.1

    # Photon number should ring down after pulse duration (500 ns)
    idx_500ns = int(500e-9 / 1e-9)
    idx_end = len(times) - 1
    assert n_photons[idx_end] < n_photons[idx_500ns] * 0.1


def test_iq_discriminator():
    c0 = -1.0 + 0.0j
    c1 = 1.0 + 0.0j
    disc = IQDiscriminator(center_0=c0, center_1=c1)

    # Test predictions
    test_pts = np.array([-0.8 + 0.1j, -0.2 - 0.3j, 0.1 + 0.0j, 0.9 - 0.2j])
    preds = disc.predict(test_pts)
    assert np.array_equal(preds, [0, 0, 1, 1])

    # Confusion matrix and fidelity for perfect separation
    true_labels = np.array([0, 0, 0, 1, 1, 1])
    assigned_labels = np.array([0, 0, 0, 1, 1, 1])
    c_mat = IQDiscriminator.compute_confusion_matrix(true_labels, assigned_labels)
    assert np.allclose(c_mat, np.eye(2))
    assert np.isclose(IQDiscriminator.compute_fidelity(c_mat), 1.0)


def test_dispersive_readout_backend_ground_and_excited():
    q = Transmon("q0", f_q=5.0e9, levels=2, omega_d=2.0 * np.pi * 50e6)
    v0_pi = np.pi / (q.omega_d * 20e-9)
    res = ReadoutResonator("r0", f_r=7.0e9, kappa=2.0 * np.pi * 3e6, chi=2.0 * np.pi * 1.5e6)

    # 1. Ground state |0> measurement
    seq_ground = PulseSequence()
    # Add a delay so sequence has duration
    seq_ground.delay(q.xy, 20e-9)

    res_ground = Measurement.run(
        qubit=q,
        sequence=seq_ground,
        backend="dispersive",
        resonator=res,
        shots=1000,
        snr_db=15.0,
        seed=123,
    )
    assert isinstance(res_ground, DispersiveResult)
    counts_0 = res_ground.counts()
    # For ground state, the vast majority should be 0
    assert counts_0.get(0, 0) > 900
    assert res_ground.fidelity > 0.90

    # 2. Excited state |1> measurement (after pi pulse on xy and explicit pulse on ro)
    seq_excited = (
        PulseSequence()
        .add(q.xy, SquarePulse(duration=20e-9, amp=v0_pi))
        .sync()
        .add(q.ro, SquarePulse(duration=1000e-9, amp=1.0))
    )
    res_excited = Measurement.run(
        qubit=q,
        sequence=seq_excited,
        backend="dispersive",
        resonator=res,
        shots=1000,
        snr_db=15.0,
        seed=456,
    )
    counts_1 = res_excited.counts()
    # For excited state, the vast majority should be 1
    assert counts_1.get(1, 0) > 900
