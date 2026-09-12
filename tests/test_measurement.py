"""Tests for unified Measurement framework and ProjectiveBackend in SQPulse."""

import numpy as np
import pytest
from sqpulse import Transmon, PulseSequence, SquarePulse, Measurement, ProjectiveBackend, ProjectiveResult


def test_measurement_projective_backend_run():
    q = Transmon("q0", f_q=5.0e9, levels=2, omega_d=2.0 * np.pi * 50e6)
    duration = 20e-9
    v0_pi = np.pi / (q.omega_d * duration)

    # Pi pulse sequence
    seq = PulseSequence().add(q.xy, SquarePulse(duration=duration, amp=v0_pi))

    # Run via Measurement.run with default backend
    res = Measurement.run(q, seq, dt=5e-10)
    assert isinstance(res, ProjectiveResult)
    assert np.isclose(res.final_population(1), 1.0, atol=1e-3)
    assert np.isclose(res.final_population(0), 0.0, atol=1e-3)

    # Run via explicit backend instance
    backend = ProjectiveBackend()
    res2 = Measurement.run(q, seq, backend=backend, dt=5e-10)
    assert np.isclose(res2.final_population(1), 1.0, atol=1e-3)


def test_projective_measurement_shots_and_counts():
    q = Transmon("q0", f_q=5.0e9, levels=2, omega_d=2.0 * np.pi * 50e6)
    duration = 20e-9
    v0_pi2 = 0.5 * np.pi / (q.omega_d * duration)

    # Pi/2 pulse prepares (|0> + |1>)/sqrt(2)
    seq = PulseSequence().add(q.xy, SquarePulse(duration=duration, amp=v0_pi2))

    res = Measurement.run(q, seq, backend="projective", shots=2000, seed=42)
    assert res.shots is not None
    assert len(res.shots) == 2000

    counts = res.counts()
    assert 0 in counts and 1 in counts
    # For equal superposition, counts should be approximately 50/50
    assert 800 < counts[0] < 1200
    assert 800 < counts[1] < 1200
