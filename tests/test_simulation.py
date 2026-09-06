"""Tests for sqpulse.simulation in SI units."""

import numpy as np
import pytest
from sqpulse.models import Transmon
from sqpulse.pulses import SquarePulse
from sqpulse.sequence import PulseSequence
from sqpulse.simulation import Simulator


def test_resonant_rabi_flip():
    """A square pi pulse of amplitude Omega (rad/s) and duration tau = pi / Omega (s) should flip |0> to |1>."""
    q = Transmon("q0", f_q=5.0e9, alpha=-300e6, levels=2)

    # In our definition:
    # H_drive_x = 0.5 * (a + a^dag) = 0.5 * sigma_x
    # Rotation angle = Omega * duration (since H = 0.5 * Omega * sigma_x -> exp(-i * Omega*t/2 * sigma_x))
    duration = 20e-9  # 20 ns in s
    # For pi rotation: Omega * duration = pi -> Omega = pi / duration rad/s
    omega = np.pi / duration  # ~ 1.57e8 rad/s
    p_pi = SquarePulse(duration=duration, amp=omega)

    seq = PulseSequence().add(q.drive, p_pi)
    res = Simulator.run(q, seq, dt=2e-10)

    # Final population of |1> should be ~1.0
    p1 = res.final_population(1)
    assert np.isclose(p1, 1.0, atol=1e-3)

    # Check Bloch vector z coordinate goes from +1 to -1
    x, y, z = res.bloch_vector()
    assert np.isclose(z[0], 1.0, atol=1e-3)
    assert np.isclose(z[-1], -1.0, atol=1e-3)
