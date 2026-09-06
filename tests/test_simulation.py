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


def test_drag_leakage_suppression():
    """Verify that a dimensionless DRAG pulse (drag=1.0) suppresses leakage to |2> by orders of magnitude."""
    from sqpulse.pulses import GaussianPulse, DRAGPulse

    q = Transmon("q0", f_q=5.0e9, alpha=-250.0e6, levels=3)
    duration = 10e-9
    amp_pi = 5.0e8

    # Pulse without DRAG
    p_nodrag = GaussianPulse(duration=duration, amp=amp_pi, drag=0.0)
    seq_nodrag = PulseSequence().add(q.drive, p_nodrag)
    res_nodrag = Simulator.run(q, seq_nodrag, dt=5e-11)
    leakage_nodrag = res_nodrag.final_population(2)

    # Pulse with dimensionless DRAG beta=1.0
    p_drag = DRAGPulse(duration=duration, amp=amp_pi, drag=1.0)
    seq_drag = PulseSequence().add(q.drive, p_drag)
    res_drag = Simulator.run(q, seq_drag, dt=5e-11)
    leakage_drag = res_drag.final_population(2)

    # DRAG should suppress leakage to |2> by at least a factor of 100
    assert leakage_nodrag > 1e-4
    assert leakage_drag < 1e-5
    assert (leakage_nodrag / leakage_drag) > 50.0
