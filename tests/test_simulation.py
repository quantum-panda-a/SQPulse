"""Tests for sqpulse.simulation in SI units."""

import numpy as np
import pytest
from sqpulse.models import Transmon
from sqpulse.pulses import SquarePulse
from sqpulse.sequence import PulseSequence
from sqpulse.simulation import Simulator


def test_resonant_rabi_flip():
    """A square pi pulse with AWG amplitude V_0 = pi / (omega_d * duration) should flip |0> to |1>."""
    # Transmon with default omega_d = 2*pi * 50 MHz
    q = Transmon("q0", f_q=5.0e9, alpha=-300e6, levels=2)

    duration = 20e-9  # 20 ns
    # For pi rotation: omega_d * V_0 * duration = pi -> V_0 = pi / (omega_d * duration) = 0.5
    v0_pi = np.pi / (q.omega_d * duration)
    assert np.isclose(v0_pi, 0.5)

    p_pi = SquarePulse(duration=duration, amp=v0_pi)
    seq = PulseSequence().add(q.drive, p_pi)
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
    seq = PulseSequence().add(q_raw.drive, p_pi)
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
    seq_nodrag = PulseSequence().add(q.drive, p_nodrag)
    res_nodrag = Simulator.run(q, seq_nodrag, dt=5e-11)
    leakage_nodrag = res_nodrag.final_population(2)

    # Pulse with dimensionless DRAG beta=1.0
    p_drag = DRAGPulse(duration=duration, amp=amp_pi, drag=1.0)
    seq_drag = PulseSequence().add(q.drive, p_drag)
    res_drag = Simulator.run(q, seq_drag, dt=5e-11)
    leakage_drag = res_drag.final_population(2)

    # DRAG should suppress leakage to |2> by at least a factor of 50
    assert leakage_nodrag > 1e-4
    assert leakage_drag < 1e-5
    assert (leakage_nodrag / leakage_drag) > 50.0
