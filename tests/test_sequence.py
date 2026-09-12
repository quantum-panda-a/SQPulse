"""Tests for sqpulse.sequence in SI units."""

import numpy as np
import pytest
from sqpulse.sequence import PulseSequence, Channel
from sqpulse.pulses import GaussianPulse, SquarePulse
from sqpulse.models import Transmon


def test_pulse_sequence_timing():
    seq = PulseSequence("test_seq")
    p1 = GaussianPulse(duration=30e-9, amp=1.0)
    p2 = SquarePulse(duration=20e-9, amp=0.5)

    seq.add("xy", p1)
    assert np.isclose(seq.duration, 30e-9)

    seq.delay("xy", 10e-9)
    assert np.isclose(seq.duration, 40e-9)

    seq.add("xy", p2)
    assert np.isclose(seq.duration, 60e-9)


def test_multichannel_sync():
    seq = PulseSequence("multi")
    seq.add("q0.xy", GaussianPulse(duration=30e-9))
    seq.add("q1.xy", GaussianPulse(duration=50e-9))

    assert np.isclose(seq._channel_clocks["q0.xy"], 30e-9)
    assert np.isclose(seq._channel_clocks["q1.xy"], 50e-9)

    seq.sync()
    assert np.isclose(seq._channel_clocks["q0.xy"], 50e-9)
    assert np.isclose(seq._channel_clocks["q1.xy"], 50e-9)


def test_sampling_and_compilation():
    q = Transmon("q0", f_q=5.0e9)
    seq = PulseSequence("compile_test")
    p = GaussianPulse(duration=20e-9, amp=0.5)
    seq.add(q.xy, p)

    times, evo = seq.to_qutip_evo(q, dt=5e-10)
    assert len(times) == 41
    assert evo is not None


def test_sequence_xy_z_ro_channels():
    q = Transmon("q0", f_q=5.0e9, d=0.2)
    seq = PulseSequence("full_seq")
    seq.add(q.xy, GaussianPulse(duration=20e-9, amp=0.5))
    seq.sync()
    seq.add(q.z, SquarePulse(duration=30e-9, amp=0.1))
    seq.sync()
    seq.add(q.ro, SquarePulse(duration=100e-9, amp=0.8))

    assert "q0.xy" in seq.channels
    assert "q0.z" in seq.channels
    assert "q0.ro" in seq.channels
    times, evo = seq.to_qutip_evo(q, dt=1e-9)
    assert len(times) > 0
    assert evo is not None
