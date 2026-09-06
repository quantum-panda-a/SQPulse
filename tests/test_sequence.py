"""Tests for sqpulse.sequence."""

import numpy as np
import pytest
from sqpulse.sequence import PulseSequence, Channel
from sqpulse.pulses import GaussianPulse, SquarePulse
from sqpulse.models import Transmon


def test_pulse_sequence_timing():
    seq = PulseSequence("test_seq")
    p1 = GaussianPulse(duration=30.0, amp=1.0)
    p2 = SquarePulse(duration=20.0, amp=0.5)

    seq.add("drive", p1)
    assert seq.duration == 30.0

    seq.delay("drive", 10.0)
    assert seq.duration == 40.0

    seq.add("drive", p2)
    assert seq.duration == 60.0


def test_multichannel_sync():
    seq = PulseSequence("multi")
    seq.add("q0.drive", GaussianPulse(duration=30.0))
    seq.add("q1.drive", GaussianPulse(duration=50.0))

    assert seq._channel_clocks["q0.drive"] == 30.0
    assert seq._channel_clocks["q1.drive"] == 50.0

    seq.sync()
    assert seq._channel_clocks["q0.drive"] == 50.0
    assert seq._channel_clocks["q1.drive"] == 50.0


def test_sampling_and_compilation():
    q = Transmon("q0", f_q=5.0)
    seq = PulseSequence("compile_test")
    p = GaussianPulse(duration=20.0, amp=0.5)
    seq.add(q.drive, p)

    times, evo = seq.to_qutip_evo(q, dt=0.5)
    assert len(times) == 41
    assert evo is not None
