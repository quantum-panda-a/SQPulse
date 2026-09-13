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


def test_align_center():
    q = Transmon("q0", f_q=5.0e9)
    seq = PulseSequence("align_center_test")
    p_z = SquarePulse(duration=120e-9, amp=0.1)
    p_xy = GaussianPulse(duration=100e-9, amp=0.5)

    seq.align_center((q.z, p_z), (q.xy, p_xy))

    # Check start and end times
    sp_z = seq._channels["q0.z"][0]
    sp_xy = seq._channels["q0.xy"][0]

    assert np.isclose(sp_z.t_start, 0.0)
    assert np.isclose(sp_z.t_end, 120e-9)
    assert np.isclose(sp_xy.t_start, 10e-9)
    assert np.isclose(sp_xy.t_end, 110e-9)

    # Check center alignment
    center_z = (sp_z.t_start + sp_z.t_end) / 2.0
    center_xy = (sp_xy.t_start + sp_xy.t_end) / 2.0
    assert np.isclose(center_z, 60e-9)
    assert np.isclose(center_xy, 60e-9)

    # Check clock advancement and total sequence duration
    assert np.isclose(seq._channel_clocks["q0.z"], 120e-9)
    assert np.isclose(seq._channel_clocks["q0.xy"], 120e-9)
    assert np.isclose(seq.duration, 120e-9)

    # Subsequent pulse starts at 120e-9
    p_next = GaussianPulse(duration=20e-9)
    seq.add(q.xy, p_next)
    sp_next = seq._channels["q0.xy"][1]
    assert np.isclose(sp_next.t_start, 120e-9)


def test_align_modes():
    seq_left = PulseSequence("align_left")
    p1 = SquarePulse(duration=80e-9)
    p2 = SquarePulse(duration=40e-9)
    seq_left.align(("ch1", p1), ("ch2", p2), mode="left")
    assert np.isclose(seq_left._channels["ch1"][0].t_start, 0.0)
    assert np.isclose(seq_left._channels["ch2"][0].t_start, 0.0)
    assert np.isclose(seq_left._channel_clocks["ch1"], 80e-9)
    assert np.isclose(seq_left._channel_clocks["ch2"], 80e-9)

    seq_right = PulseSequence("align_right")
    seq_right.align(("ch1", p1), ("ch2", p2), mode="right")
    assert np.isclose(seq_right._channels["ch1"][0].t_start, 0.0)
    assert np.isclose(seq_right._channels["ch2"][0].t_start, 40e-9)
    assert np.isclose(seq_right._channels["ch2"][0].t_end, 80e-9)
    assert np.isclose(seq_right._channel_clocks["ch1"], 80e-9)
    assert np.isclose(seq_right._channel_clocks["ch2"], 80e-9)

    # Test alignment on top of an existing baseline
    seq_offset = PulseSequence("align_offset")
    seq_offset.add("ch1", SquarePulse(duration=50e-9))
    # ch1 is at 50ns, ch2 is not yet initialized (at 0ns)
    seq_offset.align_center(("ch1", p1), ("ch2", p2))
    # Baseline must be 50ns
    assert np.isclose(seq_offset._channels["ch1"][1].t_start, 50e-9)
    assert np.isclose(seq_offset._channels["ch2"][0].t_start, 70e-9)  # 50 + (80 - 40)/2
    assert np.isclose(seq_offset._channel_clocks["ch1"], 130e-9)
    assert np.isclose(seq_offset._channel_clocks["ch2"], 130e-9)


def test_align_input_formats_and_errors():
    seq = PulseSequence("test_errors")
    p1 = SquarePulse(duration=20e-9)
    p2 = SquarePulse(duration=10e-9)

    # Pass as single list of tuples
    seq.align_center([("ch1", p1), ("ch2", p2)])
    assert np.isclose(seq.duration, 20e-9)

    # Empty input returns self
    assert seq.align() is seq

    # Invalid mode
    with pytest.raises(ValueError, match="Invalid alignment mode"):
        seq.align(("ch1", p1), mode="diagonal")

    # Invalid pair format
    with pytest.raises(ValueError, match="must be a \\(channel, pulse\\) pair"):
        seq.align("ch1")

