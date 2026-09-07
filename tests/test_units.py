"""Tests for SQPulse physical units, aliases, and parameter validation."""

import numpy as np
import pytest
from sqpulse import (
    GaussianPulse,
    CosinePulse,
    LorentzianPulse,
    SquarePulse,
    FlatTopPulse,
    SechPulse,
    DRAGPulse,
    gaussian_pulse,
    cosine_pulse,
    flattop_pulse,
    lorentzian_pulse,
    square_pulse,
    sech_pulse,
    drag_pulse,
    PulseSequence,
    s,
    ms,
    us,
    ns,
    ps,
    Hz,
    kHz,
    MHz,
    GHz,
)


def test_unit_constants():
    """Verify that unit constants accurately represent SI scale factors."""
    assert s == 1.0
    assert ms == 1e-3
    assert us == 1e-6
    assert ns == 1e-9
    assert ps == 1e-12

    assert Hz == 1.0
    assert kHz == 1e3
    assert MHz == 1e6
    assert GHz == 1e9

    # Check composition
    assert 40 * ns == 40e-9
    assert 5 * GHz == 5e9
    assert -250 * MHz == -250e6


def test_40ns_pulse_generation_and_adaptive_sample():
    """Verify pulses with duration=40e-9 sample cleanly with adaptive dt."""
    pulses = [
        GaussianPulse(duration=40e-9),
        CosinePulse(duration=40e-9),
        LorentzianPulse(duration=40e-9),
        SquarePulse(duration=40e-9),
        FlatTopPulse(duration=40e-9, ramp_time=8e-9),
        SechPulse(duration=40e-9),
        DRAGPulse(duration=40e-9),
    ]

    for p in pulses:
        # Default adaptive sampling
        t, wave = p.sample()
        assert len(t) >= 100, f"{p.name} should have >= 100 points adaptively"
        assert np.isclose(t[0], 0.0)
        assert np.isclose(t[-1], 40e-9)

        # Explicit dt in seconds (e.g. 0.1 ns = 0.1e-9 s)
        t_fine, wave_fine = p.sample(dt=0.1 * ns)
        assert len(t_fine) == 401


def test_length_alias():
    """Verify that length parameter works identically to duration."""
    p1 = GaussianPulse(duration=40 * ns, amp=0.8)
    p2 = GaussianPulse(length=40 * ns, amp=0.8)

    assert p1.duration == p2.duration
    assert p1.length == 40e-9
    assert p2.length == 40e-9

    t1, w1 = p1.sample(dt=1e-9)
    t2, w2 = p2.sample(dt=1e-9)
    assert np.allclose(w1, w2)

    # Specifying neither or both should raise ValueError
    with pytest.raises(ValueError, match="must be provided"):
        GaussianPulse()
    with pytest.raises(ValueError, match="not both"):
        GaussianPulse(duration=40e-9, length=40e-9)


def test_functional_pulse_wrappers():
    """Verify snake_case factory functions create expected pulse instances."""
    g = gaussian_pulse(duration=40 * ns, amp=1.0)
    assert isinstance(g, GaussianPulse)
    assert g.duration == 40e-9

    c = cosine_pulse(length=40 * ns)
    assert isinstance(c, CosinePulse)

    ft = flattop_pulse(duration=40 * ns, ramp_time=8 * ns)
    assert isinstance(ft, FlatTopPulse)
    assert ft.ramp_time == 8e-9

    l = lorentzian_pulse(duration=40 * ns, gamma=10 * ns)
    assert isinstance(l, LorentzianPulse)

    s_p = square_pulse(duration=20 * ns)
    assert isinstance(s_p, SquarePulse)

    se = sech_pulse(duration=30 * ns)
    assert isinstance(se, SechPulse)

    d = drag_pulse(duration=40 * ns, drag=1.0)
    assert isinstance(d, DRAGPulse)


def test_sampling_dt_validation_and_hints():
    """Passing dt >= duration should raise ValueError with informative unit hint."""
    p = CosinePulse(duration=40e-9)

    # User mistakenly passes dt=0.1 (0.1 seconds instead of 0.1 ns)
    with pytest.raises(ValueError) as exc_info:
        p.sample(dt=0.1)

    msg = str(exc_info.value)
    assert "Sampling interval dt=0.1 s cannot be greater than or equal to pulse duration" in msg
    assert "0.1e-9 s (0.1 ns)" in msg

    # Non-positive dt
    with pytest.raises(ValueError, match="must be positive"):
        p.sample(dt=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        p.sample(dt=-1e-9)


def test_flattop_ramp_time_validation_and_hints():
    """FlatTopPulse should catch ramp_time errors and give helpful SI unit hint."""
    # User passes ramp_time=8.0 (8 seconds) instead of 8e-9 (8 ns)
    with pytest.raises(ValueError) as exc_info:
        FlatTopPulse(duration=40e-9, ramp_time=8.0)

    msg = str(exc_info.value)
    assert "cannot exceed duration" in msg
    assert "ramp_time=8.0e-9 s (8.0 ns)" in msg

    # Valid usage with ns constant works perfectly
    p = FlatTopPulse(duration=40 * ns, ramp_time=8 * ns)
    assert p.ramp_time == 8e-9


def test_lorentzian_gamma_validation_and_hints():
    """LorentzianPulse should catch unusually large gamma and give hint."""
    # User passes gamma=10.0 (10 seconds) instead of 10e-9
    with pytest.raises(ValueError) as exc_info:
        LorentzianPulse(duration=40e-9, gamma=10.0)

    msg = str(exc_info.value)
    assert "gamma=10.0e-9 s (10.0 ns)" in msg

    # Valid usage
    p = LorentzianPulse(duration=40 * ns, gamma=10 * ns)
    assert p.gamma == 10e-9


def test_gaussian_sigma_validation_and_hints():
    """GaussianPulse should catch sigma exceeding duration and give hint."""
    with pytest.raises(ValueError) as exc_info:
        GaussianPulse(duration=40e-9, sigma=10.0)

    msg = str(exc_info.value)
    assert "sigma=10.0e-9 s (10.0 ns)" in msg

    # Valid usage
    p = GaussianPulse(duration=40 * ns, sigma=10 * ns)
    assert p.sigma == 10e-9


def test_pulse_sequence_sampling_validation():
    """PulseSequence.sample should validate dt against total sequence duration."""
    seq = PulseSequence("seq_test")
    p = GaussianPulse(duration=40e-9)
    seq.add("drive", p)

    # dt=0.1 s is much larger than 40 ns sequence duration
    with pytest.raises(ValueError, match="cannot be greater than or equal to total sequence duration"):
        seq.sample(dt=0.1)

    # Valid dt
    times, waves = seq.sample(dt=1e-9)
    assert len(times) == 41
