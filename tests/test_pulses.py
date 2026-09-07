"""Tests for sqpulse.pulses in SI units."""

import numpy as np
import pytest
from sqpulse.pulses import (
    GaussianPulse,
    CosinePulse,
    LorentzianPulse,
    SquarePulse,
    FlatTopPulse,
    SechPulse,
    CustomPulse,
    DRAGPulse,
    spectral_leakage,
)


def test_pulse_boundary_conditions():
    """Verify that smooth pulses start and end at zero."""
    t_pulse = 40e-9  # 40 ns in s
    pulses = [
        GaussianPulse(duration=t_pulse, amp=1.0, chop=4.0),
        CosinePulse(duration=t_pulse, amp=1.0),
        LorentzianPulse(duration=t_pulse, amp=1.0),
        SechPulse(duration=t_pulse, amp=1.0),
    ]

    for p in pulses:
        t, c_wave = p.sample(dt=1e-10)
        # Endpoints should be practically zero
        assert np.isclose(c_wave[0].real, 0.0, atol=1e-6), f"{p.name} did not start at 0"
        assert np.isclose(c_wave[-1].real, 0.0, atol=1e-6), f"{p.name} did not end at 0"
        # Peak amplitude should match amp
        assert np.isclose(np.max(np.abs(c_wave)), 1.0, atol=1e-3), f"{p.name} peak amp mismatch"


def test_square_pulse():
    p = SquarePulse(duration=20e-9, amp=0.5)
    t, c_wave = p.sample(dt=1e-9)
    assert len(t) == 21
    assert np.allclose(c_wave.real, 0.5)
    assert np.allclose(c_wave.imag, 0.0)


def test_flattop_pulse():
    p = FlatTopPulse(duration=50e-9, amp=1.0, ramp_time=10e-9, ramp_type="cosine")
    t, c_wave = p.sample(dt=5e-10)
    assert np.isclose(c_wave[0].real, 0.0, atol=1e-6)
    assert np.isclose(c_wave[-1].real, 0.0, atol=1e-6)
    # Flat top region in middle should be 1.0
    mid_idx = len(t) // 2
    assert np.isclose(c_wave[mid_idx].real, 1.0, atol=1e-5)


def test_drag_quadrature():
    """DRAG should introduce non-zero Q proportional to derivative using dimensionless beta."""
    p_no_drag = GaussianPulse(duration=40e-9, amp=1.0, drag=0.0)
    p_drag = GaussianPulse(duration=40e-9, amp=1.0, drag=1.0)

    _, wave_no_drag = p_no_drag.sample(dt=5e-10)
    _, wave_drag = p_drag.sample(dt=5e-10)

    assert np.allclose(wave_no_drag.imag, 0.0)
    assert not np.allclose(wave_drag.imag, 0.0)
    # The derivative of symmetric Gaussian is odd, so Q at center should be 0
    mid_idx = len(wave_drag) // 2
    assert np.isclose(wave_drag.imag[mid_idx], 0.0, atol=1e-4)


def test_drag_dimensionless_properties():
    """Verify dimensionless DRAG linearity, default beta=1.0 in DRAGPulse, and alpha scaling."""
    # 1. Linearity: drag=2.0 should produce exactly 2x the Q quadrature of drag=1.0
    p1 = GaussianPulse(duration=40e-9, amp=1.0, drag=1.0)
    p2 = GaussianPulse(duration=40e-9, amp=1.0, drag=2.0)
    _, w1 = p1.sample(dt=5e-10)
    _, w2 = p2.sample(dt=5e-10)
    assert np.allclose(w2.imag, 2.0 * w1.imag)

    # 2. DRAGPulse default has drag=1.0
    p_drag_class = DRAGPulse(duration=40e-9, amp=1.0)
    assert p_drag_class.drag == 1.0
    _, w_drag_class = p_drag_class.sample(dt=5e-10)
    assert np.allclose(w_drag_class.imag, w1.imag)

    # 3. Alpha scaling: doubling alpha (e.g. -500 MHz vs -250 MHz) halves the Q quadrature
    p_alpha1 = GaussianPulse(duration=40e-9, amp=1.0, drag=1.0, alpha=-250.0e6)
    p_alpha2 = GaussianPulse(duration=40e-9, amp=1.0, drag=1.0, alpha=-500.0e6)
    _, wa1 = p_alpha1.sample(dt=5e-10)
    _, wa2 = p_alpha2.sample(dt=5e-10)
    assert np.allclose(wa2.imag, 0.5 * wa1.imag)


def test_pulse_fft_and_spectral_leakage():
    """Square pulse has larger high-frequency leakage than Cosine/Gaussian."""
    duration = 40e-9  # 40 ns in s
    p_square = SquarePulse(duration=duration, amp=1.0)
    p_cosine = CosinePulse(duration=duration, amp=1.0)

    cutoff = 100e6  # 100 MHz in Hz
    leak_square = spectral_leakage(p_square, cutoff_freq=cutoff, dt=1e-10)
    leak_cosine = spectral_leakage(p_cosine, cutoff_freq=cutoff, dt=1e-10)

    # Cosine pulse should have much lower spectral leakage than square pulse
    assert leak_cosine < leak_square


def test_pulse_scaling():
    p = GaussianPulse(duration=30e-9, amp=0.2)
    p_scaled = p * 3.0
    assert np.isclose(p_scaled.amp, 0.6)


def test_pulse_amp_range_warning():
    """Verify warning is emitted when |amp| > 1.0, and no warning when amp in [-1, 1]."""
    import warnings

    # Valid amplitudes: within [-1.0, 1.0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        GaussianPulse(duration=20e-9, amp=1.0)
        GaussianPulse(duration=20e-9, amp=-1.0)
        GaussianPulse(duration=20e-9, amp=0.0)
        GaussianPulse(duration=20e-9, amp=0.5)

    # Exceeding amplitude > 1.0
    with pytest.warns(UserWarning, match=r"exceeds the normalized range \[-1, 1\]"):
        GaussianPulse(duration=20e-9, amp=1.2)

    # Exceeding amplitude < -1.0
    with pytest.warns(UserWarning, match=r"exceeds the normalized range \[-1, 1\]"):
        GaussianPulse(duration=20e-9, amp=-1.5)

    # Dynamic attribute assignment exceeding 1.0
    p = GaussianPulse(duration=20e-9, amp=0.5)
    with pytest.warns(UserWarning, match=r"exceeds the normalized range \[-1, 1\]"):
        p.amp = 2.0

    # Scaling pulse exceeding 1.0
    p_base = GaussianPulse(duration=20e-9, amp=0.6)
    with pytest.warns(UserWarning, match=r"exceeds the normalized range \[-1, 1\]"):
        _ = p_base * 2.0

