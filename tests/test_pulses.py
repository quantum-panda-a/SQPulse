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
    for ramp in ["cosine", "gaussian", "tanh"]:
        p = FlatTopPulse(duration=50e-9, amp=1.0, ramp_time=10e-9, ramp_type=ramp)
        t, c_wave = p.sample(dt=5e-10)
        assert np.isclose(c_wave[0].real, 0.0, atol=1e-6), f"{ramp} ramp did not start at 0"
        assert np.isclose(c_wave[-1].real, 0.0, atol=1e-6), f"{ramp} ramp did not end at 0"
        # Flat top region in middle should be 1.0
        mid_idx = len(t) // 2
        assert np.isclose(c_wave[mid_idx].real, 1.0, atol=1e-5), f"{ramp} ramp middle amp mismatch"

    # Test custom sigma for gaussian ramp
    p_sigma = FlatTopPulse(duration=50e-9, amp=1.0, ramp_time=10e-9, ramp_type="gaussian", sigma=4e-9)
    t, c_wave = p_sigma.sample(dt=5e-10)
    assert np.isclose(c_wave[0].real, 0.0, atol=1e-6)
    assert np.isclose(c_wave[-1].real, 0.0, atol=1e-6)


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


def test_slepian_pulse():
    """Test SlepianPulse boundary conditions, DRAG, and factory function."""
    from sqpulse.pulses import SlepianPulse, slepian_pulse

    # 1. Instance and factory
    p = SlepianPulse(duration=40e-9, amp=0.8, nw=3.0)
    p_factory = slepian_pulse(duration=40e-9, amp=0.8, nw=3.0)
    t, wave = p.sample(dt=2e-10)
    t_f, wave_f = p_factory.sample(dt=2e-10)

    assert np.allclose(wave, wave_f)
    # Endpoints with zero_offset=True should start and end at 0
    assert np.isclose(wave[0].real, 0.0, atol=1e-6)
    assert np.isclose(wave[-1].real, 0.0, atol=1e-6)
    assert np.isclose(np.max(np.abs(wave)), 0.8, atol=1e-3)

    # 2. DRAG on SlepianPulse
    p_drag = SlepianPulse(duration=40e-9, amp=1.0, nw=3.0, drag=1.0)
    _, wave_drag = p_drag.sample(dt=2e-10)
    assert not np.allclose(wave_drag.imag, 0.0)
    mid_idx = len(wave_drag) // 2
    assert np.isclose(wave_drag.imag[mid_idx], 0.0, atol=1e-4)

    # 3. Spectral leakage: Slepian pulse has substantially lower leakage than square pulse
    p_sq = SquarePulse(duration=40e-9, amp=1.0)
    leak_slepian = spectral_leakage(p, cutoff_freq=120e6, dt=1e-10)
    leak_square = spectral_leakage(p_sq, cutoff_freq=120e6, dt=1e-10)
    assert leak_slepian < leak_square

    # 4. nw validation
    with pytest.raises(ValueError, match="nw must be positive"):
        SlepianPulse(duration=40e-9, nw=-1.0)


def test_flattop_pulse_tanh():
    """Test FlatTopPulse with tanh ramp and invalid ramp_type validation."""
    p_tanh = FlatTopPulse(duration=60e-9, amp=1.0, ramp_time=15e-9, ramp_type="tanh")
    t, c_wave = p_tanh.sample(dt=5e-10)

    # Boundary conditions
    assert np.isclose(c_wave[0].real, 0.0, atol=1e-6)
    assert np.isclose(c_wave[-1].real, 0.0, atol=1e-6)

    # Flat top region in middle should be 1.0
    mid_idx = len(t) // 2
    assert np.isclose(c_wave[mid_idx].real, 1.0, atol=1e-5)

    # DRAG quadrature on FlatTop with tanh
    p_tanh_drag = FlatTopPulse(duration=60e-9, amp=1.0, ramp_time=15e-9, ramp_type="tanh", drag=1.0)
    _, wave_drag = p_tanh_drag.sample(dt=5e-10)
    # Derivative is zero in the flat region
    assert np.isclose(wave_drag.imag[mid_idx], 0.0, atol=1e-5)
    # Derivative is non-zero during ramp-up and ramp-down
    ramp_idx = int(round(7.5e-9 / 5e-10))
    assert not np.isclose(wave_drag.imag[ramp_idx], 0.0, atol=1e-3)

    # Invalid ramp_type should raise ValueError
    with pytest.raises(ValueError, match="ramp_type must be 'cosine', 'gaussian', or 'tanh'"):
        FlatTopPulse(duration=60e-9, ramp_type="invalid_ramp")


def test_noise_injection():
    """Test additive noise injection in Pulse.sample()."""
    p = GaussianPulse(duration=40e-9, amp=1.0)

    # 1. noise_sigma=0 -> identical to noiseless
    t, wave_clean = p.sample(dt=5e-10)
    _, wave_clean_explicit = p.sample(dt=5e-10, noise_sigma=0.0)
    assert np.allclose(wave_clean, wave_clean_explicit)

    # 2. noise_sigma > 0 -> noisy wave
    _, wave_noisy1 = p.sample(dt=5e-10, noise_sigma=0.05, seed=42)
    _, wave_noisy2 = p.sample(dt=5e-10, noise_sigma=0.05, seed=42)
    _, wave_noisy3 = p.sample(dt=5e-10, noise_sigma=0.05, seed=99)

    # Reproducibility with identical seed
    assert np.allclose(wave_noisy1, wave_noisy2)
    # Different seed produces different noise
    assert not np.allclose(wave_noisy1, wave_noisy3)
    # Residual noise standard deviation close to noise_sigma
    residual_real = (wave_noisy1 - wave_clean).real
    assert 0.02 < np.std(residual_real) < 0.08

    # 3. 1/f colored noise (noise_alpha=1.0)
    _, wave_pink = p.sample(dt=5e-10, noise_sigma=0.05, noise_alpha=1.0, seed=123)
    assert len(wave_pink) == len(wave_clean)
    assert not np.allclose(wave_pink, wave_clean)

    # 4. Pulse-level noise configuration
    p_noisy = GaussianPulse(duration=40e-9, amp=0.5, noise_sigma=0.04, noise_alpha=0.0, scale_noise=True)
    assert p_noisy.noise_sigma == 0.04
    assert p_noisy.scale_noise is True
    _, w_p_noisy = p_noisy.sample(dt=5e-10, seed=7)
    _, w_p_clean = GaussianPulse(duration=40e-9, amp=0.5).sample(dt=5e-10)
    assert not np.allclose(w_p_noisy, w_p_clean)


def test_idle_pulse():
    from sqpulse.pulses import IdlePulse, idle_pulse

    p = IdlePulse(duration=50e-9)
    p_factory = idle_pulse(duration=50e-9)
    t, wave = p.sample(dt=1e-9)
    t_f, wave_f = p_factory.sample(dt=1e-9)

    assert np.allclose(wave, 0.0)
    assert np.allclose(wave_f, 0.0)
    assert p.amp == 0.0
    assert len(t) == 51


def test_flattop_overshoot():
    from sqpulse.pulses import FlatTopPulse

    # Standard flattop without overshoot
    p_standard = FlatTopPulse(duration=100e-9, amp=0.8, ramp_time=20e-9, overshoot_amp=0.0)
    # Flattop with overshoot
    p_os = FlatTopPulse(duration=100e-9, amp=0.8, ramp_time=20e-9, overshoot_amp=0.2, overshoot_len=15e-9)

    t, w_std = p_standard.sample(dt=5e-10)
    t, w_os = p_os.sample(dt=5e-10)

    # Boundaries start and end at 0
    assert np.isclose(w_os[0].real, 0.0, atol=1e-6)
    assert np.isclose(w_os[-1].real, 0.0, atol=1e-6)

    # In flat plateau (e.g. at 50 ns), overshoot has zero effect
    mid_idx = len(t) // 2
    assert np.isclose(w_os[mid_idx].real, 0.8, atol=1e-5)
    assert np.isclose(w_std[mid_idx].real, 0.8, atol=1e-5)

    # Peak in the rising shoulder is higher due to overshoot
    peak_std = np.max(w_std.real)
    peak_os = np.max(w_os.real)
    assert peak_os > peak_std

    # overshoot_len validation
    with pytest.raises(ValueError, match="2 \\* overshoot_len"):
        FlatTopPulse(duration=50e-9, overshoot_len=30e-9)


def test_net_zero_pulse():
    from sqpulse.pulses import NetZeroPulse, net_zero_pulse

    # 1. ERF ramp type
    p_erf = NetZeroPulse(duration=60e-9, amp=0.9, ramp_type="erf")
    t, w_erf = p_erf.sample(dt=2e-10)

    # Boundaries zero
    assert np.isclose(w_erf[0].real, 0.0, atol=1e-5)
    assert np.isclose(w_erf[-1].real, 0.0, atol=1e-5)

    # Net-zero integral condition: \int V(t) dt == 0
    integral_erf = np.trapezoid(w_erf.real, t)
    assert np.isclose(integral_erf, 0.0, atol=1e-12)

    # Positive peak in 1st half, negative trough in 2nd half
    mid = len(t) // 2
    assert np.max(w_erf[:mid].real) > 0.8
    assert np.min(w_erf[mid:].real) < -0.8

    # 2. Cosine ramp type
    p_cos = net_zero_pulse(duration=60e-9, amp=1.0, ramp_type="cosine")
    t_c, w_cos = p_cos.sample(dt=2e-10)
    integral_cos = np.trapezoid(w_cos.real, t_c)
    assert np.isclose(integral_cos, 0.0, atol=1e-12)
    assert np.isclose(w_cos[0].real, 0.0, atol=1e-6)
    assert np.isclose(w_cos[-1].real, 0.0, atol=1e-6)

    # 3. Net-zero with overshoot
    p_os = NetZeroPulse(duration=60e-9, amp=0.8, overshoot_amp=0.15, overshoot_len=5e-9)
    t_os, w_os = p_os.sample(dt=2e-10)
    integral_os = np.trapezoid(w_os.real, t_os)
    assert np.isclose(integral_os, 0.0, atol=1e-12)


def test_cosine_hd2_drag_pulse():
    from sqpulse.pulses import CosineHD2DRAGPulse, cosine_hd2_drag_pulse

    # Instance & factory
    p = CosineHD2DRAGPulse(duration=20e-9, amp=1.0, alpha=-250e6, f_suppress=90e6)
    p_f = cosine_hd2_drag_pulse(duration=20e-9, amp=1.0, alpha=-250e6, f_suppress=90e6)
    t, wave = p.sample(dt=2e-10)
    _, wave_f = p_f.sample(dt=2e-10)
    assert np.allclose(wave, wave_f)

    # Boundary conditions: both I and Q start and end at 0
    assert np.isclose(wave[0].real, 0.0, atol=1e-6)
    assert np.isclose(wave[-1].real, 0.0, atol=1e-6)
    assert np.isclose(wave[0].imag, 0.0, atol=1e-6)
    assert np.isclose(wave[-1].imag, 0.0, atol=1e-6)

    # Q quadrature at midpoint should be 0 due to antisymmetry
    mid = len(wave) // 2
    assert np.isclose(wave[mid].imag, 0.0, atol=1e-5)

    # Non-zero Q elsewhere
    assert not np.allclose(wave.imag, 0.0)

    # Beta2 sensitivity: higher f_suppress lowers beta2
    p_high_suppress = CosineHD2DRAGPulse(duration=20e-9, amp=1.0, f_suppress=200e6)
    assert p_high_suppress.beta2 < p.beta2


def test_phase_modulated_sin_pulse():
    from sqpulse.pulses import PhaseModulatedSinPulse, phase_modulated_sin_pulse

    p = PhaseModulatedSinPulse(duration=40e-9, amp=0.8, mod_freq=50e6)
    p_f = phase_modulated_sin_pulse(duration=40e-9, amp=0.8, mod_freq=50e6)
    t, wave = p.sample(dt=2e-10)
    _, wave_f = p_f.sample(dt=2e-10)
    assert np.allclose(wave, wave_f)

    # Envelope boundaries: sin(0)=0 and sin(pi)=0
    assert np.isclose(np.abs(wave[0]), 0.0, atol=1e-6)
    assert np.isclose(np.abs(wave[-1]), 0.0, atol=1e-6)

    # The phase modulation should create both real and imaginary components
    assert not np.allclose(wave.real, 0.0)
    assert not np.allclose(wave.imag, 0.0)



