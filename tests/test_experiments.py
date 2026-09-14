"""Tests for sqpulse.experiments in SI units."""

import numpy as np
import pytest
from sqpulse import Transmon, PulseSequence, SquarePulse, GaussianPulse, Simulator
from sqpulse.experiments import (
    RabiExperiment,
    T1Experiment,
    RamseyExperiment,
    QubitSpectroscopyExperiment,
    SpectroscopyExperiment,
)


def test_amplitude_rabi_calibration():
    q = Transmon("q0", f_q=5.0e9, alpha=-250e6, levels=3)
    duration = 30e-9  # 30 ns in s

    # Perform Amplitude Rabi sweep in normalized AWG amplitude V_0 in [0, 1.0]
    rabi_res = RabiExperiment.amplitude_rabi(
        q,
        pulse_type=GaussianPulse,
        duration=duration,
        dt=5e-10,
    )

    amp_pi = rabi_res.amp_pi
    assert 0.4 < amp_pi < 0.8

    # Verify that applying amp_pi achieves state inversion (P1 ~ 1.0)
    pi_pulse = GaussianPulse(duration=duration, amp=amp_pi)
    seq = PulseSequence().add(q.xy, pi_pulse)
    sim_res = Simulator.run(q, seq, dt=5e-10)

    assert np.isclose(sim_res.final_population(1), 1.0, atol=0.03)


def test_t1_experiment():
    target_t1 = 3.0e-6  # 3 us in s
    q = Transmon("q0", f_q=5.0e9, levels=2, t1=target_t1)

    # Use square pi pulse: duration 20ns, amp = 0.5 (normalized V_0)
    duration = 20e-9
    pi_pulse = SquarePulse(duration=duration, amp=0.5)

    delays = np.linspace(0.0, 6.0e-6, 15)
    t1_res = T1Experiment.run(q, pi_pulse, delays=delays, dt=1e-9)

    # Fitted T1 should match target within 5%
    assert np.isclose(t1_res.t1_fit, target_t1, rtol=0.05)


def test_ramsey_experiment():
    target_t2 = 2.5e-6  # 2.5 us in s
    detuning = 2.0e6  # 2 MHz in Hz
    q = Transmon("q0", f_q=5.0e9, levels=2, t1=10.0e-6, t2=target_t2)

    # pi/2 pulse: duration 20ns, amp = 0.25 (normalized V_0)
    duration = 20e-9
    pi_half_pulse = SquarePulse(duration=duration, amp=0.25)

    delays = np.linspace(0.0, 2.5e-6, 35)
    ramsey_res = RamseyExperiment.run(
        q,
        pi_half_pulse=pi_half_pulse,
        detuning=detuning,
        delays=delays,
        dt=1e-9,
    )

    # Fitted detuning should match 2 MHz within 5%
    assert np.isclose(ramsey_res.fitted_detuning, detuning, rtol=0.05)
    # Fitted T2* should match target within 15%
    assert np.isclose(ramsey_res.t2_star, target_t2, rtol=0.15)


def test_spectroscopy_weak_drive():
    q = Transmon("q0", f_q=5.0e9, alpha=-250e6, levels=4)
    # Weak drive: only single-photon excitation f01, P2 ~ 0
    freqs = np.linspace(4.8e9, 5.1e9, 31)
    res = QubitSpectroscopyExperiment.run(
        q,
        freqs=freqs,
        amps=0.04,
        pulse_type=SquarePulse,
        duration=100e-9,
        dt=1e-9,
    )

    assert res.is_2d is False
    # 01 resonance near 5.0 GHz
    assert np.isclose(res.f01, 5.0e9, atol=20e6)
    # P2 should remain negligible (< 0.02) everywhere under weak drive
    assert np.max(res.p2_vals) < 0.02


def test_spectroscopy_strong_drive_two_photon():
    # 5 GHz transmon, alpha = -250 MHz
    # Two-photon transition f_02/2 = 5.0 - 0.125 = 4.875 GHz
    q = Transmon("q0", f_q=5.0e9, alpha=-250e6, levels=4)
    freqs = np.linspace(4.8e9, 5.1e9, 41)

    # Strong drive: amps = 0.5
    res = QubitSpectroscopyExperiment.run(
        q,
        freqs=freqs,
        amps=0.5,
        pulse_type=SquarePulse,
        duration=100e-9,
        dt=1e-9,
    )

    assert res.is_2d is False
    # Two-photon peak at 4.875 GHz should be detected with significant P2
    f_02_half_expected = 4.875e9
    idx_02 = np.argmin(np.abs(freqs - f_02_half_expected))
    assert res.p2_vals[idx_02] > 0.4
    assert res.has_two_photon is True

    # Measured alpha should match -250 MHz closely
    if res.alpha_measured is not None:
        assert np.isclose(res.alpha_measured, -250e6, atol=30e6)


def test_spectroscopy_1d_and_plot():
    q = Transmon("q0", f_q=5.0e9, alpha=-250e6, levels=4)
    freqs = np.linspace(4.85e9, 5.15e9, 15)

    res = QubitSpectroscopyExperiment.run(
        q,
        freqs=freqs,
        amps=0.2,
        pulse_type=GaussianPulse,
        duration=60e-9,
        dt=1e-9,
    )
    assert len(res.freqs) == 15
    assert res.is_2d is False
    assert res.pulse_type == GaussianPulse

    # Verify plot executes cleanly
    import matplotlib
    matplotlib.use("Agg")
    ax = res.plot()
    assert ax is not None


def test_power_spectroscopy_2d():
    q = Transmon("q0", f_q=5.0e9, alpha=-250e6, levels=4)
    freqs = np.linspace(4.82e9, 5.08e9, 15)
    amps = np.linspace(0.05, 0.5, 5)

    pwr_res = QubitSpectroscopyExperiment.run(
        q,
        freqs=freqs,
        amps=amps,
        pulse_type=SquarePulse,
        duration=80e-9,
        dt=1e-9,
    )

    assert pwr_res.is_2d is True
    assert pwr_res.sweep_param == "amp"
    assert pwr_res.p_exc_grid.shape == (5, 15)
    assert pwr_res.p1_grid.shape == (5, 15)
    assert pwr_res.p2_grid.shape == (5, 15)

    # At lowest amp, P2 is near zero
    assert np.max(pwr_res.p2_grid[0, :]) < 0.05
    # At highest amp, P2 is noticeably excited
    assert np.max(pwr_res.p2_grid[-1, :]) > 0.2

    # Verify plot executes cleanly
    import matplotlib
    matplotlib.use("Agg")
    ax = pwr_res.plot(observable="p2")
    assert ax is not None

    ax1, ax2 = pwr_res.plot(observable="both")
    assert ax1 is not None and ax2 is not None


def test_flux_spectroscopy_2d():
    q = Transmon("q0", f_q=5.0e9, alpha=-250e6, d=0.25, levels=4)
    freqs = np.linspace(4.4e9, 5.1e9, 21)
    flux_vals = np.linspace(-0.25, 0.25, 5)

    flux_res = QubitSpectroscopyExperiment.run(
        q,
        freqs=freqs,
        amps=0.04,
        flux=flux_vals,
        duration=80e-9,
        dt=1e-9,
    )

    assert flux_res.is_2d is True
    assert flux_res.sweep_param == "flux"
    assert flux_res.p_exc_grid.shape == (5, 21)
    assert flux_res.p1_grid.shape == (5, 21)

    # Verify plot executes cleanly with flux theory arc
    import matplotlib
    matplotlib.use("Agg")
    ax = flux_res.plot(observable="p1")
    assert ax is not None

    ax1, ax2 = flux_res.plot(observable="both")
    assert ax1 is not None and ax2 is not None


def test_spectroscopy_mutually_exclusive_2d():
    q = Transmon("q0", f_q=5.0e9, alpha=-250e6, levels=3)
    freqs = np.linspace(4.9e9, 5.1e9, 11)
    with pytest.raises(ValueError, match="Cannot sweep both"):
        QubitSpectroscopyExperiment.run(
            q,
            freqs=freqs,
            amps=np.linspace(0.1, 0.5, 3),
            flux=np.linspace(-0.1, 0.1, 3),
        )


def test_spectroscopy_levels_warning():
    # When levels < 3, warning should be triggered
    q_2lvl = Transmon("q_2lvl", f_q=5.0e9, levels=2)
    with pytest.warns(UserWarning, match="levels=2"):
        QubitSpectroscopyExperiment.run(
            q_2lvl,
            freqs=np.linspace(4.9e9, 5.1e9, 7),
            dt=1e-9,
        )


def test_rabi_with_dispersive_backend():
    # Test that RabiExperiment can execute with backend='dispersive'
    q = Transmon("q_exp_disp", f_q=5.0e9, levels=2, omega_d=50e6)
    rabi_res = RabiExperiment.amplitude_rabi(
        q,
        pulse_type=SquarePulse,
        duration=20e-9,
        amps=np.linspace(0.0, 1.0, 11),
        backend="dispersive",
        backend_kwargs=dict(shots=500, snr_db=15.0, seed=42),
    )
    assert len(rabi_res.p1_vals) == 11
    assert rabi_res.amp_pi > 0.0
    # Values should fluctuate between 0 and 1
    assert np.min(rabi_res.p1_vals) < 0.2
    assert np.max(rabi_res.p1_vals) > 0.8


