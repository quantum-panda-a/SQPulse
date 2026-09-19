"""Tests for ReadoutResonator in SQPulse."""

import numpy as np
import pytest
from sqpulse.models import ReadoutResonator


def test_readout_resonator_properties():
    f_r = 7.0e9
    kappa = 2.5e6
    chi = 1.2e6

    res = ReadoutResonator("r0", f_r=f_r, kappa=kappa, chi=chi)
    assert res.f_r == f_r
    assert np.isclose(res.kappa, 2.0 * np.pi * kappa)
    assert np.isclose(res.chi, 2.0 * np.pi * chi)
    assert np.isclose(res.kappa_hz, 2.5e6)
    assert np.isclose(res.chi_hz, 1.2e6)
    assert np.isclose(res.kappa_ext, 0.5 * (2.0 * np.pi * kappa))
    assert np.isclose(res.kappa_ext_hz, 0.5 * kappa)
    # Default is hanger (dip)
    assert res.geometry == "hanger"
    assert res.structure_type == "hanger"
    assert res.is_dip is True
    assert res.is_peak is False


def test_readout_resonator_dispersive_shift():
    f_r = 7.0e9
    chi_hz = 1.2e6
    res = ReadoutResonator("r0", f_r=f_r, chi=chi_hz)

    f0 = res.effective_frequency(qubit_state=0)
    f1 = res.effective_frequency(qubit_state=1)

    assert np.isclose(f0, f_r - chi_hz)
    assert np.isclose(f1, f_r + chi_hz)
    # The separation between |0> and |1> resonance frequency should be exactly 2*chi
    assert np.isclose(f1 - f0, 2.0 * chi_hz)


def test_readout_resonator_s21_hanger_dip():
    """Test default hanger mode produces a transmission dip at resonance."""
    f_r = 7.0e9
    # Default geometry is 'hanger' (dip)
    res = ReadoutResonator("r0", f_r=f_r, kappa=2.0e6, kappa_ext=1.0e6)

    f0 = res.effective_frequency(0)
    # At resonance, S21 = 1 - (kappa_ext / kappa) = 1 - 0.5 = 0.5 (Dip: |S21| < 1)
    s21_on_res = res.s21(np.array([f0]), qubit_state=0)[0]
    assert np.isclose(abs(s21_on_res), 0.5)

    # Far off resonance, S21 -> 1.0 (0 dB transmission line baseline)
    f_far = f0 + 200e6
    s21_far = res.s21(np.array([f_far]), qubit_state=0)[0]
    assert np.isclose(abs(s21_far), 1.0, atol=1e-3)

    # Alias 'dip' also works
    res_dip = ReadoutResonator("r_dip", f_r=f_r, geometry="dip")
    assert res_dip.geometry == "hanger"
    assert res_dip.is_dip is True


def test_readout_resonator_s21_transmission_peak():
    """Test transmission mode produces a transmission peak at resonance."""
    f_r = 7.0e9
    res = ReadoutResonator("r0", f_r=f_r, geometry="transmission")
    assert res.geometry == "transmission"
    assert res.is_peak is True
    assert res.is_dip is False

    f0 = res.effective_frequency(0)
    # At resonance of state 0, transmission for state 0 should reach maximum amplitude (|S21| = 1.0 for symmetric port)
    s21_0_at_f0 = res.s21(np.array([f0]), qubit_state=0)[0]
    assert np.isclose(abs(s21_0_at_f0), 1.0)

    # Transmission for state 1 at f0 should be significantly lower
    s21_1_at_f0 = res.s21(np.array([f0]), qubit_state=1)[0]
    assert abs(s21_1_at_f0) < abs(s21_0_at_f0)

    # Far off resonance, transmission -> 0.0
    f_far = f0 + 200e6
    s21_far = res.s21(np.array([f_far]), qubit_state=0)[0]
    assert np.isclose(abs(s21_far), 0.0, atol=1e-2)

    # Alias 'peak' also works
    res_peak = ReadoutResonator("r_peak", f_r=f_r, structure_type="peak")
    assert res_peak.geometry == "transmission"


def test_readout_resonator_flux_tuning():
    from sqpulse.models import Transmon
    q = Transmon("q_tunable", f_q=5.0e9, alpha=-250e6, d=0.25)
    r = ReadoutResonator("r0", f_r=7.05e9, kappa=2.8e6, g=50e6)

    # Sweet spot (Phi = 0.0)
    f_eff_sweet = r.effective_frequency_at_flux(q, flux=0.0)
    # Half flux (Phi = 0.5)
    f_eff_half = r.effective_frequency_at_flux(q, flux=0.5)

    # Detuning is smaller at sweet spot, so cavity frequency pull is larger
    assert f_eff_sweet > f_eff_half

    # 2D S21 vs flux
    freqs = np.linspace(7.047e9, 7.053e9, 21)
    fluxes = np.linspace(-0.5, 0.5, 11)
    s21_grid = r.s21_vs_flux(freqs, q, fluxes)
    assert s21_grid.shape == (21, 11)

