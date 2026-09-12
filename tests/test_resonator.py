"""Tests for ReadoutResonator in SQPulse."""

import numpy as np
import pytest
from sqpulse.models import ReadoutResonator


def test_readout_resonator_properties():
    f_r = 7.0e9
    kappa = 2.0 * np.pi * 2.5e6
    chi = 2.0 * np.pi * 1.2e6

    res = ReadoutResonator("r0", f_r=f_r, kappa=kappa, chi=chi)
    assert res.f_r == f_r
    assert np.isclose(res.kappa_hz, 2.5e6)
    assert np.isclose(res.chi_hz, 1.2e6)
    assert np.isclose(res.kappa_ext, 0.5 * kappa)


def test_readout_resonator_dispersive_shift():
    f_r = 7.0e9
    chi_hz = 1.2e6
    res = ReadoutResonator("r0", f_r=f_r, chi=2.0 * np.pi * chi_hz)

    f0 = res.effective_frequency(qubit_state=0)
    f1 = res.effective_frequency(qubit_state=1)

    assert np.isclose(f0, f_r - chi_hz)
    assert np.isclose(f1, f_r + chi_hz)
    # The separation between |0> and |1> resonance frequency should be exactly 2*chi
    assert np.isclose(f1 - f0, 2.0 * chi_hz)


def test_readout_resonator_s21_transmission():
    f_r = 7.0e9
    res = ReadoutResonator("r0", f_r=f_r)

    f0 = res.effective_frequency(0)
    f1 = res.effective_frequency(1)

    # At resonance of state 0, transmission for state 0 should reach maximum amplitude (|S21| = 1.0 for symmetric port)
    s21_0_at_f0 = res.s21(np.array([f0]), qubit_state=0)[0]
    assert np.isclose(abs(s21_0_at_f0), 1.0)

    # Transmission for state 1 at f0 should be significantly lower
    s21_1_at_f0 = res.s21(np.array([f0]), qubit_state=1)[0]
    assert abs(s21_1_at_f0) < abs(s21_0_at_f0)
