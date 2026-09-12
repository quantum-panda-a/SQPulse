"""Tests for sqpulse.models.Transmon in SI units."""

import numpy as np
import pytest
import qutip
from sqpulse.models import Transmon


def test_transmon_operators():
    q = Transmon("q0", f_q=5.0e9, alpha=-250e6, levels=4)

    # Fock basis states
    assert q.fock(0).shape == (4, 1)
    assert q.ground_state() == q.fock(0)

    # Commutator [a, a^dag] for |0> should give 1
    comm = q.a * q.ad - q.ad * q.a
    zero = q.fock(0)
    exp_comm = qutip.expect(comm, zero)
    assert np.isclose(exp_comm, 1.0)


def test_transmon_hamiltonian_eigenvalues():
    f_q = 5.0e9  # 5 GHz in Hz
    alpha = -250.0e6  # -250 MHz in Hz
    q = Transmon("q0", f_q=f_q, alpha=alpha, levels=3)

    # In resonant rotating frame (f_d = f_q):
    # <0|H0|0> = 0
    # <1|H0|1> = 0
    # <2|H0|2> = 2*pi * alpha (in rad/s)
    H0 = q.H0(f_d=f_q)

    e0 = qutip.expect(H0, q.fock(0))
    e1 = qutip.expect(H0, q.fock(1))
    e2 = qutip.expect(H0, q.fock(2))

    assert np.isclose(e0, 0.0, atol=1e-7)
    assert np.isclose(e1, 0.0, atol=1e-7)
    assert np.isclose(e2, 2.0 * np.pi * alpha, atol=1e-7)


def test_transmon_collapse_operators():
    q_ideal = Transmon("q0", f_q=5.0e9, levels=3)
    assert len(q_ideal.c_ops()) == 0

    q_lossy = Transmon("q0", f_q=5.0e9, levels=3, t1=10e-6, t2=8e-6)
    c_ops = q_lossy.c_ops()
    assert len(c_ops) == 2  # decay + dephasing


def test_transmon_omega_d():
    # Default omega_d should be 2*pi * 50 MHz
    q = Transmon("q0", f_q=5.0e9)
    assert np.isclose(q.omega_d, 2.0 * np.pi * 50.0e6)

    # Custom omega_d
    custom_omega = 2.0 * np.pi * 30.0e6
    q_custom = Transmon("q1", omega_d=custom_omega)
    assert np.isclose(q_custom.omega_d, custom_omega)


def test_transmon_from_circuit():
    c_d = 5.0e-17   # 0.05 fF
    c_g = 70.0e-15  # 70 fF
    f_q = 5.0e9     # 5 GHz
    q = Transmon.from_circuit(
        name="q_circ",
        c_d=c_d,
        c_g=c_g,
        f_q=f_q,
        attenuation_dB=-60.0,
        v_max=1.0,
    )
    assert q.c_d == c_d
    assert q.c_g == c_g
    assert q.c_sigma == c_d + c_g
    assert q.q_zpf > 0
    assert q.omega_chip > 1e12  # On-chip Omega is in ~1e13 rad/(s*V)
    # After -60 dB (1e-3 factor), effective omega_d is around ~1e9 rad/s
    assert 1e8 < q.omega_d < 1e10


def test_transmon_control_lines():
    q = Transmon("q0")
    assert q.xy.name == "q0.xy"
    assert q.z.name == "q0.z"
    assert q.ro.name == "q0.ro"
    # Ensure legacy attributes are completely removed
    assert not hasattr(q, "charge_line")
    assert not hasattr(q, "flux_line")
    assert not hasattr(q, "readout_line")
    assert not hasattr(q, "drive")


def test_transmon_single_junction_flux_insensitivity():
    q = Transmon("q_single", f_q=5.0e9, alpha=-250e6, d=1.0)
    assert q.d == 1.0
    for phi in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        assert np.isclose(q.frequency_at_flux(phi), 5.0e9)
        assert np.isclose(q.flux_sensitivity(phi), 0.0)


def test_transmon_squid_flux_tuning():
    f_q_max = 5.0e9
    alpha = -250e6
    d = 0.2
    q = Transmon("q_squid", f_q=f_q_max, alpha=alpha, d=d)

    # 1. Sweet spot (Phi = 0): maximum frequency and zero first-order sensitivity
    assert np.isclose(q.frequency_at_flux(0.0), f_q_max)
    assert np.isclose(q.flux_sensitivity(0.0), 0.0, atol=1e-5)

    # 2. Minimum frequency at half-integer flux (Phi = 0.5)
    ec = abs(alpha)
    expected_min_fq = (f_q_max + ec) * np.sqrt(d) - ec
    assert np.isclose(q.frequency_at_flux(0.5), expected_min_fq)
    assert np.isclose(q.flux_sensitivity(0.5), 0.0, atol=1e-5)

    # 3. Monotonic frequency decrease from 0 to 0.5
    f_prev = f_q_max
    for phi in np.linspace(0.05, 0.5, 10):
        f_curr = q.frequency_at_flux(phi)
        assert f_curr < f_prev
        f_prev = f_curr


