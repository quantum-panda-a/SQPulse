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
