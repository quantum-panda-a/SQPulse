"""Unit tests for QuantumSystem and CouplingTerm in SQPulse."""

import numpy as np
import pytest
import qutip

from sqpulse import (
    Transmon,
    QuantumSystem,
    CouplingTerm,
    PulseSequence,
    SquarePulse,
    ns,
    MHz,
    GHz,
)


def test_quantum_system_creation():
    q0 = Transmon("q0", f_q=5.0 * GHz, alpha=-250 * MHz, levels=3)
    q1 = Transmon("q1", f_q=4.6 * GHz, alpha=-240 * MHz, levels=3)
    sys = QuantumSystem([q0, q1], name="test_2q_system")

    assert sys.num_modes == 2
    assert sys.mode_names == ["q0", "q1"]
    assert sys.levels == [3, 3]
    assert sys.hilbert_dimension == 9
    assert sys["q0"] is q0
    assert sys["q1"] is q1
    assert sys.q0 is q0
    assert sys.q1 is q1
    assert "q0" in sys
    assert "q2" not in sys


def test_duplicate_mode_error():
    q0a = Transmon("q0", f_q=5.0 * GHz)
    q0b = Transmon("q0", f_q=4.8 * GHz)
    with pytest.raises(ValueError, match="Duplicate mode name"):
        QuantumSystem([q0a, q0b])


def test_tensor_with_i():
    q0 = Transmon("q0", f_q=5.0 * GHz, levels=3)
    q1 = Transmon("q1", f_q=4.6 * GHz, levels=4)
    sys = QuantumSystem([q0, q1])

    # Embed local a on q0
    a0_full = sys.tensor_with_I(q0, q0.a)
    assert a0_full.dims == [[3, 4], [3, 4]]

    # Embed local a on q1
    a1_full = sys.tensor_with_I(q1, q1.a)
    assert a1_full.dims == [[3, 4], [3, 4]]

    # Check commutation of operators on different modes: [a0, a1_dag] == 0
    comm = a0_full * a1_full.dag() - a1_full.dag() * a0_full
    assert comm.norm() < 1e-12

    # Check commutation on same mode: [a0, a0_dag] on {|0>, |1>} subspace is identity
    comm0 = a0_full * a0_full.dag() - a0_full.dag() * a0_full
    ground = sys.ground_state()
    # Expectation of [a, a^dag] in vacuum is 1
    assert np.isclose(qutip.expect(comm0, ground), 1.0)


def test_fock_basis_construction():
    q0 = Transmon("q0", f_q=5.0 * GHz, levels=3)
    q1 = Transmon("q1", f_q=4.6 * GHz, levels=3)
    sys = QuantumSystem([q0, q1])

    # Positional
    ket_01 = sys.fock(0, 1)
    assert ket_01.shape == (9, 1)
    assert ket_01.dims == [[3, 3], [1]]

    # Keyword
    ket_10 = sys.fock(q0=1, q1=0)
    assert ket_10.shape == (9, 1)
    assert ket_10.dims == [[3, 3], [1]]

    # Orthogonality
    val_ortho = ket_01.dag() * ket_10
    val_norm = ket_01.dag() * ket_01
    assert np.isclose(complex(val_ortho.full()[0, 0]) if hasattr(val_ortho, "full") else complex(val_ortho), 0.0)
    assert np.isclose(complex(val_norm.full()[0, 0]) if hasattr(val_norm, "full") else complex(val_norm), 1.0)

    # Projector
    proj_01 = sys.proj(0, 1)
    assert np.isclose(qutip.expect(proj_01, ket_01), 1.0)
    assert np.isclose(qutip.expect(proj_01, ket_10), 0.0)


def test_coupling_term():
    q0 = Transmon("q0", f_q=5.0 * GHz, levels=3)
    q1 = Transmon("q1", f_q=4.6 * GHz, levels=3)
    sys = QuantumSystem([q0, q1])

    g = 15.0 * MHz
    sys.add_capacitive_coupling(q0, q1, g=g)
    assert len(sys.coupling_terms) == 1

    term = sys.coupling_terms[0]
    h_c = term.to_qobj(sys)
    assert h_c.dims == [[3, 3], [3, 3]]
    # Must be Hermitian
    assert h_c.isherm

    # Matrix element <01| H_c |10> should be 2*pi*g
    ket_01 = sys.fock(0, 1)
    ket_10 = sys.fock(1, 0)
    prod = ket_01.dag() * h_c * ket_10
    element = complex(prod.full()[0, 0]) if hasattr(prod, "full") else complex(prod)
    assert np.isclose(element, 2.0 * np.pi * 15.0e6)

    # Add ZZ coupling
    sys.add_zz_coupling(q0, q1, zeta=100.0 * 1e3)
    assert len(sys.coupling_terms) == 2


def test_system_c_ops():
    q0 = Transmon("q0", f_q=5.0 * GHz, t1=30e-6, t2=25e-6, levels=3)
    q1 = Transmon("q1", f_q=4.6 * GHz, t1=40e-6, t2=30e-6, levels=3)
    sys = QuantumSystem([q0, q1])

    c_ops = sys.c_ops()
    assert len(c_ops) > 0
    for c in c_ops:
        assert c.dims == [[3, 3], [3, 3]]


def test_static_hamiltonian():
    q0 = Transmon("q0", f_q=5.0 * GHz, alpha=-250 * MHz, levels=3)
    q1 = Transmon("q1", f_q=4.6 * GHz, alpha=-240 * MHz, levels=3)
    sys = QuantumSystem([q0, q1])
    sys.add_capacitive_coupling(q0, q1, g=10 * MHz)

    # At individual resonant frame: detunings are zero
    h0 = sys.H0()
    assert h0.dims == [[3, 3], [3, 3]]
    assert h0.isherm
