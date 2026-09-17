"""Unit and integration tests for pulse-level two-qubit gates (FluxISWAP and FluxCZ) in SQPulse."""

import numpy as np
import pytest
import qutip

from sqpulse import (
    Transmon,
    QuantumSystem,
    PulseSequence,
    Simulator,
    FluxISWAP,
    FluxCZ,
    flux_iswap_sequence,
    flux_cz_sequence,
    evaluate_two_qubit_gate,
    ns,
    MHz,
    GHz,
)


@pytest.fixture
def coupled_two_qubit_system():
    """Create a coupled two-qubit system: q0 tunable (5.0 GHz at max), q1 fixed (4.6 GHz)."""
    q0 = Transmon(
        name="q0",
        f_q=5.0 * GHz,
        alpha=-250.0 * MHz,
        d=0.3,             # Tunable SQUID
        flux_offset=0.0,
        levels=3,
    )
    q1 = Transmon(
        name="q1",
        f_q=4.6 * GHz,
        alpha=-240.0 * MHz,
        d=1.0,             # Fixed single junction
        levels=3,
    )
    sys = QuantumSystem([q0, q1], name="test_coupled_sys")
    # Capacitive coupling g = 15 MHz
    sys.add_capacitive_coupling(q0, q1, g=15.0 * MHz)
    return sys


def test_iswap_flux_calculation(coupled_two_qubit_system):
    sys = coupled_two_qubit_system
    q0 = sys.q0
    q1 = sys.q1

    # Find flux to tune q0 to 4.6 GHz
    target_f = q1.f_q
    phi_res = FluxISWAP.calculate_resonance_flux(q0, target_f)
    freq_at_phi = q0.frequency_at_flux(phi_res)

    assert np.isclose(freq_at_phi, target_f, atol=1e3)  # within 1 kHz


def test_cz_flux_calculation(coupled_two_qubit_system):
    sys = coupled_two_qubit_system
    q0 = sys.q0
    q1 = sys.q1

    # Condition: f1(Phi) = f2 + alpha2 = 4.6 GHz - 240 MHz = 4.36 GHz
    expected_f = q1.f_q + q1.alpha
    phi_cz = FluxCZ.calculate_cz_flux(q0, q1)
    freq_at_phi = q0.frequency_at_flux(phi_cz)

    assert np.isclose(freq_at_phi, expected_f, atol=1e3)


def test_flux_iswap_dynamics(coupled_two_qubit_system):
    sys = coupled_two_qubit_system
    # g = 15 MHz -> swap time tau = 1 / (4 * 15 MHz) ~= 16.67 ns
    seq = flux_iswap_sequence(
        sys,
        q_tunable="q0",
        q_target="q1",
        pulse_shape="flattop",
        ramp_time=2.0 * ns,
    )

    # Initial state |10>
    init_state = sys.fock(1, 0)
    res = Simulator.run(target=sys, sequence=seq, init_state=init_state, dt=1e-10)

    # Multi-qubit population query
    p10 = res.population("10")
    p01 = res.population("01")
    p00 = res.population("00")

    # |10> should decrease and transfer to |01>
    assert p10[0] > 0.99
    assert p01[-1] > 0.85  # High excitation transfer into |01>
    assert p00[-1] < 0.05

    # Multi-qubit populations dict
    all_pops = res.populations()
    assert "00" in all_pops
    assert "01" in all_pops
    assert "10" in all_pops
    assert "11" in all_pops

    # Multi-qubit sampling
    shots = res.sample_shots(shots=100)
    assert len(shots) == 100
    assert all(isinstance(s, str) for s in shots)
    counts = res.counts()
    assert "01" in counts


def test_evaluate_two_qubit_gate(coupled_two_qubit_system):
    sys = coupled_two_qubit_system
    seq = flux_iswap_sequence(
        sys,
        q_tunable="q0",
        q_target="q1",
        pulse_shape="flattop",
        ramp_time=2.0 * ns,
    )

    # Ideal iSWAP target unitary
    target_iswap = np.array([
        [1, 0, 0, 0],
        [0, 0, -1j, 0],
        [0, -1j, 0, 0],
        [0, 0, 0, 1]
    ], dtype=complex)

    gate_result = evaluate_two_qubit_gate(
        system=sys,
        sequence=seq,
        gate_name="FluxISWAP",
        target_unitary=target_iswap,
        dt=2e-10,
    )

    assert gate_result.unitary.shape == (4, 4)
    # Check that |00> maps to |00>
    assert np.isclose(np.abs(gate_result.unitary[0, 0]), 1.0, atol=0.05)
    # Check that |10> has strong transition to |01> (i.e. row 1, col 2)
    assert np.abs(gate_result.unitary[1, 2]) > 0.85
    # Check average fidelity is substantial (> 80% without calibration)
    assert gate_result.average_gate_fidelity > 0.80
    assert gate_result.leakage < 0.10


def test_cz_gate_dynamics(coupled_two_qubit_system):
    sys = coupled_two_qubit_system
    seq = flux_cz_sequence(
        sys,
        q_tunable="q0",
        q_target="q1",
        pulse_shape="flattop",
        ramp_time=2.0 * ns,
    )

    # Initial state |11>
    init_state = sys.fock(1, 1)
    res = Simulator.run(target=sys, sequence=seq, init_state=init_state, dt=1e-10)

    # Check that |11> returns largely to |11> with low permanent leakage into |02>
    p11 = res.population("11")
    assert p11[0] > 0.99
    # Final state retention in |11>
    assert p11[-1] > 0.75
