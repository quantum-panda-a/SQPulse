"""Demonstration of multi-qubit composite systems and pulse-level two-qubit gates in SQPulse."""

import numpy as np
import matplotlib.pyplot as plt

from sqpulse import (
    Transmon,
    QuantumSystem,
    FluxISWAP,
    FluxCZ,
    flux_iswap_sequence,
    flux_cz_sequence,
    evaluate_two_qubit_gate,
    Simulator,
    ns,
    MHz,
    GHz,
)


def main():
    print("=" * 70)
    print("SQPulse: Multi-Qubit Composite System & Pulse-Level Two-Qubit Gates")
    print("=" * 70)

    # 1. Construct physical qubits
    # Q0: SQUID flux-tunable transmon, maximum frequency 5.0 GHz at sweet spot
    q0 = Transmon(
        name="q0",
        f_q=5.0 * GHz,
        alpha=-250.0 * MHz,
        d=0.3,              # SQUID junction asymmetry parameter (tunable)
        flux_offset=0.0,
        t1=35.0e-6,
        t2=25.0e-6,
        levels=3,
    )

    # Q1: Single-junction fixed-frequency transmon at 4.6 GHz
    q1 = Transmon(
        name="q1",
        f_q=4.6 * GHz,
        alpha=-240.0 * MHz,
        d=1.0,              # Single junction (fixed frequency)
        t1=40.0e-6,
        t2=30.0e-6,
        levels=3,
    )

    # 2. Build QuantumSystem composite container with capacitive coupling g = 15 MHz
    sys = QuantumSystem([q0, q1], name="coupled_pair")
    sys.add_capacitive_coupling(q0, q1, g=15.0 * MHz)

    print(f"\nSystem initialized: {sys}")
    print(f"Modes: {sys.mode_names}, Truncated levels: {sys.levels}, Hilbert dimension: {sys.hilbert_dimension}")
    print(f"Couplings: {sys.coupling_terms}")

    # ---------------------------------------------------------
    # Part A: Pulse-level Flux iSWAP Gate
    # ---------------------------------------------------------
    print("\n" + "-" * 50)
    print("Part A: Pulse-level Resonant Flux iSWAP Gate")
    print("-" * 50)

    # Automatically generate flat-top flux pulse sequence on q0.z
    iswap_seq = flux_iswap_sequence(
        sys,
        q_tunable="q0",
        q_target="q1",
        ramp_time=2.0 * ns,
    )
    print(f"Generated iSWAP Sequence duration: {iswap_seq.duration / ns:.2f} ns")

    # Start simulation from state |10>
    init_state = sys.fock(1, 0)
    res_iswap = Simulator.run(target=sys, sequence=iswap_seq, init_state=init_state, dt=1e-10)

    print(f"Initial population in |10>: {res_iswap.population('10')[0]:.4f}")
    print(f"Final population in |10>:   {res_iswap.final_population('10'):.4f}")
    print(f"Final population in |01>:   {res_iswap.final_population('01'):.4f} (excitation swapped!)")

    # Reconstruct 4x4 matrix and compute fidelity
    target_iswap = np.array([
        [1, 0, 0, 0],
        [0, 0, -1j, 0],
        [0, -1j, 0, 0],
        [0, 0, 0, 1]
    ], dtype=complex)

    gate_result = evaluate_two_qubit_gate(
        system=sys,
        sequence=iswap_seq,
        gate_name="FluxISWAP",
        target_unitary=target_iswap,
        dt=2e-10,
    )
    print(f"Average Gate Fidelity F_avg: {gate_result.average_gate_fidelity:.2%}")
    print(f"Subspace Leakage:            {gate_result.leakage:.2%}")

    # ---------------------------------------------------------
    # Part B: Pulse-level Controlled-Z (CZ) Gate
    # ---------------------------------------------------------
    print("\n" + "-" * 50)
    print("Part B: Pulse-level Controlled-Z (CZ) Gate (|11> <-> |02> avoided crossing)")
    print("-" * 50)

    res_cz_flux = FluxCZ.calculate_cz_flux(q0, q1)
    cz_target_freq = q0.frequency_at_flux(res_cz_flux)
    print(f"CZ resonance condition: f_q0 = f_q1 + alpha1 = {cz_target_freq / GHz:.3f} GHz")
    print(f"Calculated flux bias:   {res_cz_flux:.4f} Phi_0")

    cz_seq = flux_cz_sequence(
        sys,
        q_tunable="q0",
        q_target="q1",
        ramp_time=2.0 * ns,
    )
    print(f"Generated CZ Sequence duration: {cz_seq.duration / ns:.2f} ns")

    # Simulate with initial state |11>
    init_state_11 = sys.fock(1, 1)
    res_cz = Simulator.run(target=sys, sequence=cz_seq, init_state=init_state_11, dt=1e-10)

    print(f"Initial population in |11>: {res_cz.population('11')[0]:.4f}")
    print(f"Final population in |11>:   {res_cz.final_population('11'):.4f}")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
