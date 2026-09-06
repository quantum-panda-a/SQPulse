from sqpulse import Transmon, GaussianPulse, PulseSequence, Simulator


def main():
    print("SQPulse (SI Units) is ready.")
    q = Transmon("q0", f_q=5.0e9, alpha=-250.0e6, levels=3)
    print(f"Initialized system: {q}")

    seq = PulseSequence().add(q.drive, GaussianPulse(duration=40e-9, amp=1.0e8))
    res = Simulator.run(q, seq)
    print(f"Test simulation complete. Final P1 = {res.final_population(1):.4f}")


if __name__ == "__main__":
    main()
