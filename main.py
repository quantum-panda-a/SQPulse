from sqpulse import Transmon, GaussianPulse, PulseSequence, Simulator


def main():
    print("SQPulse is ready.")
    q = Transmon("q0", f_q=5.0, alpha=-0.25, levels=3)
    print(f"Initialized system: {q}")

    seq = PulseSequence().add(q.drive, GaussianPulse(duration=40.0, amp=0.1))
    res = Simulator.run(q, seq)
    print(f"Test simulation complete. Final P1 = {res.final_population(1):.4f}")


if __name__ == "__main__":
    main()
