"""Dispersive cavity readout measurement backend for SQPulse in SI units."""

from __future__ import annotations
from typing import Optional, Dict, Tuple, Any, List
import numpy as np
import matplotlib.pyplot as plt

from ...models.transmon import Transmon
from ...models.resonator import ReadoutResonator
from ...sequence.sequence import PulseSequence
from ...sequence.channel import normalize_channel
from ...pulses.base import Pulse
from ...pulses.shapes import FlatTopPulse
from ..base import MeasurementBackend, BaseMeasurementResult
from ..projective import ProjectiveBackend, ProjectiveResult
from .cavity import simulate_cavity_dynamics
from .demodulation import demodulate_and_integrate
from .noise import add_readout_noise
from .discrimination import IQDiscriminator


class DispersiveResult(BaseMeasurementResult):
    """Result of circuit QED dispersive readout simulation.

    Args:
        transmon: Physical Transmon model.
        sequence: The pulse sequence simulated.
        resonator: ReadoutResonator used for measurement.
        times: Readout time timestamps.
        cavity_trajectories: Dict of cavity amplitudes alpha_n(t) for states 0 and 1.
        iq_centers: Noiseless IQ centroids for states 0 and 1.
        shots_iq: 1D complex array of all single-shot measured IQ values.
        shots_labels: 1D integer array of true states for each shot.
        shots_assigned: 1D integer array of discriminator-assigned states.
        confusion_matrix: 2x2 confusion matrix.
        fidelity: Readout assignment fidelity.
        discriminator: Fitted IQDiscriminator instance.
        projective_result: Underlying ProjectiveResult from sequence evolution.
    """

    def __init__(
        self,
        transmon: Transmon,
        sequence: PulseSequence,
        resonator: ReadoutResonator,
        times: np.ndarray,
        cavity_trajectories: Dict[int, np.ndarray],
        iq_centers: Dict[int, complex],
        shots_iq: np.ndarray,
        shots_labels: np.ndarray,
        shots_assigned: np.ndarray,
        confusion_matrix: np.ndarray,
        fidelity: float,
        discriminator: IQDiscriminator,
        projective_result: Optional[ProjectiveResult] = None,
    ):
        super().__init__(transmon=transmon, sequence=sequence)
        self.resonator = resonator
        self.times = times
        self.cavity_trajectories = cavity_trajectories
        self.iq_centers = iq_centers
        self.shots_iq = shots_iq
        self.shots_labels = shots_labels
        self.shots_assigned = shots_assigned
        self.confusion_matrix = confusion_matrix
        self.fidelity = fidelity
        self.discriminator = discriminator
        self.projective_result = projective_result

    def counts(self) -> Dict[int, int]:
        """Return counts of discriminated measurement outcomes."""
        unique, counts = np.unique(self.shots_assigned, return_counts=True)
        return {int(u): int(c) for u, c in zip(unique, counts)}

    def plot_iq_plane(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (6, 6),
        max_points: int = 1500,
        title: Optional[str] = None,
    ) -> plt.Axes:
        """Plot the single-shot measurements in the complex IQ plane."""
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        n_pts = min(len(self.shots_iq), max_points)
        idx = np.random.choice(len(self.shots_iq), size=n_pts, replace=False) if len(self.shots_iq) > n_pts else np.arange(len(self.shots_iq))

        pts = self.shots_iq[idx]
        lbls = self.shots_labels[idx]

        mask0 = lbls == 0
        mask1 = lbls == 1

        ax.scatter(np.real(pts[mask0]), np.imag(pts[mask0]), alpha=0.35, color="#1f77b4", s=18, label="State |0⟩ shots")
        ax.scatter(np.real(pts[mask1]), np.imag(pts[mask1]), alpha=0.35, color="#d62728", s=18, label="State |1⟩ shots")

        # Plot centroids
        c0 = self.iq_centers[0]
        c1 = self.iq_centers[1]
        ax.scatter([np.real(c0)], [np.imag(c0)], color="#084081", edgecolor="black", s=100, marker="o", label="Centroid |0⟩", zorder=5)
        ax.scatter([np.real(c1)], [np.imag(c1)], color="#67000d", edgecolor="black", s=100, marker="s", label="Centroid |1⟩", zorder=5)

        # Plot decision boundary (perpendicular bisector)
        mid = self.discriminator.midpoint
        unit = self.discriminator.unit_vector
        perp = -1j * unit
        span = 1.5 * abs(c1 - c0)
        line_pts = mid + perp * np.linspace(-span, span, 100)
        ax.plot(np.real(line_pts), np.imag(line_pts), "--", color="black", lw=1.5, label="Decision Threshold", zorder=4)

        ax.set_xlabel("In-Phase (I)")
        ax.set_ylabel("Quadrature (Q)")
        ax.set_title(title or f"Dispersive Readout IQ Plane (Fidelity = {self.fidelity:.2%})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        ax.axis("equal")
        return ax

    def plot_trajectories(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (8, 4),
        title: Optional[str] = None,
    ) -> plt.Axes:
        """Plot the intra-cavity photon number n_cav(t) = |alpha(t)|^2 over time."""
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        t_ns = self.times * 1e9
        for st in (0, 1):
            if st in self.cavity_trajectories:
                alpha = self.cavity_trajectories[st]
                n_photons = np.abs(alpha) ** 2
                color = "#1f77b4" if st == 0 else "#d62728"
                ax.plot(t_ns, n_photons, label=f"|{st}⟩ Photon Number", color=color, lw=2)

        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Mean Intra-Cavity Photon Number ⟨n_cav⟩")
        ax.set_title(title or f"{self.resonator.name} - Readout Photon Dynamics (Ring-up / Ring-down)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        return ax


class DispersiveReadoutBackend(MeasurementBackend):
    """Dispersive cavity readout backend for circuit QED measurement simulation."""

    @property
    def name(self) -> str:
        return "dispersive"

    def run(
        self,
        transmon: Transmon,
        sequence: PulseSequence,
        resonator: Optional[ReadoutResonator] = None,
        readout_pulse: Optional[Pulse] = None,
        shots: int = 1000,
        snr_db: float = 12.0,
        f_ro: Optional[float] = None,
        dt: float = 1e-9,
        seed: Optional[int] = None,
        **kwargs,
    ) -> DispersiveResult:
        """Simulate dispersive measurement on a Transmon driven by a PulseSequence.

        Args:
            transmon: Physical Transmon model.
            sequence: Control pulse sequence executed before readout.
            resonator: Physical ReadoutResonator. If None, uses default 7 GHz cavity.
            readout_pulse: Pulse applied to readout resonator. If None, checks sequence.readout_line or uses default FlatTop.
            shots: Number of measurement shots to sample (default 1000).
            snr_db: Readout signal-to-noise ratio in dB (default 12.0 dB).
            f_ro: Readout carrier frequency in Hz (defaults to resonator bare frequency f_r).
            dt: Cavity ODE simulation time step in seconds (default 1e-9 s = 1 ns).
            seed: Optional seed for reproducible noise and shot sampling.
            **kwargs: Extra parameters passed to ProjectiveBackend for pre-readout evolution.

        Returns:
            DispersiveResult with IQ plane data, confusion matrix, and cavity trajectories.
        """
        if resonator is None:
            resonator = ReadoutResonator(
                name=f"{transmon.name}_res",
                f_r=7.0e9,
                kappa=2.0 * np.pi * 2.5e6,
                chi=2.0 * np.pi * 1.2e6,
            )

        if f_ro is None:
            f_ro = resonator.f_r

        # 1. Run sequence dynamics with ProjectiveBackend to determine pre-measurement qubit state
        proj_backend = ProjectiveBackend()
        proj_res = proj_backend.run(transmon=transmon, sequence=sequence, **kwargs)

        p0 = proj_res.final_population(0)
        p1 = proj_res.final_population(1)
        # Normalize in {|0>, |1>} subspace
        norm = max(1e-12, p0 + p1)
        prob_0 = p0 / norm
        prob_1 = p1 / norm

        # 2. Determine readout pulse
        if readout_pulse is None:
            ro_keys = [
                normalize_channel(transmon.readout_line),
                f"{transmon.name}.readout",
                f"{transmon.name}.ro",
            ]
            for rk in ro_keys:
                if rk in sequence._channels and sequence._channels[rk]:
                    readout_pulse = sequence._channels[rk][-1].pulse
                    break

        if readout_pulse is None:
            readout_pulse = FlatTopPulse(duration=1000e-9, ramp_time=20e-9, amp=1.0)

        # 3. Simulate cavity response for state |0> and |1>
        times_0, alpha_0 = simulate_cavity_dynamics(resonator, qubit_state=0, readout_pulse=readout_pulse, f_ro=f_ro, dt=dt)
        times_1, alpha_1 = simulate_cavity_dynamics(resonator, qubit_state=1, readout_pulse=readout_pulse, f_ro=f_ro, dt=dt)

        # 4. Demodulate cavity transmission to find noiseless IQ centroids S_0 and S_1
        c0 = demodulate_and_integrate(times_0, alpha_0, kappa_ext=resonator.kappa_ext)
        c1 = demodulate_and_integrate(times_1, alpha_1, kappa_ext=resonator.kappa_ext)
        iq_centers = {0: c0, 1: c1}

        # 5. Fit discriminator
        discriminator = IQDiscriminator(center_0=c0, center_1=c1)

        # 6. Sample single shots based on qubit populations
        rng = np.random.default_rng(seed)
        shots_labels = rng.choice([0, 1], size=shots, p=[prob_0, prob_1])

        # Generate noisy IQ points for sampled shots
        sep = abs(c1 - c0)
        clean_iq = np.array([iq_centers[lbl] for lbl in shots_labels], dtype=complex)
        noisy_iq = add_readout_noise(clean_iq, snr_db=snr_db, signal_separation=sep, seed=seed)

        # Discriminate
        shots_assigned = discriminator.predict(noisy_iq)

        # Compute confusion matrix and fidelity
        # Calibration shots (equal mix of |0> and |1>) to compute intrinsic confusion matrix
        cal_labels = np.array([0] * (shots // 2) + [1] * (shots // 2))
        cal_clean = np.array([iq_centers[lbl] for lbl in cal_labels], dtype=complex)
        seed_cal = (seed + 1000) if seed is not None else None
        cal_noisy = add_readout_noise(cal_clean, snr_db=snr_db, signal_separation=sep, seed=seed_cal)
        cal_assigned = discriminator.predict(cal_noisy)

        c_mat = IQDiscriminator.compute_confusion_matrix(cal_labels, cal_assigned)
        fid = IQDiscriminator.compute_fidelity(c_mat)

        return DispersiveResult(
            transmon=transmon,
            sequence=sequence,
            resonator=resonator,
            times=times_0,
            cavity_trajectories={0: alpha_0, 1: alpha_1},
            iq_centers=iq_centers,
            shots_iq=noisy_iq,
            shots_labels=shots_labels,
            shots_assigned=shots_assigned,
            confusion_matrix=c_mat,
            fidelity=fid,
            discriminator=discriminator,
            projective_result=proj_res,
        )
