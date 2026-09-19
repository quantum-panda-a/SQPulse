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
        input_fields: Dict of input drive envelopes V_in(t) for states 0 and 1.
        output_fields: Dict of transmitted output fields V_out(t) for states 0 and 1.
        s21_by_state: Dict of dynamic complex S21 transmission coefficients for states 0 and 1.
        s21: Primary dynamic complex S21 transmission coefficient.
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
        input_fields: Optional[Dict[int, np.ndarray]] = None,
        output_fields: Optional[Dict[int, np.ndarray]] = None,
        s21_by_state: Optional[Dict[int, complex]] = None,
        s21: Optional[complex] = None,
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
        self.input_fields = input_fields or {}
        self.output_fields = output_fields or {}
        self.s21_by_state = s21_by_state or {}
        self.s21 = s21

    @property
    def s21_db(self) -> Optional[float]:
        """Dynamic transmission magnitude |S21| in dB."""
        if self.s21 is None:
            return None
        return float(20.0 * np.log10(np.abs(self.s21) + 1e-15))

    @property
    def s21_phase_deg(self) -> Optional[float]:
        """Dynamic transmission phase in degrees."""
        if self.s21 is None:
            return None
        return float(np.degrees(np.angle(self.s21)))

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

    def plot_output_waveforms(
        self,
        state: int = 0,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (8, 4),
        title: Optional[str] = None,
    ) -> plt.Axes:
        """Plot time-domain input drive envelope |V_in(t)| and transmitted output field |V_out(t)|."""
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        t_ns = self.times * 1e9
        if state in self.input_fields:
            v_in = np.abs(self.input_fields[state])
            ax.plot(t_ns, v_in, label=r"Input Drive $|V_{\mathrm{in}}(t)|$", color="#2ca02c", lw=2)

        if state in self.output_fields:
            v_out = np.abs(self.output_fields[state])
            c_color = "#1f77b4" if state == 0 else "#d62728"
            ax.plot(
                t_ns,
                v_out,
                label=rf"Transmitted Output $|V_{{\mathrm{{out}}}}(t)|$ (|{state}⟩)",
                color=c_color,
                lw=2,
            )

        s21_val = self.s21_by_state.get(state, self.s21)
        s21_str = (
            f" ($|S_{{21}}| = {abs(s21_val):.3f}$ / {20*np.log10(abs(s21_val)+1e-15):.2f} dB)"
            if s21_val is not None
            else ""
        )

        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Normalized Field Amplitude")
        ax.set_title(title or f"{self.resonator.name} - Dynamic Readout Fields |{state}⟩{s21_str}")
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
        target: Any = None,
        sequence: Optional[PulseSequence] = None,
        resonator: Optional[ReadoutResonator] = None,
        readout_pulse: Optional[Pulse] = None,
        shots: int = 1000,
        snr_db: float = 12.0,
        f_ro: Optional[float] = None,
        dt: float = 1e-9,
        seed: Optional[int] = None,
        transmon: Any = None,
        g: Optional[float] = None,
        qubit_state: Optional[int] = None,
        integration_window: Optional[Tuple[float, float]] = None,
        **kwargs,
    ) -> DispersiveResult:
        """Simulate dispersive measurement on a Transmon driven by a PulseSequence.

        Args:
            target: Physical Transmon model (or transmon keyword).
            sequence: Control pulse sequence executed before and during readout.
            resonator: Physical ReadoutResonator. If None, uses default 7 GHz cavity.
            readout_pulse: Pulse applied to readout resonator. If None, checks sequence channel transmon.ro or uses default FlatTop.
            shots: Number of measurement shots to sample (default 1000).
            snr_db: Readout signal-to-noise ratio in dB (default 12.0 dB).
            f_ro: Readout carrier frequency in Hz (defaults to resonator bare frequency f_r).
            dt: Cavity ODE simulation time step in seconds (default 1e-9 s = 1 ns).
            seed: Optional seed for reproducible noise and shot sampling.
            transmon: Alias for target for backward compatibility.
            g: Optional transmon-cavity coupling strength in Hz.
            qubit_state: Optional fixed qubit state (|0> or |1>) to bypass pre-readout evolution.
            integration_window: Optional (t_start, t_end) window for dynamic S21 integration.
            **kwargs: Extra parameters passed to ProjectiveBackend for pre-readout evolution.

        Returns:
            DispersiveResult with IQ plane data, dynamic S21, confusion matrix, and cavity trajectories.
        """
        transmon = target if target is not None else transmon
        if transmon is None:
            raise ValueError("Must provide either target or transmon to DispersiveReadoutBackend.run")

        seq = sequence or PulseSequence()

        if resonator is None:
            resonator = ReadoutResonator(
                name=f"{transmon.name}_res",
                f_r=7.0e9,
                kappa=2.5e6,
                chi=1.2e6,
            )

        if f_ro is None:
            f_ro = resonator.f_r

        # 1. Determine pre-measurement qubit state populations
        if qubit_state is not None:
            prob_0 = 1.0 if qubit_state == 0 else 0.0
            prob_1 = 1.0 - prob_0
            proj_res = None
        else:
            proj_backend = ProjectiveBackend()
            proj_res = proj_backend.run(target=transmon, sequence=seq, **kwargs)
            p0 = proj_res.final_population(0)
            p1 = proj_res.final_population(1)
            norm = max(1e-12, p0 + p1)
            prob_0 = p0 / norm
            prob_1 = p1 / norm

        # 2. Determine readout pulse fallback if not in sequence
        if readout_pulse is None:
            ro_key = normalize_channel(transmon.ro)
            if ro_key in seq._channels and seq._channels[ro_key]:
                readout_pulse = seq._channels[ro_key][-1].pulse

        # 3. Simulate cavity response for state |0> and |1>
        times_0, alpha_0, vin_0, vout_0 = simulate_cavity_dynamics(
            resonator=resonator,
            qubit_state=0,
            readout_pulse=readout_pulse,
            f_ro=f_ro,
            dt=dt,
            sequence=seq,
            transmon=transmon,
            g=g,
            return_fields=True,
        )
        times_1, alpha_1, vin_1, vout_1 = simulate_cavity_dynamics(
            resonator=resonator,
            qubit_state=1,
            readout_pulse=readout_pulse,
            f_ro=f_ro,
            dt=dt,
            sequence=seq,
            transmon=transmon,
            g=g,
            return_fields=True,
        )

        # 4. Demodulate cavity transmission to find noiseless IQ centroids S_0 and S_1
        c0 = demodulate_and_integrate(times_0, alpha_0, kappa_ext=resonator.kappa_ext)
        c1 = demodulate_and_integrate(times_1, alpha_1, kappa_ext=resonator.kappa_ext)
        iq_centers = {0: c0, 1: c1}

        # Dynamic S21 extraction
        from .demodulation import extract_dynamic_s21

        s21_0 = extract_dynamic_s21(times_0, vin_0, vout_0, integration_window=integration_window)
        s21_1 = extract_dynamic_s21(times_1, vin_1, vout_1, integration_window=integration_window)
        s21_by_state = {0: s21_0, 1: s21_1}

        if qubit_state is not None:
            primary_s21 = s21_by_state[qubit_state]
        else:
            primary_s21 = s21_0 if prob_0 >= prob_1 else s21_1

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
        cal_labels = np.array([0] * (shots // 2) + [1] * (shots // 2))
        cal_clean = np.array([iq_centers[lbl] for lbl in cal_labels], dtype=complex)
        seed_cal = (seed + 1000) if seed is not None else None
        cal_noisy = add_readout_noise(cal_clean, snr_db=snr_db, signal_separation=sep, seed=seed_cal)
        cal_assigned = discriminator.predict(cal_noisy)

        c_mat = IQDiscriminator.compute_confusion_matrix(cal_labels, cal_assigned)
        fid = IQDiscriminator.compute_fidelity(c_mat)

        return DispersiveResult(
            transmon=transmon,
            sequence=seq,
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
            input_fields={0: vin_0, 1: vin_1},
            output_fields={0: vout_0, 1: vout_1},
            s21_by_state=s21_by_state,
            s21=primary_s21,
        )
