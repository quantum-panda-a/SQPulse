"""Pulse sequence orchestration and compilation for SQPulse in SI units."""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import qutip

from ..pulses.base import Pulse
from .channel import ChannelLike, normalize_channel


class ScheduledPulse:
    """Represents a pulse scheduled at a specific start time on a channel in SI units."""

    def __init__(self, t_start: float, pulse: Pulse):
        self.t_start = float(t_start)
        self.pulse = pulse

    @property
    def t_end(self) -> float:
        return self.t_start + self.pulse.duration

    def __repr__(self) -> str:
        return f"ScheduledPulse({self.pulse.name}, t=[{self.t_start:.2e}, {self.t_end:.2e}] s)"


class PulseSequence:
    """Timeline container for scheduling pulses across multiple channels in SI units (seconds).

    Args:
        name (str): Sequence identifier.
    """

    def __init__(self, name: str = "seq"):
        self.name = name
        self._channels: Dict[str, List[ScheduledPulse]] = {}
        self._channel_clocks: Dict[str, float] = {}
        self._sync_time: float = 0.0

    @property
    def channels(self) -> List[str]:
        """List of channel names present in this sequence."""
        return list(self._channels.keys())

    @property
    def duration(self) -> float:
        """Total duration of the sequence in seconds (s)."""
        if not self._channel_clocks:
            return 0.0
        return max(self._channel_clocks.values())

    def add(
        self,
        channel: ChannelLike,
        pulse: Pulse,
        t_start: Optional[float] = None,
    ) -> PulseSequence:
        """Add a pulse to a given channel.

        Args:
            channel: Target channel (str or Channel object).
            pulse: Pulse to schedule.
            t_start: Optional explicit start time in seconds. If None, appends to channel's current clock.

        Returns:
            self (for method chaining).
        """
        ch = normalize_channel(channel)
        if ch not in self._channels:
            self._channels[ch] = []
            if ch not in self._channel_clocks:
                self._channel_clocks[ch] = self._sync_time

        if t_start is None:
            t_start = self._channel_clocks[ch]
        else:
            t_start = float(t_start)

        item = ScheduledPulse(t_start=t_start, pulse=pulse)
        self._channels[ch].append(item)
        self._channel_clocks[ch] = max(self._channel_clocks[ch], item.t_end)
        return self

    def delay(self, channel: ChannelLike, duration: float) -> PulseSequence:
        """Advance the clock on a specific channel by duration (in seconds).

        Args:
            channel: Target channel.
            duration: Delay length in seconds.
        """
        ch = normalize_channel(channel)
        if ch not in self._channels:
            self._channels[ch] = []
            if ch not in self._channel_clocks:
                self._channel_clocks[ch] = self._sync_time
        self._channel_clocks[ch] += float(duration)
        return self

    def delay_all(self, duration: float) -> PulseSequence:
        """Add a delay to all channels (in seconds)."""
        self.sync()
        for ch in self._channel_clocks:
            self._channel_clocks[ch] += float(duration)
        self._sync_time += float(duration)
        return self

    def sync(self, *channels: ChannelLike) -> PulseSequence:
        """Synchronize channels to the maximum clock time among them (or all channels if none specified)."""
        if not self._channel_clocks:
            return self

        if not channels:
            t_max = max(self._channel_clocks.values())
            self._sync_time = max(self._sync_time, t_max)
            for c in self._channel_clocks:
                self._channel_clocks[c] = self._sync_time
        else:
            target_channels = [normalize_channel(c) for c in channels]
            t_max = max(self._channel_clocks.get(c, 0.0) for c in target_channels)
            for c in target_channels:
                self._channel_clocks[c] = t_max
        return self

    def align(
        self,
        *channel_pulse_pairs: Union[Tuple[ChannelLike, Pulse], List[Tuple[ChannelLike, Pulse]]],
        mode: str = "center",
    ) -> PulseSequence:
        """Align multiple pulses across different channels and schedule them concurrently.

        Args:
            *channel_pulse_pairs: Arbitrary number of (channel, pulse) pairs,
                or a single list/tuple of (channel, pulse) pairs.
            mode: Alignment mode:
                - 'center': Align the temporal midpoints of all pulses.
                - 'left': Align the start times of all pulses.
                - 'right': Align the end times of all pulses.

        Returns:
            self (PulseSequence): For method chaining.
        """
        # Handle unpacking if passed as a single list or tuple of pairs
        if len(channel_pulse_pairs) == 1 and isinstance(channel_pulse_pairs[0], (list, tuple)) and channel_pulse_pairs[0]:
            first_item = channel_pulse_pairs[0]
            if isinstance(first_item, (list, tuple)) and len(first_item) > 0 and isinstance(first_item[0], (list, tuple)):
                pairs = list(first_item)
            else:
                pairs = list(channel_pulse_pairs)
        else:
            pairs = list(channel_pulse_pairs)

        if not pairs:
            return self

        align_mode = mode.lower().strip()
        if align_mode not in ("center", "left", "right"):
            raise ValueError(
                f"Invalid alignment mode '{mode}'. Supported modes are: 'center', 'left', 'right'."
            )

        normalized_pairs = []
        for pair in pairs:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError(
                    f"Each element passed to align must be a (channel, pulse) pair, got: {pair}"
                )
            ch_like, pulse = pair
            ch_name = normalize_channel(ch_like)
            dur = getattr(pulse, "duration", None)
            if dur is None or dur < 0:
                raise ValueError(f"Pulse {pulse} must have a non-negative duration.")
            normalized_pairs.append((ch_name, pulse, float(dur)))

        # Determine reference base time: maximum clock among participating channels
        t_base = max(
            self._channel_clocks.get(ch_name, self._sync_time)
            for ch_name, _, _ in normalized_pairs
        )
        max_duration = max(dur for _, _, dur in normalized_pairs)

        # Schedule pulses according to mode
        for ch_name, pulse, dur in normalized_pairs:
            if align_mode == "center":
                t_start = t_base + (max_duration - dur) / 2.0
            elif align_mode == "left":
                t_start = t_base
            elif align_mode == "right":
                t_start = t_base + (max_duration - dur)
            self.add(ch_name, pulse, t_start=t_start)

        # Barrier synchronization: advance clocks of all participating channels to block end
        t_block_end = t_base + max_duration
        for ch_name, _, _ in normalized_pairs:
            self._channel_clocks[ch_name] = t_block_end
        self._sync_time = max(self._sync_time, t_block_end)

        return self

    def align_center(
        self,
        *channel_pulse_pairs: Union[Tuple[ChannelLike, Pulse], List[Tuple[ChannelLike, Pulse]]],
    ) -> PulseSequence:
        """Convenience alias for self.align(*channel_pulse_pairs, mode='center').

        Aligns the midpoints of all specified pulses across channels.
        """
        return self.align(*channel_pulse_pairs, mode="center")

    def sample(
        self,
        dt: Optional[float] = 1e-9,
        alpha: Optional[float] = None,
        total_time: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Sample all channels onto a uniform time grid.

        Args:
            dt: Sample time step in seconds (default 1e-9 s = 1 ns). If None, chosen adaptively.
            alpha: Optional reference anharmonicity in Hz passed to pulses for DRAG scaling.
            total_time: Optional total simulation time in seconds to extend grid beyond sequence.duration.

        Returns:
            times: 1D numpy array of time points [0, dt, ..., duration] in seconds.
            waveforms: Dict mapping channel name to complex 1D array of baseband signal.
        """
        if total_time is not None:
            tot_time = float(total_time)
        else:
            tot_time = self.duration

        if tot_time <= 0:
            return np.array([0.0]), {ch: np.array([0.0 + 0j]) for ch in self.channels}

        if dt is not None:
            dt = float(dt)
            if dt <= 0:
                raise ValueError(f"Sampling step dt must be positive, got {dt} s")
            if dt >= tot_time:
                hint = ""
                if dt > 1e-3 and tot_time < 1e-3:
                    hint = f" Did you mean dt={dt}e-9 s ({dt} ns)? All time parameters in SQPulse are strictly in SI units (seconds)."
                raise ValueError(
                    f"Sampling interval dt={dt} s cannot be greater than or equal to total sequence duration={tot_time} s.{hint}"
                )
        else:
            dt = min(tot_time / 200, 1e-9)

        n_pts = int(np.round(tot_time / dt)) + 1
        times = np.linspace(0, tot_time, n_pts, endpoint=True)

        waveforms = {}
        for ch, pulse_list in self._channels.items():
            c_wave = np.zeros(n_pts, dtype=complex)
            for item in pulse_list:
                p_t, p_wave = item.pulse.sample(dt=dt, alpha=alpha)
                idx_start = int(np.round(item.t_start / dt))
                idx_end = min(idx_start + len(p_wave), n_pts)
                n_copy = idx_end - idx_start
                if n_copy > 0:
                    c_wave[idx_start:idx_end] += p_wave[:n_copy]
            waveforms[ch] = c_wave

        return times, waveforms

    def to_qutip_evo(
        self,
        target: Any,
        dt: Optional[float] = 5e-10,
        f_d: Optional[Union[float, Dict[str, float]]] = None,
    ) -> Tuple[np.ndarray, qutip.QobjEvo]:
        """Compile this sequence for a given Transmon or QuantumSystem into a QuTiP QobjEvo time-dependent Hamiltonian.

        Args:
            target: Transmon model instance or QuantumSystem composite system instance.
            dt: Simulation time step in seconds (default 5e-10 s = 0.5 ns).
            f_d: Rotating frame drive reference frequency in Hz.
                For single Transmon: float (defaults to transmon.frequency_at_flux).
                For QuantumSystem: None, float, or dict mapping mode name to frequency.

        Returns:
            times: 1D array of times in seconds.
            evo: QuTiP QobjEvo time-dependent Hamiltonian.
        """
        eff_dt = dt if dt is not None else (min(self.duration / 400, 5e-10) if self.duration > 0 else 5e-10)

        # Branch 1: QuantumSystem (Multi-mode composite system)
        if hasattr(target, "modes") and hasattr(target, "tensor_with_I"):
            system = target
            times, waveforms = self.sample(dt=eff_dt)
            static_terms = []
            timedep_terms = []

            # 1. Process drive and detuning for each mode
            for m in system.modes:
                # Mode reference frequency
                if f_d is None:
                    # Default unified rotating frame: reference frequency of the first mode
                    m_fd = system.modes[0].frequency_at_flux(system.modes[0].flux_offset)
                elif isinstance(f_d, (int, float, np.floating)):
                    m_fd = float(f_d)
                elif isinstance(f_d, dict):
                    m_fd = float(f_d.get(m.name, m.frequency_at_flux(m.flux_offset)))
                else:
                    raise TypeError(f"Invalid f_d specification: {type(f_d)}")

                # XY microwave drive lines
                xy_ch_name = normalize_channel(m.xy)
                if xy_ch_name in waveforms:
                    drive_wave = waveforms[xy_ch_name]
                    i_coeffs = drive_wave.real
                    q_coeffs = drive_wave.imag
                    h_x_full = system.H_drive_x(m)
                    h_y_full = system.H_drive_y(m)
                    if np.any(np.abs(i_coeffs) > 1e-15):
                        timedep_terms.append([h_x_full, i_coeffs])
                    if np.any(np.abs(q_coeffs) > 1e-15):
                        timedep_terms.append([h_y_full, q_coeffs])

                # Z flux bias lines
                z_ch_name = normalize_channel(m.z)
                has_flux_pulse = False
                if z_ch_name in waveforms:
                    raw_flux_wave = waveforms[z_ch_name].real
                    flux_wave = m.voltage_to_flux(raw_flux_wave)
                    if np.any(np.abs(flux_wave) > 1e-14):
                        has_flux_pulse = True

                if has_flux_pulse and m.d < 1.0:
                    total_flux = m.flux_offset + flux_wave
                    freqs_t = np.array([m.frequency_at_flux(phi) for phi in total_flux])
                    detune_coeffs = 2.0 * np.pi * (freqs_t - m_fd)
                    # Kerr term
                    h_kerr_local = np.pi * m.alpha * (m.ad * m.ad * m.a * m.a)
                    static_terms.append(system.tensor_with_I(m, h_kerr_local))
                    timedep_terms.append([system.n(m), detune_coeffs])
                else:
                    h0_local = m.H0(f_d=m_fd, flux=m.flux_offset)
                    static_terms.append(system.tensor_with_I(m, h0_local))

            # 2. Add inter-mode coupling terms
            for c_term in system.coupling_terms:
                static_terms.append(c_term.to_qobj(system))

            h_static = sum(static_terms) if static_terms else qutip.qzero(system.levels)
            evo_terms = [h_static] + timedep_terms
            evo = qutip.QobjEvo(evo_terms, tlist=times)
            return times, evo

        # Branch 2: Single Transmon (Existing behavior, 100% backward compatible)
        transmon = target
        ref_alpha = getattr(transmon, "alpha", None)
        times, waveforms = self.sample(dt=eff_dt, alpha=ref_alpha)

        # 1. Collect XY drive waveform from xy line
        xy_ch_name = normalize_channel(transmon.xy)
        drive_wave = waveforms.get(xy_ch_name, np.zeros_like(times, dtype=complex))

        i_coeffs = drive_wave.real
        q_coeffs = drive_wave.imag

        # 2. Collect Z flux waveform from z line
        z_ch_name = normalize_channel(transmon.z)
        if z_ch_name in waveforms:
            raw_flux_wave = waveforms[z_ch_name].real
            flux_wave = transmon.voltage_to_flux(raw_flux_wave)
            has_flux_pulse = bool(np.any(np.abs(flux_wave) > 1e-14))
        else:
            flux_wave = np.zeros_like(times, dtype=float)
            has_flux_pulse = False

        ref_fd = f_d if f_d is not None else transmon.frequency_at_flux(transmon.flux_offset)
        h_x = transmon.H_drive_x
        h_y = transmon.H_drive_y

        if has_flux_pulse and transmon.d < 1.0:
            # Time-dependent detuning: 2*pi * (f_q(Phi(t)) - f_d) * n
            total_flux = transmon.flux_offset + flux_wave
            freqs_t = np.array([transmon.frequency_at_flux(phi) for phi in total_flux])
            detune_coeffs = 2.0 * np.pi * (freqs_t - ref_fd)
            h_kerr = np.pi * transmon.alpha * (transmon.ad * transmon.ad * transmon.a * transmon.a)
            evo_terms = [
                h_kerr,
                [transmon.n, detune_coeffs],
                [h_x, i_coeffs],
                [h_y, q_coeffs],
            ]
        else:
            h0 = transmon.H0(f_d=ref_fd)
            evo_terms = [
                h0,
                [h_x, i_coeffs],
                [h_y, q_coeffs],
            ]

        evo = qutip.QobjEvo(evo_terms, tlist=times)
        return times, evo

    def plot(
        self,
        dt: Optional[float] = None,
        figsize: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot multi-channel pulse schedule in time domain."""
        eff_dt = dt if dt is not None else (min(self.duration / 200, 5e-10) if self.duration > 0 else 5e-10)
        times, waveforms = self.sample(dt=eff_dt)
        channels = self.channels

        if not channels:
            fig, ax = plt.subplots(figsize=figsize or (8, 3))
            ax.set_title("Empty PulseSequence")
            return fig

        n_ch = len(channels)
        fig, axes = plt.subplots(
            n_ch, 1, figsize=figsize or (10, 2.5 * n_ch), sharex=True, squeeze=False
        )

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for i, ch in enumerate(channels):
            ax = axes[i, 0]
            wave = waveforms[ch]
            c = colors[i % len(colors)]
            ax.plot(times, wave.real, label=f"I (In-phase)", color=c, lw=2)
            ax.plot(times, wave.imag, label=f"Q (Quadrature)", color=c, lw=1.5, ls="--", alpha=0.7)
            ax.plot(times, np.abs(wave), label=f"|Envelope|", color="black", lw=1.0, ls=":", alpha=0.5)

            ax.set_ylabel(f"{ch}\nAmplitude", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

        axes[-1, 0].set_xlabel("Time (s)")
        fig.suptitle(title or f"Pulse Sequence: {self.name} (Duration: {self.duration:.2e} s)", y=1.02)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return (
            f"PulseSequence('{self.name}', channels={self.channels}, "
            f"duration={self.duration:.2e}s)"
        )
