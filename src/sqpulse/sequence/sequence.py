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
            self._channel_clocks[ch] = 0.0

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
            self._channel_clocks[ch] = 0.0
        self._channel_clocks[ch] += float(duration)
        return self

    def delay_all(self, duration: float) -> PulseSequence:
        """Add a delay to all channels (in seconds)."""
        self.sync()
        for ch in self._channel_clocks:
            self._channel_clocks[ch] += float(duration)
        return self

    def sync(self, *channels: ChannelLike) -> PulseSequence:
        """Synchronize channels to the maximum clock time among them (or all channels if none specified)."""
        if not self._channel_clocks:
            return self

        if not channels:
            target_channels = list(self._channel_clocks.keys())
        else:
            target_channels = [normalize_channel(c) for c in channels]

        t_max = max(self._channel_clocks.get(c, 0.0) for c in target_channels)
        for c in target_channels:
            self._channel_clocks[c] = t_max
        return self

    def sample(self, dt: float = 1e-9) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Sample all channels onto a uniform time grid.

        Args:
            dt: Sample time step in seconds (default 1e-9 s = 1 ns).

        Returns:
            times: 1D numpy array of time points [0, dt, ..., duration] in seconds.
            waveforms: Dict mapping channel name to complex 1D array of baseband signal.
        """
        total_time = self.duration
        if total_time <= 0:
            return np.array([0.0]), {ch: np.array([0.0 + 0j]) for ch in self.channels}

        n_pts = int(np.round(total_time / dt)) + 1
        times = np.linspace(0, total_time, n_pts, endpoint=True)

        waveforms = {}
        for ch, pulse_list in self._channels.items():
            c_wave = np.zeros(n_pts, dtype=complex)
            for item in pulse_list:
                p_t, p_wave = item.pulse.sample(dt=dt)
                idx_start = int(np.round(item.t_start / dt))
                idx_end = min(idx_start + len(p_wave), n_pts)
                n_copy = idx_end - idx_start
                if n_copy > 0:
                    c_wave[idx_start:idx_end] += p_wave[:n_copy]
            waveforms[ch] = c_wave

        return times, waveforms

    def to_qutip_evo(
        self,
        transmon,
        dt: float = 5e-10,
        f_d: Optional[float] = None,
    ) -> Tuple[np.ndarray, qutip.QobjEvo]:
        """Compile this sequence for a given Transmon into a QuTiP QobjEvo time-dependent Hamiltonian.

        Args:
            transmon: Transmon model instance.
            dt: Simulation time step in seconds (default 5e-10 s = 0.5 ns).
            f_d: Rotating frame drive reference frequency in Hz (defaults to transmon.f_q).

        Returns:
            times: 1D array of times in seconds.
            evo: QuTiP QobjEvo time-dependent Hamiltonian.
        """
        times, waveforms = self.sample(dt=dt)
        drive_ch = transmon.drive

        if drive_ch in waveforms:
            drive_wave = waveforms[drive_ch]
        else:
            drive_wave = np.zeros_like(times, dtype=complex)

        i_coeffs = drive_wave.real
        q_coeffs = drive_wave.imag

        h0 = transmon.H0(f_d=f_d)
        h_x = transmon.H_drive_x
        h_y = transmon.H_drive_y

        evo = qutip.QobjEvo(
            [h0, [h_x, i_coeffs], [h_y, q_coeffs]],
            tlist=times,
        )
        return times, evo

    def plot(
        self,
        dt: float = 5e-10,
        figsize: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot multi-channel pulse schedule in time domain."""
        times, waveforms = self.sample(dt=dt)
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
