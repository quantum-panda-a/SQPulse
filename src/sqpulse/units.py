"""Physical unit scale factors and constants for SQPulse in the International System of Units (SI).

Usage:
    from sqpulse import ns, us, ms, Hz, MHz, GHz
    p = GaussianPulse(duration=40 * ns)
    q = Transmon(f_q=5.0 * GHz, alpha=-250.0 * MHz)
"""

from __future__ import annotations

# Time units (in seconds: s)
s: float = 1.0
sec: float = 1.0
second: float = 1.0
seconds: float = 1.0

ms: float = 1e-3
millisecond: float = 1e-3
milliseconds: float = 1e-3

us: float = 1e-6
microsecond: float = 1e-6
microseconds: float = 1e-6

ns: float = 1e-9
nanosecond: float = 1e-9
nanoseconds: float = 1e-9

ps: float = 1e-12
picosecond: float = 1e-12
picoseconds: float = 1e-12

# Frequency units (in Hertz: Hz)
Hz: float = 1.0
hertz: float = 1.0

kHz: float = 1e3
kilohertz: float = 1e3

MHz: float = 1e6
megahertz: float = 1e6

GHz: float = 1e9
gigahertz: float = 1e9

# Angular frequency units (in rad/s)
rad_s: float = 1.0
krad_s: float = 1e3
Mrad_s: float = 1e6
Grad_s: float = 1e9

__all__ = [
    "s",
    "sec",
    "second",
    "seconds",
    "ms",
    "millisecond",
    "milliseconds",
    "us",
    "microsecond",
    "microseconds",
    "ns",
    "nanosecond",
    "nanoseconds",
    "ps",
    "picosecond",
    "picoseconds",
    "Hz",
    "hertz",
    "kHz",
    "kilohertz",
    "MHz",
    "megahertz",
    "GHz",
    "gigahertz",
    "rad_s",
    "krad_s",
    "Mrad_s",
    "Grad_s",
]
