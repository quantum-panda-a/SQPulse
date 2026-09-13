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

# Temperature units (in Kelvin: K)
K: float = 1.0
kelvin: float = 1.0
mK: float = 1e-3
millikelvin: float = 1e-3
uK: float = 1e-6
microkelvin: float = 1e-6

import re
from typing import Optional, Union

UNIT_SCALES = {
    # Time
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "ms": 1e-3,
    "millisecond": 1e-3,
    "milliseconds": 1e-3,
    "us": 1e-6,
    "μs": 1e-6,
    "microsecond": 1e-6,
    "microseconds": 1e-6,
    "ns": 1e-9,
    "nanosecond": 1e-9,
    "nanoseconds": 1e-9,
    "ps": 1e-12,
    "picosecond": 1e-12,
    "picoseconds": 1e-12,
    # Frequency
    "hz": 1.0,
    "hertz": 1.0,
    "khz": 1e3,
    "kilohertz": 1e3,
    "mhz": 1e6,
    "megahertz": 1e6,
    "ghz": 1e9,
    "gigahertz": 1e9,
    # Angular frequency
    "rad_s": 1.0,
    "rad/s": 1.0,
    "krad_s": 1e3,
    "krad/s": 1e3,
    "mrad_s": 1e6,
    "mrad/s": 1e6,
    "grad_s": 1e9,
    "grad/s": 1e9,
    # Voltage
    "v": 1.0,
    "volt": 1.0,
    "volts": 1.0,
    "mv": 1e-3,
    "uv": 1e-6,
    "μv": 1e-6,
    # Temperature (in Kelvin: K)
    "k": 1.0,
    "kelvin": 1.0,
    "mk": 1e-3,
    "millikelvin": 1e-3,
    "uk": 1e-6,
    "μk": 1e-6,
    "microkelvin": 1e-6,
    # Dimensionless
    "%": 0.01,
}

_QUANTITY_REGEX = re.compile(
    r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Z_μ/%]+)?$"
)


def parse_quantity(val: Union[float, int, str, None], default: Optional[float] = None) -> Optional[float]:
    """Parse a quantity into a float in SI units.

    Supports:
        - Numeric types (float, int)
        - "inf", "+inf", "-inf", "infinity"
        - None or "null" (returns default)
        - String quantities with units (e.g. "5.0 GHz", "-250 MHz", "25 us", "100 ns", "15.7 Mrad_s")

    Args:
        val: Input quantity (number, string with optional unit, or None).
        default: Fallback value if val is None or "null".

    Returns:
        Float value scaled to SI units, or default.

    Raises:
        ValueError: If string format is invalid or unit is unknown.
    """
    if val is None:
        return default

    if isinstance(val, (int, float)):
        return float(val)

    if not isinstance(val, str):
        raise TypeError(f"Expected float, int, or str for quantity, got {type(val).__name__}")

    s = val.strip()
    if not s:
        return default

    s_lower = s.lower()
    if s_lower in ("none", "null"):
        return default
    if s_lower in ("inf", "+inf", "infinity", "+infinity"):
        return float("inf")
    if s_lower in ("-inf", "-infinity"):
        return float("-inf")

    match = _QUANTITY_REGEX.match(s)
    if not match:
        raise ValueError(f"Invalid quantity format: '{val}'. Expected e.g. '5.0 GHz', '25 us', or '5e9'.")

    num_str, unit_str = match.groups()
    num = float(num_str)

    if not unit_str:
        return num

    unit_key = unit_str.lower()
    if unit_key not in UNIT_SCALES:
        # Check original casing in case of special prefixes
        if unit_str in UNIT_SCALES:
            unit_key = unit_str
        else:
            raise ValueError(
                f"Unknown unit '{unit_str}' in '{val}'. "
                f"Available units include: GHz, MHz, kHz, Hz, s, ms, us (μs), ns, ps, rad_s, Mrad_s, V, etc."
            )

    return num * UNIT_SCALES[unit_key]


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
    "K",
    "kelvin",
    "mK",
    "millikelvin",
    "uK",
    "microkelvin",
    "parse_quantity",
    "UNIT_SCALES",
]
