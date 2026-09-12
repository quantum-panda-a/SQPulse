"""Channel definitions for PulseSequence."""

from __future__ import annotations
from typing import Union, Optional


class Channel:
    """Represents a physical or virtual control line (e.g., microwave drive, flux bias).

    Args:
        name (str): Identifier for this channel (e.g. 'q0.xy').
        description (Optional[str]): Description or notes.
    """

    def __init__(self, name: str, description: Optional[str] = None):
        self.name = str(name)
        self.description = description or ""

    def __repr__(self) -> str:
        return f"Channel('{self.name}')"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Channel):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        return False


ChannelLike = Union[Channel, str]


def normalize_channel(ch: ChannelLike) -> str:
    """Normalize a Channel or str object to its string name."""
    if isinstance(ch, Channel):
        return ch.name
    elif isinstance(ch, str):
        return ch
    raise TypeError(f"Expected Channel or str, got {type(ch)}")
