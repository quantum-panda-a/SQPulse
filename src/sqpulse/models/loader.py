"""JSON configuration loader and device container for SQPulse models."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterator, Optional, Union
import json

from .transmon import Transmon
from .resonator import ReadoutResonator


class DeviceModels:
    """Container holding a collection of quantum hardware models (Transmons and Resonators).

    Provides dictionary-like access to components by name.

    Example:
        models = load_models("chip.json")
        q0 = models["q0"]
        r0 = models["r0"]
        q_list = list(models.transmons.values())
    """

    def __init__(
        self,
        transmons: Optional[Dict[str, Transmon]] = None,
        resonators: Optional[Dict[str, ReadoutResonator]] = None,
    ):
        self.transmons: Dict[str, Transmon] = dict(transmons) if transmons else {}
        self.resonators: Dict[str, ReadoutResonator] = dict(resonators) if resonators else {}

    @property
    def qubits(self) -> Dict[str, Transmon]:
        """Alias for transmons."""
        return self.transmons

    @property
    def all(self) -> Dict[str, Union[Transmon, ReadoutResonator]]:
        """Dictionary of all models combined."""
        merged: Dict[str, Union[Transmon, ReadoutResonator]] = {}
        merged.update(self.transmons)
        merged.update(self.resonators)
        return merged

    def __getitem__(self, key: str) -> Union[Transmon, ReadoutResonator]:
        if key in self.transmons:
            return self.transmons[key]
        if key in self.resonators:
            return self.resonators[key]
        raise KeyError(f"Component '{key}' not found in DeviceModels (available: {list(self.all.keys())})")

    def __contains__(self, key: str) -> bool:
        return key in self.transmons or key in self.resonators

    def __iter__(self) -> Iterator[str]:
        return iter(self.all)

    def __len__(self) -> int:
        return len(self.transmons) + len(self.resonators)

    def get(self, key: str, default: Optional[Union[Transmon, ReadoutResonator]] = None) -> Optional[Union[Transmon, ReadoutResonator]]:
        return self.transmons.get(key, self.resonators.get(key, default))

    def to_dict(self, human_readable: bool = False, include_names: bool = False) -> dict:
        """Export all models into a dictionary suitable for JSON serialization."""
        d = {}
        if self.transmons:
            d["transmons"] = {}
            for name, q in self.transmons.items():
                q_dict = q.to_dict(human_readable=human_readable)
                if not include_names:
                    q_dict.pop("name", None)
                d["transmons"][name] = q_dict
        if self.resonators:
            d["resonators"] = {}
            for name, r in self.resonators.items():
                r_dict = r.to_dict(human_readable=human_readable)
                if not include_names:
                    r_dict.pop("name", None)
                d["resonators"][name] = r_dict
        return d

    def to_json(
        self,
        filepath_or_buf: Optional[Union[str, Path]] = None,
        indent: int = 2,
        human_readable: bool = False,
    ) -> Optional[str]:
        """Serialize all models to a JSON file or JSON string."""
        data = self.to_dict(human_readable=human_readable)
        if filepath_or_buf is not None:
            p = Path(filepath_or_buf)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent)
            return None
        return json.dumps(data, indent=indent)

    def __repr__(self) -> str:
        q_names = list(self.transmons.keys())
        r_names = list(self.resonators.keys())
        return f"DeviceModels(transmons={q_names}, resonators={r_names})"


def load_models(
    source: Union[str, Path, dict],
) -> Union[DeviceModels, Transmon, ReadoutResonator]:
    """Load SQPulse models (Transmons and ReadoutResonators) from a JSON file, JSON string, or dictionary.

    Supports:
        1. Multi-component configuration files:
           {
               "transmons": { "q0": {...}, "q1": {...} },
               "resonators": { "r0": {...} }
           }
        2. Single model configuration dictionary:
           { "name": "q0", "f_q": "5.0 GHz", ... }

    Args:
        source: File path, raw JSON string, or parsed dictionary.

    Returns:
        A DeviceModels container (for multi-component configs) or a single Transmon/ReadoutResonator.
    """
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(str(source))
    elif isinstance(source, dict):
        data = source
    else:
        raise TypeError(f"Expected file path, JSON string, or dict, got {type(source).__name__}")

    # Check if this is a single Transmon dict
    model_type = data.get("type", "")
    if model_type.lower() == "transmon" or ("f_q" in data and "transmons" not in data and "qubits" not in data):
        return Transmon.from_dict(data)

    # Check if this is a single ReadoutResonator dict
    if model_type.lower() in ("readoutresonator", "resonator") or ("f_r" in data and "resonators" not in data):
        return ReadoutResonator.from_dict(data)

    # Multi-component config
    transmons: Dict[str, Transmon] = {}
    resonators: Dict[str, ReadoutResonator] = {}

    raw_transmons = data.get("transmons", data.get("qubits", {}))
    if isinstance(raw_transmons, dict):
        for name, cfg in raw_transmons.items():
            cfg_copy = dict(cfg)
            if "name" not in cfg_copy:
                cfg_copy["name"] = name
            transmons[name] = Transmon.from_dict(cfg_copy)
    elif isinstance(raw_transmons, list):
        for item in raw_transmons:
            q = Transmon.from_dict(item)
            transmons[q.name] = q

    raw_resonators = data.get("resonators", data.get("cavities", {}))
    if isinstance(raw_resonators, dict):
        for name, cfg in raw_resonators.items():
            cfg_copy = dict(cfg)
            if "name" not in cfg_copy:
                cfg_copy["name"] = name
            resonators[name] = ReadoutResonator.from_dict(cfg_copy)
    elif isinstance(raw_resonators, list):
        for item in raw_resonators:
            r = ReadoutResonator.from_dict(item)
            resonators[r.name] = r

    return DeviceModels(transmons=transmons, resonators=resonators)


def save_models(
    models: Union[DeviceModels, dict],
    filepath: Union[str, Path],
    indent: int = 2,
    human_readable: bool = False,
) -> None:
    """Save models to a JSON file.

    Args:
        models: DeviceModels instance or dict of models.
        filepath: Destination JSON file path.
        indent: Indentation level.
        human_readable: If True, outputs physical quantities with unit strings.
    """
    if isinstance(models, DeviceModels):
        models.to_json(filepath, indent=indent, human_readable=human_readable)
    elif isinstance(models, dict):
        out_dict = {}
        for k, v in models.items():
            if hasattr(v, "to_dict"):
                out_dict[k] = v.to_dict(human_readable=human_readable)
            elif isinstance(v, dict):
                sub = {}
                for sub_k, sub_v in v.items():
                    if hasattr(sub_v, "to_dict"):
                        sub_dict = sub_v.to_dict(human_readable=human_readable)
                        sub_dict.pop("name", None)
                        sub[sub_k] = sub_dict
                    else:
                        sub[sub_k] = sub_v
                out_dict[k] = sub
            else:
                out_dict[k] = v

        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, indent=indent)
    else:
        raise TypeError(f"Expected DeviceModels or dict, got {type(models).__name__}")
