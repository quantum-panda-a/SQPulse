"""JSON configuration loader and device container for SQPulse models."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterator, Optional, Union, List, Any
import json

from .transmon import Transmon
from .resonator import ReadoutResonator


class DeviceModels:
    """Container holding a collection of quantum hardware models (Transmons and Resonators).

    Provides dictionary-like access to components by name, as well as coupling configurations.

    Example:
        models = load_models("chip.json")
        q0 = models["q0"]
        r0 = models["r0"]
        q_list = list(models.transmons.values())
        c_q0_r0 = models.get_coupling("q0", "r0")
    """

    def __init__(
        self,
        transmons: Optional[Dict[str, Transmon]] = None,
        resonators: Optional[Dict[str, ReadoutResonator]] = None,
        couplings: Optional[List[Dict[str, Any]]] = None,
        chip_name: Optional[str] = None,
    ):
        self.transmons: Dict[str, Transmon] = dict(transmons) if transmons else {}
        self.resonators: Dict[str, ReadoutResonator] = dict(resonators) if resonators else {}
        self.couplings: List[Dict[str, Any]] = list(couplings) if couplings else []
        self.chip_name: Optional[str] = chip_name

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

    def get_coupling(self, mode1: Union[str, Any], mode2: Union[str, Any]) -> Optional[Dict[str, Any]]:
        """Find the coupling configuration dictionary between two modes."""
        m1 = mode1.name if hasattr(mode1, "name") else str(mode1)
        m2 = mode2.name if hasattr(mode2, "name") else str(mode2)
        target = {m1, m2}
        for coup in self.couplings:
            if set(coup.get("modes", [])) == target:
                return coup
        return None

    def to_quantum_system(
        self,
        modes: Optional[List[Union[str, Transmon, ReadoutResonator]]] = None,
        name: Optional[str] = None,
    ):
        """Construct a QuantumSystem from selected modes and defined couplings."""
        from .system import QuantumSystem, CouplingTerm
        if modes is not None:
            selected_modes = [self[m] if isinstance(m, str) else m for m in modes]
        else:
            selected_modes = list(self.transmons.values())

        sys = QuantumSystem(selected_modes, name=name or self.chip_name or "quantum_system")
        for coup in self.couplings:
            c_modes = coup.get("modes", [])
            c_type = coup.get("type", "capacitive")
            c_name = coup.get("name")
            if all(m in sys for m in c_modes):
                if c_type == "capacitive" and "g" in coup:
                    sys.add_capacitive_coupling(c_modes[0], c_modes[1], g=coup["g"], name=c_name)
                elif c_type == "zz" and "zeta" in coup:
                    sys.add_coupling(CouplingTerm.zz(c_modes[0], c_modes[1], zeta=coup["zeta"], name=c_name))
        return sys

    def to_dict(self, human_readable: bool = False, include_names: bool = False) -> dict:
        """Export all models into a dictionary suitable for JSON serialization."""
        d = {}
        if self.chip_name:
            d["chip_name"] = self.chip_name
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
        if self.couplings:
            d["couplings"] = self.couplings
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
        c_part = f", couplings={[c.get('name', str(c.get('modes', []))) for c in self.couplings]}" if self.couplings else ""
        return f"DeviceModels(transmons={q_names}, resonators={r_names}{c_part})"


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

    chip_name = data.get("chip_name", None)
    raw_couplings = data.get("couplings", [])

    from ..units import parse_quantity
    import numpy as np

    for coup in raw_couplings:
        c_modes = coup.get("modes", [])
        g_val = coup.get("g")
        chi_val = coup.get("chi")
        for m in c_modes:
            if m in resonators:
                r = resonators[m]
                if chi_val is not None:
                    r.chi = 2.0 * np.pi * float(parse_quantity(chi_val))
                if g_val is not None:
                    r.g = 2.0 * np.pi * float(parse_quantity(g_val))

    return DeviceModels(
        transmons=transmons,
        resonators=resonators,
        couplings=raw_couplings,
        chip_name=chip_name,
    )


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
