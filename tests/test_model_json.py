"""Tests for JSON configuration loading, serialization, and unit parsing in SQPulse."""

import json
from pathlib import Path
import numpy as np
import pytest

from sqpulse import (
    Transmon,
    ReadoutResonator,
    DeviceModels,
    load_models,
    save_models,
    parse_quantity,
    PulseSequence,
    GaussianPulse,
    Measurement,
    ns,
    us,
    MHz,
    GHz,
    mK,
    K,
)


def test_parse_quantity():
    # Direct numbers
    assert parse_quantity(5.0) == 5.0
    assert parse_quantity(10) == 10.0

    # None and defaults
    assert parse_quantity(None) is None
    assert parse_quantity(None, default=42.0) == 42.0
    assert parse_quantity("null", default=100.0) == 100.0
    assert parse_quantity("none", default=100.0) == 100.0

    # Infinity
    assert parse_quantity("inf") == float("inf")
    assert parse_quantity("+inf") == float("inf")
    assert parse_quantity("infinity") == float("inf")
    assert parse_quantity("-inf") == float("-inf")

    # Frequency units
    assert np.isclose(parse_quantity("5.0 GHz"), 5.0e9)
    assert np.isclose(parse_quantity("-250.0 MHz"), -250.0e6)
    assert np.isclose(parse_quantity("100 kHz"), 100.0e3)
    assert np.isclose(parse_quantity("50 Hz"), 50.0)

    # Time units
    assert np.isclose(parse_quantity("25.0 us"), 25.0e-6)
    assert np.isclose(parse_quantity("30.0 μs"), 30.0e-6)
    assert np.isclose(parse_quantity("40 ns"), 40.0e-9)
    assert np.isclose(parse_quantity("100 ps"), 100.0e-12)
    assert np.isclose(parse_quantity("10 ms"), 10.0e-3)
    assert np.isclose(parse_quantity("1.5 s"), 1.5)

    # Angular frequency
    assert np.isclose(parse_quantity("15.7 Mrad_s"), 15.7e6)
    assert np.isclose(parse_quantity("314.159 krad/s"), 314159.0)

    # Voltage
    assert np.isclose(parse_quantity("0.8 V"), 0.8)
    assert np.isclose(parse_quantity("50 mV"), 0.05)

    # Invalid formats
    with pytest.raises(ValueError):
        parse_quantity("invalid_string")

    with pytest.raises(ValueError):
        parse_quantity("5.0 LightYears")

    with pytest.raises(TypeError):
        parse_quantity([1, 2, 3])


def test_transmon_to_from_dict():
    # Create with units
    cfg = {
        "name": "q0",
        "f_q": "5.0 GHz",
        "alpha": "-250.0 MHz",
        "d": 0.25,
        "flux_offset": 0.05,
        "levels": 3,
        "t1": "25.0 us",
        "t2": "30.0 us",
        "v_phi0": "0.8 V",
        "omega_d": "60.0 MHz",
    }
    q = Transmon.from_dict(cfg)

    assert q.name == "q0"
    assert np.isclose(q.f_q, 5.0e9)
    assert np.isclose(q.alpha, -250.0e6)
    assert q.d == 0.25
    assert q.flux_offset == 0.05
    assert q.levels == 3
    assert np.isclose(q.t1, 25.0e-6)
    assert np.isclose(q.t2, 30.0e-6)
    assert np.isclose(q.v_phi0, 0.8)
    assert np.isclose(q.omega_d_hz, 60.0e6)
    assert np.isclose(q.omega_d, 2.0 * np.pi * 60.0e6)

    # Export to dict
    d_raw = q.to_dict(human_readable=False)
    assert d_raw["name"] == "q0"
    assert d_raw["f_q"] == 5.0e9
    assert np.isclose(d_raw["omega_d"], 60.0e6)

    d_human = q.to_dict(human_readable=True)
    assert "GHz" in d_human["f_q"]
    assert "MHz" in d_human["alpha"]
    assert "us" in d_human["t1"]
    assert "MHz" in d_human["omega_d"]

    # Re-import from exported dict
    q_reconstructed = Transmon.from_dict(d_human)
    assert np.isclose(q_reconstructed.f_q, q.f_q)
    assert np.isclose(q_reconstructed.t1, q.t1)
    assert np.isclose(q_reconstructed.omega_d_hz, q.omega_d_hz)


def test_transmon_json_file(tmp_path: Path):
    json_file = tmp_path / "transmon_test.json"
    q_orig = Transmon("q_file", f_q=5.1e9, alpha=-240e6, d=0.3, levels=3, t1=20e-6, t2=15e-6)

    # Save to file
    q_orig.to_json(json_file, human_readable=True)
    assert json_file.is_file()

    # Load from file
    q_loaded = Transmon.from_json(json_file)
    assert q_loaded.name == "q_file"
    assert np.isclose(q_loaded.f_q, 5.1e9)
    assert np.isclose(q_loaded.alpha, -240e6)
    assert np.isclose(q_loaded.d, 0.3)
    assert np.isclose(q_loaded.t1, 20e-6)
    assert np.isclose(q_loaded.t2, 15e-6)

    # Load from JSON string directly
    json_str = q_orig.to_json(human_readable=False)
    q_from_str = Transmon.from_json(json_str)
    assert q_from_str.name == q_orig.name
    assert np.isclose(q_from_str.f_q, q_orig.f_q)


def test_resonator_to_from_dict_and_json(tmp_path: Path):
    cfg = {
        "name": "r0",
        "f_r": "7.05 GHz",
        "kappa": "2.5 MHz",
        "chi": "1.2 MHz",
    }
    r = ReadoutResonator.from_dict(cfg)
    assert r.name == "r0"
    assert np.isclose(r.f_r, 7.05e9)
    assert np.isclose(r.kappa_hz, 2.5e6)
    assert np.isclose(r.chi_hz, 1.2e6)
    assert np.isclose(r.kappa, 2.0 * np.pi * 2.5e6)
    assert np.isclose(r.chi, 2.0 * np.pi * 1.2e6)

    # Legacy kappa_hz / chi_hz keys
    cfg_legacy = {
        "name": "r_leg",
        "f_r": "7.05 GHz",
        "kappa_hz": "2.5 MHz",
        "chi_hz": "1.2 MHz",
    }
    r_leg = ReadoutResonator.from_dict(cfg_legacy)
    assert np.isclose(r_leg.kappa_hz, 2.5e6)
    assert np.isclose(r_leg.chi_hz, 1.2e6)

    # JSON roundtrip
    json_file = tmp_path / "resonator_test.json"
    r.to_json(json_file, human_readable=True)
    r_loaded = ReadoutResonator.from_json(json_file)

    assert r_loaded.name == "r0"
    assert np.isclose(r_loaded.f_r, r.f_r)
    assert np.isclose(r_loaded.kappa, r.kappa)
    assert np.isclose(r_loaded.chi, r.chi)
    assert np.isclose(r_loaded.kappa_hz, r.kappa_hz)


def test_multi_model_chip_config(tmp_path: Path):
    chip_json = tmp_path / "chip_config.json"
    chip_data = {
        "transmons": {
            "q0": {
                "f_q": "5.0 GHz",
                "alpha": "-250.0 MHz",
                "d": 0.2,
                "levels": 3,
                "t1": "25.0 us",
                "t2": "30.0 us",
                "v_phi0": 0.8,
            },
            "q1": {
                "f_q": "5.2 GHz",
                "alpha": "-240.0 MHz",
                "d": 1.0,
                "levels": 3,
                "t1": "40.0 us",
                "t2": "35.0 us",
                "omega_d": "50.0 MHz",
            },
        },
        "resonators": {
            "r0": {
                "f_r": "7.0 GHz",
                "kappa": "2.5 MHz",
                "chi": "1.2 MHz",
            },
            "r1": {
                "f_r": "7.2 GHz",
                "kappa": "3.0 MHz",
                "chi": "1.5 MHz",
            },
        },
    }

    with open(chip_json, "w", encoding="utf-8") as f:
        json.dump(chip_data, f, indent=2)

    # Load models
    models = load_models(chip_json)
    assert isinstance(models, DeviceModels)
    assert len(models) == 4
    assert "q0" in models
    assert "q1" in models
    assert "r0" in models
    assert "r1" in models

    q0 = models["q0"]
    assert isinstance(q0, Transmon)
    assert np.isclose(q0.f_q, 5.0e9)

    r0 = models["r0"]
    assert isinstance(r0, ReadoutResonator)
    assert np.isclose(r0.f_r, 7.0e9)

    # Check container access
    assert "q0" in models.qubits
    assert "r0" in models.resonators
    assert models.get("non_existent", None) is None

    # Test save_models roundtrip
    saved_file = tmp_path / "saved_chip.json"
    save_models(models, saved_file, human_readable=True)
    reloaded_models = load_models(saved_file)
    assert len(reloaded_models) == 4
    assert np.isclose(reloaded_models["q0"].f_q, 5.0e9)


def test_single_model_auto_detect():
    # Dictionary with 'f_q' automatically detected as Transmon
    data_q = {"name": "q_single", "f_q": "5.0 GHz", "levels": 3}
    q = load_models(data_q)
    assert isinstance(q, Transmon)
    assert q.name == "q_single"

    # Dictionary with 'f_r' automatically detected as ReadoutResonator
    data_r = {"name": "r_single", "f_r": "7.1 GHz"}
    r = load_models(data_r)
    assert isinstance(r, ReadoutResonator)
    assert r.name == "r_single"


def test_simulation_with_json_loaded_model():
    cfg = {
        "name": "q0",
        "f_q": "5.0 GHz",
        "alpha": "-250.0 MHz",
        "levels": 3,
        "t1": "30.0 us",
        "t2": "20.0 us",
    }
    q = Transmon.from_dict(cfg)

    # Build a small sequence
    seq = PulseSequence()
    seq.add(q.xy, GaussianPulse(duration=20 * ns, amp=0.5))

    # Run simulation
    res = Measurement.run(q, seq, dt=1.0 * ns)
    pop0 = res.final_population(0)
    pop1 = res.final_population(1)

    assert 0.0 <= pop0 <= 1.0
    assert 0.0 <= pop1 <= 1.0
    assert np.isclose(pop0 + pop1 + res.final_population(2), 1.0, atol=1e-5)


def test_temperature_parsing_and_model():
    # Unit parsing
    assert np.isclose(parse_quantity("35 mK"), 0.035)
    assert np.isclose(parse_quantity("20.0 mk"), 0.02)
    assert np.isclose(parse_quantity("300 K"), 300.0)

    # Direct init with float temperature
    q = Transmon("q_temp", f_q=5.0 * GHz, temperature=35 * mK, t1=20 * us)
    assert np.isclose(q.temperature, 0.035)
    # Expected: nth = 1 / (exp(h*5e9 / (k*0.035)) - 1) ~ 0.001054
    assert np.isclose(q.thermal_population, 0.00105416, rtol=1e-3)
    assert "T=35.0mK" in repr(q)

    # Lindblad c_ops should have 3 operators: decay, excitation, dephasing
    ops = q.c_ops()
    assert len(ops) == 2  # decay + thermal excitation (t2=inf so no separate pure dephasing)

    # Dynamic property setter: temperature -> thermal_population
    q.temperature = 50 * mK
    assert np.isclose(q.temperature, 0.050)
    assert np.isclose(q.thermal_population, 0.008304, rtol=1e-3)

    # Dynamic property setter: thermal_population -> temperature
    q.thermal_population = 0.0010541632809600262
    assert np.isclose(q.temperature, 0.035, rtol=1e-3)

    # from_dict with string temperature
    q_dict = Transmon.from_dict({
        "name": "q_json_temp",
        "f_q": "5.0 GHz",
        "temperature": "25 mK",
    })
    assert np.isclose(q_dict.temperature, 0.025)
    assert q_dict.thermal_population > 0

    # to_dict human_readable includes temperature
    d_out = q_dict.to_dict(human_readable=True)
    assert d_out["temperature"] == "25.00 mK"

    # Zero temperature
    q_zero = Transmon("q_zero", temperature=0.0)
    assert q_zero.temperature == 0.0
    assert q_zero.thermal_population == 0.0

    # Negative temperature error
    with pytest.raises(ValueError):
        Transmon("q_err", temperature=-10 * mK)

