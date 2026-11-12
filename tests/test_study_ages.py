from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from study import ages

AT = datetime(2026, 11, 12, 12, tzinfo=UTC)


def _stamp(hours_before: float) -> str:
    return (AT - timedelta(hours=hours_before)).isoformat().replace("+00:00", "Z")


def _value(name: str, hours_before: float) -> dict[str, object]:
    return {"name": name, "value": 0.01, "unit": "", "date": _stamp(hours_before)}


def _gate(name: str, qubits: list[int], hours_before: float) -> dict[str, object]:
    return {"gate": name, "qubits": qubits, "parameters": [_value("gate_error", hours_before)]}


PROPERTIES = {
    "last_update_date": _stamp(1),
    "qubits": [
        [_value("T1", 10), _value("T2", 12), _value("prob_meas1_prep0", 50)],
        [_value("T1", 20), _value("T2", 22), _value("prob_meas1_prep0", 50)],
        [_value("T1", 400), _value("T2", 24), _value("prob_meas1_prep0", 50)],
    ],
    "gates": [
        _gate("sx", [0], 150),
        _gate("sx", [1], 150),
        _gate("sx", [2], 2),
        _gate("rz", [0], 1),
        _gate("cz", [0, 1], 9),
        _gate("cz", [1, 2], 120),
    ],
}


def test_each_kind_gets_the_ages_of_its_values() -> None:
    found = ages.value_ages(PROPERTIES, AT)

    assert found["readout"] == {"values": 3, "youngest_h": 50, "median_h": 50, "oldest_h": 50}
    assert found["t1"] == {"values": 3, "youngest_h": 10, "median_h": 20, "oldest_h": 400}
    assert found["one_qubit"]["median_h"] == 150
    assert found["two_qubit"] == {"values": 2, "youngest_h": 9, "median_h": 64.5, "oldest_h": 120}


def test_a_virtual_gate_does_not_count_as_a_measured_value() -> None:
    assert ages.value_ages(PROPERTIES, AT)["one_qubit"]["values"] == 3


def test_a_chain_gets_only_its_qubits_and_the_gates_inside_it() -> None:
    found = ages.value_ages(PROPERTIES, AT, qubits=[1, 2])

    assert found["t1"]["values"] == 2
    assert found["one_qubit"] == {"values": 2, "youngest_h": 2, "median_h": 76, "oldest_h": 150}
    assert found["two_qubit"] == {"values": 1, "youngest_h": 120, "median_h": 120, "oldest_h": 120}


def test_the_calibration_age_comes_from_the_calibration_time() -> None:
    assert ages.calibration_age(PROPERTIES, AT) == 1
    assert ages.calibration_age({}, AT) is None


def test_a_kind_with_no_dated_value_gives_none() -> None:
    found = ages.value_ages({"qubits": [[{"name": "T1", "value": 1.0}]], "gates": []}, AT)

    assert found == dict.fromkeys(ages.KINDS)


def test_a_value_measured_after_the_reference_time_has_a_negative_age() -> None:
    later = {"qubits": [[_value("T1", -3)]], "gates": []}

    assert ages.value_ages(later, AT)["t1"]["youngest_h"] == pytest.approx(-3)
