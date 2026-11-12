from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from typing import Any

import pytest
from conftest import toy
from typer.testing import CliRunner

import noisevault as nv
from noisevault.cli import app
from noisevault.sources import Origin, ibm_public
from noisevault.sources.qiskit_backend import calibration_from_properties, to_profile

ORIGIN = Origin("test properties")
CALIBRATED = "2026-11-12T04:00:00Z"
runner = CliRunner()


def _p(name: str, value: float, date: Any = None, unit: str = "") -> dict[str, Any]:
    param: dict[str, Any] = {"name": name, "value": value, "unit": unit}
    if date is not None:
        param["date"] = date
    return param


def _qubit(t1_date: str, readout_date: str) -> list[dict[str, Any]]:
    return [
        _p("T1", 100.0, t1_date, "us"),
        _p("T2", 80.0, t1_date, "us"),
        _p("readout_error", 0.02, readout_date),
        _p("prob_meas1_prep0", 0.01, readout_date),
        _p("prob_meas0_prep1", 0.03, readout_date),
        _p("readout_length", 1600, readout_date, "ns"),
    ]


def _gate(gate: str, qubits: list[int], error: float, length: float, date: str) -> dict:
    return {
        "gate": gate,
        "qubits": qubits,
        "parameters": [_p("gate_error", error, date), _p("gate_length", length, date, "ns")],
    }


# T1 and T2 are 2 h and 10 h old, readout 50 h and 52 h, sx 100 h and 300 h, cz 30 h. The
# virtual rz has the calibration time, and its error of 0 is not a measured value.
PROPERTIES: dict[str, Any] = {
    "backend_name": "ibm_test",
    "last_update_date": CALIBRATED,
    "qubits": [
        _qubit("2026-11-12T02:00:00Z", "2026-11-10T02:00:00Z"),
        _qubit("2026-11-11T18:00:00Z", "2026-11-10T00:00:00Z"),
    ],
    "gates": [
        _gate("sx", [0], 2e-4, 24, "2026-11-08T00:00:00Z"),
        _gate("sx", [1], 3e-4, 24, "2026-10-30T16:00:00Z"),
        _gate("rz", [0], 0, 0, CALIBRATED),
        _gate("rz", [1], 0, 0, CALIBRATED),
        _gate("cz", [0, 1], 4e-3, 84, "2026-11-10T22:00:00Z"),
    ],
}


def _profile(props: dict[str, Any] = PROPERTIES) -> nv.Profile:
    cal = calibration_from_properties(props, origin=ORIGIN)
    return to_profile(cal, {"source_kind": "other"}, origin=ORIGIN)


def _without_dates(props: dict[str, Any]) -> dict[str, Any]:
    props = copy.deepcopy(props)
    for params in [*props["qubits"], *(g["parameters"] for g in props["gates"])]:
        for param in params:
            param.pop("date", None)
    return props


def test_the_profile_keeps_the_oldest_median_and_newest_time_of_each_kind() -> None:
    times = _profile().provenance.extra["value_times"]
    assert times == {
        "readout": {
            "oldest": "2026-11-10T00:00:00Z",
            "median": "2026-11-10T01:00:00Z",
            "newest": "2026-11-10T02:00:00Z",
            "values": 6,
        },
        "t1": {
            "oldest": "2026-11-11T18:00:00Z",
            "median": "2026-11-11T22:00:00Z",
            "newest": "2026-11-12T02:00:00Z",
            "values": 2,
        },
        "t2": {
            "oldest": "2026-11-11T18:00:00Z",
            "median": "2026-11-11T22:00:00Z",
            "newest": "2026-11-12T02:00:00Z",
            "values": 2,
        },
        "error_1q": {
            "oldest": "2026-10-30T16:00:00Z",
            "median": "2026-11-03T20:00:00Z",
            "newest": "2026-11-08T00:00:00Z",
            "values": 2,
        },
        "error_2q": {
            "oldest": "2026-11-10T22:00:00Z",
            "median": "2026-11-10T22:00:00Z",
            "newest": "2026-11-10T22:00:00Z",
            "values": 1,
        },
    }


def test_value_ages_are_hours_before_the_calibration_time_by_default() -> None:
    ages = _profile().value_ages()
    assert ages is not None
    assert {kind: age["median_h"] for kind, age in ages.items()} == {
        "readout": 51,
        "t1": 6,
        "t2": 6,
        "error_1q": 200,
        "error_2q": 30,
    }
    assert ages["error_1q"] == {"oldest_h": 300, "median_h": 200, "newest_h": 100, "values": 2}


def test_value_ages_take_another_reference_time() -> None:
    profile = _profile()
    later = profile.value_ages("2026-11-13T04:00:00Z")
    assert later is not None and later["t1"]["newest_h"] == 26
    assert profile.value_ages(datetime(2026, 11, 12, 6, tzinfo=UTC))["t1"]["newest_h"] == 4


def test_one_note_names_each_kind_with_a_median_age_above_a_day() -> None:
    notes = [n for n in _profile().provenance.notes if "older than the calibration" in n]
    assert notes == [
        "Some values are older than the calibration time by more than 24 h (median):"
        " readout 51 h, 1q gates 200 h, 2q gates 30 h. The calibration time is the last update"
        " of the calibration, not the measurement time of each value."
    ]


def test_no_note_when_every_median_age_is_a_day_or_less() -> None:
    props = copy.deepcopy(PROPERTIES)
    props["last_update_date"] = "2026-11-11T00:00:00Z"
    for gate in props["gates"]:
        for param in gate["parameters"]:
            param["date"] = "2026-11-10T12:00:00Z"
    profile = _profile(props)
    assert profile.value_ages() is not None
    assert not any("older than the calibration" in n for n in profile.provenance.notes)


def test_a_reply_with_no_dates_gives_no_times_and_no_note() -> None:
    profile = _profile(_without_dates(PROPERTIES))
    assert "value_times" not in profile.provenance.extra
    assert profile.value_ages() is None
    assert not any("older than the calibration" in n for n in profile.provenance.notes)


def test_a_kind_with_no_dated_values_is_left_out() -> None:
    props = copy.deepcopy(PROPERTIES)
    for gate in props["gates"]:
        for param in gate["parameters"]:
            param.pop("date")
    ages = _profile(props).value_ages()
    assert ages is not None and set(ages) == {"readout", "t1", "t2"}


def test_the_fingerprint_does_not_depend_on_the_dates() -> None:
    dated, undated = _profile(), _profile(_without_dates(PROPERTIES))
    assert dated.fingerprint == undated.fingerprint
    assert dated.provenance != undated.provenance


def test_a_datetime_with_an_offset_counts_in_utc() -> None:
    props = copy.deepcopy(PROPERTIES)
    props["gates"][4]["parameters"][0]["date"] = datetime.fromisoformat("2026-11-10T17:00-05:00")
    assert _profile(props).value_ages()["error_2q"]["median_h"] == 30


@pytest.mark.parametrize("date", ["yesterday", 1731384000, True, "2026-13-01T00:00:00Z"])
def test_a_date_that_is_not_an_iso_time_is_refused(date: Any) -> None:
    props = copy.deepcopy(PROPERTIES)
    props["qubits"][1][0]["date"] = date
    with pytest.raises(nv.SourceDataError) as info:
        _profile(props)
    assert info.value.message == (
        f"test properties: T1 of qubit 1 has the date {date!r}, not an ISO 8601 time"
    )


def test_the_public_endpoint_refuses_a_bad_date_as_a_reply_of_the_wrong_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    props = copy.deepcopy(PROPERTIES)
    props["gates"][0]["parameters"][0]["date"] = "yesterday"
    monkeypatch.setattr(ibm_public, "fetch", lambda url: json.dumps(props).encode())
    with pytest.raises(nv.SourceUnavailable) as info:
        ibm_public.pull("ibm_test")
    assert "with JSON of the wrong shape at gates[0].parameters[0].date" in info.value.message


def test_the_public_endpoint_keeps_the_times(monkeypatch: pytest.MonkeyPatch) -> None:
    def fetch(url: str) -> bytes:
        if url.endswith("/properties"):
            return json.dumps(PROPERTIES).encode()
        raise ibm_public._NotFound(url)

    monkeypatch.setattr(ibm_public, "fetch", fetch)
    profile = ibm_public.pull("ibm_test")
    assert profile.provenance.extra["value_times"] == _profile().provenance.extra["value_times"]


def test_a_profile_without_times_or_with_a_damaged_entry() -> None:
    plain = nv.Profile.from_dict(toy())
    assert plain.value_ages() is None
    data = _profile().to_dict()
    data["provenance"]["extra"]["value_times"]["t1"]["median"] = "noon"
    data["provenance"]["extra"]["value_times"]["t2"] = "old"
    ages = nv.Profile.from_dict(data).value_ages()
    assert ages is not None and set(ages) == {"readout", "error_1q", "error_2q"}


def test_an_undated_profile_needs_a_reference_time() -> None:
    data = _profile().to_dict()
    data["device"]["calibrated_at"] = None
    profile = nv.Profile.from_dict(data)
    with pytest.raises(ValueError, match="no calibration time"):
        profile.value_ages()
    assert profile.value_ages("2026-11-12T04:00:00Z") == _profile().value_ages()


def test_nv_show_prints_the_median_ages_and_json_has_the_numbers(tmp_path) -> None:
    path = _profile().save(tmp_path / "ibm_test.json")
    out = runner.invoke(app, ["show", str(path)], env={"COLUMNS": "80"}).stdout
    lines = [line.rstrip() for line in out.splitlines()]
    first = lines.index(
        "value ages    readout 51 h, T1 6.0 h, T2 6.0 h, 1q gates 200 h, 2q gates 30 h"
    )
    assert lines[first + 1] == "              older than the calibration time (median)"
    assert " ".join(out.split()).count("Some values are older than the calibration time") == 1
    data = json.loads(runner.invoke(app, ["show", str(path), "--json"]).stdout)
    assert data["value_ages"]["error_1q"] == {
        "oldest_h": 300,
        "median_h": 200,
        "newest_h": 100,
        "values": 2,
    }


def test_nv_show_has_no_age_row_for_a_profile_without_times(tmp_path) -> None:
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(toy()))
    out = runner.invoke(app, ["show", str(path)]).stdout
    assert "value ages" not in out
    data = json.loads(runner.invoke(app, ["show", str(path), "--json"]).stdout)
    assert data["value_ages"] is None
