from __future__ import annotations

import copy
import csv
import io
import json
import urllib.parse
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from conftest import require
from typer.testing import CliRunner

import noisevault as nv
from noisevault import catalog, series
from noisevault.cli import app
from noisevault.sources import ibm_public
from noisevault.sources.qiskit_backend import as_utc

FIXTURES = Path(__file__).parent / "fixtures"
PROPERTIES = json.loads((FIXTURES / "ibm" / "manila_properties.json").read_bytes())
CONFIGURATION = (FIXTURES / "ibm" / "manila_configuration.json").read_bytes()
ARCHIVE = FIXTURES / "hf_archive" / "train-00000-of-00001.parquet"
LISTING = json.dumps([{"name": "ibm_manila"}, {"name": "ibm_fez"}]).encode()
PROPERTIES_URL = f"{ibm_public.BASE_URL}/ibm_manila/properties"
CALIBRATED = [datetime(2026, 11, day, 6, tzinfo=UTC) for day in (1, 2, 3)]
RANGE = ["ibm_manila", "--from", "2026-11-01T12:00", "--to", "2026-11-03T12:00", "--step", "12h"]
STEP_TIMES = [datetime(2026, 11, 1, 12, tzinfo=UTC) + n * timedelta(hours=12) for n in range(5)]

runner = CliRunner()


def _properties(when: datetime) -> bytes:
    """The manila fixture, calibrated at ``when``, with a T1 of qubit 0 that names the day."""
    props = copy.deepcopy(PROPERTIES)
    props["last_update_date"] = when.isoformat()
    t1 = next(p for p in props["qubits"][0] if p["name"] == "T1")
    t1["value"] = 100.0 + when.day
    return json.dumps(props).encode()


class Endpoint:
    """A fake of IBM's public endpoint that serves the calibrations of CALIBRATED."""

    def __init__(self) -> None:
        self.requested: list[datetime] = []
        self.fails_at: datetime | None = None

    def fetch(self, url: str) -> bytes:
        if url == ibm_public.BASE_URL:
            return LISTING
        if url == f"{ibm_public.BASE_URL}/ibm_manila/configuration":
            return CONFIGURATION
        if not url.startswith(PROPERTIES_URL):
            raise ibm_public._NotFound(url)
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        before = as_utc(query["updated_before"][0])
        self.requested.append(before)
        if before == self.fails_at:
            raise nv.SourceUnavailable(f"IBM's public endpoint answered HTTP 503 for {url}")
        shown = [when for when in CALIBRATED if when < before]
        if not shown:
            raise ibm_public._NotFound(url)
        return _properties(shown[-1])


@pytest.fixture
def endpoint(monkeypatch: pytest.MonkeyPatch) -> Endpoint:
    fake = Endpoint()
    monkeypatch.setattr(ibm_public, "fetch", fake.fetch)
    monkeypatch.setattr(series, "sleep", lambda seconds: None)
    return fake


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def test_steps_that_give_one_calibration_give_one_calibration(endpoint: Endpoint) -> None:
    found = nv.history(
        "ibm_manila", "2026-11-01T12:00", "2026-11-03T12:00", step=timedelta(hours=12)
    )
    assert endpoint.requested == STEP_TIMES
    assert [p.device.calibrated_at for p in found.profiles] == CALIBRATED
    t1 = [r for r in found.rows if r.kind == "t1" and r.qubit_a == 0]
    assert [(r.calibrated_at, r.value) for r in t1] == [
        (when, 100.0 + when.day) for when in CALIBRATED
    ]
    assert len({r.fingerprint for r in found.rows}) == 3


def test_the_wait_comes_between_two_requests(
    endpoint: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    waits: list[float] = []
    monkeypatch.setattr(series, "sleep", waits.append)
    nv.history("ibm_manila", "2026-11-01T12:00", "2026-11-03T12:00", step=timedelta(hours=12))
    assert waits == [series.PAUSE_S] * 4
    assert series.PAUSE_S > 0


def test_the_table_has_the_columns_and_units_of_each_kind(endpoint: Endpoint) -> None:
    rows = nv.history("ibm_manila", "2026-11-03T12:00", "2026-11-03T12:00").to_csv()
    assert rows.splitlines()[0] == ",".join(series.COLUMNS)
    table = _rows(rows)
    units = {(r["kind"], r["unit"]) for r in table}
    assert units == {
        ("t1", "us"),
        ("t2", "us"),
        ("p1_given_0", ""),
        ("p0_given_1", ""),
        ("gate_error", ""),
        ("gate_duration", "ns"),
    }
    by_key = {(r["kind"], r["gate"], r["qubit_a"], r["qubit_b"]): r for r in table}
    raw = {p["name"]: p["value"] for p in PROPERTIES["qubits"][1]}
    assert float(by_key["t1", "", "1", ""]["value"]) == pytest.approx(raw["T1"], rel=1e-9)
    assert float(by_key["t2", "", "1", ""]["value"]) == pytest.approx(raw["T2"], rel=1e-9)
    assert float(by_key["p1_given_0", "", "1", ""]["value"]) == pytest.approx(
        raw["prob_meas1_prep0"], rel=1e-9
    )
    assert float(by_key["p0_given_1", "", "1", ""]["value"]) == pytest.approx(
        raw["prob_meas0_prep1"], rel=1e-9
    )
    gate = next(g for g in PROPERTIES["gates"] if g["gate"] == "sx" and g["qubits"] == [2])
    values = {p["name"]: p["value"] for p in gate["parameters"]}
    assert float(by_key["gate_error", "sx", "2", ""]["value"]) == pytest.approx(
        values["gate_error"], rel=1e-9
    )
    assert float(by_key["gate_duration", "sx", "2", ""]["value"]) == pytest.approx(
        values["gate_length"]
    )
    pair = next(g for g in PROPERTIES["gates"] if g["gate"] == "cx" and g["qubits"] == [0, 1])
    error = next(p["value"] for p in pair["parameters"] if p["name"] == "gate_error")
    assert float(by_key["gate_error", "cx", "0", "1"]["value"]) == pytest.approx(error, rel=1e-9)
    assert {r["calibrated_at"] for r in table} == {"2026-11-03T06:00:00Z"}
    assert "value_time" not in table[0]


def test_a_range_of_more_than_400_steps_is_refused_before_any_request(
    endpoint: Endpoint,
) -> None:
    with pytest.raises(series.HistoryError) as info:
        nv.history("ibm_manila", "2026-01-01", "2027-02-05")
    assert info.value.message.endswith("has 401 steps of 1 day, and the limit is 400")
    assert info.value.hint == "use a step of 2 days or more"
    assert endpoint.requested == []
    start = datetime(2026, 1, 1, tzinfo=UTC)
    assert len(series.steps(start, start + timedelta(days=399), timedelta(days=1))) == 400


def test_a_failed_step_names_its_date_and_writes_no_file(
    endpoint: Endpoint, tmp_path: Path
) -> None:
    endpoint.fails_at = STEP_TIMES[2]
    out = tmp_path / "table.csv"
    result = runner.invoke(app, ["history", *RANGE, "-o", str(out)])
    assert result.exit_code == 1
    assert result.stderr.splitlines()[0].startswith(
        "error: step 3 of 5 (2026-11-02T12:00:00Z) failed: IBM's public endpoint answered HTTP 503"
    )
    assert result.stdout == ""
    assert not out.exists()
    assert endpoint.requested == STEP_TIMES[:3]


def test_a_device_the_endpoint_does_not_list_gives_one_error_with_a_hint(
    endpoint: Endpoint,
) -> None:
    result = runner.invoke(
        app, ["history", "ibm_gone", "--from", "2026-03-01", "--to", "2026-03-02"]
    )
    assert result.exit_code == 1
    error, hint = result.stderr.splitlines()
    assert error.startswith("error: step 1 of 2 (2026-03-01T00:00:00Z) failed: ibm_gone is not")
    assert error.endswith("the device may be retired")
    assert hint == (
        "hint: the public endpoint has no data for a retired device; use --source offline for the"
        " profiles that load offline, or --source archive for an archive file"
    )


def test_a_range_before_the_first_calibration_says_to_start_later(endpoint: Endpoint) -> None:
    with pytest.raises(series.StepFailed) as info:
        nv.history("ibm_manila", "2026-10-30", "2026-11-02")
    assert info.value.at == datetime(2026, 10, 30, tzinfo=UTC)
    assert info.value.hint == "start the range at a later date"


def _saved(when: datetime) -> nv.Profile:
    data = nv.load("ibm_manila").to_dict()
    data["device"]["calibrated_at"] = when.isoformat()
    data["qubits"][0]["t1_us"] = 100.0 + when.day
    profile = nv.Profile.model_validate(data)
    profile.save(catalog.vault_path(profile))
    return profile


def test_the_vault_source_reads_each_profile_in_the_range_with_no_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_network(url: str) -> bytes:
        raise AssertionError(f"requested {url}")

    monkeypatch.setattr(ibm_public, "fetch", no_network)
    inside = [_saved(datetime(2026, 5, day, 9, tzinfo=UTC)) for day in (3, 4)]
    _saved(datetime(2026, 6, 1, tzinfo=UTC))
    found = nv.history("ibm_manila", "2026-05-01", "2026-05-31", source="offline")
    assert [p.fingerprint for p in found.profiles] == [p.fingerprint for p in inside]
    assert found.step is None
    bundled = nv.history("ibm_manila", "2024-01-01", "2026-05-03T09:00", source="offline")
    assert [p.device.calibrated_at.date().isoformat() for p in bundled.profiles] == [
        "2024-05-27",
        "2026-05-03",
    ]


def test_the_vault_source_names_the_times_it_holds_when_none_is_in_the_range() -> None:
    with pytest.raises(series.HistoryError) as info:
        nv.history("ibm_manila", "2026-05-01", "2026-05-31", source="offline")
    assert info.value.hint == (
        "the only calibration time of ibm_manila that loads offline is 2024-05-27T18:27:23Z;"
        " run nv list to see them"
    )


def test_the_archive_source_gives_the_distinct_calibrations_of_each_step() -> None:
    require("pyarrow")
    found = nv.history(
        "ibm_fez",
        "2026-02-20",
        "2026-11-10",
        step=timedelta(days=30),
        source="archive",
        archive=ARCHIVE,
    )
    times = [p.device.calibrated_at for p in found.profiles]
    assert times == sorted(set(times))
    expected = {
        nv.from_calibration_archive(ARCHIVE, "ibm_fez", at=at).device.calibrated_at
        for at in series.steps(as_utc("2026-02-20"), as_utc("2026-11-10"), timedelta(days=30))
    }
    assert set(times) == expected and len(times) > 1


def test_the_command_prints_the_csv_on_standard_output(endpoint: Endpoint) -> None:
    result = runner.invoke(app, ["history", *RANGE])
    assert result.exit_code == 0, result.stderr
    assert result.stderr == ""
    expected = nv.history(
        "ibm_manila", "2026-11-01T12:00", "2026-11-03T12:00", step=timedelta(hours=12)
    )
    assert result.stdout == expected.to_csv()


def test_the_command_prints_json_with_json(endpoint: Endpoint) -> None:
    result = runner.invoke(app, ["history", *RANGE, "--json"])
    assert result.exit_code == 0, result.stderr
    data: dict[str, Any] = json.loads(result.stdout)
    assert data["source"] == "ibm" and data["step_seconds"] == 43200
    assert [c["calibrated_at"] for c in data["calibrations"]] == [
        "2026-11-01T06:00:00Z",
        "2026-11-02T06:00:00Z",
        "2026-11-03T06:00:00Z",
    ]
    assert data["columns"] == list(series.COLUMNS)
    assert all(list(row) == data["columns"] for row in data["rows"])
    assert data["rows"][0]["gate"] is None and data["rows"][0]["qubit_b"] is None


def test_with_an_output_file_the_summary_goes_to_standard_output(
    endpoint: Endpoint, tmp_path: Path
) -> None:
    out = tmp_path / "table.csv"
    result = runner.invoke(app, ["history", *RANGE, "-o", str(out)])
    assert result.exit_code == 0, result.stderr
    assert result.stderr == ""
    lines = result.stdout.splitlines()
    assert lines[:2] == [
        "ibm_manila from 2026-11-01T12:00:00Z to 2026-11-03T12:00:00Z, every 12 hours (ibm)",
        f"3 calibrations, {len(_rows(out.read_text())):,} values",
    ]
    assert lines[-1] == f"saved: {out}"
    assert out.read_text().splitlines()[0] == ",".join(series.COLUMNS)


@pytest.mark.parametrize("step", ["0d", "1x", "d"])
def test_a_bad_step_is_a_usage_error(step: str) -> None:
    result = runner.invoke(app, ["history", *RANGE[:5], "--step", step])
    assert result.exit_code == 2
    assert result.stderr.startswith("error: invalid value for --step:")
