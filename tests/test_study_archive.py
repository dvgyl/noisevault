from __future__ import annotations

import json
import math
import statistics
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import require

from study import archive

pa = require("pyarrow")
pq = require("pyarrow.parquet")

T0 = datetime(2026, 3, 1, tzinfo=UTC)
END = T0 + timedelta(hours=48)
LN2 = math.log(2)
# The readout values of 11 qubits on 2026-03-05 and on the day after, in units of 1/4096.
NOW = [0, *range(11, 20)]
NEXT = [20, *range(12, 21), 30]
HIGH = 0.05


def _row(
    prop: str,
    value: float,
    hours: float,
    qubit_a: int = 0,
    qubit_b: int | None = None,
    device: str = "ibm_fez",
    seen_after_h: float = 0.5,
) -> dict[str, Any]:
    when = T0 + timedelta(hours=hours)
    return {
        "backend": device,
        "property": prop,
        "qubit_a": qubit_a,
        "qubit_b": None if qubit_b is None else float(qubit_b),
        "value": value,
        "unit": "",
        "calibrated_time": when,
        "observed_time": when + timedelta(hours=seen_after_h),
    }


def _rows() -> list[dict[str, Any]]:
    rows = [
        # Re-stamps of one sx error, then a change 24 hours after the first value.
        *(_row("sx_gate_error", 0.001, h) for h in (0, 1, 2)),
        *(_row("sx_gate_error", 0.002, h) for h in (24, 25)),
        # A T1 that is 240 hours old at the start of the grid, with a row that a later poll
        # replaces at the same time.
        _row("T1", 1.0, -240, seen_after_h=0.5),
        _row("T1", 100e-6, -240, seen_after_h=1.0),
        _row("T1", 50e-6, 48),
        _row("prob_meas0_prep1", 0.01, 0),
        _row("cz_gate_error", 0.004, 0, qubit_a=0, qubit_b=1),
        _row("cz_gate_error", 0.004, 1, qubit_a=0, qubit_b=1),
        _row("readout_length", 1e-6, 0),
        _row("T1", 80e-6, 0, device="ibm_torino"),
    ]
    day = 4 * 24 + 12
    for q, (now, after) in enumerate(zip(NOW + [None], NEXT, strict=True)):
        rows.append(_row("prob_meas1_prep0", HIGH if now is None else now / 4096, day, q))
        rows.append(_row("prob_meas1_prep0", after / 4096, day + 24, q))
    return rows


@pytest.fixture(scope="module")
def parquet(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("archive") / "train.parquet"
    pq.write_table(pa.Table.from_pylist(_rows()), path)
    return path


@pytest.fixture(scope="module")
def data(parquet: Path) -> dict[str, Any]:
    return archive.analyze(parquet, T0, END)


def test_reads_only_the_kinds_and_excludes_the_retired_device(
    data: dict[str, Any], parquet: Path
) -> None:
    assert data["devices"] == ["ibm_fez"]
    assert data["rows"] == 5 + 2 + 1 + 2 + 22
    assert data["duplicate_times"] == {"rows": 1, "different_value": 1}
    assert archive.analyze(parquet, T0, END, include_retired=True)["devices"] == [
        "ibm_fez",
        "ibm_torino",
    ]


def test_counts_a_change_only_when_the_value_changes(data: dict[str, Any]) -> None:
    sx = data["intervals"]["ibm_fez"]["one_qubit"]
    assert sx["between_changes_h"] == {"count": 1, "median": 24.0, "p90": 24.0}
    assert sx["between_times_h"]["count"] == 4
    assert sx["between_times_h"]["median"] == 1.0
    assert sx["restamps"] == {"rows": 5, "restamps": 3, "fraction": 0.6}
    cz = data["intervals"]["ibm_fez"]["two_qubit"]
    assert cz["between_changes_h"]["count"] == 0
    assert cz["restamps"]["restamps"] == 1
    t1 = data["intervals"]["ibm_fez"]["t1"]
    assert t1["between_changes_h"] == {"count": 1, "median": 288.0, "p90": 288.0}
    readout = data["intervals_by_property"]["ibm_fez"]["prob_meas1_prep0"]
    assert readout["between_changes_h"] == {"count": 11, "median": 24.0, "p90": 24.0}


def test_gives_the_time_since_the_last_change_at_each_grid_time(data: dict[str, Any]) -> None:
    since = data["since_change"]["ibm_fez"]
    # The grid holds 9 times, 0 to 48 hours after T0. The T1 changes at -240 h and at 48 h.
    t1 = since["t1"]
    assert t1["count"] == 9
    assert t1["median"] == 258.0
    assert t1["above_24h"] == pytest.approx(8 / 9)
    assert t1["above_168h"] == pytest.approx(8 / 9)
    # The sx error changes at 0 h and at 24 h. The re-stamps do not reset the time.
    sx = since["one_qubit"]
    assert sx["count"] == 9
    assert sx["median"] == 12.0
    assert sx["above_24h"] == 0.0
    # The readout series start after the grid, but prob_meas0_prep1 starts at T0.
    assert data["since_change_by_property"]["ibm_fez"]["prob_meas0_prep1"]["count"] == 9
    assert "prob_meas1_prep0" not in data["since_change_by_property"]["ibm_fez"]


def test_gives_the_size_of_a_change_by_its_gap(data: dict[str, Any]) -> None:
    size = data["change_size"]
    assert size["one_qubit"] == {"12-36 h": {"count": 1, "median": LN2, "p90": LN2}}
    assert size["t1"][">7 d"]["median"] == pytest.approx(LN2)
    # The zero has no ratio, so 10 of the 11 readout changes count.
    logs = [abs(math.log(b / a)) for a, b in zip(NOW[1:], NEXT[1:], strict=False)]
    logs.append(abs(math.log(NEXT[-1] / 4096 / HIGH)))
    readout = size["readout"]["12-36 h"]
    assert readout["count"] == 10
    assert readout["median"] == pytest.approx(statistics.median(logs))
    assert "two_qubit" not in size


def test_finds_the_zeros_and_the_grid_of_the_readout_values(data: dict[str, Any]) -> None:
    p10 = data["readout_grid"]["ibm_fez"]["prob_meas1_prep0"]
    assert p10["values"] == 22
    assert p10["zero_fraction"] == pytest.approx(1 / 22)
    multiples = p10["multiple_of"]
    assert multiples["1/4096"] == pytest.approx(21 / 22)
    assert multiples["1/8192"] == pytest.approx(21 / 22)
    assert multiples["1/2048"] == pytest.approx(12 / 22)
    assert multiples["1/1024"] == pytest.approx(7 / 22)
    assert multiples["1/1000"] == pytest.approx(2 / 22)
    assert multiples["1/10000"] == pytest.approx(2 / 22)
    p01 = data["readout_grid"]["ibm_fez"]["prob_meas0_prep1"]
    assert p01["zero_fraction"] == 0.0


def test_shows_the_regression_to_the_mean_of_the_extreme_changes(data: dict[str, Any]) -> None:
    parts = data["regression"]["prob_meas1_prep0"]["ibm_fez"]
    assert parts["lowest tenth"] == {
        "count": 1,
        "mean_now": 0.0,
        "median_now": 0.0,
        "mean_next": 20 / 4096,
        "median_next": 20 / 4096,
    }
    assert parts["highest tenth"]["mean_now"] == HIGH
    assert parts["highest tenth"]["mean_next"] == 30 / 4096
    middle = parts["middle"]
    assert middle["count"] == 9
    assert middle["mean_now"] == pytest.approx(15 / 4096)
    assert middle["median_next"] == pytest.approx(16 / 4096)
    assert data["regression"]["cz_gate_error"] == {}


def test_ties_share_the_lowest_percent_rank() -> None:
    rank = archive._percent_rank(
        np.zeros(5, dtype=np.int64), np.array([0, 0, 0, 0, 1]), np.array([3.0, 1.0, 1.0, 2.0, 9.0])
    )
    assert rank.tolist() == [1.0, 0.0, 0.0, 2 / 3, 0.0]


def test_writes_the_json_and_the_tables(parquet: Path, tmp_path: Path) -> None:
    out = tmp_path / "results"
    assert (
        archive.main(
            [str(parquet), "--out", str(out), "--start", "2026-03-01", "--end", "2026-03-03"]
        )
        == 0
    )
    data = json.loads((out / "archive.json").read_text(encoding="utf-8"))
    text = (out / "archive.md").read_text(encoding="utf-8")
    assert data["file_sha256"] in text
    assert "phanerozoic/qiskit-calibration-drift (CC-BY-4.0)" in text
    assert "2026-03-01T00:00:00Z to 2026-03-03T00:00:00Z" in text
    assert "| ibm_fez | one_qubit | 1 | 24.0 | 24.0 | 4 | 1.0 |" in text
