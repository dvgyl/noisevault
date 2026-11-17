from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from conftest import require

from study import archive_zeros

pa = require("pyarrow")
pq = require("pyarrow.parquet")

T0 = datetime(2026, 3, 1, tzinfo=UTC)
END = T0 + timedelta(days=2)
P10, P01 = "prob_meas1_prep0", "prob_meas0_prep1"


def _row(hours: float, qubit: int, prop: str, value: float, device: str = "ibm_fez") -> dict:
    when = T0 + timedelta(hours=hours)
    return {
        "backend": device,
        "property": prop,
        "qubit_a": qubit,
        "value": value,
        "calibrated_time": when - timedelta(minutes=20),
        "observed_time": when,
    }


def _rows() -> list[dict[str, Any]]:
    return [
        # Observation A of ibm_fez: 4 readout rows, 3 zeros, 1 of P(1|0), on 2 qubits.
        _row(1, 0, P10, 0.0),
        _row(1, 0, P01, 0.0),
        _row(1, 1, P10, 0.01),
        _row(1, 2, P01, 0.0),
        _row(1, 5, "T1", 0.0),
        # Observation B, on the next day: 2 rows, 1 zero of P(1|0).
        _row(25, 1, P10, 0.02),
        _row(25, 3, P10, 0.0),
        # Observation C, in the last hour of the last day.
        _row(71.5, 0, P10, 0.0),
        # Outside the window, and a retired device.
        _row(-1, 0, P10, 0.0),
        _row(72, 0, P10, 0.0),
        _row(3, 0, P10, 0.0, "ibm_torino"),
        # The one observation of ibm_kingston, with no zero.
        _row(2, 0, P10, 0.03, "ibm_kingston"),
    ]


@pytest.fixture(scope="module")
def parquet(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("archive_zeros") / "train.parquet"
    pq.write_table(pa.Table.from_pylist(_rows()), path)
    return path


@pytest.fixture(scope="module")
def found(parquet: Path) -> archive_zeros.Observations:
    return archive_zeros.observations(parquet, T0, END)


def test_each_observation_counts_its_rows_its_zeros_and_its_qubits(found) -> None:
    assert found.devices == ["ibm_fez", "ibm_kingston"]
    assert found.device.tolist() == [0, 0, 0, 1]
    assert found.values.tolist() == [4, 2, 1, 1]
    assert found.zeros.tolist() == [3, 1, 1, 0]
    assert found.zeros_p10.tolist() == [1, 1, 1, 0]
    assert found.qubits.tolist() == [2, 1, 1, 0]
    # The 0 of qubit 0 from before the window is in force at A; the 0 of qubit 3 stays in force.
    assert found.in_force.tolist() == [3, 4, 4, 0]


def test_the_statistics_of_a_device_are_over_its_observations(found) -> None:
    fez, kingston = archive_zeros.device_statistics(found).values()
    assert fez["observations"] == 3 and fez["median_values"] == 2
    assert fez["mean_zeros"] == pytest.approx(5 / 3)
    assert (fez["median_zeros"], fez["low_tenth"], fez["most"]) == (1, 1, 3)
    assert fez["high_tenth"] == pytest.approx(2.6)
    assert fez["mean_zeros_p10"] == 1 and fez["share_with_a_zero"] == 1
    assert fez["median_qubits"] == 1
    assert kingston["share_with_a_zero"] == 0


def test_the_months_and_the_last_days_use_utc(found) -> None:
    month = archive_zeros.by_period(found, "month")
    assert list(month) == ["2026-03"]
    assert month["2026-03"]["observations"] == 4
    assert month["2026-03"]["devices"]["ibm_fez"]["mean"] == pytest.approx(5 / 3)
    days = archive_zeros.by_period(archive_zeros.last_days(found, END, 2), "day")
    assert list(days) == ["2026-03-02", "2026-03-03"]
    assert days["2026-03-03"]["devices"] == {
        "ibm_fez": {"mean": 1.0, "most": 1, "mean_in_force": 4.0}
    }


def test_a_zero_that_is_not_stated_again_stays_in_force(tmp_path: Path) -> None:
    rows = [
        _row(-5, 2, P10, 0.0),
        _row(1, 0, P10, 0.0),
        _row(1, 1, P10, 0.01),
        _row(2, 1, P10, 0.02),
        _row(3, 0, P10, 0.05),
    ]
    path = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    found = archive_zeros.observations(path, T0, END)
    assert found.zeros.tolist() == [1, 0, 0]
    assert found.in_force.tolist() == [2, 2, 1]
    fez = archive_zeros.device_statistics(found)["ibm_fez"]
    assert (fez["mean_zeros"], fez["share_with_a_zero"]) == (
        pytest.approx(1 / 3),
        pytest.approx(1 / 3),
    )
    assert (fez["mean_in_force"], fez["median_in_force"]) == (pytest.approx(5 / 3), 2)
    assert fez["share_with_a_zero_in_force"] == 1
    month = archive_zeros.by_period(found, "month")["2026-03"]["devices"]["ibm_fez"]
    assert month["mean_in_force"] == pytest.approx(5 / 3)


def test_a_row_with_no_value_is_not_a_row_of_an_observation(tmp_path: Path) -> None:
    rows = [_row(1, 0, P10, 0.0), _row(1, 1, P10, math.nan), _row(2, 1, P10, math.nan)]
    path = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    found = archive_zeros.observations(path, T0, END)
    assert (found.zeros.tolist(), found.in_force.tolist()) == ([1], [1])


def test_the_report_gives_each_device(parquet: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = [str(parquet), "--start", "2026-03-01", "--end", "2026-03-03"]
    assert archive_zeros.main(argv) == 0
    out = capsys.readouterr().out
    assert "ibm_fez: 3 observations, 2 readout values in an observation" in out
    assert "  2026-03 1.67 0.00 4" in out
    assert "  ibm_fez 2026-03-03 1.0 1" in out
    assert "ibm_torino" not in out
    assert "ibm_fez: mean 3.67, median 4; observations with a zero in force: 100.0%" in out
    assert out.rstrip().endswith("  2026-03 3.67 0.00 4")


def test_an_end_before_the_start_is_refused(parquet: Path) -> None:
    assert archive_zeros.main([str(parquet), "--start", "2026-03-03", "--end", "2026-03-01"]) == 2
