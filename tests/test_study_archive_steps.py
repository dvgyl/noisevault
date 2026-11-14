from __future__ import annotations

import math
import statistics
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from conftest import require

from study import archive, archive_steps

pa = require("pyarrow")
pq = require("pyarrow.parquet")

T0 = datetime(2026, 3, 1, tzinfo=UTC)
END = T0 + timedelta(days=4)
LN2 = math.log(2)
# The readout values of 20 qubits of ibm_fez at 36 h, 60 h and 84 h, in units of 1/4096. The
# order of the qubits stays the same from 36 h to 60 h, and reverses from 60 h to 84 h.
QUBITS = range(20)
READOUT_HOURS = (36, 60, 84)


def _readout(q: int) -> tuple[int, int, int]:
    return q + 1, q + 2, 41 - q


def _row(
    prop: str, value: float, hours: float, qubit: int = 0, device: str = "ibm_fez"
) -> dict[str, Any]:
    when = T0 + timedelta(hours=hours)
    return {
        "backend": device,
        "property": prop,
        "qubit_a": qubit,
        "qubit_b": None,
        "value": value,
        "unit": "",
        "calibrated_time": when,
        "observed_time": when + timedelta(minutes=30),
        "is_failure_ceiling": False,
    }


def _rows() -> list[dict[str, Any]]:
    rows = [
        # A T1 that changes before the window, at 36 h and at 100 h.
        _row("T1", 100.0, -12),
        _row("T1", 50.0, 36),
        _row("T1", 80.0, 100),
        # An sx error with a failure-ceiling value from 30 h to 50 h.
        _row("sx_gate_error", 0.001, 0),
        _row("sx_gate_error", 0.7, 30),
        _row("sx_gate_error", 0.002, 50),
        _row("sx_gate_error", 0.003, 100),
    ]
    for q in QUBITS:
        for hours, count in zip(READOUT_HOURS, _readout(q), strict=True):
            rows.append(_row("prob_meas1_prep0", count / 4096, hours, q))
    return rows


@pytest.fixture(scope="module")
def parquet(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("archive_steps") / "train.parquet"
    table = pa.Table.from_pylist(_rows())
    index = table.schema.get_field_index("qubit_b")
    pq.write_table(table.set_column(index, "qubit_b", table["qubit_b"].cast(pa.float64())), path)
    return path


@pytest.fixture(scope="module")
def rows(parquet: Path) -> archive.Rows:
    return archive.read_rows(parquet)


def _window_changes(rows: archive.Rows) -> archive.Rows:
    inside = (rows.time_us >= archive.micros_of(T0)) & (rows.time_us <= archive.micros_of(END))
    return rows.select(inside & rows.change)


def test_compares_the_value_in_force_on_a_day_with_the_value_days_later(
    rows: archive.Rows,
) -> None:
    found = {
        (r["kind"], r["step_days"]): r
        for r in archive_steps.fixed_steps(rows.select(rows.change), T0, END)
    }
    # The days are 0 h, 24 h, 48 h and 72 h. The T1 is 100 at 0 h and 24 h, and 50 after 36 h.
    assert found[("t1", 1)] == {
        "kind": "t1",
        "step_days": 1,
        "pairs": 4,
        "new_fraction": 0.25,
        "median_size": 0.0,
        "median_size_new": pytest.approx(LN2),
    }
    # From 0 h to 72 h and from 24 h to 96 h, the T1 goes from 100 to 50.
    assert found[("t1", 3)]["pairs"] == 2
    assert found[("t1", 3)]["new_fraction"] == 1.0
    assert found[("t1", 3)]["median_size"] == pytest.approx(LN2)
    # The pairs of the sx error at 24 h and 48 h for one day have the value 0.7, so they drop out.
    assert found[("one_qubit", 1)]["pairs"] == 2
    assert found[("one_qubit", 1)]["new_fraction"] == 0.0
    assert found[("one_qubit", 1)]["median_size_new"] is None
    assert found[("one_qubit", 3)]["pairs"] == 2
    assert found[("one_qubit", 3)]["median_size"] == pytest.approx(LN2)
    # A readout series counts only from 36 h to 84 h, so only the day at 48 h has a pair.
    assert found[("readout", 1)]["pairs"] == 20
    assert found[("readout", 1)]["new_fraction"] == 1.0
    # A step of 7 days is longer than the window.
    assert not [key for key in found if key[1] == 7]


def _file(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    path = tmp_path / "train.parquet"
    table = pa.Table.from_pylist(rows)
    index = table.schema.get_field_index("qubit_b")
    pq.write_table(table.set_column(index, "qubit_b", table["qubit_b"].cast(pa.float64())), path)
    return path


def test_a_value_of_0_stays_in_the_number_of_pairs_and_in_the_fraction(tmp_path: Path) -> None:
    # A readout error of 0 from 0 h, a new date with the same 0 at 24 h, and 1/4096 from 36 h.
    # The days 0 h, 24 h and 48 h give the pairs (0, 0) and (0, 1/4096). No pair has a size.
    values = [(0.0, 0), (0.0, 24), (1 / 4096, 36), (2 / 4096, 60)]
    path = _file(tmp_path, [_row("prob_meas1_prep0", v, hours) for v, hours in values])
    rows = archive.read_rows(path)
    found = archive_steps.fixed_steps(rows.select(rows.change), T0, T0 + timedelta(days=2), [1])
    assert found == [
        {
            "kind": "readout",
            "step_days": 1,
            "pairs": 2,
            "new_fraction": 0.5,
            "median_size": None,
            "median_size_new": None,
        }
    ]


def test_a_file_with_no_readout_value_still_gives_the_other_parts(tmp_path: Path) -> None:
    # The T1 is in force from 0 h to its last change at 30 h, so one pair exists: 0 h and 24 h.
    path = _file(tmp_path, [_row("T1", 100.0, 0), _row("T1", 110.0, 30)])
    text = archive_steps.report(path, T0, END)
    assert "t1, 1 d, 1, 0.000, 0.00%, -" in text
    assert "4. Minutes between two observation times of a device" in text


def test_gives_the_rank_correlation_of_each_day_with_20_pairs(rows: archive.Rows) -> None:
    found = archive_steps.day_correlations(_window_changes(rows))
    assert found == {"ibm_fez": {"readout": [pytest.approx(1.0), pytest.approx(-1.0)]}}


def test_gives_the_fixed_error_and_the_regression_of_the_archive(rows: archive.Rows) -> None:
    found = archive_steps.chance_regression(_window_changes(rows))
    # The mean of qubit q is (q + 44) / 3 and its median is q + 2, in units of 1/4096.
    assert found["fixed_error"]["mean"] == pytest.approx(53.5 / 3 / 4096)
    assert found["fixed_error"]["median"] == pytest.approx(11.5 / 4096)
    # Qubits 0 and 1 are the lowest tenth at 36 h and at 60 h.
    lowest = found["runs"]["archive"]["regression"]["ibm_fez"]["parts"]["lowest tenth"]
    assert lowest["count"] == 4
    assert lowest["median_now"] == pytest.approx(2 / 4096)
    assert lowest["median_next"] == pytest.approx(21.5 / 4096)
    sizes = [
        abs(math.log(b / a))
        for q in QUBITS
        for a, b in zip(_readout(q), _readout(q)[1:], strict=False)
    ]
    typical = found["runs"]["archive"]["typical_size"]
    assert typical == pytest.approx(statistics.median(sizes))


def test_chance_alone_with_many_shots_gives_no_regression(rows: archive.Rows) -> None:
    found = archive_steps.chance_regression(_window_changes(rows), shots=10**12)
    for name in ("mean", "median"):
        run = found["runs"][name]
        lowest = run["regression"]["ibm_fez"]["parts"]["lowest tenth"]
        assert lowest["count"] == 4
        assert lowest["median_next"] == pytest.approx(lowest["median_now"], rel=1e-3)
        assert run["typical_size"] < 1e-3


def test_gives_the_minutes_between_two_observation_times(tmp_path: Path) -> None:
    minutes = {"ibm_fez": [-10, 0, 30, 30, 90], "ibm_kingston": [5], "ibm_torino": [0, 1]}
    table = [
        {"backend": device, "observed_time": T0 + timedelta(minutes=m)}
        for device, times in minutes.items()
        for m in times
    ]
    path = tmp_path / "polls.parquet"
    pq.write_table(pa.Table.from_pylist(table), path)
    found = archive_steps.observation_times(path, T0, END)
    assert list(found) == ["ibm_fez", "ibm_kingston"]
    assert ((found["ibm_fez"] - archive.micros_of(T0)) / 6e7).tolist() == [0.0, 30.0, 90.0]
    assert found["ibm_kingston"].size == 1


def test_prints_the_four_parts(parquet: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert archive_steps.main([str(parquet), "--start", "2026-03-01", "--end", "2026-03-05"]) == 0
    text = capsys.readouterr().out
    assert f"file SHA-256 {archive.sha256(parquet)}" in text
    assert "t1, 1 d, 4, 0.250, 0.00%, 100.00%\n" in text
    assert "one_qubit, 1 d, 2, 0.000, 0.00%, -\n" in text
    assert "ibm_fez, readout, 2, 0.000, -0.800, 0.800\n" in text
    assert "range over the devices, readout: 0.00 to 0.00\n" in text
    assert (
        "archive, ibm_fez: 4 values, median 0.00049 (2.0 of 4096), median of the next value"
        " 0.00525 (21.5 of 4096), ratio 10.75\n" in text
    )
    assert "median over the series of the fixed error: mean 0.00435, median 0.00281\n" in text
    # Each row has a poll 30 minutes after its calibration time. The window holds the polls at
    # 0.5 h, 30.5 h, 36.5 h, 50.5 h, 60.5 h and 84.5 h.
    assert text.endswith(
        "ibm_fez: 6 observation times, median 840.0, lowest tenth 456.0, highest tenth 1656.0,"
        " share above 60 minutes 1.000\n"
    )


def test_refuses_a_missing_file(tmp_path: Path) -> None:
    assert archive_steps.main([str(tmp_path / "none.parquet")]) == 2
