from __future__ import annotations

import copy
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
# The readout values of 21 qubits of ibm_fez on 2026-03-05 and on the day after, in units of
# 1/4096. Qubit 20 has the value HIGH on 2026-03-05.
NOW = [0, *range(11, 30)]
NEXT = [20, *range(12, 31), 30]
HIGH = 0.05
# The qubit whose next readout value is a failure-ceiling value.
FLAGGED = 19
# ibm_kingston has only 19 readout changes on each day, too few to rank.
KINGSTON = range(11, 30)


def _row(
    prop: str,
    value: float,
    hours: float,
    qubit_a: int = 0,
    qubit_b: int | None = None,
    device: str = "ibm_fez",
    seen_after_h: float = 0.5,
    ceiling: bool = False,
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
        "is_failure_ceiling": ceiling,
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
        # Failure-ceiling values: a gate error of 0.7 and a flagged gate error of 1.
        _row("sx_gate_error", 0.001, 0, qubit_a=1),
        _row("sx_gate_error", 0.7, 24, qubit_a=1),
        _row("sx_gate_error", 0.002, 48, qubit_a=1),
        _row("cz_gate_error", 0.004, 0, qubit_a=2, qubit_b=3),
        _row("cz_gate_error", 1.0, 24, qubit_a=2, qubit_b=3, ceiling=True),
        _row("cz_gate_error", 0.005, 48, qubit_a=2, qubit_b=3),
        _row("readout_length", 1e-6, 0),
        _row("T1", 80e-6, 0, device="ibm_torino"),
    ]
    day = 4 * 24 + 12
    for q, (now, after) in enumerate(zip(NOW + [None], NEXT, strict=True)):
        rows.append(_row("prob_meas1_prep0", HIGH if now is None else now / 4096, day, q))
        rows.append(_row("prob_meas1_prep0", after / 4096, day + 24, q, ceiling=q == FLAGGED))
    for q, now in enumerate(KINGSTON):
        rows.append(_row("prob_meas1_prep0", now / 4096, day, q, device="ibm_kingston"))
        rows.append(_row("prob_meas1_prep0", (now + 1) / 4096, day + 24, q, device="ibm_kingston"))
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
    assert data["devices"] == ["ibm_fez", "ibm_kingston"]
    assert data["rows"] == 5 + 2 + 1 + 2 + 6 + 42 + 38
    assert data["duplicate_times"] == {"rows": 1, "different_value": 1}
    retired = archive.analyze(parquet, T0, END, include_retired=True)["devices"]
    assert retired == ["ibm_fez", "ibm_kingston", "ibm_torino"]


def test_counts_a_change_only_when_the_value_changes(data: dict[str, Any]) -> None:
    sx = data["intervals"]["ibm_fez"]["one_qubit"]
    assert sx["between_changes_h"] == {"count": 3, "median": 24.0, "p90": 24.0}
    assert sx["between_times_h"]["count"] == 6
    assert sx["between_times_h"]["median"] == 11.5
    assert sx["restamps"] == {"rows": 8, "restamps": 3, "fraction": 3 / 8}
    cz = data["intervals"]["ibm_fez"]["two_qubit"]
    assert cz["between_changes_h"]["count"] == 2
    assert cz["restamps"]["restamps"] == 1
    t1 = data["intervals"]["ibm_fez"]["t1"]
    assert t1["between_changes_h"] == {"count": 1, "median": 288.0, "p90": 288.0}
    readout = data["intervals_by_property"]["ibm_fez"]["prob_meas1_prep0"]
    assert readout["between_changes_h"] == {"count": 21, "median": 24.0, "p90": 24.0}


def test_gives_the_time_since_the_last_change_at_each_grid_time(data: dict[str, Any]) -> None:
    since = data["since_change"]["ibm_fez"]
    # The grid holds 9 times, 0 to 48 hours after T0. The T1 changes at -240 h and at 48 h.
    t1 = since["t1"]
    assert t1["count"] == 9
    assert t1["median"] == 258.0
    assert t1["above_24h"] == pytest.approx(8 / 9)
    assert t1["above_168h"] == pytest.approx(8 / 9)
    # Qubit 0 changes its sx error at 0 h and 24 h, qubit 1 at 0 h, 24 h and 48 h. The re-stamps
    # do not reset the time.
    sx = since["one_qubit"]
    assert sx["count"] == 18
    assert sx["median"] == 9.0
    assert sx["above_24h"] == 0.0
    # The readout series start after the grid, but prob_meas0_prep1 starts at T0.
    assert data["since_change_by_property"]["ibm_fez"]["prob_meas0_prep1"]["count"] == 9
    assert "prob_meas1_prep0" not in data["since_change_by_property"]["ibm_fez"]


def test_gives_the_size_of_a_change_by_its_gap(data: dict[str, Any]) -> None:
    size = data["change_size"]
    assert size["one_qubit"] == {"12-36 h": {"count": 1, "median": LN2, "p90": LN2}}
    assert size["t1"][">7 d"]["median"] == pytest.approx(LN2)
    # The zero has no ratio and the next value of qubit 19 is a failure-ceiling value, so 19 of
    # the 21 changes of ibm_fez count. The 19 changes of ibm_kingston count.
    pairs = [
        (a, b) for q, (a, b) in enumerate(zip(NOW, NEXT, strict=False)) if q not in (0, FLAGGED)
    ]
    logs = [abs(math.log(b / a)) for a, b in pairs]
    logs.append(abs(math.log(NEXT[-1] / 4096 / HIGH)))
    logs += [math.log((k + 1) / k) for k in KINGSTON]
    readout = size["readout"]["12-36 h"]
    assert readout["count"] == 38
    assert readout["median"] == pytest.approx(statistics.median(logs))
    assert "two_qubit" not in size


def test_counts_the_failure_ceiling_values_that_it_excludes(data: dict[str, Any]) -> None:
    assert data["excluded"] == {
        "ibm_fez": {"failure_ceiling": 2, "gate_error_0.5_or_more": 1},
        "ibm_kingston": {"failure_ceiling": 0, "gate_error_0.5_or_more": 0},
    }


def test_finds_the_zeros_and_the_grid_of_the_readout_values(data: dict[str, Any]) -> None:
    p10 = data["readout_grid"]["ibm_fez"]["prob_meas1_prep0"]
    assert p10["values"] == 42
    assert p10["zero_fraction"] == pytest.approx(1 / 42)
    multiples = p10["multiple_of"]
    assert multiples["1/4096"] == pytest.approx(41 / 42)
    assert multiples["1/8192"] == pytest.approx(41 / 42)
    assert multiples["1/2048"] == pytest.approx(22 / 42)
    assert multiples["1/1024"] == pytest.approx(12 / 42)
    assert multiples["1/1000"] == pytest.approx(2 / 42)
    assert multiples["1/10000"] == pytest.approx(2 / 42)
    p01 = data["readout_grid"]["ibm_fez"]["prob_meas0_prep1"]
    assert p01["zero_fraction"] == 0.0


def test_shows_the_regression_to_the_mean_of_the_extreme_changes(data: dict[str, Any]) -> None:
    # The 21 ranks of 2026-03-05 are 0, 1/20, ..., 1. Qubit 0 and qubit 1 are below 1/10.
    fez = data["regression"]["prob_meas1_prep0"]["ibm_fez"]
    parts = fez["parts"]
    assert parts["lowest tenth"] == {
        "count": 2,
        "mean_now": 5.5 / 4096,
        "median_now": 5.5 / 4096,
        "mean_next": 16 / 4096,
        "median_next": 16 / 4096,
    }
    # Qubit 19 drops out: its next value is a failure-ceiling value.
    assert fez["next_excluded"] == 1
    assert parts["highest tenth"]["count"] == 1
    assert parts["highest tenth"]["mean_now"] == HIGH
    assert parts["highest tenth"]["mean_next"] == 30 / 4096
    middle = parts["middle"]
    assert middle["count"] == 17
    assert middle["mean_now"] == pytest.approx(20 / 4096)
    assert middle["median_next"] == pytest.approx(21 / 4096)
    # The day after has 20 changes that are not failure-ceiling values, so it gives ranks too.
    assert fez["days"] == 2
    assert data["regression"]["cz_gate_error"]["ibm_fez"]["days"] == 0


def test_ranks_only_a_device_day_with_20_changes(data: dict[str, Any]) -> None:
    kingston = data["regression"]["prob_meas1_prep0"]["ibm_kingston"]
    assert kingston["days"] == 0
    assert [kingston["parts"][p]["count"] for p in archive.PARTS] == [0, 0, 0]


def test_ties_share_their_mean_percent_rank() -> None:
    rank, size = archive._percent_rank(
        np.zeros(5, dtype=np.int64), np.array([0, 0, 0, 0, 1]), np.array([3.0, 1.0, 1.0, 2.0, 9.0])
    )
    assert rank.tolist() == pytest.approx([1.0, 1 / 6, 1 / 6, 2 / 3, 0.0])
    assert size.tolist() == [4, 4, 4, 4, 1]


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
    assert "| ibm_fez | one_qubit | 3 | 24.0 | 24.0 | 6 | 11.5 |" in text


def test_shows_the_median_change_size_of_each_bin_with_100_changes(data: dict[str, Any]) -> None:
    assert data["change_size_note"] == archive.CHANGE_SIZE_NOTE
    assert data["change_size"]["one_qubit"]["12-36 h"]["p90"] == LN2
    larger = copy.deepcopy(data)
    larger["change_size"]["one_qubit"]["12-36 h"]["count"] = 100
    text = archive.markdown(larger)
    assert "| one_qubit | 12-36 h | 100 | 0.693 |\n" in text
    assert "| t1 | >7 d |" not in text
    assert "| Kind | Gap | Changes | Median |\n" in text
