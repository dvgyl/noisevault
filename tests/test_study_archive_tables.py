from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from conftest import require

from study import archive, archive_steps, archive_tables

pa = require("pyarrow")
pq = require("pyarrow.parquet")

T0 = datetime(2026, 3, 1, tzinfo=UTC)
LN2 = math.log(2)
QUBIT_SCHEMA = pa.schema(
    [
        ("backend_name", pa.string()),
        ("qubit", pa.int32()),
        ("date", pa.timestamp("us", tz="UTC")),
        ("name", pa.string()),
        ("unit", pa.string()),
        ("value", pa.float64()),
    ]
)
GATE_SCHEMA = pa.schema(
    [
        ("backend_name", pa.string()),
        ("gate_type", pa.string()),
        ("gate_name", pa.string()),
        ("qubits", pa.list_(pa.int32())),
        ("date", pa.timestamp("us", tz="UTC")),
        ("name", pa.string()),
        ("unit", pa.string()),
        ("value", pa.float64()),
    ]
)


def _qubit(
    name: str, value: float, hours: float, qubit: int = 0, unit: str = "", device: str = "ibm_fez"
) -> dict[str, Any]:
    return {
        "backend_name": device,
        "qubit": qubit,
        "date": T0 + timedelta(hours=hours),
        "name": name,
        "unit": unit,
        "value": value,
    }


def _gate(
    gate: str,
    qubits: list[int],
    value: float,
    hours: float,
    name: str = "gate_error",
    device: str = "ibm_fez",
) -> dict[str, Any]:
    return {
        "backend_name": device,
        "gate_type": gate,
        "gate_name": gate + "_".join(map(str, qubits)),
        "qubits": qubits,
        "date": T0 + timedelta(hours=hours),
        "name": name,
        "unit": "ns" if name == "gate_length" else "",
        "value": value,
    }


def _folder(path: Path, qubits: list[dict[str, Any]], gates: list[dict[str, Any]]) -> Path:
    # Row groups of 3 rows, so that the rows of one key are in more than one row group.
    path.mkdir()
    pq.write_table(pa.Table.from_pylist(qubits, QUBIT_SCHEMA), path / "qubit_properties.parquet", 3)
    pq.write_table(pa.Table.from_pylist(gates, GATE_SCHEMA), path / "gate_properties.parquet", 3)
    return path


QUBITS = [
    _qubit("T1", 100.0, 0, unit="us"),
    _qubit("T1", 100.0, 0, unit="us"),
    _qubit("T1", 120.0, 24, unit="µs"),
    _qubit("T2", 80.0, 0, unit="µs"),
    _qubit("prob_meas0_prep1", 0.01, 0),
    _qubit("prob_meas1_prep0", 0.02, 0),
    _qubit("readout_error", 0.015, 0),
    _qubit("frequency", 5.1, 0, unit="GHz"),
    _qubit("anharmonicity", -0.3, 0, unit="GHz"),
    _qubit("readout_length", 1500.0, 0, unit="ns"),
    _qubit("operational", 1.0, 0),
    _qubit("T1", 90.0, -14 * 365 * 24, qubit=1, unit="us"),
    _qubit("T1", 70.0, 0, unit="us", device="ibm_washington_"),
    _qubit("T1", 60.0, 0, unit="us", device="ibm_algiers"),
    # The same key as the row of 0.02, with another value, later in the file.
    _qubit("prob_meas1_prep0", 0.03, 0),
]
GATES = [
    _gate("sx", [0], 2e-4, 0),
    _gate("sx", [0], 2e-4, 0),
    _gate("cz", [1, 0], 5e-3, 0),
    _gate("cz", [0, 1], 4e-3, 0),
    _gate("cz", [0, 1], 68.0, 0, name="gate_length"),
    _gate("cz", [0, 1], 1.0, 0, name="operational"),
    _gate("x", [0], 2e-4, 0),
    _gate("rz", [0], 0.0, 0),
    _gate("id", [0], 2e-4, 0),
]


@pytest.fixture(scope="module")
def converted(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, archive_tables.Counts]:
    root = tmp_path_factory.mktemp("tables")
    folder = _folder(root / "tables", QUBITS, GATES)
    out = root / "clean" / "history.parquet"
    return out, archive_tables.convert(folder, out)


@pytest.fixture(scope="module")
def table(converted: tuple[Path, archive_tables.Counts]) -> pa.Table:
    return pq.read_table(converted[0])


def _keys(table: pa.Table) -> list[tuple[Any, ...]]:
    columns = ["backend", "property", "qubit_a", "qubit_b", "calibrated_time", "value"]
    return [tuple(row[c] for c in columns) for row in table.select(columns).to_pylist()]


def test_keeps_the_named_values_and_drops_the_other_names(table: pa.Table) -> None:
    assert sorted(set(table["property"].to_pylist())) == [
        "T1",
        "T2",
        "cz_gate_error",
        "prob_meas0_prep1",
        "prob_meas1_prep0",
        "readout_error",
        "sx_gate_error",
    ]


def test_gives_t1_and_t2_the_unit_us_for_both_spellings(table: pa.Table) -> None:
    rows = [r for r in table.to_pylist() if r["backend"] == "ibm_fez" and r["property"][0] == "T"]
    assert [(r["property"], r["value"], r["unit"]) for r in rows] == [
        ("T1", 100.0, "us"),
        ("T1", 120.0, "us"),
        ("T2", 80.0, "us"),
    ]
    assert set(table["unit"].to_pylist()) == {"us", ""}


def test_keeps_one_row_of_a_repeat_and_the_last_value_of_a_key(
    converted: tuple[Path, archive_tables.Counts], table: pa.Table
) -> None:
    _, counts = converted
    t1 = [k for k in _keys(table) if k[:2] == ("ibm_fez", "T1")]
    assert len(t1) == 2
    readout = [k[-1] for k in _keys(table) if k[1] == "prob_meas1_prep0"]
    assert readout == [0.03]
    assert counts.conflicts == {"ibm_algiers": 0, "ibm_fez": 1}
    report = archive_tables.report(counts)
    assert "keys with different values: 1" in report
    # 11 qubit rows and 4 gate rows have a kept name. Rule 4 drops 2 rows and rule 5 drops 3.
    assert "rows with a kept name: 15" in report
    assert "rows removed by rule 5, repeats and older values of a key: 3" in report
    assert "rows out: 10" in report


def test_drops_an_early_date_and_a_device_name_with_an_underscore(
    converted: tuple[Path, archive_tables.Counts], table: pa.Table
) -> None:
    _, counts = converted
    assert counts.early == {"ibm_fez": 1}
    assert counts.underscore == {"ibm_washington_": 1}
    assert set(table["backend"].to_pylist()) == {"ibm_algiers", "ibm_fez"}
    assert min(table["calibrated_time"].to_pylist()) == T0
    report = archive_tables.report(counts)
    assert "rows dropped with a date before 2015-01-01: 1" in report
    assert "rows dropped for a device name that ends with an underscore: 1" in report


def test_gives_a_gate_its_qubits_in_their_order(table: pa.Table) -> None:
    gates = [k[1:5] for k in _keys(table) if k[1].endswith("_gate_error")]
    assert gates == [
        ("cz_gate_error", 0, 1.0, T0),
        ("cz_gate_error", 1, 0.0, T0),
        ("sx_gate_error", 0, None, T0),
    ]


def test_writes_the_columns_and_the_order_that_the_archive_scripts_read(table: pa.Table) -> None:
    assert table.schema == archive_tables.schema()
    assert table["observed_time"].equals(table["calibrated_time"])
    assert not any(table["is_failure_ceiling"].to_pylist())
    keys = _keys(table)
    assert keys == sorted(
        keys, key=lambda k: (k[0], k[1], k[2], -1 if k[3] is None else k[3], k[4])
    )
    assert keys[0][:2] == ("ibm_algiers", "T1")


def test_reads_a_device_whose_qubit_b_is_null_in_each_row(tmp_path: Path) -> None:
    folder = _folder(tmp_path / "tables", QUBITS, GATES)
    out = tmp_path / "algiers.parquet"
    counts = archive_tables.convert(folder, out, ["ibm_algiers"])
    assert counts.rows_out == {"ibm_algiers": 1}
    rows = archive.read_rows(out)
    assert rows.devices == ["ibm_algiers"]
    assert rows.value.tolist() == [60.0]


def test_stops_at_another_unit_of_t1(tmp_path: Path) -> None:
    folder = _folder(tmp_path / "tables", [_qubit("T1", 1e-4, 0, unit="s")], GATES)
    with pytest.raises(ValueError, match="a T1 or T2 row has the unit 's'"):
        archive_tables.convert(folder, tmp_path / "out.parquet")


def test_main_writes_the_file_and_the_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = _folder(tmp_path / "tables", QUBITS, GATES)
    out = tmp_path / "history.parquet"
    assert archive_tables.main([str(folder), "--out", str(out)]) == 0
    assert "rows out: 10" in capsys.readouterr().out
    assert pq.read_metadata(out).num_rows == 10
    assert archive_tables.main([str(tmp_path), "--out", str(out)]) == 2


def test_the_archive_scripts_read_a_converted_history(tmp_path: Path) -> None:
    hf_archive = require("noisevault.sources.hf_archive")
    day = 24
    qubits = [
        # Each value is in two snapshots.
        *(_qubit("T1", v, h, unit=u) for v, h, u in [(100.0, 0, "us"), (50.0, day, "µs")] * 2),
        *(_qubit("prob_meas1_prep0", v, h) for v, h in [(0.01, 0), (0.01, day), (0.03, 2 * day)]),
    ]
    gates = [
        *(_gate("sx", [0], v, h) for v, h in [(1e-3, 0), (2e-3, day), (4e-3, 2 * day)] * 2),
        *(_gate("cx", [0, 1], v, h, device="ibmq_old") for v, h in [(0.01, 0), (0.02, day)]),
    ]
    out = tmp_path / "history.parquet"
    archive_tables.convert(_folder(tmp_path / "tables", qubits, gates), out)
    rows = archive.read_rows(out)
    # T1 2, readout 3, sx 3 and cx 2 rows. The converter left no repeat.
    assert rows.value.size == 10
    assert (rows.duplicates, rows.conflicts) == (0, 0)
    changes = rows.select(rows.change)
    # The readout value of the second day is a re-stamp.
    assert changes.value.size == 9
    old = changes.device == changes.devices.index("ibmq_old")
    assert set(changes.kind[old].tolist()) == {archive.KIND_ORDER.index("two_qubit")}
    # The sizes are ln 2 (T1), ln 3 (readout), ln 2 two times (sx) and ln 2 (cx).
    assert archive_steps.typical_size(changes) == pytest.approx(LN2)
    end = T0 + timedelta(days=3)
    data = archive.analyze(out, T0, end)
    assert data["regression"]["cx_gate_error"]["ibmq_old"]["days"] == 0
    assert "ecr_gate_error" not in data["regression"]
    assert "| cx_gate_error | ibmq_old | 0 | 0 |" in archive.markdown(data)
    spans = hf_archive.calibration_archive_devices(out)
    assert spans["ibm_fez"].first == T0
    assert spans["ibm_fez"].last == T0 + timedelta(days=2)
