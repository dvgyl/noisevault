"""scripts/build_realism_table.py: one row per counts file, as JSON and Markdown."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import noisevault as nv
from noisevault.counts import load_counts, plan, simulate
from noisevault.errors import NoiseVaultError
from noisevault.profile import ErrorFactor

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_realism_table.py"
GARNET = ROOT / "tests" / "fixtures" / "braket" / "iqm_garnet_run.json"
H2 = "quantinuum_h2-1@2025-04-30"
RUN_AT = datetime(2025, 5, 1, 12, 0, tzinfo=UTC)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_realism_table", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = _load_script()


def _simulated(folder: Path, profile: nv.Profile, name: str, gate: float, readout: float) -> Path:
    truth = profile.model_copy(
        update={"unmodeled_error": {"gates": {"factor": gate}, "readout": {"factor": readout}}}
    )
    when = profile.device.calibrated_at
    run_at = RUN_AT if when is None or when < RUN_AT else when
    path = folder / f"{name}.counts.json"
    simulate(truth, plan(profile), shots=2000, seed=7, run_at=run_at).save(path)
    return path


@pytest.fixture(scope="module")
def h2_table(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    folder = tmp_path_factory.mktemp("h2")
    _simulated(folder, nv.load(H2), "h2", 1.5, 1.2)
    return folder, script.build(folder)


def test_a_row_holds_the_fit_that_nv_compare_gives(h2_table: tuple[Path, dict[str, Any]]) -> None:
    folder, table = h2_table
    (row,) = table["rows"]
    counts = load_counts(folder / "h2.counts.json")
    result = nv.load(H2).compare(counts)
    gates, readout = result.gates, result.readout
    assert isinstance(gates, ErrorFactor) and isinstance(readout, ErrorFactor)
    assert row["device"] == "quantinuum_h2-1"
    assert (row["vendor"], row["technology"]) == ("quantinuum", "trapped_ion")
    assert (row["date"], row["run_at"]) == ("2025-05-01", "2025-05-01T12:00:00Z")
    assert row["source"] == "simulated"
    assert row["simulated_with"] == {"gates": 1.5, "readout": 1.2}
    assert row["gates"]["factor"] == pytest.approx(gates.factor, rel=1e-5)
    assert row["gates"]["low"] == pytest.approx(gates.low, rel=1e-5)
    assert row["gates"]["high"] == pytest.approx(gates.high, rel=1e-5)
    assert row["readout"]["factor"] == pytest.approx(readout.factor, rel=1e-5)
    assert row["p_value"] == pytest.approx(result.p_value, rel=1e-5)
    assert row["calibration"]["fingerprint"] == counts.profile.fingerprint
    assert row["counts"] == {"file": "h2.counts.json", "sha256": counts.sha256}
    assert row["shots"] == 2000 * len(counts.circuits)
    assert "job_ids" not in json.dumps(table)


def test_the_markdown_labels_every_simulated_row(h2_table: tuple[Path, dict[str, Any]]) -> None:
    _, table = h2_table
    page = script.markdown(table, "Realism table")
    rows = [line for line in page.splitlines() if line.startswith("| `")]
    assert len(rows) == 1
    assert rows[0].endswith("| simulated, x1.5 gates, x1.2 readout |")
    assert rows[0].startswith("| `quantinuum_h2-1` | quantinuum | trapped ion | 2025-05-01 | x")
    assert "Every row is simulated." in page


def test_the_profile_beside_the_counts_binds_a_calibration_outside_the_catalog(
    tmp_path: Path,
) -> None:
    profile = nv.from_braket(json.loads(GARNET.read_text()), device="iqm_garnet")
    path = _simulated(tmp_path, profile, "garnet", 1.0, 1.0)
    with pytest.raises(NoiseVaultError) as caught:
        script.build(tmp_path)
    assert caught.value.message == (
        f"{path}: no profile iqm_garnet nv:{profile.calibration_fingerprint[:12]}"
    )
    assert caught.value.hint == (
        "put the profile file beside the counts as garnet.profile.json, or pull the calibration"
    )
    profile.save(tmp_path / "garnet.profile.json")
    (row,) = script.build(tmp_path)["rows"]
    assert (row["device"], row["vendor"], row["technology"]) == (
        "iqm_garnet",
        "iqm",
        "superconducting",
    )


def _factor(factor: float, low: float | None, high: float | None, bound: str | None) -> Any:
    return {"factor": factor, "low": low, "high": high, "bound": bound}


def _row(**changes: Any) -> dict[str, Any]:
    base = {
        "device": "iqm_garnet",
        "vendor": "iqm",
        "technology": "superconducting",
        "date": "2026-11-01",
        "run_at": "2026-11-01T10:00:00Z",
        "source": "hardware",
        "simulated_with": None,
        "gates": _factor(1.84, 1.54, 2.12, None),
        "readout": _factor(0.9, 0.85, 0.95, None),
        "p_value": 0.4,
        "counts": {"file": "g.counts.json"},
    }
    return base | changes


@pytest.mark.parametrize(
    ("changes", "cells"),
    [
        ({}, ["x1.84 (1.54 to 2.12)", "x0.9 (0.85 to 0.95)", "p = 0.4", "hardware"]),
        (
            {"gates": _factor(0.06, None, 0.4, "lower"), "p_value": 0.003},
            ["x0.06 (at most 0.4)", "p = 0.003, poor"],
        ),
        (
            {"readout": _factor(19.0, 3.1, None, "upper"), "p_value": None},
            ["x19 (at least 3.1)", "no test"],
        ),
        ({"gates": {"not_identified": "the counts do not move it"}}, ["not identified"]),
        ({"vendor": None}, ["unknown"]),
    ],
)
def test_each_cell_says_what_the_fit_found(changes: dict[str, Any], cells: list[str]) -> None:
    page = script.markdown({"rows": [_row(**changes)]}, "Realism table")
    (line,) = [x for x in page.splitlines() if x.startswith("| `")]
    found = [cell.strip() for cell in line.strip("|").split("|")]
    assert all(cell in found for cell in cells), found
    assert "Every row is simulated." not in page


def test_a_table_of_hardware_and_simulated_rows_labels_each_row(tmp_path: Path) -> None:
    rows = [_row(), _row(source="simulated", simulated_with={"gates": 2.0, "readout": 1.0})]
    page = script.markdown({"rows": rows}, "Realism table")
    lines = [x for x in page.splitlines() if x.startswith("| `")]
    assert lines[0].endswith("| hardware |")
    assert lines[1].endswith("| simulated, x2 gates, x1 readout |")


def test_appledouble_files_and_other_files_are_not_counts(tmp_path: Path) -> None:
    for name in ("a.counts.json", "b.counts.json.gz", "._a.counts.json", "a.profile.json"):
        (tmp_path / name).write_text("{}")
    assert [p.name for p in script.counts_files(tmp_path)] == ["a.counts.json", "b.counts.json.gz"]
    with pytest.raises(NoiseVaultError, match="no counts files in"):
        script.build(tmp_path / "missing")


def test_the_committed_demo_holds_every_demo_run_and_labels_each_row_simulated() -> None:
    table = json.loads((ROOT / "docs" / "realism" / "demo.json").read_text(encoding="utf-8"))
    used = sorted(
        (nv.load(ref).id, f"{run_at:%Y-%m-%d}", gate, readout)
        for ref, run_at, gate, readout, _ in script.DEMO_RUNS
    )
    rows = table["rows"]
    assert (
        sorted(
            (r["device"], r["date"], r["simulated_with"]["gates"], r["simulated_with"]["readout"])
            for r in rows
        )
        == used
    )
    assert all(r["source"] == "simulated" for r in rows)
    assert table["realism_table"] == script.TABLE_FORMAT
    page = (ROOT / "docs" / "realism" / "demo.md").read_text(encoding="utf-8")
    assert page == script.markdown(table, "Realism table demo")
    lines = [x for x in page.splitlines() if x.startswith("| `")]
    assert len(lines) == len(rows)
    assert all("| simulated, x" in line for line in lines)


def test_the_demo_simulates_each_run_from_its_catalog_profile(tmp_path: Path) -> None:
    script.demo(tmp_path)
    paths = script.counts_files(tmp_path)
    assert len(paths) == len(script.DEMO_RUNS)
    for path, (ref, run_at, _, _, seed) in zip(paths, script.DEMO_RUNS, strict=True):
        counts = load_counts(path)
        assert counts.source == "simulated"
        assert counts.run_at == run_at
        assert counts.execution.options["seed"] == seed
        assert counts.profile.fingerprint == nv.load(ref).calibration_fingerprint


def test_check_names_each_file_that_is_out_of_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    table = {"realism_table": "1", "note": "", "rows": [_row()]}
    monkeypatch.setattr(script, "demo", lambda folder: None)
    monkeypatch.setattr(script, "build", lambda folder: table)
    output = tmp_path / "demo"
    assert script.main(["--demo", "-o", str(output), "--check"]) == 1
    err = capsys.readouterr().err
    assert f"out of date: {output}.json" in err and f"out of date: {output}.md" in err
    assert script.main(["--demo", "-o", str(output)]) == 0
    assert script.main(["--demo", "-o", str(output), "--check"]) == 0
    assert json.loads(output.with_suffix(".json").read_text()) == table


@pytest.mark.parametrize(
    "args", [[], ["folder", "--demo"], ["folder", "--check"]], ids=["none", "both", "check"]
)
def test_the_command_needs_a_folder_or_the_demo(args: list[str]) -> None:
    assert script.main(args) == 2
