"""The rows that study.analyze keeps between runs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from test_study_analyze import RUN_AT, _device

import study
from noisevault.counts import MeasuredCounts, simulate
from noisevault.errors import SourceUnavailable
from noisevault.profile import Profile
from noisevault.suite import suite
from study import analyze, models


@pytest.fixture(autouse=True)
def _fast_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "fit_factor", lambda build, counts, circuits: 1.0)


def _small_run(folder: Path, stem: str) -> MeasuredCounts:
    """A run of the suite of the 3-qubit device with few shots, saved as ``stem``."""
    calibration = _device()
    counts = simulate(calibration, suite(calibration), shots=60, seed=1, run_at=RUN_AT)
    folder.mkdir(parents=True, exist_ok=True)
    counts.save(folder / f"{stem}.counts.json")
    calibration.save(folder / f"{stem}.profile.json")
    return counts


def test_a_kept_row_has_the_chain_kind_and_the_trial_status_of_its_file(tmp_path: Path) -> None:
    rows = tmp_path / "rows"
    _small_run(tmp_path / "first", "best-deep")
    _small_run(tmp_path / "second", "median-trial")

    first = analyze.analyze(tmp_path / "first", offline=True, rows_folder=rows, samples=1000)
    second = analyze.analyze(tmp_path / "second", offline=True, rows_folder=rows, samples=1000)

    (a,), (b,) = first["rows"], second["rows"]
    assert a["counts_sha256"] == b["counts_sha256"]
    assert (a["chain"], a["trial"]) == ("best", False)
    assert (b["chain"], b["trial"]) == ("median", True)


def test_a_row_that_waits_for_a_failed_source_is_not_kept(tmp_path: Path) -> None:
    _small_run(tmp_path / "campaign", "best-deep")
    rows = tmp_path / "rows"
    calls: list[datetime] = []

    def down(device: str, at: datetime) -> Profile:
        calls.append(at)
        raise SourceUnavailable("IBM's public endpoint did not answer")

    def run() -> dict:
        return analyze.analyze(
            tmp_path / "campaign",
            cache=None,
            pull=down,
            properties=lambda device, at: None,
            rows_folder=rows,
            samples=1000,
        )

    (row,) = run()["rows"]
    assert row["failures"]["stale(7)"] == "SourceUnavailable"
    assert set(row["pending"]) >= {"stale(7)", "averaged(7)", "ages", "marked"}
    asked = len(calls)

    run()

    assert len(calls) == 2 * asked
    assert not list(rows.glob("*.json"))


def test_a_row_with_no_reason_that_can_pass_is_kept(tmp_path: Path) -> None:
    _small_run(tmp_path / "campaign", "best-deep")
    rows = tmp_path / "rows"

    data = analyze.analyze(tmp_path / "campaign", offline=True, rows_folder=rows, samples=1000)

    assert data["rows"][0]["pending"] == []
    assert len(list(rows.glob("*.json"))) == 1


def test_the_code_version_holds_the_library_constraints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("src/noisevault/a.py", "study/b.py", ".github/study-constraints.txt"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("numpy==2.5.4\n")
    monkeypatch.setattr(study, "_ROOT", tmp_path)
    study.code_version.cache_clear()
    try:
        before = study.code_version()
        (tmp_path / ".github" / "study-constraints.txt").write_text("numpy==2.5.5\n")
        study.code_version.cache_clear()
        assert study.code_version() != before
    finally:
        study.code_version.cache_clear()
