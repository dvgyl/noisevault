"""The analysis gives the same rows with one process and with several processes."""

from __future__ import annotations

from pathlib import Path

from test_study_analyze import RUN_AT, _device

from noisevault.counts import simulate
from noisevault.suite import suite
from study import analyze


def test_several_processes_give_the_rows_of_one_process(tmp_path: Path) -> None:
    profile = _device()
    circuits = suite(profile)
    for kind, seed in (("best", 1), ("median", 2), ("best-trial", 3)):
        counts = simulate(profile, circuits, shots=200, seed=seed, run_at=RUN_AT)
        counts.save(tmp_path / f"{kind}-deep.counts.json")
        profile.save(tmp_path / f"{kind}-deep.profile.json")

    one = analyze.analyze(tmp_path, offline=True, samples=1000, jobs=1)
    three = analyze.analyze(tmp_path, offline=True, samples=1000, jobs=3)

    assert len(one["rows"]) == 3
    assert three == one


def test_a_file_that_fails_in_a_process_is_a_failure_and_the_others_give_rows(
    tmp_path: Path,
) -> None:
    profile = _device()
    circuits = suite(profile)
    for kind, seed in (("best", 1), ("median", 2)):
        counts = simulate(profile, circuits, shots=200, seed=seed, run_at=RUN_AT)
        counts.save(tmp_path / f"{kind}-deep.counts.json")
    # Only the best run has its profile beside it.
    profile.save(tmp_path / "best-deep.profile.json")

    found = analyze.analyze(tmp_path, offline=True, samples=1000, jobs=2)

    assert len(found["rows"]) == 1
    assert len(found["failures"]) == 1
    assert set(found["failures"][0]) == {"error", "file"}
