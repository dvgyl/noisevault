from __future__ import annotations

import contextlib
import io
import json
import socket
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from fake_ibm import FakeIBM, load_script, noisy_fake

import noisevault as nv
from noisevault.compare import compare
from noisevault.counts import load_counts, plan

campaign = load_script("ibm_campaign")

MONDAY = date(2026, 11, 2)
THURSDAY = date(2026, 11, 5)
NEXT_MONDAY = date(2026, 11, 9)


def _run(
    root: Path,
    ibm: FakeIBM,
    today: date,
    *,
    wait: bool = True,
    reserve: float = 0.0,
    budget: bool = True,
) -> list[Any]:
    account = ibm.account(campaign)
    run = campaign.Campaign(
        root=root,
        account=account,
        budget=campaign.Budget(account.remaining_s if budget else None, reserve),
        shots=2000,
        wait=wait,
    )
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run.run(["ibm_fez"], today)


@contextlib.contextmanager
def _fake_ibm(home: Path) -> Iterator[FakeIBM]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("NOISEVAULT_HOME", str(home))
        yield FakeIBM(monkeypatch, {"ibm_fez": noisy_fake()})


@dataclass
class TwoRuns:
    root: Path
    ibm: FakeIBM
    first: list[Any]
    first_jobs: list[str]
    rerun: list[Any]


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TwoRuns]:
    folder = tmp_path_factory.mktemp("campaign")
    with _fake_ibm(folder / "nv_home") as ibm:
        root = folder / "campaign"
        first = _run(root, ibm, MONDAY)
        first_jobs = list(ibm.submitted)
        rerun = _run(root, ibm, THURSDAY)
        yield TwoRuns(root, ibm, first, first_jobs, rerun)


def _states(statuses: list[Any]) -> list[tuple[str, str, str]]:
    return [(s.week, s.chain, s.state) for s in statuses]


def test_the_first_run_of_a_week_saves_counts_and_profiles_for_both_chains(
    two_runs: TwoRuns,
) -> None:
    week = two_runs.root / "ibm_fez" / "2026-11-02"
    assert _states(two_runs.first) == [
        ("2026-11-02", "best", "collected"),
        ("2026-11-02", "median", "collected"),
    ]
    assert len(two_runs.first_jobs) == 2
    names = {p.name for p in week.iterdir()}
    assert {
        "calibration.json",
        "chains.json",
        "best.counts.json",
        "best.profile.json",
        "median.counts.json",
        "median.profile.json",
    } <= names
    assert not any(name.endswith(".job.json") for name in names)
    for kind in campaign.CHAINS:
        counts = load_counts(week / f"{kind}.counts.json")
        profile = nv.load(week / f"{kind}.profile.json")
        assert counts.profile.fingerprint == profile.calibration_fingerprint
        assert counts.execution.job_ids[0] in two_runs.first_jobs
        assert compare(profile, counts).counts is counts


def test_the_counts_run_on_the_best_and_the_median_chain(two_runs: TwoRuns) -> None:
    week = two_runs.root / "ibm_fez" / "2026-11-02"
    profile = nv.load(week / "calibration.json")
    chains = campaign.pick_chains(profile)
    saved = json.loads((week / "chains.json").read_text())
    assert saved["calibration"] == profile.calibration_fingerprint
    for kind in campaign.CHAINS:
        assert tuple(saved[kind]["qubits"]) == chains[kind].qubits
        counts = load_counts(week / f"{kind}.counts.json")
        assert [c.qubits for c in counts.circuits] == [
            c.qubits for c in plan(profile, chains[kind].qubits)
        ]
    assert chains["best"].qubits == max((c.qubits for c in plan(profile)), key=len)
    assert chains["best"].error < chains["median"].error


def test_the_median_chain_is_the_median_of_the_chains_that_plan_the_same_circuits() -> None:
    profile = nv.from_qiskit_backend(noisy_fake())
    best, candidates = campaign.eligible_chains(profile)
    chains = campaign.pick_chains(profile)
    assert chains["best"] == best
    assert (chains["median"].error, chains["median"].qubits) == candidates[
        (len(candidates) - 1) // 2
    ]
    shape = campaign._shape(plan(profile))
    for _, path in candidates[:: max(1, len(candidates) // 10)]:
        assert campaign._shape(plan(profile, path)) == shape
    ranked = campaign._ranked_chains(profile, len(best.qubits))
    assert candidates[0][0] <= best.error + 1e-12
    assert len(candidates) <= len(ranked)


def test_a_rerun_in_the_same_week_submits_nothing(two_runs: TwoRuns) -> None:
    assert _states(two_runs.rerun) == [
        ("2026-11-02", "best", "done"),
        ("2026-11-02", "median", "done"),
    ]
    assert two_runs.ibm.submitted == two_runs.first_jobs
    assert not (two_runs.root / "ibm_fez" / "2026-11-05").exists()


def test_each_iso_week_has_its_own_folder(tmp_path: Path) -> None:
    device = tmp_path / "ibm_fez"
    assert campaign.week_folder(device, THURSDAY) == device / "2026-11-05"
    (device / "2026-11-02").mkdir(parents=True)
    (device / "._2026-11-03").mkdir()
    (device / "notes").mkdir()
    assert campaign.week_folder(device, THURSDAY) == device / "2026-11-02"
    assert campaign.week_folder(device, date(2026, 11, 8)) == device / "2026-11-02"
    assert campaign.week_folder(device, NEXT_MONDAY) == device / "2026-11-09"


def test_no_wait_submits_and_a_rerun_in_the_next_week_collects_first(tmp_path: Path) -> None:
    with _fake_ibm(tmp_path / "nv_home") as ibm:
        root = tmp_path / "campaign"
        submitted = _run(root, ibm, MONDAY, wait=False)
        assert [(s.chain, s.state, s.detail) for s in submitted] == [
            ("best", "submitted", "now"),
            ("median", "submitted", "now"),
        ]
        week = root / "ibm_fez" / "2026-11-02"
        assert (week / "best.job.json").exists() and (week / "median.job.json").exists()
        assert not (week / "best.counts.json").exists()
        jobs = list(ibm.submitted)

        later = _run(root, ibm, NEXT_MONDAY)
        assert _states(later) == [
            ("2026-11-02", "best", "collected"),
            ("2026-11-02", "median", "collected"),
            ("2026-11-09", "best", "collected"),
            ("2026-11-09", "median", "collected"),
        ]
        assert set(jobs) <= set(ibm.opened)
        assert len(ibm.submitted) == 4
        assert (week / "best.counts.json").exists() and not (week / "best.job.json").exists()


def test_the_budget_stops_a_job_that_does_not_fit(tmp_path: Path) -> None:
    with _fake_ibm(tmp_path / "nv_home") as ibm:
        ibm.remaining = 6.0
        statuses = _run(tmp_path / "campaign", ibm, MONDAY)
        assert _states(statuses) == [
            ("2026-11-02", "median", "error"),
            ("2026-11-02", "best", "collected"),
        ]
        detail = statuses[0].detail
        assert "s is free (6.0 s left, " in detail and " s reserve), so the campaign" in detail
        assert "s for jobs that have not run yet" in detail
        assert len(ibm.submitted) == 1
        week = tmp_path / "campaign" / "ibm_fez" / "2026-11-02"
        assert not (week / "median.job.json").exists()

        ibm.remaining = 600.0
        again = _run(tmp_path / "campaign", ibm, THURSDAY)
        assert _states(again) == [
            ("2026-11-02", "best", "done"),
            ("2026-11-02", "median", "collected"),
        ]
        assert len(ibm.submitted) == 2


def test_the_reserve_counts_against_the_budget(tmp_path: Path) -> None:
    with _fake_ibm(tmp_path / "nv_home") as ibm:
        statuses = _run(tmp_path / "campaign", ibm, MONDAY, reserve=598.0)
        assert [s.state for s in statuses] == ["error", "error"]
        assert "598 s reserve" in statuses[0].detail
        assert ibm.submitted == []


def test_an_unknown_budget_submits_nothing(tmp_path: Path) -> None:
    with _fake_ibm(tmp_path / "nv_home") as ibm:
        ibm.remaining = None
        statuses = _run(tmp_path / "campaign", ibm, MONDAY)
        assert [s.state for s in statuses] == ["error", "error"]
        assert "--no-budget-check" in statuses[0].detail
        assert ibm.submitted == []


def test_a_job_that_cannot_be_collected_keeps_its_job_file(tmp_path: Path) -> None:
    with _fake_ibm(tmp_path / "nv_home") as ibm:
        root = tmp_path / "campaign"
        _run(root, ibm, MONDAY, wait=False)

        def lost(job_id: str) -> Any:
            raise ConnectionError("network is unreachable")

        ibm.open_job = lost  # type: ignore[method-assign]
        statuses = _run(root, ibm, THURSDAY)
        assert [s.state for s in statuses] == ["error", "error"]
        assert "network is unreachable" in statuses[0].detail
        assert (root / "ibm_fez" / "2026-11-02" / "best.job.json").exists()
        assert len(ibm.submitted) == 2


def test_the_status_table_lists_each_error() -> None:
    statuses = [
        campaign.Status("ibm_fez", "2026-11-02", "best", "collected", "job a"),
        campaign.Status("ibm_fez", "2026-11-02", "median", "error", "no time"),
    ]
    assert campaign.table(statuses) == (
        "device   week        chain   state\n"
        "ibm_fez  2026-11-02  best    collected\n"
        "ibm_fez  2026-11-02  median  error\n"
        "error: ibm_fez 2026-11-02 median: no time"
    )


@pytest.mark.parametrize("value", ["-1", "nan", "x"])
def test_the_reserve_must_be_seconds_of_zero_or_more(
    value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        campaign._parser().parse_args(["ibm_fez", "--reserve", value])
    assert "is not a number of seconds of 0 or more" in capsys.readouterr().err


def test_the_budget_counts_jobs_of_earlier_runs_until_they_have_run(tmp_path: Path) -> None:
    with _fake_ibm(tmp_path / "nv_home") as ibm:
        root = tmp_path / "campaign"
        _run(root, ibm, MONDAY, wait=False)
        week = root / "ibm_fez" / "2026-11-02"
        estimates = [
            json.loads((week / f"{kind}.usage.json").read_text())["usage_s"]
            for kind in campaign.CHAINS
        ]
        first = list(ibm.submitted)
        ibm.remaining = sum(estimates) + 1.0
        statuses = _run(root, ibm, NEXT_MONDAY, wait=False)
        refused = [s for s in statuses if s.week == "2026-11-09"]
        assert [s.state for s in refused] == ["error", "error"]
        assert f"{sum(estimates):.1f} s for jobs that have not run yet" in refused[0].detail
        assert ibm.submitted == first

        for job_id in first:
            ibm.states[job_id] = "DONE"
        statuses = _run(root, ibm, NEXT_MONDAY, wait=False)
        assert [s.state for s in statuses if s.week == "2026-11-09"][:1] == ["submitted"]
        assert len(ibm.submitted) > len(first)


def test_a_job_file_saved_under_another_name_is_collected_and_blocks_a_new_job(
    tmp_path: Path,
) -> None:
    with _fake_ibm(tmp_path / "nv_home") as ibm:
        root = tmp_path / "campaign"
        _run(root, ibm, MONDAY, wait=False)
        week = root / "ibm_fez" / "2026-11-02"
        job_id = ibm.submitted[0]
        (week / "best.job.json").rename(week / f"best.{job_id}.job.json")
        statuses = _run(root, ibm, THURSDAY)
        assert _states(statuses) == [
            ("2026-11-02", "best", "collected"),
            ("2026-11-02", "median", "collected"),
        ]
        assert len(ibm.submitted) == 2
        assert load_counts(week / "best.counts.json").execution.job_ids == (job_id,)
        assert not list(week.glob("*.job.json"))


def test_a_lock_of_another_run_stops_the_device(tmp_path: Path) -> None:
    with _fake_ibm(tmp_path / "nv_home") as ibm:
        root = tmp_path / "campaign"
        lock = root / "ibm_fez" / campaign.LOCK
        lock.parent.mkdir(parents=True)
        lock.write_text("another-computer 1\n")
        statuses = _run(root, ibm, MONDAY)
        assert [(s.week, s.state) for s in statuses] == [("-", "error")]
        assert "another campaign run may be at work on ibm_fez" in statuses[0].detail
        assert ibm.submitted == [] and lock.exists()


def test_a_lock_that_a_stopped_process_left_is_taken_over(tmp_path: Path) -> None:
    with _fake_ibm(tmp_path / "nv_home") as ibm:
        root = tmp_path / "campaign"
        lock = root / "ibm_fez" / campaign.LOCK
        lock.parent.mkdir(parents=True)
        lock.write_text(f"{socket.gethostname()} 99999999\n")
        statuses = _run(root, ibm, MONDAY, wait=False)
        assert [s.state for s in statuses] == ["submitted", "submitted"]
        assert not lock.exists()
