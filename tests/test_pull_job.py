"""nv pull DEVICE --job JOB_ID: the calibration in effect when an IBM Quantum job ran."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

pytest.importorskip("qiskit")

from test_counts_from_qiskit import FakeBackend, FakeJob, kingston
from typer.testing import CliRunner

import noisevault as nv
from noisevault.cli import app
from noisevault.profile import Profile

STARTED = datetime(2026, 4, 16, 9, 30, 2, 500000, tzinfo=UTC)
runner = CliRunner()


@pytest.fixture
def pulls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, datetime | None]]:
    """IBM's public endpoint, answered with the bundled calibration of ibm_kingston."""
    from noisevault.sources import ibm_public

    seen: list[tuple[str, datetime | None]] = []

    def pull(device: str, at: datetime | None = None) -> Profile:
        seen.append((device, at))
        return kingston()

    monkeypatch.setattr(ibm_public, "pull", pull)
    return seen


class FakeService:
    def __init__(self, backend: str) -> None:
        self._backend = backend

    def job(self, job_id: str) -> FakeJob:
        return FakeJob([], [], backend=FakeBackend(self._backend))


def _account(monkeypatch: pytest.MonkeyPatch, backend: str = "ibm_kingston") -> None:
    from noisevault.sources import ibm_account

    monkeypatch.setattr(ibm_account, "_service", lambda: FakeService(backend))


def test_pull_job_saves_the_calibration_of_the_job_time_and_prints_the_time(
    monkeypatch, pulls
) -> None:
    _account(monkeypatch)
    result = runner.invoke(app, ["pull", "ibm_kingston", "--job", "d1job"])
    assert result.exit_code == 0, result.output
    assert pulls == [("ibm_kingston", STARTED)]
    assert result.stdout.splitlines()[0] == (
        "job d1job started at 2026-04-16 09:30:02Z. This is the calibration in effect at that time."
    )
    assert result.stdout.splitlines()[-1].startswith("saved: ")


def test_pull_job_in_python_takes_the_job_time(monkeypatch, pulls) -> None:
    _account(monkeypatch)
    assert nv.pull("ibm_kingston", job="d1job").fingerprint == kingston().fingerprint
    assert pulls == [("ibm_kingston", STARTED)]


def test_pull_job_refuses_a_device_that_is_not_the_backend_of_the_job(monkeypatch, pulls) -> None:
    _account(monkeypatch, backend="ibm_fez")
    result = runner.invoke(app, ["pull", "ibm_kingston", "--job", "d1job"])
    assert result.exit_code == 1
    assert "error: job d1job ran on ibm_fez, not on ibm_kingston" in result.stderr
    assert "hint: pull ibm_fez, the device of the job" in result.stderr
    assert pulls == []


def test_pull_takes_at_or_job_but_not_both(monkeypatch, pulls) -> None:
    _account(monkeypatch)
    result = runner.invoke(app, ["pull", "ibm_kingston", "--job", "d1job", "--at", "2026-04-16"])
    assert result.exit_code == 1
    assert "error: give --at or --job, not both" in result.stderr
    assert pulls == []
