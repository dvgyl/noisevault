from __future__ import annotations

import contextlib
import io
import json
import shutil
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from fake_ibm import FakeIBM, load_script, noisy_fake

import noisevault as nv
from noisevault.compare import compare
from noisevault.counts import load_counts

campaign = load_script("ibm_campaign")
drift = load_script("drift_report")

WEEKS = (date(2026, 11, 2), date(2026, 11, 9), date(2026, 11, 16))
# IBM recalibrates before weeks 2 and 3: (error scale, calibration time). Each time is in the
# past, because counts must run after their calibration.
RECALIBRATIONS = {
    date(2026, 11, 9): (1.5, datetime(2026, 1, 12, 6, tzinfo=UTC)),
    date(2026, 11, 16): (0.8, datetime(2026, 1, 19, 6, tzinfo=UTC)),
}


@dataclass
class Weeks:
    root: Path
    report: dict[str, Any]


@pytest.fixture(scope="module")
def weeks(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Weeks]:
    """Three weekly campaign runs on a fake ibm_fez. IBM recalibrates before weeks 2 and 3."""
    folder = tmp_path_factory.mktemp("drift")
    root = folder / "campaign"
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("NOISEVAULT_HOME", str(folder / "nv_home"))
        ibm = FakeIBM(monkeypatch, {"ibm_fez": noisy_fake()})
        for week in WEEKS:
            if week in RECALIBRATIONS:
                ibm.recalibrate("ibm_fez", *RECALIBRATIONS[week])
            account = ibm.account(campaign)
            run = campaign.Campaign(
                root=root, account=account, budget=campaign.Budget(None, 0.0), shots=2000
            )
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                statuses = run.run(["ibm_fez"], week)
            assert {s.state for s in statuses} == {"collected"}, statuses
        yield Weeks(root, drift.report(root))


def test_each_date_and_chain_has_a_gate_and_a_readout_factor(weeks: Weeks) -> None:
    dates = weeks.report["devices"]["ibm_fez"]["dates"]
    assert [d["date"] for d in dates] == [w.isoformat() for w in WEEKS]
    for entry in dates:
        week = weeks.root / "ibm_fez" / entry["date"]
        chains = json.loads((week / "chains.json").read_text())
        for kind in ("best", "median"):
            fit = entry["chains"][kind]
            assert fit["state"] == "fitted"
            assert fit["qubits"] == chains[kind]["qubits"]
            for axis in ("gates", "readout"):
                factor = fit[axis]
                assert "not_identified" not in factor, factor
                assert factor["factor"] > 0
                low, high = factor["low"], factor["high"]
                assert (low is None or low <= factor["factor"]) and (
                    high is None or factor["factor"] <= high
                )


def test_consecutive_dates_show_the_calibration_change(weeks: Weeks) -> None:
    changes = weeks.report["devices"]["ibm_fez"]["changes"]
    assert [(c["from"], c["to"]) for c in changes] == [
        ("2026-11-02", "2026-11-09"),
        ("2026-11-09", "2026-11-16"),
    ]
    for change, scale in zip(changes, (1.5, 0.8), strict=True):
        assert not change["identical"]
        medians = {m["metric"]: m for m in change["medians"]}
        assert medians["error_1q"]["relative"] == pytest.approx(scale - 1, abs=0.02)
        assert medians["readout_error"]["relative"] == pytest.approx(scale - 1, abs=0.02)
        assert change["qubits"] and len(change["qubits"]) <= drift.TOP
        assert change["summary"].startswith("ibm_fez@")


def test_the_markdown_has_a_row_per_date_and_chain_and_each_change(weeks: Weeks) -> None:
    text = drift.markdown(weeks.report)
    assert text.startswith("# Drift report\n")
    for week in WEEKS:
        for kind in ("best", "median"):
            assert f"| {week.isoformat()} | {kind} | " in text
    assert "#### 2026-11-02 to 2026-11-09" in text
    assert "#### 2026-11-09 to 2026-11-16" in text
    assert "(95% interval" in text


def test_main_writes_json_and_markdown(weeks: Weeks, tmp_path: Path) -> None:
    out = tmp_path / "out"
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        assert drift.main(["--root", str(weeks.root), "-o", str(out)]) == 0
    data = json.loads((out / "drift.json").read_text())
    assert data["nv_drift"] == drift.REPORT_FORMAT
    assert list(data["devices"]) == ["ibm_fez"]
    assert (out / "drift.md").read_text() == drift.markdown(data)
    assert printed.getvalue() == f"wrote {out / 'drift.json'}\nwrote {out / 'drift.md'}\n"


def test_a_pending_job_and_a_missing_profile_show_in_the_report(
    weeks: Weeks, tmp_path: Path
) -> None:
    root = tmp_path / "campaign"
    shutil.copytree(weeks.root / "ibm_fez" / "2026-11-02", root / "ibm_fez" / "2026-11-02")
    week = root / "ibm_fez" / "2026-11-02"
    (week / "median.counts.json").rename(week / "median.job.json")
    (week / "best.profile.json").unlink()
    (week / "._best.counts.json").write_bytes(b"")
    (root / "ibm_fez" / "._2026-11-09").mkdir()
    found = drift.report(root)["devices"]["ibm_fez"]
    (entry,) = found["dates"]
    assert entry["chains"]["median"] == {"state": "submitted"}
    assert entry["chains"]["best"]["state"] == "error"
    assert "no profile file best.profile.json" in entry["chains"]["best"]["error"]
    assert found["changes"] == []
    assert "One date only" in drift.markdown(drift.report(root))


def test_main_names_a_missing_campaign_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert drift.main(["--root", str(tmp_path / "none")]) == 1
    assert capsys.readouterr().err.startswith(f"error: no campaign folder {tmp_path / 'none'}\n")


def test_each_fit_is_the_nv_compare_fit_of_the_counts_on_their_bound_profile(
    weeks: Weeks,
) -> None:
    for entry in weeks.report["devices"]["ibm_fez"]["dates"]:
        week = weeks.root / "ibm_fez" / entry["date"]
        for kind in ("best", "median"):
            profile = nv.load(week / f"{kind}.profile.json")
            expected = compare(profile, load_counts(week / f"{kind}.counts.json")).to_dict()
            fit = entry["chains"][kind]
            assert fit == {"state": "fitted", **{k: expected[k] for k in drift.FIT_KEYS}}


def test_the_fit_reads_the_bound_profile_and_the_change_reads_calibration_json(
    weeks: Weeks, tmp_path: Path
) -> None:
    source = weeks.root / "ibm_fez"
    root = tmp_path / "campaign"
    shutil.copytree(source / "2026-11-02", root / "ibm_fez" / "2026-11-02")
    shutil.copytree(source / "2026-11-02", root / "ibm_fez" / "2026-11-03")
    shutil.copy(source / "2026-11-09" / "calibration.json", root / "ibm_fez" / "2026-11-03")
    found = drift.report(root)["devices"]["ibm_fez"]
    first, second = found["dates"]
    assert first["chains"] == second["chains"]
    assert (
        second["calibration"]
        == nv.load(source / "2026-11-09" / "calibration.json").calibration_fingerprint
    )
    (change,) = found["changes"]
    assert not change["identical"]
    assert (
        change["after_fingerprint"]
        == nv.load(source / "2026-11-09" / "calibration.json").fingerprint
    )


def test_damaged_files_give_errors_and_the_report_goes_on(weeks: Weeks, tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    for name in ("2026-11-02", "2026-11-09"):
        shutil.copytree(weeks.root / "ibm_fez" / name, root / "ibm_fez" / name)
    week = root / "ibm_fez" / "2026-11-09"
    (week / "calibration.json").write_text('{"noisevault": ')
    (week / "best.profile.json").write_text("[]")
    (week / "median.counts.json").write_text("{")
    found = drift.report(root)["devices"]["ibm_fez"]
    second = found["dates"][1]
    assert second["calibration"] is None and "calibration.json" in second["calibration_error"]
    assert second["chains"]["best"]["state"] == "error"
    assert "best.profile.json" in second["chains"]["best"]["error"]
    assert second["chains"]["median"]["state"] == "error"
    assert found["dates"][0]["chains"]["best"]["state"] == "fitted"
    (change,) = found["changes"]
    assert "calibration.json" in change["error"]
    drift.markdown(drift.report(root))


def _fitted(gates: dict[str, Any], readout: dict[str, Any]) -> dict[str, Any]:
    return {"state": "fitted", "qubits": [1, 2], "p_value": 0.5, "gates": gates, "readout": readout}


def test_the_markdown_shows_bounds_unidentified_factors_and_escaped_errors() -> None:
    data = {
        "generated_at": "2026-11-09T00:00:00Z",
        "devices": {
            "ibm_fez": {
                "dates": [
                    {
                        "date": "2026-11-02",
                        "chains": {
                            "best": _fitted(
                                {"factor": 1.2, "low": None, "high": 1.5, "bound": "lower"},
                                {"factor": 2.0, "low": 1.5, "high": None, "bound": "upper"},
                            ),
                            "median": {"state": "error", "error": "a\\|b <c>"},
                        },
                    },
                    {
                        "date": "2026-11-09",
                        "chains": {
                            "best": _fitted({"not_identified": "x | y"}, {"factor": 1.0}),
                            "median": {"state": "submitted"},
                        },
                    },
                ],
                "changes": [],
            },
            "ibm_torino": {"dates": [], "changes": []},
        },
    }
    lines = drift.markdown(data).splitlines()
    assert (
        "| 2026-11-02 | best | 1-2 | x1.2 (95% interval, at most 1.5) |"
        + (" x2 (95% interval, at least 1.5) | 0.5 |")
        in lines
    )
    assert "| 2026-11-02 | median | | a\\\\\\|b &lt;c&gt; | | |" in lines
    assert "| 2026-11-09 | best | 1-2 | not identified: x \\| y | x1 | 0.5 |" in lines
    assert "| 2026-11-09 | median | | submitted | | |" in lines
    assert "No dates yet." in lines
    for line in lines:
        if line.startswith("| 2026"):
            assert len(line.replace("\\|", "").split("|")) == 8, line
