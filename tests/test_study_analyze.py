from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from noisevault.counts import MeasuredCounts, plan, simulate
from noisevault.errors import SourceUnavailable
from noisevault.profile import Profile
from noisevault.suite import role, suite
from study import analyze, models

RUN_AT = datetime(2026, 4, 16, 9, 30, tzinfo=UTC)


def _device(**readout: float) -> Profile:
    profile = Profile.uniform(
        "ibm_line",
        technology="superconducting",
        num_qubits=3,
        connectivity=[(0, 1), (1, 2)],
        one_qubit_error=4e-3,
        two_qubit_error=3e-2,
        readout_error=3e-2,
        t1_us=100.0,
        t2_us=80.0,
        one_qubit_ns=30.0,
        two_qubit_ns=60.0,
    )
    data = profile.to_dict()
    data["device"] |= {"vendor": "ibm", "calibrated_at": (RUN_AT - timedelta(hours=30)).isoformat()}
    data["gates"] = {k: v for k, v in data["gates"].items() if k in ("rz", "sx", "x", "cz")}
    data["qubits"] = [
        {"index": 0, "readout": {"p1_given_0": 0.0, "p0_given_1": 0.04}},
        {"index": 1, "readout": {"p1_given_0": 0.02, "p0_given_1": 0.05}},
    ]
    return Profile.model_validate(data | readout)


class FakeSource:
    """IBM's public endpoint: C, an older calibration, and a calibration with no cz gate."""

    def __init__(self, calibration: Profile) -> None:
        def at(days: float, **changes: Any) -> Profile:
            data = calibration.to_dict()
            data["device"]["calibrated_at"] = (RUN_AT - timedelta(days=days)).isoformat()
            return Profile.model_validate(data | changes)

        broken = at(100).to_dict()
        broken["gates"].pop("cz")
        older = at(10, readout={"error": 0.06})
        self.calibrations = [calibration, older, Profile.model_validate(broken)]
        self.asked: list[datetime] = []

    def __call__(self, device: str, when: datetime) -> Profile:
        self.asked.append(when)
        older = [p for p in self.calibrations if p.device.calibrated_at < when]
        if not older:
            raise SourceUnavailable(f"IBM's public endpoint has no {device} calibration older than")
        return max(older, key=lambda p: p.device.calibrated_at)


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """One run simulated from C with gate factor 2, analyzed with spies on every fit."""
    root = tmp_path_factory.mktemp("analysis")
    calibration = _device()
    truth = models.floored(calibration).model_copy(
        update={"unmodeled_error": {"gates": {"factor": 2.0}}}
    )
    counts = simulate(truth, suite(calibration), shots=1000, seed=4, run_at=RUN_AT)
    counts = counts.model_copy(
        update={
            "profile": {"id": calibration.id, "fingerprint": calibration.calibration_fingerprint}
        }
    )
    week = root / "campaign" / "ibm_line" / "2026-04-13"
    week.mkdir(parents=True)
    counts.save(week / "best-deep.counts.json")
    calibration.save(week / "best-deep.profile.json")
    simulate(calibration, plan(calibration), shots=100, seed=1, run_at=RUN_AT).save(
        week / "best.counts.json"
    )
    calibration.save(week / "best.profile.json")
    (week / "._best-deep.counts.json").write_bytes(b"\x00\x05")

    seen: dict[str, list[list[str]]] = {}

    def spy(name: str) -> Any:
        real = getattr(models, name)

        def wrapper(profile: Profile, counts: MeasuredCounts, *args: Any) -> Any:
            seen.setdefault(name, []).append([c.name for c in counts.circuits])
            return real(profile, counts, *args)

        return wrapper

    source = FakeSource(calibration)
    with pytest.MonkeyPatch.context() as patch:
        for name in ("two_factor", "readout_measured", "gate_factor", "locus"):
            patch.setattr(models, name, spy(name))
        data = analyze.analyze(
            root / "campaign",
            cache=root / "cache",
            pull=source,
            properties=lambda device, at: None,
        )
    out = root / "results"
    analyze.write(data, out)
    yield {"data": data, "seen": seen, "out": out, "root": root, "source": source}


def _row(result: dict[str, Any]) -> dict[str, Any]:
    (row,) = result["data"]["rows"]
    return row


def _test(row: dict[str, Any], a: str, b: str, group: str) -> dict[str, Any]:
    return next(t for t in row["paired"] if (t["a"], t["b"], t["group"]) == (a, b, group))


def test_only_the_run_of_the_deep_set_is_analyzed(result: dict[str, Any]) -> None:
    data = result["data"]
    assert data["failures"] == []
    assert data["skipped"] == 1
    row = _row(result)
    assert row["chain"] == "best"
    assert row["shots"] == 1000
    assert row["calibration_age_hours"] == pytest.approx(30.0)
    assert row["marked"] is False


def test_fitted_models_never_receive_a_test_circuit(result: dict[str, Any]) -> None:
    seen = result["seen"]
    assert set(seen) == {"two_factor", "readout_measured", "gate_factor", "locus"}
    for calls in seen.values():
        for names in calls:
            assert {role(name) for name in names} == {"train"}


def test_two_factor_beats_the_calibration_when_the_device_has_more_gate_error(
    result: dict[str, Any],
) -> None:
    row = _row(result)
    test = _test(row, "two_factor", "calibration", "test_near")
    assert test["winner"] == "two_factor"
    assert row["fits"]["two_factor"]["gates"]["factor"] == pytest.approx(2.0, rel=0.15)
    assert row["fits"]["readout_measured"]["gates"] == pytest.approx(2.0, rel=0.15)
    assert (
        row["scores"]["two_factor"]["test_near"]["log_score"]
        > (row["scores"]["calibration"]["test_near"]["log_score"])
    )


def test_the_stated_calibration_is_ruled_out_and_the_floored_one_is_not(
    result: dict[str, Any],
) -> None:
    row = _row(result)
    assert row["stated"]["zero_readouts"] == 1
    assert row["stated"]["ruled_out"]["all"] > 0
    assert row["scores"]["calibration"]["all"]["ruled_out"] == 0


def test_each_model_has_a_score_or_a_reason(result: dict[str, Any]) -> None:
    row = _row(result)
    assert set(row["scores"]) | set(row["failures"]) == set(analyze.MODELS)
    assert not set(row["scores"]) & set(row["failures"])
    assert set(row["scores"]) >= {"ideal", "generic", "medians", "stale(7)", "locus"}
    assert row["failures"]["stale(1)"].startswith("the source has no other calibration")
    assert row["failures"]["stale(180)"].startswith("the source has no other calibration")
    assert "cz" in row["failures"]["stale(30)"]
    assert row["scores"]["ideal"]["all"]["ruled_out"] > 0


def test_stale_comparisons_use_every_circuit_and_the_cache(result: dict[str, Any]) -> None:
    row = _row(result)
    groups = {t["group"] for t in row["paired"] if t["b"] == "stale(7)"}
    assert groups == {"all"}
    assert {t["group"] for t in row["paired"] if t["b"] == "medians"} == {
        "all",
        "test_near",
        "test_far",
    }
    asked = len(result["source"].asked)
    again = models.stale(
        _device(), RUN_AT, 7, result["root"] / "cache", lambda d, a: pytest.fail("no cache")
    )
    assert again is not None
    assert len(result["source"].asked) == asked


def test_the_deep_table_gives_the_ideal_result_of_each_deep_circuit(
    result: dict[str, Any],
) -> None:
    row = _row(result)
    names = [d["name"] for d in row["deep"]]
    assert any(n.startswith("deep_") for n in names)
    assert any(n.startswith("periodic_") for n in names)
    assert "readout" not in names
    for deep in row["deep"]:
        assert deep["models"]["ideal"] == pytest.approx(1.0)
        assert 0 < deep["measured"] <= 1
        assert deep["models"]["calibration"] < 1


def _keys(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


def test_the_output_holds_no_counts_and_no_path(result: dict[str, Any]) -> None:
    out, root = result["out"], result["root"]
    text = (out / "analysis.json").read_text() + (out / "analysis.md").read_text()
    data = json.loads((out / "analysis.json").read_text())
    assert "counts" not in set(_keys(data))
    assert "circuits" not in set(_keys(data["rows"][0]["scores"]))
    assert str(root) not in text
    assert "best-deep" not in text
    assert ".json" not in text
    assert "| two_factor | calibration | test_near | two_factor |" in text


def test_the_command_line_skips_a_tree_with_no_deep_run(tmp_path: Path) -> None:
    calibration = _device()
    simulate(calibration, plan(calibration), shots=100, seed=1, run_at=RUN_AT).save(
        tmp_path / "best.counts.json"
    )
    calibration.save(tmp_path / "best.profile.json")

    assert analyze.main([str(tmp_path), "--out", str(tmp_path / "out"), "--offline"]) == 1
    data = json.loads((tmp_path / "out" / "analysis.json").read_text())
    assert data["rows"] == [] and data["skipped"] == 1
    assert analyze.main([str(tmp_path / "missing")]) == 2


def test_the_primary_comparisons_are_the_five_of_the_plan(result: dict[str, Any]) -> None:
    row = _row(result)
    primary = [(t["a"], t["b"], t["group"]) for t in row["paired"] if t["primary"]]
    assert set(primary) <= set(analyze.PRIMARY)
    assert ("two_factor", "calibration", "test") in primary
    assert ("locus", "two_factor", "test") in primary
    assert ("calibration", "medians", "all") in primary
    secondary = {(t["a"], t["b"], t["group"]) for t in row["paired"] if not t["primary"]}
    assert ("two_factor", "calibration", "test_near") in secondary
    assert row["circuits"]["test"] == row["circuits"]["test_near"] + row["circuits"]["test_far"]


def test_the_ages_of_a_run_come_from_the_reply_in_effect_at_each_time() -> None:
    calibration = _device()
    counts = simulate(calibration, plan(calibration), shots=10, seed=1, run_at=RUN_AT)
    asked: list[datetime] = []

    def properties(device: str, at: datetime) -> dict[str, Any] | None:
        asked.append(at)
        if at < RUN_AT - timedelta(days=100):
            return None
        stamp = (at - timedelta(hours=5)).isoformat()
        value = {"name": "prob_meas1_prep0", "value": 0.01, "date": stamp}
        return {"last_update_date": at.isoformat(), "qubits": [[value], [value]], "gates": []}

    found = analyze._ages(calibration, counts, None, properties)

    assert asked[0] == RUN_AT and len(asked) == 1 + len(analyze.STALE_DAYS)
    assert set(found) == {"calibration", "stale(1)", "stale(7)", "stale(30)", "stale(90)"}
    assert found["calibration"]["calibration_h"] == 0
    assert found["calibration"]["values"]["readout"]["median_h"] == 5
    assert found["stale(7)"]["values"]["readout"]["median_h"] == 7 * 24 + 5
    assert found["stale(7)"]["values"]["t1"] is None


def test_a_device_of_another_vendor_has_no_ages() -> None:
    data = _device().to_dict()
    data["device"]["vendor"] = "iqm"
    other = Profile.model_validate(data)
    counts = simulate(other, plan(other), shots=10, seed=1, run_at=RUN_AT)
    assert analyze._ages(other, counts, None, lambda device, at: pytest.fail("asked")) == {}


def test_each_model_starts_from_the_stated_calibration_and_is_floored_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _device().to_dict()
    data["qubits"] = [
        {"index": 0, "readout": {"p1_given_0": 0.0, "p0_given_1": 0.04}},
        {"index": 1, "readout": {"p1_given_0": 0.02, "p0_given_1": 0.08}},
        {"index": 2, "readout": {"p1_given_0": 0.0, "p0_given_1": 0.03}},
    ]
    calibration = Profile.model_validate(data)
    counts = simulate(calibration, suite(calibration), shots=50, seed=1)
    monkeypatch.setattr(models, "fit_factor", lambda build, counts, circuits: 1.0)

    built, _, _ = analyze.build(calibration, counts, cache=None, offline=True)

    assert built["symmetric"].table.qubit(0).readout == pytest.approx((0.02, 0.02))
    assert built["medians"].table.qubit(0).readout == pytest.approx((0.02, 0.04))
    for name in ("medians", "symmetric", "calibration", "two_factor", "readout_measured", "locus"):
        assert models.zero_readouts(built[name]) == 0
