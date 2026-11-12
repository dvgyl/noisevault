from __future__ import annotations

import copy
import json
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import require

from noisevault.counts import MeasuredCounts, load_counts, plan, simulate
from noisevault.errors import SourceUnavailable
from noisevault.profile import Profile
from noisevault.reference import MAX_QUBITS
from noisevault.suite import role, suite
from study import analyze, models, summarize, wide

RUN_AT = datetime(2026, 4, 16, 9, 30, tzinfo=UTC)
CLIFFORD_SAMPLES = 200_000


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


AXIS = ["axis_d64", "axis_d256"]


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """One run simulated from C with gate factor 2, analyzed with spies on every fit. The suite
    holds the probe circuits."""
    root = tmp_path_factory.mktemp("analysis")
    calibration = _device()
    truth = models.floored(calibration).model_copy(
        update={"unmodeled_error": {"gates": {"factor": 2.0}}}
    )
    circuits = suite(calibration)
    assert [c.name for c in circuits if c.name.startswith("axis_")] == AXIS
    counts = simulate(truth, circuits, shots=1000, seed=4, run_at=RUN_AT)
    counts = counts.model_copy(
        update={
            "profile": {"id": calibration.id, "fingerprint": calibration.calibration_fingerprint}
        }
    )
    week = root / "campaign" / "ibm_line" / "2026-04-13"
    week.mkdir(parents=True)
    counts.save(week / "best-deep.counts.json")
    counts.save(week / "best-deep.counts.json.gz")
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
    assert data["duplicates"] == 1
    row = _row(result)
    assert row["chain"] == "best" and row["trial"] is False
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
    assert row["stated"]["zero_on_chain"] is True
    assert row["stated"]["ruled_out"]["all4"] > 0
    assert row["scores"]["calibration"]["all4"]["ruled_out"] == 0


def test_a_zero_off_the_chain_is_not_a_zero_on_the_chain() -> None:
    calibration = _device()
    assert analyze.zero_on(calibration, [0]) is True
    assert analyze.zero_on(calibration, [1, 2]) is False


def test_a_fresh_deep_circuit_is_a_test_circuit_of_the_chain() -> None:
    calibration = _device()
    counts = simulate(calibration, suite(calibration, fresh=5), shots=20, seed=1, run_at=RUN_AT)

    named = analyze.groups(counts)

    fresh = [c.name for c in counts.circuits if c.name.endswith("_f5")]
    assert fresh and all(n.startswith("deep_") for n in fresh)
    assert set(fresh) <= set(named["test_near"]) & set(named["test4"]) & set(named["all4"])


def test_each_model_has_a_score_or_a_reason(result: dict[str, Any]) -> None:
    row = _row(result)
    assert set(row["scores"]) | set(row["failures"]) == set(analyze.MODELS)
    assert not set(row["scores"]) & set(row["failures"])
    assert set(row["scores"]) >= {"ideal", "generic", "medians", "stale(7)", "locus"}
    assert row["failures"]["stale(1)"].startswith("the source has no other calibration")
    assert row["failures"]["stale(180)"].startswith("the source has no other calibration")
    # The reason gives the class of the error and the class of its first cause.
    assert row["failures"]["stale(30)"] == "CountsError from MissingCalibrationError"
    assert row["scores"]["ideal"]["all4"]["ruled_out"] > 0


def test_stale_comparisons_use_every_circuit_of_the_chain_and_the_cache(
    result: dict[str, Any],
) -> None:
    row = _row(result)
    groups = {t["group"] for t in row["paired"] if t["b"] == "stale(7)"}
    assert groups == {"all4"}
    assert {t["group"] for t in row["paired"] if t["b"] == "medians"} == {
        "all4",
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
    # A probe circuit, such as the periodic circuit, is in no group, so it has no deep row.
    periodic = [c["name"] for c in row["probes"]["circuits"] if c["name"].startswith("periodic_")]
    assert bool(periodic) == (role("periodic_d1") == "probe")
    assert not any(n.startswith("periodic_") for n in names) or not periodic
    assert not any(role(n) == "probe" for n in names)
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
    assert ("two_factor", "calibration", "test4") in primary
    assert ("locus", "two_factor", "test4") in primary
    assert ("calibration", "medians", "all4") in primary
    secondary = {(t["a"], t["b"], t["group"]) for t in row["paired"] if not t["primary"]}
    assert ("two_factor", "calibration", "test_near") in secondary
    assert row["circuits"]["test4"] == row["circuits"]["test_near"] + row["circuits"]["test_far"]


def test_the_groups_hold_only_circuits_of_the_chain_and_no_probe(result: dict[str, Any]) -> None:
    root = result["root"] / "campaign" / "ibm_line" / "2026-04-13"
    counts = load_counts(root / "best-deep.counts.json")
    named = analyze.groups(counts)
    chain = set(analyze.chain_of(counts))
    by_name = {c.name: c for c in counts.circuits}

    expected = [
        c.name for c in counts.circuits if set(c.qubits) <= chain and role(c.name) != "probe"
    ]
    assert named["all4"] == expected
    assert named["test4"] == [n for n in expected if role(n) in ("test_near", "test_far")]
    assert set(AXIS) <= set(by_name)
    for names in named.values():
        assert not set(AXIS) & set(names)
    assert _row(result)["circuits"] == {g: len(names) for g, names in named.items()}


def test_a_wide_circuit_and_a_probe_circuit_are_in_no_primary_group(clifford_result) -> None:
    (row,) = clifford_result["data"]["rows"]
    named = clifford_result["groups"]
    wide_names = [n for n in clifford_result["small"] if n.startswith("wide_")]

    assert wide_names and named["wide"] == wide_names
    for group in ("all4", "test4", "test_near", "test_far"):
        assert not set(wide_names) & set(named[group])
        assert not any(role(n) == "probe" for n in named[group])
    for test in row["paired"]:
        if test["primary"]:
            assert test["group"] in ("all4", "test4")
    assert row["wide_chain_overlap"] == len(
        set(clifford_result["wide_qubits"]) & set(clifford_result["chain"])
    )


class FakeProperties:
    """IBM's properties endpoint. The device was calibrated 30 h before the run (C), 2 h before
    the run, and 1 h before each stale time. Each value is 5 h older than its calibration. A
    calibration is published ``delay`` after its time."""

    def __init__(
        self,
        *,
        without: datetime | None = None,
        delay: timedelta = timedelta(0),
        extra: Sequence[datetime] = (),
    ) -> None:
        self.times = [RUN_AT - timedelta(hours=2), RUN_AT - timedelta(hours=30), *extra]
        self.times += [RUN_AT - timedelta(days=d, hours=1) for d in analyze.STALE_DAYS if d < 100]
        self.times = [t for t in self.times if t != without]
        self.delay = delay
        self.asked: list[datetime] = []

    def __call__(self, device: str, at: datetime) -> dict[str, Any] | None:
        self.asked.append(at)
        older = [t for t in self.times if t + self.delay < at]
        if not older:
            return None
        newest = max(older)
        stamp = (newest - timedelta(hours=5)).isoformat()
        value = {"name": "prob_meas1_prep0", "value": 0.01, "date": stamp}
        return {"last_update_date": newest.isoformat(), "qubits": [[value], [value]], "gates": []}


def test_the_ages_of_a_run_come_from_the_reply_of_each_calibration() -> None:
    calibration = _device()
    counts = simulate(calibration, plan(calibration), shots=10, seed=1, run_at=RUN_AT)
    properties = FakeProperties()

    found = analyze._ages(calibration, counts, None, properties)

    c = calibration.device.calibrated_at
    assert properties.asked[0] == c + timedelta(seconds=1)
    assert len(properties.asked) == 1 + len(analyze.STALE_DAYS)
    assert set(found) == {"calibration", "stale(1)", "stale(7)", "stale(30)", "stale(90)"}
    assert found["calibration"]["calibration_h"] == pytest.approx(30)
    assert found["calibration"]["values"]["readout"]["median_h"] == pytest.approx(35)
    assert found["stale(7)"]["values"]["readout"]["median_h"] == pytest.approx(7 * 24 + 6)
    assert found["stale(7)"]["values"]["t1"] is None


def test_the_calibration_has_no_ages_when_the_source_has_no_reply_of_c() -> None:
    calibration = _device()
    counts = simulate(calibration, plan(calibration), shots=10, seed=1, run_at=RUN_AT)
    c = calibration.device.calibrated_at

    found = analyze._ages(calibration, counts, None, FakeProperties(without=c))

    assert "calibration" not in found
    assert found["missing"] == {"calibration": "other_calibration"}
    assert "stale(7)" in found


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


def test_the_week_model_takes_the_calibrations_in_effect_at_the_run_time_and_6_days_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = _device()
    counts = simulate(calibration, plan(calibration), shots=10, seed=1, run_at=RUN_AT)
    source = FakeSource(calibration)
    given: list[list[datetime | None]] = []
    real = models.averaged

    def spy(profile: Profile, calibrations: list[Profile]) -> Profile:
        given.append([p.device.calibrated_at for p in calibrations])
        return real(profile, calibrations)

    def skipped(*args: Any) -> Any:
        raise ValueError("not needed")

    monkeypatch.setattr(models, "averaged", spy)
    for name in ("two_factor", "readout_measured", "locus"):
        monkeypatch.setattr(models, name, skipped)

    built, _, _ = analyze.build(calibration, counts, cache=None, offline=False, pull=source)

    c, older = (
        source.calibrations[0].device.calibrated_at,
        source.calibrations[1].device.calibrated_at,
    )
    assert given == [[c, c] + [older] * 5]
    assert source.asked[-7:] == [RUN_AT - timedelta(days=d) for d in range(7)]
    assert "averaged(7)" in built


def test_a_failure_reason_holds_no_message_and_no_input_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    secret = "job-d4c0ffee-secret"
    data = _device().to_dict()
    data["provenance"]["extra"] = {"job_id": secret}
    calibration = Profile.model_validate(data)
    counts = simulate(calibration, suite(calibration), shots=20, seed=1, run_at=RUN_AT)
    path = tmp_path / f"best-{secret}.counts.json"

    def invalid(profile: Profile) -> Profile:
        broken = profile.to_dict()
        broken["qubits"] = [{"index": 99, "readout": {"error": 0.0123}}]
        return Profile.model_validate(broken)

    def unreadable(profile: Profile) -> Profile:
        raise SourceUnavailable(f"cannot read {path}: value 0.0123")

    def fails(profile: Profile, counts: MeasuredCounts) -> Any:
        raise ValueError(f"{secret} 0.0123")

    monkeypatch.setattr(models, "generic", invalid)
    monkeypatch.setattr(models, "bundled", unreadable)
    monkeypatch.setattr(models, "two_factor", fails)
    monkeypatch.setattr(models, "fit_factor", lambda build, counts, circuits: 1.0)

    _, missing, _ = analyze.build(calibration, counts, cache=None, offline=True)

    assert missing["generic"] == "ValidationError at (profile)"
    assert missing["bundled"] == "SourceUnavailable"
    assert missing["two_factor"] == "ValueError"
    text = json.dumps(missing)
    assert secret not in text and "0.0123" not in text and str(tmp_path) not in text


def _line_24() -> Profile:
    """24 IBM qubits in a line, so that the suite holds a Clifford circuit on 20 qubits."""
    data = Profile.uniform(
        "ibm_line",
        technology="superconducting",
        num_qubits=24,
        connectivity=[(q, q + 1) for q in range(23)],
        one_qubit_error=2e-3,
        two_qubit_error=2e-2,
        readout_error=2e-2,
        t1_us=100.0,
        t2_us=80.0,
        one_qubit_ns=32.0,
        two_qubit_ns=68.0,
    ).to_dict()
    data["device"] |= {"vendor": "ibm", "calibrated_at": (RUN_AT - timedelta(hours=30)).isoformat()}
    data["gates"] = {k: v for k, v in data["gates"].items() if k in ("rz", "sx", "x", "cz")}
    return Profile.model_validate(data)


@pytest.fixture(scope="module")
def clifford_result(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One run on 24 qubits simulated with gate factor 2: the small circuits from the exact
    simulator and the Clifford circuit from the stabilizer simulator."""
    require("stim")
    root = tmp_path_factory.mktemp("clifford")
    calibration = _line_24()
    truth = models.floored(calibration).model_copy(
        update={"unmodeled_error": {"gates": {"factor": 2.0}}}
    )
    circuits = suite(calibration)
    small = [c for c in circuits if len(c.qubits) <= MAX_QUBITS]
    large = [c for c in circuits if len(c.qubits) > MAX_QUBITS]
    exact = simulate(truth, small, shots=1000, seed=4, run_at=RUN_AT).to_dict()
    stabilized = wide.simulate(truth, large, shots=1000, seed=5, run_at=RUN_AT).to_dict()
    data = exact | {
        "profile": {"id": calibration.id, "fingerprint": calibration.calibration_fingerprint},
        "circuits": [*exact["circuits"], *stabilized["circuits"]],
    }
    week = root / "campaign" / "ibm_line" / "2026-04-13"
    week.mkdir(parents=True)
    counts = MeasuredCounts.model_validate(data)
    counts.save(week / "best-deep.counts.json")
    calibration.save(week / "best-deep.profile.json")
    analysis = analyze.analyze(root / "campaign", offline=True, samples=CLIFFORD_SAMPLES)
    return {
        "data": analysis,
        "small": [c.name for c in small],
        "large": [c.name for c in large],
        "groups": analyze.groups(counts),
        "chain": analyze.chain_of(counts),
        "wide_qubits": {q for c in small if c.name.startswith("wide_") for q in c.qubits},
    }


def test_the_clifford_block_ranks_two_factor_above_the_calibration(clifford_result) -> None:
    (row,) = clifford_result["data"]["rows"]
    block = row["clifford"]

    assert [c["name"] for c in block["circuits"]] == clifford_result["large"]
    (circuit,) = block["circuits"]
    assert circuit["qubits"] == 20 and circuit["shots"] == 1000
    assert set(circuit["models"]) == {
        "generic",
        "medians",
        "symmetric",
        "calibration",
        "two_factor",
    }
    calibration = circuit["models"]["calibration"]
    assert calibration["predicted"]["wrong_fraction"] < circuit["measured"]["wrong_fraction"]
    assert set(calibration["scores"]) == {
        "success_log",
        "success_quadratic",
        "wrong_log",
        "wrong_quadratic",
        "qubit_log",
    }
    together = next(
        t
        for t in block["paired"]
        if (t["a"], t["b"], t["circuits"]) == ("two_factor", "calibration", "all")
    )
    assert together["winner"] == "two_factor"
    assert together["qubit"]["log"]["better"] == "a"
    assert 0 < together["qubit"]["sampling_ratio"] < 1
    assert together["wrong"]["sampling_ratio"] > 0
    assert together["success"]["verdict"] is True
    assert block["samples"] == CLIFFORD_SAMPLES
    assert {(t["a"], t["b"]) for t in block["paired"]} == {
        ("calibration", "medians"),
        ("two_factor", "calibration"),
    }


def test_a_model_that_fails_on_a_wide_circuit_keeps_its_scores_on_the_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calibration = _line_24()
    small = [c for c in suite(calibration) if len(c.qubits) <= MAX_QUBITS]
    chain = next(c.qubits for c in small if c.name == "readout")
    wide_chain = next(c.qubits for c in small if c.name.startswith("wide_"))
    pairs = [(a, b) for a, b in zip(wide_chain, wide_chain[1:], strict=False)]
    pair = next(p for p in pairs if not set(p) <= set(chain))
    counts = simulate(calibration, small, shots=200, seed=2, run_at=RUN_AT)
    counts.save(tmp_path / "best-deep.counts.json")
    calibration.save(tmp_path / "best-deep.profile.json")
    real = models.medians

    def no_pair(profile: Profile) -> Profile:
        """The medians model with the cz gate of one pair of the wide chain disabled."""
        data = real(profile).to_dict()
        data["calibrations"] = [
            *data.get("calibrations", ()),
            *({"gate": "cz", "qubits": list(q), "disabled": True} for q in (pair, pair[::-1])),
        ]
        return Profile.model_validate(data)

    monkeypatch.setattr(models, "medians", no_pair)
    monkeypatch.setattr(models, "fit_factor", lambda build, counts, circuits: 1.0)

    row = analyze.analyze_run(tmp_path / "best-deep.counts.json", cache=None, offline=True)

    assert set(row["scores"]["medians"]) == {"all4", "test4", "test_near", "test_far"}
    assert row["failures"]["medians (wide)"].startswith("CountsError")
    assert "medians" not in row["failures"]
    assert set(row["scores"]["calibration"]) == {"all4", "test4", "test_near", "test_far", "wide"}
    tests = {(t["a"], t["b"], t["group"]) for t in row["paired"]}
    assert ("calibration", "medians", "all4") in tests
    assert ("calibration", "medians", "wide") not in tests
    assert all("medians" in d["models"] for d in row["deep"] if not d["name"].startswith("wide"))


def test_the_groups_of_the_exact_simulator_hold_no_circuit_above_10_qubits(
    clifford_result,
) -> None:
    (row,) = clifford_result["data"]["rows"]

    # Each circuit of 10 qubits or fewer is on the chain, a wide circuit or a probe circuit.
    probes = [n for n in clifford_result["small"] if role(n) == "probe"]
    assert [n for n in probes if n.startswith("axis_")] == AXIS
    assert row["circuits"]["all4"] + row["circuits"]["wide"] + len(probes) == len(
        clifford_result["small"]
    )
    assert [c["name"] for c in row["probes"]["circuits"]] == probes
    assert "calibration" not in row["failures"]
    assert len(row["qubits"]) < 20


def test_the_twirl_check_compares_the_two_forecasts_of_each_clifford_model(
    clifford_result,
) -> None:
    (row,) = clifford_result["data"]["rows"]

    names = [name for name in clifford_result["small"] if name.startswith("wide_")]
    models_ = [m for m in analyze.TWIRL_MODELS if m != "averaged(7)"]
    assert models_ == ["calibration", "medians", "two_factor"]
    assert [(t["model"], t["name"]) for t in row["twirl_check"]] == [
        (m, n) for m in models_ for n in names
    ]
    for check in row["twirl_check"]:
        assert check["difference"] == pytest.approx(check["stabilizer_log"] - check["exact_log"])
        assert abs(check["difference"]) < 0.05


def test_the_summary_of_a_pilot_run_has_the_twirl_check_and_no_clifford_value(
    clifford_result,
) -> None:
    summary = summarize.summarize(clifford_result["data"])

    text = summarize.markdown(summary)
    assert summary["periods"]["pilot"] == 1
    assert all(c["device_weeks"] == [] for c in summary["clifford"])
    assert len(summary["twirl_check"]) == len(clifford_result["data"]["rows"][0]["twirl_check"])
    assert "| ibm_line | best |" in text
    assert [(c["a"], c["b"]) for c in summary["primary"]] == [(a, b) for a, b, _ in analyze.PRIMARY]


def test_a_run_with_no_clifford_circuit_has_an_empty_clifford_block(result: dict[str, Any]) -> None:
    row = _row(result)

    assert row["clifford"] == {
        "samples": wide.SAMPLES,
        "circuits": [],
        "paired": [],
        "failures": {},
    }
    assert row["twirl_check"] == []


def test_the_probe_table_gives_the_measured_and_the_forecast_wrong_bits(
    result: dict[str, Any],
) -> None:
    row = _row(result)
    root = result["root"] / "campaign" / "ibm_line" / "2026-04-13"
    counts = load_counts(root / "best-deep.counts.json")
    table = row["probes"]
    chain = analyze.chain_of(counts)

    assert table["chain"] == list(chain)
    readout = next(c for c in counts.circuits if c.name == "readout")
    for i, base in enumerate(table["baseline"]):
        ones = sum(n for key, n in readout.counts.items() if key[i] == "1")
        assert base == {"qubit": chain[i], "measured": pytest.approx(ones / readout.shots)}
    probes = [c.name for c in counts.circuits if role(c.name) == "probe"]
    assert [c["name"] for c in table["circuits"]] == probes
    for probe in table["circuits"]:
        circuit = next(c for c in counts.circuits if c.name == probe["name"])
        if probe["name"].startswith("axis_"):
            # An axis circuit is a whole number of full rotations, so its ideal result is all 0.
            assert probe["ideal"] == "0" * len(chain)
        _check_probe(probe, circuit, chain)
    axis = [c for c in table["circuits"] if c["name"] in AXIS]
    assert [c["name"] for c in axis] == AXIS
    assert axis[1]["forecast"]["calibration"] < axis[0]["forecast"]["calibration"]


def _check_probe(probe: dict[str, Any], circuit: Any, chain: Sequence[int]) -> None:
    """The values of one probe of the table, computed again from the counts. A wrong bit of a
    qubit is a bit that differs from the ideal bit of that qubit."""
    shots = circuit.shots
    assert probe["measured"] == pytest.approx(circuit.counts.get(probe["ideal"], 0) / shots)
    assert set(probe["forecast"]) == {"calibration", "two_factor"}
    for i, q in enumerate(probe["qubits"]):
        wrong = sum(n for key, n in circuit.counts.items() if key[i] != probe["ideal"][i])
        assert q["qubit"] == chain[i]
        assert q["measured"] == pytest.approx(wrong / shots)
        assert set(q["forecast"]) == {"calibration", "two_factor"}
        assert 0 < q["forecast"]["calibration"] < q["forecast"]["two_factor"] < 1


def test_a_probe_with_an_ideal_result_that_is_not_all_0_uses_the_ideal_bit_of_each_qubit(
    result: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = result["root"] / "campaign" / "ibm_line" / "2026-04-13"
    counts = load_counts(root / "best-deep.counts.json")
    calibration = _device()
    chain = analyze.chain_of(counts)
    # The periodic circuit ends in a bit string that is not all 0. On main it is a probe.
    monkeypatch.setattr(
        analyze, "role", lambda name: "probe" if name.startswith("periodic_") else role(name)
    )
    built = {
        "ideal": models.ideal(calibration),
        "calibration": models.floored(calibration),
        "two_factor": models.floored(models.with_factors(calibration, 2.0)),
    }

    table = analyze._probes(counts, built, chain, {})

    (periodic,) = [c for c in table["circuits"] if c["name"].startswith("periodic_")]
    assert periodic["ideal"] != "0" * len(chain)
    _check_probe(periodic, next(c for c in counts.circuits if c.name == periodic["name"]), chain)
    probability = analyze._forecast_wrong(np.array([0.0, 0.7, 0.2, 0.1]), "01")
    # Outcomes 00, 01, 10 and 11: qubit 0 is wrong in 10 and 11, qubit 1 in 00 and 10.
    assert probability == pytest.approx([0.3, 0.2])
    # The growth ratio uses only the axis circuits, also when the periodic circuit has fewer
    # cycles than the shortest axis circuit.
    periodic["name"] = "periodic_d2"
    row = {"device": "d", "chain": "best", "run_at": RUN_AT.isoformat(), "probes": table}
    (found,) = summarize.probes([row])
    base = {b["qubit"]: b["measured"] for b in table["baseline"]}
    small, large = [c for c in table["circuits"] if c["name"] in AXIS]
    for g, s, big in zip(found["growth"], small["qubits"], large["qubits"], strict=True):
        divisor = s["measured"] - base[s["qubit"]]
        expected = (big["measured"] - base[s["qubit"]]) / divisor if divisor > 0 else None
        assert g["ratio"] == (None if expected is None else pytest.approx(expected))


def test_the_summary_gives_the_growth_ratio_of_each_probe_qubit(result: dict[str, Any]) -> None:
    row = _row(result)

    (found,) = summarize.probes([row])

    base = {b["qubit"]: b["measured"] for b in row["probes"]["baseline"]}
    small, large = [c for c in row["probes"]["circuits"] if c["name"] in AXIS]
    for g, s, big in zip(found["growth"], small["qubits"], large["qubits"], strict=True):
        divisor = s["measured"] - base[s["qubit"]]
        expected = (big["measured"] - base[s["qubit"]]) / divisor if divisor > 0 else None
        assert g["qubit"] == s["qubit"]
        assert g["ratio"] == (None if expected is None else pytest.approx(expected))
    text = summarize.markdown(summarize.summarize(result["data"]))
    assert "| axis_d256 |" in text and "Growth ratio" in text


def test_a_growth_ratio_with_a_divisor_not_above_0_has_no_value() -> None:
    def probe(name: str, wrong: float) -> dict[str, Any]:
        return {"name": name, "qubits": [{"qubit": 5, "measured": wrong, "forecast": {}}]}

    row = {
        "device": "ibm_fez",
        "chain": "best",
        "run_at": "2026-11-17T03:00:00Z",
        "probes": {
            "baseline": [{"qubit": 5, "measured": 0.02}],
            "circuits": [probe("axis_d64", 0.1), probe("axis_d8", 0.02)],
        },
    }
    assert summarize.probes([row])[0]["growth"] == [{"qubit": 5, "ratio": None}]
    row["probes"]["circuits"][1] = probe("axis_d8", 0.04)
    assert summarize.probes([row])[0]["growth"] == [{"qubit": 5, "ratio": pytest.approx(4.0)}]


def test_the_misfit_replaces_the_p_value_of_a_fitted_model(result: dict[str, Any]) -> None:
    row = _row(result)

    for name in analyze.FITTED:
        assert all("p_value" not in by_group for by_group in row["scores"][name].values())
    assert "p_value" in row["scores"]["calibration"]["all4"]
    assert set(row["misfit"]) == {"locus", "two_factor", "calibration"}
    for name, found in row["misfit"].items():
        scores = row["scores"][name]["test_far"]
        assert found["excess"] == pytest.approx(scores["excess"])
        assert found["misfit"] == pytest.approx(scores["excess"] - scores["excess_noise_mean"])
        assert 0 <= abs(found["rare"]) <= abs(found["excess"]) + 1e-9


def test_the_log_verdict_is_the_winner_of_each_comparison(result: dict[str, Any]) -> None:
    row = _row(result)
    for test in row["paired"]:
        expected = {"a": test["a"], "b": test["b"], "neither": None}[test["log"]["better"]]
        assert test["winner"] == expected


def test_the_chi_square_merges_rare_cells_and_has_k_minus_1_degrees_of_freedom() -> None:
    table = np.array([[50.0, 30.0, 18.0, 2.0, 0.0], [45.0, 35.0, 17.0, 1.0, 2.0]])

    statistic, df = analyze.chi_square(table)

    merged = np.array([[50.0, 30.0, 18.0, 2.0], [45.0, 35.0, 17.0, 3.0]])
    expected = merged.sum(axis=1, keepdims=True) * merged.sum(axis=0) / merged.sum()
    assert df == 3
    assert statistic == pytest.approx(((merged - expected) ** 2 / expected).sum())


def test_the_check_of_independent_shots_uses_only_the_same_circuits(tmp_path: Path) -> None:
    calibration = _device()
    circuits = suite(calibration)
    first = simulate(calibration, circuits, shots=4000, seed=1, run_at=RUN_AT)
    second = simulate(calibration, circuits, shots=4000, seed=2, run_at=RUN_AT)
    data = second.to_dict()
    data["circuits"][0]["ops"] = data["circuits"][0]["ops"][:-1]
    changed = MeasuredCounts.model_validate(data)
    paths = {}
    rows = []
    for kind, counts in (("best", first), ("median", changed)):
        path = tmp_path / f"{kind}-deep.counts.json"
        counts.save(path)
        paths[counts.sha256] = path
        rows.append(
            {"device": "ibm_line", "run_at": "2026-04-16T09:30:00Z", "chain": kind}
            | {"counts_sha256": counts.sha256}
        )

    (check,) = analyze._shot_checks(rows, paths)

    assert check["device"] == "ibm_line" and check["week"] == "2026-04-13"
    assert check["not_the_same"] == 1
    assert len(check["circuits"]) == len(circuits) - 1
    statistic = sum(c["statistic"] for c in check["circuits"])
    df = sum(c["df"] for c in check["circuits"])
    # Two runs of one distribution give a ratio near 1.
    assert 0.6 < statistic / df < 1.5
    for c in check["circuits"]:
        assert c["ratio"] == pytest.approx(c["statistic"] / c["df"])
    assert analyze._shot_checks(rows[:1], paths) == []
    # A trial run in the device-week does not change the check of the two normal runs.
    trial = rows[0] | {"trial": True, "counts_sha256": "trial"}
    assert analyze._shot_checks([*rows, trial], paths) == [check]


def test_a_trial_run_is_named_by_its_file_and_is_never_the_earlier_run() -> None:
    assert analyze.is_trial(Path("best-trial.counts.json"))
    assert analyze.is_trial(Path("median-trial.counts.json.gz"))
    assert not analyze.is_trial(Path("best-deep.counts.json"))
    assert analyze.chain_kind(Path("median-trial.counts.json")) == "median"
    row = {"device": "d", "chain": "best", "run_at": "2026-11-24T03:00:00Z"}
    trial = row | {"run_at": "2026-11-17T03:00:00Z", "trial": True}
    assert analyze.earlier_run(row, [trial, row]) is None


def test_the_ages_of_c_wait_for_the_publication_of_c() -> None:
    calibration = _device()
    counts = simulate(calibration, plan(calibration), shots=10, seed=1, run_at=RUN_AT)
    c = calibration.device.calibrated_at
    properties = FakeProperties(delay=timedelta(minutes=10))

    found = analyze._ages(calibration, counts, None, properties)

    assert properties.asked[:2] == [c + timedelta(seconds=1), c + timedelta(minutes=15)]
    assert found["calibration"]["calibration_h"] == pytest.approx(30)
    assert "missing" not in found


def test_the_ages_of_c_stop_at_a_later_calibration() -> None:
    calibration = _device()
    counts = simulate(calibration, plan(calibration), shots=10, seed=1, run_at=RUN_AT)
    c = calibration.device.calibrated_at
    later = c + timedelta(minutes=20)
    properties = FakeProperties(without=c, extra=[later])

    found = analyze._ages(calibration, counts, None, properties)

    waits = [c + wait for wait in analyze.PUBLICATION_WAITS]
    assert properties.asked[:3] == waits[:3]
    assert properties.asked[3] not in waits
    assert found["missing"] == {"calibration": "other_calibration"}


def test_the_ages_of_c_with_no_reply_at_all() -> None:
    calibration = _device()
    counts = simulate(calibration, plan(calibration), shots=10, seed=1, run_at=RUN_AT)

    found = analyze._ages(calibration, counts, None, lambda device, at: None)

    assert found == {"missing": {"calibration": "no_reply"}}


def test_a_run_alone_has_no_model_with_the_factors_of_an_earlier_run(
    result: dict[str, Any],
) -> None:
    assert _row(result)["before"] == {"missing": "no_earlier_run"}


def test_the_model_before_takes_the_factors_of_the_run_5_to_9_days_before(
    result: dict[str, Any],
) -> None:
    row = copy.deepcopy(_row(result))
    path = result["root"] / "campaign" / "ibm_line" / "2026-04-13" / "best-deep.counts.json"
    own = row["fits"]["two_factor"]

    def run(days: float, chain: str = "best", **fits: Any) -> dict[str, Any]:
        when = RUN_AT - timedelta(days=days)
        stamp = when.isoformat().replace("+00:00", "Z")
        return {"device": "ibm_line", "chain": chain, "run_at": stamp, "fits": fits}

    same = {k: own[k] for k in ("gates", "readout")}
    others = [run(4.9, **{"two_factor": same}), run(7, "median", **{"two_factor": same})]
    assert analyze.earlier_run(row, [*others, row]) is None
    week = run(7, two_factor=same)
    older = run(9, two_factor={"gates": {"factor": 5.0}, "readout": {"factor": 1.0}})
    assert analyze.earlier_run(row, [older, week, row]) is week

    analyze._add_before(row, [older, week, row], path)

    assert row["before"]["days"] == pytest.approx(7.0)
    assert row["before"]["factors"] == {k: own[k]["factor"] for k in ("gates", "readout")}
    with_own = _test(row, "two_factor_before", "two_factor", "test4")
    assert with_own["log"]["outcome"] == "identical"
    against = _test(row, "two_factor_before", "calibration", "test4")
    first = _test(row, "two_factor", "calibration", "test4")
    assert against["log"]["difference"] == pytest.approx(first["log"]["difference"], rel=1e-4)
    assert "p_value" not in row["scores"]["two_factor_before"]["test4"]

    row = copy.deepcopy(_row(result))
    analyze._add_before(row, [older, row], path)
    assert row["before"]["factors"] == {"gates": 5.0, "readout": 1.0}
    assert _test(row, "two_factor_before", "two_factor", "test4")["winner"] == "two_factor"
