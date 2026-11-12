from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

import noisevault as nv
from noisevault.cli import app
from noisevault.counts import MeasuredCounts, PlannedCircuit, load_counts, plan, simulate
from noisevault.errors import CountsError
from noisevault.profile import Profile
from noisevault.reference import Op, probabilities
from noisevault.score import LeftOut, _grouped, age_text, paired, probe_report, rank, score
from noisevault.suite import role, suite

KINGSTON = "ibm_kingston@2026-04-15"
RUN_AT = datetime(2026, 4, 16, 9, 30, tzinfo=UTC)
HOURS_3 = timedelta(hours=3)
slow = pytest.mark.skipif(
    os.environ.get("NOISEVAULT_SLOW") != "1", reason="set NOISEVAULT_SLOW=1 to run (2 minutes)"
)
runner = CliRunner()


def scaled(profile: Profile, gate: float | None = None, readout: float | None = None) -> Profile:
    factors = {k: {"factor": v} for k, v in (("gates", gate), ("readout", readout)) if v}
    return profile.model_copy(update={"unmodeled_error": factors})


def one_qubit(readout_error: float | None) -> Profile:
    return Profile.uniform(
        "toy",
        technology="superconducting",
        num_qubits=1,
        one_qubit_error=0.01,
        readout_error=readout_error,
    )


def toy_one_qubit() -> Profile:
    return Profile.uniform(
        "toy",
        technology="superconducting",
        num_qubits=1,
        one_qubit_error=0.01,
        readout_error=0.02,
    )


FLIPS = (PlannedCircuit(name="flips", qubits=(0,), ops=(Op("x", (0,)),) * 9),)


def written(profile: Profile, circuits: Sequence[tuple[str, dict[str, int]]]) -> MeasuredCounts:
    """Counts of circuits with no ops on qubit 0, bound to ``profile``."""
    return MeasuredCounts.model_validate(
        {
            "nv_counts": "1.0",
            "source": "simulated",
            "profile": {"id": profile.id, "fingerprint": profile.calibration_fingerprint},
            "backend": profile.device.name,
            "run_at": RUN_AT.isoformat(),
            "bit_order": "clbit0_left",
            "execution": {
                "client": "hand-written",
                "transpiled": False,
                "gate_twirling": False,
                "measure_twirling": False,
                "dynamical_decoupling": False,
                "init_qubits": True,
            },
            "circuits": [
                {
                    "name": name,
                    "qubits": [0],
                    "ops": [],
                    "shots": sum(counts.values()),
                    "counts": counts,
                }
                for name, counts in circuits
            ],
        }
    )


def kingston_counts(shots: int = 4000, seed: int = 1) -> MeasuredCounts:
    kingston = nv.load(KINGSTON)
    truth = scaled(kingston, 1.8, 1.3)
    return simulate(truth, plan(kingston), shots=shots, seed=seed, run_at=RUN_AT)


def test_scores_equal_the_formulas_on_one_qubit() -> None:
    profile = one_qubit(0.1)
    counts = written(profile, [("readout", {"0": 70, "1": 30}), ("again", {"0": 9, "1": 1})])
    result = score(profile, counts)
    p = np.array([0.9, 0.1])
    for c, (n0, n1) in zip(result.circuits, [(70, 30), (9, 1)], strict=True):
        f = np.array([n0, n1]) / (n0 + n1)
        assert c.probabilities == pytest.approx(p, abs=1e-15)
        assert c.log_score == pytest.approx(f @ np.log(p), abs=1e-12)
        assert c.quadratic_score == pytest.approx(2 * f @ p - p @ p, abs=1e-12)
        assert c.excess == pytest.approx(f @ np.log(f / p), abs=1e-12)
        assert c.ruled_out == 0
    assert result.shots == 110
    assert result.log_score == pytest.approx((79 * math.log(0.9) + 31 * math.log(0.1)) / 110)
    expected = (100 * result.circuits[0].excess + 10 * result.circuits[1].excess) / 110
    assert result.excess == pytest.approx(expected, abs=1e-12)
    assert result.same_calibration and result.calibration_age is None


def test_paired_equals_the_formulas_on_one_qubit() -> None:
    a, b = one_qubit(0.1), one_qubit(0.2)
    counts = written(a, [("readout", {"0": 70, "1": 30}), ("again", {"0": 9, "1": 1})])
    test = paired(score(a, counts), score(b, counts))
    pa, pb = np.array([0.9, 0.1]), np.array([0.8, 0.2])
    for rule, s in (
        (test.log, np.log),
        (test.quadratic, lambda p: 2 * p - p @ p),
    ):
        d = s(pa) - s(pb)
        total = variance = 0.0
        for n in (np.array([70, 30]), np.array([9, 1])):
            f = n / n.sum()
            total += n @ d
            variance += n.sum() * (f @ d**2 - (f @ d) ** 2)
        z = total / math.sqrt(variance)
        assert rule.outcome == "tested"
        assert rule.difference == pytest.approx(total / 110, abs=1e-12)
        assert rule.standard_error == pytest.approx(math.sqrt(variance) / 110, abs=1e-12)
        assert rule.z == pytest.approx(z, abs=1e-12)
        assert rule.p_value == pytest.approx(math.erfc(abs(z) / math.sqrt(2)), abs=1e-12)
    assert test.log.better == "b" and test.better == "b"


def test_the_true_model_scores_higher_than_a_wrong_one() -> None:
    kingston = nv.load(KINGSTON)
    counts = simulate(scaled(kingston, 1.8), plan(kingston), shots=200_000, seed=3, run_at=RUN_AT)
    truth, wrong = score(scaled(kingston, 1.8), counts), score(kingston, counts)
    assert truth.log_score > wrong.log_score
    assert truth.quadratic_score > wrong.quadratic_score
    assert truth.excess < wrong.excess
    assert truth.p_value > 0.05 > wrong.p_value
    assert paired(truth, wrong).better == "a"


def test_excess_of_the_true_model_is_shot_noise() -> None:
    counts = kingston_counts()
    truth = score(scaled(nv.load(KINGSTON), 1.8, 1.3), counts)
    assert 0 < truth.excess < truth.excess_noise_95
    assert 0 < truth.excess_noise_mean < truth.excess_noise_95
    assert truth.p_value > 0.05
    again = score(scaled(nv.load(KINGSTON), 1.8, 1.3), counts)
    assert again.p_value == truth.p_value and again.excess_noise_95 == truth.excess_noise_95


def test_a_profile_of_another_date_and_device_scores_the_counts() -> None:
    counts = kingston_counts()
    kingston = nv.load(KINGSTON)
    later = kingston.model_copy(
        update={"device": kingston.device.model_copy(update={"calibrated_at": RUN_AT + HOURS_3})}
    )
    result = score(later, counts)
    assert result.calibration_age == -HOURS_3 and age_text(result) == "-3 h"
    assert result.same_device
    fez = score(nv.load("ibm_fez"), counts)
    assert not fez.same_calibration and not fez.same_device and age_text(fez) == "other device"
    assert math.isfinite(fez.quadratic_score)


def test_a_wrong_reading_on_a_qubit_with_readout_error_0_rules_out_the_profile() -> None:
    exact, honest = one_qubit(0.0), one_qubit(0.05)
    counts = written(exact, [("readout", {"0": 95, "1": 5})])
    ruled = score(exact, counts)
    assert ruled.ruled_out == 5 and ruled.circuits[0].ruled_out == 5
    assert ruled.log_score == -math.inf and ruled.excess == math.inf and ruled.p_value == 0
    assert ruled.to_dict()["log_score"] is None and ruled.to_dict()["excess"] is None
    test = paired(ruled, score(honest, counts))
    assert test.log.outcome == "ruled_out" and test.log.z is None and test.log.better == "b"
    assert test.quadratic.outcome == "tested" and math.isfinite(test.quadratic.z or math.nan)
    assert test.better == "b"
    json.dumps(test.to_dict(), allow_nan=False)


def test_with_both_models_ruled_out_the_quadratic_rule_decides() -> None:
    counts = written(one_qubit(0.0), [("readout", {"0": 90, "1": 10})])
    zero = score(one_qubit(0.0), counts)
    none = score(one_qubit(None), counts)
    test = paired(zero, none)
    assert test.log.outcome == "both_ruled_out" and test.log.better == "neither"
    assert test.quadratic.outcome == "identical" and test.better == "neither"


def test_identical_models_give_no_difference() -> None:
    profile = one_qubit(0.1)
    counts = written(profile, [("readout", {"0": 70, "1": 30})])
    test = paired(score(profile, counts), score(profile, counts))
    assert test.log.outcome == test.quadratic.outcome == "identical"
    assert test.log.z is None and test.better == "neither"


def test_shots_with_one_outcome_still_test_models_that_expect_another_outcome() -> None:
    # 100 of 100 shots read 0. That is more likely with a readout error of 0.1 than of 0.2.
    counts = written(one_qubit(0.1), [("readout", {"0": 100})])
    test = paired(score(one_qubit(0.1), counts), score(one_qubit(0.2), counts))
    assert test.log.outcome == "tested"
    assert test.log.z is not None and test.log.z > 3
    assert test.better == "a"
    # The outcome 1 has no shot. Its part of the variance is (0 + 3) / 100 times its squared
    # deviation from the mean score difference.
    d0, d1 = math.log(0.9 / 0.8), math.log(0.1 / 0.2)
    error = math.sqrt(100 * 0.03 * (d1 - d0) ** 2)
    assert test.log.z == pytest.approx(100 * d0 / error, rel=1e-9)
    assert test.quadratic.outcome == "tested" and test.quadratic.difference != 0


def test_the_log_rule_gives_no_test_when_one_model_excludes_an_outcome_the_other_expects() -> None:
    counts = written(one_qubit(0.1), [("readout", {"0": 100})])
    test = paired(score(one_qubit(0.1), counts), score(one_qubit(0.0), counts))
    assert test.log.outcome == "unbounded" and test.log.z is None
    assert test.quadratic.outcome == "tested"
    assert test.better == test.quadratic.better


def two_qubits(error_0: float, error_1: float) -> Profile:
    """Two qubits with a symmetric readout error each."""
    data = Profile.uniform(
        "toy",
        technology="superconducting",
        num_qubits=2,
        one_qubit_error=0.01,
        two_qubit_error=0.02,
        readout_error=0.1,
    ).to_dict()
    data["qubits"] = [
        {"index": q, "readout": {"p1_given_0": e, "p0_given_1": e}}
        for q, e in enumerate((error_0, error_1))
    ]
    return Profile.model_validate(data)


def test_a_rare_outcome_that_few_runs_show_does_not_make_the_test_reject_too_often() -> None:
    # Both models have the same expected log score under these true probabilities. Model b
    # gives the rare outcome of qubit 0 almost no probability, and most runs do not show it.
    a, b = two_qubits(0.001, 0.1), two_qubits(1e-12, 0.10001)
    truth = (4.8276755196e-5, 0.100004999926)
    rng = np.random.default_rng(42)
    runs, rejected = 300, 0
    for _ in range(runs):
        ones = [int(rng.binomial(2000, p)) for p in truth]
        counts = written(a, [("q0", {"0": 2000 - ones[0], "1": ones[0]})]).model_copy(
            update={
                "circuits": [
                    {"name": f"q{q}", "qubits": [q], "ops": [], "shots": 2000, "counts": found}
                    for q, found in enumerate({"0": 2000 - k, "1": k} for k in ones)
                ]
            }
        )
        rejected += paired(score(a, counts), score(b, counts)).log.better != "neither"
    assert rejected <= 0.05 * runs


def test_an_outcome_that_both_models_call_almost_impossible_does_not_give_a_false_verdict() -> None:
    # Both models have the same expected log score under these true probabilities. Qubit 1
    # reads wrong in 1 of 20,000 shots, and model a gives that outcome almost no probability.
    a, b = two_qubits(2e-8, 1e-8 * math.exp(-700)), two_qubits(1e-8, 1e-8)
    truth = (0.05, 4.951051361071759e-5)
    rng = np.random.default_rng(42)
    runs, rejected = 300, 0
    for _ in range(runs):
        ones = [int(rng.binomial(2000, p)) for p in truth]
        counts = written(a, [("q0", {"0": 2000})]).model_copy(
            update={
                "circuits": [
                    {"name": f"q{q}", "qubits": [q], "ops": [], "shots": 2000, "counts": found}
                    for q, found in enumerate({"0": 2000 - k, "1": k} for k in ones)
                ]
            }
        )
        rejected += paired(score(a, counts), score(b, counts)).log.better != "neither"
    assert rejected <= 0.05 * runs


def test_a_model_far_from_shots_with_one_outcome_loses_the_test() -> None:
    # 10,000 of 10,000 shots read 0. A readout error of 0.1 is then very unlikely.
    near, far = one_qubit(1e-12), one_qubit(0.1)
    counts = written(near, [("readout", {"0": 10_000})])
    test = paired(score(near, counts), score(far, counts))
    assert test.log.outcome == "tested" and test.log.better == "a"
    assert test.log.z is not None and test.log.z > 10


def test_rare_outcomes_of_a_circuit_count_as_one_outcome_in_the_log_rule() -> None:
    measured = np.array([1990, 10, 0, 0])
    a = np.array([0.99, 0.0098, 1.5e-4, 5e-5])
    b = np.array([0.99, 0.0099, 2e-5, 8e-5])
    grouped = _grouped(measured, a, b)
    assert [list(part) for part in grouped] == [
        [1990, 10, 0],
        [0.99, 0.0098, pytest.approx(2e-4)],
        [0.99, 0.0099, pytest.approx(1e-4)],
    ]
    one_rare = _grouped(np.array([1990, 10, 0]), a[:3], b[:3])
    assert len(one_rare[0]) == 3


def test_a_very_small_probability_gives_a_finite_excess() -> None:
    profile = two_qubits(1e-320, 0.1)
    counts = written(profile, [("readout", {"0": 99, "1": 1})])
    forecast = score(profile, counts)
    assert forecast.ruled_out == 0
    assert forecast.excess == pytest.approx(7.3122708746, rel=1e-9)
    assert math.isfinite(forecast.excess_noise_95)


def test_split_refuses_a_circuit_with_more_shots_than_the_sampler_divides() -> None:
    counts = written(one_qubit(0.1), [("readout", {"0": 900_000_000, "1": 100_000_000})])
    with pytest.raises(CountsError, match="split divides at most 999999999 shots"):
        counts.split(0.5, seed=0)


def test_paired_refuses_forecasts_of_different_counts_or_circuits() -> None:
    profile = one_qubit(0.1)
    first = written(profile, [("readout", {"0": 70, "1": 30}), ("again", {"0": 9, "1": 1})])
    second = written(profile, [("readout", {"0": 71, "1": 29}), ("again", {"0": 9, "1": 1})])
    with pytest.raises(CountsError, match="different counts"):
        paired(score(profile, first), score(profile, second))
    with pytest.raises(CountsError, match="different circuits"):
        paired(score(profile, first), score(profile, first, circuits=["again"]))


def test_score_of_named_circuits_only() -> None:
    profile = one_qubit(0.1)
    counts = written(profile, [("readout", {"0": 70, "1": 30}), ("again", {"0": 9, "1": 1})])
    result = profile.score(counts, circuits=["again"])
    assert [c.name for c in result.circuits] == ["again"] and result.shots == 10
    with pytest.raises(CountsError, match="no circuit 'agian'"):
        score(profile, counts, circuits=["agian"])


def test_a_profile_that_cannot_simulate_a_circuit_is_refused() -> None:
    counts = kingston_counts()
    with pytest.raises(CountsError, match="circuit ghz_chain") as info:
        score(one_qubit(0.1), counts)
    # The error keeps its first cause, so a caller can give both classes.
    assert type(info.value.__cause__).__name__ == "LayoutError"


def test_select_keeps_the_named_circuits_in_run_order() -> None:
    counts = kingston_counts()
    chosen = counts.select(["readout", "ghz_chain"])
    assert [c.name for c in chosen.circuits] == ["ghz_chain", "readout"]
    assert chosen.circuits[0] == counts.circuits[0]
    assert chosen.profile == counts.profile and chosen.run_at == counts.run_at
    assert chosen.sha256 != counts.sha256
    with pytest.raises(CountsError, match="no circuit"):
        counts.select(["nothing"])


def test_split_divides_each_circuit_and_the_parts_sum_to_the_counts() -> None:
    counts = kingston_counts()
    first, second = counts.split(0.3, seed=5)
    for whole, x, y in zip(counts.circuits, first.circuits, second.circuits, strict=True):
        assert x.shots == round(0.3 * whole.shots) and x.shots + y.shots == whole.shots
        assert (x.vector() + y.vector() == whole.vector()).all()
        assert (x.name, x.qubits, x.ops) == (whole.name, whole.qubits, whole.ops)
    assert counts.split(0.3, seed=5) == (first, second)
    assert counts.split(0.3, seed=6)[0].sha256 != first.sha256
    assert len({counts.sha256, first.sha256, second.sha256}) == 3


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.5, 1.5, math.nan, 0.0001, 0.9999])
def test_split_refuses_a_part_with_no_shots(fraction: float) -> None:
    with pytest.raises(CountsError):
        kingston_counts().split(fraction, seed=1)


def test_parts_have_the_sha256_of_their_saved_file_and_still_bind(tmp_path: Path) -> None:
    profile = toy_one_qubit()
    counts = simulate(profile, (*FLIPS, *plan(profile)), shots=2000, seed=2, run_at=RUN_AT)
    first, _ = counts.split(0.5, seed=1)
    chosen = counts.select(["flips"])
    for part in (first, chosen):
        loaded = load_counts(part.save(tmp_path / "part.counts.json"))
        assert loaded.sha256 == part.sha256 and loaded == part
    assert profile.compare(chosen).counts is chosen
    assert profile.compare(first).counts is first


def test_held_out_shots_test_a_model_fitted_on_the_other_shots() -> None:
    profile = toy_one_qubit()
    circuits = (*FLIPS, *plan(profile))
    truth = scaled(profile, 3.0, 2.0)
    counts = simulate(truth, circuits, shots=20_000, seed=4, run_at=RUN_AT)
    train, held_out = counts.split(0.5, seed=1)
    fitted = profile.compare(train).fitted_profile()
    result = paired(score(fitted, held_out), score(profile, held_out))
    assert result.better == "a"


def _probs(profile: Profile, circuits: Sequence[PlannedCircuit]) -> list[np.ndarray]:
    return [
        probabilities(profile, c.ops, len(c.qubits), layout=c.qubits, unknown_gates="error")
        for c in circuits
    ]


@slow
def test_the_log_rule_test_rejects_at_its_level_and_finds_a_wrong_model() -> None:
    toy = Profile.uniform(
        "toy",
        technology="superconducting",
        num_qubits=2,
        one_qubit_error=0.02,
        two_qubit_error=0.05,
        readout_error=0.02,
    )
    circuits = (
        PlannedCircuit(name="bell", qubits=(0, 1), ops=(Op("h", (0,)), Op("cx", (0, 1)))),
        PlannedCircuit(
            name="flip",
            qubits=(0, 1),
            ops=(Op("x", (0,)), Op("x", (1,)), Op("cx", (0, 1)), Op("cx", (0, 1))),
        ),
    )
    truth = scaled(toy, 1.5)
    exact = _probs(truth, circuits)

    def divergence(factor: float) -> float:
        model = _probs(scaled(toy, factor), circuits)
        return sum(float(t @ np.log(t / m)) for t, m in zip(exact, model, strict=True))

    below, low, high = 1.1, 1.5, 20.0
    target = divergence(below)
    while high - low > 1e-9:
        middle = (low + high) / 2
        low, high = (middle, high) if divergence(middle) < target else (low, middle)
    above = (low + high) / 2
    assert divergence(above) == pytest.approx(target, rel=1e-6)

    models = {"truth": truth, "below": scaled(toy, below), "above": scaled(toy, above)}
    models["plain"] = toy
    sets, size_rejects, power_rejects = 2000, 0, 0
    for seed in range(sets):
        counts = simulate(truth, circuits, shots=4000, seed=seed, run_at=RUN_AT)
        scored = {name: score(model, counts) for name, model in models.items()}
        size_rejects += paired(scored["below"], scored["above"]).log.better != "neither"
        power_rejects += paired(scored["truth"], scored["plain"]).log.better == "a"
    size, power = size_rejects / sets, power_rejects / sets
    print(f"type-I error {size:.4f}, power {power:.4f}, factors {below} and {above:.4f}")
    assert 0.03 <= size <= 0.07
    assert power >= 0.95


def test_nv_score_prints_one_row_per_model_best_first(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    kingston = nv.load(KINGSTON)
    truth = scaled(kingston, 1.8, 1.3).save("truth.json")
    counts = kingston_counts().save("run.counts.json")
    args = ["score", KINGSTON, "ibm_fez", str(truth), str(counts)]
    result = runner.invoke(app, args, env={"COLUMNS": "120"})
    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert lines[0] == f"counts {counts}, simulated, sha256:{load_counts(counts).sha256[7:19]}"
    assert lines[1] == "run 2026-04-16 09:30Z on ibm_kingston, qubits 148-149-150-151"
    head, *rows = lines[3:7]
    assert (
        head.split()
        == "model fingerprint age log score excess noise 95% quadratic ruled out".split()
    )
    assert [row.split()[0] for row in rows] == [str(truth), KINGSTON, "ibm_fez"]
    assert rows[0].split()[2:4] == ["26", "h"] and "other device" in rows[2]
    assert rows[2].split()[-1] != "0" and rows[2].split()[-5:-3] == ["-inf", "inf"]
    assert f"{KINGSTON} against {truth}:" in lines
    assert (
        f"  log score: ibm_fez is ruled out by {rows[2].split()[-1]} shots, so {truth} is better"
        in lines
    )
    narrow = runner.invoke(app, args, env={"COLUMNS": "80"})
    assert max(len(line) for line in narrow.stdout.splitlines()) <= 80


def test_nv_score_json_holds_every_number(tmp_path: Path) -> None:
    counts = kingston_counts().save(tmp_path / "run.counts.json")
    result = runner.invoke(app, ["score", "ibm_fez", KINGSTON, str(counts), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert [m["ref"] for m in data["models"]] == [KINGSTON, "ibm_fez"]
    best = data["models"][0]
    assert best["same_calibration"] and best["draws"] == 200 and len(best["circuits"]) == 4
    assert data["models"][1]["log_score"] is None and data["models"][1]["ruled_out"] > 0
    (test,) = data["paired"]
    assert test["best"] == KINGSTON and test["other"] == "ibm_fez"
    assert test["log"]["outcome"] == "ruled_out" and test["better"] == "a"
    assert data["counts"] == load_counts(counts).sha256


def _with_a_stated_zero(tmp_path: Path) -> tuple[Path, Path, int]:
    """The Kingston profile with P(1|0) of exactly 0 on the first qubit of its chain, and counts
    of a device that reads that qubit wrong."""
    kingston = nv.load(KINGSTON)
    chain = plan(kingston)[0].qubits
    data = kingston.to_dict()
    readout = data["qubits"][chain[0]]["readout"]
    readout.pop("error", None)
    readout.update({"p1_given_0": 0.0, "p0_given_1": 0.01})
    stated = Profile.model_validate(data)
    truth = scaled(stated.floored(), readout=30.0)
    counts = simulate(truth, plan(stated, chain), shots=4000, seed=3, run_at=RUN_AT)
    return (
        stated.save(tmp_path / "stated.json"),
        counts.save(tmp_path / "run.counts.json"),
        chain[0],
    )


def test_nv_score_names_a_stated_zero_and_floor_replaces_it(tmp_path: Path) -> None:
    stated, counts, qubit = _with_a_stated_zero(tmp_path)

    plain = runner.invoke(app, ["score", str(stated), str(counts), "--json"])
    floored = runner.invoke(app, ["score", str(stated), str(counts), "--floor", "--json"])
    text = runner.invoke(app, ["score", str(stated), str(counts)], env={"COLUMNS": "200"})

    assert plain.exit_code == floored.exit_code == text.exit_code == 0, plain.output
    first, second = json.loads(plain.stdout), json.loads(floored.stdout)
    assert first["floor"] is False and second["floor"] is True
    assert first["zero_readout_qubits"] == {str(stated): [qubit]}
    assert first["models"][0]["ruled_out"] > 0 and first["models"][0]["log_score"] is None
    assert second["models"][0]["ruled_out"] == 0 and second["models"][0]["log_score"] < 0
    assert f"hint: a readout error of exactly 0 is stated by {stated} on 1 qubit {qubit}" in (
        " ".join(text.stderr.split())
    )
    assert "--floor" in text.stderr


def test_nv_score_gives_no_hint_for_a_zero_on_a_qubit_that_the_counts_do_not_measure(
    tmp_path: Path,
) -> None:
    kingston = nv.load(KINGSTON)
    measured = set(plan(kingston)[0].qubits)
    other = next(q for q in range(kingston.device.num_qubits) if q not in measured)
    data = kingston.to_dict()
    data["qubits"][other]["readout"] = {"error": 0.0}
    profile = Profile.model_validate(data).save(tmp_path / "zero.json")
    counts = kingston_counts().save(tmp_path / "run.counts.json")

    result = runner.invoke(app, ["score", str(profile), str(counts)])

    assert result.exit_code == 0, result.output
    assert "hint" not in result.stderr


def test_nv_score_wants_the_counts_file_last(tmp_path: Path) -> None:
    counts = kingston_counts().save(tmp_path / "run.counts.json")
    result = runner.invoke(app, ["score", str(counts), KINGSTON], env={"COLUMNS": "80"})
    assert result.exit_code == 1
    assert "nv score takes the profiles first and the counts file last" in result.stderr


def test_rank_puts_ruled_out_models_last_by_quadratic_score() -> None:
    counts = written(one_qubit(0.0), [("readout", {"0": 90, "1": 10})])
    named = [
        ("none", score(one_qubit(None), counts)),
        ("far", score(one_qubit(0.4), counts)),
        ("near", score(one_qubit(0.1), counts)),
        ("zero", score(one_qubit(0.0), counts)),
    ]
    ranking = rank(named)
    assert [name for name, _ in ranking.models][:2] == ["near", "far"]
    assert {name for name, _ in ranking.models[2:]} == {"none", "zero"}
    assert len(ranking.tests) == 3


def test_the_scoring_page_shows_the_output_of_a_current_run(tmp_path: Path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    page = (root / "docs" / "scoring.md").read_text(encoding="utf-8")
    shown = page.split("```text\n$ nv score ", 1)[1].split("\n```", 1)[0]
    command, expected = shown.split("\n", 1)
    shutil.copy(root / "examples" / "kingston-simulated.counts.json", tmp_path)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["score", *command.split()], env={"COLUMNS": "80"})
    assert result.exit_code == 0, result.output
    assert result.stdout.rstrip("\n") == expected, result.stdout


def test_select_gives_the_forecast_of_the_named_circuits_with_no_new_simulation(
    monkeypatch,
) -> None:
    kingston = nv.load(KINGSTON)
    counts = kingston_counts(shots=500)
    whole = score(kingston, counts)
    monkeypatch.setattr("noisevault.score.probabilities", lambda *a, **k: pytest.fail("simulated"))

    part = whole.select(["readout", "mirror"])

    monkeypatch.undo()
    assert part == score(kingston, counts, circuits=["mirror", "readout"])
    assert [c.name for c in part.circuits] == ["mirror", "readout"]
    with pytest.raises(CountsError, match="no circuit 'missing'"):
        whole.select(["missing"])
    with pytest.raises(CountsError, match="names no circuit"):
        whole.select([])


def two_qubit_device() -> Profile:
    return Profile.uniform(
        "toy",
        technology="superconducting",
        num_qubits=2,
        one_qubit_error=0.01,
        two_qubit_error=0.02,
        readout_error=0.03,
    )


def run_of(
    profile: Profile, circuits: Sequence[tuple[str, int, Sequence[Op], dict[str, int]]]
) -> MeasuredCounts:
    """Counts of circuits made by hand: name, number of qubits, ops and counts."""
    data = written(profile, [("readout", {"0": 1})]).to_dict()
    data["circuits"] = [
        {
            "name": name,
            "qubits": list(range(n)),
            "ops": [[op.name, list(op.qubits), list(op.params)] for op in ops],
            "shots": sum(counts.values()),
            "counts": counts,
        }
        for name, n, ops, counts in circuits
    ]
    return MeasuredCounts.model_validate(data)


def x_times(repeats: int) -> list[Op]:
    return [Op("x", (q,)) for _ in range(repeats) for q in (0, 1)]


READOUT = ("readout", 2, (), {"00": 90, "01": 6, "10": 4})
AXIS_D4 = ("axis_d4", 2, x_times(4), {"00": 80, "01": 10, "10": 6, "11": 4})
AXIS_D8 = ("axis_d8", 2, x_times(8), {"00": 50, "01": 20, "10": 22, "11": 8})
WIDE = ("clifford_n12_d1", 12, (), {"0" * 12: 10})


def test_score_leaves_out_probe_and_wide_circuits_and_names_them() -> None:
    profile = two_qubit_device()
    counts = run_of(profile, [READOUT, AXIS_D4, WIDE])

    result = score(profile, counts)

    assert [c.name for c in result.circuits] == ["readout"]
    assert result.left_out == (LeftOut("axis_d4", "probe"), LeftOut("clifford_n12_d1", "too_wide"))
    assert result.to_dict()["left_out"] == [
        {"name": "axis_d4", "reason": "probe"},
        {"name": "clifford_n12_d1", "reason": "too_wide"},
    ]
    assert result.summary().splitlines()[-2:] == [
        "1 probe circuit left out: axis_d4",
        "1 circuit on more than 10 qubits left out: clifford_n12_d1",
    ]
    assert paired(result, score(scaled(profile, readout=2.0), counts)).log.outcome == "tested"


def test_named_circuits_keep_a_probe_circuit_and_refuse_a_wide_one() -> None:
    profile = two_qubit_device()
    counts = run_of(profile, [READOUT, AXIS_D4, WIDE])

    named = score(profile, counts, circuits=["readout", "axis_d4"])

    assert [c.name for c in named.circuits] == ["readout", "axis_d4"] and named.left_out == ()
    with pytest.raises(CountsError, match="circuit clifford_n12_d1 has 12 qubits"):
        score(profile, counts, circuits=["readout", "clifford_n12_d1"])
    with pytest.raises(CountsError, match="different circuits"):
        paired(score(profile, counts), named)
    with pytest.raises(CountsError, match="only probe circuits") as info:
        score(profile, counts.select(["axis_d4", "clifford_n12_d1"]))
    assert info.value.hint == "name the circuits to score with circuits="


def test_a_long_list_of_left_out_circuits_is_cut() -> None:
    profile = two_qubit_device()
    wide = [(f"w{i}", 12, (), {"0" * 12: 1}) for i in range(6)]
    result = score(profile, run_of(profile, [READOUT, *wide]))
    assert result.summary().splitlines()[-1] == (
        "6 circuits on more than 10 qubits left out: w0, w1, w2, w3 and 2 more"
    )


def wrong_forecast(profile: Profile, ops: Sequence[Op], ideal: str) -> list[float]:
    """The forecast fraction of wrong bits of each of 2 qubits, from the outcome probabilities."""
    probs = probabilities(profile, list(ops), 2, layout=(0, 1), readout=True)
    keys = ["00", "01", "10", "11"]
    return [
        sum(p for key, p in zip(keys, probs, strict=True) if key[i] != ideal[i]) for i in range(2)
    ]


def test_the_probe_report_equals_a_hand_computation() -> None:
    profile = two_qubit_device()
    counts = run_of(profile, [READOUT, AXIS_D4, AXIS_D8])

    report = probe_report(profile, counts)

    assert report is not None and report == profile.probe_report(counts)
    d4, d8 = report.circuits
    assert (d4.name, d4.qubits, d4.shots, d4.ideal_result) == ("axis_d4", (0, 1), 100, "00")
    assert d4.measured == pytest.approx(0.80) and d8.measured == pytest.approx(0.50)
    probs = probabilities(profile, x_times(4), 2, layout=(0, 1), readout=True)
    assert d4.forecast == pytest.approx(probs[0])
    measured = {"axis_d4": [0.10, 0.14], "axis_d8": [0.30, 0.28]}
    for c, ops in ((d4, x_times(4)), (d8, x_times(8))):
        assert [q.qubit for q in c.wrong_bits] == [0, 1]
        assert [q.ideal_bit for q in c.wrong_bits] == ["0", "0"]
        assert [q.measured for q in c.wrong_bits] == pytest.approx(measured[c.name])
        forecast = wrong_forecast(profile, ops, "00")
        assert [q.forecast for q in c.wrong_bits] == pytest.approx(forecast)
        assert [q.baseline for q in c.wrong_bits] == pytest.approx([0.04, 0.06])
    growth = report.growth
    assert growth is not None
    assert (growth.fewer, growth.more, growth.gates) == ("axis_d4", "axis_d8", 2)
    assert growth.qubits == (0, 1)
    assert growth.ratios == pytest.approx([0.26 / 0.06, 0.22 / 0.08])
    assert report.left_out == ()
    data = report.to_dict()
    assert data["growth"]["ratios"][0] == {"qubit": 0, "ratio": pytest.approx(0.26 / 0.06)}
    assert data["circuits"][1]["wrong_bits"][1]["measured"] == pytest.approx(0.28)
    text = " ".join(report.summary("toy").split())
    assert "a systematic rotation error gives about 4, and an error with no direction" in text
    narrow = report.summary("toy", width=46).splitlines()
    table = narrow[narrow.index("wrong bits, measured / forecast") :]
    assert max(map(len, table)) <= 46
    assert [line.split()[2:] for line in narrow if line.startswith("qubit")] == [
        ["axis_d4"],
        ["axis_d8", "growth"],
    ]


def test_growth_has_no_ratio_where_the_fewer_gates_give_no_wrong_bits_above_baseline() -> None:
    profile = two_qubit_device()
    at_baseline = ("axis_d4", 2, x_times(4), {"00": 90, "01": 6, "10": 4})
    report = probe_report(profile, run_of(profile, [READOUT, at_baseline, AXIS_D8]))
    assert report is not None and report.growth is not None
    assert report.growth.ratios == (None, None)
    rows = [line.split() for line in report.summary().splitlines() if line[:2] in ("0 ", "1 ")]
    assert [row[-1] for row in rows] == ["-", "-"]
    no_readout = probe_report(profile, run_of(profile, [AXIS_D4, AXIS_D8]))
    assert no_readout is not None and no_readout.growth is not None
    assert [q.baseline for q in no_readout.circuits[0].wrong_bits] == [None, None]
    assert no_readout.growth.ratios == pytest.approx([0.30 / 0.10, 0.28 / 0.14])


def test_a_probe_circuit_whose_ideal_result_is_not_all_0() -> None:
    profile = two_qubit_device()
    ops = [Op("x", (0,)), Op("sx", (1,)), Op("sx", (1,))]
    periodic = ("periodic_d2", 2, ops, {"11": 90, "01": 5, "10": 3, "00": 2})
    uncertain = ("periodic_d3", 2, [Op("h", (0,))], {"00": 5, "10": 5})

    report = probe_report(profile, run_of(profile, [READOUT, periodic, uncertain, WIDE]))

    assert report is not None and report.growth is None
    (c,) = report.circuits
    assert c.ideal_result == "11" and c.measured == pytest.approx(0.90)
    assert [q.ideal_bit for q in c.wrong_bits] == ["1", "1"]
    assert [q.measured for q in c.wrong_bits] == pytest.approx([0.07, 0.05])
    assert [q.forecast for q in c.wrong_bits] == pytest.approx(wrong_forecast(profile, ops, "11"))
    assert report.left_out == (LeftOut("periodic_d3", "no_ideal_result"),)
    assert "1 circuit with no certain result left out: periodic_d3" in report.summary()


def test_counts_with_no_probe_circuit_that_the_simulator_runs_give_no_probe_report() -> None:
    profile = two_qubit_device()
    wide_probe = ("axis_d64", 12, (), {"0" * 12: 10})
    assert probe_report(profile, run_of(profile, [READOUT, wide_probe])) is None
    assert probe_report(profile, run_of(profile, [READOUT])) is None


def toy_suite_run(tmp_path: Path) -> tuple[Path, Path]:
    """A profile of a small device and simulated counts of its deep circuit set."""
    device = Profile.uniform(
        "toy",
        technology="superconducting",
        num_qubits=3,
        one_qubit_error=0.002,
        two_qubit_error=0.02,
        readout_error=0.02,
        one_qubit_ns=30,
        two_qubit_ns=60,
        connectivity=[(0, 1), (1, 2)],
    )
    counts = simulate(scaled(device, 1.5), suite(device), shots=1000, seed=1, run_at=RUN_AT)
    return device.save(tmp_path / "toy.json"), counts.save(tmp_path / "run.counts.json")


def test_nv_score_of_a_deep_circuit_set_prints_the_probe_report(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    profile, counts = (path.name for path in toy_suite_run(tmp_path))

    shown = {
        columns: runner.invoke(app, ["score", profile, counts], env={"COLUMNS": columns})
        for columns in ("100", "80")
    }

    for columns, result in shown.items():
        assert result.exit_code == 0, result.output
        lines = result.stdout.splitlines()
        assert max(len(line) for line in lines) <= int(columns), result.stdout
        text = " ".join(result.stdout.split())
        assert "3 probe circuits left out: periodic_d8, axis_d64, axis_d256" in text
        assert f"probe circuits, not in the score; forecast by {profile}" in text
        assert any(line.split()[:3] == ["axis_d64", "1000", "000"] for line in lines)
        assert "a systematic rotation error gives about 16" in text


def test_nv_score_with_a_wide_circuit_scores_the_others(tmp_path: Path) -> None:
    profile = two_qubit_device()
    path = run_of(profile, [READOUT, AXIS_D4, AXIS_D8, WIDE]).save(tmp_path / "run.counts.json")
    saved = profile.save(tmp_path / "toy.json")

    result = runner.invoke(app, ["score", str(saved), str(path)], env={"COLUMNS": "80"})

    assert result.exit_code == 0, result.output
    assert "run 2026-04-16 09:30Z on toy, qubits 0-1" in result.stdout.splitlines()
    assert "1 circuit on more than 10 qubits left out: clifford_n12_d1" in result.stdout
    assert "growth" in result.stdout


def test_nv_score_json_holds_the_probe_report_of_the_best_profile(tmp_path: Path) -> None:
    profile, counts = toy_suite_run(tmp_path)
    worse = scaled(Profile.load(profile), 4.0).save(tmp_path / "worse.json")

    result = runner.invoke(app, ["score", str(worse), str(profile), str(counts), "--json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    best = data["models"][0]
    assert data["probes"]["ref"] == best["ref"] == str(profile)
    assert data["probes"]["fingerprint"] == best["fingerprint"]
    names = [c["name"] for c in data["probes"]["circuits"]]
    assert names == ["periodic_d8", "axis_d64", "axis_d256"]
    assert data["probes"]["growth"]["gates"] == 4
    assert {item["reason"] for item in best["left_out"]} == {"probe"}


def test_nv_score_prints_no_probe_section_for_counts_with_no_probe_circuit(tmp_path: Path) -> None:
    counts = kingston_counts().save(tmp_path / "run.counts.json")

    text = runner.invoke(app, ["score", KINGSTON, str(counts)], env={"COLUMNS": "100"})
    data = runner.invoke(app, ["score", KINGSTON, str(counts), "--json"])

    assert text.exit_code == data.exit_code == 0, text.output
    assert "probe circuits" not in text.stdout and "left out" not in text.stdout
    assert json.loads(data.stdout)["probes"] is None


def test_only_a_complete_probe_name_of_the_suite_is_a_probe() -> None:
    profile = two_qubit_device()
    counts = run_of(
        profile,
        [
            READOUT,
            ("axis_custom", 2, x_times(4), {"00": 80, "01": 10, "10": 6, "11": 4}),
            ("periodic_test", 2, x_times(2), {"00": 90, "01": 6, "10": 4}),
            AXIS_D4,
        ],
    )

    result = score(profile, counts)

    assert [c.name for c in result.circuits] == ["readout", "axis_custom", "periodic_test"]
    assert result.left_out == (LeftOut("axis_d4", "probe"),)
    assert role("axis_d64") == role("periodic_d48") == "probe"
    assert role("axis_custom") == role("axis_d") == role("periodic_d8x") == "test_far"


def test_nv_score_prints_the_ranking_when_the_profile_cannot_simulate_a_probe(
    tmp_path: Path,
) -> None:
    kingston = nv.load(KINGSTON)
    not_native = [Op("h", (0,)), Op("h", (0,))]
    probe = ("axis_d4", 2, not_native, {"00": 80, "01": 10, "10": 6, "11": 4})
    counts = run_of(kingston, [READOUT, probe]).save(tmp_path / "run.counts.json")

    text = runner.invoke(app, ["score", KINGSTON, str(counts)], env={"COLUMNS": "200"})
    data = runner.invoke(app, ["score", KINGSTON, str(counts), "--json"])

    assert text.exit_code == data.exit_code == 0, text.output
    assert "1 probe circuit left out: axis_d4" in text.stdout
    assert "probe circuits, not in the score" not in text.stdout
    warning = "warning: no probe report: circuit axis_d4: h on qubit 0: h is not defined"
    assert warning in text.stderr and warning in data.stderr
    found = json.loads(data.stdout)
    assert found["probes"] is None and len(found["models"]) == 1


def test_nv_score_says_when_a_profile_has_no_value_for_a_floor(tmp_path: Path) -> None:
    exact = Profile.uniform(
        "exact",
        technology="superconducting",
        num_qubits=2,
        one_qubit_error=0.01,
        two_qubit_error=0.02,
        readout_error=0.0,
    )
    path = exact.save(tmp_path / "exact.json")
    counts = run_of(exact, [("readout", 2, (), {"00": 100})]).save(tmp_path / "run.counts.json")

    assert exact.floored() is exact
    for extra in ([], ["--floor"]):
        result = runner.invoke(app, ["score", str(path), str(counts), *extra])
        assert result.exit_code == 0, result.output
        hint = " ".join(result.stderr.split())
        assert f"hint: {path} states no readout error above 0." in hint
        assert "--floor has no value to replace a 0" in hint
        assert "Add --floor" not in hint


def test_nv_score_floor_gives_the_probe_forecast_of_the_floored_profile(tmp_path: Path) -> None:
    data = nv.load(KINGSTON).to_dict()
    data["qubits"][0]["readout"] = {"error": 0.0}
    stated = Profile.model_validate(data)
    saved = stated.save(tmp_path / "stated.json")
    path = run_of(stated, [READOUT, AXIS_D4, AXIS_D8]).save(tmp_path / "run.counts.json")

    plain = runner.invoke(app, ["score", str(saved), str(path), "--json"])
    floored = runner.invoke(app, ["score", str(saved), str(path), "--json", "--floor"])

    assert plain.exit_code == floored.exit_code == 0, plain.output
    first, second = json.loads(plain.stdout)["probes"], json.loads(floored.stdout)["probes"]
    assert first["fingerprint"] == stated.fingerprint
    assert second["fingerprint"] == stated.floored().fingerprint
    expected = probe_report(stated.floored(), load_counts(path))
    assert expected is not None
    wrong = expected.circuits[0].wrong_bits[0].forecast
    assert second["circuits"][0]["wrong_bits"][0]["forecast"] == pytest.approx(wrong)
    assert first["circuits"][0]["wrong_bits"][0]["forecast"] != pytest.approx(wrong)
