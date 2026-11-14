from __future__ import annotations

import json
import math
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from conftest import require

stim = require("stim")

from noisevault import events  # noqa: E402
from noisevault.counts import MeasuredCircuit, PlannedCircuit  # noqa: E402
from noisevault.errors import (  # noqa: E402
    CountsError,
    MissingCalibrationError,
    NoiseApproximationWarning,
    NoiseVaultError,
)
from noisevault.profile import Profile  # noqa: E402
from noisevault.reference import Op  # noqa: E402
from study import wide as study  # noqa: E402

RUN_AT = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)
WIDTH = 12


def _factor(profile: Profile, factor: float) -> Profile:
    return profile.model_copy(update={"unmodeled_error": {"gates": {"factor": factor}}})


@pytest.fixture(scope="module")
def line_9() -> Profile:
    return Profile.uniform(
        "line_9",
        technology="superconducting",
        num_qubits=9,
        connectivity=[(q, q + 1) for q in range(8)],
        one_qubit_error=1e-3,
        two_qubit_error=1e-2,
        readout_error=2e-2,
        t1_us=200.0,
        t2_us=150.0,
        one_qubit_ns=32.0,
        two_qubit_ns=68.0,
    )


@pytest.fixture(scope="module")
def code() -> stim.Circuit:
    return stim.Circuit.generated("repetition_code:memory", distance=5, rounds=5)


def _detections(profile: Profile, circuit: stim.Circuit, shots: int, seed: int) -> np.ndarray:
    return profile.to_stim(circuit).compile_detector_sampler(seed=seed).sample(shots)


# the move of the study code ------------------------------------------------------------------


def _draws(n: int, rate: float, seed: int) -> np.ndarray:
    r = np.random.default_rng(seed)
    common = r.random(n) < 0.05
    return (r.random((n, WIDTH)) < rate) | (common[:, None] & (r.random((n, WIDTH)) < 0.5))


def _counts(bits: np.ndarray) -> dict[str, int]:
    rows, times = np.unique(bits.astype(np.uint8), axis=0, return_counts=True)
    return {"".join(map(str, row)): int(k) for row, k in zip(rows, times, strict=True)}


def test_the_package_gives_the_numbers_of_the_study_code_on_a_fixed_case() -> None:
    # The values come from study/wide.py before its method moved into noisevault.events.
    circuit = MeasuredCircuit(
        name="clifford_fixed",
        qubits=tuple(range(WIDTH)),
        ops=(),
        shots=3000,
        counts=_counts(_draws(3000, 0.04, 1)),
    )
    bits_a, bits_b = _draws(20_000, 0.04, 2), _draws(20_000, 0.025, 3)
    a = events.accumulate([bits_a[:7000], bits_a[7000:]], WIDTH)
    b = events.accumulate([bits_b], WIDTH)
    measured = events.circuit_events(circuit, "0" * WIDTH)

    result = events.compare([measured], [a], [b])
    old = study.compare(
        [circuit],
        [study.Forecast(circuit.name, "0" * WIDTH, a)],
        [study.Forecast(circuit.name, "0" * WIDTH, b)],
    )

    exact = {"rel": 1e-12, "abs": 0.0}
    assert a.none == pytest.approx(0.5814959252037398, **exact)
    assert float(a.rate.sum()) == pytest.approx(0.7613619319034047, **exact)
    assert float(a.covariance.sum()) == pytest.approx(2.1202267900000002, **exact)
    assert vars(result.a) == pytest.approx(
        {
            "none_log": -0.6804062355064037,
            "none_quadratic": 0.5126868631819647,
            "count_log": -1.1537574240803292,
            "count_quadratic": 0.42387116682766995,
            "bit_log": -2.9357870428458384,
        },
        **exact,
    )
    assert result.b.bit_log == pytest.approx(-2.96578669315047, **exact)
    assert result.count_log.z == pytest.approx(7.406009749115345, **exact)
    assert result.count_log.better == "neither"
    assert result.bit_log.z == pytest.approx(3.6961267110943288, **exact)
    assert result.bit_log.p_value == pytest.approx(0.00021891374412477934, **exact)
    assert result.bit_log.better == "a"
    assert result.bit_sampling.variance == pytest.approx(123.47005455662966, **exact)
    assert result.bit_sampling.shot_variance == pytest.approx(469.42959195276603, **exact)
    assert result.count_sampling.variance == pytest.approx(53.6092427231064, **exact)
    assert result.count_sampling.reason == "few_samples"
    assert events.scores([measured], [b]).count_log == pytest.approx(-1.2016901712511143, **exact)
    assert not events.few_rate_samples(a)
    assert events.few_count_samples(b, measured, a, b)
    assert old.qubit_log == result.bit_log and old.wrong_log == result.count_log
    assert old.a.qubit_log == result.a.bit_log and old.b.success_log == result.b.none_log


# detection events ----------------------------------------------------------------------------


def test_the_paired_test_prefers_the_true_profile_on_detection_events(
    line_9: Profile, code: stim.Circuit
) -> None:
    truth = _factor(line_9, 2.0)
    measured = _detections(truth, code, 20_000, seed=1)

    report = events.score_detectors({"calibration": line_9, "device": truth}, code, measured)

    assert [m.name for m in report.models] == ["device", "calibration"]
    (test,) = report.tests
    assert test.better == "device"
    assert test.bit.p_value < 1e-6 and test.bit_sampling.reason is None
    # The calibration forecast has few samples in the cells of many events, so the count score
    # gives only a direction.
    assert test.count.z > 0
    device, calibration = report.models
    assert calibration.events < 0.8 * report.events < device.events
    assert [b.name for b in report.bits] == [f"D{j}" for j in range(code.num_detectors)]
    assert report.bits[0].coords == (1.0, 0.0)
    text = report.summary()
    assert "calibration: too few events, 0.7" in text
    # The true model has no proved difference in the number of events, and no bit whose
    # difference is more than chance over 24 bits: its line has no line of bits after it.
    assert re.search(
        r"device: [01]\.\d\d times the measured number of events, no proved difference\n"
        r"calibration: too few events",
        text,
    )
    assert "device is better by" in text


def test_the_report_gives_each_detector_and_each_test_to_programs(
    line_9: Profile, code: stim.Circuit
) -> None:
    measured = _detections(_factor(line_9, 2.0), code, 2000, seed=2)

    found = events.score_detectors(
        {"calibration": line_9, "device": _factor(line_9, 2.0)}, code, measured, samples=50_000
    ).to_dict()

    assert set(found) == {
        "kind",
        "counts",
        "circuits",
        "shots",
        "level",
        "measured",
        "models",
        "paired",
        "bits",
    }
    assert found["kind"] == "detectors" and found["shots"] == 2000
    assert len(found["bits"]) == code.num_detectors
    bit = found["bits"][3]
    assert bit["measured"] == pytest.approx(measured[:, 3].mean())
    assert set(bit["forecast"]) == set(bit["error"]) == {"calibration", "device"}
    assert found["paired"][0]["best"] == found["models"][0]["name"]
    assert set(found["paired"][0]["bit"]) >= {"z", "p_value", "sampling_ratio", "no_verdict"}


def test_two_equal_models_give_no_verdict_on_one_set_of_samples(
    line_9: Profile, code: stim.Circuit
) -> None:
    measured = _detections(line_9, code, 2000, seed=3)
    noisy = line_9.to_stim(code)
    same = events.detector_forecast(noisy, samples=40_000, seed=4)
    found = events.Events.of(measured)

    result = events.compare([found], [same], [same])

    assert result.bit_log.outcome == result.count_log.outcome == "identical"
    assert result.a == result.b


@pytest.mark.slow
def test_two_equal_models_give_a_verdict_in_about_the_share_of_the_level(
    line_9: Profile,
) -> None:
    small = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=3)
    runs, verdicts = 40, 0
    for run in range(runs):
        measured = _detections(line_9, small, 1000, seed=100 + run)
        report = events.score_detectors(
            {"one": line_9, "two": line_9}, small, measured, samples=200_000, seed=run
        )
        verdicts += report.tests[0].better is not None
    # With a level of 5%, 7 or more verdicts in 40 runs has a probability of about 0.003.
    assert verdicts <= 6


def test_a_file_of_detection_events_gives_the_report_of_the_array(
    line_9: Profile, code: stim.Circuit, tmp_path: Path
) -> None:
    measured = _detections(line_9, code, 500, seed=5)
    path = tmp_path / "events.b8"
    stim.write_shot_data_file(data=measured, path=str(path), format="b8", num_detectors=24)
    (tmp_path / "code.stim").write_text(str(code), encoding="utf-8")

    from_file = events.score_detectors(line_9, tmp_path / "code.stim", path, samples=5000)
    from_array = events.score_detectors(line_9, code, measured, samples=5000)

    assert from_file.to_dict() == from_array.to_dict()
    assert from_file.models[0].name == "line_9"


@pytest.mark.parametrize(
    ("circuit", "shape", "message"),
    [
        ("H 0\nM 0", (3, 0), "the circuit has no DETECTOR instruction"),
        (None, (10, 3), "the data has 3 bits in each shot, and the circuit has 24 detectors"),
    ],
)
def test_a_circuit_with_no_detector_or_other_detectors_than_the_data_is_refused(
    line_9: Profile, code: stim.Circuit, circuit, shape, message: str
) -> None:
    with pytest.raises(NoiseVaultError, match=message):
        events.score_detectors(line_9, code if circuit is None else circuit, np.zeros(shape))


def test_a_gate_that_the_profile_cannot_give_noise_to_is_refused(line_9: Profile) -> None:
    unknown = "C_XYZ 0\nM 0\nDETECTOR rec[-1]"
    far = "CX 0 5\nM 0 5\nDETECTOR rec[-1]"

    with pytest.raises(MissingCalibrationError, match="model line_9: C_XYZ 0 .*not defined"):
        events.score_detectors(line_9, unknown, np.zeros((3, 1)))
    with pytest.raises(MissingCalibrationError, match="cx has no calibration on qubits 0-5"):
        events.score_detectors(line_9, far, np.zeros((3, 1)))
    with pytest.warns(NoiseApproximationWarning, match="uses the noise of h instead"):
        typical = events.score_detectors(
            line_9, unknown, np.zeros((3, 1)), unknown_gates="typical", samples=1000
        )
    assert typical.models[0].events > 0


def test_a_circuit_with_noise_and_events_that_are_not_bits_are_refused(
    line_9: Profile, code: stim.Circuit
) -> None:
    noisy = stim.Circuit.generated(
        "repetition_code:memory", distance=5, rounds=5, after_clifford_depolarization=0.01
    )
    zeros = np.zeros((10, 24))

    with pytest.raises(NoiseVaultError, match=r"already has noise") as caught:
        events.score_detectors(line_9, noisy, zeros)
    assert "without_noise()" in caught.value.hint
    with pytest.raises(NoiseVaultError, match="the events hold the value 2; give 0 or 1"):
        events.score_detectors(line_9, code, np.full((10, 24), 2))
    with pytest.raises(NoiseVaultError, match="names no Stim data format"):
        events.score_detectors(line_9, code, "events.npy")


def test_without_stim_the_error_names_the_install_extra(
    line_9: Profile, code: stim.Circuit, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "stim", None)

    with pytest.raises(ImportError, match=r"noisevault\[stim\]"):
        events.score_detectors(line_9, "M 0\nDETECTOR rec[-1]", np.zeros((3, 1)))
    with pytest.raises(ImportError, match=r"noisevault\[stim\]"):
        events.score_wide(line_9, None)


def _rare(shots: int, events_on_bit_0: int) -> tuple[np.ndarray, np.ndarray]:
    """Measured bits with events on bit 0 only, and samples with a rate of 0.001 on each bit."""
    measured = np.zeros((shots, 2), dtype=bool)
    measured[:events_on_bit_0, 0] = True
    samples = np.zeros((100_000, 2), dtype=bool)
    samples[:100, 0] = True
    samples[100:200, 1] = True
    return measured, samples


def test_no_measured_event_does_not_contradict_a_forecast_of_a_rare_event() -> None:
    # 100 shots show no event with a probability of 0.82 when each bit has the rate 0.001.
    measured, samples = _rare(100, 0)
    text = events.score_events(measured, {"model": samples}).summary()
    assert "model: no proved difference in the number of events" in text
    assert "largest differences" not in text


def test_many_measured_events_on_one_bit_show_in_the_line_of_the_bits() -> None:
    # 30 events in 1000 shots against a rate of 0.001: the exact test of the bit proves the
    # difference. The bound for the number of events on 2 bits needs a larger difference.
    measured, samples = _rare(1000, 30)
    text = events.score_events(measured, {"model": samples}).summary()
    assert "model: 0.07 times the measured number of events, no proved difference" in text
    assert "largest differences, forecast against measured: b0 0.001 against 0.030" in text
    assert "b1" not in text.split("largest differences")[1].splitlines()[0]


def test_one_measured_event_gives_no_verdict_on_the_number_of_events() -> None:
    # One event in 100 shots occurs in 9.5% of the experiments at the forecast rate of 0.001.
    measured = np.zeros((100, 1), dtype=bool)
    measured[0, 0] = True
    samples = np.zeros((100_000, 1), dtype=bool)
    samples[:100, 0] = True
    text = events.score_events(measured, {"model": samples}).summary()
    assert "too few events" not in text and "largest differences" not in text
    assert "model: 0.10 times the measured number of events, no proved difference" in text


def test_rare_events_on_two_sides_give_no_verdict_on_the_number_of_events() -> None:
    # One bit has an event in almost each shot, and one bit has a rare event. One shot with
    # no event occurs in 21% of the experiments, and a normal test calls it a difference.
    measured = np.zeros((100, 2), dtype=bool)
    measured[1:, 0] = True
    samples = np.zeros((400_000, 2), dtype=bool)
    samples[500:, 0] = True
    samples[500:1000, 1] = True
    text = events.score_events(measured, {"model": samples}).summary()
    assert "model: 1.01 times the measured number of events, no proved difference" in text


def test_the_smoothing_of_the_rates_gives_no_verdict_on_the_number_of_events() -> None:
    # No shot and no sample has an event. The smoothed rate of the bit is 0.5 / 1001, far above
    # the measured rate of 0 in 200,000 shots. The verdict uses the counted samples, so it
    # sees no difference.
    measured, samples = np.zeros((200_000, 1), dtype=bool), np.zeros((1000, 1), dtype=bool)
    score = events.score_events(measured, {"model": samples})
    assert score.models[0].events_sampled == 0.0
    assert score.models[0].events == pytest.approx(0.5 / 1001)
    assert "model: no proved difference in the number of events" in score.summary()


def _half_width(counts: np.ndarray, bits: int, intervals: int) -> float:
    """The empirical Bernstein bound for the mean of ``counts``, each from 0 to ``bits``."""
    n, log_term = len(counts), math.log(4 * intervals / events.LEVEL)
    return math.sqrt(2 * counts.var(ddof=1) * log_term / n) + 7 * bits * log_term / (3 * (n - 1))


def test_a_large_difference_in_the_number_of_events_gets_a_verdict() -> None:
    rng = np.random.default_rng(0)
    measured, samples = rng.random((2000, 8)) < 0.2, rng.random((200_000, 8)) < 0.1
    score = events.score_events(measured, {"model": samples})
    model = score.models[0]
    by_hand = _half_width(measured.sum(axis=1), 8, 2) + _half_width(samples.sum(axis=1), 8, 2)
    assert model.events_margin == pytest.approx(by_hand, rel=1e-9)
    assert model.events_margin == pytest.approx(0.132, abs=0.001)
    assert model.events_sampled == pytest.approx(samples.sum() / 200_000, rel=1e-12)
    assert "model: too few events, 0.50 times the measured number (z = -41.8)" in score.summary()
    assert score.events_verdict(model) == "too few"
    entry = score.to_dict()["models"][0]
    assert entry["events_verdict"] == "too few"
    assert entry["events_sampled"] == model.events_sampled
    assert entry["events_margin"] == model.events_margin
    near = events.score_events(rng.random((2000, 8)) < 0.108, {"model": samples})
    assert near.to_dict()["models"][0]["events_verdict"] is None
    assert "times the measured number of events, no proved difference" in near.summary()


def test_the_bound_for_ten_shots_is_the_one_computed_by_hand() -> None:
    # Ten shots with 0, 0, 0, 0, 0, 0, 1, 1, 2 and 2 events on 2 bits: mean 0.6, sample
    # variance 6.4 / 9. One sample with the same numbers gives the same half width.
    counts = np.array([0, 0, 0, 0, 0, 0, 1, 1, 2, 2])
    rows = np.array([[k >= 1, k == 2] for k in counts])
    log_term = math.log(160)
    by_hand = math.sqrt(2 * (6.4 / 9) * log_term / 10) + 7 * 2 * log_term / (3 * 9)
    score = events.score_events(rows, {"model": rows})
    assert score.models[0].events_margin == pytest.approx(2 * by_hand, rel=1e-12)
    assert score.models[0].events_sampled == pytest.approx(0.6)


def test_the_bound_is_wider_when_a_wide_score_has_two_circuits(line_14: Profile) -> None:
    def margin(circuits: list[PlannedCircuit]) -> float:
        counts = events.simulate_wide(line_14, circuits, shots=300, seed=2, run_at=RUN_AT)
        return (
            events.score_wide({"stated": line_14}, counts, samples=20_000).models[0].events_margin
        )

    one, two = margin([_mirror(12, 2)]), margin([_mirror(12, 2), _mirror(12, 3)])
    # Two circuits need four intervals at one time, so each interval is wider.
    assert two > one * math.sqrt(math.log(320) / math.log(160)) * 0.9


def test_one_shot_and_one_sample_that_disagree_give_a_summary() -> None:
    score = events.score_events([[1]], {"model": np.array([[0]])})
    text = score.summary()
    assert "model: 0.00 times the measured number of events, no proved difference" in text
    # One shot gives no sample variance, so the margin is infinite and the JSON gives null.
    assert score.to_dict()["models"][0]["events_margin"] is None
    assert json.dumps(score.to_dict())


def _tail_p_value(events_: int, shots: int, sampled: int, samples: int) -> float:
    """Two times the smaller tail of the hypergeometric distribution, from binomial numbers."""
    total = events_ + sampled
    ways = [
        math.comb(shots, k) * math.comb(samples, total - k) if 0 <= total - k <= samples else 0
        for k in range(min(total, shots) + 1)
    ]
    below, above = sum(ways[: events_ + 1]), sum(ways[events_:])
    return min(1.0, 2 * min(below, above) / math.comb(shots + samples, total))


@pytest.mark.parametrize(
    "case",
    [(0, 100, 100, 5000), (25, 5000, 0, 1000), (13, 500, 20, 2000), (500, 500, 1999, 2000)],
)
def test_the_test_of_a_rate_is_the_exact_test_of_two_counts(
    case: tuple[int, int, int, int],
) -> None:
    assert events.two_sample_p_value(*case) == pytest.approx(_tail_p_value(*case), rel=1e-9)


def test_the_test_of_a_rate_holds_its_level_for_a_rare_event_and_few_samples() -> None:
    # 25 events in 5000 shots against no event in 1000 samples is no contradiction at the
    # level of 100 bits: at one rate, all 25 events are in the shots with a chance of 1%.
    p = events.two_sample_p_value(25, 5000, 0, 1000)
    assert p == pytest.approx(0.020756, rel=1e-4) and p > events.LEVEL / 100
    assert events.two_sample_p_value(4975, 5000, 1000, 1000) == pytest.approx(p)
    assert events.two_sample_p_value(13, 5000, 200, 200_000) == pytest.approx(0.004827, rel=1e-3)
    assert events.two_sample_p_value(0, 7, 0, 9) == 1.0
    # The tail goes on past a measured number that is far above the mean.
    far = events.two_sample_p_value(100, 1000, 900, 999_000)
    assert far == pytest.approx(3.4267707384723345e-163, rel=1e-9)


# wide circuits -------------------------------------------------------------------------------


def _mirror(n: int, depth: int) -> PlannedCircuit:
    """A Clifford circuit on ``n`` qubits whose one ideal result has a 1 on each odd qubit."""
    ops = [Op("x", (q,)) for q in range(1, n, 2)]
    for _ in range(depth):
        for start in (0, 1):
            pairs = [Op("cz", (q, q + 1)) for q in range(start, n - 1, 2)]
            ops += pairs + pairs
        ops += [Op("sx", (q,)) for q in range(n) for _ in range(4)]
    return PlannedCircuit(name=f"clifford_n{n}_d{depth}", qubits=tuple(range(n)), ops=tuple(ops))


@pytest.fixture(scope="module")
def line_14() -> Profile:
    data = Profile.uniform(
        "line_14",
        technology="superconducting",
        num_qubits=14,
        connectivity=[(q, q + 1) for q in range(13)],
        one_qubit_error=1e-3,
        two_qubit_error=1e-2,
        readout_error=2e-2,
        t1_us=200.0,
        t2_us=150.0,
        one_qubit_ns=32.0,
        two_qubit_ns=68.0,
    ).to_dict()
    data["gates"] = {k: v for k, v in data["gates"].items() if k in ("rz", "sx", "x", "cz")}
    return Profile.model_validate(data)


def test_the_true_profile_scores_best_on_the_wide_circuits_of_a_counts_file(
    line_14: Profile,
) -> None:
    truth = _factor(line_14, 2.0)
    counts = events.simulate_wide(truth, [_mirror(12, 3)], shots=3000, seed=6, run_at=RUN_AT)

    report = events.score_wide([line_14, truth], counts, samples=200_000)

    names = [m.name for m in report.models]
    assert names[0].startswith("line_14") and names[0] != names[1]
    assert report.models[0].profile is truth
    assert report.tests[0].better == names[0]
    assert report.circuits == ("clifford_n12_d3",)
    assert report.bits[1].name == "q1" and report.bits[1].circuit == "clifford_n12_d3"
    assert report.to_dict()["counts"] == counts.sha256
    assert "wide circuits: clifford_n12_d3, 3000 shots" in report.summary()


def test_a_counts_file_with_no_wide_circuit_or_an_unknown_name_is_refused(
    line_14: Profile,
) -> None:
    narrow = events.simulate_wide(line_14, [_mirror(8, 1)], shots=100, seed=7, run_at=RUN_AT)

    with pytest.raises(CountsError, match="the counts hold no circuit on more than 10 qubits"):
        events.score_wide(line_14, narrow)
    with pytest.raises(CountsError, match="the counts have no circuit 'deep_d1_s0'"):
        events.score_wide(line_14, narrow, circuits=["deep_d1_s0"])
    named = events.score_wide(line_14, narrow, circuits=["clifford_n8_d1"], samples=2000)
    assert named.circuits == ("clifford_n8_d1",)


def test_samples_from_any_simulator_give_the_same_report() -> None:
    measured = _draws(2000, 0.04, 8)

    report = events.score_events(
        measured, {"right": _draws(100_000, 0.04, 9), "low": iter([_draws(100_000, 0.02, 10)])}
    )

    assert [m.name for m in report.models] == ["right", "low"]
    assert report.tests[0].better == "right"
    assert report.models[1].profile is None
    assert "low: 0.74 times the measured number of events, no proved difference" in report.summary()


@pytest.mark.timing
def test_one_80_qubit_circuit_of_the_circuit_set_scores_two_models_in_under_a_minute() -> None:
    from noisevault.suite import ASSUMED_GATE_FACTOR, _clifford_circuits

    data = Profile.uniform(
        "line_84",
        technology="superconducting",
        num_qubits=84,
        connectivity=[(q, q + 1) for q in range(83)],
        one_qubit_error=1e-3,
        two_qubit_error=1e-2,
        readout_error=2e-2,
        t1_us=200.0,
        t2_us=150.0,
        one_qubit_ns=32.0,
        two_qubit_ns=68.0,
    ).to_dict()
    data["gates"] = {k: v for k, v in data["gates"].items() if k in ("rz", "sx", "x", "cz")}
    profile = Profile.model_validate(data)
    (circuit,) = (
        c
        for c in _clifford_circuits(profile, _factor(profile, ASSUMED_GATE_FACTOR), 0)
        if len(c.qubits) == 80
    )
    truth = _factor(profile, 2.0)
    counts = events.simulate_wide(truth, [circuit], shots=4000, seed=1, run_at=RUN_AT)

    start, cpu = time.perf_counter(), time.process_time()
    report = events.score_wide({"calibration": profile, "device": truth}, counts)
    elapsed, used = time.perf_counter() - start, time.process_time() - cpu

    print(f"score_wide of one 80-qubit circuit, two models: {elapsed:.1f} s, {used:.1f} s of CPU")
    assert report.tests[0].better == "device"
    # A shared machine stretches the wall time, so the budget is on the CPU time of the process.
    assert used < 60


def test_nv_score_scores_the_wide_circuits_only_with_the_option(
    line_14: Profile, tmp_path: Path
) -> None:
    from typer.testing import CliRunner

    from noisevault.cli import app

    truth = _factor(line_14, 2.0)
    narrow = PlannedCircuit(
        name="narrow", qubits=(0, 1, 2, 3), ops=(Op("x", (1,)), Op("cz", (0, 1)), Op("sx", (2,)))
    )
    counts = events.simulate_wide(
        truth, [narrow, _mirror(12, 3)], shots=3000, seed=6, run_at=RUN_AT
    )
    files = {name: tmp_path / f"{name}.json" for name in ("stated", "truth", "counts")}
    line_14.save(files["stated"])
    truth.save(files["truth"])
    counts.save(files["counts"])
    args = ["score", str(files["stated"]), str(files["truth"]), str(files["counts"])]
    runner = CliRunner()

    plain = runner.invoke(app, args, env={"COLUMNS": "200"})
    assert plain.exit_code == 0, plain.output
    assert "wide circuits:" not in plain.stdout
    assert "hint: add --wide to score the 1 circuit on more than 10 qubits" in plain.stderr
    assert json.loads(runner.invoke(app, [*args, "--json"]).stdout)["wide"] is None

    wide = runner.invoke(app, [*args, "--wide", "--samples", "100000"], env={"COLUMNS": "200"})
    assert wide.exit_code == 0, wide.output
    assert (
        "wide circuits: clifford_n12_d3, 3000 shots, forecasts from 100000 samples" in wide.stdout
    )
    assert "hint: add --wide" not in wide.stderr
    data = json.loads(runner.invoke(app, [*args, "--wide", "--samples", "100000", "--json"]).stdout)
    assert data["wide"]["circuits"] == ["clifford_n12_d3"]
    assert data["wide"]["models"][0]["name"] == str(files["truth"])


def test_nv_score_with_the_option_scores_a_file_of_only_wide_circuits(
    line_14: Profile, tmp_path: Path
) -> None:
    from typer.testing import CliRunner

    from noisevault.cli import app

    counts = events.simulate_wide(line_14, [_mirror(12, 3)], shots=500, seed=6, run_at=RUN_AT)
    profile, path = tmp_path / "stated.json", tmp_path / "counts.json"
    line_14.save(profile)
    counts.save(path)
    args = ["score", str(profile), str(path)]
    runner = CliRunner()

    plain = runner.invoke(app, args, env={"COLUMNS": "200"})
    assert plain.exit_code == 1
    assert "circuit clifford_n12_d3 has 12 qubits, more than the 10" in plain.stderr
    assert "--wide" in plain.stderr

    wide = runner.invoke(app, [*args, "--wide", "--samples", "20000"], env={"COLUMNS": "200"})
    assert wide.exit_code == 0, wide.output
    assert "wide circuits: clifford_n12_d3, 500 shots" in wide.stdout
    data = json.loads(runner.invoke(app, [*args, "--wide", "--samples", "20000", "--json"]).stdout)
    assert data["wide"]["circuits"] == ["clifford_n12_d3"]
    assert data["models"] == [] and data["counts"] == counts.sha256


def test_nv_score_says_when_the_option_for_wide_circuits_has_no_effect(
    line_14: Profile, tmp_path: Path
) -> None:
    from typer.testing import CliRunner

    from noisevault.cli import app

    narrow = events.simulate_wide(line_14, [_mirror(8, 1)], shots=200, seed=7, run_at=RUN_AT)
    profile, counts = tmp_path / "stated.json", tmp_path / "counts.json"
    line_14.save(profile)
    narrow.save(counts)

    result = CliRunner().invoke(app, ["score", str(profile), str(counts), "--wide"])
    assert result.exit_code == 0, result.output
    assert "warning: --wide has no effect" in result.stderr
