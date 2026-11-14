from __future__ import annotations

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
    # The true model has no bit whose difference is more than chance over 24 bits.
    assert "device: the number of events agrees with the measured number, within chance\n" in text
    assert "within chance\ncalibration: too few events" in text
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
    assert "low: too few events, 0.7" in report.summary()


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
