from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from conftest import require

require("stim")

from noisevault.counts import load_counts  # noqa: E402
from noisevault.errors import CountsError  # noqa: E402
from noisevault.profile import Profile  # noqa: E402
from noisevault.suite import ASSUMED_GATE_FACTOR, _clifford_circuits  # noqa: E402
from study import wide  # noqa: E402

RUN_AT = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)
SHOTS = 2000


@pytest.fixture(scope="module")
def line_24() -> Profile:
    """24 qubits in a line with the IBM gate set, so that a chain of 20 exists."""
    data = Profile.uniform(
        "line",
        technology="superconducting",
        num_qubits=24,
        connectivity=[(q, q + 1) for q in range(23)],
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


def _factor(profile: Profile, factor: float) -> Profile:
    return profile.model_copy(update={"unmodeled_error": {"gates": {"factor": factor}}})


@pytest.fixture(scope="module")
def counts(line_24: Profile):
    truth = _factor(line_24, ASSUMED_GATE_FACTOR)
    circuits = _clifford_circuits(line_24, truth, 0)
    assert [len(c.qubits) for c in circuits] == [20]
    return wide.simulate(truth, circuits, shots=SHOTS, seed=4, run_at=RUN_AT)


def test_the_model_of_the_truth_scores_above_the_calibration_on_each_score(
    line_24: Profile, counts
) -> None:
    result = wide.paired(_factor(line_24, 2.0), line_24, counts)

    tests = [
        result.success_log,
        result.success_quadratic,
        result.wrong_log,
        result.wrong_quadratic,
        result.qubit_log,
    ]
    assert [t.better for t in tests] == ["a"] * 5
    assert all(t.outcome == "tested" and t.p_value < 0.05 for t in tests)
    assert result.a.success_log > result.b.success_log
    assert result.a.wrong_log > result.b.wrong_log
    assert result.a.qubit_log > result.b.qubit_log


def test_two_equal_models_give_no_verdict(line_24: Profile, counts) -> None:
    result = wide.paired(line_24, line_24, counts, samples=20_000)

    for test in (result.success_log, result.wrong_quadratic, result.qubit_log):
        assert test.outcome == "identical"
        assert test.better == "neither"
    assert result.a == result.b


def test_a_forecast_without_noise_gives_the_ideal_result_with_add_half_smoothing() -> None:
    profile = Profile.uniform(
        "ideal",
        technology="superconducting",
        num_qubits=24,
        connectivity=[(q, q + 1) for q in range(23)],
        one_qubit_error=1e-3,
        two_qubit_error=1e-2,
    )
    (circuit,) = _clifford_circuits(profile, profile, 0)
    noiseless = Profile.uniform(
        "ideal",
        technology="superconducting",
        num_qubits=24,
        connectivity=[(q, q + 1) for q in range(23)],
        one_qubit_error=0.0,
        two_qubit_error=0.0,
    )

    found = wide.forecast(noiseless, circuit, samples=1000)

    assert found.success == pytest.approx(1000.5 / 1001)
    assert found.qubit == pytest.approx(np.full(20, 0.5 / 1001))
    assert found.wrong.sum() == pytest.approx(1.0)
    assert found.wrong[0] == pytest.approx(1000.5 / (1000 + 21 / 2))


def test_the_rare_wrong_cells_become_one_last_cell() -> None:
    measured = np.array([50.0, 40.0, 9.0, 1.0, 0.0])
    a = np.array([0.5, 0.4, 0.08, 0.019, 0.001])
    b = np.array([0.6, 0.3, 0.06, 0.039, 0.001])

    cells, merged_a, merged_b = wide._merged(measured, a, b, 100)

    assert cells.tolist() == [50.0, 40.0, 9.0, 1.0]
    assert merged_a == pytest.approx([0.5, 0.4, 0.08, 0.02])
    assert merged_b == pytest.approx([0.6, 0.3, 0.06, 0.04])


def test_the_qubit_score_of_a_shot_sums_the_log_marginal_of_each_bit() -> None:
    wrong = np.array([[True, False, False], [False, False, True]])
    marginals = np.array([0.1, 0.2, 0.3])

    found = wide._qubit_scores(wrong, marginals)

    assert found == pytest.approx([np.log(0.1 * 0.8 * 0.7), np.log(0.9 * 0.8 * 0.3)])


def test_the_paired_z_test_of_the_shots_uses_the_sample_variance_of_the_differences() -> None:
    times = np.array([3.0, 1.0])
    a, b = np.array([-1.0, -2.0]), np.array([-1.5, -1.0])

    found = wide._shot_test([(times, a, b)])

    d = np.array([0.5, 0.5, 0.5, -1.0])
    assert found.difference == pytest.approx(d.mean())
    assert found.standard_error == pytest.approx(d.std(ddof=1) / np.sqrt(4))


def test_a_wide_counts_file_saves_loads_selects_and_splits(counts, tmp_path: Path) -> None:
    name = counts.circuits[0].name

    loaded = load_counts(counts.save(tmp_path / "wide.counts.json"))
    first, second = loaded.split(0.5, seed=0)

    assert loaded == counts
    assert loaded.select([name]).sha256 == counts.sha256
    assert first.circuits[0].shots == second.circuits[0].shots == SHOTS // 2
    assert name.startswith("clifford_n20_d")


def test_one_chunk_gives_the_bits_of_the_stabilizer_sample(line_24: Profile, counts) -> None:
    from noisevault import wide as stabilizer

    circuit = counts.circuits[0]
    bits = stabilizer.sample(line_24, circuit, 5000, seed=9)
    wrong = bits != np.array([b == "1" for b in stabilizer.ideal_bits(circuit)])

    one = wide.forecast(line_24, circuit, samples=5000, seed=9, chunk=5000)
    many = wide.forecast(line_24, circuit, samples=5000, seed=9, chunk=700)

    assert one.qubit == pytest.approx((wrong.sum(axis=0) + 0.5) / 5001)
    assert one.success == pytest.approx((int((~wrong.any(axis=1)).sum()) + 0.5) / 5001)
    assert many.wrong.sum() == pytest.approx(1.0)
    assert np.abs(many.qubit - one.qubit).max() < 0.03


def test_the_qubit_test_holds_the_sampling_variance_of_both_forecasts(
    line_24: Profile, counts
) -> None:
    circuit = counts.circuits[0]
    samples = 3000
    a = wide.forecast(_factor(line_24, 2.0), circuit, samples=samples, seed=1)
    b = wide.forecast(_factor(line_24, 1.5), circuit, samples=samples, seed=2)

    result = wide.compare([circuit], [a], [b])

    keys, times = wide._rows(circuit)
    f = times @ (keys != np.array([bit == "1" for bit in a.ideal])) / circuit.shots
    expected = sum(
        circuit.shots**2 / samples * ((f - q) ** 2 / (q * (1 - q))).sum()
        for q in (a.qubit, b.qubit)
    )
    assert result.qubit_sampling.variance == pytest.approx(expected)
    plain = wide._shot_test(wide._cells(circuit, a, b, merge=True)[2:])
    shot_variance = (plain.standard_error * circuit.shots) ** 2
    assert result.qubit_sampling.shot_variance == pytest.approx(shot_variance)
    assert result.qubit_sampling.ratio == pytest.approx(expected / shot_variance)
    assert result.qubit_log.difference == pytest.approx(plain.difference)
    assert result.qubit_log.standard_error == pytest.approx(
        np.sqrt(shot_variance + expected) / circuit.shots
    )
    assert result.qubit_log.standard_error > plain.standard_error
    measured, pa, pb = wide._cells(circuit, a, b, merge=True)[1]
    cells = sum(
        circuit.shots**2 / samples * (((measured / circuit.shots) ** 2 / p).sum() - 1)
        for p in (pa, pb)
    )
    assert result.wrong_sampling.variance == pytest.approx(cells)


def test_a_test_with_more_sampling_variance_than_shot_variance_gives_no_verdict() -> None:
    from noisevault.score import RuleTest

    clear = RuleTest("log", "tested", 0.01, 0.001, 10.0, 1e-23, "a")

    kept, small = wide.with_sampling(clear, 0.5 * (0.001 * 1000) ** 2, 1000)
    lost, large = wide.with_sampling(clear, 1.5 * (0.001 * 1000) ** 2, 1000)

    assert small.ratio == pytest.approx(0.5) and kept.better == "a"
    assert kept.z == pytest.approx(10 / np.sqrt(1.5))
    assert large.ratio == pytest.approx(1.5) and lost.better == "neither"
    assert lost.z == pytest.approx(10 / np.sqrt(2.5)) and lost.p_value < 0.05


def test_the_ideal_bit_string_gives_no_verdict_on_40_qubits() -> None:
    data = Profile.uniform(
        "line",
        technology="superconducting",
        num_qubits=44,
        connectivity=[(q, q + 1) for q in range(43)],
        one_qubit_error=1e-3,
        two_qubit_error=1e-2,
        readout_error=2e-2,
        t1_us=200.0,
        t2_us=150.0,
        one_qubit_ns=32.0,
        two_qubit_ns=68.0,
    ).to_dict()
    data["gates"] = {k: v for k, v in data["gates"].items() if k in ("rz", "sx", "x", "cz")}
    line = Profile.model_validate(data)
    truth = _factor(line, ASSUMED_GATE_FACTOR)
    circuits = _clifford_circuits(line, truth, 0)
    assert [len(c.qubits) for c in circuits] == [20, 40]
    found = wide.simulate(truth, circuits, shots=500, seed=4, run_at=RUN_AT)

    both = wide.paired(_factor(line, 3.0), line, found, samples=20_000)
    small = wide.paired(
        _factor(line, 3.0), line, found, circuits=[circuits[0].name], samples=20_000
    )

    assert both.success_verdict is False and both.success_log.better == "neither"
    assert both.success_quadratic.better == "neither"
    assert small.success_verdict is True


def test_a_run_without_a_clifford_circuit_is_refused(line_24: Profile, counts) -> None:
    with pytest.raises(CountsError, match="the counts have no circuit 'deep_d1_s0'"):
        wide.paired(line_24, line_24, counts, circuits=["deep_d1_s0"])
