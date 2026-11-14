from __future__ import annotations

import math

import numpy as np
import pytest

from noisevault.profile import Profile
from noisevault.score import LEVEL
from noisevault.suite import role
from study import score_checks


@pytest.fixture(scope="module")
def hidden() -> dict:
    return score_checks.hidden_check(jobs=300, seed=3)


def test_the_hidden_device_gives_the_two_models_the_same_expected_score() -> None:
    truth = score_checks.hidden_truth()
    a = score_checks.hidden_model(score_checks.A_RARE)
    b = score_checks.hidden_model(score_checks.B_RARE)
    assert truth.sum() == pytest.approx(1.0)
    assert score_checks.B_RARE < truth[1] < score_checks.A_RARE
    assert abs(float(truth @ (np.log(a) - np.log(b)))) < 1e-15


def test_step_2_holds_the_level_and_step_1_takes_it_away(hidden: dict) -> None:
    verdicts = hidden["verdicts"]
    # With no shot of R, step 1 hides R and the usual result favors model b.
    assert verdicts["steps 1 and 2"]["b"] > 0.8
    assert verdicts["no step"]["b"] > 0.8
    assert verdicts["step 2 only"] == {"a": 0.0, "b": 0.0}
    assert hidden["difference_rare"] == pytest.approx(math.log(0.0004 / 1e-12))


def test_a_result_that_no_shot_showed_misleads_only_the_natural_test() -> None:
    a, b = 0.001, 1e-12
    wrong = score_checks.matched_wrong(a, b)
    expected = wrong * math.log(a / b) + (1 - wrong) * math.log((1 - a) / (1 - b))
    assert wrong == pytest.approx(4.8276755196e-5, rel=1e-9)
    assert expected == pytest.approx(0.0, abs=1e-15)
    found = score_checks.unseen_check(jobs=200, shots=2000, seed=42)
    assert found["truth"][1] == pytest.approx(0.100004999926, rel=1e-9)
    assert found["verdicts"]["steps 1 and 2"] == {"a": 0.0, "b": 0.0}
    assert found["verdicts"]["no step"]["a"] == 0.0
    assert found["verdicts"]["no step"]["b"] > 0.85
    assert found["verdicts"]["no step"]["b"] == pytest.approx(1 - found["occurred"], abs=0.02)


def test_the_natural_test_adds_the_circuits() -> None:
    one = (np.array([900, 100]), np.array([1.0, -1.0]))
    flat = (np.array([10, 0]), np.array([0.5, 9.0]))
    assert score_checks.natural_verdict_of([flat, flat]) == "neither"
    assert score_checks.natural_verdict_of([one, flat]) == "a"


def test_the_natural_test_has_no_verdict_with_no_variance() -> None:
    assert score_checks.natural_verdict(np.array([10, 0]), np.array([0.5, 9.0])) == "neither"
    assert score_checks.natural_verdict(np.array([900, 100]), np.array([1.0, -1.0])) == "a"


def test_the_smallest_difference_meets_the_hoeffding_bound() -> None:
    low, high = score_checks.shot_range()
    assert low == pytest.approx(4 * math.log(0.99 / 0.99988))
    assert high == pytest.approx(4 * math.log(0.01 / 0.00012))
    t = score_checks.smallest_difference(high - low, 1250)
    assert 2 * math.exp(-2 * 1250 * t**2 / (high - low) ** 2) == pytest.approx(LEVEL)
    assert round(t, 3) == 0.681


def test_safe_report_gives_both_shot_numbers() -> None:
    text = score_checks.safe_report()
    assert "width 17.73" in text
    assert "36250 shots" in text and "verdict 0.126" in text


def _device() -> Profile:
    return Profile.uniform(
        "toy",
        technology="superconducting",
        num_qubits=2,
        one_qubit_error=0.01,
        two_qubit_error=0.02,
        readout_error=0.03,
        connectivity=[(0, 1)],
    )


@pytest.fixture(scope="module")
def chain() -> dict:
    profile = _device()
    return score_checks.chain_check(profile, (0, 1), fresh=5, shots=200, jobs=2)


def test_the_chain_circuits_leave_out_the_probe_circuits() -> None:
    circuits = score_checks.chain_circuits(_device(), (0, 1), fresh=5)
    assert circuits
    assert all(set(c.qubits) <= {0, 1} and role(c.name) != "probe" for c in circuits)
    assert "deep_d8_f5" in [c.name for c in circuits]


def test_the_matched_factor_gives_the_same_expected_loss(chain: dict) -> None:
    assert chain["above"] > score_checks.TRUTH
    assert chain["loss_above"] == pytest.approx(chain["loss_below"], rel=1e-6)
    for name in ("level", "power_near", "wrong_near", "power_plain"):
        assert chain[name] in (0.0, 0.5, 1.0)


def test_a_result_that_the_truth_cannot_give_adds_no_loss() -> None:
    truth, model = [np.array([0.5, 0.5, 0.0])], [np.array([0.25, 0.75, 0.0])]
    by_hand = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    assert score_checks.divergence(truth, model) == pytest.approx(by_hand)
    assert score_checks.divergence(truth, [np.array([0.5, 0.0, 0.5])]) == math.inf


def test_a_readout_error_of_0_still_gives_two_models_with_the_same_loss() -> None:
    profile = Profile.uniform(
        "exact_readout",
        technology="superconducting",
        num_qubits=2,
        one_qubit_error=0.01,
        two_qubit_error=0.02,
        readout_error=0.0,
        connectivity=[(0, 1)],
    )
    circuits = score_checks.chain_circuits(profile, (0, 1), fresh=5)
    above, target = score_checks.matched_factor(profile, circuits)
    exact = score_checks.outcome_probabilities(
        score_checks.gate_factor(profile, score_checks.TRUTH), circuits
    )
    found = score_checks.outcome_probabilities(score_checks.gate_factor(profile, above), circuits)
    assert target > 0 and above > score_checks.TRUTH
    assert score_checks.divergence(exact, found) == pytest.approx(target, rel=1e-6)


def test_the_check_refuses_when_no_factor_has_the_same_loss(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # With errors near full depolarization, no factor up to the limit loses as much as 1.2.
    profile = Profile.uniform(
        "noisy",
        technology="superconducting",
        num_qubits=2,
        one_qubit_error=0.49,
        two_qubit_error=0.74,
        readout_error=0.03,
        connectivity=[(0, 1)],
    )
    circuits = score_checks.chain_circuits(profile, (0, 1), fresh=5)
    with pytest.raises(ValueError, match="no gate factor from 1.5 to 20.0 has the expected"):
        score_checks.matched_factor(profile, circuits)


def test_the_plain_model_has_no_correction_of_the_profile(chain: dict) -> None:
    corrected = score_checks.gate_factor(_device(), score_checks.TRUTH)
    again = score_checks.chain_check(corrected, (0, 1), fresh=5, shots=200, jobs=2)
    assert again == chain


def test_main_runs_one_check(capsys: pytest.CaptureFixture[str]) -> None:
    assert score_checks.main(["hidden", "--jobs", "20", "--shots", "500"]) == 0
    out = capsys.readouterr().out
    assert "500 shots, 20 jobs, seed 11" in out
    assert out.count("verdict for a") == 3
