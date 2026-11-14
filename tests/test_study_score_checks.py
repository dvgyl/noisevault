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


def test_main_runs_one_check(capsys: pytest.CaptureFixture[str]) -> None:
    assert score_checks.main(["hidden", "--jobs", "20", "--shots", "500"]) == 0
    out = capsys.readouterr().out
    assert "500 shots, 20 jobs, seed 11" in out
    assert out.count("verdict for a") == 3
