from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from study import correlation_rules as rules
from study import estimate as est
from study.error_rates import series

MONDAY = date(2026, 11, 16)


def test_the_rules_plan_and_none_are_the_estimate_of_the_plan() -> None:
    y = series(np.random.default_rng(7), 200, 0.4, rules.SPREAD, mu=0.1)
    rho_hat, found = rules.corrections(y)
    for study in range(200):
        values = {
            f"device{d}": {
                MONDAY + timedelta(days=7 * w): float(v) for w, v in enumerate(y[study, d])
            }
            for d in range(rules.DEVICES)
        }
        expected = est.estimate(values)
        assert rho_hat[study] == pytest.approx(expected.rho_hat, abs=1e-12)
        assert found["plan"].rho[study] == pytest.approx(expected.rho, abs=1e-12)
        assert found["plan"].t[study] == pytest.approx(expected.corrected.t, rel=1e-9)
        assert found["none"].t[study] == pytest.approx(expected.plain.t, rel=1e-9)


@pytest.mark.parametrize(
    ("rule", "bias"),
    [("inverse", lambda rho, n: (1 + 4 * rho) / n), ("inverse3", lambda rho, n: (1 + 3 * rho) / n)],
)
def test_an_inverse_rule_gives_back_the_true_correlation(rule: str, bias) -> None:
    n, rho = 8, 0.4
    measured = np.array([rho - bias(rho, n)])
    assert rules.RULES[rule](measured, n)[0] == pytest.approx(rho)


def test_each_rule_keeps_rho_in_the_limits_of_the_plan() -> None:
    y = series(np.random.default_rng(2), 2000, 0.7, rules.SPREAD)
    _, found = rules.corrections(y)
    for corrected in found.values():
        assert corrected.rho.min() >= est.RHO_LIMITS[0]
        assert corrected.rho.max() <= est.RHO_LIMITS[1]
    assert np.all(found["none"].rho == 0)


def test_the_measured_correlation_of_independent_weeks_is_near_minus_one_eighth() -> None:
    y = series(np.random.default_rng(3), 20_000, 0.0, rules.SPREAD)
    rho_hat, _ = rules.corrections(y)
    assert rho_hat.mean() == pytest.approx(-1 / rules.WEEKS, abs=0.01)


def test_below_counts_the_studies_past_the_two_sided_limit() -> None:
    limit = est.t_quantile(0.975, 21)
    t = np.array([limit + 0.01, -(limit + 0.01), limit - 0.01, 0.0])
    assert rules.below(t, 0.05, 21) == 0.5


def test_the_report_has_four_parts_and_each_rule(capsys: pytest.CaptureFixture[str]) -> None:
    assert rules.main(["--studies", "2000", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("2000 studies for each line, 3 devices, 8 weeks, seed 1")
    for part in ("\n1. ", "\n2. ", "\n3. ", "\n4. "):
        assert part in out
    for name in rules.RULES:
        assert out.count(f" {name} ") >= 1
