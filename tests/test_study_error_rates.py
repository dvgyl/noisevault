from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from study import error_rates
from study import estimate as est

MONDAY = date(2026, 11, 16)


@pytest.mark.parametrize("min_rho", [0.0, est.CAREFUL_RHO])
def test_the_fast_statistic_is_the_statistic_of_the_estimate(min_rho: float) -> None:
    rng = np.random.default_rng(3)
    y = error_rates.series(rng, 40, 0.3, (1.0, 2.0, 3.0), mu=0.2)
    fast = error_rates.statistics(y, min_rho)
    for study in range(40):
        values = {
            f"device{d}": {
                MONDAY + timedelta(days=7 * w): float(v) for w, v in enumerate(y[study, d])
            }
            for d in range(3)
        }
        assert est.estimate(values, min_rho=min_rho).corrected.t == pytest.approx(fast[study])


def test_the_series_has_the_correlation_and_the_spread_that_it_names() -> None:
    y = error_rates.series(np.random.default_rng(5), 40_000, 0.5, (1.0, 3.0))
    assert y[:, 0].std() == pytest.approx(1.0, abs=0.02)
    assert y[:, 1].std() == pytest.approx(3.0, abs=0.06)
    neighbours = np.corrcoef(y[:, 0, 3], y[:, 0, 4])[0, 1]
    two_apart = np.corrcoef(y[:, 0, 2], y[:, 0, 4])[0, 1]
    assert neighbours == pytest.approx(0.5, abs=0.02)
    assert two_apart == pytest.approx(0.25, abs=0.02)


def test_the_plan_keeps_its_level_only_for_independent_weeks() -> None:
    df = error_rates.DEVICES * (error_rates.WEEKS - 1)
    spread = error_rates.SPREADS["1, 1, 1"]
    found = {}
    for rho in (0.0, 0.5):
        y = error_rates.series(np.random.default_rng([7, round(rho * 10)]), 40_000, rho, spread)
        found[rho] = {
            rule: error_rates.rates(error_rates.statistics(y, min_rho), df)
            for rule, min_rho in error_rates.RULES.items()
        }
    assert 0.03 < found[0.0]["plan"][0] < 0.05
    assert 0.09 < found[0.5]["plan"][0] < 0.14
    assert 0.17 < found[0.5]["plan"][2] < 0.28
    assert 0.03 < found[0.5]["careful"][2] < 0.08
    assert found[0.0]["careful"][2] < 0.005


def test_the_holm_fraction_counts_a_set_with_one_small_p_value() -> None:
    df = 21
    first = est.t_quantile(1 - 0.05 / 5 / 2, df)
    t = np.array([0.0, 0.0, first + 0.01, 0.0, 0.0, 0.0, 0.0, first - 0.01, 0.0, 0.0])
    assert error_rates.rates(t, df)[2] == 0.5


def test_the_report_names_each_rule(capsys: pytest.CaptureFixture[str]) -> None:
    assert error_rates.main(["--studies", "2000", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "2000 studies for each line, 3 devices, 8 weeks, seed 1" in out
    assert out.count("plan") >= 10 and out.count("careful") >= 10
