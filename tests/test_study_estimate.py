from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from study import estimate as est

MONDAY = date(2026, 11, 16)


def _weeks(values: list[float], start: int = 0) -> dict[date, float]:
    return {MONDAY + timedelta(days=7 * (start + i)): v for i, v in enumerate(values)}


def _t_quantile(p: float, df: int) -> float:
    stats = pytest.importorskip("scipy.stats")
    return float(stats.t.ppf(p, df))


def _independent(table: dict[str, dict[date, float]], f_one: bool = False) -> dict[str, float]:
    """The formulas of the plan again, in plain numpy, for the devices with 4 weeks or more."""
    stats = pytest.importorskip("scipy.stats")
    kept = {d: w for d, w in table.items() if len(w) >= 4}
    means = {d: np.mean(list(w.values())) for d, w in kept.items()}
    variances = {d: np.var(list(w.values()), ddof=1) for d, w in kept.items()}
    top = bottom = 0.0
    for d, w in kept.items():
        for k, y in w.items():
            r = y - means[d]
            bottom += r * r
            before = k - timedelta(days=7)
            if before in w:
                top += r * (w[before] - means[d])
    rho_hat = top / bottom
    n_bar = np.mean([len(w) for w in kept.values()])
    rho = float(np.clip(rho_hat + (1 + 3 * rho_hat) / n_bar, 0.0, 0.9))
    f = {}
    for d, w in kept.items():
        n = len(w)
        k = np.arange(1, n)
        g = float(((1 - k / n) * rho**k).sum())
        f[d] = 1.0 if f_one else (1 + 2 * g) / (1 - 2 * g / (n - 1))
    count = len(kept)
    value = np.mean(list(means.values()))
    se = np.sqrt(sum(f[d] * variances[d] / len(w) for d, w in kept.items())) / count
    df = sum(len(w) - 1 for w in kept.values())
    t = value / se
    half = stats.t.ppf(0.975, df) * se
    return {
        "estimate": value,
        "se": se,
        "t": t,
        "df": df,
        "p": 2 * stats.t.sf(abs(t), df),
        "low": value - half,
        "high": value + half,
        "rho_hat": rho_hat,
        "rho": rho,
    }


def _check(found: est.Estimate, table: dict[str, dict[date, float]]) -> None:
    expected = _independent(table)
    plain = _independent(table, f_one=True)
    assert found.estimate == pytest.approx(expected["estimate"])
    assert found.df == expected["df"]
    assert found.rho_hat == pytest.approx(expected["rho_hat"])
    assert found.rho == pytest.approx(expected["rho"])
    assert found.corrected.standard_error == pytest.approx(expected["se"])
    assert found.corrected.t == pytest.approx(expected["t"])
    assert found.corrected.p_value == pytest.approx(expected["p"])
    assert found.corrected.low == pytest.approx(expected["low"])
    assert found.corrected.high == pytest.approx(expected["high"])
    assert found.plain.standard_error == pytest.approx(plain["se"])
    assert found.plain.p_value == pytest.approx(plain["p"])


def test_the_estimate_with_equal_n_matches_an_independent_computation() -> None:
    table = {
        "ibm_fez": _weeks([0.004, 0.002, 0.005, 0.001, 0.003]),
        "ibm_kingston": _weeks([0.001, -0.002, 0.000, 0.003, 0.002]),
        "ibm_marrakesh": _weeks([0.006, 0.004, 0.007, 0.002, 0.005]),
    }

    found = est.estimate(table)

    _check(found, table)
    assert [d.n for d in found.devices] == [5, 5, 5]
    assert found.device_weeks == 15
    assert found.sign == (13, 1, est.sign_test(13, 1))


def test_the_estimate_with_unequal_n_matches_an_independent_computation() -> None:
    table = {
        "ibm_fez": _weeks([0.004, 0.002, 0.005, 0.001, 0.003, 0.006, 0.000, 0.002]),
        "ibm_kingston": _weeks([0.001, -0.002, 0.002, 0.003], start=2),
        # Weeks 0, 1, 3, 4, 6: only the pairs (0, 1) and (3, 4) are 7 days apart.
        "ibm_marrakesh": {
            MONDAY + timedelta(days=7 * i): v
            for i, v in zip((0, 1, 3, 4, 6), (0.006, 0.001, 0.007, 0.002, 0.004), strict=True)
        },
    }

    found = est.estimate(table)

    _check(found, table)
    assert {d.name: d.n for d in found.devices} == {
        "ibm_fez": 8,
        "ibm_kingston": 4,
        "ibm_marrakesh": 5,
    }


def test_a_device_with_fewer_than_4_weeks_is_not_in_the_estimate() -> None:
    table = {
        "ibm_fez": _weeks([0.004, 0.002, 0.005, 0.001]),
        "ibm_kingston": _weeks([0.001, -0.002, 0.000, 0.003]),
        "ibm_marrakesh": _weeks([0.5, 0.4, 0.3]),
    }

    found = est.estimate(table, {"missing_run": 2})

    _check(found, table)
    assert [d.name for d in found.devices] == ["ibm_fez", "ibm_kingston"]
    assert found.excluded == {"ibm_marrakesh": 3}
    assert found.incomplete == {"missing_run": 2}
    assert found.device_weeks == 8


def test_no_device_with_4_weeks_gives_no_estimate_and_no_test() -> None:
    found = est.estimate({"ibm_fez": _weeks([0.1, 0.2, 0.3])}, {"missing_run": 1})

    assert found.estimate is None and found.corrected is None and found.plain is None
    data = found.to_dict()
    assert data["test"] is None and data["devices_with_too_few_weeks"] == {"ibm_fez": 3}
    assert est.estimate({}).to_dict()["estimate"] is None


def test_rho_is_limited_at_0_and_at_0_9() -> None:
    # Values that change sign each week give a negative pooled correlation, so rho is 0.
    alternating = {d: _weeks([1.0, -1.0] * 4) for d in ("a", "b", "c")}
    low = est.estimate(alternating)
    _check(low, alternating)
    assert low.rho_hat < -0.5 and low.rho == 0.0
    assert all(d.f == pytest.approx(1.0) for d in low.devices)
    # Values that rise each week give a large correlation, so rho is 0.9.
    rising = {d: _weeks([float(i) for i in range(8)]) for d in ("a", "b", "c")}
    high = est.estimate(rising)
    _check(high, rising)
    assert high.rho == 0.9
    assert all(d.f > 1 for d in high.devices)


def test_the_correction_factor_and_the_interval_for_a_new_week() -> None:
    assert est.correction(0.0, 8) == pytest.approx(1.0)
    k = np.arange(1, 6)
    g = ((1 - k / 6) * 0.3**k).sum()
    assert est.correction(0.3, 6) == pytest.approx((1 + 2 * g) / (1 - 2 * g / 5))
    table = {"ibm_fez": _weeks([1.0, 2.0, 4.0, 3.0, 5.0])}

    (device,) = est.estimate(table).devices

    half = _t_quantile(0.975, 4) * np.std([1, 2, 4, 3, 5], ddof=1) * np.sqrt(1 + 1 / 5)
    assert device.m == pytest.approx(3.0)
    assert (device.low, device.high) == pytest.approx((3.0 - half, 3.0 + half))


def test_the_student_t_functions_match_scipy() -> None:
    stats = pytest.importorskip("scipy.stats")
    for df in (1, 2, 5, 21, 200):
        for t in (0.0, 0.3, 1.7, 4.3, 12.0):
            assert est.t_sf(t, df) == pytest.approx(stats.t.sf(t, df), abs=1e-12)
        for p in (0.025, 0.5, 0.975, 0.995):
            assert est.t_quantile(p, df) == pytest.approx(stats.t.ppf(p, df), abs=1e-9)


def test_holm_and_the_sign_test() -> None:
    assert est.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert est.holm([0.002, 1.0, 1.0, 1.0, 1.0]) == pytest.approx([0.01, 1.0, 1.0, 1.0, 1.0])
    assert est.sign_test(6, 0) == pytest.approx(2 / 64)


def test_the_t_test_keeps_its_level_for_independent_device_weeks() -> None:
    """3 devices, 8 weeks, independent normal values with mean 0. The fraction of studies with p
    below 0.05 must be near 0.05. The correction makes the test a little careful when the weeks
    are independent, so the bounds are loose."""
    rng = np.random.default_rng(20261116)
    studies = 2000
    rejected = plain = 0
    for _ in range(studies):
        table = {d: _weeks(list(rng.standard_normal(8))) for d in ("a", "b", "c")}
        found = est.estimate(table)
        rejected += found.corrected.p_value < 0.05
        plain += found.plain.p_value < 0.05
    assert 0.02 < rejected / studies < 0.07
    assert 0.03 < plain / studies < 0.075
