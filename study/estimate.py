"""The estimate over device-weeks of the plan entry "the unit of the primary analysis".

A device-week is the set of runs of one device in one ISO week. Y is one value for each
complete device-week, for example the mean of delta over its two runs. :func:`estimate` takes
the values Y of each device, keyed by the Monday of the ISO week, and gives:

1. For each device with n of 4 or more weeks: m, the mean of Y, and s^2, its sample variance.
2. The estimate: the mean of m over these G devices.
3. rho: the pooled correlation of Y - m between weeks 7 days apart, plus (1 + 3 rho) / n with
   the mean n of the devices, limited to the range from 0 to 0.9.
4. For each device: g = sum for k from 1 to n - 1 of (1 - k / n) rho^k, and
   f = (1 + 2 g) / (1 - 2 g / (n - 1)).
5. The standard error: the square root of the sum over the devices of f s^2 / n, divided by G.
6. The two-sided t-test with the sum of n - 1 over the devices as degrees of freedom, and the
   95% interval.

The same estimate with f = 1, the interval for Y of a new week of each device, and the sign
test of Y are descriptions. A device with fewer than 4 weeks is not in the estimate. With no
such device, the result has no estimate and no test.

The module uses only the standard library, so that the Student t distribution has no extra
dependency.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from math import comb
from typing import Any

MIN_WEEKS = 4
RHO_LIMITS = (0.0, 0.9)
LEVEL = 0.05
CONFIDENCE = 0.95
SMALL = 0.001
WEEK = timedelta(days=7)


@dataclass(frozen=True)
class Test:
    """The standard error, the t-test and the 95% interval of one estimate."""

    standard_error: float
    t: float | None
    p_value: float | None
    low: float | None
    high: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "standard_error": self.standard_error,
            "t": self.t,
            "p_value": self.p_value,
            "interval": None if self.low is None else [self.low, self.high],
        }


@dataclass(frozen=True)
class Device:
    """The values of one device in the estimate.

    ``low`` and ``high`` give the interval in which Y of a new week of the device is expected:
    m +/- t(0.975, n - 1) s sqrt(1 + 1/n).
    """

    name: str
    n: int
    m: float
    s: float
    f: float
    low: float
    high: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.name,
            "n": self.n,
            "m": self.m,
            "s": self.s,
            "f": self.f,
            "new_week": [self.low, self.high],
        }


@dataclass(frozen=True)
class Estimate:
    """The result of :func:`estimate`.

    ``estimate`` is None when no device has ``MIN_WEEKS`` weeks. ``corrected`` is the test with
    the factor f, and ``plain`` the test with f = 1. ``excluded`` gives the number of weeks of
    each device that is not in the estimate. ``incomplete`` gives the number of device-weeks
    that are not complete, for each reason. ``sign`` gives the device-weeks of the devices in
    the estimate with Y above 0 and below 0, and the two-sided p-value of the sign test.
    """

    estimate: float | None
    df: int
    rho_hat: float | None
    rho: float | None
    corrected: Test | None
    plain: Test | None
    devices: tuple[Device, ...]
    excluded: Mapping[str, int]
    device_weeks: int
    incomplete: Mapping[str, int] = field(default_factory=dict)
    sign: tuple[int, int, float] = (0, 0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate,
            "df": self.df,
            "rho_hat": self.rho_hat,
            "rho": self.rho,
            "test": None if self.corrected is None else self.corrected.to_dict(),
            "test_with_f_1": None if self.plain is None else self.plain.to_dict(),
            "devices": [d.to_dict() for d in self.devices],
            "devices_with_too_few_weeks": dict(self.excluded),
            "device_weeks": self.device_weeks,
            "incomplete": dict(self.incomplete),
            "sign_test": {
                "above_0": self.sign[0],
                "below_0": self.sign[1],
                "p_value": self.sign[2],
            },
        }


def week_of(day: date) -> date:
    """The Monday of the ISO week of ``day``."""
    return day - timedelta(days=day.weekday())


def correction(rho: float, n: int) -> float:
    """f for a device with n weeks: (1 + 2 g) / (1 - 2 g / (n - 1)), with g the sum for k from
    1 to n - 1 of (1 - k / n) rho^k."""
    g = sum((1 - k / n) * rho**k for k in range(1, n))
    return (1 + 2 * g) / (1 - 2 * g / (n - 1))


def estimate(
    values: Mapping[str, Mapping[date, float]],
    incomplete: Mapping[str, int] | None = None,
) -> Estimate:
    """The estimate of the plan over the values Y of each device, keyed by the week.

    Weeks 7 days apart are consecutive ISO weeks of one device. Each value must be finite.
    """
    incomplete = dict(incomplete or {})
    excluded = {d: len(w) for d, w in sorted(values.items()) if 0 < len(w) < MIN_WEEKS}
    kept = {d: dict(sorted(w.items())) for d, w in sorted(values.items()) if len(w) >= MIN_WEEKS}
    if not kept:
        return Estimate(None, 0, None, None, None, None, (), excluded, 0, incomplete)
    means = {d: sum(w.values()) / len(w) for d, w in kept.items()}
    residuals = {d: {k: y - means[d] for k, y in w.items()} for d, w in kept.items()}
    variances = {
        d: sum(r * r for r in res.values()) / (len(res) - 1) for d, res in residuals.items()
    }
    cross = sum(
        r * res[k - WEEK] for res in residuals.values() for k, r in res.items() if k - WEEK in res
    )
    square = sum(r * r for res in residuals.values() for r in res.values())
    rho_hat = cross / square if square > 0 else 0.0
    n_bar = sum(len(w) for w in kept.values()) / len(kept)
    rho = min(RHO_LIMITS[1], max(RHO_LIMITS[0], rho_hat + (1 + 3 * rho_hat) / n_bar))
    factors = {d: correction(rho, len(w)) for d, w in kept.items()}
    count = len(kept)
    value = sum(means.values()) / count
    df = sum(len(w) - 1 for w in kept.values())

    def test(f: Mapping[str, float]) -> Test:
        error = math.sqrt(sum(f[d] * variances[d] / len(w) for d, w in kept.items())) / count
        if error <= 0:
            return Test(0.0, None, None, None, None)
        t = value / error
        half = t_quantile((1 + CONFIDENCE) / 2, df) * error
        return Test(error, t, 2 * t_sf(abs(t), df), value - half, value + half)

    devices = []
    for d, w in kept.items():
        n, s = len(w), math.sqrt(variances[d])
        half = t_quantile((1 + CONFIDENCE) / 2, n - 1) * s * math.sqrt(1 + 1 / n)
        devices.append(Device(d, n, means[d], s, factors[d], means[d] - half, means[d] + half))
    ys = [y for w in kept.values() for y in w.values()]
    above, below = sum(y > 0 for y in ys), sum(y < 0 for y in ys)
    return Estimate(
        estimate=value,
        df=df,
        rho_hat=rho_hat,
        rho=rho,
        corrected=test(factors),
        plain=test(dict.fromkeys(kept, 1.0)),
        devices=tuple(devices),
        excluded=excluded,
        device_weeks=len(ys),
        incomplete=incomplete,
        sign=(above, below, sign_test(above, below)),
    )


def sign_test(first: int, second: int) -> float:
    """The two-sided p-value of ``first`` wins against ``second`` wins under equal chances."""
    n = first + second
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(first, second) + 1)) / 2**n
    return min(1.0, 2 * tail)


def holm(p_values: Sequence[float]) -> list[float]:
    """The Holm-adjusted p-values, in the order of ``p_values``."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    largest = 0.0
    for rank, i in enumerate(order):
        largest = max(largest, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = largest
    return adjusted


def t_sf(t: float, df: float) -> float:
    """P(T > t) for the Student t distribution with ``df`` degrees of freedom."""
    tail = 0.5 * _beta_regularized(df / 2, 0.5, df / (df + t * t))
    return tail if t >= 0 else 1 - tail


def t_quantile(p: float, df: float) -> float:
    """The value t with P(T <= t) = p, for 0 < p < 1."""
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -t_quantile(1 - p, df)
    low, high = 0.0, 1.0
    while t_sf(high, df) > 1 - p:
        high *= 2
    for _ in range(200):
        middle = (low + high) / 2
        if t_sf(middle, df) > 1 - p:
            low = middle
        else:
            high = middle
        if high - low < 1e-12 * max(1.0, high):
            break
    return (low + high) / 2


def _beta_regularized(a: float, b: float, x: float) -> float:
    """The regularized incomplete beta function I_x(a, b), by its continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1) / (a + b + 2):
        return math.exp(log_front) * _fraction(a, b, x) / a
    return 1 - math.exp(log_front) * _fraction(b, a, 1 - x) / b


def _fraction(a: float, b: float, x: float) -> float:
    """The continued fraction of the incomplete beta function (modified Lentz method)."""
    tiny = 1e-300
    c, d = 1.0, 1 - (a + b) * x / (a + 1)
    d = 1 / (d if abs(d) > tiny else tiny)
    result = d
    for m in range(1, 1000):
        for numerator in (
            m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1)),
        ):
            d = 1 + numerator * d
            d = 1 / (d if abs(d) > tiny else tiny)
            c = 1 + numerator / c
            c = c if abs(c) > tiny else tiny
            result *= c * d
        if abs(c * d - 1) < 1e-15:
            break
    return result
