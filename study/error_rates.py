"""The error rates of the test of the plan in simulated studies with no true difference.

    python -m study.error_rates [--studies 200000] [--seed 11]

One simulated study has 3 devices and 8 weeks. The value Y of a device in a week is noise: an
AR(1) series in its steady state, with the correlation ``rho`` between neighbouring weeks. The
devices are independent. The spread of Y is 1 for each device, or 1, 2 and 3.

Two rules give a p-value for one comparison:

- ``plan``: :func:`study.estimate.estimate`, the rule of the plan.
- ``careful``: the same estimate with a lower limit of ``CAREFUL_RHO`` for rho, the second
  report of the plan entry of 2026-11-14.

For each rule, the output gives the fraction of studies with p below 0.05 and below 0.01 for
one comparison. It also gives the fraction of sets of five independent comparisons with one or
more confirmations by the Holm rule at 5%. The five comparisons of the study use the same runs,
so they are not independent; five independent comparisons are a simple case to state.

The last part gives the power for independent weeks: the fraction of studies with p below 0.05
when Y has the mean ``mu``, in units of the standard deviation of Y.

The module computes many studies at one time with numpy. :func:`statistics` holds the formulas
of ``study.estimate`` again, and ``tests/test_study_error_rates.py`` checks the two against
each other.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence

import numpy as np

from study.estimate import CAREFUL_RHO, RHO_LIMITS, t_quantile

DEVICES, WEEKS, COMPARISONS = 3, 8, 5
CORRELATIONS = (0.0, 0.2, 0.3, 0.5, 0.7)
SPREADS = {"1, 1, 1": (1.0, 1.0, 1.0), "1, 2, 3": (1.0, 2.0, 3.0)}
MEANS = (0.5, 0.7, 0.85, 1.0, 1.25, 1.5)
RULES = {"plan": RHO_LIMITS[0], "careful": CAREFUL_RHO}


def series(
    rng: np.random.Generator, studies: int, rho: float, spread: Sequence[float], mu: float = 0.0
) -> np.ndarray:
    """Y of ``studies`` studies, with the shape (studies, devices, weeks)."""
    devices = len(spread)
    y = np.empty((studies, devices, WEEKS))
    y[..., 0] = rng.standard_normal((studies, devices))
    fresh = math.sqrt(1 - rho * rho)
    for week in range(1, WEEKS):
        y[..., week] = rho * y[..., week - 1] + fresh * rng.standard_normal((studies, devices))
    return mu + y * np.asarray(spread)[None, :, None]


def statistics(y: np.ndarray, min_rho: float = RHO_LIMITS[0]) -> np.ndarray:
    """The t statistic of the estimate of each study of ``y``. Each device has each week."""
    devices, weeks = y.shape[1], y.shape[2]
    mean = y.mean(axis=2)
    residual = y - mean[..., None]
    square = (residual * residual).sum(axis=2)
    cross = (residual[..., 1:] * residual[..., :-1]).sum(axis=(1, 2))
    rho_hat = cross / square.sum(axis=1)
    rho = np.clip(rho_hat + (1 + 3 * rho_hat) / weeks, min_rho, RHO_LIMITS[1])
    k = np.arange(1, weeks)
    g = ((1 - k / weeks)[None, :] * rho[:, None] ** k[None, :]).sum(axis=1)
    factor = (1 + 2 * g) / (1 - 2 * g / (weeks - 1))
    variance = square / (weeks - 1)
    error = np.sqrt(factor * variance.sum(axis=1) / weeks) / devices
    return mean.mean(axis=1) / error


def rates(t: np.ndarray, df: int) -> tuple[float, float, float]:
    """The fraction with p below 0.05, with p below 0.01, and the fraction of sets of
    ``COMPARISONS`` studies with one or more confirmations by the Holm rule at 5%.

    The Holm rule confirms one or more comparisons of a set exactly when the smallest p-value
    of the set is below 0.05 / COMPARISONS.
    """
    size = np.abs(t)
    at_5, at_1 = t_quantile(0.975, df), t_quantile(0.995, df)
    first = t_quantile(1 - 0.05 / COMPARISONS / 2, df)
    sets = size[: size.size // COMPARISONS * COMPARISONS].reshape(-1, COMPARISONS)
    return (
        float(np.mean(size > at_5)),
        float(np.mean(size > at_1)),
        float(np.mean(sets.max(axis=1) > first)),
    )


def report(studies: int, seed: int) -> str:
    df = DEVICES * (WEEKS - 1)
    lines = [
        f"{studies} studies for each line, {DEVICES} devices, {WEEKS} weeks, seed {seed}",
        "No true difference. Fraction with p below 0.05, with p below 0.01, and of sets of"
        f" {COMPARISONS} independent comparisons with one or more confirmations (Holm, 5%):",
        "spread    rho  rule     p<0.05  p<0.01  Holm",
    ]
    for name, spread in SPREADS.items():
        for rho in CORRELATIONS:
            rng = np.random.default_rng([seed, round(rho * 10), len(name)])
            y = series(rng, studies, rho, spread)
            for rule, min_rho in RULES.items():
                at_5, at_1, family = rates(statistics(y, min_rho), df)
                lines.append(
                    f"{name:<8}  {rho:.1f}  {rule:<7}  {100 * at_5:5.1f}%  {100 * at_1:5.1f}%"
                    f"  {100 * family:4.1f}%"
                )
    lines += [
        "",
        "Independent weeks, spread 1, 1, 1. Fraction with p below 0.05 for a mean mu of Y:",
        "mu    plan    careful",
    ]
    for mu in MEANS:
        rng = np.random.default_rng([seed + 2, round(mu * 100)])
        y = series(rng, studies // 4, 0.0, SPREADS["1, 1, 1"], mu)
        found = [rates(statistics(y, min_rho), df)[0] for min_rho in RULES.values()]
        lines.append(f"{mu:<4}  {100 * found[0]:5.1f}%  {100 * found[1]:5.1f}%")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--studies", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)
    print(report(args.studies, args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
