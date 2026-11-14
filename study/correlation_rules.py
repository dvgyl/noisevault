"""Five rules for the correlation of neighbouring weeks, compared in simulated studies.

    python -m study.correlation_rules [--studies 400000] [--seed 11]

One simulated study has 3 devices and 8 weeks, as in ``study.error_rates``, and it uses
:func:`study.error_rates.series`. The value Y of a device in a week is mu plus noise: an AR(1)
series in its steady state, with the variance 1 and the true correlation ``rho_true`` between
neighbouring weeks. The devices are independent.

The measured correlation is the value of the plan:

    rho' = (sum over the devices and the pairs of neighbouring weeks of r_w r_(w-1))
           / (sum over all weeks of r_w^2),

with r the difference between Y and the mean of its device. A rule changes rho' into the rho of
the factor F of each device. The n in the rules is the number of weeks, 8:

    none      rho = 0, so F = 1 (no correction)
    plan      rho = rho' + (1 + 3 rho') / n, the rule of the plan
    four      rho = rho' + (1 + 4 rho') / n, the first-order bias of rho' for one device
    inverse   rho = (rho' + 1 / n) / (1 - 4 / n), the solution of rho' = rho - (1 + 4 rho) / n
    inverse3  rho = (rho' + 1 / n) / (1 - 3 / n), the solution of rho' = rho - (1 + 3 rho) / n

For D devices in one sum, the first-order bias of rho' is (1 + (2 + 2 / D) rho) / n. The value
is (1 + 4 rho) / n for one device, and (1 + 8 rho / 3) / n for the three devices of the study.
The rules ``four`` and ``inverse`` are thus candidates, not exact corrections.

Each rule limits rho to the range of ``study.estimate.RHO_LIMITS``, 0 to 0.9. The rest is the
estimate of the plan: F, the standard error and the two-sided t-test with 21 degrees of freedom.
``tests/test_study_correlation_rules.py`` checks the rule ``plan`` and the rule ``none``
against :func:`study.estimate.estimate`.

The output has four parts:

1. For each rho_true: the mean of rho' over the studies, and the two first-order formulas
   rho - (1 + 3 rho) / n and rho - (1 + 4 rho) / n.
2. No true difference: for each rho_true and each rule, the fraction of studies with p below
   0.05 and below 0.01.
3. Independent weeks: the standard error of each rule divided by the standard error of the rule
   ``none``, and the fraction of studies in which the rule gives rho = 0.
4. Independent weeks: the fraction of studies with p below 0.05 and below 0.01 for a true mean
   mu of Y, in units of the standard deviation of Y.

Parts 1 and 2 use the same studies for one rho_true. Each part takes its random numbers from the
seed and from the value of rho_true or mu, so a line does not depend on the other lines.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np

from study.error_rates import DEVICES, WEEKS, measured, series, standard_error
from study.estimate import RHO_LIMITS, t_quantile

CORRELATIONS = (0.0, 0.2, 0.3, 0.5, 0.7)
MEANS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
LEVELS = (0.05, 0.01)
SPREAD = (1.0,) * DEVICES
INDEPENDENT_STREAM = 99
MEAN_STREAM = 50
RULES: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "none": lambda r, n: np.zeros_like(r),
    "plan": lambda r, n: r + (1 + 3 * r) / n,
    "four": lambda r, n: r + (1 + 4 * r) / n,
    "inverse": lambda r, n: (r + 1 / n) / (1 - 4 / n),
    "inverse3": lambda r, n: (r + 1 / n) / (1 - 3 / n),
}


class Corrected(NamedTuple):
    """The result of one rule for each study."""

    rho: np.ndarray
    standard_error: np.ndarray
    t: np.ndarray


def corrections(y: np.ndarray) -> tuple[np.ndarray, dict[str, Corrected]]:
    """The measured correlation rho' of each study of ``y``, and the result of each rule."""
    weeks = y.shape[2]
    mean, rho_hat, variance = measured(y)
    found = {}
    for name, rule in RULES.items():
        rho = np.clip(rule(rho_hat, weeks), *RHO_LIMITS)
        error = standard_error(variance, rho, weeks)
        found[name] = Corrected(rho, error, mean / error)
    return rho_hat, found


def below(t: np.ndarray, level: float, df: int) -> float:
    """The fraction of the studies with a two-sided p-value below ``level``."""
    return float(np.mean(np.abs(t) > t_quantile(1 - level / 2, df)))


def _tenths(value: float) -> int:
    return round(value * 10)


def _cells(found: dict[str, Corrected], df: int) -> str:
    cells = [
        f"{100 * below(c.t, LEVELS[0], df):5.1f}% ({100 * below(c.t, LEVELS[1], df):4.1f}%)"
        for c in found.values()
    ]
    return "  ".join(f"{cell:>15s}" for cell in cells)


def report(studies: int, seed: int) -> str:
    """The four parts as text, with ``studies`` studies for each line."""
    df = DEVICES * (WEEKS - 1)
    header = "  ".join(f"{name:>15s}" for name in RULES)
    lines = [f"{studies} studies for each line, {DEVICES} devices, {WEEKS} weeks, seed {seed}"]
    part_1, part_2 = [], []
    for rho in CORRELATIONS:
        rho_hat, found = corrections(
            series(np.random.default_rng([seed, _tenths(rho)]), studies, rho, SPREAD)
        )
        part_1.append(
            f"  {rho:.1f}     {rho_hat.mean():+.4f}    {rho - (1 + 3 * rho) / WEEKS:+.4f}"
            f"         {rho - (1 + 4 * rho) / WEEKS:+.4f}"
        )
        part_2.append(f"  {rho:.1f}     " + _cells(found, df))
    lines += [
        "",
        "1. The measured correlation rho' and two first-order formulas for its mean",
        "rho_true  mean rho'  rho-(1+3rho)/n  rho-(1+4rho)/n",
        *part_1,
        "",
        "2. No true difference: fraction of studies with p below 0.05 (and below 0.01)",
        "rho_true  " + header,
        *part_2,
        "",
        "3. Independent weeks: standard error of a rule divided by the standard error with no"
        " correction, and the fraction of studies with rho = 0",
    ]
    _, found = corrections(
        series(np.random.default_rng([seed, INDEPENDENT_STREAM]), studies, 0.0, SPREAD)
    )
    for name, corrected in found.items():
        ratio = corrected.standard_error / found["none"].standard_error
        lines.append(
            f"  {name:8s} mean ratio {ratio.mean():.3f}, median {np.median(ratio):.3f},"
            f" 95th percentile {np.percentile(ratio, 95):.3f};"
            f" rho = 0 in {100 * np.mean(corrected.rho == 0):.1f}%"
        )
    lines += [
        "",
        "4. Independent weeks: fraction of studies with p below 0.05 (and below 0.01) for a true"
        " mean mu, in units of the standard deviation of Y",
        "  mu    " + header,
    ]
    for mu in MEANS:
        rng = np.random.default_rng([seed, MEAN_STREAM + _tenths(mu)])
        y = series(rng, studies, 0.0, SPREAD, mu)
        lines.append(f"  {mu:.1f}   " + _cells(corrections(y)[1], df))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--studies", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)
    print(report(args.studies, args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
