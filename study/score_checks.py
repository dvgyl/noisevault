"""Three checks of the paired test of two models in ``noisevault.score``.

    python -m study.score_checks hidden [--jobs 4000] [--shots 2000] [--seed 11]
    python -m study.score_checks safe
    python -m study.score_checks chain PROFILE --chain 142,143,144,145 [--fresh 202646]
        [--shots 1250] [--jobs 2000]

Each check writes text to standard output.

``hidden``: a constructed case for the weak point of step 1 of the test. Step 1 takes the rare
results of a circuit together as one result. A result is rare when no shot showed it and each
model expects it less than one time in the shots. The case hides one rare result with a very
large score difference among other rare results. One circuit has ``HIDDEN_OTHERS + 2`` results:

- result 0, the usual result;
- result R: model a gives it ``A_RARE`` and model b gives it ``B_RARE``;
- ``HIDDEN_OTHERS`` other results: the device and the two models give each one ``EACH``.

The device gives R the probability for which the two models have the same expected log score,
so each verdict of the test is false. The check runs :func:`noisevault.score.rule_test` with
steps 1 and 2 (the test of ``paired``) and with step 2 only. Step 2 is the variance part for the
results that no shot showed. It also runs the natural test with no step:
:func:`natural_verdict`.

``safe``: the smallest mean score difference that a test with a guarantee can show. Model a
gives a wrong readout of a qubit the probability ``A_WRONG``, and model b gives it ``B_WRONG``.
A circuit reads ``SAFE_QUBITS`` qubits, so the score difference of one shot is in a range of the
width R. The bound of Hoeffding for the mean of N independent values in this range: the mean is
more than t from its expected value with a probability of at most 2 exp(-2 N t^2 / R^2). A
two-sided test at the level ``LEVEL`` that uses only this bound gives a verdict when the mean
score difference is above t = R sqrt(ln(2 / LEVEL) / (2 N)).

``chain``: the level and the power of the test of two models on one job, for the circuits of
the study on one chain. The circuits are the deep circuit set of the profile for the chain and
the random stream ``--fresh``. The check keeps each circuit on the qubits of the chain that is
not a probe circuit (the group ``all4`` of the plan). The simulated device has ``TRUTH`` times
the stated gate errors, through the gate factor of ``unmodeled_error``. Four wrong models:

- ``below``: ``BELOW`` times the stated gate errors.
- ``above``: the factor above ``TRUTH`` that gives ``below`` and ``above`` the same expected log
  score on this device. A test between them has no right verdict.
- ``near``: ``NEAR`` times, and ``plain``: the profile as stated. The check takes the profile
  with no ``unmodeled_error``.

The level is the fraction of simulated jobs with a verdict between ``below`` and ``above``. The
power is the fraction with a verdict for the truth against ``near``, and against ``plain``. Job
``i`` has the seed ``i``.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np

import noisevault as nv
from noisevault.counts import PlannedCircuit, simulate
from noisevault.profile import Profile
from noisevault.reference import probabilities
from noisevault.score import LEVEL, Choice, paired, rule_test, score
from noisevault.suite import role, suite

HIDDEN_SHOTS, HIDDEN_JOBS, HIDDEN_SEED, HIDDEN_OTHERS = 2000, 4000, 11, 50
A_RARE, B_RARE, EACH = 0.0004, 1e-12, 0.0002
HIDDEN_FORMS = ("steps 1 and 2", "step 2 only", "no step")

A_WRONG, B_WRONG, SAFE_QUBITS = 0.01, 0.00012, 4
SAFE_SHOTS = (
    (1250, "one circuit of a job"),
    (36250, "29 circuits of 1250 shots, as in one job of the study"),
)

CHAIN_SHOTS, CHAIN_JOBS = 1250, 2000
TRUTH, BELOW, NEAR = 1.5, 1.2, 1.4
FACTOR_LIMIT, FACTOR_TOLERANCE = 20.0, 1e-9


def hidden_model(rare: float) -> np.ndarray:
    """The result probabilities of the hidden case when result R has the probability ``rare``."""
    rest = HIDDEN_OTHERS * EACH
    return np.array([1 - rare - rest, rare, *([EACH] * HIDDEN_OTHERS)])


def hidden_truth() -> np.ndarray:
    """The device of the hidden case: the probability of R that gives the two models the same
    expected log score, found by bisection."""
    difference = np.log(hidden_model(A_RARE)) - np.log(hidden_model(B_RARE))
    low, high = 0.0, A_RARE
    while high - low > 1e-18:
        middle = (low + high) / 2
        if float(hidden_model(middle) @ difference) < 0:
            low = middle
        else:
            high = middle
    return hidden_model((low + high) / 2)


def natural_verdict(counts: np.ndarray, difference: np.ndarray) -> Choice:
    """The verdict of the test with no step: the variance comes from the measured frequencies
    only, with no part for the results that no shot showed."""
    shots = int(counts.sum())
    seen = counts > 0
    frequency = counts[seen] / shots
    d = difference[seen]
    mean = float(frequency @ d)
    variance = shots * float(frequency @ (d - mean) ** 2)
    if variance == 0:
        return "neither"
    z = float(counts[seen] @ d) / math.sqrt(variance)
    if math.erfc(abs(z) / math.sqrt(2)) >= LEVEL:
        return "neither"
    return "a" if z > 0 else "b"


def hidden_check(
    jobs: int = HIDDEN_JOBS, shots: int = HIDDEN_SHOTS, seed: int = HIDDEN_SEED
) -> dict[str, Any]:
    """The verdicts of the three forms of the test on ``jobs`` simulated jobs of the hidden case.

    ``verdicts`` gives, for each form, the fraction of jobs with a verdict for a and for b.
    """
    a, b, truth = hidden_model(A_RARE), hidden_model(B_RARE), hidden_truth()
    difference = np.log(a) - np.log(b)
    found: dict[str, list[Choice]] = {form: [] for form in HIDDEN_FORMS}
    occurred = 0
    rng = np.random.default_rng(seed)
    for _ in range(jobs):
        counts = rng.multinomial(shots, truth)
        occurred += int(counts[1] > 0)
        for form, group_rare in zip(HIDDEN_FORMS[:2], (True, False), strict=True):
            test = rule_test("log", [(counts, a, b)], np.log, group_rare=group_rare)
            found[form].append(test.better)
        found["no step"].append(natural_verdict(counts, difference))
    return {
        "jobs": jobs,
        "shots": shots,
        "seed": seed,
        "rare_truth": float(truth[1]),
        "difference_rare": float(difference[1]),
        "difference_usual": float(difference[0]),
        "expected_difference": float(truth @ difference),
        "occurred": occurred / jobs,
        "verdicts": {
            form: {"a": choices.count("a") / jobs, "b": choices.count("b") / jobs}
            for form, choices in found.items()
        },
    }


def shot_range(
    a_wrong: float = A_WRONG, b_wrong: float = B_WRONG, qubits: int = SAFE_QUBITS
) -> tuple[float, float]:
    """The lowest and the highest log score difference of one shot: each qubit read right, and
    each qubit read wrong."""
    low = qubits * math.log((1 - a_wrong) / (1 - b_wrong))
    high = qubits * math.log(a_wrong / b_wrong)
    return low, high


def smallest_difference(width: float, shots: int, level: float = LEVEL) -> float:
    """The smallest mean score difference with a verdict of the test from the bound of
    Hoeffding, for ``shots`` values in a range of ``width``."""
    return width * math.sqrt(math.log(2 / level) / (2 * shots))


def chain_circuits(
    profile: Profile, chain: Sequence[int], fresh: int | None = None
) -> list[PlannedCircuit]:
    """The circuits of the deep circuit set on the qubits of ``chain``, with no probe circuit."""
    wanted = set(chain)
    return [
        c
        for c in suite(profile, tuple(chain), fresh=fresh)
        if set(c.qubits) <= wanted and role(c.name) != "probe"
    ]


def gate_factor(profile: Profile, factor: float) -> Profile:
    """``profile`` with the gate factor ``factor`` in ``unmodeled_error``."""
    return profile.model_copy(update={"unmodeled_error": {"gates": {"factor": factor}}})


def outcome_probabilities(profile: Profile, circuits: Sequence[PlannedCircuit]) -> list[np.ndarray]:
    return [
        probabilities(profile, c.ops, len(c.qubits), layout=c.qubits, unknown_gates="error")
        for c in circuits
    ]


def divergence(truth: Sequence[np.ndarray], model: Sequence[np.ndarray]) -> float:
    """The expected loss of log score of ``model`` against ``truth``, summed over the circuits.

    A result that the truth gives the probability 0 adds nothing. The loss is infinite when the
    model gives the probability 0 to a result that the truth can give.
    """
    total = 0.0
    for t, m in zip(truth, model, strict=True):
        possible = t > 0
        if np.any(m[possible] <= 0):
            return math.inf
        total += float(t[possible] @ np.log(t[possible] / m[possible]))
    return total


def matched_factor(profile: Profile, circuits: Sequence[PlannedCircuit]) -> tuple[float, float]:
    """The gate factor above ``TRUTH`` with the expected loss of log score of the factor
    ``BELOW``, found by bisection, and that loss.

    Raises ValueError when no factor from ``TRUTH`` to ``FACTOR_LIMIT`` has that loss. The
    check has no level then.
    """
    exact = outcome_probabilities(gate_factor(profile, TRUTH), circuits)

    def loss(factor: float) -> float:
        return divergence(exact, outcome_probabilities(gate_factor(profile, factor), circuits))

    target = loss(BELOW)
    low, high = TRUTH, FACTOR_LIMIT
    if not (math.isfinite(target) and 0 < target <= loss(high)):
        raise ValueError(
            f"no gate factor from {TRUTH} to {FACTOR_LIMIT} has the expected log score of the"
            f" factor {BELOW} on these circuits"
        )
    while high - low > FACTOR_TOLERANCE:
        middle = (low + high) / 2
        if loss(middle) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2, target


def chain_check(
    profile: Profile,
    chain: Sequence[int],
    *,
    fresh: int | None = None,
    shots: int = CHAIN_SHOTS,
    jobs: int = CHAIN_JOBS,
) -> dict[str, Any]:
    """The level and the power of the paired test on ``jobs`` simulated jobs of the chain."""
    profile = profile.uncorrected()
    circuits = chain_circuits(profile, chain, fresh)
    above, target = matched_factor(profile, circuits)
    models = {
        "truth": gate_factor(profile, TRUTH),
        "below": gate_factor(profile, BELOW),
        "above": gate_factor(profile, above),
        "near": gate_factor(profile, NEAR),
        "plain": profile,
    }
    exact = outcome_probabilities(models["truth"], circuits)
    level = near = wrong_near = plain = 0
    for seed in range(jobs):
        counts = simulate(
            models["truth"], circuits, shots=shots, seed=seed, run_at=profile.device.calibrated_at
        )
        found = {name: score(model, counts) for name, model in models.items()}
        level += paired(found["below"], found["above"]).log.better != "neither"
        verdict = paired(found["truth"], found["near"]).log.better
        near += verdict == "a"
        wrong_near += verdict == "b"
        plain += paired(found["truth"], found["plain"]).log.better == "a"
    return {
        "circuits": len(circuits),
        "chain": list(chain),
        "shots": shots,
        "jobs": jobs,
        "above": above,
        "loss_below": target,
        "loss_above": divergence(exact, outcome_probabilities(models["above"], circuits)),
        "level": level / jobs,
        "power_near": near / jobs,
        "wrong_near": wrong_near / jobs,
        "power_plain": plain / jobs,
    }


def hidden_report(jobs: int, shots: int, seed: int) -> str:
    found = hidden_check(jobs, shots, seed)
    lines = [
        f"noisevault {nv.__version__}; {shots} shots, {jobs} jobs, seed {seed}",
        f"probability of R: device {found['rare_truth']:.4e}, model a {A_RARE}, model b {B_RARE}",
        f"score difference of a shot with R: {found['difference_rare']:.2f};"
        f" of a shot with result 0: {found['difference_usual']:.6f}",
        f"expected score difference for each shot: {found['expected_difference']:.2e}",
        f"jobs in which R occurred: {found['occurred']:.4f}",
    ]
    for form, shares in found["verdicts"].items():
        lines.append(
            f"{form}: verdict for a {shares['a']:.4f}, verdict for b {shares['b']:.4f},"
            f" all verdicts {shares['a'] + shares['b']:.4f}"
        )
    return "\n".join(lines)


def safe_report() -> str:
    low, high = shot_range()
    width = high - low
    lines = [f"score difference of one shot: from {low:.4f} to {high:.2f}; width {width:.2f}"]
    for shots, what in SAFE_SHOTS:
        lines.append(
            f"{shots} shots ({what}): smallest mean score difference with a verdict"
            f" {smallest_difference(width, shots):.3f}"
        )
    return "\n".join(lines)


def chain_report(
    profile: Profile, chain: Sequence[int], *, fresh: int | None, shots: int, jobs: int
) -> str:
    found = chain_check(profile, chain, fresh=fresh, shots=shots, jobs=jobs)
    return "\n".join(
        [
            f"profile {profile.id} {profile.short_fingerprint}; noisevault {nv.__version__}",
            f"{found['circuits']} circuits on qubits {found['chain']}, {shots} shots for each,"
            f" {jobs} jobs, random stream {fresh}",
            f"gate factors: truth {TRUTH}, below {BELOW}, above {found['above']:.4f},"
            f" near {NEAR}, plain 1.0",
            "expected loss of log score for each circuit, summed:"
            f" below {found['loss_below']:.6f}, above {found['loss_above']:.6f}",
            f"level (verdict between below and above): {found['level']:.4f}",
            f"power, truth against near ({NEAR}): {found['power_near']:.4f};"
            f" verdict for the wrong model: {found['wrong_near']:.4f}",
            f"power, truth against plain (1.0): {found['power_plain']:.4f}",
        ]
    )


def report(args: argparse.Namespace) -> str:
    """The text of the check that ``args.check`` names."""
    if args.check == "hidden":
        return hidden_report(args.jobs, args.shots, args.seed)
    if args.check == "safe":
        return safe_report()
    chain = [int(q) for q in args.chain.split(",")]
    return chain_report(
        nv.load(args.profile), chain, fresh=args.fresh, shots=args.shots, jobs=args.jobs
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    checks = parser.add_subparsers(dest="check", required=True)
    hidden = checks.add_parser("hidden", help="a rare result that hides among rare results")
    hidden.add_argument("--jobs", type=int, default=HIDDEN_JOBS)
    hidden.add_argument("--shots", type=int, default=HIDDEN_SHOTS)
    hidden.add_argument("--seed", type=int, default=HIDDEN_SEED)
    checks.add_parser("safe", help="the smallest difference of a test from the Hoeffding bound")
    chain = checks.add_parser("chain", help="the level and the power on the circuits of a chain")
    chain.add_argument("profile", help="a profile reference, such as ibm_fez@2026-11-12, or file")
    chain.add_argument("--chain", required=True, help="the qubits of the chain, such as 1,2,3,4")
    chain.add_argument("--fresh", type=int, default=None, help="the random stream of the week")
    chain.add_argument("--shots", type=int, default=CHAIN_SHOTS)
    chain.add_argument("--jobs", type=int, default=CHAIN_JOBS)
    try:
        print(report(parser.parse_args(argv)))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
