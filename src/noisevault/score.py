"""Score a profile as a forecast of measured counts, with no fit.

A noise model gives a probability to each outcome of a circuit. A proper scoring rule rewards
the model that gives the measured outcomes the most probability. :func:`score` applies the log
rule and the quadratic rule to one profile. :func:`paired` tests which of two profiles scores
better on the same counts.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .compare import _duration, _wrap
from .errors import CountsError, NoiseVaultError, did_you_mean, joined, plural, qubit_loci
from .profile import iso_z, ref_on_day
from .reference import probabilities

if TYPE_CHECKING:
    from .counts import MeasuredCircuit, MeasuredCounts
    from .profile import Profile

DRAWS = 200
LEVEL = 0.05
_WIDTH = 80
_DIGITS = 5
_DROPPABLE = ("fingerprint", "noise 95%", "age")

Rule = Literal["log", "quadratic"]
Choice = Literal["a", "b", "neither"]
Outcome = Literal["tested", "identical", "ruled_out", "both_ruled_out", "unbounded"]


@dataclass(frozen=True)
class CircuitForecast:
    """The scores of one circuit, as means over its shots.

    ``log_score`` is -inf and ``excess`` is inf when ``ruled_out`` shots have probability 0.
    ``probabilities`` holds the model's outcome probabilities, big-endian, as in the reference
    model.
    """

    name: str
    qubits: tuple[int, ...]
    shots: int
    log_score: float
    quadratic_score: float
    ruled_out: int
    excess: float
    probabilities: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True)
class Forecast:
    """The scores of ``profile`` on ``counts``. Each total is a mean over all shots.

    ``p_value`` compares ``excess`` with the excess of counts drawn from the profile itself. It
    is 0 when the profile rules out a shot.
    """

    profile: Profile
    counts: MeasuredCounts = field(repr=False)
    circuits: tuple[CircuitForecast, ...]
    shots: int
    log_score: float
    quadratic_score: float
    ruled_out: int
    excess: float
    excess_noise_mean: float
    excess_noise_95: float
    p_value: float

    @property
    def same_calibration(self) -> bool:
        """True when the counts were planned from the calibration of the profile."""
        return self.profile.calibration_fingerprint == self.counts.profile.fingerprint

    @property
    def same_device(self) -> bool:
        return self.profile.device.name == self.counts.backend

    @property
    def calibration_age(self) -> timedelta | None:
        """The run time minus the calibration time. It is negative for a later calibration."""
        when = self.profile.device.calibrated_at
        return None if when is None else self.counts.run_at - when

    def summary(self, *, width: int = _WIDTH) -> str:
        """The scores as text, one row for each circuit and one row for the total."""
        profile, counts = self.profile, self.counts
        ref = ref_on_day(profile.id, profile.device.calibrated_at)
        head = [
            f"{ref} {profile.short_fingerprint}, calibration age {age_text(self)}",
            f"counts sha256:{counts.sha256.removeprefix('sha256:')[:12]}, {counts.source},"
            f" run {counts.run_at:%Y-%m-%d %H:%MZ} on {counts.backend}"
            f" {qubit_loci(counts.qubits)}",
        ]
        lines = [part for text in head for part in _wrap(text, min(width, _WIDTH), "  ")]
        rows = [
            (c.name, str(c.shots), _number(c.log_score), _number(c.excess), str(c.ruled_out))
            for c in self.circuits
        ]
        rows.append(
            (
                "total",
                str(self.shots),
                _number(self.log_score),
                _number(self.excess),
                str(self.ruled_out),
            )
        )
        titles = ("circuit", "shots", "log score", "excess", "ruled out")
        lines += ["", *_columns(titles, rows)]
        if self.ruled_out:
            fit = f"ruled out (p = 0), {plural(self.ruled_out, 'shot')} have probability 0"
        else:
            fit = (
                f"excess {_number(self.excess)}, shot noise {_number(self.excess_noise_95)}"
                f" at 95% (p = {self.p_value:.2g})"
            )
        lines += ["", f"quadratic score {_number(self.quadratic_score)}", f"fit {fit}"]
        return "\n".join(lines)

    __str__ = summary

    def to_dict(self) -> dict[str, Any]:
        age = self.calibration_age
        return {
            "profile_id": self.profile.id,
            "fingerprint": self.profile.fingerprint,
            "calibration": self.profile.calibration_fingerprint,
            "same_calibration": self.same_calibration,
            "same_device": self.same_device,
            "counts": self.counts.sha256,
            "run_at": iso_z(self.counts.run_at),
            "calibration_age_hours": None if age is None else age.total_seconds() / 3600,
            "shots": self.shots,
            "log_score": _finite(self.log_score),
            "quadratic_score": self.quadratic_score,
            "ruled_out": self.ruled_out,
            "excess": _finite(self.excess),
            "excess_noise_mean": self.excess_noise_mean,
            "excess_noise_95": self.excess_noise_95,
            "p_value": self.p_value,
            "draws": DRAWS,
            "circuits": [
                {
                    "name": c.name,
                    "qubits": list(c.qubits),
                    "shots": c.shots,
                    "log_score": _finite(c.log_score),
                    "quadratic_score": c.quadratic_score,
                    "ruled_out": c.ruled_out,
                    "excess": _finite(c.excess),
                }
                for c in self.circuits
            ],
        }


@dataclass(frozen=True)
class RuleTest:
    """The paired test of one scoring rule.

    ``difference`` is the mean score of model a minus the mean score of model b, per shot, and
    ``standard_error`` is its standard error. ``outcome`` says why a value is None:
    "identical" (the models give the same scores), "ruled_out" (one model gives a measured
    shot probability 0), "both_ruled_out" or "unbounded" (the log rule only: one model gives
    probability 0 to an outcome that the other model expects, so one more shot can change the
    difference without limit).
    """

    rule: Rule
    outcome: Outcome
    difference: float | None
    standard_error: float | None
    z: float | None
    p_value: float | None
    better: Choice

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "difference": self.difference,
            "standard_error": self.standard_error,
            "z": self.z,
            "p_value": self.p_value,
            "better": self.better,
        }


@dataclass(frozen=True)
class PairedTest:
    """Which of two forecasts of the same counts scores better, at the 5% level.

    The log rule decides. When both models rule out shots, or when the log rule is unbounded,
    the quadratic rule decides. The test needs independent shots and enough shots for a normal
    approximation. It is valid when both models were fixed before the counts were seen, or when
    their parameters were fitted on other shots or other circuits. A model fitted on the same
    counts has an advantage that the test does not correct.
    """

    a: Forecast = field(repr=False)
    b: Forecast = field(repr=False)
    log: RuleTest
    quadratic: RuleTest

    @property
    def better(self) -> Choice:
        if self.log.outcome in ("both_ruled_out", "unbounded"):
            return self.quadratic.better
        return self.log.better

    def summary(self, names: tuple[str, str] | None = None, *, width: int = _WIDTH) -> str:
        """The test as text. ``names`` names model a and model b."""
        a, b = names or _names(self.a.profile, self.b.profile)
        lines = []
        for test in (self.log, self.quadratic):
            text = f"{test.rule} score: {rule_phrase(test, self, a, b)}"
            lines += _wrap(text, min(width, _WIDTH), "  ")
        verdict = {"a": f"{a} is better", "b": f"{b} is better", "neither": "no verdict"}
        lines.append(f"verdict: {verdict[self.better]}")
        return "\n".join(lines)

    __str__ = summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": {"profile_id": self.a.profile.id, "fingerprint": self.a.profile.fingerprint},
            "b": {"profile_id": self.b.profile.id, "fingerprint": self.b.profile.fingerprint},
            "counts": self.a.counts.sha256,
            "level": LEVEL,
            "log": self.log.to_dict(),
            "quadratic": self.quadratic.to_dict(),
            "better": self.better,
        }


def score(
    profile: Profile, counts: MeasuredCounts, *, circuits: Iterable[str] | None = None
) -> Forecast:
    """Score ``profile`` as a forecast of ``counts``, or of only the named ``circuits``.

    The profile can be of any date or device, or written by hand, if the reference simulator
    can run the circuits with it. The profile's ``unmodeled_error`` factors apply. Raises
    CountsError for an unknown circuit name or for a circuit that the profile cannot simulate.
    """
    chosen = _chosen(counts, circuits)
    rng = np.random.default_rng(int(counts.sha256[7:23], 16))
    forecasts, noise = [], np.zeros(DRAWS)
    for c in chosen:
        probs = _probabilities(profile, c)
        measured = c.vector()
        forecasts.append(_circuit(c, probs, measured))
        drawn = rng.multinomial(c.shots, _normalized(probs), size=DRAWS)
        noise += c.shots * _excess(drawn, probs)
    shots = sum(c.shots for c in chosen)
    noise /= shots
    ruled_out = sum(c.ruled_out for c in forecasts)
    excess = _mean([c.excess for c in forecasts], forecasts, shots)
    p_value = 0.0 if ruled_out else float((1 + np.sum(noise >= excess)) / (1 + DRAWS))
    return Forecast(
        profile=profile,
        counts=counts,
        circuits=tuple(forecasts),
        shots=shots,
        log_score=_mean([c.log_score for c in forecasts], forecasts, shots),
        quadratic_score=_mean([c.quadratic_score for c in forecasts], forecasts, shots),
        ruled_out=ruled_out,
        excess=excess,
        excess_noise_mean=float(noise.mean()),
        excess_noise_95=float(np.percentile(noise, 95)),
        p_value=p_value,
    )


def paired(a: Forecast, b: Forecast) -> PairedTest:
    """Test whether forecast ``a`` or forecast ``b`` scores better on their counts.

    For each rule, the test sums the per-shot score differences over all shots and divides the
    sum by its standard error. The two-sided p-value comes from the standard normal
    distribution. The variance of the difference in each circuit is the largest of three
    estimates: from the measured frequencies, from the probabilities of model a and from the
    probabilities of model b. The measured frequencies alone give no variance to an outcome
    that no shot showed. A rare outcome with a large score difference then makes the test
    reject too often.

    The test needs independent shots. It is valid when both models were fixed before the counts
    were seen, or when their parameters were fitted on other shots or other circuits. A model
    fitted on the same counts has an advantage that the test does not correct.

    Raises CountsError when the forecasts score different counts or different circuits.
    """
    if a.counts.sha256 != b.counts.sha256:
        raise CountsError(
            f"the forecasts score different counts, {_digest(a)} and {_digest(b)}",
            hint="score both profiles on the same counts",
        )
    names_a, names_b = [c.name for c in a.circuits], [c.name for c in b.circuits]
    if names_a != names_b:
        raise CountsError(
            f"the forecasts score different circuits, {joined(names_a)} and {joined(names_b)}",
            hint="give both forecasts the same circuits=",
        )
    measured = {c.name: c.vector() for c in a.counts.circuits}
    pairs = [
        (measured[x.name], x.probabilities, y.probabilities)
        for x, y in zip(a.circuits, b.circuits, strict=True)
    ]
    if a.ruled_out and b.ruled_out:
        log = RuleTest("log", "both_ruled_out", None, None, None, None, "neither")
    elif a.ruled_out or b.ruled_out:
        log = RuleTest("log", "ruled_out", None, None, None, None, "b" if a.ruled_out else "a")
    else:
        log = _test("log", pairs, _log)
    quadratic = _test("quadratic", pairs, _quadratic)
    return PairedTest(a=a, b=b, log=log, quadratic=quadratic)


@dataclass(frozen=True)
class Ranking:
    """Named forecasts of the same counts, best first, and the paired test of the best one
    against each other one.

    The order puts models with no ruled-out shot first, by log score. Ruled-out models follow,
    by quadratic score.
    """

    models: tuple[tuple[str, Forecast], ...]
    tests: tuple[PairedTest, ...]

    def summary(self, *, counts_file: str | None = None, width: int = _WIDTH) -> str:
        """The text that ``nv score`` prints."""
        prose = min(width, _WIDTH)
        counts = self.models[0][1].counts
        digest = counts.sha256.removeprefix("sha256:")[:12]
        head = (
            f"{counts.source} counts sha256:{digest}"
            if counts_file is None
            else f"counts {counts_file}, {counts.source}, sha256:{digest}",
            f"run {counts.run_at:%Y-%m-%d %H:%MZ} on {counts.backend}, {qubit_loci(counts.qubits)}",
        )
        lines = [part for text in head for part in _wrap(text, prose, "  ")]
        lines += ["", *self._table(width)]
        best = self.models[0][0]
        for (name, _), test in zip(self.models[1:], self.tests, strict=True):
            lines += ["", f"{name} against {best}:"]
            for rule in (test.log, test.quadratic):
                text = f"{rule.rule} score: {rule_phrase(rule, test, best, name)}"
                lines += ["  " + part for part in _wrap(text, prose - 2, "  ")]
        return "\n".join(lines)

    __str__ = summary

    def to_dict(self) -> dict[str, Any]:
        counts = self.models[0][1].counts
        best = self.models[0][0]
        return {
            "counts": counts.sha256,
            "source": counts.source,
            "backend": counts.backend,
            "run_at": iso_z(counts.run_at),
            "qubits": list(counts.qubits),
            "models": [{"ref": name, **forecast.to_dict()} for name, forecast in self.models],
            "paired": [
                {"best": best, "other": name, **test.to_dict()}
                for (name, _), test in zip(self.models[1:], self.tests, strict=True)
            ],
        }

    def _table(self, width: int) -> list[str]:
        cells: dict[str, Callable[[str, Forecast], str]] = {
            "model": lambda name, f: name,
            "fingerprint": lambda name, f: f.profile.short_fingerprint,
            "age": lambda name, f: age_text(f),
            "log score": lambda name, f: _number(f.log_score),
            "excess": lambda name, f: _number(f.excess),
            "noise 95%": lambda name, f: _number(f.excess_noise_95),
            "quadratic": lambda name, f: _number(f.quadratic_score),
        }
        if any(f.ruled_out for _, f in self.models):
            cells["ruled out"] = lambda name, f: str(f.ruled_out)
        dropped: list[str] = []
        while True:
            titles = [title for title in cells if title not in dropped]
            rows = [[cells[t](name, f) for t in titles] for name, f in self.models]
            table = _columns(titles, rows)
            if max(map(len, table)) <= width or len(dropped) == len(_DROPPABLE):
                return table
            dropped.append(_DROPPABLE[len(dropped)])


def rank(models: Sequence[tuple[str, Forecast]]) -> Ranking:
    """Order named forecasts of the same counts, best first, and test the best one against each
    other one with :func:`paired`."""
    if not models:
        raise ValueError("rank needs at least one forecast")

    def key(named: tuple[str, Forecast]) -> tuple[bool, float]:
        f = named[1]
        return bool(f.ruled_out), -(f.quadratic_score if f.ruled_out else f.log_score)

    ordered = tuple(sorted(models, key=key))
    best = ordered[0][1]
    return Ranking(ordered, tuple(paired(best, f) for _, f in ordered[1:]))


def age_text(forecast: Forecast) -> str:
    """The calibration age as ``nv score`` shows it, such as "26 h" or "-3 h" for a calibration
    after the run. A profile of another device gives "other device"."""
    if not forecast.same_device:
        return "other device"
    age = forecast.calibration_age
    if age is None:
        return "unknown"
    return f"-{_duration(-age)}" if age < timedelta(0) else _duration(age)


def rule_phrase(test: RuleTest, pair: PairedTest, a: str, b: str) -> str:
    """One rule's result in words, with model a named ``a`` and model b named ``b``."""
    if test.outcome == "ruled_out":
        out, kept = (b, a) if test.better == "a" else (a, b)
        shots = pair.b.ruled_out if test.better == "a" else pair.a.ruled_out
        return f"{out} is ruled out by {plural(shots, 'shot')}, so {kept} is better"
    if test.outcome == "both_ruled_out":
        return "both models are ruled out, so the quadratic rule decides"
    if test.outcome == "unbounded":
        return (
            "no test (one model gives probability 0 to an outcome that the other model"
            " expects), so the quadratic rule decides"
        )
    if test.outcome == "identical":
        return "no difference (the models give the same scores)"
    assert test.difference is not None and test.z is not None and test.p_value is not None
    stats = (
        f"z = {test.z:.2f}, p = {test.p_value:.2g},"
        f" standard error {_number(test.standard_error or 0.0)}"
    )
    if test.better == "neither":
        return f"no difference at the 5% level ({stats})"
    winner, loser = (a, b) if test.better == "a" else (b, a)
    return f"{winner} is better by {_number(abs(test.difference))} per shot than {loser} ({stats})"


def _chosen(counts: MeasuredCounts, names: Iterable[str] | None) -> list[MeasuredCircuit]:
    if names is None:
        return list(counts.circuits)
    wanted = list(dict.fromkeys(names))
    known = [c.name for c in counts.circuits]
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise CountsError(
            f"the counts have no circuit {unknown[0]!r}",
            hint=f"{did_you_mean(unknown[0], known)}give circuit names from {joined(known)}",
        )
    if not wanted:
        raise CountsError("circuits= names no circuit", hint="give at least one circuit name")
    return [c for c in counts.circuits if c.name in wanted]


def _probabilities(profile: Profile, c: MeasuredCircuit) -> np.ndarray:
    try:
        probs = probabilities(
            profile, c.ops, len(c.qubits), layout=c.qubits, readout=True, unknown_gates="error"
        )
    except NoiseVaultError as exc:
        raise CountsError(
            f"circuit {c.name}: {exc.message}",
            hint="score only circuits that the profile calibrates on their qubits",
        ) from None
    probs = np.where(probs > 0, probs, 0.0)
    probs.setflags(write=False)
    return probs


def _circuit(c: MeasuredCircuit, probs: np.ndarray, measured: np.ndarray) -> CircuitForecast:
    seen = measured > 0
    ruled_out = int(measured[seen & (probs == 0)].sum())
    n, p = measured[seen], probs[seen]
    if ruled_out:
        log_score, excess = -math.inf, math.inf
    else:
        log_score = float(n @ np.log(p)) / c.shots
        # A ratio n / (shots * p) overflows for a very small p. The logs do not.
        excess = float(n @ (np.log(n) - math.log(c.shots) - np.log(p))) / c.shots
    quadratic = float(2 * (measured @ probs) / c.shots - probs @ probs)
    return CircuitForecast(
        name=c.name,
        qubits=tuple(c.qubits),
        shots=c.shots,
        log_score=log_score,
        quadratic_score=quadratic,
        ruled_out=ruled_out,
        excess=excess,
        probabilities=probs,
    )


def _excess(drawn: np.ndarray, probs: np.ndarray) -> np.ndarray:
    shots = drawn.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log(drawn) - np.log(shots) - np.log(probs)
        terms = np.where(drawn > 0, drawn * logs, 0.0)
    return terms.sum(axis=1) / shots[:, 0]


def _normalized(probs: np.ndarray) -> np.ndarray:
    return probs / probs.sum()


def _mean(values: Sequence[float], forecasts: Sequence[CircuitForecast], shots: int) -> float:
    if any(math.isinf(v) for v in values):
        return next(v for v in values if math.isinf(v))
    return sum(v * c.shots for v, c in zip(values, forecasts, strict=True)) / shots


def _log(probs: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log(probs)


def _quadratic(probs: np.ndarray) -> np.ndarray:
    return 2 * probs - probs @ probs


def _test(
    rule: Rule,
    circuits: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    rule_score: Callable[[np.ndarray], np.ndarray],
) -> RuleTest:
    """The paired test of one rule. Each circuit gives its counts and the outcome
    probabilities of model a and of model b. No measured outcome has probability 0."""
    total, variance, scale, shots = 0.0, 0.0, 0.0, 0
    for measured, probs_a, probs_b in circuits:
        with np.errstate(invalid="ignore"):
            difference = rule_score(probs_a) - rule_score(probs_b)
        n = int(measured.sum())
        spreads = []
        for weights in (measured / n, probs_a, probs_b):
            expected = weights > 0
            d = difference[expected]
            if not np.all(np.isfinite(d)):
                return RuleTest(rule, "unbounded", None, None, None, None, "neither")
            mean = float(weights[expected] @ d)
            spreads.append(float(weights[expected] @ (d - mean) ** 2))
            scale += n * float(np.abs(d).max())
        seen = measured > 0
        total += float(measured[seen] @ difference[seen])
        variance += n * max(spreads)
        shots += n
    error = math.sqrt(variance)
    if error <= 1e-12 * scale:
        return RuleTest(rule, "identical", 0.0, 0.0, None, None, "neither")
    z = total / error
    p_value = math.erfc(abs(z) / math.sqrt(2))
    better: Choice = "neither" if p_value >= LEVEL else "a" if z > 0 else "b"
    return RuleTest(rule, "tested", total / shots, error / shots, z, p_value, better)


def _names(a: Profile, b: Profile) -> tuple[str, str]:
    first = ref_on_day(a.id, a.device.calibrated_at)
    second = ref_on_day(b.id, b.device.calibrated_at)
    if first == second:
        return f"{first} {a.short_fingerprint}", f"{second} {b.short_fingerprint}"
    return first, second


def _digest(forecast: Forecast) -> str:
    return "sha256:" + forecast.counts.sha256.removeprefix("sha256:")[:12]


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _number(value: float) -> str:
    if math.isinf(value):
        return "-inf" if value < 0 else "inf"
    return f"{value:.{_DIGITS}f}"


def _columns(titles: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [max(len(t), *(len(r[i]) for r in rows)) for i, t in enumerate(titles)]
    lines = []
    for cells in (titles, *rows):
        first = cells[0].ljust(widths[0])
        rest = "".join(f"  {cell:>{w}}" for cell, w in zip(cells[1:], widths[1:], strict=True))
        lines.append(first + rest)
    return lines
