"""Score a profile as a forecast of measured counts, with no fit.

A noise model gives a probability to each outcome of a circuit. A proper scoring rule rewards
the model that gives the measured outcomes the most probability. :func:`score` applies the log
rule and the quadratic rule to one profile. :func:`paired` tests which of two profiles scores
better on the same counts. :func:`probe_report` shows the probe circuits of the deep circuit set,
which a score leaves out because no noise model from a calibration can forecast them.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .compare import _duration, _wrap
from .counts import refuse_wide
from .errors import CountsError, NoiseVaultError, did_you_mean, joined, plural, qubit_loci
from .profile import iso_z, ref_on_day
from .reference import MAX_QUBITS, probabilities
from .suite import ideal_result, role

if TYPE_CHECKING:
    from .counts import MeasuredCircuit, MeasuredCounts
    from .profile import Profile

DRAWS = 200
LEVEL = 0.05
UNSEEN_SHOTS = 3
_WIDTH = 80
_DIGITS = 5
_DROPPABLE = ("fingerprint", "noise 95%", "age")

Rule = Literal["log", "quadratic"]
Choice = Literal["a", "b", "neither"]
Outcome = Literal["tested", "identical", "ruled_out", "both_ruled_out", "unbounded"]
Reason = Literal["probe", "too_wide", "no_ideal_result"]
# The most names that a "left out" line shows. A run of the deep circuit set leaves out 3 of
# each kind.
_SHOWN_NAMES = 4


@dataclass(frozen=True)
class LeftOut:
    """A circuit of the counts that a result does not use, and the reason.

    ``reason`` is "probe" (a ``probe`` circuit of the deep circuit set), "too_wide" (more qubits
    than the reference simulator holds) or "no_ideal_result" (a probe circuit with no certain
    result).
    """

    name: str
    reason: Reason

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "reason": self.reason}


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
    is 0 when the profile rules out a shot. ``left_out`` holds each circuit of the counts that
    the default selection of :func:`score` did not score.
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
    left_out: tuple[LeftOut, ...] = ()

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

    def select(self, names: Iterable[str]) -> Forecast:
        """The forecast of only the named circuits, with no new simulation.

        The result equals ``score(profile, counts, circuits=names)``. Raises CountsError for a
        name that this forecast does not hold.
        """
        wanted = list(dict.fromkeys(names))
        known = [c.name for c in self.circuits]
        unknown = next((name for name in wanted if name not in known), None)
        if unknown is not None:
            raise CountsError(
                f"the forecast has no circuit {unknown!r}",
                hint=f"{did_you_mean(unknown, known)}give circuit names from {joined(known)}",
            )
        if not wanted:
            raise CountsError("select names no circuit", hint="give at least one circuit name")
        kept = [c for c in self.circuits if c.name in wanted]
        return _total(self.profile, self.counts, kept)

    def summary(self, *, width: int = _WIDTH) -> str:
        """The scores as text, one row for each circuit and one row for the total."""
        profile, counts = self.profile, self.counts
        ref = ref_on_day(profile.id, profile.device.calibrated_at)
        head = [
            f"{ref} {profile.short_fingerprint}, calibration age {age_text(self)}",
            f"counts sha256:{counts.sha256.removeprefix('sha256:')[:12]}, {counts.source},"
            f" run {counts.run_at:%Y-%m-%d %H:%MZ} on {counts.backend}"
            f" {qubit_loci(_measured(self))}",
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
        lines += _left_out_lines(self.left_out, min(width, _WIDTH))
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
            "left_out": [item.to_dict() for item in self.left_out],
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
    can run the circuits with it. The profile's ``unmodeled_error`` factors apply.

    With no ``circuits``, the score leaves out two kinds of circuits and records them in
    ``Forecast.left_out``: the ``probe`` circuits of the deep circuit set (see
    :func:`probe_report`), and each circuit on more qubits than the reference simulator holds.
    Named ``circuits`` are scored as named, a probe circuit too.

    Raises CountsError for an unknown circuit name, for a circuit that the profile cannot
    simulate, and when the default selection leaves no circuit. A named circuit on more qubits
    than the reference simulator holds is one that the profile cannot simulate.
    """
    chosen, left_out = _chosen(counts, circuits)
    refuse_wide(chosen, "score can simulate")
    forecasts = [_circuit(c, _probabilities(profile, c), c.vector()) for c in chosen]
    return _total(profile, counts, forecasts, left_out)


def _total(
    profile: Profile,
    counts: MeasuredCounts,
    forecasts: Sequence[CircuitForecast],
    left_out: Sequence[LeftOut] = (),
) -> Forecast:
    """The forecast of the circuits in ``forecasts``, with the shot-noise reference of its
    excess. The draws depend only on the counts and on these circuits."""
    rng = np.random.default_rng(int(counts.sha256[7:23], 16))
    noise = np.zeros(DRAWS)
    for c in forecasts:
        drawn = rng.multinomial(c.shots, _normalized(c.probabilities), size=DRAWS)
        noise += c.shots * _excess(drawn, c.probabilities)
    shots = sum(c.shots for c in forecasts)
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
        left_out=tuple(left_out),
    )


def paired(a: Forecast, b: Forecast) -> PairedTest:
    """Test whether forecast ``a`` or forecast ``b`` scores better on their counts.

    For each rule, the test sums the per-shot score differences over all shots and divides the
    sum by its standard error. The two-sided p-value comes from the standard normal
    distribution. The variance of the difference in each circuit comes from the measured
    frequencies, plus a part for the outcomes that no shot showed. For the log rule, the test
    first takes the rare outcomes of a circuit together as one outcome: the outcomes that no
    shot showed and that each model expects less than one time.

    No test of this kind is safe against every possible device. A device can have an outcome
    that both models call almost impossible, with a very large score difference. The part for
    the outcomes that no shot showed limits this risk, and it makes the test careful.

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
    measured = {c.name: c.vector() for c in a.counts.circuits if c.name in names_a}
    pairs = [
        (measured[x.name], x.probabilities, y.probabilities)
        for x, y in zip(a.circuits, b.circuits, strict=True)
    ]
    if a.ruled_out and b.ruled_out:
        log = RuleTest("log", "both_ruled_out", None, None, None, None, "neither")
    elif a.ruled_out or b.ruled_out:
        log = RuleTest("log", "ruled_out", None, None, None, None, "b" if a.ruled_out else "a")
    else:
        log = _test("log", pairs, _log, group_rare=True)
    quadratic = _test("quadratic", pairs, _quadratic, group_rare=False)
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
            f"run {counts.run_at:%Y-%m-%d %H:%MZ} on {counts.backend},"
            f" {qubit_loci(_measured(self.models[0][1]))}",
        )
        lines = [part for text in head for part in _wrap(text, prose, "  ")]
        lines += ["", *self._table(width)]
        lines += _left_out_lines(self.models[0][1].left_out, prose)
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


@dataclass(frozen=True)
class ProbeQubit:
    """The wrong bits of one qubit in one probe circuit, as fractions of the shots.

    A bit is wrong when it differs from ``ideal_bit``. ``baseline`` is the fraction of wrong bits
    of the qubit in the ``readout`` circuit, or None when the counts have no such circuit with no
    gate on the qubit.
    """

    qubit: int
    ideal_bit: str
    measured: float
    forecast: float
    baseline: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "qubit": self.qubit,
            "ideal_bit": self.ideal_bit,
            "measured": self.measured,
            "forecast": self.forecast,
            "baseline": self.baseline,
        }


@dataclass(frozen=True)
class ProbeCircuit:
    """One probe circuit: the probability of its ideal result, measured and forecast, and the
    wrong bits of each qubit."""

    name: str
    qubits: tuple[int, ...]
    shots: int
    ideal_result: str
    measured: float
    forecast: float
    wrong_bits: tuple[ProbeQubit, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "qubits": list(self.qubits),
            "shots": self.shots,
            "ideal_result": self.ideal_result,
            "measured": self.measured,
            "forecast": self.forecast,
            "wrong_bits": [q.to_dict() for q in self.wrong_bits],
        }


@dataclass(frozen=True)
class Growth:
    """How the measured wrong bits above the baseline grow from the ``axis_*`` circuit with the
    fewest gates to the one with the most gates.

    ``ratios`` gives, for each qubit in ``qubits``, (wrong fraction minus baseline) of ``more``
    divided by the same of ``fewer``. A missing baseline counts as 0. A ratio is None when the
    divisor is not above 0. With ``gates`` times more gates, a systematic rotation error gives a
    ratio of about ``gates**2`` and an error with no direction about ``gates``, when the
    fractions are small.
    """

    fewer: str
    more: str
    gates: float
    qubits: tuple[int, ...]
    ratios: tuple[float | None, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fewer": self.fewer,
            "more": self.more,
            "gates": self.gates,
            "ratios": [
                {"qubit": q, "ratio": r} for q, r in zip(self.qubits, self.ratios, strict=True)
            ],
        }


@dataclass(frozen=True)
class ProbeReport:
    """The probe circuits of a counts file, measured and forecast by ``profile``.

    A probe circuit repeats gates so that a small systematic error of a gate adds up. A noise
    model from a calibration holds no such error, so it cannot forecast these circuits, and
    :func:`score` leaves them out. ``left_out`` holds the probe circuits that the report cannot
    show: "too_wide" or "no_ideal_result".
    """

    profile: Profile
    counts: MeasuredCounts = field(repr=False)
    circuits: tuple[ProbeCircuit, ...]
    growth: Growth | None
    left_out: tuple[LeftOut, ...]

    def summary(self, name: str | None = None, *, width: int = _WIDTH) -> str:
        """The report as text. ``name`` names the profile."""
        prose = min(width, _WIDTH)
        if name is None:
            name = ref_on_day(self.profile.id, self.profile.device.calibrated_at)
        lines = _wrap(f"probe circuits, not in the score; forecast by {name}", prose, "  ")
        if self.circuits:
            rows = [
                (
                    c.name,
                    str(c.shots),
                    c.ideal_result,
                    _fraction(c.measured),
                    _fraction(c.forecast),
                )
                for c in self.circuits
            ]
            titles = ("circuit", "shots", "ideal result", "measured", "forecast")
            lines += ["", *_columns(titles, rows)]
            lines += ["", "wrong bits, measured / forecast", *self._wrong_table(width)]
        if self.growth is not None:
            g = self.growth
            text = (
                f"growth: wrong bits above the readout value in {g.more}, divided by the same in"
                f" {g.fewer}. When the fractions are small, a systematic rotation error gives"
                f" about {g.gates**2:g}, and an error with no direction about {g.gates:g}."
            )
            lines += ["", *_wrap(text, prose, "  ")]
        lines += _left_out_lines(self.left_out, prose)
        return "\n".join(lines)

    __str__ = summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile.id,
            "fingerprint": self.profile.fingerprint,
            "counts": self.counts.sha256,
            "circuits": [c.to_dict() for c in self.circuits],
            "growth": None if self.growth is None else self.growth.to_dict(),
            "left_out": [item.to_dict() for item in self.left_out],
        }

    def _wrong_table(self, width: int) -> list[str]:
        """One row for each qubit, one column for each probe circuit. When the table is wider
        than ``width``, the circuits go in more than one table."""
        qubits = list(dict.fromkeys(q.qubit for c in self.circuits for q in c.wrong_bits))
        cells = {(c.name, q.qubit): q for c in self.circuits for q in c.wrong_bits}
        baseline = {q.qubit: q.baseline for c in self.circuits for q in c.wrong_bits}
        ratio = (
            {}
            if self.growth is None
            else dict(zip(self.growth.qubits, self.growth.ratios, strict=True))
        )

        def table(names: Sequence[str], last: bool) -> list[str]:
            titles = ["qubit", "readout", *names]
            if last and self.growth is not None:
                titles.append("growth")
            rows = []
            for qubit in qubits:
                row = [str(qubit), _optional(baseline[qubit])]
                for circuit in names:
                    cell = cells.get((circuit, qubit))
                    row.append(
                        "-"
                        if cell is None
                        else f"{_fraction(cell.measured)} / {_fraction(cell.forecast)}"
                    )
                if last and self.growth is not None:
                    found = ratio.get(qubit)
                    row.append("-" if found is None else f"{found:.1f}")
                rows.append(row)
            return _columns(titles, rows)

        names = [c.name for c in self.circuits]
        for size in range(len(names), 0, -1):
            parts = [names[i : i + size] for i in range(0, len(names), size)]
            tables = [table(part, i == len(parts) - 1) for i, part in enumerate(parts)]
            if size == 1 or all(max(map(len, t)) <= width for t in tables):
                return [line for i, t in enumerate(tables) for line in ([""] if i else []) + t]
        return []


def probe_report(profile: Profile, counts: MeasuredCounts) -> ProbeReport | None:
    """The probe circuits of ``counts``, measured and forecast by ``profile``.

    For each probe circuit (see :func:`noisevault.suite.role`), the report gives the measured
    and the forecast probability of its ideal result, and the fraction of wrong bits of each
    qubit. The circuit ``readout`` of the same counts gives the baseline of each qubit. For the
    ``axis_*`` circuits, :class:`Growth` compares the circuit with the most gates and the one
    with the fewest gates (the number after ``_d`` in the name).

    Returns None when the counts hold no probe circuit that the reference simulator can run.
    Raises CountsError for a probe circuit that the profile cannot simulate.
    """
    probes = [c for c in counts.circuits if role(c.name) == "probe"]
    runnable = [c for c in probes if len(c.qubits) <= MAX_QUBITS]
    if not runnable:
        return None
    left_out = [LeftOut(c.name, "too_wide") for c in probes if len(c.qubits) > MAX_QUBITS]
    baselines = _baselines(counts)
    circuits = []
    for c in runnable:
        try:
            ideal = ideal_result(c)
        except NoiseVaultError:
            left_out.append(LeftOut(c.name, "no_ideal_result"))
            continue
        circuits.append(_probe(c, ideal, _probabilities(profile, c), baselines))
    return ProbeReport(
        profile=profile,
        counts=counts,
        circuits=tuple(circuits),
        growth=_growth(circuits),
        left_out=tuple(left_out),
    )


def _probe(
    c: MeasuredCircuit, ideal: str, probs: np.ndarray, baselines: Mapping[int, float]
) -> ProbeCircuit:
    n = len(c.qubits)
    # Axis i of the reshaped array is circuit qubit i, the first character of an outcome.
    grid = probs.reshape((2,) * n)
    wrong = []
    for i, qubit in enumerate(c.qubits):
        others = tuple(j for j in range(n) if j != i)
        marginal = grid.sum(axis=others) if others else grid
        bad = "1" if ideal[i] == "0" else "0"
        measured = sum(k for key, k in c.counts.items() if key[i] == bad) / c.shots
        wrong.append(
            ProbeQubit(
                qubit=qubit,
                ideal_bit=ideal[i],
                measured=measured,
                forecast=float(marginal[int(bad)]),
                baseline=baselines.get(qubit),
            )
        )
    return ProbeCircuit(
        name=c.name,
        qubits=tuple(c.qubits),
        shots=c.shots,
        ideal_result=ideal,
        measured=c.counts.get(ideal, 0) / c.shots,
        forecast=float(probs[int(ideal, 2)]),
        wrong_bits=tuple(wrong),
    )


def _baselines(counts: MeasuredCounts) -> dict[int, float]:
    """The fraction of 1 bits of each qubit in the circuit ``readout``, when it has no gate."""
    found = next((c for c in counts.circuits if c.name == "readout"), None)
    if found is None or any(op.name != "delay" for op in found.ops):
        return {}
    return {
        qubit: sum(k for key, k in found.counts.items() if key[i] == "1") / found.shots
        for i, qubit in enumerate(found.qubits)
    }


def _growth(circuits: Sequence[ProbeCircuit]) -> Growth | None:
    gates = {}
    for c in circuits:
        found = re.fullmatch(r"axis_d(\d+)", c.name)
        if found and int(found[1]) > 0:
            gates[c.name] = int(found[1])
    if len(set(gates.values())) < 2:
        return None
    fewer = min(gates, key=gates.__getitem__)
    more = max(gates, key=gates.__getitem__)
    by_name = {c.name: {q.qubit: q for q in c.wrong_bits} for c in circuits}
    qubits = tuple(q for q in by_name[more] if q in by_name[fewer])
    ratios = []
    for qubit in qubits:
        top, bottom = by_name[more][qubit], by_name[fewer][qubit]
        divisor = bottom.measured - (bottom.baseline or 0.0)
        above = top.measured - (top.baseline or 0.0)
        ratios.append(above / divisor if divisor > 0 else None)
    return Growth(fewer, more, gates[more] / gates[fewer], qubits, tuple(ratios))


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


def _chosen(
    counts: MeasuredCounts, names: Iterable[str] | None
) -> tuple[list[MeasuredCircuit], list[LeftOut]]:
    """The circuits to score, and the circuits that the default selection leaves out."""
    if names is None:
        chosen, left_out = [], []
        for c in counts.circuits:
            reason = _left_out_reason(c)
            if reason is None:
                chosen.append(c)
            else:
                left_out.append(LeftOut(c.name, reason))
        if not chosen:
            raise CountsError(
                "the counts hold only probe circuits and circuits on more than"
                f" {MAX_QUBITS} qubits, which a score leaves out",
                hint="name the circuits to score with circuits=",
            )
        return chosen, left_out
    return _named(counts, names), []


def _left_out_reason(c: MeasuredCircuit) -> Reason | None:
    if role(c.name) == "probe":
        return "probe"
    if len(c.qubits) > MAX_QUBITS:
        return "too_wide"
    return None


def _left_out_lines(left_out: Sequence[LeftOut], width: int) -> list[str]:
    """One line for each reason, after a blank line, or no line. The names are not in
    parentheses, so that a long list can break between them."""
    kinds = {
        "probe": "",
        "too_wide": f" on more than {MAX_QUBITS} qubits",
        "no_ideal_result": " with no certain result",
    }
    lines = []
    for reason, kind in kinds.items():
        names = [item.name for item in left_out if item.reason == reason]
        if names:
            shown = ", ".join(names[:_SHOWN_NAMES])
            if len(names) > _SHOWN_NAMES:
                shown += f" and {len(names) - _SHOWN_NAMES} more"
            noun = plural(len(names), "probe circuit" if reason == "probe" else "circuit")
            text = f"{noun}{kind} left out: {shown}"
            lines += _wrap(text, width, "  ")
    return ["", *lines] if lines else []


def _named(counts: MeasuredCounts, names: Iterable[str]) -> list[MeasuredCircuit]:
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
        ) from exc
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
    *,
    group_rare: bool,
) -> RuleTest:
    """The paired test of one rule. Each circuit gives its counts and the outcome
    probabilities of model a and of model b. No measured outcome has probability 0.

    The variance of the score difference in a circuit has two parts. The first part comes from
    the measured frequencies. The second part is for the outcomes that no shot showed: the
    probability that they can have together, times the largest squared deviation of the score
    difference among them. Without the second part, a rare outcome with a large score
    difference gives no variance until a shot shows it, and the test rejects too often.
    """
    total, variance, scale, shots = 0.0, 0.0, 0.0, 0
    for measured, probs_a, probs_b in circuits:
        n = int(measured.sum())
        if group_rare:
            measured, probs_a, probs_b = _grouped(measured, probs_a, probs_b)
        with np.errstate(divide="ignore", invalid="ignore"):
            difference = rule_score(probs_a) - rule_score(probs_b)
        seen = measured > 0
        unseen = ~seen & ((probs_a > 0) | (probs_b > 0))
        if not np.all(np.isfinite(difference[unseen])):
            return RuleTest(rule, "unbounded", None, None, None, None, "neither")
        d = difference[seen]
        frequency = measured[seen] / n
        mean = float(frequency @ d)
        spread = float(frequency @ (d - mean) ** 2)
        if unseen.any():
            spread += _unseen_mass(measured, n) * float(((difference[unseen] - mean) ** 2).max())
        total += float(measured[seen] @ d)
        variance += n * spread
        scale += n * float(np.abs(difference[seen | unseen]).max())
        shots += n
    error = math.sqrt(variance)
    if error <= 1e-12 * scale:
        return RuleTest(rule, "identical", 0.0, 0.0, None, None, "neither")
    z = total / error
    p_value = math.erfc(abs(z) / math.sqrt(2))
    better: Choice = "neither" if p_value >= LEVEL else "a" if z > 0 else "b"
    return RuleTest(rule, "tested", total / shots, error / shots, z, p_value, better)


def _grouped(
    measured: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The outcomes of one circuit with the rare outcomes as one last outcome.

    An outcome is rare when no shot showed it and each model expects it less than one time in
    the shots of the circuit. Counts of this size cannot test the probability of one such
    outcome, so the test takes them together.
    """
    n = int(measured.sum())
    rare = (measured == 0) & (n * np.maximum(probs_a, probs_b) < 1)
    if rare.sum() < 2:
        return measured, probs_a, probs_b
    keep = ~rare
    return (
        np.append(measured[keep], 0),
        np.append(probs_a[keep], probs_a[rare].sum()),
        np.append(probs_b[keep], probs_b[rare].sum()),
    )


def _unseen_mass(measured: np.ndarray, shots: int) -> float:
    """An upper estimate of the probability of all outcomes that no shot showed.

    The Good-Turing estimate is the number of outcomes with exactly one shot, divided by the
    shots. With no such outcome, an event that 3 / shots can hide stays possible at the 5%
    level (the rule of three). The estimate adds both.
    """
    return min(1.0, (int((measured == 1).sum()) + UNSEEN_SHOTS) / shots)


def _names(a: Profile, b: Profile) -> tuple[str, str]:
    first = ref_on_day(a.id, a.device.calibrated_at)
    second = ref_on_day(b.id, b.device.calibrated_at)
    if first == second:
        return f"{first} {a.short_fingerprint}", f"{second} {b.short_fingerprint}"
    return first, second


def _measured(forecast: Forecast) -> tuple[int, ...]:
    """The qubits of the scored circuits, in first-seen order. A circuit that the score leaves
    out can measure many more qubits, and they would make the line very long."""
    return tuple(dict.fromkeys(q for c in forecast.circuits for q in c.qubits))


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


def _fraction(value: float) -> str:
    return f"{value:.3f}"


def _optional(value: float | None) -> str:
    return "-" if value is None else _fraction(value)
