"""Score a noise model on event bits, from samples of a stabilizer simulator.

An event is a bit with the value 1. In a wide Clifford circuit, an event is a wrong bit: a qubit
whose measured bit differs from the one ideal result. In an error-correction experiment, an
event is a detector that fired. The measured events are an array of shots by bits. Each model
gives samples of the same bits from Stim, so the circuit can have hundreds of qubits.

For each model, the report gives four results:

- ``rate``: the event rate of each bit, measured and forecast, with the sampling error of the
  forecast.
- ``bit`` score: the sum over the bits of the log probability of each bit under the marginal
  of the model for that bit. The report gives the mean over the shots.
- ``count`` score: one cell for each number of events in a shot, 0 to n. The cells that each of
  two models expects fewer than ``MIN_EXPECTED`` times in the measured shots become one last cell.
- ``none``: the share of shots with no event.

The bit score treats the bits as independent. Thus it does not see a correlation between bits,
for example two detectors that one error fires together. The count score sees a part of the
correlations. At the same event rates, correlated events give a wider spread of the number of
events in a shot.

Each forecast probability comes from ``samples`` Stim shots with add-half smoothing:
(k + 1/2) / (samples + 1) for a probability of two outcomes. For the count cells, the smoothing
adds 1/2 to each cell and divides by the sum, so that the cells add up to 1. The samples come in
chunks, so that the memory stays low.

Each model and each circuit has its own seed, so the sampling errors of two forecasts are
independent. The sampling error of a forecast is the same for each measured shot, so it does not
become smaller with more shots. The tests of the bit score and of the count score therefore add
the sampling variance of both forecasts to the shot variance (see :func:`compare`). A test gives
no verdict when its sampling variance is larger than its shot variance. A test also gives no
verdict when a forecast has fewer than ``MIN_SAMPLES`` samples where the variance needs them.

This module needs Stim (the ``stim`` extra) for :func:`score_wide` and :func:`score_detectors`.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from . import __version__, wide
from .compare import _wrap
from .conversion import UnknownGates
from .counts import COUNTS_FORMAT, Execution, MeasuredCounts, ProfileBinding
from .errors import CountsError, NoiseVaultError, joined, plural
from .profile import Profile, ref_on_day
from .reference import MAX_QUBITS
from .score import LEVEL, Choice, RuleTest, _columns, _log, _number, _test

if TYPE_CHECKING:
    from .counts import MeasuredCircuit, PlannedCircuit

SAMPLES = 1_000_000
CHUNK = 250_000
MIN_EXPECTED = 5
# The fewest samples on each side of a marginal, and in a cell with a measured shot, for which
# the delta method of the sampling variance is valid.
MIN_SAMPLES = 100
# The largest chunk of samples, in bits. A chunk of 2^25 bits takes 128 MiB as float32.
_CHUNK_BITS = 2**25
_SHOWN_BITS = 3
_WIDTH = 80
_DETECTOR_HINT = "give the detection events of this circuit, with no observable"

Kind = Literal["wide", "detectors", "bits"]
NoVerdict = Literal["few_samples", "sampling_variance", "no_variance"]
Models = Profile | Sequence[Profile] | Mapping[str, Profile]
DATA_FORMATS = {
    ".01": "01",
    ".b8": "b8",
    ".r8": "r8",
    ".ptb64": "ptb64",
    ".hits": "hits",
    ".dets": "dets",
}


@dataclass(frozen=True)
class Events:
    """Measured event bits: each distinct row of bits, and the shots of each row."""

    rows: np.ndarray
    times: np.ndarray
    shots: int

    @classmethod
    def of(cls, bits: Any) -> Events:
        """The events of an array of shots by bits, one row for each shot."""
        rows = np.asarray(bits, dtype=bool)
        return cls(rows, np.ones(len(rows)), len(rows))

    @property
    def rates(self) -> np.ndarray:
        return self.times @ self.rows / self.shots


@dataclass(frozen=True)
class Sampled:
    """The forecast of one model for one set of bits, from ``samples`` samples.

    ``none`` is the probability of no event, ``count[k]`` the probability of k events, and
    ``rate[j]`` the probability of an event on bit j. ``covariance`` is the covariance matrix of
    the bits over the samples, without smoothing. None means independent bits.
    ``rate_samples[j]`` is the number of samples with an event on bit j, and ``count_samples[k]``
    the number of samples with k events. None means not known.
    """

    samples: int
    none: float
    count: np.ndarray
    rate: np.ndarray
    covariance: np.ndarray | None = None
    rate_samples: np.ndarray | None = None
    count_samples: np.ndarray | None = None


@dataclass(frozen=True)
class Scores:
    """The mean scores of one model per shot, over the shots of the scored bit sets."""

    none_log: float
    none_quadratic: float
    count_log: float
    count_quadratic: float
    bit_log: float


@dataclass(frozen=True)
class Sampling:
    """The sampling variance of the two forecasts in one test, the shot variance of the test,
    and their ratio. Both variances are of the sum of the score differences over the shots.
    ``reason`` says why the samples stop a verdict, or it is None:

    - ``few_samples``: a marginal or a cell has fewer than ``MIN_SAMPLES`` samples.
    - ``sampling_variance``: the sampling variance is larger than the shot variance.
    - ``no_variance``: both variances are 0, so there is no test.
    """

    variance: float
    shot_variance: float
    reason: NoVerdict | None = None

    @property
    def ratio(self) -> float | None:
        return None if self.shot_variance <= 0 else self.variance / self.shot_variance


@dataclass(frozen=True)
class Paired:
    """The scores of models a and b, and the paired test of the two scores that give a verdict.

    Only the bit score and the log score of the count cells give a verdict. Only their tests
    hold the sampling variance of both forecasts. The scores of no event and the quadratic count
    score are values only, in ``a`` and ``b``.
    """

    a: Scores
    b: Scores
    count_log: RuleTest
    bit_log: RuleTest
    bit_sampling: Sampling
    count_sampling: Sampling


def accumulate(chunks: Iterable[np.ndarray], width: int) -> Sampled:
    """The smoothed forecast of ``width`` bits from chunks of samples, each of shots by bits."""
    cells = np.zeros(width + 1)
    rate = np.zeros(width)
    both = np.zeros((width, width))
    samples = 0
    for events in chunks:
        cells += np.bincount(events.sum(axis=1), minlength=width + 1)
        rate += events.sum(axis=0)
        # float32 counts each sum of one chunk exactly, because a chunk has fewer than 2^24 rows.
        values = events.astype(np.float32)
        both += values.T @ values
        samples += len(events)
    if samples == 0:
        raise ValueError("a forecast needs at least one sample")
    smoothed = cells + 0.5
    mean = rate / samples
    return Sampled(
        samples=samples,
        none=_smoothed(int(cells[0]), samples),
        count=smoothed / smoothed.sum(),
        rate=(rate + 0.5) / (samples + 1),
        covariance=both / samples - np.outer(mean, mean),
        rate_samples=rate,
        count_samples=cells,
    )


def circuit_events(circuit: MeasuredCircuit, ideal: str) -> Events:
    """The wrong bits of the measured shots of ``circuit``, whose one ideal result is ``ideal``."""
    keys, times = _rows(circuit)
    return Events(keys != _bits(ideal), times, circuit.shots)


def circuit_forecast(
    profile: Profile,
    circuit: PlannedCircuit,
    *,
    samples: int = SAMPLES,
    seed: int = 0,
    chunk: int = CHUNK,
) -> Sampled:
    """The smoothed forecast of ``profile`` for the wrong bits of the Clifford ``circuit``.

    The readout error of each qubit applies to the sampled bits as in ``noisevault.wide.sample``,
    which gives the same bits for one chunk.
    """
    target = _bits(wide.ideal_bits(circuit))
    noisy, readout = wide.noisy_circuit(profile, circuit)
    rng = np.random.default_rng(seed)
    sampler = noisy.compile_sampler(seed=int(rng.integers(2**63)))
    size = _chunk(chunk, len(target))

    def chunks() -> Iterator[np.ndarray]:
        left = samples
        while left > 0:
            n = min(size, left)
            bits = sampler.sample(n)
            for q, (to_one, to_zero) in enumerate(readout):
                bits[:, q] ^= rng.random(n) < np.where(bits[:, q], to_zero, to_one)
            yield bits != target
            left -= n

    return accumulate(chunks(), len(target))


def detector_forecast(
    circuit: Any, *, samples: int = SAMPLES, seed: int = 0, chunk: int = CHUNK
) -> Sampled:
    """The smoothed forecast of the detectors of the noisy Stim ``circuit``."""
    width = circuit.num_detectors
    rng = np.random.default_rng(seed)
    sampler = circuit.compile_detector_sampler(seed=int(rng.integers(2**63)))
    size = _chunk(chunk, width)

    def chunks() -> Iterator[np.ndarray]:
        left = samples
        while left > 0:
            n = min(size, left)
            yield sampler.sample(n)
            left -= n

    return accumulate(chunks(), width)


def compare(measured: Sequence[Events], a: Sequence[Sampled], b: Sequence[Sampled]) -> Paired:
    """The scores and the paired tests of models a and b, with one forecast of each model for
    each set of measured bits. Each test sums over the sets, which are independent.

    The bit score uses a paired z-test. Each shot gives one difference d = s_a - s_b of the two
    per-shot scores. The test divides the sum of d by its standard error. The standard error is
    the square root of the sum over the sets of the shots times the sample variance of d in the
    set. The two-sided p-value comes from the standard normal distribution. The log rule of the
    count cells uses the paired test of ``noisevault.score`` without its grouping of rare
    outcomes. The count cells merge their rare cells first. The tests assume:

    - The shots are independent, and the shots of one set have one distribution.
    - Both models came before the measured bits, or a fit made them from other data.
    - Each set has enough shots that the mean of d is near normal. Each smoothed probability is
      above 0, so d has a limit and a finite variance.

    Both tests add the sampling variance of both forecasts to the variance of the summed
    difference. For one model and one set with N shots and S samples, the term is:

    - bit score: a^T C a / S (see :func:`bit_sampling_variance`).
    - count score: (N^2 / S) (sum over the cells of f^2 / p - 1), with the measured fraction f
      and the forecast p of a cell.
    """
    cells = [_cells(m, fa, fb, merge=True) for m, fa, fb in zip(measured, a, b, strict=True)]
    none, count, bit = (list(kind) for kind in zip(*cells, strict=True))
    bit_variance = count_variance = 0.0
    for m, fa, fb, (per_cell, pa, pb) in zip(measured, a, b, count, strict=True):
        for found in (fa, fb):
            bit_variance += bit_sampling_variance(
                m.shots, found.samples, m.rates, found.rate, found.covariance
            )
        for found, probs in ((fa, pa), (fb, pb)):
            count_variance += cell_sampling_variance(m.shots, found.samples, per_cell, probs)
    total = float(sum(m.shots for m in measured))
    few_bits = any(few_rate_samples(found) for found in (*a, *b))
    few_cells = any(
        few_count_samples(found, m, fa, fb)
        for m, fa, fb in zip(measured, a, b, strict=True)
        for found in (fa, fb)
    )
    bit_test, bit_sampling = with_sampling(_shot_test(bit), bit_variance, total, few=few_bits)
    count_test, count_sampling = with_sampling(
        _test("log", count, _log, group_rare=False), count_variance, total, few=few_cells
    )
    return Paired(
        a=_scores(none, count, bit, 1),
        b=_scores(none, count, bit, 2),
        count_log=count_test,
        bit_log=bit_test,
        bit_sampling=bit_sampling,
        count_sampling=count_sampling,
    )


def scores(measured: Sequence[Events], found: Sequence[Sampled]) -> Scores:
    """The mean scores of one model. The function does not merge the count cells, so the scores
    of two models on the same bits use the same cells."""
    cells = [_cells(m, f, f, merge=False) for m, f in zip(measured, found, strict=True)]
    none, count, bit = (list(kind) for kind in zip(*cells, strict=True))
    return _scores(none, count, bit, 1)


def bit_sampling_variance(
    shots: float,
    samples: int,
    measured: np.ndarray,
    forecast: np.ndarray,
    covariance: np.ndarray | None = None,
) -> float:
    """The sampling variance of the summed bit score of one model on one set of bits.

    The value is a^T C a / S, with a_j = N (f_j - q_j) / (q_j (1 - q_j)). Here f is the
    measured rate and q the forecast of bit j. C is the covariance matrix of the bits over the
    samples. The marginals come from one set of samples, so their errors have a correlation.
    Without ``covariance``, C is diagonal with q (1 - q). The value is then (N^2 / S) times the
    sum over the bits of (f - q)^2 / (q (1 - q)).
    """
    f, q = np.asarray(measured, dtype=float), np.asarray(forecast, dtype=float)
    a = shots * (f - q) / (q * (1 - q))
    if covariance is None:
        return float((a**2 * q * (1 - q)).sum()) / samples
    return float(a @ np.asarray(covariance, dtype=float) @ a) / samples


def cell_sampling_variance(
    shots: float, samples: int, measured: np.ndarray, forecast: np.ndarray
) -> float:
    """The sampling variance of the summed log score of one model on the cells of one set.

    The value is (N^2 / S) (sum over the cells of f^2 / p - 1). ``measured`` gives the shots of
    each cell.
    """
    f = np.asarray(measured, dtype=float) / shots
    p = np.asarray(forecast, dtype=float)
    return shots**2 / samples * max(0.0, float((f**2 / p).sum()) - 1)


def few_rate_samples(found: Sampled) -> bool:
    """True when a bit of the forecast has fewer than ``MIN_SAMPLES`` sampled events or sampled
    shots with no event on the bit."""
    if found.rate_samples is None:
        return False
    events = np.asarray(found.rate_samples)
    return bool((np.minimum(events, found.samples - events) < MIN_SAMPLES).any())


def few_count_samples(found: Sampled, measured: Events, fa: Sampled, fb: Sampled) -> bool:
    """True when a count cell of ``measured`` holds a measured shot and has fewer than
    ``MIN_SAMPLES`` samples in the forecast ``found``. The cells are the merged cells of the test
    of ``fa`` and ``fb``."""
    if found.count_samples is None:
        return False
    width = measured.rows.shape[1]
    shots = np.bincount(measured.rows.sum(axis=1), weights=measured.times, minlength=width + 1)
    samples = np.asarray(found.count_samples, dtype=float)
    # The rule of _merged: the cells that each model expects fewer than MIN_EXPECTED times
    # become one last cell, with the sum of their shots and of their samples.
    rare = measured.shots * np.maximum(fa.count, fb.count) < MIN_EXPECTED
    if rare.any():
        shots = np.append(shots[~rare], shots[rare].sum())
        samples = np.append(samples[~rare], samples[rare].sum())
    return bool(((shots > 0) & (samples < MIN_SAMPLES)).any())


def with_sampling(
    test: RuleTest, variance: float, shots: float, *, few: bool = False
) -> tuple[RuleTest, Sampling]:
    """``test`` over ``shots`` shots, with the sampling ``variance`` added to the variance of its
    summed difference.

    The test gives no verdict when ``variance`` is larger than the shot variance. It also gives
    no verdict when ``few`` says that a marginal or a cell has too few samples. A test with no
    value stays as it is.
    """
    if test.outcome != "tested" or test.difference is None or test.standard_error is None:
        return test, Sampling(variance, 0.0, "few_samples" if few else None)
    total = test.difference * shots
    shot_variance = (test.standard_error * shots) ** 2
    error = math.sqrt(shot_variance + variance)
    if error == 0:
        # No shot variance and no sampling variance: no test, but the difference stays.
        reason: NoVerdict = "few_samples" if few else "no_variance"
        found = RuleTest(test.rule, "tested", test.difference, 0.0, None, None, "neither")
        return found, Sampling(variance, shot_variance, reason)
    z = total / error
    p_value = math.erfc(abs(z) / math.sqrt(2))
    too_large = variance > shot_variance
    why: NoVerdict | None = "few_samples" if few else "sampling_variance" if too_large else None
    sampling = Sampling(variance, shot_variance, why)
    better: Choice = "neither" if p_value >= LEVEL or why else "a" if z > 0 else "b"
    found = RuleTest(test.rule, "tested", test.difference, error / shots, z, p_value, better)
    return found, sampling


def seed_of(digest: str, model: str, circuit: str) -> int:
    """A seed from the SHA-256 of a counts file, a model name and a circuit name. Thus a result
    repeats, and each model and circuit has its own sampling error."""
    return int(hashlib.sha256(f"{digest}\n{model}\n{circuit}".encode()).hexdigest()[:16], 16)


# reports ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Bit:
    """One scored bit: a qubit of a wide circuit, or a detector.

    ``name`` is ``q`` and the physical qubit, or ``D`` and the detector index. ``circuit`` is the
    name of the wide circuit, or None. ``coords`` holds the detector coordinates of the circuit.
    ``measured`` is the measured event rate.
    """

    name: str
    circuit: str | None
    coords: tuple[float, ...]
    shots: int
    measured: float


@dataclass(frozen=True)
class ModelScore:
    """The forecast of one model, over all scored bits.

    ``rates`` and ``errors`` hold the forecast event rate of each bit of the report and its
    sampling standard error. ``events`` is the forecast mean number of events in a shot, and
    ``events_error`` its sampling standard error. ``none`` is the forecast share of shots with
    no event. ``few_samples`` is True when a bit or a measured count cell has fewer than
    ``MIN_SAMPLES`` samples. Then a test with this model can give no verdict.
    """

    name: str
    profile: Profile | None = field(repr=False)
    samples: int
    rates: tuple[float, ...] = field(repr=False)
    errors: tuple[float, ...] = field(repr=False)
    events: float
    events_error: float
    none: float
    scores: Scores
    few_samples: bool


@dataclass(frozen=True)
class EventTest:
    """The paired test of model ``a`` (the best) against model ``b`` on the bit score and on
    the count score. The bit score decides ``better``."""

    a: str
    b: str
    bit: RuleTest
    count: RuleTest
    bit_sampling: Sampling
    count_sampling: Sampling

    @property
    def better(self) -> str | None:
        return {"a": self.a, "b": self.b, "neither": None}[self.bit.better]


@dataclass(frozen=True)
class EventScore:
    """The score of one or more models on measured event bits.

    ``models`` is in the order of the bit score, best first. ``tests`` holds the paired test of
    the best model against each other model. ``events`` is the measured mean number of events
    in a shot, ``events_error`` its standard error, and ``none`` the measured share of shots
    with no event. A wide score sums over its circuits: each circuit has its own bits and shots.
    """

    kind: Kind
    counts: str | None
    circuits: tuple[str, ...]
    shots: int
    bits: tuple[Bit, ...] = field(repr=False)
    events: float
    events_error: float
    none: float
    models: tuple[ModelScore, ...]
    tests: tuple[EventTest, ...]

    def summary(self, *, width: int = _WIDTH) -> str:
        """The report as text: the measured events, the forecast of each model, the bits with
        the largest differences, and the paired tests."""
        prose = min(width, _WIDTH)
        lines = _wrap(self._head(), prose, "  ")
        measured = (
            f"measured: {_count(self.events)} events per shot,"
            f" no event in {_share(self.none)} of the shots"
        )
        lines += _wrap(measured, prose, "  ")
        rows = [
            [
                m.name,
                _count(m.events),
                _ratio(m.events, self.events),
                _share(m.none),
                _number(m.scores.bit_log),
                _number(m.scores.count_log),
            ]
            for m in self.models
        ]
        titles = ["model", "events", "ratio", "no event", "bit score", "count score"]
        lines += ["", *_columns(titles, rows), ""]
        for m in self.models:
            lines += _wrap(f"{m.name}: {self._events_phrase(m)}", prose, "  ")
            shown = self._different_bits(m)
            if shown:
                text = f"largest differences, forecast against measured: {shown}"
                lines += ["  " + part for part in _wrap(text, prose - 2, "  ")]
        for test in self.tests:
            lines += ["", f"{test.b} against {test.a}:"]
            for name, rule, sampling, few in (
                ("bit score", test.bit, test.bit_sampling, "of 0 or of 1 on a bit"),
                (
                    "count score",
                    test.count,
                    test.count_sampling,
                    "of a number of events that the measured shots show",
                ),
            ):
                text = f"{name}: {_test_phrase(rule, sampling, test.a, test.b, few)}"
                lines += ["  " + part for part in _wrap(text, prose - 2, "  ")]
        return "\n".join(lines)

    __str__ = summary

    def to_dict(self) -> dict[str, Any]:
        names = [m.name for m in self.models]
        return {
            "kind": self.kind,
            "counts": self.counts,
            "circuits": list(self.circuits),
            "shots": self.shots,
            "level": LEVEL,
            "measured": {
                "events_per_shot": self.events,
                "events_error": self.events_error,
                "no_event": self.none,
            },
            "models": [
                {
                    "name": m.name,
                    "profile_id": None if m.profile is None else m.profile.id,
                    "fingerprint": None if m.profile is None else m.profile.fingerprint,
                    "samples": m.samples,
                    "events_per_shot": m.events,
                    "events_error": m.events_error,
                    "ratio": m.events / self.events if self.events > 0 else None,
                    "no_event": m.none,
                    "few_samples": m.few_samples,
                    "scores": {
                        "bit": m.scores.bit_log,
                        "count": m.scores.count_log,
                        "count_quadratic": m.scores.count_quadratic,
                        "no_event_log": m.scores.none_log,
                        "no_event_quadratic": m.scores.none_quadratic,
                    },
                }
                for m in self.models
            ],
            "paired": [
                {
                    "best": t.a,
                    "other": t.b,
                    "bit": _test_dict(t.bit, t.bit_sampling),
                    "count": _test_dict(t.count, t.count_sampling),
                    "better": t.better,
                }
                for t in self.tests
            ],
            "bits": [
                {
                    "name": bit.name,
                    "circuit": bit.circuit,
                    "coords": list(bit.coords),
                    "shots": bit.shots,
                    "measured": bit.measured,
                    "forecast": {n: m.rates[i] for n, m in zip(names, self.models, strict=True)},
                    "error": {n: m.errors[i] for n, m in zip(names, self.models, strict=True)},
                }
                for i, bit in enumerate(self.bits)
            ],
        }

    def _head(self) -> str:
        samples = sorted({m.samples for m in self.models})
        drawn = f"forecasts from {' or '.join(map(str, samples))} samples"
        if self.kind == "detectors":
            what = f"detection events: {plural(len(self.bits), 'detector')}"
        elif self.kind == "wide":
            what = f"wide circuits: {joined(list(self.circuits))}"
        else:
            what = f"event bits: {plural(len(self.bits), 'bit')}"
        return f"{what}, {plural(self.shots, 'shot')}, {drawn}"

    def _events_phrase(self, m: ModelScore) -> str:
        error = math.hypot(self.events_error, m.events_error)
        z = 0.0 if error == 0 else (m.events - self.events) / error
        if math.erfc(abs(z) / math.sqrt(2)) >= LEVEL:
            return "the number of events agrees with the measured number, within chance"
        side = "too many" if z > 0 else "too few"
        ratio = _ratio(m.events, self.events)
        return f"{side} events, {ratio} times the measured number (z = {z:.1f})"

    def _different_bits(self, m: ModelScore) -> str:
        """The bits whose forecast rate differs most from the measured rate. Each difference
        must be more than chance after a Bonferroni correction for the number of bits."""
        f = np.array([b.measured for b in self.bits])
        n = np.array([b.shots for b in self.bits], dtype=float)
        q, e = np.array(m.rates), np.array(m.errors)
        error = np.sqrt(f * (1 - f) / n + e**2)
        z = np.divide(q - f, error, out=np.zeros_like(q), where=error > 0)
        p = np.array([math.erfc(abs(v) / math.sqrt(2)) for v in z])
        chosen = [i for i in np.argsort(-np.abs(z)) if p[i] < LEVEL / len(self.bits)]
        many = len(self.circuits) > 1
        shown = []
        for i in chosen[:_SHOWN_BITS]:
            bit = self.bits[i]
            name = f"{bit.name} of {bit.circuit}" if many else bit.name
            shown.append(f"{name} {_share(q[i])} against {_share(f[i])}")
        return ", ".join(shown)


def score_events(measured: Any, sampled: Mapping[str, Any | Iterable[Any]]) -> EventScore:
    """Score the sampled bits of each model as a forecast of the ``measured`` bits.

    ``measured`` is an array of shots by bits, with 1 for an event. ``sampled`` maps a model name
    to its samples of the same bits: an array of samples by bits, or an iterable of such chunks.
    Use this function for samples from a simulator of your own. :func:`score_wide` and
    :func:`score_detectors` make the samples with Stim.
    """
    if not sampled:
        raise NoiseVaultError("give the samples of at least one model")
    events = _event_array(measured, None)
    width = events.rows.shape[1]
    forecasts: dict[str, tuple[Profile | None, list[Sampled]]] = {}
    for name, found in sampled.items():
        chunks = [found] if isinstance(found, np.ndarray) else found
        expected = f"the measured events have {plural(width, 'bit')}"
        arrays = (_event_array(chunk, width, expected).rows for chunk in chunks)
        forecasts[name] = (None, [accumulate(arrays, width)])
    bits = [Bit(f"b{j}", None, (), events.shots, float(r)) for j, r in enumerate(events.rates)]
    return _report("bits", None, [(None, events, bits)], forecasts)


def score_wide(
    models: Models,
    counts: MeasuredCounts,
    *,
    circuits: Iterable[str] | None = None,
    samples: int = SAMPLES,
    seed: int = 0,
) -> EventScore:
    """Score each model on the wide Clifford circuits of ``counts``, from Stim samples.

    Without ``circuits``, the score takes each circuit on more qubits than the reference
    simulator holds (``noisevault.reference.MAX_QUBITS``). An event is a wrong bit: a measured
    bit that differs from the one ideal result of the circuit. ``models`` is one profile, a
    list of profiles, or a dict of name to profile. ``samples`` is the number of Stim samples of
    each model for each circuit. The same counts, models, ``samples`` and ``seed`` give the same
    report.

    Raises ImportError with the install command when Stim is missing. Raises CountsError for an
    unknown circuit name or for no wide circuit. Raises NoiseVaultError for a gate outside
    ``noisevault.wide.CLIFFORD`` and for a circuit with no certain result.
    """
    wide.stim_module()
    _check_samples(samples)
    named = _named(models)
    chosen = _wide_circuits(counts, circuits)
    groups = []
    for c in chosen:
        ideal = wide.ideal_bits(c)
        events = circuit_events(c, ideal)
        rates = events.rates
        bits = [
            Bit(f"q{q}", c.name, (), c.shots, float(r))
            for q, r in zip(c.qubits, rates, strict=True)
        ]
        groups.append((c.name, events, bits))
    digest = f"{counts.sha256}\n{seed}"
    forecasts: dict[str, tuple[Profile | None, list[Sampled]]] = {}
    for name, profile in named.items():
        found = []
        for c in chosen:
            try:
                found.append(
                    circuit_forecast(
                        profile, c, samples=samples, seed=seed_of(digest, name, c.name)
                    )
                )
            except NoiseVaultError as exc:
                raise type(exc)(f"model {name}: {exc.message}", hint=exc.hint) from None
        forecasts[name] = (profile, found)
    return _report("wide", counts.sha256, groups, forecasts)


def score_detectors(
    models: Models,
    circuit: Any,
    detection_events: Any,
    *,
    layout: Any = None,
    tick_ns: float | None = None,
    unknown_gates: UnknownGates = "error",
    data_format: str | None = None,
    samples: int = SAMPLES,
    seed: int = 0,
) -> EventScore:
    """Score each model as a forecast of the measured detection events of an experiment.

    ``circuit`` is the Stim circuit of the experiment, with its DETECTOR instructions and with no
    noise. Give a ``stim.Circuit``, Stim program text or the path of a ``.stim`` file. For each
    model, ``profile.to_stim`` adds the noise of the profile on the physical qubits of
    ``layout``, with ``tick_ns`` and ``unknown_gates`` as in ``to_stim``. By default, a gate
    that the profile does not define is an error. Each measured result flips with the mean of
    P(1|0) and P(0|1).

    ``detection_events`` is an array of shots by detectors, with 1 for a detector that fired, or
    the path of a file that Stim can read. ``data_format`` names the Stim format of the file
    ("01", "b8", "r8", "ptb64", "hits" or "dets"); by default the file suffix names it. The
    report has one bit for each detector, ``D`` and its index.

    Raises ImportError with the install command when Stim is missing. Raises NoiseVaultError for
    a circuit with no detector or with noise, and for data with another number of detectors.
    Raises MissingCalibrationError for a gate that a profile cannot give noise to.
    """
    stim = wide.stim_module()
    from .frameworks.stim import ExistingNoiseError

    _check_samples(samples)
    named = _named(models)
    if isinstance(circuit, os.PathLike):
        circuit = Path(circuit).read_text(encoding="utf-8")
    if isinstance(circuit, str):
        circuit = stim.Circuit(circuit)
    if circuit.num_detectors == 0:
        raise NoiseVaultError(
            "the circuit has no DETECTOR instruction, so it gives no detection events",
            hint="give the circuit of the experiment with its DETECTOR instructions",
        )
    width = circuit.num_detectors
    events = _detection_data(stim, detection_events, width, data_format)
    coords = circuit.get_detector_coordinates()
    bits = [
        Bit(f"D{j}", None, tuple(map(float, coords.get(j, ()))), events.shots, float(r))
        for j, r in enumerate(events.rates)
    ]
    forecasts: dict[str, tuple[Profile | None, list[Sampled]]] = {}
    for name, profile in named.items():
        try:
            noisy = profile.to_stim(
                circuit, layout=layout, tick_ns=tick_ns, unknown_gates=unknown_gates
            )
        except ExistingNoiseError as exc:
            raise NoiseVaultError(
                exc.message,
                hint="give the circuit with no noise, for example circuit.without_noise();"
                " score_detectors adds the noise of each profile",
            ) from None
        except NoiseVaultError as exc:
            raise type(exc)(f"model {name}: {exc.message}", hint=exc.hint) from None
        found = detector_forecast(
            noisy, samples=samples, seed=seed_of(str(seed), name, "detectors")
        )
        forecasts[name] = (profile, [found])
    return _report("detectors", None, [(None, events, bits)], forecasts)


def simulate_wide(
    truth: Profile,
    circuits: Sequence[PlannedCircuit],
    *,
    shots: int,
    seed: int,
    run_at: datetime | None = None,
) -> MeasuredCounts:
    """Counts of the Clifford ``circuits`` sampled from the stabilizer forecast of ``truth``,
    bound to its calibration, as ``noisevault.counts.simulate`` binds them. ``truth`` stands for
    the device, so a test of a model can use counts with a known noise model."""
    measured = [
        {
            "name": c.name,
            "qubits": c.qubits,
            "ops": c.ops,
            "shots": shots,
            "counts": wide.counts(wide.sample(truth, c, shots, seed=seed + i)),
        }
        for i, c in enumerate(circuits)
    ]
    return MeasuredCounts.model_validate(
        {
            "nv_counts": COUNTS_FORMAT,
            "source": "simulated",
            "profile": ProfileBinding(id=truth.id, fingerprint=truth.calibration_fingerprint),
            "backend": truth.device.name,
            "run_at": datetime.now(UTC) if run_at is None else run_at,
            "bit_order": "clbit0_left",
            "execution": Execution(
                client=f"noisevault {__version__} stabilizer simulate",
                transpiled=False,
                gate_twirling=False,
                measure_twirling=False,
                dynamical_decoupling=False,
                init_qubits=True,
                options={"seed": seed, "unmodeled_error": truth.to_dict().get("unmodeled_error")},
            ),
            "circuits": measured,
        }
    )


# helpers ---------------------------------------------------------------------------------------


def _report(
    kind: Kind,
    counts: str | None,
    groups: Sequence[tuple[str | None, Events, list[Bit]]],
    forecasts: Mapping[str, tuple[Profile | None, list[Sampled]]],
) -> EventScore:
    measured = [events for _, events, _ in groups]
    shots = sum(m.shots for m in measured)
    hist = [_histogram(m) for m in measured]
    events = sum(h @ np.arange(len(h)) for h in hist) / shots
    spread = sum(m.shots * _variance(h / m.shots) for m, h in zip(measured, hist, strict=True))
    models = []
    for name, (profile, found) in forecasts.items():
        rates = np.concatenate([f.rate for f in found])
        samples = np.concatenate([np.full(len(f.rate), f.samples) for f in found])
        forecast_events = sum(
            m.shots * float(f.rate.sum()) for m, f in zip(measured, found, strict=True)
        )
        forecast_spread = sum(
            m.shots**2 * _variance(f.count) / f.samples
            for m, f in zip(measured, found, strict=True)
        )
        few = any(few_rate_samples(f) for f in found) or any(
            few_count_samples(f, m, f, f) for m, f in zip(measured, found, strict=True)
        )
        models.append(
            ModelScore(
                name=name,
                profile=profile,
                samples=max(f.samples for f in found),
                rates=tuple(map(float, rates)),
                errors=tuple(map(float, np.sqrt(rates * (1 - rates) / samples))),
                events=forecast_events / shots,
                events_error=math.sqrt(forecast_spread) / shots,
                none=sum(m.shots * f.none for m, f in zip(measured, found, strict=True)) / shots,
                scores=scores(measured, found),
                few_samples=few,
            )
        )
    models.sort(key=lambda m: -m.scores.bit_log)
    by_name = {name: found for name, (_, found) in forecasts.items()}
    best = models[0].name
    tests = []
    for other in models[1:]:
        result = compare(measured, by_name[best], by_name[other.name])
        tests.append(
            EventTest(
                best,
                other.name,
                result.bit_log,
                result.count_log,
                result.bit_sampling,
                result.count_sampling,
            )
        )
    return EventScore(
        kind=kind,
        counts=counts,
        circuits=tuple(name for name, _, _ in groups if name is not None),
        shots=shots,
        bits=tuple(bit for _, _, bits in groups for bit in bits),
        events=float(events),
        events_error=math.sqrt(spread) / shots,
        none=sum(float(h[0]) for h in hist) / shots,
        models=tuple(models),
        tests=tuple(tests),
    )


def _named(models: Models) -> dict[str, Profile]:
    """Each model by name. A list names each profile by its ref on its calibration day, with
    the short fingerprint when two refs are the same."""
    if isinstance(models, Profile):
        models = [models]
    if isinstance(models, Mapping):
        named = dict(models)
    else:
        refs = [ref_on_day(p.id, p.device.calibrated_at) for p in models]
        names = [
            f"{ref} {p.short_fingerprint}" if refs.count(ref) > 1 else ref
            for ref, p in zip(refs, models, strict=True)
        ]
        if len(set(names)) < len(names):
            raise NoiseVaultError(
                "two of the profiles are the same, so their names are the same",
                hint="give a dict of name to profile",
            )
        named = dict(zip(names, models, strict=True))
    if not named:
        raise NoiseVaultError("give at least one profile")
    return named


def _wide_circuits(counts: MeasuredCounts, names: Iterable[str] | None) -> list[MeasuredCircuit]:
    if names is None:
        chosen = [c for c in counts.circuits if len(c.qubits) > MAX_QUBITS]
        if not chosen:
            raise CountsError(
                f"the counts hold no circuit on more than {MAX_QUBITS} qubits",
                hint="score the counts with nv score, or name the circuits with circuits=",
            )
        return chosen
    by_name = {c.name: c for c in counts.circuits}
    names = list(dict.fromkeys(names))
    unknown = next((name for name in names if name not in by_name), None)
    if unknown is not None:
        raise CountsError(f"the counts have no circuit {unknown!r}")
    if not names:
        raise CountsError("give at least one circuit name in circuits=")
    return [by_name[name] for name in names]


def _detection_data(stim: Any, data: Any, width: int, data_format: str | None) -> Events:
    if isinstance(data, str | os.PathLike):
        path = Path(data)
        fmt = data_format or DATA_FORMATS.get(path.suffix)
        if fmt is None:
            raise NoiseVaultError(
                f"the suffix of {path.name} names no Stim data format",
                hint=f"give data_format=, one of {joined(list(DATA_FORMATS.values()), 'or')}",
            )
        try:
            data = stim.read_shot_data_file(path=str(path), format=fmt, num_detectors=width)
        except (ValueError, RuntimeError) as exc:
            raise NoiseVaultError(
                f"Stim cannot read {path.name} as {fmt} data of {plural(width, 'detector')}: {exc}",
                hint=_DETECTOR_HINT,
            ) from None
    return _event_array(data, width, f"the circuit has {plural(width, 'detector')}", _DETECTOR_HINT)


def _event_array(
    data: Any, width: int | None, expected: str = "", hint: str | None = None
) -> Events:
    """The events of an array of shots by bits. ``expected`` says what has ``width`` bits."""
    rows = np.asarray(data)
    if rows.ndim != 2:
        raise NoiseVaultError(f"the events have shape {rows.shape}; give an array of shots by bits")
    if width is not None and rows.shape[1] != width:
        raise NoiseVaultError(
            f"the data has {plural(rows.shape[1], 'bit')} in each shot, and {expected}", hint=hint
        )
    if len(rows) == 0:
        raise NoiseVaultError("the events hold no shot")
    if rows.dtype != bool:
        wrong = rows[~np.isin(rows, (0, 1))]
        if wrong.size:
            raise NoiseVaultError(
                f"the events hold the value {wrong[0].item()!r}; give 0 or 1 for each bit"
            )
    return Events.of(rows)


def _check_samples(samples: int) -> None:
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError(f"samples={samples!r}: give a number of samples of 1 or more")


def _histogram(m: Events) -> np.ndarray:
    return np.bincount(m.rows.sum(axis=1), weights=m.times, minlength=m.rows.shape[1] + 1)


def _variance(probs: np.ndarray) -> float:
    k = np.arange(len(probs))
    mean = float(probs @ k)
    return max(0.0, float(probs @ k**2) - mean**2)


def _test_dict(test: RuleTest, sampling: Sampling) -> dict[str, Any]:
    return {
        **test.to_dict(),
        "sampling_ratio": sampling.ratio,
        "no_verdict": sampling.reason,
    }


def _test_phrase(test: RuleTest, sampling: Sampling, a: str, b: str, few: str) -> str:
    """One test in words. ``few`` says where a forecast can have too few samples."""
    if test.outcome == "identical":
        return "no difference (the forecasts give the same scores)"
    stats = (
        ""
        if test.z is None or test.p_value is None
        else f" (z = {test.z:.2f}, p = {test.p_value:.2g})"
    )
    if sampling.reason == "few_samples":
        return (
            f"no verdict, because a forecast has fewer than {MIN_SAMPLES} samples {few};"
            f" use more samples{stats}"
        )
    if sampling.reason == "sampling_variance":
        return (
            "no verdict, because the sampling error of the forecasts is larger than the shot"
            f" error; use more samples{stats}"
        )
    if test.difference is None or test.z is None:
        return "no test, because the score difference is the same in each shot"
    if test.better == "neither":
        return f"no difference at the 5% level{stats}"
    winner, loser = (a, b) if test.better == "a" else (b, a)
    return f"{winner} is better by {_number(abs(test.difference))} per shot than {loser}{stats}"


def _count(value: float) -> str:
    return f"{value:.3f}" if value >= 0.1 else f"{value:.3g}"


def _share(value: float) -> str:
    return f"{value:.3f}" if value >= 0.001 or value == 0 else f"{value:.2g}"


def _ratio(value: float, measured: float) -> str:
    return "-" if measured <= 0 else f"{value / measured:.2f}"


def _chunk(chunk: int, width: int) -> int:
    return max(1, min(chunk, _CHUNK_BITS // max(width, 1)))


def _bits(ideal: str) -> np.ndarray:
    return np.array([bit == "1" for bit in ideal])


def _smoothed(k: int, samples: int) -> float:
    return (k + 0.5) / (samples + 1)


def _rows(circuit: MeasuredCircuit) -> tuple[np.ndarray, np.ndarray]:
    """The measured bit strings as rows of booleans, and the shots of each row."""
    keys = list(circuit.counts)
    raw = np.frombuffer("".join(keys).encode("ascii"), dtype=np.uint8)
    rows = raw.reshape(len(keys), len(circuit.qubits)) == ord("1")
    return rows, np.array([circuit.counts[k] for k in keys], dtype=float)


def _merged(
    measured: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray, shots: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The cells that a model expects ``MIN_EXPECTED`` times or more, then one last cell for
    the other cells together, when there are other cells."""
    rare = shots * np.maximum(probs_a, probs_b) < MIN_EXPECTED
    if not rare.any():
        return measured, probs_a, probs_b
    keep = ~rare
    return (
        np.append(measured[keep], measured[rare].sum()),
        np.append(probs_a[keep], probs_a[rare].sum()),
        np.append(probs_b[keep], probs_b[rare].sum()),
    )


def _cells(
    m: Events, fa: Sampled, fb: Sampled, *, merge: bool
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    """The ``none``, ``count`` and ``bit`` cells of one set of bits for models a and b."""
    per_shot = _histogram(m)
    right = float(per_shot[0])
    none = (
        np.array([right, m.shots - right]),
        np.array([fa.none, 1 - fa.none]),
        np.array([fb.none, 1 - fb.none]),
    )
    count = (
        _merged(per_shot, fa.count, fb.count, m.shots) if merge else (per_shot, fa.count, fb.count)
    )
    bit = (m.times, _bit_scores(m.rows, fa.rate), _bit_scores(m.rows, fb.rate))
    return none, count, bit


def _bit_scores(rows: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """The per-shot score of each measured row: the sum of the log marginal of each bit. The
    rows go in blocks, so that the memory stays low."""
    event, none = np.log(probs), np.log1p(-probs)
    block = max(1, _CHUNK_BITS // max(rows.shape[1], 1))
    parts = [
        np.where(rows[i : i + block], event, none).sum(axis=1) for i in range(0, len(rows), block)
    ]
    return np.concatenate(parts) if parts else np.zeros(0)


def _scores(
    none: list[tuple[np.ndarray, ...]],
    count: list[tuple[np.ndarray, ...]],
    bit: list[tuple[np.ndarray, ...]],
    model: int,
) -> Scores:
    def mean(cells: list[tuple[np.ndarray, ...]], rule: str) -> float:
        total = shots = 0.0
        for row in cells:
            measured, probs = row[0], row[model]
            score = np.log(probs) if rule == "log" else 2 * probs - probs @ probs
            total += float(measured @ score)
            shots += float(measured.sum())
        return total / shots

    shots = sum(float(times.sum()) for times, *_ in bit)
    return Scores(
        none_log=mean(none, "log"),
        none_quadratic=mean(none, "quadratic"),
        count_log=mean(count, "log"),
        count_quadratic=mean(count, "quadratic"),
        bit_log=sum(float(row[0] @ row[model]) for row in bit) / shots,
    )


def _shot_test(sets: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> RuleTest:
    """The paired z-test of the per-shot score differences. Each set gives the shots of each
    measured row and the score of each row under model a and under model b.

    The outcome is ``identical`` only when each difference is 0. Each shot can have the same
    difference that is not 0. Then the shot variance is 0, and the result keeps the difference
    with the standard error 0 and no z. :func:`with_sampling` then adds the sampling variance
    and decides on the test."""
    total, variance, shots, scale = 0.0, 0.0, 0.0, 0.0
    for times, score_a, score_b in sets:
        d = score_a - score_b
        n = float(times.sum())
        mean = float(times @ d) / n
        spread = float(times @ (d - mean) ** 2) / (n - 1) if n > 1 else 0.0
        total += float(times @ d)
        variance += n * spread
        shots += n
        scale += n * float(np.abs(d).max())
    error = math.sqrt(variance)
    if scale == 0:
        return RuleTest("log", "identical", 0.0, 0.0, None, None, "neither")
    if error <= 1e-12 * scale:
        return RuleTest("log", "tested", total / shots, 0.0, None, None, "neither")
    z = total / error
    p_value = math.erfc(abs(z) / math.sqrt(2))
    better: Choice = "neither" if p_value >= LEVEL else "a" if z > 0 else "b"
    return RuleTest("log", "tested", total / shots, error / shots, z, p_value, better)
