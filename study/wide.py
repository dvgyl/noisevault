"""The stabilizer score of two noise models on the wide Clifford circuits of a run.

A Clifford circuit of ``noisevault.suite`` has more qubits than the reference simulator holds,
so the forecast of each model comes from Stim samples. ``noisevault.events`` holds the method:
an event is a wrong bit, a measured bit that differs from the one ideal bit string of the
circuit. This module keeps the names of the study for the three scores:

- ``success``: two cells, the ideal bit string and every other bit string (``none`` in
  ``noisevault.events``).
- ``wrong``: one cell for each number of wrong bits, 0 to n (``count``).
- ``qubit``: the sum over the qubits of the log probability of the bit of each qubit under the
  marginal of the model (``bit``).

``success`` and ``wrong`` use the log rule and the quadratic rule on their cells. Only the log
rule of ``wrong`` and ``qubit`` give a verdict, because only their tests hold the sampling
variance of both forecasts. ``study.analyze`` gives each model its own seed, so the sampling
errors of two forecasts are independent. The smoothing, the chunks and the tests are those of
``noisevault.events``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from noisevault import events, wide
from noisevault.counts import MeasuredCircuit, MeasuredCounts, PlannedCircuit
from noisevault.errors import CountsError
from noisevault.events import (
    CHUNK,
    MIN_EXPECTED,
    MIN_SAMPLES,
    Sampling,
    _merged,
    _rows,
    _shot_test,
    cell_sampling_variance,
    seed_of,
    with_sampling,
)
from noisevault.events import _bit_scores as _qubit_scores
from noisevault.events import bit_sampling_variance as qubit_sampling_variance
from noisevault.events import simulate_wide as simulate
from noisevault.profile import Profile
from noisevault.score import RuleTest

SAMPLES = 4_000_000
PREFIX = "clifford_"

__all__ = [
    "CHUNK",
    "MIN_EXPECTED",
    "MIN_SAMPLES",
    "SAMPLES",
    "Forecast",
    "Paired",
    "Sampling",
    "Scores",
    "_merged",
    "_qubit_scores",
    "_rows",
    "_shot_test",
    "cell_sampling_variance",
    "compare",
    "few_cell_samples",
    "few_qubit_samples",
    "forecast",
    "measured",
    "paired",
    "qubit_sampling_variance",
    "scores",
    "seed_of",
    "simulate",
    "with_sampling",
]


@dataclass(frozen=True)
class Forecast:
    """The forecast of one model for one circuit, from ``samples`` Stim shots.

    ``success`` is the probability of the ideal bit string. ``wrong[k]`` is the probability of k
    wrong bits. ``qubit[q]`` is the probability that the bit of circuit qubit q is wrong.
    ``covariance`` is the covariance matrix of the wrong-bit values of the qubits over the
    samples, without smoothing. ``qubit_samples[q]`` is the number of samples with a wrong bit
    of qubit q, and ``wrong_samples[k]`` the number of samples with k wrong bits.
    """

    circuit: str
    ideal: str
    sampled: events.Sampled

    @property
    def samples(self) -> int:
        return self.sampled.samples

    @property
    def success(self) -> float:
        return self.sampled.none

    @property
    def wrong(self) -> np.ndarray:
        return self.sampled.count

    @property
    def qubit(self) -> np.ndarray:
        return self.sampled.rate

    @property
    def covariance(self) -> np.ndarray | None:
        return self.sampled.covariance

    @property
    def qubit_samples(self) -> np.ndarray | None:
        return self.sampled.rate_samples

    @property
    def wrong_samples(self) -> np.ndarray | None:
        return self.sampled.count_samples


@dataclass(frozen=True)
class Scores:
    """The mean scores of one model per shot, over the shots of the scored circuits."""

    success_log: float
    success_quadratic: float
    wrong_log: float
    wrong_quadratic: float
    qubit_log: float


@dataclass(frozen=True)
class Paired:
    """The scores of models a and b, and the paired test of each score that gives a verdict.

    The two tests are of the score of each qubit (``qubit_log``) and of the log score of the
    number of wrong bits (``wrong_log``). ``qubit_sampling`` and ``wrong_sampling`` give the size
    of the sampling variance.
    """

    a: Scores
    b: Scores
    wrong_log: RuleTest
    qubit_log: RuleTest
    qubit_sampling: Sampling
    wrong_sampling: Sampling


def forecast(
    profile: Profile,
    circuit: PlannedCircuit,
    *,
    samples: int = SAMPLES,
    seed: int = 0,
    chunk: int = CHUNK,
) -> Forecast:
    """The smoothed forecast of ``profile`` for the Clifford circuit ``circuit``."""
    found = events.circuit_forecast(profile, circuit, samples=samples, seed=seed, chunk=chunk)
    return Forecast(circuit=circuit.name, ideal=wide.ideal_bits(circuit), sampled=found)


def paired(
    a: Profile,
    b: Profile,
    counts: MeasuredCounts,
    *,
    circuits: Sequence[str] | None = None,
    samples: int = SAMPLES,
    seed: int = 0,
) -> Paired:
    """Score models ``a`` and ``b`` on the named wide Clifford ``circuits`` of ``counts``.

    Without ``circuits``, the score takes every circuit whose name starts with ``clifford_``.
    Each forecast takes its own seed from ``seed``, the model (a or b) and the name of the
    circuit. Thus the sampling errors of two forecasts are independent.

    Raises CountsError for a name that ``counts`` does not hold or for no circuit.
    """
    chosen = _chosen(counts, circuits)
    found = {
        label: [
            forecast(model, c, samples=samples, seed=seed_of(str(seed), label, c.name))
            for c in chosen
        ]
        for label, model in (("a", a), ("b", b))
    }
    return compare(chosen, found["a"], found["b"])


def compare(
    circuits: Sequence[MeasuredCircuit], a: Sequence[Forecast], b: Sequence[Forecast]
) -> Paired:
    """The scores and the paired tests of :func:`paired` from forecasts made before.

    Give one forecast of model a and one of model b for each circuit. The method is
    ``noisevault.events.compare``.
    """
    found = events.compare(
        [_events(c, fa) for c, fa in zip(circuits, a, strict=True)],
        [f.sampled for f in a],
        [f.sampled for f in b],
    )
    return Paired(
        a=_study(found.a),
        b=_study(found.b),
        wrong_log=found.count_log,
        qubit_log=found.bit_log,
        qubit_sampling=found.bit_sampling,
        wrong_sampling=found.count_sampling,
    )


def few_qubit_samples(found: Forecast) -> bool:
    """True when a qubit of the forecast has fewer than ``MIN_SAMPLES`` sampled wrong bits or
    sampled right bits."""
    return events.few_rate_samples(found.sampled)


def few_cell_samples(found: Forecast, circuit: MeasuredCircuit, fa: Forecast, fb: Forecast) -> bool:
    """True when a cell of the ``wrong`` score of ``circuit`` holds a measured shot and has fewer
    than ``MIN_SAMPLES`` samples in the forecast ``found``."""
    return events.few_count_samples(found.sampled, _events(circuit, found), fa.sampled, fb.sampled)


def scores(circuit: MeasuredCircuit, found: Forecast) -> Scores:
    """The mean scores of one model on one circuit. The function does not merge the ``wrong``
    cells, so the scores of two models on one circuit use the same cells."""
    return _study(events.scores([_events(circuit, found)], [found.sampled]))


def measured(circuit: MeasuredCircuit, ideal: str) -> tuple[float, float]:
    """The measured frequency of the ideal bit string ``ideal`` and the measured mean fraction
    of wrong bits."""
    keys, times = _rows(circuit)
    wrong_bits = keys != np.array([bit == "1" for bit in ideal])
    right = float(times[~wrong_bits.any(axis=1)].sum())
    return right / circuit.shots, float(times @ wrong_bits.mean(axis=1)) / circuit.shots


def _cells(
    circuit: MeasuredCircuit, fa: Forecast, fb: Forecast, *, merge: bool
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    """The ``success``, ``wrong`` and ``qubit`` cells of one circuit for models a and b."""
    return events._cells(_events(circuit, fa), fa.sampled, fb.sampled, merge=merge)


def _events(circuit: MeasuredCircuit, found: Forecast) -> events.Events:
    return events.circuit_events(circuit, found.ideal)


def _study(found: events.Scores) -> Scores:
    return Scores(
        success_log=found.none_log,
        success_quadratic=found.none_quadratic,
        wrong_log=found.count_log,
        wrong_quadratic=found.count_quadratic,
        qubit_log=found.bit_log,
    )


def _chosen(counts: MeasuredCounts, names: Sequence[str] | None) -> list[MeasuredCircuit]:
    if names is None:
        names = [c.name for c in counts.circuits if c.name.startswith(PREFIX)]
    by_name = {c.name: c for c in counts.circuits}
    unknown = next((name for name in names if name not in by_name), None)
    if unknown is not None:
        raise CountsError(f"the counts have no circuit {unknown!r}")
    if not names:
        raise CountsError(
            "the counts hold no wide Clifford circuit",
            hint="give counts of a run of noisevault.suite on a device of 20 qubits or more",
        )
    return [by_name[name] for name in dict.fromkeys(names)]
