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

This module needs Stim (the ``stim`` extra) for :func:`circuit_forecast` and
:func:`detector_forecast`.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from . import __version__, wide
from .counts import COUNTS_FORMAT, Execution, MeasuredCounts, ProfileBinding
from .profile import Profile
from .score import LEVEL, Choice, RuleTest, _log, _test

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

NoVerdict = Literal["few_samples", "sampling_variance", "no_variance"]


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


def _histogram(m: Events) -> np.ndarray:
    return np.bincount(m.rows.sum(axis=1), weights=m.times, minlength=m.rows.shape[1] + 1)


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
