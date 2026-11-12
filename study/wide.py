"""The stabilizer score of two noise models on the wide Clifford circuits of a run.

A Clifford circuit of ``noisevault.suite`` has more qubits than the reference simulator holds,
so ``noisevault.wide`` samples the forecast of each model in Stim. A circuit has one ideal bit
string. The score compares the measured shots with the forecast in three ways:

- ``success``: two cells, the ideal bit string and every other bit string.
- ``wrong``: one cell for each number of wrong bits, 0 to n. The cells that each model expects
  fewer than ``MIN_EXPECTED`` times in the measured shots become one last cell.
- ``qubit``: the score of a shot is the sum over the qubits of the log probability of its bit
  (right or wrong) under the marginal of the model for that qubit.

``success`` and ``wrong`` use the log rule and the quadratic rule on their cells. Only the log
rule of ``wrong`` (the paired test of ``noisevault.score``) and ``qubit`` (a paired z-test of
the per-shot score differences, see :func:`paired`) give a verdict, because only their tests
hold the sampling variance. The other scores are values with no verdict, on each size.

Each forecast probability comes from ``samples`` Stim shots with add-half smoothing:
(k + 1/2) / (samples + 1) for a probability of two outcomes. For the ``wrong`` cells, the
smoothing adds 1/2 to each cell and divides by the sum, so that the cells add up to 1. The
samples come in chunks of ``CHUNK`` shots, so that the memory stays low.

The seed of a forecast is a parameter. ``study.analyze`` gives each model its own seed, so the
sampling errors of two forecasts are independent. The sampling error of a forecast is the same
for each measured shot, so it does not become smaller with more shots. The tests of the
``qubit`` score and of the log rule of the ``wrong`` score therefore add the sampling variance of
both forecasts to the shot variance (see :func:`compare`).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from noisevault import __version__, wide
from noisevault.counts import (
    COUNTS_FORMAT,
    Execution,
    MeasuredCircuit,
    MeasuredCounts,
    PlannedCircuit,
    ProfileBinding,
)
from noisevault.errors import CountsError
from noisevault.profile import Profile
from noisevault.score import LEVEL, Choice, RuleTest, _log, _test

SAMPLES = 4_000_000
CHUNK = 250_000
MIN_EXPECTED = 5
PREFIX = "clifford_"


@dataclass(frozen=True)
class Forecast:
    """The forecast of one model for one circuit, from ``samples`` Stim shots.

    ``success`` is the probability of the ideal bit string, ``wrong[k]`` the probability of k
    wrong bits, and ``qubit[q]`` the probability that the bit of circuit qubit q is wrong.
    ``covariance`` is the covariance matrix of the wrong-bit values of the qubits over the
    samples (each value is 0 or 1), without smoothing. None means independent qubits.
    ``qubit_samples[q]`` is the number of samples with a wrong bit of qubit q, and
    ``wrong_samples[k]`` the number of samples with k wrong bits. None means not known.
    """

    circuit: str
    ideal: str
    samples: int
    success: float
    wrong: np.ndarray
    qubit: np.ndarray
    covariance: np.ndarray | None = None
    qubit_samples: np.ndarray | None = None
    wrong_samples: np.ndarray | None = None


# The fewest samples on each side of a marginal, and in a cell with a measured shot, for which
# the delta method of the sampling variance is valid.
MIN_SAMPLES = 100


@dataclass(frozen=True)
class Scores:
    """The mean scores of one model per shot, over the shots of the scored circuits."""

    success_log: float
    success_quadratic: float
    wrong_log: float
    wrong_quadratic: float
    qubit_log: float


@dataclass(frozen=True)
class Sampling:
    """The sampling variance of the two forecasts in one test, the shot variance of the test,
    and their ratio. Both variances are of the sum of the score differences over the shots.
    ``reason`` says why the test gives no verdict for a reason of the samples:
    ``few_samples`` (a marginal or a cell has fewer than ``MIN_SAMPLES`` samples) or
    ``sampling_variance`` (the sampling variance is larger than the shot variance) or
    ``no_variance`` (both variances are 0, so there is no test). It is None otherwise."""

    variance: float
    shot_variance: float
    reason: str | None = None

    @property
    def ratio(self) -> float | None:
        return None if self.shot_variance <= 0 else self.variance / self.shot_variance


@dataclass(frozen=True)
class Paired:
    """The scores of models a and b, and the paired test of each score that gives a verdict.

    Only two scores give a verdict, because only their tests hold the sampling variance of both
    forecasts: the score of each qubit (``qubit_log``) and the log score of the number of wrong
    bits (``wrong_log``). ``qubit_sampling`` and ``wrong_sampling`` give the size of the
    sampling variance. A test whose sampling variance is larger than its shot variance gives no
    verdict. The score of the ideal bit string and the quadratic score of the number of wrong
    bits are values only: the scores ``a`` and ``b`` of each model.
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
    """The smoothed forecast of ``profile`` for the Clifford circuit ``circuit``.

    The samples come in chunks of ``chunk`` shots. The readout error of each qubit applies to
    the sampled bits as in ``noisevault.wide.sample``, which gives the same bits for one chunk.
    """
    ideal = wide.ideal_bits(circuit)
    target = np.array([bit == "1" for bit in ideal])
    noisy, readout = wide.noisy_circuit(profile, circuit)
    rng = np.random.default_rng(seed)
    sampler = noisy.compile_sampler(seed=int(rng.integers(2**63)))
    cells = np.zeros(len(ideal) + 1)
    qubit = np.zeros(len(ideal))
    both = np.zeros((len(ideal), len(ideal)))
    left = samples
    while left > 0:
        size = min(chunk, left)
        bits = sampler.sample(size)
        for q, (to_one, to_zero) in enumerate(readout):
            bits[:, q] ^= rng.random(size) < np.where(bits[:, q], to_zero, to_one)
        wrong = bits != target
        cells += np.bincount(wrong.sum(axis=1), minlength=len(ideal) + 1)
        qubit += wrong.sum(axis=0)
        # float32 counts each sum of one chunk exactly, because a chunk has fewer than 2^24 rows.
        values = wrong.astype(np.float32)
        both += values.T @ values
        left -= size
    smoothed = cells + 0.5
    mean = qubit / samples
    return Forecast(
        circuit=circuit.name,
        ideal=ideal,
        samples=samples,
        success=_smoothed(int(cells[0]), samples),
        wrong=smoothed / smoothed.sum(),
        qubit=(qubit + 0.5) / (samples + 1),
        covariance=both / samples - np.outer(mean, mean),
        qubit_samples=qubit,
        wrong_samples=cells,
    )


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
    The log rule of ``wrong`` uses the paired test of ``noisevault.score`` (its ``_test``,
    without the grouping of rare outcomes, because ``wrong`` merges its rare cells first).

    ``qubit`` uses a paired z-test. Each shot gives one difference d = s_a - s_b of the two
    per-shot scores. The test divides the sum of d by its standard error, the square root of
    the sum over the circuits of the shots times the sample variance of d in the circuit, and
    takes the two-sided p-value from the standard normal distribution. The test assumes:

    - The shots are independent, and the shots of one circuit have one distribution.
    - Both models were fixed before the counts were seen, or fitted on other counts.
    - Each circuit has enough shots that the mean of d is near normal. Each smoothed
      probability is above 0, so d is bounded and has a finite variance.

    Each forecast takes its own seed from ``seed``, the model (a or b) and the name of the
    circuit, so that the sampling errors of two forecasts are independent.

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
    """The scores and the paired tests of :func:`paired` from forecasts made before, one of
    model a and one of model b for each circuit. Each test sums over the circuits, which are
    independent.

    The test of the ``qubit`` score and the log rule of the ``wrong`` score add the sampling
    variance of both forecasts to the variance of the summed difference. For one model and one
    circuit with N shots and S samples, the term is:

    - ``qubit``: a^T C a / S, with a_j = N (f_j - q_j) / (q_j (1 - q_j)) for the measured
      fraction f and the forecast q of a wrong bit, and C the covariance matrix of the wrong-bit
      values of the qubits over the samples (see :func:`qubit_sampling_variance`).
    - ``wrong``: (N^2 / S) (sum over the cells of f^2 / p - 1), with the measured fraction f and
      the forecast p of a cell.

    A test gives no verdict when the sampling variance is larger than the shot variance. The
    ``success`` scores and the quadratic ``wrong`` score are values with no test.
    """
    cells = [_cells(c, fa, fb, merge=True) for c, fa, fb in zip(circuits, a, b, strict=True)]
    success, wrong, qubit = (list(kind) for kind in zip(*cells, strict=True))
    qubit_variance = wrong_variance = 0.0
    for c, fa, fb, (measured, pa, pb) in zip(circuits, a, b, wrong, strict=True):
        keys, times = _rows(c)
        fraction = times @ (keys != np.array([bit == "1" for bit in fa.ideal])) / c.shots
        for found in (fa, fb):
            qubit_variance += qubit_sampling_variance(
                c.shots, found.samples, fraction, found.qubit, found.covariance
            )
        for found, probs in ((fa, pa), (fb, pb)):
            wrong_variance += cell_sampling_variance(c.shots, found.samples, measured, probs)
    shots = float(sum(c.shots for c in circuits))
    few_qubits = any(few_qubit_samples(found) for found in (*a, *b))
    few_cells = any(
        few_cell_samples(found, c, fa, fb)
        for c, fa, fb in zip(circuits, a, b, strict=True)
        for found in (fa, fb)
    )
    qubit_test, qubit_sampling = with_sampling(
        _shot_test(qubit), qubit_variance, shots, few=few_qubits
    )
    wrong_test, wrong_sampling = with_sampling(
        _test("log", wrong, _log, group_rare=False), wrong_variance, shots, few=few_cells
    )
    return Paired(
        a=_scores(success, wrong, qubit, 1),
        b=_scores(success, wrong, qubit, 2),
        wrong_log=wrong_test,
        qubit_log=qubit_test,
        qubit_sampling=qubit_sampling,
        wrong_sampling=wrong_sampling,
    )


def qubit_sampling_variance(
    shots: float,
    samples: int,
    measured: np.ndarray,
    forecast: np.ndarray,
    covariance: np.ndarray | None = None,
) -> float:
    """The sampling variance of the summed ``qubit`` score of one model on one circuit.

    The value is a^T C a / S, with a_j = N (f_j - q_j) / (q_j (1 - q_j)) for the measured
    fraction f and the forecast q of a wrong bit of qubit j. C is the covariance matrix of the
    wrong-bit values of the qubits over the samples. The marginals come from one set of samples,
    so their errors are correlated. Without ``covariance``, C is diagonal with q (1 - q), and the
    value is (N^2 / S) times the sum over the qubits of (f - q)^2 / (q (1 - q)).
    """
    f, q = np.asarray(measured, dtype=float), np.asarray(forecast, dtype=float)
    a = shots * (f - q) / (q * (1 - q))
    if covariance is None:
        return float((a**2 * q * (1 - q)).sum()) / samples
    return float(a @ np.asarray(covariance, dtype=float) @ a) / samples


def cell_sampling_variance(
    shots: float, samples: int, measured: np.ndarray, forecast: np.ndarray
) -> float:
    """The sampling variance of the summed log score of one model on the cells of one circuit:
    (N^2 / S) (sum over the cells of f^2 / p - 1). ``measured`` gives the shots of each cell."""
    f = np.asarray(measured, dtype=float) / shots
    p = np.asarray(forecast, dtype=float)
    return shots**2 / samples * max(0.0, float((f**2 / p).sum()) - 1)


def few_qubit_samples(found: Forecast) -> bool:
    """True when a qubit of the forecast has fewer than ``MIN_SAMPLES`` sampled wrong bits or
    sampled right bits."""
    if found.qubit_samples is None:
        return False
    wrong = np.asarray(found.qubit_samples)
    return bool((np.minimum(wrong, found.samples - wrong) < MIN_SAMPLES).any())


def few_cell_samples(found: Forecast, circuit: MeasuredCircuit, fa: Forecast, fb: Forecast) -> bool:
    """True when a cell of the ``wrong`` score of ``circuit`` holds a measured shot and has fewer
    than ``MIN_SAMPLES`` samples in the forecast ``found``. The cells are the merged cells of
    the test of ``fa`` and ``fb``."""
    if found.wrong_samples is None:
        return False
    keys, times = _rows(circuit)
    wrong_bits = keys != np.array([bit == "1" for bit in found.ideal])
    measured = np.bincount(wrong_bits.sum(axis=1), weights=times, minlength=len(found.ideal) + 1)
    samples = np.asarray(found.wrong_samples, dtype=float)
    # The rule of _merged: the cells that each model expects fewer than MIN_EXPECTED times
    # become one last cell, with the sum of their shots and of their samples.
    rare = circuit.shots * np.maximum(fa.wrong, fb.wrong) < MIN_EXPECTED
    if rare.any():
        measured = np.append(measured[~rare], measured[rare].sum())
        samples = np.append(samples[~rare], samples[rare].sum())
    return bool(((measured > 0) & (samples < MIN_SAMPLES)).any())


def with_sampling(
    test: RuleTest, variance: float, shots: float, *, few: bool = False
) -> tuple[RuleTest, Sampling]:
    """``test`` over ``shots`` shots, with the sampling ``variance`` added to the variance of its
    summed difference.

    The test gives no verdict when ``variance`` is larger than the shot variance, or when
    ``few`` says that a marginal or a cell has too few samples for the sampling variance. A test
    with no value stays as it is.
    """
    if test.outcome != "tested" or test.difference is None or test.standard_error is None:
        return test, Sampling(variance, 0.0, "few_samples" if few else None)
    total = test.difference * shots
    shot_variance = (test.standard_error * shots) ** 2
    error = math.sqrt(shot_variance + variance)
    if error == 0:
        # No shot variance and no sampling variance: no test, but the difference stays.
        reason = "few_samples" if few else "no_variance"
        found = RuleTest(test.rule, "tested", test.difference, 0.0, None, None, "neither")
        return found, Sampling(variance, shot_variance, reason)
    z = total / error
    p_value = math.erfc(abs(z) / math.sqrt(2))
    too_large = variance > shot_variance
    reason = "few_samples" if few else "sampling_variance" if too_large else None
    sampling = Sampling(variance, shot_variance, reason)
    better: Choice = "neither" if p_value >= LEVEL or reason else "a" if z > 0 else "b"
    found = RuleTest(test.rule, "tested", test.difference, error / shots, z, p_value, better)
    return found, sampling


def scores(circuit: MeasuredCircuit, found: Forecast) -> Scores:
    """The mean scores of one model on one circuit. The ``wrong`` cells are not merged, so the
    scores of two models on one circuit use the same cells."""
    success, wrong, qubit = _cells(circuit, found, found, merge=False)
    return _scores([success], [wrong], [qubit], 1)


def measured(circuit: MeasuredCircuit, ideal: str) -> tuple[float, float]:
    """The measured frequency of the ideal bit string ``ideal`` and the measured mean fraction
    of wrong bits."""
    keys, times = _rows(circuit)
    wrong_bits = keys != np.array([bit == "1" for bit in ideal])
    right = float(times[~wrong_bits.any(axis=1)].sum())
    return right / circuit.shots, float(times @ wrong_bits.mean(axis=1)) / circuit.shots


def seed_of(digest: str, model: str, circuit: str) -> int:
    """A seed from the SHA-256 of a counts file, a model name and a circuit name, so that a
    result repeats and each model and circuit has its own sampling error."""
    return int(hashlib.sha256(f"{digest}\n{model}\n{circuit}".encode()).hexdigest()[:16], 16)


def _cells(
    circuit: MeasuredCircuit, fa: Forecast, fb: Forecast, *, merge: bool
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    """The ``success``, ``wrong`` and ``qubit`` cells of one circuit for models a and b."""
    keys, times = _rows(circuit)
    wrong_bits = keys != np.array([bit == "1" for bit in fa.ideal])
    per_shot = np.bincount(wrong_bits.sum(axis=1), weights=times, minlength=len(fa.ideal) + 1)
    right = float(per_shot[0])
    success = (
        np.array([right, circuit.shots - right]),
        np.array([fa.success, 1 - fa.success]),
        np.array([fb.success, 1 - fb.success]),
    )
    wrong = (
        _merged(per_shot, fa.wrong, fb.wrong, circuit.shots)
        if merge
        else (per_shot, fa.wrong, fb.wrong)
    )
    qubit = (times, _qubit_scores(wrong_bits, fa.qubit), _qubit_scores(wrong_bits, fb.qubit))
    return success, wrong, qubit


def simulate(
    truth: Profile,
    circuits: Sequence[PlannedCircuit],
    *,
    shots: int,
    seed: int,
    run_at: datetime | None = None,
) -> MeasuredCounts:
    """Counts of the Clifford ``circuits`` sampled from the stabilizer forecast of ``truth``,
    bound to its calibration, as ``noisevault.counts.simulate`` binds them."""
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


def _smoothed(k: int, samples: int) -> float:
    return (k + 0.5) / (samples + 1)


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


def _qubit_scores(wrong_bits: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """The per-shot score of each measured row: the sum of the log marginal of each bit."""
    return np.where(wrong_bits, np.log(probs), np.log1p(-probs)).sum(axis=1)


def _scores(
    success: list[tuple[np.ndarray, ...]],
    wrong: list[tuple[np.ndarray, ...]],
    qubit: list[tuple[np.ndarray, ...]],
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

    shots = sum(float(times.sum()) for times, *_ in qubit)
    return Scores(
        success_log=mean(success, "log"),
        success_quadratic=mean(success, "quadratic"),
        wrong_log=mean(wrong, "log"),
        wrong_quadratic=mean(wrong, "quadratic"),
        qubit_log=sum(float(row[0] @ row[model]) for row in qubit) / shots,
    )


def _shot_test(circuits: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> RuleTest:
    """The paired z-test of the per-shot score differences. Each circuit gives the shots of
    each measured row and the score of each row under model a and under model b.

    The outcome is ``identical`` only when each difference is 0. When each shot has the same
    difference that is not 0, the shot variance is 0 and the result keeps the difference, with
    the standard error 0 and no z. :func:`with_sampling` then decides on the test, after it adds
    the sampling variance."""
    total, variance, shots, scale = 0.0, 0.0, 0.0, 0.0
    for times, score_a, score_b in circuits:
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
