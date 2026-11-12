"""The stabilizer score of two noise models on the wide Clifford circuits of a run.

A Clifford circuit of ``noisevault.suite`` has more qubits than the reference simulator holds,
so ``noisevault.wide`` samples the forecast of each model in Stim. A circuit has one ideal bit
string. The score compares the measured shots with the forecast in three ways:

- ``success``: two cells, the ideal bit string and every other bit string.
- ``wrong``: one cell for each number of wrong bits, 0 to n. The cells that each model expects
  fewer than ``MIN_EXPECTED`` times in the measured shots become one last cell.
- ``qubit``: the score of a shot is the sum over the qubits of the log probability of its bit
  (right or wrong) under the marginal of the model for that qubit.

``success`` and ``wrong`` use the log rule, the quadratic rule and the paired test of
``noisevault.score`` on their cells. ``qubit`` uses a paired z-test of the per-shot score
differences (see :func:`paired`).

Each forecast probability comes from ``samples`` Stim shots with add-half smoothing:
(k + 1/2) / (samples + 1) for a probability of two outcomes. For the ``wrong`` cells, the
smoothing adds 1/2 to each cell and divides by the sum, so that the cells add up to 1. Both
models use the same seed, which makes the sampling error of the two forecasts similar.
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
from noisevault.score import LEVEL, Choice, RuleTest, _log, _quadratic, _test

SAMPLES = 200_000
MIN_EXPECTED = 5
PREFIX = "clifford_"


@dataclass(frozen=True)
class Forecast:
    """The forecast of one model for one circuit, from ``samples`` Stim shots.

    ``success`` is the probability of the ideal bit string, ``wrong[k]`` the probability of k
    wrong bits, and ``qubit[q]`` the probability that the bit of circuit qubit q is wrong.
    """

    circuit: str
    ideal: str
    samples: int
    success: float
    wrong: np.ndarray
    qubit: np.ndarray


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
    """The scores of models a and b and the paired test of each score."""

    a: Scores
    b: Scores
    success_log: RuleTest
    success_quadratic: RuleTest
    wrong_log: RuleTest
    wrong_quadratic: RuleTest
    qubit_log: RuleTest


def forecast(
    profile: Profile, circuit: PlannedCircuit, *, samples: int = SAMPLES, seed: int = 0
) -> Forecast:
    """The smoothed forecast of ``profile`` for the Clifford circuit ``circuit``."""
    ideal = wide.ideal_bits(circuit)
    bits = wide.sample(profile, circuit, samples, seed=seed)
    wrong = bits != np.array([bit == "1" for bit in ideal])
    per_shot = wrong.sum(axis=1)
    cells = np.bincount(per_shot, minlength=len(ideal) + 1) + 0.5
    return Forecast(
        circuit=circuit.name,
        ideal=ideal,
        samples=samples,
        success=_smoothed(int(np.sum(per_shot == 0)), samples),
        wrong=cells / cells.sum(),
        qubit=(wrong.sum(axis=0) + 0.5) / (samples + 1),
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
    ``success`` and ``wrong`` use the paired test of ``noisevault.score`` (its ``_test``,
    without the grouping of rare outcomes, because ``wrong`` merges its rare cells first).

    ``qubit`` uses a paired z-test. Each shot gives one difference d = s_a - s_b of the two
    per-shot scores. The test divides the sum of d by its standard error, the square root of
    the sum over the circuits of the shots times the sample variance of d in the circuit, and
    takes the two-sided p-value from the standard normal distribution. The test assumes:

    - The shots are independent, and the shots of one circuit have one distribution.
    - Both models were fixed before the counts were seen, or fitted on other counts.
    - Each circuit has enough shots that the mean of d is near normal. Each smoothed
      probability is above 0, so d is bounded and has a finite variance.

    Raises CountsError for a name that ``counts`` does not hold or for no circuit.
    """
    chosen = _chosen(counts, circuits)
    a_forecasts = [forecast(a, c, samples=samples, seed=seed) for c in chosen]
    b_forecasts = [forecast(b, c, samples=samples, seed=seed) for c in chosen]
    return compare(chosen, a_forecasts, b_forecasts)


def compare(
    circuits: Sequence[MeasuredCircuit], a: Sequence[Forecast], b: Sequence[Forecast]
) -> Paired:
    """The scores and the paired tests of :func:`paired` from forecasts made before, one of
    model a and one of model b for each circuit. Each test sums over the circuits, which are
    independent."""
    cells = [_cells(c, fa, fb, merge=True) for c, fa, fb in zip(circuits, a, b, strict=True)]
    success, wrong, qubit = (list(kind) for kind in zip(*cells, strict=True))
    return Paired(
        a=_scores(success, wrong, qubit, 1),
        b=_scores(success, wrong, qubit, 2),
        success_log=_test("log", success, _log, group_rare=False),
        success_quadratic=_test("quadratic", success, _quadratic, group_rare=False),
        wrong_log=_test("log", wrong, _log, group_rare=False),
        wrong_quadratic=_test("quadratic", wrong, _quadratic, group_rare=False),
        qubit_log=_shot_test(qubit),
    )


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


def seed_of(digest: str, model: str) -> int:
    """A seed from the SHA-256 of a counts file and a model name, so that a result repeats."""
    return int(hashlib.sha256(f"{digest}\n{model}".encode()).hexdigest()[:16], 16)


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
    each measured row and the score of each row under model a and under model b."""
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
    if error <= 1e-12 * scale:
        return RuleTest("log", "identical", 0.0, 0.0, None, None, "neither")
    z = total / error
    p_value = math.erfc(abs(z) / math.sqrt(2))
    better: Choice = "neither" if p_value >= LEVEL else "a" if z > 0 else "b"
    return RuleTest("log", "tested", total / shots, error / shots, z, p_value, better)
