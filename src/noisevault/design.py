"""Choose the circuits and the shots of a hardware run for a budget.

A device with a price for each shot cannot run every circuit of ``noisevault.suite.suite``.
``design`` takes the circuits that tell the most about the gate factor for the money. The
measure is the Fisher information: how much one shot of a circuit narrows the estimate of the
gate factor and of the readout factor, when the profile with assumed factors is the device.

The result always holds the readout circuits and at least one test circuit. Every circuit gets
the same number of shots.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .counts import PlannedCircuit
from .errors import NoiseVaultError
from .metrics import METRIC_KEYS
from .reference import probabilities
from .suite import ASSUMED_GATE_FACTOR, role

if TYPE_CHECKING:
    from .profile import Profile

MIN_SHOTS = 100
_STEP = 1e-3
# The finite differences of a rank-1 matrix leave 1 - correlation**2 near 1e-10.
_RANK_TOLERANCE = 1e-6
_READOUT = ("readout", "readout_1")


@dataclass(frozen=True)
class Price:
    """The price of a run, in one currency: ``task`` for each task and ``shot`` for each shot.

    With ``one_task``, all circuits run in one task. Otherwise each circuit is one task.
    """

    task: float
    shot: float
    one_task: bool = False

    def cost(self, circuits: int, shots: int) -> float:
        tasks = 1 if self.one_task else circuits
        return tasks * self.task + circuits * shots * self.shot


@dataclass(frozen=True)
class Design:
    """The circuits of a run, the shots for each circuit, and what the run is predicted to give.

    ``gate_error`` and ``readout_error`` are the predicted relative standard errors of the gate
    factor and of the readout factor. Each one is None when the circuits do not identify the
    factor.
    """

    circuits: tuple[PlannedCircuit, ...]
    shots: int
    cost: float
    gate_error: float | None
    readout_error: float | None

    def summary(self) -> str:
        lines = [f"{len(self.circuits)} circuits, {self.shots} shots each, cost {self.cost:.2f}"]
        lines += [f"  {c.name} ({role(c.name)})" for c in self.circuits]
        for axis, error in (("gate", self.gate_error), ("readout", self.readout_error)):
            found = "not identified" if error is None else f"{100 * error:.1f}%"
            lines.append(f"predicted standard error of the {axis} factor: {found}")
        return "\n".join(lines)


def information(
    profile: Profile,
    circuit: PlannedCircuit,
    *,
    gate_factor: float = ASSUMED_GATE_FACTOR,
    readout_factor: float = 1.0,
) -> np.ndarray:
    """The Fisher information of one shot of ``circuit`` on the log gate factor and the log
    readout factor, as a 2 by 2 matrix, when the device is ``profile`` with the two factors.

    A profile that states no error of a kind has no factor for that kind, and that axis gets
    zero information.
    """
    base = profile.uncorrected()
    has_gates = _states_gate_error(base)
    has_readout = base.readout is not None or any(q.readout is not None for q in base.qubits)

    def probs(log_gate: float, log_readout: float) -> np.ndarray:
        block: dict[str, dict[str, float]] = {}
        if has_gates:
            block["gates"] = {"factor": gate_factor * math.exp(log_gate)}
        if has_readout:
            block["readout"] = {"factor": readout_factor * math.exp(log_readout)}
        scaled = base.model_copy(update={"unmodeled_error": block or None})
        return probabilities(scaled, circuit.ops, len(circuit.qubits), layout=circuit.qubits)

    def slope(present: bool, step: tuple[float, float]) -> np.ndarray:
        if not present:
            return np.zeros_like(center)
        return (probs(*step) - probs(-step[0], -step[1])) / (2 * _STEP)

    center = probs(0.0, 0.0)
    slopes = np.stack([slope(has_gates, (_STEP, 0.0)), slope(has_readout, (0.0, _STEP))])
    seen = center > 1e-15
    return (slopes[:, seen] / center[seen]) @ slopes[:, seen].T


def design(
    profile: Profile,
    circuits: Sequence[PlannedCircuit],
    *,
    budget: float,
    price: Price,
    gate_factor: float = ASSUMED_GATE_FACTOR,
    tests: int = 1,
    min_shots: int = MIN_SHOTS,
    max_shots: int | None = None,
) -> Design:
    """The circuits from ``circuits`` and the shots that give the smallest predicted error of
    the gate factor for at most ``budget``.

    ``circuits`` are the circuits of ``suite``. The design keeps the readout circuits and
    ``tests`` test circuits, and adds ``train`` circuits while they lower the predicted error.
    Raises NoiseVaultError when ``budget`` does not pay ``min_shots`` shots of the smallest
    design.
    """
    by_name = {c.name: c for c in circuits}
    fixed = [by_name[name] for name in _READOUT if name in by_name]
    fixed += _tests(circuits, tests)
    pool = [c for c in circuits if role(c.name) == "train" and c.name not in _READOUT]
    if not pool:
        raise NoiseVaultError(
            "the circuits hold no train circuit with gates",
            hint="give the circuits of noisevault.suite.suite",
        )
    info = {c.name: information(profile, c, gate_factor=gate_factor) for c in (*fixed, *pool)}
    fits = [c for c in fixed if role(c.name) == "train"]

    chosen: list[PlannedCircuit] = []
    best: Design | None = None
    while pool:
        added = min(pool, key=lambda c: _gate_variance([*fits, *chosen, c], info))
        pool.remove(added)
        chosen.append(added)
        total = len(fixed) + len(chosen)
        shots = _shots(budget, price, total, max_shots)
        if shots < min_shots:
            break
        errors = _errors([*fits, *chosen], info, shots)
        found = Design(
            circuits=tuple(c for c in circuits if c in fixed or c in chosen),
            shots=shots,
            cost=price.cost(total, shots),
            gate_error=errors[0],
            readout_error=errors[1],
        )
        if best is None or _below(found.gate_error, best.gate_error):
            best = found
    if best is None:
        least = price.cost(len(fixed) + 1, min_shots)
        raise NoiseVaultError(
            f"a budget of {budget:g} does not pay {min_shots} shots of {len(fixed) + 1} circuits,"
            f" which cost {least:.2f}",
            hint="give a larger budget, or fewer shots with min_shots",
        )
    return best


def _states_gate_error(profile: Profile) -> bool:
    specs = [*profile.gates.values(), *profile.calibrations]
    return any(getattr(spec, key, None) is not None for spec in specs for key in METRIC_KEYS)


def _tests(circuits: Sequence[PlannedCircuit], count: int) -> list[PlannedCircuit]:
    """``count`` test circuits: first the deep circuits of the test seed, from the middle depth
    out, then the other test circuits in their order."""
    near = [c for c in circuits if role(c.name) == "test_near"]
    middle = (len(near) - 1) / 2
    from_middle = sorted(enumerate(near), key=lambda found: abs(found[0] - middle))
    far = [c for c in circuits if role(c.name) == "test_far"]
    return [*(c for _, c in from_middle), *far][:count]


def _shots(budget: float, price: Price, circuits: int, max_shots: int | None) -> int:
    tasks = 1 if price.one_task else circuits
    left = budget - tasks * price.task
    if left < 0:
        return 0
    shots = math.floor(left / (circuits * price.shot) + 1e-9) if price.shot > 0 else 10**9
    return shots if max_shots is None else min(shots, max_shots)


def _matrix(circuits: Sequence[PlannedCircuit], info: dict[str, np.ndarray]) -> np.ndarray:
    return sum((info[c.name] for c in circuits), np.zeros((2, 2)))


def _gate_variance(circuits: Sequence[PlannedCircuit], info: dict[str, np.ndarray]) -> float:
    variance = _variances(_matrix(circuits, info))[0]
    return math.inf if variance is None else variance


def _errors(
    circuits: Sequence[PlannedCircuit], info: dict[str, np.ndarray], shots: int
) -> tuple[float | None, float | None]:
    gate, readout = _variances(_matrix(circuits, info) * shots)
    return (
        None if gate is None else math.sqrt(gate),
        None if readout is None else math.sqrt(readout),
    )


def _variances(matrix: np.ndarray) -> tuple[float | None, float | None]:
    """The variance of each log factor from its Fisher information, with the other factor
    unknown. A factor that the information does not identify gives None.

    A diagonal entry of exactly 0 means that the circuits do not see that factor, so the other
    factor alone is identified. Otherwise the rank test uses normalized axes: when the
    correlation ``cross / sqrt(gate * readout)`` is 1 within ``_RANK_TOLERANCE``, the matrix has
    rank 1 and neither factor is identified.
    """
    gate, readout, cross = matrix[0, 0], matrix[1, 1], matrix[0, 1]
    if gate <= 0 and readout <= 0:
        return None, None
    if readout <= 0:
        return 1 / gate, None
    if gate <= 0:
        return None, 1 / readout
    det = gate * readout - cross * cross
    if det <= _RANK_TOLERANCE * gate * readout:
        return None, None
    return readout / det, gate / det


def _below(a: float | None, b: float | None) -> bool:
    return a is not None and (b is None or a < b)
