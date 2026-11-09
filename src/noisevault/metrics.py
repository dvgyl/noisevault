from __future__ import annotations

import math
from collections.abc import Sequence
from functools import cache
from itertools import product
from typing import Any, Literal

import numpy as np

MetricKind = Literal["avg_infidelity", "process_infidelity", "depolarizing_param", "pauli"]
METRIC_KEYS: tuple[MetricKind, ...] = (
    "avg_infidelity",
    "process_infidelity",
    "depolarizing_param",
    "pauli",
)
_TOL = 1e-12
_RATE_TOL = 1e-12


def dim(num_qubits: int) -> int:
    return 2**num_qubits


def max_avg_infidelity(num_qubits: int) -> float:
    d = dim(num_qubits)
    return d / (d + 1)


def max_depolarizing_param(num_qubits: int) -> float:
    """Largest lambda of a CPTP depolarizing channel: d^2 / (d^2 - 1)."""
    d = dim(num_qubits)
    return d * d / (d * d - 1)


def process_from_avg(r: float, num_qubits: int) -> float:
    d = dim(num_qubits)
    return r * (d + 1) / d


def avg_from_process(e_pro: float, num_qubits: int) -> float:
    d = dim(num_qubits)
    return e_pro * d / (d + 1)


def avg_from_depolarizing(lam: float, num_qubits: int) -> float:
    d = dim(num_qubits)
    return lam * (d - 1) / d


def depolarizing_from_avg(r: float, num_qubits: int) -> float:
    d = dim(num_qubits)
    return r * d / (d - 1)


def pauli_arity(length: int) -> int:
    """Number of qubits of a Pauli vector with ``length`` non-identity entries."""
    n = round(math.log(length + 1, 4)) if length > 0 else 0
    if n < 1 or 4**n - 1 != length:
        raise ValueError(f"a Pauli vector has 3 (1 qubit) or 15 (2 qubits) entries, got {length}")
    return n


def avg_from_pauli(pauli: Sequence[float]) -> float:
    return avg_from_process(math.fsum(pauli), pauli_arity(len(pauli)))


def to_avg_infidelity(kind: MetricKind, value: float | Sequence[float], num_qubits: int) -> float:
    if kind == "pauli":
        return avg_from_pauli(value)  # type: ignore[arg-type]
    if kind == "process_infidelity":
        return avg_from_process(float(value), num_qubits)  # type: ignore[arg-type]
    if kind == "depolarizing_param":
        return avg_from_depolarizing(float(value), num_qubits)  # type: ignore[arg-type]
    return float(value)  # type: ignore[arg-type]


def check_metric(kind: MetricKind, value: float | Sequence[float], num_qubits: int) -> None:
    """Raise ValueError when ``value`` is outside the physical range of ``kind``."""
    d = dim(num_qubits)
    if kind == "pauli":
        entries = list(value)  # type: ignore[arg-type]
        if len(entries) != d * d - 1:
            raise ValueError(
                f"pauli for a {num_qubits}-qubit gate needs {d * d - 1} entries, got {len(entries)}"
            )
        if not all(p >= 0 for p in entries):  # also refuses NaN, which no comparison accepts
            raise ValueError("pauli entries must be >= 0")
        if any(p > 1 for p in entries):  # also keeps math.fsum from an overflow
            raise ValueError("pauli entries must be <= 1")
        if math.fsum(entries) > 1 + _TOL:
            raise ValueError(f"pauli entries sum to {math.fsum(entries)!r} > 1")
        return
    upper = {
        "avg_infidelity": max_avg_infidelity(num_qubits),
        "process_infidelity": 1.0,
        "depolarizing_param": max_depolarizing_param(num_qubits),
    }[kind]
    if not 0 <= value <= upper + _TOL:  # type: ignore[operator]
        raise ValueError(
            f"{kind} of a {num_qubits}-qubit gate must be in [0, {upper:.6g}], got {value!r}"
        )


def pauli_labels(num_qubits: int) -> tuple[str, ...]:
    """Non-identity Pauli labels in the order Pauli vectors use."""
    return tuple("".join(p) for p in product("IXYZ", repeat=num_qubits))[1:]


PAULI_1Q = pauli_labels(1)
PAULI_2Q = pauli_labels(2)


def uniform_pauli(r: float, num_qubits: int) -> tuple[float, ...]:
    """Pauli vector of the depolarizing channel with average infidelity ``r``."""
    d = dim(num_qubits)
    return (process_from_avg(r, num_qubits) / (d * d - 1),) * (d * d - 1)


_SWAPPED_2Q = tuple(PAULI_2Q.index(label[::-1]) for label in PAULI_2Q)


def swap_pauli_2q(pauli: Sequence[float]) -> tuple[float, ...]:
    """Pauli vector of the same channel with its two operands exchanged."""
    if len(pauli) != len(PAULI_2Q):
        raise ValueError(f"expected 15 entries, got {len(pauli)}")
    return tuple(pauli[i] for i in _SWAPPED_2Q)


def _anticommute(p: str, q: str) -> bool:
    return sum("I" not in (x, y) and x != y for x, y in zip(p, q, strict=True)) % 2 == 1


@cache
def _anticommutation(num_qubits: int) -> np.ndarray:
    labels = pauli_labels(num_qubits)
    return np.array([[_anticommute(p, q) for q in labels] for p in labels], dtype=float)


def unscalable(kind: MetricKind, value: Any, num_qubits: int) -> str | None:
    if kind == "pauli":
        rates = pauli_rates(value)
        if rates is None:
            return "at or past full depolarization"
        return "it has a negative Pauli-Lindblad rate" if min(rates) < -_RATE_TOL else None
    r = to_avg_infidelity(kind, value, num_qubits)
    return "at or past full depolarization" if depolarizing_from_avg(r, num_qubits) >= 1 else None


def readout_unscalable(pair: tuple[float, float]) -> str | None:
    return "no better than chance" if sum(pair) >= 1 else None


def scale_avg_infidelity(r: float, num_qubits: int, factor: float) -> float:
    if unscalable("avg_infidelity", r, num_qubits):
        return r
    lam = depolarizing_from_avg(r, num_qubits)
    return avg_from_depolarizing(-math.expm1(factor * math.log1p(-lam)), num_qubits)


def pauli_rates(pauli: Sequence[float]) -> tuple[float, ...] | None:
    anti = _anticommutation(pauli_arity(len(pauli)))
    deficit = 2 * anti @ np.asarray(pauli, dtype=float)
    if deficit.max() >= 1:
        return None
    return tuple(np.linalg.solve(anti, -np.log1p(-deficit) / 2).tolist())


def scale_pauli(pauli: Sequence[float], factor: float) -> tuple[float, ...]:
    if unscalable("pauli", pauli, pauli_arity(len(pauli))):
        return tuple(pauli)
    anti = _anticommutation(pauli_arity(len(pauli)))
    log_fidelities = np.log1p(-2 * anti @ np.asarray(pauli, dtype=float))
    deficit = -np.expm1(factor * log_fidelities)
    scaled = (2 * anti - 1) @ deficit / (len(pauli) + 1)
    return tuple(np.maximum(scaled, 0.0).tolist())


def scale_readout(pair: tuple[float, float], factor: float) -> tuple[float, float]:
    a, b = pair
    total = a + b
    if total == 0 or readout_unscalable(pair):
        return pair
    k = -math.expm1(factor * math.log1p(-total)) / total
    return (a * k, b * k)
