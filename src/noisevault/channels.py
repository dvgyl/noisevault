"""Channels built from resolved noise, keeping the 0.1 math and Qiskit Aer's conventions.

Kraus operators are big-endian over their ``wires`` (the first wire is the most significant
tensor factor). Superoperators use column stacking, vec(K rho K^dagger) = (conj(K) (x) K) vec(rho),
which is also Qiskit's ``SuperOp`` convention.

A gate with a stated average infidelity gets depolarizing noise, then zero-temperature thermal
relaxation over its duration. NoiseVault solves for the depolarizing strength that gives the
composed channel the stated infidelity (Aer's residual rule). Sometimes no depolarizing
strength gives the stated infidelity. Relaxation alone can be more than the stated infidelity,
or the strongest depolarizing noise with relaxation can be less. Then NoiseVault keeps the
nearest channel and :attr:`GateChannels.inexact` is true, so the report records the stated and
achieved errors. A missing T2 means T2 = 2 T1 (no pure dephasing). NoiseVault clamps a T2
above 2 T1 to 2 T1. A ``pauli`` spec is the whole channel, so NoiseVault adds no relaxation
to it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from itertools import product
from typing import Literal

import numpy as np

from . import metrics
from .table import GateNoise, QubitNoise, refuse_disabled

_I2 = np.eye(2, dtype=complex)
_PAULIS = {
    "I": _I2,
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
_MATCH_TOL = 1e-12


@dataclass(frozen=True)
class ChannelSpec:
    kind: Literal["depolarizing", "thermal_relaxation", "pauli"]
    wires: tuple[int, ...]  # physical qubits, big-endian
    kraus: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class GateChannels:
    """Channels for one gate application, in the order they act, with bookkeeping."""

    gate: GateNoise
    channels: tuple[ChannelSpec, ...]
    requested: float | None  # stated average infidelity
    achieved: float  # average infidelity of the composed channel
    relaxation: float  # average infidelity of relaxation alone
    t2_clamped: tuple[int, ...]  # qubits whose T2 NoiseVault clamped to 2 T1

    @property
    def inexact(self) -> bool:
        """True when the composed channel cannot have the stated error, in either direction."""
        return self.requested is not None and abs(self.achieved - self.requested) > _MATCH_TOL


def gate_channels(gate: GateNoise, qubits: Sequence[QubitNoise]) -> GateChannels:
    """Channels for ``gate``. ``qubits`` are the resolved qubits in ``gate.qubits`` order."""
    if [q.index for q in qubits] != list(gate.qubits):
        raise ValueError(f"qubits {[q.index for q in qubits]} do not match {gate.qubits}")
    refuse_disabled(gate)
    if gate.state == "ideal":
        return GateChannels(gate, (), None, 0.0, 0.0, ())
    if gate.pauli is not None:
        r = metrics.avg_from_pauli(gate.pauli)
        spec = ChannelSpec("pauli", gate.qubits, tuple(pauli_kraus(gate.pauli)))
        return GateChannels(gate, (spec,), r, r, 0.0, ())

    n = len(qubits)
    duration = gate.duration_ns or 0.0
    thermal = [
        thermal_relaxation_kraus(q.t1_ns, q.t2_ns, duration, q.dephasing_rate_per_s) for q in qubits
    ]
    relaxation = _tensor_kraus(thermal)
    relaxation_error = max(0.0, 1.0 - average_gate_fidelity(relaxation))
    residual = _residual_depolarizing(gate.avg_infidelity, relaxation, n)
    depolarizing = depolarizing_kraus(residual, n)

    channels = []
    if len(depolarizing) > 1:
        channels.append(ChannelSpec("depolarizing", gate.qubits, tuple(depolarizing)))
    for q, kraus in zip(qubits, thermal, strict=True):
        if len(kraus) > 1:
            channels.append(ChannelSpec("thermal_relaxation", (q.index,), tuple(kraus)))
    achieved = 1.0 - average_gate_fidelity(_compose(relaxation, depolarizing))
    clamped = tuple(q.index for q in qubits if q.t2_clamped)
    return GateChannels(
        gate, tuple(channels), gate.avg_infidelity, max(achieved, 0.0), relaxation_error, clamped
    )


def thermal_relaxation_kraus(
    t1_ns: float | None,
    t2_ns: float | None,
    duration_ns: float,
    dephasing_rate_per_s: float | None = None,
) -> list[np.ndarray]:
    """Amplitude damping then pure dephasing at rate 1/T2 - 1/(2 T1), zero temperature.

    A ``dephasing_rate_per_s`` adds Z errors at that rate, so the Z error probability is
    (1 - exp(-2 rate duration)) / 2. Two idle times in sequence then give the same channel as
    one idle time of their sum.
    """
    rate = dephasing_rate_per_s or 0.0
    if duration_ns <= 0 or (t1_ns is None and t2_ns is None and rate == 0):
        return [_I2.copy()]
    t1 = np.inf if t1_ns is None else max(float(t1_ns), 1e-12)
    if t2_ns is None:
        t2 = np.inf if not np.isfinite(t1) else 2.0 * t1
    else:
        t2 = max(float(t2_ns), 1e-12)
        if np.isfinite(t1):
            t2 = min(t2, 2.0 * t1)
    duration = float(duration_ns)

    # The survival amplitude comes from exp, not from sqrt(1 - gamma): when duration >> T1,
    # 1 - gamma rounds to 0 and the small remaining coherence is lost.
    survival = 1.0 if not np.isfinite(t1) else float(np.exp(-duration / (2.0 * t1)))
    gamma = 0.0 if not np.isfinite(t1) else float(np.clip(-np.expm1(-duration / t1), 0.0, 1.0))
    amplitude = [
        np.array([[1.0, 0.0], [0.0, survival]], dtype=complex),
        np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=complex),
    ]
    inverse_t1 = 0.0 if not np.isfinite(t1) else 1.0 / t1
    inverse_t2 = 0.0 if not np.isfinite(t2) else 1.0 / t2
    pure_rate = max(0.0, inverse_t2 - inverse_t1 / 2.0)
    z_error = -float(np.expm1(-2.0 * rate * duration * 1e-9)) / 2.0
    coherence = float(np.exp(-duration * pure_rate)) * (1.0 - 2.0 * z_error)
    flip = float(np.clip((1.0 - coherence) / 2.0, 0.0, 0.5))
    phase = [np.sqrt(1.0 - flip) * _I2, np.sqrt(flip) * _PAULIS["Z"]]
    return _compose(phase, amplitude)


def depolarizing_kraus(avg_infidelity: float | None, num_qubits: int) -> list[np.ndarray]:
    """(1 - lambda) rho + lambda I/d with r = lambda (d - 1)/d, clipped to the CPTP range."""
    if avg_infidelity is None or avg_infidelity <= 0:
        return [np.eye(2**num_qubits, dtype=complex)]
    r = float(np.clip(avg_infidelity, 0.0, metrics.max_avg_infidelity(num_qubits)))
    lam = min(
        metrics.depolarizing_from_avg(r, num_qubits), metrics.max_depolarizing_param(num_qubits)
    )
    paulis = _pauli_basis(num_qubits)
    weights = [lam / len(paulis)] * len(paulis)
    weights[0] += 1.0 - lam
    return [np.sqrt(max(w, 0.0)) * p for w, p in zip(weights, paulis, strict=True)]


def pauli_kraus(pauli: Sequence[float]) -> list[np.ndarray]:
    """Kraus operators of a Pauli channel given its non-identity probabilities."""
    n = metrics.pauli_arity(len(pauli))
    probs = [1.0 - float(np.sum(pauli)), *map(float, pauli)]
    return [np.sqrt(max(p, 0.0)) * op for p, op in zip(probs, _pauli_basis(n), strict=True)]


def average_gate_fidelity(kraus: Iterable[np.ndarray]) -> float:
    """Average gate fidelity to the identity of a trace-preserving channel."""
    operators = [np.asarray(op, dtype=complex) for op in kraus]
    d = operators[0].shape[0]
    entanglement = sum(abs(np.trace(op)) ** 2 for op in operators) / d**2
    return float((d * entanglement + 1.0) / (d + 1.0))


def superoperator(channels: Iterable[ChannelSpec], wires: Sequence[int]) -> np.ndarray:
    """Column-stacking superoperator of ``channels`` applied in order, over ``wires``."""
    wires = tuple(wires)
    position = {w: i for i, w in enumerate(wires)}
    n = len(wires)
    total = np.eye(4**n, dtype=complex)
    for channel in channels:
        places = [position[w] for w in channel.wires]
        ops = channel.kraus
        if places != list(range(n)):
            ops = tuple(_embed(k, places, n) for k in channel.kraus)
        total = sum(np.kron(np.conj(k), k) for k in ops) @ total
    return total


def pauli_twirl(channels: Iterable[ChannelSpec], wires: Sequence[int]) -> tuple[float, ...]:
    """Non-identity Pauli probabilities of the twirled composition (metrics label order)."""
    n = len(wires)
    d = 2**n
    s = superoperator(channels, wires).reshape(d, d, d, d)
    choi = s.transpose(3, 1, 2, 0).reshape(d * d, d * d)
    out = []
    for label in metrics.pauli_labels(n):
        v = _kron_all(_PAULIS[c] for c in label).flatten(order="F")
        out.append(max(float(np.real(np.conj(v) @ choi @ v)) / d**2, 0.0))
    return tuple(out)


def readout_matrix(qubit: QubitNoise) -> np.ndarray | None:
    """M[measured, prepared] = [[1-a, b], [a, 1-b]] with a = P(1|0), b = P(0|1); None if unknown."""
    if qubit.readout is None:
        return None
    a, b = qubit.readout
    return np.array([[1.0 - a, b], [a, 1.0 - b]])


def _residual_depolarizing(
    requested: float | None, relaxation: list[np.ndarray], num_qubits: int
) -> float | None:
    """Average infidelity of the depolarizing part so the composition matches ``requested``."""
    fidelity = average_gate_fidelity(relaxation)
    relaxation_error = max(0.0, 1.0 - fidelity)
    if requested is None or requested <= relaxation_error:
        return None
    d = 2**num_qubits
    target = min(float(requested), metrics.max_avg_infidelity(num_qubits))
    denominator = d * fidelity - 1.0
    if denominator <= 0:
        return None
    strength = d * (target - relaxation_error) / denominator
    strength = float(np.clip(strength, 0.0, metrics.max_depolarizing_param(num_qubits)))
    return metrics.avg_from_depolarizing(strength, num_qubits)


def _compose(after: Iterable[np.ndarray], before: Iterable[np.ndarray]) -> list[np.ndarray]:
    before = list(before)
    return [np.asarray(a) @ np.asarray(b) for a in after for b in before]


def _tensor_kraus(channels: list[list[np.ndarray]]) -> list[np.ndarray]:
    result = [np.array([[1.0 + 0.0j]])]
    for channel in channels:
        result = [np.kron(left, right) for left in result for right in channel]
    return result


def _kron_all(ops: Iterable[np.ndarray]) -> np.ndarray:
    result = np.array([[1.0 + 0.0j]])
    for op in ops:
        result = np.kron(result, op)
    return result


@cache
def _pauli_basis(num_qubits: int) -> tuple[np.ndarray, ...]:
    # Each gate resolution needs the basis, so the basis is built one time. The arrays are shared.
    basis = [_kron_all(_PAULIS[c] for c in labels) for labels in product("IXYZ", repeat=num_qubits)]
    for op in basis:
        op.setflags(write=False)
    return tuple(basis)


def _embed(op: np.ndarray, places: list[int], n: int) -> np.ndarray:
    """``op`` acting on qubit positions ``places`` (in that order) of an n-qubit register."""
    rest = [i for i in range(n) if i not in places]
    full = np.kron(op, np.eye(2 ** len(rest)))
    perm = list(np.argsort(places + rest))
    tensor = full.reshape([2] * (2 * n)).transpose(perm + [n + p for p in perm])
    return tensor.reshape(2**n, 2**n)
