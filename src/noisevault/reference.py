from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import pi
from typing import Any

import numpy as np

from . import gates
from .channels import readout_matrix
from .conversion import UnknownGates, idle_channel, native_name, resolve_op
from .layout import normalize_layout
from .profile import Profile
from .report import Report
from .table import refuse_disabled

MAX_QUBITS = 10
# The rotations an ms gate equals at some phases, as the exports read XX and YY gates.
_MS_ROTATIONS = ("rxx", "ryy")


@dataclass(frozen=True)
class Op:
    """One gate on circuit qubits ``qubits`` (indices ``0..n-1``) with numeric ``params``.

    ``Op("delay", (q,), (duration_ns,))`` idles circuit qubit ``q`` for ``duration_ns`` nanoseconds.
    """

    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()


def probabilities(
    profile: Profile,
    ops: Iterable[Op],
    num_qubits: int,
    *,
    layout: Mapping[Any, int] | Sequence[int] | None = None,
    unknown_gates: UnknownGates = "typical",
    readout: bool = True,
    report: Report | None = None,
) -> np.ndarray:
    """Outcome probabilities of the noisy circuit, all qubits measured at the end."""
    if not 1 <= num_qubits <= MAX_QUBITS:
        raise ValueError(f"the reference simulator handles 1..{MAX_QUBITS} qubits")
    ops = list(ops)
    physical = normalize_layout(range(num_qubits), layout, profile)
    circuit_of = {p: c for c, p in physical.items()}
    report = report or Report.start(profile, "reference", None, unknown_gates=unknown_gates)
    report.record_effects(profile.effects)
    table = profile.table

    rho = np.zeros((2,) * (2 * num_qubits), dtype=complex)
    rho[(0,) * (2 * num_qubits)] = 1.0
    for op in ops:
        if op.name == "delay":
            (q,), (duration_ns,) = op.qubits, op.params
            refuse_disabled(table.gate("delay", (physical[q],)))
            channel = idle_channel(table, physical[q], duration_ns, report)
            if channel is not None:
                rho = _apply(rho, channel.kraus, (q,), num_qubits)
            continue
        info = gates.lookup(op.name)
        if info is None or info.unitary is None:
            raise ValueError(f"the reference simulator has no unitary for {op.name!r}")
        unitary = info.unitary(*op.params)
        rho = _apply(rho, [unitary], op.qubits, num_qubits)
        targets = tuple(physical[q] for q in op.qubits)
        built = resolve_op(
            table,
            charged_as(profile, op.name, unitary),
            targets,
            unknown_gates=unknown_gates,
            report=report,
        )
        for channel in built.channels:
            wires = tuple(circuit_of[w] for w in channel.wires)
            rho = _apply(rho, channel.kraus, wires, num_qubits)
    for c in range(num_qubits):
        refuse_disabled(table.gate("measure", (physical[c],)))

    dim = 2**num_qubits
    probs = np.real(np.diagonal(rho.reshape(dim, dim))).reshape((2,) * num_qubits)
    if readout:
        for c in range(num_qubits):
            matrix = readout_matrix(table.qubit(physical[c]))
            if matrix is not None:
                probs = np.moveaxis(np.tensordot(matrix, probs, axes=([1], [c])), 0, c)
    probs = np.clip(probs.reshape(dim), 0.0, None)
    return probs / probs.sum()


def outcome_bits(outcome: int, num_qubits: int) -> str:
    return format(outcome, f"0{num_qubits}b")


def charged_as(profile: Profile, name: str, unitary: np.ndarray) -> str:
    """The gate whose calibration an operation ``name`` with ``unitary`` takes, as the exports
    decide it."""
    return native_name(name, profile.gates, _rotation(name, unitary))


def _rotation(name: str, unitary: np.ndarray) -> str | None:
    """The rotation an ``ms`` gate with this unitary equals, or None."""
    if name != "ms":
        return None
    for rotation in _MS_ROTATIONS:
        for theta in (pi / 2, -pi / 2):
            v = gates.unitary(rotation, (theta,))
            if abs(abs(np.vdot(v, unitary)) - len(unitary)) < 1e-9:  # equal up to global phase
                return rotation
    return None


def _apply(
    rho: np.ndarray, kraus: Sequence[np.ndarray], wires: Sequence[int], n: int
) -> np.ndarray:
    """rho -> sum_k K rho K^dagger with each K acting on ``wires`` (big-endian)."""
    k = len(wires)
    row_axes = list(wires)
    col_axes = [n + w for w in wires]
    out = np.zeros_like(rho)
    for op in kraus:
        tensor = np.asarray(op, dtype=complex).reshape((2,) * (2 * k))
        left = np.tensordot(tensor, rho, axes=(list(range(k, 2 * k)), row_axes))
        left = np.moveaxis(left, list(range(k)), row_axes)
        right = np.tensordot(tensor.conj(), left, axes=(list(range(k, 2 * k)), col_axes))
        out += np.moveaxis(right, list(range(k)), col_axes)
    return out
