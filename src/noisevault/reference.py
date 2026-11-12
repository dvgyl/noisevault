from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import inf, pi
from typing import Any, cast

import numpy as np

from . import gates
from .channels import readout_matrix
from .conversion import UnknownGates, idle_channel, native_name, resolve_op
from .errors import TooWideError
from .layout import normalize_layout
from .profile import Profile
from .report import Report
from .table import refuse_disabled

MAX_QUBITS = 10
# The rotations an ms gate equals at some phases, as the exports read XX and YY gates.
_MS_ROTATIONS = ("rxx", "ryy")
_I4 = np.eye(4, dtype=complex)

# A superoperator here acts on the density matrix of its wires. Each wire has a pair of indices
# (row, column), and the pairs come in wire order. Then the superoperator of one wire is a 4 x 4
# matrix, and np.kron of such matrices acts on several wires.
# A piece is a superoperator and the circuit wires it acts on.
_Piece = tuple[np.ndarray, tuple[int, ...]]


@dataclass(frozen=True)
class Op:
    """One gate on circuit qubits ``qubits`` (indices ``0..n-1``) with numeric ``params``.

    ``Op("delay", (q,), (duration_ns,))`` idles circuit qubit ``q`` for ``duration_ns`` nanoseconds.
    The duration is finite and 0 or more.
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
    """Outcome probabilities of the noisy circuit, all qubits measured at the end.

    Raises TooWideError for more than ``MAX_QUBITS`` qubits, because the density matrix has
    4**n entries.
    """
    if num_qubits > MAX_QUBITS:
        raise TooWideError(
            f"the reference simulator holds 1 to {MAX_QUBITS} qubits, not {num_qubits}",
            hint="give a circuit on fewer qubits, or use the stabilizer forecast of"
            " noisevault.wide for a Clifford circuit",
        )
    if num_qubits < 1:
        raise ValueError(f"the reference simulator handles 1..{MAX_QUBITS} qubits")
    ops = list(ops)
    physical = normalize_layout(range(num_qubits), layout, profile)
    circuit_of = {p: c for c, p in physical.items()}
    # A caller that gives a report gets one event for each op. Each gate on each locus is
    # resolved one time, and a later op on that locus gives the report the same calls again.
    replay = report is not None
    report = report or Report.start(profile, "reference", None, unknown_gates=unknown_gates)
    report.record_effects(profile.effects)
    table = profile.table
    unitaries: dict[tuple[str, tuple[float, ...]], tuple[np.ndarray, str]] = {}
    resolved: dict[tuple[str, tuple[int, ...]], tuple[Any, _Tape]] = {}
    steps: dict[tuple[Any, ...], tuple[tuple[_Piece, ...], _Tape]] = {}

    state = _State(num_qubits)
    for op in ops:
        if op.name == "delay":
            (q,), (duration_ns,) = op.qubits, op.params
            if not 0 <= duration_ns < inf:
                raise ValueError(
                    f"the delay on circuit qubit {q} needs a finite duration of 0 ns or more, "
                    f"not {duration_ns} ns"
                )
            wait = ("delay", physical[q], duration_ns)
            if wait not in steps:
                refuse_disabled(table.gate("delay", (physical[q],)))
                tape = _Tape(report)
                channel = idle_channel(table, physical[q], duration_ns, cast(Report, tape))
                pieces = () if channel is None else ((_superop(channel.kraus, 1), (q,)),)
                steps[wait] = (pieces, tape)
            elif replay:
                steps[wait][1].replay(report)
            state.apply(steps[wait][0])
            continue
        key = (op.name, tuple(op.params))
        if key not in unitaries:
            info = gates.lookup(op.name)
            if info is None or info.unitary is None:
                raise ValueError(f"the reference simulator has no unitary for {op.name!r}")
            unitary = info.unitary(*op.params)
            unitaries[key] = (unitary, charged_as(profile, op.name, unitary))
        step = (*key, op.qubits)
        if step not in steps:
            unitary, charged = unitaries[key]
            if not _fits(unitary, op.qubits, num_qubits):
                # The unitary acts before the noise is resolved. Thus a malformed op raises the
                # error of _apply, as before the superoperators.
                _apply(state.flush(), [unitary], op.qubits, num_qubits)
            targets = tuple(physical[q] for q in op.qubits)
            locus = (charged, targets)
            if locus not in resolved:
                tape = _Tape(report)
                built = resolve_op(
                    table, charged, targets, unknown_gates=unknown_gates, report=cast(Report, tape)
                )
                resolved[locus] = (built, tape)
            elif replay:
                resolved[locus][1].replay(report)
            built, tape = resolved[locus]
            channels = [
                (channel.kraus, tuple(circuit_of[w] for w in channel.wires))
                for channel in built.channels
            ]
            steps[step] = (_compose([((unitary,), op.qubits), *channels]), tape)
        elif replay:
            steps[step][1].replay(report)
        state.apply(steps[step][0])
    rho = state.flush()
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


class _Tape:
    """A report that keeps each call it passes on, so that a later op can repeat the calls.

    A report counts some events once for each call, so an op that reuses a resolution must give
    the report the same calls that a new resolution gives.
    """

    def __init__(self, report: Report) -> None:
        self._report = report
        self._calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Any:
        found = getattr(self._report, name)
        if not callable(found):
            return found

        def call(*args: Any, **kwargs: Any) -> Any:
            self._calls.append((name, args, kwargs))
            return found(*args, **kwargs)

        return call

    def replay(self, report: Report) -> None:
        for name, args, kwargs in self._calls:
            getattr(report, name)(*args, **kwargs)


class _State:
    """The density matrix, with the 1-wire superoperators that wait for the next op on a wire.

    A 1-wire superoperator waits, and joins the next superoperator on its wire. Then a deep
    circuit needs about one contraction of the density matrix for each multi-qubit gate.
    """

    def __init__(self, n: int) -> None:
        self.n = n
        self.rho = np.zeros((2,) * (2 * n), dtype=complex)
        self.rho[(0,) * (2 * n)] = 1.0
        self.waiting: dict[int, np.ndarray] = {}

    def apply(self, pieces: Iterable[_Piece]) -> None:
        for matrix, wires in pieces:
            if len(wires) == 1:
                (w,) = wires
                before = self.waiting.get(w)
                self.waiting[w] = matrix if before is None else matrix @ before
                continue
            before = [self.waiting.pop(w, None) for w in wires]
            if any(b is not None for b in before):
                matrix = matrix @ _kron_all(_I4 if b is None else b for b in before)
            self.rho = _contract(self.rho, matrix, wires, self.n)

    def flush(self) -> np.ndarray:
        for w, matrix in self.waiting.items():
            self.rho = _contract(self.rho, matrix, (w,), self.n)
        self.waiting.clear()
        return self.rho


def _contract(rho: np.ndarray, matrix: np.ndarray, wires: Sequence[int], n: int) -> np.ndarray:
    k = len(wires)
    axes = [a for w in wires for a in (w, n + w)]
    out = np.tensordot(matrix.reshape((2,) * (4 * k)), rho, axes=(list(range(2 * k, 4 * k)), axes))
    return np.moveaxis(out, list(range(2 * k)), axes)


def _compose(parts: Sequence[tuple[Sequence[np.ndarray], tuple[int, ...]]]) -> tuple[_Piece, ...]:
    """Pieces for Kraus operator sets that act in order on their wires.

    A set on wires that the open piece already holds joins it. Another set starts a new piece.
    """
    pieces: list[_Piece] = []
    for kraus, wires in parts:
        if pieces and set(wires) <= set(pieces[-1][1]):
            matrix, held = pieces[-1]
            places = [held.index(w) for w in wires]
            pieces[-1] = (_superop(kraus, len(held), places) @ matrix, held)
        else:
            pieces.append((_superop(kraus, len(wires)), wires))
    return tuple(pieces)


def _superop(
    kraus: Iterable[np.ndarray], k: int, places: Sequence[int] | None = None
) -> np.ndarray:
    """The superoperator over ``k`` wires of Kraus operators on wire positions ``places``."""
    kraus = [np.asarray(op, dtype=complex) for op in kraus]
    m = len(kraus[0]).bit_length() - 1
    # sum K (x) conj(K) has the row indices of all wires first. Each wire's pair comes together.
    total = sum(np.kron(op, op.conj()) for op in kraus)
    order = [i for w in range(m) for i in (w, m + w)]
    matrix = (
        np.asarray(total)
        .reshape((2,) * (4 * m))
        .transpose(order + [2 * m + i for i in order])
        .reshape(4**m, 4**m)
    )
    places = list(range(m)) if places is None else list(places)
    if places == list(range(k)):
        return matrix
    rest = [i for i in range(k) if i not in places]
    full = np.kron(matrix, np.eye(4 ** len(rest), dtype=complex))
    perm = list(np.argsort(places + rest))
    return full.reshape([4] * (2 * k)).transpose(perm + [k + p for p in perm]).reshape(4**k, 4**k)


def _kron_all(matrices: Iterable[np.ndarray]) -> np.ndarray:
    # np.kron has a large fixed cost, and a deep circuit needs this product for most gates.
    result, *rest = matrices
    for matrix in rest:
        rows, cols = result.shape
        result = (result[:, None, :, None] * matrix[None, :, None, :]).reshape(
            rows * len(matrix), cols * len(matrix)
        )
    return result


def _fits(unitary: np.ndarray, qubits: tuple[int, ...], n: int) -> bool:
    return (
        unitary.shape == (2 ** len(qubits),) * 2
        and len(set(qubits)) == len(qubits)
        and all(0 <= q < n for q in qubits)
    )
