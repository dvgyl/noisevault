from __future__ import annotations

import re
import sys
from collections import Counter
from collections.abc import Callable, Container, Hashable, Iterable, Mapping, Sequence
from itertools import compress
from math import pi
from types import FrameType
from typing import Any, NamedTuple

import numpy as np

from ..errors import DisabledGateError, LayoutError, NoiseVaultError, install_hint

try:
    import pennylane as qml
except ImportError as exc:
    raise ImportError(f"the PennyLane export needs PennyLane: {install_hint('pennylane')}") from exc

from pennylane.measurements import (
    ClassicalShadowMP,
    CountsMP,
    ExpectationMP,
    MidMeasureMP,
    ProbabilityMP,
    SampleMP,
    ShadowExpvalMP,
    VarianceMP,
)
from pennylane.operation import Channel, Operation, Operator, StatePrepBase
from pennylane.ops.op_math import Adjoint, CompositeOp, Conditional, ControlledOp, SymbolicOp

from .. import gates
from ..channels import readout_matrix
from ..conversion import UnknownGates, native_name, resolve_op
from ..layout import can_measure, normalize_layout
from ..profile import Profile
from ..report import Report
from ..table import refuse_disabled

Layout = Mapping[Hashable, int] | Sequence[int]
Kraus = tuple[np.ndarray, ...]
PhysicalChannel = tuple[Kraus, tuple[int, ...]]  # Kraus operators (big-endian) and their qubits
EventCounts = tuple[tuple[str, str, int], ...]  # (event, key, count) added to the report
CacheEntry = tuple[tuple[PhysicalChannel, ...], EventCounts]


class Charge(NamedTuple):
    gate: str
    wires: tuple[Hashable, ...]


_CANONICAL = {info.pennylane: info.name for info in gates.GATES.values() if info.pennylane}
# Registry gates PennyLane has only as another operation at a fixed angle: operation name ->
# (gate, the angle read from the operation's parameters, the values that make it that gate).
# Angles compare modulo 2 pi, where both operations repeat up to a global phase.
_AT_ANGLE: dict[str, tuple[str, Callable[[Sequence[Any]], Any], tuple[float, ...]]] = {
    "IsingZZ": ("zz", lambda p: p[0], (pi / 2,)),
    "IsingXX": ("ms", lambda p: p[0], (pi / 2, -pi / 2)),  # ms(0, 0) and ms(pi, 0)
    "IsingYY": ("ms", lambda p: p[0], (pi / 2, -pi / 2)),  # ms(pi/2, pi/2) and ms(pi/2, -pi/2)
    "Rot": ("r", lambda p: p[0] + p[2], (0.0,)),  # Rot(a, theta, -a) = r(theta, pi/2 - a)
}


def _ms(phi0: float, phi1: float, wires: Any) -> Operator:
    if phi0 != 0 or phi1 != 0:
        raise ValueError(
            f"PennyLane has no Molmer-Sorensen gate, so operation_for('ms') builds only ms(0, 0),"
            f" which is qml.IsingXX(pi/2). Got ms({phi0}, {phi1})"
        )
    return qml.IsingXX(pi / 2, wires=wires)


_BUILDERS: dict[str, Callable[..., Operator]] = {
    "zz": lambda wires: qml.IsingZZ(pi / 2, wires=wires),
    "r": lambda theta, phi, wires: qml.Rot(pi / 2 - phi, theta, phi - pi / 2, wires=wires),
    "ms": _ms,
}
_ANGLE_TOL = 1e-9
_NOT_GATES = frozenset({"Barrier", "Snapshot", "GlobalPhase", "WireCut"})
_SHADOW_MEASUREMENTS = (ClassicalShadowMP, ShadowExpvalMP)
_READOUT_MEASUREMENTS = (
    ExpectationMP,
    VarianceMP,
    ProbabilityMP,
    SampleMP,
    CountsMP,
    *_SHADOW_MEASUREMENTS,
)
_EIGENVALUE_MEASUREMENTS = (ExpectationMP, VarianceMP, SampleMP, CountsMP)
_ROTATED_PAULIS = {"X": qml.PauliX, "Y": qml.PauliY}
_ADD_NOISE = ("pennylane.noise.add_noise", "add_noise")  # module and name of its tape transform
_EXECUTE = ("pennylane.workflow.execution", "execute")  # its `device` runs the tape
_NO_BASIS_FIX = (
    "measure Pauli words or computational-basis probabilities. For a Hamiltonian, wrap the"
    " QNode in qml.transforms.split_non_commuting before qml.add_noise"
)


class NoiseVaultPennyLaneModel(qml.NoiseModel):
    """A PennyLane noise model built from a NoiseVault profile.

    ``.report`` says what the export reproduced, approximated or omitted. The report counts
    events each time a circuit runs. ``.profile`` is the source profile.
    """

    def __init__(
        self,
        profile: Profile,
        *,
        layout: Layout | None = None,
        unknown_gates: UnknownGates = "typical",
        readout: bool = True,
    ) -> None:
        if unknown_gates not in ("typical", "error"):
            raise ValueError(f"unknown_gates={unknown_gates!r}: choose 'typical' or 'error'")
        self.profile = profile
        self.report = Report.start(
            profile,
            "pennylane",
            qml.__version__,
            layout=layout,
            unknown_gates=unknown_gates,
            readout=readout,
        )
        self.report.record_effects(profile.effects)
        _describe(self.report, readout)
        self._unknown_gates: UnknownGates = unknown_gates
        self._explicit_layout = layout is not None
        self._list_layout = self._explicit_layout and not isinstance(layout, Mapping)
        self._layout: dict[Hashable, int] = (
            {} if layout is None else normalize_layout(_labels(layout), layout, profile)
        )
        self._cache: dict[tuple[str, tuple[int, ...]], CacheEntry] = {}
        gate_map = {
            qml.BooleanFn(_is_operation, "NoiseVaultWireCheck"): self._check_wires,
            qml.BooleanFn(_is_gate, "NoiseVaultGate"): self._gate_noise,
            qml.BooleanFn(_is_mid_measure, "NoiseVaultMidMeasure"): self._mid_measure_noise,
        }
        meas_map = {qml.BooleanFn(_reads_out, "NoiseVaultReadout"): self._readout_noise}
        super().__init__(gate_map, meas_map=meas_map if readout else None)

    @property
    def model_map(self) -> dict:
        # add_noise reads this for every tape, so a tape with no operation and no readout
        # callback (measurements only, or readout=False) still has its wires checked.
        if _outer_frame(_ADD_NOISE) is not None:
            self._noised_tape()
        return super().model_map

    def _noised_tape(self) -> qml.tape.QuantumScript:
        """The tape qml.add_noise is noising, after checking every wire it uses against the layout
        and every qubit that a measurement reads against the profile's ``measure`` entry.

        Noise functions see one operation or measurement, but readout needs the whole tape and only
        add_noise's frame holds the tape. Each call gets the tape from that frame and does not
        keep it on the model. A composed model is a new plain qml.NoiseModel that shares only
        these functions.
        """
        frame = _outer_frame(_ADD_NOISE)
        if frame is None:
            raise RuntimeError(
                f"NoiseVault noise ran outside qml.add_noise (PennyLane {qml.__version__}),"
                " so NoiseVault cannot see the circuit's wires. Use qml.add_noise(qnode, model)"
            )
        # The first two parameters of add_noise are the tape and the model that add_noise
        # applies. That model, not self, decides readout, because composing can remove or add a
        # measurement map.
        tape, model = (frame.f_locals[name] for name in frame.f_code.co_varnames[:2])
        if model.meas_map and tape.shots.has_partitioned_shots:
            raise ValueError(
                "qml.add_noise keeps only part of a shot vector's results when readout noise is"
                " on. Run each shot count separately or pass readout=False"
            )
        for wire in tape.wires:
            self._physical(wire, len(tape.wires))
        read = [mp for mp in tape.measurements if _reads_out(mp)]
        measured = [wire for mp in read for wire in _read_wires(mp, tape)]
        if any(not mp.wires for mp in read):
            measured += self._device_wires(frame, tape)
        width = len({*tape.wires, *measured})
        for wire in measured:
            qubit = self._physical(wire, width)
            refuse_disabled(self.profile.table.gate("measure", (qubit,)))
        return tape

    def _device_wires(self, frame: FrameType, tape: qml.tape.QuantumScript) -> list[Hashable]:
        """The wires of the device, which a measurement without wires reads.

        Only a QNode run shows the device. Without the device, raise if a wire that the circuit
        does not use can map to a qubit that cannot measure.
        """
        execute = _outer_frame(_EXECUTE, frame)
        if execute is not None:
            return list(execute.f_locals["device"].wires or ())
        table = self.profile.table
        used = {self._layout[wire] for wire in tape.wires}
        possible = self._layout.values() if self._explicit_layout else range(table.num_qubits)
        for qubit in possible:
            if qubit in used or (can_measure(table, qubit) and not table.qubit(qubit).disabled):
                continue
            raise DisabledGateError(
                "a measurement without wires reads every device wire, and qml.add_noise on a"
                f" tape cannot see the device. A device wire can map to qubit {qubit}, which"
                " cannot measure in this profile",
                hint="pass wires= to the measurement, or apply qml.add_noise to the QNode",
            )
        return []

    def _check_wires(self, _: Operator, **__: Any) -> None:
        self._noised_tape()

    def physical_qubit(self, wire: Hashable) -> int:
        """The device qubit a circuit wire maps to. Integer wire ``i`` is qubit ``i`` by default."""
        return self._physical(wire, 0)

    def _physical(self, wire: Hashable, width: int) -> int:
        """``physical_qubit`` for a wire of a circuit with ``width`` wires."""
        if wire not in self._layout:
            if self._list_layout:
                last = len(self._layout) - 1
                raise LayoutError(
                    f"wire {wire!r} is not in the layout. A list layout covers wires 0 to {last}",
                    hint="extend the list",
                )
            if self._explicit_layout:
                raise LayoutError(
                    f"wire {wire!r} is not in the layout",
                    hint=f"add the wire: layout={{..., {wire!r}: <physical qubit>}}",
                )
            self._layout.update(normalize_layout([wire], None, self.profile, width=width))
        return self._layout[wire]

    def _gate_noise(self, op: Operator, **_: Any) -> None:
        gate = _unconditional(op)
        if gate is not op:
            self.report.approximate(
                "conditional gates",
                "gate noise applied whether or not the condition holds",
                "default.mixed cannot condition a channel on a mid-circuit measurement",
            )
        noise: list[tuple[Kraus, list[Hashable]]] = []
        noisy_wires: set[Hashable] = set()
        moved = False
        for name, wires in _charges(gate, self.profile.gates):
            moved |= not noisy_wires.isdisjoint(wires)
            physical = tuple(self.physical_qubit(w) for w in wires)
            wire_of = dict(zip(physical, wires, strict=True))
            for kraus, qubits in self._channels(name, physical):
                targets = [wire_of[q] for q in qubits]
                noise.append((kraus, targets))
                noisy_wires.update(targets)
        if moved:
            self.report.approximate(
                "operator arithmetic",
                "the noise of each gate in the decomposition, after the whole operator",
                "apply the gates separately to put the noise of each gate directly after that gate",
            )
        for kraus, wires in noise:
            qml.QubitChannel(list(kraus), wires=wires)

    def _mid_measure_noise(self, op: MidMeasureMP, **_: Any) -> None:
        qubit = self.physical_qubit(op.wires[0])
        refuse_disabled(self.profile.table.gate("measure", (qubit,)))
        if not op.reset:
            return
        refuse_disabled(self.profile.table.gate("reset", (qubit,)))
        error = self.profile.table.qubit(qubit).prep_error
        if error is None:
            self.report.mark_unknown(f"reset error on qubit {qubit}")
        elif error > 0:
            qml.BitFlip(error, wires=op.wires)

    def _channels(self, name: str, physical: tuple[int, ...]) -> tuple[PhysicalChannel, ...]:
        """Channels of one gate, from a cache. A cache hit adds the report events again.

        Channels do not depend on gate angles, so the key leaves them out and a trained
        circuit keeps hitting the cache as its parameters change.
        """
        key = (name, physical)
        if key in self._cache:
            channels, events = self._cache[key]
            for event, what, n in events:
                self.report.count(event, what, n)
            return channels
        before = {event: Counter(counts) for event, counts in self.report.events.items()}
        built = resolve_op(
            self.profile.table,
            name,
            physical,
            unknown_gates=self._unknown_gates,
            report=self.report,
        )
        channels = tuple((c.kraus, c.wires) for c in built.channels)
        self._cache[key] = (channels, _added_events(before, self.report.events))
        return channels

    def _readout_noise(self, mp: Any, **_: Any) -> None:
        if isinstance(mp, _SHADOW_MEASUREMENTS):
            self.report.omit("readout on classical shadow measurements")
            self.report.warn_once(
                "readout_skipped:shadow",
                f"no readout noise applied to {mp}. A classical shadow picks a random measurement"
                " basis for each shot. The noise model acts before the measurement, so the noise"
                " model cannot flip the bit read in that basis. To fix: measure the Pauli words"
                " you need with qml.expval or qml.sample, which get readout noise",
            )
            return
        if not mp.wires:
            self.report.approximate(
                "readout of measurements without wires",
                "applied to every wire the circuit's operations or measurements use",
                "a device wire that the circuit does not use reads out without error. Pass"
                " wires= to give that wire readout error",
            )
        basis = _measured_basis(mp.obs)
        if basis is None:
            self.report.omit("readout on observables not measured in one product basis")
            self.report.warn_once(
                f"readout_skipped:{mp.obs}",
                f"no readout noise applied to {mp}, because its observable is not measured in"
                f" one product basis. To fix: {_NO_BASIS_FIX}",
            )
            return
        if basis:
            self.report.approximate(
                "measurement basis change",
                "ideal rotation around the readout confusion",
                "add the rotation to the circuit to give it gate noise",
            )
        tape = self._noised_tape()
        for wire in _read_wires(mp, tape):
            if self._readout_matrix(wire) is None:
                self.report.mark_unknown(f"readout on qubit {self.physical_qubit(wire)}")
        rotations, wires = _shared_readout(mp, basis, tape)
        with qml.QueuingManager.stop_recording():
            undo = [qml.adjoint(gate, lazy=False) for gate in reversed(rotations)]
        for gate in rotations:
            qml.apply(gate)
        for wire in wires:
            matrix = self._readout_matrix(wire)
            if matrix is not None and not np.array_equal(matrix, np.eye(2)):
                qml.QubitChannel(confusion_kraus(matrix), wires=[wire])
        for gate in undo:
            qml.apply(gate)

    def _readout_matrix(self, wire: Hashable) -> np.ndarray | None:
        return readout_matrix(self.profile.table.qubit(self.physical_qubit(wire)))


def to_pennylane(
    profile: Profile,
    *,
    layout: Layout | None = None,
    unknown_gates: UnknownGates = "typical",
    readout: bool = True,
) -> NoiseVaultPennyLaneModel:
    """A noise model for ``qml.add_noise(qnode, model)`` on ``default.mixed``.

    ``layout`` maps circuit wires to physical qubits (a mapping, or a sequence where wire ``i``
    maps to ``layout[i]``). By default, an integer wire maps to the same qubit, and other wire
    labels need a layout. ``unknown_gates`` decides what a gate gets when the profile does not
    calibrate it: ``"typical"`` noise (with a warning and a report entry) or an error.
    ``readout=False`` leaves out readout errors.

    ``qml.add_noise`` at its default ``level="user"`` decomposes ``qml.adjoint`` gates and
    templates first, so each gate in the decomposition gets its own noise. To noise
    ``Adjoint(SX)``, ``Adjoint(S)`` and ``Adjoint(T)`` as the profile's sxdg, sdg and tdg, pass
    ``level="top"``.
    Operator arithmetic, such as ``qml.prod``, ``@``, ``qml.pow``, ``qml.exp`` or ``qml.ctrl``,
    gets the noise of the gates it decomposes into, after the whole operator.
    ``qml.pow(qml.RX(0.3, 0), 2)`` gets the noise of ``rx``. An operator with its own gate name,
    such as ``qml.CNOT`` or ``qml.CRX``, gets the noise of one gate. Arithmetic with no
    decomposition into gates, such as ``qml.sum``, raises ValueError. State preparation, such as
    ``qml.StatePrep`` or ``qml.QubitDensityMatrix``, is noiseless.

    ``qml.IsingZZ(pi/2)`` gets the noise of a profile's ``zz``. ``qml.IsingXX(+-pi/2)`` and
    ``qml.IsingYY(+-pi/2)`` get the noise of its ``ms``, and ``qml.Rot(a, theta, -a)`` gets the
    noise of its ``r``. :func:`operation_for` builds zz, r and ms(0, 0). See :func:`gate_name`.
    NoiseVault cannot compare traced angles, as under ``jax.jit``, so those operations then get
    ``rzz``, ``rxx`` or ``ryy`` noise or the typical-noise rule. A broadcast whose angles need the
    noise of different gates raises ValueError. Apply ``qml.transforms.broadcast_expand`` before
    ``qml.add_noise``.
    """
    return NoiseVaultPennyLaneModel(
        profile, layout=layout, unknown_gates=unknown_gates, readout=readout
    )


def gate_name(op: Operator, defined: Container[str] = ()) -> str:
    """The registry name of ``op`` under :func:`~noisevault.conversion.native_name`.

    ``defined`` is a profile's gate names. An operation that equals a registry gate at its
    angle takes the name of that gate. For example, ``IsingZZ(pi/2)`` is ``zz``,
    ``IsingXX(pi/2)`` and ``IsingYY(pi/2)`` are ``ms``, and ``Rot(a, theta, -a)`` is ``r``. A
    fixed gate takes the name of the rotation it equals when the profile has only that rotation
    (``SX`` as ``rx``).
    """
    own = _CANONICAL.get(op.name, op.name)
    names = {
        native_name(own, defined) if native is None else native_name(native, defined, own)
        for native in _natives_at_angle(op)
    }
    if len(names) > 1:
        raise ValueError(
            f"{op.name} is broadcast over angles that get the noise of different gates"
            f" ({', '.join(sorted(names))}), but one operation takes one noise channel. Expand"
            " the broadcast first: qml.add_noise(qml.transforms.broadcast_expand(qnode), model)"
        )
    return names.pop()


def operation_for(name: str) -> Callable[..., Operator] | None:
    """The PennyLane operation for registry gate ``name``, called with the gate's parameters
    and ``wires=``, or None if PennyLane has no such operation.
    ``operation_for("r")(theta, phi, wires=0)`` is a ``qml.Rot`` with the unitary of ``r``,
    which the noise model recognizes as ``r``. ``operation_for("ms")`` builds only ms(0, 0), as
    ``qml.IsingXX(pi/2)``, and raises ValueError at other phases. ``operation_for("sdg")``
    builds ``qml.adjoint(qml.S(wires))``."""
    if name in _BUILDERS:
        return _BUILDERS[name]
    info = gates.lookup(name)
    if info is None or not info.pennylane:
        return None
    adjoint = re.fullmatch(r"Adjoint\((\w+)\)", info.pennylane)
    if adjoint is None:
        return getattr(qml, info.pennylane, None)
    base = getattr(qml, adjoint[1])
    return lambda *params, wires: qml.adjoint(base(*params, wires=wires))


def confusion_kraus(matrix: np.ndarray) -> list[np.ndarray]:
    """Kraus operators sqrt(M[j, i]) |j><i| that turn populations p into M p, for any column-
    stochastic M (including P(1|0) + P(0|1) > 1, which a generalized amplitude damping cannot)."""
    return [
        np.sqrt(matrix[j, i]) * np.outer(np.eye(2)[j], np.eye(2)[i])
        for i in range(2)
        for j in range(2)
        if matrix[j, i] > 0
    ]


def _describe(report: Report, readout: bool) -> None:
    report.mark_exact("gate errors")
    if readout:
        report.mark_exact("readout errors")
    else:
        report.omit("readout errors (readout=False)")
    report.approximate(
        "initial state",
        "ideal |0...0>",
        "state preparation operations (BasisState, StatePrep, QubitDensityMatrix) are noiseless",
    )
    report.omit("idle time between gates (PennyLane circuits are not scheduled)")
    report.omit("readout on mid-circuit measurements")
    report.approximate(
        "adjoint gates and templates",
        "noised through their decomposition at qml.add_noise's default level='user'",
        "pass level='top' to qml.add_noise to noise Adjoint(SX), Adjoint(S) and Adjoint(T) whole",
    )


def _natives_at_angle(op: Operator) -> set[str | None]:
    """The native gate that each angle of ``op`` equals, or None for an angle that equals no
    native gate. A broadcast operation has one angle per element."""
    if op.name not in _AT_ANGLE:
        return {None}
    native, angle_of, values = _AT_ANGLE[op.name]
    angle = angle_of(op.parameters)
    if qml.math.is_abstract(angle):
        return {None}
    return {
        native if any(_same_angle(a, value) for value in values) else None
        for a in np.ravel(qml.math.toarray(angle))
    }


def _same_angle(a: float, b: float) -> bool:
    gap = (float(a) - b) % (2 * pi)
    return min(gap, 2 * pi - gap) < _ANGLE_TOL


def _outer_frame(key: tuple[str, str], frame: FrameType | None = None) -> FrameType | None:
    frame = frame or sys._getframe(1)
    while frame and (frame.f_globals.get("__name__"), frame.f_code.co_name) != key:
        frame = frame.f_back
    return frame


def _labels(layout: Layout) -> list[Hashable]:
    return list(layout) if isinstance(layout, Mapping) else list(range(len(layout)))


def _unconditional(op: Operator) -> Operator:
    return op.base if isinstance(op, Conditional) else op


def _is_operation(_: Operator) -> bool:
    return True


def _is_gate(op: Operator) -> bool:
    op = _unconditional(op)
    return (
        isinstance(op, Operation | CompositeOp | SymbolicOp)
        and not isinstance(op, Channel | StatePrepBase | qml.QubitDensityMatrix)
        and op.name not in _NOT_GATES
    )


def _is_named_controlled_gate(op: Operator) -> bool:
    return isinstance(op, ControlledOp) and type(op) is not ControlledOp


def _is_arithmetic(op: Operator) -> bool:
    named = op.name in _CANONICAL or _is_named_controlled_gate(op)
    return isinstance(op, CompositeOp | SymbolicOp) and not named


def _charges(op: Operator, defined: Container[str]) -> list[Charge]:
    charges: list[Charge] = []
    for gate in _decomposed(op) if _is_arithmetic(op) else [op]:
        name = gate_name(gate, defined)
        if isinstance(gate, qml.Identity):
            charges += [Charge(name, (wire,)) for wire in gate.wires]
        else:
            charges.append(Charge(name, tuple(gate.wires)))
    return charges


def _decomposed(op: Operator) -> list[Operator]:
    if not op.has_decomposition:
        raise ValueError(
            f"{op} has no decomposition into gates, so the operator gets no gate noise and"
            " default.mixed cannot run the operator. Apply a unitary operator as"
            " qml.QubitUnitary(qml.matrix(op), wires=...)"
        )
    with qml.QueuingManager.stop_recording():
        parts = op.decomposition()
    return [
        gate
        for part in parts
        if _is_gate(part)
        for gate in (_decomposed(part) if _splits_at_user_level(part) else [part])
    ]


def _splits_at_user_level(op: Operator) -> bool:
    return _is_arithmetic(op) or (isinstance(op, Adjoint) and op.has_decomposition)


def _is_mid_measure(op: Operator) -> bool:
    return isinstance(op, MidMeasureMP)


def _reads_out(mp: Any) -> bool:
    return isinstance(mp, _READOUT_MEASUREMENTS) and getattr(mp, "mv", None) is None


def _measured_basis(obs: Operator | None) -> tuple[Operator, ...] | None:
    """Single-qubit rotations into the measured basis, or () for the computational basis.

    None means that no single product basis measures the observable, for example Pauli terms
    that disagree on a wire, or an entangled eigenbasis.
    """
    if obs is None:
        return ()
    words = _pauli_terms(obs)
    if words is not None:
        return _pauli_basis(words)
    try:
        with qml.QueuingManager.stop_recording():
            basis = tuple(obs.diagonalizing_gates())
    except qml.exceptions.DiagGatesUndefinedError:
        return None
    return basis if all(len(gate.wires) == 1 for gate in basis) else None


def _pauli_terms(obs: Operator | None) -> qml.pauli.PauliSentence | None:
    """``default.mixed`` measures the simplified observable. The simplification removes each word
    whose coefficient is at most 1e-8. The copy keeps the Pauli form of ``obs`` unchanged.
    """
    words = getattr(obs, "pauli_rep", None)
    if words is None:
        return None
    words = qml.pauli.PauliSentence(words)
    words.simplify()
    return words


def _pauli_basis(words: Iterable[qml.pauli.PauliWord]) -> tuple[Operator, ...] | None:
    """Rotations that make every +1 eigenstate read 0, from the Pauli letter of each wire.

    The rotations come from the letters and not from the observable's own diagonalizing gates.
    For a sum, those gates follow eigh order and can map a -1 eigenstate to |0>.
    """
    letters: dict[Hashable, str] = {}
    for word in words:
        for wire, letter in word.items():
            if letters.setdefault(wire, letter) != letter:
                return None
    with qml.QueuingManager.stop_recording():
        return tuple(
            gate
            for wire, letter in letters.items()
            if letter != "Z"
            for gate in _ROTATED_PAULIS[letter]([wire]).diagonalizing_gates()
        )


def _shared_readout(
    mp: Any, basis: tuple[Operator, ...], tape: qml.tape.QuantumScript
) -> tuple[tuple[Operator, ...], list[Hashable]]:
    """qml.add_noise puts measurements with different readout operations on separate tapes, and
    separate tapes get separate shots. With shots, default.mixed gives Pauli words that commute
    on each wire the same shots. Thus each such word gets the readout of its whole group.
    Confusion on a wire that a measurement does not read leaves its results alone. Thus
    computational-basis readout goes on every tape wire, and those measurements share a tape.
    """
    rotations, wires = basis, _read_wires(mp, tape)
    if tape.shots and _word(mp) is not None:
        words = [m for m in tape.measurements if _word(m) is not None]
        rotations, wires = _shot_group(mp, words, tape)
    return rotations, wires if rotations else list(tape.wires)


def _shot_group(
    mp: Any, words: list[Any], tape: qml.tape.QuantumScript
) -> tuple[tuple[Operator, ...], list[Hashable]]:
    letters = [_word(m) for m in words]
    inside = [_commute(_word(mp), word) for word in letters]
    grown = True
    while grown:
        joined = [
            not member and any(_commute(word, other) for other in compress(letters, inside))
            for member, word in zip(inside, letters, strict=True)
        ]
        grown = any(joined)
        inside = [a or b for a, b in zip(inside, joined, strict=True)]
    reader: dict[Hashable, tuple[int, Any, str]] = {}
    rotations: list[Operator] = []
    wires: list[Hashable] = []
    for i, (m, word) in enumerate(compress(zip(words, letters, strict=True), inside)):
        for wire, letter in word.items():
            _, other, other_letter = reader.setdefault(wire, (i, m, letter))
            if other_letter != letter:
                raise NoiseVaultError(
                    f"with shots, {other} and {m} read wire {wire!r} in different bases, and"
                    " other measurements commute with both of them. default.mixed decides which"
                    " of these measurements share shots, and a noise model cannot see that"
                    " decision",
                    hint="wrap the QNode in qml.transforms.split_non_commuting before"
                    " qml.add_noise",
                )
        rotations += [g for g in _pauli_basis([word]) if reader[g.wires[0]][0] == i]
        wires += [wire for wire in _read_wires(m, tape) if wire not in wires]
    return tuple(rotations), wires


def _read_wires(mp: Any, tape: qml.tape.QuantumScript) -> list[Hashable]:
    """The wires whose results ``mp`` reads. An eigenvalue of a Pauli observable reads only the
    wires of its simplified words. ``qml.probs(op=...)`` gives outcomes on every observable wire.
    """
    words = _pauli_terms(mp.obs) if isinstance(mp, _EIGENVALUE_MEASUREMENTS) else None
    if words is None:
        return list(mp.wires or tape.wires)
    return [wire for wire in mp.wires if wire in words.wires]


def _word(mp: Any) -> qml.pauli.PauliWord | None:
    """The Pauli word by which default.mixed puts ``mp`` in a group with shared shots, or None.

    A zero or identity observable has the empty word. Only probabilities keep the empty word,
    because the shared basis does not change an eigenvalue of the identity.
    """
    words = _pauli_terms(mp.obs)
    if not _reads_out(mp) or words is None or len(words) > 1:
        return None
    word = next(iter(words), qml.pauli.PauliWord({}))
    return word if word or isinstance(mp, ProbabilityMP) else None


def _commute(a: Mapping[Hashable, str], b: Mapping[Hashable, str]) -> bool:
    return all(a[wire] == b[wire] for wire in a.keys() & b.keys())


def _added_events(before: dict[str, Counter[str]], after: dict[str, Counter[str]]) -> EventCounts:
    return tuple(
        (event, what, n)
        for event, counts in after.items()
        for what, n in (counts - before.get(event, Counter())).items()
    )
