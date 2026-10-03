from __future__ import annotations

import numbers
import re
from collections import defaultdict
from collections.abc import Container, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any, get_args

import numpy as np

from ..errors import LayoutError, install_hint, qubit_loci

try:
    import cirq
except ImportError as exc:
    raise ImportError(f"the Cirq export needs Cirq: {install_hint('cirq')}") from exc

from .. import gates
from ..channels import readout_matrix
from ..conversion import UnknownGates, idle_channel, native_name, resolve_op
from ..layout import normalize_layout
from ..profile import Profile, QubitRecord
from ..report import Report
from ..table import refuse_disabled

CirqLayout = Mapping[Any, int] | Sequence[int]
_ANGLE_TOL = 1e-9


@dataclass(frozen=True)
class _PowFamily:
    """Canonical names of a Cirq EigenGate class by exponent, compared modulo ``period``.

    ``names`` are the fixed gates at their exponents. ``other`` is the registry gate with a
    parameter that the class equals at every exponent. For ``ZPowGate``, ``other`` is ``p``
    exactly. For ``XPowGate``, ``other`` is ``rx`` up to global phase.
    :func:`~noisevault.conversion.native_name` decides which calibration each one takes on a
    profile. When ``other`` is also None, the gate is unknown at other exponents and the report
    shows it as ``base**exponent``.
    """

    base: str
    period: float
    names: tuple[tuple[float, str], ...]
    other: str | None

    def name_for(self, exponent: Any, defined: Container[str]) -> str:
        fixed = self._fixed(exponent)
        if fixed is not None:
            return native_name(fixed, defined, self.other)
        if self.other is not None:
            return native_name(self.other, defined)
        shown = f"{exponent:.6g}" if isinstance(exponent, numbers.Real) else str(exponent)
        return f"{self.base}**{shown}"

    def _fixed(self, exponent: Any) -> str | None:
        if isinstance(exponent, numbers.Real):
            for value, name in self.names:
                gap = (float(exponent) - value) % self.period
                if min(gap, self.period - gap) < _ANGLE_TOL:
                    return name
        return None


# Up to a global phase these gates repeat with period 2 in the exponent (iSWAP with 4), so
# inverses such as X**-1, S**-1 or CZ**-1 from cirq.inverse map to their gates. MS(pi, 0) is
# XX**-0.5 and MS(pi/2, +-pi/2) is YY**+-0.5, so both signs are the native MS. MSGate is an
# XXPowGate and gets its name the same way.
_XX = _PowFamily("rxx", 2, ((0.5, "ms"), (-0.5, "ms")), "rxx")
_POW: dict[type, _PowFamily] = {
    cirq.XPowGate: _PowFamily("x", 2, ((1, "x"), (0.5, "sx"), (-0.5, "sxdg")), "rx"),
    cirq.YPowGate: _PowFamily("y", 2, ((1, "y"),), "ry"),
    cirq.ZPowGate: _PowFamily(
        "z", 2, ((1, "z"), (0.5, "s"), (-0.5, "sdg"), (0.25, "t"), (-0.25, "tdg")), "p"
    ),
    cirq.HPowGate: _PowFamily("h", 2, ((1, "h"),), None),
    cirq.CXPowGate: _PowFamily("cx", 2, ((1, "cx"),), None),
    cirq.CZPowGate: _PowFamily("cz", 2, ((1, "cz"),), None),
    cirq.SwapPowGate: _PowFamily("swap", 2, ((1, "swap"),), None),
    # ISWAP**-0.5 is the same coupler pulse with the opposite sign. Google calibrates the two
    # as one gate, and cirq_google gives both the sqrt_iswap error.
    cirq.ISwapPowGate: _PowFamily(
        "iswap", 4, ((1, "iswap"), (0.5, "sqrt_iswap"), (-0.5, "sqrt_iswap")), None
    ),
    cirq.ZZPowGate: _PowFamily("rzz", 2, ((0.5, "zz"),), "rzz"),
    cirq.XXPowGate: _XX,
    cirq.MSGate: _XX,
    cirq.YYPowGate: _PowFamily("ryy", 2, ((0.5, "ms"), (-0.5, "ms")), "ryy"),
    cirq.CCXPowGate: _PowFamily("ccx", 2, ((1, "ccx"),), None),
}


@cirq.value_equality
class ECRGate(cirq.Gate):
    """IBM's echoed cross-resonance gate ``ecr``, with the gate registry's unitary.

    Cirq has no ECR gate, so the noise model gives ECRGate the ``ecr`` noise of a profile.
    """

    def _num_qubits_(self) -> int:
        return 2

    def _unitary_(self) -> np.ndarray:
        return gates.GATES["ecr"].unitary()

    def _value_equality_values_(self) -> tuple:
        return ()

    def _circuit_diagram_info_(self, args: Any) -> tuple[str, str]:
        return ("ECR", "ECR")

    def __repr__(self) -> str:
        return "noisevault.frameworks.cirq.ECRGate()"


def _registry_by_class() -> dict[str, str]:
    """Registry gates that are the only registry gate of their Cirq class, for example Rx -> rx."""
    names: dict[str, list[str]] = defaultdict(list)
    for info in gates.GATES.values():
        if info.cirq and info.unitary is not None:
            names[info.cirq].append(info.name)
    return {cls: found[0] for cls, found in names.items() if len(found) == 1}


_BY_CLASS = _registry_by_class()
_OWN_GATES: dict[type, str] = {ECRGate: "ecr"}
_RESOLVE_FIRST = (
    "Resolve the parameters before you add noise: cirq.resolve_parameters(circuit,"
    " params).with_noise(model). You can also give the parameterized circuit and its sweep to a"
    " simulator made with noise=model, which resolves them first"
)


def gate_name(gate: cirq.Gate, defined: Container[str] = ()) -> str:
    """The canonical NoiseVault name of a Cirq unitary gate.

    The most specific class decides the name. The order is this module's own gates
    (:class:`ECRGate`), an exponent table for the power gates, then the gate registry's Cirq
    column. ``defined`` is a profile's gate names. A power gate at a fixed angle has the name
    of the fixed gate: ``ZZ**0.5`` is ``zz``, ``X`` is ``x``, ``XX**0.5`` and ``YY**0.5`` are
    ``ms``. The power gate takes its rotation's name (``rzz``, ``rx``, ``rxx``, ``ryy``) when
    ``defined`` has the rotation but not the fixed gate (see
    :func:`~noisevault.conversion.native_name`). ``Z**t`` is exactly ``p(pi t)``, so a profile's
    ``p`` comes before ``rz`` at every exponent, and after the fixed gate (``s``, ``t``, ...)
    at the exponent of that gate. Any other gate takes the name of its class in snake case
    without the ``Gate`` suffix (``FSimGate`` -> ``fsim``, ``MatrixGate`` -> ``matrix``). A
    profile can calibrate the gate under that name. If the profile does not, the gate gets the
    typical-noise rule. A power gate whose name depends on an unresolved exponent (``X**t`` is
    ``x`` at t=1) raises ValueError.
    """
    for cls in type(gate).__mro__:
        if cls in _OWN_GATES:
            return _OWN_GATES[cls]
        family = _POW.get(cls)
        if family is not None:
            if family.names and cirq.is_parameterized(gate):
                raise ValueError(f"the gate of {gate!r} depends on its exponent. {_RESOLVE_FIRST}")
            return family.name_for(gate.exponent, defined)  # type: ignore[attr-defined]
        if cls.__name__ in _BY_CLASS:
            return _BY_CLASS[cls.__name__]
    stem = re.sub(r"(Pow)?Gate$", "", type(gate).__name__) or type(gate).__name__
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", stem).lower()
    return type(gate).__name__ if name in gates.GATES else name


class _QubitMap:
    """Cirq qubits to device qubits: the layout, else LineQubit index or GridQubit coords."""

    def __init__(self, profile: Profile, layout: CirqLayout | None) -> None:
        self._profile = profile
        self._explicit = layout is not None
        self._coords: dict[tuple[float, ...], list[QubitRecord]] = defaultdict(list)
        for record in profile.qubits:
            if record.coords is not None:
                self._coords[tuple(record.coords)].append(record)
        self._known: dict[cirq.Qid, int] = {}
        if layout is not None:
            keyed = _keyed_layout(layout)
            self._known = dict(normalize_layout(list(keyed), keyed, profile))  # type: ignore[arg-type]

    def physical(self, qids: Sequence[cirq.Qid]) -> tuple[int, ...]:
        """Device qubits of ``qids``, which must map to distinct ones."""
        indices = tuple(self._one(qid) for qid in qids)
        owner: dict[int, cirq.Qid] = {}
        for qid, index in zip(qids, indices, strict=True):
            first = owner.setdefault(index, qid)
            if first != qid:
                raise LayoutError(f"{first!r} and {qid!r} both map to device qubit {index}")
        return indices

    def _one(self, qid: cirq.Qid) -> int:
        if qid in self._known:
            return self._known[qid]
        if self._explicit:
            raise LayoutError(
                f"layout has no device qubit for {qid!r}", hint="map every circuit qubit in layout="
            )
        if qid.dimension != 2:
            raise LayoutError(
                f"{qid!r} has dimension {qid.dimension}, but a profile describes qubits"
            )
        (index,) = normalize_layout(
            [qid], {qid: self._default_index(qid)}, self._profile, width=0
        ).values()
        self._known[qid] = index
        return index

    def _default_index(self, qid: cirq.Qid) -> int:
        if isinstance(qid, cirq.LineQubit):
            return qid.x
        fix = f"pass layout={{{qid!r}: <device qubit>, ...}} covering every circuit qubit"
        if isinstance(qid, cirq.GridQubit) and self._coords:
            records = self._coords.get((qid.row, qid.col))
            if records is None:
                usable = [
                    f"GridQubit{c}"
                    for c, found in self._coords.items()
                    if sum(not r.disabled for r in found) == 1
                ]
                raise LayoutError(
                    f"{self._profile.id} has no qubit at coords ({qid.row}, {qid.col})",
                    hint=f"use the device's coords, for example {', '.join(usable[:3])}, or {fix}"
                    if usable
                    else fix,
                )
            enabled = [r.index for r in records if not r.disabled]
            if len(enabled) > 1:
                raise LayoutError(
                    f"{self._profile.id} has {qubit_loci(*((i,) for i in enabled), limit=None)}"
                    f" at coords ({qid.row}, {qid.col}), so {qid!r} has no single device qubit",
                    hint=fix,
                )
            return enabled[0] if enabled else records[0].index
        why = (
            "records no qubit coords"
            if isinstance(qid, cirq.GridQubit)
            else "cannot place this type of qubit"
        )
        raise LayoutError(
            f"{qid!r} needs a layout, because {self._profile.id} {why} and only LineQubit(i)"
            " maps to device qubit i by default",
            hint=fix,
        )


def _keyed_layout(layout: CirqLayout) -> dict[cirq.Qid, int]:
    """A layout keyed by Cirq qubits. A sequence or an integer key means LineQubit(i)."""
    if not isinstance(layout, Mapping):
        return {cirq.LineQubit(i): p for i, p in enumerate(layout)}
    given: dict[cirq.Qid, Any] = {}
    for key in layout:
        qid = cirq.LineQubit(key) if isinstance(key, int) and not isinstance(key, bool) else key
        if not isinstance(qid, cirq.Qid):
            raise LayoutError(f"layout key {key!r} is not a Cirq qubit or an integer")
        if qid in given:
            raise LayoutError(
                f"layout keys {given[qid]!r} and {key!r} both name {qid!r}",
                hint="keep one of the two keys",
            )
        given[qid] = key
    return {qid: layout[key] for qid, key in given.items()}


class NoiseVaultNoiseModel(cirq.NoiseModel):
    """A profile's noise as a Cirq noise model. ``.report`` says what the export reproduced.

    Use the model with ``cirq.DensityMatrixSimulator(noise=model)`` (exact) or
    ``cirq.Simulator(noise=model)`` (sampled trajectories). The report adds events, such as
    gates that got typical noise, each time Cirq simulates a circuit.

    Readout error is classical: a mid-circuit measurement reports a flipped bit and leaves the
    qubit in its true state (a ``confusion_map``). A terminal measurement gets the same
    assignment probabilities as a channel directly before it. Cirq samples terminal
    measurements from the gates of the original circuit and does not use a confusion map
    there. Sampled results are exact in both cases. But after a terminal measurement, the
    state that ``simulate()`` returns includes the flips. To inspect states, use
    ``readout=False`` or remove the final measurements.

    Report events count each noisy circuit that the model builds, not shots or runs. Cirq asks
    for a noisy circuit once per run or once per part of a run. A repeat of the last circuit
    uses the noisy circuit again.
    """

    def __init__(
        self,
        profile: Profile,
        report: Report,
        qubits: _QubitMap,
        *,
        unknown_gates: UnknownGates,
        readout: bool,
    ) -> None:
        self.profile = profile
        self.report = report
        self._table = profile.table
        self._qubits = qubits
        self._unknown_gates = unknown_gates
        self._readout = readout
        self._last: tuple[tuple[cirq.Moment, ...], list[cirq.OP_TREE]] | None = None

    def __repr__(self) -> str:
        return f"NoiseVaultNoiseModel({self.profile.id}, nv:{self.profile.fingerprint[:12]})"

    def noisy_moments(
        self, moments: Iterable[cirq.Moment], system_qubits: Sequence[cirq.Qid]
    ) -> Sequence[cirq.OP_TREE]:
        moments = tuple(moments)
        # Cirq's per-shot paths (trajectories, mid-circuit measurements) ask for the same
        # moments on every repetition. The model returns its last result again, so these paths
        # stay fast and report events do not count shots.
        if self._last is not None and self._last[0] == moments:
            return self._last[1]
        self._qubits.physical(system_qubits)
        later: set[cirq.Qid] = set()
        terminal: set[tuple[int, cirq.Operation]] = set()
        for i in reversed(range(len(moments))):
            for op in moments[i]:
                if cirq.is_measurement(op) and later.isdisjoint(op.qubits):
                    terminal.add((i, op))
            later.update(moments[i].qubits)
        noisy: list[cirq.OP_TREE] = [
            [self._noisy(op, terminal=(i, op) in terminal) for op in moment]
            for i, moment in enumerate(moments)
        ]
        self._last = (moments, noisy)
        return noisy

    def noisy_operation(self, operation: cirq.Operation) -> cirq.OP_TREE:
        """The operation followed by its noise. A measurement gets a ``confusion_map``."""
        return self._noisy(operation, terminal=False)

    def _noisy(self, operation: cirq.Operation, *, terminal: bool) -> cirq.OP_TREE:
        gate = operation.gate
        if not operation.qubits:
            return operation
        if isinstance(operation, cirq.ClassicallyControlledOperation):
            raise ValueError(
                f"{operation!r} is classically controlled, and the Cirq export does not support"
                " classical control. Replace the feed-forward with a quantum-controlled gate and a"
                " final measurement, or simulate the parts before and after separately"
            )
        if gate is None:
            raise ValueError(
                f"{operation!r} is not a plain gate operation. Flatten the circuit first, for"
                " example with cirq.Circuit(cirq.decompose(circuit, keep=lambda op: op.gate is not"
                " None))"
            )
        physical = self._qubits.physical(operation.qubits)
        if cirq.is_measurement(gate):
            self._refuse_disabled("measure", physical)
            if isinstance(gate, cirq.MeasurementGate):
                return self._measure(operation, gate, physical, terminal=terminal)
            return self._other_measurement(operation)
        if isinstance(gate, cirq.ResetChannel):
            self._refuse_disabled("reset", physical)
            return self._reset(operation, physical)
        if isinstance(gate, cirq.WaitGate):
            self._refuse_disabled("delay", physical)
            return self._wait(operation, gate, physical)
        if isinstance(gate, cirq.IdentityGate):
            pairs = zip(operation.qubits, physical, strict=True)
            return [operation, *(op for q, p in pairs for op in self._noise("id", [q], [p]))]
        if isinstance(gate, cirq.PhasedXZGate) and "phased_xz" not in self.profile.gates:
            return self._phased_xz(gate, operation.qubits, physical)
        if not (cirq.has_unitary(gate) or cirq.is_parameterized(gate)):
            self.report.count("circuit_channel_kept", type(gate).__name__)
            return operation
        name = gate_name(gate, self.profile.gates)
        return [operation, *self._noise(name, operation.qubits, physical)]

    def _refuse_disabled(self, name: str, physical: Sequence[int]) -> None:
        for index in physical:
            refuse_disabled(self._table.gate(name, (index,)))

    def _noise(
        self, name: str, qids: Sequence[cirq.Qid], physical: Sequence[int]
    ) -> list[cirq.Operation]:
        built = resolve_op(
            self._table, name, physical, unknown_gates=self._unknown_gates, report=self.report
        )
        qid_of = dict(zip(physical, qids, strict=True))
        return [
            cirq.KrausChannel(list(channel.kraus)).on(*(qid_of[w] for w in channel.wires))
            for channel in built.channels
        ]

    def _phased_xz(
        self, gate: cirq.PhasedXZGate, qids: Sequence[cirq.Qid], physical: tuple[int, ...]
    ) -> list[cirq.Operation]:
        """PhasedXZ is exactly the r gate (PhasedXPow) then ``Z**z``, each with its noise. The
        Z part gets the same name as a separate ``Z**z``.

        Google calibrates PhasedXZ as r with a virtual Z, so it needs no typical noise there.
        Only a profile without its own ``phased_xz`` gets here.
        """
        x_part = cirq.PhasedXPowGate(
            phase_exponent=gate.axis_phase_exponent, exponent=gate.x_exponent
        )
        out = [x_part.on(*qids), *self._noise("r", qids, physical)]
        if cirq.is_parameterized(gate.z_exponent) or gate.z_exponent != 0:
            z_part = cirq.Z**gate.z_exponent
            name = gate_name(z_part, self.profile.gates)
            out += [z_part.on(*qids), *self._noise(name, qids, physical)]
        return out

    def _other_measurement(self, operation: cirq.Operation) -> cirq.Operation:
        if self._readout:
            raise ValueError(
                f"{operation!r} is not a cirq.measure, so NoiseVault cannot add its readout"
                " error. Rotate into the Z basis and use cirq.measure, or pass readout=False to"
                " keep the measurement noiseless"
            )
        return operation

    def _measure(
        self,
        operation: cirq.Operation,
        gate: cirq.MeasurementGate,
        physical: tuple[int, ...],
        *,
        terminal: bool,
    ) -> cirq.OP_TREE:
        if not self._readout:
            return operation
        if gate.confusion_map:
            raise ValueError(
                f"{operation!r} already has a confusion_map. Remove the confusion_map, or pass"
                " readout=False to keep your own readout model"
            )
        matrices = {}
        for position, index in enumerate(physical):
            matrix = readout_matrix(self._table.qubit(index))
            if matrix is None:
                self.report.mark_unknown(f"readout of qubit {index}")
            else:
                matrices[position] = matrix
        if not matrices:
            return operation
        if terminal:
            self.report.approximate(
                "state after a terminal measurement",
                "includes the readout flips",
                "sampled results are exact. Inspect states with readout=False",
            )
            flips = [
                cirq.KrausChannel(_assignment_kraus(m)).on(operation.qubits[position])
                for position, m in matrices.items()
            ]
            return [*flips, operation]
        noisy = cirq.MeasurementGate(
            gate.num_qubits(),
            key=gate.mkey,
            invert_mask=gate.invert_mask,
            qid_shape=cirq.qid_shape(gate),
            confusion_map={(position,): m.T for position, m in matrices.items()},  # rows: truth
        )
        return noisy.on(*operation.qubits).with_tags(*operation.tags)

    def _reset(self, operation: cirq.Operation, physical: tuple[int, ...]) -> cirq.OP_TREE:
        error = self._table.qubit(physical[0]).prep_error
        if error is None:
            self.report.mark_unknown(f"preparation error of qubit {physical[0]}")
        if not error:
            return operation
        return [operation, cirq.bit_flip(error).on(*operation.qubits)]

    def _wait(
        self, operation: cirq.Operation, gate: cirq.WaitGate, physical: tuple[int, ...]
    ) -> list[cirq.Operation]:
        if cirq.is_parameterized(gate):
            raise ValueError(f"{operation!r} has an unresolved duration. {_RESOLVE_FIRST}")
        duration = float(gate.duration.total_nanos())
        out = [operation]
        for qid, index in zip(operation.qubits, physical, strict=True):
            channel = idle_channel(self._table, index, duration, self.report, label="WaitGate")
            if channel is not None:
                out.append(cirq.KrausChannel(list(channel.kraus)).on(qid))
        return out


def _assignment_kraus(matrix: np.ndarray) -> list[np.ndarray]:
    """Kraus operators sqrt(M[m, p]) |m><p|: a Z measurement after them reports m with M[m, p]."""
    ops = []
    for measured, prepared in product(range(2), repeat=2):
        op = np.zeros((2, 2), dtype=complex)
        op[measured, prepared] = np.sqrt(matrix[measured, prepared])
        ops.append(op)
    return ops


def to_cirq(
    profile: Profile,
    *,
    layout: CirqLayout | None = None,
    unknown_gates: UnknownGates = "typical",
    readout: bool = True,
) -> NoiseVaultNoiseModel:
    """A Cirq noise model of ``profile``, carrying ``.report`` and ``.profile``.

    ``layout`` maps circuit qubits (Cirq qubits, or integers meaning ``LineQubit(i)``, or a
    sequence indexed by LineQubit) to device qubits. The keys ``i`` and ``LineQubit(i)`` in one
    layout raise LayoutError. Without ``layout``, ``LineQubit(i)`` is qubit ``i`` and
    ``GridQubit(r, c)`` is the qubit at coords ``(r, c)`` if the profile records coords. If two
    enabled qubits have those coords, the model raises LayoutError.
    ``unknown_gates="typical"`` gives gates the profile does not calibrate the noise of its
    typical native gate, with one report entry and one warning per gate. ``"error"`` raises
    an error instead. ``readout=False`` leaves measurements noiseless.
    """
    if unknown_gates not in get_args(UnknownGates):
        raise ValueError(f"unknown_gates={unknown_gates!r}: choose 'typical' or 'error'")
    if not isinstance(readout, bool):
        raise TypeError(
            f"readout={readout!r}: pass True to add readout error or False to leave"
            " measurements noiseless"
        )
    report = Report.start(
        profile,
        "cirq",
        cirq.__version__,
        layout=layout,
        unknown_gates=unknown_gates,
        readout=readout,
    )
    report.record_effects(profile.effects)
    qubits = _QubitMap(profile, layout)
    report.mark_exact("gate noise as Kraus channels after each gate")
    if readout:
        report.mark_exact(
            "readout assignment error (confusion_map mid-circuit, an equivalent channel before"
            " terminal measurements)"
        )
    else:
        report.omit("readout error (readout=False)")
    report.mark_exact("preparation error after each reset")
    report.mark_exact("thermal relaxation during WaitGate")
    report.omit("preparation error of the initial state (only resets get preparation error)")
    report.omit("idle noise outside WaitGate (unscheduled idle time)")
    return NoiseVaultNoiseModel(
        profile, report, qubits, unknown_gates=unknown_gates, readout=readout
    )
