"""Stim export: a noisy copy of a Stim circuit that carries its report.

After every Clifford gate, the export adds the Pauli twirl of the channel that the shared
conversion rules give the gate. The twirl is PAULI_CHANNEL_1 or PAULI_CHANNEL_2. For a Pauli
product on three or more qubits, the twirl is a CORRELATED_ERROR chain with the same
probabilities. Twirling keeps the average fidelity of each gate but drops relaxation's bias
toward |0>. Thus the export matches the twirled model, not the full channel. Measurements get
readout error, and resets get preparation error. With ``tick_ns``, every qubit that stays idle
in a TICK layer gets twirled relaxation for that time.
Annotations, REPEAT blocks, detectors and observables pass through unchanged.
"""

from __future__ import annotations

import functools
import warnings
from collections import Counter
from collections.abc import Callable, Container, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple, NoReturn, get_args

import numpy as np

from ..errors import (
    DisabledGateError,
    LayoutError,
    LociText,
    MissingCalibrationError,
    NoiseVaultError,
    install_hint,
    qubit_loci,
)

try:
    import stim
except ImportError as exc:
    raise ImportError(f"the Stim export needs stim: {install_hint('stim')}") from exc

from .. import gates, metrics
from ..channels import ChannelSpec, pauli_twirl
from ..conversion import UnknownGates, idle_channel, native_name, resolve_op
from ..layout import can_measure, normalize_layout
from ..profile import Profile
from ..report import Report
from ..table import GateNoise, NoiseTable, refuse_disabled

Readout = Literal["symmetrize", "exact", "none"]
ExistingNoise = Literal["error", "keep", "strip"]
Kind = Literal["tick", "annotation", "pad", "noise", "measure", "reset", "gate"]
Layout = Mapping[int, int] | Sequence[int] | None
EventCounts = tuple[tuple[str, str, int], ...]  # (event, key, count) that one gate adds

_ANNOTATIONS = frozenset({"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"})
_HERALDED = frozenset({"HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1"})
_COMBINED = frozenset({"MPP", "SPP", "SPP_DAG"})  # combiners join the targets of a group
_X, _Z = 1, 2
_Y = _X | _Z
_I_POWER = {(_X, _Y): 1, (_Y, _Z): 1, (_Z, _X): 1, (_Y, _X): 3, (_Z, _Y): 3, (_X, _Z): 3}
# The registry gate each Stim gate equals and, for an MS gate at fixed phases, the rotation it
# also equals. conversion.native_name picks the gate that a profile calibrates.
_NAMES: dict[str, tuple[str, str | None]] = {
    **{name: (row.name, None) for row in gates.GATES.values() for name in row.stim},
    "SQRT_Y": ("ry", None),
    "SQRT_Y_DAG": ("ry", None),
    "SQRT_ZZ_DAG": ("rzz", None),
    "ISWAP_DAG": ("iswap", None),
    "SQRT_XX": ("ms", "rxx"),
    "SQRT_XX_DAG": ("ms", "rxx"),
    "SQRT_YY": ("ms", "ryy"),
    "SQRT_YY_DAG": ("ms", "ryy"),
}
_NOISELESS = frozenset({"II"})  # a 2-qubit identity is no entangling gate
_CONTROLLED_PAULI = {
    ("CX", 1): "X",
    ("CY", 1): "Y",
    ("CZ", 0): "Z",
    ("CZ", 1): "Z",
    ("XCZ", 0): "X",
    ("YCZ", 0): "Y",
}
# The Pauli that undoes each reset: a failed Z reset leaves |1>, a failed X reset |->.
_PREP_FLIP = {
    "R": "X_ERROR",
    "RX": "Z_ERROR",
    "RY": "X_ERROR",
    "MR": "X_ERROR",
    "MRX": "Z_ERROR",
    "MRY": "X_ERROR",
}
_PAULI_CHANNEL = {3: "PAULI_CHANNEL_1", 15: "PAULI_CHANNEL_2"}


class ExistingNoiseError(NoiseVaultError, ValueError):
    """The input circuit already has noise and the caller did not say what to do with it."""


class NoiseVaultStimCircuit(stim.Circuit):
    """A noisy ``stim.Circuit`` carrying ``.report``, ``.profile`` and ``.layout``.

    ``readout="exact"`` sets ``.readout_flips``, with one row per measurement record,
    (P(flip | recorded 0), P(flip | recorded 1)), which :func:`sample_with_readout` applies.
    Stim methods that build a new circuit (``copy``, ``flattened``, ``+``) return a plain
    ``stim.Circuit`` without these attributes. Pickling and ``copy.deepcopy`` keep them, with
    probabilities at the 6 significant digits of Stim's own circuit pickling.
    """

    report: Report
    profile: Profile
    layout: dict[int, int]
    readout_flips: np.ndarray | None

    def __getstate__(self) -> tuple[Any, dict[str, Any]]:
        return super().__getstate__(), dict(self.__dict__)

    def __setstate__(self, state: tuple[Any, dict[str, Any]]) -> None:
        program, attributes = state
        super().__setstate__(program)
        self.__dict__.update(attributes)

    def detector_error_model(
        self, *, approximate_disjoint_errors: bool | float = True, **options: Any
    ) -> stim.DetectorErrorModel:
        """Stim's detector error model, reading PAULI_CHANNEL components as independent.

        Stim refuses PAULI_CHANNEL_1/2 and ELSE_CORRELATED_ERROR without
        ``approximate_disjoint_errors``, an O(p^2) approximation. Thus this method turns the
        option on by default. Pass False to get Stim's refusal.
        """
        return super().detector_error_model(
            approximate_disjoint_errors=approximate_disjoint_errors, **options
        )


def to_stim(
    profile: Profile,
    circuit: stim.Circuit | str,
    *,
    layout: Layout = None,
    readout: Readout = "symmetrize",
    tick_ns: float | None = None,
    existing_noise: ExistingNoise = "error",
    unknown_gates: UnknownGates = "typical",
) -> NoiseVaultStimCircuit:
    """A copy of ``circuit`` with the profile's noise, and a report of what it approximated.

    ``layout`` maps Stim qubit indices to physical qubits (identity by default, or
    :func:`layout_from_coords` makes a layout from QUBIT_COORDS). ``readout="symmetrize"``
    flips each result with the mean of P(1|0) and P(0|1). ``"exact"`` leaves measurements
    perfect and keeps the asymmetric error for :func:`sample_with_readout`. ``"none"`` adds no
    readout error. ``tick_ns`` is the duration of one TICK layer, and qubits idle in a layer
    relax for that duration. ``existing_noise`` says what to do with noise already in the
    circuit. ``"error"`` raises an error, ``"keep"`` keeps the noise, and ``"strip"`` removes
    the noise.
    """
    _check_choice("readout", readout, Readout)
    _check_choice("existing_noise", existing_noise, ExistingNoise)
    _check_choice("unknown_gates", unknown_gates, UnknownGates)
    if tick_ns is not None and not (np.isfinite(tick_ns) and tick_ns > 0):
        raise ValueError(f"tick_ns={tick_ns!r}: give the TICK layer duration in ns (> 0) or None")
    circuit = _as_circuit(circuit)
    found = _scan(circuit)
    _check_existing_noise(found, existing_noise)
    if readout == "exact":
        _check_exact_readout(found)

    qubits = found.qubits | found.noise_qubits if existing_noise == "keep" else found.qubits
    physical = normalize_layout(sorted(qubits), layout, profile)
    report = Report.start(
        profile,
        "stim",
        stim.__version__,
        layout=layout,
        readout=readout,
        tick_ns=tick_ns,
        existing_noise=existing_noise,
        unknown_gates=unknown_gates,
    )
    report.record_effects(profile.effects)
    _describe(report, found, readout, tick_ns, existing_noise)
    exporter = _Exporter(profile, physical, report, readout, tick_ns, existing_noise, unknown_gates)
    lines, _ = exporter.block(circuit, set())
    flips = exporter.record_flips(circuit) if readout == "exact" else None
    exporter.finish()

    out = NoiseVaultStimCircuit("\n".join(lines))
    out.report, out.profile, out.layout = report, profile, physical
    out.readout_flips = flips
    return out


def sample_with_readout(
    circuit: NoiseVaultStimCircuit, shots: int, *, seed: int | None = None
) -> np.ndarray:
    """Measurement samples, shape (shots, num_measurements), with exact asymmetric readout.

    The circuit must come from ``to_stim(..., readout="exact")``: Stim samples perfect
    measurements and each recorded bit then flips with the probability for its value.
    """
    flips = getattr(circuit, "readout_flips", None)
    if flips is None:
        raise ValueError(
            "sample_with_readout needs a circuit exported with readout='exact':"
            " profile.to_stim(circuit, readout='exact')"
        )
    rng = np.random.default_rng(seed)
    bits = circuit.compile_sampler(seed=int(rng.integers(2**63))).sample(shots)
    flip = rng.random(bits.shape) < np.where(bits, flips[:, 1], flips[:, 0])
    return bits ^ flip


def layout_from_coords(circuit: stim.Circuit | str, profile: Profile) -> dict[int, int]:
    """Place the circuit on the device by matching its QUBIT_COORDS to the profile's coords.

    Tries the eight rotations and reflections of the square lattice and every translation. It
    also tries them after a 45 degree turn, because Stim's rotated surface codes put neighbors
    on diagonals. A placement must put every qubit on an enabled device qubit and every 2-qubit
    gate on a pair with a calibrated native gate. A placement also must not put an operation on
    qubits where the profile disables the operation, for example ``M`` on a qubit whose
    ``measure`` is disabled. From the valid placements, the function returns the one with the
    lowest summed 2-qubit gate error and mean readout error. If a placement puts a qubit on
    coords that two enabled device qubits have, the function raises LayoutError.
    """
    circuit = _as_circuit(circuit)
    scanned: set[tuple[str, tuple[int, ...]]] = set()
    found = _scan(circuit, _Scan(operations=scanned))
    table = profile.table
    enabled = sum(not table.qubit(q).disabled for q in range(table.num_qubits))
    placeable = len(found.qubits) <= enabled and all(
        _placeable(table, name, len(qubits))
        for stim_name, qubits in scanned
        for name in _profile_names(stim_name, profile.gates)
    )
    coords = circuit.get_final_qubit_coordinates()
    missing = sorted(q for q in found.qubits if len(coords.get(q, ())) < 2)
    if missing:
        raise LayoutError(
            f"the circuit gives no 2D QUBIT_COORDS for {qubit_loci(*((q,) for q in missing))}",
            hint="add QUBIT_COORDS for each qubit or pass layout=" if placeable else None,
        )
    device = _Device(profile, placeable)
    labels = sorted(found.qubits)
    if not labels:
        return {}
    points = np.array([coords[q][:2] for q in labels], dtype=float)
    column = {q: i for i, q in enumerate(labels)}
    pairs = np.array([(column[a], column[b]) for a, b in sorted(found.pairs)], dtype=int)
    used: dict[tuple[str, int], list[tuple[int, ...]]] = {}
    for stim_name, qubits in sorted(scanned):
        for name in _profile_names(stim_name, profile.gates):
            used.setdefault((name, len(qubits)), []).append(tuple(column[q] for q in qubits))
    operations = [(name, np.array(columns, dtype=int)) for (name, _), columns in used.items()]
    cost = _PlacementCost(profile, pairs.reshape(-1, 2), operations)
    best: tuple[float, np.ndarray] | None = None
    for transform in _TRANSFORMS:
        placements = device.placements(points @ transform.T)
        if not len(placements):
            continue
        totals = cost(placements)
        i = int(np.argmin(totals))
        if np.isfinite(totals[i]) and (best is None or totals[i] < best[0]):
            best = (float(totals[i]), placements[i])
    if best is None:
        raise LayoutError(
            f"no rotation or shift of the circuit's QUBIT_COORDS fits {profile.id}'s qubit"
            " coords, puts every 2-qubit gate on a connected pair and avoids disabled operations",
            hint="pass layout= explicitly" if placeable else None,
        )
    return dict(zip(labels, map(int, best[1]), strict=True))


# circuit scan -----------------------------------------------------------------------------


@dataclass
class _Scan:
    qubits: set[int] = field(default_factory=set)  # every qubit an operation touches
    noise_qubits: set[int] = field(default_factory=set)  # qubits that noise instructions target
    pairs: set[tuple[int, int]] = field(default_factory=set)  # 2-qubit gate targets
    noise: str | None = None  # first noise instruction, as text
    herald: str | None = None  # first heralded noise instruction (it adds records)
    feedback: str | None = None  # first gate controlled by a measurement record
    product: str | None = None  # first multi-qubit measurement
    operations: set[tuple[str, tuple[int, ...]]] | None = None


def _scan(circuit: stim.Circuit, found: _Scan | None = None) -> _Scan:
    found = found or _Scan()
    for item in circuit:
        if isinstance(item, stim.CircuitRepeatBlock):
            _scan(item.body_copy(), found)
            continue
        kind = _kind(item.name)
        if kind == "pad" and item.gate_args_copy():
            found.noise = found.noise or _short(item)
        if kind in ("tick", "annotation", "pad"):
            continue
        if kind == "noise" or (kind == "measure" and item.gate_args_copy()):
            found.noise = found.noise or _short(item)
        if item.name in _HERALDED:
            found.herald = found.herald or _short(item)
        if kind == "noise":
            qubits = (t.qubit_value for t in item.targets_copy())
            found.noise_qubits.update(q for q in qubits if q is not None)
            continue
        for group in item.target_groups():
            qubits = [t.qubit_value for t in group if t.qubit_value is not None]
            found.qubits.update(qubits)
            if found.operations is not None:
                found.operations.update(_operations(item.name, kind, group))
            if any(t.is_measurement_record_target for t in group):
                found.feedback = found.feedback or _short(item)
            elif kind == "gate" and len(acted := _acted_on(item.name, group)) == 2:
                found.pairs.add((acted[0], acted[1]))
            if kind == "measure" and len(_measured_product(item.name, group).qubits) > 1:
                found.product = found.product or _short(item)
    return found


def _operations(
    name: str, kind: Kind, group: Sequence[stim.GateTarget]
) -> Iterator[tuple[str, tuple[int, ...]]]:
    """The (Stim gate, qubits) of each check the export makes for a disabled operation."""
    if any(t.is_measurement_record_target or t.is_sweep_bit_target for t in group):
        for position, target in enumerate(group):
            pauli = _CONTROLLED_PAULI.get((name, position))
            if pauli and target.qubit_value is not None:
                yield pauli, (target.value,)
    elif kind == "gate":
        if qubits := _acted_on(name, group):
            yield name, tuple(qubits)
    else:
        yield from ((name, (q,)) for q in _acted_on(name, group))


def _profile_names(stim_name: str, defined: Container[str]) -> tuple[str, ...]:
    kind = _kind(stim_name)
    if kind == "measure":
        return ("measure", "reset") if stim_name in _PREP_FLIP else ("measure",)
    if kind == "reset":
        return ("reset",)
    return (gate_name(stim_name, defined),)


def _check_existing_noise(found: _Scan, policy: ExistingNoise) -> None:
    if found.noise is not None and policy == "error":
        raise ExistingNoiseError(
            f"the circuit already has noise ({found.noise})",
            hint="pass existing_noise='strip' to replace it with the profile's noise, or"
            " existing_noise='keep' to add to it",
        )
    if found.herald is not None and policy == "strip":
        raise ExistingNoiseError(
            f"{found.herald} adds measurement records that later rec[] targets count, so"
            " NoiseVault cannot strip the instruction",
            hint="remove it from the circuit or pass existing_noise='keep'",
        )


def _check_exact_readout(found: _Scan) -> None:
    problem = None
    if found.feedback:
        problem = f"{found.feedback} feeds a measurement back into the circuit"
    if found.product:
        problem = f"{found.product} measures a multi-qubit product"
    if problem:
        raise ValueError(
            f"readout='exact' flips recorded bits after sampling, which is exact only for"
            f" single-qubit measurements that nothing reads during the circuit, but {problem}."
            " Use readout='symmetrize'"
        )


# emission ---------------------------------------------------------------------------------


class _Resolved(NamedTuple):
    events: EventCounts
    prefix: str = ""
    lines: tuple[str, ...] = ()


class _Exporter:
    """Writes Stim program text, because Stim parses text much faster than stim.Circuit.append."""

    def __init__(
        self,
        profile: Profile,
        physical: dict[int, int],
        report: Report,
        readout: Readout,
        tick_ns: float | None,
        existing_noise: ExistingNoise,
        unknown_gates: UnknownGates,
    ) -> None:
        self.profile = profile
        self.table = profile.table
        self.defined = profile.gates
        self.physical = physical
        self.qubits = frozenset(physical)
        self.report = report
        self.readout = readout
        self.tick_ns = tick_ns
        self.keep_noise = existing_noise == "keep"
        self.unknown_gates = unknown_gates
        self.unknown: dict[str, set[int]] = {"readout": set(), "prep": set(), "idle": set()}
        self._gate_noise: dict[tuple[str, tuple[int, ...]], _Resolved] = {}
        # Gate applications per (Stim gate, qubits); a REPEAT body counts once per pass.
        self._applied: Counter[tuple[str, tuple[int, ...]]] = Counter()
        self._passes = 1
        self._idle: dict[int, str] = {}
        self._twirls: dict[tuple, str] = {}
        self._handlers = {
            "tick": self._tick,
            "annotation": self._copy,
            "pad": self._pad,
            "noise": self._noise,
            "measure": self._measure,
            "reset": self._reset,
            "gate": self._gate,
        }

    def block(self, circuit: stim.Circuit, busy: set[int]) -> tuple[list[str], set[int]]:
        """Lines for ``circuit`` given the qubits already busy in the current TICK layer."""
        lines: list[str] = []
        for item in circuit:
            if isinstance(item, stim.CircuitRepeatBlock):
                busy = self._repeat(item, busy, lines)
            else:
                busy = self._handlers[_kind(item.name)](item, busy, lines)
        return lines, busy

    def _repeat(self, block: stim.CircuitRepeatBlock, busy: set[int], lines: list[str]) -> set[int]:
        body, count, outer = block.body_copy(), block.repeat_count, self._passes
        peel = self.tick_ns is not None and count > 1
        self._passes = outer * (1 if peel else count)
        first, after = self.block(body, set(busy))
        again = first
        if peel:
            # Idle noise at the body's first TICK depends on what ran before it, which differs
            # between the first pass and later ones. Thus the exporter writes the first pass once.
            # The two walks count 1 and count - 1 passes, so events total count either way.
            self._passes = outer * (count - 1)
            again, after = self.block(body, set(after))
        self._passes = outer
        tag = f"[{block.tag}]" if block.tag else ""
        if again != first:
            lines.extend(first)
            count -= 1
        lines.extend([f"REPEAT{tag} {count} {{", *again, "}"])
        return after

    def _copy(self, inst: stim.CircuitInstruction, busy: set[int], lines: list[str]) -> set[int]:
        lines.append(str(inst))
        return busy

    def _pad(self, inst: stim.CircuitInstruction, busy: set[int], lines: list[str]) -> set[int]:
        keep = self.keep_noise or not inst.gate_args_copy()
        lines.append(str(inst) if keep else _text(inst, inst.target_groups(), []))
        return busy

    def _noise(self, inst: stim.CircuitInstruction, busy: set[int], lines: list[str]) -> set[int]:
        if self.keep_noise:
            lines.append(str(inst))
        return busy

    def _tick(self, inst: stim.CircuitInstruction, busy: set[int], lines: list[str]) -> set[int]:
        if self.tick_ns is not None:
            noise: dict[str, list[int]] = {}
            for q in sorted(self.qubits - busy):
                channel = self._idle_noise(q)
                if channel:
                    noise.setdefault(channel, []).append(q)
            lines.extend(_noise_lines(noise))
        lines.append(str(inst))
        return set()

    def _gate(self, inst: stim.CircuitInstruction, busy: set[int], lines: list[str]) -> set[int]:
        chunks = _disjoint_chunks(inst.target_groups())
        for chunk in chunks:
            lines.append(str(inst) if len(chunks) == 1 else _text(inst, chunk))
            noise: dict[str, list[int]] = {}
            for group in chunk:
                if any(t.is_measurement_record_target or t.is_sweep_bit_target for t in group):
                    self._refuse_controlled_pauli(inst, group)
                    self.report.approximate(
                        "classically controlled Paulis", "no gate noise", "Pauli-frame updates"
                    )
                    continue
                qubits = tuple(_acted_on(inst.name, group))
                if not qubits:
                    continue
                busy.update(qubits)
                resolved = self._gate_channel(inst.name, qubits)
                lines.extend(resolved.lines)
                if resolved.prefix:
                    noise.setdefault(resolved.prefix, []).extend(qubits)
            lines.extend(_noise_lines(noise))
        return busy

    def _measure(self, inst: stim.CircuitInstruction, busy: set[int], lines: list[str]) -> set[int]:
        stated = inst.gate_args_copy()
        stated_flip = stated[0] if stated and self.keep_noise else 0.0
        resets = inst.name in _PREP_FLIP
        # A qubit measured and reset twice must get its preparation error between the two.
        groups = inst.target_groups()
        for chunk in _disjoint_chunks(groups) if resets else [groups]:
            runs: list[tuple[float, list[list[stim.GateTarget]]]] = []
            measured: list[int] = []
            for group in chunk:
                qubits = _acted_on(inst.name, group)
                self._refuse_disabled("measure", inst.name, qubits)
                if resets:
                    self._refuse_disabled("reset", inst.name, qubits)
                measured += qubits
                flip = _either(stated_flip, self._readout_flip(qubits))
                if runs and runs[-1][0] == flip:
                    runs[-1][1].append(group)
                else:
                    runs.append((flip, [group]))
            busy.update(measured)
            # Runs of equal flip probability keep the targets, and so the record order, unchanged.
            lines.extend(_text(inst, run, [flip] if flip else []) for flip, run in runs)
            if resets:
                self._prep(inst.name, measured, lines)
        return busy

    def _reset(self, inst: stim.CircuitInstruction, busy: set[int], lines: list[str]) -> set[int]:
        chunks = _disjoint_chunks(inst.target_groups())
        for chunk in chunks:
            qubits = [t.value for group in chunk for t in group]
            self._refuse_disabled("reset", inst.name, qubits)
            busy.update(qubits)
            lines.append(str(inst) if len(chunks) == 1 else _text(inst, chunk))
            self._prep(inst.name, qubits, lines)
        return busy

    def _refuse_controlled_pauli(
        self, inst: stim.CircuitInstruction, group: list[stim.GateTarget]
    ) -> None:
        for position, target in enumerate(group):
            pauli = _CONTROLLED_PAULI.get((inst.name, position))
            if pauli and target.qubit_value is not None:
                wires = (self.physical[target.value],)
                self._refuse(gate_name(pauli, self.defined), _text(inst, [group]), wires)

    def _refuse_disabled(self, name: str, stim_name: str, qubits: Iterable[int]) -> None:
        for q in qubits:
            self._refuse(name, f"{stim_name} {q}", (self.physical[q],))

    def _refuse(self, name: str, instruction: str, wires: tuple[int, ...]) -> None:
        try:
            refuse_disabled(self.table.gate(name, wires))
        except DisabledGateError as exc:
            raise self._explain(instruction, name, wires, exc, self._enabled) from None

    def _prep(self, name: str, qubits: list[int], lines: list[str]) -> None:
        noise: dict[str, list[int]] = {}
        for q in qubits:
            error = self.table.qubit(self.physical[q]).prep_error
            if error is None:
                self.unknown["prep"].add(self.physical[q])
            elif error > 0:
                noise.setdefault(f"{_PREP_FLIP[name]}({error!r})", []).append(q)
        lines.extend(_noise_lines(noise))

    # noise values -----------------------------------------------------------------------

    def _gate_channel(self, stim_name: str, qubits: tuple[int, ...]) -> _Resolved:
        key = (stim_name, qubits)
        if key not in self._gate_noise:
            self._gate_noise[key] = self._resolve(stim_name, qubits)
        resolved = self._gate_noise[key]
        if resolved.events:
            self._applied[key] += self._passes
        return resolved

    def _resolve(self, stim_name: str, qubits: tuple[int, ...]) -> _Resolved:
        """Twirled channel of one gate and the report events that resolving it counted."""
        wires = tuple(self.physical[q] for q in qubits)
        name = gate_name(stim_name, self.defined)
        instruction = f"{stim_name} {' '.join(map(str, qubits))}"
        if stim_name in _NOISELESS:
            self._refuse(name, instruction, wires)
            return _Resolved(())
        before = {event: Counter(counts) for event, counts in self.report.events.items()}
        try:
            built = resolve_op(
                self.table, name, wires, unknown_gates=self.unknown_gates, report=self.report
            )
        except (MissingCalibrationError, DisabledGateError, LayoutError) as exc:
            raise self._explain(instruction, name, wires, exc, self._resolves) from exc
        events = _added_events(before, self.report.events)
        if len(qubits) > 2:
            probs = pauli_twirl(built.channels, wires)
            return _Resolved(events, lines=_correlated_errors(probs, qubits))
        return _Resolved(events, prefix=self._twirl(built.channels, wires))

    def _explain(
        self,
        instruction: str,
        name: str,
        wires: tuple[int, ...],
        exc: NoiseVaultError,
        runs: Callable[[str, tuple[int, ...]], bool],
    ) -> NoiseVaultError:
        steps = [exc.hint] if exc.hint else []
        if len(wires) == 2 and (hint := self._layout_hint(name, wires, runs)):
            steps.append(hint)
        return type(exc)(
            f"{instruction} (physical {qubit_loci(wires)}): {exc.message}",
            hint="; ".join(steps) or None,
        )

    def _layout_hint(
        self, name: str, wires: tuple[int, ...], runs: Callable[[str, tuple[int, ...]], bool]
    ) -> str | None:
        """A layout step only when ``name`` runs on a pair the step can choose."""
        table = self.table
        pairs = table.edges() if table.all_to_all else table.listed_pairs()
        sides = [
            side
            for pair in pairs
            if can_measure(table, pair[0]) and can_measure(table, pair[1])
            for side in (pair, pair[::-1])
            if runs(name, side)
        ]
        if not sides:
            return None
        both = set(sides) & {side[::-1] for side in sides}
        if not all(pair in both for pair in pairs if self._usable_pair(pair)):
            return f"pass layout= to put {name} on {qubit_loci(sides[0])}, where it runs"
        if not (table.all_to_all or tuple(sorted(wires)) in table.listed_pairs()):
            return (
                "pass layout= to put 2-qubit gates on connected pairs"
                " (profile.suggest_layout(n) proposes a layout, and"
                " noisevault.stim.layout_from_coords matches the circuit's QUBIT_COORDS)"
            )
        return (
            "pass layout= to put 2-qubit gates on pairs with a usable 2-qubit gate"
            " (profile.suggest_layout(n) proposes a chain of such pairs)"
        )

    def _enabled(self, name: str, wires: tuple[int, ...]) -> bool:
        try:
            refuse_disabled(self.table.gate(name, wires))
        except DisabledGateError:
            return False
        return True

    @functools.cached_property
    def _scratch(self) -> Report:
        return Report.start(self.profile, "stim", stim.__version__)

    def _resolves(self, name: str, wires: tuple[int, ...]) -> bool:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                resolve_op(
                    self.table, name, wires, unknown_gates=self.unknown_gates, report=self._scratch
                )
        except (NoiseVaultError, ValueError):
            return False
        return True

    def _usable_pair(self, pair: tuple[int, int]) -> bool:
        table = self.table
        if not (can_measure(table, pair[0]) and can_measure(table, pair[1])):
            return False
        return any(isinstance(table.typical(2, side), GateNoise) for side in (pair, pair[::-1]))

    def _twirl(self, channels: Sequence[ChannelSpec], wires: tuple[int, ...]) -> str:
        # Devices with shared defaults repeat the same channel on every pair: twirl it once.
        key = tuple(
            (c.kind, tuple(wires.index(w) for w in c.wires), b"".join(k.tobytes() for k in c.kraus))
            for c in channels
        )
        if key not in self._twirls:
            self._twirls[key] = _pauli_channel(pauli_twirl(channels, wires) if channels else ())
        return self._twirls[key]

    def _idle_noise(self, q: int) -> str:
        if q not in self._idle:
            index = self.physical[q]
            channel = None
            if self.table.qubit(index).relaxation_unknown:
                self.unknown["idle"].add(index)
            else:
                channel = idle_channel(self.table, index, float(self.tick_ns or 0.0), self.report)
            self._idle[q] = "" if channel is None else self._twirl([channel], (index,))
        return self._idle[q]

    def _readout_flip(self, qubits: Iterable[int]) -> float:
        """Symmetric flip of the recorded parity of ``qubits``, each read out independently."""
        if self.readout != "symmetrize":
            return 0.0
        parity = 1.0
        for q in qubits:
            a, b = self._readout(q)
            parity *= 1.0 - (a + b)
        return (1.0 - parity) / 2.0

    def _readout(self, q: int) -> tuple[float, float]:
        pair = self.table.qubit(self.physical[q]).readout
        if pair is None:
            self.unknown["readout"].add(self.physical[q])
            return 0.0, 0.0
        return pair

    def record_flips(self, circuit: stim.Circuit) -> np.ndarray:
        """(P(flip | recorded 0), P(flip | recorded 1)) for every measurement record, in order."""
        rows: list[tuple[float, float]] = []
        for inst in circuit.flattened():
            kind = _kind(inst.name)
            if inst.name in _HERALDED or inst.name == "MPAD":
                rows += [(0.0, 0.0)] * len(inst.targets_copy())
            elif kind == "measure":
                for group in inst.target_groups():
                    product = _measured_product(inst.name, group)
                    p1_given_0, p0_given_1 = (
                        self._readout(product.qubits[0]) if product.qubits else (0.0, 0.0)
                    )
                    rows.append(
                        (p0_given_1, p1_given_0) if product.negative else (p1_given_0, p0_given_1)
                    )
        return np.array(rows, dtype=float).reshape(-1, 2)

    def finish(self) -> None:
        """Count every gate application's events and report the unknown values."""
        for key, applied in self._applied.items():
            for event, what, n in self._gate_noise[key].events:
                self.report.count(event, what, n * (applied - 1))  # resolving counted one
        texts = {
            "readout": "readout error",
            "prep": "reset error",
            "idle": "T1/T2 for idle noise",
        }
        for key, qubits in self.unknown.items():
            if qubits:
                self.report.mark_unknown(
                    LociText(f"{texts[key]} of physical ", [(q,) for q in sorted(qubits)])
                )


def _describe(
    report: Report,
    found: _Scan,
    readout: Readout,
    tick_ns: float | None,
    existing_noise: ExistingNoise,
) -> None:
    report.approximate(
        "gate noise",
        "Pauli twirl of each gate's channel",
        "keeps average gate fidelity but drops relaxation's bias toward |0>",
    )
    report.approximate(
        "detector_error_model()",
        "Pauli channel components treated as independent",
        "Stim's approximate_disjoint_errors, an O(p^2) change",
    )
    if readout == "symmetrize":
        report.approximate(
            "readout error",
            "symmetric flip with the mean of P(1|0) and P(0|1)",
            "Stim flips results symmetrically, but readout='exact' keeps the asymmetry",
        )
    elif readout == "exact":
        report.mark_exact("readout error, applied by sample_with_readout")
        report.approximate(
            "readout error inside the circuit",
            "none",
            "Stim's own samplers and detector_error_model() see perfect readout",
        )
    else:
        report.omit("readout error (readout='none')")
    report.omit("initial state preparation error (qubits start in |0> unless reset)")
    if tick_ns is None:
        report.omit("idle noise (pass tick_ns= to add relaxation at each TICK)")
    else:
        report.approximate(
            "idle noise",
            f"twirled relaxation for {tick_ns} ns on each qubit idle in a TICK layer",
            "qubits busy in a layer get only their gate or readout noise, and the layer after"
            " the last TICK gets no idle noise",
        )
    if found.noise is not None:
        if existing_noise == "keep":
            report.approximate(
                "noise already in the circuit", "kept, with the profile's noise added"
            )
        else:
            report.omit("noise already in the circuit (stripped)")


# helpers ----------------------------------------------------------------------------------


@functools.cache
def _kind(name: str) -> Kind:
    if name == "TICK":
        return "tick"
    if name in _ANNOTATIONS:
        return "annotation"
    if name == "MPAD":
        return "pad"
    data = stim.gate_data(name)
    if data.is_noisy_gate and (name in _HERALDED or not data.produces_measurements):
        return "noise"
    if data.produces_measurements:
        return "measure"
    if data.is_reset:
        return "reset"
    if data.is_unitary:
        return "gate"
    raise ValueError(f"the Stim export does not handle {name} instructions")


def gate_name(stim_name: str, defined: Container[str] = ()) -> str:
    """The canonical NoiseVault name a Stim gate takes its noise from.

    ``defined`` is a profile's gate names. A gate equal to a rotation at a fixed angle takes the
    rotation's name when ``defined`` has the rotation but not the gate (see
    :func:`~noisevault.conversion.native_name`). ``SQRT_X`` is ``sx``, or ``rx`` on a profile
    with ``rx`` but no ``sx``. ``SQRT_XX`` is ``ms`` when the profile has ``ms``, else ``rxx``.
    Gates outside the registry keep their lowercased Stim name and get the typical-noise rule.
    """
    if stim_name not in _NAMES:
        return stim_name.lower()
    name, rotation = _NAMES[stim_name]
    return native_name(name, defined, rotation)


def _pauli_channel(probs: Sequence[float]) -> str:
    """PAULI_CHANNEL_1/2 prefix. The label order of metrics is Stim's, first letter on target 0."""
    if not any(probs):
        return ""
    return f"{_PAULI_CHANNEL[len(probs)]}({','.join(map(repr, probs))})"


def _correlated_errors(probs: Sequence[float], qubits: Sequence[int]) -> tuple[str, ...]:
    """The Pauli channel ``probs`` on 3+ ``qubits``, which no PAULI_CHANNEL instruction takes.

    An ELSE_CORRELATED_ERROR fires only when no earlier error in its chain fired. Thus each
    error takes its probability divided by the chance that no earlier error fired.
    """
    lines: list[str] = []
    untouched = 1.0
    for label, p in zip(metrics.pauli_labels(len(qubits)), probs, strict=True):
        if p > 0:
            targets = " ".join(f"{c}{q}" for c, q in zip(label, qubits, strict=True) if c != "I")
            given = p / untouched if p < untouched else 1.0
            lines.append(f"{'ELSE_' if lines else ''}CORRELATED_ERROR({given!r}) {targets}")
            untouched -= p
    return tuple(lines)


def _noise_lines(noise: dict[str, list[int]]) -> list[str]:
    return [f"{prefix} {' '.join(map(str, qubits))}" for prefix, qubits in noise.items()]


def _disjoint_chunks(groups: list[list[stim.GateTarget]]) -> list[list[list[stim.GateTarget]]]:
    """Split target groups so no qubit repeats in a chunk, keeping each gate's noise in order."""
    chunks: list[list[list[stim.GateTarget]]] = [[]]
    used: set[int] = set()
    for group in groups:
        qubits = {t.qubit_value for t in group if t.qubit_value is not None}
        if used & qubits:
            chunks.append([])
            used = set()
        chunks[-1].append(group)
        used |= qubits
    return chunks


def _acted_on(name: str, group: Sequence[stim.GateTarget]) -> list[int]:
    if name not in _COMBINED:
        return [t.value for t in group if t.qubit_value is not None]
    return _pauli_product(group).qubits


class _PauliProduct(NamedTuple):
    qubits: list[int]
    negative: bool


def _measured_product(name: str, group: Sequence[stim.GateTarget]) -> _PauliProduct:
    if name in _COMBINED:
        return _pauli_product(group)
    inverted = sum(t.is_inverted_result_target for t in group)
    return _PauliProduct([t.value for t in group], inverted % 2 == 1)


def _pauli_product(group: Sequence[stim.GateTarget]) -> _PauliProduct:
    paulis: dict[int, int] = {}
    power, negative = 0, False
    for t in group:
        pauli = _Y if t.is_y_target else _X if t.is_x_target else _Z if t.is_z_target else 0
        before = paulis.get(t.value, 0)
        power += _I_POWER.get((before, pauli), 0)
        paulis[t.value] = before ^ pauli
        negative ^= t.is_inverted_result_target
    # Stim's simulators refuse a product with an imaginary phase, such as ``X0*Z0``, so its
    # sign does not matter.
    return _PauliProduct([q for q, p in paulis.items() if p], negative != (power % 4 == 2))


def _text(
    inst: stim.CircuitInstruction,
    groups: list[list[stim.GateTarget]],
    args: list[float] | None = None,
) -> str:
    targets: list[stim.GateTarget] = []
    for group in groups:
        for i, target in enumerate(group):
            if i and inst.name in _COMBINED:
                targets.append(stim.target_combiner())
            targets.append(target)
    args = inst.gate_args_copy() if args is None else args
    return str(stim.CircuitInstruction(inst.name, targets, args, tag=inst.tag))


def _added_events(before: dict[str, Counter[str]], after: dict[str, Counter[str]]) -> EventCounts:
    return tuple(
        (event, what, n)
        for event, counts in after.items()
        for what, n in (counts - before.get(event, Counter())).items()
    )


def _either(p: float, q: float) -> float:
    """Probability that exactly one of two independent flips happens."""
    return p + q - 2.0 * p * q


def _short(inst: stim.CircuitInstruction) -> str:
    text = str(inst)
    return text if len(text) <= 60 else text[:57] + "..."


def _as_circuit(circuit: Any) -> stim.Circuit:
    if isinstance(circuit, str):
        return stim.Circuit(circuit)
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(f"expected a stim.Circuit or Stim program text, got {type(circuit)!r}")
    return circuit


def _check_choice(name: str, value: Any, choices: Any) -> None:
    options = get_args(choices)
    if value not in options:
        raise ValueError(f"{name}={value!r}: choose one of {', '.join(map(repr, options))}")


_D8 = [np.array(m, dtype=float) for m in (
    [[1, 0], [0, 1]], [[0, -1], [1, 0]], [[-1, 0], [0, -1]], [[0, 1], [-1, 0]],
    [[1, 0], [0, -1]], [[-1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1], [-1, 0]],
)]  # fmt: skip
_TURN_45 = 0.5 * np.array([[1.0, 1.0], [1.0, -1.0]])  # diagonal neighbors become grid neighbors
_TRANSFORMS = _D8 + [d @ _TURN_45 for d in _D8]


def _placeable(table: NoiseTable, name: str, arity: int) -> bool:
    """False when no enabled qubit or pair of the device allows ``name``."""
    if arity == 1:
        sides: Iterable[tuple[int, ...]] = ((q,) for q in range(table.num_qubits))
    elif arity == 2:
        sides = (side for pair in table.edges() for side in (pair, pair[::-1]))
    else:
        return True
    for side in sides:
        found = table.gate(name, side)
        disabled = isinstance(found, GateNoise) and found.state == "disabled"
        if not (disabled or any(table.qubit(q).disabled for q in side)):
            return True
    return False


class _Device:
    """Enabled device qubits with coords, looked up by position (to 1e-6) in bulk."""

    def __init__(self, profile: Profile, placeable: bool) -> None:
        self.hint = "pass layout={stim qubit: physical qubit}" if placeable else None
        usable = [
            (q.coords[:2], q.index)
            for q in profile.qubits
            if q.coords is not None and len(q.coords) >= 2 and not q.disabled
        ]
        if not usable:
            raise LayoutError(
                f"{profile.id} records no qubit coords",
                hint=self.hint,
            )
        self.profile_id = profile.id
        self.index = np.array([i for _, i in usable], dtype=int)
        self.coords = np.array([c for c, _ in usable], dtype=float)
        self.grid = _grid_units(self.coords)
        self.low, self.high = self.grid.min(axis=0), self.grid.max(axis=0)
        keys = self._keys(self.grid)
        self.order = np.argsort(keys, kind="stable")
        self.sorted_keys = keys[self.order]
        self.shared = np.unique(self.sorted_keys[1:][np.diff(self.sorted_keys) == 0])

    def placements(self, points: np.ndarray) -> np.ndarray:
        """Physical qubits of every shift that lands all points on device qubits.

        Shape (shifts, points); a shift puts points[0] on a device qubit, in device order.
        """
        offsets = _grid_units(points - points[0])
        # Shifts that would push the circuit's bounding box off the device cannot fit.
        fits = (self.grid + offsets.min(axis=0) >= self.low).all(axis=1)
        fits &= (self.grid + offsets.max(axis=0) <= self.high).all(axis=1)
        keys = self._keys(self.grid[fits][:, None, :] + offsets[None, :, :])
        at = np.searchsorted(self.sorted_keys, keys).clip(max=len(self.sorted_keys) - 1)
        whole = (self.sorted_keys[at] == keys).all(axis=1)
        matched = keys[whole]
        hit = np.isin(matched, self.shared)
        if hit.any():
            self._refuse_shared(int(matched[hit][0]))
        return self.index[self.order[at[whole]]]

    def _refuse_shared(self, key: int) -> NoReturn:
        rows = self.order[self.sorted_keys == key]
        x, y = self.coords[rows[0]]
        qubits = qubit_loci(*((int(i),) for i in self.index[rows]), limit=None)
        raise LayoutError(
            f"{self.profile_id} has {qubits}"
            f" at coords ({x:g}, {y:g}), so the circuit's QUBIT_COORDS have no single device"
            " qubit there",
            hint=self.hint,
        )

    def _keys(self, grid: np.ndarray) -> np.ndarray:
        """One integer per point inside the device's bounding box."""
        inside = grid - self.low
        return inside[..., 0] * (self.high[1] - self.low[1] + 1) + inside[..., 1]


def _distinct_rows(rows: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """The distinct rows of ``rows``, with entries in 0..n-1, and the index of each row in them."""
    if n ** rows.shape[1] > np.iinfo(np.int64).max:
        return np.unique(rows, axis=0, return_inverse=True)
    codes = rows @ n ** np.arange(rows.shape[1], dtype=np.int64)
    _, first, inverse = np.unique(codes, return_index=True, return_inverse=True)
    return rows[first], inverse


def _grid_units(points: np.ndarray) -> np.ndarray:
    return np.rint(points * 1e6).astype(np.int64)


class _PlacementCost:
    """Summed 2-qubit error of the circuit's pairs plus mean readout error, or inf if unusable.

    A placement is unusable when a pair has no usable 2-qubit gate, or when an operation lands
    where the profile disables it.
    """

    def __init__(
        self, profile: Profile, pairs: np.ndarray, operations: list[tuple[str, np.ndarray]]
    ) -> None:
        self.table = profile.table
        self.pairs = pairs  # (pairs, 2) columns of a placement
        disabled = {name for name, spec in profile.gates.items() if spec.disabled}
        disabled |= {record.gate for record in profile.calibrations if record.disabled}
        self.operations = [(name, columns) for name, columns in operations if name in disabled]
        self._disabled: dict[tuple[str, tuple[int, ...]], bool] = {}
        self.readout = np.array(
            [
                sum(r) / 2 if (r := self.table.qubit(i).readout) else 0.0
                for i in range(self.table.num_qubits)
            ],
            dtype=float,
        )
        self._pair: dict[int, float] = {}
        # Pairs that no 2-qubit gate can use skip the (slow) typical-gate lookup.
        self.maybe: np.ndarray | None = None
        if not self.table.all_to_all:
            n = self.table.num_qubits
            linked = self.table.listed_pairs()
            codes = [a * n + b for a, b in linked] + [b * n + a for a, b in linked]
            self.maybe = np.array(codes, dtype=np.int64)

    def __call__(self, placements: np.ndarray) -> np.ndarray:
        total = self.readout[placements].sum(axis=1)
        if len(self.pairs):
            n = self.table.num_qubits
            codes = placements[:, self.pairs[:, 0]] * n + placements[:, self.pairs[:, 1]]
            unique, inverse = np.unique(codes, return_inverse=True)
            errors = np.full(len(unique), np.nan)
            maybe = (
                np.ones(len(unique), bool) if self.maybe is None else np.isin(unique, self.maybe)
            )
            errors[maybe] = [self._pair_error(int(c), n) for c in unique[maybe]]
            total = total + errors[inverse.reshape(codes.shape)].sum(axis=1)
        return np.where(np.isnan(total) | self._blocked(placements), np.inf, total)

    def _blocked(self, placements: np.ndarray) -> np.ndarray:
        blocked = np.zeros(len(placements), dtype=bool)
        for name, columns in self.operations:
            wires = placements[:, columns].reshape(-1, columns.shape[1])
            unique, inverse = _distinct_rows(wires, self.table.num_qubits)
            off = np.array([self._is_disabled(name, tuple(map(int, row))) for row in unique])
            blocked |= off[inverse.reshape(len(placements), -1)].any(axis=1)
        return blocked

    def _is_disabled(self, name: str, wires: tuple[int, ...]) -> bool:
        key = (name, wires)
        if key not in self._disabled:
            found = self.table.gate(name, wires)
            self._disabled[key] = isinstance(found, GateNoise) and found.state == "disabled"
        return self._disabled[key]

    def _pair_error(self, code: int, n: int) -> float:
        if code not in self._pair:
            found = self.table.typical(2, divmod(code, n))
            ok = isinstance(found, GateNoise)
            self._pair[code] = (found.avg_infidelity or 0.0) if ok else np.nan  # type: ignore[union-attr]
        return self._pair[code]
