"""Counts files from the benchmark data that Quantinuum publishes.

    python -m study.quantinuum DATA_ROOT --out FOLDER

Quantinuum publishes the raw shots of its benchmarks under Apache-2.0 at
https://github.com/Quantinuum/quantinuum-hardware-specifications. A dataset is a folder
``data/<machine>/<YYYY_MM_DD>/`` with one JSON file for each benchmark, such as ``SQ_RB.json``
and ``TQ_RB.json``. A file holds an OpenQASM 2.0 program and the shots of each circuit.

The command writes, for each machine, date and benchmark, a counts file and the profile file
beside it: ``<machine>/<date>/<benchmark>.counts.json`` and ``<benchmark>.profile.json``.
``study/derive.py`` reads the output folder. A file with more outcomes than one counts file
holds goes into parts, ``<benchmark>_part<k>``.

One program runs the same benchmark on several qubits or pairs at the same time. The converter
makes one small circuit for each group of qubits that the gates connect: a qubit, or a pair,
with the ancilla qubits of its leakage check. The counts of a small circuit are the marginal
counts of its own classical bits. The noise model of a profile has no crosstalk, so the model
gives each group the same marginal as a simulation of the whole program. The reference
simulator runs at most 10 qubits, so the small circuits make the simulation possible.

The SQ_RB and TQ_RB files check each data qubit for leakage after the sequence. An ancilla
qubit gets ``x``, ``h``, two ZZ gates with the data qubit and ``h``. Then a measurement writes
the ancilla to the register ``l``. A bit of 1 in ``l`` flags leakage. Each shot of the files
holds the bits of ``c`` and of ``l``. Thus the default, ``--leakage measure``, keeps the ancilla
and the gates of the check in each circuit, so the ops are the ops that the device ran.
``--leakage postselect`` keeps only the shots in which the ancillas read 0, and leaves out the
ancillas and their gates.

``role`` gives ``train`` to a sequence with an even index and ``test_near`` to an odd index.

The counts bind to the bundled profile of the same machine whose calibration date is the date
of the folder. ``source`` is ``hardware``: the counts format has only ``hardware`` and
``simulated``. ``execution.client`` and ``execution.options`` say that the counts are published
vendor data and name the file that they come from.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import statistics
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

from noisevault import catalog, gates, metrics
from noisevault.counts import COUNTS_FORMAT, MeasuredCounts
from noisevault.profile import Profile
from noisevault.reference import Op
from noisevault.table import GateNoise

REPOSITORY = "https://github.com/Quantinuum/quantinuum-hardware-specifications"
CLIENT = "Quantinuum hardware specifications, published vendor data"
HERALD = "l"
MAX_QUBITS = 10
MAX_OUTCOMES = 4096
Leakage = Literal["measure", "postselect"]
Role = Literal["train", "test_near"]

_INCLUDES = frozenset({"qelib1.inc", "hqslib1.inc", "hqslib1_dev.inc"})
_FILE_NAMES = {"spam": "SPAM"}


class QasmError(ValueError):
    """A statement that the parser does not know, or a program that it cannot convert."""


# the gates ------------------------------------------------------------------------------------

Mapping1 = Callable[[tuple[float, ...], tuple[int, ...]], list[Op]]


def _r(theta: float, phi: float) -> Mapping1:
    return lambda p, q: [Op("r", q, (theta, phi))]


def _u1q(p: tuple[float, ...], q: tuple[int, ...]) -> list[Op]:
    return [Op("r", q, (p[0], p[1]))]


def _rx(p: tuple[float, ...], q: tuple[int, ...]) -> list[Op]:
    return [Op("r", q, (p[0], 0.0))]


def _ry(p: tuple[float, ...], q: tuple[int, ...]) -> list[Op]:
    return [Op("r", q, (p[0], math.pi / 2))]


def _rz(p: tuple[float, ...], q: tuple[int, ...]) -> list[Op]:
    return [Op("rz", q, (p[0],))]


def _h(p: tuple[float, ...], q: tuple[int, ...]) -> list[Op]:
    return [Op("r", q, (math.pi / 2, -math.pi / 2)), Op("rz", q, (math.pi,))]


def _zz(p: tuple[float, ...], q: tuple[int, ...]) -> list[Op]:
    return [Op("zz", q, ())]


def _rzz(p: tuple[float, ...], q: tuple[int, ...]) -> list[Op]:
    # RZZ(pi/2) is the native ZZ gate, which the profiles calibrate as zz.
    if math.isclose(math.remainder(p[0] - math.pi / 2, 2 * math.pi), 0.0, abs_tol=1e-12):
        return [Op("zz", q, ())]
    return [Op("rzz", q, (p[0],))]


@dataclass(frozen=True)
class QasmGate:
    arity: int
    params: int
    ops: Mapping1
    definition: str


# Each row maps a gate of qelib1.inc or hqslib1_dev.inc to registry gates. The definition
# column repeats the vendor definition. tests/test_study_quantinuum.py checks each row.
# hqslib1_dev.inc declares ZZ opaque. Its commented decomposition is ZZ followed by Rz(pi) on
# q1. The expected outputs of the vendor files agree with exp(-i pi/4 Z Z), not with the comment.
GATES: Mapping[str, QasmGate] = {
    "U1q": QasmGate(1, 2, _u1q, "U(theta, phi-pi/2, pi/2-phi)"),
    "Rz": QasmGate(1, 1, _rz, "U(0, 0, lam)"),
    "rz": QasmGate(1, 1, _rz, "Rz(phi)"),
    "u1": QasmGate(1, 1, _rz, "Rz(lam)"),
    "rx": QasmGate(1, 1, _rx, "U1q(theta, 0)"),
    "ry": QasmGate(1, 1, _ry, "U1q(theta, pi/2)"),
    "x": QasmGate(1, 0, _r(math.pi, 0.0), "U1q(pi, 0)"),
    "y": QasmGate(1, 0, _r(math.pi, math.pi / 2), "U1q(pi, pi/2)"),
    "z": QasmGate(1, 0, lambda p, q: [Op("rz", q, (math.pi,))], "Rz(pi)"),
    "h": QasmGate(1, 0, _h, "U1q(pi/2, -pi/2); Rz(pi)"),
    "id": QasmGate(1, 0, _r(0.0, math.pi / 2), "u3(0, 0, 0) = Rz(0); U1q(0, pi/2); Rz(0)"),
    "ZZ": QasmGate(2, 0, _zz, "RZZ(pi/2) = exp(-i pi/4 Z Z)"),
    "RZZ": QasmGate(2, 1, _rzz, "exp(-i theta/2 Z Z)"),
    "rzz": QasmGate(2, 1, _rzz, "RZZ(theta)"),
}  # fmt: skip


# the parser -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Program:
    """A parsed program. Physical qubit i is the i-th qubit over the qregs in declaration order.

    ``ops`` act on physical qubits. ``measured`` maps each measured qubit to its classical bit,
    as (creg, index).
    """

    qregs: Mapping[str, tuple[int, int]]
    cregs: Mapping[str, int]
    ops: tuple[Op, ...]
    measured: Mapping[int, tuple[str, int]]

    @property
    def num_qubits(self) -> int:
        return sum(size for _, size in self.qregs.values())


_STATEMENT = re.compile(r"^([A-Za-z_]\w*)\s*(?:\((.*)\))?\s*(.*)$", re.S)
_IF = re.compile(r"^if\s*\(\s*([A-Za-z_]\w*)\s*==\s*(\d+)\s*\)\s*(.+)$", re.S)
_DECLARATION = re.compile(r"^(qreg|creg)\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]$")
_INCLUDE = re.compile(r'^include\s+"([^"]+)"$')
_ARGUMENT = re.compile(r"^([A-Za-z_]\w*)\s*(?:\[\s*(\d+)\s*\])?$")
_DIRECTIVE = re.compile(r"^(order|group)(\d+)$")


def parse_qasm(text: str) -> Program:
    """The ops and measurements of an OpenQASM 2.0 program in the subset of the vendor files.

    The parser knows the gates of ``GATES``, ``measure``, ``barrier``, the ``orderN`` and
    ``groupN`` directives of hqslib1_dev.inc and ``if`` on a classical register that no
    measurement has written. It raises QasmError for every other statement. ``barrier``,
    ``orderN`` and ``groupN`` change only the schedule, so they give no op.
    """
    parser = _Parser()
    statements = [s.strip() for s in re.sub(r"//[^\n]*", "", text).split(";")]
    if not statements or statements[0] != "OPENQASM 2.0":
        raise QasmError("the program does not start with OPENQASM 2.0")
    for statement in statements[1:]:
        if statement:
            parser.statement(" ".join(statement.split()))
    return Program(dict(parser.qregs), dict(parser.cregs), tuple(parser.ops), dict(parser.measured))


class _Parser:
    def __init__(self) -> None:
        self.qregs: dict[str, tuple[int, int]] = {}
        self.cregs: dict[str, int] = {}
        self.ops: list[Op] = []
        self.measured: dict[int, tuple[str, int]] = {}
        self.written: set[str] = set()

    def statement(self, text: str) -> None:
        if match := _INCLUDE.match(text):
            if match.group(1) not in _INCLUDES:
                raise QasmError(f"{text!r}: the parser does not know {match.group(1)}")
            return
        if match := _DECLARATION.match(text):
            self._declare(match.group(1), match.group(2), int(match.group(3)))
            return
        if match := _IF.match(text):
            self._conditional(match.group(1), int(match.group(2)), match.group(3))
            return
        if text.startswith("measure "):
            self._measure(text)
            return
        match = _STATEMENT.match(text)
        if match is None:
            raise QasmError(f"{text!r} is not a statement that the parser knows")
        name, params, arguments = match.groups()
        if name == "barrier":
            self._qubits(arguments, text, whole=True)
            return
        if directive := _DIRECTIVE.match(name):
            found = self._qubits(arguments, text, whole=False)
            if params or len(found) != int(directive.group(2)):
                raise QasmError(f"{text!r}: {name} takes {directive.group(2)} qubits")
            return
        gate = GATES.get(name)
        if gate is None:
            raise QasmError(f"{text!r}: the parser does not know the statement {name}")
        values = tuple(_angle(p, text) for p in _split(params)) if params else ()
        qubits = self._qubits(arguments, text, whole=False)
        if len(values) != gate.params or len(qubits) != gate.arity:
            raise QasmError(
                f"{text!r}: {name} takes {gate.params} parameters and {gate.arity} qubits"
            )
        if len(set(qubits)) != len(qubits):
            raise QasmError(f"{text!r}: the qubits of {name} must be distinct")
        if (used := next((q for q in qubits if q in self.measured), None)) is not None:
            raise QasmError(f"{text!r}: qubit {used} has a gate after its measurement")
        self.ops.extend(gate.ops(values, qubits))

    def _declare(self, kind: str, name: str, size: int) -> None:
        if name in self.qregs or name in self.cregs:
            raise QasmError(f"the register {name} is declared two times")
        if kind == "qreg":
            self.qregs[name] = (sum(s for _, s in self.qregs.values()), size)
        else:
            self.cregs[name] = size

    def _conditional(self, register: str, value: int, body: str) -> None:
        if register not in self.cregs:
            raise QasmError(f"if ({register}=={value}): {register} is not a classical register")
        if register in self.written:
            raise QasmError(
                f"if ({register}=={value}) follows a measurement into {register}, so the"
                " parser cannot know if the gate runs"
            )
        # Before a measurement, every classical bit is 0.
        if value == 0:
            self.statement(body)
            return
        skipped = _Parser()
        skipped.qregs, skipped.cregs = self.qregs, self.cregs
        skipped.statement(body)

    def _measure(self, text: str) -> None:
        match = re.match(r"^measure (.+?) -> (.+)$", text)
        if match is None:
            raise QasmError(f"{text!r} is not a measurement")
        qubits = self._qubits(match.group(1), text, whole=True)
        creg = _ARGUMENT.match(match.group(2).strip())
        if creg is None or creg.group(1) not in self.cregs:
            raise QasmError(f"{text!r}: {match.group(2)} is not a classical bit")
        name, size = creg.group(1), self.cregs[creg.group(1)]
        whole = creg.group(2) is None
        if whole != ("[" not in match.group(1)):
            raise QasmError(f"{text!r}: measure a qubit into a bit or a register into a register")
        if whole and len(qubits) != size:
            raise QasmError(f"{text!r}: the registers have different sizes")
        bits = [(name, k) for k in range(size)] if whole else [(name, int(creg.group(2)))]
        for qubit, bit in zip(qubits, bits, strict=True):
            if bit[1] >= size:
                raise QasmError(f"{text!r}: {name} has no bit {bit[1]}")
            if qubit in self.measured:
                raise QasmError(f"{text!r}: qubit {qubit} is measured two times")
            if bit in self.measured.values():
                raise QasmError(f"{text!r}: the bit {name}[{bit[1]}] is written two times")
            self.measured[qubit] = bit
        self.written.add(name)

    def _qubits(self, arguments: str, text: str, *, whole: bool) -> tuple[int, ...]:
        found: list[int] = []
        for argument in _split(arguments):
            match = _ARGUMENT.match(argument)
            if match is None or match.group(1) not in self.qregs:
                raise QasmError(f"{text!r}: {argument!r} is not a qubit")
            offset, size = self.qregs[match.group(1)]
            if match.group(2) is None:
                if not whole:
                    raise QasmError(f"{text!r}: give one qubit, not the register {argument}")
                found.extend(range(offset, offset + size))
                continue
            index = int(match.group(2))
            if index >= size:
                raise QasmError(f"{text!r}: {match.group(1)} has no qubit {index}")
            found.append(offset + index)
        return tuple(found)


def _split(text: str) -> list[str]:
    return [part.strip() for part in text.split(",")] if text.strip() else []


def _angle(text: str, statement: str) -> float:
    """The value of an angle with numbers, pi, + - * / and parentheses. Nothing else runs."""
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        raise QasmError(f"{statement!r}: {text!r} is not an angle") from None

    def value(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id == "pi":
            return math.pi
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
            inner = value(node.operand)
            return -inner if isinstance(node.op, ast.USub) else inner
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, ast.Add | ast.Sub | ast.Mult | ast.Div
        ):
            a, b = value(node.left), value(node.right)
            operations = {ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b}
            return operations[type(node.op)] if type(node.op) in operations else a / b
        raise QasmError(f"{statement!r}: {text!r} is not an angle")

    return value(tree.body)


# the converter --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Group:
    """The qubits of one small circuit, in the order of their classical bits."""

    qubits: tuple[int, ...]
    bits: tuple[tuple[str, int], ...]


def groups(program: Program) -> list[Group]:
    """The groups of qubits that the gates connect, each with at least one measured qubit.

    A group with no measured qubit has no counts, so it gives no circuit. A group with a qubit
    that no measurement reads cannot be a circuit of a counts file, so it raises QasmError.
    """
    parent = list(range(program.num_qubits))

    def root(q: int) -> int:
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    used: set[int] = set(program.measured)
    for op in program.ops:
        used.update(op.qubits)
        for other in op.qubits[1:]:
            parent[root(other)] = root(op.qubits[0])
    members: dict[int, list[int]] = {}
    for q in sorted(used):
        members.setdefault(root(q), []).append(q)
    order = list(program.cregs)
    found = []
    for qubits in members.values():
        measured = [q for q in qubits if q in program.measured]
        if not measured:
            continue
        if len(measured) != len(qubits):
            unread = next(q for q in qubits if q not in program.measured)
            raise QasmError(f"qubit {unread} has gates with measured qubits but no measurement")
        if len(qubits) > MAX_QUBITS:
            raise QasmError(
                f"the gates connect {len(qubits)} qubits, more than the {MAX_QUBITS} that the"
                " reference simulator runs"
            )
        measured.sort(key=lambda q: (order.index(program.measured[q][0]), program.measured[q][1]))
        found.append(Group(tuple(measured), tuple(program.measured[q] for q in measured)))
    found.sort(key=lambda g: (order.index(g.bits[0][0]), g.bits[0][1]))
    return found


def marginal(
    shots: Mapping[str, Sequence[str]],
    cregs: Mapping[str, int],
    bits: Sequence[tuple[str, int]],
    keep: Sequence[tuple[str, int]] = (),
) -> dict[str, int]:
    """The counts of ``bits``, keyed with ``bits[0]`` leftmost, over the shots in which every
    bit of ``keep`` is 0.

    A vendor bit string of a register of n bits holds bit k at character n - 1 - k, as Qiskit
    prints it.
    """
    registers = {name for name, _ in (*bits, *keep)}
    lengths = {len(shots[name]) for name in registers}
    if len(lengths) != 1:
        raise QasmError("the classical registers have different numbers of shots")
    counts: dict[str, int] = {}
    for i in range(lengths.pop()):
        if any(shots[name][i][cregs[name] - 1 - k] != "0" for name, k in keep):
            continue
        key = "".join(shots[name][i][cregs[name] - 1 - k] for name, k in bits)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _shot_strings(raw: Any, cregs: Mapping[str, int], where: str) -> dict[str, list[str]]:
    if isinstance(raw, list):
        if len(cregs) != 1:
            raise QasmError(f"{where}: one list of shots, but the program has {len(cregs)} cregs")
        raw = {next(iter(cregs)): raw}
    if not isinstance(raw, Mapping) or set(raw) != set(cregs):
        raise QasmError(f"{where}: the shots do not name the classical registers {sorted(cregs)}")
    for name, strings in raw.items():
        good = isinstance(strings, list) and all(
            isinstance(s, str) and len(s) == cregs[name] and set(s) <= {"0", "1"} for s in strings
        )
        if not good:
            raise QasmError(f"{where}: the shots of {name} are not strings of {cregs[name]} bits")
    return dict(raw)


_LABEL = re.compile(r"\(\s*(\d+)\s*,\s*'?([01]+|\d+)'?\s*\)|^(\d+)\s*[,-]\s*(\d+)$")


def label(key: str) -> tuple[str, str]:
    """(sequence length, sequence index) from a circuit key of a vendor file.

    The keys have the forms ``SQ_RB: (128, 3) [15]``, ``SQ_RB (128, 3)``, ``128, 3`` and
    ``32-28``. In SPAM files the second value is the prepared state.
    """
    match = _LABEL.search(key.strip())
    if match is None:
        raise QasmError(f"the circuit key {key!r} has no (length, index)")
    first, second = (match.group(1), match.group(2)) if match.group(1) else match.group(3, 4)
    return first, second


def _same_circuit(key: str) -> str:
    """A qasm key and a raw_data key of one circuit give the same text: ``SQ_RB: (2, 0) [17]``
    and ``SQ_RB (2, 0)`` both give ``SQ_RB (2, 0)``."""
    return " ".join(re.sub(r"\s*\[\d+\]\s*$", "", key).replace(":", " ").split())


def circuit_name(benchmark: str, data_bits: Sequence[int], length: str, index: str) -> str:
    """Such as ``TQ_RB_q2-3_m64_s5``: the benchmark, the qubits of the counts, the sequence
    length and the sequence index."""
    return f"{benchmark}_q{'-'.join(map(str, data_bits))}_m{length}_s{index}"


def role(name: str) -> Role:
    """``train`` for a sequence with an even index, ``test_near`` for an odd index."""
    match = re.search(r"_s(\d+)$", name)
    if match is None:
        raise ValueError(f"{name!r} is not a circuit name of study.quantinuum")
    return "train" if int(match.group(1)) % 2 == 0 else "test_near"


def circuits(
    doc: Mapping[str, Any], benchmark: str, *, leakage: Leakage = "measure", where: str = ""
) -> list[dict[str, Any]]:
    """The small circuits of one vendor file, as dicts of the counts format.

    With ``leakage="measure"``, a circuit holds its ancilla qubits, the gates of the leakage
    check and the ancilla bits (creg ``l``), so the ops are the ops that the device ran. With
    ``leakage="postselect"``, a circuit keeps only the shots in which its ancilla bits are 0,
    and it leaves out the ancillas and every gate on them. The gates of the check also act on
    the data qubits, so that form leaves out their error.
    """
    qasm = doc.get("qasm")
    raw = doc.get("raw_data", doc.get("raw data"))
    if not isinstance(qasm, Mapping) or not isinstance(raw, Mapping):
        raise QasmError(f"{where}: the file has no qasm and raw_data, only summary counts")
    shots_by_key = {_same_circuit(key): value for key, value in raw.items()}
    if len(shots_by_key) != len(raw) or set(shots_by_key) != {_same_circuit(k) for k in qasm}:
        raise QasmError(f"{where}: the keys of qasm and raw_data do not name the same circuits")
    result = []
    for key in sorted(qasm, key=lambda k: tuple(int(x) if x.isdigit() else x for x in label(k))):
        length, index = label(key)
        program = parse_qasm(qasm[key])
        shots = _shot_strings(shots_by_key[_same_circuit(key)], program.cregs, f"{where}: {key}")
        for group in groups(program):
            data = [b for b in group.bits if b[0] != HERALD]
            heralds = [b for b in group.bits if b[0] == HERALD]
            if not data:
                continue
            kept = list(group.qubits)
            keep: list[tuple[str, int]] = []
            if leakage == "postselect":
                kept = [q for q, b in zip(group.qubits, group.bits, strict=True) if b[0] != HERALD]
                keep = heralds
            bits = [program.measured[q] for q in kept]
            position = {q: i for i, q in enumerate(kept)}
            ops = [
                Op(op.name, tuple(position[q] for q in op.qubits), op.params)
                for op in program.ops
                if set(op.qubits) <= set(kept)
            ]
            counts = marginal(shots, program.cregs, bits, keep)
            if not counts:
                raise QasmError(f"{where}: {key}: no shot passes the leakage check")
            result.append(
                {
                    "name": circuit_name(benchmark, [b[1] for b in data], length, index),
                    "qubits": kept,
                    "ops": ops,
                    "shots": sum(counts.values()),
                    "counts": counts,
                }
            )
    return result


def counts_files(
    doc: Mapping[str, Any],
    benchmark: str,
    profile: Profile,
    *,
    run_at: datetime,
    file: str,
    file_sha256: str,
    leakage: Leakage = "measure",
) -> list[MeasuredCounts]:
    """The counts files of one vendor file, bound to ``profile``. ``file`` is the path of the
    vendor file in the repository, such as ``data/H1-1/2025_05_02/TQ_RB.json``."""
    found = circuits(doc, benchmark, leakage=leakage, where=file)
    if not found:
        raise QasmError(f"{file}: the file has no measured circuit")
    names = [c["name"] for c in found]
    if len(set(names)) != len(names):
        raise QasmError(f"{file}: two circuits get the same name, so the keys hold another label")
    widest = max(q for c in found for q in c["qubits"])
    if widest >= profile.device.num_qubits:
        raise QasmError(
            f"{file}: qubit {widest} is outside the {profile.device.num_qubits} qubits of"
            f" {profile.id}"
        )
    parts: list[list[dict[str, Any]]] = [[]]
    outcomes = 0
    for circuit in found:
        size = 2 ** len(circuit["qubits"])
        if outcomes + size > MAX_OUTCOMES:
            parts.append([])
            outcomes = 0
        parts[-1].append(circuit)
        outcomes += size
    options = {
        "repository": REPOSITORY,
        "file": file,
        "file_sha256": file_sha256,
        "license": "Apache-2.0",
        "leakage": leakage,
        "circuits": "one circuit per group of connected qubits, with marginal counts",
    }
    return [
        MeasuredCounts.model_validate(
            {
                "nv_counts": COUNTS_FORMAT,
                "source": "hardware",
                "profile": {"id": profile.id, "fingerprint": profile.calibration_fingerprint},
                "backend": profile.device.name,
                "run_at": run_at,
                "bit_order": "clbit0_left",
                "execution": {
                    "client": CLIENT,
                    "transpiled": False,
                    "gate_twirling": False,
                    "measure_twirling": False,
                    "dynamical_decoupling": False,
                    "init_qubits": True,
                    "options": {**options, **({"part": k + 1} if len(parts) > 1 else {})},
                },
                "circuits": part,
            }
        )
        for k, part in enumerate(parts)
    ]


# the profiles ---------------------------------------------------------------------------------


_ERROR_METRICS = ("avg_infidelity", "process_infidelity", "depolarizing_param", "pauli")


def without_one_qubit_gate_error(profile: Profile) -> Profile:
    """``profile`` with the error of every 1-qubit gate set to 0.

    A 2-qubit error from 2-qubit randomized benchmarking already holds the error of the 1-qubit
    gates in each Clifford (``includes`` has ``1q_dressing``). The reference simulator also
    charges each 1-qubit gate of a circuit, so that part of the error counts two times. This
    variant charges it one time, in the 2-qubit gate. Relaxation, readout, preparation and the
    2-qubit gates do not change. A virtual or disabled gate does not change.
    """
    data = profile.to_dict()
    specs = data["gates"]

    def one_qubit(name: str) -> bool:
        info = gates.lookup(name)
        arity = specs.get(name, {}).get("qubits") or (info.arity if info else None)
        return arity == 1

    def cleared(entry: dict[str, Any]) -> dict[str, Any]:
        if not any(key in entry for key in _ERROR_METRICS):
            return entry
        kept = {k: v for k, v in entry.items() if k not in (*_ERROR_METRICS, "stderr")}
        return {**kept, "avg_infidelity": 0.0}

    data["gates"] = {n: cleared(g) if one_qubit(n) else g for n, g in specs.items()}
    if "calibrations" in data:
        data["calibrations"] = [
            cleared(r) if len(r["qubits"]) == 1 else r for r in data["calibrations"]
        ]
    return _noted(
        data, "every 1-qubit gate error is set to 0 (study.quantinuum.without_one_qubit_gate_error)"
    )


def dressing_gates(doc: Mapping[str, Any], *, where: str = "") -> float:
    """k of a 2-qubit RB file: the mean number of charged 1-qubit gates on each qubit for each
    2-qubit gate of the sequences.

    The vendor fits the decay over the sequence length. Thus k is a growth with the length: the
    slope of the 1-qubit gates on one qubit, divided by the slope of the 2-qubit gates on the
    pair. The gates that a circuit has at each length do not count. An ``rz`` op does not
    count, because it is virtual in the bundled profiles.
    """
    lengths, ones, twos = [], [], []
    for circuit in circuits(doc, "TQ_RB", leakage="postselect", where=where):
        ops = circuit["ops"]
        lengths.append(float(circuit["name"].rsplit("_m", 1)[1].split("_s")[0]))
        charged = sum(len(op.qubits) == 1 and op.name != "rz" for op in ops)
        ones.append(charged / len(circuit["qubits"]))
        twos.append(float(sum(len(op.qubits) == 2 for op in ops)))
    if len(set(lengths)) < 2:
        raise QasmError(f"{where}: the gate count needs two sequence lengths")
    growth = statistics.linear_regression(lengths, twos).slope
    if growth <= 0:
        raise QasmError(f"{where}: the 2-qubit gates do not grow with the sequence length")
    return statistics.linear_regression(lengths, ones).slope / growth


def net_of_one_qubit_dressing(profile: Profile, gates_per_qubit: float) -> Profile:
    """``profile`` with the 1-qubit gates of the benchmark taken out of each 2-qubit gate error.

    A 2-qubit error with ``1q_dressing`` in ``includes`` holds the error of the 1-qubit gates
    in the benchmark. ``gates_per_qubit`` is k, the mean number of those gates on each qubit for
    each 2-qubit gate, as ``dressing_gates`` counts it. Process infidelities add to first order.
    Thus the net process infidelity is e - 2 k e_1, where e_1 is the process infidelity of the
    typical 1-qubit native gate. As average infidelities, r_net = r - (6 / 5) 2 k r_1. A net
    error below 0 becomes 0. A ``pauli`` vector keeps its shape: each entry gets the factor
    e_net / e. The 1-qubit gates and the readout do not change, so a circuit with no 2-qubit
    gate gets the same noise. A k below 0 adds error. Only a fit of k uses such a value.

    The profile must give each gate one error for the whole device. A calibration record with
    an error raises ValueError.
    """
    if not math.isfinite(gates_per_qubit):
        raise ValueError(f"gates_per_qubit must be a finite number, not {gates_per_qubit!r}")
    data = _uniform(profile, "net_of_one_qubit_dressing")
    stated = profile.uncorrected().table
    working = [q for q in range(stated.num_qubits) if not stated.qubit(q).disabled]
    one = stated.typical(1, working[:1])
    if not isinstance(one, GateNoise) or one.avg_infidelity is None:
        raise ValueError("the profile has no calibrated 1-qubit native gate")
    dressing = gates_per_qubit * 2 * metrics.process_from_avg(one.avg_infidelity, 1)
    for name, spec in data["gates"].items():
        metric = profile.gates[name].metric
        if stated.arity(name) != 2 or "1q_dressing" not in spec.get("includes", ()) or not metric:
            continue
        kind, value = metric
        stated_error = metrics.process_from_avg(metrics.to_avg_infidelity(kind, value, 2), 2)
        net = max(stated_error - dressing, 0.0)
        if kind == "pauli":
            factor = net / stated_error if stated_error > 0 else 1.0
            data["gates"][name] = _changed(spec, "1q_dressing", pauli=[p * factor for p in value])
        else:
            error = metrics.avg_from_process(net, 2)
            data["gates"][name] = _changed(spec, "1q_dressing", avg_infidelity=error)
    return _noted(
        data,
        f"each 2-qubit error with 1q_dressing is net of {gates_per_qubit:.6g} gates of {one.gate}"
        " on each qubit (study.quantinuum.net_of_one_qubit_dressing)",
    )


def without_leakage(profile: Profile) -> Profile:
    """``profile`` with the leakage taken out of each gate error that holds it.

    The counts of ``--leakage postselect`` hold no shot with a leakage flag. Thus their gates
    show the error without the leakage. A gate error with ``leakage`` in ``includes`` holds the
    leakage. Quantinuum states r = r_all + L / d, where r_all comes from the decay of all shots
    and L is the leakage of one gate, the probability of its ``leakage`` effect. If a leaked
    shot gives a random result, r_all holds L (d - 1) / d. Thus the average infidelity without
    the leakage is r - L, and 0 when L is more than r. In the datasets of the five bundled
    profiles, the decay of the kept shots gives r minus 0.9 L to 1.1 L. The ``leakage`` effect
    of the gate goes out of the profile. A gate with a ``pauli`` vector does not change.

    The profile must give each gate one error for the whole device. A calibration record with
    an error raises ValueError.
    """
    data = _uniform(profile, "without_leakage")
    specs = data["gates"]
    net: dict[str, float] = {}
    kept = []
    for effect in data.get("effects", ()):
        name = effect.get("gate")
        whole = effect["type"] == "leakage" and "prob" in effect and "qubits" not in effect
        stated = _default_error(profile, name) if whole and name in specs else None
        if stated is not None and "leakage" in specs[name].get("includes", ()):
            net[name] = net.get(name, stated) - effect["prob"]
        else:
            kept.append(effect)
    for name, error in net.items():
        specs[name] = _changed(specs[name], "leakage", avg_infidelity=max(error, 0.0))
    data.pop("effects", None)
    if kept:
        data["effects"] = kept
    return _noted(
        data,
        "each gate error with leakage is net of the leakage of its effect"
        " (study.quantinuum.without_leakage)",
    )


def _default_error(profile: Profile, name: str) -> float | None:
    """The average infidelity of the device-wide error of gate ``name``. None for a gate with
    no error or with a ``pauli`` vector."""
    metric, arity = profile.gates[name].metric, profile.table.arity(name)
    if metric is None or arity is None or metric[0] == "pauli":
        return None
    return metrics.to_avg_infidelity(*metric, arity)


def _uniform(profile: Profile, caller: str) -> dict[str, Any]:
    """The dict of ``profile``, which must have no calibration record with an error."""
    data = profile.to_dict()
    if any(key in record for record in data.get("calibrations", ()) for key in _ERROR_METRICS):
        raise ValueError(
            f"{caller} needs one error for each gate on the whole device, but the profile has a"
            " calibration record with an error"
        )
    return data


def _changed(spec: Mapping[str, Any], included: str, **metric: Any) -> dict[str, Any]:
    """``spec`` with a new error that no longer holds ``included``. The stderr of the stated
    error does not describe the new error, so it goes out."""
    kept = {k: v for k, v in spec.items() if k not in (*_ERROR_METRICS, "stderr", "includes")}
    includes = [item for item in spec.get("includes", ()) if item != included]
    return kept | metric | ({"includes": includes} if includes else {})


def _noted(data: dict[str, Any], note: str) -> Profile:
    provenance = data["provenance"]
    provenance["notes"] = [*provenance.get("notes", []), note]
    return Profile.from_dict(data)


def profile_id(machine: str) -> str:
    return f"quantinuum_{machine.lower()}"


def bundled_profile(machine: str, day: date) -> Profile | None:
    """The bundled profile of ``machine`` with the calibration date ``day``, or None.

    The match is the UTC date of ``device.calibrated_at``, as ``nv.load("id@YYYY-MM-DD")``
    matches it. Only bundled profiles count, so the result does not depend on the vault.
    """
    wanted = profile_id(machine)
    for info in catalog.bundled_profiles():
        if info.id == wanted and info.calibrated_at and info.calibrated_at.date() == day:
            return info.load()
    return None


def folder_date(name: str) -> date:
    return datetime.strptime(name, "%Y_%m_%d").replace(tzinfo=UTC).date()


# the command ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Written:
    machine: str
    day: str
    benchmark: str
    paths: tuple[Path, ...]


def datasets(root: Path) -> Iterator[tuple[str, str, Path]]:
    """(machine, folder date, folder) of each dataset below ``root`` or ``root/data``."""
    data = root / "data" if (root / "data").is_dir() else root
    for machine in sorted(p for p in data.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for folder in sorted(p for p in machine.iterdir() if p.is_dir()):
            if re.fullmatch(r"\d{4}_\d{2}_\d{2}", folder.name):
                yield machine.name, folder.name, folder


def convert_folder(
    root: Path, out: Path, *, leakage: Leakage = "measure", log: Callable[[str], None] = print
) -> list[Written]:
    """Write the counts and profile files of every dataset below ``root`` to ``out``."""
    written = []
    for machine, day, folder in datasets(root):
        profile = bundled_profile(machine, folder_date(day))
        if profile is None:
            log(f"skip {machine} {day}: no bundled profile of {profile_id(machine)} has that date")
            continue
        for path in sorted(folder.glob("*.json")):
            if path.name.startswith("._"):
                continue
            benchmark = _FILE_NAMES.get(path.stem, path.stem)
            raw = path.read_bytes()
            file = f"data/{machine}/{day}/{path.name}"
            try:
                files = counts_files(
                    json.loads(raw),
                    benchmark,
                    profile,
                    run_at=datetime.combine(folder_date(day), datetime.min.time(), UTC),
                    file=file,
                    file_sha256="sha256:" + hashlib.sha256(raw).hexdigest(),
                    leakage=leakage,
                )
            except QasmError as exc:
                log(f"skip {file}: {exc}")
                continue
            target = out / machine / day
            target.mkdir(parents=True, exist_ok=True)
            paths = []
            for k, counts in enumerate(files):
                stem = benchmark if len(files) == 1 else f"{benchmark}_part{k + 1}"
                paths.append(counts.save(target / f"{stem}.counts.json"))
                profile.save(target / f"{stem}.profile.json")
            written.append(Written(machine, day, benchmark, tuple(paths)))
            log(f"wrote {len(files)} counts files for {file}")
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", type=Path, help="the clone of the vendor repository")
    parser.add_argument("--out", type=Path, required=True, help="the output folder")
    parser.add_argument(
        "--leakage",
        choices=["measure", "postselect"],
        default="measure",
        help="keep the leakage check in the circuits, or keep only the shots that pass it",
    )
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        print(f"error: {args.root} is not a folder", file=sys.stderr)
        return 2
    written = convert_folder(args.root, args.out, leakage=args.leakage)
    print(f"wrote {sum(len(w.paths) for w in written)} counts files to {args.out}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
