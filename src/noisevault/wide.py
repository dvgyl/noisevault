"""The stabilizer forecast of a Clifford circuit on more qubits than the reference holds.

The reference simulator holds 10 qubits. A Clifford circuit with Pauli noise runs in Stim on
hundreds of qubits. :func:`noisy_circuit` writes a planned circuit as a Stim circuit with the
noise of a profile. For each op, the circuit gets the Clifford instruction of the op and then
the Pauli twirl of each channel that the reference applies for the op: the gate channels of
``resolve_op`` and the wait channel of ``idle_channel``. Thus the forecast and the reference use
one noise model. The only difference is the twirl, which keeps the Pauli part of relaxation and
drops its bias toward 0. :func:`sample` applies the asymmetric readout error of each qubit to
the sampled bits, as ``frameworks.stim.sample_with_readout`` does.

This module needs Stim (the ``stim`` extra).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import pi
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from . import gates
from .channels import pauli_twirl
from .conversion import idle_channel, resolve_op
from .errors import NoiseVaultError, install_hint
from .reference import Op, charged_as
from .report import Report
from .table import refuse_disabled

if TYPE_CHECKING:
    from .counts import PlannedCircuit
    from .profile import Profile

# The Clifford parameter sets of each native gate and the Stim instruction of each. A test
# checks that each registry unitary equals the Stim tableau up to a global phase. ecr is not
# here: its Stim form needs more than one instruction.
CLIFFORD: Mapping[tuple[str, tuple[float, ...]], str] = MappingProxyType(
    {
        ("id", ()): "I",
        ("x", ()): "X",
        ("sx", ()): "SQRT_X",
        ("rz", (pi / 2,)): "S",
        ("r", (pi / 2, 0.0)): "SQRT_X",
        ("r", (pi / 2, pi / 2)): "SQRT_Y",
        ("rx", (pi / 2,)): "SQRT_X",
        ("ry", (pi / 2,)): "SQRT_Y",
        ("cz", ()): "CZ",
        ("zz", ()): "SQRT_ZZ",
        ("rzz", (pi / 2,)): "SQRT_ZZ",
        ("ms", (0.0, 0.0)): "SQRT_XX",
        ("rxx", (pi / 2,)): "SQRT_XX",
        ("iswap", ()): "ISWAP",
        ("cx", ()): "CX",
    }
)
_PAULI_CHANNEL = {1: "PAULI_CHANNEL_1", 2: "PAULI_CHANNEL_2"}
_SAME_ANGLE = 1e-9


def stim_module() -> Any:
    """The Stim module. Raises ImportError with the install command when Stim is missing."""
    try:
        import stim
    except ImportError as exc:
        raise ImportError(f"the stabilizer forecast needs stim: {install_hint('stim')}") from exc
    return stim


def has_stim() -> bool:
    try:
        stim_module()
    except ImportError:
        return False
    return True


def clifford_ops(name: str, qubits: tuple[int, ...]) -> list[Op]:
    """Each Clifford op of gate ``name`` on ``qubits`` that the table gives, in table order."""
    return [Op(name, qubits, params) for (gate, params) in CLIFFORD if gate == name]


def instruction(op: Op) -> str | None:
    """The Stim instruction of ``op``, or None when the table has no such Clifford op."""
    for (name, params), found in CLIFFORD.items():
        if name == op.name and len(params) == len(op.params):
            if all(abs(a - b) < _SAME_ANGLE for a, b in zip(params, op.params, strict=True)):
                return found
    return None


def noisy_circuit(profile: Profile, circuit: PlannedCircuit) -> tuple[Any, np.ndarray]:
    """The Stim circuit of ``circuit`` with the twirled noise of ``profile``, and its readout.

    The Stim circuit measures every circuit qubit at the end, circuit qubit i into record i,
    with no readout error. The array has one row for each circuit qubit: P(1 | 0) and P(0 | 1).
    A gate with no calibration on its qubits raises MissingCalibrationError, as for
    ``simulate``. An op that is not in ``CLIFFORD`` raises NoiseVaultError.
    """
    stim = stim_module()
    table, physical = profile.table, circuit.qubits
    report = Report.start(profile, "reference", None, unknown_gates="error")
    twirls: dict[tuple[Any, ...], tuple[float, ...] | None] = {}
    out = stim.Circuit()
    for op in circuit.ops:
        if op.name == "delay":
            (q,), (duration_ns,) = op.qubits, op.params
            key: tuple[Any, ...] = ("delay", physical[q], duration_ns)
            if key not in twirls:
                refuse_disabled(table.gate("delay", (physical[q],)))
                channel = idle_channel(table, physical[q], duration_ns, report)
                twirls[key] = None if channel is None else pauli_twirl([channel], (physical[q],))
            _noise(out, twirls[key], op.qubits)
            continue
        name = instruction(op)
        if name is None:
            raise NoiseVaultError(
                f"circuit {circuit.name}: {op.name}{_angles(op)} is not a Clifford gate of the"
                " stabilizer forecast",
                hint="use only the gates and angles of noisevault.wide.CLIFFORD",
            )
        out.append(name, list(op.qubits))
        targets = tuple(physical[q] for q in op.qubits)
        key = (charged_as(profile, op.name, gates.unitary(op.name, op.params)), targets)
        if key not in twirls:
            built = resolve_op(table, key[0], targets, unknown_gates="error", report=report)
            twirls[key] = pauli_twirl(built.channels, targets) if built.channels else None
        _noise(out, twirls[key], op.qubits)
    for p in physical:
        refuse_disabled(table.gate("measure", (p,)))
    out.append("M", list(range(len(physical))))
    readout = np.array([table.qubit(p).readout or (0.0, 0.0) for p in physical], dtype=float)
    return out, readout


def sample(profile: Profile, circuit: PlannedCircuit, shots: int, *, seed: int) -> np.ndarray:
    """``shots`` results of ``circuit`` under ``profile``, shape (shots, qubits), circuit qubit
    0 in column 0. The same seed, Stim version and machine give the same bits."""
    noisy, readout = noisy_circuit(profile, circuit)
    rng = np.random.default_rng(seed)
    bits = noisy.compile_sampler(seed=int(rng.integers(2**63))).sample(shots)
    for q, (to_one, to_zero) in enumerate(readout):
        flip = rng.random(shots) < np.where(bits[:, q], to_zero, to_one)
        bits[:, q] ^= flip
    return bits


def ideal_bits(circuit: PlannedCircuit) -> str:
    """The one result of ``circuit`` without noise, circuit qubit 0 first.

    Raises NoiseVaultError when the result is not certain, or for an op outside ``CLIFFORD``.
    """
    stim = stim_module()
    simulator = stim.TableauSimulator()
    simulator.set_num_qubits(len(circuit.qubits))
    for op in circuit.ops:
        if op.name == "delay":
            continue
        name = instruction(op)
        if name is None:
            raise NoiseVaultError(
                f"circuit {circuit.name}: {op.name}{_angles(op)} is not a Clifford gate of the"
                " stabilizer forecast",
                hint="use only the gates and angles of noisevault.wide.CLIFFORD",
            )
        simulator.do(stim.Circuit(f"{name} {' '.join(map(str, op.qubits))}"))
    signs = [simulator.peek_z(q) for q in range(len(circuit.qubits))]
    if 0 in signs:
        raise NoiseVaultError(
            f"circuit {circuit.name} has no certain result: circuit qubit {signs.index(0)} ends"
            " in a superposition",
            hint="give a mirror circuit, whose ideal result is one bit string",
        )
    return "".join("0" if sign > 0 else "1" for sign in signs)


def counts(bits: np.ndarray) -> dict[str, int]:
    """The counts of the rows of ``bits``, as a counts file keys them (circuit qubit 0 left)."""
    rows, times = np.unique(np.asarray(bits, dtype=np.uint8), axis=0, return_counts=True)
    return {"".join(map(str, row)): int(k) for row, k in zip(rows, times, strict=True)}


def _noise(out: Any, probs: Sequence[float] | None, qubits: Sequence[int]) -> None:
    if probs is not None and any(probs):
        out.append(_PAULI_CHANNEL[len(qubits)], list(qubits), list(probs))


def _angles(op: Op) -> str:
    return f"({', '.join(f'{p:g}' for p in op.params)})" if op.params else ""
