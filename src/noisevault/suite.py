"""The deep circuit set for a hardware run.

``plan`` gives the short ``nv check`` circuits. A short circuit has few errors, so most shots
carry no information on the error rates. ``suite`` adds deep mirror circuits. The depth of each
circuit comes from the profile: ``suite`` takes the depths at which the predicted probability of
the ideal result has fallen to fixed parts of the way from 1 to the probability of a random bit
string (see ``CHAIN_TARGETS``).

A mirror circuit has three parts: layers of random gates, one random Pauli gate on each qubit,
and the layers undone in the opposite order. Each gate is a Clifford gate, so the ideal result is
one bit string, and the Pauli gates in the middle make this bit string random.

A noise model from a calibration holds errors that have no direction. A real gate can also have
a small systematic error, for example a rotation that is a little too large. A circuit design
decides what such an error does:

- When each gate turns about one axis, the errors of all gates add up. The error probability
  then grows with the square of the number of gates.
- When the second half is the exact mirror of the first half and nothing is between them, the
  error of each gate cancels the error of its partner.

A circuit of the usual kind does neither. Three rules give a mirror circuit this property:

- On a device whose 1-qubit gates all turn about one axis (``sx`` and ``x``, for example), a
  random virtual Z rotation comes before each gate. The gates then turn about changing axes.
- The inverse of a gate is the gate itself in another frame where the device permits (``sx`` is
  undone by ``rz(pi) sx rz(pi)``), and not three more of the same gate.
- The random Pauli gates in the middle give the errors of the two halves random signs with
  respect to each other.

This is the design of randomized mirror circuits (Proctor et al., Nature Physics 18, 75, 2022).

Two kinds of circuits are the opposite case on purpose. An ``axis_*`` circuit applies one gate
many times on each qubit, and it shows the size of the systematic rotation error of that gate.
The ``periodic_*`` circuit is a mirror circuit that repeats two layers, with the same frames in
each repeat, so a systematic error of a 1-qubit gate or of a 2-qubit gate adds up in it. No
model sees these circuits in a fit, and no comparison of models uses them.

The three rules need a device that permits them. A device whose 1-qubit gates turn about one
axis and that has no virtual Z rotation gets no frames, and the Pauli gates in the middle are
then only the ones that its gates can make. On such a device, a systematic rotation error still
adds up in each mirror circuit. No bundled device is of this kind.

The set has four roles, which ``role`` gives for each circuit name:

- ``train``: the readout circuits, mirror circuits on the chain, on each pair and on each qubit.
  A fit uses these circuits.
- ``test_near``: mirror circuits on the chain from other random seeds.
- ``test_far``: the ``nv check`` circuits, a circuit with 1-qubit gates on all qubits at the
  same time, mirror circuits on a chain of ``WIDE_QUBITS`` qubits, and one mirror circuit on a
  chain of each size in ``CLIFFORD_QUBITS``. No fit sees these circuits.
- ``probe``: the ``axis_*`` circuits and the ``periodic_*`` circuit.

A family takes one depth for each target. When no depth of ``DEPTHS`` reaches a target, the
family takes the largest depth. When one depth reaches two targets, the family has one circuit
for both, so a device with very large errors gets fewer circuits.

A circuit on a chain of ``CLIFFORD_QUBITS`` has more qubits than the reference simulator holds,
so the stabilizer forecast of ``noisevault.wide`` gives its depth. On many qubits, readout error
alone makes the ideal result rare, so the depth comes from the mean probability of a wrong bit,
not from the ideal result (see ``CLIFFORD_WRONG``). Such a circuit uses only the gates and angles
of ``noisevault.wide.CLIFFORD``. Without Stim, ``suite`` makes no such circuit.

Each layer starts at the same time on every qubit of the circuit, and an explicit delay holds
every wait. A wait is 0 ns or at least ``MIN_WAIT_NS``, because a device refuses a delay that is
shorter than its time grid allows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import reduce
from itertools import product
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from . import check, wide
from .counts import PlannedCircuit, _delay_disabled, _duration, plan
from .errors import LayoutError, NoiseVaultError
from .layout import suggest_layout
from .reference import Op, probabilities
from .table import GateNoise

if TYPE_CHECKING:
    from .profile import Profile

Role = Literal["train", "test_near", "test_far", "probe"]
Layer = tuple[Op, ...]
Key = tuple[int, ...]

# The version of the circuit design. Version 1 undid a gate by more of the same gate and had
# no virtual rotations and no Pauli gates in the middle.
SUITE_VERSION = 2
MIN_WAIT_NS = 16.0
# Candidate depths in layers. A circuit of depth m holds m layers and then their inverses.
DEPTHS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
# The depths come from the stated errors. Deep circuits on IBM Heron devices gave their ideal
# result about as often as the stated errors predict in four of six runs, and about as often as
# two times the stated errors predict in one run. The targets below suit this range.
ASSUMED_GATE_FACTOR = 1.0
# The targets at which each family takes a depth. A target is a value of the polarization: the
# predicted probability p of the ideal result on n qubits as (p - 1/2^n) / (1 - 1/2^n). The
# polarization is 1 for a certain result and 0 for random bits, so one target means the same on
# 1 qubit and on 8 qubits. A circuit tells the most about the size of the gate errors at a
# polarization from 0.2 to 0.45. The lowest target is 0.4, so that a device with two times the
# stated errors is still in this range.
CHAIN_TARGETS = (0.8, 0.6, 0.4)
PAIR_TARGETS = (0.75, 0.4)
QUBIT_TARGETS = (0.7,)
# The wide circuits test a model on twice the qubits of the chain, most of them other qubits.
WIDE_QUBITS = 8
WIDE_TARGETS = (0.6, 0.35)
# These circuits test a model on more qubits than the reference simulator holds.
CLIFFORD_QUBITS = (20, 40, 80)
# The depth of such a circuit is the first whole depth at which the profile with
# ASSUMED_GATE_FACTOR predicts a wrong bit with this mean probability over the qubits, or more.
CLIFFORD_WRONG = 0.15
# The depth search samples the stabilizer forecast. This many samples give the mean probability
# of a wrong bit to better than 0.003.
CLIFFORD_SAMPLES = 20_000
TRAIN_SEEDS = (0, 1)
TEST_SEED = 2
# The number of gates on each qubit of each ``axis_*`` circuit. A systematic rotation error
# gives 16 times more wrong bits at 256 gates than at 64 gates, and an error with no direction
# gives 4 times more.
AXIS_REPEATS = (64, 256)
# The exact ideal result of a circuit comes from a state vector up to this number of qubits.
_STATE_QUBITS = 10
_CHAIN, _PAIR, _QUBIT, _MIDDLE, _WIDE, _CLIFFORD, _FRESH = 1, 2, 3, 4, 5, 6, 7
_TRAIN_PREFIXES = ("readout", "pair", "qubit")
_ONLY_IN_SUITE = (
    "readout_1",
    "deep_",
    "pair",
    "qubit",
    "periodic_",
    "wide_",
    "clifford_",
    "axis_",
)
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_TOLERANCE = 1e-9


@dataclass(frozen=True)
class _Natives:
    """The gates that a mirror circuit on one chain can use.

    Each op in ``turns`` and ``frames`` is on qubit 0. A turn is a 1-qubit gate that the device
    applies with a pulse. A frame is a virtual Z rotation, which takes no time and has no error.
    ``runs`` gives, for each 1-qubit Clifford gate that the turns and frames can make, the
    shortest run of them in the order of time.
    """

    turns: tuple[Op, ...]
    frames: tuple[Op, ...]
    pairs: tuple[Op | None, ...]
    runs: Mapping[Key, tuple[Op, ...]]

    def virtual(self, op: Op) -> bool:
        return any(op.name == frame.name for frame in self.frames)


def suite(
    profile: Profile,
    layout: Mapping[Any, int] | Sequence[int] | None = None,
    *,
    seed: int = 0,
    fresh: int | None = None,
) -> tuple[PlannedCircuit, ...]:
    """The circuits of ``plan`` and the deep circuits, on the same chain.

    The same profile, layout, seed and ``fresh`` give the same circuits. A profile with
    unmodeled-error factors gives the circuits of its calibration. With ``fresh``, the set also
    has ``test_near`` circuits ``deep_d{m}_f{fresh}`` from a random stream that only this number
    gives. A campaign that passes a new number for each run tests new random circuits each time.
    """
    base = profile.uncorrected()
    assumed = base.model_copy(
        update={"unmodeled_error": {"gates": {"factor": ASSUMED_GATE_FACTOR}}}
    )
    planned = plan(base, layout)
    chain = max((c.qubits for c in planned), key=len)
    natives = _natives(base, chain)
    if not natives.turns:
        return planned
    circuits = list(planned)
    n = len(chain)
    flip = natives.runs.get(_key(_X))
    if flip is not None:
        ones = _columns({q: [_on(op, q) for op in flip] for q in range(n)}, natives)
        circuits.append(_circuit(base, "readout_1", chain, ones))

    middle = None
    if n >= 2 and all(natives.pairs):
        streams = {f"s{s}": (seed, _CHAIN, s) for s in (*TRAIN_SEEDS, TEST_SEED)}
        if fresh is not None:
            streams[f"f{fresh}"] = (seed, _CHAIN, _FRESH, fresh)
        for tag, stream in streams.items():
            for depth, built in _family(assumed, chain, natives, CHAIN_TARGETS, stream):
                circuits.append(built.model_copy(update={"name": f"deep_d{depth}_{tag}"}))
        depths = _depths(assumed, chain, natives, CHAIN_TARGETS, (seed, _CHAIN, TRAIN_SEEDS[0]))
        middle = depths[len(depths) // 2]
        for i in range(n - 1):
            sub = chain[i : i + 2]
            found = _natives(base, sub)
            for depth, built in _family(assumed, sub, found, PAIR_TARGETS, (seed, _PAIR, i)):
                circuits.append(built.model_copy(update={"name": f"pair{i}_d{depth}_s0"}))

    longest = 0
    for i in range(n):
        sub = chain[i : i + 1]
        found = _natives(base, sub)
        for depth, built in _family(assumed, sub, found, QUBIT_TARGETS, (seed, _QUBIT, i)):
            circuits.append(built.model_copy(update={"name": f"qubit{i}_d{depth}"}))
            longest = max(longest, depth)
    if n >= 2 and longest:
        only_ones = replace(natives, pairs=(None,) * (n - 1))
        layers = _layers(only_ones, n, longest, (seed, _QUBIT, n))
        circuits.append(_mirror(base, f"qubits_d{longest}", chain, only_ones, layers, (seed, n)))
    if middle is not None:
        first, second = _cycles(natives, n, 2, (seed, _CHAIN, TEST_SEED + 1))
        layers = [layer for i in range(middle) for layer in (second if i % 2 else first)]
        circuits.append(_mirror(base, f"periodic_d{middle}", chain, natives, layers, (seed, 0)))
    circuits += _axis_circuits(base, chain, natives)
    eight = _wide_chain(base) if middle is not None else None
    if eight is not None:
        found = _natives(base, eight)
        if found.turns and all(found.pairs):
            for depth, built in _family(assumed, eight, found, WIDE_TARGETS, (seed, _WIDE, 0)):
                circuits.append(built.model_copy(update={"name": f"wide_d{depth}_s0"}))
    if wide.has_stim():
        circuits += _clifford_circuits(base, assumed, seed)
    return tuple(circuits)


def role(name: str) -> Role:
    """The role of the suite circuit ``name``. A name that ``suite`` does not make is a test.

    A ``probe`` circuit is not in a fit and not in a comparison of models. Its result is given
    alone.
    """
    if name.startswith(("axis_", "periodic_")):
        return "probe"
    if name.startswith("deep_"):
        train = name.endswith(tuple(f"_s{s}" for s in TRAIN_SEEDS))
        return "train" if train else "test_near"
    if name.startswith(("qubits_", "wide_", "clifford_")):
        return "test_far"
    return "train" if name.startswith(_TRAIN_PREFIXES) else "test_far"


def train_names(names: Sequence[str]) -> list[str] | None:
    """The names of the ``train`` circuits of a run of ``suite``, in their order.

    Returns None when ``names`` hold no circuit that only ``suite`` makes. Such a run holds the
    circuits of ``plan``, and a fit uses all of them.
    """
    if not any(name.startswith(_ONLY_IN_SUITE) for name in names):
        return None
    return [name for name in names if role(name) == "train"]


def ideal_result(circuit: PlannedCircuit) -> str:
    """The one result of ``circuit`` without noise, circuit qubit 0 first.

    A circuit on more than 10 qubits needs Stim. Raises NoiseVaultError when the result is not
    certain.
    """
    n = len(circuit.qubits)
    if n > _STATE_QUBITS:
        return wide.ideal_bits(circuit)
    state = np.zeros((2,) * n, dtype=complex)
    state[(0,) * n] = 1.0
    tensors: dict[tuple[str, tuple[float, ...]], np.ndarray] = {}
    for op in circuit.ops:
        if op.name == "delay":
            continue
        k = len(op.qubits)
        key = (op.name, tuple(op.params))
        if key not in tensors:
            tensors[key] = check._unitary(op).reshape((2,) * (2 * k))
        moved = np.tensordot(tensors[key], state, axes=(list(range(k, 2 * k)), list(op.qubits)))
        state = np.moveaxis(moved, list(range(k)), list(op.qubits))
    weights = np.abs(state.reshape(-1)) ** 2
    index = int(weights.argmax())
    if weights[index] < 1 - 1e-6:
        raise NoiseVaultError(
            f"circuit {circuit.name} has no certain result: its most probable result has the"
            f" probability {weights[index]:.3g}",
            hint="give a mirror circuit, whose ideal result is one bit string",
        )
    return format(index, f"0{n}b")


def _wide_chain(profile: Profile) -> tuple[int, ...] | None:
    """The suggested chain of ``WIDE_QUBITS`` qubits, or None when the device has none."""
    if profile.device.num_qubits < WIDE_QUBITS:
        return None
    try:
        found = suggest_layout(
            profile, WIDE_QUBITS, usable_pair=lambda a, b: check._usable_pair(profile, a, b)
        )
    except LayoutError:
        return None
    return tuple(found.values())


def _clifford_circuits(base: Profile, assumed: Profile, seed: int) -> list[PlannedCircuit]:
    """One mirror circuit on a chain of each size in ``CLIFFORD_QUBITS`` that fits.

    A size is left out when the device has no such chain with a gate of
    ``noisevault.wide.CLIFFORD`` on each pair, when a qubit of the chain has no such 1-qubit
    gate, or when a qubit of the chain must wait but disables delay.
    """
    built: list[PlannedCircuit] = []
    for n in CLIFFORD_QUBITS:
        if base.device.num_qubits < n:
            continue
        try:
            found = suggest_layout(base, n, usable_pair=lambda a, b: _clifford_pair(base, a, b))
            chain = tuple(found.values())
            natives = _natives(base, chain, stim_only=True)
            if not natives.turns or not all(natives.pairs):
                continue
            stream = (seed, _CLIFFORD, n)
            depth = _clifford_depth(assumed, chain, natives, stream)
            layers = _layers(natives, n, depth, stream)
            circuit = _mirror(base, f"clifford_n{n}_d{depth}", chain, natives, layers, stream)
        except LayoutError:
            continue
        built.append(circuit)
    return built


def _clifford_depth(
    profile: Profile, qubits: Sequence[int], natives: _Natives, stream: Sequence[int]
) -> int:
    """The first whole depth from 1 to the last of ``DEPTHS`` at which ``profile`` predicts a
    wrong bit with a mean probability of ``CLIFFORD_WRONG`` or more, or the last of ``DEPTHS``.

    The search tries the depths of ``DEPTHS`` and then each whole depth in the step where the
    prediction first reaches the target. This is the first whole depth when the prediction grows
    with the depth.
    """

    def reaches(depth: int) -> bool:
        layers = _layers(natives, len(qubits), depth, stream)
        built = _mirror(profile, "deep", qubits, natives, layers, stream)
        bits = np.array([bit == "1" for bit in ideal_result(built)])
        sampled = wide.sample(profile, built, CLIFFORD_SAMPLES, seed=0)
        return float(np.mean(sampled != bits)) >= CLIFFORD_WRONG

    previous = 0
    for depth in DEPTHS:
        if reaches(depth):
            return next((d for d in range(previous + 1, depth) if reaches(d)), depth)
        previous = depth
    return DEPTHS[-1]


def _clifford_pair(profile: Profile, a: int, b: int) -> bool:
    return any(
        wide.clifford_ops(op.name, op.qubits) for op in check._two_qubit_ops(profile, (a, b), 0)
    )


def _natives(profile: Profile, chain: Sequence[int], *, stim_only: bool = False) -> _Natives:
    """The gates of the mirror circuits on ``chain``.

    The turns are the calibrated 1-qubit natives with no parameter, without the identity. A
    device with none of these gives the Clifford angles of its natives with parameters. Frames
    are in use only when all turns are rotations about one axis. With ``stim_only``, a 2-qubit
    gate must be in ``noisevault.wide.CLIFFORD``. Without it, each 2-qubit Clifford native is
    usable.
    """
    table = profile.table
    fixed: list[Op] = []
    angled: list[Op] = []
    for name in check._unitary_natives(profile, 1):
        if not all(check._calibrated(table.gate(name, (p,))) for p in chain):
            continue
        for op in wide.clifford_ops(name, (0,)):
            if _key(check._unitary(op)) != _key(_I):
                (angled if op.params else fixed).append(op)
    turns = _distinct(fixed or angled)
    frames: tuple[Op, ...] = ()
    if turns and _one_axis(turns) and _virtual_z(profile, chain):
        frames = tuple(
            op for op in wide.clifford_ops("rz", (0,)) if _key(check._unitary(op)) != _key(_I)
        )
    partial = _Natives(turns, frames, (), _runs(turns, frames))
    pairs: list[Op | None] = []
    for i in range(len(chain) - 1):
        usable = None
        for op in check._two_qubit_ops(profile, chain, i):
            options = wide.clifford_ops(op.name, op.qubits)
            if not options and not stim_only and _is_clifford(check._unitary(op)):
                options = [op]
            usable = next((o for o in options if _undo_two(partial, o) is not None), None)
            if usable is not None:
                break
        pairs.append(usable)
    return replace(partial, pairs=tuple(pairs))


def _virtual_z(profile: Profile, chain: Sequence[int]) -> bool:
    """True when ``rz`` is a virtual gate on each qubit of ``chain``."""
    found = [profile.table.gate("rz", (p,)) for p in chain]
    return all(isinstance(gate, GateNoise) and gate.state == "ideal" for gate in found)


def _distinct(ops: Sequence[Op]) -> tuple[Op, ...]:
    """``ops`` without an op that does the same as an earlier op."""
    kept: dict[Key, Op] = {}
    for op in ops:
        key = _key(check._unitary(op))
        if key is not None:
            kept.setdefault(key, op)
    return tuple(kept.values())


def _one_axis(ops: Sequence[Op]) -> bool:
    """True when all ``ops`` are rotations about one axis, so that each pair commutes."""
    unitaries = [check._unitary(op) for op in ops]
    return all(np.allclose(a @ b, b @ a, atol=_TOLERANCE) for a in unitaries for b in unitaries)


def _key(unitary: np.ndarray) -> Key | None:
    """How a 1-qubit Clifford gate moves X, Y and Z, or None when the gate is not a Clifford.

    Two gates that differ only by a global phase have the same key.
    """
    adjoint = unitary.conj().T
    paulis = (_X, _Y, _Z)
    moved = np.array(
        [[0.5 * np.trace(p @ unitary @ q @ adjoint).real for q in paulis] for p in paulis]
    )
    whole = np.rint(moved)
    if not np.allclose(moved, whole, atol=1e-6) or (np.abs(whole).sum(axis=1) != 1).any():
        return None
    return tuple(int(value) for value in whole.reshape(-1))


def _is_clifford(unitary: np.ndarray) -> bool:
    """True when the gate moves each Pauli operator to a Pauli operator."""
    n = len(unitary).bit_length() - 1
    products = [reduce(np.kron, combo) for combo in product((_I, _X, _Y, _Z), repeat=n)]
    adjoint = unitary.conj().T
    for position in range(n):
        for pauli in (_X, _Z):
            factors = [pauli if q == position else _I for q in range(n)]
            image = unitary @ reduce(np.kron, factors) @ adjoint
            overlaps = [abs(np.trace(p @ image)) / len(unitary) for p in products]
            if abs(max(overlaps) - 1) > 1e-6:
                return False
    return True


def _runs(turns: Sequence[Op], frames: Sequence[Op]) -> dict[Key, tuple[Op, ...]]:
    """For each 1-qubit Clifford gate that products of the ops give: the run with the fewest
    turns, and then with the fewest ops. The ops of a run are in the order of time."""
    steps = [(op, check._unitary(op), 1) for op in turns]
    steps += [(op, check._unitary(op), 0) for op in frames]
    start = _key(_I)
    assert start is not None
    found: dict[Key, tuple[tuple[Op, ...], np.ndarray, tuple[int, int]]] = {start: ((), _I, (0, 0))}
    changed = True
    while changed:
        changed = False
        for run, unitary, (cost, length) in list(found.values()):
            for op, step, pulses in steps:
                after = step @ unitary
                key = _key(after)
                rank = (cost + pulses, length + 1)
                if key is not None and (key not in found or rank < found[key][2]):
                    found[key] = ((*run, op), after, rank)
                    changed = True
    return {key: run for key, (run, _, _) in found.items()}


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    """True when two unitaries differ only by a global phase."""
    overlap = np.trace(a.conj().T @ b) / len(a)
    return bool(abs(abs(overlap) - 1) < 1e-6)


def _local(unitary: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """The two 1-qubit gates of a 2-qubit gate that acts on each qubit alone, or None."""
    blocks = unitary.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
    left, sizes, right = np.linalg.svd(blocks)
    if sizes[1] > 1e-6:
        return None
    scale = np.sqrt(sizes[0])
    return (left[:, 0] * scale).reshape(2, 2), (right[0] * scale).reshape(2, 2)


def _undo_two(natives: _Natives, op: Op) -> list[Layer] | None:
    """The layers that undo a 2-qubit gate, or None.

    A gate that is its own inverse is undone by itself. When two of the gate are 1-qubit gates
    (``iswap`` two times is Z on each qubit), the gate and the inverses of these 1-qubit gates
    undo it. Each other gate is undone by more of the same gate.
    """
    unitary = check._unitary(op)
    square = unitary @ unitary
    if _same(square, np.eye(4)):
        return [(op,)]
    parts = _local(square)
    if parts is not None:
        keys = [_key(part.conj().T) for part in parts]
        runs = [None if key is None else natives.runs.get(key) for key in keys]
        if all(run is not None for run in runs):
            fix = {
                q: [_on(g, q) for g in run or ()] for q, run in zip(op.qubits, runs, strict=True)
            }
            return [(op,), *_columns(fix, natives)]
    again = check._inverse(op)
    return None if again is None else [(op,)] * len(again)


def _columns(runs: Mapping[int, Sequence[Op]], natives: _Natives) -> list[Layer]:
    """The layers of one run of ops for each qubit.

    A layer holds only frames or only gates with a pulse, so that the gates with a pulse of all
    qubits start at the same time.
    """
    pending = {q: list(run) for q, run in sorted(runs.items())}
    layers: list[Layer] = []
    while any(pending.values()):
        first = [q for q, run in pending.items() if run and natives.virtual(run[0])]
        if not first:
            first = [q for q, run in pending.items() if run]
        layers.append(tuple(pending[q].pop(0) for q in first))
    return layers


def _cycles(natives: _Natives, n: int, depth: int, stream: Sequence[int]) -> list[list[Layer]]:
    """``depth`` cycles. A cycle is one random 1-qubit gate on each qubit, then 2-qubit gates.

    With frames, a random frame or no frame comes before the gate of each qubit. The 2-qubit
    gates take the even pairs of the chain in one cycle and the odd pairs in the next. A chain of
    two qubits has its one pair in each cycle. The first ``m`` cycles of a stream are the same
    for each ``depth`` of ``m`` or more.
    """
    rng = np.random.default_rng([*stream])
    cycles: list[list[Layer]] = []
    for index in range(depth):
        layers: list[Layer] = []
        if natives.frames:
            picks = rng.integers(len(natives.frames) + 1, size=n)
            frame = tuple(_on(natives.frames[k - 1], q) for q, k in enumerate(picks) if k)
            if frame:
                layers.append(frame)
        picks = rng.integers(len(natives.turns), size=n)
        layers.append(tuple(_on(natives.turns[k], q) for q, k in enumerate(picks)))
        parity = 0 if n == 2 else index % 2
        twos = tuple(op for op in natives.pairs[parity::2] if op is not None)
        if twos:
            layers.append(twos)
        cycles.append(layers)
    return cycles


def _layers(natives: _Natives, n: int, depth: int, stream: Sequence[int]) -> list[Layer]:
    """The layers of ``depth`` cycles of ``_cycles``, in their order."""
    return [layer for cycle in _cycles(natives, n, depth, stream) for layer in cycle]


def _on(op: Op, qubit: int) -> Op:
    return Op(op.name, (qubit,), op.params)


def _undone(natives: _Natives, layers: Sequence[Layer]) -> list[Layer]:
    """The layers that undo ``layers``, last layer first."""
    undo: list[Layer] = []
    for layer in reversed(layers):
        if any(len(op.qubits) == 2 for op in layer):
            parts = [_undo_two(natives, op) or [] for op in layer]
            for step in range(max(len(part) for part in parts)):
                undo.append(tuple(op for part in parts if step < len(part) for op in part[step]))
            continue
        runs = {}
        for op in layer:
            key = _key(check._unitary(op).conj().T)
            run = None if key is None else natives.runs.get(key)
            if run is None:
                raise LayoutError(f"the 1-qubit natives cannot undo {op.name}")
            runs[op.qubits[0]] = [_on(g, op.qubits[0]) for g in run]
        undo += _columns(runs, natives)
    return undo


def _middle(natives: _Natives, n: int, stream: Sequence[int]) -> list[Layer]:
    """One random Pauli gate, or no gate, on each qubit. Only the Pauli gates that the natives
    can make are in the choice."""
    paulis = [run for run in (natives.runs.get(_key(p)) for p in (_X, _Y, _Z)) if run]
    if not paulis:
        return []
    picks = np.random.default_rng([*stream, _MIDDLE]).integers(len(paulis) + 1, size=n)
    runs = {q: [_on(g, q) for g in paulis[k - 1]] for q, k in enumerate(picks) if k}
    return _columns(runs, natives)


def _mirror(
    profile: Profile,
    name: str,
    qubits: Sequence[int],
    natives: _Natives,
    layers: Sequence[Layer],
    stream: Sequence[int],
) -> PlannedCircuit:
    """``layers``, random Pauli gates, and the inverses of ``layers``."""
    every = [*layers, *_middle(natives, len(qubits), stream), *_undone(natives, layers)]
    return _circuit(profile, name, qubits, every)


def _axis_circuits(
    profile: Profile, chain: Sequence[int], natives: _Natives
) -> list[PlannedCircuit]:
    """For each number in ``AXIS_REPEATS``: one turn that many times on each qubit at the same
    time. The turn is the one that needs the most repeats for a full rotation, and a number of
    repeats that is not a whole number of full rotations is left out. The ideal result is all 0.
    """
    orders = {}
    for op in natives.turns:
        again = check._inverse(op)
        if again is not None:
            orders[op] = len(again) + 1
    if not orders:
        return []
    turn = max(orders, key=lambda op: orders[op])
    layer = tuple(_on(turn, q) for q in range(len(chain)))
    return [
        _circuit(profile, f"axis_d{repeats}", chain, [layer] * repeats)
        for repeats in AXIS_REPEATS
        if repeats % orders[turn] == 0
    ]


def _circuit(
    profile: Profile, name: str, qubits: Sequence[int], layers: Sequence[Layer]
) -> PlannedCircuit:
    chain = tuple(qubits)
    ops = _synchronous(profile, chain, layers)
    for q in sorted({q for op in ops if op.name == "delay" for q in op.qubits}):
        if _delay_disabled(profile, chain[q]):
            raise LayoutError(
                f"circuit {name!r}: qubit {chain[q]} waits, but {profile.id} disables delay on"
                " that qubit",
                hint="pass layout= with qubits that allow delay",
            )
    return PlannedCircuit(name=name, qubits=chain, ops=tuple(ops))


def _synchronous(profile: Profile, qubits: tuple[int, ...], layers: Sequence[Layer]) -> list[Op]:
    """The ops of ``layers`` with delays, so that each layer starts at one time on all qubits."""
    ops: list[Op] = []
    for layer in layers:
        busy = [0.0] * len(qubits)
        for op in layer:
            for q in op.qubits:
                busy[q] = float(_duration(profile, op, qubits))
        length = max(busy)
        if any(0 < length - t < MIN_WAIT_NS for t in busy):
            length += MIN_WAIT_NS
        ops += layer
        ops += [Op("delay", (q,), (length - t,)) for q, t in enumerate(busy) if length > t]
    return ops


def _polarization(circuit: PlannedCircuit, profile: Profile) -> float:
    """The polarization that ``profile`` predicts for ``circuit``: 1 when the ideal result is
    certain, and 0 when it is as probable as each other bit string."""
    n = len(circuit.qubits)
    index = int(ideal_result(circuit), 2)
    found = float(probabilities(profile, circuit.ops, n, layout=circuit.qubits)[index])
    random = 0.5**n
    return (found - random) / (1 - random)


def _depths(
    profile: Profile,
    qubits: Sequence[int],
    natives: _Natives,
    targets: Sequence[float],
    stream: Sequence[int],
) -> list[int]:
    """For each target, the first depth at which the predicted polarization is that low."""
    found: list[int] = []
    remaining = sorted(targets, reverse=True)
    for depth in DEPTHS:
        layers = _layers(natives, len(qubits), depth, stream)
        built = _mirror(profile, "deep", qubits, natives, layers, stream)
        predicted = _polarization(built, profile)
        while remaining and predicted <= remaining[0]:
            remaining.pop(0)
            if depth not in found:
                found.append(depth)
        if not remaining:
            break
    if remaining and DEPTHS[-1] not in found:
        found.append(DEPTHS[-1])
    return found


def _family(
    profile: Profile,
    qubits: Sequence[int],
    natives: _Natives,
    targets: Sequence[float],
    stream: Sequence[int],
) -> list[tuple[int, PlannedCircuit]]:
    if not natives.turns:
        return []
    built = []
    for depth in _depths(profile, qubits, natives, targets, stream):
        layers = _layers(natives, len(qubits), depth, stream)
        built.append((depth, _mirror(profile, "deep", qubits, natives, layers, stream)))
    return built


def describe(circuits: Sequence[PlannedCircuit]) -> list[dict[str, Any]]:
    """One row for each circuit: its name, role, qubits and gate counts by arity."""
    rows = []
    for circuit in circuits:
        gates = [op for op in circuit.ops if op.name != "delay"]
        rows.append(
            {
                "name": circuit.name,
                "role": role(circuit.name),
                "qubits": list(circuit.qubits),
                "one_qubit_gates": sum(len(op.qubits) == 1 for op in gates),
                "two_qubit_gates": sum(len(op.qubits) == 2 for op in gates),
            }
        )
    return rows
