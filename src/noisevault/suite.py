"""The deep circuit set for a hardware run.

``plan`` gives the short ``nv check`` circuits. A short circuit has few errors, so most shots
carry no information on the error rates. ``suite`` adds deep mirror circuits: layers of random
native gates, then the same layers undone, so the ideal result is one known bit string. The
depth of each circuit comes from the profile. The profile predicts the probability of the
ideal result, and ``suite`` takes the depths where that probability reaches fixed targets.

The set has three roles, which ``role`` gives for each circuit name:

- ``train``: the readout circuits, deep circuits on the chain, on each pair and on each qubit.
  A fit uses these circuits.
- ``test_near``: deep circuits on the chain from another random seed.
- ``test_far``: the ``nv check`` circuits, a periodic circuit and a circuit with 1-qubit gates
  on all qubits at the same time. No fit sees these circuits.

Each layer starts at the same time on every qubit of the circuit, and an explicit delay holds
every wait. A wait is 0 ns or at least ``MIN_WAIT_NS``, because a device refuses a delay that is
shorter than its time grid allows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from . import check
from .counts import PlannedCircuit, _delay_disabled, _duration, plan
from .errors import LayoutError, NoiseVaultError
from .reference import Op, probabilities

if TYPE_CHECKING:
    from .profile import Profile

Role = Literal["train", "test_near", "test_far"]
Layer = tuple[Op, ...]

SUITE = "1"
MIN_WAIT_NS = 16.0
# Candidate depths in layers. A circuit of depth m holds m layers and then their inverses.
DEPTHS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
# The predicted probability of the ideal result at which each family takes a depth.
CHAIN_TARGETS = (0.9, 0.75, 0.55)
PAIR_TARGETS = (0.85, 0.6)
QUBIT_TARGETS = (0.9,)
TRAIN_SEEDS = (0, 1)
TEST_SEED = 2
_CHAIN, _PAIR, _QUBIT, _FLIP = 1, 2, 3, 4
_TRAIN_PREFIXES = ("readout", "pair", "qubit")
_FLIPPED = 1 - 1e-9


@dataclass(frozen=True)
class _Natives:
    """The gates that a deep circuit on one chain can use."""

    ones: tuple[str, ...]
    pairs: tuple[Op | None, ...]
    flip: tuple[Op, ...] | None


def suite(
    profile: Profile,
    layout: Mapping[Any, int] | Sequence[int] | None = None,
    *,
    seed: int = 0,
) -> tuple[PlannedCircuit, ...]:
    """The circuits of ``plan`` and the deep circuits, on the same chain.

    The same profile, layout and seed give the same circuits. A profile with unmodeled-error
    factors gives the circuits of its calibration.
    """
    base = profile.uncorrected()
    planned = plan(base, layout)
    chain = max((c.qubits for c in planned), key=len)
    natives = _natives(base, chain)
    if not natives.ones:
        return planned
    circuits = list(planned)
    n = len(chain)
    if natives.flip is not None:
        circuits.append(_circuit(base, "readout_1", chain, _flip_layers(natives, [1] * n)))

    middle = None
    if n >= 2 and all(natives.pairs):
        for s in (*TRAIN_SEEDS, TEST_SEED):
            for depth, built in _family(base, chain, natives, CHAIN_TARGETS, (seed, _CHAIN, s)):
                circuits.append(built.model_copy(update={"name": f"deep_d{depth}_s{s}"}))
        depths = _depths(base, chain, natives, CHAIN_TARGETS, (seed, _CHAIN, TRAIN_SEEDS[0]))
        middle = depths[len(depths) // 2]
        for i in range(n - 1):
            sub = chain[i : i + 2]
            found = _natives(base, sub)
            for depth, built in _family(base, sub, found, PAIR_TARGETS, (seed, _PAIR, i)):
                circuits.append(built.model_copy(update={"name": f"pair{i}_d{depth}_s0"}))

    longest = 0
    for i in range(n):
        sub = chain[i : i + 1]
        found = _natives(base, sub)
        for depth, built in _family(base, sub, found, QUBIT_TARGETS, (seed, _QUBIT, i)):
            circuits.append(built.model_copy(update={"name": f"qubit{i}_d{depth}"}))
            longest = max(longest, depth)
    if n >= 2 and longest:
        only_ones = _Natives(natives.ones, (None,) * (n - 1), natives.flip)
        layers = _layers(only_ones, n, longest, (seed, _QUBIT, n))
        circuits.append(_mirror(base, f"qubits_d{longest}", chain, natives, layers, (seed, n)))
    if middle is not None:
        period = _layers(natives, n, 2, (seed, _CHAIN, TEST_SEED + 1))
        layers = period * (middle // 2 or 1)
        circuits.append(_mirror(base, f"periodic_d{middle}", chain, natives, layers, (seed, 0)))
    return tuple(circuits)


def role(name: str) -> Role:
    """The role of the suite circuit ``name``. A name that ``suite`` does not make is a test."""
    if name.startswith("deep_"):
        return "test_near" if name.endswith(f"_s{TEST_SEED}") else "train"
    if name.startswith("qubits_"):
        return "test_far"
    return "train" if name.startswith(_TRAIN_PREFIXES) else "test_far"


def _natives(profile: Profile, chain: Sequence[int]) -> _Natives:
    table = profile.table
    n = len(chain)
    ones = tuple(
        name
        for name in check._unitary_natives(profile, 1)
        if all(check._calibrated(table.gate(name, (p,))) for p in chain)
        and check._inverse(check._op(name, (0,))) is not None
    )
    pairs: list[Op | None] = []
    for i in range(n - 1):
        usable = [
            op for op in check._two_qubit_ops(profile, chain, i) if check._inverse(op) is not None
        ]
        pairs.append(usable[0] if usable else None)
    return _Natives(ones, tuple(pairs), _flip(ones))


def _flip(ones: Sequence[str]) -> tuple[Op, ...] | None:
    """The shortest run of one native gate that takes |0> to |1>, or None."""
    best: tuple[Op, ...] | None = None
    for name in ones:
        op = check._op(name, (0,))
        unitary = check._unitary(op)
        power = np.eye(2, dtype=complex)
        for k in range(1, 5):
            power = unitary @ power
            if abs(power[1, 0]) ** 2 > _FLIPPED and (best is None or k < len(best)):
                best = (op,) * k
                break
    return best


def _layers(natives: _Natives, n: int, depth: int, stream: Sequence[int]) -> list[Layer]:
    """``depth`` layers. A layer is one random 1-qubit gate on each qubit, then 2-qubit gates.

    The 2-qubit gates take the even pairs of the chain in one layer and the odd pairs in the
    next. A chain of two qubits has its one pair in each layer. The first ``m`` layers of a
    stream are the same for each ``depth`` of ``m`` or more.
    """
    rng = np.random.default_rng([*stream])
    layers: list[Layer] = []
    for index in range(depth):
        picks = rng.integers(len(natives.ones), size=n)
        layers.append(tuple(check._op(natives.ones[k], (q,)) for q, k in enumerate(picks)))
        parity = 0 if n == 2 else index % 2
        twos = tuple(op for op in natives.pairs[parity::2] if op is not None)
        if twos:
            layers.append(twos)
    return layers


def _undone(layers: Sequence[Layer]) -> list[Layer]:
    """The layers that undo ``layers``. A gate of order k is undone by k - 1 more of it."""
    undo: list[Layer] = []
    for layer in reversed(layers):
        repeats = [(op, len(check._inverse(op) or ())) for op in layer]
        for step in range(max((times for _, times in repeats), default=0)):
            undo.append(tuple(op for op, times in repeats if times > step))
    return undo


def _flip_layers(natives: _Natives, bits: Sequence[int]) -> list[Layer]:
    if natives.flip is None:
        return []
    return [
        tuple(Op(op.name, (q,), op.params) for q, bit in enumerate(bits) if bit)
        for op in natives.flip
    ]


def _mirror(
    profile: Profile,
    name: str,
    qubits: Sequence[int],
    natives: _Natives,
    layers: Sequence[Layer],
    stream: Sequence[int],
) -> PlannedCircuit:
    """``layers``, their inverses, and flips to a random ideal bit string."""
    bits = np.random.default_rng([*stream, _FLIP]).integers(2, size=len(qubits))
    every = [*layers, *_undone(layers), *_flip_layers(natives, bits)]
    return _circuit(profile, name, qubits, every)


def _circuit(
    profile: Profile, name: str, qubits: Sequence[int], layers: Sequence[Layer]
) -> PlannedCircuit:
    chain = tuple(qubits)
    ops = _synchronous(profile, chain, layers)
    waits = next((q for op in ops if op.name == "delay" for q in op.qubits), None)
    if waits is not None and _delay_disabled(profile, chain[waits]):
        raise LayoutError(
            f"circuit {name!r}: qubit {chain[waits]} waits, but {profile.id} disables delay on"
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


def _ideal(circuit: PlannedCircuit, profile: Profile) -> float:
    """The probability that ``profile`` gives the most likely result of ``circuit``."""
    n = len(circuit.qubits)
    return float(probabilities(profile, circuit.ops, n, layout=circuit.qubits).max())


def _depths(
    profile: Profile,
    qubits: Sequence[int],
    natives: _Natives,
    targets: Sequence[float],
    stream: Sequence[int],
) -> list[int]:
    """For each target, the first depth at which the predicted ideal result is that rare."""
    found: list[int] = []
    remaining = sorted(targets, reverse=True)
    for depth in DEPTHS:
        layers = _layers(natives, len(qubits), depth, stream)
        predicted = _ideal(_mirror(profile, "deep", qubits, natives, layers, stream), profile)
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
    if not natives.ones:
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


def require_deep(circuits: Sequence[PlannedCircuit], profile: Profile) -> None:
    """Raise NoiseVaultError when ``circuits`` hold no deep circuit."""
    if not any(c.name.startswith("deep_") or c.name.startswith("qubit") for c in circuits):
        raise NoiseVaultError(
            f"{profile.id} has no native gate that a deep circuit can undo on this chain",
            hint="use the circuits of plan(), or pass layout= with other qubits",
        )
