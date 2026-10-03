"""Circuit-to-device qubit mappings: validation shared by every framework, and a chain helper."""

from __future__ import annotations

import operator
import warnings
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from . import gates
from .errors import LayoutError, NoiseVaultWarning
from .report import warn_from_caller
from .table import GateNoise

if TYPE_CHECKING:
    from .profile import Profile
    from .table import NoiseTable

_BEAMS = (32, 128, 512)


@dataclass(frozen=True, order=True)
class _Cost:
    """A chain's score, in order: qubits missing a 1-qubit native of the device, qubits with no
    usable 1-qubit gate, qubits with unknown readout, summed error."""

    incomplete: int
    no_gate: int
    no_readout: int
    error: float

    def __add__(self, other: _Cost) -> _Cost:
        return _Cost(
            self.incomplete + other.incomplete,
            self.no_gate + other.no_gate,
            self.no_readout + other.no_readout,
            self.error + other.error,
        )


def _int_label(label: Hashable) -> int | None:
    if isinstance(label, bool):
        return None
    try:
        return operator.index(label)  # type: ignore[arg-type]
    except TypeError:
        return None


def normalize_layout(
    labels: Iterable[Hashable],
    layout: Mapping[Hashable, int] | Sequence[int] | None,
    profile: Profile,
    *,
    index_of: Callable[[Hashable], int | None] = _int_label,
    width: int | None = None,
) -> dict[Hashable, int]:
    """Map every circuit qubit label to a usable physical qubit, or raise LayoutError.

    With no ``layout``, labels that ``index_of`` turns into integers map to themselves
    (integers by default). An export can pass its own rule, for example for Cirq ``LineQubit``.
    A sequence layout maps label ``i`` to ``layout[i]``. The result must be complete and
    injective, and use in-range, enabled qubits. ``width`` is the number of qubits in the
    circuit, by default ``len(labels)``. A caller that does not know the number passes 0. The
    hint for a disabled qubit names ``suggest_layout`` only when the search finds a chain of
    ``width`` qubits.
    """
    labels = list(dict.fromkeys(labels))
    table = profile.table
    if layout is None:
        mapping: dict[Hashable, int] = {}
        for label in labels:
            index = index_of(label)
            if index is None:
                raise LayoutError(
                    f"qubit {label!r} has no integer index",
                    hint=f"pass layout={{{label!r}: <physical qubit>, ...}} covering every"
                    " circuit qubit",
                )
            mapping[label] = index
    elif isinstance(layout, Mapping):
        mapping = dict(layout)
    else:
        mapping = dict(enumerate(layout))

    missing = [label for label in labels if label not in mapping]
    if missing:
        raise LayoutError(
            f"layout has no physical qubit for {missing!r}", hint="map every circuit qubit"
        )
    result: dict[Hashable, int] = {}
    owner: dict[int, Hashable] = {}
    for label in labels:
        physical = _int_label(mapping[label])
        if physical is None:
            raise LayoutError(f"layout maps {label!r} to {mapping[label]!r}, not a qubit index")
        if not 0 <= physical < table.num_qubits:
            raise LayoutError(
                f"layout maps {label!r} to qubit {physical}, but {profile.id} has qubits"
                f" 0..{table.num_qubits - 1}"
            )
        if table.qubit(physical).disabled:
            n = len(labels) if width is None else width
            raise LayoutError(
                f"layout maps {label!r} to qubit {physical}, which {profile.id} marks disabled",
                hint=f"choose another qubit (profile.suggest_layout({n}) proposes a usable chain)"
                if has_chain(profile, n)
                else None,
            )
        if physical in owner:
            raise LayoutError(
                f"layout maps both {owner[physical]!r} and {label!r} to qubit {physical}"
            )
        owner[physical] = label
        result[label] = physical
    return result


def has_chain(profile: Profile, n: int) -> bool:
    """Whether ``suggest_layout(profile, n)`` finds a chain."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoiseVaultWarning)
        try:
            suggest_layout(profile, n)
        except LayoutError:
            return False
    return True


def suggest_layout(
    profile: Profile, n: int, *, usable_pair: Callable[[int, int], bool] | None = None
) -> dict[int, int]:
    """A connected chain of ``n`` well-calibrated qubits as ``{0: p0, 1: p1, ...}``.

    A deterministic beam search finds the chain. The search minimizes the summed average
    infidelity of the typical 1-qubit gate, mean readout error and typical 2-qubit gate along
    the chain. Two qubits can be consecutive in the chain if and only if the pair has a usable
    2-qubit gate. The pair counts when connectivity lists it, and also when only a calibration
    record lists it. ``usable_pair(a, b)``, with ``a < b``, can remove more pairs: the chain
    then has no consecutive pair for which ``usable_pair`` returns False. The chain has no qubit
    whose measurement the profile disables.

    When such a chain exists, every qubit in the chain has every 1-qubit native that the device
    has. Those natives are the unitary 1-qubit gates, other than the identity, that are usable
    on at least one qubit after calibration records apply. A gate disabled by default counts
    when a record enables it on one qubit or more. A gate disabled everywhere does not count.
    A qubit without one of those natives is incomplete, even when its other gates could make the
    missing gate (an IBM qubit without x alone). Thus the chain has the same basis as the rest
    of the device. A transpiler that compiles for the device then also compiles for the chain.
    When no chain of complete qubits exists, the chain uses as few incomplete qubits as possible.
    A NoiseVaultWarning names those qubits and their missing gates.

    Missing calibration ranks next, whatever the calibrated errors. A chain with fewer qubits
    without a calibrated 1-qubit gate always wins. Next, a chain with fewer unknown readout
    errors wins. On an all-to-all device, the search takes the ``n`` qubits with the lowest
    1-qubit and readout cost when every consecutive pair is usable. If not, the search
    works as on any other device. The result is a starting point for small experiments, not a
    circuit placer.
    """
    table = profile.table
    if not 1 <= n <= table.num_qubits:
        raise LayoutError(f"cannot choose {n} qubits on {profile.id} ({table.num_qubits} qubits)")
    enabled = [q for q in range(table.num_qubits) if not table.qubit(q).disabled]
    if len(enabled) < n:
        raise LayoutError(f"{profile.id} has only {len(enabled)} usable qubits, not {n}")
    usable = [q for q in enabled if can_measure(table, q)]
    if len(usable) < n:
        raise LayoutError(
            f"{profile.id} has only {len(usable)} enabled qubits that can measure, not {n}"
        )
    required = [
        name
        for name in table.profile.gates
        if table.arity(name) == 1
        and _needed(name)
        and any(table.allowed(name, (q,)) for q in usable)
    ]
    lacking = {q: tuple(g for g in required if not table.allowed(g, (q,))) for q in usable}
    qubit_cost = {q: _qubit_cost(table, q, lacking[q]) for q in usable}
    edge_costs: dict[tuple[int, int], _Cost | None] = {}

    def edge(a: int, b: int) -> _Cost | None:
        key = (min(a, b), max(a, b))
        if key not in edge_costs:
            ruled_out = usable_pair is not None and not usable_pair(*key)
            edge_costs[key] = None if ruled_out else _edge_cost(table, *key)
        return edge_costs[key]

    complete = [q for q in usable if not lacking[q]]
    path = _chain(table, complete, qubit_cost, edge, n) or _chain(
        table, usable, qubit_cost, edge, n
    )
    if path is None:
        raise LayoutError(f"{profile.id} has no connected chain of {n} usable qubits")
    short = [f"{q} ({', '.join(lacking[q])} disabled)" for q in path if lacking[q]]
    if short:
        warn_from_caller(
            f"{profile.id} has no connected chain of {n} qubits that each have every 1-qubit"
            f" native the device has, so this chain includes qubit{'s' if len(short) > 1 else ''}"
            f" {', '.join(short)}. A transpiler can fail to place 1-qubit gates on those qubits",
            NoiseVaultWarning,
        )
    return dict(enumerate(path))


def _chain(
    table: NoiseTable,
    pool: list[int],
    qubit_cost: dict[int, _Cost],
    edge: Callable[[int, int], _Cost | None],
    n: int,
) -> tuple[int, ...] | None:
    if len(pool) < n:
        return None
    if table.all_to_all:
        chain = tuple(sorted(pool, key=lambda q: (qubit_cost[q], q))[:n])
        if all(edge(a, b) is not None for a, b in zip(chain, chain[1:], strict=False)):
            return chain
    neighbors = _neighbors(table, pool)
    for width in _BEAMS:  # a wider beam only when a narrow one walks into dead ends
        path = _beam_search(pool, qubit_cost, neighbors, edge, n, width)
        if path is not None:
            return path
    return None


def _beam_search(
    usable: list[int],
    qubit_cost: dict[int, _Cost],
    neighbors: dict[int, list[int]],
    edge: Callable[[int, int], _Cost | None],
    n: int,
    width: int,
) -> tuple[int, ...] | None:
    beam = sorted((qubit_cost[q], (q,)) for q in usable)[:width]
    for _ in range(n - 1):
        grown: dict[tuple[int, ...], _Cost] = {}
        for cost, path in beam:
            members = set(path)
            for at_tail, end in ((True, path[-1]), (False, path[0])):
                for nb in neighbors[end]:
                    step = None if nb in members else edge(end, nb)
                    if step is None:
                        continue
                    new = path + (nb,) if at_tail else (nb, *path)
                    key = min(new, new[::-1])
                    total = cost + step + qubit_cost[nb]
                    if key not in grown or total < grown[key]:
                        grown[key] = total
        if not grown:
            return None
        beam = sorted((cost, path) for path, cost in grown.items())[:width]
    return beam[0][1]


def _qubit_cost(table: NoiseTable, q: int, lacking: tuple[str, ...]) -> _Cost:
    one = table.typical(1, (q,))
    readout = table.qubit(q).readout
    gate_error = one.avg_infidelity if isinstance(one, GateNoise) else None
    readout_error = None if readout is None else sum(readout) / 2
    return _Cost(
        int(bool(lacking)),
        int(gate_error is None),
        int(readout_error is None),
        (gate_error or 0.0) + (readout_error or 0.0),
    )


def _edge_cost(table: NoiseTable, a: int, b: int) -> _Cost | None:
    costs = [
        found.avg_infidelity
        for found in (table.typical(2, (a, b)), table.typical(2, (b, a)))
        if isinstance(found, GateNoise) and found.avg_infidelity is not None
    ]
    return _Cost(0, 0, 0, min(costs)) if costs else None


def _neighbors(table: NoiseTable, usable: list[int]) -> dict[int, list[int]]:
    allowed = set(usable)
    out: dict[int, set[int]] = {q: set() for q in usable}
    pairs = combinations(usable, 2) if table.all_to_all else table.listed_pairs()
    for a, b in pairs:
        if a in allowed and b in allowed:
            out[a].add(b)
            out[b].add(a)
    return {q: sorted(nbs) for q, nbs in out.items()}


def can_measure(table: NoiseTable, q: int) -> bool:
    found = table.gate("measure", (q,))
    return not (isinstance(found, GateNoise) and found.state == "disabled")


def _needed(name: str) -> bool:
    """A 1-qubit gate a transpiler may need: unitary and not the identity, or unknown."""
    info = gates.lookup(name)
    if info is None or info.params:
        return True
    return info.unitary is not None and not np.allclose(info.unitary(), np.eye(2))
