"""The reference simulator gives the results of its first version, kept here as _old_probabilities.

The first version applies each Kraus operator alone and, with a report, resolves the noise of
each op again. The present version must give the same probabilities to 1e-12, the same report
and the same warnings, and raise the same errors.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from math import inf, pi
from typing import Any

import numpy as np
import pytest

import noisevault as nv
from noisevault import gates
from noisevault.channels import readout_matrix
from noisevault.conversion import UnknownGates, idle_channel, resolve_op
from noisevault.layout import normalize_layout
from noisevault.reference import Op, charged_as, probabilities
from noisevault.report import Report
from noisevault.table import refuse_disabled

# the first simulator -------------------------------------------------------------------------


def _old_probabilities(
    profile: nv.Profile,
    ops: Iterable[Op],
    num_qubits: int,
    *,
    layout: Mapping[Any, int] | Sequence[int] | None = None,
    unknown_gates: UnknownGates = "typical",
    readout: bool = True,
    report: Report | None = None,
) -> np.ndarray:
    ops = list(ops)
    physical = normalize_layout(range(num_qubits), layout, profile)
    circuit_of = {p: c for c, p in physical.items()}
    reuse = report is None
    report = report or Report.start(profile, "reference", None, unknown_gates=unknown_gates)
    report.record_effects(profile.effects)
    table = profile.table
    unitaries: dict[tuple[str, tuple[float, ...]], tuple[np.ndarray, str]] = {}
    resolved: dict[tuple[str, tuple[int, ...]], Any] = {}
    idles: dict[tuple[int, float], Any] = {}

    rho = np.zeros((2,) * (2 * num_qubits), dtype=complex)
    rho[(0,) * (2 * num_qubits)] = 1.0
    for op in ops:
        if op.name == "delay":
            (q,), (duration_ns,) = op.qubits, op.params
            if not 0 <= duration_ns < inf:
                raise ValueError(
                    f"the delay on circuit qubit {q} needs a finite duration of 0 ns or more, "
                    f"not {duration_ns} ns"
                )
            wait = (physical[q], duration_ns)
            if not reuse or wait not in idles:
                refuse_disabled(table.gate("delay", (physical[q],)))
                idles[wait] = idle_channel(table, physical[q], duration_ns, report)
            channel = idles[wait]
            if channel is not None:
                rho = _old_apply(rho, channel.kraus, (q,), num_qubits)
            continue
        key = (op.name, tuple(op.params))
        if key not in unitaries:
            info = gates.lookup(op.name)
            if info is None or info.unitary is None:
                raise ValueError(f"the reference simulator has no unitary for {op.name!r}")
            unitary = info.unitary(*op.params)
            unitaries[key] = (unitary, charged_as(profile, op.name, unitary))
        unitary, charged = unitaries[key]
        rho = _old_apply(rho, [unitary], op.qubits, num_qubits)
        targets = tuple(physical[q] for q in op.qubits)
        locus = (charged, targets)
        if not reuse or locus not in resolved:
            resolved[locus] = resolve_op(
                table, charged, targets, unknown_gates=unknown_gates, report=report
            )
        for channel in resolved[locus].channels:
            wires = tuple(circuit_of[w] for w in channel.wires)
            rho = _old_apply(rho, channel.kraus, wires, num_qubits)
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


def _old_apply(rho, kraus, wires, n):
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


# random circuits ----------------------------------------------------------------------------

_UNMODELED = {"gates": {"factor": 1.7}, "readout": {"factor": 1.3}}


def _uniform() -> nv.Profile:
    return nv.Profile.uniform(
        "uniform",
        technology="superconducting",
        num_qubits=6,
        one_qubit_error=2e-3,
        two_qubit_error=2e-2,
        readout_error=0.02,
        t1_us=80,
        t2_us=60,
        one_qubit_ns=40,
        two_qubit_ns=300,
        connectivity=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
    )


_PROFILES = {
    "ibm_cz": lambda: nv.load("ibm_kingston"),
    "ibm_ecr": lambda: nv.load("ibm_kyiv"),
    "quantinuum": lambda: nv.load("quantinuum_h1-1"),
    "uniform": _uniform,
}
# Gates that no bundled profile defines, so that "typical" gives them the typical gate's noise.
_FOREIGN = {1: ("h", "ry"), 2: ("cx", "iswap")}


def _chain(profile: nv.Profile, n: int, rng: np.random.Generator) -> list[int]:
    """``n`` connected physical qubits, in a random order."""
    edges = profile.connectivity.edges if profile.connectivity != "all_to_all" else None
    if edges is None:
        found = list(rng.choice(profile.table.num_qubits, size=n, replace=False))
    else:
        near: dict[int, set[int]] = {}
        for a, b in edges:
            near.setdefault(a, set()).add(b)
            near.setdefault(b, set()).add(a)
        while True:
            found = [int(rng.choice(sorted(near)))]
            while len(found) < n:
                step = sorted(near[found[-1]] - set(found))
                if not step:
                    break
                found.append(int(rng.choice(step)))
            if len(found) == n:
                break
    return [int(q) for q in rng.permutation(found)]


def _params(name: str, rng: np.random.Generator) -> tuple[float, ...]:
    return tuple(float(x) for x in rng.uniform(-pi, pi, len(gates.lookup(name).params)))


def _pairs(profile: nv.Profile, layout: list[int], foreign: bool) -> list[tuple[int, int]]:
    """The circuit pairs that a circuit uses: all of them, or only the connected ones."""
    n = len(layout)
    every = [(a, b) for a in range(n) for b in range(n) if a != b]
    if foreign or profile.connectivity == "all_to_all":
        return every
    edges = {frozenset(e) for e in profile.connectivity.edges}
    return [(a, b) for a, b in every if frozenset((layout[a], layout[b])) in edges]


def _circuit(
    profile: nv.Profile,
    layout: list[int],
    depth: int,
    rng: np.random.Generator,
    foreign: bool,
) -> list[Op]:
    n = len(layout)
    pairs = _pairs(profile, layout, foreign)
    natives = {
        arity: sorted(
            name
            for name in profile.gates
            if (info := gates.lookup(name)) and info.unitary and info.arity == arity
        )
        for arity in (1, 2)
    }
    if foreign:
        natives = {a: [*names, *_FOREIGN[a]] for a, names in natives.items()}
    angles = {name: [_params(name, rng) for _ in range(2)] for name in natives[1] + natives[2]}
    ops = []
    for _ in range(depth):
        kind = rng.uniform()
        if kind < 0.1:
            duration = float(rng.choice([0.0, 50.0, 1234.5, 20_000.0]))
            ops.append(Op("delay", (int(rng.integers(n)),), (duration,)))
        elif kind < 0.65 or not pairs:
            name = str(rng.choice(natives[1]))
            ops.append(Op(name, (int(rng.integers(n)),), angles[name][rng.integers(2)]))
        else:
            name = str(rng.choice(natives[2]))
            pair = pairs[int(rng.integers(len(pairs)))]
            ops.append(Op(name, pair, angles[name][rng.integers(2)]))
    return ops


def _run(simulate, profile, ops, n, report, **options):
    """The outcome of one simulator: probabilities or the error, and the warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            got: Any = simulate(profile, ops, n, report=report, **options)
        except Exception as exc:  # noqa: BLE001 - both simulators must raise the same error
            got = (type(exc), str(exc))
    return got, [(w.category, str(w.message)) for w in caught]


def _same_report(old: Report, new: Report) -> None:
    assert new.to_dict() == old.to_dict()
    assert new.summary() == old.summary()
    assert new.clamped == old.clamped
    assert [list(c.items()) for c in new.events.values()] == [
        list(c.items()) for c in old.events.values()
    ]
    assert new._warned == old._warned


_CASES = [
    (name, unmodeled, seed)
    for name in sorted(_PROFILES)
    for unmodeled in (False, True)
    for seed in range(6)
]


@pytest.mark.parametrize(("name", "unmodeled", "seed"), _CASES)
def test_the_simulator_gives_the_results_of_its_first_version(name, unmodeled, seed) -> None:
    profile = _PROFILES[name]()
    if unmodeled:
        profile = profile.model_copy(update={"unmodeled_error": _UNMODELED})
    rng = np.random.default_rng([seed, len(name), unmodeled])
    n = seed + 1
    layout = _chain(profile, n, rng)
    foreign = seed % 2 == 1
    unknown_gates = "typical" if foreign or seed % 4 == 0 else "error"
    options = {"layout": layout, "unknown_gates": unknown_gates, "readout": seed % 3 != 0}
    circuits = [_circuit(profile, layout, int(rng.integers(5, 60)), rng, foreign) for _ in range(2)]

    for ops in circuits:
        old, old_warned = _run(_old_probabilities, profile, ops, n, None, **options)
        new, new_warned = _run(probabilities, profile, ops, n, None, **options)
        _same_outcome(old, new)
        assert new_warned == old_warned

    # One report for both circuits, as nv compare passes one report for a step of a fit.
    old_report = Report.start(profile, "reference", None)
    new_report = Report.start(profile, "reference", None)
    for ops in circuits:
        old, old_warned = _run(_old_probabilities, profile, ops, n, old_report, **options)
        new, new_warned = _run(probabilities, profile, ops, n, new_report, **options)
        _same_outcome(old, new)
        assert new_warned == old_warned
    _same_report(old_report, new_report)


def _same_outcome(old: Any, new: Any) -> None:
    if isinstance(old, tuple):
        assert new == old
    else:
        assert isinstance(new, np.ndarray)
        assert np.max(np.abs(new - old)) <= 1e-12


def test_a_deep_circuit_gives_the_results_of_the_first_version() -> None:
    profile = nv.load("ibm_kingston").model_copy(update={"unmodeled_error": _UNMODELED})
    rng = np.random.default_rng(7)
    layout = _chain(profile, 5, rng)
    ops = _circuit(profile, layout, 1500, rng, foreign=True)
    old_report = Report.start(profile, "reference", None)
    new_report = Report.start(profile, "reference", None)
    old, _ = _run(_old_probabilities, profile, ops, 5, old_report, layout=layout)
    new, _ = _run(probabilities, profile, ops, 5, new_report, layout=layout)
    _same_outcome(old, new)
    _same_report(old_report, new_report)
