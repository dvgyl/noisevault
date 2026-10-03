"""Conformance check: every framework export against the NoiseVault reference on small circuits.

``check(profile)`` builds a few circuits from the native gates of the profile on a well
calibrated chain of qubits. The framework-free reference simulator computes the outcome
probabilities of these circuits, readout included. The check then runs the same circuits
through each installed export, with the simulator and readout mechanism of that framework.

The check compares Qiskit, Cirq and PennyLane exactly. The check samples Stim and compares
the samples against the Pauli-twirled reference, which is the model that Stim implements.
The Stim samples use exact readout. On the widest circuit and the measurement-only circuit, the
check also samples through the default symmetrized readout of the Stim export.

A pass certifies that the exports implement the same noise model as the reference on these
circuits. A pass says nothing about how closely that model matches the hardware.
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from math import pi
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from . import gates
from .channels import pauli_kraus, pauli_twirl, readout_matrix
from .conversion import resolve_op
from .errors import (
    LayoutError,
    LociText,
    NoiseApproximationWarning,
    NoiseVaultError,
    NoiseVaultWarning,
    install_hint,
    qubit_loci,
)
from .layout import can_measure, normalize_layout, suggest_layout
from .reference import Op, _apply, charged_as
from .reference import probabilities as reference_probabilities
from .report import Report
from .table import GateNoise

if TYPE_CHECKING:
    from .profile import Profile
    from .table import NoiseTable

FRAMEWORKS = ("qiskit", "cirq", "pennylane", "stim")
EXACT_TOLERANCE = 1e-9
SIGMAS = 5.0
MAX_QUBITS = 4
NOTE = (
    "A pass means each export implements the same noise model as the NoiseVault reference on"
    " these circuits. A pass does not measure how well the model matches the hardware."
)
_EXTRAS = {"qiskit": "qiskit", "cirq": "cirq", "pennylane": "pennylane", "stim": "stim"}
# Gate angles for check circuits: pi/2 unless listed. r keeps a phase off 0 so Cirq does not
# turn it into an X rotation. p takes an angle no fixed gate has: Cirq cannot tell p(pi/2)
# from S and charges a profile's s for it.
_ANGLES: dict[str, tuple[float, ...]] = {
    "r": (pi / 2, pi / 4),
    "p": (2 * pi / 3,),
    "ms": (0.0, 0.0),
}
# Rotations that at pi/2 equal a fixed native, whose noise the exports then charge when the
# profile defines that native. So the check runs these rotations at pi/4 (see _two_qubit_ops).
_FIXED_AT_HALF_PI = {"rzz": "zz", "rxx": "ms", "ryy": "ms"}
# Fixed z-family gates, which a profile without their own calibration charges as p, else rz.
_FIXED_PHASES = ("s", "z", "sdg", "t", "tdg")
_MAX_ORDER = 8
_CHECK_ONLY = frozenset({"chain_mirror"})


@dataclass(frozen=True)
class Circuit:
    """A check circuit on circuit qubits ``0..num_qubits-1`` (layout qubits in order)."""

    name: str
    num_qubits: int
    ops: tuple[Op, ...]


@dataclass(frozen=True)
class CircuitCheck:
    circuit: str
    num_qubits: int
    gates: tuple[str, ...]
    tvd: float
    deviation: float
    tolerance: float
    sampled: bool = False

    @property
    def passed(self) -> bool:
        return self.deviation <= self.tolerance


@dataclass(frozen=True)
class NotRun:
    """A planned check circuit, or part of one, that a framework did not run.

    ``ran_without`` names the gates that a reduced version of the circuit left out.
    ``ran_without`` is empty when the circuit did not run at all.
    """

    circuit: str
    reason: str
    ran_without: tuple[str, ...] = ()

    def describe(self) -> str:
        if self.ran_without:
            return f"{self.circuit} ran without {', '.join(self.ran_without)}: {self.reason}"
        return f"{self.circuit} not run: {self.reason}"


@dataclass(frozen=True)
class FrameworkCheck:
    framework: str
    version: str
    method: str
    circuits: tuple[CircuitCheck, ...]
    not_run: tuple[NotRun, ...]  # what of the plan this framework cannot express
    reports: tuple[Report, ...]

    @property
    def passed(self) -> bool:
        return bool(self.circuits) and all(c.passed for c in self.circuits)

    @property
    def worst(self) -> CircuitCheck:
        return max(self.circuits, key=lambda c: c.deviation / c.tolerance)

    @property
    def max_tvd(self) -> float:
        return max(c.tvd for c in self.circuits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "version": self.version,
            "method": self.method,
            "passed": self.passed,
            "max_tvd": self.max_tvd,
            "worst": {
                "circuit": self.worst.circuit,
                "tvd": self.worst.tvd,
                "deviation": self.worst.deviation,
                "tolerance": self.worst.tolerance,
            },
            "circuits": [
                {**c.__dict__, "gates": list(c.gates), "passed": c.passed} for c in self.circuits
            ],
            "not_run": [
                {"circuit": n.circuit, "reason": n.reason, "ran_without": list(n.ran_without)}
                for n in self.not_run
            ],
            "reports": [r.to_dict() for r in self.reports],
        }


@dataclass(frozen=True)
class CheckResult:
    profile_id: str
    fingerprint: str
    layout: dict[int, int]
    shots: int
    seed: int | None
    circuits: tuple[Circuit, ...]
    frameworks: tuple[FrameworkCheck, ...]
    skipped: tuple[tuple[str, str], ...] = ()  # (framework, reason)

    @property
    def passed(self) -> bool:
        """At least one framework ran, and every one that ran passed."""
        return bool(self.frameworks) and all(f.passed for f in self.frameworks)

    def summary(self) -> str:
        qubits = qubit_loci([self.layout[i] for i in range(len(self.layout))])
        lines = [
            f"check {self.profile_id} nv:{self.fingerprint[:12]}: {len(self.circuits)} circuits"
            f" on {qubits}"
        ]
        for f in self.frameworks:
            verdict, worst = "pass" if f.passed else "FAIL", f.worst
            lines.append(
                f"  {f.framework:<10} {verdict}  deviation {worst.deviation:.2g} on {worst.circuit}"
                f" (tolerance {worst.tolerance:.2g}), {len(f.circuits)} circuits, {f.method}"
            )
            lines += [f"    {n.describe()}" for n in f.not_run]
        lines += [f"  {name:<10} skipped: {why}" for name, why in self.skipped]
        lines.append(NOTE)
        return "\n".join(lines)

    __str__ = summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "fingerprint": self.fingerprint,
            "passed": self.passed,
            "layout": {str(k): v for k, v in self.layout.items()},
            "shots": self.shots,
            "seed": self.seed,
            "circuits": [
                {
                    "name": c.name,
                    "num_qubits": c.num_qubits,
                    "ops": [[op.name, list(op.qubits), list(op.params)] for op in c.ops],
                }
                for c in self.circuits
            ],
            "frameworks": [f.to_dict() for f in self.frameworks],
            "skipped": [{"framework": n, "reason": r} for n, r in self.skipped],
            "note": NOTE,
        }


def check(
    profile: Profile,
    *,
    frameworks: Sequence[str] | None = None,
    layout: Mapping[Hashable, int] | Sequence[int] | None = None,
    shots: int = 20_000,
    seed: int | None = 0,
) -> CheckResult:
    """Run the check circuits through every installed export (or ``frameworks``).

    ``layout`` gives the chain of physical qubits to use: 1 to 4 qubits that can measure, with
    connected neighbors. The default chain comes from the ``profile.suggest_layout(n)`` search
    for the largest ``n`` that has a check circuit on its chain. Here the search also requires a
    calibrated 2-qubit native with a known unitary on each pair of neighbors. ``shots`` and
    ``seed`` apply to sampled frameworks (Stim).
    """
    names = list(FRAMEWORKS if frameworks is None else frameworks)
    unknown = [n for n in names if n not in FRAMEWORKS]
    if unknown:
        raise ValueError(f"unknown framework {unknown[0]!r}. Choose from {', '.join(FRAMEWORKS)}")
    validate_shots(shots)
    chain, circuits = plan_circuits(profile, layout, purpose="check")
    expected = _Expected(profile, chain)
    results, skipped = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoiseApproximationWarning)  # each report records it
        for name in names:
            try:
                outcome = _run(_RUNNERS[name](profile, chain), circuits, expected, shots, seed)
            except ImportError:
                outcome = f"not installed: {install_hint(_EXTRAS[name])}"
            except NoiseVaultError as exc:
                outcome = f"the export refused this profile: {exc}"
            if isinstance(outcome, str):
                skipped.append((name, outcome))
            else:
                results.append(outcome)
    return CheckResult(
        profile_id=profile.id,
        fingerprint=profile.fingerprint,
        layout=dict(enumerate(chain)),
        shots=shots,
        seed=seed,
        circuits=circuits,
        frameworks=tuple(results),
        skipped=tuple(skipped),
    )


# circuits -----------------------------------------------------------------------------------


def validate_shots(shots: int) -> None:
    if not isinstance(shots, int) or shots < 1:
        raise ValueError(f"shots={shots!r}: give a positive number of shots")


def plan_circuits(
    profile: Profile,
    layout: Mapping[Hashable, int] | Sequence[int] | None,
    *,
    purpose: Literal["check", "run"],
) -> tuple[list[int], tuple[Circuit, ...]]:
    chain, circuits = _chain_and_circuits(profile, layout)
    if not circuits:
        raise NoiseVaultError(
            f"{profile.id} has no calibrated native gate with a known unitary on"
            f" {qubit_loci(chain)}, so there is nothing to {purpose}",
            hint=_layout_hint(profile, "pass layout= with other qubits"),
        )
    if purpose == "run":
        circuits = tuple(c for c in circuits if c.name not in _CHECK_ONLY)
    return chain, circuits


def _chain_and_circuits(
    profile: Profile, layout: Mapping[Hashable, int] | Sequence[int] | None
) -> tuple[list[int], tuple[Circuit, ...]]:
    if layout is not None:
        chain = _chain(profile, layout)
        return chain, build_circuits(profile, chain)
    found = _connected_chain(profile)
    if found is not None:
        return found
    chain = list(profile.suggest_layout(1).values())
    return chain, build_circuits(profile, chain)


def _connected_chain(profile: Profile) -> tuple[list[int], tuple[Circuit, ...]] | None:
    """The longest suggested chain of 2 or more qubits that has check circuits."""
    for n in range(min(MAX_QUBITS, profile.device.num_qubits), 1, -1):
        try:
            suggested = suggest_layout(
                profile, n, usable_pair=lambda a, b: _usable_pair(profile, a, b)
            )
            chain = _chain(profile, suggested)
        except LayoutError:
            continue
        circuits = build_circuits(profile, chain)
        if circuits:
            return chain, circuits
    return None


def _chain(profile: Profile, layout: Mapping[Hashable, int] | Sequence[int]) -> list[int]:
    n = len(layout)
    if not 1 <= n <= MAX_QUBITS:
        raise LayoutError(f"a check layout has 1 to {MAX_QUBITS} qubits, got {n}")
    mapping = normalize_layout(range(n), layout, profile)
    chain = [mapping[i] for i in range(n)]
    table = profile.table
    unmeasured = [(q,) for q in chain if not can_measure(table, q)]
    if unmeasured:
        measurable = any(_measurable(table, q) for q in range(table.num_qubits))
        raise LayoutError(
            f"{profile.id} disables measure on {qubit_loci(*unmeasured)}, but every check"
            " circuit measures all its qubits",
            hint=_layout_hint(profile, "use qubits that can measure") if measurable else None,
        )
    for a, b in zip(chain, chain[1:], strict=False):
        if not _usable_pair(profile, a, b):
            raise LayoutError(
                f"qubits {a} and {b} share no calibrated 2-qubit native gate",
                hint=_layout_hint(profile, "pass a layout whose neighbors share one"),
            )
    return chain


_NO_ONE_QUBIT = "use a profile that calibrates a 1-qubit native gate with a known unitary"


def _layout_hint(profile: Profile, step: str) -> str:
    """``step`` with a layout that has check circuits, or the calibration that no layout has."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoiseVaultWarning)
        found = _connected_chain(profile)
    table = profile.table
    singles = (
        [q]
        for q in range(table.num_qubits)
        if _measurable(table, q) and build_circuits(profile, [q])
    )
    layout = found[0] if found else next(singles, None)
    return _NO_ONE_QUBIT if layout is None else f"{step}, such as layout={layout}"


def _measurable(table: NoiseTable, q: int) -> bool:
    return not table.qubit(q).disabled and can_measure(table, q)


def _usable_pair(profile: Profile, a: int, b: int) -> bool:
    return not _unitary_natives(profile, 2) or bool(_two_qubit_ops(profile, (a, b), 0))


def build_circuits(
    profile: Profile, chain: Sequence[int], expressible: Callable[[Op], bool] = lambda op: True
) -> tuple[Circuit, ...]:
    table = profile.table
    n = len(chain)
    ones = [
        name
        for name in _unitary_natives(profile, 1)
        if all(_calibrated(table.gate(name, (p,))) for p in chain)
        and all(expressible(_op(name, (q,))) for q in range(n))
    ]
    mix = next((g for g in ones if _mixes(_op(g, (0,)))), None)
    pairs = [
        [op for op in _two_qubit_ops(profile, chain, i) if expressible(op)] for i in range(n - 1)
    ]
    entangles = n >= 2 and all(pairs)
    # Without a gate that makes a superposition, the entangling and mirror circuits act only
    # on |0...0>, where neither a 2-qubit gate's unitary nor coherent noise has any effect.
    circuits: list[Circuit | None] = []
    if mix is not None and entangles:
        circuits += [_ghz_chain(mix, pairs, n), _chain_mirror(mix, pairs, n)]
    if mix is not None and (entangles or n == 1 or not _unitary_natives(profile, 2)):
        k = min(3, n)
        circuits.append(_mirror(mix, _entangle(pairs, k) if entangles else [], ones, k))
    if ones:
        circuits.append(_single_qubit(ones, min(2, n)))
    if mix is not None and n >= 2 and len(pairs[0]) > 1:
        circuits.append(_two_qubit_natives(mix, pairs[0]))
    phase = _fixed_phase_gate(profile, chain[0])
    if mix is not None and phase is not None and expressible(phase):
        circuits.append(_fixed_phase(mix, phase))
    built = [c for c in circuits if c is not None]
    if built:
        built.append(Circuit("readout", n, ()))
    return tuple(built)


def _ghz_chain(mix: str, pairs: list[list[Op]], n: int) -> Circuit:
    return Circuit("ghz_chain", n, (*_layer(mix, n), *_entangle(pairs, n), *_layer(mix, n)))


def _chain_mirror(mix: str, pairs: list[list[Op]], n: int) -> Circuit | None:
    chain = _entangle(pairs, n)
    qubit_0_superposed = _undone([*_layer(mix, 1), *chain])
    every_qubit_superposed = _undone([*_layer(mix, n), *chain])
    if qubit_0_superposed is None or every_qubit_superposed is None:
        return None
    return Circuit("chain_mirror", n, (*qubit_0_superposed, *every_qubit_superposed))


def _mirror(mix: str, entangling: list[Op], ones: list[str], k: int) -> Circuit | None:
    forward = [*_layer(mix, k), *entangling, *(_op(g, (q,)) for q in range(k) for g in ones)]
    inverses = {op: ops for op in forward if (ops := _inverse(op)) is not None}
    forward = [op for op in forward if op in inverses]
    if not forward:
        return None
    backward = [inv for op in reversed(forward) for inv in inverses[op]]
    return Circuit("mirror", k, (*forward, *backward))


def _single_qubit(ones: list[str], k: int) -> Circuit:
    return Circuit(
        "single_qubit", k, tuple(_op(g, (q,)) for q in range(k) for _ in range(2) for g in ones)
    )


def _two_qubit_natives(mix: str, natives: list[Op]) -> Circuit:
    return Circuit("two_qubit_natives", 2, (*_layer(mix, 2), *natives, *_layer(mix, 2)))


def _fixed_phase(mix: str, phase: Op) -> Circuit | None:
    # Undone like the mirror circuit, so the phase gate's Z errors show as |1>.
    undone = _undone([*_layer(mix, 1), phase])
    return None if undone is None else Circuit("fixed_phase", 1, tuple(undone))


def _layer(mix: str, k: int) -> list[Op]:
    return [_op(mix, (q,)) for q in range(k)]


def _entangle(pairs: list[list[Op]], k: int) -> list[Op]:
    return [pairs[i][0] for i in range(k - 1)]


def _fixed_phase_gate(profile: Profile, qubit: int) -> Op | None:
    """A fixed z-family gate that the profile leaves to its ``p`` calibration.

    Every export must then charge the ``p`` noise for this gate, ahead of the ``rz`` noise.
    Returns None when ``p`` has no calibration on ``qubit``.
    """
    if not _calibrated(profile.table.gate("p", (qubit,))):
        return None
    name = next((g for g in _FIXED_PHASES if g not in profile.gates), None)
    return None if name is None else _op(name, (0,))


def _unitary_natives(profile: Profile, arity: int) -> list[str]:
    """Natives of ``arity`` with a registry unitary, most calibration records first."""
    records = Counter(r.gate for r in profile.calibrations)
    names = [n for n in profile.table.natives(arity) if gates.lookup(n) is not None]
    return sorted(names, key=lambda n: (-records[n], n))


def _two_qubit_ops(profile: Profile, chain: Sequence[int], i: int) -> list[Op]:
    """Each calibrated 2-qubit native on circuit pair (i, i+1), in an operand order it allows."""
    table = profile.table
    out = []
    for name in _unitary_natives(profile, 2):
        for a, b in ((i, i + 1), (i + 1, i)):
            found = table.gate(name, (chain[a], chain[b]))
            if _calibrated(found):
                op = _op(name, (a, b))
                if _FIXED_AT_HALF_PI.get(name) in profile.gates:
                    op = Op(name, op.qubits, (pi / 4,))
                out.append(op)
                break
    return out


def _calibrated(found: Any) -> bool:
    return isinstance(found, GateNoise) and found.state == "calibrated"


def _op(name: str, qubits: tuple[int, ...]) -> Op:
    info = gates.GATES[name]
    return Op(name, qubits, _ANGLES.get(name, (pi / 2,) * len(info.params)))


def _unitary(op: Op) -> np.ndarray:
    return gates.unitary(op.name, op.params)


def _mixes(op: Op) -> bool:
    """True when the gate takes |0> to a superposition."""
    return 0.1 < abs(_unitary(op)[0, 0]) ** 2 < 0.9


def _inverse(op: Op) -> list[Op] | None:
    """The gate repeated until the product is the identity up to phase, or None."""
    u = _unitary(op)
    power = u
    for k in range(1, _MAX_ORDER + 1):
        phase = power[0, 0]
        if abs(abs(phase) - 1) < 1e-9 and np.allclose(power, phase * np.eye(len(u)), atol=1e-9):
            return [op] * (k - 1)
        power = u @ power
    return None


def _undone(forward: list[Op]) -> list[Op] | None:
    undo: list[Op] = []
    for op in reversed(forward):
        ops = _inverse(op)
        if ops is None:
            return None
        undo += ops
    return [*forward, *undo]


# expected probabilities ---------------------------------------------------------------------


class _Expected:
    """Reference probabilities per circuit, plain and Pauli-twirled, computed once each.

    ``symmetrized`` gives the twirled reference each qubit's mean readout error in both
    directions, as Stim's default export flips results.
    """

    def __init__(self, profile: Profile, chain: Sequence[int]) -> None:
        self.profile = profile
        self.chain = list(chain)
        self._cache: dict[tuple[int, tuple[Op, ...], bool, bool], np.ndarray] = {}

    def __call__(self, circuit: Circuit, *, twirled: bool, symmetrized: bool = False) -> np.ndarray:
        key = (circuit.num_qubits, circuit.ops, twirled, symmetrized)
        if key not in self._cache:
            layout = self.chain[: circuit.num_qubits]
            if twirled:
                self._cache[key] = _twirled(self.profile, circuit, layout, symmetrized)
            else:
                self._cache[key] = reference_probabilities(
                    self.profile, circuit.ops, circuit.num_qubits, layout=layout, readout=True
                )
        return self._cache[key]


def _twirled(
    profile: Profile, circuit: Circuit, layout: list[int], symmetrized: bool
) -> np.ndarray:
    """The reference with each gate's channel replaced by its Pauli twirl (Stim's model)."""
    n, table = circuit.num_qubits, profile.table
    report = Report.start(profile, "reference", None)
    rho = np.zeros((2,) * (2 * n), dtype=complex)
    rho[(0,) * (2 * n)] = 1.0
    for op in circuit.ops:
        unitary = _unitary(op)
        rho = _apply(rho, [unitary], op.qubits, n)
        wires = tuple(layout[q] for q in op.qubits)
        name = charged_as(profile, op.name, unitary)
        built = resolve_op(table, name, wires, unknown_gates="error", report=report)
        if built.channels:
            rho = _apply(rho, pauli_kraus(pauli_twirl(built.channels, wires)), op.qubits, n)
    probs = np.real(np.diagonal(rho.reshape(2**n, 2**n)))
    matrices = [readout_matrix(table.qubit(q)) for q in layout]
    if symmetrized:
        matrices = [None if m is None else (m + m[::-1, ::-1]) / 2 for m in matrices]
    probs = _with_readout(probs, matrices)
    # Rounding can leave a certain outcome at 1 + 1e-16, whose sampling variance is negative.
    probs = np.clip(probs, 0.0, None)
    return probs / probs.sum()


def _with_readout(probs: np.ndarray, matrices: Sequence[np.ndarray | None]) -> np.ndarray:
    """Apply M[measured, prepared] per qubit to big-endian probabilities (None: no error)."""
    probs = np.asarray(probs, dtype=float).reshape((2,) * len(matrices))
    for axis, matrix in enumerate(matrices):
        if matrix is not None:
            probs = np.moveaxis(np.tensordot(matrix, probs, axes=([1], [axis])), 0, axis)
    return probs.reshape(-1)


def _tvd(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


# running ------------------------------------------------------------------------------------


class _Runner:
    """One framework: which ops it can express, and how it runs a circuit through its export."""

    framework: str
    version: str
    sampled = False
    twirled = False
    symmetrized = False  # sample_measured flips each bit with the qubit's mean readout error

    def __init__(self, profile: Profile, chain: list[int]) -> None:
        self.profile = profile
        self.chain = chain

    def cannot_express(self, op: Op) -> str | None:
        raise NotImplementedError

    def run(self, circuit: Circuit, shots: int, seed: int | None) -> np.ndarray:
        """Exact probabilities, or sampled frequencies, big-endian over circuit qubits."""
        raise NotImplementedError

    def sample_measured(self, circuit: Circuit, shots: int, seed: int | None) -> np.ndarray | None:
        """Frequencies through the framework's own measurement, when ``run`` bypasses it."""
        return None

    def reports(self) -> list[Report]:
        raise NotImplementedError


def _run(
    runner: _Runner,
    plan: Sequence[Circuit],
    expected: _Expected,
    shots: int,
    seed: int | None,
) -> FrameworkCheck | str:
    """The framework's check on the circuits it can express, or why it ran none."""
    circuits = build_circuits(
        runner.profile, runner.chain, lambda op: not runner.cannot_express(op)
    )
    built = {c.name: c for c in circuits}
    not_run = []
    for planned in plan:
        ran = built.get(planned.name)
        # A reduced version keeps the circuit's name, so compare the operations themselves.
        left_out = list((Counter(planned.ops) - Counter(ran.ops if ran else ())).elements())
        if left_out:
            reason = "; ".join(dict.fromkeys(filter(None, map(runner.cannot_express, left_out))))
            names = tuple(dict.fromkeys(op.name for op in left_out)) if ran else ()
            not_run.append(NotRun(planned.name, reason, names))
    checks = [
        _compare(
            c,
            runner.run(c, shots, seed),
            expected(c, twirled=runner.twirled),
            shots,
            runner.sampled,
        )
        for c in circuits
    ]
    if not checks:
        reasons = dict.fromkeys(part for n in not_run for part in n.reason.split("; "))
        return " ".join(f"{reason[:1].upper()}{reason[1:]}." for reason in reasons)
    # The widest circuit's outcomes can be uniform, which readout error leaves unchanged, so
    # the check also samples the measurement-only circuit through the framework's measurement.
    widest = max((c for c in circuits if c.ops), key=lambda c: c.num_qubits)
    measured_names = []
    for circuit in (widest, *(c for c in circuits if not c.ops)):
        measured = runner.sample_measured(circuit, shots, seed)
        if measured is None:
            break
        want = expected(circuit, twirled=runner.twirled, symmetrized=runner.symmetrized)
        checks.append(_compare(circuit, measured, want, shots, True))
        measured_names.append(circuit.name)
    names = " and ".join(measured_names)
    if runner.sampled:
        method = f"sampled {shots} shots against the twirled reference"
        if names:
            method += f", and {names} with the export's default symmetrized readout"
        method += f", {SIGMAS:g} sigma"
    elif not names:
        method = "exact"
    else:
        method = (
            f"exact, and {names} sampled {shots} shots through the framework's"
            f" measurement, {SIGMAS:g} sigma"
        )
    return FrameworkCheck(
        framework=runner.framework,
        version=runner.version,
        method=method,
        circuits=tuple(checks),
        not_run=tuple(not_run),
        reports=_merged(runner.reports()),
    )


def _merged(reports: Sequence[Report]) -> tuple[Report, ...]:
    merged: list[Report] = []
    for report in reports:
        into = next((m for m in merged if m.options == report.options), None)
        if into is None:
            into = replace(
                report, exact=[], approximated=[], omitted=[], unknown=[], clamped=[], events={}
            )
            merged.append(into)
        for what in report.exact:
            into.mark_exact(what)
        for a in report.approximated:
            into.approximate(a.what, a.how, a.detail)
        into.omitted[:] = _merge_loci(into.omitted, report.omitted)
        into.unknown[:] = _merge_loci(into.unknown, report.unknown)
        into.clamped += [c for c in report.clamped if c not in into.clamped]
        for event, counts in report.events.items():
            for key, n in counts.items():
                into.count(event, key, n)
    return tuple(merged)


def _merge_loci(entries: Sequence[LociText], more: Sequence[LociText]) -> list[LociText]:
    out = list(entries)
    for new in more:
        i = next((i for i, old in enumerate(out) if _template(old) == _template(new)), None)
        if i is None:
            out.append(new)
        else:
            parts = zip(out[i].parts, new.parts, strict=True)
            out[i] = LociText(*(a if isinstance(a, str) else sorted({*a, *b}) for a, b in parts))
    return out


def _template(text: LociText) -> tuple[str | None, ...]:
    return tuple(part if isinstance(part, str) else None for part in text.parts)


def _compare(
    circuit: Circuit, got: np.ndarray, want: np.ndarray, shots: int, sampled: bool
) -> CircuitCheck:
    tvd = _tvd(got, want)
    deviation, tolerance = tvd, EXACT_TOLERANCE
    if sampled:
        off = np.abs(got - want)
        bound = SIGMAS * np.sqrt(want * (1 - want) / shots) + SIGMAS / shots
        worst = int(np.argmax(off / bound))
        deviation, tolerance = float(off[worst]), float(bound[worst])
    names = tuple(sorted({op.name for op in circuit.ops}))
    return CircuitCheck(circuit.name, circuit.num_qubits, names, tvd, deviation, tolerance, sampled)


def _registry_missing(op: Op, framework: str, column: str | None) -> str | None:
    return None if column else f"{framework} has no {op.name} gate"


class _Qiskit(_Runner):
    framework = "qiskit"

    def __init__(self, profile: Profile, chain: list[int]) -> None:
        super().__init__(profile, chain)
        import qiskit
        import qiskit_aer
        from qiskit.circuit import library

        from .frameworks.qiskit import ALIASES, gate_class, to_qiskit

        self.version = f"{qiskit.__version__} (qiskit-aer {qiskit_aer.__version__})"
        self._library, self._aliases, self._gate_class = library, ALIASES, gate_class
        self.sim = to_qiskit(profile)
        self.sim.set_options(method="density_matrix")
        self._readout = {
            e["gate_qubits"][0][0]: np.array(e["probabilities"]).T  # Aer rows: prepared state
            for e in self.sim.noise_model.to_dict()["errors"]
            if e["type"] == "roerror"
        }

    def _gate(self, op: Op) -> Any:
        alias = self._aliases.get(op.name)
        if alias is not None:
            if alias in self.profile.gates or op.params not in ((), (0.0, 0.0)):
                return None
            return getattr(self._library, gates.GATES[alias].qiskit_class)(pi / 2)
        cls = self._gate_class(op.name)
        return None if cls is None else cls(*op.params)

    def cannot_express(self, op: Op) -> str | None:
        gate = self._gate(op)
        if gate is None:
            return f"Qiskit has no {op.name} gate"
        qargs = tuple(self.chain[q] for q in op.qubits)
        if not self.sim.target.instruction_supported(gate.name, qargs):
            return f"the Qiskit export has no {gate.name} on {qubit_loci(qargs)}"
        return None

    def _circuit(self, circuit: Circuit, clbits: int = 0) -> Any:
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(self.sim.target.num_qubits, clbits)
        for op in circuit.ops:
            qc.append(self._gate(op), [self.chain[q] for q in op.qubits])
        return qc

    def run(self, circuit: Circuit, shots: int, seed: int | None) -> np.ndarray:
        # Exact: Aer's probabilities before measurement, then the exported ReadoutError
        # matrices. sample_measured covers Aer applying them at a real measurement.
        qc = self._circuit(circuit)
        measured = self.chain[: circuit.num_qubits]
        qc.save_probabilities(measured)
        little = np.asarray(self.sim.run(qc).result().data()["probabilities"])
        n = circuit.num_qubits
        big = little.reshape((2,) * n).transpose(range(n - 1, -1, -1)).reshape(-1)
        return _with_readout(big, [self._readout.get(q) for q in measured])

    def sample_measured(self, circuit: Circuit, shots: int, seed: int | None) -> np.ndarray:
        n = circuit.num_qubits
        qc = self._circuit(circuit, n)
        qc.measure(self.chain[:n], range(n))
        counts = self.sim.run(qc, shots=shots, seed_simulator=seed).result().get_counts()
        freq = np.zeros(2**n)
        for bits, count in counts.items():
            freq[int(bits[::-1], 2)] = count / shots  # clbit 0 is the rightmost character
        return freq

    def reports(self) -> list[Report]:
        return [self.sim.report]


class _Cirq(_Runner):
    framework = "cirq"

    def __init__(self, profile: Profile, chain: list[int]) -> None:
        super().__init__(profile, chain)
        import cirq

        from .frameworks.cirq import ECRGate, to_cirq

        self.cirq, self.version = cirq, cirq.__version__
        self.qubits = cirq.LineQubit.range(len(chain))
        self.model = to_cirq(profile, layout=dict(zip(self.qubits, chain, strict=True)))
        c = cirq
        self._gates: dict[str, Callable[..., Any]] = {
            "id": lambda: c.I,
            "x": lambda: c.X,
            "y": lambda: c.Y,
            "z": lambda: c.Z,
            "h": lambda: c.H,
            "s": lambda: c.S,
            "sdg": lambda: c.S**-1,
            "t": lambda: c.T,
            "tdg": lambda: c.T**-1,
            "sx": lambda: c.X**0.5,
            "sxdg": lambda: c.X**-0.5,
            "rx": c.rx,
            "ry": c.ry,
            "rz": c.rz,
            "p": lambda lam: c.ZPowGate(exponent=lam / pi),
            "r": lambda t, p: c.PhasedXPowGate(phase_exponent=p / pi, exponent=t / pi),
            "cx": lambda: c.CNOT,
            "cz": lambda: c.CZ,
            "ecr": ECRGate,
            "swap": lambda: c.SWAP,
            "iswap": lambda: c.ISWAP,
            "sqrt_iswap": lambda: c.ISWAP**0.5,
            "zz": lambda: c.ZZ**0.5,
            "rzz": lambda t: c.ZZPowGate(exponent=t / pi),
            "rxx": lambda t: c.XXPowGate(exponent=t / pi),
            "ryy": lambda t: c.YYPowGate(exponent=t / pi),
            "ms": lambda p0, p1: c.ms(pi / 4) if p0 == p1 == 0 else None,
        }

    def cannot_express(self, op: Op) -> str | None:
        make = self._gates.get(op.name)
        return (
            None
            if make is not None and make(*op.params) is not None
            else (f"Cirq has no {op.name} gate")
        )

    def run(self, circuit: Circuit, shots: int, seed: int | None) -> np.ndarray:
        cirq = self.cirq
        measured = self.qubits[: circuit.num_qubits]
        ops = [
            self._gates[op.name](*op.params).on(*(self.qubits[q] for q in op.qubits))
            for op in circuit.ops
        ]
        noisy = cirq.Circuit([*ops, cirq.measure(*measured, key="m")]).with_noise(self.model)
        # Nothing after the measurement can change its outcome, so the state before the
        # measurement, readout channels included, gives the exact outcome probabilities.
        end = next(i for i, m in enumerate(noisy) if any(cirq.is_measurement(op) for op in m))
        quantum = noisy[:end]
        simulator = cirq.DensityMatrixSimulator(dtype=np.complex128)
        rho = simulator.simulate(quantum, qubit_order=measured).final_density_matrix
        return np.real(np.diag(rho))

    def reports(self) -> list[Report]:
        return [self.model.report]


class _PennyLane(_Runner):
    framework = "pennylane"

    def __init__(self, profile: Profile, chain: list[int]) -> None:
        super().__init__(profile, chain)
        import pennylane as qml

        from .frameworks.pennylane import operation_for, to_pennylane

        self.qml, self.version = qml, qml.__version__
        self.model = to_pennylane(profile, layout=chain)
        self._operation_for = operation_for

    def _cls(self, op: Op) -> Any:
        return self._operation_for(op.name)

    def cannot_express(self, op: Op) -> str | None:
        return None if self._cls(op) is not None else f"PennyLane has no {op.name} gate"

    def run(self, circuit: Circuit, shots: int, seed: int | None) -> np.ndarray:
        qml, n = self.qml, circuit.num_qubits

        @qml.qnode(qml.device("default.mixed", wires=n))
        def run() -> Any:
            for op in circuit.ops:
                self._cls(op)(*op.params, wires=list(op.qubits))
            return qml.probs(wires=range(n))

        return np.asarray(qml.add_noise(run, self.model, level="top")(), dtype=float)

    def reports(self) -> list[Report]:
        return [self.model.report]


class _Stim(_Runner):
    framework = "stim"
    sampled = True
    twirled = True
    symmetrized = True

    def __init__(self, profile: Profile, chain: list[int]) -> None:
        super().__init__(profile, chain)
        import stim

        from .frameworks.stim import gate_name, sample_with_readout, to_stim

        self.version = stim.__version__
        self._to_stim, self._sample, self._gate_name = to_stim, sample_with_readout, gate_name
        self._reports: list[Report] = []
        self._stim_gates = [
            (name, _big_endian(data.unitary_matrix))
            for name, data in stim.gate_data().items()
            if data.is_unitary and data.unitary_matrix is not None  # SPP has no fixed size
        ]

    def _equal_gates(self, op: Op) -> list[str]:
        """Stim gates equal to ``op`` up to global phase (Stim stores them in single precision)."""
        u = _unitary(op)
        return [
            name
            for name, v in self._stim_gates
            if v.shape == u.shape and abs(abs(np.vdot(u, v)) - len(u)) < 1e-6
        ]

    def _instruction(self, op: Op) -> str | None:
        """A Stim gate equal to ``op``, or None.

        The export must charge the noise of ``op.name`` for the Stim gate. Or the Stim gate
        must be ``op`` itself, when the profile leaves ``op`` to the calibration of another gate.
        """
        defined = self.profile.gates
        return next(
            (
                s
                for s in self._equal_gates(op)
                if op.name in (self._gate_name(s, defined), self._gate_name(s))
            ),
            None,
        )

    def cannot_express(self, op: Op) -> str | None:
        if self._instruction(op) is not None:
            return None
        equal = self._equal_gates(op)
        if equal:
            charged = self._gate_name(equal[0], self.profile.gates)
            return (
                f"Stim's {equal[0]} equals {op.name} here but takes the profile's {charged} noise"
            )
        if gates.GATES[op.name].params:
            return (
                f"Stim simulates only Clifford gates, and this profile's {op.name} gate is not"
                " Clifford at the check angles"
            )
        return f"Stim has no {op.name} instruction"

    def run(self, circuit: Circuit, shots: int, seed: int | None) -> np.ndarray:
        return self._frequencies(circuit, shots, seed, "exact")

    def sample_measured(self, circuit: Circuit, shots: int, seed: int | None) -> np.ndarray:
        return self._frequencies(circuit, shots, seed, "symmetrize")

    def _frequencies(
        self, circuit: Circuit, shots: int, seed: int | None, readout: str
    ) -> np.ndarray:
        n = circuit.num_qubits
        lines = [f"{self._instruction(op)} {' '.join(map(str, op.qubits))}" for op in circuit.ops]
        lines.append("M " + " ".join(map(str, range(n))))
        noisy = self._to_stim(
            self.profile, "\n".join(lines), layout=dict(enumerate(self.chain)), readout=readout
        )
        self._reports.append(noisy.report)
        if readout == "exact":
            bits = self._sample(noisy, shots, seed=seed)
        else:
            bits = noisy.compile_sampler(seed=seed).sample(shots)
        index = bits.astype(int) @ (1 << np.arange(n)[::-1])
        return np.bincount(index, minlength=2**n) / shots

    def reports(self) -> list[Report]:
        return self._reports


def _big_endian(u: np.ndarray) -> np.ndarray:
    """A Stim unitary, which puts the first target in the lowest bit, in registry order."""
    k = round(np.log2(len(u)))
    axes = [*range(k - 1, -1, -1), *range(2 * k - 1, k - 1, -1)]
    return u.reshape((2,) * (2 * k)).transpose(axes).reshape(len(u), len(u))


_RUNNERS: dict[str, type[_Runner]] = {
    "qiskit": _Qiskit,
    "cirq": _Cirq,
    "pennylane": _PennyLane,
    "stim": _Stim,
}
