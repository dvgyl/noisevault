"""Leakage, atom loss and erasure as an erasure channel, shared by the Stim and Cirq exports.

An erasure replaces the state of one qubit with I/2, the same as a random Pauli I, X, Y or Z
with probability 1/4 each. docs/profile-format.md (section "Effects") gives the rules: where each
effect acts, which number it uses, and when the report calls it exact or approximated.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import permutations
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .profile import Effect, GateSpec, Profile
    from .table import GateNoise

Level = Literal["omit", "approximate", "exact"]
RANK: dict[Level, int] = {"omit": 0, "approximate": 1, "exact": 2}
ERASING = frozenset({"leakage", "atom_loss", "erasure"})
_NOT_GATES = frozenset({"measure", "reset", "delay"})  # operations the exports handle apart
_HOW = {
    "erasure": ("an erasure (the qubit becomes I/2)", ""),
    "atom_loss": (
        "an erasure (the qubit becomes I/2)",
        "the atom comes back at once; on the device the trap stays empty",
    ),
    "leakage": (
        "an erasure (the qubit becomes I/2)",
        "the population comes back at once; on the device it stays outside the qubit",
    ),
}


@dataclass(frozen=True)
class Treatment:
    """How one export treats one effect: the level, and the words the report uses."""

    level: Level
    how: str = ""
    detail: str = ""


@dataclass(frozen=True)
class Erasure:
    """One effect that erases a qubit with probability ``prob``."""

    type: str
    prob: float
    heralded: bool


def treatment(
    effect: Effect, profile: Profile, framework: str, *, idle: Level, readout: bool
) -> Treatment:
    """The treatment that an export with an erasure channel gives ``effect``.

    ``idle`` is the level of the export's idle model ("omit" when it models no idle time).
    An exact idle model still gives an approximated idle effect on a qubit with relaxation.
    ``readout`` is False when the export adds no readout error, and so no readout effects.
    """
    if effect.type not in ERASING:
        return Treatment("omit")
    if effect.gate in _NOT_GATES:
        return Treatment(
            "omit", detail=f"use on: readout or on: idle, not gate: {effect.gate}, for this effect"
        )
    counted = _counted_in_gate(effect, profile)
    if counted == "all":
        return Treatment(
            "approximate",
            f"counted in the stated error of {effect.gate} as depolarizing noise",
            "the gate's includes lists leakage, so the export does not add the effect again",
        )
    number = "rate_per_s" if effect.on == "idle" else "prob"
    if getattr(effect, number) is None:
        return Treatment("omit", detail=f"the effect gives no {number}")
    if effect.on == "readout" and not readout:
        return Treatment("omit", detail=f"the {framework} export adds no readout error")
    if effect.on == "idle" and idle == "omit":
        return Treatment("omit", detail=f"the {framework} export models no idle time")
    how, detail = _HOW[effect.type]
    if effect.heralded:
        how = "a heralded " + how.removeprefix("an ")
    if counted == "some":
        detail = "; ".join(
            [
                detail,
                "loci whose record lists leakage in includes get no erasure, because"
                " their stated error already counts it",
            ]
        )
    level: Level = "exact" if effect.type == "erasure" else "approximate"
    if effect.on == "idle" and (note := _idle_note(effect, profile, idle)):
        level = "approximate"
        detail = "; ".join(filter(None, [detail, note]))
    return Treatment(level, how, detail)


def _idle_note(effect: Effect, profile: Profile, idle: Level) -> str:
    """Why an idle effect is approximated, or "" when the export applies it exactly."""
    if idle == "approximate":
        return "only where the export applies idle noise"
    table = profile.table
    qubits = effect.qubits or range(table.num_qubits)
    if all(table.qubit(q).relaxation_unknown for q in qubits):
        return ""
    return (
        "the erasure acts after the relaxation of each idle time; on the device both act"
        " at the same time"
    )


class Erasures:
    """The erasures that a profile's effects put on each gate, measurement and idle time."""

    def __init__(self, profile: Profile, *, readout: bool) -> None:
        self._gate: list[Effect] = []
        self._readout: list[Effect] = []
        self._idle: list[Effect] = []
        # An effect without its number still replaces a default; _pick then drops it.
        for effect in profile.effects:
            if effect.type not in ERASING or effect.gate in _NOT_GATES:
                continue
            if _counted_in_gate(effect, profile) == "all":
                continue
            if effect.gate is not None:
                self._gate.append(effect)
            elif effect.on == "readout" and readout:
                self._readout.append(effect)
            elif effect.on == "idle":
                self._idle.append(effect)

    @property
    def any(self) -> bool:
        return bool(self._gate or self._readout or self._idle)

    @property
    def heralded(self) -> bool:
        """True when some effect can add a herald."""
        effects = [*self._readout, *self._idle, *self._gate]
        return any(e.heralded and (e.prob or e.rate_per_s) for e in effects)

    @property
    def on_gates(self) -> bool:
        return bool(self._gate)

    @property
    def on_readout(self) -> bool:
        return bool(self._readout)

    @property
    def on_idle(self) -> bool:
        return bool(self._idle)

    def gate(self, name: str, wires: Sequence[int], noise: GateNoise) -> tuple[Erasure, ...]:
        """Erasures on each qubit of ``wires`` after one application of gate ``name``, whose
        noise comes from the record ``noise``. See :func:`gate_effects` for the rules."""
        return _erasures(self.gate_effects(name, wires, noise), _prob)

    def gate_effects(self, name: str, wires: Sequence[int], noise: GateNoise) -> list[Effect]:
        """The effects behind :meth:`gate`, also those without a probability."""
        return gate_effects(self._gate, name, wires, noise) if self._gate else []

    def readout(self, qubit: int) -> tuple[Erasure, ...]:
        """Erasures on ``qubit`` directly before one measurement of it."""
        return _pick(self._readout, lambda qubits: qubit in qubits, _prob)

    def idle(self, qubit: int, duration_ns: float) -> tuple[Erasure, ...]:
        """Erasures on ``qubit`` for ``duration_ns`` of idle time."""
        seconds = duration_ns * 1e-9
        return _pick(
            self._idle,
            lambda qubits: qubit in qubits,
            lambda e: -math.expm1(-(e.rate_per_s or 0.0) * seconds),
        )


def gate_effects(
    effects: Iterable[Effect], name: str, wires: Sequence[int], noise: GateNoise
) -> list[Effect]:
    """The effects of ``effects`` that act on one application of gate ``name`` on ``wires``.

    ``noise`` is the record that gives the gate its noise there. A gate without its own noise
    takes the noise of the typical native gate. For each effect type, such a gate takes the
    effects of that native gate too, unless an effect of the type names ``name``. An effect
    with ``qubits`` acts on that locus only, where it replaces the effects of its type without
    ``qubits``. A leakage effect on the gate that gives the noise does not act when the record
    lists leakage in ``includes``, because the stated error already counts it.
    """
    by_type: dict[str, list[Effect]] = {}
    for effect in effects:
        if effect.gate in (name, noise.gate):
            by_type.setdefault(effect.type, []).append(effect)
    locus = sorted(wires)
    acting: list[Effect] = []
    for kind, found in by_type.items():
        source = name if any(e.gate == name for e in found) else noise.gate
        if kind == "leakage" and source == noise.gate and _counts_leakage(noise.spec):
            continue
        named = [e for e in found if e.gate == source]
        acting += _chosen(named, lambda qubits: sorted(qubits) == locus)
    return acting


def _pick(
    effects: Iterable[Effect],
    applies: Callable[[tuple[int, ...]], bool],
    prob: Callable[[Effect], float],
) -> tuple[Erasure, ...]:
    """The erasures of ``effects`` at a place where ``applies`` tells if ``qubits`` match."""
    return _erasures(_chosen(effects, applies), prob)


def _chosen(effects: Iterable[Effect], applies: Callable[[tuple[int, ...]], bool]) -> list[Effect]:
    """An effect with matching ``qubits`` replaces an effect of the same type without
    ``qubits``."""
    chosen: dict[str, list[Effect]] = {}
    general: dict[str, list[Effect]] = {}
    for effect in effects:
        if effect.qubits:
            if applies(effect.qubits):
                chosen.setdefault(effect.type, []).append(effect)
        else:
            general.setdefault(effect.type, []).append(effect)
    for kind, found in general.items():
        chosen.setdefault(kind, found)
    return [effect for found in chosen.values() for effect in found]


def _erasures(effects: Iterable[Effect], prob: Callable[[Effect], float]) -> tuple[Erasure, ...]:
    return tuple(
        Erasure(effect.type, p, bool(effect.heralded))
        for effect in effects
        if (p := prob(effect)) > 0
    )


def _prob(effect: Effect) -> float:
    return effect.prob or 0.0


def _counted_in_gate(effect: Effect, profile: Profile) -> Literal["all", "some", "none"]:
    """Whether the records of the loci that ``effect`` acts on already count it in their error.

    A locus effect acts on its qubits in either order, and each order can resolve to its own
    record. An effect without ``qubits`` acts on every locus: the definition and each record.
    """
    if effect.type != "leakage" or effect.gate is None:
        return "none"
    from .table import GateNoise

    table = profile.table
    if effect.qubits:
        loci = set(permutations(effect.qubits))
    else:
        records = [r.qubits for r in profile.calibrations if r.gate == effect.gate]
        loci = {order for qubits in records for order in permutations(qubits)}
    counted = set() if effect.qubits else {_counts_leakage(profile.gates[effect.gate])}
    for locus in loci:
        found = table.gate(effect.gate, locus)
        if isinstance(found, GateNoise) and found.gate == effect.gate:
            counted.add(_counts_leakage(found.spec))
    return "all" if counted == {True} else "some" if True in counted else "none"


def _counts_leakage(spec: GateSpec) -> bool:
    """True when the stated error of ``spec`` contains leakage. A gate without a stated error
    takes the noise of another gate, so its ``includes`` describes no error."""
    return spec.metric is not None and "leakage" in (spec.includes or ())
