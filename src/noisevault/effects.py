"""Leakage, atom loss and erasure as an erasure channel, shared by the Stim and Cirq exports.

An erasure replaces the state of one qubit with I/2, the same as a random Pauli I, X, Y or Z
with probability 1/4 each. docs/profile-format.md (section "Effects") gives the rules: where each
effect acts, which number it uses, and when the report calls it exact or approximated.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import permutations
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .profile import Effect, Profile

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
    ``readout`` is False when the export adds no readout error, and so no readout effects.
    """
    if effect.type not in ERASING:
        return Treatment("omit", detail=f"the {framework} export does not model {effect.type}")
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
    if effect.on == "idle" and idle == "approximate":
        level = "approximate"
        detail = "; ".join(filter(None, [detail, "only where the export applies idle noise"]))
    return Treatment(level, how, detail)


class Erasures:
    """The erasures that a profile's effects put on each gate, measurement and idle time."""

    def __init__(self, profile: Profile, *, readout: bool) -> None:
        self._gate: dict[str, list[Effect]] = {}
        self._readout: list[Effect] = []
        self._idle: list[Effect] = []
        # An effect without its number still replaces a default; _pick then drops it.
        for effect in profile.effects:
            if effect.type not in ERASING or effect.gate in _NOT_GATES:
                continue
            if _counted_in_gate(effect, profile) == "all":
                continue
            if effect.gate is not None:
                self._gate.setdefault(effect.gate, []).append(effect)
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
        effects = [*self._readout, *self._idle, *(e for v in self._gate.values() for e in v)]
        return any(e.heralded and (e.prob or e.rate_per_s) for e in effects)

    @property
    def on_readout(self) -> bool:
        return bool(self._readout)

    def gate(
        self, name: str, wires: tuple[int, ...], includes: Iterable[str] | None = ()
    ) -> tuple[Erasure, ...]:
        """Erasures on each qubit of ``wires`` after one application of gate ``name``.

        ``includes`` is the qualifier of the gate's record on ``wires``. When it lists leakage,
        the stated error already counts the leakage effects, so they add nothing.
        """
        effects = self._gate.get(name)
        if effects and "leakage" in (includes or ()):
            effects = [e for e in effects if e.type != "leakage"]
        if not effects:
            return ()
        locus = sorted(wires)
        return _pick(effects, lambda qubits: sorted(qubits) == locus, _prob)

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


def _pick(
    effects: Iterable[Effect],
    applies: Callable[[tuple[int, ...]], bool],
    prob: Callable[[Effect], float],
) -> tuple[Erasure, ...]:
    """The erasures of ``effects`` here: an effect with matching ``qubits`` replaces an effect
    of the same type without ``qubits``."""
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
    return tuple(
        Erasure(effect.type, p, bool(effect.heralded))
        for found in chosen.values()
        for effect in found
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
    counted = set() if effect.qubits else {"leakage" in (profile.gates[effect.gate].includes or ())}
    for locus in loci:
        found = table.gate(effect.gate, locus)
        if isinstance(found, GateNoise) and found.gate == effect.gate:
            counted.add("leakage" in (found.spec.includes or ()))
    return "all" if counted == {True} else "some" if True in counted else "none"
