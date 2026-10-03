"""The report every framework export carries as ``.report``.

Each framework export fills one Report for each exported object. The Report adds events while
the framework processes circuits. The same helpers decide how the Report records clamps,
effects and repeated warnings, so every framework reports the same way.
"""

from __future__ import annotations

import sys
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .errors import LociText, NoiseApproximationWarning, UnsupportedEffect, qubit_loci
from .profile import unmodeled_note

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable
    from types import FrameType

    from .channels import GateChannels
    from .profile import Effect, GateSpec, Profile

HONESTY = "Calibration-derived models approximate the hardware. They are not a digital twin."
_INCLUDES = {
    "1q_dressing": (
        "single-qubit gate error",
        "explicit single-qubit gates in the circuit add their own error on top",
    ),
    "leakage": ("leakage", "applied as depolarizing noise, so no population leaves the qubit"),
    "spam": (
        "state preparation and measurement error",
        "readout and preparation noise, where applied, add their own error on top",
    ),
}
_MODELS = {"reference": ("the reference simulator", "leave the effect out")}
_EVENTS = {  # event -> how summary() states one key's count
    "typical_noise_used": "{key} took the typical native gate's noise {times}",
    "reversed_record_used": (
        "{key} took the calibration recorded for the opposite qubit order {times}"
    ),
    "circuit_channel_kept": (
        "the export kept the circuit's own {key} as written, with no noise added, {times}"
    ),
}


@dataclass(frozen=True)
class Approximation:
    what: str
    how: str
    detail: str = ""


@dataclass(frozen=True)
class Clamp:
    gate: str
    qubits: tuple[int, ...]
    requested: float
    achieved: float


@dataclass
class Report:
    profile_id: str
    fingerprint: str
    framework: str
    framework_version: str | None
    noisevault_version: str
    unmodeled_error: LociText | None = None
    options: dict[str, Any] = field(default_factory=dict)
    exact: list[str] = field(default_factory=list)
    approximated: list[Approximation] = field(default_factory=list)
    omitted: list[LociText] = field(default_factory=list)
    unknown: list[LociText] = field(default_factory=list)
    clamped: list[Clamp] = field(default_factory=list)
    events: dict[str, Counter[str]] = field(default_factory=dict)
    _warned: set[str] = field(default_factory=set, repr=False, compare=False)

    @classmethod
    def start(
        cls, profile: Profile, framework: str, framework_version: str | None, **options: Any
    ) -> Report:
        from . import __version__

        note = unmodeled_note(profile)
        return cls(
            profile_id=profile.id,
            fingerprint=profile.fingerprint,
            framework=framework,
            framework_version=framework_version,
            noisevault_version=__version__,
            unmodeled_error=LociText("; ").join(note) if note else None,
            options=options,
        )

    # recording ------------------------------------------------------------------------------

    def mark_exact(self, what: str) -> None:
        _append_new(self.exact, what)

    def approximate(self, what: str, how: str, detail: str = "") -> None:
        _append_new(self.approximated, Approximation(what, how, detail))

    def omit(self, what: str | LociText) -> None:
        _append_new(self.omitted, LociText(what))

    def mark_unknown(self, what: str | LociText) -> None:
        _append_new(self.unknown, LociText(what))

    def count(self, event: str, key: str, n: int = 1) -> None:
        self.events.setdefault(event, Counter())[key] += n

    def record_channels(self, built: GateChannels) -> None:
        """Record the bookkeeping of one gate's channels: clamps, qualifiers, reversed records."""
        gate = built.gate
        if built.inexact:
            if not any(c.gate == gate.gate and c.qubits == gate.qubits for c in self.clamped):
                self.clamped.append(
                    Clamp(gate.gate, gate.qubits, built.requested, built.achieved)  # type: ignore[arg-type]
                )
        for q in built.t2_clamped:
            self.record_t2_clamp(q)
        self._record_qualifiers(gate.gate, gate.spec)
        if gate.origin == "reversed_record":
            self.count("reversed_record_used", gate.gate)

    def record_t2_clamp(self, qubit: int) -> None:
        self.approximate(f"T2 of qubit {qubit}", "clamped to 2*T1", "the stated T2 exceeds 2*T1")

    def _record_qualifiers(self, name: str, spec: GateSpec) -> None:
        """Qualifiers that make the stated number differ from the error of the gate alone."""
        what = f"{name} error"
        if spec.scope == "cycle":
            self.approximate(
                what,
                "a per-cycle error applied to each gate",
                "the stated error also counts the surrounding layer",
            )
        for item in spec.includes or ():
            name, effect = _INCLUDES[item]
            self.approximate(what, f"the stated error already includes {name}", effect)
        if spec.statistic in ("median", "mean"):
            self.approximate(what, f"a device {spec.statistic} applied to every locus")
        if spec.assumption:
            self.approximate(what, "read under an importer assumption", spec.assumption)

    def record_effects(self, effects: Iterable[Effect]) -> None:
        """Omit or refuse each effect, because no export and no reference simulator models
        effects in this release."""
        model, fix = _MODELS.get(
            self.framework, (f"{self.framework} export", "export without the effect")
        )
        for effect in effects:
            target = effect.gate or effect.on
            if effect.allow != "omit":
                raise UnsupportedEffect(
                    f"effect {effect.type} on {target} asks for allow={effect.allow!r}, but"
                    f" {model} does not model effects yet",
                    hint=f"set allow to 'omit' to {fix}",
                )
            self.omit(f"effect {effect.type} on {target}")

    def warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            # every export's framework label is also its top-level package name
            warn_from_caller(message, NoiseApproximationWarning, packages=(self.framework,))

    # output ---------------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        unmodeled = {"unmodeled_error": self.unmodeled_error.full} if self.unmodeled_error else {}
        return {
            "profile_id": self.profile_id,
            "fingerprint": self.fingerprint,
            "framework": self.framework,
            "framework_version": self.framework_version,
            "noisevault_version": self.noisevault_version,
            **unmodeled,
            "options": _jsonable(self.options),
            "exact": list(self.exact),
            "approximated": [a.__dict__.copy() for a in self.approximated],
            "omitted": [what.full for what in self.omitted],
            "unknown": [what.full for what in self.unknown],
            "clamped": [
                {
                    "gate": c.gate,
                    "qubits": list(c.qubits),
                    "requested": c.requested,
                    "achieved": c.achieved,
                }
                for c in self.clamped
            ],
            "events": {name: dict(counts) for name, counts in self.events.items()},
        }

    def summary(self) -> str:
        version = f" {self.framework_version}" if self.framework_version else ""
        lines = [
            f"NoiseVault {self.noisevault_version} -> {self.framework}{version}:"
            f" {self.profile_id} (nv:{self.fingerprint[:12]})"
        ]
        if self.unmodeled_error:
            lines.append(f"unmodeled error: {self.unmodeled_error.short}")
        if self.options:
            lines.append("options: " + ", ".join(f"{k}={v!r}" for k, v in self.options.items()))
        if self.exact:
            lines.append("exact: " + ", ".join(self.exact))
        for a in self.approximated:
            detail = f" ({a.detail})" if a.detail else ""
            lines.append(f"approximated: {a.what}: {a.how}{detail}")
        if self.omitted:
            lines.append("omitted: " + ", ".join(what.short for what in self.omitted))
        if self.unknown:
            lines.append(
                "unknown (no noise applied): " + ", ".join(what.short for what in self.unknown)
            )
        noisier = [c for c in self.clamped if c.achieved > c.requested]
        quieter = [c for c in self.clamped if c.achieved < c.requested]
        if noisier:
            lines.append(
                f"clamped: {_gates(len(noisier))} noisier than stated because relaxation"
                f" alone exceeds the stated error. The largest is {_worst(noisier)}"
            )
        if quieter:
            lines.append(
                f"clamped: {_gates(len(quieter))} less noisy than stated because relaxation plus"
                f" the strongest depolarizing noise stays below the stated error. The largest"
                f" is {_worst(quieter)}"
            )
        if self.events:
            lines.append(
                "used: "
                + "; ".join(
                    _count_sentence(name, key, n)
                    for name, counts in self.events.items()
                    for key, n in counts.most_common()
                )
            )
        lines.append(HONESTY)
        return "\n".join(lines)


def warn_from_caller(message: str, category: type[Warning], packages: Collection[str] = ()) -> None:
    """Warn at the first calling frame outside NoiseVault and ``packages`` (top-level names).

    An export runs inside the framework's own calls (``with_noise``, ``qml.add_noise``), so a
    fixed stacklevel points at framework code. The user can act only on their own line.
    """
    skipped = {"noisevault", *packages}
    frame, level = sys._getframe(1), 2
    while frame.f_back is not None and _package(frame) in skipped:
        frame, level = frame.f_back, level + 1
    warnings.warn(message, category, stacklevel=level)


def _package(frame: FrameType) -> str:
    return frame.f_globals.get("__name__", "").partition(".")[0]


def _gates(n: int) -> str:
    return "1 gate" if n == 1 else f"{n} gates"


def _worst(clamps: list[Clamp]) -> str:
    c = max(clamps, key=lambda c: abs(c.achieved - c.requested))
    return f"{c.gate} on {qubit_loci(c.qubits)}, {c.requested:.3g} -> {c.achieved:.3g}"


def _count_sentence(event: str, key: str, n: int) -> str:
    template = _EVENTS.get(event, event.replace("_", " ") + ": {key} {times}")
    return template.format(key=key, times="once" if n == 1 else f"{n} times")


def _append_new(items: list, item: Any) -> None:
    if item not in items:
        items.append(item)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        if len({str(k) for k in value}) == len(value):
            return {str(k): _jsonable(v) for k, v in value.items()}
        # Keys equal as strings, such as wire 0 and wire "0", would merge into one entry.
        return [[_jsonable(k), _jsonable(v)] for k, v in value.items()]
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return repr(value)
