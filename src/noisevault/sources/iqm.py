"""IQM Resonance quality metric sets that a user saved with iqm-client.

Save the set with ``IQMClient.get_quality_metric_set().model_dump_json()``. Optionally, also save
``IQMClient.get_dynamic_quantum_architecture().model_dump_json()`` of the same calibration set.
The architecture tells which qubits and pairs IQM calibrated, so the profile can disable the
others. IQM's terms of use are not public, so profiles stay local. docs/data-sources.md gives
the mapping from each IQM field to the profile.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from .. import __version__, units
from ..errors import SourceDataError
from ..profile import Profile
from .saved import Saved, clean, read_saved, shape, utc_time

_SAVE = "save IQMClient.get_quality_metric_set().model_dump_json() and pass that file"
_SAVE_DQA = (
    "save IQMClient.get_dynamic_quantum_architecture().model_dump_json() of the same calibration"
    " set and pass that file as architecture="
)
_QUBIT = re.compile(r"QB([1-9][0-9]*)")
_TIME_UNITS = ("s", "ms", "us", "ns")
_COHERENCE = {"t1_time": "t1", "t2_echo_time": "t2_echo", "t2_time": "t2"}
_READOUT_GATES = ("measure", "measure_fidelity")
_READOUT_QUANTITIES = ("fidelity", "error_0_to_1", "error_1_to_0")
# IQM gate -> (profile gate, the methods NoiseVault reads, most preferred first)
_GATES = {"prx": ("r", ("rb", "irb", "srb")), "cz": ("cz", ("irb", "rb", "srb"))}
_ASSUMPTIONS = {
    "r": "IQM does not state if its RB fidelity of prx is per gate or per Clifford. NoiseVault"
    " reads 1 - fidelity as the average gate infidelity of one prx",
    "cz": "NoiseVault reads 1 - IQM's fidelity of cz as the average gate infidelity of one cz."
    " The method says how IQM measured it, such as irb for interleaved RB",
}
_STDERR = (
    ". IQM does not say that its uncertainty is a standard error, and NoiseVault stores it as"
    " stderr"
)
_SHOWN = 5
_ARITY = {"prx": 1, "measure": 1, "measure_fidelity": 1, "cz": 2}


class _Shape(BaseModel):
    model_config = ConfigDict(strict=True)


class _Observation(_Shape):
    dut_field: str
    value: Any
    unit: str
    uncertainty: Any = None
    invalid: bool = False


class _MetricSet(_Shape):
    observations: list[_Observation]
    invalid: bool = False
    observation_set_id: str | None = None
    describes_id: str | None = None
    created_timestamp: str | None = None
    end_timestamp: str | None = None


class _Implementation(_Shape):
    loci: list[list[str]]


class _GateInfo(_Shape):
    implementations: dict[str, _Implementation]
    default_implementation: str
    override_default_implementation: dict[str, str] = {}


class _Architecture(_Shape):
    calibration_set_id: str | None = None
    qubits: list[str]
    gates: dict[str, _GateInfo]


@dataclass(frozen=True)
class _Value:
    """One observation that NoiseVault reads, in profile units."""

    field: str
    value: float
    stderr: float | None


Locus = tuple[int, ...]


@dataclass
class _Readings:
    """The observations of a metric set, by what they describe."""

    coherence: dict[tuple[int, str], _Value]
    # (gate, impl, qubit) -> quantity -> value
    readout: dict[tuple[str, str, int], dict[str, _Value]]
    # (gate, impl, locus) -> method -> value
    fidelity: dict[tuple[str, str, Locus], dict[str, _Value]]
    # (gate, impl, locus) -> duration in ns
    duration: dict[tuple[str, str, Locus], _Value]
    unread: set[str]
    invalid: list[str]
    repeated: list[str]
    not_qubits: set[str]


def bundled_profiles() -> list[Profile]:
    """None, because IQM's terms of use are not public. Import your own saved file."""
    return []


def from_iqm(
    metrics: str | Path | Mapping[str, Any],
    *,
    architecture: str | Path | Mapping[str, Any] | None = None,
    device: str | None = None,
) -> Profile:
    """A profile from a saved IQM quality metric set (a JSON file path or the parsed dict).

    ``architecture`` is the dynamic quantum architecture of the same calibration set. With it,
    the profile disables the qubits and pairs that IQM did not calibrate. ``device`` names the
    profile. The default name is the file name without its suffix.
    """
    saved = read_saved(metrics, save=_SAVE)
    shape(saved, _MetricSet, save=_SAVE)
    data = saved.data
    if data.get("invalid"):
        raise SourceDataError(
            f"{saved.source}: IQM marks this quality metric set invalid, so NoiseVault does not"
            " use it",
            hint="save the metrics of another calibration set",
        )
    origin = saved.origin
    notes = [f"source_hash is over {saved.hashed}"]
    dqa, arch = _architecture(architecture, data, saved)
    if arch is not None:
        notes.append(f"the dynamic quantum architecture came from {arch.source} ({arch.sha256})")
    readings = _read(data["observations"], saved)
    built = _Build(readings, dqa, notes)
    profile_fields = built.fields()
    calibrated_at = _calibrated_at(data, notes, saved)
    _note_readings(readings, notes)
    set_id = data.get("observation_set_id")
    extra: dict[str, Any] = {}
    if data.get("describes_id"):
        extra["calibration_set_id"] = data["describes_id"]
    if arch is not None:
        extra["architecture_hash"] = arch.sha256
    return origin.profile(
        {
            "noisevault": "1.0",
            "device": {
                "vendor": "iqm",
                "name": device or saved.stem or "iqm_device",
                "technology": "superconducting",
                "num_qubits": profile_fields.pop("num_qubits"),
                "calibrated_at": calibrated_at,
            },
            **profile_fields,
            "provenance": {
                "data_kind": "measured",
                "source_kind": "user_file",
                "source": "IQM Resonance quality metric set"
                + (f" {set_id}" if set_id else "")
                + (" with its dynamic quantum architecture" if arch is not None else ""),
                "license": "IQM Resonance terms of use (not an open license)",
                "attribution": "IQM",
                "redistributable": "unknown",
                "source_hash": saved.sha256,
                "tool": f"noisevault {__version__}",
                "notes": notes,
                "extra": extra,
            },
        }
    )


def _architecture(
    architecture: str | Path | Mapping[str, Any] | None, data: Mapping[str, Any], saved: Saved
) -> tuple[Mapping[str, Any] | None, Saved | None]:
    if architecture is None:
        return None, None
    arch = read_saved(architecture, save=_SAVE_DQA)
    shape(arch, _Architecture, save=_SAVE_DQA)
    for gate, info in arch.data["gates"].items():
        arity = _ARITY.get(gate)
        for impl, found in info["implementations"].items():
            for i, labels in enumerate(found["loci"]):
                if arity is not None and len(labels) != arity:
                    where = f"gates.{gate}.implementations.{impl}.loci[{i}]"
                    raise arch.origin.refuse(f"{where} is {labels}, not {arity} components")
    described, calibrated = data.get("describes_id"), arch.data.get("calibration_set_id")
    if described and calibrated and described != calibrated:
        raise SourceDataError(
            f"{saved.source} has the metrics of calibration set {described}, but {arch.source} is"
            f" the architecture of calibration set {calibrated}",
            hint=_SAVE_DQA,
        )
    return arch.data, arch


# reading -----------------------------------------------------------------------------------


def _read(observations: Sequence[Mapping[str, Any]], saved: Saved) -> _Readings:
    readings = _Readings({}, {}, {}, {}, set(), [], [], set())
    seen: set[str] = set()
    for obs in observations:
        name = obs["dut_field"]
        if obs.get("invalid"):
            readings.invalid.append(name)
            continue
        path = name.partition(":")[0]
        if path in seen:
            readings.repeated.append(name)
            continue
        seen.add(path)
        _read_one(path.split("."), obs, readings, saved)
    return readings


def _read_one(parts: list[str], obs: Mapping[str, Any], readings: _Readings, saved: Saved) -> None:
    match parts:
        case ["characterization", "model", component, quantity] if quantity in _COHERENCE:
            qubit = _qubit(component, readings)
            if qubit is not None:
                readings.coherence[(qubit, _COHERENCE[quantity])] = _time(obs, saved)
        case ["metrics", "ssro", gate, impl, component, quantity] if (
            gate in _READOUT_GATES and quantity in _READOUT_QUANTITIES
        ):
            qubit = _qubit(component, readings)
            if qubit is not None:
                quantities = readings.readout.setdefault((gate, impl, qubit), {})
                quantities[quantity] = _ratio(obs, saved)
        case ["metrics", method, gate, impl, locus_text, "fidelity"] if (
            gate in _GATES and method in _GATES[gate][1]
        ):
            locus = _locus(locus_text, readings)
            if locus is not None and len(locus) == (1 if gate == "prx" else 2):
                key = (gate, impl, tuple(sorted(locus)))
                methods = readings.fidelity.setdefault(key, {})
                if method in methods:
                    readings.repeated.append(obs["dut_field"])
                else:
                    methods[method] = _ratio(obs, saved)
        case ["gates", gate, impl, locus_text, "duration"]:
            locus = _locus(locus_text, readings)
            if locus is not None:
                readings.duration[(gate, impl, tuple(sorted(locus)))] = _time(obs, saved, "ns")
        case ["metrics", method, kind, *_]:
            readings.unread.add(f"metrics.{method}.{kind}")


def _qubit(component: str, readings: _Readings) -> int | None:
    match = _QUBIT.fullmatch(component)
    if match is None:
        readings.not_qubits.add(component)
        return None
    return int(match[1]) - 1


def _locus(text: str, readings: _Readings) -> Locus | None:
    qubits = [_qubit(component, readings) for component in text.split("__")]
    if any(q is None for q in qubits) or len(set(qubits)) != len(qubits):
        return None
    return tuple(q for q in qubits if q is not None)


def _number(obs: Mapping[str, Any], saved: Saved) -> float:
    value = obs["value"]
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise saved.origin.refuse(f"{obs['dut_field']} is {value!r}, not a finite number")
    return float(value)


def _stderr(obs: Mapping[str, Any]) -> float | None:
    error = obs.get("uncertainty")
    if isinstance(error, bool) or not isinstance(error, int | float):
        return None
    return float(error) if math.isfinite(error) and error >= 0 else None


def _time(obs: Mapping[str, Any], saved: Saved, to: str = "us") -> _Value:
    unit = obs["unit"]
    if unit not in _TIME_UNITS:
        raise saved.origin.refuse(
            f"{obs['dut_field']} has the unit {unit!r}, not one of {', '.join(_TIME_UNITS)}"
        )
    value = clean(units.convert(_number(obs, saved), unit, to))
    return _Value(obs["dut_field"], value, None)


def _ratio(obs: Mapping[str, Any], saved: Saved) -> _Value:
    if obs["unit"] != "":
        raise saved.origin.refuse(
            f"{obs['dut_field']} has the unit {obs['unit']!r}, not '' (no unit)"
        )
    return _Value(obs["dut_field"], _number(obs, saved), _stderr(obs))


# building ----------------------------------------------------------------------------------


class _Build:
    """The physics fields of the profile from the readings and the optional architecture."""

    def __init__(
        self, readings: _Readings, dqa: Mapping[str, Any] | None, notes: list[str]
    ) -> None:
        self.readings, self.dqa, self.notes = readings, dqa, notes
        self.choices: list[str] = []
        self.unselected: list[str] = []
        self.symmetric: list[str] = []
        self.placeholders: list[str] = []
        self.broken: list[str] = []

    def fields(self) -> dict[str, Any]:
        pairs = self._pairs()
        num_qubits = self._num_qubits(pairs)
        qubits = [self._qubit(i) for i in range(num_qubits)]
        records = [*self._gate_records("prx"), *self._gate_records("cz", pairs)]
        gates: dict[str, Any] = {
            "rz": {"virtual": True},
            "r": {"assumption": _ASSUMPTIONS["r"] + _STDERR},
        }
        if pairs or any(r["gate"] == "cz" for r in records):
            gates["cz"] = {"qubits": 2, "assumption": _ASSUMPTIONS["cz"] + _STDERR}
        self._note_choices()
        return {
            "num_qubits": num_qubits,
            "connectivity": {"edges": sorted(pairs), "directed": False},
            "gates": gates,
            "qubits": qubits,
            "calibrations": records,
        }

    # layout

    def _dqa_loci(self, gate: str) -> set[Locus]:
        if self.dqa is None or gate not in self.dqa["gates"]:
            return set()
        loci = set()
        for impl in self.dqa["gates"][gate]["implementations"].values():
            for labels in impl["loci"]:
                locus = _locus("__".join(labels), self.readings)
                if locus is not None:
                    loci.add(tuple(sorted(locus)))
        return loci

    def _pairs(self) -> set[Locus]:
        if self.dqa is not None:
            return self._dqa_loci("cz")
        return {key[2] for key in self.readings.fidelity if key[0] == "cz"}

    def _dqa_qubits(self) -> set[int] | None:
        if self.dqa is None:
            return None
        named = (_qubit(label, self.readings) for label in self.dqa["qubits"])
        return {q for q in named if q is not None}

    def _num_qubits(self, pairs: set[Locus]) -> int:
        r = self.readings
        named: set[int] = {
            *(q for q, _ in r.coherence),
            *(key[2] for key in r.readout),
            *(q for key in [*r.fidelity, *r.duration] for q in key[2]),
            *(q for pair in pairs for q in pair),
            *(self._dqa_qubits() or ()),
            *(q for locus in self._dqa_loci("prx") for q in locus),
        }
        if not named:
            raise SourceDataError("the quality metric set names no qubit QB1, QB2, ...", hint=_SAVE)
        self.named = named
        unnamed = sorted(set(range(max(named) + 1)) - named)
        if unnamed:
            self.notes.append(
                f"no observation names {_labels(unnamed)}, so NoiseVault disables those qubits"
            )
        return max(named) + 1

    # qubits

    def _qubit(self, index: int) -> dict[str, Any]:
        record: dict[str, Any] = {"index": index, "label": f"QB{index + 1}"}
        calibrated = self._dqa_qubits()
        if index not in self.named:
            return record | {"disabled": True}
        if calibrated is not None and index not in calibrated:
            self.notes.append(
                f"QB{index + 1} is not in the architecture's qubits, so NoiseVault disables it"
            )
            record["disabled"] = True
        record |= self._coherence(index)
        readout = self._readout(index)
        if readout is not None:
            record["readout"] = readout
        return record

    def _coherence(self, index: int) -> dict[str, Any]:
        found: dict[str, Any] = {}
        coherence = self.readings.coherence
        t1 = coherence.get((index, "t1"))
        if t1 is not None:
            found |= self._positive("t1_us", t1)
        for kind, t2_kind in (("t2_echo", "echo"), ("t2", "ramsey")):
            t2 = coherence.get((index, kind))
            if t2 is not None:
                positive = self._positive("t2_us", t2)
                if positive:
                    return found | positive | {"t2_kind": t2_kind}
        return found

    def _positive(self, key: str, value: _Value) -> dict[str, float]:
        if value.value > 0:
            return {key: value.value}
        self.placeholders.append(f"{value.field} ({value.value})")
        return {}

    def _readout(self, index: int) -> dict[str, Any] | None:
        options = {(g, i): v for (g, i, q), v in self.readings.readout.items() if q == index}
        if not options:
            return None
        labels = (f"QB{index + 1}",)
        chosen = self._choose(options, _READOUT_GATES, "measure", (index,))
        if chosen is None:
            return None
        gate, impl = chosen
        quantities = options[(gate, impl)]
        readout: dict[str, Any] = {}
        e01, e10 = quantities.get("error_0_to_1"), quantities.get("error_1_to_0")
        fidelity = quantities.get("fidelity")
        if e01 is not None and e10 is not None:
            readout = {"p1_given_0": e01.value, "p0_given_1": e10.value}
        elif fidelity is not None and self._usable(fidelity):
            readout = {"error": clean(1 - fidelity.value)}
            self.symmetric.append(labels[0])
        names = (gate, *(g for g in _READOUT_GATES if g != gate))
        durations = (self.readings.duration.get((g, impl, (index,))) for g in names)
        duration = next((d for d in durations if d is not None), None)
        if readout and duration is not None:
            readout["duration_ns"] = duration.value
        return readout or None

    # gates

    def _gate_records(
        self, iqm_gate: str, pairs: AbstractSet[Locus] = frozenset()
    ) -> list[dict[str, Any]]:
        gate, methods = _GATES[iqm_gate]
        arity = _ARITY[iqm_gate]
        # locus -> implementation -> method -> fidelity; a duration alone gives an empty dict
        by_locus: dict[Locus, dict[str, dict[str, _Value]]] = {}
        for (g, impl, locus), found in self.readings.fidelity.items():
            if g == iqm_gate:
                by_locus.setdefault(locus, {})[impl] = found
        for g, impl, locus in self.readings.duration:
            if g == iqm_gate and len(locus) == arity:
                by_locus.setdefault(locus, {}).setdefault(impl, {})
        calibrated = self._dqa_loci(iqm_gate) if self.dqa is not None else None
        records = []
        with_fidelity: set[Locus] = set()
        for locus in sorted(by_locus):
            labels = tuple(f"QB{q + 1}" for q in locus)
            record: dict[str, Any] = {"gate": gate, "qubits": list(locus)}
            if calibrated is not None and locus not in calibrated:
                records.append(record | {"disabled": True})
                self.notes.append(
                    f"{iqm_gate} on {'__'.join(labels)} has data but no locus in the"
                    " architecture, so NoiseVault disables it"
                )
                continue
            options = {(iqm_gate, impl): found for impl, found in by_locus[locus].items()}
            chosen = self._choose(options, (iqm_gate,), iqm_gate, locus)
            if chosen is None:
                continue
            impl = chosen[1]
            duration = self.readings.duration.get((iqm_gate, impl, locus))
            if duration is not None:
                record["duration_ns"] = duration.value
            found = by_locus[locus][impl]
            method = next((m for m in methods if m in found), None)
            spec = None if method is None else self._spec(found[method], method, arity)
            if spec is not None:
                with_fidelity.add(locus)
                if spec.get("disabled"):
                    record = {"gate": gate, "qubits": list(locus)}
                record |= spec
            if len(record) > 2:
                records.append(record)
        if iqm_gate == "cz":
            missing = sorted(pairs - with_fidelity)
            if missing:
                shown = [f"QB{a + 1}__QB{b + 1}" for a, b in missing]
                self.notes.append(f"cz has no fidelity on {_listed(shown)}, so it is uncalibrated")
        return records

    def _spec(self, value: _Value, method: str, arity: int) -> dict[str, Any] | None:
        """The record fields of a fidelity, or None for a placeholder."""
        if not self._usable(value):
            return None
        if value.value <= 1 / (2**arity + 1):
            self.broken.append(f"{value.field} ({value.value})")
            return {"disabled": True}
        spec: dict[str, Any] = {"avg_infidelity": clean(1 - value.value), "method": method}
        if value.stderr is not None:
            spec["stderr"] = value.stderr
        return spec

    def _usable(self, fidelity: _Value) -> bool:
        """False for IQM's placeholder of an element it did not characterize."""
        if 0 < fidelity.value < 1:
            return True
        self.placeholders.append(f"{fidelity.field} ({fidelity.value})")
        return False

    def _selected(self, gate: str, labels: tuple[str, ...]) -> str | None:
        """The implementation that the architecture selects for ``gate`` on ``labels``."""
        if self.dqa is None or gate not in self.dqa["gates"]:
            return None
        info = self.dqa["gates"][gate]
        overrides = info.get("override_default_implementation") or {}
        for key in (",".join(labels), ",".join(reversed(labels))):
            if key in overrides:
                return str(overrides[key])
        return str(info["default_implementation"])

    def _choose(
        self,
        options: Mapping[tuple[str, str], Any],
        gates: Sequence[str],
        dqa_gate: str,
        locus: Locus,
    ) -> tuple[str, str] | None:
        """The (gate, implementation) of ``options`` to read.

        With an architecture, only the implementation that it selects, and only when that
        implementation lists the locus, because data of another implementation describes a
        different pulse. Else the first in alphabetical order.
        """
        labels = tuple(f"QB{q + 1}" for q in locus)
        where = f"{dqa_gate} on {'__'.join(labels)}"
        selected = self._selected(dqa_gate, labels)
        if selected is not None:
            found = [(gate, selected) for gate in gates if (gate, selected) in options]
            if not found or locus not in self._impl_loci(dqa_gate, selected):
                self.unselected.append(f"{where} ({selected})")
                return None
            return found[0]
        chosen = sorted(options)[0]
        if len(options) > 1:
            self.choices.append(f"{where} ({'.'.join(chosen)})")
        return chosen

    def _impl_loci(self, gate: str, impl: str) -> set[Locus]:
        assert self.dqa is not None
        found = self.dqa["gates"][gate]["implementations"].get(impl)
        loci = (
            _locus("__".join(labels), self.readings) for labels in (found or {}).get("loci", ())
        )
        return {tuple(sorted(locus)) for locus in loci if locus is not None}

    def _note_choices(self) -> None:
        if self.choices:
            self.notes.append(
                "with no architecture to select an implementation, NoiseVault used the first one"
                f" in alphabetical order for {_listed(self.choices)}"
            )
        if self.unselected:
            self.notes.append(
                "the implementation that the architecture selects has no data or does not list"
                f" the locus, so these stay unknown: {_listed(self.unselected)}"
            )
        if self.symmetric:
            self.notes.append(
                "IQM gives only a readout fidelity for these qubits. NoiseVault stores 1 -"
                " fidelity as one symmetric readout error, which is correct for the mean of the"
                f" two error probabilities only: {_listed(self.symmetric)}"
            )
        if self.placeholders:
            self.notes.append(
                "NoiseVault reads these values as placeholders for elements that IQM did not"
                f" characterize, and leaves them unknown: {_listed(self.placeholders)}"
            )
        if self.broken:
            self.notes.append(
                "a fidelity at or below the physical bound 1/(d+1) disables the gate:"
                f" {_listed(self.broken)}"
            )


def _calibrated_at(data: Mapping[str, Any], notes: list[str], saved: Saved) -> Any:
    for key in ("end_timestamp", "created_timestamp"):
        value = data.get(key)
        if value is not None:
            notes.append(f"calibrated_at is the {key} of the quality metric set")
            return utc_time(value, key, notes, saved.origin)
    return None


def _note_readings(readings: _Readings, notes: list[str]) -> None:
    if readings.invalid:
        notes.append(f"IQM marks these observations invalid: {_listed(readings.invalid)}")
    if readings.repeated:
        notes.append(
            "NoiseVault kept the first of observations that differ only in the suffix, and"
            f" skipped {_listed(readings.repeated)}"
        )
    if readings.unread:
        notes.append(f"NoiseVault does not read {_listed(sorted(readings.unread))}")
    if readings.not_qubits:
        notes.append(
            "the profile format has no computational resonators, so NoiseVault does not read"
            f" the loci with {_listed(sorted(readings.not_qubits))}"
        )
    notes.append(
        "T2 is the echo T2 (t2_echo_time) where IQM gives one, else the Ramsey T2 (t2_time)"
    )


def _labels(indices: Sequence[int]) -> str:
    return _listed([f"QB{i + 1}" for i in indices])


def _listed(items: Sequence[str]) -> str:
    shown = ", ".join(items[:_SHOWN])
    return shown + (f" and {len(items) - _SHOWN} more" if len(items) > _SHOWN else "")
