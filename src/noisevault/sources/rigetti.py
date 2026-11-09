"""Rigetti QCS instruction set architectures (ISAs) that a user saved.

Save the ISA with ``qcs_sdk.qpu.isa.get_instruction_set_architecture(id).json()``. Rigetti's
terms of service forbid redistribution, so profiles stay local. docs/data-sources.md gives the
mapping from each ISA field to the profile.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from .. import __version__
from ..errors import plural
from ..profile import Profile
from .saved import Saved, clean, read_saved, shape, utc_time

_SAVE = "save qcs_sdk.qpu.isa.get_instruction_set_architecture(id).json() and pass that file"
# ISA instruction -> (profile gate, its fidelity characteristic)
_TWO_QUBIT = {"CZ": ("cz", "fCZ"), "ISWAP": ("iswap", "fISWAP")}
# benchmark -> (method, measured), most preferred first
_RB = {
    "randomized_benchmark_1q": ("rb", "isolated"),
    "randomized_benchmark_simultaneous_1q": ("srb", "simultaneous"),
}
_COHERENCE = {"FreeInversionRecovery": ("T1", "t1_us"), "FreeInductionDecay": ("T2", "t2_us")}
# pyquil marks a node dead when none of these one-qubit instructions has a site on it
_LIVE = ("RX", "RZ", "MEASURE", "_")
# the number of nodes in each site of a known instruction, as pyquil requires
_ARITY = {
    "RX": 1,
    "RZ": 1,
    "MEASURE": 1,
    "RESET": 1,
    "I": 1,
    "CZ": 2,
    "ISWAP": 2,
    "XY": 2,
    "CPHASE": 2,
}
_DEAD = (0.5, 1.0)  # Rigetti's (value, error) of a dead site
_EPOCH_YEAR = 1970
_ASSUMPTIONS = {
    "rx": "Rigetti calibrates RX only at plus or minus pi/2 and pi, and does not state if fRB is"
    " per gate or per Clifford. NoiseVault reads 1 - fRB as the average gate infidelity of one"
    " rx at every angle",
    "two": "Rigetti does not state the fidelity metric. NoiseVault reads 1 - fidelity as the"
    " average gate infidelity",
}
_SHOWN = 5


class _Shape(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)


_NodeId = Annotated[int, Field(ge=0)]


class _Characteristic(_Shape):
    name: str
    value: float
    timestamp: str
    error: float | None = None
    node_ids: list[_NodeId] | None = None


class _Site(_Shape):
    node_ids: list[_NodeId]
    characteristics: list[_Characteristic]


class _Operation(_Shape):
    name: str
    sites: list[_Site]


class _Node(_Shape):
    node_id: _NodeId


class _Edge(_Shape):
    node_ids: Annotated[list[_NodeId], Field(min_length=2, max_length=2)]


class _Architecture(_Shape):
    family: str | None = None
    nodes: list[_Node]
    edges: list[_Edge]


class _Isa(_Shape):
    name: str
    architecture: _Architecture
    instructions: list[_Operation]
    benchmarks: list[_Operation]


@dataclass
class _Notes:
    notes: list[str]
    times: list[datetime] = field(default_factory=list)
    placeholders: list[str] = field(default_factory=list)
    dead_sites: list[str] = field(default_factory=list)
    broken: list[str] = field(default_factory=list)
    uncalibrated: list[str] = field(default_factory=list)
    zoneless: int = 0


def bundled_profiles() -> list[Profile]:
    """None, because Rigetti's terms forbid redistribution. Import your own saved file."""
    return []


def from_rigetti(
    path_or_dict: str | Path | Mapping[str, Any], *, device: str | None = None
) -> Profile:
    """A profile from a saved Rigetti QCS ISA (a JSON file path or the parsed dict).

    ``device`` names the profile. The default name is the ISA's ``name``, such as ``Ankaa-3``.
    """
    saved = read_saved(path_or_dict, save=_SAVE)
    shape(saved, _Isa, save=_SAVE)
    isa = saved.data
    log = _Notes([f"source_hash is over {saved.hashed}"])
    nodes, edges = _layout(isa, saved)
    instructions = {op["name"]: op for op in isa["instructions"]}
    _check_sites(isa, nodes, edges, saved)
    num_qubits = max(nodes) + 1
    live = {
        site["node_ids"][0]
        for name in _LIVE
        for site in instructions.get(name, {"sites": []})["sites"]
        if len(site["node_ids"]) == 1
    }
    gates: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    if "RZ" in instructions:
        gates["rz"] = {"virtual": True}
        records += _rz_records(instructions["RZ"], nodes, live)
    if "RX" in instructions:
        gates["rx"] = {"assumption": _ASSUMPTIONS["rx"]}
        records += _rx_records(isa, instructions["RX"], nodes, live, log, saved)
    for name, (gate, characteristic) in _TWO_QUBIT.items():
        if name in instructions:
            gates[gate] = {"qubits": 2, "assumption": _ASSUMPTIONS["two"]}
            records += _pair_records(instructions[name], gate, characteristic, edges, log, saved)
    qubits = _qubits(isa, instructions, nodes, live, num_qubits, log, saved)
    _note_unread(isa, log)
    name = isa["name"]
    family = isa["architecture"].get("family")
    return saved.origin.profile(
        {
            "noisevault": "1.0",
            "device": {
                "vendor": "rigetti",
                "name": device or name,
                "technology": "superconducting",
                "num_qubits": num_qubits,
                "processor": family if family not in (None, "None", "Full") else None,
                "calibrated_at": _calibrated_at(log),
            },
            "connectivity": {"edges": sorted(edges), "directed": False},
            "gates": gates,
            "qubits": qubits,
            "calibrations": records,
            "provenance": {
                "data_kind": "measured",
                "source_kind": "user_file",
                "source": f"Rigetti QCS instruction set architecture of {name}",
                "license": "Rigetti QCS terms of service (not an open license)",
                "attribution": "Rigetti Computing",
                "redistributable": "no",
                "source_hash": saved.sha256,
                "tool": f"noisevault {__version__}",
                "notes": _finished(log),
            },
        }
    )


# layout ------------------------------------------------------------------------------------


def _layout(isa: Mapping[str, Any], saved: Saved) -> tuple[set[int], set[tuple[int, int]]]:
    arch = isa["architecture"]
    nodes = {node["node_id"] for node in arch["nodes"]}
    if not nodes:
        raise saved.origin.refuse("architecture.nodes is empty, so the ISA has no qubit")
    edges = set()
    for i, edge in enumerate(arch["edges"]):
        a, b = edge["node_ids"]
        if a == b or not {a, b} <= nodes:
            raise saved.origin.refuse(
                f"architecture.edges[{i}] is {[a, b]}, not two different architecture nodes"
            )
        edges.add((min(a, b), max(a, b)))
    return nodes, edges


def _check_sites(
    isa: Mapping[str, Any], nodes: set[int], edges: set[tuple[int, int]], saved: Saved
) -> None:
    """Refuse a site of a known operation on a node or edge that the architecture lacks, a site
    of a known instruction with the wrong number of nodes, and a characteristic that names a
    node outside its site.
    """
    for kind in ("instructions", "benchmarks"):
        for op in isa[kind]:
            arity = _ARITY.get(op["name"]) if kind == "instructions" else None
            for site in op["sites"]:
                ids = site["node_ids"]
                if arity is not None and (len(ids) != arity or len(set(ids)) != arity):
                    raise saved.origin.refuse(
                        f"{kind} {op['name']} has a site on {ids}, not on {arity} different nodes"
                    )
                for item in site["characteristics"]:
                    named = item.get("node_ids")
                    if named is not None and not set(named) <= set(ids):
                        raise saved.origin.refuse(
                            f"{kind} {op['name']} site {ids} has a {item['name']} on nodes"
                            f" {named}, outside the site"
                        )
                missing = sorted(set(ids) - nodes)
                if missing:
                    raise saved.origin.refuse(
                        f"{kind} {op['name']} has a site on node {missing[0]}, which is not an"
                        " architecture node"
                    )
                if op["name"] in _TWO_QUBIT and _pair(ids) not in edges:
                    raise saved.origin.refuse(
                        f"instructions {op['name']} has a site on {ids}, which is not an"
                        " architecture edge"
                    )


def _pair(ids: Sequence[int]) -> tuple[int, int] | None:
    if len(ids) != 2 or ids[0] == ids[1]:
        return None
    return (min(ids), max(ids))


# values ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Reading:
    """One characteristic: where it is, and its value as the profile reads it."""

    where: str
    value: float
    stderr: float | None
    dead: bool


def _reading(characteristic: Mapping[str, Any], where: str, log: _Notes, saved: Saved) -> _Reading:
    zoneless: list[str] = []
    when = utc_time(
        characteristic["timestamp"], f"the timestamp of {where}", zoneless, saved.origin
    )
    log.zoneless += len(zoneless)
    if when.year > _EPOCH_YEAR:
        log.times.append(when)
    value, error = characteristic["value"], characteristic.get("error")
    dead = (value, error) == _DEAD
    stderr = error if error is not None and math.isfinite(error) and error >= 0 else None
    return _Reading(where, float(value), None if dead else stderr, dead)


def _fidelity_spec(reading: _Reading, arity: int, log: _Notes) -> dict[str, Any] | None:
    """The record fields for a fidelity, or None when it is a placeholder (unknown)."""
    if reading.dead:
        log.dead_sites.append(reading.where)
        return {"disabled": True}
    if not 0 < reading.value < 1:
        log.placeholders.append(f"{reading.where} ({reading.value})")
        return None
    if reading.value <= 1 / (2**arity + 1):
        log.broken.append(f"{reading.where} ({reading.value})")
        return {"disabled": True}
    spec: dict[str, Any] = {"avg_infidelity": clean(1 - reading.value)}
    if reading.stderr is not None:
        spec["stderr"] = reading.stderr
    return spec


def _per_node(
    op: Mapping[str, Any], characteristic: str, saved: Saved
) -> dict[int, Mapping[str, Any]]:
    """The first ``characteristic`` of ``op`` on each node.

    A characteristic names its node in its own ``node_ids``, or else its site has one node.
    """
    found: dict[int, Mapping[str, Any]] = {}
    for site in op["sites"]:
        for item in site["characteristics"]:
            if item["name"] != characteristic:
                continue
            ids = item.get("node_ids") or site["node_ids"]
            if len(ids) != 1:
                raise saved.origin.refuse(
                    f"{op['name']} has a {characteristic} on nodes {ids}, not on one node"
                )
            found.setdefault(ids[0], item)
    return found


# gates -------------------------------------------------------------------------------------


def _rx_records(
    isa: Mapping[str, Any],
    rx: Mapping[str, Any],
    nodes: set[int],
    live: set[int],
    log: _Notes,
    saved: Saved,
) -> list[dict[str, Any]]:
    sited = {site["node_ids"][0] for site in rx["sites"] if len(site["node_ids"]) == 1}
    benchmarks = {op["name"]: op for op in isa["benchmarks"]}
    candidates: dict[int, list[tuple[Mapping[str, Any], str, str, str]]] = {}
    for name, (method, measured) in _RB.items():
        if name in benchmarks:
            for node, item in _per_node(benchmarks[name], "fRB", saved).items():
                candidates.setdefault(node, []).append((item, name, method, measured))
    records = []
    for node in sorted(nodes & live):
        record: dict[str, Any] = {"gate": "rx", "qubits": [node]}
        if node not in sited:
            records.append(record | {"disabled": True})
            continue
        if node not in candidates:
            log.uncalibrated.append(f"rx on {node}")
            continue
        for item, name, method, measured in candidates[node]:
            reading = _reading(item, f"{name} fRB of node {node}", log, saved)
            spec = _fidelity_spec(reading, 1, log)
            if spec is not None:  # else a placeholder, so try the next benchmark
                if "avg_infidelity" in spec:
                    spec |= {"method": method, "measured": measured}
                records.append(record | spec)
                break
    return records


def _rz_records(rz: Mapping[str, Any], nodes: set[int], live: set[int]) -> list[dict[str, Any]]:
    """Disabled rz on each live node that has no RZ site, because pyquil allows RZ only there."""
    sited = {site["node_ids"][0] for site in rz["sites"]}
    return [
        {"gate": "rz", "qubits": [node], "disabled": True}
        for node in sorted((nodes & live) - sited)
    ]


def _pair_records(
    op: Mapping[str, Any],
    gate: str,
    characteristic: str,
    edges: set[tuple[int, int]],
    log: _Notes,
    saved: Saved,
) -> list[dict[str, Any]]:
    records = []
    sited: set[tuple[int, int]] = set()
    for site in op["sites"]:
        pair = _pair(site["node_ids"])
        assert pair is not None  # _check_sites() refused sites off the edges
        if pair in sited:
            continue
        sited.add(pair)
        record: dict[str, Any] = {"gate": gate, "qubits": list(pair)}
        items = [c for c in site["characteristics"] if c["name"] == characteristic]
        if not items:
            log.uncalibrated.append(f"{gate} on {pair[0]}-{pair[1]}")
            continue
        where = f"{characteristic} of {pair[0]}-{pair[1]}"
        spec = _fidelity_spec(_reading(items[0], where, log, saved), 2, log)
        if spec is None:
            continue
        if "avg_infidelity" in spec:
            spec["method"] = "vendor"
        records.append(record | spec)
    missing = sorted(edges - sited)
    if missing:
        records += [{"gate": gate, "qubits": list(pair), "disabled": True} for pair in missing]
        shown = [f"{a}-{b}" for a, b in missing]
        log.notes.append(
            f"{op['name']} has no site on the edges {_listed(shown)}, so NoiseVault disables"
            f" {gate} on them"
        )
    return records


# qubits ------------------------------------------------------------------------------------


def _qubits(
    isa: Mapping[str, Any],
    instructions: Mapping[str, Mapping[str, Any]],
    nodes: set[int],
    live: set[int],
    num_qubits: int,
    log: _Notes,
    saved: Saved,
) -> list[dict[str, Any]]:
    readout = _per_node(instructions["MEASURE"], "fRO", saved) if "MEASURE" in instructions else {}
    benchmarks = {op["name"]: op for op in isa["benchmarks"]}
    coherence = {
        key: _per_node(benchmarks[name], characteristic, saved)
        for name, (characteristic, key) in _COHERENCE.items()
        if name in benchmarks
    }
    records = []
    for index in range(num_qubits):
        record: dict[str, Any] = {"index": index, "label": str(index)}
        if index not in nodes or index not in live:
            records.append(record | {"disabled": True})
            continue
        for key, found in coherence.items():
            if index in found:
                record |= _time(found[index], key, index, log, saved)
        if "t2_us" in record:
            record["t2_kind"] = "ramsey"
        if index in readout:
            reading = _reading(readout[index], f"fRO of node {index}", log, saved)
            if reading.dead:
                log.dead_sites.append(reading.where)
            elif 0 < reading.value < 1:
                record["readout"] = {"error": clean(1 - reading.value)}
            else:
                log.placeholders.append(f"{reading.where} ({reading.value})")
        records.append(record)
    absent = sorted(set(range(num_qubits)) - nodes)
    dead = sorted(nodes - live)
    if absent:
        log.notes.append(f"indices {_ints(absent)} are not architecture nodes and are disabled")
    if dead:
        log.notes.append(
            f"nodes {_ints(dead)} have no site of {', '.join(_LIVE)}, so pyquil and NoiseVault"
            " treat them as dead and NoiseVault disables them"
        )
    return records


def _time(
    item: Mapping[str, Any], key: str, index: int, log: _Notes, saved: Saved
) -> dict[str, float]:
    reading = _reading(item, f"{item['name']} of node {index}", log, saved)
    if reading.dead:
        log.dead_sites.append(reading.where)
        return {}
    if 0 < reading.value < 1:
        return {key: clean(reading.value * 1e6)}
    log.placeholders.append(f"{reading.where} ({reading.value} s)")
    return {}


# notes -------------------------------------------------------------------------------------


def _note_unread(isa: Mapping[str, Any], log: _Notes) -> None:
    known = {*_TWO_QUBIT, *_LIVE}
    skipped = sorted({op["name"] for op in isa["instructions"]} - known)
    if skipped:
        reasons = {
            "I": "I is not a calibrated gate",
            "RESET": "the RESET fidelity fAR possibly includes readout error",
            "XY": "XY takes an angle and no registry gate matches it",
            "CPHASE": "CPHASE takes an angle and no registry gate matches it",
        }
        why = [reasons.get(name, f"{name} is not a known Rigetti instruction") for name in skipped]
        log.notes.append(f"NoiseVault does not read these instructions: {'; '.join(why)}")
    benchmarks = sorted({op["name"] for op in isa["benchmarks"]} - {*_RB, *_COHERENCE})
    if benchmarks:
        log.notes.append(f"NoiseVault does not read these benchmarks: {_listed(benchmarks)}")


def _calibrated_at(log: _Notes) -> datetime | None:
    if not log.times:
        log.notes.append(
            "no characteristic has a real timestamp (1970-01-01 is a placeholder), so the"
            " profile has no calibrated_at"
        )
        return None
    newest, oldest = max(log.times), min(log.times)
    log.notes.append(
        "calibrated_at is the newest timestamp of a characteristic that NoiseVault reads. The"
        f" oldest is {oldest.isoformat()}"
    )
    return newest


def _finished(log: _Notes) -> list[str]:
    notes = list(log.notes)
    if log.zoneless:
        notes.append(
            f"NoiseVault reads {plural(log.zoneless, 'timestamp')} with no time zone as UTC"
        )
    notes.append(
        "fRO is the mean of P(0|0) and P(1|1). NoiseVault stores 1 - fRO as one symmetric readout"
        " error, which is correct for the mean of the two error probabilities only"
    )
    notes.append(
        "NoiseVault prefers the isolated fRB (randomized_benchmark_1q) to the simultaneous one"
    )
    if log.dead_sites:
        notes.append(
            "value 0.5 with error 1.0 is Rigetti's marker for a dead site:"
            f" {_listed(log.dead_sites)}"
        )
    if log.placeholders:
        notes.append(
            f"NoiseVault reads these values as placeholders and leaves them unknown:"
            f" {_listed(log.placeholders)}"
        )
    if log.broken:
        notes.append(f"a fidelity at or below 1/(d+1) disables the gate: {_listed(log.broken)}")
    if log.uncalibrated:
        notes.append(f"no fidelity, so uncalibrated: {_listed(log.uncalibrated)}")
    return notes


def _ints(values: Sequence[int]) -> str:
    return _listed([str(v) for v in values])


def _listed(items: Sequence[str]) -> str:
    shown = ", ".join(items[:_SHOWN])
    return shown + (f" and {len(items) - _SHOWN} more" if len(items) > _SHOWN else "")
