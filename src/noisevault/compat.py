from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any, NoReturn

from pydantic import TypeAdapter, ValidationError

from . import gates, metrics
from .errors import NoiseVaultError

_IBM_PROVIDERS = {"ibm", "qiskit_fake"}
_KINDS = {  # 0.1 provider -> (data_kind, source_kind)
    "qiskit_fake": ("measured", "package_snapshot"),
    "ibm": ("measured", "account_api"),
    "demo": ("hypothetical", "hand_written"),
}
# 0.1 raw_hash values (canonical JSON of configuration + properties) checked against the snapshot
# that qiskit-ibm-runtime ships (FakeManilaV2 in 0.49.0). Any other qiskit_fake file can hold
# refreshed IBM service data.
_PACKAGE_SNAPSHOTS = {"sha256:f79216df928d9dba24b98d2f65f3bbf4b918c46e61e7a6455f0c8218ae4179a0"}
_UNVERIFIED_FAKE = (
    "The 0.1 file did not prove that its data was the snapshot qiskit-ibm-runtime ships. A"
    " refreshed fake holds IBM Quantum service data, so NoiseVault claims no license."
)


def pennylane_mid_measure() -> type:
    """PennyLane's mid-circuit measurement class.

    PennyLane 0.45 moved ``pennylane.measurements.MidMeasureMP`` to ``pennylane.ops.MidMeasure``
    and keeps the old name as an alias. PennyLane 0.43, the floor, has only the old name.
    """
    try:
        from pennylane.ops import MidMeasure  # type: ignore[import-untyped]
    except ImportError:
        from pennylane.measurements import (  # type: ignore[import-untyped]
            MidMeasureMP as MidMeasure,
        )
    return MidMeasure


def ibm_sampler() -> type:
    """The server-side SamplerV2 of qiskit-ibm-runtime, which ``scripts/run_on_ibm.py`` uses.

    qiskit-ibm-runtime 0.52 moved it to ``qiskit_ibm_runtime.server_side``. The top-level
    ``SamplerV2`` became a client-side sampler with other pre- and post-processing.
    """
    try:
        from qiskit_ibm_runtime.server_side import (  # type: ignore[import-untyped]
            SamplerV2,
        )
    except ImportError:
        from qiskit_ibm_runtime import SamplerV2  # type: ignore[import-untyped]
    return SamplerV2


def is_v01(data: Any) -> bool:
    return isinstance(data, dict) and data.get("schema_version") == "0.1"


# Fields that the upgrade reads, and the JSON type of each. None means any type.
_FIELDS: dict[str, type | None] = {
    "provider": str,
    "backend_name": str,
    "num_qubits": None,
    "captured_at": str,
    "basis_gates": list,
    "coupling_map": list,
    "gates": list,
    "qubits": list,
    "provenance": dict,
}
_JSON_TYPES = {dict: "an object", list: "a list", str: "a string"}


def upgrade_v01(old: dict[str, Any]) -> dict[str, Any]:
    old = _parse(old)
    provider = old["provider"]
    by_name: dict[str, list[dict[str, Any]]] = {}
    for gate in old["gates"]:
        by_name.setdefault(gate["name"], []).append(gate)

    definitions: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for name in dict.fromkeys([*old["basis_gates"], *by_name]):
        entries = by_name.get(name, [])
        converted = [_record(e) for e in entries]
        live = [e for e, r in zip(entries, converted, strict=True) if not r.get("disabled")]
        if name == "rz" and live and all(_explicitly_free(e) for e in live):
            definitions[name] = {"virtual": True}
            records += [
                {"gate": name, "qubits": r["qubits"], "disabled": True}
                for r in converted
                if r.get("disabled")
            ]
            continue
        arity = len(entries[0]["qubits"]) if entries else None
        definitions[name] = {} if gates.lookup(name) else {"qubits": arity or 1}
        if provider in _IBM_PROVIDERS and any(e.get("error") is not None for e in entries):
            definitions[name]["method"] = "rb"
        records += converted

    data_kind, source_kind = _KINDS.get(provider, ("unknown", "other"))
    prov = old["provenance"]
    licensing = _licensing(provider, prov.get("raw_hash"))
    gate_metadata, dropped = _gate_metadata(old["gates"])
    notes = [
        *prov.get("notes", []),
        "Upgraded from NoiseVault format 0.1.",
        *_coherence_notes(old["qubits"]),
        *dropped,
    ]
    if provider == "qiskit_fake" and licensing["redistributable"] != "yes":
        source_kind = "other"
        notes.append(_UNVERIFIED_FAKE)
    extra = {"migrated_from": "0.1"}
    if prov.get("harvester_version"):
        extra["v01_harvester_version"] = prov["harvester_version"]
    extensions = dict(old.get("extensions") or {})
    frequencies = {
        str(q["index"]): q["frequency_ghz"]
        for q in old["qubits"]
        if q.get("frequency_ghz") is not None
    }
    if frequencies:
        extensions["v01_qubit_frequency_ghz"] = frequencies
    metadata = {str(q["index"]): q["metadata"] for q in old["qubits"] if q.get("metadata")}
    if metadata:
        extensions["v01_qubit_metadata"] = metadata
    if gate_metadata:
        extensions["v01_gate_metadata"] = gate_metadata

    return {
        "noisevault": "1.0",
        "device": {
            "name": _device_name(provider, old["backend_name"]),
            "vendor": "ibm" if provider in _IBM_PROVIDERS else provider,
            "technology": "superconducting",
            "num_qubits": old["num_qubits"],
            "calibrated_at": prov.get("source_timestamp"),
        },
        "connectivity": {"edges": [list(e) for e in old["coupling_map"]], "directed": True},
        "gates": definitions,
        "idle": {"t2_kind": "echo"} if provider in _IBM_PROVIDERS else None,
        "qubits": [_qubit(q) for q in old["qubits"]],
        "calibrations": records,
        "provenance": {
            "data_kind": data_kind,
            "source_kind": source_kind,
            "source": prov.get("source"),
            "source_url": prov.get("source_url"),
            "source_hash": prov.get("raw_hash"),
            "retrieved_at": old["captured_at"],
            "tool": f"noisevault {prov.get('harvester_version', '0.1')}",
            "notes": notes,
            "extra": extra,
            **licensing,
        },
        "extensions": extensions,
    }


class _InvalidV01(NoiseVaultError, ValueError): ...


def _fail(problem: str) -> NoReturn:
    raise _InvalidV01(
        f"not a valid NoiseVault 0.1 file: {problem}",
        hint="fix that field or pull the device again",
    )


def _parse(old: dict[str, Any]) -> dict[str, Any]:
    for key, kind in _FIELDS.items():
        if key not in old:
            _fail(f"{key} is missing")
        if kind is not None and not isinstance(old[key], kind):
            _fail(f"{key} should be {_JSON_TYPES[kind]}, not {_json_type(old[key])}")
    if not _aware_time(old["captured_at"]):
        _fail(
            "captured_at should be an ISO 8601 time with a timezone,"
            f" not {_shown(old['captured_at'])}"
        )
    gate_entries = []
    for i, gate in enumerate(old["gates"]):
        if not (
            isinstance(gate, dict)
            and isinstance(gate.get("name"), str)
            and isinstance(gate.get("qubits"), list)
        ):
            _fail(f"gates[{i}] needs a name and a list of qubits")
        error = gate.get("error")
        if isinstance(error, bool) or not isinstance(error, int | float | None):
            _fail(f"gates[{i}]: error should be a number, not {_json_type(error)}")
        if isinstance(error, float) and not math.isfinite(error):
            _fail(f"gates[{i}]: error should be a finite number, not {_shown(error)}")
        if not isinstance(gate.get("metadata", {}), dict):
            _fail(f"gates[{i}]: metadata should be an object, not {_json_type(gate['metadata'])}")
        gate_entries.append({**gate, "operational": _flag(gate, f"gates[{i}]")})
    qubit_entries = []
    for i, qubit in enumerate(old["qubits"]):
        if not (isinstance(qubit, dict) and isinstance(qubit.get("index"), int)):
            _fail(f"qubits[{i}] needs an integer index")
        if not isinstance(qubit.get("frequency_ghz"), int | float | None):
            _fail(
                f"qubits[{i}]: frequency_ghz should be a number,"
                f" not {_json_type(qubit['frequency_ghz'])}"
            )
        if not isinstance(qubit.get("metadata", {}), dict):
            _fail(f"qubits[{i}]: metadata should be an object, not {_json_type(qubit['metadata'])}")
        qubit_entries.append({**qubit, "operational": _flag(qubit, f"qubits[{i}]")})
    for i, edge in enumerate(old["coupling_map"]):
        if not isinstance(edge, list):
            _fail(f"coupling_map[{i}] should be a list of two qubits, not {_json_type(edge)}")
    prov = old["provenance"]
    if not isinstance(prov.get("notes", []), list):
        _fail("provenance.notes should be a list")
    if not isinstance(prov.get("raw_hash"), str | None):
        _fail("provenance.raw_hash should be a string")
    if not isinstance(prov.get("harvester_version", ""), str):
        _fail(
            "provenance.harvester_version should be a string,"
            f" not {_json_type(prov['harvester_version'])}"
        )
    if not isinstance(prov.get("source_timestamp"), str | None):
        _fail(
            "provenance.source_timestamp should be a string,"
            f" not {_json_type(prov['source_timestamp'])}"
        )
    if not isinstance(old.get("extensions"), dict | None):
        _fail("extensions should be an object")
    return {**old, "gates": gate_entries, "qubits": qubit_entries}


def _aware_time(text: str) -> bool:
    try:
        return datetime.fromisoformat(text).tzinfo is not None
    except ValueError:
        return False


_V01_LAX_BOOL = TypeAdapter(bool)


def _flag(entry: dict[str, Any], where: str) -> bool:
    value = entry.get("operational", True)
    try:
        return _V01_LAX_BOOL.validate_python(value)
    except ValidationError:
        _fail(f"{where}: operational should be true or false, not {_shown(value)}")


def _shown(value: Any) -> str:
    return json.dumps(value) if isinstance(value, str | int | float) else _json_type(value)


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true or false"
    if isinstance(value, int | float):
        return "a number"
    return _JSON_TYPES.get(type(value), type(value).__name__)


def _device_name(provider: str, backend_name: str) -> str:
    if provider == "qiskit_fake" and backend_name.startswith("fake_"):
        return "ibm_" + backend_name.removeprefix("fake_")
    return backend_name


def _licensing(provider: str, raw_hash: Any) -> dict[str, Any]:
    if provider == "qiskit_fake" and raw_hash in _PACKAGE_SNAPSHOTS:
        return {
            "license": "Apache-2.0",
            "attribution": "IBM Quantum (via qiskit-ibm-runtime)",
            "redistributable": "yes",
        }
    if provider == "demo":
        return {"redistributable": "yes"}
    if provider in _IBM_PROVIDERS:
        return {"attribution": "IBM Quantum", "redistributable": "unknown"}
    return {"redistributable": "unknown"}


def _explicitly_free(entry: dict[str, Any]) -> bool:
    return entry.get("error") == 0 and entry.get("duration_ns") == 0


def _gate_metadata(entries: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """The metadata of each gate, keyed like "cx:0-1", and a note for each item not kept."""
    from .profile import _freeze  # profile imports this module

    kept: dict[str, dict[str, Any]] = {}
    notes = []
    for entry in entries:
        locus = f"{entry['name']}:{'-'.join(str(q) for q in entry['qubits'])}"
        for key, value in entry.get("metadata", {}).items():
            try:
                # depth 3: extensions > v01_gate_metadata > locus
                _freeze({key: value}, depth=3)
            except ValueError as exc:
                notes.append(
                    f"Not converted: 0.1 gate metadata {json.dumps(key)} of {locus} ({exc})."
                )
            else:
                kept.setdefault(locus, {})[key] = value
    return kept, notes


def _record(entry: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {"gate": entry["name"], "qubits": entry["qubits"]}
    error = entry.get("error")
    # No channel has an average infidelity above d/(d+1), the value of full depolarization.
    # IBM writes such a value, often 1, to mark a broken gate.
    limit = metrics.max_avg_infidelity(len(entry["qubits"]))
    if not entry["operational"] or (error is not None and error >= limit):
        record["disabled"] = True
    elif error is not None:
        record["avg_infidelity"] = error
    if entry.get("duration_ns") is not None:
        record["duration_ns"] = entry["duration_ns"]
    return record


_COHERENCE = {"t1_us": "T1", "t2_us": "T2"}


def _invalid_time(value: Any) -> bool:
    """A dead qubit can report T1 = 0. The value counts as missing, as in the IBM importer."""
    if isinstance(value, float) and not math.isfinite(value):
        return True
    return isinstance(value, int | float) and not isinstance(value, bool) and value <= 0


def _coherence_notes(qubits: list[dict[str, Any]]) -> list[str]:
    return [
        f"Qubit {q['index']} reported {label} = {_shown(q[key])} us."
        " NoiseVault treats the value as missing."
        for q in qubits
        for key, label in _COHERENCE.items()
        if _invalid_time(q.get(key))
    ]


def _qubit(q: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"index": q["index"]}
    for key in _COHERENCE:
        if q.get(key) is not None and not _invalid_time(q[key]):
            out[key] = q[key]
    a, b = q.get("prob_meas1_prep0"), q.get("prob_meas0_prep1")
    if a is not None and b is not None:
        out["readout"] = {"p1_given_0": a, "p0_given_1": b}
    elif q.get("readout_error") is not None:
        out["readout"] = {"error": q["readout_error"]}
    if not q["operational"]:
        out["disabled"] = True
    return out
