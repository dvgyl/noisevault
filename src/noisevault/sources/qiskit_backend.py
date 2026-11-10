"""Profiles from Qiskit backends and from IBM calibration data.

Every IBM source goes first into a :class:`Calibration`: a BackendV2 Target, BackendProperties
JSON from the public endpoint or an account, and a calibration CSV. Then :func:`to_profile`
turns the Calibration into a Profile. As a result, the IBM conventions (dead-gate sentinel,
virtual ``rz``, medians as device defaults, RB qualifiers) are in one place. Only the functions
that read a backend import Qiskit, so the other IBM sources work on a core install.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import importlib
import importlib.metadata
import json
import math
import re
import statistics
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, NewType

from .. import __version__, gates, metrics, units
from ..errors import SourceDataError, plural, qubit_loci
from ..profile import FORMAT_VERSION, Profile, Technology
from . import Origin, source_json

RUNTIME_REPO = "https://github.com/Qiskit/qiskit-ibm-runtime"
IBM_ATTRIBUTION = "IBM Quantum, via qiskit-ibm-runtime"

# Not gates of a noise model: scheduling, control flow, and IBM's mid-circuit variants of
# measure and reset, which a circuit reaches only through dynamic-circuit instructions.
_NOT_GATES = frozenset(
    {
        "delay",
        "barrier",
        "if_else",
        "while_loop",
        "for_loop",
        "switch_case",
        "box",
        "measure_2",
        "reset_2",
        "measure_reset",
        "measure_reset_2",
    }
)
# IBM documents its error numbers this way: RB throughout, 2-qubit errors measured on isolated
# pairs and including the single-qubit Clifford layers, 1-qubit errors measured simultaneously.
_IBM_QUALIFIERS: Mapping[int, Mapping[str, Any]] = {
    1: {"method": "rb", "measured": "simultaneous"},
    2: {"method": "rb", "measured": "isolated", "includes": ["1q_dressing"]},
}
_QISKIT_TO_CANONICAL = {info.qiskit: info.name for info in gates.GATES.values() if info.qiskit}

# The curated bundle: modern Heron and Eagle r3 snapshots, the two real Nighthawk snapshots, and
# FakeManilaV2 for 5-qubit demos. The bundle leaves out FakeNighthawk, because its package says
# its values are not typical of the device. The bundle also leaves out FakeFractionalBackend and
# other test backends.
BUNDLED_FAKES = (
    "FakeAachen",
    "FakeBerlin",
    "FakeBoston",
    "FakeBrisbane",
    "FakeBrussels",
    "FakeCusco",
    "FakeFez",
    "FakeKawasaki",
    "FakeKingston",
    "FakeKyiv",
    "FakeManilaV2",
    "FakeMarrakesh",
    "FakeMiami",
    "FakePittsburgh",
    "FakeQuebec",
    "FakeSherbrooke",
    "FakeStrasbourg",
    "FakeTorino",
)


@dataclass(frozen=True)
class Instruction:
    """One calibrated instruction on specific qubits, in the source's own gate name."""

    name: str
    qubits: tuple[int, ...]
    error: float | None = None
    duration_ns: float | None = None
    operational: bool = True


@dataclass(frozen=True)
class QubitCalibration:
    t1_us: float | None = None
    t2_us: float | None = None
    p1_given_0: float | None = None
    p0_given_1: float | None = None
    prep_error: float | None = None
    operational: bool = True


@dataclass(frozen=True)
class Calibration:
    """A device calibration in IBM's shape: per-instruction errors and per-qubit properties.

    ``measure`` entries carry the readout duration and the symmetric readout error used when a
    qubit has no asymmetric pair. ``supported`` maps a gate to every locus it runs on when the
    source lists them, as a Target does. A gate it leaves out may run on any qubit or pair.
    """

    name: str
    num_qubits: int
    instructions: tuple[Instruction, ...]
    qubits: Mapping[int, QubitCalibration] = field(default_factory=dict)
    vendor: str | None = "ibm"
    technology: Technology = "superconducting"
    processor: str | None = None
    calibrated_at: datetime | None = None
    skipped: tuple[str, ...] = ()  # source instruction names deliberately not converted
    supported: Mapping[str, frozenset[tuple[int, ...]]] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


# Qiskit backends -------------------------------------------------------------------------------


def from_qiskit_backend(backend: Any) -> Profile:
    """A profile from any Qiskit BackendV2, such as a qiskit-ibm-runtime fake or a live backend.

    Gate errors, durations and T1/T2 come from the backend's Target. The asymmetric readout pair,
    preparation error and calibration time come from ``backend.properties()`` if it exists.
    """
    properties = _properties(backend)
    target = _target(backend, properties)
    if target is None:
        raise TypeError(f"{backend!r} is not a Qiskit BackendV2: it has no target")
    if target.num_qubits is None:
        raise SourceDataError(
            f"{backend.name} has no fixed qubit count, so it has no device calibration",
            hint="pass a device backend, such as a qiskit-ibm-runtime fake or a live IBM backend",
        )
    origin = Origin(f"backend {backend.name}")
    cal = calibration_from_target(target, name=_device_name(backend))
    if properties is not None:
        from_props = calibration_from_properties(properties.to_dict(), origin=origin)
        # IBM's Target converter drops non-operational gates and every gate on a faulty qubit.
        names = {_QISKIT_TO_CANONICAL.get(n, n) for n in target.operation_names}
        in_target = {(i.name, i.qubits) for i in cal.instructions}
        dropped = tuple(
            i
            for i in from_props.instructions
            if i.name in names and (i.name, i.qubits) not in in_target
        )
        cal = replace(
            cal,
            instructions=cal.instructions + dropped,
            calibrated_at=from_props.calibrated_at,
            qubits={
                i: _overlay(cal.qubits.get(i, QubitCalibration()), from_props.qubits.get(i))
                for i in sorted(cal.qubits.keys() | from_props.qubits.keys())
            },
        )
    cal = replace(
        cal,
        vendor=_vendor(backend),
        technology=_technology(backend),
        processor=_processor(backend),
    )
    return to_profile(cal, _backend_provenance(backend), origin=origin)


def calibration_from_target(target: Any, *, name: str) -> Calibration:
    """Read a Qiskit Target: per-qargs error and duration, and per-qubit T1/T2.

    A global 1-qubit instruction (``qargs=None``) becomes an instruction on every qubit. A
    global instruction on more qubits is not converted, and ``notes`` names it.
    """
    from qiskit.circuit import Gate

    instructions, skipped, supported, notes = [], set(), {}, []
    for op_name in sorted(target.operation_names):
        operation = target.operation_from_name(op_name)
        keep = isinstance(operation, Gate) or op_name in ("measure", "reset")
        if op_name in _NOT_GATES or not keep:
            skipped.add(op_name)
            continue
        loci = dict(target[op_name])
        if None in loci:
            props = loci.pop(None)
            if operation.num_qubits == 1:
                loci |= {(q,): props for q in range(target.num_qubits) if (q,) not in loci}
            else:
                notes.append(
                    f"Not converted: {op_name}, a global instruction on any"
                    f" {operation.num_qubits} qubits. A profile records a gate on 2 or more"
                    " qubits only on the qubits that the source lists."
                )
        else:
            supported[_QISKIT_TO_CANONICAL.get(op_name, op_name)] = frozenset(loci)
        for qargs, props in loci.items():
            instructions.append(
                Instruction(
                    _QISKIT_TO_CANONICAL.get(op_name, op_name),
                    tuple(qargs),
                    error=None if props is None else props.error,
                    duration_ns=_seconds_to(props and props.duration, "ns"),
                )
            )
    qubit_props = target.qubit_properties or []
    qubits = {
        i: QubitCalibration(t1_us=_seconds_to(p.t1, "us"), t2_us=_seconds_to(p.t2, "us"))
        for i, p in enumerate(qubit_props)
        if p is not None
    }
    control_flow = {"if_else", "while_loop", "for_loop", "switch_case", "box", "barrier"}
    return Calibration(
        name=name,
        num_qubits=target.num_qubits,
        instructions=tuple(instructions),
        qubits=qubits,
        skipped=tuple(sorted(skipped - control_flow - {"delay"})),
        supported=supported,
        notes=tuple(notes),
    )


def bundled_profiles() -> list[Profile]:
    """The curated IBM set from qiskit-ibm-runtime's packaged snapshots."""
    from qiskit_ibm_runtime import fake_provider

    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # fake backends warn about deprecations on creation
        for class_name in BUNDLED_FAKES:
            out.append(from_qiskit_backend(getattr(fake_provider, class_name)()))
    return out


def _properties(backend: Any) -> Any:
    method = getattr(backend, "properties", None)
    return None if method is None else method()


def _target(backend: Any, properties: Any) -> Any:
    """The Target of the same calibration as ``properties``, or None for a non-BackendV2.

    A qiskit-ibm-runtime backend keeps its first Target when a refresh changes only its
    properties. For this reason, this function builds the Target again from the snapshot that
    the rest of the profile reads.
    """
    if properties is not None and type(backend).__module__.startswith("qiskit_ibm_runtime"):
        from qiskit_ibm_runtime.utils.backend_converter import convert_to_target

        return convert_to_target(configuration=backend.configuration(), properties=properties)
    return getattr(backend, "target", None)


def _overlay(base: QubitCalibration, extra: QubitCalibration | None) -> QubitCalibration:
    """Target values plus what only BackendProperties has: readout pair, prep, operational."""
    if extra is None:
        return base
    return replace(
        base,
        p1_given_0=extra.p1_given_0,
        p0_given_1=extra.p0_given_1,
        prep_error=extra.prep_error,
        operational=extra.operational,
    )


def _shipped_props(backend: Any) -> bytes | None:
    """The properties file qiskit-ibm-runtime ships for this fake, or None when the backend's
    data is not what the package ships.

    ``refresh()`` overwrites the installed files with account data or reads a temporary copy.
    For this reason, this function checks the files against the wheel's RECORD, and the backend
    against a new instance.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # fake backends warn about deprecations on creation
        fresh = type(backend)()
    folder = Path(fresh.dirname)
    recorded = _recorded_sha256()
    raw = {}
    for name in (fresh.conf_filename, fresh.props_filename):
        path = (folder / name).resolve()
        raw[name] = path.read_bytes()
        if recorded.get(path) != _record_digest(raw[name]):
            return None
    same = (
        backend.configuration().to_dict() == fresh.configuration().to_dict()
        and backend.properties().to_dict() == fresh.properties().to_dict()
    )
    return raw[fresh.props_filename] if same else None


@functools.cache
def _recorded_sha256() -> dict[Path, str]:
    dist = importlib.metadata.distribution("qiskit-ibm-runtime")
    return {
        Path(dist.locate_file(f)).resolve(): f.hash.value
        for f in dist.files or ()
        if f.hash is not None and f.hash.mode == "sha256"
    }


def _record_digest(raw: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode()


def _is_fake(backend: Any) -> bool:
    return type(backend).__module__.startswith("qiskit_ibm_runtime.fake_provider")


def _vendor(backend: Any) -> str | None:
    module = type(backend).__module__
    return (
        "ibm"
        if module.startswith("qiskit_ibm_runtime") or _device_name(backend).startswith("ibm_")
        else None
    )


def _technology(backend: Any) -> Technology:
    if _vendor(backend) == "ibm":
        return "superconducting"
    package = type(backend).__module__.partition(".")[0]
    if package in ("qiskit_ionq", "qiskit_quantinuum") or _device_name(backend).startswith("ionq_"):
        return "trapped_ion"
    return "other"


_AER_WRAPPED = re.compile(r"aer_simulator_from\((?P<inner>.+)\)")


def _device_name(backend: Any) -> str:
    """The device name: ``fake_fez`` and ``aer_simulator_from(fake_fez)`` both give ibm_fez."""
    name = backend.name
    wrapped = _AER_WRAPPED.fullmatch(name)
    if wrapped:
        name = wrapped["inner"]
    if _is_fake(backend) or (wrapped and name.startswith("fake_")):
        name = "ibm_" + name.removeprefix("fake_")
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _processor(backend: Any) -> str | None:
    return processor_name(getattr(backend, "processor_type", None))


def processor_name(processor_type: Mapping[str, Any] | None) -> str | None:
    """IBM's ``{"family": "Heron", "revision": "2"}`` as ``"Heron r2"``."""
    if not processor_type or not processor_type.get("family"):
        return None
    family, revision = processor_type["family"], processor_type.get("revision")
    return f"{family} r{revision}" if revision not in (None, "") else str(family)


def _backend_provenance(backend: Any) -> dict[str, Any]:
    class_name = type(backend).__name__
    if _is_fake(backend):
        version = importlib.import_module("qiskit_ibm_runtime").__version__
        shipped = _shipped_props(backend)
        if shipped is not None:
            caveat = _model_caveat(backend, shipped, version)
            return {
                "data_kind": "measured" if caveat is None else "vendor_model",
                "source_kind": "package_snapshot",
                "source": f"qiskit-ibm-runtime {version} {class_name}",
                "source_url": RUNTIME_REPO,
                "license": "Apache-2.0",
                "attribution": IBM_ATTRIBUTION,
                "redistributable": "yes",
                "source_hash": sha256_bytes(shipped),
                "notes": [] if caveat is None else [caveat],
            }
        props = backend.properties().to_dict()
        return {
            "data_kind": "measured",
            "source_kind": "account_api",
            "source": f"IBM Quantum backend data in qiskit-ibm-runtime {version} {class_name}",
            "attribution": "IBM Quantum",
            "redistributable": "unknown",
            "source_hash": sha256_bytes(json.dumps(props, sort_keys=True, default=str).encode()),
            "notes": [
                f"This data is not the snapshot that qiskit-ibm-runtime {version} ships for"
                f" {class_name}, for example after refresh(). The data is IBM Quantum service data"
                " under IBM's terms, and the package's Apache-2.0 license does not cover it."
            ],
        }
    live = type(backend).__module__.startswith("qiskit_ibm_runtime")
    if live or backend.name.startswith("ibm_"):  # not a simulator wrapping a device's numbers
        return {
            "data_kind": "measured",
            "source_kind": "account_api",
            "source": f"IBM Quantum backend {backend.name} ({class_name})",
            "attribution": "IBM Quantum",
            "redistributable": "unknown",
            "retrieved_at": now_utc(),
        }
    return {"source_kind": "other", "source": f"Qiskit backend {backend.name} ({class_name})"}


def _model_caveat(backend: Any, shipped: bytes, version: str) -> str | None:
    """A note when a fake's snapshot is a model rather than a device's calibration, else None.

    A snapshot taken from a device has the device name (``ibm_fez``, ``ibmq_manila``). The
    package's modeled backends have their own names (``fake_nighthawk``, ``fake_fractional``).
    """
    name = source_json(shipped, backend.props_filename).get("backend_name") or ""
    if not name.startswith("fake_"):
        return None
    note = (
        f"qiskit-ibm-runtime {version} {type(backend).__name__} is a model, not a calibration"
        f" of an IBM device. Its snapshot names the backend {name}."
    )
    # The package's own words on what the model is and is not, from the class docstring's
    # prose (examples, lists and directives follow it).
    prose = re.split(r"\n\s*(?:#|\*|\.\.|```)", type(backend).__doc__ or "")[0]
    sentences = re.split(r"(?<=\.)\s+", " ".join(prose.split()))
    said = " ".join(s for s in sentences if re.search(r"model|represent", s, re.IGNORECASE))
    return f'{note} The package says: "{said}"' if said else note


# BackendProperties JSON ------------------------------------------------------------------------

_QUBIT_PARAMETERS = {
    "t1_us": "T1",
    "t2_us": "T2",
    "p1_given_0": "prob_meas1_prep0",
    "p0_given_1": "prob_meas0_prep1",
    "prep_error": "init_error",
}
_READOUT_PARAMETERS = {"error": "readout_error", "duration_ns": "readout_length"}
_GATE_PARAMETERS = {"error": "gate_error", "duration_ns": "gate_length"}


def calibration_from_properties(props: Mapping[str, Any], *, origin: Origin) -> Calibration:
    """Read IBM BackendProperties as a dict (``properties().to_dict()`` or the REST JSON).

    The readout error and length on each qubit become its ``measure`` instruction. The
    ``measure`` entries of the gate list repeat those numbers, so this function skips them. A
    time in an unknown unit raises a SourceDataError from ``origin``. A value that is not a number
    also raises one, and so does an ``operational`` flag that is not 0 or 1. A gate on the wrong
    number of qubits, or on a qubit that ``qubits`` does not list, also raises one.
    """
    qubits: dict[int, QubitCalibration] = {}
    instructions: list[Instruction] = []
    for index, params in enumerate(props.get("qubits") or []):
        owner = f"qubit {index}"
        values = _parameters(params, origin, owner)
        qubits[index] = QubitCalibration(
            t1_us=_in_unit(values.get("T1"), "us", origin, owner),
            t2_us=_in_unit(values.get("T2"), "us", origin, owner),
            p1_given_0=_fraction(values.get("prob_meas1_prep0"), origin, owner),
            p0_given_1=_fraction(values.get("prob_meas0_prep1"), origin, owner),
            prep_error=_fraction(values.get("init_error"), origin, owner),
            operational=_operational(values, origin, owner),
        )
        error, length = values.get("readout_error"), values.get("readout_length")
        if error is not None or length is not None:
            instructions.append(
                Instruction(
                    "measure",
                    (index,),
                    _fraction(error, origin, owner),
                    _in_unit(length, "ns", origin, owner),
                )
            )
    skipped = set()
    arities: dict[str, int] = {}
    for entry in props.get("gates") or []:
        name = entry["gate"]
        _check_indices(name, entry["qubits"], len(qubits), origin)
        if name == "measure":
            continue
        if name in _NOT_GATES:
            skipped.add(name)
            continue
        canonical = _QISKIT_TO_CANONICAL.get(name, name)
        locus = _locus(name, canonical, entry["qubits"], arities, origin)
        owner = f"{name} on {qubit_loci(locus)}"
        values = _parameters(entry.get("parameters") or [], origin, owner)
        instructions.append(
            Instruction(
                canonical,
                locus,
                error=_fraction(values.get("gate_error"), origin, owner),
                duration_ns=_in_unit(values.get("gate_length"), "ns", origin, owner),
                operational=_operational(values, origin, owner),
            )
        )
    stamp = props.get("last_update_date")
    return Calibration(
        name=props.get("backend_name") or "unknown",
        num_qubits=len(qubits),
        instructions=tuple(instructions),
        qubits=qubits,
        calibrated_at=None if stamp is None else as_utc(stamp),
        skipped=tuple(sorted(skipped)),
    )


def _parameters(
    params: Iterable[Mapping[str, Any]], origin: Origin, owner: str
) -> dict[str, Mapping[str, Any]]:
    values: dict[str, Mapping[str, Any]] = {}
    for param in params:
        name, value, unit = param.get("name"), param.get("value"), param.get("unit")
        if not isinstance(name, str):
            raise origin.refuse(f"a parameter of {owner} has the name {name!r}, not a string")
        if value is not None and (isinstance(value, bool) or not isinstance(value, int | float)):
            raise origin.refuse(f"{name} of {owner} is {value!r}, not a number")
        if unit is not None and not isinstance(unit, str):
            raise origin.refuse(f"{name} of {owner} has the unit {unit!r}, not a string")
        first = values.setdefault(name, param)
        if (first.get("value"), first.get("unit") or None) != (value, unit or None):
            raise origin.refuse(
                f"{name} of {owner} has two different values, {_shown(first)} and {_shown(param)}"
            )
    return values


def _shown(param: Mapping[str, Any]) -> str:
    return f"{param.get('value')!r} {param.get('unit') or ''}".rstrip()


def _check_indices(name: str, qubits: Sequence[Any], num_qubits: int, origin: Origin) -> None:
    """``origin`` refuses a qubit that is not an integer from 0 to ``num_qubits - 1``.

    The check includes the entries that the importer skips and the virtual gates that a profile
    does not record.
    """
    if any(isinstance(q, bool) or not isinstance(q, int) or q < 0 for q in qubits):
        raise origin.refuse(
            f"gate {name} is on {list(qubits)}; a qubit index is an integer 0 or more"
        )
    if any(q >= num_qubits for q in qubits):
        raise origin.refuse(
            f"gate {name} is on {list(qubits)}; the calibration has {plural(num_qubits, 'qubit')}"
        )


def _locus(
    name: str, canonical: str, qubits: Sequence[Any], arities: dict[str, int], origin: Origin
) -> tuple[int, ...]:
    """``qubits`` of one gate entry; a gate the registry does not know keeps the qubit count of
    its first entry.
    """
    info = gates.lookup(canonical)
    arity = arities.setdefault(canonical, len(qubits)) if info is None else info.arity
    if len(qubits) != arity:
        expected = f"{name} acts on" if info else f"the first {name} entry is on"
        raise origin.refuse(
            f"gate {name} is on {plural(len(qubits), 'qubit')} {list(qubits)}; {expected}"
            f" {plural(arity, 'qubit')}"
        )
    return tuple(qubits)


def _value(param: Mapping[str, Any] | None) -> float | None:
    return None if param is None else param.get("value")


def _fraction(param: Mapping[str, Any] | None, origin: Origin, owner: str) -> float | None:
    """An error or a probability from 0 to 1. ``origin`` refuses a unit that is not ``%``."""
    if param is None or param.get("value") is None or not param.get("unit"):
        return _value(param)
    if param["unit"] == "%":
        return param["value"] / 100
    raise origin.refuse(
        f"{param['name']} of {owner} has the unknown unit {param['unit']!r}, not % or no unit"
    )


def _operational(values: Mapping[str, Mapping[str, Any]], origin: Origin, owner: str) -> bool:
    """qiskit-ibm-runtime's ``BackendProperties`` reads a flag of ``nan`` as operational and a
    flag of ``null`` as not operational, so ``origin`` refuses a flag that is not 0 or 1.
    """
    param = values.get("operational")
    if param is None:
        return True
    value = param.get("value")
    if value not in (0, 1):
        raise origin.refuse(f"operational of {owner} is {value!r}, not 0 or 1")
    return value == 1


_UNIT_ALIASES = {"µs": "us", "μs": "us", "sec": "s"}


def _in_unit(
    param: Mapping[str, Any] | None, unit: str, origin: Origin, owner: str
) -> float | None:
    value = _value(param)
    if value is None:
        return None
    given = param.get("unit") or unit  # type: ignore[union-attr]
    try:
        return units.convert(float(value), _UNIT_ALIASES.get(given, given), unit)
    except KeyError:
        raise origin.refuse(
            f"{param['name']} of {owner} has the unknown time unit {given!r}, not one of ns, us,"
            " µs, ms or s"
        ) from None


# Calibration -> Profile ------------------------------------------------------------------------


_Checked = NewType("_Checked", Calibration)


def to_profile(cal: Calibration, provenance: Mapping[str, Any], *, origin: Origin) -> Profile:
    return origin.profile(_profile_data(_checked(cal, origin), provenance))


def used_parameters(cal: Calibration) -> set[tuple[str | None, str, tuple[int, ...]]]:
    """Each BackendProperties value of ``cal`` that its profile uses, as (gate, parameter, qubits).

    ``gate`` is None for a value of a qubit. A disabled gate uses only the error that disables it.
    A virtual gate uses no value.
    """
    used, _ = _in_use(cal)
    found = {
        (None, name, (index,))
        for index, qubit in used.qubits.items()
        for field, name in _QUBIT_PARAMETERS.items()
        if getattr(qubit, field) is not None
    }
    for inst in used.instructions:
        gate, names = (
            (None, _READOUT_PARAMETERS) if inst.name == "measure" else (inst.name, _GATE_PARAMETERS)
        )
        found |= {
            (gate, name, inst.qubits)
            for field, name in names.items()
            if getattr(inst, field) is not None
        }
    return found


def _in_use(cal: Calibration) -> tuple[Calibration, frozenset[str]]:
    """``cal`` with None in each value that its profile does not use, and the virtual gates.

    A readout pair has priority over the readout error. A readout duration needs one of the two.
    A disabled gate keeps only the error that disables it. A T1 or T2 that is not a positive
    finite number is missing. A gate is virtual when each working entry has error 0 and no
    duration. ``virtual: true`` then replaces the values of each entry that is not disabled.
    """
    qubits, _ = _without_invalid_coherence(cal)
    paired = {i for i, q in qubits.items() if q.p1_given_0 is not None and q.p0_given_1 is not None}
    qubits = {
        i: q if i in paired else replace(q, p1_given_0=None, p0_given_1=None)
        for i, q in qubits.items()
    }
    arity = {inst.name: len(inst.qubits) for inst in cal.instructions}
    runs_on = {
        name: _both_ways(name, arity[name], support)
        for name, support in cal.supported.items()
        if name in arity
    }
    instructions = []
    for inst in cal.instructions:
        if inst.name == "measure":
            if inst.qubits[0] in paired:
                inst = replace(inst, error=None)
            elif inst.error is None:
                inst = replace(inst, duration_ns=None)
        else:
            sentinel = _is_sentinel(inst.error, len(inst.qubits))
            unsupported = inst.name in runs_on and inst.qubits not in runs_on[inst.name]
            if sentinel or unsupported or not inst.operational:
                error = inst.error if sentinel else None
                inst = replace(inst, error=error, duration_ns=None, operational=False)
        instructions.append(inst)
    disabled_qubits = {i for i, q in cal.qubits.items() if not q.operational}
    working: dict[str, list[Instruction]] = {}
    for inst in instructions:
        if inst.name != "measure":
            working.setdefault(inst.name, [])
            if inst.operational and disabled_qubits.isdisjoint(inst.qubits):
                working[inst.name].append(inst)
    virtual = frozenset(
        name
        for name, entries in working.items()
        if entries and all(e.error == 0 and not e.duration_ns for e in entries)
    )
    instructions = [
        replace(inst, error=None, duration_ns=None)
        if inst.name in virtual and inst.operational
        else inst
        for inst in instructions
    ]
    return replace(cal, qubits=qubits, instructions=tuple(instructions)), virtual


def _profile_data(cal: _Checked, provenance: Mapping[str, Any]) -> dict[str, Any]:
    ibm = cal.vendor == "ibm"
    used, virtual = _in_use(cal)
    by_name: dict[str, list[Instruction]] = {}
    for inst in used.instructions:
        by_name.setdefault(inst.name, []).append(inst)
    measure = {inst.qubits[0]: inst for inst in by_name.pop("measure", [])}
    connectivity = _connectivity(cal.instructions)
    disabled_qubits = {i for i, q in cal.qubits.items() if not q.operational}

    definitions: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    gate_notes: list[str] = []
    for name, entries in by_name.items():
        arity = len(entries[0].qubits)
        listed = _both_ways(name, arity, (e.qubits for e in entries))
        unlisted = _unlisted(arity, listed, cal.num_qubits, connectivity)
        if name in cal.supported:
            entries = entries + [Instruction(name, locus, operational=False) for locus in unlisted]
            unpublished = []
        else:
            unpublished = unlisted
        definition, gate_records = _gate(
            name, arity, entries, ibm, disabled_qubits, virtual=name in virtual
        )
        definitions[name] = definition
        records += gate_records
        unpublished += [e.qubits for e in entries if e.operational and e.error is None]
        note = _unpublished_note(name, arity, unpublished, definition, disabled_qubits)
        gate_notes += [note] if note else []

    _, invalid = _without_invalid_coherence(cal)
    qubit_records = [
        _qubit_record(i, used.qubits.get(i, QubitCalibration()), measure.get(i))
        for i in range(cal.num_qubits)
    ]
    working = [q for q in qubit_records if not q.get("disabled")]
    qubit_records = [q for q in qubit_records if len(q) > 1]
    notes = list(provenance.get("notes", ()))
    if cal.skipped:
        notes.append(f"Not converted: {', '.join(cal.skipped)}.")
    notes += cal.notes
    notes += gate_notes
    for key, label in _COHERENCE.items():
        if any(key in q for q in working):
            notes += [
                f"Qubit {index} reported {label} = {value:g} us. NoiseVault treats the value as"
                " missing, so the device median applies."
                for index, value in invalid[key]
            ]
        elif on_working := [(i, v) for i, v in invalid[key] if i not in disabled_qubits]:
            index, value = on_working[0]
            raise SourceDataError(
                f"{cal.name} reports no valid {label} on any working qubit (for example, qubit"
                f" {index}: {label} = {value:g} us). {label} must be a positive number of"
                " microseconds"
            )
    explained = {key: {index for index, _ in found} for key, found in invalid.items()}
    lacking = {
        key: [
            q["index"] for q in working if key not in q and q["index"] not in explained.get(key, ())
        ]
        for key in ("readout", "prep", "t1_us", "t2_us")
    }
    readout, prep = _median_readout(working), _median_prep(working)
    if readout and lacking["readout"]:
        readout = None
        notes.append(
            f"Qubits {lacking['readout']} have no readout calibration, so their readout is unknown."
        )
    if prep and lacking["prep"]:
        prep = None
        notes.append(
            f"Qubits {lacking['prep']} have no prep calibration, so their prep is unknown."
        )
    idle = _median_idle(working, ibm)
    for key, label in (("t1_us", "T1"), ("t2_us", "T2")):
        if idle and key in idle and lacking[key]:
            notes.append(
                f"Qubits {lacking[key]} have no {label}; the device median applies to them."
            )
    data = {
        "noisevault": FORMAT_VERSION,
        "device": {
            "name": cal.name,
            "vendor": cal.vendor,
            "technology": cal.technology,
            "num_qubits": cal.num_qubits,
            "processor": cal.processor,
            "calibrated_at": cal.calibrated_at,
        },
        "connectivity": connectivity,
        "gates": definitions,
        "readout": readout,
        "prep": prep,
        "idle": idle,
        "qubits": qubit_records,
        "calibrations": records,
        "provenance": {"tool": f"noisevault {__version__}", **provenance, "notes": notes},
    }
    return data


_COHERENCE = {"t1_us": "T1", "t2_us": "T2"}


def _checked(cal: Calibration, origin: Origin) -> _Checked:
    for where, value, most in _values_except_coherence(cal):
        if value is None or (math.isfinite(value) and 0 <= value <= most):
            continue
        if not math.isfinite(value):
            problem = "a finite number"
        elif value < 0:
            problem = "greater than or equal to 0"
        else:
            problem = f"less than or equal to {most:g}"
        raise origin.refuse(f"{where}: Input should be {problem}, got {value!r}")
    return _Checked(cal)


def _values_except_coherence(cal: Calibration) -> Iterator[tuple[str, float | None, float]]:
    for inst in cal.instructions:
        if inst.name == "measure":
            owner = f"qubit {inst.qubits[0]}"
            yield f"readout.error of {owner}", inst.error, 1
            yield f"readout.duration_ns of {owner}", inst.duration_ns, math.inf
        else:
            owner = f"{inst.name} on {qubit_loci(inst.qubits)}"
            yield f"avg_infidelity of {owner}", inst.error, 1
            yield f"duration_ns of {owner}", inst.duration_ns, math.inf
    for index, qubit in sorted(cal.qubits.items()):
        yield f"readout.p1_given_0 of qubit {index}", qubit.p1_given_0, 1
        yield f"readout.p0_given_1 of qubit {index}", qubit.p0_given_1, 1
        yield f"prep.error of qubit {index}", qubit.prep_error, 1


def _without_invalid_coherence(
    cal: Calibration,
) -> tuple[dict[int, QubitCalibration], dict[str, list[tuple[int, float]]]]:
    """The qubits with nonpositive or nonfinite T1/T2 cleared, and what each cleared one said.

    A dead qubit can report T1 = 0. If the value counts as missing, the rest of the device stays
    usable.
    """
    qubits = dict(cal.qubits)
    invalid: dict[str, list[tuple[int, float]]] = {key: [] for key in _COHERENCE}
    for index, qubit in sorted(cal.qubits.items()):
        for key in _COHERENCE:
            value = getattr(qubit, key)
            if value is not None and not (math.isfinite(value) and value > 0):
                invalid[key].append((index, value))
                qubits[index] = replace(qubits[index], **{key: None})
    return qubits, invalid


def _gate(
    name: str,
    arity: int,
    entries: Sequence[Instruction],
    ibm: bool,
    disabled_qubits: set[int],
    *,
    virtual: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dead = [not e.operational for e in entries]
    working = [
        e
        for e, is_dead in zip(entries, dead, strict=True)
        if not is_dead and disabled_qubits.isdisjoint(e.qubits)
    ]
    definition: dict[str, Any] = {} if gates.lookup(name) else {"qubits": arity}
    if virtual:
        definition["virtual"] = True
    else:
        errors = [e.error for e in working if e.error is not None]
        if errors:
            definition |= {"avg_infidelity": statistics.median(errors), "statistic": "median"}
            if ibm:
                definition |= _IBM_QUALIFIERS.get(arity, {})
        durations = [e.duration_ns for e in working if e.duration_ns is not None]
        if durations:
            definition["duration_ns"] = _clean(statistics.median(durations))

    records = []
    for entry, is_dead in zip(entries, dead, strict=True):
        record: dict[str, Any] = {"gate": name, "qubits": list(entry.qubits)}
        if is_dead:
            record["disabled"] = True
        elif virtual:
            continue
        else:
            if entry.error is not None:
                record |= {"avg_infidelity": entry.error, "statistic": "individual"}
            if entry.duration_ns is not None and _clean(entry.duration_ns) != definition.get(
                "duration_ns"
            ):
                record["duration_ns"] = _clean(entry.duration_ns)
        records.append(record)
    if arity == 2 and gates.is_symmetric(name):
        records = _one_per_pair(records)
    return definition, records


def _both_ways(name: str, arity: int, loci: Iterable[tuple[int, ...]]) -> set[tuple[int, ...]]:
    out = set(loci)
    if arity == 2 and gates.is_symmetric(name):
        out |= {qubits[::-1] for qubits in out}
    return out


def _unlisted(
    arity: int,
    listed: set[tuple[int, ...]],
    num_qubits: int,
    connectivity: Mapping[str, Any],
) -> list[tuple[int, ...]]:
    if arity == 1:
        loci = [(q,) for q in range(num_qubits)]
    elif arity == 2:
        loci = [tuple(edge) for edge in connectivity["edges"]]
    else:
        return []
    return [locus for locus in loci if locus not in listed]


def _unpublished_note(
    name: str,
    arity: int,
    loci: Iterable[tuple[int, ...]],
    definition: Mapping[str, Any],
    disabled_qubits: set[int],
) -> str | None:
    if "avg_infidelity" not in definition:
        return None
    if arity == 2 and gates.is_symmetric(name):
        loci = (tuple(sorted(qubits)) for qubits in loci)
    shown = sorted({qubits for qubits in loci if not disabled_qubits.intersection(qubits)})
    if not shown:
        return None
    listed = [q for (q,) in shown] if arity == 1 else shown
    label = {1: "Qubits", 2: "Pairs"}.get(arity, "Qubit sets")
    return f"{label} {listed} have no {name} error; the device median applies to them."


def _is_sentinel(error: float | None, arity: int) -> bool:
    return error is not None and error >= metrics.max_avg_infidelity(arity)


def _one_per_pair(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the (b, a) record of a symmetric gate when it only repeats (a, b)."""
    by_locus = {tuple(r["qubits"]): r for r in records}
    out = []
    for r in records:
        a, b = r["qubits"]
        twin = by_locus.get((b, a))
        same = twin is not None and {**twin, "qubits": r["qubits"]} == r
        if not (same and a > b):
            out.append(r)
    return out


def _qubit_record(
    index: int, qubit: QubitCalibration, measure: Instruction | None
) -> dict[str, Any]:
    record: dict[str, Any] = {"index": index}
    if qubit.t1_us is not None:
        record["t1_us"] = _clean(qubit.t1_us)
    if qubit.t2_us is not None:
        record["t2_us"] = _clean(qubit.t2_us)
    error, duration = (None, None) if measure is None else (measure.error, measure.duration_ns)
    readout = {
        "p1_given_0": qubit.p1_given_0,
        "p0_given_1": qubit.p0_given_1,
        "error": error,
        "duration_ns": None if duration is None else _clean(duration),
    }
    readout = {key: value for key, value in readout.items() if value is not None}
    if readout:
        record["readout"] = readout
    if qubit.prep_error is not None:
        record["prep"] = {"error": qubit.prep_error}
    if not qubit.operational:
        record["disabled"] = True
    return record


def _connectivity(instructions: Iterable[Instruction]) -> dict[str, Any]:
    """Pairs of every 2-qubit instruction, directed if any 2-qubit gate is directional."""
    pairs = {inst.qubits for inst in instructions if len(inst.qubits) == 2}
    names = {inst.name for inst in instructions if len(inst.qubits) == 2}
    directed = any(not gates.is_symmetric(name) for name in names)
    if directed:
        return {"edges": sorted(pairs), "directed": True}
    return {"edges": sorted({(min(a, b), max(a, b)) for a, b in pairs}), "directed": False}


def _median_prep(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    errors = [r["prep"]["error"] for r in records if "prep" in r]
    return {"error": statistics.median(errors)} if errors else None


def _median_readout(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    readouts = [r["readout"] for r in records if "readout" in r]
    pairs = [r for r in readouts if "p1_given_0" in r]
    if pairs:
        out = {
            "p1_given_0": statistics.median(r["p1_given_0"] for r in pairs),
            "p0_given_1": statistics.median(r["p0_given_1"] for r in pairs),
        }
    elif readouts:
        out = {"error": statistics.median(r["error"] for r in readouts)}
    else:
        return None
    durations = [r["duration_ns"] for r in readouts if "duration_ns" in r]
    if durations:
        out["duration_ns"] = _clean(statistics.median(durations))
    return out


def _median_idle(records: Sequence[Mapping[str, Any]], ibm: bool) -> dict[str, Any] | None:
    idle: dict[str, Any] = {}
    for key in ("t1_us", "t2_us"):
        values = [r[key] for r in records if key in r]
        if values:
            idle[key] = _clean(statistics.median(values))
    if ibm:
        idle["t2_kind"] = "echo"  # IBM reports T2 from a Hahn echo
    return idle or None


# small shared helpers --------------------------------------------------------------------------


def _clean(value: float) -> float:
    """Twelve significant digits, without the float residue of s -> ns/us conversions.

    As a result, a calibration read from a Target or from BackendProperties gets one fingerprint.
    """
    return float(f"{value:.12g}")


def _seconds_to(value: float | None, unit: str) -> float | None:
    return None if value is None else units.convert(float(value), "s", unit)


def as_utc(value: str | date | datetime, *, name: str = "at") -> datetime:
    """A timezone-aware UTC datetime from a datetime, a date (midnight) or an ISO 8601 string.

    Naive means UTC. ``name`` is the parameter the value came from, for the error message.
    """
    try:  # a date's str() is its ISO form, so it parses as midnight
        stamp = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).strip())
    except ValueError:
        raise ValueError(
            f"{name}={value!r} is not an ISO 8601 date or time,"
            " such as '2025-06-01' or '2025-06-01T12:00:00Z'"
        ) from None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return stamp.astimezone(UTC)


def now_utc() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def sha256_bytes(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()
