"""Profiles made from other profiles: errors scaled by group, and typical values of a history."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from statistics import median
from typing import Any

from . import metrics
from .errors import qubit_loci
from .profile import Profile, iso_z
from .table import GateNoise, QubitNoise

_READOUT_KEYS = ("error", "p1_given_0", "p0_given_1")


def scaled(
    profile: Profile, *, one_qubit: float = 1.0, two_qubit: float = 1.0, readout: float = 1.0
) -> Profile:
    """``profile`` with each stated error multiplied by the factor of its group.

    ``one_qubit`` multiplies the error metric of each 1-qubit gate, ``two_qubit`` the metric of
    each 2-qubit gate, and ``readout`` each readout probability. A Pauli metric has each entry
    multiplied. A stated standard error is multiplied with its value. A value of exactly 0 stays
    0. Gates on 3 or more qubits, T1, T2, durations, preparation errors, effects and
    ``unmodeled_error`` do not change. Two kinds of value also stay as stated, because they mark
    a failure and not a level of noise: a gate error at or past full depolarization, and the
    readout of a qubit with a P(1|0) or P(0|1) of 0.5 or more. ``unusable_qubits`` names such
    qubits.

    A factor of 1.0 for each group gives ``profile`` itself. Any other factors give a new
    fingerprint, ``provenance.derived_from`` set to the fingerprint of ``profile``, and a note
    with the factors and the number of failed values that stay as stated. A factor that puts a
    value outside the valid range of its field raises ValueError, which names the value and the
    largest valid factor of the group.
    """
    factors = {"one_qubit": one_qubit, "two_qubit": two_qubit, "readout": readout}
    for name, factor in factors.items():
        if isinstance(factor, bool) or not isinstance(factor, int | float):
            raise TypeError(f"{name} must be a number, got {factor!r}")
        if not math.isfinite(factor) or factor < 0:
            raise ValueError(f"{name} must be a finite number of 0 or more, got {factor!r}")
    if all(factor == 1 for factor in factors.values()):
        return profile
    if profile.unmodeled_error is not None and profile.unmodeled_error.fit is not None:
        raise ValueError(
            "the unmodeled_error factors of this profile are fitted to its calibration, so they"
            " do not apply to scaled errors. Scale profile.uncorrected() instead"
        )
    table = profile.table
    data = profile.to_dict()
    limits: dict[str, list[tuple[float, str]]] = {name: [] for name in factors}
    group = {1: "one_qubit", 2: "two_qubit"}
    kept = 0

    def gate(spec: dict[str, Any], arity: int | None, where: str) -> dict[str, Any]:
        nonlocal kept
        name = group.get(arity or 0)
        stated = _metric_of(spec)
        if name is None or stated is None or factors[name] == 1:
            return spec
        kind, value = stated
        if metrics.unscalable(kind, value, arity or 0):
            kept += 1
            return spec
        total = math.fsum(value) if kind == "pauli" else value
        upper = 1.0 if kind in ("pauli", "process_infidelity") else _upper(kind, arity or 0)
        if total > 0:
            limits[name].append((upper / total, f"{where} ({kind} {total:.3g})"))
        out = dict(spec)
        factor = factors[name]
        out[kind] = [p * factor for p in value] if kind == "pauli" else value * factor
        if "stderr" in out:
            out["stderr"] = out["stderr"] * factor
        return out

    data["gates"] = {
        name: gate(spec, table.arity(name), f"the default {name} error")
        for name, spec in data["gates"].items()
    }
    if "calibrations" in data:
        data["calibrations"] = [
            gate(r, len(r["qubits"]), f"{r['gate']} on {qubit_loci(tuple(r['qubits']))}")
            for r in data["calibrations"]
        ]

    def read(spec: dict[str, Any], where: str) -> dict[str, Any]:
        nonlocal kept
        if any(spec.get(key, 0) >= 0.5 for key in _READOUT_KEYS):
            kept += 1
            return spec
        out = dict(spec)
        for key in _READOUT_KEYS:
            if key in out:
                if out[key] > 0:
                    limits["readout"].append((1 / out[key], f"{key} of {where} ({out[key]:.3g})"))
                out[key] = out[key] * readout
        return out

    if readout != 1:
        if "readout" in data:
            data["readout"] = read(data["readout"], "the default readout")
        for q in data.get("qubits", ()):
            if "readout" in q:
                q["readout"] = read(q["readout"], f"qubit {q['index']}")

    for name, factor in factors.items():
        if not limits[name]:
            continue
        largest, where = min(limits[name])
        if factor > largest:
            raise ValueError(
                f"{name}={factor:g} puts {where} outside its valid range. The largest valid"
                f" {name} factor for this profile is {_floor(largest)}"
            )

    provenance = data.setdefault("provenance", {})
    provenance["source_kind"] = "derived"
    provenance["data_kind"] = "hypothetical"
    provenance["derived_from"] = f"sha256:{profile.fingerprint}"
    note = (
        f"The stated errors are multiplied by {one_qubit:g} for 1-qubit gates, {two_qubit:g} for"
        f" 2-qubit gates and {readout:g} for readout."
    )
    if kept:
        values = (
            "1 failed gate or readout stays"
            if kept == 1
            else f"{kept} failed gates or readouts stay"
        )
        note += f" {values} as stated."
    provenance["notes"] = [*provenance.get("notes", ()), note]
    return Profile.model_validate(data)


def typical(profiles: Sequence[Profile], *, structure: Profile | None = None) -> Profile:
    """One profile with the median of each value over ``profiles``, calibrations of one device.

    A value is the T1, T2, P(1|0) or P(0|1) of a qubit, or the average infidelity of a gate on
    one locus. ``structure`` gives the records, the connectivity, the disabled marks and the
    durations. By default, ``structure`` is the last profile, so give the profiles oldest first.
    Only the records of ``structure`` for one qubit or one locus change. Device-wide default
    values stay as ``structure`` states them.

    A profile counts for a value only when it states the value and does not disable the gate or
    the qubit. A profile that is in ``profiles`` two times counts two times. A value that no
    profile states stays as ``structure`` states it. The medians use the stated values, so the
    result has no ``unmodeled_error``. A provenance note gives the number of profiles and the
    range of their calibration times, and ``provenance.derived_from`` names ``structure``.
    """
    if not profiles:
        raise ValueError("typical needs at least one profile")
    base = (profiles[-1] if structure is None else structure).uncorrected()
    other = next((p.id for p in profiles if p.id != base.id), None)
    if other is not None:
        raise ValueError(f"typical needs profiles of one device, got {base.id} and {other}")
    every = [p.uncorrected() for p in profiles]
    data = base.to_dict()

    def over(read: Callable[[Profile], float | None]) -> float | None:
        found = [value for value in map(read, every) if value is not None]
        return median(found) if found else None

    def gate(name: str, qubits: tuple[int, ...]) -> Callable[[Profile], float | None]:
        def read(p: Profile) -> float | None:
            if name not in p.gates or any(q >= p.device.num_qubits for q in qubits):
                return None
            found = p.table.gate(name, qubits)
            calibrated = isinstance(found, GateNoise) and found.state == "calibrated"
            return found.avg_infidelity if calibrated else None

        return read

    def qubit(
        index: int, pick: Callable[[QubitNoise], float | None]
    ) -> Callable[[Profile], float | None]:
        def read(p: Profile) -> float | None:
            if index >= p.device.num_qubits or p.table.qubit(index).disabled:
                return None
            return pick(p.table.qubit(index))

        return read

    records = []
    for record in data.get("calibrations", ()):
        middle = None
        if not record.get("disabled"):
            middle = over(gate(record["gate"], tuple(record["qubits"])))
        records.append(record if middle is None else _with_metric(record, middle))
    if records:
        data["calibrations"] = records
    for q in data.get("qubits", ()):
        index = q["index"]
        for key, pick in (
            ("t1_us", lambda found: None if found.t1_ns is None else found.t1_ns / 1000),
            ("t2_us", lambda found: None if found.t2_ns is None else found.t2_ns / 1000),
        ):
            middle = over(qubit(index, pick))
            if key in q and middle is not None:
                q[key] = middle
        if "readout" in q:
            low = over(qubit(index, lambda found: found.readout and found.readout[0]))
            high = over(qubit(index, lambda found: found.readout and found.readout[1]))
            if low is not None and high is not None:
                q["readout"] = _readout(q["readout"], low, high)

    provenance = data.setdefault("provenance", {})
    provenance["source_kind"] = "derived"
    provenance["derived_from"] = f"sha256:{base.fingerprint}"
    provenance["notes"] = [*provenance.get("notes", ()), _typical_note(profiles)]
    return Profile.model_validate(data)


def _typical_note(profiles: Sequence[Profile]) -> str:
    times = sorted(p.device.calibrated_at for p in profiles if p.device.calibrated_at is not None)
    count = "1 profile" if len(profiles) == 1 else f"{len(profiles)} profiles"
    note = f"Each T1, T2, readout probability and gate error is the median over {count}"
    if not times:
        return f"{note}, none with a calibration time."
    undated = len(profiles) - len(times)
    span = iso_z(times[0]) if times[0] == times[-1] else f"{iso_z(times[0])} to {iso_z(times[-1])}"
    return f"{note}, calibrated {span}" + (f", and {undated} undated." if undated else ".")


def _metric_of(spec: Mapping[str, Any]) -> tuple[str, Any] | None:
    for key in metrics.METRIC_KEYS:
        if key in spec:
            return key, spec[key]
    return None


def _upper(kind: str, arity: int) -> float:
    if kind == "avg_infidelity":
        return metrics.max_avg_infidelity(arity)
    return metrics.max_depolarizing_param(arity)


def _floor(value: float) -> str:
    """``value`` rounded down to 4 significant digits, so that the shown factor is valid."""
    digits = 3 - math.floor(math.log10(value))
    return f"{math.floor(value * 10**digits) / 10**digits:g}"


def _with_metric(spec: Mapping[str, Any], avg_infidelity: float) -> dict[str, Any]:
    kept = {k: v for k, v in spec.items() if k not in metrics.METRIC_KEYS}
    return kept | {"avg_infidelity": avg_infidelity}


def _readout(readout: Mapping[str, Any], low: float, high: float) -> dict[str, Any]:
    kept = {k: v for k, v in readout.items() if k == "duration_ns"}
    if low == high:
        return kept | {"error": low}
    return kept | {"p1_given_0": low, "p0_given_1": high}
