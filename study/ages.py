"""The age of each value of an IBM calibration.

IBM gives one time for a calibration (``last_update_date``) and one time for each value in it.
A value can be much older than the calibration: a calibration from today can hold a gate error
that IBM measured two weeks ago. ``value_ages`` gives the age of each kind of value at a
reference time, for the whole device or for one chain of qubits.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

KINDS = ("readout", "t1", "t2", "one_qubit", "two_qubit")
_QUBIT_KINDS = {
    "T1": "t1",
    "T2": "t2",
    "readout_error": "readout",
    "prob_meas0_prep1": "readout",
    "prob_meas1_prep0": "readout",
}
# A virtual gate has no measured error. IBM gives its error of 0 the time of the calibration.
_VIRTUAL = frozenset({"rz"})


def value_times(
    properties: Mapping[str, Any], qubits: Sequence[int] | None = None
) -> dict[str, list[datetime]]:
    """The measurement time of each value, by kind.

    ``properties`` is the reply of IBM's properties endpoint. With ``qubits``, the result holds
    only the values of those qubits and of the gates that act on those qubits alone.
    """
    wanted = None if qubits is None else set(qubits)
    times: dict[str, list[datetime]] = {kind: [] for kind in KINDS}
    for index, entries in enumerate(properties.get("qubits") or []):
        if wanted is not None and index not in wanted:
            continue
        for entry in entries:
            kind = _QUBIT_KINDS.get(entry.get("name"))
            when = _time(entry.get("date"))
            if kind is not None and when is not None:
                times[kind].append(when)
    for gate in properties.get("gates") or []:
        acts_on = gate.get("qubits") or []
        if gate.get("gate") in _VIRTUAL or len(acts_on) not in (1, 2):
            continue
        if wanted is not None and not set(acts_on) <= wanted:
            continue
        kind = "one_qubit" if len(acts_on) == 1 else "two_qubit"
        for entry in gate.get("parameters") or []:
            when = _time(entry.get("date"))
            if entry.get("name") == "gate_error" and when is not None:
                times[kind].append(when)
    return times


def value_ages(
    properties: Mapping[str, Any], at: datetime, qubits: Sequence[int] | None = None
) -> dict[str, dict[str, float | int] | None]:
    """For each kind, the youngest, median and oldest age in hours at the time ``at``.

    A kind with no dated value gives None. An age is negative when IBM measured the value
    after ``at``.
    """
    ages: dict[str, dict[str, float | int] | None] = {}
    for kind, times in value_times(properties, qubits).items():
        hours = [_hours(at, when) for when in times]
        ages[kind] = _summary(hours) if hours else None
    return ages


def calibration_age(properties: Mapping[str, Any], at: datetime) -> float | None:
    """The age in hours of the calibration time (``last_update_date``) at the time ``at``."""
    when = _time(properties.get("last_update_date"))
    return None if when is None else _hours(at, when)


def _summary(hours: Iterable[float]) -> dict[str, float | int]:
    ordered = sorted(hours)
    return {
        "values": len(ordered),
        "youngest_h": ordered[0],
        "median_h": statistics.median(ordered),
        "oldest_h": ordered[-1],
    }


def _hours(at: datetime, when: datetime) -> float:
    return (at - when).total_seconds() / 3600


def _time(stamp: Any) -> datetime | None:
    if not isinstance(stamp, str):
        return None
    try:
        when = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return when.replace(tzinfo=UTC) if when.tzinfo is None else when.astimezone(UTC)
