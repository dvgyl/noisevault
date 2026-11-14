"""The calibration values of one device over a range of dates, as one long table.

:func:`history` collects the distinct calibrations of a range from one source. It gives one
:class:`Row` for each value. A value is a T1, T2 or readout value of a qubit, or an error or a
duration of a gate. Two calibrations with the same calibration time are one calibration.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from time import sleep
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, get_args

from . import metrics
from .errors import NoiseVaultError
from .profile import Profile, exact_ref, iso_z, write_atomically
from .sources.qiskit_backend import as_utc

if TYPE_CHECKING:
    from .catalog import ProfileInfo

Kind = Literal["t1", "t2", "p1_given_0", "p0_given_1", "gate_error", "gate_duration"]
Source = Literal["ibm", "vault", "archive"]
UNITS: Mapping[Kind, str] = MappingProxyType(
    {
        "t1": "us",
        "t2": "us",
        "p1_given_0": "",
        "p0_given_1": "",
        "gate_error": "",
        "gate_duration": "ns",
    }
)
COLUMNS = (
    "device",
    "calibrated_at",
    "fingerprint",
    "kind",
    "gate",
    "qubit_a",
    "qubit_b",
    "value",
    "unit",
)
MAX_STEPS = 400
PAUSE_S = 0.5
"""The wait between two requests to IBM's public endpoint."""

_KIND_ORDER = {kind: n for n, kind in enumerate(get_args(Kind))}
Progress = Callable[[int, int, datetime], None]


class HistoryError(NoiseVaultError, ValueError):
    """A range, step or source that :func:`history` cannot use."""


class StepFailed(NoiseVaultError):
    """One step of a range failed. ``at`` is the time of that step."""

    def __init__(self, *args: object, hint: str | None = None, at: datetime) -> None:
        super().__init__(*args, hint=hint)
        self.at = at


@dataclass(frozen=True)
class Row:
    device: str
    calibrated_at: datetime
    fingerprint: str
    kind: Kind
    gate: str | None
    qubit_a: int
    qubit_b: int | None
    value: float

    @property
    def unit(self) -> str:
        return UNITS[self.kind]

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "calibrated_at": iso_z(self.calibrated_at),
            "fingerprint": self.fingerprint,
            "kind": self.kind,
            "gate": self.gate,
            "qubit_a": self.qubit_a,
            "qubit_b": self.qubit_b,
            "value": self.value,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class History:
    device: str
    source: Source
    start: datetime
    end: datetime
    step: timedelta | None
    profiles: tuple[Profile, ...]
    rows: tuple[Row, ...]

    def summary(self) -> str:
        every = "" if self.step is None else f", every {describe_step(self.step)}"
        lines = [
            f"{self.device} from {iso_z(self.start)} to {iso_z(self.end)}{every} ({self.source})",
            f"{_count(len(self.profiles), 'calibration')}, {_count(len(self.rows), 'value')}",
        ]
        lines += [f"  {iso_z(_time(p))}  {p.short_fingerprint}" for p in self.profiles]
        return "\n".join(lines)

    __str__ = summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "source": self.source,
            "from": iso_z(self.start),
            "to": iso_z(self.end),
            "step_seconds": None if self.step is None else self.step.total_seconds(),
            "calibrations": [
                {
                    "calibrated_at": iso_z(_time(p)),
                    "fingerprint": p.short_fingerprint,
                    "ref": exact_ref(p.id, p.device.calibrated_at),
                }
                for p in self.profiles
            ],
            "columns": list(COLUMNS),
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, allow_nan=False)

    def to_csv(self, path: str | Path | None = None) -> str | None:
        """The table as CSV text, or, with ``path``, write the CSV to ``path`` and return None."""
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(COLUMNS)
        for row in self.rows:
            writer.writerow("" if value is None else value for value in row.to_dict().values())
        text = buffer.getvalue()
        if path is None:
            return text
        write_atomically(Path(path), text.encode())
        return None


def history(
    device: str,
    start: str | date | datetime,
    end: str | date | datetime,
    *,
    step: timedelta = timedelta(days=1),
    source: str = "ibm",
    archive: str | Path | None = None,
    progress: Progress | None = None,
) -> History:
    """The calibrations of ``device`` from ``start`` to ``end``, and a table of their values.

    ``source="ibm"`` asks IBM's public endpoint for the calibration in effect at each step from
    ``start`` to ``end``. ``source="archive"`` does the same with a local copy of the IBM
    calibration archive at ``archive``. ``source="vault"`` reads each profile that loads offline
    with a calibration time from ``start`` to ``end``, and ignores ``step``. A date or a time
    with no time zone is UTC. Before each step, ``history`` calls
    ``progress(step_number, steps, time)``.
    """
    name = device.strip().lower()
    begin, finish = as_utc(start, name="start"), as_utc(end, name="end")
    if finish < begin:
        raise HistoryError(f"the range ends ({iso_z(finish)}) before it starts ({iso_z(begin)})")
    kind = source.strip().lower()
    if kind not in get_args(Source):
        raise HistoryError(f"source={source!r} is not 'ibm', 'vault' or 'archive'")
    if (kind == "archive") != (archive is not None):
        raise HistoryError(
            "source='archive' needs archive=PATH"
            if archive is None
            else f"archive={str(archive)!r} needs source='archive'"
        )
    if kind == "vault":
        found = _saved(name, begin, finish)
        return History(name, kind, begin, finish, None, found, _table(found))
    times = steps(begin, finish, step)
    if kind == "ibm":
        from .sources import ibm_public

        found = _stepped(lambda at: ibm_public.pull(name, at=at), times, PAUSE_S, progress)
    else:
        from .sources.hf_archive import from_calibration_archive

        path = Path(str(archive))
        found = _stepped(lambda at: from_calibration_archive(path, name, at=at), times, 0, progress)
    return History(name, kind, begin, finish, step, found, _table(found))


def steps(start: datetime, end: datetime, step: timedelta) -> list[datetime]:
    """Each time from ``start`` to ``end``, ``step`` apart. Refuses more than MAX_STEPS."""
    if step <= timedelta(0):
        raise HistoryError(f"the step must be more than 0, not {step}")
    count = (end - start) // step + 1
    if count > MAX_STEPS:
        raise HistoryError(
            f"the range from {iso_z(start)} to {iso_z(end)} has {count} steps of"
            f" {describe_step(step)}, and the limit is {MAX_STEPS}",
            hint=f"use a step of {describe_step(_fitting_step(end - start))} or more",
        )
    return [start + n * step for n in range(count)]


def describe_step(step: timedelta) -> str:
    seconds = step.total_seconds()
    for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60)):
        if seconds % size == 0:
            return _count(int(seconds // size), unit)
    return _count(round(seconds), "second")


def _fitting_step(span: timedelta) -> timedelta:
    hours = math.ceil(span / timedelta(hours=1) / (MAX_STEPS - 1))
    return timedelta(days=math.ceil(hours / 24)) if hours > 12 else timedelta(hours=hours)


def _stepped(
    pull_at: Callable[[datetime], Profile],
    times: list[datetime],
    pause_s: float,
    progress: Progress | None,
) -> tuple[Profile, ...]:
    by_time: dict[float, Profile] = {}
    for number, at in enumerate(times, 1):
        if progress is not None:
            progress(number, len(times), at)
        if number > 1 and pause_s:
            sleep(pause_s)
        try:
            profile = pull_at(at)
        except NoiseVaultError as exc:
            raise _failed(number, len(times), at, exc) from exc
        if profile.device.calibrated_at is None:
            raise StepFailed(
                f"step {number} of {len(times)} ({iso_z(at)}) gave a calibration with no time",
                at=at,
            )
        by_time.setdefault(profile.device.calibrated_at.timestamp(), profile)
    return tuple(by_time[key] for key in sorted(by_time))


def _failed(number: int, count: int, at: datetime, exc: NoiseVaultError) -> StepFailed:
    from .sources.ibm_public import NotServed

    hint = exc.hint
    if isinstance(exc, NotServed):
        hint = (
            "start the range at a later date"
            if exc.listed
            else "the public endpoint has no data for a retired device; use source='vault'"
            " for the profiles that load offline, or source='archive' for an archive file"
        )
    return StepFailed(
        f"step {number} of {count} ({iso_z(at)}) failed: {exc.message}", hint=hint, at=at
    )


def _saved(name: str, start: datetime, end: datetime) -> tuple[Profile, ...]:
    from .catalog import profiles

    held = [
        (info.calibrated_at, info)
        for info in profiles()
        if info.id == name and info.calibrated_at is not None
    ]
    if not held:
        raise HistoryError(
            f"no profile of {name} loads offline",
            hint=f"pull calibrations with nv pull {name} --at DATE, or use source='ibm'",
        )
    by_time: dict[datetime, ProfileInfo] = {}
    for when, info in sorted(held, key=lambda pair: pair[1].location != "vault"):
        if start <= when <= end:
            by_time.setdefault(when, info)
    if not by_time:
        times = sorted({iso_z(when) for when, _ in held})
        where = (
            f"the only calibration time of {name} that loads offline is {times[0]}"
            if len(times) == 1
            else f"the {name} profiles that load offline are from {times[0]} to {times[-1]}"
        )
        raise HistoryError(
            f"no profile of {name} has a calibration time from {iso_z(start)} to {iso_z(end)}",
            hint=f"{where}; run nv list to see them",
        )
    return tuple(by_time[key].load() for key in sorted(by_time))


def _table(found: tuple[Profile, ...]) -> tuple[Row, ...]:
    rows = [row for profile in found for row in _rows(profile)]
    return tuple(
        sorted(
            rows,
            key=lambda r: (
                r.calibrated_at,
                _KIND_ORDER[r.kind],
                r.gate or "",
                r.qubit_a,
                -1 if r.qubit_b is None else r.qubit_b,
            ),
        )
    )


def _rows(profile: Profile) -> list[Row]:
    base = (profile.id, _time(profile), profile.short_fingerprint)
    rows: list[Row] = []
    for q in profile.qubits:
        values: list[tuple[Kind, float | None]] = [("t1", q.t1_us), ("t2", q.t2_us)]
        if q.readout is not None:
            p1_given_0, p0_given_1 = q.readout.pair
            values += [("p1_given_0", p1_given_0), ("p0_given_1", p0_given_1)]
        rows += [Row(*base, k, None, q.index, None, v) for k, v in values if v is not None]
    for record in profile.calibrations:
        if len(record.qubits) > 2:
            continue
        a, b = record.qubits[0], record.qubits[1] if len(record.qubits) == 2 else None
        metric = record.metric
        if metric is not None:
            error = metrics.to_avg_infidelity(*metric, len(record.qubits))
            rows.append(Row(*base, "gate_error", record.gate, a, b, error))
        duration = record.duration_ns
        if duration is None and record.gate in profile.gates:
            duration = profile.gates[record.gate].duration_ns
        if duration is not None and not record.disabled:
            rows.append(Row(*base, "gate_duration", record.gate, a, b, duration))
    return rows


def _time(profile: Profile) -> datetime:
    when = profile.device.calibrated_at
    if when is None:
        raise HistoryError(f"{profile.id} has no calibration time")
    return when


def _count(n: int, noun: str) -> str:
    return f"{n:,} {noun}{'' if n == 1 else 's'}"
