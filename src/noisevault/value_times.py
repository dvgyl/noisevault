"""The measurement times of the values of a calibration, and their ages.

A source can give a calibration one time and each value its own time. IBM does this: the
calibration time is ``last_update_date``, and a value keeps the time of its last measurement. An
importer keeps the oldest, median and newest time of each kind of value in
``provenance.extra["value_times"]``. The fingerprint does not include provenance, so the times do
not change it.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .profile import ValueAge, iso_z

if TYPE_CHECKING:
    from .profile import Profile

KEY = "value_times"
KINDS = ("readout", "t1", "t2", "error_1q", "error_2q")
LABELS = {
    "readout": "readout",
    "t1": "T1",
    "t2": "T2",
    "error_1q": "1q gates",
    "error_2q": "2q gates",
}
OLD_HOURS = 24.0
"""A median age above this value gives a provenance note at import."""

_ENDS = ("oldest", "median", "newest")


def summary(times: Mapping[str, Sequence[datetime]]) -> dict[str, dict[str, Any]]:
    """For each kind with times, the oldest, median and newest time and the number of values."""
    out: dict[str, dict[str, Any]] = {}
    for kind in KINDS:
        found = sorted(times.get(kind, ()))
        if not found:
            continue
        middle = round(statistics.median(t.timestamp() for t in found))
        out[kind] = {
            "oldest": iso_z(found[0]),
            "median": iso_z(datetime.fromtimestamp(middle, UTC)),
            "newest": iso_z(found[-1]),
            "values": len(found),
        }
    return out


def value_ages(profile: Profile, at: datetime | None = None) -> dict[str, ValueAge] | None:
    """The age in hours of each kind at ``at``, or None when the profile has no times.

    ``at`` defaults to the calibration time. A kind whose entry does not have the shape that
    :func:`summary` writes is left out.
    """
    stored = profile.provenance.extra.get(KEY)
    if not isinstance(stored, Mapping):
        return None
    if at is None:
        at = profile.device.calibrated_at
        if at is None:
            raise ValueError("the profile has no calibration time, so give the time at=")
    ages = {kind: _ages(stored.get(kind), at) for kind in KINDS}
    found = {kind: age for kind, age in ages.items() if age is not None}
    return found or None


def old_note(times: Mapping[str, Any], calibrated_at: datetime | None) -> str | None:
    """One note when the median age of a kind at ``calibrated_at`` is above OLD_HOURS."""
    if calibrated_at is None:
        return None
    ages = {kind: _ages(times.get(kind), calibrated_at) for kind in KINDS}
    old = [
        f"{LABELS[kind]} {hours(age['median_h'])}"
        for kind, age in ages.items()
        if age is not None and age["median_h"] > OLD_HOURS
    ]
    if not old:
        return None
    return (
        f"Some values are older than the calibration time by more than {OLD_HOURS:g} h (median):"
        f" {', '.join(old)}. The calibration time is the last update of the calibration, not the"
        " measurement time of each value."
    )


def hours(value: float) -> str:
    """An age such as ``151 h`` or ``2.5 h``."""
    return f"{value:.0f} h" if abs(value) >= 10 else f"{value:.1f} h"


def _ages(entry: Any, at: datetime) -> ValueAge | None:
    if not isinstance(entry, Mapping):
        return None
    count = entry.get("values")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        return None
    try:
        times = [_utc(entry.get(end)) for end in _ENDS]
    except (TypeError, ValueError, OverflowError):
        return None
    oldest, median, newest = ((at - t).total_seconds() / 3600 for t in times)
    return ValueAge(oldest_h=oldest, median_h=median, newest_h=newest, values=count)


def _utc(stamp: Any) -> datetime:
    when = datetime.fromisoformat(stamp)  # raises TypeError for a value that is not a string
    if when.tzinfo is None:
        raise ValueError(f"{stamp} has no timezone")
    return when.astimezone(UTC)
