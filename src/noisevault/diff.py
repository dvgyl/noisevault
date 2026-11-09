"""Drift between two profiles: device-wide medians, the largest per-qubit and per-pair changes,
gates and qubits that became disabled or re-enabled, and qubits added or removed.

Every value is the resolved one (a record, else the device default), looked up in both profiles
for the same locus. The typical 1-qubit and 2-qubit errors are those of the native gate that the
exports use on that locus (:meth:`NoiseTable.typical`), on each ordered pair. A pair whose two
directions agree in both profiles is one row. On all-to-all devices, each pair with a 2-qubit
record is one row. All other pairs carry the device default and share one "default" row, which
counts once per pair in the medians. A disabled gate definition shows as "<gate> default".
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import combinations
from math import comb
from typing import TYPE_CHECKING, Any

from .errors import plural
from .profile import DeviceMedians, qubit_medians, ref_on_day
from .table import GateNoise

if TYPE_CHECKING:
    from .profile import Profile
    from .table import NoiseTable

Pair = tuple[int, int]
DEFAULT = "default"  # all-to-all pairs without a 2-qubit record
# metric name, unit for display, and whether a larger value is better
METRICS: dict[str, tuple[str, bool]] = {
    "t1_us": ("T1 (us)", True),
    "t2_us": ("T2 (us)", True),
    "error_1q": ("1q avg infidelity", False),
    "error_2q": ("2q avg infidelity", False),
    "readout_error": ("readout error", False),
}


@dataclass(frozen=True)
class Change:
    """One value before and after. ``where`` is "" for a device median, else the qubit or pair."""

    metric: str
    before: float | None
    after: float | None
    where: str = ""

    @property
    def relative(self) -> float | None:
        if self.before is None or self.after is None or self.before == 0:
            return None
        return (self.after - self.before) / self.before

    @property
    def worse(self) -> bool:
        """True when the change makes the device noisier."""
        if self.before is None or self.after is None:
            return False
        if METRICS[self.metric][1]:
            return self.after < self.before
        return self.after > self.before

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "where": self.where or None,
            "before": self.before,
            "after": self.after,
            "relative": self.relative,
        }


@dataclass(frozen=True)
class ProfileDiff:
    before: str  # ref of the first profile
    after: str
    before_fingerprint: str
    after_fingerprint: str
    time_delta: timedelta | None
    medians: tuple[Change, ...]
    qubits: tuple[Change, ...]  # largest per-qubit relative changes, largest first
    pairs: tuple[Change, ...]
    newly_disabled: tuple[str, ...]
    reenabled: tuple[str, ...]
    pairs_added: tuple[str, ...]  # pairs with a 2-qubit gate only in the second profile
    pairs_removed: tuple[str, ...]
    qubits_added: tuple[int, ...]
    qubits_removed: tuple[int, ...]
    warnings: tuple[str, ...]

    @property
    def identical(self) -> bool:
        return self.before_fingerprint == self.after_fingerprint

    def summary(self) -> str:
        lines = [f"{self.before} -> {self.after} ({describe_delta(self.time_delta)})"]
        lines += [f"warning: {w}" for w in self.warnings]
        if self.identical:
            lines.append("same physics: the fingerprints are equal")
            return "\n".join(lines)
        lines.append(f"  {'device median':<22}{'before':>12}{'after':>12}{'change':>10}")
        for c in self.medians:
            lines.append(
                f"  {METRICS[c.metric][0]:<22}{fmt_metric(c.metric, c.before):>12}"
                f"{fmt_metric(c.metric, c.after):>12}{fmt_relative(c.relative):>10}"
            )
        for title, changes in (("qubits", self.qubits), ("pairs", self.pairs)):
            if changes:
                lines.append(f"largest changes by {title[:-1]}:")
                lines += [
                    f"  {_where(title, c.where):<14} {METRICS[c.metric][0]:<17}"
                    f" {fmt_metric(c.metric, c.before)} -> {fmt_metric(c.metric, c.after)}"
                    f" ({fmt_relative(c.relative)})"
                    for c in changes
                ]
        for label, items in (
            ("newly disabled", self.newly_disabled),
            ("re-enabled", self.reenabled),
            ("pairs added", self.pairs_added),
            ("pairs removed", self.pairs_removed),
            ("qubits added", tuple(map(str, self.qubits_added))),
            ("qubits removed", tuple(map(str, self.qubits_removed))),
        ):
            if items:
                lines.append(f"{label}: {', '.join(items)}")
        return "\n".join(lines)

    __str__ = summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "before": self.before,
            "after": self.after,
            "before_fingerprint": self.before_fingerprint,
            "after_fingerprint": self.after_fingerprint,
            "identical": self.identical,
            "time_delta_seconds": None
            if self.time_delta is None
            else self.time_delta.total_seconds(),
            "medians": [c.to_dict() for c in self.medians],
            "qubits": [c.to_dict() for c in self.qubits],
            "pairs": [c.to_dict() for c in self.pairs],
            "newly_disabled": list(self.newly_disabled),
            "reenabled": list(self.reenabled),
            "pairs_added": list(self.pairs_added),
            "pairs_removed": list(self.pairs_removed),
            "qubits_added": list(self.qubits_added),
            "qubits_removed": list(self.qubits_removed),
            "warnings": list(self.warnings),
        }


def diff(a: Profile, b: Profile, *, top: int = 5) -> ProfileDiff:
    """How ``b`` differs from ``a``. ``top`` limits the per-qubit and per-pair lists."""
    if top < 0:
        raise ValueError(f"top={top}: pass 0 or more")
    warnings = []
    if a.id != b.id:
        warnings.append(
            f"the two profiles are from different devices ({a.id} and {b.id}), so the diff"
            " matches qubits and pairs by index"
        )
    when_a, when_b = a.device.calibrated_at, b.device.calibrated_at
    delta = when_b - when_a if when_a and when_b else None
    if a.id == b.id and delta is not None and delta < timedelta(0):
        first, second = (f"{a.id}@{stamp}" for stamp in distinguishing_stamps(when_a, when_b))
        warnings.append(
            f"{second} is older than {first}. Before and after follow the argument order, not time"
        )

    qa, qb = _qubit_values(a.table), _qubit_values(b.table)
    ma, mb = _medians(a, qa), _medians(b, qb)
    medians = tuple(Change(m, ma[m], mb[m]) for m in METRICS)
    pa, pb = _pair_values(a.table, b.table)
    newly_disabled, reenabled = _availability(a, b)
    pairs_added, pairs_removed = _pair_changes(a.table, b.table)
    common = range(min(a.device.num_qubits, b.device.num_qubits))
    return ProfileDiff(
        before=ref_on_day(a.id, a.device.calibrated_at),
        after=ref_on_day(b.id, b.device.calibrated_at),
        before_fingerprint=a.fingerprint,
        after_fingerprint=b.fingerprint,
        time_delta=delta,
        medians=medians,
        qubits=_largest(qa, qb, lambda q: str(q), top),
        pairs=_largest(pa, pb, lambda p: p if p == DEFAULT else f"{p[0]}-{p[1]}", top),
        newly_disabled=newly_disabled,
        reenabled=reenabled,
        pairs_added=pairs_added,
        pairs_removed=pairs_removed,
        qubits_added=tuple(q for q in range(b.device.num_qubits) if q not in common),
        qubits_removed=tuple(q for q in range(a.device.num_qubits) if q not in common),
        warnings=tuple(warnings),
    )


# values -------------------------------------------------------------------------------------


def _qubit_values(table: NoiseTable) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for i in range(table.num_qubits):
        q = table.qubit(i)
        if q.disabled:
            continue
        values = {
            "t1_us": None if q.t1_ns is None else q.t1_ns / 1000,
            "t2_us": None if q.t2_ns is None else q.t2_ns / 1000,
            "error_1q": _error(table.typical(1, (i,))),
            "readout_error": None if q.readout is None else sum(q.readout) / 2,
        }
        out[i] = {k: v for k, v in values.items() if v is not None}
    return out


def _pair_values(
    a: NoiseTable, b: NoiseTable
) -> tuple[dict[Pair | str, dict[str, float]], dict[Pair | str, dict[str, float]]]:
    """The typical 2-qubit error on the same loci of ``a`` and ``b``.

    The values of a profile leave out each locus with no usable 2-qubit gate, so the diff does
    not compare that locus. The diff reports a disabled or re-enabled gate as availability, and
    a pair that one profile does not have as an added or removed pair, not as drift.
    """
    listed, rest, _ = _pair_loci((a, b), _shared_qubits(a, b))
    before: dict[Pair | str, dict[str, float]] = {}
    after: dict[Pair | str, dict[str, float]] = {}
    for pair in listed:
        x, y = _entries(a, pair), _entries(b, pair)
        loci = [pair] if x[0] == x[1] and y[0] == y[1] else [pair, (pair[1], pair[0])]
        for i, locus in enumerate(loci):
            _put(before, locus, x[i])
            _put(after, locus, y[i])
    if rest is not None:
        _put(before, DEFAULT, _entries(a, rest)[0])
        _put(after, DEFAULT, _entries(b, rest)[0])
    return before, after


def _shared_qubits(a: NoiseTable, b: NoiseTable) -> list[int]:
    """The qubits that both profiles have and enable."""
    common = range(min(a.num_qubits, b.num_qubits))
    return [q for q in common if not a.qubit(q).disabled and not b.qubit(q).disabled]


def _pair_changes(a: NoiseTable, b: NoiseTable) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The pairs with a 2-qubit gate in ``b`` only, and the pairs with one in ``a`` only.

    A pair has a 2-qubit gate when a native 2-qubit gate resolves on it in one direction, also
    a disabled gate. Thus a coupler that the connectivity gains or loses is here, and a
    disabled coupler is in the availability lists. A pair of a qubit that one profile disables
    or does not have is not here, because the diff lists that qubit.
    """
    listed, _, _ = _pair_loci((a, b), _shared_qubits(a, b))
    had, has = ({pair for pair in listed if _has_gate(t, pair)} for t in (a, b))
    return (
        tuple(f"{x}-{y}" for x, y in sorted(has - had)),
        tuple(f"{x}-{y}" for x, y in sorted(had - has)),
    )


def _has_gate(table: NoiseTable, pair: Pair) -> bool:
    return any(
        isinstance(table.gate(gate, locus), GateNoise)
        for gate in table.natives(2)
        for locus in (pair, pair[::-1])
    )


def _pair_errors(table: NoiseTable) -> list[float]:
    """The typical 2-qubit error of every usable ordered pair."""
    usable = [q for q in range(table.num_qubits) if not table.qubit(q).disabled]
    listed, rest, count = _pair_loci((table,), usable)
    errors = [e for pair in listed for e in _directions(table, pair)]
    if rest is not None:
        errors += list(_directions(table, rest)) * count
    return [e for e in errors if e is not None]


def _pair_loci(
    tables: tuple[NoiseTable, ...], qubits: list[int]
) -> tuple[list[Pair], Pair | None, int]:
    """Pairs among ``qubits`` to look up, as (a, b) with a < b.

    Every connected pair and every pair with a 2-qubit record in one of ``tables``. On
    all-to-all devices: the recorded pairs, one pair that stands for all other pairs, and the
    count of those other pairs. The other pairs all resolve to the device default.
    """
    allowed = set(qubits)
    listed = sorted({p for t in tables for p in t.listed_pairs() if allowed.issuperset(p)})
    if not all(t.all_to_all for t in tables):
        return listed, None, 0
    taken = set(listed)
    rest = next((p for p in combinations(qubits, 2) if p not in taken), None)
    return listed, rest, comb(len(qubits), 2) - len(listed)


def _directions(table: NoiseTable, pair: Pair) -> tuple[float | None, float | None]:
    a, b = pair
    return _error(table.typical(2, (a, b))), _error(table.typical(2, (b, a)))


def _entries(table: NoiseTable, pair: Pair) -> list[dict[str, float] | None]:
    """Per direction: the typical error, {} when only uncalibrated gates are usable, or None."""
    out: list[dict[str, float] | None] = []
    for locus, error in zip((pair, pair[::-1]), _directions(table, pair), strict=True):
        if error is not None:
            out.append({"error_2q": error})
        else:
            usable = any(table.allowed(g, locus) for g in table.natives(2))
            out.append({} if usable else None)
    return out


def _put(
    values: dict[Pair | str, dict[str, float]], locus: Pair | str, entry: dict[str, float] | None
) -> None:
    if entry is not None:
        values[locus] = entry


def _error(found: Any) -> float | None:
    return found.avg_infidelity if isinstance(found, GateNoise) else None


def device_medians(profile: Profile) -> DeviceMedians:
    """See :meth:`noisevault.Profile.medians`."""
    found = _medians(profile, _qubit_values(profile.table))
    return DeviceMedians(
        t1_us=found["t1_us"],
        t2_us=found["t2_us"],
        error_1q=found["error_1q"],
        error_2q=found["error_2q"],
        readout_error=found["readout_error"],
    )


def _medians(profile: Profile, qubits: dict[int, dict[str, float]]) -> dict[str, float | None]:
    shown = qubit_medians(profile)
    return {
        "t1_us": shown.t1_us,
        "t2_us": shown.t2_us,
        "error_1q": _median([v["error_1q"] for v in qubits.values() if "error_1q" in v]),
        "error_2q": _median(_pair_errors(profile.table)),
        "readout_error": shown.readout_error,
    }


def _median(found: list[float]) -> float | None:
    return statistics.median(found) if found else None


def _largest(
    before: dict[Any, dict[str, float]],
    after: dict[Any, dict[str, float]],
    label: Callable[[Any], str],
    top: int,
) -> tuple[Change, ...]:
    moved = [
        (key, Change(metric, before[key].get(metric), after[key].get(metric), label(key)))
        for key in before.keys() & after.keys()
        for metric in METRICS
        if before[key].get(metric) != after[key].get(metric)
    ]
    moved = _collapse_uniform(moved, before.keys() & after.keys())
    moved.sort(key=lambda kc: (*_size(kc[1]), *_order(kc[0]), kc[1].metric))
    return tuple(c for _, c in moved[:top])


def _size(change: Change) -> tuple[bool, float]:
    """Largest first. A value that appears, disappears or leaves zero comes before any ratio."""
    rel = change.relative
    return (False, 0.0) if rel is None else (True, -abs(rel))


def _order(key: Any) -> tuple[int, Any]:
    return (0, ()) if key == "all" else (1, ()) if key == DEFAULT else (2, key)


def _collapse_uniform(moved: list[tuple[Any, Change]], keys: set[Any]) -> list[tuple[Any, Change]]:
    """One "all" row per metric that changed the same way everywhere (device-wide values)."""
    out = []
    for metric in dict.fromkeys(c.metric for _, c in moved):
        rows = [(key, c) for key, c in moved if c.metric == metric]
        if len(rows) == len(keys) and len({(c.before, c.after) for _, c in rows}) == 1:
            c = rows[0][1]
            out.append(("all", Change(metric, c.before, c.after, "all")))
        else:
            out += rows
    return out


def _availability(a: Profile, b: Profile) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Gate loci and qubits usable in ``a`` and disabled in ``b``, and the reverse.

    The diff compares every locus that a record names in either profile. So the removal of an
    override that enabled a pair under a disabled definition counts. A profile can lack a locus
    completely, for example a directed gate calibrated in the other direction, or a qubit past
    the end. Such a locus is in neither list, because the locus was not usable before or is not
    usable now. The diff compares both orders of the pair of a symmetric gate. The two orders
    are one entry, under the lower qubit first, when they move the same way.
    """
    moves = {k: (_state(a, k), _state(b, k)) for k in sorted({*_loci(a), *_loci(b)})}
    labels = [
        (f"{gate or 'qubit'} {'-'.join(map(str, qubits)) if qubits else DEFAULT}", move)
        for (gate, qubits), move in moves.items()
        if not _same_as_lower_order(gate, qubits, moves, a, b)
    ]
    return (
        tuple(label for label, move in labels if move == ("usable", "disabled")),
        tuple(label for label, move in labels if move == ("disabled", "usable")),
    )


def _same_as_lower_order(
    gate: str,
    qubits: tuple[int, ...],
    moves: dict[tuple[str, tuple[int, ...]], tuple[str | None, str | None]],
    a: Profile,
    b: Profile,
) -> bool:
    if len(qubits) != 2 or qubits[0] < qubits[1]:
        return False
    symmetric = a.table.symmetric(gate) and b.table.symmetric(gate)
    return symmetric and moves.get((gate, qubits[::-1])) == moves[gate, qubits]


def _loci(profile: Profile) -> set[tuple[str, tuple[int, ...]]]:
    """Recorded gate loci, disabled gate definitions and disabled qubits of ``profile``.

    A gate locus is (gate, qubits), in both orders for a symmetric pair. A disabled gate
    definition is (gate, ()). A disabled qubit is ("", (index,)).
    """
    table = profile.table
    gates = {(r.gate, r.qubits) for r in profile.calibrations}
    gates |= {
        (r.gate, r.qubits[::-1])
        for r in profile.calibrations
        if len(r.qubits) == 2 and table.symmetric(r.gate)
    }
    gates |= {(name, ()) for name, spec in profile.gates.items() if spec.disabled}
    return gates | {("", (q.index,)) for q in profile.qubits if q.disabled}


def _state(profile: Profile, locus: tuple[str, tuple[int, ...]]) -> str | None:
    """ "usable", "disabled", or None when the profile has no such gate, gate locus or qubit."""
    gate, qubits = locus
    table = profile.table
    if not qubits:
        spec = profile.gates.get(gate)
        return None if spec is None else "disabled" if spec.disabled else "usable"
    if not gate:
        if qubits[0] >= table.num_qubits:
            return None
        return "disabled" if table.qubit(qubits[0]).disabled else "usable"
    found = table.gate(gate, qubits)
    if not isinstance(found, GateNoise):
        return None
    return "disabled" if found.state == "disabled" else "usable"


def _where(title: str, where: str) -> str:
    return f"all {title}" if where == "all" else f"{title[:-1]} {where}"


def distinguishing_stamps(a: datetime | None, b: datetime | None) -> tuple[str | None, str | None]:
    if a and b:
        a, b = a.astimezone(UTC), b.astimezone(UTC)
        if a.date() == b.date() and a != b:
            seconds = a.strftime("%H:%M") == b.strftime("%H:%M")
            clock = "%Y-%m-%dT%H:%M:%SZ" if seconds else "%Y-%m-%dT%H:%MZ"
            return a.strftime(clock), b.strftime(clock)
    return (a.date().isoformat() if a else None, b.date().isoformat() if b else None)


# formatting ---------------------------------------------------------------------------------


def fmt_error(value: float | None) -> str:
    return "-" if value is None else f"{value:.2e}"


def fmt_us(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def fmt_metric(metric: str, value: float | None) -> str:
    return fmt_us(value) if metric in ("t1_us", "t2_us") else fmt_error(value)


def fmt_relative(value: float | None) -> str:
    return "-" if value is None else f"{value:+.1%}"


def describe_delta(delta: timedelta | None) -> str:
    if delta is None:
        return "calibration time unknown"
    if delta == timedelta(0):
        return "same calibration time"
    span = abs(delta)
    days, hours = span.days, span.seconds // 3600
    minutes = span.seconds % 3600 // 60
    parts = [(days, "day"), (hours, "hour")] if days or hours else [(minutes, "minute")]
    text = " ".join(plural(n, unit) for n, unit in parts if n) or "under a minute"
    return f"{text} {'later' if delta > timedelta(0) else 'earlier'}"
