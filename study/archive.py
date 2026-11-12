"""Refresh times, value ages and value changes in the public archive of IBM calibrations.

    python -m study.archive train.parquet --out results --start 2026-02-15 --end 2026-11-10

The archive is the Hugging Face dataset phanerozoic/qiskit-calibration-drift (CC-BY-4.0). Its
poller reads the calibration of each IBM device each 30 minutes. The script reads the parquet
file of the dataset with pyarrow and writes ``archive.json`` and ``archive.md`` to the output
folder.

A series is one property of one qubit or of one ordered pair of one device, ordered by
``calibrated_time``, with one row for each distinct ``calibrated_time``. When two rows of a series
have the same ``calibrated_time``, the series keeps the row with the latest ``observed_time``.
A change is a row whose value is not the value of the row before it in the series. The first row
of a series is a change. A re-stamp is a row that is not a change: IBM gives a gate error a new
time with the same value about each hour, so a new ``calibrated_time`` is not always a new
measurement.

The change sizes and the regression to the mean exclude the failure-ceiling values: a row with
``is_failure_ceiling`` true, and a gate error of 0.5 or more. The output gives the number of
changes that each exclusion removes.

The script excludes ibm_torino, which IBM retired in April 2026, unless ``--include-retired`` is
given. ``--start`` and ``--end`` set the grid of times for the time since the last change. The
other statistics use all rows of the file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pyarrow as pa

DATASET = "phanerozoic/qiskit-calibration-drift"
LICENSE = "CC-BY-4.0"
FORMAT = "1"
KINDS = {
    "T1": "t1",
    "T2": "t2",
    "prob_meas0_prep1": "readout",
    "prob_meas1_prep0": "readout",
    "sx_gate_error": "one_qubit",
    "cz_gate_error": "two_qubit",
    "rzz_gate_error": "rzz",
}
KIND_ORDER = ("t1", "t2", "readout", "one_qubit", "two_qubit", "rzz")
GATE_KINDS = ("one_qubit", "two_qubit", "rzz")
GATE_CEILING = 0.5
READOUT = ("prob_meas0_prep1", "prob_meas1_prep0")
RANKED = ("prob_meas1_prep0", "cz_gate_error")
RETIRED = ("ibm_torino",)
GRID_STEP = timedelta(hours=6)
GAP_BINS = ("<12 h", "12-36 h", "36-72 h", "3-7 d", ">7 d")
GAP_EDGES_H = (12.0, 36.0, 72.0, 168.0)
GRID_DENOMINATORS = (1000, 1024, 2048, 4096, 8192, 10000)
GRID_TOLERANCE = 1e-6
PARTS = ("lowest tenth", "middle", "highest tenth")
_COLUMNS = (
    "backend",
    "property",
    "qubit_a",
    "qubit_b",
    "value",
    "calibrated_time",
    "observed_time",
    "is_failure_ceiling",
)
_US_PER_HOUR = 3_600_000_000
_US_PER_DAY = 24 * _US_PER_HOUR
# A key of a series and a time in seconds in one int64: the seconds stay below 2**34 to year 2514.
_SERIES_STRIDE = 2**34


@dataclass(frozen=True)
class Rows:
    """The rows of all series, sorted by series and then by ``calibrated_time``.

    ``device`` and ``prop`` index ``devices`` and ``properties``. ``first`` marks the first row of
    each series. ``time_us`` is ``calibrated_time`` in microseconds since 1970-01-01 UTC.
    ``flagged`` marks the rows with ``is_failure_ceiling`` true.
    ``duplicates`` counts the rows that the series dropped for a time that another row has, and
    ``conflicts`` counts the dropped rows with a different value.
    """

    devices: list[str]
    properties: list[str]
    device: np.ndarray
    prop: np.ndarray
    first: np.ndarray
    time_us: np.ndarray
    value: np.ndarray
    flagged: np.ndarray
    duplicates: int
    conflicts: int

    @property
    def change(self) -> np.ndarray:
        previous = np.concatenate(([np.nan], self.value[:-1]))
        return self.first | (self.value != previous)

    @property
    def kind(self) -> np.ndarray:
        # The dictionary of the file also names the properties that the filter drops.
        codes = np.array(
            [KIND_ORDER.index(KINDS.get(p, KIND_ORDER[0])) for p in self.properties], dtype=np.int8
        )
        return codes[self.prop]

    @property
    def high_gate(self) -> np.ndarray:
        """The gate errors of ``GATE_CEILING`` or more."""
        gates = np.isin(self.kind, [KIND_ORDER.index(k) for k in GATE_KINDS])
        return gates & (self.value >= GATE_CEILING)

    @property
    def excluded(self) -> np.ndarray:
        """The failure-ceiling values."""
        return self.flagged | self.high_gate

    def select(self, mask: np.ndarray) -> Rows:
        """The rows of ``mask``. A row is first when no row of its series comes before it."""
        series = np.cumsum(self.first)[mask]
        first = np.concatenate(([True], series[1:] != series[:-1])) if series.size else series > 0
        return Rows(
            self.devices,
            self.properties,
            self.device[mask],
            self.prop[mask],
            first,
            self.time_us[mask],
            self.value[mask],
            self.flagged[mask],
            self.duplicates,
            self.conflicts,
        )


def read_rows(path: Path, include_retired: bool = False) -> Rows:
    """The rows of the properties of ``KINDS`` in the parquet file at ``path``."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    wanted = (
        pc.field("property").isin(list(KINDS))
        & pc.field("calibrated_time").is_valid()
        & pc.field("value").is_valid()
        & pc.field("qubit_a").is_valid()
    )
    if not include_retired:
        wanted = wanted & ~pc.field("backend").isin(list(RETIRED))
    table = pq.read_table(
        path, columns=list(_COLUMNS), filters=wanted, read_dictionary=["backend", "property"]
    ).unify_dictionaries()

    devices, device = _codes(table["backend"])
    properties, prop = _codes(table["property"])
    qubit_a = table["qubit_a"].to_numpy().astype(np.int32)
    qubit_b = pc.fill_null(table["qubit_b"], -1.0).to_numpy().astype(np.int32)
    value = table["value"].to_numpy()
    flagged = pc.fill_null(table["is_failure_ceiling"], False).to_numpy(zero_copy_only=False)
    time_us = _micros(table["calibrated_time"])
    observed = _micros(table["observed_time"])
    del table
    order = np.lexsort((observed, time_us, qubit_b, qubit_a, prop, device))
    del observed
    device, prop, qubit_a, qubit_b = device[order], prop[order], qubit_a[order], qubit_b[order]
    time_us, value, flagged = time_us[order], value[order], flagged[order]
    del order
    new_series = np.ones(device.size, dtype=bool)
    new_series[1:] = (
        (device[1:] != device[:-1])
        | (prop[1:] != prop[:-1])
        | (qubit_a[1:] != qubit_a[:-1])
        | (qubit_b[1:] != qubit_b[:-1])
    )
    del qubit_a, qubit_b
    repeated = np.zeros(device.size, dtype=bool)
    repeated[:-1] = ~new_series[1:] & (time_us[1:] == time_us[:-1])
    conflicts = int(np.count_nonzero(repeated[:-1] & (value[:-1] != value[1:])))
    rows = Rows(
        devices,
        properties,
        device,
        prop,
        new_series,
        time_us,
        value,
        flagged,
        int(np.count_nonzero(repeated)),
        conflicts,
    )
    return rows.select(~repeated)


def _codes(column: pa.ChunkedArray) -> tuple[list[str], np.ndarray]:
    """The names of a dictionary column and the index of each row's name."""
    if not column.chunks:
        return [], np.zeros(0, dtype=np.int16)
    names = column.chunks[0].dictionary.to_pylist()
    return names, np.concatenate([c.indices.to_numpy() for c in column.chunks]).astype(np.int16)


def _micros(column: pa.ChunkedArray) -> np.ndarray:
    import pyarrow as pa
    import pyarrow.compute as pc

    stamps = pc.cast(column, pa.timestamp("us", tz="UTC"))
    return pc.cast(stamps, pa.int64()).to_numpy()


def analyze(
    path: Path, start: datetime, end: datetime, include_retired: bool = False
) -> dict[str, Any]:
    """Every statistic of the archive file at ``path``, with its count."""
    rows = read_rows(path, include_retired)
    changes = rows.select(rows.change)
    return {
        "format": FORMAT,
        "dataset": DATASET,
        "license": LICENSE,
        "file_sha256": _sha256(path),
        "devices": sorted(set(rows.devices[d] for d in np.unique(rows.device).tolist())),
        "rows": int(rows.value.size),
        "changes": int(changes.value.size),
        "duplicate_times": {"rows": rows.duplicates, "different_value": rows.conflicts},
        "calibrated_from": _iso(int(rows.time_us.min())) if rows.value.size else None,
        "calibrated_to": _iso(int(rows.time_us.max())) if rows.value.size else None,
        "grid": {"start": _iso_dt(start), "end": _iso_dt(end), "step_h": 6},
        "intervals": _intervals(rows, changes, _kind_label),
        "intervals_by_property": _intervals(rows, changes, _property_label),
        "since_change": _since_change(changes, start, end, _kind_label),
        "since_change_by_property": _since_change(changes, start, end, _property_label),
        "excluded": _excluded(changes),
        "change_size": _change_size(changes),
        "readout_grid": _readout_grid(rows),
        "regression": {p: _regression(changes, p) for p in RANKED},
    }


Label = Callable[[Rows], tuple[np.ndarray, Callable[[int], tuple[str, str]]]]


def _kind_label(rows: Rows) -> tuple[np.ndarray, Callable[[int], tuple[str, str]]]:
    width = len(KIND_ORDER)
    labels = rows.device.astype(np.int32) * width + rows.kind

    def name(label: int) -> tuple[str, str]:
        return rows.devices[label // width], KIND_ORDER[label % width]

    return labels, name


def _property_label(rows: Rows) -> tuple[np.ndarray, Callable[[int], tuple[str, str]]]:
    width = len(rows.properties)
    labels = rows.device.astype(np.int32) * width + rows.prop

    def name(label: int) -> tuple[str, str]:
        return rows.devices[label // width], rows.properties[label % width]

    return labels, name


def _intervals(rows: Rows, changes: Rows, label: Label) -> dict[str, dict[str, Any]]:
    """Definition 3: the hours between two changes, the hours between two times and the
    fraction of rows that are re-stamps, for each device and group."""
    found: dict[str, dict[str, Any]] = {}
    labels, name = label(rows)
    later = ~rows.first
    row_gaps = _split(labels[later], np.diff(rows.time_us, prepend=0)[later] / _US_PER_HOUR)
    restamps = _split(labels, (~rows.change).astype(np.float64))
    change_labels, _ = label(changes)
    later = ~changes.first
    gaps = _split(change_labels[later], np.diff(changes.time_us, prepend=0)[later] / _US_PER_HOUR)
    for key, flags in sorted(restamps.items()):
        device, group = name(key)
        found.setdefault(device, {})[group] = {
            "between_changes_h": _quantiles(gaps.get(key, np.zeros(0)), (0.5, 0.9)),
            "between_times_h": _quantiles(row_gaps.get(key, np.zeros(0)), (0.5, 0.9)),
            "restamps": {
                "rows": int(flags.size),
                "restamps": int(flags.sum()),
                "fraction": float(flags.mean()),
            },
        }
    return found


def _since_change(
    changes: Rows, start: datetime, end: datetime, label: Label
) -> dict[str, dict[str, Any]]:
    """Definition 4: at each grid time, the hours since the last change of each series that has
    a change at or before the grid time."""
    grid = np.arange(_seconds(start), _seconds(end) + 1, int(GRID_STEP.total_seconds()))
    series = np.cumsum(changes.first) - 1
    keys = series * _SERIES_STRIDE + changes.time_us // 1_000_000
    labels, name = label(changes)
    series_labels = labels[changes.first]
    ages: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    for block in np.array_split(np.arange(series_labels.size), max(1, series_labels.size // 512)):
        query = (block[:, None] * _SERIES_STRIDE + grid[None, :]).ravel()
        last = np.searchsorted(keys, query, side="right") - 1
        valid = (last >= 0) & (series[np.maximum(last, 0)] == np.repeat(block, grid.size))
        ages.append(((query - keys[np.maximum(last, 0)]) / 3600.0)[valid].astype(np.float64))
        owners.append(series_labels[np.repeat(block, grid.size)][valid])
    grouped = _split(np.concatenate(owners), np.concatenate(ages)) if ages else {}
    found: dict[str, dict[str, Any]] = {}
    for key, hours in sorted(grouped.items()):
        device, group = name(key)
        found.setdefault(device, {})[group] = {
            **_quantiles(hours, (0.5, 0.9, 0.99)),
            "above_24h": float(np.mean(hours > 24)),
            "above_168h": float(np.mean(hours > 168)),
        }
    return found


def _excluded(changes: Rows) -> dict[str, dict[str, int]]:
    """The number of changes of each device that each exclusion removes. A flagged gate error of
    ``GATE_CEILING`` or more counts only as flagged."""
    found: dict[str, dict[str, int]] = {}
    for code in np.unique(changes.device).tolist():
        here = changes.device == code
        found[changes.devices[code]] = {
            "failure_ceiling": int(np.count_nonzero(here & changes.flagged)),
            "gate_error_0.5_or_more": int(
                np.count_nonzero(here & changes.high_gate & ~changes.flagged)
            ),
        }
    return found


def _change_size(changes: Rows) -> dict[str, dict[str, Any]]:
    """Definition 5: ``abs(ln(value / previous value))`` of each change after a change, by kind
    and by the hours between the two changes. Both values are above 0 and are not failure-ceiling
    values."""
    later = np.flatnonzero(~changes.first)
    now, before = changes.value[later], changes.value[later - 1]
    excluded = changes.excluded
    kept = (now > 0) & (before > 0) & ~excluded[later] & ~excluded[later - 1]
    later, now, before = later[kept], now[kept], before[kept]
    gap_h = (changes.time_us[later] - changes.time_us[later - 1]) / _US_PER_HOUR
    bins = np.digitize(gap_h, GAP_EDGES_H)
    labels = changes.kind[later].astype(np.int32) * len(GAP_BINS) + bins
    grouped = _split(labels, np.abs(np.log(now / before)))
    found: dict[str, dict[str, Any]] = {}
    for key, sizes in sorted(grouped.items()):
        kind, gap = KIND_ORDER[key // len(GAP_BINS)], GAP_BINS[key % len(GAP_BINS)]
        found.setdefault(kind, {})[gap] = _quantiles(sizes, (0.5, 0.9))
    return found


def _readout_grid(rows: Rows) -> dict[str, dict[str, Any]]:
    """Definition 6: the fraction of zeros and of multiples of 1/N in each readout series."""
    found: dict[str, dict[str, Any]] = {}
    labels, name = _property_label(rows)
    readout = np.isin(rows.prop, [i for i, p in enumerate(rows.properties) if p in READOUT])
    for key, values in sorted(_split(labels[readout], rows.value[readout]).items()):
        device, prop = name(key)
        found.setdefault(device, {})[prop] = {
            "values": int(values.size),
            "zero_fraction": float(np.mean(values == 0)),
            "multiple_of": {
                f"1/{n}": float(np.mean(np.abs(values * n - np.round(values * n)) < GRID_TOLERANCE))
                for n in GRID_DENOMINATORS
            },
        }
    return found


def _regression(changes: Rows, prop: str) -> dict[str, dict[str, Any]]:
    """Definition 7: the changes of ``prop`` by their percent rank among the changes of ``prop``
    on the same device and UTC day, with the next change of the same series. The ranks and the
    pairs exclude the failure-ceiling values."""
    if prop not in changes.properties:
        return {}
    picked = changes.select(changes.prop == changes.properties.index(prop))
    kept = ~picked.excluded
    rank = np.full(kept.size, np.nan)
    rank[kept] = _percent_rank(
        picked.device[kept].astype(np.int64),
        picked.time_us[kept] // _US_PER_DAY,
        picked.value[kept],
    )
    following = np.flatnonzero(~picked.first) - 1
    following = following[kept[following] & kept[following + 1]]
    now, rank, device = picked.value[following], rank[following], picked.device[following]
    after = picked.value[following + 1]
    part = np.where(rank < 0.1, 0, np.where(rank > 0.9, 2, 1))
    found: dict[str, dict[str, Any]] = {}
    for code in np.unique(device).tolist():
        here = device == code
        found[picked.devices[code]] = {
            name: _pair(now[here & (part == index)], after[here & (part == index)])
            for index, name in enumerate(PARTS)
        }
    return found


def _percent_rank(device: np.ndarray, day: np.ndarray, value: np.ndarray) -> np.ndarray:
    """The SQL ``percent_rank`` of each value within its device and day: (rank - 1) / (n - 1),
    where tied values share the lowest rank, and 0 for a group of one."""
    order = np.lexsort((value, day, device))
    group_device, group_day, ordered = device[order], day[order], value[order]
    size = ordered.size
    new_group = np.ones(size, dtype=bool)
    new_group[1:] = (group_device[1:] != group_device[:-1]) | (group_day[1:] != group_day[:-1])
    tie = np.zeros(size, dtype=bool)
    tie[1:] = ~new_group[1:] & (ordered[1:] == ordered[:-1])
    index = np.arange(size)
    group_start = np.maximum.accumulate(np.where(new_group, index, 0))
    rank_start = np.maximum.accumulate(np.where(tie, 0, index))
    starts = np.flatnonzero(new_group)
    counts = np.diff(np.append(starts, size))
    group_size = np.repeat(counts, counts)
    rank = np.zeros(size)
    many = group_size > 1
    rank[many] = (rank_start - group_start)[many] / (group_size[many] - 1)
    found = np.empty(size)
    found[order] = rank
    return found


def _pair(now: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    empty = now.size == 0
    return {
        "count": int(now.size),
        "mean_now": None if empty else float(now.mean()),
        "median_now": None if empty else float(np.median(now)),
        "mean_next": None if empty else float(after.mean()),
        "median_next": None if empty else float(np.median(after)),
    }


def _split(labels: np.ndarray, values: np.ndarray) -> dict[int, np.ndarray]:
    order = np.argsort(labels, kind="stable")
    keys, starts = np.unique(labels[order], return_index=True)
    return dict(zip(keys.tolist(), np.split(values[order], starts[1:]), strict=True))


def _quantiles(values: np.ndarray, levels: Sequence[float]) -> dict[str, Any]:
    found: dict[str, Any] = {"count": int(values.size)}
    for level in levels:
        key = "median" if level == 0.5 else f"p{round(level * 100)}"
        found[key] = float(np.quantile(values, level)) if values.size else None
    return found


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _seconds(when: datetime) -> int:
    return int(when.timestamp())


def _iso(micros: int) -> str:
    return _iso_dt(datetime.fromtimestamp(micros / 1_000_000, UTC))


def _iso_dt(when: datetime) -> str:
    return when.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def markdown(data: Mapping[str, Any]) -> str:
    """The compact tables of ``data``, the result of ``analyze``."""
    grid = data["grid"]
    lines = [
        "# Calibration archive",
        "",
        f"- Dataset: {data['dataset']} ({data['license']})",
        f"- File SHA-256: `{data['file_sha256']}`",
        f"- Devices: {', '.join(data['devices'])}",
        f"- Values: calibrated from {data['calibrated_from']} to {data['calibrated_to']},"
        f" {data['rows']} rows, {data['changes']} changes",
        f"- Grid of times: {grid['start']} to {grid['end']}, each {grid['step_h']} hours",
        "",
        "## Hours between two changes and between two times",
        "",
        "| Device | Kind | Changes | Median | P90 | Times | Median | P90 | Re-stamps |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for device, kinds in _by_kind(data["intervals"]):
        for kind, row in kinds:
            change, times = row["between_changes_h"], row["between_times_h"]
            lines.append(
                f"| {device} | {kind} | {change['count']} | {_f(change['median'], 1)}"
                f" | {_f(change['p90'], 1)} | {times['count']} | {_f(times['median'], 1)}"
                f" | {_f(times['p90'], 1)} | {_f(row['restamps']['fraction'], 3)} |"
            )
    lines += [
        "",
        "## Hours since the last change at a grid time",
        "",
        "| Device | Kind | Points | Median | P90 | P99 | > 24 h | > 168 h |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for device, kinds in _by_kind(data["since_change"]):
        for kind, row in kinds:
            lines.append(
                f"| {device} | {kind} | {row['count']} | {_f(row['median'], 1)}"
                f" | {_f(row['p90'], 1)} | {_f(row['p99'], 0)} | {_f(row['above_24h'], 3)}"
                f" | {_f(row['above_168h'], 3)} |"
            )
    lines += [
        "",
        "## Changes excluded from the change sizes and the regression to the mean",
        "",
        "| Device | is_failure_ceiling | Gate error of 0.5 or more |",
        "|---|---:|---:|",
    ]
    for device, row in sorted(data["excluded"].items()):
        lines.append(f"| {device} | {row['failure_ceiling']} | {row['gate_error_0.5_or_more']} |")
    lines += [
        "",
        "## Size of a change, abs(ln(value / previous value)), by the time since the previous"
        " change",
        "",
        "| Kind | Gap | Changes | Median | P90 |",
        "|---|---|---:|---:|---:|",
    ]
    for kind in KIND_ORDER:
        for gap in GAP_BINS:
            row = data["change_size"].get(kind, {}).get(gap)
            if row is not None:
                lines.append(
                    f"| {kind} | {gap} | {row['count']} | {_f(row['median'], 3)}"
                    f" | {_f(row['p90'], 3)} |"
                )
    denominators = [f"1/{n}" for n in GRID_DENOMINATORS]
    lines += [
        "",
        "## Readout values: fraction of zeros and of multiples of 1/N",
        "",
        "| Device | Property | Values | Zeros | " + " | ".join(denominators) + " |",
        "|---|---|---:|---:|" + "---:|" * len(denominators),
    ]
    for device, props in sorted(data["readout_grid"].items()):
        for prop, row in sorted(props.items()):
            cells = " | ".join(_f(row["multiple_of"][d], 3) for d in denominators)
            lines.append(
                f"| {device} | {prop} | {row['values']} | {_f(row['zero_fraction'], 4)} | {cells} |"
            )
    lines += [
        "",
        "## Regression to the mean: changes by rank in their device and day, and the next change",
        "",
        "| Property | Device | Part | Changes | Mean now | Median now | Mean next | Median next |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for prop in RANKED:
        for device, parts in sorted(data["regression"][prop].items()):
            for part in PARTS:
                row = parts[part]
                lines.append(
                    f"| {prop} | {device} | {part} | {row['count']} | {_f(row['mean_now'], 5)}"
                    f" | {_f(row['median_now'], 5)} | {_f(row['mean_next'], 5)}"
                    f" | {_f(row['median_next'], 5)} |"
                )
    return "\n".join(lines) + "\n"


def _by_kind(
    table: Mapping[str, Mapping[str, Any]],
) -> list[tuple[str, list[tuple[str, Any]]]]:
    return [
        (device, [(k, kinds[k]) for k in KIND_ORDER if k in kinds])
        for device, kinds in sorted(table.items())
    ]


def _f(number: float | None, digits: int) -> str:
    if number is None or math.isnan(number):
        return "-"
    return f"{number:.{digits}f}"


def _date(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def write(data: Mapping[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "archive.json").write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    (out / "archive.md").write_text(markdown(data), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("parquet", type=Path, help="the parquet file of the dataset")
    parser.add_argument("--out", type=Path, default=Path("results"), help="the output folder")
    parser.add_argument(
        "--start", type=_date, default=_date("2026-02-15"), help="the first grid day"
    )
    parser.add_argument("--end", type=_date, default=_date("2026-11-10"), help="the last grid day")
    parser.add_argument(
        "--include-retired", action="store_true", help="keep the rows of ibm_torino"
    )
    args = parser.parse_args(argv)
    if not args.parquet.is_file():
        print(f"error: {args.parquet} is not a file", file=sys.stderr)
        return 2
    if args.end < args.start:
        print("error: --end is before --start", file=sys.stderr)
        return 2
    data = analyze(args.parquet, args.start, args.end, args.include_retired)
    write(data, args.out)
    print(f"wrote {data['rows']} rows and {data['changes']} changes to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
