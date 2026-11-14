"""Four more statistics of the public archive of IBM calibrations.

    python -m study.archive_steps train.parquet --start 2026-02-15 --end 2026-11-10

The script reads the parquet file with :func:`study.archive.read_rows`. It uses the same series,
changes, window and failure-ceiling exclusions as ``study.archive``, and it writes text to
standard output.

1. Fixed steps. For each series and each day at 00:00 UTC in the window, the script finds the
   value in force on that day and 1, 3 and 7 days later. The value in force at a time is the
   value of the last change at or before that time. A series counts only from its first change
   to its last change. The size of a difference is ``abs(ln(later / earlier))``. The output
   gives a size as the percent ``exp(size) - 1``. A pair with a value of 0 counts as a pair,
   but it has no size, so the two medians do not use it.
2. Rank correlation inside one day. The pairs are the pairs of ``study.archive``: a change and
   the next change of the same series, 12 to 36 hours later. The script gives the Spearman
   correlation of the pairs of one device, one kind and one UTC day, if the day has
   ``MIN_DAY_CHANGES`` or more pairs. The output gives the median over the days.
3. Regression to the mean with chance alone. Each ``prob_meas1_prep0`` series gets one fixed
   error: the mean of its values, or in a second run their median. Each change of the series
   gets a new value: the number of wrong readouts in ``SHOTS`` shots with this error, divided
   by ``SHOTS``. The script then computes the regression to the mean of ``study.archive`` on
   these values.
4. The minutes between two observation times of a device, from ``observed_time``. The archive
   has rows only for a poll that saw a new calibration, so the statistic does not hold a poll
   that saw no new calibration.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from study.archive import (
    KIND_ORDER,
    MIN_DAY_CHANGES,
    PERSISTENCE_GAP_H,
    RETIRED,
    SERIES_STRIDE,
    Rows,
    micros,
    micros_of,
    read_rows,
    regression,
    sha256,
    spearman,
    utc_date,
)

STEPS_DAYS = (1, 3, 7)
SHOTS, SEED = 4096, 5
CHANCE_PROPERTY = "prob_meas1_prep0"
DAY_S = 86_400


def fixed_steps(
    changes: Rows, start: datetime, end: datetime, steps: Sequence[int] = STEPS_DAYS
) -> list[dict[str, Any]]:
    """For each step in days and each kind: the number of pairs of a value in force on a day and
    the value in force ``step`` days later, the fraction of pairs with a new value, the median
    size of all pairs and the median size of the pairs with a new value. The two medians use
    only the pairs with two values above 0.

    ``changes`` holds only changes, as ``rows.select(rows.change)`` gives them.
    """
    seconds = changes.time_us // 1_000_000
    series = np.cumsum(changes.first) - 1
    key = series.astype(np.int64) * SERIES_STRIDE + seconds
    starts = np.flatnonzero(changes.first)
    ends = np.append(starts[1:], changes.value.size) - 1
    alive_from, alive_to = seconds[starts], seconds[ends]
    kind_of = changes.kind[starts]
    excluded = changes.excluded
    ids = np.arange(starts.size, dtype=np.int64)
    first_day, last_day = int(start.timestamp()) // DAY_S, int(end.timestamp()) // DAY_S

    def in_force(when: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        wanted = ids[:, None] * SERIES_STRIDE + when[None, :]
        index = np.searchsorted(key, wanted.ravel(), side="right").reshape(wanted.shape) - 1
        alive = (when[None, :] >= alive_from[:, None]) & (when[None, :] <= alive_to[:, None])
        return np.where(alive, index, 0), alive

    found: list[dict[str, Any]] = []
    for step in steps:
        days = np.arange(first_day, last_day - step + 1, dtype=np.int64) * DAY_S
        now, alive_now = in_force(days)
        later, alive_later = in_force(days + step * DAY_S)
        before, after = changes.value[now], changes.value[later]
        kept = alive_now & alive_later & ~excluded[now] & ~excluded[later]
        sized = kept & (before > 0) & (after > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            size = np.abs(np.log(after / before))
        new = later != now
        for code, kind in enumerate(KIND_ORDER):
            here = kept & (kind_of[:, None] == code)
            if not here.any():
                continue
            found.append(
                {
                    "kind": kind,
                    "step_days": step,
                    "pairs": int(here.sum()),
                    "new_fraction": float(np.mean(new[here])),
                    "median_size": _median(size[here & sized]),
                    "median_size_new": _median(size[here & sized & new]),
                }
            )
    return found


def day_correlations(changes: Rows) -> dict[str, dict[str, list[float]]]:
    """For each device and kind: the Spearman correlation of the pairs of each UTC day with
    ``MIN_DAY_CHANGES`` or more pairs. The day of a pair is the day of its first change.

    ``changes`` holds only the changes in the window.
    """
    later = np.flatnonzero(~changes.first)
    gap_h = (changes.time_us[later] - changes.time_us[later - 1]) / 3.6e9
    out = changes.excluded
    low, high = PERSISTENCE_GAP_H
    kept = later[(gap_h >= low) & (gap_h < high) & ~out[later] & ~out[later - 1]]
    day = changes.time_us[kept - 1] // (DAY_S * 1_000_000)
    found: dict[str, dict[str, list[float]]] = {}
    for code in sorted(np.unique(changes.device).tolist(), key=lambda c: changes.devices[c]):
        for kind_code, kind in enumerate(KIND_ORDER):
            here = (changes.device[kept] == code) & (changes.kind[kept] == kind_code)
            values = []
            for one_day in np.unique(day[here]).tolist():
                index = kept[here & (day == one_day)]
                if index.size >= MIN_DAY_CHANGES:
                    value = spearman(changes.value[index - 1], changes.value[index])
                    if value is not None:
                        values.append(value)
            if values:
                found.setdefault(changes.devices[code], {})[kind] = values
    return found


def chance_regression(
    changes: Rows, prop: str = CHANCE_PROPERTY, shots: int = SHOTS, seed: int = SEED
) -> dict[str, Any]:
    """The regression to the mean of ``prop`` in the archive and with chance alone.

    ``fixed_error`` gives the median over the series of the mean and of the median of each
    series. Each entry of ``runs`` gives the result of :func:`study.archive.regression` and
    ``typical_size``: the archive, then chance alone with the mean and with the median of each
    series as its fixed error. ``changes`` holds only the changes in the window.
    """
    code = changes.properties.index(prop) if prop in changes.properties else -1
    picked = changes.select(changes.prop == code)
    series = np.cumsum(picked.first) - 1
    usable = ~picked.excluded
    count = int(series.max()) + 1 if series.size else 0
    used_series, used_value = series[usable], picked.value[usable]
    numbers = np.maximum(np.bincount(used_series, minlength=count), 1)
    mean_level = np.bincount(used_series, used_value, minlength=count) / numbers
    median_level, has = _series_median(used_series, used_value, count)
    runs = {"archive": _run(picked, prop)}
    for name, level in (("mean", mean_level), ("median", median_level)):
        rng = np.random.default_rng(seed)
        drawn = rng.binomial(shots, np.clip(level[series], 0, 1)) / shots
        simulated = Rows(
            picked.devices,
            picked.properties,
            picked.device,
            picked.prop,
            picked.first,
            picked.time_us,
            np.where(usable, drawn, picked.value),
            picked.flagged,
            0,
            0,
        )
        # A value equal to the value before is not a change in the archive, so it is not one here.
        runs[name] = _run(simulated.select(simulated.change), prop)
    return {
        "fixed_error": {
            "mean": _median(mean_level[has]),
            "median": _median(median_level[has]),
        },
        "runs": runs,
    }


def observation_times(path: Path, start: datetime, end: datetime) -> dict[str, np.ndarray]:
    """For each device: the distinct ``observed_time`` values in the window, in microseconds."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(
        path,
        columns=["backend", "observed_time"],
        filters=~pc.field("backend").isin(list(RETIRED)),
    )
    observed = micros(table["observed_time"])
    codes = pc.dictionary_encode(table["backend"]).combine_chunks()
    labels = codes.indices.to_numpy()
    low, high = micros_of(start), micros_of(end)
    found: dict[str, np.ndarray] = {}
    for code, device in enumerate(codes.dictionary.to_pylist()):
        times = np.unique(observed[labels == code])
        found[device] = times[(times >= low) & (times <= high)]
    return dict(sorted(found.items()))


def _run(changes: Rows, prop: str) -> dict[str, Any]:
    return {"regression": regression(changes, prop), "typical_size": typical_size(changes)}


def typical_size(changes: Rows) -> float | None:
    """The median of ``abs(ln(value / value before))`` over the changes after a change. Both
    values are above 0 and are not failure-ceiling values."""
    later = np.flatnonzero(~changes.first)
    now, before = changes.value[later], changes.value[later - 1]
    out = changes.excluded
    kept = (now > 0) & (before > 0) & ~out[later] & ~out[later - 1]
    return _median(np.abs(np.log(now[kept] / before[kept])))


def _series_median(
    series: np.ndarray, value: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """The median value of each of ``count`` series, 0 for a series with no value, and a mask of
    the series with a value."""
    order = np.lexsort((value, series))
    ordered_series, ordered = series[order], value[order]
    first = np.searchsorted(ordered_series, np.arange(count))
    last = np.searchsorted(ordered_series, np.arange(count), side="right") - 1
    has = last >= first
    if ordered.size == 0:
        return np.zeros(count), has
    top = ordered.size - 1
    low, high = np.minimum((first + last) // 2, top), np.minimum((first + last + 1) // 2, top)
    return np.where(has, (ordered[low] + ordered[high]) / 2, 0.0), has


def _median(values: np.ndarray) -> float | None:
    return float(np.median(values)) if values.size else None


def _percent(size: float | None) -> str:
    return "-" if size is None else f"{100 * (math.exp(size) - 1):.2f}%"


def report(path: Path, start: datetime, end: datetime) -> str:
    """The four statistics of the archive file at ``path`` as text."""
    rows = read_rows(path)
    changes = rows.select(rows.change)
    inside = (rows.time_us >= micros_of(start)) & (rows.time_us <= micros_of(end))
    window_changes = rows.select(inside & rows.change)
    del rows
    lines = [
        f"file SHA-256 {sha256(path)}",
        f"window {start:%Y-%m-%d} to {end:%Y-%m-%d}",
        "",
        "1. Value in force on a day and N days later (all devices together)",
        "kind, step, pairs, share with a new value, median difference of all pairs,"
        " of the pairs with a new value",
    ]
    for row in fixed_steps(changes, start, end):
        lines.append(
            f"{row['kind']}, {row['step_days']} d, {row['pairs']}, {row['new_fraction']:.3f}, "
            f"{_percent(row['median_size'])}, {_percent(row['median_size_new'])}"
        )
    del changes
    lines += [
        "",
        "2. Rank correlation of a new value and the next new value 12 to 36 hours later",
        f"device, kind, days with {MIN_DAY_CHANGES} or more pairs, median over the days,"
        " lowest and highest tenth of the days",
    ]
    medians: dict[str, list[float]] = {}
    for device, kinds in day_correlations(window_changes).items():
        for kind, values in kinds.items():
            medians.setdefault(kind, []).append(float(np.median(values)))
            lines.append(
                f"{device}, {kind}, {len(values)}, {np.median(values):.3f}, "
                f"{np.quantile(values, 0.1):.3f}, {np.quantile(values, 0.9):.3f}"
            )
    for kind in sorted(medians, key=KIND_ORDER.index):
        lines.append(
            f"range over the devices, {kind}: {min(medians[kind]):.2f} to {max(medians[kind]):.2f}"
        )
    lines += [
        "",
        f"3. {CHANCE_PROPERTY}, lowest tenth of a device and day, and the next value"
        f" (counts of {SHOTS} shots)",
    ]
    chance = chance_regression(window_changes)
    titles = {
        "archive": "archive",
        "mean": "chance alone, fixed error = mean of the series",
        "median": "chance alone, fixed error = median of the series",
    }
    for name, run in chance["runs"].items():
        for device, found in run["regression"].items():
            part = found["parts"]["lowest tenth"]
            if part["count"] == 0:
                lines.append(f"{titles[name]}, {device}: 0 values")
                continue
            now, after = part["median_now"], part["median_next"]
            lines.append(
                f"{titles[name]}, {device}: {part['count']} values, median {now:.5f}"
                f" ({now * SHOTS:.1f} of {SHOTS}), median of the next value {after:.5f}"
                f" ({after * SHOTS:.1f} of {SHOTS}), ratio {_ratio(after, now)}"
            )
        lines.append(
            f"{titles[name]}: median difference between a value and the value before"
            f" {_percent(run['typical_size'])}"
        )
        if name == "archive":
            fixed = chance["fixed_error"]
            lines.append(
                f"median over the series of the fixed error: mean {_f5(fixed['mean'])},"
                f" median {_f5(fixed['median'])}"
            )
    del window_changes
    lines += ["", "4. Minutes between two observation times of a device (observed_time)"]
    for device, times in observation_times(path, start, end).items():
        gaps = np.diff(times) / 6e7
        if gaps.size == 0:
            lines.append(f"{device}: {times.size} observation times")
            continue
        lines.append(
            f"{device}: {times.size} observation times, median {np.median(gaps):.1f},"
            f" lowest tenth {np.quantile(gaps, 0.1):.1f},"
            f" highest tenth {np.quantile(gaps, 0.9):.1f},"
            f" share above 60 minutes {np.mean(gaps > 60):.3f}"
        )
    return "\n".join(lines)


def _ratio(after: float, now: float) -> str:
    return f"{after / now:.2f}" if now else "-"


def _f5(number: float | None) -> str:
    return "-" if number is None else f"{number:.5f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("parquet", type=Path, help="the parquet file of the dataset")
    parser.add_argument(
        "--start", type=utc_date, default=utc_date("2026-02-15"), help="the first day"
    )
    parser.add_argument("--end", type=utc_date, default=utc_date("2026-11-10"), help="the last day")
    args = parser.parse_args(argv)
    if not args.parquet.is_file():
        print(f"error: {args.parquet} is not a file", file=sys.stderr)
        return 2
    if args.end < args.start:
        print("error: --end is before --start", file=sys.stderr)
        return 2
    print(report(args.parquet, args.start, args.end))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
