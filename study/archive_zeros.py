"""The readout probabilities of exactly 0 in each observation of the archive of IBM calibrations.

    python -m study.archive_zeros train.parquet --start 2026-02-15 --end 2026-11-10

The archive is the parquet file of ``study.archive``. Its poller reads the calibration of each
device about each 30 minutes, at the ``observed_time`` of the poll. The archive has a row for a
value only when IBM gives the value a new ``calibrated_time``, with a new value or with the same
value. Thus a row of a poll is a value that IBM stated again since the poll before.

An observation is one ``observed_time`` of one device with one or more readout rows: rows of
``prob_meas1_prep0`` (P(1|0)) or ``prob_meas0_prep1`` (P(0|1)). An observation counts when its
``observed_time`` is at 00:00 UTC of ``--start`` or later and before 00:00 UTC of the day after
``--end``. The script excludes the devices of ``study.archive.RETIRED``. For each observation,
the script counts:

- the readout rows of the observation;
- the zeros: the readout rows with a value of exactly 0;
- the zeros of P(1|0);
- the qubits with one or more zeros;
- the zeros in force: the readout values in force at the observation that are exactly 0.

The script uses two definitions of a zero of an observation:

- A zero stated in the observation is a readout row of the observation with the value 0. A
  readout value that IBM did not state again at the observation has no row there, so a 0 from
  an earlier observation does not count.
- A zero in force at the observation is a readout series whose value in force is 0. The value
  in force of a series at a time is the value of its last row with an ``observed_time`` at or
  before that time, also a row from before ``--start``. A user who reads the calibration at
  that time gets these values.

The output gives, for each device, the statistics of the stated zeros over its observations.
It then gives the mean number of stated zeros of an observation for each UTC month of
``observed_time``, and the mean and the largest number for each UTC day of the last
``LAST_DAYS`` days of the window. The last two parts give the zeros in force for each device
and for each UTC month.

The script reads only six columns of the file with pyarrow, and it needs no other database.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from study.archive import READOUT, RETIRED, micros, micros_of, utc_date

LAST_DAYS = 7
P10 = "prob_meas1_prep0"
_US_PER_DAY = 86_400 * 1_000_000
# A device code and a time in microseconds in one int64: the times stay below 2**53.
_DEVICE_STRIDE = 2**53


class Observations(NamedTuple):
    """One entry for each observation, sorted by device and then by time.

    ``device`` indexes ``devices``. ``time_us`` is ``observed_time`` in microseconds since
    1970-01-01 UTC. ``values`` counts the readout rows, ``zeros`` the rows of value 0,
    ``zeros_p10`` the zeros of P(1|0), and ``qubits`` the qubits with one or more zeros.
    ``in_force`` counts the readout values in force at the observation that are 0.
    """

    devices: list[str]
    device: np.ndarray
    time_us: np.ndarray
    values: np.ndarray
    zeros: np.ndarray
    zeros_p10: np.ndarray
    qubits: np.ndarray
    in_force: np.ndarray


def observations(path: Path, start: datetime, end: datetime) -> Observations:
    """The counts of each observation of the file at ``path`` from ``start`` to the end of the
    day ``end``. The values in force also use the rows before ``start``."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    stamp = pa.timestamp("us", tz="UTC")
    wanted = (
        pc.field("property").isin(list(READOUT))
        & ~pc.field("backend").isin(list(RETIRED))
        & (pc.field("observed_time") < pa.scalar(end + timedelta(days=1), stamp))
    )
    columns = ["backend", "property", "qubit_a", "value", "observed_time", "calibrated_time"]
    table = pq.read_table(path, columns=columns, filters=wanted)
    encoded = pc.dictionary_encode(table["backend"]).combine_chunks()
    names = encoded.dictionary.to_pylist()
    devices = sorted(names)
    recode = np.array([devices.index(name) for name in names], dtype=np.int64)
    device = recode[encoded.indices.to_numpy()]
    time_us = micros(table["observed_time"])
    stated_us = micros(pc.fill_null(table["calibrated_time"], table["observed_time"]))
    zero = pc.fill_null(pc.equal(table["value"], 0.0), False).to_numpy(zero_copy_only=False)
    p10 = pc.equal(table["property"], P10).to_numpy(zero_copy_only=False)
    qubit = table["qubit_a"].to_numpy(zero_copy_only=False)
    del table
    _, series = np.unique(np.stack([device, p10, qubit]), axis=1, return_inverse=True)
    inside = time_us >= micros_of(start)
    order = np.lexsort((time_us[inside], device[inside]))
    d, t, z, z10, q = (a[inside][order] for a in (device, time_us, zero, p10, qubit))
    new = np.ones(d.size, dtype=bool)
    new[1:] = (d[1:] != d[:-1]) | (t[1:] != t[:-1])
    group = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    count = starts.size
    pairs = np.unique(np.stack([group[z], q[z]]), axis=1)
    return Observations(
        devices,
        d[starts],
        t[starts],
        np.bincount(group, minlength=count),
        np.bincount(group, weights=z, minlength=count).astype(np.int64),
        np.bincount(group, weights=z & z10, minlength=count).astype(np.int64),
        np.bincount(pairs[0], minlength=count),
        zeros_in_force(series.ravel(), device, time_us, stated_us, zero, d[starts], t[starts]),
    )


def zeros_in_force(
    series: np.ndarray,
    device: np.ndarray,
    time_us: np.ndarray,
    stated_us: np.ndarray,
    zero: np.ndarray,
    at_device: np.ndarray,
    at_time_us: np.ndarray,
) -> np.ndarray:
    """For each pair of ``at_device`` and ``at_time_us``: the number of series of that device
    whose value in force at that time is 0.

    Each row has a ``series``, a ``device``, an ``observed_time`` ``time_us``, a
    ``calibrated_time`` ``stated_us`` and a mask ``zero`` of the value 0. The value in force of a
    series at a time is the value of its last row with ``time_us`` at or before that time. Of two
    rows with the same ``time_us``, the row with the later ``stated_us`` is the last row.
    """
    order = np.lexsort((stated_us, time_us, series))
    one = series[order]
    now = zero[order].astype(np.int64)
    before = np.concatenate(([0], now[:-1]))
    before[np.concatenate(([True], one[1:] != one[:-1]))] = 0
    # The sum of the steps of a series up to a time is 1 when its last value is 0, else 0.
    step = now - before
    by_time = np.lexsort((time_us[order], device[order]))
    key = device[order][by_time] * _DEVICE_STRIDE + time_us[order][by_time]
    total = np.concatenate(([0], np.cumsum(step[by_time])))
    upto = np.searchsorted(key, at_device * _DEVICE_STRIDE + at_time_us, side="right")
    begin = np.searchsorted(key, at_device * _DEVICE_STRIDE, side="left")
    return total[upto] - total[begin]


def device_statistics(found: Observations) -> dict[str, dict[str, Any]]:
    """For each device: the number of observations, the median number of readout rows of an
    observation, the statistics of its zeros, and the median number of qubits with a zero."""
    out = {}
    for code, name in enumerate(found.devices):
        here = found.device == code
        if not here.any():
            continue
        zeros = found.zeros[here]
        out[name] = {
            "observations": int(here.sum()),
            "median_values": float(np.median(found.values[here])),
            "mean_zeros": float(zeros.mean()),
            "median_zeros": float(np.median(zeros)),
            "low_tenth": float(np.quantile(zeros, 0.1)),
            "high_tenth": float(np.quantile(zeros, 0.9)),
            "most": int(zeros.max()),
            "mean_zeros_p10": float(found.zeros_p10[here].mean()),
            "share_with_a_zero": float(np.mean(zeros > 0)),
            "median_qubits": float(np.median(found.qubits[here])),
            "mean_in_force": float(found.in_force[here].mean()),
            "median_in_force": float(np.median(found.in_force[here])),
            "share_with_a_zero_in_force": float(np.mean(found.in_force[here] > 0)),
        }
    return out


def by_period(found: Observations, period: str) -> dict[str, dict[str, Any]]:
    """For each UTC period of ``observed_time``, ``month`` or ``day``: the mean and the largest
    number of zeros of an observation of each device, the mean number of zeros in force, and the
    number of observations."""
    unit = "M" if period == "month" else "D"
    keys = found.time_us.astype("datetime64[us]").astype(f"datetime64[{unit}]").astype(str)
    out: dict[str, dict[str, Any]] = {}
    for key in np.unique(keys).tolist():
        here = keys == key
        entry: dict[str, Any] = {"observations": int(here.sum()), "devices": {}}
        for code, name in enumerate(found.devices):
            one = here & (found.device == code)
            zeros = found.zeros[one]
            if zeros.size:
                entry["devices"][name] = {
                    "mean": float(zeros.mean()),
                    "most": int(zeros.max()),
                    "mean_in_force": float(found.in_force[one].mean()),
                }
        out[key] = entry
    return out


def last_days(found: Observations, end: datetime, days: int = LAST_DAYS) -> Observations:
    """The observations of the last ``days`` UTC days up to the day ``end``."""
    kept = found.time_us >= micros_of(end) - (days - 1) * _US_PER_DAY
    return Observations(found.devices, *(a[kept] for a in found[1:]))


def report(path: Path, start: datetime, end: datetime) -> str:
    """The zeros of each observation of the file at ``path`` as text."""
    found = observations(path, start, end)
    lines = [
        f"window {start:%Y-%m-%d} to {end:%Y-%m-%d}, observed_time in UTC;"
        " one observation = one observed_time of one device with readout rows"
    ]
    for name, s in device_statistics(found).items():
        lines.append(
            f"{name}: {s['observations']} observations, {s['median_values']:.0f} readout values"
            f" in an observation; zeros in an observation: mean {s['mean_zeros']:.2f},"
            f" median {s['median_zeros']:.0f}, 10% {s['low_tenth']:.0f},"
            f" 90% {s['high_tenth']:.0f}, most {s['most']}; of them P(1|0): mean"
            f" {s['mean_zeros_p10']:.2f}; observations with a zero:"
            f" {100 * s['share_with_a_zero']:.1f}%; qubits with a zero: median"
            f" {s['median_qubits']:.0f}"
        )
    lines.append(
        "mean zeros in an observation, by month: month, "
        + ", ".join(found.devices)
        + ", observations of all devices"
    )
    for month, entry in by_period(found, "month").items():
        means = [entry["devices"].get(name) for name in found.devices]
        cells = ["-" if m is None else f"{m['mean']:.2f}" for m in means]
        lines.append(f"  {month} {' '.join(cells)} {entry['observations']}")
    lines.append(
        f"last {LAST_DAYS} days of the window, zeros in each observation (device, day, mean, most):"
    )
    days = by_period(last_days(found, end), "day")
    for name in found.devices:
        for day, entry in days.items():
            one = entry["devices"].get(name)
            if one is not None:
                lines.append(f"  {name} {day} {one['mean']:.1f} {one['most']}")
    lines.append(
        "zeros in force at an observation (the last row of each readout series at or before"
        " the observed_time):"
    )
    for name, s in device_statistics(found).items():
        lines.append(
            f"{name}: mean {s['mean_in_force']:.2f}, median {s['median_in_force']:.0f};"
            f" observations with a zero in force: {100 * s['share_with_a_zero_in_force']:.1f}%"
        )
    lines.append(
        "mean zeros in force at an observation, by month: month, "
        + ", ".join(found.devices)
        + ", observations of all devices"
    )
    for month, entry in by_period(found, "month").items():
        means = [entry["devices"].get(name) for name in found.devices]
        cells = ["-" if m is None else f"{m['mean_in_force']:.2f}" for m in means]
        lines.append(f"  {month} {' '.join(cells)} {entry['observations']}")
    return "\n".join(lines)


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
