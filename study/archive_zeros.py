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
- the qubits with one or more zeros.

A readout value that IBM did not state again at an observation has no row there. Thus a 0 that
is in force from an earlier observation does not count. The output gives, for each device, the
statistics of these counts over its observations. It then gives the mean number of zeros of an
observation for each UTC month of ``observed_time``, and the mean and the largest number for
each UTC day of the last ``LAST_DAYS`` days of the window.

The script reads only five columns of the file with pyarrow, and it needs no other database.
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


class Observations(NamedTuple):
    """One entry for each observation, sorted by device and then by time.

    ``device`` indexes ``devices``. ``time_us`` is ``observed_time`` in microseconds since
    1970-01-01 UTC. ``values`` counts the readout rows, ``zeros`` the rows of value 0,
    ``zeros_p10`` the zeros of P(1|0), and ``qubits`` the qubits with one or more zeros.
    """

    devices: list[str]
    device: np.ndarray
    time_us: np.ndarray
    values: np.ndarray
    zeros: np.ndarray
    zeros_p10: np.ndarray
    qubits: np.ndarray


def observations(path: Path, start: datetime, end: datetime) -> Observations:
    """The counts of each observation of the file at ``path`` from ``start`` to the end of the
    day ``end``."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    stamp = pa.timestamp("us", tz="UTC")
    wanted = (
        pc.field("property").isin(list(READOUT))
        & ~pc.field("backend").isin(list(RETIRED))
        & (pc.field("observed_time") >= pa.scalar(start, stamp))
        & (pc.field("observed_time") < pa.scalar(end + timedelta(days=1), stamp))
    )
    table = pq.read_table(
        path, columns=["backend", "property", "qubit_a", "value", "observed_time"], filters=wanted
    )
    encoded = pc.dictionary_encode(table["backend"]).combine_chunks()
    names = encoded.dictionary.to_pylist()
    devices = sorted(names)
    recode = np.array([devices.index(name) for name in names], dtype=np.int64)
    device = recode[encoded.indices.to_numpy()]
    time_us = micros(table["observed_time"])
    zero = pc.fill_null(pc.equal(table["value"], 0.0), False).to_numpy(zero_copy_only=False)
    p10 = pc.equal(table["property"], P10).to_numpy(zero_copy_only=False)
    qubit = table["qubit_a"].to_numpy(zero_copy_only=False)
    del table
    order = np.lexsort((time_us, device))
    device, time_us, zero, p10, qubit = (a[order] for a in (device, time_us, zero, p10, qubit))
    new = np.ones(device.size, dtype=bool)
    new[1:] = (device[1:] != device[:-1]) | (time_us[1:] != time_us[:-1])
    group = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    count = starts.size
    pairs = np.unique(np.stack([group[zero], qubit[zero]]), axis=1)
    return Observations(
        devices,
        device[starts],
        time_us[starts],
        np.bincount(group, minlength=count),
        np.bincount(group, weights=zero, minlength=count).astype(np.int64),
        np.bincount(group, weights=zero & p10, minlength=count).astype(np.int64),
        np.bincount(pairs[0], minlength=count),
    )


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
        }
    return out


def by_period(found: Observations, period: str) -> dict[str, dict[str, Any]]:
    """For each UTC period of ``observed_time``, ``month`` or ``day``: the mean and the largest
    number of zeros of an observation of each device, and the number of observations."""
    unit = "M" if period == "month" else "D"
    keys = found.time_us.astype("datetime64[us]").astype(f"datetime64[{unit}]").astype(str)
    out: dict[str, dict[str, Any]] = {}
    for key in np.unique(keys).tolist():
        here = keys == key
        entry: dict[str, Any] = {"observations": int(here.sum()), "devices": {}}
        for code, name in enumerate(found.devices):
            zeros = found.zeros[here & (found.device == code)]
            if zeros.size:
                entry["devices"][name] = {"mean": float(zeros.mean()), "most": int(zeros.max())}
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
