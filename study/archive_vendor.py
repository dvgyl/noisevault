"""The values of the public archive of IBM calibrations against the values that IBM serves.

    python -m study.archive_vendor train.parquet

The script picks a fixed random sample of times in a window. Each time is inside the span of
its device in the archive. For each device and time, it makes two profiles:

- from the archive, with :func:`noisevault.sources.hf_archive.from_calibration_archive`;
- from IBM's public calibration endpoint, with :func:`noisevault.sources.ibm_public.pull`. This
  needs the network, but no account.

The script compares each value that the two profiles hold: T1, T2, the two readout probabilities
of each qubit, and the error of each gate on each qubit or pair. The relative difference of two
values is ``abs(a - b) / max(abs(a), abs(b))``. Two values agree when their relative difference
is ``TOLERANCE`` or less. The script writes text to standard output.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

import noisevault as nv
from noisevault.profile import Profile
from noisevault.sources import hf_archive, ibm_public
from study.archive_steps import observation_times

DEVICES = ("ibm_fez", "ibm_kingston", "ibm_marrakesh")
START, END = datetime(2026, 2, 20, tzinfo=UTC), datetime(2026, 11, 9, tzinfo=UTC)
TIMES_FOR_EACH, SEED, TOLERANCE = 4, 3, 1e-9
LEVELS = (1e-9, 1e-6, 1e-3, 1e-2)

Key = tuple[Any, ...]
Pull = Callable[..., Profile]


def values(profile: Profile) -> dict[Key, float]:
    """Each value of ``profile`` that the comparison uses, by its name and its qubits."""
    data = profile.model_dump(mode="json")
    found: dict[Key, float] = {}
    for qubit in data["qubits"]:
        index = qubit["index"]
        for name in ("t1_us", "t2_us"):
            if qubit.get(name) is not None:
                found[(name, index)] = qubit[name]
        for name in ("p1_given_0", "p0_given_1"):
            value = (qubit.get("readout") or {}).get(name)
            if value is not None:
                found[(name, index)] = value
    for entry in data["calibrations"]:
        if entry.get("avg_infidelity") is not None:
            found[(entry["gate"], *entry["qubits"])] = entry["avg_infidelity"]
    return found


def kind(key: Key) -> str:
    """The kind of the value of ``key``: t1, t2, readout, one-qubit gate or two-qubit gate."""
    name = key[0]
    if name in ("t1_us", "t2_us"):
        return name[:2]
    if name in ("p1_given_0", "p0_given_1"):
        return "readout"
    return "two-qubit gate" if len(key) == 3 else "one-qubit gate"


def sample_times(
    devices: Sequence[str],
    times_for_each: int,
    seed: int,
    start: datetime,
    end: datetime,
    spans: Mapping[str, tuple[datetime, datetime]] | None = None,
) -> list[tuple[str, datetime]]:
    """``times_for_each`` random times for each device, to the minute, between ``start`` and
    ``end``. The times of one device are in order.

    ``spans`` gives the first and the last observation time of each device in the archive. A
    time outside the span of its device is drawn again inside the span, because the archive
    holds no complete calibration before the span, and no new calibration after it.
    """
    rng = np.random.default_rng(seed)
    minutes = int((end - start).total_seconds() // 60)
    found = []
    for device in devices:
        first, last = (spans or {}).get(device, (start, end))
        # The whole minutes from ``start`` that are in the window and in the span.
        low = max(0, math.ceil((first - start).total_seconds() / 60))
        high = min(minutes - 1, math.floor((last - start).total_seconds() / 60))
        if low > high:
            raise ValueError(
                f"the archive holds no calibration of {device} between {start:%Y-%m-%d %H:%M}"
                f" and {end:%Y-%m-%d %H:%M}"
            )
        drawn = rng.integers(0, minutes, times_for_each).tolist()
        again = [m if low <= m <= high else int(rng.integers(low, high + 1)) for m in drawn]
        found += [(device, start + timedelta(minutes=m)) for m in sorted(again)]
    return found


def archive_spans(path: Path) -> dict[str, tuple[datetime, datetime]]:
    """The first and the last observation time of each device in the archive file."""
    whole = (datetime(1970, 1, 2, tzinfo=UTC), datetime(2200, 1, 1, tzinfo=UTC))
    return {
        device: tuple(datetime.fromtimestamp(int(t) / 1e6, UTC) for t in (times[0], times[-1]))
        for device, times in observation_times(path, *whole).items()
        if times.size
    }


def compare(
    path: Path,
    device: str,
    at: datetime,
    pull: Pull = ibm_public.pull,
) -> dict[str, Any]:
    """The relative differences of the values of ``device`` at ``at`` in the archive file and
    from ``pull``, by kind, and the number of values that only one side holds."""
    archive = values(hf_archive.from_calibration_archive(path, device, at=at))
    vendor = values(pull(device, at=at))
    gaps: dict[str, list[float]] = {}
    for key in sorted(archive.keys() & vendor.keys(), key=str):
        a, b = archive[key], vendor[key]
        gap = abs(a - b) / max(abs(a), abs(b)) if a != b else 0.0
        gaps.setdefault(kind(key), []).append(gap)
    return {
        "device": device,
        "at": at,
        "gaps": dict(sorted(gaps.items())),
        "only_archive": len(archive.keys() - vendor.keys()),
        "only_vendor": len(vendor.keys() - archive.keys()),
    }


def report(
    path: Path,
    devices: Sequence[str] = DEVICES,
    times_for_each: int = TIMES_FOR_EACH,
    seed: int = SEED,
    pull: Pull = ibm_public.pull,
) -> str:
    """The comparison at each sampled time, and the share of values that agree for each kind."""
    lines = [
        f"noisevault {nv.__version__}; seed {seed}; {times_for_each} times for each device",
        "device, time, values in both, share that agree, only in the archive, only from the vendor",
    ]
    by_kind: dict[str, list[float]] = {}
    spans = archive_spans(path)
    for device, at in sample_times(devices, times_for_each, seed, START, END, spans):
        found = compare(path, device, at, pull)
        both = sum(len(g) for g in found["gaps"].values())
        agree = sum(g <= TOLERANCE for gaps in found["gaps"].values() for g in gaps)
        shares = ", ".join(
            f"{name} {np.mean(np.array(gaps) <= TOLERANCE):.2f}"
            for name, gaps in found["gaps"].items()
        )
        lines.append(
            f"{device}, {at:%Y-%m-%dT%H:%MZ}, {both}, {_share(agree, both)},"
            f" {found['only_archive']}, {found['only_vendor']}; {shares}"
        )
        for name, gaps in found["gaps"].items():
            by_kind.setdefault(name, []).extend(gaps)
    lines.append(
        "kind, values in both, share within "
        + ", ".join(f"1e{round(math.log10(level))}" for level in LEVELS)
        + " (relative), median relative difference of the values that differ"
    )
    everything: list[float] = []
    for name, gaps in sorted(by_kind.items()):
        array = np.array(gaps)
        everything += gaps
        differ = array[array > TOLERANCE]
        shares = ", ".join(f"{np.mean(array <= level):.4f}" for level in LEVELS)
        lines.append(
            f"{name}, {array.size}, {shares}, {np.median(differ) if differ.size else 0:.3g}"
        )
    array = np.array(everything)
    if array.size:
        lines.append(
            f"all, {array.size}, " + ", ".join(f"{np.mean(array <= level):.4f}" for level in LEVELS)
        )
    return "\n".join(lines)


def _share(part: int, total: int) -> str:
    return f"{part / total:.4f}" if total else "-"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("parquet", type=Path, help="the parquet file of the dataset")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--times", type=int, default=TIMES_FOR_EACH, help="times for each device")
    args = parser.parse_args(argv)
    if not args.parquet.is_file():
        print(f"error: {args.parquet} is not a file", file=sys.stderr)
        return 2
    print(report(args.parquet, times_for_each=args.times, seed=args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
