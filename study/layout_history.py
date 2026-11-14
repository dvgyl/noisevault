"""Does a history of seven days give a better chain of qubits than the calibration of the day?

    python -m study.layout_history train.parquet --device ibm_kingston
        [--start 2026-06-08] [--end 2026-10-26] [--step 5] [--week 7] [--qubits 4]

The input is the parquet file of the public archive of IBM calibrations. The script first
copies the rows of the device to a temporary parquet file with pyarrow. A day d gives one
profile, ``nv.from_calibration_archive(copy, device, at=<d> 12:00 UTC)``. A profile from the
copy has the same values as a profile from the full file, but another ``source_hash``.

The dates are each ``--step`` days from ``--start`` to ``--end``. On each date d:

- the day choice is ``profile(d).suggest_layout(n)``;
- the history choice is ``profile(d).suggest_layout(n, history=[profile(d - week), ...,
  profile(d - 1)])``.

The calibration of the next day, profile(d + 1), measures both choices:

(a) For each chosen qubit and pair, the ratio of the next-day value to the stated value. The
    values are the mean readout error of a qubit, the error of the typical 1-qubit gate of a
    qubit, and the lower error of the typical 2-qubit gates of a pair in its two directions.
    The stated value comes from profile(d), and for the history choice also from the typical
    values ``nv.typical([*history, profile(d)], structure=profile(d))``. The output gives the
    median and the geometric mean of the ratios.
(b) The next-day errors of the chosen chain: the sum of each qubit's 1-qubit gate error and
    mean readout error, plus each pair's 2-qubit gate error, and the three parts of the sum.
    A chain whose qubit or pair the next day disables or does not state has no sum.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import noisevault as nv
from noisevault.profile import Profile
from noisevault.table import GateNoise
from study.archive import sha256, utc_date

KEYS = ("readout", "1q", "2q")
METHODS = ("day", "history", "history_typical")
LABELS = {
    "day": "day choice, against the day",
    "history": "history choice, against the day",
    "history_typical": "history choice, against the typical values",
}
NOON = timedelta(hours=12)

Values = dict[str, list[float | None]]


def device_copy(path: Path, device: str, folder: Path) -> Path:
    """A parquet file in ``folder`` with all columns of the rows of ``device`` in ``path``."""
    import pyarrow.parquet as pq

    out = folder / path.name
    pq.write_table(pq.read_table(path, filters=[("backend", "=", device)]), out)
    return out


def chain_values(profile: Profile, chain: Sequence[int]) -> Values:
    """The readout, 1-qubit and 2-qubit values of ``chain`` that ``profile`` states. A value is
    None when the profile disables the qubit or does not state the value."""
    table = profile.table
    found: Values = {key: [] for key in KEYS}
    for q in chain:
        if table.qubit(q).disabled:
            found["readout"].append(None)
            found["1q"].append(None)
            continue
        readout = table.qubit(q).readout
        one = table.typical(1, (q,))
        found["readout"].append(None if readout is None else sum(readout) / 2)
        found["1q"].append(one.avg_infidelity if isinstance(one, GateNoise) else None)
    for a, b in zip(chain, chain[1:], strict=False):
        errors = [
            g.avg_infidelity
            for g in (table.typical(2, (a, b)), table.typical(2, (b, a)))
            if isinstance(g, GateNoise) and g.avg_infidelity is not None
        ]
        found["2q"].append(min(errors) if errors else None)
    return found


def chain_sums(values: Values) -> dict[str, float] | None:
    """The sum of each part and of all parts, or None when a value is missing."""
    if any(v is None for part in values.values() for v in part):
        return None
    sums = {key: math.fsum(v for v in values[key] if v is not None) for key in KEYS}
    return sums | {"all": math.fsum(sums.values())}


def dates(start: datetime, end: datetime, step: int) -> list[datetime]:
    """Each ``step`` days from ``start`` to ``end``, at 12:00 UTC."""
    first, count = start + NOON, (end - start).days // step + 1
    return [first + timedelta(days=step * i) for i in range(count)]


def evaluate(
    profile_on: Callable[[datetime], Profile], on: Sequence[datetime], week: int, n: int
) -> dict[str, Any]:
    """The two choices on each date of ``on`` and their measures.

    ``profile_on(day)`` gives the profile of a day. ``rows`` gives, for each date, the number
    of distinct calibrations in the history and the day, the two chains, and the stated and
    next-day sums of each chain. ``ratios`` gives the next-day / stated ratios of each method
    and part, and ``unchanged`` the number of next-day values equal to the stated value. A
    next-day value of 0 gives the ratio 0. A stated value of 0 has no ratio, and ``no_ratio``
    gives the number of these values for each method.
    """
    ratios: dict[str, dict[str, list[float]]] = {m: {key: [] for key in KEYS} for m in METHODS}
    unchanged = dict.fromkeys(METHODS, 0)
    no_ratio = dict.fromkeys(METHODS, 0)
    rows = []
    for d in on:
        today, after = profile_on(d), profile_on(d + timedelta(days=1))
        history = [profile_on(d - timedelta(days=k)) for k in range(week, 0, -1)]
        distinct = len({p.calibration_fingerprint for p in [*history, today]})
        chains = {
            "day": tuple(today.suggest_layout(n).values()),
            "history": tuple(today.suggest_layout(n, history=history).values()),
        }
        typical = nv.typical([*history, today], structure=today)
        next_day = {m: chain_values(after, chains[m]) for m in chains}
        stated = {
            "day": chain_values(today, chains["day"]),
            "history": chain_values(today, chains["history"]),
            "history_typical": chain_values(typical, chains["history"]),
        }
        for m in METHODS:
            later = next_day["day" if m == "day" else "history"]
            for key in KEYS:
                for s, x in zip(stated[m][key], later[key], strict=True):
                    if s is None or x is None:
                        continue
                    if s == 0:
                        no_ratio[m] += 1
                        continue
                    ratios[m][key].append(x / s)
                    unchanged[m] += x == s
        sums = {m: (chain_sums(stated[m]), chain_sums(next_day[m])) for m in chains}
        rows.append({"date": d, "distinct": distinct, "chains": chains, "sums": sums})
    return {"rows": rows, "ratios": ratios, "unchanged": unchanged, "no_ratio": no_ratio}


def geometric_mean(values: Sequence[float]) -> float:
    """The geometric mean, which is 0 when one value is 0."""
    if any(x == 0 for x in values):
        return 0.0
    return math.exp(statistics.fmean(math.log(x) for x in values))


def _sum(sums: Mapping[str, float] | None) -> str:
    return "broken" if sums is None else f"{sums['all']:.4f}"


def report(
    path: Path,
    device: str,
    start: datetime,
    end: datetime,
    *,
    step: int = 5,
    week: int = 7,
    n: int = 4,
) -> str:
    """The table of each date and the two measures, as text."""
    on = dates(start, end, step)
    profiles: dict[datetime, Profile] = {}
    with tempfile.TemporaryDirectory() as folder:
        copy = device_copy(path, device, Path(folder))

        def profile_on(day: datetime) -> Profile:
            if day not in profiles:
                profiles[day] = nv.from_calibration_archive(copy, device, at=day)
            return profiles[day]

        found = evaluate(profile_on, on, week, n)
    rows = found["rows"]
    lines = [
        f"file SHA-256 {sha256(path)}",
        f"device {device}, {len(on)} dates, chains of {n} qubits, history of {week} days",
        "",
        "date        distinct  day chain             history chain         stated sum day / hist"
        "   next-day sum day / hist",
    ]
    for row in rows:
        chains, sums = row["chains"], row["sums"]
        lines.append(
            f"{row['date']:%Y-%m-%d}  {row['distinct']:>8}  {str(chains['day']):<20}"
            f"  {str(chains['history']):<20}"
            f"  {_sum(sums['day'][0])} / {_sum(sums['history'][0])}"
            f"        {_sum(sums['day'][1])} / {_sum(sums['history'][1])}"
        )
    same = sum(
        r["chains"]["day"] in (r["chains"]["history"], r["chains"]["history"][::-1]) for r in rows
    )
    lines += [
        "",
        f"distinct: the number of different calibrations in the {week + 1} profiles of the history",
        f"dates where both choices give the same chain: {same} of {len(rows)}",
        "",
        "(a) next-day value / stated value, over the chosen qubits and pairs of all dates",
    ]
    for m in METHODS:
        for key in KEYS:
            xs = found["ratios"][m][key]
            if not xs:
                lines.append(f"  {LABELS[m]:<44} {key:<8} n=  0")
                continue
            lines.append(
                f"  {LABELS[m]:<44} {key:<8} n={len(xs):>3}  median {statistics.median(xs):.3f}"
                f"  geometric mean {geometric_mean(xs):.3f}"
            )
        count = sum(len(xs) for xs in found["ratios"][m].values())
        lines.append(
            f"  {LABELS[m]}: {found['unchanged'][m]} of {count} next-day values equal the stated"
            " value"
        )
        if found["no_ratio"][m]:
            lines.append(f"  {LABELS[m]}: {found['no_ratio'][m]} stated values of 0 have no ratio")
    lines += [
        "",
        f"(b) next-day errors of the chosen chain (sums over {n} qubits and {n - 1} pairs)",
    ]
    complete = [
        r["sums"] for r in rows if None not in (r["sums"]["day"][1], r["sums"]["history"][1])
    ]
    for part in ("all", *KEYS):
        pairs = [(s["day"][1][part], s["history"][1][part]) for s in complete]
        if not pairs:
            lines.append(f"  {part:<8} n=0")
            continue
        day, hist = [p[0] for p in pairs], [p[1] for p in pairs]
        lower = sum(h < a for a, h in pairs)
        higher = sum(h > a for a, h in pairs)
        lines.append(
            f"  {part:<8} n={len(pairs)}  median day {statistics.median(day):.4f}"
            f"  median history {statistics.median(hist):.4f}"
            f"  mean day {statistics.fmean(day):.4f}  mean history {statistics.fmean(hist):.4f}"
            f"  history lower on {lower}, higher on {higher},"
            f" equal on {len(pairs) - lower - higher}"
        )
    broken = {m: sum(r["sums"][m][1] is None for r in rows) for m in ("day", "history")}
    lines.append(
        "  chains with a qubit or pair that the next day disables or does not state:"
        f" day {broken['day']}, history {broken['history']}"
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("parquet", type=Path, help="the parquet file of the dataset")
    parser.add_argument("--device", required=True, help="the device, such as ibm_kingston")
    parser.add_argument("--start", type=utc_date, default=utc_date("2026-06-08"))
    parser.add_argument("--end", type=utc_date, default=utc_date("2026-10-26"))
    parser.add_argument("--step", type=int, default=5, help="the days between two dates")
    parser.add_argument("--week", type=int, default=7, help="the days of the history")
    parser.add_argument("--qubits", type=int, default=4, help="the qubits of a chain")
    args = parser.parse_args(argv)
    if not args.parquet.is_file():
        print(f"error: {args.parquet} is not a file", file=sys.stderr)
        return 2
    if args.end < args.start:
        print("error: --end is before --start", file=sys.stderr)
        return 2
    print(
        report(
            args.parquet,
            args.device,
            args.start,
            args.end,
            step=args.step,
            week=args.week,
            n=args.qubits,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
