"""Summarize the analysis over all runs, as the plan states.

    python -m study.summarize results/analysis.json --out results

The script reads ``analysis.json`` of ``study.analyze`` and writes ``summary.json`` and
``summary.md``.

The runs that count are the IBM runs from the week of 2026-11-16 to the week of 2027-01-04. A
run before that window is a pilot run, a run after it is a late run, and a run of another vendor
in the window is a secondary run. A trial run (``best-trial.*`` or ``median-trial.*``) is one
more run in a week that has its runs. The primary estimates use only the runs that count, and a
trial run never counts.

For each primary comparison of ``PLAN.md``, the result is the estimate over device-weeks of
``study.estimate``, with the Holm correction of the five tests. A device-week is the set of runs
of one device in one ISO week, and Y is the mean of delta over its run on the best chain and its
run on the median chain. The summary also gives the descriptions of the plan: the estimate with
f = 1, m and s of each device, the interval for a new week, the median standard error of delta
of a run, the sign test of Y, the verdict counts of the runs, the estimate with the quadratic
score and without the marked runs, and the strata. The secondary results use the same estimate.

The verdict counts of the pilot runs, of the late runs and of the trial runs are exploratory.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime
from math import log, log10, sqrt
from pathlib import Path
from typing import Any

from noisevault.compare import FACTOR_RANGE
from study.analyze import (
    CHAIN_KINDS,
    CLIFFORD_COMPARISONS,
    MISFIT_MODELS,
    PARTS_OF_5,
    PRIMARY,
    PROBE_MODELS,
    SECONDARY,
    STALE,
    STALE_DAYS,
    WIDE_COMPARISONS,
    device_week,
)
from study.estimate import LEVEL, SMALL, Estimate, estimate, holm, sign_test

FORMAT = "2"
AGE_KINDS = ("readout", "one_qubit", "two_qubit")
MIN_POINTS = 5
# The runs that count: from the first week to the end of the last week, in UTC.
FIRST_WEEK = datetime(2026, 11, 16, tzinfo=UTC)
AFTER_LAST_WEEK = datetime(2027, 1, 11, tzinfo=UTC)
PERIODS = ("pilot", "counted", "late", "secondary", "trial")
# A fitted factor is at a limit of the search when ln f is this near to the ln of a limit.
LIMIT_TOLERANCE = 0.01
# The forms of a device-week value: the mean of the two runs, the run of one chain kind, the
# run on the best chain minus the run on the median chain, or the mean of the runs given.
Mode = str
NO_ESTIMATE = (
    "No device has 4 complete device-weeks of runs that count, so the summary gives no estimate"
    " and no test."
)


def period(row: Mapping[str, Any]) -> str:
    """``trial`` for a trial run, at any time. Else ``pilot`` before the first week, ``late``
    after the last week, ``counted`` for an IBM run in the window and ``secondary`` for a run of
    another vendor in the window. A trial run is in no estimate, so it never makes a device-week
    complete or not complete."""
    if row.get("trial"):
        return "trial"
    when = _time(row["run_at"])
    if when < FIRST_WEEK:
        return "pilot"
    if when >= AFTER_LAST_WEEK:
        return "late"
    return "counted" if row.get("vendor", "ibm") == "ibm" else "secondary"


def unique(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """The first row of each ``counts_sha256``, and the number of rows removed.

    One run must count one time, also when the analysis comes from another tool.
    """
    found: dict[str, dict[str, Any]] = {}
    for row in rows:
        found.setdefault(row["counts_sha256"], row)
    return list(found.values()), len(rows) - len(found)


def delta(row: Mapping[str, Any], a: str, b: str, group: str, rule: str = "log") -> float | None:
    """The score of model a minus the score of model b, per shot, on ``group`` of one run. None
    when the run has no such test or the difference is not finite."""
    test = _test(row, a, b, group)
    if test is None:
        return None
    return _finite(test[rule]["difference"])


def device_weeks(
    rows: Sequence[Mapping[str, Any]],
    value: Callable[[Mapping[str, Any]], float | None],
    mode: Mode = "pair",
) -> tuple[dict[str, dict[date, float]], dict[str, int]]:
    """The value Y of each complete device-week, and the number of the other device-weeks for
    each reason.

    ``mode`` gives Y from the values of the runs of a device-week:

    - ``pair``: the mean of the run on the best chain and the run on the median chain.
    - ``best`` or ``median``: the value of the run on that chain.
    - ``contrast``: the run on the best chain minus the run on the median chain.
    - ``mean``: the mean of all runs of the device-week.

    The reasons are ``missing_run`` (a run of the mode is not there), ``extra_run`` (more than
    one run of a chain kind, or a run with no chain kind) and ``no_value`` (a run of the mode
    has no finite value).
    """
    weeks: dict[tuple[str, date], list[Mapping[str, Any]]] = {}
    for row in rows:
        weeks.setdefault(device_week(row), []).append(row)
    values: dict[str, dict[date, float]] = {}
    incomplete: dict[str, int] = {}
    for (device, week), found in sorted(weeks.items()):
        y, reason = _week_value(found, value, mode)
        if y is None:
            incomplete[reason] = incomplete.get(reason, 0) + 1
        else:
            values.setdefault(device, {})[week] = y
    return values, incomplete


def summarize(analysis: Mapping[str, Any]) -> dict[str, Any]:
    rows, duplicates = unique(analysis["rows"])
    by_period: dict[str, list[dict[str, Any]]] = {p: [] for p in PERIODS}
    for row in rows:
        by_period[period(row)].append(row)
    counted = by_period["counted"]
    primary = _primary(counted)
    return {
        "nv_study_summary": FORMAT,
        "runs": len(rows),
        "duplicate_rows_removed": duplicates,
        "devices": sorted({r["device"] for r in rows}),
        "level": LEVEL,
        "window": [FIRST_WEEK.isoformat(), AFTER_LAST_WEEK.isoformat()],
        "periods": {p: len(found) for p, found in by_period.items()},
        "stated_calibration": _stated(counted),
        "primary": primary,
        "primary_note": None
        if any(c["estimate"]["estimate"] is not None for c in primary)
        else NO_ESTIMATE,
        "runs_marked": sum(1 for r in counted if r.get("marked") is True),
        "secondary": [_secondary(counted, a, b, group) for a, b, group in SECONDARY],
        "staleness": staleness(counted),
        "staleness_ages": _staleness_ages(rows),
        "age_against_error": age_against_error(counted),
        "age_against_error_without_marked_runs": age_against_error(
            [r for r in counted if r.get("marked") is not True]
        ),
        "misfit": misfit(counted),
        "probes": probes(rows),
        "wide_overlap": wide_overlap(counted),
        "fits": fit_notes(counted),
        "clifford": clifford(counted),
        "twirl_check": twirl_check(rows),
        "shot_checks": shot_checks(analysis.get("shot_checks", ())),
        "exploratory": {
            name: verdict_family(by_period[name])
            for name in ("pilot", "late", "secondary", "trial")
        },
    }


def verdict_family(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The verdict counts of the five primary comparisons over ``rows``, with the sign test over
    runs and its Holm correction."""
    comparisons = [verdicts(rows, a, b, group) for a, b, group in PRIMARY]
    adjusted = holm([c["p_value"] for c in comparisons])
    for found, p_holm in zip(comparisons, adjusted, strict=True):
        found["p_holm"] = p_holm
        found["confirmed"] = p_holm < LEVEL
    return comparisons


def verdicts(rows: Sequence[dict[str, Any]], a: str, b: str, group: str) -> dict[str, Any]:
    """The runs in which the first model is better, the second model is better, or neither, by
    the verdict of the log score, for each device, and the sign test over the runs."""
    by_device: dict[str, dict[str, int]] = {}
    for row in rows:
        test = _test(row, a, b, group)
        if test is None:
            continue
        counts = by_device.setdefault(row["device"], {"a": 0, "b": 0, "neither": 0})
        counts[test["log"]["better"]] += 1
    total = {
        key: sum(counts[key] for counts in by_device.values()) for key in ("a", "b", "neither")
    }
    return {
        "a": a,
        "b": b,
        "group": group,
        "by_device": dict(sorted(by_device.items())),
        "total": total,
        "p_value": sign_test(total["a"], total["b"]),
    }


def staleness(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The staleness curve for the mean of the two runs and for each chain kind.

    A point is the estimate for delta of ``stale(d)`` against ``calibration`` on ``all4``, with
    its 95% interval and the number of device-weeks. The trend is the estimate for the
    least-squares slope of delta against log10(d) of each run. For the two runs together, Y of a
    device-week is the mean of the slopes of its two runs.
    """
    found: dict[str, Any] = {}
    for kind, mode in (("both", "pair"), *((k, k) for k in CHAIN_KINDS)):
        points = []
        for days, name in zip(STALE_DAYS, STALE, strict=True):

            def value(row: Mapping[str, Any], name: str = name) -> float | None:
                d = delta(row, "calibration", name, "all4")
                return None if d is None else -d

            result = estimate(*device_weeks(rows, value, mode))
            points.append({"days": days, **_brief(result)})
        slope = estimate(*device_weeks(rows, stale_slope, mode))
        found[kind] = {"points": points, "slope": _brief(slope)}
    return found


def stale_slope(row: Mapping[str, Any]) -> float | None:
    """The least-squares slope of delta of ``stale(d)`` against ``calibration`` on ``all4``
    against log10(d), over the ages that the run has. None for fewer than 2 ages."""
    points = []
    for days, name in zip(STALE_DAYS, STALE, strict=True):
        d = delta(row, "calibration", name, "all4")
        if d is not None:
            points.append((log10(days), -d))
    if len(points) < 2:
        return None
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    spread = sum((x - mean_x) ** 2 for x, _ in points)
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / spread


def spearman(first: Sequence[float], second: Sequence[float]) -> float | None:
    """The Spearman rank correlation, with the mean rank for tied values.

    The result is None for fewer than ``MIN_POINTS`` points or when one input is constant.
    """
    if len(first) < MIN_POINTS:
        return None
    a, b = _ranks(first), _ranks(second)
    if len(set(a)) == 1 or len(set(b)) == 1:
        return None
    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    cross = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True))
    spread_a = sum((x - mean_a) ** 2 for x in a)
    spread_b = sum((y - mean_b) ** 2 for y in b)
    return cross / sqrt(spread_a * spread_b)


def at_limit(factor: float | None) -> bool:
    """True when a fitted factor is at a limit of the search of ``nv compare``."""
    if factor is None or factor <= 0:
        return False
    return any(abs(log(factor) - log(limit)) < LIMIT_TOLERANCE for limit in FACTOR_RANGE)


def age_against_error(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each device, and for the qubits and the pairs separately, the Spearman rank
    correlation between the age of a value at the run time and the absolute value of ln f.

    f is the factor of that qubit or pair in the ``locus`` model. The age of a qubit is the age
    of its 1-qubit gate errors, and the age of a pair is the age of its 2-qubit gate errors. The
    result has no p-value, because the points of one device are not independent. It gives the
    number of factors at a limit of the search, and the correlation also without them.
    """
    points: dict[tuple[str, str], list[tuple[float, float, bool]]] = {}
    for row in rows:
        loci = (row.get("ages", {}).get("calibration") or {}).get("loci")
        fits = row.get("fits", {}).get("locus")
        if not loci or not fits:
            continue
        found = [
            (str(q["qubit"]), "qubits", "one_qubit", q["factor"]) for q in fits.get("qubits", ())
        ]
        found += [
            ("-".join(map(str, p["qubits"])), "pairs", "two_qubit", p["factor"])
            for p in fits.get("pairs", ())
        ]
        for key, kind, age_kind, factor in found:
            age = loci.get(kind, {}).get(key, {}).get(age_kind)
            if age is None or factor is None or factor <= 0:
                continue
            entry = (age, abs(log(factor)), at_limit(factor))
            points.setdefault((row["device"], kind), []).append(entry)
    result = []
    for (device, kind), found_points in sorted(points.items()):
        ages, errors, limits = zip(*found_points, strict=True)
        inside = [(x, y) for x, y, limit in found_points if not limit]
        result.append(
            {
                "device": device,
                "loci": kind,
                "points": len(found_points),
                "spearman": spearman(ages, errors),
                "at_limit": sum(limits),
                "spearman_without_limits": spearman(*zip(*inside, strict=True)) if inside else None,
            }
        )
    return result


def misfit(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each device and each model of the misfit, the median over the runs of the misfit per
    shot and of the part of the excess from outcomes with fewer than 1 expected shot."""
    found: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for row in rows:
        for model, values in (row.get("misfit") or {}).items():
            if values.get("misfit") is None:
                continue
            found.setdefault((row["device"], model), []).append((values["misfit"], values["rare"]))
    result = []
    for model in MISFIT_MODELS:
        for (device, name), values in sorted(found.items()):
            if name != model:
                continue
            result.append(
                {
                    "device": device,
                    "model": model,
                    "runs": len(values),
                    "median_misfit": statistics.median(v for v, _ in values),
                    "median_rare": statistics.median(r for _, r in values),
                }
            )
    return result


def probes(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The probe table of each run that has one, and for each chain qubit the growth ratio.

    The growth ratio is (wrong fraction minus baseline) of the probe with the largest m, divided
    by the same value of the probe with the smallest m. m is the number after ``_d`` in the name.
    The baseline is the wrong fraction of the qubit in the ``readout`` circuit. The ratio is None
    when the divisor is not above 0.
    """
    result = []
    for row in rows:
        table = row.get("probes") or {}
        circuits = table.get("circuits") or []
        if not circuits:
            continue
        baseline = {b["qubit"]: b["measured"] for b in table.get("baseline", ())}
        ordered = sorted(circuits, key=lambda c: _depth(c["name"]))
        smallest, largest = ordered[0], ordered[-1]
        growth = []
        if _depth(largest["name"]) > _depth(smallest["name"]):
            small = {q["qubit"]: q["measured"] for q in smallest["qubits"]}
            for q in largest["qubits"]:
                base = baseline.get(q["qubit"])
                if base is None or q["qubit"] not in small:
                    continue
                divisor = small[q["qubit"]] - base
                ratio = (q["measured"] - base) / divisor if divisor > 0 else None
                growth.append({"qubit": q["qubit"], "ratio": ratio})
        result.append(
            {
                "device": row["device"],
                "chain": row["chain"],
                "run_at": row["run_at"],
                "period": period(row),
                "baseline": table.get("baseline", []),
                "circuits": circuits,
                "growth": growth,
            }
        )
    return result


def wide_overlap(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each run with a wide comparison: the number of qubits of the 8-qubit chain that are in
    the chain of the run, and delta of each wide comparison."""
    result = []
    for row in rows:
        found = {f"{a} - {b}": delta(row, a, b, group) for a, b, group in WIDE_COMPARISONS}
        if all(v is None for v in found.values()):
            continue
        result.append(
            {
                "device": row["device"],
                "chain": row["chain"],
                "run_at": row["run_at"],
                "overlap": row.get("wide_chain_overlap"),
                "delta": found,
            }
        )
    return result


def fit_notes(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The number of runs with a fitted factor that the fit does not identify, the number of
    runs with a fitted factor at a limit of the search, and the number of runs for each number
    of different calibrations in the median of ``averaged(7)``."""
    unidentified = limited = 0
    calibrations: dict[str, int] = {}
    for row in rows:
        fits = row.get("fits") or {}
        two = fits.get("two_factor") or {}
        if any("not_identified" in (two.get(k) or {}) for k in ("gates", "readout")):
            unidentified += 1
        factors = [(two.get(k) or {}).get("factor") for k in ("gates", "readout")]
        factors.append((fits.get("readout_measured") or {}).get("gates"))
        locus = fits.get("locus") or {}
        factors += [x["factor"] for kind in ("qubits", "pairs") for x in locus.get(kind, ())]
        if any(at_limit(f) for f in factors):
            limited += 1
        week = fits.get("averaged(7)")
        if week:
            key = str(week["calibrations"])
            calibrations[key] = calibrations.get(key, 0) + 1
    return {
        "runs_with_a_factor_not_identified": unidentified,
        "runs_with_a_factor_at_a_limit": limited,
        "averaged_7_calibrations": dict(sorted(calibrations.items())),
    }


def clifford(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each Clifford comparison, one value for each device-week: the mean over its two runs
    of the difference of the qubit score per shot on the Clifford circuits together. The result
    gives the values, the verdict of each run, and the estimate over the device-weeks."""
    result = []
    for a, b in CLIFFORD_COMPARISONS:

        def value(row: Mapping[str, Any], a: str = a, b: str = b) -> float | None:
            test = _clifford_test(row, a, b)
            return None if test is None else _finite(test["qubit"]["log"]["difference"])

        values, incomplete = device_weeks(rows, value, "pair")
        weeks = []
        for device, found in sorted(values.items()):
            for week, y in sorted(found.items()):
                runs = [r for r in rows if device_week(r) == (device, week)]
                tests = [_clifford_test(r, a, b) for r in runs]
                weeks.append(
                    {
                        "device": device,
                        "week": week.isoformat(),
                        "value": y,
                        "verdicts": [t["qubit"]["log"]["better"] for t in tests if t],
                        "largest_sampling_ratio": _largest(
                            t["qubit"].get("sampling_ratio") for t in tests if t
                        ),
                    }
                )
        result.append(
            {
                "a": a,
                "b": b,
                "device_weeks": weeks,
                "estimate": estimate(values, incomplete).to_dict(),
            }
        )
    return result


def twirl_check(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The twirl check of each model and each 8-qubit circuit of each run: the log score from
    the exact simulator and from the stabilizer forecast, per shot."""
    return [
        {"device": row["device"], "chain": row["chain"], "run_at": row["run_at"], **check}
        for row in rows
        for check in row.get("twirl_check", ())
    ]


def shot_checks(checks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """For each device-week of the analysis, the summed chi-square statistic of the circuits
    that both runs hold, its degrees of freedom and their ratio."""
    result = []
    for check in checks:
        circuits = check.get("circuits") or []
        statistic = sum(c["statistic"] for c in circuits)
        df = sum(c["df"] for c in circuits)
        result.append(
            {
                "device": check["device"],
                "week": check["week"],
                "circuits": len(circuits),
                "not_the_same": check.get("not_the_same"),
                "statistic": statistic,
                "df": df,
                "ratio": statistic / df if df else None,
            }
        )
    return result


def markdown(summary: Mapping[str, Any]) -> str:
    stated = summary["stated_calibration"]
    periods = summary["periods"]
    lines = [
        f"{summary['runs']} runs on {', '.join(summary['devices']) or 'no device'}:"
        f" {periods['counted']} runs that count, {periods['pilot']} pilot runs,"
        f" {periods['late']} late runs, {periods['secondary']} runs of other vendors,"
        f" {periods['trial']} trial runs.",
        f"Of the runs that count, the calibration as stated is ruled out in"
        f" {stated['runs_ruled_out']} runs and has a readout probability of 0 on the chain in"
        f" {stated['runs_with_a_zero_on_the_chain']} runs.",
    ]
    if summary["duplicate_rows_removed"]:
        lines.append(f"Duplicate rows removed: {summary['duplicate_rows_removed']}.")
    lines += ["", "## Primary comparisons", ""]
    if summary["primary_note"]:
        lines += [summary["primary_note"], ""]
    lines += _primary_table(summary["primary"])
    for c in summary["primary"]:
        lines += _primary_details(c)
    lines += ["", "## Secondary comparisons", "", *_estimate_head("Comparison")]
    for c in summary["secondary"]:
        lines.append(_estimate_line(f"{c['a']} - {c['b']} on {c['group']}", c["estimate"]))
    lines += _staleness_md(summary)
    lines += _age_md(summary)
    lines += _misfit_md(summary["misfit"])
    lines += _probes_md(summary["probes"])
    lines += _wide_md(summary["wide_overlap"])
    lines += _fits_md(summary["fits"])
    lines += _clifford_md(summary["clifford"])
    lines += _twirl_md(summary["twirl_check"])
    lines += _shots_md(summary["shot_checks"])
    titles = {
        "pilot": "Pilot runs, exploratory",
        "late": "Late runs, exploratory",
        "secondary": "Runs of other vendors, secondary",
        "trial": "Trial runs, exploratory",
    }
    for name, comparisons in summary["exploratory"].items():
        if not periods[name]:
            continue
        lines += ["", f"## {titles[name]}", "", *_verdict_table(comparisons)]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("analysis", type=Path, help="analysis.json of study.analyze")
    parser.add_argument("--out", type=Path, default=Path("results"), help="the output folder")
    args = parser.parse_args(argv)
    if not args.analysis.is_file():
        print(f"error: {args.analysis} is not a file", file=sys.stderr)
        return 2
    summary = summarize(json.loads(args.analysis.read_text(encoding="utf-8")))
    args.out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary, indent=1, allow_nan=False) + "\n"
    (args.out / "summary.json").write_text(text, encoding="utf-8")
    (args.out / "summary.md").write_text(markdown(summary), encoding="utf-8")
    print(f"wrote the summary of {summary['runs']} runs to {args.out}")
    return 0


def _primary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The five primary comparisons over the runs that count, with the Holm correction."""
    result = []
    unmarked = [r for r in rows if r.get("marked") is not True]
    for a, b, group in PRIMARY:

        def value(row: Mapping[str, Any], a: str = a, b: str = b, group: str = group) -> Any:
            return delta(row, a, b, group)

        def quadratic(row: Mapping[str, Any], a: str = a, b: str = b, group: str = group) -> Any:
            return delta(row, a, b, group, "quadratic")

        main_estimate = estimate(*device_weeks(rows, value))
        errors = [
            _test(r, a, b, group)["log"]["standard_error"]
            for r in rows
            if value(r) is not None and _test(r, a, b, group)["log"]["standard_error"] is not None
        ]
        strata = {kind: estimate(*device_weeks(rows, value, kind)) for kind in CHAIN_KINDS}
        for name, zero in (("zero_on_chain", True), ("no_zero_on_chain", False)):
            chosen = [r for r in rows if r["stated"].get("zero_on_chain") is zero]
            strata[name] = estimate(*device_weeks(chosen, value, "mean"))
        result.append(
            {
                "a": a,
                "b": b,
                "group": group,
                "estimate": main_estimate.to_dict(),
                "median_run_standard_error": statistics.median(errors) if errors else None,
                "quadratic": estimate(*device_weeks(rows, quadratic)).to_dict(),
                "without_marked_runs": estimate(*device_weeks(unmarked, value)).to_dict(),
                "strata": {name: found.to_dict() for name, found in strata.items()},
                "best_minus_median": estimate(*device_weeks(rows, value, "contrast")).to_dict(),
                "verdicts": verdicts(rows, a, b, group),
                "_p": _p(main_estimate),
            }
        )
    adjusted = holm([c.pop("_p") for c in result])
    for found, p_holm in zip(result, adjusted, strict=True):
        tested = (
            found["estimate"]["test"] is not None and found["estimate"]["test"]["t"] is not None
        )
        found["p_holm"] = p_holm if tested else None
        found["confirmed"] = tested and p_holm < LEVEL
        value = found["estimate"]["estimate"]
        found["small"] = bool(found["confirmed"] and abs(value) < SMALL)
    return result


def _secondary(rows: Sequence[dict[str, Any]], a: str, b: str, group: str) -> dict[str, Any]:
    def value(row: Mapping[str, Any]) -> float | None:
        return delta(row, a, b, group)

    found = estimate(*device_weeks(rows, value))
    return {
        "a": a,
        "b": b,
        "group": group,
        "part_of_5": (a, b, group) in PARTS_OF_5,
        "estimate": found.to_dict(),
    }


def _p(found: Estimate) -> float:
    """The p-value of the test, or 1 when there is no test. The family always has five tests."""
    if found.corrected is None or found.corrected.p_value is None:
        return 1.0
    return found.corrected.p_value


def _brief(found: Estimate) -> dict[str, Any]:
    test = found.corrected
    return {
        "estimate": found.estimate,
        "interval": None if test is None or test.low is None else [test.low, test.high],
        "device_weeks": found.device_weeks,
    }


def _week_value(
    runs: Sequence[Mapping[str, Any]],
    value: Callable[[Mapping[str, Any]], float | None],
    mode: Mode,
) -> tuple[float | None, str]:
    if mode == "mean":
        values = [value(r) for r in runs]
        if any(v is None for v in values):
            return None, "no_value"
        return sum(values) / len(values), ""  # type: ignore[arg-type]
    kinds: dict[str, list[Mapping[str, Any]]] = {}
    for run in runs:
        kinds.setdefault(run.get("chain") or "", []).append(run)
    needed = CHAIN_KINDS if mode in ("pair", "contrast") else (mode,)
    if any(len(found) > 1 for found in kinds.values()) or "" in kinds:
        return None, "extra_run"
    if any(kind not in kinds for kind in needed):
        return None, "missing_run"
    found = {kind: value(kinds[kind][0]) for kind in needed}
    if any(v is None for v in found.values()):
        return None, "no_value"
    if mode == "pair":
        return (found["best"] + found["median"]) / 2, ""  # type: ignore[operator]
    if mode == "contrast":
        return found["best"] - found["median"], ""  # type: ignore[operator]
    return found[mode], ""


def _stated(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    return {
        "runs_ruled_out": sum(1 for r in rows if r["stated"]["ruled_out"].get("all4", 0)),
        "runs_with_a_zero_readout": sum(1 for r in rows if r["stated"]["zero_readouts"]),
        "runs_with_a_zero_on_the_chain": sum(
            1 for r in rows if r["stated"].get("zero_on_chain") is True
        ),
    }


def _staleness_ages(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The note of the staleness curve: for the calibration of each run and for each stale
    model, the age of the calibration time and the median age of each kind of value."""
    points = []
    for row in rows:
        for model in ("calibration", *STALE):
            ages = row.get("ages", {}).get(model)
            if ages is None:
                continue
            points.append(
                {
                    "device": row["device"],
                    "chain": row["chain"],
                    "run_at": row["run_at"],
                    "period": period(row),
                    "model": model,
                    "calibration_age_h": ages["calibration_h"],
                    "median_age_h": {
                        kind: None if found is None else found["median_h"]
                        for kind, found in ages["values"].items()
                        if kind in AGE_KINDS
                    },
                }
            )
    return points


def _clifford_test(row: Mapping[str, Any], a: str, b: str) -> dict[str, Any] | None:
    return next(
        (
            t
            for t in (row.get("clifford") or {}).get("paired", ())
            if (t["a"], t["b"], t["circuits"]) == (a, b, "all")
        ),
        None,
    )


def _test(row: Mapping[str, Any], a: str, b: str, group: str) -> dict[str, Any] | None:
    for test in row["paired"]:
        if (test["a"], test["b"], test["group"]) == (a, b, group):
            return test
    return None


def _ranks(values: Sequence[float]) -> list[float]:
    """The rank of each value from 1, in the order of ``values``. Tied values share their mean
    rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for i in order[start : end + 1]:
            ranks[i] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def _depth(name: str) -> int:
    """The number m of a circuit name with ``_d{m}``, such as ``axis_d64``."""
    tail = name.rsplit("_d", 1)[-1]
    digits = "".join(ch for ch in tail.split("_")[0] if ch.isdigit())
    return int(digits) if digits else 0


def _largest(values: Any) -> float | None:
    found = [v for v in values if v is not None]
    return max(found) if found else None


def _time(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(UTC)


def _finite(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


# The Markdown parts.


def _estimate_head(first: str) -> list[str]:
    return [
        f"| {first} | Estimate | 95% interval | p | Device-weeks | Not complete |",
        "| --- | --- | --- | --- | --- | --- |",
    ]


def _estimate_line(label: str, found: Mapping[str, Any]) -> str:
    test = found["test"] or {}
    interval = test.get("interval")
    span = "-" if interval is None else f"{interval[0]:+.5f} to {interval[1]:+.5f}"
    incomplete = sum(found["incomplete"].values())
    return (
        f"| {label} | {_signed(found['estimate'])} | {span} |"
        f" {_number(test.get('p_value'), '.3g')} | {found['device_weeks']} | {incomplete} |"
    )


def _primary_table(primary: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "Delta is the log score of the first model minus the log score of the second model, in"
        " nats for each shot. The test is the t-test over device-weeks.",
        "",
        "| First model | Second model | Circuits | Estimate | Standard error | t | df | p |"
        " p (Holm) | Confirmed | Small |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in primary:
        found = c["estimate"]
        test = found["test"] or {}
        lines.append(
            f"| {c['a']} | {c['b']} | {c['group']} | {_signed(found['estimate'])} |"
            f" {_number(test.get('standard_error'), '.5f')} | {_signed(test.get('t'), '.2f')} |"
            f" {found['df']} | {_number(test.get('p_value'), '.3g')} |"
            f" {_number(c['p_holm'], '.3g')} | {'yes' if c['confirmed'] else 'no'} |"
            f" {'yes' if c['small'] else 'no'} |"
        )
    return lines


def _primary_details(c: Mapping[str, Any]) -> list[str]:
    found = c["estimate"]
    lines = ["", f"### {c['a']} against {c['b']} on {c['group']}, descriptions", ""]
    plain = found["test_with_f_1"] or {}
    sign = found["sign_test"]
    lines += [
        f"rho {_number(found['rho'], '.2f')}. With f = 1: standard error"
        f" {_number(plain.get('standard_error'), '.5f')}, p {_number(plain.get('p_value'), '.3g')}."
        " Median standard error of delta of a run:"
        f" {_number(c['median_run_standard_error'], '.5f')}."
        f" Sign test of Y: {sign['above_0']} above 0, {sign['below_0']} below 0,"
        f" p {sign['p_value']:.3g}.",
    ]
    if found["incomplete"]:
        reasons = ", ".join(f"{k} {v}" for k, v in found["incomplete"].items())
        lines.append(f"Device-weeks not complete: {reasons}.")
    if found["devices_with_too_few_weeks"]:
        few = ", ".join(f"{k} {v}" for k, v in found["devices_with_too_few_weeks"].items())
        lines.append(f"Devices with fewer than 4 complete device-weeks: {few}.")
    lines += [
        "",
        "| Device | n | m | s | f | Interval for a new week |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for d in found["devices"]:
        low, high = d["new_week"]
        lines.append(
            f"| {d['device']} | {d['n']} | {_signed(d['m'])} | {d['s']:.5f} | {d['f']:.2f} |"
            f" {low:+.5f} to {high:+.5f} |"
        )
    lines += ["", *_estimate_head("Form")]
    forms = [
        ("quadratic score", c["quadratic"]),
        ("without the marked runs", c["without_marked_runs"]),
        *((f"stratum {name}", value) for name, value in c["strata"].items()),
        ("best chain minus median chain, secondary", c["best_minus_median"]),
    ]
    lines += [_estimate_line(label, value) for label, value in forms]
    total = c["verdicts"]["total"]
    lines += [
        "",
        f"Verdict counts of the runs (log score): first better {total['a']}, second better"
        f" {total['b']}, neither {total['neither']}.",
    ]
    return lines


def _verdict_table(comparisons: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| First model | Second model | Circuits | First better | Second better | Neither |"
        " p (sign test) | p (Holm) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in comparisons:
        total = c["total"]
        lines.append(
            f"| {c['a']} | {c['b']} | {c['group']} | {total['a']} | {total['b']} |"
            f" {total['neither']} | {c['p_value']:.3g} | {c['p_holm']:.3g} |"
        )
    return lines


def _staleness_md(summary: Mapping[str, Any]) -> list[str]:
    lines = [
        "",
        "## Staleness curve",
        "",
        "Delta of stale(d) against calibration on all4, for each nominal age d.",
        "",
        "| Runs | d (days) | Estimate | 95% interval | Device-weeks |",
        "| --- | --- | --- | --- | --- |",
    ]
    for kind, found in summary["staleness"].items():
        for p in found["points"]:
            lines.append(
                f"| {kind} | {p['days']} | {_signed(p['estimate'])} | {_interval(p['interval'])} |"
                f" {p['device_weeks']} |"
            )
        slope = found["slope"]
        lines.append(
            f"| {kind} | slope against log10(d) | {_signed(slope['estimate'])} |"
            f" {_interval(slope['interval'])} | {slope['device_weeks']} |"
        )
    lines += [
        "",
        "Note: the median age of the values of each model at the run time.",
        "",
        "| Device | Chain | Run | Model | Age of the calibration (h) | Readout (h) |"
        " 1-qubit (h) | 2-qubit (h) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for p in summary["staleness_ages"]:
        ages = " | ".join(_hours(p["median_age_h"].get(kind)) for kind in AGE_KINDS)
        lines.append(
            f"| {p['device']} | {p['chain'] or '-'} | {p['run_at']} | {p['model']} |"
            f" {_hours(p['calibration_age_h'])} | {ages} |"
        )
    return lines


def _age_md(summary: Mapping[str, Any]) -> list[str]:
    lines = []
    for title, key in (
        ("Age of a value against its fitted factor", "age_against_error"),
        (
            "Age of a value against its fitted factor, without the marked runs",
            "age_against_error_without_marked_runs",
        ),
    ):
        if key.endswith("marked_runs") and not summary["runs_marked"]:
            continue
        lines += [
            "",
            f"## {title}",
            "",
            "| Device | Loci | Points | Spearman | At a limit | Spearman without them |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for c in summary[key]:
            lines.append(
                f"| {c['device']} | {c['loci']} | {c['points']} | {_signed(c['spearman'], '.3f')} |"
                f" {c['at_limit']} | {_signed(c['spearman_without_limits'], '.3f')} |"
            )
    return lines


def _misfit_md(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "",
        "## Misfit on the test_far circuits of the chain",
        "",
        "The excess minus the mean excess of counts drawn from the model, per shot. For a fitted"
        " model this value is a little too high.",
        "",
        "| Device | Model | Runs | Median misfit | Median part below 1 expected shot |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['device']} | {r['model']} | {r['runs']} | {r['median_misfit']:+.5f} |"
            f" {r['median_rare']:+.5f} |"
        )
    return lines


def _probes_md(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return []
    models = " | ".join(PROBE_MODELS)
    lines = [
        "",
        "## Probe circuits",
        "",
        "Fraction of wrong bits of each qubit: measured, then the forecast of each model.",
        "",
        f"| Device | Chain | Run | Circuit | Qubit | Measured | {models} | Baseline |",
        "| --- | --- | --- | --- | --- | --- |" + " --- |" * (len(PROBE_MODELS) + 1),
    ]
    for r in rows:
        baseline = {b["qubit"]: b["measured"] for b in r["baseline"]}
        for c in r["circuits"]:
            forecast = " | ".join(_number(c["forecast"].get(m), ".4f") for m in PROBE_MODELS)
            lines.append(
                f"| {r['device']} | {r['chain'] or '-'} | {r['run_at']} | {c['name']} | ideal"
                f" {c['ideal']} | {c['measured']:.4f} | {forecast} | - |"
            )
            for q in c["qubits"]:
                values = " | ".join(_number(q["forecast"].get(m), ".4f") for m in PROBE_MODELS)
                lines.append(
                    f"| {r['device']} | {r['chain'] or '-'} | {r['run_at']} | {c['name']} |"
                    f" {q['qubit']} | {q['measured']:.4f} | {values} |"
                    f" {_number(baseline.get(q['qubit']), '.4f')} |"
                )
    lines += [
        "",
        "Growth ratio of each qubit: (wrong fraction minus baseline) of the probe with the largest"
        " m, divided by the same value of the probe with the smallest m.",
        "",
        "| Device | Chain | Run | Qubit | Growth ratio |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        for g in r["growth"]:
            lines.append(
                f"| {r['device']} | {r['chain'] or '-'} | {r['run_at']} | {g['qubit']} |"
                f" {_number(g['ratio'], '.2f')} |"
            )
    return lines


def _wide_md(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return []
    names = [f"{a} - {b}" for a, b, _ in WIDE_COMPARISONS]
    lines = [
        "",
        "## Wide comparisons of each run",
        "",
        "Overlap is the number of qubits of the 8-qubit chain that are in the chain of the run.",
        "",
        "| Device | Chain | Run | Overlap | " + " | ".join(names) + " |",
        "| --- | --- | --- | --- |" + " --- |" * len(names),
    ]
    for r in rows:
        values = " | ".join(_signed(r["delta"][n]) for n in names)
        overlap = "-" if r["overlap"] is None else str(r["overlap"])
        lines.append(
            f"| {r['device']} | {r['chain'] or '-'} | {r['run_at']} | {overlap} | {values} |"
        )
    return lines


def _fits_md(fits: Mapping[str, Any]) -> list[str]:
    counts = ", ".join(f"{k}: {v} runs" for k, v in fits["averaged_7_calibrations"].items())
    return [
        "",
        "## Fits",
        "",
        f"Runs with a factor that the fit does not identify:"
        f" {fits['runs_with_a_factor_not_identified']}.",
        f"Runs with a factor at a limit of the search: {fits['runs_with_a_factor_at_a_limit']}.",
        f"Different calibrations in the median of averaged(7): {counts or 'no run'}.",
    ]


def _clifford_md(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "",
        "## Clifford circuits on 20 to 80 qubits, secondary",
        "",
        "One value for each device-week: the mean over its two runs of the difference of the"
        " qubit score per shot.",
        "",
        "| First model | Second model | Device | Week | Value | Verdicts of the runs |"
        " Largest sampling ratio |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in rows:
        for w in c["device_weeks"]:
            lines.append(
                f"| {c['a']} | {c['b']} | {w['device']} | {w['week']} | {w['value']:+.5f} |"
                f" {', '.join(w['verdicts'])} | {_number(w['largest_sampling_ratio'], '.3f')} |"
            )
    lines += ["", *_estimate_head("Comparison")]
    for c in rows:
        lines.append(_estimate_line(f"{c['a']} - {c['b']}", c["estimate"]))
    return lines


def _twirl_md(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "",
        "## Twirl check on the 8-qubit circuits",
        "",
        "| Device | Chain | Run | Model | Circuit | Exact | Stabilizer | Difference |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for t in rows:
        lines.append(
            f"| {t['device']} | {t['chain'] or '-'} | {t['run_at']} |"
            f" {t.get('model', 'calibration')} | {t['name']} | {_signed(t['exact_log'])} |"
            f" {_signed(t['stabilizer_log'])} | {_signed(t['difference'])} |"
        )
    return lines


def _shots_md(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "",
        "## Check of independent shots",
        "",
        "| Device | Week | Circuits | Not the same | Chi-square | df | Ratio |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['device']} | {r['week']} | {r['circuits']} | {r['not_the_same']} |"
            f" {r['statistic']:.1f} | {r['df']} | {_number(r['ratio'], '.2f')} |"
        )
    return lines


def _interval(value: Sequence[float] | None) -> str:
    return "-" if value is None else f"{value[0]:+.5f} to {value[1]:+.5f}"


def _hours(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def _signed(value: float | None, form: str = ".5f") -> str:
    return "-" if value is None else format(value, "+" + form)


def _number(value: float | None, form: str) -> str:
    return "-" if value is None else format(value, form)


if __name__ == "__main__":
    raise SystemExit(main())
