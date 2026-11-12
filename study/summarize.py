"""Summarize the analysis over all runs, as the plan states.

    python -m study.summarize results/analysis.json --out results

The script reads ``analysis.json`` of ``study.analyze``. For each primary comparison of
``PLAN.md``, it counts the runs in which the first model is better, the second model is better,
or neither. It tests the counts with a two-sided sign test and applies the Holm correction to the
five tests. It also lists the points of the staleness curve, and the rank correlation of the
age of a value with the fitted factor of the ``locus`` model. For the Clifford circuits on 20
to 80 qubits, it counts the verdicts of each comparison of ``CLIFFORD_COMPARISONS`` on the
circuits of a run together. These counts are descriptive and have no sign test. It also lists
the twirl check of each run. The script writes ``summary.json`` and ``summary.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from math import comb, log, sqrt
from pathlib import Path
from typing import Any

from study.analyze import CLIFFORD_COMPARISONS, PRIMARY, STALE

FORMAT = "1"
LEVEL = 0.05
AGE_KINDS = ("readout", "one_qubit", "two_qubit")
MIN_POINTS = 5


def sign_test(first: int, second: int) -> float:
    """The two-sided p-value of ``first`` wins against ``second`` wins under equal chances."""
    n = first + second
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(first, second) + 1)) / 2**n
    return min(1.0, 2 * tail)


def holm(p_values: Sequence[float]) -> list[float]:
    """The Holm-adjusted p-values, in the order of ``p_values``."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    largest = 0.0
    for rank, i in enumerate(order):
        largest = max(largest, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = largest
    return adjusted


def unique(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """The first row of each ``counts_sha256``, and the number of rows removed.

    One run must count one time in a sign test, also when the analysis comes from another tool.
    """
    found: dict[str, dict[str, Any]] = {}
    for row in rows:
        found.setdefault(row["counts_sha256"], row)
    return list(found.values()), len(rows) - len(found)


def summarize(analysis: dict[str, Any]) -> dict[str, Any]:
    rows, duplicates = unique(analysis["rows"])
    comparisons = _family(rows)
    unmarked_rows = [r for r in rows if r.get("marked") is not True]
    unmarked = _family(unmarked_rows)
    stated = [r["stated"]["ruled_out"].get("all", 0) for r in rows]
    return {
        "nv_study_summary": FORMAT,
        "runs": len(rows),
        "duplicate_rows_removed": duplicates,
        "devices": sorted({r["device"] for r in rows}),
        "level": LEVEL,
        "stated_calibration": {
            "runs_ruled_out": sum(1 for shots in stated if shots),
            "runs_with_a_zero_readout": sum(1 for r in rows if r["stated"]["zero_readouts"]),
        },
        "primary": comparisons,
        "runs_marked": sum(1 for r in rows if r.get("marked") is True),
        "primary_without_marked_runs": unmarked,
        "staleness": _staleness(rows),
        "age_against_error": age_against_error(rows),
        "age_against_error_without_marked_runs": age_against_error(unmarked_rows),
        "clifford": clifford(rows),
        "twirl_check": twirl_check(rows),
    }


def clifford(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each Clifford comparison, the runs in which the first model is better, the second
    model is better, or neither, by the qubit score on the Clifford circuits of a run together.
    """
    result = []
    for a, b in CLIFFORD_COMPARISONS:
        by_device: dict[str, dict[str, int]] = {}
        for row in rows:
            test = next(
                (
                    t
                    for t in row.get("clifford", {}).get("paired", ())
                    if (t["a"], t["b"], t["circuits"]) == (a, b, "all")
                ),
                None,
            )
            if test is None:
                continue
            counts = by_device.setdefault(row["device"], {"a": 0, "b": 0, "neither": 0})
            counts[test["qubit"]["log"]["better"]] += 1
        total = {
            key: sum(counts[key] for counts in by_device.values()) for key in ("a", "b", "neither")
        }
        result.append(
            {"a": a, "b": b, "by_device": dict(sorted(by_device.items())), "total": total}
        )
    return result


def twirl_check(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The twirl check of each 8-qubit circuit of each run: the log score of ``calibration``
    from the exact simulator and from the stabilizer forecast, per shot."""
    return [
        {"device": row["device"], "chain": row["chain"], "run_at": row["run_at"], **check}
        for row in rows
        for check in row.get("twirl_check", ())
    ]


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


def age_against_error(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each device, and for the qubits and the pairs separately, the Spearman rank
    correlation between the age of a value at the run time and the absolute value of ln f.

    f is the factor of that qubit or pair in the ``locus`` model. The age of a qubit is the age
    of its 1-qubit gate errors, and the age of a pair is the age of its 2-qubit gate errors.
    """
    points: dict[tuple[str, str], list[tuple[float, float]]] = {}
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
            points.setdefault((row["device"], kind), []).append((age, abs(log(factor))))
    result = []
    for (device, kind), pairs in sorted(points.items()):
        ages, errors = zip(*pairs, strict=True)
        result.append(
            {
                "device": device,
                "loci": kind,
                "points": len(pairs),
                "spearman": spearman(ages, errors),
            }
        )
    return result


def markdown(summary: dict[str, Any]) -> str:
    stated = summary["stated_calibration"]
    lines = [
        f"{summary['runs']} runs on {', '.join(summary['devices']) or 'no device'}.",
        f"The calibration as stated is ruled out in {stated['runs_ruled_out']} runs.",
    ]
    if summary["duplicate_rows_removed"]:
        lines.append(f"Duplicate rows removed: {summary['duplicate_rows_removed']}.")
    tables = [("All runs", summary["primary"])]
    if summary["runs_marked"]:
        tables.append(
            (
                f"Without the {summary['runs_marked']} marked runs",
                summary["primary_without_marked_runs"],
            )
        )
    for title, comparisons in tables:
        lines += [
            "",
            f"{title}:",
            "",
            "| First model | Second model | Circuits | First better | Second better | Neither |"
            " p (sign test) | p (Holm) | Confirmed |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for c in comparisons:
            total = c["total"]
            lines.append(
                f"| {c['a']} | {c['b']} | {c['group']} | {total['a']} | {total['b']} |"
                f" {total['neither']} | {c['p_value']:.3g} | {c['p_holm']:.3g} |"
                f" {'yes' if c['confirmed'] else 'no'} |"
            )
    lines += [
        "",
        "| Device | Chain | Run | Model | Age of the calibration (h) | Readout (h) |"
        " 1-qubit (h) | 2-qubit (h) | Log score minus calibration | z |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for p in summary["staleness"]:
        ages = " | ".join(_hours(p["median_age_h"].get(kind)) for kind in AGE_KINDS)
        lines.append(
            f"| {p['device']} | {p['chain'] or '-'} | {p['run_at']} | {p['model']} |"
            f" {_hours(p['calibration_age_h'])} | {ages} | {_signed(p['difference'])} |"
            f" {_signed(p['z'], '.2f')} |"
        )
    for title, key in (
        ("Age of a value against its error, all runs", "age_against_error"),
        ("Without the marked runs", "age_against_error_without_marked_runs"),
    ):
        if key == "age_against_error_without_marked_runs" and not summary["runs_marked"]:
            continue
        lines += [
            "",
            f"{title}:",
            "",
            "| Device | Loci | Points | Spearman |",
            "| --- | --- | --- | --- |",
        ]
        for c in summary[key]:
            lines.append(
                f"| {c['device']} | {c['loci']} | {c['points']} | {_signed(c['spearman'], '.3f')} |"
            )
    lines += [
        "",
        "Clifford circuits on 20 to 80 qubits, qubit score, descriptive:",
        "",
        "| First model | Second model | Device | First better | Second better | Neither |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for c in summary["clifford"]:
        for device, counts in [*c["by_device"].items(), ("all devices", c["total"])]:
            lines.append(
                f"| {c['a']} | {c['b']} | {device} | {counts['a']} | {counts['b']} |"
                f" {counts['neither']} |"
            )
    lines += [
        "",
        "Twirl check of calibration on the 8-qubit circuits (log score per shot):",
        "",
        "| Device | Chain | Run | Circuit | Exact | Stabilizer | Difference |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for t in summary["twirl_check"]:
        lines.append(
            f"| {t['device']} | {t['chain'] or '-'} | {t['run_at']} | {t['name']} |"
            f" {_signed(t['exact_log'])} | {_signed(t['stabilizer_log'])} |"
            f" {_signed(t['difference'])} |"
        )
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


def _family(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The five primary comparisons over ``rows``, with the Holm correction of their tests."""
    comparisons = [_comparison(rows, a, b, group) for a, b, group in PRIMARY]
    adjusted = holm([c["p_value"] for c in comparisons])
    for found, p_holm in zip(comparisons, adjusted, strict=True):
        found["p_holm"] = p_holm
        found["confirmed"] = p_holm < LEVEL
    return comparisons


def _comparison(rows: Sequence[dict[str, Any]], a: str, b: str, group: str) -> dict[str, Any]:
    by_device: dict[str, dict[str, int]] = {}
    for row in rows:
        test = _test(row, a, b, group)
        if test is None:
            continue
        counts = by_device.setdefault(row["device"], {"a": 0, "b": 0, "neither": 0})
        # The plan takes the verdict of the log score. The quadratic score decides nothing here.
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


def _test(row: dict[str, Any], a: str, b: str, group: str) -> dict[str, Any] | None:
    for test in row["paired"]:
        if (test["a"], test["b"], test["group"]) == (a, b, group):
            return test
    return None


def _staleness(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One point for the calibration of each run and for each stale model: the ages at the run
    time, and the log score of the model minus the log score of the calibration, per shot."""
    points = []
    for row in rows:
        for model in ("calibration", *STALE):
            ages = row.get("ages", {}).get(model)
            test = None if model == "calibration" else _test(row, "calibration", model, "all")
            if ages is None or (model != "calibration" and test is None):
                continue
            log = {"difference": 0.0, "standard_error": 0.0, "z": None}
            if test is not None:
                log = test["log"]
            difference = log["difference"]
            points.append(
                {
                    "device": row["device"],
                    "chain": row["chain"],
                    "run_at": row["run_at"],
                    "model": model,
                    "calibration_age_h": ages["calibration_h"],
                    "median_age_h": {
                        kind: None if found is None else found["median_h"]
                        for kind, found in ages["values"].items()
                    },
                    "outcome": "tested" if test is None else test["log"]["outcome"],
                    "difference": None if difference is None else -difference,
                    "standard_error": log["standard_error"],
                    "z": None if log["z"] is None else -log["z"],
                }
            )
    return points


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


def _hours(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def _signed(value: float | None, form: str = ".5f") -> str:
    return "-" if value is None else format(value, "+" + form)


if __name__ == "__main__":
    raise SystemExit(main())
