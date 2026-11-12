"""Summarize the analysis over all runs, as the plan states.

    python -m study.summarize results/analysis.json --out results

The script reads ``analysis.json`` of ``study.analyze``. For each primary comparison of
``PLAN.md``, it counts the runs in which the first model is better, the second model is better,
or neither. It tests the counts with a two-sided sign test and applies the Holm correction to the
five tests. It also lists the points of the staleness curve. The script writes ``summary.json``
and ``summary.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from math import comb
from pathlib import Path
from typing import Any

from study.analyze import PRIMARY, STALE

FORMAT = "1"
LEVEL = 0.05
AGE_KINDS = ("readout", "one_qubit", "two_qubit")


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


def summarize(analysis: dict[str, Any]) -> dict[str, Any]:
    rows = analysis["rows"]
    comparisons = _family(rows)
    unmarked = _family([r for r in rows if r.get("marked") is not True])
    stated = [r["stated"]["ruled_out"].get("all", 0) for r in rows]
    return {
        "nv_study_summary": FORMAT,
        "runs": len(rows),
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
    }


def markdown(summary: dict[str, Any]) -> str:
    stated = summary["stated_calibration"]
    lines = [
        f"{summary['runs']} runs on {', '.join(summary['devices']) or 'no device'}.",
        f"The calibration as stated is ruled out in {stated['runs_ruled_out']} runs.",
    ]
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


def _hours(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def _signed(value: float | None, form: str = ".5f") -> str:
    return "-" if value is None else format(value, "+" + form)


if __name__ == "__main__":
    raise SystemExit(main())
