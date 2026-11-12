from __future__ import annotations

import itertools
import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from study import estimate, summarize
from study.analyze import PARTS_OF_5, PRIMARY, SECONDARY, STALE, STALE_DAYS

_HASHES = itertools.count(1)
FIRST = datetime(2026, 11, 17, 3, 0)
DEVICES = ("ibm_fez", "ibm_kingston", "ibm_marrakesh")


def _test(a: str, b: str, group: str, difference: float | None, better: str = "neither") -> dict:
    log = {
        "outcome": "tested" if difference is not None else "ruled_out",
        "difference": difference,
        "standard_error": None if difference is None else 0.0004,
        "z": None if difference is None else difference / 0.0004,
        "better": better,
    }
    quadratic = log | {"difference": None if difference is None else difference / 4}
    return {
        "a": a,
        "b": b,
        "group": group,
        "primary": (a, b, group) in PRIMARY,
        "log": log,
        "quadratic": quadratic,
    }


def _row(
    device: str,
    when: datetime | str = "2026-11-12T03:00:00Z",
    chain: str | None = "best",
    *,
    deltas: dict[tuple[str, str, str], float | None] | None = None,
    better: dict[tuple[str, str, str], str] | None = None,
    zero: bool = False,
    **more: Any,
) -> dict[str, Any]:
    deltas = deltas if deltas is not None else {}
    better = better or {}
    keys = dict.fromkeys([*PRIMARY, *deltas])
    paired = [
        _test(a, b, g, deltas.get((a, b, g), 0.01), better.get((a, b, g), "neither"))
        for a, b, g in keys
    ]
    stamp = when if isinstance(when, str) else when.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "device": device,
        "vendor": "ibm",
        "counts_sha256": f"{next(_HASHES):064x}",
        "chain": chain,
        "run_at": stamp,
        "stated": {"zero_readouts": 2, "zero_on_chain": zero, "ruled_out": {"all4": 7}},
        "paired": paired,
        "ages": {},
        **more,
    }


def test_the_sign_test_is_the_exact_two_sided_binomial_test() -> None:
    assert summarize.sign_test(0, 0) == 1.0
    assert summarize.sign_test(5, 5) == 1.0
    assert summarize.sign_test(6, 0) == pytest.approx(2 / 64)
    assert summarize.sign_test(0, 6) == pytest.approx(2 / 64)
    assert summarize.sign_test(9, 1) == pytest.approx(2 * 11 / 1024)


def test_holm_multiplies_the_smallest_p_value_by_the_number_of_tests_and_keeps_the_order() -> None:
    assert summarize.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert summarize.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert summarize.holm([]) == []


def test_the_window_makes_pilot_runs_runs_that_count_and_late_runs() -> None:
    assert summarize.period(_row("ibm_fez", "2026-11-15T23:59:59Z")) == "pilot"
    assert summarize.period(_row("ibm_fez", "2026-11-16T00:00:00Z")) == "counted"
    assert summarize.period(_row("ibm_fez", "2027-01-10T23:59:59Z")) == "counted"
    assert summarize.period(_row("ibm_fez", "2027-01-11T00:00:00Z")) == "late"
    other = _row("iqm_garnet", "2026-11-20T00:00:00Z") | {"vendor": "iqm"}
    assert summarize.period(other) == "secondary"


def test_the_summary_of_zero_runs_is_complete(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis.json"
    analysis.write_text(json.dumps({"rows": []}))

    assert summarize.main([str(analysis), "--out", str(tmp_path)]) == 0

    summary = json.loads((tmp_path / "summary.json").read_text())
    text = (tmp_path / "summary.md").read_text()
    assert summary["runs"] == 0 and summary["primary_note"] == summarize.NO_ESTIMATE
    assert all(c["estimate"]["estimate"] is None for c in summary["primary"])
    assert all(c["p_holm"] is None and c["confirmed"] is False for c in summary["primary"])
    assert summarize.NO_ESTIMATE in text
    assert "Pilot runs" not in text


def test_the_summary_of_pilot_runs_only_has_no_test_and_an_exploratory_table() -> None:
    first = PRIMARY[0]
    rows = [_row("ibm_fez", better={first: "a"}) for _ in range(6)]
    rows += [_row("ibm_kingston", better={first: "a"}) for _ in range(4)]
    rows.append(_row("ibm_kingston", better={first: "b"}))

    summary = summarize.summarize({"rows": rows})

    assert summary["periods"] == {"pilot": 11, "counted": 0, "late": 0, "secondary": 0}
    assert summary["primary_note"] == summarize.NO_ESTIMATE
    pilot = summary["exploratory"]["pilot"][0]
    assert pilot["by_device"]["ibm_fez"] == {"a": 6, "b": 0, "neither": 0}
    assert pilot["total"] == {"a": 10, "b": 1, "neither": 0}
    assert pilot["p_value"] == pytest.approx(2 * 12 / 2048)
    assert pilot["p_holm"] == pytest.approx(5 * 2 * 12 / 2048)
    text = summarize.markdown(summary)
    assert "## Pilot runs, exploratory" in text
    assert "| calibration | medians | all4 | 10 | 1 | 0 |" in text
    assert "## Late runs" not in text


def test_a_late_run_is_only_in_the_exploratory_table() -> None:
    rows = [_row("ibm_fez", "2027-01-12T03:00:00Z", better={PRIMARY[0]: "a"})]

    summary = summarize.summarize({"rows": rows})

    assert summary["periods"]["late"] == 1
    assert summary["exploratory"]["late"][0]["total"] == {"a": 1, "b": 0, "neither": 0}
    assert all(c["verdicts"]["total"]["a"] == 0 for c in summary["primary"])
    assert "## Late runs, exploratory" in summarize.markdown(summary)


def _study(seed: int = 3, weeks: int = 4) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """3 devices, ``weeks`` weeks, two runs in each device-week, with a delta for each primary
    comparison that has a device effect and a week effect."""
    rng = np.random.default_rng(seed)
    rows, truth = [], {}
    for d, device in enumerate(DEVICES):
        for w in range(weeks):
            when = FIRST + timedelta(days=7 * w + d)
            shared = rng.normal(0.002, 0.001)
            for c, chain in enumerate(("best", "median")):
                deltas = {
                    key: shared + 0.001 * i + 0.0005 * c + rng.normal(0, 0.0002)
                    for i, key in enumerate(PRIMARY)
                }
                deltas |= {key: rng.normal(0.001, 0.001) for key in PARTS_OF_5}
                zero = (w + c) % 2 == 0
                row = _row(device, when + timedelta(minutes=c), chain, deltas=deltas, zero=zero)
                rows.append(row)
                truth[row["counts_sha256"]] = deltas
    return rows, truth


def _y(rows: list[dict[str, Any]], key: tuple[str, str, str], kind: str = "pair") -> dict:
    """Y of each device-week, computed again from the rows."""
    weeks: dict[tuple[str, date], dict[str, float]] = {}
    for row in rows:
        day = datetime.fromisoformat(row["run_at"].replace("Z", "+00:00")).date()
        week = day - timedelta(days=day.weekday())
        delta = next(t for t in row["paired"] if (t["a"], t["b"], t["group"]) == key)
        weeks.setdefault((row["device"], week), {})[row["chain"]] = delta["log"]["difference"]
    values: dict[str, dict[date, float]] = {}
    for (device, week), by_chain in weeks.items():
        if kind == "pair":
            y = (by_chain["best"] + by_chain["median"]) / 2
        elif kind == "contrast":
            y = by_chain["best"] - by_chain["median"]
        else:
            y = by_chain[kind]
        values.setdefault(device, {})[week] = y
    return values


def test_the_primary_estimate_of_3_devices_and_4_weeks() -> None:
    rows, _ = _study()

    summary = summarize.summarize({"rows": rows})

    assert summary["periods"]["counted"] == 24 and summary["primary_note"] is None
    p_values = []
    for found, key in zip(summary["primary"], PRIMARY, strict=True):
        expected = estimate.estimate(_y(rows, key))
        assert found["estimate"]["estimate"] == pytest.approx(expected.estimate)
        assert found["estimate"]["df"] == 9 and found["estimate"]["device_weeks"] == 12
        assert found["estimate"]["test"]["p_value"] == pytest.approx(expected.corrected.p_value)
        assert found["median_run_standard_error"] == pytest.approx(0.0004)
        assert found["verdicts"]["total"] == {"a": 0, "b": 0, "neither": 24}
        quadratic = estimate.estimate(
            {d: {w: y / 4 for w, y in v.items()} for d, v in _y(rows, key).items()}
        )
        assert found["quadratic"]["estimate"] == pytest.approx(quadratic.estimate)
        for kind in ("best", "median", "contrast"):
            name = "best_minus_median" if kind == "contrast" else kind
            part = found[name] if kind == "contrast" else found["strata"][name]
            assert part["estimate"] == pytest.approx(
                estimate.estimate(_y(rows, key, kind)).estimate
            )
        assert found["best_minus_median"]["estimate"] == pytest.approx(-0.0005, abs=0.0003)
        p_values.append(expected.corrected.p_value)
    adjusted = estimate.holm(p_values)
    assert [c["p_holm"] for c in summary["primary"]] == pytest.approx(adjusted)
    for c in summary["primary"]:
        assert c["confirmed"] == (c["p_holm"] < 0.05)
        assert c["small"] == (c["confirmed"] and abs(c["estimate"]["estimate"]) < 0.001)
    text = summarize.markdown(summary)
    assert "| calibration | medians | all4 |" in text
    assert "Interval for a new week" in text
    assert "stratum zero_on_chain" in text


def test_the_stratum_of_a_stated_zero_takes_the_mean_of_the_runs_in_the_stratum() -> None:
    rows, _ = _study()
    key = PRIMARY[0]

    (found,) = [c for c in summarize.summarize({"rows": rows})["primary"] if c["a"] == key[0]][:1]

    for name, zero in (("zero_on_chain", True), ("no_zero_on_chain", False)):
        chosen = [r for r in rows if r["stated"]["zero_on_chain"] is zero]
        values: dict[str, dict[date, list[float]]] = {}
        for r in chosen:
            device, week = summarize.device_week(r)
            values.setdefault(device, {}).setdefault(week, []).append(summarize.delta(r, *key))
        expected = estimate.estimate(
            {d: {w: sum(v) / len(v) for w, v in by.items()} for d, by in values.items()}
        )
        assert found["strata"][name]["estimate"] == pytest.approx(expected.estimate)
        assert found["strata"][name]["device_weeks"] == 12


def test_the_parts_of_comparison_5_are_secondary_estimates() -> None:
    rows, _ = _study()

    summary = summarize.summarize({"rows": rows})

    parts = [c for c in summary["secondary"] if c["part_of_5"]]
    assert [(c["a"], c["b"], c["group"]) for c in parts] == list(PARTS_OF_5)
    for c in parts:
        expected = estimate.estimate(_y(rows, (c["a"], c["b"], c["group"])))
        assert c["estimate"]["estimate"] == pytest.approx(expected.estimate)
    assert [(c["a"], c["b"], c["group"]) for c in summary["secondary"]] == list(SECONDARY)


def test_a_device_week_that_is_not_complete_is_counted_with_its_reason() -> None:
    rows, _ = _study(weeks=5)
    rows = [
        r
        for r in rows
        if not (
            r["device"] == "ibm_fez"
            and r["run_at"].startswith("2026-11-17")
            and r["chain"] == "median"
        )
    ]
    ruled = rows[5]
    ruled["paired"][0]["log"]["difference"] = None
    rows.append(_row("ibm_kingston", rows[-1]["run_at"], "best"))
    key = PRIMARY[0]

    values, incomplete = summarize.device_weeks(rows, lambda r: summarize.delta(r, *key))

    assert incomplete == {"missing_run": 1, "no_value": 1, "extra_run": 1}
    assert sum(len(v) for v in values.values()) == 15 - 3
    found = summarize.summarize({"rows": rows})["primary"][0]["estimate"]
    assert found["incomplete"] == incomplete


def test_the_marked_runs_leave_the_estimate_without_marked_runs() -> None:
    rows, _ = _study()
    rows[0]["marked"] = True

    found = summarize.summarize({"rows": rows})["primary"][0]

    assert found["estimate"]["device_weeks"] == 12
    # ibm_fez keeps 3 complete device-weeks, so it leaves the estimate without the marked runs.
    assert found["without_marked_runs"]["device_weeks"] == 8
    assert found["without_marked_runs"]["devices_with_too_few_weeks"] == {"ibm_fez": 3}
    assert found["without_marked_runs"]["incomplete"] == {"missing_run": 1}


def test_the_staleness_curve_has_a_point_for_each_age_and_a_slope() -> None:
    rows = []
    for d, device in enumerate(DEVICES):
        for w in range(4):
            for chain in ("best", "median"):
                when = FIRST + timedelta(days=7 * w + d)
                # stale(d) is worse than the calibration by 0.001 log10(d) + a week effect.
                deltas = {
                    ("calibration", name, "all4"): 0.001 * math.log10(days) + 0.0001 * w
                    for days, name in zip(STALE_DAYS, STALE, strict=True)
                }
                rows.append(_row(device, when, chain, deltas=deltas))

    curve = summarize.summarize({"rows": rows})["staleness"]

    for kind in ("both", "best", "median"):
        points = curve[kind]["points"]
        assert [p["days"] for p in points] == list(STALE_DAYS)
        for p, days in zip(points, STALE_DAYS, strict=True):
            assert p["estimate"] == pytest.approx(-0.001 * math.log10(days) - 0.00015)
            assert p["device_weeks"] == 12
        assert curve[kind]["slope"]["estimate"] == pytest.approx(-0.001)


def test_the_age_of_a_value_is_compared_with_its_fitted_factor() -> None:
    rows = []
    for run in range(3):
        ages = {
            "qubits": {str(q): {"one_qubit": 10.0 * q + run, "readout": 1.0} for q in range(3)},
            "pairs": {"0-1": {"two_qubit": 5.0 + run}, "1-2": {"two_qubit": 50.0 + run}},
        }
        fits = {
            "qubits": [{"qubit": q, "factor": math.exp(0.1 * q + 0.01 * run)} for q in range(3)],
            "pairs": [{"qubits": [0, 1], "factor": 20.0}, {"qubits": [1, 2], "factor": None}],
        }
        more = {"ages": {"calibration": {"calibration_h": 1.0, "values": {}, "loci": ages}}}
        rows.append(_row("ibm_fez", FIRST, fits={"locus": fits}, **more))

    summary = summarize.summarize({"rows": rows})

    assert summary["age_against_error"] == [
        {
            "device": "ibm_fez",
            "loci": "pairs",
            "points": 3,
            "spearman": None,
            "at_limit": 3,
            "spearman_without_limits": None,
        },
        {
            "device": "ibm_fez",
            "loci": "qubits",
            "points": 9,
            "spearman": pytest.approx(1.0),
            "at_limit": 0,
            "spearman_without_limits": pytest.approx(1.0),
        },
    ]
    assert "| ibm_fez | qubits | 9 | +1.000 | 0 | +1.000 |" in summarize.markdown(summary)
    assert summary["fits"]["runs_with_a_factor_at_a_limit"] == 3


def test_a_factor_is_at_a_limit_of_the_search_near_0_05_or_20() -> None:
    assert summarize.at_limit(19.959) and summarize.at_limit(0.0501)
    assert not summarize.at_limit(19.0) and not summarize.at_limit(1.0)
    assert not summarize.at_limit(None)


def test_spearman_gives_tied_values_their_mean_rank() -> None:
    assert summarize.spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 100]) == pytest.approx(1.0)
    assert summarize.spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1.0)
    # Ranks (1, 2.5, 2.5, 4, 5) and (1, 2, 3, 4, 5): the Pearson correlation of the ranks.
    assert summarize.spearman([1, 2, 2, 3, 4], [1, 2, 3, 4, 5]) == pytest.approx(0.9746794)
    assert summarize.spearman([1, 2, 3, 4], [1, 2, 3, 4]) is None
    assert summarize.spearman([1, 1, 1, 1, 1], [1, 2, 3, 4, 5]) is None


def test_a_run_that_is_in_the_analysis_two_times_counts_one_time() -> None:
    first = PRIMARY[0]
    copy = _row("ibm_fez", better={first: "a"})
    rows = [copy] * 8 + [_row("ibm_fez", better={first: "b"})]

    summary = summarize.summarize({"rows": rows})

    assert summary["runs"] == 2
    assert summary["duplicate_rows_removed"] == 7
    assert summary["exploratory"]["pilot"][0]["total"] == {"a": 1, "b": 1, "neither": 0}
    assert "Duplicate rows removed: 7." in summarize.markdown(summary)


def _clifford_test(difference: float, ratio: float, better: str) -> dict[str, Any]:
    return {
        "a": "two_factor",
        "b": "calibration",
        "circuits": "all",
        "qubit": {"log": {"difference": difference, "better": better}, "sampling_ratio": ratio},
    }


def test_a_device_week_gives_one_clifford_value_the_mean_of_its_two_runs() -> None:
    rows = []
    for w in range(4):
        for chain, value in (("best", 0.02 + w * 0.001), ("median", 0.04)):
            block = {"paired": [_clifford_test(value, 0.1 * (w + 1), "a")]}
            rows.append(_row("ibm_fez", FIRST + timedelta(days=7 * w), chain, clifford=block))

    found = next(
        c
        for c in summarize.summarize({"rows": rows})["clifford"]
        if (c["a"], c["b"]) == ("two_factor", "calibration")
    )

    assert [w["value"] for w in found["device_weeks"]] == pytest.approx(
        [(0.02 + w * 0.001 + 0.04) / 2 for w in range(4)]
    )
    assert found["device_weeks"][0]["verdicts"] == ["a", "a"]
    assert found["device_weeks"][3]["largest_sampling_ratio"] == pytest.approx(0.4)
    assert found["estimate"]["device_weeks"] == 4


def test_the_check_of_independent_shots_gives_the_ratio_of_each_device_week() -> None:
    checks = [
        {
            "device": "ibm_fez",
            "week": "2026-11-16",
            "not_the_same": 1,
            "circuits": [
                {"name": "wide_d12_s0", "statistic": 30.0, "df": 20, "ratio": 1.5},
                {"name": "clifford_n20_d3", "statistic": 10.0, "df": 5, "ratio": 2.0},
            ],
        }
    ]

    (found,) = summarize.summarize({"rows": [], "shot_checks": checks})["shot_checks"]

    assert found == {
        "device": "ibm_fez",
        "week": "2026-11-16",
        "circuits": 2,
        "not_the_same": 1,
        "statistic": 40.0,
        "df": 25,
        "ratio": pytest.approx(1.6),
    }


def test_the_notes_of_the_fits_count_the_runs() -> None:
    fits = [
        {"two_factor": {"gates": {"not_identified": "flat"}, "readout": {"factor": 1.2}}},
        {"two_factor": {"gates": {"factor": 1.9}}, "averaged(7)": {"pulls": 7, "calibrations": 3}},
        {"readout_measured": {"gates": 0.05}, "averaged(7)": {"pulls": 7, "calibrations": 3}},
    ]
    rows = [_row("ibm_fez", FIRST, fits=f) for f in fits]

    found = summarize.summarize({"rows": rows})["fits"]

    assert found == {
        "runs_with_a_factor_not_identified": 1,
        "runs_with_a_factor_at_a_limit": 1,
        "averaged_7_calibrations": {"3": 2},
    }


def test_the_misfit_gives_the_median_of_each_device_and_model() -> None:
    rows = [
        _row("ibm_fez", FIRST, misfit={"locus": {"misfit": m, "rare": m / 10}})
        for m in (0.01, 0.03, 0.02)
    ]
    rows.append(_row("ibm_fez", FIRST, misfit={"locus": {"misfit": None, "rare": None}}))

    (found,) = summarize.summarize({"rows": rows})["misfit"]

    assert found == {
        "device": "ibm_fez",
        "model": "locus",
        "runs": 3,
        "median_misfit": pytest.approx(0.02),
        "median_rare": pytest.approx(0.002),
    }
