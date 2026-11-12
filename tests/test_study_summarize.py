from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import pytest

from study import summarize
from study.analyze import PRIMARY

_HASHES = itertools.count(1)


def _row(device: str, better: dict[tuple[str, str, str], str], **more: Any) -> dict[str, Any]:
    paired = [
        {
            "a": a,
            "b": b,
            "group": group,
            "primary": True,
            "better": "neither",
            "log": {
                "outcome": "tested",
                "difference": 0.01,
                "standard_error": 0.002,
                "z": 5.0,
                "better": better.get((a, b, group), "neither"),
            },
        }
        for a, b, group in PRIMARY
    ]
    return {
        "device": device,
        "counts_sha256": f"{next(_HASHES):064x}",
        "chain": "best",
        "run_at": "2026-11-12T03:00:00Z",
        "stated": {"zero_readouts": 2, "ruled_out": {"all": 7}},
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


def test_a_comparison_counts_the_runs_of_each_device_and_is_confirmed_after_holm() -> None:
    first = PRIMARY[0]
    wins = {first: "a"}
    rows = [_row("ibm_fez", wins) for _ in range(6)] + [
        _row("ibm_kingston", wins) for _ in range(4)
    ]
    rows.append(_row("ibm_kingston", {first: "b"}))

    summary = summarize.summarize({"rows": rows})

    found = summary["primary"][0]
    assert found["by_device"]["ibm_fez"] == {"a": 6, "b": 0, "neither": 0}
    assert found["total"] == {"a": 10, "b": 1, "neither": 0}
    assert found["p_value"] == pytest.approx(2 * 12 / 2048)
    assert found["p_holm"] == pytest.approx(5 * 2 * 12 / 2048)
    assert found["confirmed"] is False
    assert all(c["total"]["neither"] == 11 and c["p_value"] == 1.0 for c in summary["primary"][1:])
    assert summary["stated_calibration"] == {"runs_ruled_out": 11, "runs_with_a_zero_readout": 11}
    assert summary["devices"] == ["ibm_fez", "ibm_kingston"]


def test_the_staleness_points_give_the_model_minus_the_calibration(tmp_path: Path) -> None:
    values = {"readout": {"median_h": 51.0}, "one_qubit": {"median_h": 151.0}, "two_qubit": None}
    ages = {
        "calibration": {"calibration_h": 0.5, "values": values},
        "stale(7)": {"calibration_h": 168.5, "values": values},
    }
    row = _row("ibm_fez", {}, ages=ages)
    row["paired"].append(
        {
            "a": "calibration",
            "b": "stale(7)",
            "group": "all",
            "primary": False,
            "better": "a",
            "log": {
                "outcome": "tested",
                "difference": 0.02,
                "standard_error": 0.004,
                "z": 5.0,
                "better": "a",
            },
        }
    )
    analysis = tmp_path / "analysis.json"
    analysis.write_text(json.dumps({"rows": [row]}))

    assert summarize.main([str(analysis), "--out", str(tmp_path)]) == 0

    points = json.loads((tmp_path / "summary.json").read_text())["staleness"]
    assert [p["model"] for p in points] == ["calibration", "stale(7)"]
    assert points[0]["difference"] == 0.0
    assert points[1]["difference"] == pytest.approx(-0.02) and points[1]["z"] == -5.0
    assert points[1]["median_age_h"] == {"readout": 51.0, "one_qubit": 151.0, "two_qubit": None}
    assert "| ibm_fez | best |" in (tmp_path / "summary.md").read_text()


def test_the_summary_also_gives_the_comparisons_without_the_marked_runs() -> None:
    first = PRIMARY[0]
    rows = [_row("ibm_fez", {first: "a"}) for _ in range(3)]
    rows.append(_row("ibm_fez", {first: "b"}, marked=True))
    rows.append(_row("ibm_fez", {first: "a"}, marked=None))

    summary = summarize.summarize({"rows": rows})

    assert summary["runs_marked"] == 1
    assert summary["primary"][0]["total"] == {"a": 4, "b": 1, "neither": 0}
    assert summary["primary_without_marked_runs"][0]["total"] == {"a": 4, "b": 0, "neither": 0}
    assert "p_holm" in summary["primary_without_marked_runs"][0]


def test_spearman_gives_tied_values_their_mean_rank() -> None:
    assert summarize.spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 100]) == pytest.approx(1.0)
    assert summarize.spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1.0)
    # Ranks (1, 2.5, 2.5, 4, 5) and (1, 2, 3, 4, 5): the Pearson correlation of the ranks.
    assert summarize.spearman([1, 2, 2, 3, 4], [1, 2, 3, 4, 5]) == pytest.approx(0.9746794)
    assert summarize.spearman([1, 2, 3, 4], [1, 2, 3, 4]) is None
    assert summarize.spearman([1, 1, 1, 1, 1], [1, 2, 3, 4, 5]) is None


def test_the_age_of_a_value_is_compared_with_its_fitted_factor() -> None:
    rows = []
    for run in range(3):
        ages = {
            "qubits": {str(q): {"one_qubit": 10.0 * q + run, "readout": 1.0} for q in range(3)},
            "pairs": {"0-1": {"two_qubit": 5.0 + run}, "1-2": {"two_qubit": 50.0 + run}},
        }
        fits = {
            "qubits": [{"qubit": q, "factor": math.exp(0.1 * q + 0.01 * run)} for q in range(3)],
            "pairs": [{"qubits": [0, 1], "factor": 2.0}, {"qubits": [1, 2], "factor": None}],
        }
        more = {"ages": {"calibration": {"calibration_h": 1.0, "values": {}, "loci": ages}}}
        rows.append(_row("ibm_fez", {}, fits={"locus": fits}, **more))

    summary = summarize.summarize({"rows": rows})

    assert summary["age_against_error"] == [
        {"device": "ibm_fez", "loci": "pairs", "points": 3, "spearman": None},
        {"device": "ibm_fez", "loci": "qubits", "points": 9, "spearman": pytest.approx(1.0)},
    ]
    assert "| ibm_fez | qubits | 9 | +1.000 |" in summarize.markdown(summary)


def test_a_run_that_is_in_the_analysis_two_times_counts_one_time() -> None:
    first = PRIMARY[0]
    copy = _row("ibm_fez", {first: "a"})
    rows = [copy] * 8 + [_row("ibm_fez", {first: "b"})]

    summary = summarize.summarize({"rows": rows})

    assert summary["runs"] == 2
    assert summary["duplicate_rows_removed"] == 7
    assert summary["primary"][0]["total"] == {"a": 1, "b": 1, "neither": 0}
    assert "Duplicate rows removed: 7." in summarize.markdown(summary)
