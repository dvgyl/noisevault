from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from study import summarize
from study.analyze import PRIMARY


def _row(device: str, better: dict[tuple[str, str, str], str], **more: Any) -> dict[str, Any]:
    paired = [
        {
            "a": a,
            "b": b,
            "group": group,
            "primary": True,
            "better": better.get((a, b, group), "neither"),
            "log": {"outcome": "tested", "difference": 0.01, "standard_error": 0.002, "z": 5.0},
        }
        for a, b, group in PRIMARY
    ]
    return {
        "device": device,
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
            "log": {"outcome": "tested", "difference": 0.02, "standard_error": 0.004, "z": 5.0},
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
