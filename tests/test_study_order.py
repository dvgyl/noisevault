"""The row of a run does not depend on the order of its circuits.

The campaign runs the circuits of a run in a random order, so a counts file holds them in that
order. The row of such a run must equal the row of the same run in the order of the suite.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from test_study_analyze import RUN_AT, FakeSource, _device

from noisevault.counts import MeasuredCounts, simulate
from noisevault.suite import suite
from study import analyze, models

# The fields that depend on the bytes of the counts file, and so on the order of its circuits:
# its SHA-256, and the values whose random draws take a seed from it (the reference of the
# excess, and so the misfit).
FROM_THE_BYTES = {
    "counts_sha256",
    "excess_noise_mean",
    "excess_noise_95",
    "p_value",
    "noise_mean",
    "misfit",
    # The interval of a factor of nv compare comes from draws with a seed from the train counts.
    "low",
    "high",
}


def _properties(device: str, at: datetime) -> dict[str, Any]:
    """IBM's properties endpoint for the 3-qubit line: each value of qubit q and of pair (a, b)
    is q or a + b hours older than the calibration, 30 h before the run."""
    calibrated = RUN_AT - timedelta(hours=30)
    if at <= calibrated:
        return None  # type: ignore[return-value]

    def date(hours: float) -> str:
        return (calibrated - timedelta(hours=hours)).isoformat()

    qubits = [[{"name": "prob_meas1_prep0", "value": 0.01, "date": date(q)}] for q in range(3)]
    gates = [
        {
            "gate": "cz",
            "qubits": [a, b],
            "parameters": [{"name": "gate_error", "date": date(a + b)}],
        }
        for a, b in ((0, 1), (1, 2))
    ]
    return {"last_update_date": calibrated.isoformat(), "qubits": qubits, "gates": gates}


def _row(folder: Path, counts: MeasuredCounts) -> dict[str, Any]:
    folder.mkdir(parents=True)
    counts.save(folder / "best-deep.counts.json")
    _device().save(folder / "best-deep.profile.json")
    return analyze.analyze_run(
        folder / "best-deep.counts.json",
        cache=None,
        offline=False,
        pull=FakeSource(_device()),
        properties=_properties,
        samples=1000,
    )


def _same(a: Any, b: Any, where: str = "row") -> None:
    """Assert that two rows are equal, apart from the fields of ``FROM_THE_BYTES`` and the order
    of the lists of named items. Floats can differ by the order of a sum."""
    if isinstance(a, dict):
        assert set(a) == set(b), where
        for key in a:
            # The block "misfit" stays; only the misfit value of each model is skipped.
            if key not in FROM_THE_BYTES or (where, key) == ("row", "misfit"):
                _same(a[key], b[key], f"{where}.{key}")
    elif isinstance(a, list):
        assert len(a) == len(b), where
        if a and all(isinstance(x, dict) and "name" in x for x in a):
            a = sorted(a, key=lambda x: (x["name"], str(x.get("model"))))
            b = sorted(b, key=lambda x: (x["name"], str(x.get("model"))))
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _same(x, y, f"{where}[{i}]")
    elif isinstance(a, float) and isinstance(b, float):
        assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9), (where, a, b)
    else:
        assert a == b, (where, a, b)


def test_a_run_in_a_random_order_gives_the_row_of_the_run_in_suite_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(models, "fit_factor", lambda build, counts, circuits: 1.0)
    calibration = _device()
    counts = simulate(calibration, suite(calibration), shots=200, seed=3, run_at=RUN_AT)
    data = counts.to_dict()
    # Put the readout circuit last and qubit 1 first, as a random order can do.
    order = list(data["circuits"])
    random.Random(5).shuffle(order)
    order.sort(key=lambda c: c["name"] == "readout")
    first = next(i for i, c in enumerate(order) if c["qubits"][0] == 1)
    order.insert(0, order.pop(first))
    shuffled = MeasuredCounts.model_validate(data | {"circuits": order})

    ordered = _row(tmp_path / "suite", counts)
    random_order = _row(tmp_path / "random", shuffled)

    assert random_order["qubits"] == ordered["qubits"] == [0, 1, 2]
    loci = random_order["ages"]["calibration"]["loci"]
    assert set(loci["pairs"]) == {"0-1", "1-2"}
    assert loci["pairs"]["1-2"]["two_qubit"] == pytest.approx(30 + 3)
    _same(ordered, random_order)
