from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from conftest import require

import noisevault as nv
from study import layout_history

require("pyarrow")

FIXTURE = Path(__file__).parent / "fixtures" / "hf_archive" / "train-00000-of-00001.parquet"
# The fixture has the calibration A of ibm_fez on 2026-06-01 and B on 2026-06-02. B gives cz on
# (0, 1) the error 0.002 in place of 0.003, and it disables qubit 3.
DAY = datetime(2026, 6, 1, 12, tzinfo=UTC)


def _profile_on(day: datetime) -> nv.Profile:
    return nv.from_calibration_archive(FIXTURE, "ibm_fez", at=day)


@pytest.fixture(scope="module")
def two_qubits() -> dict:
    return layout_history.evaluate(_profile_on, [DAY], week=2, n=2)


def test_the_sums_of_a_chain_are_the_values_of_the_fixture(two_qubits: dict) -> None:
    (row,) = two_qubits["rows"]
    assert row["chains"] == {"day": (0, 1), "history": (0, 1)}
    stated, next_day = row["sums"]["day"]
    # Mean readout error 0.01 on each qubit, sx 2e-4 and 2.1e-4, and cz 0.003, then 0.002.
    assert stated["readout"] == pytest.approx(0.02)
    assert stated["1q"] == pytest.approx(0.00041)
    assert stated["all"] == pytest.approx(0.02341)
    assert next_day["2q"] == pytest.approx(0.002)
    assert next_day["all"] == pytest.approx(0.02241)


def test_the_ratios_compare_the_next_day_with_the_stated_value(two_qubits: dict) -> None:
    ratios = two_qubits["ratios"]
    assert ratios["day"]["2q"] == [pytest.approx(2 / 3)]
    assert ratios["day"]["readout"] == [1.0, 1.0]
    # Each readout and 1-qubit value stays the same, and the cz value changes.
    assert two_qubits["unchanged"]["day"] == 4


def test_a_next_day_value_of_0_gives_the_ratio_0_and_a_stated_0_gives_none() -> None:
    def with_readout(day: datetime) -> nv.Profile:
        data = _profile_on(DAY).to_dict()
        for qubit in data["qubits"]:
            error = 0.0 if (day > DAY) == (qubit["index"] != 0) else 0.03
            qubit["readout"] = {"error": error}
        return nv.Profile.model_validate(data)

    # Up to the day, qubit 0 states a readout error of 0 and each other qubit states 0.03. On
    # the next day, qubit 0 states 0.03 and each other qubit states 0.
    found = layout_history.evaluate(with_readout, [DAY], week=1, n=2)
    chain = found["rows"][0]["chains"]["day"]
    assert 0 in chain
    assert sorted(found["ratios"]["day"]["readout"]) == [0.0]
    assert found["no_ratio"]["day"] == 1
    assert layout_history.geometric_mean([0.0, 2.0]) == 0.0
    assert layout_history.geometric_mean([0.5, 2.0]) == pytest.approx(1.0)


def test_a_chain_with_a_qubit_that_the_next_day_disables_has_no_sum() -> None:
    (row,) = layout_history.evaluate(_profile_on, [DAY], week=2, n=4)["rows"]
    assert sorted(row["chains"]["day"]) == [0, 1, 2, 3]
    assert row["sums"]["day"][0] is not None
    assert row["sums"]["day"][1] is None


def test_chain_sums_needs_each_value() -> None:
    values = {"readout": [0.01, 0.02], "1q": [0.001, 0.002], "2q": [0.005]}
    assert layout_history.chain_sums(values) == pytest.approx(
        {"readout": 0.03, "1q": 0.003, "2q": 0.005, "all": 0.038}
    )
    assert layout_history.chain_sums({**values, "2q": [None]}) is None


def test_the_dates_are_at_noon_with_the_step() -> None:
    found = layout_history.dates(DAY - timedelta(hours=12), datetime(2026, 6, 11, tzinfo=UTC), 5)
    assert found == [DAY, DAY + timedelta(days=5), DAY + timedelta(days=10)]
    assert layout_history.geometric_mean([1.0, 4.0]) == pytest.approx(2.0)
    assert math.isclose(layout_history.geometric_mean([3.0]), 3.0)


def test_main_prints_the_table(capsys: pytest.CaptureFixture[str]) -> None:
    argv = [str(FIXTURE), "--device", "ibm_fez", "--start", "2026-06-01", "--end", "2026-06-01"]
    assert layout_history.main([*argv, "--week", "2", "--qubits", "2"]) == 0
    out = capsys.readouterr().out
    assert "device ibm_fez, 1 dates, chains of 2 qubits, history of 2 days" in out
    assert "2026-06-01         2  (0, 1)                (0, 1)" in out
    assert "0.0234 / 0.0234        0.0224 / 0.0224" in out
    assert "dates where both choices give the same chain: 1 of 1" in out


def test_main_refuses_a_missing_file(tmp_path: Path) -> None:
    assert layout_history.main([str(tmp_path / "none.parquet"), "--device", "ibm_fez"]) == 2
