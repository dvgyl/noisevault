from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from conftest import require

from noisevault.profile import Profile
from noisevault.sources import hf_archive
from study import archive_vendor

require("pyarrow")

FIXTURE = Path(__file__).parent / "fixtures" / "hf_archive" / "train-00000-of-00001.parquet"
AT = datetime(2026, 6, 1, 20, tzinfo=UTC)


def _same(device: str, *, at: datetime) -> Profile:
    return hf_archive.from_calibration_archive(FIXTURE, device, at=at)


def _other_readout(device: str, *, at: datetime) -> Profile:
    data = _same(device, at=at).model_dump(mode="json")
    data["qubits"][0]["readout"]["p1_given_0"] = 0.01
    return Profile.model_validate(data)


def test_finds_that_each_value_agrees() -> None:
    found = archive_vendor.compare(FIXTURE, "ibm_fez", AT, pull=_same)
    # The fixture has 4 qubits, 3 one-qubit gates on each qubit, and 7 two-qubit gates.
    sizes = {name: len(gaps) for name, gaps in found["gaps"].items()}
    assert sizes == {"one-qubit gate": 12, "readout": 8, "t1": 4, "t2": 4, "two-qubit gate": 7}
    assert all(gap == 0.0 for gaps in found["gaps"].values() for gap in gaps)
    assert (found["only_archive"], found["only_vendor"]) == (0, 0)


def test_finds_the_one_readout_value_that_differs() -> None:
    found = archive_vendor.compare(FIXTURE, "ibm_fez", AT, pull=_other_readout)
    # The archive gives 0.008 for qubit 0, so the relative difference is 0.002 / 0.01.
    readout = found["gaps"]["readout"]
    assert sorted(readout) == [0.0] * 7 + [pytest.approx(0.2)]
    assert all(
        gap == 0.0 for name, gaps in found["gaps"].items() if name != "readout" for gap in gaps
    )


def test_reports_the_share_that_agrees() -> None:
    text = archive_vendor.report(FIXTURE, ("ibm_fez",), times_for_each=1, pull=_other_readout)
    lines = text.splitlines()
    # With one time for each device, seed 3 gives 2026-09-20T14:44Z first. The fixture ends on
    # 2026-06-02, so the time is drawn again: 2026-03-14T10:33Z. The archive then gives a
    # calibration with 27 values. One of them differs.
    assert lines[2] == (
        "ibm_fez, 2026-03-14T10:33Z, 27, 0.9630, 0, 0; one-qubit gate 1.00, readout 0.88,"
        " t1 1.00, t2 1.00, two-qubit gate 1.00"
    )
    assert lines[3] == (
        "kind, values in both, share within 1e-9, 1e-6, 1e-3, 1e-2 (relative),"
        " median relative difference of the values that differ"
    )
    assert "readout, 8, 0.8750, 0.8750, 0.8750, 0.8750, " in text


def test_takes_the_same_times_for_the_same_seed() -> None:
    times = archive_vendor.sample_times(
        ("ibm_fez", "ibm_kingston"), 4, 3, archive_vendor.START, archive_vendor.END
    )
    assert [f"{at:%Y-%m-%dT%H:%MZ}" for device, at in times if device == "ibm_fez"] == [
        "2026-03-14T10:33Z",
        "2026-04-08T00:19Z",
        "2026-04-23T01:03Z",
        "2026-09-20T14:44Z",
    ]
    assert [device for device, _ in times[4:]] == ["ibm_kingston"] * 4


def test_draws_a_time_again_when_the_archive_has_no_calibration_of_the_device_then() -> None:
    first, last = datetime(2026, 5, 1, tzinfo=UTC), datetime(2026, 5, 20, tzinfo=UTC)
    spans = {"ibm_fez": (first, last)}
    times = archive_vendor.sample_times(
        ("ibm_fez", "ibm_kingston"), 4, 3, archive_vendor.START, archive_vendor.END, spans
    )
    assert all(first <= at <= last for device, at in times if device == "ibm_fez")
    assert [device for device, _ in times] == ["ibm_fez"] * 4 + ["ibm_kingston"] * 4
    assert archive_vendor.archive_spans(FIXTURE) == {
        "ibm_fez": (
            datetime(2026, 1, 31, 17, 5, 3, tzinfo=UTC),
            datetime(2026, 6, 2, 9, 0, tzinfo=UTC),
        )
    }
    outside = {"ibm_fez": (datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 2, 1, tzinfo=UTC))}
    with pytest.raises(ValueError, match="the archive holds no calibration of ibm_fez between"):
        archive_vendor.sample_times(
            ("ibm_fez",), 1, 3, archive_vendor.START, archive_vendor.END, outside
        )
