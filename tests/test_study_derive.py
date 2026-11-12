from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import noisevault as nv
from noisevault.counts import plan, simulate
from study import derive

RUN_AT = datetime(2026, 4, 16, 9, 30, tzinfo=UTC)


def _run(folder: Path, name: str, gate: float, readout: float, seed: int) -> None:
    profile = nv.load("ibm_kingston@2026-04-15")
    truth = profile.model_copy(
        update={"unmodeled_error": {"gates": {"factor": gate}, "readout": {"factor": readout}}}
    )
    folder.mkdir(parents=True)
    simulate(truth, plan(profile), shots=4000, seed=seed, run_at=RUN_AT).save(
        folder / f"{name}.counts.json"
    )
    profile.save(folder / f"{name}.profile.json")


def test_derive_fits_each_counts_file_below_the_root(tmp_path: Path) -> None:
    _run(tmp_path / "ibm_kingston" / "2026-04-13", "best", 1.8, 1.3, seed=1)
    _run(tmp_path / "ibm_kingston" / "2026-04-20", "median", 1.0, 1.0, seed=2)
    (tmp_path / "ibm_kingston" / "2026-04-13" / "._best.counts.json").write_bytes(b"\x00\x05")

    data = derive.derive(tmp_path)

    assert data["failures"] == []
    high, low = sorted(data["rows"], key=lambda r: -r["gates"]["factor"])
    assert high["gates"]["low"] < 1.8 < high["gates"]["high"]
    assert low["gates"]["low"] < 1.0 < low["gates"]["high"]
    assert high["vendor"] == "ibm"
    assert high["technology"] == "superconducting"


def test_the_output_holds_no_counts_and_no_file_name(tmp_path: Path) -> None:
    _run(tmp_path / "campaign" / "week", "best", 1.8, 1.3, seed=1)
    out = tmp_path / "results"

    assert derive.main([str(tmp_path / "campaign"), "--out", str(out)]) == 0

    text = (out / "fits.json").read_text() + (out / "fits.md").read_text()
    data = json.loads((out / "fits.json").read_text())
    assert "best.counts.json" not in text
    assert str(tmp_path) not in text
    assert all(set(c) == _CIRCUIT_KEYS for row in data["rows"] for c in row["circuits"])


def test_a_counts_file_with_no_profile_beside_it_is_a_failure(tmp_path: Path) -> None:
    _run(tmp_path / "week", "best", 1.0, 1.0, seed=1)
    (tmp_path / "week" / "best.profile.json").unlink()

    assert derive.main([str(tmp_path), "--out", str(tmp_path / "results")]) == 1
    data = json.loads((tmp_path / "results" / "fits.json").read_text())
    assert data["rows"] == []
    assert len(data["failures"]) == 1


_CIRCUIT_KEYS = {
    "name",
    "qubits",
    "shots",
    "tvd_profile",
    "tvd_fitted",
    "shot_noise_95",
    "impossible",
}
