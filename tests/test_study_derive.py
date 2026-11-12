from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import noisevault as nv
from noisevault.counts import plan, simulate
from noisevault.suite import role, suite
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

    data, table = derive.derive(tmp_path)

    assert data["failures"] == []
    assert [r["gates"]["factor"] for r in table["rows"]] == [
        float(f"{r['gates']['factor']:.6g}") for r in data["rows"]
    ]
    assert {r["source"] for r in table["rows"]} == {"simulated"}
    high, low = sorted(data["rows"], key=lambda r: -r["gates"]["factor"])
    assert high["gates"]["low"] < 1.8 < high["gates"]["high"]
    assert low["gates"]["low"] < 1.0 < low["gates"]["high"]
    assert high["vendor"] == "ibm"
    assert high["technology"] == "superconducting"


def test_the_output_holds_no_counts_and_no_file_name(tmp_path: Path) -> None:
    _run(tmp_path / "campaign" / "week", "best", 1.8, 1.3, seed=1)
    out = tmp_path / "results"

    assert derive.main([str(tmp_path / "campaign"), "--out", str(out)]) == 0

    text = "".join((out / name).read_text() for name in ("fits.json", "fits.md", "table.md"))
    assert json.loads((out / "table.json").read_text())["rows"][0]["device"] == "ibm_kingston"
    data = json.loads((out / "fits.json").read_text())
    assert "best.counts.json" not in text
    assert str(tmp_path) not in text
    assert all(set(c) == _CIRCUIT_KEYS for row in data["rows"] for c in row["circuits"])


def test_a_counts_file_with_no_profile_beside_it_is_a_failure(tmp_path: Path) -> None:
    _run(tmp_path / "week", "best", 1.0, 1.0, seed=1)
    (tmp_path / "week" / "best.profile.json").unlink()

    out, rows = str(tmp_path / "results"), str(tmp_path / "rows")
    assert derive.main([str(tmp_path / "week"), "--out", out, "--rows", rows]) == 1
    data = json.loads((tmp_path / "results" / "fits.json").read_text())
    assert data["rows"] == []
    assert len(data["failures"]) == 1


def test_a_failure_gives_no_file_name_and_no_error_message(tmp_path: Path) -> None:
    week = tmp_path / "campaign" / "week"
    week.mkdir(parents=True)
    (week / "best-JOB-PRIVATE-123.counts.json").write_text("{}")
    out = tmp_path / "results"

    assert derive.main([str(tmp_path / "campaign"), "--out", str(out)]) == 1

    text = (out / "fits.json").read_text() + (out / "fits.md").read_text()
    (failure,) = json.loads((out / "fits.json").read_text())["failures"]
    assert set(failure) == {"error", "file"}
    assert failure["file"].startswith("sha256:")
    assert "PRIVATE" not in text and "week" not in text and str(tmp_path) not in text


_CIRCUIT_KEYS = {
    "name",
    "qubits",
    "shots",
    "tvd_profile",
    "tvd_fitted",
    "shot_noise_95",
    "impossible",
}


def test_a_run_of_the_deep_circuit_set_is_fitted_on_its_train_circuits_only() -> None:
    profile = nv.Profile.uniform(
        "line",
        technology="superconducting",
        num_qubits=2,
        one_qubit_error=1e-3,
        two_qubit_error=1e-2,
        readout_error=2e-2,
    )
    counts = simulate(profile, suite(profile), shots=100, seed=1, run_at=RUN_AT)

    used, fit_on = derive.fit_counts(counts)

    assert fit_on == "train"
    assert {role(c.name) for c in counts.circuits} == {"train", "test_near", "test_far", "probe"}
    assert {role(c.name) for c in used.circuits} == {"train"}
    assert derive.fit_counts(counts.select(["readout", "ghz_chain"]))[1] == "all"


def test_a_kept_fit_is_used_again_for_the_same_files_and_the_same_code(
    tmp_path: Path, monkeypatch
) -> None:
    _run(tmp_path / "campaign" / "week", "best", 1.8, 1.3, seed=1)
    rows = tmp_path / "rows"
    first, _ = derive.derive(tmp_path / "campaign", rows)
    calls = []
    real = derive.fit
    monkeypatch.setattr(derive, "fit", lambda path: calls.append(path) or real(path))

    again, table = derive.derive(tmp_path / "campaign", rows)

    assert calls == [] and again == first and len(table["rows"]) == 1
    monkeypatch.setattr("study.code_version", lambda: "another version")
    derive.derive(tmp_path / "campaign", rows)
    assert len(calls) == 1
