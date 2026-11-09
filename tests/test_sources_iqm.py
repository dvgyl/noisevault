"""Profiles from IQM Resonance quality metric sets saved by the user."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import noisevault as nv
from noisevault.cli import app
from noisevault.sources.iqm import from_iqm

FIXTURES = Path(__file__).parent / "fixtures" / "iqm"
METRICS = FIXTURES / "metrics.json"
ARCHITECTURE = FIXTURES / "architecture.json"


def _metrics() -> dict[str, Any]:
    return json.loads(METRICS.read_text())


def _architecture() -> dict[str, Any]:
    return json.loads(ARCHITECTURE.read_text())


def _with(field: str, **changes: Any) -> dict[str, Any]:
    data = _metrics()
    (obs,) = [o for o in data["observations"] if o["dut_field"] == field]
    obs.update(changes)
    return data


def _added(*observations: dict[str, Any]) -> dict[str, Any]:
    data = _metrics()
    data["observations"] += [{"unit": "", "invalid": False, **o} for o in observations]
    return data


def _notes(profile: nv.Profile) -> str:
    return " ".join(profile.provenance.notes)


@pytest.fixture(scope="module")
def garnet() -> nv.Profile:
    return from_iqm(METRICS, architecture=ARCHITECTURE, device="garnet")


def test_qubit_numbers_and_device(garnet: nv.Profile) -> None:
    assert (garnet.id, garnet.device.technology) == ("iqm_garnet", "superconducting")
    assert garnet.device.num_qubits == 5
    assert [q.label for q in garnet.qubits] == ["QB1", "QB2", "QB3", "QB4", "QB5"]
    assert garnet.device.calibrated_at.isoformat() == "2026-10-28T06:40:00+00:00"
    assert garnet.provenance.extra["calibration_set_id"] == "9e8d7c6b-5a49-4382-9170-6f5e4d3c2b1a"


def test_coherence_is_converted_from_seconds_and_prefers_the_echo_t2(garnet: nv.Profile) -> None:
    first, fourth = garnet.qubits[0], garnet.qubits[3]
    assert (first.t1_us, first.t2_us, first.t2_kind) == (41.0, 22.0, "echo")
    assert (fourth.t1_us, fourth.t2_us, fourth.t2_kind) == (29.0, 12.0, "ramsey")


def test_readout_errors_keep_both_directions(garnet: nv.Profile) -> None:
    readout = garnet.qubits[0].readout
    assert (readout.p1_given_0, readout.p0_given_1, readout.duration_ns) == (0.012, 0.031, 1500)
    assert garnet.qubits[4].readout is None  # the architecture lists no measure on QB5
    assert "measure on QB5 (constant)" in _notes(garnet)
    alone = from_iqm(METRICS)
    assert alone.qubits[4].readout.error == pytest.approx(0.038)  # only a fidelity
    assert "symmetric readout error" in _notes(alone) and "QB5" in _notes(alone)


def test_gate_fidelities_become_average_infidelities(garnet: nv.Profile) -> None:
    r = {rec.qubits: rec for rec in garnet.calibrations if rec.gate == "r"}
    first = r[(0,)]
    assert (first.avg_infidelity, first.stderr, first.method, first.duration_ns) == (
        0.0009,
        0.0001,
        "rb",
        20,
    )
    cz = {rec.qubits: rec for rec in garnet.calibrations if rec.gate == "cz"}
    assert (cz[(0, 1)].avg_infidelity, cz[(0, 1)].stderr, cz[(0, 1)].method) == (
        0.0048,
        0.0008,
        "irb",
    )
    assert "per gate or per Clifford" in garnet.gates["r"].assumption
    assert "irb for interleaved RB" in garnet.gates["cz"].assumption
    assert "stderr" in garnet.gates["cz"].assumption


def test_the_architecture_selects_the_implementation(garnet: nv.Profile) -> None:
    cz = {rec.qubits: rec for rec in garnet.calibrations if rec.gate == "cz"}
    assert (cz[(1, 2)].avg_infidelity, cz[(1, 2)].duration_ns) == (0.0052, 44)  # crf override
    arch = _architecture()
    arch["gates"]["cz"]["override_default_implementation"] = {}
    default = from_iqm(METRICS, architecture=arch)
    assert default.table.gate("cz", (1, 2)).avg_infidelity == pytest.approx(0.0069)  # tgss


def test_an_implementation_without_data_stays_unknown() -> None:
    arch = _architecture()
    arch["gates"]["cz"]["override_default_implementation"] = {"QB2,QB3": "slepian"}
    profile = from_iqm(METRICS, architecture=arch)
    assert not any(r.gate == "cz" and r.qubits == (1, 2) for r in profile.calibrations)
    assert profile.table.gate("cz", (1, 2)).state == "uncalibrated"
    assert "cz on QB2__QB3 (slepian)" in _notes(profile)


def test_the_selected_implementation_must_list_the_locus() -> None:
    arch = _architecture()
    tgss, crf = (arch["gates"]["cz"]["implementations"][i]["loci"] for i in ("tgss", "crf"))
    tgss.remove(["QB1", "QB2"])
    crf.append(["QB1", "QB2"])
    profile = from_iqm(METRICS, architecture=arch)
    assert profile.table.gate("cz", (0, 1)).state == "uncalibrated"  # only tgss has data
    assert "cz on QB1__QB2 (tgss)" in _notes(profile)


def test_a_duration_without_a_fidelity_is_kept() -> None:
    data = _metrics()
    data["observations"] = [
        o
        for o in data["observations"]
        if not o["dut_field"].startswith("metrics.rb.prx.drag_crf.QB1")
    ]
    profile = from_iqm(data, architecture=ARCHITECTURE)
    first = profile.table.gate("r", (0,))
    assert (first.state, first.duration_ns) == ("uncalibrated", 20)


def test_a_readout_duration_under_measure_fidelity_is_read() -> None:
    data = _with("gates.measure.constant.QB1.duration")
    for o in data["observations"]:
        if o["dut_field"] == "gates.measure.constant.QB1.duration":
            o["dut_field"] = "gates.measure_fidelity.constant.QB1.duration"
    profile = from_iqm(data, architecture=ARCHITECTURE)
    assert profile.qubits[0].readout.duration_ns == 1500


def test_the_method_comes_from_the_observation_name() -> None:
    data = _metrics()
    for o in data["observations"]:
        if o["dut_field"] == "metrics.irb.cz.tgss.QB1__QB2.fidelity:par=d2":
            o["dut_field"] = "metrics.rb.cz.tgss.QB1__QB2.fidelity:par=d2"
    profile = from_iqm(data, architecture=ARCHITECTURE)
    assert profile.table.gate("cz", (0, 1)).spec.method == "rb"


def test_an_architecture_locus_of_the_wrong_size_is_refused() -> None:
    arch = _architecture()
    arch["gates"]["cz"]["implementations"]["tgss"]["loci"].append(["QB1"])
    with pytest.raises(nv.SourceDataError) as info:
        from_iqm(METRICS, architecture=arch)
    assert info.value.message == (
        "the dict passed in: gates.cz.implementations.tgss.loci[4] is ['QB1'], not 2 components"
    )


def test_pairs_and_qubits_the_architecture_leaves_out_are_disabled(garnet: nv.Profile) -> None:
    assert garnet.connectivity.edges == ((0, 1), (0, 3), (1, 2), (2, 3))
    table = garnet.table
    records = {r.qubits: r for r in garnet.calibrations if r.gate == "cz"}
    assert records[(3, 4)].disabled  # a fidelity, but no DQA locus
    assert not table.allowed("cz", (3, 4))
    assert table.qubit(4).disabled  # QB5 is not in the DQA qubits
    assert {r.qubits: r for r in garnet.calibrations if r.gate == "r"}[(4,)].disabled
    assert table.gate("cz", (0, 3)).state == "uncalibrated"  # a DQA locus with no fidelity
    notes = _notes(garnet)
    assert "cz on QB4__QB5 has data but no locus in the architecture" in notes
    assert "cz has no fidelity on QB1__QB4" in notes


def test_placeholders_and_invalid_values_stay_unknown(garnet: nv.Profile) -> None:
    assert garnet.table.gate("cz", (2, 3)).state == "uncalibrated"  # fidelity exactly 1
    assert garnet.qubits[4].t1_us is None  # its T1 is marked invalid
    notes = _notes(garnet)
    assert "metrics.irb.cz.tgss.QB3__QB4.fidelity:par=d2 (1.0)" in notes
    assert "invalid: characterization.model.QB5.t1_time" in notes


def test_repeated_and_unread_observations_are_named(garnet: nv.Profile) -> None:
    notes = _notes(garnet)
    assert "skipped metrics.rb.prx.drag_crf.QB1.fidelity" in notes
    assert "does not read metrics.qndness.measure, metrics.rb.clifford" in notes


def test_without_an_architecture_the_pairs_with_data_are_connected() -> None:
    profile = from_iqm(METRICS)
    assert profile.id == "iqm_metrics"
    assert profile.connectivity.edges == ((0, 1), (1, 2), (2, 3), (3, 4))
    assert profile.table.gate("cz", (3, 4)).avg_infidelity == pytest.approx(0.0093)
    assert not profile.table.qubit(4).disabled
    assert "for cz on QB2__QB3 (cz.crf)" in _notes(profile)


def test_a_qubit_no_observation_names_is_disabled() -> None:
    data = _added({"dut_field": "characterization.model.QB8.t1_time", "value": 3e-5, "unit": "s"})
    profile = from_iqm(data)
    assert profile.device.num_qubits == 8
    assert [q.index for q in profile.qubits if q.disabled] == [5, 6]
    assert "no observation names QB6, QB7" in _notes(profile)


@pytest.mark.parametrize(("fidelity", "state"), [(0.2, "disabled"), (0.21, "calibrated")])
def test_a_fidelity_at_the_physical_bound_disables_the_gate(fidelity: float, state: str) -> None:
    data = _with("metrics.irb.cz.tgss.QB1__QB2.fidelity:par=d2", value=fidelity)
    profile = from_iqm(data, architecture=ARCHITECTURE)
    assert profile.table.gate("cz", (0, 1)).state == state


def test_loci_with_a_computational_resonator_are_named_and_left_out() -> None:
    data = _added({"dut_field": "metrics.irb.cz.tgss.QB1__COMP_R.fidelity", "value": 0.99})
    profile = from_iqm(data)
    assert all(rec.qubits != (0,) for rec in profile.calibrations if rec.gate == "cz")
    assert "no computational resonators" in _notes(profile) and "COMP_R" in _notes(profile)


def test_a_dict_and_the_public_api() -> None:
    from_dict = nv.from_iqm(_metrics(), architecture=_architecture(), device="garnet")
    from_file = nv.from_iqm(METRICS, architecture=ARCHITECTURE, device="garnet")
    assert from_dict.fingerprint == from_file.fingerprint
    assert from_iqm(_metrics()).id == "iqm_device"


def test_provenance_hashes_what_was_read(garnet: nv.Profile) -> None:
    import hashlib

    digest = hashlib.sha256(METRICS.read_bytes()).hexdigest()
    assert garnet.provenance.source_hash == f"sha256:{digest}"
    assert garnet.provenance.redistributable == "unknown"
    assert garnet.provenance.source_kind == "user_file"
    assert garnet.provenance.extra["architecture_hash"].startswith("sha256:")


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({**_metrics(), "invalid": True}, "IQM marks this quality metric set invalid"),
        (
            _with("characterization.model.QB1.t1_time", unit="Hz"),
            "characterization.model.QB1.t1_time has the unit 'Hz'",
        ),
        (
            _with("metrics.irb.cz.tgss.QB1__QB2.fidelity:par=d2", unit="%"),
            "metrics.irb.cz.tgss.QB1__QB2.fidelity:par=d2 has the unit '%'",
        ),
        (
            _with("metrics.irb.cz.tgss.QB1__QB2.fidelity:par=d2", value="0.99"),
            "metrics.irb.cz.tgss.QB1__QB2.fidelity:par=d2 is '0.99', not a finite number",
        ),
        ({"observations": [{"dut_field": "x", "value": 1}]}, "observations[0] has no 'unit'"),
        ({"observations": "none"}, "observations is not a JSON array"),
        ({"observations": []}, "names no qubit"),
    ],
    ids=["invalid-set", "time-unit", "ratio-unit", "string-value", "no-unit", "shape", "empty"],
)
def test_data_it_cannot_read_raises_source_data_error(data: dict, message: str) -> None:
    with pytest.raises(nv.SourceDataError) as info:
        from_iqm(data)
    assert message in info.value.message


def test_metrics_and_architecture_of_different_calibration_sets_are_refused() -> None:
    arch = _architecture() | {"calibration_set_id": "00000000-0000-0000-0000-000000000000"}
    with pytest.raises(nv.SourceDataError) as info:
        from_iqm(METRICS, architecture=arch)
    assert "the architecture of calibration set 00000000" in info.value.message
    assert "architecture=" in (info.value.hint or "")


def test_a_file_that_is_not_json_names_the_file(tmp_path: Path) -> None:
    bad = tmp_path / "metrics.json"
    bad.write_text("{not json")
    with pytest.raises(nv.SourceDataError) as info:
        from_iqm(bad)
    assert info.value.message.startswith("metrics.json is not JSON")


def test_the_profile_passes_nv_check(garnet: nv.Profile) -> None:
    result = garnet.check()
    assert result.passed, result.summary()


def test_nv_check_runs_on_the_saved_profile(garnet: nv.Profile, tmp_path: Path) -> None:
    path = tmp_path / "garnet.json"
    garnet.save(path)
    result = CliRunner().invoke(app, ["check", str(path)])
    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("iqm_garnet@2026-10-28T06:40:00Z")


def test_the_fixture_is_not_changed_by_the_importer() -> None:
    data = _metrics()
    before = copy.deepcopy(data)
    from_iqm(data)
    assert data == before
