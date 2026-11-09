"""Profiles from Rigetti QCS instruction set architectures saved by the user."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import noisevault as nv
from noisevault.cli import app
from noisevault.sources.rigetti import from_rigetti

FIXTURES = Path(__file__).parent / "fixtures" / "rigetti"
ANKAA = FIXTURES / "ankaa_isa.json"
CEPHEUS = FIXTURES / "cepheus_isa.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _notes(profile: nv.Profile) -> str:
    return " ".join(profile.provenance.notes)


def _records(profile: nv.Profile, gate: str) -> dict[tuple[int, ...], Any]:
    return {r.qubits: r for r in profile.calibrations if r.gate == gate}


@pytest.fixture(scope="module")
def ankaa() -> nv.Profile:
    return from_rigetti(ANKAA)


@pytest.fixture(scope="module")
def cepheus() -> nv.Profile:
    return from_rigetti(CEPHEUS)


def test_device_and_qubit_numbers(ankaa: nv.Profile) -> None:
    assert (ankaa.id, ankaa.device.processor) == ("rigetti_ankaa-synthetic", "Ankaa")
    assert ankaa.device.num_qubits == 7
    assert ankaa.device.calibrated_at.isoformat() == "2026-10-30T07:15:00+00:00"
    assert "The oldest is 2026-10-29T22:05:00+00:00" in _notes(ankaa)
    assert sorted(ankaa.gates) == ["iswap", "rx", "rz"]
    assert ankaa.gates["rz"].virtual


def test_gaps_and_dead_nodes_are_disabled(ankaa: nv.Profile) -> None:
    assert [q.index for q in ankaa.qubits if q.disabled] == [4, 6]
    notes = _notes(ankaa)
    assert "indices 4 are not architecture nodes" in notes
    assert "nodes 6 have no site of RX, RZ, MEASURE, _" in notes


def test_readout_is_one_minus_fro_and_sentinels_stay_unknown(ankaa: nv.Profile) -> None:
    by_index = {q.index: q for q in ankaa.qubits}
    assert by_index[0].readout.error == pytest.approx(0.038)
    assert by_index[3].readout is None  # fRO of exactly 1 is a placeholder
    assert by_index[5].readout is None  # 0.5 with error 1.0 is a dead site
    notes = _notes(ankaa)
    assert "fRO of node 5" in notes and "symmetric readout error" in notes


def test_isolated_rb_wins_over_simultaneous(ankaa: nv.Profile) -> None:
    rx = _records(ankaa, "rx")
    first = rx[(0,)]
    assert (first.avg_infidelity, first.stderr, first.method, first.measured) == (
        0.0009,
        0.0001,
        "rb",
        "isolated",
    )
    sixth = rx[(5,)]
    assert (sixth.avg_infidelity, sixth.method, sixth.measured) == (0.0042, "srb", "simultaneous")
    assert rx[(3,)].disabled  # fRB 0.3 is below 1/3
    assert "per gate or per Clifford" in ankaa.gates["rx"].assumption


def test_two_qubit_sites_sentinels_and_missing_edges(ankaa: nv.Profile) -> None:
    assert ankaa.connectivity.edges == ((0, 1), (1, 2), (1, 5), (2, 3), (2, 6), (3, 5))
    iswap = _records(ankaa, "iswap")
    assert (iswap[(0, 1)].avg_infidelity, iswap[(0, 1)].stderr) == (0.009, 0.002)
    assert iswap[(1, 5)].avg_infidelity == pytest.approx(0.013)  # the site lists [5, 1]
    table = ankaa.table
    assert table.gate("iswap", (1, 2)).state == "disabled"  # 0.5 with error 1.0
    assert table.gate("iswap", (3, 5)).state == "disabled"  # an edge with no ISWAP site
    dead = table.gate("iswap", (2, 6))  # a calibrated site, but node 6 is dead
    assert (dead.kind, dead.reason) == ("qubit_disabled", "qubit 6 is disabled")
    assert table.gate("iswap", (2, 3)).state == "uncalibrated"  # a site with no fISWAP
    notes = _notes(ankaa)
    assert "ISWAP has no site on the edges 3-5," in notes
    assert "fISWAP of 1-2" in notes
    assert "no fidelity, so uncalibrated: iswap on 2-3" in notes


def _ankaa_with(change: Any) -> dict[str, Any]:
    data = _load(ANKAA)
    change({op["name"]: op for op in [*data["instructions"], *data["benchmarks"]]})
    return data


def test_a_placeholder_isolated_fidelity_falls_back_to_the_simultaneous_one() -> None:
    def placeholder(ops: dict[str, Any]) -> None:
        ops["randomized_benchmark_1q"]["sites"][0]["characteristics"][0]["value"] = 1.0

    rx = _records(from_rigetti(_ankaa_with(placeholder)), "rx")[(0,)]
    assert (rx.avg_infidelity, rx.method) == (0.0028, "srb")


def test_a_live_node_without_an_rz_site_has_rz_disabled() -> None:
    def no_rz(ops: dict[str, Any]) -> None:
        ops["RZ"]["sites"] = [s for s in ops["RZ"]["sites"] if s["node_ids"] != [0]]

    profile = from_rigetti(_ankaa_with(no_rz))
    assert profile.table.gate("rz", (0,)).state == "disabled"
    assert profile.table.gate("rz", (1,)).state == "ideal"


def test_a_characteristic_outside_its_site_is_refused() -> None:
    def moved(ops: dict[str, Any]) -> None:
        ops["MEASURE"]["sites"][0]["characteristics"][0]["node_ids"] = [1]

    with pytest.raises(nv.SourceDataError) as info:
        from_rigetti(_ankaa_with(moved))
    assert info.value.message.endswith(
        "instructions MEASURE site [0] has a fRO on nodes [1], outside the site"
    )


@pytest.mark.parametrize("ids", [[], [0, 1], [0, 0]])
def test_an_instruction_site_with_the_wrong_number_of_nodes_is_refused(ids: list[int]) -> None:
    def malformed(ops: dict[str, Any]) -> None:
        ops["RX"]["sites"][0]["node_ids"] = ids

    with pytest.raises(nv.SourceDataError) as info:
        from_rigetti(_ankaa_with(malformed))
    assert f"instructions RX has a site on {ids}, not on 1 different nodes" in info.value.message


def test_a_dead_site_marker_on_t1_leaves_t1_unknown() -> None:
    data = _load(CEPHEUS)
    t1 = data["benchmarks"][1]["sites"][0]["characteristics"][0]
    t1.update(value=0.5, error=1.0)
    profile = from_rigetti(data)
    assert profile.qubits[0].t1_us is None
    assert "T1 of node 0" in _notes(profile)


def test_instructions_it_does_not_read_are_named(ankaa: nv.Profile) -> None:
    notes = _notes(ankaa)
    assert "I is not a calibrated gate" in notes
    assert "fAR possibly includes readout error" in notes
    assert "XY takes an angle" in notes


def test_a_cz_device_with_coherence_benchmarks(cepheus: nv.Profile) -> None:
    assert cepheus.device.processor is None  # family "None"
    assert sorted(cepheus.gates) == ["cz", "rx", "rz"]
    first = cepheus.qubits[0]
    assert (first.t1_us, first.t2_us, first.t2_kind) == (28.0, 19.0, "ramsey")
    assert cepheus.qubits[3].t1_us is None  # T1 of 0 s
    cz = _records(cepheus, "cz")
    assert (cz[(0, 1)].avg_infidelity, cz[(0, 1)].stderr, cz[(0, 1)].method) == (
        0.015,
        0.004,
        "vendor",
    )
    assert cz[(2, 3)].disabled  # fCZ 0.15 is below 1/5
    assert "Rigetti does not state the fidelity metric" in cepheus.gates["cz"].assumption
    assert "CPHASE takes an angle" in _notes(cepheus)


def test_epoch_timestamps_are_placeholders() -> None:
    data = _load(CEPHEUS)
    for kind in ("instructions", "benchmarks"):
        for op in data[kind]:
            for site in op["sites"]:
                for item in site["characteristics"]:
                    item["timestamp"] = "1970-01-01T00:00:00+00:00"
    profile = from_rigetti(data)
    assert profile.device.calibrated_at is None
    assert "1970-01-01 is a placeholder" in _notes(profile)


def test_a_timestamp_with_no_zone_is_utc() -> None:
    data = _load(CEPHEUS)
    data["instructions"][2]["sites"][0]["characteristics"][0]["timestamp"] = "2026-11-01T09:00:00"
    profile = from_rigetti(data)
    assert profile.device.calibrated_at.isoformat() == "2026-11-01T09:00:00+00:00"
    assert "NoiseVault reads 1 timestamp with no time zone as UTC" in _notes(profile)


def test_device_names_the_profile_and_dicts_import() -> None:
    assert from_rigetti(ANKAA, device="ankaa-3").id == "rigetti_ankaa-3"
    from_dict = nv.from_rigetti(_load(ANKAA))
    assert from_dict.fingerprint == nv.from_rigetti(ANKAA).fingerprint


def test_provenance(ankaa: nv.Profile) -> None:
    digest = hashlib.sha256(ANKAA.read_bytes()).hexdigest()
    assert ankaa.provenance.source_hash == f"sha256:{digest}"
    assert (ankaa.provenance.redistributable, ankaa.provenance.source_kind) == ("no", "user_file")


def _changed(path: Path, change: Any) -> dict[str, Any]:
    data = _load(path)
    change(data)
    return data


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"name": "x"}, "the top level has no 'architecture'"),
        (
            _changed(CEPHEUS, lambda d: d["architecture"]["edges"].append({"node_ids": [0, 9]})),
            "architecture.edges[3] is [0, 9], not two different architecture nodes",
        ),
        (
            _changed(
                CEPHEUS,
                lambda d: d["instructions"][3]["sites"].append(
                    {"node_ids": [0, 3], "characteristics": []}
                ),
            ),
            "instructions CZ has a site on [0, 3], which is not an architecture edge",
        ),
        (
            _changed(CEPHEUS, lambda d: d["instructions"][2]["sites"][0].update(node_ids=[7])),
            "instructions MEASURE has a site on node 7, which is not an architecture node",
        ),
        (
            _changed(CEPHEUS, lambda d: d["architecture"].update(nodes=[], edges=[])),
            "architecture.nodes is empty",
        ),
        (
            _changed(
                CEPHEUS,
                lambda d: d["instructions"][2]["sites"][0]["characteristics"][0].update(
                    value="0.95"
                ),
            ),
            "instructions[2].sites[0].characteristics[0].value is '0.95'",
        ),
        (
            _changed(
                CEPHEUS,
                lambda d: d["instructions"][2]["sites"][0]["characteristics"][0].update(
                    timestamp="yesterday"
                ),
            ),
            "is 'yesterday', not an ISO 8601 time",
        ),
    ],
    ids=["shape", "edge", "site-edge", "site-node", "no-nodes", "string-value", "timestamp"],
)
def test_data_it_cannot_read_raises_source_data_error(data: dict, message: str) -> None:
    with pytest.raises(nv.SourceDataError) as info:
        from_rigetti(data)
    assert message in info.value.message


@pytest.mark.parametrize("path", [ANKAA, CEPHEUS], ids=["ankaa", "cepheus"])
def test_the_profile_passes_nv_check(path: Path, tmp_path: Path) -> None:
    profile = from_rigetti(path)
    assert profile.check().passed
    saved = tmp_path / "isa_profile.json"
    profile.save(saved)
    result = CliRunner().invoke(app, ["check", str(saved)])
    assert result.exit_code == 0, result.output
