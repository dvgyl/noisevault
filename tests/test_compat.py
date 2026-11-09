from __future__ import annotations

import hashlib
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
from conftest import MANILA_V01, V01, migrated, require
from typer.testing import CliRunner

from noisevault.channels import gate_channels, readout_matrix, superoperator
from noisevault.cli import app
from noisevault.errors import MigrationWarning, NoiseVaultError
from noisevault.profile import Profile, load_file
from noisevault.table import Unavailable

BASELINE = np.load(V01 / "baseline_superops.npz")
META = json.loads(str(BASELINE["_meta"]))
KEYS = sorted(k for k in BASELINE.files if k != "_meta")
PROFILES = {path.name: migrated(path) for path in sorted(V01.glob("*.json"))}


def test_baseline_covers_every_snapshot() -> None:
    assert {key.split("|")[0] for key in KEYS} == set(PROFILES)
    assert sum("|readout|" not in key for key in KEYS) == 99


@pytest.mark.parametrize("key", [k for k in KEYS if "|readout|" not in k])
def test_gate_superoperators_match_0_1(key: str) -> None:
    file, op, qubits_text = key.split("|")
    qubits = tuple(int(q) for q in qubits_text.split(","))
    table = PROFILES[file].table
    # 0.1 borrowed calibrations through aliases (h -> sx); 1.0 names the calibration it uses.
    found = table.gate(META[key]["calibration"], qubits)
    built = gate_channels(found, [table.qubit(q) for q in qubits])
    assert np.abs(superoperator(built.channels, qubits) - BASELINE[key]).max() <= 1e-12


@pytest.mark.parametrize("key", [k for k in KEYS if "|readout|" in k])
def test_readout_matrices_match_0_1(key: str) -> None:
    file, _, qubit = key.split("|")
    assert np.array_equal(readout_matrix(PROFILES[file].table.qubit(int(qubit))), BASELINE[key])


def test_aliases_are_gone() -> None:
    manila = PROFILES[MANILA_V01.name]
    assert isinstance(manila.table.gate("h", (0,)), Unavailable)


def test_loading_0_1_warns() -> None:
    with pytest.warns(MigrationWarning, match="0.1"):
        load_file(MANILA_V01)


def test_migrated_manila_fields() -> None:
    manila = PROFILES[MANILA_V01.name]
    assert manila.id == "ibm_manila"
    assert manila.device.calibrated_at.isoformat() == "2024-05-27T18:27:23+00:00"
    assert manila.provenance.retrieved_at.isoformat() == "2024-05-27T18:27:23+00:00"
    assert manila.gates["rz"].virtual
    assert manila.table.gate("reset", (0,)).state == "uncalibrated"
    assert manila.idle.t2_kind == "echo"
    assert manila.provenance.redistributable == "yes"
    assert manila.provenance.license == "Apache-2.0"
    assert manila.provenance.extra["migrated_from"] == "0.1"
    assert manila.connectivity.directed


def _v01(**changes) -> dict:
    data = json.loads(MANILA_V01.read_text())
    data.update(changes)
    return data


def _upgrade(data: dict) -> Profile:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MigrationWarning)
        return Profile.from_dict(data)


def test_ibm_sentinel_error_disables_the_gate() -> None:
    data = _v01()
    for gate in data["gates"]:
        if gate["name"] == "cx" and gate["qubits"] == [0, 1]:
            gate["error"] = 1.0
        if gate["name"] == "sx" and gate["qubits"] == [3]:
            gate["operational"] = False
    table = _upgrade(data).table
    assert table.gate("cx", (0, 1)).state == "disabled"
    assert table.gate("sx", (3,)).state == "disabled"
    assert table.gate("cx", (1, 0)).state == "calibrated"


def _small_v01() -> dict:
    """A 2-qubit IBM file in format 0.1, small enough to check each value by hand."""
    return {
        "schema_version": "0.1",
        "provider": "ibm",
        "backend_name": "ibm_small",
        "num_qubits": 2,
        "captured_at": "2024-01-02T00:00:00Z",
        "basis_gates": ["sx", "cx"],
        "coupling_map": [[0, 1]],
        "gates": [
            {"name": "sx", "qubits": [0], "error": 1e-3, "duration_ns": 32.0},
            {"name": "sx", "qubits": [1], "error": 1e-3, "duration_ns": 32.0},
            {"name": "cx", "qubits": [0, 1], "error": 1e-2, "duration_ns": 300.0},
        ],
        "qubits": [
            {"index": 0, "t1_us": 100.0, "t2_us": 80.0},
            {"index": 1, "t1_us": 100.0, "t2_us": 80.0},
        ],
        "provenance": {"source_timestamp": "2024-01-01T00:00:00Z", "notes": []},
    }


def _small_gate(data: dict, name: str, qubits: list[int]) -> dict:
    return next(g for g in data["gates"] if g["name"] == name and g["qubits"] == qubits)


# The largest average infidelity of a channel is d/(d+1): 2/3 for 1 qubit, 4/5 for 2 qubits.
@pytest.mark.parametrize(
    ("name", "qubits", "error", "state"),
    [
        ("sx", [0], 0.6, "calibrated"),
        ("sx", [0], 2 / 3, "disabled"),
        ("sx", [0], 0.7, "disabled"),
        ("cx", [0, 1], 0.75, "calibrated"),
        ("cx", [0, 1], 0.8, "disabled"),
        ("cx", [0, 1], 0.85, "disabled"),
        ("cx", [0, 1], 2.0, "disabled"),
    ],
)
def test_an_error_at_the_largest_infidelity_of_its_arity_disables_the_gate(
    name: str, qubits: list[int], error: float, state: str
) -> None:
    data = _small_v01()
    _small_gate(data, name, qubits)["error"] = error
    found = _upgrade(data).table.gate(name, tuple(qubits))
    assert found.state == state
    if state == "calibrated":
        assert found.avg_infidelity == error


def test_a_coherence_time_of_0_or_less_counts_as_missing_with_a_note() -> None:
    data = _small_v01()
    data["qubits"][1].update(t1_us=0, t2_us=-1.5)
    profile = _upgrade(data)
    assert profile.table.qubit(1).t1_ns is None
    assert profile.table.qubit(1).t2_ns is None
    assert profile.table.qubit(0).t1_ns == 100_000
    assert [n for n in profile.provenance.notes if n.startswith("Qubit 1")] == [
        "Qubit 1 reported T1 = 0 us. NoiseVault treats the value as missing.",
        "Qubit 1 reported T2 = -1.5 us. NoiseVault treats the value as missing.",
    ]


def test_gate_metadata_is_kept_by_gate_and_qubits() -> None:
    data = _small_v01()
    _small_gate(data, "sx", [0])["metadata"] = {"rb_stderr": 1e-5}
    _small_gate(data, "sx", [1])["metadata"] = {}
    _small_gate(data, "cx", [0, 1])["metadata"] = {"date": "2024-01-01", "runs": [1, 2]}
    profile = _upgrade(data)
    assert profile.extensions["v01_gate_metadata"] == {
        "sx:0": {"rb_stderr": 1e-5},
        "cx:0-1": {"date": "2024-01-01", "runs": (1, 2)},
    }
    assert not [n for n in profile.provenance.notes if "metadata" in n]
    assert Profile.from_dict(profile.to_dict()).extensions == profile.extensions


def test_gate_metadata_that_format_1_0_cannot_hold_is_named_in_the_notes() -> None:
    data = _small_v01()
    _small_gate(data, "sx", [0])["metadata"] = {"fit": [0.5, float("nan")], "rb_stderr": 1e-5}
    _small_gate(data, "cx", [0, 1])["metadata"] = {"bad": "\ud800", "\ud800": 1}
    profile = _upgrade(data)
    assert profile.extensions["v01_gate_metadata"] == {"sx:0": {"rb_stderr": 1e-5}}
    surrogate = (
        "holds the unpaired surrogate \\ud800, which UTF-8 cannot encode. Remove the surrogate or"
        " write the whole character"
    )
    assert [n for n in profile.provenance.notes if "metadata" in n] == [
        'Not converted: 0.1 gate metadata "fit" of sx:0 (fit[1]: nan is not a finite number).',
        f'Not converted: 0.1 gate metadata "bad" of cx:0-1 (bad: the string {surrogate}).',
        f"Not converted: 0.1 gate metadata \"\\ud800\" of cx:0-1 (the key '\\ud800' {surrogate}).",
    ]


def test_the_capture_time_is_the_retrieval_time_not_the_calibration_time() -> None:
    profile = _upgrade(_small_v01())
    assert profile.device.calibrated_at.isoformat() == "2024-01-01T00:00:00+00:00"
    assert profile.provenance.retrieved_at.isoformat() == "2024-01-02T00:00:00+00:00"


def test_a_file_with_no_source_timestamp_is_undated() -> None:
    data = _small_v01()
    del data["provenance"]["source_timestamp"]
    profile = _upgrade(data)
    assert profile.device.calibrated_at is None
    assert profile.provenance.retrieved_at.isoformat() == "2024-01-02T00:00:00+00:00"
    assert repr(profile).startswith("<Profile ibm_small@undated ")


def test_rz_stays_calibrated_when_it_has_an_error() -> None:
    data = _v01()
    for gate in data["gates"]:
        if gate["name"] == "rz" and gate["qubits"] == [2]:
            gate["error"] = 1e-4
    profile = _upgrade(data)
    assert not profile.gates["rz"].virtual
    assert profile.table.gate("rz", (2,)).avg_infidelity == 1e-4


@pytest.mark.parametrize("dead", [{"operational": False}, {"error": 1.0}])
def test_disabled_rz_stays_disabled_when_rz_becomes_virtual(dead: dict) -> None:
    data = _v01()
    for gate in data["gates"]:
        if gate["name"] == "rz" and gate["qubits"] == [0]:
            gate.update(dead)
    profile = _upgrade(data)
    assert profile.gates["rz"].virtual
    assert profile.table.gate("rz", (0,)).state == "disabled"
    assert profile.table.gate("rz", (1,)).state == "ideal"


def test_missing_values_stay_missing() -> None:
    data = _v01()
    data["qubits"][1].update(t2_us=None, prob_meas0_prep1=None, readout_error=0.03)
    data["qubits"][2].update(prob_meas0_prep1=None, prob_meas1_prep0=None, readout_error=None)
    table = _upgrade(data).table
    assert table.qubit(1).t2_ns is None
    assert table.qubit(1).readout == (0.03, 0.03)
    assert table.qubit(2).readout is None


LEGACY_FLAGS = [
    ("false", "disabled"),
    ("no", "disabled"),
    ("off", "disabled"),
    ("0", "disabled"),
    (0, "disabled"),
    (0.0, "disabled"),
    ("true", "calibrated"),
    ("yes", "calibrated"),
    (1, "calibrated"),
]


@pytest.mark.parametrize(("flag", "state"), LEGACY_FLAGS, ids=repr)
def test_a_gate_flag_is_read_as_0_1_read_it(flag: object, state: str) -> None:
    data = _v01()
    for gate in data["gates"]:
        if gate["name"] == "sx" and gate["qubits"] == [0]:
            gate["operational"] = flag
    assert _upgrade(data).table.gate("sx", (0,)).state == state


@pytest.mark.parametrize(("flag", "state"), LEGACY_FLAGS, ids=repr)
def test_a_qubit_flag_is_read_as_0_1_read_it(flag: object, state: str) -> None:
    data = _v01()
    data["qubits"][0]["operational"] = flag
    assert _upgrade(data).table.qubit(0).disabled is (state == "disabled")


def test_a_zero_frequency_is_kept() -> None:
    profile = _upgrade(_first_qubit(frequency_ghz=0))
    assert profile.extensions["v01_qubit_frequency_ghz"]["0"] == 0


def test_a_capture_time_without_a_timezone_gives_a_note_and_no_retrieval_time() -> None:
    profile = _upgrade(_v01(captured_at="2024-05-27T15:27:23"))
    assert profile.provenance.retrieved_at is None
    assert (
        'Not converted: captured_at "2024-05-27T15:27:23", which is not an ISO 8601 time with a'
        " timezone." in profile.provenance.notes
    )
    assert profile.fingerprint == _upgrade(_v01()).fingerprint


def test_an_empty_source_timestamp_is_not_replaced_by_the_capture_time() -> None:
    with pytest.raises(ValueError, match="calibrated_at"):
        _upgrade(_prov(source_timestamp=""))


def _first_gate(**changes) -> dict:
    data = _v01()
    data["gates"][0].update(changes)
    return data


def _first_qubit(**changes) -> dict:
    data = _v01()
    data["qubits"][0].update(changes)
    return data


def _prov(**changes) -> dict:
    data = _v01()
    data["provenance"].update(changes)
    return data


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"schema_version": "0.1"}, "provider is missing"),
        (_v01(gates=None), "gates should be a list, not null"),
        (_v01(provenance=[]), "provenance should be an object, not a list"),
        (_v01(qubits=[{"t1_us": 1.0}]), "qubits[0] needs an integer index"),
        (_v01(gates=[{"name": "sx"}]), "gates[0] needs a name and a list of qubits"),
        (_first_gate(error="high"), "gates[0]: error should be a number"),
        (_first_gate(error=True), "gates[0]: error should be a number, not true or false"),
        (
            _first_gate(error=float("inf")),
            "gates[0]: error should be a finite number, not Infinity",
        ),
        (_first_gate(error=float("nan")), "gates[0]: error should be a finite number, not NaN"),
        (_first_gate(metadata=[1]), "gates[0]: metadata should be an object, not a list"),
        (_v01(provenance={"raw_hash": ["x"]}), "provenance.raw_hash should be a string"),
        (
            _first_gate(operational="maybe"),
            'gates[0]: operational should be true or false, not "maybe"',
        ),
        (_first_gate(operational=None), "gates[0]: operational should be true or false, not null"),
        (_first_qubit(operational=2), "qubits[0]: operational should be true or false, not 2"),
        (
            _first_qubit(operational=[]),
            "qubits[0]: operational should be true or false, not a list",
        ),
        (
            _first_qubit(frequency_ghz="4.9"),
            "qubits[0]: frequency_ghz should be a number, not a string",
        ),
        (_first_qubit(metadata="x"), "qubits[0]: metadata should be an object, not a string"),
        (
            _prov(harvester_version=1),
            "provenance.harvester_version should be a string, not a number",
        ),
        (
            _prov(source_timestamp=123),
            "provenance.source_timestamp should be a string, not a number",
        ),
        (_v01(captured_at=123), "captured_at should be a string, not a number"),
    ],
    ids=[
        "empty",
        "null-gates",
        "list-provenance",
        "qubit-index",
        "gate-qubits",
        "gate-error",
        "gate-error-bool",
        "gate-error-infinite",
        "gate-error-nan",
        "gate-metadata",
        "raw-hash",
        "gate-flag-word",
        "gate-flag-null",
        "qubit-flag-number",
        "qubit-flag-list",
        "frequency",
        "metadata",
        "harvester-version",
        "source-timestamp",
        "captured-at",
    ],
)
def test_a_malformed_0_1_file_is_refused_with_the_field_to_fix(data: dict, message: str) -> None:
    with pytest.raises(
        ValueError, match=re.escape(f"not a valid NoiseVault 0.1 file: {message}")
    ) as info:
        _upgrade(data)
    assert isinstance(info.value, NoiseVaultError)
    assert info.value.hint == "fix that field or pull the device again"


def test_nv_show_prints_the_fix_for_a_malformed_0_1_file_on_a_hint_line(tmp_path: Path) -> None:
    data = _v01()
    del data["coupling_map"]
    path = tmp_path / "old.json"
    path.write_text(json.dumps(data))
    result = CliRunner().invoke(app, ["show", str(path)])
    assert result.stderr.splitlines() == [
        f"error: {path}: not a valid NoiseVault 0.1 file: coupling_map is missing",
        "hint: fix that field or pull the device again",
    ]
    assert result.exit_code == 1


@pytest.mark.parametrize(
    "missing", [{"error": None, "duration_ns": None}, {"error": None}], ids=["both", "error"]
)
def test_rz_with_missing_calibration_stays_uncalibrated(missing: dict) -> None:
    data = _v01()
    for gate in data["gates"]:
        if gate["name"] == "rz":
            gate.update(missing)
    table = _upgrade(data).table
    assert not table.profile.gates["rz"].virtual
    assert table.gate("rz", (0,)).state == "uncalibrated"


def test_rz_without_records_stays_uncalibrated() -> None:
    data = _v01()
    data["gates"] = [g for g in data["gates"] if g["name"] != "rz"]
    profile = _upgrade(data)
    assert "rz" in data["basis_gates"] and not profile.gates["rz"].virtual
    assert profile.table.gate("rz", (0,)).state == "uncalibrated"


@pytest.mark.parametrize("raw_hash", ["sha256:" + "0" * 64, None], ids=["other-data", "no-hash"])
def test_an_unverified_fake_snapshot_claims_no_license(raw_hash: str | None) -> None:
    data = _v01()
    data["provenance"]["raw_hash"] = raw_hash
    prov = _upgrade(data).provenance
    assert (prov.license, prov.attribution, prov.redistributable) == (
        None,
        "IBM Quantum",
        "unknown",
    )
    assert prov.source_kind != "package_snapshot"
    assert any("did not prove" in note for note in prov.notes)


def test_the_verified_hash_is_the_0_1_payload_of_the_shipped_fake_manila() -> None:
    runtime = require("qiskit_ibm_runtime")
    if runtime.__version__ != "0.49.0":
        pytest.skip(f"the hash was taken from qiskit-ibm-runtime 0.49.0, not {runtime.__version__}")
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2

    backend = FakeManilaV2()
    raw = {
        "configuration": backend.configuration().to_dict(),
        "properties": backend.properties().to_dict(),
    }
    text = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    payload_hash = "sha256:" + hashlib.sha256(text.encode()).hexdigest()
    assert payload_hash == json.loads(MANILA_V01.read_text())["provenance"]["raw_hash"]
    assert PROFILES[MANILA_V01.name].provenance.redistributable == "yes"
