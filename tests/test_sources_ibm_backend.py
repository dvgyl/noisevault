from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import FAKES_ADDED_IN, needs_runtime, require

import noisevault as nv
from noisevault.reference import Op, probabilities
from noisevault.sources import Origin, qiskit_backend
from noisevault.sources.qiskit_backend import BUNDLED_FAKES
from noisevault.table import GateNoise, Unavailable

FIXTURES = Path(__file__).parent / "fixtures" / "ibm"
ROUND_TRIP = ("FakeFez", "FakeSherbrooke", "FakeManilaV2")


def _backend(class_name: str):
    fake_provider = require("qiskit_ibm_runtime.fake_provider")
    if class_name in FAKES_ADDED_IN:
        needs_runtime(FAKES_ADDED_IN[class_name])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return getattr(fake_provider, class_name)()


def _profile(class_name: str) -> tuple[object, nv.Profile]:
    backend = _backend(class_name)
    return backend, nv.from_qiskit_backend(backend)


@pytest.mark.parametrize("class_name", ROUND_TRIP)
def test_every_target_entry_round_trips(class_name: str) -> None:
    backend, profile = _profile(class_name)
    target, table = backend.target, profile.table
    checked = 0
    for name in ("id", "x", "sx", "cz", "ecr", "cx"):
        for qargs, props in target[name].items() if name in target else []:
            found = table.gate(name, qargs)
            assert isinstance(found, GateNoise), (name, qargs, found)
            if props.error >= 1:
                assert found.state == "disabled"
                continue
            assert found.avg_infidelity == props.error
            assert found.duration_ns == pytest.approx(props.duration * 1e9, rel=1e-11)
            checked += 1
    assert checked > 5 * backend.num_qubits // 2
    for q, props in enumerate(target.qubit_properties):
        noise = table.qubit(q)
        assert noise.t1_ns == pytest.approx(props.t1 * 1e9, rel=1e-11)
        assert noise.t2_ns == pytest.approx(props.t2 * 1e9, rel=1e-11)
        measure = target["measure"][(q,)]
        assert profile.qubits[q].readout.duration_ns == pytest.approx(measure.duration * 1e9)
    assert profile.gates["rz"].virtual


@pytest.mark.parametrize("class_name", ROUND_TRIP)
def test_asymmetric_readout_comes_from_properties(class_name: str) -> None:
    backend, profile = _profile(class_name)
    props = backend.properties()
    asymmetric = 0
    for q in range(backend.num_qubits):
        a, b = (
            props.qubit_property(q)["prob_meas1_prep0"][0],
            props.qubit_property(q)["prob_meas0_prep1"][0],
        )
        assert profile.table.qubit(q).readout == (a, b)
        asymmetric += a != b
    assert asymmetric > backend.num_qubits // 2


def test_calibrated_at_is_the_snapshot_time_in_utc() -> None:
    assert _profile("FakeFez")[1].device.calibrated_at.isoformat() == "2025-02-26T20:16:25+00:00"
    manila = _profile("FakeManilaV2")[1]
    assert manila.device.calibrated_at.isoformat() == "2024-05-27T18:27:23+00:00"


def test_fez_dead_cz_entries_become_disabled() -> None:
    backend, profile = _profile("FakeFez")
    dead = {q for q, p in backend.target["cz"].items() if p.error >= 1}
    assert len(dead) == 14
    disabled = {
        (a, b)
        for a, b in backend.target["cz"]
        if profile.table.gate("cz", (a, b)).state == "disabled"
    }
    assert disabled == dead


def test_heron_cz_is_symmetric_and_eagle_ecr_is_directed() -> None:
    backend, fez = _profile("FakeFez")
    assert not fez.connectivity.directed
    assert len(fez.connectivity.edges) == len(backend.target["cz"]) // 2
    stored = sum(1 for r in fez.calibrations if r.gate == "cz")
    assert stored == len(backend.target["cz"]) // 2  # both directions agree, so one record
    for (a, b), props in backend.target["cz"].items():
        if props.error < 1:
            assert fez.table.gate("cz", (b, a)).avg_infidelity == props.error

    backend, sherbrooke = _profile("FakeSherbrooke")
    assert sherbrooke.connectivity.directed
    pairs = set(backend.target["ecr"])
    a, b = next(iter(pairs))
    assert (b, a) not in pairs
    assert isinstance(sherbrooke.table.gate("ecr", (b, a)), Unavailable)


def test_ibm_qualifiers_and_device_defaults() -> None:
    _, profile = _profile("FakeFez")
    cz, sx = profile.gates["cz"], profile.gates["sx"]
    assert (cz.method, cz.measured, cz.includes, cz.statistic) == (
        "rb",
        "isolated",
        ("1q_dressing",),
        "median",
    )
    assert (sx.method, sx.measured) == ("rb", "simultaneous")
    working = [r.avg_infidelity for r in profile.calibrations if r.gate == "cz" and not r.disabled]
    assert cz.avg_infidelity == pytest.approx(float(np.median(working)))
    assert {r.statistic for r in profile.calibrations if r.gate == "cz" and not r.disabled} == {
        "individual"
    }
    assert profile.device.processor == "Heron r2" and profile.idle.t2_kind == "echo"
    assert profile.gates["reset"].duration_ns == 1584


def test_fake_backend_provenance() -> None:
    qiskit_ibm_runtime = require("qiskit_ibm_runtime")

    _, profile = _profile("FakeFez")
    prov = profile.provenance
    assert profile.id == "ibm_fez" and profile.device.vendor == "ibm"
    assert prov.source == f"qiskit-ibm-runtime {qiskit_ibm_runtime.__version__} FakeFez"
    assert (prov.data_kind, prov.source_kind, prov.license, prov.redistributable) == (
        "measured",
        "package_snapshot",
        "Apache-2.0",
        "yes",
    )
    assert prov.attribution == "IBM Quantum, via qiskit-ibm-runtime"
    assert prov.source_url == "https://github.com/Qiskit/qiskit-ibm-runtime"


def _refresh(backend, *, persist: bool) -> None:
    """Run the SDK's own refresh() against an IBM Quantum account serving a newer calibration."""
    runtime = require("qiskit_ibm_runtime")
    needs_runtime("0.49")  # refresh(persist=...)
    from qiskit_ibm_runtime.utils.backend_decoder import properties_from_server_data

    folder = Path(backend.dirname)
    props = json.loads((folder / backend.props_filename).read_text(encoding="utf-8"))
    props["last_update_date"] = "2026-09-01T00:00:00+00:00"
    [t1] = [p for p in props["qubits"][0] if p["name"] == "T1"]
    t1["value"] = 123.0
    conf = (folder / backend.conf_filename).read_text(encoding="utf-8")
    device = SimpleNamespace(
        properties=lambda refresh=False: properties_from_server_data(copy.deepcopy(props))
    )
    service = runtime.QiskitRuntimeService.__new__(runtime.QiskitRuntimeService)
    service.backends = lambda name, **_: [device]
    service._get_api_client = lambda: SimpleNamespace(
        backend_configuration=lambda name, refresh=False: json.loads(conf)
    )
    backend.refresh(service, persist=persist)


def _assert_account_data(profile: nv.Profile, shipped_props: Path) -> None:
    assert profile.device.calibrated_at.isoformat() == "2026-09-01T00:00:00+00:00"
    assert profile.qubits[0].t1_us == 123.0
    prov = profile.provenance
    assert (prov.source_kind, prov.license, prov.redistributable, prov.attribution) == (
        "account_api",
        None,
        "unknown",
        "IBM Quantum",
    )
    assert prov.source_hash not in (None, "sha256:" + _sha256(shipped_props.read_bytes()))


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def test_refreshed_fake_in_memory_is_account_data_not_the_package_snapshot() -> None:
    backend = _backend("FakeManilaV2")
    shipped = Path(backend.dirname) / backend.props_filename
    _refresh(backend, persist=False)
    _assert_account_data(nv.from_qiskit_backend(backend), shipped)


def test_refreshed_fake_on_disk_is_account_data_not_the_package_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_provider = require("qiskit_ibm_runtime.fake_provider")
    installed = Path(fake_provider.FakeManilaV2.dirname)
    shipped = installed / fake_provider.FakeManilaV2.props_filename
    copied = tmp_path / installed.name
    shutil.copytree(installed, copied)
    # persist=True overwrites the package's own files; the copy keeps the install untouched.
    monkeypatch.setattr(fake_provider.FakeManilaV2, "dirname", str(copied))
    backend = _backend("FakeManilaV2")
    _refresh(backend, persist=True)
    _assert_account_data(nv.from_qiskit_backend(backend), shipped)
    _assert_account_data(nv.from_qiskit_backend(_backend("FakeManilaV2")), shipped)


def test_backend_without_properties_uses_the_symmetric_measure_error() -> None:
    require("qiskit")
    from qiskit.providers.fake_provider import GenericBackendV2

    backend = GenericBackendV2(num_qubits=3, seed=7)
    profile = nv.from_qiskit_backend(backend)
    error = backend.target["measure"][(1,)].error
    assert profile.table.qubit(1).readout == (error, error)
    assert profile.device.vendor is None and profile.device.calibrated_at is None


@pytest.mark.parametrize("class_name", ("FakeFez", "FakeSherbrooke", "FakeKingston", "FakeTorino"))
def test_matches_aer_from_backend_noise_model(class_name: str) -> None:
    """The profile reproduces Aer's own NoiseModel.from_backend, applied exactly.

    The test evolves the Aer model with qiskit.quantum_info, not with AerSimulator. The
    density-matrix execution of AerSimulator departs from its model by about 1e-9.
    """
    require("qiskit_aer")
    from qiskit.quantum_info import DensityMatrix, Operator, SuperOp
    from qiskit_aer.noise import NoiseModel

    backend, profile = _profile(class_name)
    two = "cz" if "cz" in backend.target else "ecr"
    live = sorted(q for q, p in backend.target[two].items() if p.error < 1)
    chain = next((a, b, d) for a, b in live for (c, d) in live if c == b and d != a)
    ops = [Op("sx", (0,)), Op("rz", (0,), (0.3,)), Op("x", (1,)), Op("sx", (1,))]
    ops += [Op(two, (0, 1)), Op("sx", (2,)), Op(two, (1, 2)), Op("id", (2,))]
    ours = probabilities(profile, ops, 3, layout=chain, unknown_gates="error", readout=False)

    noise = NoiseModel.from_backend(backend, readout_error=False)
    rho = DensityMatrix.from_label("000")
    for op in ops:
        gate = backend.target.operation_from_name(op.name)
        gate = gate.__class__(*op.params) if op.params else gate
        rho = rho.evolve(Operator(gate), qargs=list(op.qubits))
        error = noise._local_quantum_errors.get(op.name, {}).get(tuple(chain[q] for q in op.qubits))
        if error is not None:
            rho = rho.evolve(SuperOp(error), qargs=list(op.qubits))
    aer = rho.probabilities().reshape((2,) * 3).transpose(2, 1, 0).reshape(8)  # little-endian
    assert 0.5 * np.abs(ours - aer).sum() < 1e-12


@pytest.mark.parametrize(
    ("class_name", "caveat"),
    [
        ("FakeFractionalBackend", "modeled based on FakeLima"),
        ("FakeNighthawk", "not intended to represent typical Nighthawk error values"),
    ],
)
def test_a_modeled_fake_is_a_vendor_model_with_the_package_caveat(
    class_name: str, caveat: str
) -> None:
    fake_provider = require("qiskit_ibm_runtime.fake_provider")
    if not hasattr(fake_provider, class_name):
        pytest.skip(f"this qiskit-ibm-runtime has no {class_name}")
    prov = _profile(class_name)[1].provenance
    assert prov.data_kind == "vendor_model"
    assert any(caveat in note for note in prov.notes), prov.notes


def test_a_shipped_snapshot_with_a_key_twice_is_refused() -> None:
    backend = _backend("FakeManilaV2")
    shipped = b'{"backend_name": "fake_manila", "backend_name": "ibmq_manila"}'
    with pytest.raises(nv.SourceDataError) as caught:
        qiskit_backend._model_caveat(backend, shipped, "0.49.0")
    assert caught.value.message == f"{backend.props_filename} has the key backend_name twice"


def test_a_device_snapshot_is_measured_without_a_model_caveat() -> None:
    prov = _profile("FakeManilaV2")[1].provenance
    assert prov.data_kind == "measured"
    assert not any("model" in note for note in prov.notes), prov.notes


def test_bundle_curation() -> None:
    assert "FakeNighthawk" not in BUNDLED_FAKES and "FakeFractionalBackend" not in BUNDLED_FAKES
    assert "FakeManilaV2" in BUNDLED_FAKES and len(BUNDLED_FAKES) == 18


def test_gate_marked_not_operational_is_disabled_despite_a_normal_error() -> None:
    from noisevault.sources.qiskit_backend import calibration_from_properties, to_profile

    props = json.loads((FIXTURES / "manila_properties.json").read_bytes())
    [entry] = [e for e in props["gates"] if e["gate"] == "cx" and e["qubits"] == [3, 4]]
    entry["parameters"].append({"name": "operational", "unit": "", "value": 0})
    origin = Origin("manila_properties.json")
    profile = to_profile(
        calibration_from_properties(props, origin=origin), {"source_kind": "other"}, origin=origin
    )
    assert profile.table.gate("cx", (3, 4)).state == "disabled"
    assert profile.table.gate("cx", (4, 3)).state == "calibrated"


def test_an_ideal_gate_outside_the_registry_keeps_its_arity() -> None:
    from noisevault.sources.qiskit_backend import calibration_from_properties, to_profile

    props = json.loads((FIXTURES / "manila_properties.json").read_bytes())
    zero = [{"name": "gate_error", "value": 0}, {"name": "gate_length", "value": 0, "unit": "ns"}]
    props["gates"] += [{"gate": "phase", "qubits": [q], "parameters": zero} for q in range(5)]
    origin = Origin("manila_properties.json")
    profile = to_profile(
        calibration_from_properties(props, origin=origin), {"source_kind": "other"}, origin=origin
    )
    assert (profile.gates["phase"].qubits, profile.gates["phase"].virtual) == (1, True)
    assert profile.table.gate("phase", (3,)).state == "ideal"


def _with_units(gate_error: dict, prob_meas1_prep0: dict) -> nv.Profile:
    from noisevault.sources.qiskit_backend import calibration_from_properties, to_profile

    props = json.loads((FIXTURES / "manila_properties.json").read_bytes())
    [entry] = [e for e in props["gates"] if e["gate"] == "sx" and e["qubits"] == [0]]
    next(p for p in entry["parameters"] if p["name"] == "gate_error").update(gate_error)
    next(p for p in props["qubits"][0] if p["name"] == "prob_meas1_prep0").update(prob_meas1_prep0)
    origin = Origin("manila_properties.json")
    return to_profile(
        calibration_from_properties(props, origin=origin), {"source_kind": "other"}, origin=origin
    )


def test_an_error_in_percent_becomes_a_fraction() -> None:
    profile = _with_units({"value": 0.5, "unit": "%"}, {"value": 2, "unit": "%"})
    assert profile.table.gate("sx", (0,)).avg_infidelity == 0.005
    assert profile.table.qubit(0).readout[0] == 0.02


def test_an_error_with_no_unit_keeps_its_value() -> None:
    profile = _with_units({"value": 0.005, "unit": None}, {"value": 0.02})
    assert profile.table.gate("sx", (0,)).avg_infidelity == 0.005
    assert profile.table.qubit(0).readout[0] == 0.02


def test_an_error_in_an_unknown_unit_is_refused() -> None:
    with pytest.raises(nv.SourceDataError) as info:
        _with_units({"value": 500, "unit": "ppm"}, {})
    assert info.value.message == (
        "manila_properties.json: gate_error of sx on qubit 0 has the unknown unit 'ppm', not %"
        " or no unit"
    )


def test_a_parameter_with_no_name_is_refused() -> None:
    from noisevault.sources.qiskit_backend import calibration_from_properties

    props = json.loads((FIXTURES / "manila_properties.json").read_bytes())
    del props["qubits"][0][0]["name"]
    with pytest.raises(nv.SourceDataError) as info:
        calibration_from_properties(props, origin=Origin("manila_properties.json"))
    assert info.value.message == (
        "manila_properties.json: a parameter of qubit 0 has the name None, not a string"
    )


def test_aer_simulator_from_a_fake_models_that_device() -> None:
    aer = require("qiskit_aer")
    profile = nv.from_qiskit_backend(aer.AerSimulator.from_backend(_backend("FakeManilaV2")))
    assert profile.id == "ibm_manila" and profile.device.vendor == "ibm"
    assert profile.provenance.source_kind == "other"  # not a live IBM pull
    assert profile.fingerprint != nv.load("ibm_manila").fingerprint  # Aer drops the readout pair


def test_backend_with_no_qubit_count_says_what_to_pass() -> None:
    require("qiskit")
    from qiskit.providers.basic_provider import BasicSimulator

    with pytest.raises(nv.SourceDataError) as info:
        nv.from_qiskit_backend(BasicSimulator())
    assert info.value.message == (
        "basic_simulator has no fixed qubit count, so it has no device calibration"
    )
    assert info.value.hint == (
        "pass a device backend, such as a qiskit-ibm-runtime fake or a live IBM backend"
    )


def test_gates_the_target_drops_as_non_operational_stay_disabled() -> None:
    backend = _backend("FakeManilaV2")
    props = backend.properties().to_dict()
    for entry in props["gates"]:
        if (entry["gate"], entry["qubits"]) in (("sx", [3]), ("cx", [3, 4])):
            stamp = entry["parameters"][0]["date"]
            entry["parameters"].append(
                {"name": "operational", "unit": "", "value": 0, "date": stamp}
            )
    backend._props_dict = props
    assert (3,) not in backend.target["sx"] and (3, 4) not in backend.target["cx"]
    profile = nv.from_qiskit_backend(backend)
    assert profile.table.gate("sx", (3,)).state == "disabled"
    assert profile.table.gate("cx", (3, 4)).state == "disabled"
    assert profile.table.gate("cx", (4, 3)).state == "calibrated"


def test_virtual_gate_the_target_drops_as_non_operational_stays_disabled() -> None:
    backend = _backend("FakeManilaV2")
    props = backend.properties().to_dict()
    [entry] = [e for e in props["gates"] if e["gate"] == "rz" and e["qubits"] == [3]]
    stamp = entry["parameters"][0]["date"]
    entry["parameters"].append({"name": "operational", "unit": "", "value": 0, "date": stamp})
    backend._props_dict = props
    assert (3,) not in backend.target["rz"] and (3,) in backend.target["sx"]
    profile = nv.from_qiskit_backend(backend)
    assert profile.gates["rz"].virtual
    assert profile.table.gate("rz", (3,)).state == "disabled"
    assert profile.table.gate("rz", (2,)).state == "ideal"
    assert not profile.table.qubit(3).disabled


def test_a_gate_the_properties_omit_on_a_configured_qubit_takes_the_median() -> None:
    backend = _backend("FakeManilaV2")
    props = backend.properties().to_dict()
    props["gates"] = [e for e in props["gates"] if (e["gate"], e["qubits"]) != ("sx", [0])]
    backend._props_dict = props
    assert backend.target["sx"][(0,)] is None
    errors = [
        p["value"]
        for e in props["gates"]
        if e["gate"] == "sx"
        for p in e["parameters"]
        if p["name"] == "gate_error"
    ]
    profile = nv.from_qiskit_backend(backend)
    sx = profile.table.gate("sx", (0,))
    assert (sx.state, sx.avg_infidelity) == ("calibrated", statistics.median(errors))
    note = "Qubits [0] have no sx error; the device median applies to them."
    assert note in profile.provenance.notes


def test_each_pair_of_a_mixed_cx_and_ecr_device_allows_only_its_own_gate() -> None:
    backend, profile = _profile("FakeCairoV2")
    target = backend.target
    pairs = {qargs for name in ("cx", "ecr") for qargs in target[name]}
    assert target.instruction_supported("ecr", (1, 2))
    assert not target.instruction_supported("cx", (1, 2))
    for name in ("cx", "ecr"):
        for a, b in pairs:
            for qubits in ((a, b), (b, a)):
                props = target[name].get(qubits)
                supported = props is not None and props.error < 1  # 1 is IBM's dead gate
                assert profile.table.allowed(name, qubits) == supported, (name, qubits)


def test_ibm_backends_are_superconducting() -> None:
    _, profile = _profile("FakeManilaV2")
    assert profile.device.technology == "superconducting"
    aer = require("qiskit_aer")
    wrapped = nv.from_qiskit_backend(aer.AerSimulator.from_backend(_backend("FakeManilaV2")))
    assert wrapped.device.technology == "superconducting"


def _ibm_backend(served: list[dict]):
    """A real IBMBackend whose API client serves the last calibration in `served`."""
    runtime = require("qiskit_ibm_runtime")
    from qiskit_ibm_runtime.utils.backend_decoder import configuration_from_server_data

    fake = _backend("FakeManilaV2")
    conf = json.loads((Path(fake.dirname) / fake.conf_filename).read_text(encoding="utf-8"))
    client = SimpleNamespace(backend_properties=lambda name, **_: copy.deepcopy(served[-1]))
    return runtime.IBMBackend(configuration_from_server_data(conf), None, client)


def test_refreshed_ibm_backend_properties_give_one_consistent_snapshot() -> None:
    old = json.loads((FIXTURES / "manila_properties.json").read_bytes())
    new = copy.deepcopy(old)
    new["last_update_date"] = "2026-09-30T00:00:00+00:00"
    for entry in new["gates"]:
        if (entry["gate"], entry["qubits"]) == ("sx", [0]):
            [error] = [p for p in entry["parameters"] if p["name"] == "gate_error"]
            error["value"] = 0.01
        if (entry["gate"], entry["qubits"]) == ("cx", [3, 4]):
            stamp = entry["parameters"][0]["date"]
            entry["parameters"].append(
                {"name": "operational", "unit": "", "value": 0, "date": stamp}
            )
    served = [old]
    backend = _ibm_backend(served)
    assert backend.target["sx"][(0,)].error == pytest.approx(0.000155, rel=1e-3)
    served.append(new)
    backend.properties(refresh=True)

    profile = nv.from_qiskit_backend(backend)
    assert profile.device.calibrated_at.isoformat() == "2026-09-30T00:00:00+00:00"
    assert profile.table.gate("sx", (0,)).avg_infidelity == 0.01
    assert profile.table.gate("cx", (3, 4)).state == "disabled"
    assert profile.table.gate("cx", (4, 3)).state == "calibrated"
    assert profile.fingerprint == nv.from_qiskit_backend(_ibm_backend([new])).fingerprint


_UNREADABLE_ON_QUBIT_0 = {
    "a T1 in minutes": (
        "T1",
        {"unit": "min"},
        "T1 of qubit 0 has the unknown time unit 'min', not one of ns, us, µs, ms or s",
    ),
    "a readout error above 1": (
        "prob_meas0_prep1",
        {"value": 1.5},
        "readout.p0_given_1 of qubit 0: Input should be less than or equal to 1, got 1.5",
    ),
}


@pytest.mark.parametrize("case", list(_UNREADABLE_ON_QUBIT_0))
def test_a_backend_value_it_cannot_read_names_the_backend_and_the_value(case: str) -> None:
    field, change, problem = _UNREADABLE_ON_QUBIT_0[case]
    props = json.loads((FIXTURES / "manila_properties.json").read_bytes())
    next(p for p in props["qubits"][0] if p["name"] == field).update(change)
    backend = _ibm_backend([props])
    with pytest.raises(nv.SourceDataError) as info:
        nv.from_qiskit_backend(backend)
    assert (info.value.message, info.value.hint) == (f"backend {backend.name}: {problem}", None)


_SAVE_EACH = """
import sys, warnings
from pathlib import Path
import noisevault as nv
from qiskit_ibm_runtime import fake_provider
fixtures, out = Path(sys.argv[1]), Path(sys.argv[2])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    nv.from_qiskit_backend(fake_provider.FakeFez()).save(out / "fez.json")
archive = fixtures / "hf_archive" / "train-00000-of-00001.parquet"
nv.from_calibration_archive(archive, "ibm_fez").save(out / "archive.json")
nv.from_braket(fixtures / "braket" / "iqm_capabilities_v1.json").save(out / "braket.json")
"""


def test_a_profile_saves_the_same_bytes_under_any_hash_seed(tmp_path: Path) -> None:
    require("qiskit_ibm_runtime")
    require("pyarrow")
    saved = []
    for seed in ("1", "2"):
        out = tmp_path / seed
        out.mkdir()
        fixtures = Path(__file__).parent / "fixtures"
        subprocess.run(
            [sys.executable, "-c", _SAVE_EACH, str(fixtures), str(out)],
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        saved.append({path.name: path.read_bytes() for path in sorted(out.iterdir())})
    assert sorted(saved[0]) == ["archive.json", "braket.json", "fez.json"]
    assert saved[0] == saved[1]
