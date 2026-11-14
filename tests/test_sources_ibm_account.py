from __future__ import annotations

import statistics
import sys
import warnings
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import require

import noisevault as nv
from noisevault.errors import install_hint
from noisevault.sources import ibm_account


class _Backend:
    """The parts of an IBMBackend the account source uses, backed by FakeManilaV2's data."""

    name = "ibm_manila"
    processor_type = {"family": "Falcon", "revision": "5.11"}

    def __init__(self, calls: list[Any]) -> None:
        fake_provider = require("qiskit_ibm_runtime.fake_provider")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fake = fake_provider.FakeManilaV2()
        self._calls = calls

    def properties(self, refresh: bool = False, datetime: datetime | None = None) -> Any:
        self._calls.append(datetime)
        return self._fake.properties()


class _Service:
    def __init__(self, calls: list[Any], **options: Any) -> None:
        calls.append(options)
        self._calls = calls

    def backend(self, name: str) -> _Backend:
        if name != "ibm_manila":
            raise RuntimeError(f"No backend matches the criteria: {name}")
        return _Backend(self._calls)


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    runtime = require("qiskit_ibm_runtime")
    seen: list[Any] = []
    monkeypatch.setattr(runtime, "QiskitRuntimeService", lambda **o: _Service(seen, **o))
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    return seen


def test_pull_reads_the_account_snapshot(calls: list[Any]) -> None:
    profile = ibm_account.pull("ibm_manila", at="2025-06-01")
    assert calls[0] == {"channel": "ibm_quantum_platform"}
    assert calls[1] == datetime(2025, 6, 1, tzinfo=UTC)
    assert profile.fingerprint == nv.load("ibm_manila").fingerprint
    prov = profile.provenance
    assert (prov.source_kind, prov.redistributable) == ("account_api", "unknown")
    assert prov.source_hash.startswith("sha256:") and prov.retrieved_at is not None


def test_a_gate_the_snapshot_omits_on_a_qubit_takes_the_median_and_says_so(
    calls: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = require("qiskit_ibm_runtime")
    props = _Backend(calls)._fake.properties().to_dict()
    props["gates"] = [e for e in props["gates"] if (e["gate"], e["qubits"]) != ("sx", [0])]
    snapshot = runtime.models.BackendProperties.from_dict(props)
    monkeypatch.setattr(_Backend, "properties", lambda self, **_: snapshot)
    errors = [
        p["value"]
        for e in props["gates"]
        if e["gate"] == "sx"
        for p in e["parameters"]
        if p["name"] == "gate_error"
    ]
    assert len(errors) == 4
    profile = ibm_account.pull("ibm_manila")
    sx = profile.table.gate("sx", (0,))
    assert (sx.state, sx.origin, sx.avg_infidelity) == (
        "calibrated",
        "default",
        statistics.median(errors),
    )
    assert profile.provenance.notes == (
        "Qubits [0] have no sx error; the device median applies to them.",
    )


def test_token_from_the_environment(calls: list[Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IBM_QUANTUM_TOKEN", "test-key")
    monkeypatch.setenv("IBM_QUANTUM_INSTANCE", "test-instance")
    ibm_account.pull("ibm_manila")
    assert calls[0] == {
        "channel": "ibm_quantum_platform",
        "token": "test-key",
        "instance": "test-instance",
    }
    assert calls[1] is None


def test_missing_account_explains_the_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = require("qiskit_ibm_runtime")

    missing = runtime.accounts.AccountNotFoundError("Unable to find account.")

    def no_account(**options: Any) -> None:
        raise missing

    monkeypatch.setattr(runtime, "QiskitRuntimeService", no_account)
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    with pytest.raises(nv.SourceUnavailable) as info:
        nv.pull("ibm_manila", source="ibm-account")
    assert info.value.message == f"could not open your IBM Quantum account ({missing})"
    assert info.value.hint == (
        "set IBM_QUANTUM_TOKEN to your IBM Quantum API key, or pull without an account with"
        " source='ibm'"
    )


_RETRY = "try again later, or pull without an account with source='ibm'"


@pytest.mark.parametrize(
    ("visible", "hint"),
    [
        (
            ["ibm_torino", "ibm_manila"],
            "pull a device that your account can see: ibm_manila, ibm_torino",
        ),
        ([], "pull without an account with source='ibm', because your account can see no device"),
        (["ibm_nowhere"], _RETRY),
        (ConnectionError("connection reset"), _RETRY),
    ],
    ids=["visible", "none-visible", "listed-but-not-opened", "listing-fails"],
)
def test_device_the_account_cannot_see(
    calls: list[Any],
    monkeypatch: pytest.MonkeyPatch,
    visible: list[str] | Exception,
    hint: str,
) -> None:
    def backends(self: _Service) -> list[Any]:
        if isinstance(visible, Exception):
            raise visible
        return [SimpleNamespace(name=name) for name in visible]

    monkeypatch.setattr(_Service, "backends", backends, raising=False)
    with pytest.raises(nv.SourceUnavailable) as info:
        ibm_account.pull("ibm_nowhere")
    assert info.value.message == (
        "your IBM account cannot open ibm_nowhere (No backend matches the criteria: ibm_nowhere)"
    )
    assert info.value.hint == hint


@pytest.mark.parametrize(
    ("at", "message", "hint"),
    [
        (
            None,
            "IBM returned no calibration for ibm_manila. Retired devices have no calibration",
            "run nv list to see every profile you can load offline",
        ),
        (
            "2019-01-01",
            "IBM returned no calibration for ibm_manila before 2019-01-01",
            "pick a later date",
        ),
    ],
)
def test_no_calibration_says_what_to_try(
    calls: list[Any], monkeypatch: pytest.MonkeyPatch, at: str | None, message: str, hint: str
) -> None:
    monkeypatch.setattr(_Backend, "properties", lambda self, **_: None)
    with pytest.raises(nv.SourceUnavailable) as info:
        ibm_account.pull("ibm_manila", at=at)
    assert (info.value.message, info.value.hint) == (message, hint)


def test_missing_runtime_gives_an_install_command_that_works(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", None)
    with pytest.raises(nv.SourceUnavailable) as info:
        nv.pull("ibm_manila", source="ibm-account")
    assert info.value.message == "source='ibm-account' needs qiskit-ibm-runtime"
    assert info.value.hint == install_hint("ibm")


def test_a_calibration_request_ibm_rejects_is_an_error_and_a_hint(
    calls: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from noisevault.cli import app

    exceptions = require("qiskit_ibm_runtime.exceptions")

    error = exceptions.IBMBackendApiProtocolError("Unexpected return value from the server.")

    def rejected(self: _Backend, refresh: bool = False, datetime: datetime | None = None) -> Any:
        raise error

    monkeypatch.setattr(_Backend, "properties", rejected)
    with pytest.raises(nv.SourceUnavailable, match="Unexpected return value"):
        ibm_account.pull("ibm_manila")
    result = CliRunner().invoke(app, ["pull", "ibm_manila", "--source", "ibm-account"])
    assert result.exit_code == 1 and result.exception.__class__ is SystemExit
    assert result.stderr.splitlines() == [
        f"error: IBM did not return the calibration of ibm_manila ({error})",
        "hint: try again later, or pull without an account with --source ibm",
    ]


_ON_QUBIT_0 = {
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
    "a readout probability as text": (
        "prob_meas0_prep1",
        {"value": "0.0548"},
        "prob_meas0_prep1 of qubit 0 is '0.0548', not a number",
    ),
    "a readout error of true": (
        "readout_error",
        {"value": True},
        "readout_error of qubit 0 is True, not a number",
    ),
    "a unit that is not a string": (
        "readout_length",
        {"unit": ["ns"]},
        "readout_length of qubit 0 has the unit ['ns'], not a string",
    ),
}


@pytest.mark.parametrize("case", list(_ON_QUBIT_0))
def test_a_snapshot_value_it_cannot_read_names_the_device_and_the_value(
    calls: list[Any], monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    runtime = require("qiskit_ibm_runtime")
    field, change, problem = _ON_QUBIT_0[case]
    props = _Backend(calls)._fake.properties().to_dict()
    next(p for p in props["qubits"][0] if p["name"] == field).update(change)
    snapshot = runtime.models.BackendProperties.from_dict(props)
    monkeypatch.setattr(_Backend, "properties", lambda self, **_: snapshot)
    with pytest.raises(nv.SourceDataError) as info:
        ibm_account.pull("ibm_manila")
    assert (info.value.message, info.value.hint) == (
        f"IBM's calibration of ibm_manila: {problem}",
        "pass an earlier at= to use an older calibration",
    )


def test_a_new_variant_of_measure_is_not_a_gate(
    calls: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # IBM added measure_terminal in November 2026, with a duration and no error.
    runtime = require("qiskit_ibm_runtime")
    props = _Backend(calls)._fake.properties().to_dict()
    length = next(p for p in props["gates"][0]["parameters"] if p["name"] == "gate_length")
    props["gates"].append(
        {
            "gate": "measure_terminal",
            "name": "measure_terminal0",
            "qubits": [0],
            "parameters": [{**length, "value": 1340}],
        }
    )
    snapshot = runtime.models.BackendProperties.from_dict(props)
    monkeypatch.setattr(_Backend, "properties", lambda self, **_: snapshot)
    profile = ibm_account.pull("ibm_manila")
    assert "measure_terminal" not in profile.gates
    assert not [r for r in profile.calibrations if r.gate == "measure_terminal"]
    assert "Not converted: measure_terminal." in profile.provenance.notes


def test_a_gate_on_a_qubit_that_is_not_an_index_names_the_gate(
    calls: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = require("qiskit_ibm_runtime")
    props = _Backend(calls)._fake.properties().to_dict()
    next(e for e in props["gates"] if e["gate"] == "cx")["qubits"] = ["0", 1]
    snapshot = runtime.models.BackendProperties.from_dict(props)
    monkeypatch.setattr(_Backend, "properties", lambda self, **_: snapshot)
    with pytest.raises(nv.SourceDataError) as info:
        ibm_account.pull("ibm_manila")
    assert (info.value.message, info.value.hint) == (
        "IBM's calibration of ibm_manila: gate cx is on ['0', 1]; a qubit index is an integer 0 or"
        " more",
        "pass an earlier at= to use an older calibration",
    )
