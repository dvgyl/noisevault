from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest
from conftest import require

import noisevault as nv

require("qiskit")

from qiskit.circuit import Measure  # noqa: E402
from qiskit.circuit.library import (  # noqa: E402
    CXGate,
    CZGate,
    ECRGate,
    SwapGate,
    SXGate,
    UnitaryGate,
    XGate,
    YGate,
)
from qiskit.providers import BackendV2, Options  # noqa: E402
from qiskit.transpiler import InstructionProperties, Target  # noqa: E402

MANILA_PROPERTIES = Path(__file__).parent / "fixtures" / "ibm" / "manila_properties.json"
MANILA_EDGES = ((0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3))


class _Backend(BackendV2):
    def __init__(self, target: Target, name: str = "toy", properties=None) -> None:
        super().__init__(name=name)
        self._target = target
        self._properties = properties

    @property
    def target(self) -> Target:
        return self._target

    def properties(self):
        return self._properties

    @property
    def max_circuits(self) -> None:
        return None

    @classmethod
    def _default_options(cls) -> Options:
        return Options()

    def run(self, run_input, **options):
        raise NotImplementedError


def _props(error: float, duration: float) -> InstructionProperties:
    return InstructionProperties(error=error, duration=duration)


def _target(num_qubits: int, *gates: tuple) -> Target:
    target = Target(num_qubits=num_qubits)
    for gate, loci in gates:
        target.add_instruction(gate, loci)
    target.add_instruction(Measure(), {(q,): _props(0.02, 1e-6) for q in range(num_qubits)})
    return target


def _assert_allows_exactly_what_the_target_allows(target: Target, profile: nv.Profile) -> None:
    table = profile.table
    for name in target.operation_names:
        if name == "measure":
            continue
        arity = target.operation_from_name(name).num_qubits
        for qubits in permutations(range(target.num_qubits), arity):
            supported = target.instruction_supported(name, qubits)
            if arity == 2 and table.symmetric(name):
                supported = supported or target.instruction_supported(name, qubits[::-1])
            assert table.allowed(name, qubits) == supported, (name, qubits)


def test_a_one_qubit_gate_on_some_qubits_is_disabled_on_the_others() -> None:
    target = _target(
        3,
        (SXGate(), {(0,): _props(0.001, 35e-9)}),
        (XGate(), {(q,): _props(0.002, 35e-9) for q in range(3)}),
        (CZGate(), {(0, 1): _props(0.01, 60e-9), (1, 2): _props(0.01, 60e-9)}),
    )
    assert not target.instruction_supported("sx", (1,))
    profile = nv.from_qiskit_backend(_Backend(target))
    assert profile.table.gate("sx", (0,)).avg_infidelity == 0.001
    assert profile.table.gate("sx", (1,)).state == "disabled"
    assert profile.table.gate("sx", (2,)).state == "disabled"
    _assert_allows_exactly_what_the_target_allows(target, profile)
    require("qiskit_aer")
    exported = profile.to_qiskit().target
    assert set(exported["sx"]) == {(0,)}


def test_a_target_error_above_1_is_refused_and_an_error_of_1_disables_the_gate() -> None:
    def profile(error: float) -> nv.Profile:
        target = _target(
            2,
            (XGate(), {(q,): _props(0.002, 35e-9) for q in range(2)}),
            (CZGate(), {(0, 1): _props(error, 60e-9)}),
        )
        return nv.from_qiskit_backend(_Backend(target))

    assert profile(1).table.gate("cz", (0, 1)).state == "disabled"
    with pytest.raises(nv.SourceDataError) as info:
        profile(1.5)
    assert info.value.message == (
        "backend toy: avg_infidelity of cz on qubits 0-1: Input should be less than or equal to 1,"
        " got 1.5"
    )


def test_properties_do_not_enable_a_gate_on_qubits_the_target_leaves_out() -> None:
    models = require("qiskit_ibm_runtime.models")
    properties = models.BackendProperties.from_dict(json.loads(MANILA_PROPERTIES.read_bytes()))
    assert {tuple(g.qubits) for g in properties.gates if g.gate == "sx"} == {(q,) for q in range(5)}
    target = _target(
        5,
        (SXGate(), {(0,): _props(0.001, 35e-9)}),
        (XGate(), {(q,): _props(0.002, 35e-9) for q in range(5)}),
        (CXGate(), {edge: _props(0.01, 300e-9) for edge in MANILA_EDGES}),
    )
    profile = nv.from_qiskit_backend(_Backend(target, properties=properties))
    states = [profile.table.gate("sx", (q,)).state for q in range(5)]
    assert states == ["calibrated", "disabled", "disabled", "disabled", "disabled"]
    assert profile.table.gate("sx", (0,)).avg_infidelity == 0.001
    assert profile.device.calibrated_at.isoformat() == "2024-05-27T18:27:23+00:00"
    _assert_allows_exactly_what_the_target_allows(target, profile)
    require("qiskit_aer")
    assert set(profile.to_qiskit().target["sx"]) == {(0,)}


def test_properties_reach_qubits_the_target_has_no_properties_for() -> None:
    models = require("qiskit_ibm_runtime.models")
    raw = json.loads(MANILA_PROPERTIES.read_bytes())
    stamp = raw["last_update_date"]
    raw["qubits"][3].append({"name": "operational", "value": 0, "unit": "", "date": stamp})
    properties = models.BackendProperties.from_dict(raw)
    working = (0, 1, 2, 4)
    target = Target(num_qubits=5)
    target.add_instruction(SXGate(), {(q,): _props(0.001, 35e-9) for q in working})
    target.add_instruction(CXGate(), {e: _props(0.01, 300e-9) for e in MANILA_EDGES if 3 not in e})
    target.add_instruction(Measure(), {(q,): _props(0.02, 1e-6) for q in working})
    assert target.qubit_properties is None
    profile = nv.from_qiskit_backend(_Backend(target, properties=properties))
    published = {p["name"]: p["value"] for p in raw["qubits"][0]}
    assert profile.table.qubit(0).readout == (
        published["prob_meas1_prep0"],
        published["prob_meas0_prep1"],
    )
    assert profile.table.qubit(3).disabled
    _assert_allows_exactly_what_the_target_allows(target, profile)
    require("qiskit_aer")
    assert set(profile.to_qiskit().target["measure"]) == set(target["measure"])


def test_a_directed_gate_in_one_direction_of_a_pair_is_disabled_in_the_other() -> None:
    cz = {(a, b): _props(0.01, 60e-9) for a, b in ((0, 1), (1, 0), (1, 2), (2, 1))}
    target = _target(
        3,
        (XGate(), {(q,): _props(0.002, 35e-9) for q in range(3)}),
        (CXGate(), {(0, 1): _props(0.02, 300e-9)}),
        (CZGate(), cz),
    )
    profile = nv.from_qiskit_backend(_Backend(target))
    assert profile.table.gate("cx", (0, 1)).avg_infidelity == 0.02
    assert profile.table.gate("cx", (1, 0)).state == "disabled"
    assert profile.table.gate("cx", (1, 2)).state == "disabled"
    _assert_allows_exactly_what_the_target_allows(target, profile)
    require("qiskit_aer")
    assert set(profile.to_qiskit().target["cx"]) == {(0, 1)}


def test_a_symmetric_gate_missing_from_a_pair_is_disabled_on_it() -> None:
    target = _target(
        3,
        (XGate(), {(q,): _props(0.002, 35e-9) for q in range(3)}),
        (ECRGate(), {(0, 1): _props(0.02, 500e-9), (2, 1): _props(0.02, 500e-9)}),
        (CZGate(), {(1, 2): _props(0.01, 60e-9)}),
    )
    profile = nv.from_qiskit_backend(_Backend(target))
    assert profile.table.gate("cz", (0, 1)).state == "disabled"
    assert profile.table.gate("cz", (2, 1)).avg_infidelity == 0.01
    assert profile.table.gate("ecr", (1, 2)).state == "disabled"
    _assert_allows_exactly_what_the_target_allows(target, profile)


def test_a_gate_missing_from_an_undirected_pair_is_disabled_on_it() -> None:
    target = _target(
        3,
        (XGate(), {(q,): _props(0.002, 35e-9) for q in range(3)}),
        (CZGate(), {(0, 1): _props(0.01, 60e-9), (1, 2): _props(0.01, 60e-9)}),
        (SwapGate(), {(1, 0): _props(0.03, 180e-9)}),
    )
    profile = nv.from_qiskit_backend(_Backend(target))
    assert not profile.connectivity.directed
    assert profile.table.gate("swap", (0, 1)).avg_infidelity == 0.03
    assert profile.table.gate("swap", (2, 1)).state == "disabled"
    _assert_allows_exactly_what_the_target_allows(target, profile)


def test_a_gate_with_no_locus_is_not_in_the_profile() -> None:
    target = _target(
        2,
        (XGate(), {(q,): _props(0.002, 35e-9) for q in range(2)}),
        (YGate(), {}),
        (CZGate(), {(0, 1): _props(0.01, 60e-9)}),
    )
    profile = nv.from_qiskit_backend(_Backend(target))
    assert "y" not in profile.gates
    _assert_allows_exactly_what_the_target_allows(target, profile)


def test_a_global_one_qubit_gate_is_uncalibrated_on_every_qubit() -> None:
    target = _target(2, (SXGate(), {(q,): _props(0.001, 32e-9) for q in range(2)}))
    target.add_instruction(XGate())
    assert target["x"] == {None: None}
    profile = nv.from_qiskit_backend(_Backend(target))
    assert [profile.table.gate("x", (q,)).state for q in range(2)] == ["uncalibrated"] * 2
    _assert_allows_exactly_what_the_target_allows(target, profile)


def test_a_global_two_qubit_gate_is_named_in_the_notes() -> None:
    target = _target(
        3,
        (SXGate(), {(q,): _props(0.001, 32e-9) for q in range(3)}),
        (CXGate(), {(0, 1): _props(0.01, 300e-9)}),
    )
    target.add_instruction(CZGate())
    profile = nv.from_qiskit_backend(_Backend(target))
    assert "cz" not in profile.gates
    assert (
        "Not converted: cz, a global instruction on any 2 qubits. A profile records a gate on 2"
        " or more qubits only on the qubits that the source lists." in profile.provenance.notes
    )


def test_an_ideal_custom_gate_keeps_its_arity_and_resolves_as_ideal() -> None:
    target = Target(num_qubits=2)
    target.add_instruction(
        UnitaryGate(np.diag([1, 1, 1, -1])), {(0, 1): _props(0, 0)}, name="unitary"
    )
    profile = nv.from_qiskit_backend(_Backend(target))
    assert (profile.gates["unitary"].qubits, profile.gates["unitary"].virtual) == (2, True)
    assert profile.table.gate("unitary", (0, 1)).state == "ideal"
    _assert_allows_exactly_what_the_target_allows(target, profile)


def test_an_ionq_backend_is_trapped_ion() -> None:
    class IonQQPUBackend(_Backend):
        pass

    IonQQPUBackend.__module__ = "qiskit_ionq.ionq_backend"
    target = _target(
        2,
        (XGate(), {(q,): _props(0.0002, 130e-6) for q in range(2)}),
        (CXGate(), {(0, 1): _props(0.005, 970e-6), (1, 0): _props(0.005, 970e-6)}),
    )
    for name in ("ionq_qpu", "ionq_qpu.forte-1"):
        profile = nv.from_qiskit_backend(IonQQPUBackend(target, name=name))
        assert profile.device.technology == "trapped_ion", name
    assert nv.from_qiskit_backend(_Backend(target, name="ionq_qpu")).device.technology == (
        "trapped_ion"
    )


def test_a_backend_that_names_no_known_provider_is_other_technology() -> None:
    from qiskit.providers.fake_provider import GenericBackendV2

    profile = nv.from_qiskit_backend(GenericBackendV2(num_qubits=3, seed=7))
    assert profile.device.technology == "other"
