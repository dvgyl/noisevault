from __future__ import annotations

import numpy as np
import pytest

from noisevault.counts import PlannedCircuit
from noisevault.design import Price, _variances, design, information
from noisevault.errors import NoiseVaultError
from noisevault.profile import Profile
from noisevault.reference import Op
from noisevault.suite import role, suite


@pytest.fixture(scope="module")
def device() -> Profile:
    return Profile.uniform(
        "pair",
        technology="trapped_ion",
        num_qubits=2,
        one_qubit_error=5e-4,
        two_qubit_error=8e-3,
        readout_error=5e-3,
    )


@pytest.fixture(scope="module")
def circuits(device: Profile) -> tuple:
    return suite(device)


def test_a_deeper_circuit_tells_more_about_the_gate_factor(device, circuits):
    by_name = {c.name: c for c in circuits}
    family = sorted(
        (c for c in circuits if c.name.startswith("deep_") and c.name.endswith("_s0")),
        key=lambda c: len(c.ops),
    )
    deep = information(device, family[-1], gate_factor=1.0)
    assert deep[0, 0] > 2 * information(device, family[0], gate_factor=1.0)[0, 0]
    assert information(device, by_name["readout"])[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert information(device, by_name["readout"])[1, 1] > 0


def test_the_information_is_a_symmetric_matrix_with_no_negative_eigenvalue(device, circuits):
    for circuit in circuits[:6]:
        info = information(device, circuit)
        assert info == pytest.approx(info.T)
        assert np.linalg.eigvalsh(info).min() >= -1e-9


def test_a_design_keeps_the_readout_circuits_and_a_test_circuit_within_the_budget(device, circuits):
    price = Price(task=0.30, shot=0.08)
    found = design(device, circuits, budget=320.0, price=price)

    names = [c.name for c in found.circuits]
    assert {"readout", "readout_1"} <= set(names)
    assert sum(role(name) == "test_near" for name in names) == 1
    assert any(name.startswith(("deep_", "pair", "qubit")) for name in names)
    assert found.cost == pytest.approx(price.cost(len(names), found.shots))
    assert found.cost <= 320.0
    assert price.cost(len(names), found.shots + 1) > 320.0
    assert 0 < found.gate_error < 0.2
    assert 0 < found.readout_error


def test_a_larger_budget_gives_a_smaller_predicted_error(device, circuits):
    price = Price(task=0.30, shot=0.0235, one_task=True)
    small = design(device, circuits, budget=100.0, price=price)
    large = design(device, circuits, budget=400.0, price=price)

    assert large.gate_error < small.gate_error
    assert large.cost <= 400.0


def test_one_task_for_all_circuits_pays_the_task_price_one_time():
    assert Price(0.30, 0.01, one_task=True).cost(5, 100) == pytest.approx(5.30)
    assert Price(0.30, 0.01).cost(5, 100) == pytest.approx(6.50)


def test_a_budget_below_the_smallest_design_is_refused(device, circuits):
    with pytest.raises(NoiseVaultError, match="does not pay 100 shots"):
        design(device, circuits, budget=5.0, price=Price(task=0.30, shot=0.08))


def _two_x_gates(readout_error: float) -> np.ndarray:
    profile = Profile.uniform(
        "one",
        technology="superconducting",
        num_qubits=1,
        one_qubit_error=0.01,
        readout_error=readout_error,
    )
    circuit = PlannedCircuit(name="xs", qubits=(0,), ops=(Op("x", (0,)), Op("x", (0,))))
    return information(profile, circuit)


def test_one_binary_probability_identifies_neither_factor():
    for readout_error in (1e-9, 1e-3, 0.01):
        assert _variances(_two_x_gates(readout_error)) == (None, None)


def test_a_factor_with_zero_information_leaves_the_other_factor_identified():
    gate, readout = _variances(np.array([[0.02, 0.0], [0.0, 0.0]]))
    assert gate == pytest.approx(50.0) and readout is None
    gate, readout = _variances(np.array([[0.0, 0.0], [0.0, 0.01]]))
    assert gate is None and readout == pytest.approx(100.0)
    gate, readout = _variances(np.array([[2.0, 1e-5], [1e-5, 1e-9]]))
    assert gate is not None and readout is not None
    assert gate == pytest.approx(1e-9 / (2e-9 - 1e-10))


def test_task_fees_above_the_budget_pay_no_shots_also_when_shots_are_free(device, circuits):
    with pytest.raises(NoiseVaultError, match="does not pay 100 shots"):
        design(device, circuits, budget=1.0, price=Price(task=0.30, shot=0.0))


def test_a_profile_with_no_readout_error_gives_the_readout_axis_zero_information():
    profile = Profile.uniform(
        "one", technology="superconducting", num_qubits=1, one_qubit_error=0.01
    )
    circuit = PlannedCircuit(name="xs", qubits=(0,), ops=(Op("x", (0,)), Op("x", (0,))))

    info = information(profile, circuit)

    assert info[0, 0] > 0
    assert info[0, 1] == info[1, 0] == info[1, 1] == 0
    gate, readout = _variances(info)
    assert gate == pytest.approx(1 / info[0, 0]) and readout is None


def test_max_shots_limits_the_shots_of_each_circuit(device, circuits):
    found = design(
        device, circuits, budget=1000.0, price=Price(task=0.0, shot=0.001), max_shots=500
    )
    assert found.shots == 500
