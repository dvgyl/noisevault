from __future__ import annotations

import numpy as np
import pytest

from noisevault.counts import PlannedCircuit
from noisevault.profile import Profile
from noisevault.reference import Op
from study import overrotation


def _circuit(*ops: Op, qubits: tuple[int, ...] = (0,)) -> PlannedCircuit:
    return PlannedCircuit(name="hand", qubits=qubits, ops=ops)


def test_four_sx_gates_with_an_extra_turn_give_the_right_result_with_cos_squared() -> None:
    circuit = _circuit(*(Op("sx", (0,)),) * 4, Op("delay", (0,), (100.0,)))
    expected = [np.cos(2 * e) ** 2 for e in overrotation.TURNS]
    assert overrotation.right_results(circuit, 0.0) == pytest.approx(expected)
    assert overrotation.right_results(circuit, 1.0) == pytest.approx(expected)


def test_the_x_error_acts_only_in_the_second_case() -> None:
    circuit = _circuit(Op("x", (0,)), Op("x", (0,)))
    assert overrotation.right_results(circuit, 0.0) == pytest.approx([1.0] * 5)
    expected = [np.cos(e) ** 2 for e in overrotation.TURNS]
    assert overrotation.right_results(circuit, 1.0) == pytest.approx(expected)


def test_the_right_result_is_the_ideal_bit_string_with_qubit_0_first() -> None:
    circuit = _circuit(Op("x", (1,)), Op("cz", (0, 1)), qubits=(3, 5))
    found = overrotation.probabilities(circuit, 0.0, 0.0)
    assert found == pytest.approx([0.0, 1.0, 0.0, 0.0])
    assert overrotation.right_results(circuit, 1.0)[-1] == pytest.approx(np.cos(0.01) ** 2)


def test_the_report_gives_each_chain_circuit_in_both_cases() -> None:
    pair = Profile.uniform(
        "pair",
        technology="superconducting",
        num_qubits=2,
        one_qubit_error=5e-4,
        two_qubit_error=8e-3,
        readout_error=5e-3,
    )
    circuits = overrotation.chain_circuits(pair)
    lines = overrotation.report(pair).splitlines()
    assert lines[0].startswith("profile: pair")
    for c in circuits:
        rows = [line for line in lines if line.split(" ", 1)[0] == c.name]
        assert len(rows) == len(overrotation.CASES)
        values = [float(x) for row in rows for x in row.split()[4:]]
        assert len(values) == len(overrotation.CASES) * len(overrotation.TURNS)
        assert all(0 <= x <= 1 for x in values)
