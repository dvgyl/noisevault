from __future__ import annotations

from math import pi

import numpy as np
import pytest
from conftest import require

stim = require("stim")

from noisevault import check, gates  # noqa: E402
from noisevault.counts import PlannedCircuit  # noqa: E402
from noisevault.errors import NoiseVaultError  # noqa: E402
from noisevault.profile import Profile  # noqa: E402
from noisevault.reference import Op, probabilities  # noqa: E402
from noisevault.wide import CLIFFORD, counts, ideal_bits, noisy_circuit, sample  # noqa: E402

SHOTS = 1_000_000
QUBITS = (0, 1, 2, 3)
# Each op once, both operand orders of cx, and waits of two lengths.
OPS = (
    Op("sx", (0,)),
    Op("x", (1,)),
    Op("delay", (2,), (2000.0,)),
    Op("cx", (0, 1)),
    Op("rz", (2,), (pi / 2,)),
    Op("sx", (2,)),
    Op("cx", (2, 1)),
    Op("delay", (0,), (900.0,)),
    Op("iswap", (2, 3)),
    Op("sx", (3,)),
    Op("rzz", (0, 1), (pi / 2,)),
    Op("cz", (3, 2)),
    Op("sx", (0,)),
)
_LINE = [(0, 1), (1, 2), (2, 3)]


def _line(t1_us: float | None, t2_us: float | None) -> Profile:
    """Four qubits with readout error of two sizes and asymmetric Pauli noise on cx and sx."""
    data = Profile.uniform(
        "line",
        technology="superconducting",
        num_qubits=4,
        connectivity=_LINE,
        one_qubit_error=0.01,
        two_qubit_error=0.05,
        t1_us=t1_us,
        t2_us=t2_us,
        one_qubit_ns=40.0,
        two_qubit_ns=300.0,
    ).to_dict()
    data["readout"] = {"p1_given_0": 0.02, "p0_given_1": 0.07}
    if t1_us is None:
        cx = [0.0] * 15
        cx[0], cx[12] = 0.06, 0.03  # IX: X on the target. ZI: Z on the control.
        data["gates"]["cx"] = {"pauli": cx, "duration_ns": 300.0}
        data["gates"]["sx"] = {"pauli": [0.03, 0.0, 0.005], "duration_ns": 40.0}
    return Profile.model_validate(data)


def _frequencies(bits: np.ndarray) -> np.ndarray:
    found = np.zeros(2 ** bits.shape[1])
    for key, k in counts(bits).items():
        found[int(key, 2)] = k
    return found / len(bits)


def _largest_z(expected: np.ndarray, found: np.ndarray) -> float:
    """The largest deviation of an outcome frequency, in units of its sampling error."""
    sigma = np.sqrt(expected * (1 - expected) / SHOTS)
    assert np.all(found[sigma == 0] == 0)
    seen = sigma > 0
    return float(np.max(np.abs(found[seen] - expected[seen]) / sigma[seen]))


@pytest.mark.parametrize(("name", "params"), list(CLIFFORD))
def test_each_clifford_row_equals_the_stim_tableau_up_to_a_global_phase(name, params) -> None:
    unitary = gates.unitary(name, params)

    found = stim.Tableau.from_unitary_matrix(unitary, endian="big")

    assert found == stim.Tableau.from_named_gate(CLIFFORD[name, params])


def test_with_pauli_noise_the_forecast_equals_the_reference_on_every_outcome() -> None:
    profile = _line(None, 40.0)
    circuit = PlannedCircuit(name="mix", qubits=QUBITS, ops=OPS)

    found = _frequencies(sample(profile, circuit, SHOTS, seed=1))

    expected = probabilities(profile, OPS, 4, layout=QUBITS, unknown_gates="error")
    assert _largest_z(expected, found) < 4.5


def test_with_relaxation_the_forecast_equals_the_twirled_reference_on_every_outcome() -> None:
    profile = _line(30.0, 40.0)
    ops = tuple(op for op in OPS if op.name != "delay")
    circuit = PlannedCircuit(name="gates", qubits=QUBITS, ops=ops)

    found = _frequencies(sample(profile, circuit, SHOTS, seed=2))

    twirled = check._twirled(profile, check.Circuit("gates", 4, ops), list(QUBITS), False)
    exact = probabilities(profile, ops, 4, layout=QUBITS, unknown_gates="error")
    assert _largest_z(twirled, found) < 4.5
    assert 0.001 < 0.5 * np.abs(twirled - exact).sum() < 0.05


def test_readout_flips_each_bit_with_the_probability_for_its_value() -> None:
    data = Profile.uniform(
        "ideal", technology="superconducting", num_qubits=2, one_qubit_error=0, two_qubit_error=0
    ).to_dict()
    profile = Profile.model_validate(data | {"readout": {"p1_given_0": 0.1, "p0_given_1": 0.3}})
    circuit = PlannedCircuit(name="x0", qubits=(0, 1), ops=(Op("x", (0,)),))

    bits = sample(profile, circuit, 200_000, seed=3)

    assert bits[:, 0].mean() == pytest.approx(0.7, abs=0.005)
    assert bits[:, 1].mean() == pytest.approx(0.1, abs=0.005)


def test_the_circuit_holds_one_twirl_after_each_noisy_op_and_ideal_measurements() -> None:
    profile = _line(None, 40.0)
    circuit = PlannedCircuit(name="mix", qubits=QUBITS, ops=OPS)

    noisy, readout = noisy_circuit(profile, circuit)

    names = [line.name for line in noisy]
    assert names.count("PAULI_CHANNEL_2") == 5
    assert names[-1] == "M"
    assert readout.tolist() == [[0.02, 0.07]] * 4
    first = next(line for line in noisy if line.name == "PAULI_CHANNEL_1")
    assert first.gate_args_copy() == pytest.approx([0.03, 0.0, 0.005])


def test_ideal_bits_gives_the_one_result_without_noise_and_refuses_a_superposition() -> None:
    flipped = PlannedCircuit(name="x", qubits=(5, 6), ops=(Op("sx", (1,)), Op("sx", (1,))))
    mixed = PlannedCircuit(name="half", qubits=(5,), ops=(Op("sx", (0,)),))

    assert ideal_bits(flipped) == "01"
    with pytest.raises(NoiseVaultError, match="circuit half has no certain result"):
        ideal_bits(mixed)


def test_an_op_outside_the_clifford_table_is_refused_by_name() -> None:
    circuit = PlannedCircuit(name="t", qubits=(0,), ops=(Op("rz", (0,), (pi / 4,)),))

    with pytest.raises(NoiseVaultError, match=r"circuit t: rz\(0.785398\) is not a Clifford"):
        noisy_circuit(_line(None, 40.0), circuit)
