from __future__ import annotations

from collections import Counter
from math import pi

import numpy as np
import pytest
from conftest import require

import noisevault as nv
from noisevault import wide
from noisevault.counts import PlannedCircuit, plan, simulate
from noisevault.profile import Profile
from noisevault.reference import Op, probabilities
from noisevault.suite import (
    ASSUMED_GATE_FACTOR,
    CHAIN_TARGETS,
    CLIFFORD_SAMPLES,
    CLIFFORD_WRONG,
    DEPTHS,
    MIN_WAIT_NS,
    TEST_SEED,
    _circuit,
    _clifford_circuits,
    _layers,
    _mirror,
    _natives,
    describe,
    role,
    suite,
    train_names,
)

LINE = [(0, 1), (1, 2), (2, 3)]


def _device(**changes: float) -> Profile:
    values = {
        "one_qubit_error": 1e-3,
        "two_qubit_error": 1e-2,
        "readout_error": 2e-2,
        "t1_us": 200.0,
        "t2_us": 150.0,
        "one_qubit_ns": 30.0,
        "two_qubit_ns": 34.0,
    }
    return Profile.uniform(
        "line", technology="superconducting", num_qubits=4, connectivity=LINE, **(values | changes)
    )


@pytest.fixture(scope="module")
def device() -> Profile:
    return _device()


@pytest.fixture(scope="module")
def circuits(device: Profile) -> tuple:
    return suite(device)


def _noiseless(device: Profile) -> Profile:
    return _device(
        one_qubit_error=0.0, two_qubit_error=0.0, readout_error=0.0, t1_us=1e12, t2_us=1e12
    )


def test_the_suite_holds_the_plan_circuits_and_each_family(device: Profile, circuits) -> None:
    names = [c.name for c in circuits]
    assert len(set(names)) == len(names)
    assert names[: len(plan(device))] == [c.name for c in plan(device)]
    families = Counter(name.split("_")[0].rstrip("0123456789") for name in names)
    assert families["deep"] == 3 * len({n.split("_")[1] for n in names if n.startswith("deep_")})
    assert families["pair"] >= 3
    assert families["qubit"] == 4
    assert "readout_1" in names
    assert any(name.startswith("periodic_d") for name in names)
    assert any(name.startswith("qubits_d") for name in names)


def test_each_deep_circuit_gives_one_bit_string_without_noise(device: Profile, circuits) -> None:
    ideal = _noiseless(device)
    added = circuits[len(plan(device)) :]
    assert len(added) >= 15
    for circuit in added:
        probs = probabilities(ideal, circuit.ops, len(circuit.qubits), layout=circuit.qubits)
        assert probs.max() == pytest.approx(1.0, abs=1e-6), circuit.name


def test_readout_1_prepares_each_qubit_in_1(device: Profile, circuits) -> None:
    ideal = _noiseless(device)
    circuit = next(c for c in circuits if c.name == "readout_1")
    probs = probabilities(ideal, circuit.ops, 4, layout=circuit.qubits)
    assert int(np.argmax(probs)) == 2**4 - 1


def test_some_deep_circuit_ends_in_a_bit_string_other_than_all_0(device, circuits) -> None:
    ideal = _noiseless(device)
    ends = set()
    for circuit in circuits:
        if circuit.name.startswith("deep_"):
            probs = probabilities(ideal, circuit.ops, 4, layout=circuit.qubits)
            ends.add(int(np.argmax(probs)))
    assert ends - {0}


def test_each_chain_depth_reaches_its_target_at_the_assumed_gate_factor(device, circuits) -> None:
    ideal = _noiseless(device)
    assumed = device.model_copy(
        update={"unmodeled_error": {"gates": {"factor": ASSUMED_GATE_FACTOR}}}
    )
    deep = {c.name: c for c in circuits if c.name.endswith("_s0") and c.name.startswith("deep_")}
    depths = sorted(int(name.split("_")[1][1:]) for name in deep)
    assert len(depths) == len(set(depths)) <= len(CHAIN_TARGETS)
    assert set(depths) <= set(DEPTHS)
    for depth, target in zip(depths, sorted(CHAIN_TARGETS, reverse=True), strict=False):
        circuit = deep[f"deep_d{depth}_s0"]
        wanted = int(np.argmax(probabilities(ideal, circuit.ops, 4, layout=circuit.qubits)))
        predicted = probabilities(assumed, circuit.ops, 4, layout=circuit.qubits)[wanted]
        assert predicted <= target
        assert probabilities(device, circuit.ops, 4, layout=circuit.qubits)[wanted] > predicted


def test_the_depth_comes_from_the_ideal_result_and_not_from_the_most_likely_result() -> None:
    # With this much relaxation, a qubit that should end in 1 is most likely read as 0. The
    # probability of the ideal result is then far below every target at depth 1.
    data = Profile.uniform(
        "hot",
        technology="superconducting",
        num_qubits=1,
        one_qubit_error=0.3,
        readout_error=0.0,
        t1_us=0.01,
        t2_us=0.01,
        one_qubit_ns=1000.0,
    ).to_dict()
    data["gates"] = {"x": data["gates"]["x"]}
    hot = Profile.model_validate(data)
    names = [c.name for c in suite(hot)]
    assert "qubit0_d1" in names


def test_a_wait_on_a_later_qubit_that_disables_delay_is_refused() -> None:
    data = _device().to_dict()
    data["gates"]["delay"] = {}
    data["calibrations"] = [{"gate": "delay", "qubits": [2], "disabled": True}]
    profile = Profile.model_validate(data)
    # Qubit 0 and qubit 2 wait while qubit 1 runs its gate. Only qubit 2 disables delay.
    with pytest.raises(nv.errors.LayoutError, match="qubit 2 waits"):
        _circuit(profile, "one_gate", (0, 1, 2), [(Op("x", (1,)),)])


def test_train_names_gives_the_fit_circuits_of_each_run_that_suite_makes(circuits) -> None:
    names = [c.name for c in circuits]
    assert train_names(names) == [name for name in names if role(name) == "train"]
    assert train_names(["ghz_chain", "mirror", "single_qubit", "readout"]) is None
    assert train_names(["mirror", "single_qubit", "readout", "qubit0_d6"]) == [
        "readout",
        "qubit0_d6",
    ]


def test_a_wait_is_never_shorter_than_the_minimum(circuits) -> None:
    waits = [op.params[0] for c in circuits for op in c.ops if op.name == "delay"]
    assert waits
    assert min(waits) >= MIN_WAIT_NS


def test_each_qubit_of_a_deep_circuit_is_busy_for_the_same_time(device: Profile, circuits) -> None:
    table = device.table
    for circuit in circuits:
        if not circuit.name.startswith(("deep_", "pair", "periodic_")):
            continue
        busy = [0.0] * len(circuit.qubits)
        for op in circuit.ops:
            if op.name == "delay":
                busy[op.qubits[0]] += op.params[0]
                continue
            found = table.gate(op.name, [circuit.qubits[q] for q in op.qubits])
            for q in op.qubits:
                busy[q] += found.duration_ns or 0.0
        assert max(busy) - min(busy) < 1e-6, circuit.name


def test_the_same_seed_gives_the_same_circuits_and_another_seed_gives_others(device) -> None:
    again = suite(device)
    other = suite(device, seed=1)
    first = suite(device)
    assert again == first
    assert [c.ops for c in other if c.name.startswith("deep_")] != [
        c.ops for c in first if c.name.startswith("deep_")
    ]


def test_the_roles_separate_the_fit_circuits_from_the_test_circuits(circuits) -> None:
    roles = {c.name: role(c.name) for c in circuits}
    assert set(roles.values()) == {"train", "test_near", "test_far"}
    assert roles["readout"] == roles["readout_1"] == "train"
    assert roles["ghz_chain"] == roles["mirror"] == "test_far"
    for name, found in roles.items():
        if name.startswith("deep_"):
            assert (found == "test_near") == name.endswith(f"_s{TEST_SEED}")
        if name.startswith(("qubits_", "periodic_")):
            assert found == "test_far"
        if name.startswith(("pair", "qubit")) and not name.startswith("qubits_"):
            assert found == "train"
    assert [row["role"] for row in describe(circuits)] == list(roles.values())


def test_simulated_counts_of_the_suite_load_as_a_counts_file(device, circuits, tmp_path) -> None:
    counts = simulate(device, circuits, shots=200, seed=3)
    path = counts.save(tmp_path / "suite.counts.json")
    loaded = nv.counts.load_counts(path)
    assert [c.name for c in loaded.circuits] == [c.name for c in circuits]


def test_a_profile_with_factors_gives_the_circuits_of_its_calibration(device, circuits) -> None:
    scaled = device.model_copy(update={"unmodeled_error": {"gates": {"factor": 3.0}}})
    assert suite(scaled) == circuits


@pytest.fixture(scope="module")
def long_line() -> Profile:
    """Ten qubits in a line with the IBM gate set, so that a wide chain exists."""
    data = Profile.uniform(
        "long",
        technology="superconducting",
        num_qubits=10,
        connectivity=[(q, q + 1) for q in range(9)],
        one_qubit_error=1e-3,
        two_qubit_error=1e-2,
        readout_error=2e-2,
        t1_us=200.0,
        t2_us=150.0,
        one_qubit_ns=32.0,
        two_qubit_ns=68.0,
    ).to_dict()
    data["gates"] = {k: v for k, v in data["gates"].items() if k in ("rz", "sx", "x", "cz")}
    return Profile.model_validate(data)


def test_a_device_with_a_long_chain_gets_wide_test_circuits(long_line: Profile) -> None:
    circuits = suite(long_line)
    wide = [c for c in circuits if c.name.startswith("wide_")]
    assert wide and all(len(c.qubits) == 8 and role(c.name) == "test_far" for c in wide)
    assert len({c.qubits for c in wide}) == 1
    pairs = {
        tuple(sorted((a, b))) for c in wide for a, b in zip(c.qubits, c.qubits[1:], strict=False)
    }
    assert all(b - a == 1 for a, b in pairs)
    assert train_names([c.name for c in circuits]) == [
        c.name for c in circuits if role(c.name) == "train"
    ]


def test_a_device_with_fewer_qubits_than_the_wide_chain_gets_no_wide_circuit(circuits) -> None:
    assert not any(c.name.startswith("wide_") for c in circuits)


def test_a_device_with_fewer_than_20_qubits_gets_no_clifford_circuit(circuits, long_line) -> None:
    assert not any(c.name.startswith("clifford_") for c in circuits)
    assert not _clifford_circuits(long_line, long_line, 0)


def _ibm_line(num_qubits: int, natives: tuple[str, ...] = ("rz", "sx", "x", "cz")) -> Profile:
    data = Profile.uniform(
        "line",
        technology="superconducting",
        num_qubits=num_qubits,
        connectivity=[(q, q + 1) for q in range(num_qubits - 1)],
        one_qubit_error=1e-3,
        two_qubit_error=1e-2,
        readout_error=2e-2,
        t1_us=200.0,
        t2_us=150.0,
        one_qubit_ns=32.0,
        two_qubit_ns=68.0,
    ).to_dict()
    data["gates"] = {k: v for k, v in data["gates"].items() if k in natives}
    return Profile.model_validate(data)


@pytest.fixture(scope="module")
def line_24() -> Profile:
    """24 qubits in a line with the IBM gate set, so that a chain of 20 exists."""
    require("stim")
    return _ibm_line(24)


@pytest.fixture(scope="module")
def clifford_suite(line_24: Profile) -> tuple:
    return suite(line_24)


def test_a_device_with_a_chain_of_20_gets_one_clifford_circuit_of_20(clifford_suite) -> None:
    found = [c for c in clifford_suite if c.name.startswith("clifford_")]

    assert [c.name.rsplit("_d", 1)[0] for c in found] == ["clifford_n20"]
    (circuit,) = found
    assert len(circuit.qubits) == 20
    assert all(abs(a - b) == 1 for a, b in zip(circuit.qubits, circuit.qubits[1:], strict=False))
    assert {op.name for op in circuit.ops} <= {"sx", "x", "cz", "delay"}
    assert all(wide.instruction(op) for op in circuit.ops if op.name != "delay")


def test_each_clifford_circuit_gives_one_bit_string_in_stim_without_noise(clifford_suite) -> None:
    noiseless = Profile.uniform(
        "ideal",
        technology="superconducting",
        num_qubits=24,
        connectivity=[(q, q + 1) for q in range(23)],
        one_qubit_error=0.0,
        two_qubit_error=0.0,
    )
    for circuit in clifford_suite:
        if circuit.name.startswith("clifford_"):
            bits = wide.ideal_bits(circuit)
            assert wide.counts(wide.sample(noiseless, circuit, 1000, seed=0)) == {bits: 1000}
            assert set(bits) == {"0", "1"}


def test_the_clifford_depth_is_the_first_whole_depth_that_reaches_the_target(
    line_24, clifford_suite
) -> None:
    (circuit,) = [c for c in clifford_suite if c.name.startswith("clifford_")]
    depth = int(circuit.name.rsplit("_d", 1)[1])
    assumed = line_24.model_copy(
        update={"unmodeled_error": {"gates": {"factor": ASSUMED_GATE_FACTOR}}}
    )
    bits = np.array([b == "1" for b in wide.ideal_bits(circuit)])
    natives = _natives(line_24, circuit.qubits, clifford=True)
    stream = (0, 6, 20)

    def built(at: int) -> PlannedCircuit:
        layers = _layers(natives, 20, at, stream)
        return _mirror(line_24, "deep", circuit.qubits, natives, layers, stream)

    def wrong(at: int) -> float:
        return float(np.mean(wide.sample(assumed, built(at), CLIFFORD_SAMPLES, seed=0) != bits))

    assert built(depth).ops == circuit.ops
    assert wrong(depth) >= CLIFFORD_WRONG > wrong(depth - 1)


def test_the_clifford_circuits_are_far_tests_that_no_fit_sees(clifford_suite) -> None:
    names = [c.name for c in clifford_suite]
    clifford = [name for name in names if name.startswith("clifford_")]

    assert clifford and all(role(name) == "test_far" for name in clifford)
    assert not set(clifford) & set(train_names(names) or [])
    assert train_names(["readout", *clifford]) == ["readout"]


def test_without_stim_the_suite_holds_no_clifford_circuit(line_24, clifford_suite, monkeypatch):
    monkeypatch.setattr(wide, "has_stim", lambda: False)

    assert suite(line_24) == tuple(c for c in clifford_suite if not c.name.startswith("clifford_"))


def test_a_chain_without_a_clifford_native_on_a_pair_gets_no_clifford_circuit() -> None:
    require("stim")
    no_clifford_pair = _ibm_line(20, ("rz", "sx", "x", "ecr"))

    assert not _clifford_circuits(no_clifford_pair, no_clifford_pair, 0)


def test_the_clifford_natives_take_each_clifford_angle_of_a_gate_with_parameters() -> None:
    ions = Profile.uniform(
        "ions",
        technology="trapped_ion",
        num_qubits=3,
        connectivity="all_to_all",
        one_qubit_error=1e-3,
        two_qubit_error=1e-2,
    ).to_dict()
    ions["gates"] = {k: v for k, v in ions["gates"].items() if k in ("r", "ms")}

    found = _natives(Profile.model_validate(ions), (0, 1, 2), clifford=True)

    assert [(op.name, op.params) for op in found.ones] == [
        ("r", (pi / 2, 0.0)),
        ("r", (pi / 2, pi / 2)),
    ]
    assert [(op.name, op.params) for op in found.pairs] == [("ms", (0.0, 0.0))] * 2
