"""The Qiskit export: Target, per-locus channels, readout, reset, delays, validation, report."""

from __future__ import annotations

import json
import time
import warnings
from typing import Any

import numpy as np
import pytest
from conftest import MANILA_V01, migrated, require, toy

import noisevault as nv
from noisevault.channels import ChannelSpec, gate_channels, superoperator
from noisevault.conversion import resolve_op
from noisevault.errors import (
    DisabledGateError,
    MissingCalibrationError,
    NoiseApproximationWarning,
    NoiseVaultError,
    UnsupportedEffect,
)
from noisevault.profile import Profile
from noisevault.reference import Op
from noisevault.reference import probabilities as reference
from noisevault.report import Report

require("qiskit_aer")

from qiskit import QuantumCircuit, transpile  # noqa: E402
from qiskit.circuit import Parameter  # noqa: E402
from qiskit.circuit.library import iSwapGate  # noqa: E402
from qiskit.quantum_info import SuperOp  # noqa: E402
from qiskit.transpiler import CouplingMap, PassManager  # noqa: E402
from qiskit.transpiler.exceptions import TranspilerError  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402
from qiskit_aer.noise.passes import RelaxationNoisePass  # noqa: E402

from noisevault import gates  # noqa: E402
from noisevault.frameworks.qiskit import (  # noqa: E402
    CircuitNotNativeError,
    NoiseVaultSimulator,
    SqrtISwapGate,
    UnsupportedDevice,
    gate_error,
    to_qiskit,
)


def quiet_export(profile: Profile, **options: Any) -> NoiseVaultSimulator:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoiseApproximationWarning)
        return to_qiskit(profile, **options)


def ops_of(circuit: QuantumCircuit, layout: list[int]) -> list[Op]:
    """A transpiled circuit's gates as reference ops on circuit qubits ``0..len(layout)-1``."""
    position = {q: i for i, q in enumerate(layout)}
    return [
        Op(
            i.operation.name,
            tuple(position[circuit.find_bit(q).index] for q in i.qubits),
            tuple(float(p) for p in i.operation.params),
        )
        for i in circuit.data
        if i.operation.name != "barrier"
    ]


def aer_probabilities(sim: AerSimulator, circuit: QuantumCircuit, layout: list[int]) -> np.ndarray:
    """Exact pre-measurement probabilities, big-endian over ``layout`` like the reference."""
    circuit = circuit.copy()
    circuit.save_probabilities(layout)
    little = np.asarray(sim.run(circuit).result().data()["probabilities"])
    n = len(layout)
    return little.reshape((2,) * n).transpose(range(n - 1, -1, -1)).reshape(-1)


def with_exported_readout(
    probs: np.ndarray, sim: NoiseVaultSimulator, layout: list[int]
) -> np.ndarray:
    """Apply the simulator's readout errors as Aer defines them (rows = prepared state)."""
    rows = {
        tuple(e["gate_qubits"][0]): np.array(e["probabilities"])
        for e in sim.noise_model.to_dict()["errors"]
        if e["type"] == "roerror"
    }
    probs = probs.reshape((2,) * len(layout))
    for axis, q in enumerate(layout):
        probs = np.moveaxis(np.tensordot(rows[(q,)].T, probs, axes=([1], [axis])), 0, axis)
    return probs.reshape(-1)


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def ghz(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"ghz{n}")
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


@pytest.fixture(scope="module")
def manila() -> Profile:
    return migrated(MANILA_V01)


def ring() -> Profile:
    """Four qubits on a ring with distinct coherence, a Pauli record and an uncalibrated x."""
    return Profile.model_validate(
        toy(
            device={
                "name": "ring",
                "vendor": "test",
                "technology": "superconducting",
                "num_qubits": 4,
            },
            connectivity={"edges": [[0, 1], [1, 2], [2, 3], [1, 3]]},
            gates={
                "rz": {"virtual": True},
                "sx": {"avg_infidelity": 1e-3, "duration_ns": 35},
                "x": {},
                "cz": {"avg_infidelity": 1e-2, "duration_ns": 70},
                "ecr": {"qubits": 2, "avg_infidelity": 2e-2, "duration_ns": 300},
            },
            idle={"t1_us": 80, "t2_us": 60},
            qubits=[
                {"index": 1, "t1_us": 30, "t2_us": 50, "dephasing_rate_per_s": 2000},
                {"index": 3, "t1_us": 120, "t2_us": 20},
            ],
            calibrations=[
                {"gate": "cz", "qubits": [1, 3], "avg_infidelity": 3e-2, "duration_ns": 90},
                {"gate": "cz", "qubits": [2, 3], "pauli": [1e-3 * (k + 1) for k in range(15)]},
                {"gate": "ecr", "qubits": [0, 1], "avg_infidelity": 1.5e-2},
            ],
            readout={"p1_given_0": 0.01, "p0_given_1": 0.04, "duration_ns": 800},
            prep={"error": 0.02},
        )
    )


# per-gate channels ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "qargs"),
    [
        ("sx", (2,)),
        ("x", (0,)),  # uncalibrated: the typical 1-qubit native's channels
        ("cz", (1, 3)),
        ("cz", (3, 1)),  # reversed operands of a record, non-contiguous qubits
        ("cz", (3, 2)),  # reversed Pauli record: labels must follow the operands
    ],
)
def test_gate_superoperator_equals_the_shared_channels(name: str, qargs: tuple) -> None:
    profile = ring()
    sim = quiet_export(profile)
    sim.set_options(method="superop", enable_truncation=False)
    circuit = QuantumCircuit(4)
    gate = sim.target.operation_from_name(name)
    circuit.append(gate, list(qargs))
    circuit.save_superop()
    actual = np.asarray(sim.run(circuit).result().data()["superop"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoiseApproximationWarning)
        built = resolve_op(
            profile.table,
            name,
            qargs,
            unknown_gates="typical",
            report=Report.start(profile, "t", None),
        )
    ideal = ChannelSpec("pauli", qargs, (gates.lookup(name).unitary(),))
    expected = superoperator([ideal, *built.channels], wires=(3, 2, 1, 0))  # Qiskit order
    assert built.channels
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_multi_qubit_kraus_channels_are_reordered_for_qiskit() -> None:
    # No built-in channel is a 2-qubit non-Pauli map yet; a CX written as one Kraus operator on
    # wires (5, 2) must act with 5 as control once placed on qargs (2, 5).
    cx = gates.lookup("cx").unitary()
    channel = ChannelSpec("pauli", (5, 2), (cx,))
    actual = SuperOp(gate_error([channel], (2, 5))).data
    np.testing.assert_allclose(actual, superoperator([channel], wires=(5, 2)), atol=1e-12)


# circuits against the reference --------------------------------------------------------------


@pytest.mark.parametrize("layout", [[0, 1, 2], [4, 3, 2], [2, 1, 0]])
def test_native_circuits_with_readout_match_the_reference(manila: Profile, layout: list) -> None:
    sim = quiet_export(manila)
    sim.set_options(method="density_matrix")
    circuit = ghz(3)
    circuit.rx(0.7, 2)
    circuit.cx(2, 1)
    compiled = transpile(circuit, sim, initial_layout=layout, seed_transpiler=5)
    ours = with_exported_readout(aer_probabilities(sim, compiled, layout), sim, layout)
    ref = reference(manila, ops_of(compiled, layout), 3, layout=layout, readout=True)
    assert tvd(ours, ref) <= 1e-9


def test_directed_reversed_and_typical_gates_match_the_reference() -> None:
    profile = ring()
    sim = quiet_export(profile)
    sim.set_options(method="density_matrix")
    circuit = QuantumCircuit(4)
    circuit.sx(0)
    circuit.sx(3)
    circuit.x(2)  # uncalibrated: typical noise
    circuit.ecr(1, 0)  # directed gate against its record's direction: the definition
    circuit.cz(3, 2)  # Pauli record on the reversed pair
    circuit.rz(0.3, 3)
    circuit.cz(3, 1)
    layout = [0, 1, 2, 3]
    ours = with_exported_readout(aer_probabilities(sim, circuit, layout), sim, layout)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoiseApproximationWarning)
        ref = reference(profile, ops_of(circuit, layout), 4, readout=True)
    assert tvd(ours, ref) <= 1e-9


def test_asymmetric_readout_is_exported_exactly(manila: Profile) -> None:
    sim = quiet_export(manila)
    rows = {
        tuple(e["gate_qubits"][0]): e["probabilities"]
        for e in sim.noise_model.to_dict()["errors"]
        if e["type"] == "roerror"
    }
    for q in range(5):
        a, b = manila.table.qubit(q).readout
        assert a != b
        np.testing.assert_array_equal(rows[(q,)], [[1 - a, a], [b, 1 - b]])


def test_aer_reads_the_readout_rows_as_prepared_states(manila: Profile) -> None:
    # Aer's from_backend replaces qubit 0's P(0|1) = 0.0548 and P(1|0) = 0.0158 by their mean;
    # sampling separates the right orientation from the transposed one by about 80 sigma.
    sim = quiet_export(manila)
    circuit = QuantumCircuit(1, 1)
    circuit.x(0)
    circuit.measure(0, 0)
    shots = 200_000
    counts = sim.run(circuit, shots=shots, seed_simulator=7).result().get_counts()
    expected = reference(manila, [Op("x", (0,))], 1, layout=[0], readout=True)[0]
    sigma = np.sqrt(expected * (1 - expected) / shots)
    assert abs(counts.get("0", 0) / shots - expected) < 5 * sigma


def test_all_to_all_trapped_ion_device_transpiles_and_runs() -> None:
    profile = Profile.uniform(
        "ion8",
        technology="trapped_ion",
        num_qubits=8,
        one_qubit_error=2e-4,
        two_qubit_error=4e-3,
        readout_error=3e-3,
        t1_us=1e8,
        t2_us=1e6,
        one_qubit_ns=10_000,
        two_qubit_ns=200_000,
    )
    sim = quiet_export(profile)
    assert len(sim.target["rxx"]) == 8 * 7  # every ordered pair of an all-to-all device

    circuit = ghz(8)
    circuit.measure_all()
    compiled = transpile(circuit, sim, seed_transpiler=2)
    counts = sim.run(compiled, shots=4000, seed_simulator=3).result().get_counts()
    assert counts.get("0" * 8, 0) + counts.get("1" * 8, 0) > 0.9 * 4000

    sim.set_options(method="density_matrix")
    small = transpile(ghz(4), sim, initial_layout=[6, 1, 4, 0], seed_transpiler=2)
    layout = [6, 1, 4, 0]
    ours = aer_probabilities(sim, small, layout)
    ref = reference(profile, ops_of(small, layout), 4, layout=layout, readout=False)
    assert tvd(ours, ref) <= 1e-9


@pytest.mark.timing
def test_large_all_to_all_exports_fast() -> None:
    # Eight entanglers on 48 qubits: building every ordered pair's channels took 47 s.
    profile = Profile.uniform(
        "ion48",
        technology="trapped_ion",
        num_qubits=48,
        one_qubit_error=1e-4,
        two_qubit_error=2e-3,
        t1_us=1e7,
        t2_us=1e6,
        two_qubit_ns=200_000,
    )
    start = time.perf_counter()
    sim = quiet_export(profile)
    assert time.perf_counter() - start < 10
    assert len(sim.target["cz"]) == 48 * 47


@pytest.mark.slow
@pytest.mark.parametrize("ref", [info.id for info in nv.profiles()])
def test_every_bundled_profile_exports_transpiles_and_runs(ref: str) -> None:
    profile = nv.load(ref)
    sim = quiet_export(profile)
    circuit = ghz(4)
    circuit.measure_all()
    compiled = transpile(circuit, sim, seed_transpiler=2)
    counts = sim.run(compiled, shots=1000, seed_simulator=1).result().get_counts()
    assert counts.get("0000", 0) + counts.get("1111", 0) > 800


def test_natives_aer_lacks_run_as_labelled_unitaries_with_their_noise() -> None:
    profile = Profile.uniform(
        "grid",
        technology="superconducting",
        num_qubits=2,
        one_qubit_error=1e-3,
        two_qubit_error=2e-2,
        t1_us=50,
        two_qubit_ns=100,
    )
    sim = quiet_export(profile)
    sim.set_options(method="density_matrix")
    circuit = QuantumCircuit(2)
    circuit.sx(0)
    circuit.append(iSwapGate(), [0, 1])
    ours = aer_probabilities(sim, circuit, [0, 1])
    ref = reference(profile, [Op("sx", (0,)), Op("iswap", (0, 1))], 2, readout=False)
    assert tvd(ours, ref) <= 1e-9


def test_ms_exports_as_rxx_with_the_ms_noise() -> None:
    profile = Profile.model_validate(
        toy(
            device={"name": "ions", "vendor": "test", "technology": "trapped_ion", "num_qubits": 3},
            connectivity="all_to_all",
            gates={
                "rz": {"virtual": True},
                "rx": {"avg_infidelity": 5e-4},
                "ry": {"avg_infidelity": 5e-4},
                "ms": {"avg_infidelity": 1e-2},
            },
        )
    )
    sim = quiet_export(profile)
    sim.set_options(method="density_matrix")
    compiled = transpile(ghz(3), sim, seed_transpiler=1)
    assert "rxx" in compiled.count_ops()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoiseApproximationWarning)
        ref = reference(profile, ops_of(compiled, [0, 1, 2]), 3, readout=False)
    assert tvd(aer_probabilities(sim, compiled, [0, 1, 2]), ref) <= 1e-9
    assert any(a.what == "gate ms" for a in sim.report.approximated)


# target and validation -----------------------------------------------------------------------


def test_transpile_routes_around_a_disabled_pair() -> None:
    profile = Profile.model_validate(
        toy(
            device={
                "name": "sq",
                "vendor": "test",
                "technology": "superconducting",
                "num_qubits": 4,
            },
            connectivity={"edges": [[0, 1], [1, 2], [2, 3], [3, 0]]},
            calibrations=[{"gate": "cz", "qubits": [0, 1], "disabled": True}],
        )
    )
    sim = quiet_export(profile)
    assert {(0, 1), (1, 0)}.isdisjoint(sim.target["cz"])
    circuit = ghz(2)
    circuit.measure_all()
    compiled = transpile(circuit, sim, initial_layout=[0, 1], seed_transpiler=4)
    used = {
        tuple(sorted(compiled.find_bit(q).index for q in i.qubits))
        for i in compiled.data
        if i.operation.num_qubits == 2
    }
    assert used and (0, 1) not in used
    assert sim.run(compiled, shots=10).result().success


_LINE_GATES = {
    "rz": {"virtual": True},
    "sx": {"avg_infidelity": 1e-3, "duration_ns": 35},
    "x": {"avg_infidelity": 1e-3, "duration_ns": 35},
    "cz": {"avg_infidelity": 1e-2, "duration_ns": 70},
}
_QUBIT_0_UNUSABLE = {
    "qubit off": {"qubits": [{"index": 0, "disabled": True}]},
    "1q gates off": {
        "calibrations": [
            {"gate": "sx", "qubits": [0], "disabled": True},
            {"gate": "x", "qubits": [0], "disabled": True},
        ]
    },
    "only pair off": {"calibrations": [{"gate": "cz", "qubits": [0, 1], "disabled": True}]},
    "sx off": {"calibrations": [{"gate": "sx", "qubits": [0], "disabled": True}]},
}


@pytest.mark.parametrize("level", [0, 1, 2, 3])
@pytest.mark.parametrize("case", list(_QUBIT_0_UNUSABLE))
def test_suggested_layout_transpiles_around_disabled_parts_at_every_level(
    case: str, level: int
) -> None:
    # Level 0 places circuit qubit i on physical qubit i whatever the Target allows (Qiskit's
    # own Targets fail the same way), so the supported route there is an explicit layout.
    profile = Profile.model_validate(
        toy(
            device={
                "name": "line",
                "vendor": "test",
                "technology": "superconducting",
                "num_qubits": 4,
            },
            connectivity={"edges": [[0, 1], [1, 2], [2, 3]]},
            gates=_LINE_GATES,
            **_QUBIT_0_UNUSABLE[case],
        )
    )
    sim = quiet_export(profile)
    circuit = QuantumCircuit(2)
    circuit.h([0, 1])
    circuit.cx(0, 1)
    circuit.measure_all()
    layout = list(profile.suggest_layout(2).values())
    compiled = transpile(
        circuit, sim, initial_layout=layout, optimization_level=level, seed_transpiler=3
    )
    used = {compiled.find_bit(q).index for i in compiled.data for q in i.qubits}
    assert used and 0 not in used
    assert sim.run(compiled, shots=10).result().success


_QUBIT_0_SHORT = {
    "sx re-enabled elsewhere": {
        "gates": {**_LINE_GATES, "sx": {"avg_infidelity": 2e-3, "disabled": True}},
        "calibrations": [{"gate": "sx", "qubits": [q], "disabled": False} for q in (1, 2)],
    },
    "rx off, sx and ry left": {
        "gates": {
            "sx": {"avg_infidelity": 2e-3},
            "ry": {"avg_infidelity": 2e-3},
            "rx": {"avg_infidelity": 1e-3},
            "cz": {"avg_infidelity": 1e-2},
        },
        "calibrations": [{"gate": "rx", "qubits": [0], "disabled": True}],
        "qubits": [{"index": q, "readout": {"error": e}} for q, e in enumerate([1e-3, 0.2, 0.3])],
    },
}


@pytest.mark.parametrize("level", [0, 1, 2, 3])
@pytest.mark.parametrize("case", list(_QUBIT_0_SHORT))
def test_suggested_layout_transpiles_around_a_qubit_missing_a_native(case: str, level: int) -> None:
    profile = Profile.model_validate(
        toy(
            device={
                "name": "tri",
                "vendor": "test",
                "technology": "superconducting",
                "num_qubits": 3,
            },
            connectivity="all_to_all",
            **_QUBIT_0_SHORT[case],
        )
    )
    sim = quiet_export(profile)
    circuit = QuantumCircuit(2)
    circuit.h([0, 1])
    circuit.cx(0, 1)
    circuit.measure_all()
    layout = list(profile.suggest_layout(2).values())
    compiled = transpile(
        circuit, sim, initial_layout=layout, optimization_level=level, seed_transpiler=3
    )
    assert sim.run(compiled, shots=10).result().success


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_a_qubit_without_x_is_used_only_when_needed_and_still_transpiles(level: int) -> None:
    profile = Profile.model_validate(
        toy(
            device={
                "name": "tri",
                "vendor": "test",
                "technology": "superconducting",
                "num_qubits": 3,
            },
            connectivity="all_to_all",
            gates=_LINE_GATES,
            calibrations=[{"gate": "x", "qubits": [0], "disabled": True}],
            qubits=[{"index": 0, "readout": {"error": 1e-4}}],
        )
    )
    sim = quiet_export(profile)
    assert 0 not in profile.suggest_layout(2).values()
    with pytest.warns(nv.NoiseVaultWarning, match=r"qubit 0 \(x disabled\)"):
        layout = list(profile.suggest_layout(3).values())
    circuit = QuantumCircuit(3)
    circuit.h([0, 1, 2])
    circuit.x(range(3))
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.measure_all()
    compiled = transpile(
        circuit, sim, initial_layout=layout, optimization_level=level, seed_transpiler=3
    )
    assert sim.run(compiled, shots=10).result().success


def test_target_carries_errors_and_durations_per_locus(manila: Profile) -> None:
    sim = quiet_export(manila)
    target = sim.target
    record = next(r for r in manila.calibrations if (r.gate, r.qubits) == ("cx", (1, 2)))
    assert target["cx"][(1, 2)].duration == pytest.approx(record.duration_ns * 1e-9)
    assert target["cx"][(1, 2)].error == pytest.approx(record.avg_infidelity, rel=1e-9)
    assert (0, 2) not in target["cx"]
    a, b = manila.table.qubit(3).readout
    assert target["measure"][(3,)].error == pytest.approx((a + b) / 2)
    assert target["rz"][(0,)].error == 0.0


_TRANSPILE_FIRST = (
    "transpile the circuit for this simulator first: from qiskit import transpile;"
    " sim.run(transpile(circuit, sim, seed_transpiler=0))"
)


def test_run_rejects_an_untranspiled_circuit_with_the_fix(manila: Profile) -> None:
    sim = quiet_export(manila)
    with pytest.raises(
        CircuitNotNativeError,
        match=r"h on qubit 0 is not.*transpile\(circuit, sim, seed_transpiler=0\)",
    ) as caught:
        sim.run(ghz(2))
    assert caught.value.hint == _TRANSPILE_FIRST
    wrong_pair = QuantumCircuit(3, name="wrong_pair")
    wrong_pair.cx(0, 2)
    with pytest.raises(CircuitNotNativeError) as caught:
        sim.run(wrong_pair)
    assert caught.value.message == (
        "circuit 'wrong_pair': cx on qubits 0-2 is not available on ibm_manila (the device"
        " does not provide cx on that locus)"
    )
    assert caught.value.hint == _TRANSPILE_FIRST
    assert sim.run(transpile(ghz(2), sim, seed_transpiler=0), shots=10).result().success


def _two_qubits(*ops: str) -> QuantumCircuit:
    circuit = QuantumCircuit(2, 2, name="two")
    circuit.h(0)
    for op in ops:
        if op == "cx":
            circuit.cx(0, 1)
        elif op == "save":
            circuit.save_statevector()
        elif op == "delay":
            circuit.delay(Parameter("t"), 0, unit="ns")
        else:
            circuit.measure(range(2), range(2))
    return circuit


def _three_qubits(**sections: Any) -> Profile:
    device = {"name": "tri", "vendor": "test", "technology": "superconducting", "num_qubits": 3}
    return Profile.model_validate(toy(device=device, gates=_LINE_GATES, **sections))


_LINE = {"edges": [[0, 1], [1, 2]]}


@pytest.mark.parametrize(
    ("profile", "circuit", "failure"),
    [
        (
            Profile.model_validate(
                toy(
                    device={
                        "name": "pair",
                        "vendor": "test",
                        "technology": "superconducting",
                        "num_qubits": 2,
                    },
                    connectivity="all_to_all",
                    gates=_LINE_GATES,
                    qubits=[{"index": 1, "disabled": True}],
                )
            ),
            _two_qubits("cx", "measure"),
            BaseException,  # Qiskit panics in Rust (pyo3 PanicException)
        ),
        (
            _three_qubits(connectivity="all_to_all", qubits=[{"index": 0, "disabled": True}]),
            _two_qubits("cx", "measure"),
            None,
        ),
        (
            _three_qubits(
                connectivity=_LINE,
                calibrations=[
                    {"gate": "sx", "qubits": [0], "disabled": True},
                    {"gate": "x", "qubits": [0], "disabled": True},
                ],
            ),
            _two_qubits("cx", "measure"),
            TranspilerError,
        ),
        (_three_qubits(connectivity=_LINE), _two_qubits("cx", "save"), TranspilerError),
        (_three_qubits(connectivity=_LINE), _two_qubits("delay"), CircuitNotNativeError),
    ],
    ids=["one enabled qubit", "a disabled qubit", "a qubit without 1q gates", "save", "delay"],
)
def test_the_transpile_fix_is_offered_only_when_it_runs(
    profile: Profile, circuit: QuantumCircuit, failure: type[BaseException] | None
) -> None:
    sim = quiet_export(profile)
    with pytest.raises(CircuitNotNativeError, match="^circuit 'two': h on qubit 0") as caught:
        sim.run(circuit)
    assert caught.value.hint == (None if failure else _TRANSPILE_FIRST)
    if failure is None:
        compiled = transpile(circuit, sim, seed_transpiler=0)
        assert sim.run(compiled, shots=10).result().success
    else:
        with pytest.raises(failure):
            sim.run(transpile(circuit, sim, seed_transpiler=0), shots=10)


def _ghz_for_a_line_of_8(n: int) -> QuantumCircuit:
    return _for_a_line_of_8(ghz(n))


def _for_a_line_of_8(circuit: QuantumCircuit) -> QuantumCircuit:
    return transpile(
        circuit,
        coupling_map=CouplingMap.from_line(8),
        basis_gates=["cx", "rz", "sx", "x"],
        seed_transpiler=1,
    )


def _bell_on(width: int, name: str) -> QuantumCircuit:
    circuit = QuantumCircuit(width, name=name)
    circuit.h(0)
    circuit.cx(0, 1)
    return circuit


def _saved_on_all(width: int) -> QuantumCircuit:
    circuit = _bell_on(width, "saved")
    circuit.save_statevector()
    return circuit


def _fails(sim: NoiseVaultSimulator, circuit: QuantumCircuit) -> None:
    with pytest.raises(TranspilerError):
        transpile(circuit, sim, seed_transpiler=0)


def _five_of_eight_qubits() -> QuantumCircuit:
    circuit = QuantumCircuit(8, 5, name="sparse")
    circuit.h(0)
    circuit.cx([0, 2, 4, 6], [2, 4, 6, 7])
    circuit.barrier()
    circuit.measure([0, 2, 4, 6, 7], range(5))
    return circuit


def _five_qubits() -> QuantumCircuit:
    circuit = QuantumCircuit(5, 5, name="dense")
    circuit.h(0)
    circuit.cx([0, 1, 2, 3], [1, 2, 3, 4])
    circuit.barrier()
    circuit.measure(range(5), range(5))
    return circuit


def _runs(sim: NoiseVaultSimulator, circuit: QuantumCircuit) -> None:
    assert sim.run(transpile(circuit, sim, seed_transpiler=0), shots=10).result().success


def _runs_on_8_qubits(circuit: QuantumCircuit) -> None:
    _runs(quiet_export(_line(8)), circuit)


_BUILD_NARROWER = (
    "transpile also counts idle qubits, so remove the idle qubits from the circuit. Then run"
    " sim.run(transpile(circuit, sim, seed_transpiler=0))"
)


@pytest.mark.parametrize(
    ("build", "message", "hint", "follow"),
    [
        (
            lambda: _ghz_for_a_line_of_8(3),
            "circuit 'ghz3' has 8 qubits but ibm_manila has 5",
            "the circuit is transpiled for a backend with 8 qubits, so transpile the original"
            " circuit for this simulator instead: sim.run(transpile(original, sim,"
            " seed_transpiler=0))",
            lambda sim: _runs(sim, ghz(3)),
        ),
        (
            lambda: _ghz_for_a_line_of_8(6),
            "circuit 'ghz6' has 8 qubits but ibm_manila has 5",
            "the circuit is transpiled for a backend with 8 qubits from a circuit with 6."
            " Transpile the original circuit for a profile with at least 6 enabled qubits (nv list"
            " shows how many qubits each profile has)",
            lambda sim: _runs_on_8_qubits(ghz(6)),
        ),
        (
            _five_of_eight_qubits,
            "circuit 'sparse' has 8 qubits but ibm_manila has 5",
            _BUILD_NARROWER,
            lambda sim: _runs(sim, _five_qubits()),
        ),
        (
            lambda: QuantumCircuit(6, name="empty"),
            "circuit 'empty' has 6 qubits but ibm_manila has 5",
            _BUILD_NARROWER,
            lambda sim: _runs(sim, QuantumCircuit(0)),
        ),
        (
            lambda: ghz(6),
            "circuit 'ghz6' has 6 qubits but ibm_manila has 5",
            "the circuit acts on 6 qubits, so run the circuit on a profile with at least 6"
            " enabled qubits (nv list shows how many qubits each profile has)",
            lambda sim: _runs_on_8_qubits(ghz(6)),
        ),
        (
            lambda: _saved_on_all(6),
            "circuit 'saved' has 6 qubits but ibm_manila has 5",
            None,
            lambda sim: _fails(sim, _saved_on_all(2)),
        ),
        (
            lambda: _for_a_line_of_8(_bell_on(6, "bell")),
            "circuit 'bell' has 8 qubits but ibm_manila has 5",
            "the circuit is transpiled for a backend with 8 qubits from a circuit with 6."
            " Transpile the original circuit for a profile with at least 6 enabled qubits (nv list"
            " shows how many qubits each profile has)",
            lambda sim: _runs_on_8_qubits(_bell_on(6, "bell")),
        ),
    ],
    ids=[
        "transpiled elsewhere, fits",
        "transpiled elsewhere, too many",
        "idle qubits",
        "no operations",
        "too many",
        "saved on idle qubits",
        "transpiled elsewhere from idle qubits",
    ],
)
def test_a_circuit_wider_than_the_device_gets_a_step_that_can_work(
    build, message, hint, follow
) -> None:
    sim = quiet_export(nv.load("ibm_manila"))
    with pytest.raises(CircuitNotNativeError) as caught:
        sim.run(build())
    assert caught.value.message == message
    assert caught.value.hint == hint
    follow(sim)


def _chain_on(width: int, length: int) -> QuantumCircuit:
    circuit = QuantumCircuit(width, length, name="chain")
    circuit.h(0)
    for q in range(length - 1):
        circuit.cx(q, q + 1)
    circuit.measure(range(length), range(length))
    return circuit


@pytest.mark.parametrize(
    ("profile", "hint"),
    [
        (
            _three_qubits(connectivity="all_to_all", qubits=[{"index": 2, "disabled": True}]),
            "the circuit acts on 3 qubits, so run the circuit on a profile with at least 3 enabled"
            " qubits (nv list shows how many qubits each profile has)",
        ),
        (
            _three_qubits(
                connectivity=_LINE,
                calibrations=[{"gate": "cz", "qubits": [0, 1], "disabled": True}],
            ),
            None,
        ),
    ],
    ids=["too few enabled qubits", "no chain of 3"],
)
def test_a_wide_circuit_gets_no_narrowing_step_that_cannot_run(
    profile: Profile, hint: str | None
) -> None:
    sim = quiet_export(profile)
    with pytest.raises(CircuitNotNativeError, match="^circuit 'chain' has 4 qubits") as caught:
        sim.run(_chain_on(4, 3))
    assert caught.value.hint == hint
    with pytest.raises(TranspilerError):
        transpile(_chain_on(3, 3), sim, seed_transpiler=0)


_IDLE = "idle time outside explicit delays"


def test_the_report_names_alap_scheduling_only_when_every_instruction_has_a_duration() -> None:
    timed = quiet_export(nv.load("ibm_manila"))
    assert f"{_IDLE} (insert delays with transpile(circuit, sim, scheduling_method='alap'))" in (
        timed.report.omitted
    )
    scheduled = transpile(_chain_on(3, 3), timed, scheduling_method="alap", seed_transpiler=0)
    assert any(i.operation.name == "delay" for i in scheduled.data)
    untimed = quiet_export(_three_qubits(connectivity=_LINE))
    assert _IDLE in untimed.report.omitted
    assert not any(e.startswith(f"{_IDLE} (") for e in untimed.report.omitted)
    with pytest.raises(TranspilerError, match="Duration of"):
        transpile(_chain_on(3, 3), untimed, scheduling_method="alap", seed_transpiler=0)


# reset and delays ----------------------------------------------------------------------------


def test_reset_flips_with_the_preparation_error() -> None:
    profile = ring()
    sim = quiet_export(profile, readout=False)
    sim.set_options(method="density_matrix")
    circuit = QuantumCircuit(4)
    circuit.reset(2)
    assert aer_probabilities(sim, circuit, [2]) == pytest.approx([0.98, 0.02], abs=1e-12)


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_scheduling_gives_a_reset_the_profile_reset_duration(level: int) -> None:
    profile = nv.load("ibm_manila")
    sim = quiet_export(profile)
    circuit = QuantumCircuit(1, 1)
    circuit.x(0)
    circuit.reset(0)
    circuit.measure(0, 0)
    scheduled = transpile(
        circuit,
        sim,
        initial_layout=[3],
        scheduling_method="alap",
        optimization_level=level,
        seed_transpiler=1,
    )
    x, reset = (profile.table.gate(name, (3,)) for name in ("x", "reset"))
    readout = next(r.readout for r in profile.qubits if r.index == 3)
    assert reset.duration_ns == pytest.approx(5514.66666667)
    expected_ns = x.duration_ns + reset.duration_ns + readout.duration_ns
    assert scheduled.estimate_duration(sim.target, unit="s") == pytest.approx(expected_ns * 1e-9)


def _prepared(gates: dict[str, Any], *preps: dict[str, float] | None) -> Profile:
    return Profile.model_validate(
        toy(
            device={
                "name": "prep",
                "vendor": "test",
                "technology": "superconducting",
                "num_qubits": len(preps),
            },
            connectivity="all_to_all",
            gates=gates,
            readout={"error": 0.0, "duration_ns": 800},
            qubits=[{"index": q, "prep": p} for q, p in enumerate(preps) if p is not None],
        )
    )


@pytest.mark.parametrize("level", [2, 3])
def test_transpile_places_a_reset_on_the_qubit_with_the_lower_preparation_error(level: int) -> None:
    gates = {
        "x": {"avg_infidelity": 1e-3, "duration_ns": 10},
        "h": {"avg_infidelity": 1e-3, "duration_ns": 10},
        "reset": {"duration_ns": 1000},
    }
    sim = quiet_export(_prepared(gates, {"error": 0.04}, {"error": 0.0}))
    circuit = QuantumCircuit(1, 1)
    circuit.x(0)
    circuit.reset(0)
    circuit.measure(0, 0)
    placed = transpile(circuit, sim, optimization_level=level, seed_transpiler=19)
    assert placed.layout.initial_index_layout(filter_ancillas=True) == [1]
    assert sim.run(placed, shots=10_000, seed_simulator=19).result().get_counts() == {"0": 10_000}


def test_a_reset_keeps_its_preparation_error_without_a_duration() -> None:
    profile = _prepared({"x": {"avg_infidelity": 1e-3}}, {"error": 0.04}, {"error": 0.0}, None)
    reset = quiet_export(profile).target["reset"]
    assert {q: None if p is None else (p.error, p.duration) for (q,), p in reset.items()} == {
        0: (0.04, None),
        1: (0.0, None),
        2: None,
    }


def _turned_off(gate: str, *qubits: int) -> Profile:
    return Profile.model_validate(
        toy(
            device={
                "name": "two",
                "vendor": "test",
                "technology": "superconducting",
                "num_qubits": 2,
            },
            connectivity="all_to_all",
            gates={
                **_LINE_GATES,
                "x": {"avg_infidelity": 0.0, "duration_ns": 35},
                "reset": {"duration_ns": 1000},
                gate: {},
            },
            readout={"error": 0.0, "duration_ns": 800},
            calibrations=[{"gate": gate, "qubits": [q], "disabled": True} for q in qubits],
        )
    )


def _probe(gate: str, q: int) -> QuantumCircuit:
    circuit = QuantumCircuit(2, 1, name="probe")
    circuit.x(q)
    if gate == "reset":
        circuit.reset(q)
    elif gate == "delay":
        circuit.delay(100, q, unit="ns")
    circuit.measure(q, 0)
    return circuit


_SINGLE_QUBIT_OPS = ["measure", "reset", "delay"]


@pytest.mark.parametrize("gate", _SINGLE_QUBIT_OPS)
def test_a_measure_reset_or_delay_the_profile_disables_on_a_qubit_is_refused_there(gate) -> None:
    sim = quiet_export(_turned_off(gate, 0))
    assert list(sim.target[gate]) == [(1,)]
    assert sim.run(_probe(gate, 1), shots=10).result().get_counts() == {
        "0" if gate == "reset" else "1": 10
    }
    message = f"^circuit 'probe': {gate} on qubit 0 is disabled in this profile$"
    with pytest.raises(DisabledGateError, match=message):
        sim.run(_probe(gate, 0), shots=10)
    with pytest.raises(TranspilerError):
        transpile(_probe(gate, 0), sim, initial_layout=[0, 1], optimization_level=0)


@pytest.mark.parametrize("gate", _SINGLE_QUBIT_OPS)
def test_a_measure_reset_or_delay_the_profile_disables_everywhere_is_refused(gate) -> None:
    sim = quiet_export(_turned_off(gate, 0, 1))
    assert gate not in sim.target.operation_names
    message = f"^circuit 'probe': {gate} on qubit 1 is disabled in this profile$"
    with pytest.raises(DisabledGateError, match=message):
        sim.run(_probe(gate, 1), shots=10)
    with pytest.raises(TranspilerError):
        transpile(_probe(gate, 1), sim, optimization_level=0)


def _target_entries(sim: NoiseVaultSimulator) -> dict[tuple[str, Any], Any]:
    return {
        (name, qargs): None if props is None else (props.error, props.duration)
        for name in sim.target.operation_names
        for qargs, props in sim.target[name].items()
    }


def _noise_entries(sim: NoiseVaultSimulator) -> list[dict[str, Any]]:
    """The noise model's errors without their random ids."""
    errors = sim.noise_model.to_dict(serializable=True)["errors"]
    return [{key: value for key, value in error.items() if key != "id"} for error in errors]


@pytest.mark.parametrize("gate", _SINGLE_QUBIT_OPS)
def test_disabling_a_measure_reset_or_delay_on_a_qubit_changes_only_that_entry(gate) -> None:
    turned_off, allowed = quiet_export(_turned_off(gate, 0)), quiet_export(_turned_off(gate))
    expected = _target_entries(allowed)
    del expected[(gate, (0,))]
    assert _target_entries(turned_off) == expected
    assert _noise_entries(turned_off) == _noise_entries(allowed)


def test_scheduling_leaves_out_a_delay_the_profile_disables() -> None:
    circuit = QuantumCircuit(2)
    circuit.x(0)
    for _ in range(3):
        circuit.sx(1)
    circuit.measure_all()

    def delayed(profile: Profile) -> set[int]:
        sim = quiet_export(profile)
        scheduled = transpile(
            circuit, sim, initial_layout=[0, 1], optimization_level=0, scheduling_method="alap"
        )
        sim.run(scheduled, shots=1)
        return {scheduled.find_bit(i.qubits[0]).index for i in scheduled.data if i.name == "delay"}

    assert delayed(_turned_off("delay")) == {0}
    assert delayed(_turned_off("delay", 0)) == set()


def _delayed(prepare: str, ns: float) -> QuantumCircuit:
    circuit = QuantumCircuit(4)
    getattr(circuit, prepare)(1)
    if ns:
        circuit.delay(ns, 1, unit="ns")
    circuit.save_density_matrix([1])
    return circuit


def test_delays_relax_and_dephase() -> None:
    profile = ring()
    sim = quiet_export(profile)
    sim.set_options(method="density_matrix")
    q1 = profile.table.qubit(1)
    ns = 4000.0

    def rho(prepare: str, delay: float) -> np.ndarray:
        result = sim.run(_delayed(prepare, delay)).result()
        return np.asarray(result.data()["density_matrix"])

    excited = rho("x", ns)[1, 1] / rho("x", 0)[1, 1]
    assert excited == pytest.approx(np.exp(-ns / q1.t1_ns), rel=1e-9)
    coherence = abs(rho("sx", ns)[0, 1]) / abs(rho("sx", 0)[0, 1])
    dephasing = 1 - 2 * q1.dephasing_rate_per_s * ns * 1e-9
    assert coherence == pytest.approx(np.exp(-ns / q1.t2_ns) * dephasing, rel=1e-9)


def test_delays_match_aers_relaxation_pass_without_extra_dephasing(manila: Profile) -> None:
    sim = quiet_export(manila)
    sim.set_options(method="density_matrix")
    circuit = _delayed("sx", 3000)
    t1s = [manila.table.qubit(q).t1_ns * 1e-9 for q in range(4)]
    t2s = [
        min(manila.table.qubit(q).t2_ns, 2 * manila.table.qubit(q).t1_ns) * 1e-9 for q in range(4)
    ]
    from qiskit.circuit import Delay

    relaxed = PassManager([RelaxationNoisePass(t1s, t2s, op_types=Delay)]).run(circuit)
    plain = AerSimulator(method="density_matrix", noise_model=sim.noise_model)
    ours = np.asarray(sim.run(circuit).result().data()["density_matrix"])
    aers = np.asarray(plain.run(relaxed).result().data()["density_matrix"])
    np.testing.assert_allclose(ours, aers, rtol=0, atol=1e-12)
    assert abs(ours[0, 1]) < abs(
        np.asarray(plain.run(circuit).result().data()["density_matrix"])[0, 1]
    )


def test_scheduled_circuits_get_idle_relaxation() -> None:
    # Scheduling against this Target (no dt) writes delays in seconds: they must relax too.
    sim = quiet_export(ring(), readout=False)
    sim.set_options(method="density_matrix")
    layout = [0, 1, 3]
    plain = transpile(ghz(3), sim, initial_layout=layout, seed_transpiler=1)
    scheduled = transpile(
        ghz(3), sim, initial_layout=layout, scheduling_method="alap", seed_transpiler=1
    )
    delays = [i.operation for i in scheduled.data if i.operation.name == "delay"]
    assert delays and {d.unit for d in delays} == {"s"}

    def purity(circuit: QuantumCircuit) -> float:
        circuit = circuit.copy()
        circuit.save_density_matrix(layout)
        rho = np.asarray(sim.run(circuit).result().data()["density_matrix"])
        return float(np.real(np.trace(rho @ rho)))

    assert purity(scheduled) < purity(plain) - 1e-3


def test_delay_in_device_ticks_says_how_to_fix_it() -> None:
    sim = quiet_export(ring())
    circuit = QuantumCircuit(1)
    circuit.delay(100, 0)
    with pytest.raises(CircuitNotNativeError) as caught:
        sim.run(circuit)
    assert caught.value.message == "delay on qubit 0 has duration 100 dt"
    assert caught.value.hint == (
        "the profile has no sample time, so give delays a time unit (s, ms, us, ns, ps), for"
        " example qc.delay(100, q, unit='ns')"
    )
    timed = QuantumCircuit(1)
    timed.delay(100, 0, unit="ns")
    assert sim.run(timed, shots=1).result().success


def test_unbound_delay_duration_says_to_bind_it_first() -> None:
    sim = quiet_export(ring())
    t = Parameter("t")
    circuit = QuantumCircuit(1)
    circuit.delay(t, 0, unit="ns")
    with pytest.raises(CircuitNotNativeError) as caught:
        sim.run(circuit, parameter_binds=[{t: [100.0]}])
    assert caught.value.message == "delay on qubit 0 has the unbound duration t"
    assert caught.value.hint == (
        "delays relax before parameter_binds apply, so bind delay durations before run:"
        " sim.run(circuit.assign_parameters({...}))"
    )
    bound = circuit.assign_parameters({t: 100.0})
    assert sim.run(bound, shots=1).result().success


def _coherence_on_qubit_0(**coherence: float) -> Profile:
    return Profile.model_validate(
        toy(
            gates={
                "rz": {"virtual": True},
                "x": {"avg_infidelity": 0},
                "sx": {"avg_infidelity": 0},
                "cz": {"avg_infidelity": 1e-2, "duration_ns": 70},
            },
            qubits=[{"index": 0, **coherence}],
            prep={"error": 0},
        )
    )


def test_delays_report_each_qubit_without_relaxation_data_as_unknown() -> None:
    sim = quiet_export(_coherence_on_qubit_0(t1_us=50, t2_us=40), readout=False)
    sim.set_options(method="density_matrix")
    assert sim.report.unknown == []
    circuit = QuantumCircuit(3)
    circuit.x([0, 1])
    circuit.delay(100_000, [0, 1], unit="ns")
    decayed = np.exp(-100_000 / 50_000)
    assert aer_probabilities(sim, circuit, [0, 1]) == pytest.approx(
        [0, 1 - decayed, 0, decayed], abs=1e-12
    )
    sim.run(circuit).result()
    sim.run([circuit, circuit]).result()
    unknown_1 = "T1 and T2 of qubit 1 (no delay relaxation)"
    assert sim.report.unknown == [unknown_1]
    assert "delay: thermal relaxation and dephasing over its duration" in sim.report.exact
    other = QuantumCircuit(3)
    other.delay(500, 2, unit="ns")
    sim.run(other).result()
    assert sim.report.unknown == [unknown_1, "T1 and T2 of qubit 2 (no delay relaxation)"]
    assert f"unknown (no noise applied): {unknown_1}, T1 and T2 of qubit 2" in (
        sim.report.summary()
    )


@pytest.mark.parametrize(
    ("coherence", "factor"),
    [
        ({"t1_us": 50}, np.exp(-10_000 / (2 * 50_000))),
        ({"t2_us": 40}, np.exp(-10_000 / 40_000)),
        ({"dephasing_rate_per_s": 2000}, 1 - 2 * 2000 * 10_000e-9),
    ],
    ids=["t1-only", "t2-only", "dephasing-only"],
)
def test_delays_relax_with_partial_coherence_data_and_report_nothing_unknown(
    coherence: dict, factor: float
) -> None:
    sim = quiet_export(_coherence_on_qubit_0(**coherence), readout=False)
    sim.set_options(method="density_matrix")
    circuit = QuantumCircuit(3)
    circuit.sx(0)
    circuit.delay(10_000, 0, unit="ns")
    circuit.save_density_matrix([0])
    rho = np.asarray(sim.run(circuit).result().data()["density_matrix"])
    assert abs(rho[0, 1]) == pytest.approx(0.5 * factor, rel=1e-9)
    assert sim.report.unknown == []


# report --------------------------------------------------------------------------------------


def _reported_profile(**extra: Any) -> Profile:
    return Profile.model_validate(
        toy(
            gates={
                "rz": {"virtual": True},
                "sx": {"avg_infidelity": 1e-3, "duration_ns": 35},
                "x": {},
                "cz": {"avg_infidelity": 1e-4, "duration_ns": 400},
                "zz": {"avg_infidelity": 1e-2},
            },
            idle={"t1_us": 20, "t2_us": 30},
            **extra,
        )
    )


def test_typical_noise_warnings_point_at_the_callers_line() -> None:
    with pytest.warns(NoiseApproximationWarning, match="x on qubit 0: ") as caught:
        to_qiskit(_reported_profile())
    assert [w.filename for w in caught] == [__file__] * len(caught)


def test_report_lists_what_the_export_did() -> None:
    profile = _reported_profile(effects=[{"type": "leakage", "gate": "cz", "prob": 1e-4}])
    with pytest.warns(NoiseApproximationWarning, match="x on qubit 0: ") as caught:
        sim = to_qiskit(profile)
    assert len([w for w in caught if "x on qubit 0: " in str(w.message)]) == 1
    report = sim.report
    assert report.framework == "qiskit" and report.options == {
        "unknown_gates": "typical",
        "readout": True,
    }
    assert any(e.startswith("gate noise") for e in report.exact)
    approximated = {a.what: a.how for a in report.approximated}
    assert approximated["gate x"] == "noise of the typical 1-qubit native gate"
    assert approximated["gate zz"] == "exported as Qiskit rzz"
    assert "effect leakage on cz" in report.omitted
    assert "readout error of qubits 0, 1 and 2" in report.unknown
    assert {(c.gate, c.qubits) for c in report.clamped} >= {("cz", (0, 1)), ("cz", (1, 2))}
    assert report.events == {}

    circuit = QuantumCircuit(3)
    circuit.x(0)
    circuit.x(0)
    circuit.x(2)
    sim.run(circuit, shots=1)
    assert report.events["typical_noise_used"]["x"] == 3


def test_unknown_gates_error_leaves_uncalibrated_natives_to_transpile_around() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", NoiseApproximationWarning)
        sim = to_qiskit(ring(), unknown_gates="error")
    assert "x" not in sim.target.operation_names
    assert (
        "native x: no error metric on qubits 0, 1, 2 and 3, and unknown_gates='error', so"
        " transpile does not use this native there (unknown_gates='typical' gives those loci"
        " the typical native's noise)"
    ) in sim.report.omitted
    circuit = QuantumCircuit(2)
    circuit.x(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    compiled = transpile(circuit, sim, seed_transpiler=1)
    assert "x" not in compiled.count_ops()
    assert sim.run(compiled, shots=10).result().success
    assert sim.report.events == {}  # no guessed noise anywhere


def test_unknown_gates_error_refuses_when_no_entangler_is_calibrated() -> None:
    profile = Profile.model_validate(
        toy(gates={"rz": {"virtual": True}, "sx": {"avg_infidelity": 1e-3}, "cz": {}})
    )
    with pytest.raises(UnsupportedDevice, match=r"no two-qubit native.*cz: no error metric"):
        to_qiskit(profile, unknown_gates="error")


_ONE_QUBIT_NATIVES = {"rz": {"virtual": True}, "sx": {"avg_infidelity": 1e-3}}
_NO_EDGES = {"edges": []}


_UNCALIBRATED = ", and unknown_gates='error', so transpile does not use this native there"
_TYPICAL_FITS = " (unknown_gates='typical' gives those loci the typical native's noise)"
_NO_METRIC = (
    ". No {}-qubit native has an error metric on any {}, so profile.to_cirq() cannot run one"
)
_CZ_LOCI = "cz: no error metric on qubits 0-1, 1-0, 1-2 and 2-1"


@pytest.mark.parametrize(
    ("sections", "unknown_gates", "ending", "hint", "fix"),
    [
        (
            {
                "gates": {**_ONE_QUBIT_NATIVES, "ms": {"avg_infidelity": 1e-2}},
                "connectivity": _NO_EDGES,
            },
            "typical",
            "compile to. The profile connectivity allows no pair of enabled qubits, so"
            " profile.to_cirq() cannot run a two-qubit gate either",
            None,
            None,
        ),
        (
            {
                "gates": {
                    **_ONE_QUBIT_NATIVES,
                    "cz": {"avg_infidelity": 1e-2},
                    "ms": {"avg_infidelity": 1e-2},
                },
                "calibrations": [{"gate": "cz", "qubits": [0, 1], "disabled": True}],
                "connectivity": _NO_EDGES,
            },
            "typical",
            "compile to (cz: disabled on every locus; ms: ms has no calibration on qubits 0-1,"
            " and the connectivity does not allow ms there). The profile allows no two-qubit"
            " native on any pair of enabled qubits, so profile.to_cirq() cannot run one either",
            None,
            None,
        ),
        (
            {"gates": {"sx": {"disabled": True}, "cz": {"avg_infidelity": 1e-2}}},
            "typical",
            "compile to (sx: disabled on every locus). The profile allows no one-qubit native on"
            " any enabled qubit, so profile.to_cirq() cannot run one either",
            None,
            None,
        ),
        (
            {"gates": {**_ONE_QUBIT_NATIVES, "cz": {}}},
            "error",
            f"compile to ({_CZ_LOCI}{_UNCALIBRATED})"
            + _NO_METRIC.format("two", "pair of enabled qubits")
            + " either",
            "give the profile a calibrated two-qubit native that Qiskit provides",
            {"cz": {"avg_infidelity": 1e-2}},
        ),
        (
            {"gates": {"sx": {}, "cz": {"avg_infidelity": 1e-2}}},
            "error",
            f"compile to (sx: no error metric on qubits 0, 1 and 2{_UNCALIBRATED})"
            + _NO_METRIC.format("one", "enabled qubit")
            + " either",
            "give the profile a calibrated one-qubit native that Qiskit provides",
            {"sx": {"avg_infidelity": 1e-3}},
        ),
        (
            {"gates": {**_ONE_QUBIT_NATIVES, "cz": {}, "swapcx": {"avg_infidelity": 1e-2}}},
            "error",
            f"compile to ({_CZ_LOCI}{_UNCALIBRATED}{_TYPICAL_FITS}; swapcx: the Qiskit export has"
            " no instruction for this native)",
            "simulate the profile with profile.to_cirq() instead, or give the profile a calibrated"
            " two-qubit native that Qiskit provides",
            {"cz": {"avg_infidelity": 1e-2}},
        ),
    ],
    ids=[
        "no pair",
        "disabled or unconnected",
        "one-qubit disabled",
        "uncalibrated",
        "one-qubit uncalibrated",
        "uncalibrated beside a native Cirq runs",
    ],
)
def test_a_refusal_names_only_next_steps_that_run(
    sections, unknown_gates, ending, hint, fix
) -> None:
    profile = Profile.model_validate(toy(**sections))
    with pytest.raises(UnsupportedDevice) as refused:
        to_qiskit(profile, unknown_gates=unknown_gates)
    message = refused.value.message
    assert message.endswith(ending), message
    assert refused.value.hint == hint
    arity = 1 if "has no one-qubit native" in message else 2
    runs = _cirq_runs_a_gate(profile, arity)
    assert runs is ("cannot run" not in message)
    assert runs is ("profile.to_cirq() instead" in (hint or ""))
    if _TYPICAL_FITS in message:
        _run_natives(profile, unknown_gates="typical")
    if fix is not None:
        fixed = Profile.model_validate(toy(**{**sections, "gates": {**sections["gates"], **fix}}))
        _run_natives(fixed, unknown_gates=unknown_gates)


def test_a_refusal_offers_cirq_for_an_ideal_native_cirq_runs() -> None:
    cirq = require("cirq")
    profile = Profile.model_validate(
        toy(
            device={**toy()["device"], "num_qubits": 2},
            connectivity="all_to_all",
            gates={"h": {"avg_infidelity": 0}, "fsim": {"qubits": 2, "virtual": True}},
            qubits=[{"index": q, "readout": {"error": 0}} for q in range(2)],
        )
    )
    with pytest.raises(UnsupportedDevice) as refused:
        to_qiskit(profile)
    assert "cannot run" not in refused.value.message
    assert refused.value.hint == (
        "simulate the profile with profile.to_cirq() instead, or give the profile a calibrated"
        " two-qubit native that Qiskit provides"
    )
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(a), cirq.FSimGate(0.5, 0.3)(a, b), cirq.measure(a, b, key="m"))
    simulator = cirq.DensityMatrixSimulator(noise=profile.to_cirq())
    assert len(simulator.run(circuit, repetitions=10).measurements["m"]) == 10


def _cirq_runs_a_gate(profile: Profile, arity: int) -> bool:
    cirq = require("cirq")
    a, b = cirq.LineQubit.range(2)
    gate = cirq.CZ(a, b) if arity == 2 else cirq.X(a)
    circuit = cirq.Circuit(gate, cirq.measure(a, b, key="m"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoiseApproximationWarning)
        try:
            simulator = cirq.DensityMatrixSimulator(noise=profile.to_cirq())
            simulator.run(circuit, repetitions=10)
        except NoiseVaultError:
            return False
    return True


def _run_natives(profile: Profile, **options: Any) -> None:
    sim = quiet_export(profile, **options)
    circuit = QuantumCircuit(2)
    circuit.sx([0, 1])
    circuit.cz(0, 1)
    circuit.measure_all()
    assert sim.run(transpile(circuit, sim, seed_transpiler=1), shots=10).result().success


@pytest.mark.parametrize(
    ("extra", "offered"),
    [
        ({"iswap": {}}, True),
        ({"swap": {}}, False),
        ({"iswap": {}, "swap": {}}, False),
    ],
    ids=["typical fits", "swap needs several entanglers", "one native blocks the other"],
)
def test_the_report_offers_typical_noise_only_when_unknown_gates_typical_runs(
    extra, offered
) -> None:
    profile = Profile.model_validate(
        toy(gates={**_ONE_QUBIT_NATIVES, "cz": {"avg_infidelity": 1e-2}, **extra})
    )
    sim = to_qiskit(profile, unknown_gates="error")
    entries = [e for e in sim.report.omitted if e.startswith(tuple(f"native {n}:" for n in extra))]
    assert len(entries) == len(extra)
    assert all(e.endswith(_TYPICAL_FITS) is offered for e in entries), entries
    if offered:
        _run_natives(profile, unknown_gates="typical")
    else:
        with pytest.raises(MissingCalibrationError):
            quiet_export(profile, unknown_gates="typical")


def test_the_report_names_a_native_that_no_listed_pair_allows() -> None:
    gates = {**_ONE_QUBIT_NATIVES, "cz": {}, "ms": {"avg_infidelity": 1e-2}}
    cz = [{"gate": "cz", "qubits": [0, 1], "avg_infidelity": 2e-2}]
    profile = Profile.model_validate(toy(gates=gates, calibrations=cz, connectivity=_NO_EDGES))
    sim = to_qiskit(profile)
    assert sorted(sim.target.operation_names) == ["cz", "delay", "measure", "reset", "rz", "sx"]
    omission = (
        "native ms: ms has no calibration on qubits 0-1, and the connectivity does not allow ms"
        " there"
    )
    assert omission in sim.report.omitted


def _line(num_qubits: int, **sections: Any) -> Profile:
    device = {"name": "line", "vendor": "test", "technology": "superconducting"}
    return Profile.model_validate(
        toy(
            device={**device, "num_qubits": num_qubits},
            connectivity={"edges": [[q, q + 1] for q in range(num_qubits - 1)]},
            **sections,
        )
    )


def test_a_saved_report_names_every_affected_qubit() -> None:
    readout = {"p1_given_0": 0.01, "p0_given_1": 0.02}
    profile = _line(
        8,
        gates={**_ONE_QUBIT_NATIVES, "x": {}, "cz": {"avg_infidelity": 1e-2}},
        qubits=[{"index": q, "readout": readout} for q in (3, 4, 5)],
    )
    report = to_qiskit(profile, unknown_gates="error").report
    saved = json.loads(json.dumps(report.to_dict()))
    assert saved["unknown"] == [
        "readout error of qubits 0, 1, 2, 6 and 7",
        "preparation (reset) error of qubits 0, 1, 2, 3, 4, 5, 6 and 7",
    ]
    assert (
        "native x: no error metric on qubits 0, 1, 2, 3, 4, 5, 6 and 7"
        + _UNCALIBRATED
        + _TYPICAL_FITS
    ) in saved["omitted"]
    summary = report.summary()
    assert (
        "unknown (no noise applied): readout error of qubits 0, 1, 2 and 2 more,"
        " preparation (reset) error of qubits 0, 1, 2 and 5 more"
    ) in summary.splitlines()
    assert "omitted: native x: no error metric on qubits 0, 1, 2 and 5 more, and" in summary


def test_the_summary_stays_short_on_a_156_qubit_device() -> None:
    report = to_qiskit(_line(156)).report
    every = "qubits " + ", ".join(map(str, range(155))) + " and 155"
    assert report.to_dict()["unknown"] == [
        f"readout error of {every}",
        f"preparation (reset) error of {every}",
    ]
    assert (
        "unknown (no noise applied): readout error of qubits 0, 1, 2 and 153 more,"
        " preparation (reset) error of qubits 0, 1, 2 and 153 more"
    ) in report.summary().splitlines()


def test_a_refusal_shortens_a_long_list_of_uncalibrated_loci() -> None:
    with pytest.raises(UnsupportedDevice) as refused:
        to_qiskit(_line(6, gates={**_ONE_QUBIT_NATIVES, "cz": {}}), unknown_gates="error")
    assert "(cz: no error metric on qubits 0-1, 1-0, 1-2 and 7 more, and" in refused.value.message


def test_readout_must_be_a_bool() -> None:
    with pytest.raises(ValueError, match="readout='symmetrize': pass True or False"):
        to_qiskit(ring(), readout="symmetrize")  # type: ignore[arg-type]


def test_effects_that_demand_modelling_are_refused() -> None:
    effect = {"type": "leakage", "gate": "cz", "prob": 1e-4, "allow": "approximate"}
    with pytest.raises(UnsupportedEffect, match="allow"):
        to_qiskit(_reported_profile(effects=[effect]))


def test_profile_method_forwards_options(manila: Profile) -> None:
    sim = manila.to_qiskit(readout=False)
    assert isinstance(sim, NoiseVaultSimulator) and sim.profile is manila
    assert "readout error (readout=False)" in sim.report.omitted
    assert not [e for e in sim.noise_model.to_dict()["errors"] if e["type"] == "roerror"]


@pytest.mark.parametrize("level", [2, 3])
def test_readout_false_places_circuits_by_the_gate_noise_alone(level: int) -> None:
    profile = Profile.model_validate(
        toy(
            device={
                "name": "two",
                "vendor": "test",
                "technology": "superconducting",
                "num_qubits": 2,
            },
            connectivity="all_to_all",
            gates=_LINE_GATES,
            calibrations=[{"gate": "x", "qubits": [1], "avg_infidelity": 0.1}],
            qubits=[
                {"index": 0, "readout": {"error": 0.49, "duration_ns": 900}},
                {"index": 1, "readout": {"error": 0.0, "duration_ns": 900}},
            ],
        )
    )
    sim = quiet_export(profile, readout=False)
    measure = sim.target["measure"]
    assert [(p.error, p.duration) for p in measure.values()] == [(0.0, pytest.approx(9e-7))] * 2
    circuit = QuantumCircuit(1, 1)
    circuit.x(0)
    circuit.measure(0, 0)
    compiled = transpile(circuit, sim, optimization_level=level, seed_transpiler=1)
    assert compiled.layout.initial_index_layout(filter_ancillas=True) == [0]
    assert quiet_export(profile).target["measure"][(0,)].error == pytest.approx(0.49)


# sqrt_iswap ----------------------------------------------------------------------------------


def test_sqrt_iswap_gate_and_its_cx_rule_are_exact() -> None:
    from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
    from qiskit.circuit.library import CXGate
    from qiskit.quantum_info import Operator

    registry = gates.GATES["sqrt_iswap"].unitary()
    assert np.allclose(Operator(SqrtISwapGate()).data, registry, rtol=0, atol=1e-12)
    assert np.allclose(Operator(SqrtISwapGate().definition).data, registry, rtol=0, atol=1e-12)
    (rule,) = [
        c for c in SessionEquivalenceLibrary.get_entry(CXGate()) if "sqrt_iswap" in c.count_ops()
    ]
    assert rule.count_ops()["sqrt_iswap"] == 2
    assert np.allclose(Operator(rule).data, Operator(CXGate()).data, rtol=0, atol=1e-12)


def _sqrt_iswap_line() -> Profile:
    """Google's gate set on three qubits; sqrt_iswap has an asymmetric Pauli error."""
    pauli = [0.0] * 15
    pauli[0], pauli[14] = 0.015, 0.004
    return Profile.model_validate(
        toy(
            gates={
                "rz": {"virtual": True},
                "r": {"avg_infidelity": 1e-3, "duration_ns": 25},
                "sqrt_iswap": {"pauli": pauli, "duration_ns": 32},
            },
            idle={"t1_us": 20, "t2_us": 15},
            readout={"p1_given_0": 0.01, "p0_given_1": 0.04},
        )
    )


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_circuits_transpile_to_sqrt_iswap_and_match_the_reference(level: int) -> None:
    from qiskit.quantum_info import Operator, random_unitary

    profile = _sqrt_iswap_line()
    sim = quiet_export(profile)
    sim.set_options(method="density_matrix")
    circuit = ghz(3)
    circuit.rzz(0.4, 1, 2)
    circuit.unitary(random_unitary(4, seed=3), [0, 1])
    layout = [0, 1, 2]
    compiled = transpile(
        circuit, sim, initial_layout=layout, optimization_level=level, seed_transpiler=4
    )
    assert set(compiled.count_ops()) <= {"r", "rz", "sqrt_iswap"}
    assert Operator.from_circuit(compiled).equiv(Operator(circuit))
    ours = with_exported_readout(aer_probabilities(sim, compiled, layout), sim, layout)
    ref = reference(profile, ops_of(compiled, layout), 3, layout=layout, readout=True)
    assert tvd(ours, ref) <= 1e-9


def test_google_profiles_export_sqrt_iswap_and_report_the_gate_count_cost() -> None:
    sim = quiet_export(nv.load("google_weber"))
    assert {"r", "rz", "sqrt_iswap"} <= set(sim.target.operation_names)
    assert "sycamore" not in sim.target.operation_names
    assert any(a.what == "gate count of transpiled circuits" for a in sim.report.approximated)
    assert (
        "native sycamore: the Qiskit export has no instruction for this native"
        in sim.report.omitted
    )
    (qiskit,) = nv.load("google_weber").check(frameworks=["qiskit"]).frameworks
    assert qiskit.passed and not qiskit.not_run


def test_the_sqrt_iswap_cost_note_names_steps_that_run() -> None:
    cirq = require("cirq")
    profile = _sqrt_iswap_line()
    sim = quiet_export(profile)
    (note,) = [a for a in sim.report.approximated if a.what == "gate count of transpiled circuits"]
    assert note.detail.endswith(
        "Build circuits in sqrt_iswap directly, or use profile.to_cirq() to compile for Google"
    )
    direct = QuantumCircuit(2)
    direct.append(SqrtISwapGate(), [0, 1])
    direct.measure_all()
    compiled = transpile(direct, sim, seed_transpiler=1)
    assert compiled.count_ops()["sqrt_iswap"] == 1
    assert sim.run(compiled, shots=10).result().success
    a, b = cirq.LineQubit.range(2)
    google = cirq.optimize_for_target_gateset(
        cirq.Circuit(cirq.CNOT(a, b), cirq.measure(a, b, key="m")),
        gateset=cirq.SqrtIswapTargetGateset(),
    )
    assert sum(op.gate == cirq.SQRT_ISWAP for op in google.all_operations()) == 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoiseApproximationWarning)
        cirq.DensityMatrixSimulator(noise=profile.to_cirq()).run(google, repetitions=10)


def _t2_above_2_t1(**gates: dict) -> Profile:
    """Qubit 0 states T2 above 2*T1; qubit 1 does not."""
    return Profile.model_validate(
        toy(
            device={
                "name": "t2",
                "vendor": "test",
                "technology": "superconducting",
                "num_qubits": 2,
            },
            connectivity={"edges": []},
            gates={"rz": {"virtual": True}, **gates},
            qubits=[
                {"index": 0, "t1_us": 10, "t2_us": 100},
                {"index": 1, "t1_us": 10, "t2_us": 15},
            ],
        )
    )


def _gate_path_t2_entries() -> list:
    profile = _t2_above_2_t1(sx={"avg_infidelity": 1e-3, "duration_ns": 35})
    report = Report.start(profile, "test", None)
    report.record_channels(gate_channels(profile.table.gate("sx", (0,)), [profile.table.qubit(0)]))
    return report.approximated


def test_delay_relaxation_reports_the_t2_clamp_like_the_gate_path() -> None:
    sim = quiet_export(_t2_above_2_t1())
    circuit = QuantumCircuit(2)
    circuit.delay(1000, [0, 1], unit="ns")
    sim.run(circuit).result()
    t2 = [a for a in sim.report.approximated if a.what.startswith("T2")]
    assert t2 == _gate_path_t2_entries()


@pytest.mark.parametrize("directed", [False, True])
def test_the_target_holds_every_pair_the_table_allows_off_the_connectivity(directed) -> None:
    profile = Profile.model_validate(
        toy(
            connectivity={"edges": [[0, 1]], "directed": directed},
            calibrations=[{"gate": "cz", "qubits": [2, 1], "avg_infidelity": 0.03}],
        )
    )
    target = quiet_export(profile).target
    pairs = [(a, b) for a in range(3) for b in range(3) if a != b]
    assert [p for p in pairs if p in target["cz"]] == [
        p for p in pairs if profile.table.allowed("cz", p)
    ]
    assert (1, 2) in target["cz"]


@pytest.mark.parametrize(
    "extra", [{}, {"cz": {"avg_infidelity": 2e-2}}], ids=["uniform", "plus-cz"]
)
def test_a_one_qubit_profile_runs_a_one_qubit_circuit(extra: dict) -> None:
    data = Profile.uniform(
        "u",
        technology="superconducting",
        num_qubits=1,
        one_qubit_error=1e-2,
        two_qubit_error=2e-2,
        readout_error=0.05,
    ).to_dict()
    data["gates"].update(extra)
    sim = Profile.model_validate(data).to_qiskit()
    assert "cz" not in sim.target.operation_names
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.measure(0, 0)
    counts = sim.run(transpile(qc, sim), shots=4000, seed_simulator=1).result().get_counts()
    assert 0.03 < counts.get("0", 0) / 4000 < 0.09
