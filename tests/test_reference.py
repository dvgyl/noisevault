from math import exp, pi

import numpy as np
import pytest
from conftest import require, toy

import noisevault as nv
from noisevault.errors import MissingCalibrationError, UnsupportedEffect
from noisevault.reference import Op, _apply, probabilities
from noisevault.report import Report


def _dense_apply(rho, kraus, wires, n):
    """Embed each Kraus operator into the full space with explicit permutations."""
    dim = 2**n
    rest = [q for q in range(n) if q not in wires]
    order = list(wires) + rest
    perm = np.zeros((dim, dim))
    for index in range(dim):
        bits = [(index >> (n - 1 - q)) & 1 for q in range(n)]
        permuted = [bits[q] for q in order]
        perm[int("".join(map(str, permuted)), 2), index] = 1
    full = [perm.T @ np.kron(k, np.eye(2 ** len(rest))) @ perm for k in kraus]
    flat = rho.reshape(dim, dim)
    return sum(k @ flat @ k.conj().T for k in full)


@pytest.mark.parametrize("wires", [(0,), (2,), (2, 0), (1, 2), (0, 1)])
def test_apply_matches_dense_embedding(wires):
    rng = np.random.default_rng(len(wires) * 10 + wires[0])
    n = 3
    a = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    rho = a @ a.conj().T
    rho /= np.trace(rho)
    d = 2 ** len(wires)
    kraus = [rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)) for _ in range(3)]
    got = _apply(rho.reshape((2,) * 6), kraus, wires, n).reshape(8, 8)
    assert np.allclose(got, _dense_apply(rho, kraus, list(wires), n))


def test_noiseless_ghz_and_readout_confusion():
    ideal = nv.Profile.uniform(
        "ideal", technology="trapped_ion", num_qubits=3, one_qubit_error=0.0, two_qubit_error=0.0
    )
    ghz = [Op("h", (0,)), Op("cx", (0, 1)), Op("cx", (1, 2))]
    probs = probabilities(ideal, ghz, 3)
    assert probs[0] == pytest.approx(0.5) and probs[7] == pytest.approx(0.5)

    noisy_readout = nv.Profile.uniform(
        "readout",
        technology="trapped_ion",
        num_qubits=1,
        one_qubit_error=0.0,
        two_qubit_error=0.0,
        readout_error=0.03,
    )
    probs = probabilities(noisy_readout, [Op("x", (0,))], 1)
    assert probs == pytest.approx([0.03, 0.97])


def test_matches_aer_from_backend_on_native_circuit():
    require("qiskit_aer")
    require("qiskit_ibm_runtime")
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import DensityMatrix
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2

    profile = nv.load("ibm_manila")
    ops = [Op("sx", (0,)), Op("rz", (0,), (0.3,)), Op("x", (1,)), Op("sx", (1,))]
    ops += [Op("cx", (1, 0)), Op("sx", (2,)), Op("cx", (1, 2))]
    ours = probabilities(profile, ops, 3, unknown_gates="error", readout=False)

    qc = QuantumCircuit(3)
    for op in ops:
        getattr(qc, op.name)(*op.params, *op.qubits)
    qc.save_density_matrix()
    noise = NoiseModel.from_backend(FakeManilaV2(), readout_error=False)
    sim = AerSimulator(method="density_matrix", noise_model=noise)
    rho = sim.run(qc).result().data()["density_matrix"]
    aer = DensityMatrix(rho).probabilities()  # little-endian: qubit 0 is the least significant
    aer = aer.reshape((2,) * 3).transpose(2, 1, 0).reshape(8)
    assert 0.5 * np.abs(ours - aer).sum() < 1e-9


# fixed-angle gates on a profile that calibrates only rotations --------------------------------

# Each rotation has its own error, so the outcome shows which calibration a fixed gate took.
_ROTATIONS = toy(
    device={"name": "rot", "vendor": "test", "technology": "superconducting", "num_qubits": 2},
    connectivity={"edges": [[0, 1]]},
    gates={
        "rz": {"avg_infidelity": 2e-3},
        "rx": {"avg_infidelity": 3e-2},
        "rxx": {"avg_infidelity": 4e-2},
        "ryy": {"avg_infidelity": 5e-2},
        "rzz": {"avg_infidelity": 6e-2},
    },
)

# Each fixed gate and the calibrated rotation it equals up to global phase.
_FIXED = {
    "sx": (Op("sx", (0,)), Op("rx", (0,), (pi / 2,))),
    "x": (Op("x", (0,)), Op("rx", (0,), (pi,))),
    "s": (Op("s", (0,)), Op("rz", (0,), (pi / 2,))),
    "t": (Op("t", (0,)), Op("rz", (0,), (pi / 4,))),
    "zz": (Op("zz", (0, 1)), Op("rzz", (0, 1), (pi / 2,))),
    "ms_xx": (Op("ms", (0, 1), (0.0, 0.0)), Op("rxx", (0, 1), (pi / 2,))),
    "ms_yy": (Op("ms", (0, 1), (pi / 2, pi / 2)), Op("ryy", (0, 1), (pi / 2,))),
}


def _sandwiched(gate: Op) -> list[Op]:
    """``gate`` between rx(pi/2) layers, so a diagonal gate's unitary shows in the outcome."""
    layer = [Op("rx", (q,), (pi / 2,)) for q in range(len(gate.qubits))]
    return [*layer, gate, *layer]


@pytest.mark.parametrize("unknown_gates", ["typical", "error"])
@pytest.mark.parametrize("case", sorted(_FIXED))
def test_a_fixed_gate_takes_the_rotation_it_equals(case, unknown_gates) -> None:
    profile = nv.Profile.model_validate(_ROTATIONS)
    gate, rotation = _FIXED[case]
    n = len(gate.qubits)
    want = probabilities(profile, _sandwiched(rotation), n, readout=False)
    got = probabilities(profile, _sandwiched(gate), n, unknown_gates=unknown_gates, readout=False)
    assert got == pytest.approx(want, abs=1e-12)


def test_an_ms_gate_equal_to_no_rotation_takes_no_rotation() -> None:
    profile = nv.Profile.model_validate(_ROTATIONS)
    with pytest.raises(MissingCalibrationError, match="ms on qubits"):
        probabilities(profile, [Op("ms", (0, 1), (0.0, pi / 2))], 2, unknown_gates="error")


def _cirq_probs(profile, case: str, n: int) -> np.ndarray:
    cirq = require("cirq")
    gate = {
        "sx": cirq.X**0.5,
        "x": cirq.X,
        "s": cirq.S,
        "t": cirq.T,
        "zz": cirq.ZZ**0.5,
        "ms_xx": cirq.ms(pi / 4),
        "ms_yy": cirq.YY**0.5,
    }[case]
    qubits = cirq.LineQubit.range(n)
    layer = [cirq.rx(pi / 2).on(q) for q in qubits]
    model = profile.to_cirq(
        layout=dict(zip(qubits, range(n), strict=True)), unknown_gates="error", readout=False
    )
    circuit = cirq.Circuit([*layer, gate.on(*qubits), *layer]).with_noise(model)
    simulator = cirq.DensityMatrixSimulator(dtype=np.complex128)
    rho = simulator.simulate(circuit, qubit_order=qubits).final_density_matrix
    return np.real(np.diag(rho))


def _pennylane_probs(profile, case: str, n: int) -> np.ndarray:
    qml = require("pennylane")
    make = {
        "sx": qml.SX,
        "x": qml.PauliX,
        "s": qml.S,
        "t": qml.T,
        "zz": lambda wires: qml.IsingZZ(pi / 2, wires=wires),
        "ms_xx": lambda wires: qml.IsingXX(pi / 2, wires=wires),
        "ms_yy": lambda wires: qml.IsingYY(pi / 2, wires=wires),
    }[case]
    wires = list(range(n))

    @qml.qnode(qml.device("default.mixed", wires=n))
    def run():
        for w in wires:
            qml.RX(pi / 2, wires=w)
        make(wires=wires)
        for w in wires:
            qml.RX(pi / 2, wires=w)
        return qml.probs(wires=wires)

    model = profile.to_pennylane(layout=wires, unknown_gates="error", readout=False)
    return np.asarray(qml.add_noise(run, model)(), dtype=float)


def _qiskit_probs(profile, case: str, n: int) -> np.ndarray:
    require("qiskit_aer")
    from qiskit import QuantumCircuit, transpile

    sim = profile.to_qiskit(unknown_gates="error", readout=False)
    sim.set_options(method="density_matrix")
    circuit = QuantumCircuit(n)
    for q in range(n):
        circuit.rx(pi / 2, q)
    {
        "sx": lambda: circuit.sx(0),
        "x": lambda: circuit.x(0),
        "s": lambda: circuit.s(0),
        "t": lambda: circuit.t(0),
        "zz": lambda: circuit.rzz(pi / 2, 0, 1),
        "ms_xx": lambda: circuit.rxx(pi / 2, 0, 1),
        "ms_yy": lambda: circuit.ryy(pi / 2, 0, 1),
    }[case]()
    for q in range(n):
        circuit.rx(pi / 2, q)
    # Level 0 keeps each gate a single native, so no rx layer merges into the gate under test.
    native = transpile(circuit, sim, initial_layout=list(range(n)), optimization_level=0)
    native.save_probabilities(list(range(n)))
    little = np.asarray(sim.run(native).result().data()["probabilities"])
    return little.reshape((2,) * n).transpose(range(n - 1, -1, -1)).reshape(-1)


_STIM_SHOTS = 200_000


def _stim_probs(profile, case: str, n: int) -> np.ndarray:
    require("stim")
    from noisevault.frameworks.stim import to_stim

    gate = {
        "sx": "SQRT_X",
        "x": "X",
        "s": "S",
        "zz": "SQRT_ZZ",
        "ms_xx": "SQRT_XX",
        "ms_yy": "SQRT_YY",
    }[case]
    targets = " ".join(map(str, range(n)))
    text = f"SQRT_X {targets}\n{gate} {targets}\nSQRT_X {targets}\nM {targets}"
    noisy = to_stim(profile, text, unknown_gates="error", readout="none")
    bits = noisy.compile_sampler(seed=0).sample(_STIM_SHOTS)
    index = bits.astype(int) @ (1 << np.arange(n)[::-1])
    return np.bincount(index, minlength=2**n) / _STIM_SHOTS


_EXPORT_PROBS = {
    "cirq": _cirq_probs,
    "pennylane": _pennylane_probs,
    "qiskit": _qiskit_probs,
    "stim": _stim_probs,
}
# T is no Clifford gate, so Stim cannot hold it.
_EXPORT_CASES = [
    (framework, case)
    for framework in sorted(_EXPORT_PROBS)
    for case in sorted(_FIXED)
    if not (framework == "stim" and case == "t")
]


@pytest.mark.parametrize(("framework", "case"), _EXPORT_CASES)
def test_every_export_agrees_with_the_reference_on_fixed_gates(framework, case) -> None:
    profile = nv.Profile.model_validate(_ROTATIONS)
    gate, _ = _FIXED[case]
    n = len(gate.qubits)
    want = probabilities(profile, _sandwiched(gate), n, unknown_gates="error", readout=False)
    got = _EXPORT_PROBS[framework](profile, case, n)
    if framework == "stim":
        # Depolarizing channels are their own Pauli twirl, so Stim samples the plain reference.
        bound = 5 * np.sqrt(want * (1 - want) / _STIM_SHOTS) + 5 / _STIM_SHOTS
        assert np.all(np.abs(got - want) <= bound), (got, want)
    else:
        assert got == pytest.approx(want, abs=1e-9)


_IDLE = toy(
    device={"name": "idle", "vendor": "test", "technology": "superconducting", "num_qubits": 3},
    connectivity={"edges": []},
    gates={"x": {"avg_infidelity": 0}, "h": {"avg_infidelity": 0}},
    qubits=[
        {"index": 0, "t1_us": 50, "t2_us": 40, "dephasing_rate_per_s": 2000},
        {"index": 1, "t1_us": 10, "t2_us": 100},
    ],
)
_NS = 20_000.0
_RAMSEY = (Op("h", (0,)), Op("delay", (0,), (_NS,)), Op("h", (0,)))


def test_a_delay_relaxes_and_dephases_its_physical_qubit() -> None:
    profile = nv.Profile.model_validate(_IDLE)
    ops = [Op("x", (0,)), Op("x", (1,)), Op("delay", (0,), (_NS,))]
    excited = exp(-_NS / 10_000)
    assert probabilities(profile, ops, 2, layout=[1, 0]) == pytest.approx(
        [0, 1 - excited, 0, excited], abs=1e-12
    )
    coherence = exp(-_NS / 40_000) * (1 - 2 * 2000 * _NS * 1e-9)
    assert probabilities(profile, _RAMSEY, 1) == pytest.approx(
        [(1 + coherence) / 2, (1 - coherence) / 2], abs=1e-12
    )


def test_a_delay_on_a_qubit_without_relaxation_data_adds_no_noise_and_is_unknown() -> None:
    profile = nv.Profile.model_validate(_IDLE)
    report = Report.start(profile, "reference", None)
    assert probabilities(profile, _RAMSEY, 1, layout=[2], report=report) == pytest.approx(
        [1, 0], abs=1e-15
    )
    assert report.to_dict()["unknown"] == ["T1 and T2 of qubit 2 (no delay relaxation)"]


def test_a_delay_clamps_t2_above_2_t1_and_reports_it() -> None:
    profile = nv.Profile.model_validate(_IDLE)
    coherence = exp(-_NS / (2 * 10_000))
    assert probabilities(profile, _RAMSEY, 1, layout=[1]) == pytest.approx(
        [(1 + coherence) / 2, (1 - coherence) / 2], abs=1e-12
    )
    report = Report.start(profile, "reference", None)
    probabilities(profile, [Op("delay", (0,), (_NS,))], 1, layout=[1], report=report)
    t2 = [(a.what, a.how) for a in report.approximated if a.what.startswith("T2")]
    assert t2 == [("T2 of qubit 1", "clamped to 2*T1")]


@pytest.mark.parametrize("name", ["measure", "reset"])
def test_measure_and_reset_still_have_no_place_in_the_reference(name) -> None:
    profile = nv.Profile.model_validate(_IDLE)
    with pytest.raises(ValueError, match=f"has no unitary for '{name}'"):
        probabilities(profile, [Op("x", (0,)), Op(name, (0,))], 1)


_REVIEW_CIRCUIT = (Op("h", (0,)), Op("delay", (0,), (50_000.0,)), Op("h", (0,)))


def _h_qubit(**coherence) -> nv.Profile:
    return nv.Profile.model_validate(
        toy(
            device={
                "name": "h",
                "vendor": "test",
                "technology": "superconducting",
                "num_qubits": 1,
            },
            connectivity={"edges": []},
            gates={"h": {"avg_infidelity": 1e-3}},
            qubits=[{"index": 0, **coherence}],
        )
    )


def test_the_reviews_circuit_gives_its_closed_form() -> None:
    h_shrink = 1 - 2 * 1e-3
    flipped = (1 - h_shrink**2 * exp(-50_000 / 1_000_000)) / 2
    assert probabilities(_h_qubit(t2_us=1000), _REVIEW_CIRCUIT, 1) == pytest.approx(
        [1 - flipped, flipped], abs=1e-12
    )


@pytest.mark.parametrize(
    "coherence",
    [{"t2_us": 1000}, {"t1_us": 30, "t2_us": 40, "dephasing_rate_per_s": 2000}],
    ids=["t2-only", "t1-t2-dephasing"],
)
def test_a_delay_matches_the_qiskit_export_through_aer(coherence) -> None:
    require("qiskit_aer")
    from qiskit import QuantumCircuit

    profile = _h_qubit(**coherence)
    sim = profile.to_qiskit()
    sim.set_options(method="density_matrix")
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.delay(50, 0, unit="us")
    circuit.h(0)
    circuit.save_probabilities([0])
    aer = np.asarray(sim.run(circuit).result().data()["probabilities"])
    assert probabilities(profile, _REVIEW_CIRCUIT, 1) == pytest.approx(aer, abs=1e-12)


def _with_effect(allow: str) -> nv.Profile:
    effect = {"type": "atom_loss", "on": "readout", "prob": 1e-3, "allow": allow}
    return nv.Profile.model_validate(toy(effects=[effect]))


def test_the_reference_leaves_out_an_omitted_effect_and_reports_it() -> None:
    profile = _with_effect("omit")
    report = Report.start(profile, "reference", None)
    got = probabilities(profile, [Op("sx", (0,))], 1, report=report)
    plain = probabilities(nv.Profile.model_validate(toy()), [Op("sx", (0,))], 1)
    assert np.array_equal(got, plain)
    assert report.to_dict()["omitted"] == ["effect atom_loss on readout"]


@pytest.mark.parametrize("allow", ["exact", "approximate"])
def test_the_reference_refuses_an_effect_that_it_cannot_leave_out(allow: str) -> None:
    with pytest.raises(UnsupportedEffect, match="the reference simulator does not model effects"):
        probabilities(_with_effect(allow), [Op("sx", (0,))], 1)
