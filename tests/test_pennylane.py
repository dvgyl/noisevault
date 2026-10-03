from __future__ import annotations

import time
import warnings
from functools import partial
from types import ModuleType

import numpy as np
import pytest
from conftest import MANILA_V01, migrated, require, toy

from noisevault.channels import ChannelSpec, superoperator
from noisevault.conversion import resolve_op
from noisevault.errors import (
    DisabledGateError,
    LayoutError,
    MissingCalibrationError,
    NoiseApproximationWarning,
    NoiseVaultError,
    UnsupportedEffect,
)
from noisevault.gates import GATES
from noisevault.profile import Profile
from noisevault.reference import Op, probabilities
from noisevault.report import Report

pytestmark = pytest.mark.filterwarnings("ignore::noisevault.errors.NoiseApproximationWarning")

GHZ = [Op("h", (0,)), Op("cx", (0, 1)), Op("cx", (1, 2))]
MIRROR_HALF = [
    Op("sx", (0,)),
    Op("rz", (0,), (0.7,)),
    Op("x", (1,)),
    Op("sx", (2,)),
    Op("cx", (0, 1)),
    Op("rz", (1,), (-1.1,)),
    Op("cx", (2, 1)),
    Op("sx", (1,)),
]


@pytest.fixture
def qml() -> ModuleType:
    return require("pennylane")


@pytest.fixture
def manila() -> Profile:
    return migrated(MANILA_V01)


def _ion() -> Profile:
    return Profile.uniform(
        "ion",
        technology="trapped_ion",
        num_qubits=5,
        one_qubit_error=3e-4,
        two_qubit_error=6e-3,
        readout_error=4e-3,
        t1_us=2e5,
        t2_us=5e3,
        one_qubit_ns=1e4,
        two_qubit_ns=2e5,
    )


def _asymmetric_toy() -> Profile:
    """Two qubits, ideal basis changes (virtual h, z family) and very different readouts."""
    data = toy(
        gates={
            "rz": {"virtual": True},
            "h": {"virtual": True},
            "sx": {"avg_infidelity": 2e-3, "duration_ns": 35},
            "cz": {"avg_infidelity": 2e-2, "duration_ns": 70},
        },
        idle={"t1_us": 20, "t2_us": 15},
        qubits=[
            {"index": 0, "readout": {"p1_given_0": 0.02, "p0_given_1": 0.11}},
            {"index": 1, "readout": {"p1_given_0": 0.07, "p0_given_1": 0.01}},
        ],
    )
    return Profile.model_validate(data)


def _mirror() -> list[Op]:
    """MIRROR_HALF then its inverse, in native gates, so the ideal output is |000>."""
    undo: list[Op] = []
    for op in reversed(MIRROR_HALF):
        if op.name == "rz":
            undo.append(Op("rz", op.qubits, (-op.params[0],)))
        elif op.name == "sx":  # sx^-1 = rz(pi) sx rz(pi) up to a global phase
            undo += [Op("rz", op.qubits, (np.pi,)), op, Op("rz", op.qubits, (np.pi,))]
        else:
            undo.append(op)
    return MIRROR_HALF + undo


def _pl_ops(qml: ModuleType, ops: list[Op], wires: list | None = None) -> None:
    classes = {
        "h": qml.Hadamard,
        "x": qml.PauliX,
        "sx": qml.SX,
        "rz": qml.RZ,
        "rx": qml.RX,
        "cx": qml.CNOT,
        "cz": qml.CZ,
    }
    for op in ops:
        targets = list(op.qubits) if wires is None else [wires[q] for q in op.qubits]
        classes[op.name](*op.params, wires=targets)


def _noisy_probs(qml, model, ops: list[Op], wires: list) -> np.ndarray:
    @qml.qnode(qml.device("default.mixed", wires=wires))
    def circuit():
        _pl_ops(qml, ops, wires)
        return qml.probs(wires=wires)

    return np.asarray(qml.add_noise(circuit, model)())


def _tvd(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.abs(np.asarray(a) - np.asarray(b)).sum())


# per-gate channels ---------------------------------------------------------------------------


def _pauli_device() -> Profile:
    """A cx whose Pauli channel is not symmetric under swapping its operands."""
    pauli = [0.0] * 15
    pauli[0], pauli[11] = 0.02, 0.005  # IX and ZI: the first letter acts on qubits[0]
    data = toy(
        device={"name": "pauli", "technology": "other", "num_qubits": 5},
        connectivity="all_to_all",
        gates={"rz": {"virtual": True}, "cx": {"qubits": 2, "pauli": pauli}},
    )
    return Profile.model_validate(data)


@pytest.mark.parametrize(
    ("device", "gate", "wires", "layout"),
    [
        ("manila", "sx", (0,), None),
        ("manila", "cx", (0, 1), None),
        ("manila", "cx", (1, 0), None),
        ("manila", "cx", (0, 1), {0: 3, 1: 2}),
        ("manila", "cx", (1, 0), {0: 3, 1: 2}),
        ("manila", "h", (1,), {0: 3, 1: 4}),
        ("pauli", "cx", (0, 1), None),
        ("pauli", "cx", (1, 0), {0: 4, 1: 1}),
    ],
    ids=[
        "sx",
        "cx",
        "cx-reversed",
        "cx-layout",
        "cx-layout-reversed",
        "typical-layout",
        "pauli-cx",
        "pauli-cx-layout-reversed",
    ],
)
def test_gate_superoperators_equal_core_channels(qml, manila, device, gate, wires, layout) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    profile = manila if device == "manila" else _pauli_device()
    model = to_pennylane(profile, layout=layout, readout=False)
    pl_gate = {"sx": qml.SX, "cx": qml.CNOT, "h": qml.Hadamard}[gate](wires=list(wires))
    [tape], _ = qml.noise.add_noise(qml.tape.QuantumScript([pl_gate]), model)
    physical = tuple(wires if layout is None else (layout[w] for w in wires))
    to_physical = dict(zip(wires, physical, strict=True))

    inserted = [
        ChannelSpec("pauli", tuple(to_physical[w] for w in op.wires), tuple(op.kraus_matrices()))
        for op in tape.operations[1:]
    ]
    assert [op.name for op in tape.operations[1:]] == ["QubitChannel"] * len(inserted)
    report = Report.start(profile, "t", None)
    core = resolve_op(profile.table, gate, physical, unknown_gates="typical", report=report)
    assert core.channels
    expected = superoperator(core.channels, physical)
    assert np.abs(superoperator(inserted, physical) - expected).max() < 1e-10


def test_virtual_rz_gets_no_channels(qml, manila) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    tape = qml.tape.QuantumScript([qml.RZ(0.3, wires=0), qml.PhaseShift(0.2, wires=1)])
    [noisy], _ = qml.noise.add_noise(tape, to_pennylane(manila, readout=False))
    assert [op.name for op in noisy.operations] == ["RZ", "PhaseShift"]


# circuits against the reference --------------------------------------------------------------


@pytest.mark.parametrize("readout", [True, False], ids=["readout", "no-readout"])
@pytest.mark.parametrize("circuit", ["ghz", "mirror"])
@pytest.mark.parametrize("device", ["manila", "ion"])
def test_noisy_qnode_matches_reference(qml, manila, device, circuit, readout) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    profile = manila if device == "manila" else _ion()
    ops = GHZ if circuit == "ghz" else _mirror()
    layout = [4, 3, 2] if device == "manila" else [1, 4, 0]
    model = to_pennylane(profile, layout=layout, readout=readout)

    got = _noisy_probs(qml, model, ops, [0, 1, 2])
    expected = probabilities(profile, ops, 3, layout=layout, readout=readout)
    ideal = probabilities(
        Profile.uniform(
            "ideal", technology="other", num_qubits=3, one_qubit_error=0.0, two_qubit_error=0.0
        ),
        ops,
        3,
    )
    assert _tvd(got, expected) <= 1e-9
    assert _tvd(got, ideal) > 1e-3  # the noise is really there


def test_string_wires_follow_the_layout(qml, manila) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    labels = ["anc", "data", "flag"]
    model = to_pennylane(manila, layout={"anc": 2, "data": 1, "flag": 0})
    got = _noisy_probs(qml, model, _mirror(), labels)
    expected = probabilities(manila, _mirror(), 3, layout=[2, 1, 0])
    assert _tvd(got, expected) <= 1e-9
    assert model.physical_qubit("data") == 1


def test_wires_the_layout_cannot_place_are_layout_errors(qml, manila) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    with pytest.raises(LayoutError, match="layout="):
        _noisy_probs(qml, to_pennylane(manila), GHZ, ["a", "b", "c"])
    with pytest.raises(LayoutError, match="'c' is not in the layout") as caught:
        _noisy_probs(qml, to_pennylane(manila, layout={"a": 0, "b": 1}), GHZ, ["a", "b", "c"])
    assert caught.value.hint == "add the wire: layout={..., 'c': <physical qubit>}"
    with pytest.raises(LayoutError, match="wires 0 to 1; extend the list") as caught:
        _noisy_probs(qml, to_pennylane(manila, layout=[3, 4]), GHZ, [0, 1, 2])
    assert caught.value.hint == "extend the list"
    with pytest.raises(LayoutError, match="both"):
        to_pennylane(manila, layout={"a": 0, "b": 0})


def _one_disabled() -> Profile:
    return Profile.model_validate(toy(qubits=[{"index": 1, "disabled": True}]))


def _qubit_1_disabled(num_qubits: int) -> Profile:
    device = {**toy()["device"], "num_qubits": num_qubits}
    disabled = [{"index": 1, "disabled": True}]
    return Profile.model_validate(toy(device=device, connectivity="all_to_all", qubits=disabled))


_CHAIN_OF_2 = "choose another qubit (profile.suggest_layout(2) proposes a usable chain)"
_CHAIN_OF_3 = "choose another qubit (profile.suggest_layout(3) proposes a usable chain)"


@pytest.mark.parametrize(("num_qubits", "hint"), [(2, None), (3, _CHAIN_OF_2)])
def test_a_disabled_wire_hint_needs_a_chain_as_wide_as_the_circuit(qml, num_qubits, hint) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    @qml.qnode(qml.device("default.mixed", wires=2))
    def circuit():
        qml.SX(wires=1)
        return qml.probs(wires=[0])

    with pytest.raises(LayoutError, match="marks disabled") as caught:
        qml.add_noise(circuit, to_pennylane(_qubit_1_disabled(num_qubits)))()
    assert caught.value.hint == hint


@pytest.mark.parametrize(("num_qubits", "hint"), [(3, None), (4, _CHAIN_OF_3)])
def test_a_disabled_device_wire_hint_counts_every_device_wire(qml, num_qubits, hint) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    @qml.qnode(qml.device("default.mixed", wires=3))
    def circuit():
        qml.SX(wires=0)
        return qml.probs()

    with pytest.raises(LayoutError, match="marks disabled") as caught:
        qml.add_noise(circuit, to_pennylane(_qubit_1_disabled(num_qubits)))()
    assert caught.value.hint == hint


@pytest.mark.parametrize(
    "prepare",
    [
        lambda q, wire: q.BasisState(np.array([1]), wires=[wire]),
        lambda q, wire: q.QubitDensityMatrix(np.diag([0.0, 1.0]), wires=[wire]),
    ],
    ids=["basis state", "density matrix"],
)
@pytest.mark.parametrize(
    ("wire", "match"),
    [(1, "marks disabled"), (3, "has qubits 0..2"), ("anc", "layout=")],
)
def test_noiseless_operations_are_checked_against_the_layout(qml, prepare, wire, match) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    @qml.qnode(qml.device("default.mixed", wires=[0, wire]))
    def circuit():
        prepare(qml, wire)
        return qml.probs(wires=[0])

    with pytest.raises(LayoutError, match=match):
        qml.add_noise(circuit, to_pennylane(_one_disabled(), readout=False))()


@pytest.mark.parametrize(
    ("measure", "readout"),
    [("density_matrix", True), ("density_matrix", False), ("probs", False)],
)
def test_measured_wires_are_checked_against_the_layout(qml, measure, readout) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    @qml.qnode(qml.device("default.mixed", wires=2))
    def circuit():
        qml.PauliX(0)
        return getattr(qml, measure)(wires=[1])

    with pytest.raises(LayoutError, match="marks disabled"):
        qml.add_noise(circuit, to_pennylane(_one_disabled(), readout=readout))()


_NOT_A_PRODUCT_BASIS = np.array([[1, 0, 0, 0.5], [0, -1, 0, 0], [0, 0, -1, 0], [0.5, 0, 0, 1]])


@pytest.mark.parametrize("readout", [True, False])
@pytest.mark.parametrize("measure", ["probs", "density_matrix", "hermitian"])
@pytest.mark.parametrize(
    ("wire", "layout", "match"),
    [
        (1, None, "marks disabled"),
        (3, None, "has qubits 0..2"),
        ("anc", None, r"'anc' has no integer index; pass layout="),
        ("b", {"a": 0}, r"wire 'b' is not in the layout; add the wire"),
    ],
)
def test_measurement_only_circuits_are_checked_against_the_layout(
    qml, wire, layout, match, measure, readout
) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    other = 0 if layout is None else "a"

    @qml.qnode(qml.device("default.mixed", wires=[other, wire]))
    def circuit():
        if measure == "hermitian":
            return qml.expval(qml.Hermitian(_NOT_A_PRODUCT_BASIS, wires=[other, wire]))
        return getattr(qml, measure)(wires=[wire])

    model = to_pennylane(_one_disabled(), layout=layout, readout=readout)
    with pytest.raises(LayoutError, match=match):
        qml.add_noise(circuit, model)()


# readout -------------------------------------------------------------------------------------


def test_asymmetric_readout_follows_the_profile_convention(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    model = to_pennylane(_asymmetric_toy())
    dev = qml.device("default.mixed", wires=2)

    @qml.qnode(dev)
    def prepared_01():
        return qml.probs(wires=[0, 1])

    # both qubits in |0>: qubit 0 reads 1 with P(1|0) = 0.02, qubit 1 with 0.07
    got = np.asarray(qml.add_noise(prepared_01, model)())
    assert got == pytest.approx(np.kron([0.98, 0.02], [0.93, 0.07]), abs=1e-12)


def test_tuple_wire_labels_get_their_readout(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    profile = _asymmetric_toy()
    results = []
    for a, b in ((0, 1), (("reg", 0), ("reg", 1))):

        @qml.qnode(qml.device("default.mixed", wires=[a, b]))
        def circuit(a=a, b=b):
            qml.SX(wires=[a])
            qml.CZ(wires=[a, b])
            qml.SX(wires=[b])
            return qml.probs(wires=[a, b]), qml.probs(), qml.expval(qml.X([a]) @ qml.Y([b]))

        model = to_pennylane(profile, layout={a: 0, b: 1})
        results.append([np.asarray(r) for r in qml.add_noise(circuit, model)()])
    for by_index, by_tuple in zip(*results, strict=True):
        assert by_tuple == pytest.approx(by_index, abs=1e-12)


@pytest.mark.parametrize(("a", "b"), [(0.0, 0.0), (0.7, 0.6), (0.3, 0.0)])
def test_any_confusion_matrix_is_reproduced(qml, a, b) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    data = toy(
        gates={"rz": {"virtual": True}, "rx": {"avg_infidelity": 0.0}},
        qubits=[{"index": 0, "readout": {"p1_given_0": a, "p0_given_1": b}}],
    )
    model = to_pennylane(Profile.model_validate(data))

    @qml.qnode(qml.device("default.mixed", wires=1))
    def circuit(flip):
        qml.RX(np.pi * flip, wires=0)
        return qml.probs(wires=[0])

    noisy = qml.add_noise(circuit, model)
    assert np.asarray(noisy(0.0)) == pytest.approx([1 - a, a], abs=1e-12)
    assert np.asarray(noisy(1.0)) == pytest.approx([b, 1 - b], abs=1e-12)


def test_pauli_word_readout_is_applied_in_the_measured_basis(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    profile = _asymmetric_toy()
    ops = [Op("sx", (0,)), Op("cz", (0, 1)), Op("sx", (1,)), Op("rz", (1,), (0.4,)), Op("sx", (1,))]

    @qml.qnode(qml.device("default.mixed", wires=2))
    def circuit():
        _pl_ops(qml, ops)
        return qml.expval(qml.X(0) @ qml.Z(1)), qml.expval(qml.Y(1)), qml.var(qml.X(0))

    xz, y, var_x = qml.add_noise(circuit, to_pennylane(profile))()
    signs = np.array([1, -1, -1, 1])
    in_x0 = probabilities(profile, ops + [Op("h", (0,))], 2)
    in_y1 = probabilities(profile, ops + [Op("sdg", (1,)), Op("h", (1,))], 2)
    x0 = float(in_x0 @ np.array([1, 1, -1, -1]))
    assert float(xz) == pytest.approx(float(in_x0 @ signs), abs=1e-12)
    assert float(y) == pytest.approx(float(in_y1 @ np.array([1, -1, 1, -1])), abs=1e-12)
    assert float(var_x) == pytest.approx(1 - x0**2, abs=1e-12)


def test_sums_get_readout_in_their_shared_basis_and_conflicts_are_reported(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    profile = _asymmetric_toy()
    ops = [Op("sx", (0,)), Op("cz", (0, 1)), Op("sx", (1,))]
    model = to_pennylane(profile)

    @qml.qnode(qml.device("default.mixed", wires=2))
    def circuit(observable):
        _pl_ops(qml, ops)
        return qml.expval(observable)

    noisy = qml.add_noise(circuit, model)
    probs = probabilities(profile, ops, 2)
    in_x0 = probabilities(profile, ops + [Op("h", (0,))], 2)
    x0, z0, z1 = in_x0 @ [1, 1, -1, -1], probs @ [1, 1, -1, -1], probs @ [1, -1, 1, -1]
    z_sum = float(probs @ np.array([2, 0, -2, 0]))  # Z0 + Z0 Z1
    assert float(noisy(qml.Z(0) + qml.Z(0) @ qml.Z(1))) == pytest.approx(z_sum, abs=1e-12)
    assert float(noisy(qml.X(0) + qml.Z(1))) == pytest.approx(x0 + z1, abs=1e-12)

    with pytest.warns(NoiseApproximationWarning, match="split_non_commuting"):
        mixed = float(noisy(qml.X(0) + qml.Z(0)))
    x0_ideal = probabilities(profile, ops + [Op("h", (0,))], 2, readout=False) @ [1, 1, -1, -1]
    z0_ideal = probabilities(profile, ops, 2, readout=False) @ [1, 1, -1, -1]
    assert mixed == pytest.approx(x0_ideal + z0_ideal, abs=1e-12)
    assert (
        "readout on observables not measured in one product basis"
        in model.report.to_dict()["omitted"]
    )

    split = qml.add_noise(qml.transforms.split_non_commuting(circuit), to_pennylane(profile))
    with warnings.catch_warnings():
        warnings.simplefilter("error", NoiseApproximationWarning)
        fixed = float(split(qml.X(0) + qml.Z(0)))
    assert fixed == pytest.approx(x0 + z0, abs=1e-12)


def test_repeated_pauli_terms_read_out_like_the_scaled_word(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    profile = _asymmetric_toy()
    ops = [Op("sx", (0,)), Op("cz", (0, 1)), Op("sx", (1,)), Op("rz", (1,), (0.4,)), Op("sx", (1,))]

    @qml.qnode(qml.device("default.mixed", wires=2))
    def circuit():
        _pl_ops(qml, ops)
        return (
            qml.expval(qml.X(0) + qml.X(0)),
            qml.expval(qml.dot([0.5, 0.5], [qml.Y(1), qml.Y(1)])),
        )

    xx, yy = qml.add_noise(circuit, to_pennylane(profile))()
    x0 = probabilities(profile, ops + [Op("h", (0,))], 2) @ [1, 1, -1, -1]
    y1 = probabilities(profile, ops + [Op("sdg", (1,)), Op("h", (1,))], 2) @ [1, -1, 1, -1]
    assert float(xx) == pytest.approx(2 * x0, abs=1e-12)
    assert float(yy) == pytest.approx(y1, abs=1e-12)


@pytest.mark.parametrize(
    "measure",
    [
        lambda qml: qml.classical_shadow(wires=[0], seed=10),
        lambda qml: qml.shadow_expval(qml.Z(0), seed=10),
    ],
    ids=["classical_shadow", "shadow_expval"],
)
def test_shadow_measurements_are_reported_and_warned_without_readout(qml, measure) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    profile = Profile.uniform(
        "readout", technology="other", num_qubits=1, one_qubit_error=0.0, readout_error=0.25
    )

    def run(model):
        @qml.qnode(qml.device("default.mixed", wires=1, seed=10))
        def circuit():
            return measure(qml)

        return qml.set_shots(qml.add_noise(circuit, model), shots=3000)()

    model = to_pennylane(profile)
    with pytest.warns(NoiseApproximationWarning, match="random measurement basis"):
        got = run(model)
    assert "readout on classical shadow measurements" in model.report.to_dict()["omitted"]
    np.testing.assert_array_equal(got, run(to_pennylane(profile, readout=False)))


def test_computational_basis_measurements_share_one_simulation(qml, manila) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    ops = [Op("sx", (w,)) for w in range(4)] + [Op("cx", (w, w + 1)) for w in range(3)]
    dev = qml.device("default.mixed", wires=4)

    @qml.qnode(dev)
    def circuit():
        _pl_ops(qml, ops)
        return [qml.expval(qml.Z(w)) for w in range(4)] + [qml.probs(wires=[2, 3])]

    noisy = qml.add_noise(circuit, to_pennylane(manila))
    with qml.Tracker(dev) as tracker:
        *z, probs_23 = noisy()
    assert tracker.totals["simulations"] == 1
    expected = probabilities(manila, ops, 4).reshape([2] * 4)
    for w in range(4):
        marginal = expected.sum(axis=tuple(a for a in range(4) if a != w))
        assert float(z[w]) == pytest.approx(marginal[0] - marginal[1], abs=1e-9)
    assert _tvd(probs_23, expected.sum(axis=(0, 1)).ravel()) <= 1e-9


def test_sampled_counts_include_readout(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    profile = _asymmetric_toy()
    ops = [Op("sx", (0,)), Op("cz", (0, 1)), Op("sx", (1,))]
    shots = 200_000

    @qml.qnode(qml.device("default.mixed", wires=2, seed=11))
    def circuit():
        _pl_ops(qml, ops)
        return qml.counts(wires=[0, 1])

    noisy = qml.set_shots(qml.add_noise(circuit, to_pennylane(profile)), shots=shots)
    counts = noisy()
    got = np.array([counts.get(k, 0) for k in ("00", "01", "10", "11")]) / shots
    expected = probabilities(profile, ops, 2)
    assert np.abs(got - expected).max() < 5 * np.sqrt(0.25 / shots)
    assert np.abs(got - probabilities(profile, ops, 2, readout=False)).max() > 0.02


def test_probs_without_wires_equals_probs_on_every_wire(qml, manila) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    @qml.qnode(qml.device("default.mixed", wires=3))
    def circuit():
        _pl_ops(qml, MIRROR_HALF)
        return qml.probs(), qml.probs(wires=[0, 1, 2])

    model = to_pennylane(manila)
    implicit, explicit = qml.add_noise(circuit, model)()
    expected = probabilities(manila, MIRROR_HALF, 3)
    assert _tvd(np.asarray(explicit), expected) <= 1e-9
    assert _tvd(np.asarray(implicit), np.asarray(explicit)) <= 1e-12
    assert any(a.what == "readout of measurements without wires" for a in model.report.approximated)


@pytest.mark.parametrize("measure", ["counts", "sample"])
def test_sampled_measurements_without_wires_include_readout(qml, manila, measure) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    shots = 100_000
    results = {}
    for wires in (None, [0, 1, 2]):

        @qml.qnode(qml.device("default.mixed", wires=3, seed=5))
        def circuit(wires=wires):
            _pl_ops(qml, MIRROR_HALF)
            return getattr(qml, measure)(wires=wires)

        out = qml.set_shots(qml.add_noise(circuit, to_pennylane(manila)), shots=shots)()
        results[str(wires)] = _frequencies(out, 3, measure)
    expected = probabilities(manila, MIRROR_HALF, 3)
    without_readout = probabilities(manila, MIRROR_HALF, 3, readout=False)
    sigma = np.sqrt(expected * (1 - expected) / shots)
    for got in results.values():
        assert np.all(np.abs(got - expected) <= 5 * sigma + 5 / shots)
        assert _tvd(got, without_readout) > 0.02
    assert results["None"] == pytest.approx(results["[0, 1, 2]"], abs=1e-12)


@pytest.mark.parametrize("wireless_first", [True, False])
def test_readout_without_wires_does_not_depend_on_measurement_order(qml, wireless_first) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    profile = _asymmetric_toy()

    @qml.qnode(qml.device("default.mixed", wires=2))
    def circuit():
        qml.SX(0)
        if wireless_first:
            return qml.probs(), qml.probs(wires=[1])
        return qml.probs(wires=[1]), qml.probs()

    out = [np.asarray(r) for r in qml.add_noise(circuit, to_pennylane(profile))()]
    every_wire, wire_1 = out if wireless_first else out[::-1]
    expected = probabilities(profile, [Op("sx", (0,))], 2)
    assert every_wire == pytest.approx(expected, abs=1e-12)
    assert wire_1 == pytest.approx(expected.reshape(2, 2).sum(axis=0), abs=1e-12)


BELL_THEN_H = [Op("h", (0,)), Op("cx", (0, 1)), Op("h", (1,))]


def _ideal_gates(readout_error: float) -> Profile:
    data = toy(
        gates={"h": {"avg_infidelity": 0.0}, "cx": {"avg_infidelity": 0.0}},
        qubits=[
            {"index": q, "readout": {"p1_given_0": readout_error, "p0_given_1": readout_error}}
            for q in range(2)
        ],
    )
    return Profile.model_validate(data)


@pytest.mark.parametrize("readout_error", [0.01, 0.0])
def test_commuting_pauli_samples_come_from_the_same_shots(qml, readout_error) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    profile = _ideal_gates(readout_error)
    shots = 100_000

    @qml.qnode(qml.device("default.mixed", wires=2, seed=19))
    def circuit():
        _pl_ops(qml, BELL_THEN_H)
        return qml.sample(qml.Z(0)), qml.sample(qml.X(1))

    z0, x1 = qml.set_shots(qml.add_noise(circuit, to_pennylane(profile)), shots=shots)()
    expected = probabilities(profile, BELL_THEN_H + [Op("h", (1,))], 2) @ [1, -1, -1, 1]
    product = np.mean(np.asarray(z0) * np.asarray(x1))
    assert product == pytest.approx(expected, abs=5 / np.sqrt(shots))


def test_samples_that_share_a_wire_get_one_readout_in_the_shared_basis(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    profile = _asymmetric_toy()
    ops = [Op("sx", (0,)), Op("cz", (0, 1)), Op("sx", (1,))]
    shots = 100_000

    @qml.qnode(qml.device("default.mixed", wires=2, seed=23))
    def circuit():
        _pl_ops(qml, ops)
        return qml.sample(qml.X(0)), qml.sample(qml.X(0) @ qml.Y(1))

    x0, x0y1 = qml.set_shots(qml.add_noise(circuit, to_pennylane(profile)), shots=shots)()
    in_x0_y1 = probabilities(profile, ops + [Op("h", (0,)), Op("sdg", (1,)), Op("h", (1,))], 2)
    sigma = 5 / np.sqrt(shots)
    assert np.mean(x0) == pytest.approx(float(in_x0_y1 @ [1, 1, -1, -1]), abs=sigma)
    product = np.mean(np.asarray(x0) * np.asarray(x0y1))
    assert product == pytest.approx(float(in_x0_y1 @ [1, -1, 1, -1]), abs=sigma)


@pytest.mark.parametrize("measure", ["expval", "var", "probs", "counts", "sample"])
def test_commuting_pauli_measurements_of_every_kind_share_shots(qml, measure) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    @qml.qnode(qml.device("default.mixed", wires=2, seed=29))
    def circuit():
        _pl_ops(qml, BELL_THEN_H)
        if measure == "probs":
            return qml.probs(op=qml.Z(0)), qml.probs(op=qml.X(1))
        return getattr(qml, measure)(qml.Z(0)), getattr(qml, measure)(qml.X(1))

    noisy = qml.add_noise(circuit, to_pennylane(_ideal_gates(0.0)))
    z0, x1 = qml.set_shots(noisy, shots=1000)()
    if measure == "counts":
        assert z0 == x1
    else:
        np.testing.assert_array_equal(z0, x1)


def test_shots_refuse_pauli_words_whose_shared_shots_default_mixed_decides(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    shots = 20_000
    model = to_pennylane(_ideal_gates(0.01))

    def run(measurements, transform=lambda qnode: qnode):
        @qml.qnode(qml.device("default.mixed", wires=2, seed=31))
        def circuit():
            _pl_ops(qml, BELL_THEN_H)
            return measurements()

        return qml.set_shots(qml.add_noise(transform(circuit), model), shots=shots)()

    def ambiguous():
        return qml.sample(qml.Z(0)), qml.sample(qml.X(1)), qml.sample(qml.X(0))

    def through_identity_probabilities():
        return qml.probs(op=qml.I(0) @ qml.I(1)), qml.sample(qml.X(1)), qml.sample(qml.Z(1))

    with pytest.raises(NoiseVaultError, match="read wire 0 in different bases") as raised:
        run(ambiguous)
    assert "qml.transforms.split_non_commuting" in raised.value.hint
    with pytest.raises(NoiseVaultError, match="read wire 1 in different bases"):
        run(through_identity_probabilities)
    split = run(through_identity_probabilities, qml.transforms.split_non_commuting)
    assert [np.asarray(r).shape for r in split] == [(4,), (shots,), (shots,)]
    split = run(ambiguous, qml.transforms.split_non_commuting)
    assert [np.asarray(r).shape for r in split] == [(shots,)] * 3

    x0, z0, probs_1 = run(
        lambda: (qml.expval(qml.X(0)), qml.expval(qml.Z(0)), qml.probs(wires=[1]))
    )
    in_x0 = probabilities(_ideal_gates(0.01), BELL_THEN_H + [Op("h", (0,))], 2)
    in_z0 = probabilities(_ideal_gates(0.01), BELL_THEN_H, 2)
    sigma = 5 / np.sqrt(shots)
    assert float(x0) == pytest.approx(in_x0 @ [1, 1, -1, -1], abs=sigma)
    assert float(z0) == pytest.approx(in_z0 @ [1, 1, -1, -1], abs=sigma)
    assert np.asarray(probs_1) == pytest.approx(in_z0.reshape(2, 2).sum(axis=0), abs=sigma)


@pytest.mark.parametrize("readout_error", [0.01, 0.0])
def test_a_zero_coefficient_term_does_not_split_shared_shots(qml, readout_error) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    profile = _ideal_gates(readout_error)
    bell = [Op("h", (0,)), Op("cx", (0, 1))]
    shots = 10_000

    @qml.qnode(qml.device("default.mixed", wires=2, seed=19))
    def circuit():
        _pl_ops(qml, bell)
        return qml.sample(qml.X(0) + 0 * qml.Y(1)), qml.sample(qml.X(1))

    x0, x1 = qml.set_shots(qml.add_noise(circuit, to_pennylane(profile)), shots=shots)()
    expected = probabilities(profile, bell + [Op("h", (0,)), Op("h", (1,))], 2) @ [1, -1, -1, 1]
    product = np.mean(np.asarray(x0) * np.asarray(x1))
    assert product == pytest.approx(expected, abs=5 / np.sqrt(shots))


_IDENTITY_PROBS = {
    "identity product": lambda qml: (qml.probs(op=qml.I(0) @ qml.I(1)), qml.sample(qml.X(1))),
    "identity on one wire": lambda qml: (qml.probs(op=qml.I(1)), qml.counts(qml.X(1))),
    "zero observable": lambda qml: (
        qml.probs(op=0 * qml.Z(0) + 0 * qml.X(1)),
        qml.sample(qml.Y(1)),
    ),
}


@pytest.mark.parametrize("readout", [True, False])
@pytest.mark.parametrize("case", list(_IDENTITY_PROBS))
def test_identity_probabilities_share_the_basis_of_their_shot_group(qml, case, readout) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    def run(model=None):
        @qml.qnode(qml.device("default.mixed", wires=2, seed=19))
        def circuit():
            _pl_ops(qml, BELL_THEN_H)
            return _IDENTITY_PROBS[case](qml)

        noisy = circuit if model is None else qml.add_noise(circuit, model)
        return qml.set_shots(noisy, shots=10_000)()

    (raw_probs, raw_other), (probs, other) = (
        run(),
        run(to_pennylane(_ideal_gates(0.0), readout=readout)),
    )
    assert np.asarray(probs) == pytest.approx(np.asarray(raw_probs), abs=1e-12)
    if isinstance(raw_other, dict):
        assert other == raw_other
    else:
        np.testing.assert_array_equal(other, raw_other)
    if case == "identity product":
        assert np.asarray(probs) == pytest.approx([0.5, 0, 0, 0.5], abs=0.02)


def test_a_zero_coefficient_term_does_not_hide_the_measured_basis(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    data = toy(
        gates={"h": {"avg_infidelity": 0.0}},
        qubits=[{"index": 0, "readout": {"p1_given_0": 0.2, "p0_given_1": 0.2}}],
    )
    profile = Profile.model_validate(data)
    model = to_pennylane(profile)

    @qml.qnode(qml.device("default.mixed", wires=1))
    def circuit():
        qml.Hadamard(0)
        return qml.expval(qml.dot([1, 0], [qml.X(0), qml.Z(0)]))

    in_x0 = probabilities(profile, [Op("h", (0,)), Op("h", (0,))], 1)
    assert float(qml.add_noise(circuit, model)()) == pytest.approx(in_x0 @ [1, -1], abs=1e-12)
    assert (
        "readout on observables not measured in one product basis"
        not in model.report.to_dict()["omitted"]
    )


def test_shot_vectors_with_readout_raise_instead_of_dropping_results(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    @qml.qnode(qml.device("default.mixed", wires=1, seed=3))
    def circuit():
        qml.SX(0)
        return qml.probs(wires=[0])

    shot_vector = qml.set_shots(qml.add_noise(circuit, to_pennylane(_asymmetric_toy())), [100, 200])
    with pytest.raises(ValueError, match="Run each shot count separately or pass readout=False"):
        shot_vector()
    without_readout = to_pennylane(_asymmetric_toy(), readout=False)
    out = qml.set_shots(qml.add_noise(circuit, without_readout), [100, 200])()
    assert [np.asarray(r).shape for r in out] == [(2,), (2,)]


def _two_readouts() -> Profile:
    data = toy(
        gates={"x": {"avg_infidelity": 0.0}},
        qubits=[
            {"index": 0, "readout": {"p1_given_0": 0.1, "p0_given_1": 0.3}},
            {"index": 1, "readout": {"p1_given_0": 0.2, "p0_given_1": 0.05}},
        ],
    )
    return Profile.model_validate(data)


def _composed(qml, model, how: str):
    flip = {qml.noise.op_eq(qml.PauliX): qml.noise.partial_wires(qml.BitFlip, 0.05)}
    if how == "dict":
        return model + flip
    other = qml.NoiseModel(flip)
    return model + other if how == "model first" else other + model


@pytest.mark.parametrize("how", ["model first", "model second", "dict"])
def test_composed_models_keep_readout_on_every_tape(qml, how) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    composed = _composed(qml, to_pennylane(_two_readouts()), how)
    # X then BitFlip(0.05) leaves P(1) = 0.95; readout gives P(0) = 0.05 P(0|0) + 0.95 P(0|1)
    expected = {
        0: [0.05 * 0.9 + 0.95 * 0.3, 0.05 * 0.1 + 0.95 * 0.7],
        1: [0.05 * 0.8 + 0.95 * 0.05, 0.05 * 0.2 + 0.95 * 0.95],
    }
    for wire in (0, 1):

        @qml.qnode(qml.device("default.mixed", wires=2))
        def circuit(wire=wire):
            qml.PauliX(wire)
            return qml.probs(wires=[wire])

        got = np.asarray(qml.add_noise(circuit, composed)())
        assert got == pytest.approx(expected[wire], abs=1e-12)


def test_composed_models_still_refuse_shot_vectors_with_readout(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    @qml.qnode(qml.device("default.mixed", wires=1, seed=3))
    def circuit():
        qml.PauliX(0)
        return qml.probs(wires=[0])

    composed = _composed(qml, to_pennylane(_two_readouts()), "model second")
    with pytest.raises(ValueError, match="Run each shot count separately"):
        qml.set_shots(qml.add_noise(circuit, composed), [100, 200])()


def test_a_model_stripped_of_readout_runs_shot_vectors(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    @qml.qnode(qml.device("default.mixed", wires=1, seed=3))
    def circuit():
        qml.PauliX(0)
        return qml.probs(wires=[0])

    model = to_pennylane(_two_readouts())
    stripped = model - {"meas_map": model.meas_map}
    out = qml.set_shots(qml.add_noise(circuit, stripped), [100, 200])()
    # x is ideal, so without readout every shot reads 1
    assert [np.asarray(r).tolist() for r in out] == [[0.0, 1.0], [0.0, 1.0]]


def test_noise_functions_outside_add_noise_refuse_to_guess_the_circuit(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    [readout] = to_pennylane(_two_readouts()).meas_map.values()
    with pytest.raises(RuntimeError, match=r"^[^\n]*qml\.add_noise[^\n]*$"):
        readout(qml.probs(wires=[0]))


def test_device_wires_no_operation_touches_read_out_ideally(qml, manila) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    @qml.qnode(qml.device("default.mixed", wires=3))
    def circuit():
        qml.PauliX(0)
        qml.PauliX(1)
        return qml.probs()

    probs = np.asarray(qml.add_noise(circuit, to_pennylane(manila))()).reshape(2, 2, 2)
    assert probs[:, :, 1].sum() == 0
    expected = probabilities(manila, [Op("x", (0,)), Op("x", (1,))], 2)
    assert _tvd(probs[:, :, 0].ravel(), expected) <= 1e-9


def _frequencies(out, n: int, measure: str) -> np.ndarray:
    if measure == "counts":
        keys = [format(i, f"0{n}b") for i in range(2**n)]
        total = sum(out.values())
        return np.array([out.get(k, 0) for k in keys]) / total
    index = np.asarray(out, dtype=int) @ (1 << np.arange(n)[::-1])
    return np.bincount(index, minlength=2**n) / len(index)


def test_unknown_readout_is_reported_not_invented(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    profile = Profile.model_validate(toy())
    model = to_pennylane(profile)

    @qml.qnode(qml.device("default.mixed", wires=1))
    def circuit():
        qml.PauliX(0)
        return qml.probs(wires=[0])

    got = np.asarray(qml.add_noise(circuit, model)())
    assert got == pytest.approx(
        probabilities(profile, [Op("x", (0,))], 1, readout=False), abs=1e-12
    )
    assert "readout on qubit 0" in model.report.to_dict()["unknown"]


def test_reset_gets_the_preparation_error(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    data = toy(gates={"rz": {"virtual": True}, "x": {"avg_infidelity": 0.0}}, prep={"error": 0.05})
    model = to_pennylane(Profile.model_validate(data), readout=False)

    @qml.qnode(qml.device("default.mixed", wires=2))  # deferred measurement needs a spare wire
    def circuit():
        qml.PauliX(0)
        qml.measure(0, reset=True)
        return qml.probs(wires=[0])

    assert np.asarray(qml.add_noise(circuit, model)()) == pytest.approx([0.95, 0.05], abs=1e-12)

    unknown = to_pennylane(Profile.model_validate(toy()), readout=False)
    assert np.asarray(qml.add_noise(circuit, unknown)()) == pytest.approx([1, 0], abs=1e-12)
    assert "reset error on qubit 0" in unknown.report.to_dict()["unknown"]


@pytest.mark.parametrize("where", ["record", "definition"])
def test_a_reset_the_profile_disables_is_refused(qml, where) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    data = toy(gates={"x": {"avg_infidelity": 0.0}}, prep={"error": 0.03})
    if where == "record":
        data["gates"]["reset"] = {}
        data["calibrations"] = [{"gate": "reset", "qubits": [2], "disabled": True}]
    else:
        data["gates"]["reset"] = {"disabled": True}
    model = to_pennylane(Profile.model_validate(data), layout=[0, 2, 1], readout=False)

    def run(wire):
        @qml.qnode(qml.device("default.mixed", wires=3))  # deferred measurement needs a spare wire
        def circuit():
            qml.PauliX(wire)
            qml.measure(wire, reset=True)
            return qml.probs(wires=[wire])

        return np.asarray(qml.add_noise(circuit, model)())

    with pytest.raises(DisabledGateError, match=r"^reset on qubit 2 is disabled in this profile$"):
        run(1)
    if where == "record":
        assert run(0) == pytest.approx([0.97, 0.03], abs=1e-12)


def _measure_off(where: str | None) -> Profile:
    readout = {"p1_given_0": 0.02, "p0_given_1": 0.1}
    data = toy(gates={"x": {"avg_infidelity": 0.0}}, readout=readout)
    if where == "record":
        data["gates"]["measure"] = {}
        data["calibrations"] = [{"gate": "measure", "qubits": [2], "disabled": True}]
    elif where == "definition":
        data["gates"]["measure"] = {"disabled": True}
    return Profile.model_validate(data)


_MEASUREMENTS = {
    "mid-circuit": lambda qml, w: qml.probs(op=qml.measure(w)),
    "mid-circuit with reset": lambda qml, w: qml.probs(op=qml.measure(w, reset=True)),
    "probs": lambda qml, w: qml.probs(wires=[w]),
    "probs of every wire": lambda qml, w: qml.probs(),
    "expval": lambda qml, w: qml.expval(qml.PauliZ(w)),
    "sample": lambda qml, w: qml.sample(wires=[w]),
}


def _measured(qml, model, kind: str, wire: int):
    @qml.qnode(qml.device("default.mixed", wires=3))
    def circuit():
        qml.PauliX(wire)
        return _MEASUREMENTS[kind](qml, wire)

    noisy = qml.add_noise(circuit, model)
    return np.asarray(qml.set_shots(noisy, shots=20)() if kind == "sample" else noisy())


@pytest.mark.parametrize("readout", [True, False])
@pytest.mark.parametrize("where", ["record", "definition"])
@pytest.mark.parametrize("kind", list(_MEASUREMENTS))
def test_a_measurement_the_profile_disables_is_refused(qml, kind, where, readout) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    model = to_pennylane(_measure_off(where), layout=[0, 2, 1], readout=readout)
    message = r"^measure on qubit 2 is disabled in this profile$"
    with pytest.raises(DisabledGateError, match=message):
        _measured(qml, model, kind, 1)


@pytest.mark.parametrize(
    "kind", [k for k in _MEASUREMENTS if k not in ("sample", "probs of every wire")]
)
def test_a_measurement_where_the_profile_allows_it_keeps_its_noise(qml, kind) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    noisy, plain = (
        _measured(qml, to_pennylane(_measure_off(where), layout=[0, 2, 1]), kind, 0)
        for where in ("record", None)
    )
    assert noisy == pytest.approx(plain, abs=1e-12)
    if kind == "probs":
        assert noisy == pytest.approx([0.1, 0.9], abs=1e-12)


def _measure_off_on_1() -> Profile:
    data = toy(
        gates={"x": {"avg_infidelity": 0.0}, "measure": {}},
        qubits=[{"index": 0, "readout": {"p1_given_0": 0.1, "p0_given_1": 0.2}}],
        calibrations=[{"gate": "measure", "qubits": [1], "disabled": True}],
    )
    return Profile.model_validate(data)


@pytest.mark.parametrize("readout", [True, False])
@pytest.mark.parametrize("measure", ["probs", "sample", "counts"])
def test_a_measurement_without_wires_reads_every_device_wire(qml, measure, readout) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    def run(profile, device_wires, wires=None):
        @qml.qnode(qml.device("default.mixed", wires=device_wires, seed=7))
        def circuit():
            qml.PauliX(0)
            return getattr(qml, measure)(wires=wires)

        noisy = qml.add_noise(circuit, to_pennylane(profile, layout=[0, 1], readout=readout))
        return qml.set_shots(noisy, shots=200)() if measure != "probs" else noisy()

    with pytest.raises(
        DisabledGateError, match=r"^measure on qubit 1 is disabled in this profile$"
    ):
        run(_measure_off_on_1(), 2)
    with pytest.raises(LayoutError, match="marks disabled"):
        run(_one_disabled(), 2)
    explicit = run(_measure_off_on_1(), [0], wires=[0])
    for device_wires in ([0], None):
        got = run(_measure_off_on_1(), device_wires)
        if measure == "counts":
            assert got == explicit
        else:
            np.testing.assert_array_equal(got, explicit)


_TAPE_CASES = {
    "measure off, default layout": (_measure_off_on_1, None, True),
    "measure off, list layout": (_measure_off_on_1, [0, 1], True),
    "measure off, layout without it": (_measure_off_on_1, {0: 0}, False),
    "qubit off, default layout": (_one_disabled, None, True),
    "every qubit measures": (lambda: Profile.model_validate(toy()), None, False),
}


@pytest.mark.parametrize("case", list(_TAPE_CASES))
def test_a_measurement_without_wires_on_a_tape_is_refused_when_a_qubit_cannot_measure(
    qml, case
) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    profile, layout, refused = _TAPE_CASES[case]
    model = to_pennylane(profile(), layout=layout)

    def run(wires=None):
        tape = qml.tape.QuantumScript([qml.PauliX(0)], [qml.probs(wires=wires)])
        [noisy], _ = qml.noise.add_noise(tape, model)
        return qml.device("default.mixed", wires=[0]).execute(noisy)

    explicit = run(wires=[0])
    if not refused:
        assert run() == pytest.approx(explicit, abs=1e-12)
        return
    with pytest.raises(
        DisabledGateError, match="can map to qubit 1, which cannot measure"
    ) as raised:
        run()
    assert (
        raised.value.hint == "pass wires= to the measurement, or apply qml.add_noise to the QNode"
    )


_ONLY_WIRE_0 = {
    "a zero term": lambda qml: qml.X(0) + 0 * qml.Z(1),
    "an identity factor": lambda qml: qml.X(0) @ qml.I(1),
    "a Hamiltonian": lambda qml: qml.Hamiltonian([1.0, 0.0], [qml.X(0), qml.Z(1)]),
}


def _wire_0_reads(qml, model, measure: str, observable):
    @qml.qnode(qml.device("default.mixed", wires=2, seed=5))
    def circuit():
        qml.Hadamard(0)
        return getattr(qml, measure)(op=observable)

    noisy = qml.add_noise(circuit, model)
    return qml.set_shots(noisy, shots=50)() if measure in ("sample", "counts") else noisy()


@pytest.mark.parametrize("readout", [True, False])
@pytest.mark.parametrize("measure", ["expval", "var", "sample", "counts"])
@pytest.mark.parametrize("obs", list(_ONLY_WIRE_0))
def test_a_pauli_observable_reads_only_the_wires_of_its_simplified_words(
    qml, obs, measure, readout
) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    data = toy(
        gates={"h": {"avg_infidelity": 0.0}, "measure": {}},
        qubits=[{"index": 0, "readout": {"p1_given_0": 0.1, "p0_given_1": 0.1}}],
        calibrations=[{"gate": "measure", "qubits": [1], "disabled": True}],
    )
    model = to_pennylane(Profile.model_validate(data), readout=readout)
    got = _wire_0_reads(qml, model, measure, _ONLY_WIRE_0[obs](qml))
    assert got == pytest.approx(_wire_0_reads(qml, model, measure, qml.X(0)), abs=1e-12)
    if measure == "expval":
        assert got == pytest.approx(0.8 if readout else 1.0, abs=1e-12)


@pytest.mark.parametrize("readout", [True, False])
@pytest.mark.parametrize("measure", ["probs", "expval"])
def test_a_measurement_of_every_wire_of_its_observable_is_refused(qml, measure, readout) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    data = toy(
        gates={"h": {"avg_infidelity": 0.0}, "measure": {}},
        calibrations=[{"gate": "measure", "qubits": [1], "disabled": True}],
    )
    model = to_pennylane(Profile.model_validate(data), readout=readout)
    x0 = np.kron([[0, 1], [1, 0]], np.eye(2))
    observable = qml.X(0) @ qml.I(1) if measure == "probs" else qml.Hermitian(x0, wires=[0, 1])
    message = r"^measure on qubit 1 is disabled in this profile$"
    with pytest.raises(DisabledGateError, match=message):
        _wire_0_reads(qml, model, measure, observable)


def test_readout_on_a_wire_with_only_zero_terms_is_not_unknown(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    data = toy(
        gates={"h": {"avg_infidelity": 0.0}},
        qubits=[{"index": 0, "readout": {"p1_given_0": 0.1, "p0_given_1": 0.1}}],
    )
    model = to_pennylane(Profile.model_validate(data))
    assert _wire_0_reads(qml, model, "expval", qml.X(0) + 0 * qml.Z(1)) == pytest.approx(0.8)
    assert model.report.to_dict()["unknown"] == []
    _wire_0_reads(qml, model, "expval", qml.X(0) + qml.Z(1))
    assert model.report.to_dict()["unknown"] == ["readout on qubit 1"]


@pytest.mark.parametrize("shots", [None, 100])
def test_readout_goes_only_on_the_wires_of_the_simplified_words(qml, shots) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    readout = {"p1_given_0": 0.1, "p0_given_1": 0.1}
    data = toy(
        gates={"h": {"avg_infidelity": 0.0}},
        qubits=[{"index": q, "readout": readout} for q in range(2)],
    )
    measured = [qml.expval(qml.X(0) + 0 * qml.Z(1)), qml.var(qml.X(0) @ qml.I(1))]
    tape = qml.tape.QuantumScript([qml.Hadamard(0)], measured, shots=shots)
    [noisy], _ = qml.noise.add_noise(tape, to_pennylane(Profile.model_validate(data)))
    assert [(op.name, op.wires.tolist()) for op in noisy.operations] == [
        ("Hadamard", [0]),
        ("Hadamard", [0]),
        ("QubitChannel", [0]),
        ("Hadamard", [0]),
    ]


# unknown gates, report -----------------------------------------------------------------------


def test_typical_noise_warnings_point_at_the_callers_line(qml, manila) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    @qml.qnode(qml.device("default.mixed", wires=2))
    def circuit():
        qml.Hadamard(0)
        qml.CZ([0, 1])
        return qml.expval(qml.PauliX(0) @ qml.PauliZ(1) + qml.PauliZ(0))

    with pytest.warns(NoiseApproximationWarning) as caught:
        qml.add_noise(circuit, to_pennylane(manila))()
    assert len(caught) == 3  # h, cz, and the readout the observable rules out
    assert [w.filename for w in caught] == [__file__] * 3


def test_unknown_gate_warns_once_and_counts_every_use(qml, manila) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    model = to_pennylane(manila)

    @qml.qnode(qml.device("default.mixed", wires=2))
    def circuit():
        qml.Hadamard(0)
        qml.Hadamard(0)
        qml.Hadamard(1)
        return qml.probs(wires=[0, 1])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        qml.add_noise(circuit, model)()
    messages = [str(w.message) for w in caught if issubclass(w.category, NoiseApproximationWarning)]
    assert len(messages) == 1 and "h on qubit 0: " in messages[0]
    assert model.report.events["typical_noise_used"]["h"] == 3


def test_unknown_gates_error_raises(qml, manila) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    @qml.qnode(qml.device("default.mixed", wires=1))
    def circuit():
        qml.Hadamard(0)
        return qml.probs(wires=[0])

    with pytest.raises(MissingCalibrationError, match="h on qubit 0: "):
        qml.add_noise(circuit, to_pennylane(manila, unknown_gates="error"))()
    with pytest.raises(ValueError, match="choose 'typical' or 'error'"):
        to_pennylane(manila, unknown_gates="ignore")


def test_conditional_gates_get_the_noise_of_their_gate(qml, manila) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    model = to_pennylane(manila)

    @qml.qnode(qml.device("default.mixed", wires=3))  # deferred measurement needs a spare wire
    def circuit():
        qml.PauliX(0)
        qml.cond(qml.measure(0), qml.PauliX)(1)
        return qml.probs(wires=[1])

    with warnings.catch_warnings():
        warnings.simplefilter("error", NoiseApproximationWarning)
        qml.add_noise(circuit, model)()
    assert "typical_noise_used" not in model.report.events
    assert "conditional gates" in [a.what for a in model.report.approximated]


def test_level_top_noises_adjoint_gates_as_written(qml, manila) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    @qml.qnode(qml.device("default.mixed", wires=1))
    def circuit():
        qml.adjoint(qml.SX(0))
        return qml.probs(wires=[0])

    user, top = to_pennylane(manila), to_pennylane(manila)
    qml.add_noise(circuit, user)()
    qml.add_noise(circuit, top, level="top")()
    assert "ry" in user.report.events["typical_noise_used"]
    assert "adjoint gates and templates" in [a.what for a in user.report.approximated]
    assert dict(top.report.events["typical_noise_used"]) == {"sxdg": 1}


@pytest.mark.parametrize("level", ["user", "top"])
def test_adjoints_of_other_gates_get_the_noise_of_their_decomposition(qml, level) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    model = to_pennylane(_uniform(1), unknown_gates="error", readout=False)
    got = _noisy_qnode_probs(qml, model, lambda: qml.adjoint(qml.RX(0.3, 0)), 1, level)
    want = probabilities(_uniform(1), [Op("rx", (0,), (-0.3,))], 1, readout=False)
    assert got == pytest.approx(want, abs=1e-12)


def test_model_is_a_pennylane_noise_model_with_report(qml, manila) -> None:
    from noisevault.frameworks.pennylane import NoiseVaultPennyLaneModel

    model = manila.to_pennylane(layout=[0, 1])
    assert isinstance(model, qml.NoiseModel) and isinstance(model, NoiseVaultPennyLaneModel)
    assert model.profile is manila
    report = model.report.to_dict()
    assert report["framework"] == "pennylane" and report["framework_version"] == qml.__version__
    assert report["options"] == {"layout": [0, 1], "unknown_gates": "typical", "readout": True}


def test_effects_that_cannot_be_omitted_refuse_to_convert(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    data = toy(effects=[{"type": "leakage", "gate": "cz", "prob": 1e-4, "allow": "approximate"}])
    with pytest.raises(UnsupportedEffect, match="pennylane"):
        to_pennylane(Profile.model_validate(data))
    data["effects"][0]["allow"] = "omit"
    assert (
        "effect leakage on cz"
        in to_pennylane(Profile.model_validate(data)).report.to_dict()["omitted"]
    )


# operator arithmetic, identities and state preparation ---------------------------------------


def _uniform(num_qubits: int) -> Profile:
    return Profile.uniform(
        "uniform",
        technology="other",
        num_qubits=num_qubits,
        one_qubit_error=0.1,
        two_qubit_error=0.2 if num_qubits > 1 else None,
    )


def _noisy_qnode_probs(qml, model, apply, num_wires: int, level: str = "user") -> np.ndarray:
    @qml.qnode(qml.device("default.mixed", wires=num_wires))
    def circuit():
        apply()
        return qml.probs(wires=range(num_wires))

    return np.asarray(qml.add_noise(circuit, model, level=level)())


def _inserted_channels(qml, model, ops: list) -> list[tuple[list, np.ndarray]]:
    [tape], _ = qml.noise.add_noise(qml.tape.QuantumScript(ops), model)
    return [
        (op.wires.tolist(), np.stack(op.kraus_matrices()))
        for op in tape.operations
        if op.name == "QubitChannel"
    ]


@pytest.mark.parametrize("unknown_gates", ["typical", "error"])
def test_a_gate_product_gets_the_noise_of_its_gates(qml, unknown_gates) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    model = to_pennylane(_uniform(1), unknown_gates=unknown_gates, readout=False)
    got = _noisy_qnode_probs(qml, model, lambda: qml.prod(qml.X(0), qml.Z(0)), 1)
    assert got == pytest.approx([0.1, 0.9], abs=1e-12)
    assert "typical_noise_used" not in model.report.events
    assert "operator arithmetic" not in [a.what for a in model.report.approximated]


@pytest.mark.parametrize(
    ("arithmetic", "gates", "noise_moves"),
    [
        (lambda q: q.prod(q.X(0), q.Z(0)), lambda q: [q.Z(0), q.X(0)], False),
        (lambda q: q.X(0) @ q.SX(1), lambda q: [q.SX(1), q.X(0)], False),
        (lambda q: q.s_prod(1j, q.SX(0)), lambda q: [q.SX(0)], False),
        (
            lambda q: q.pow(q.SX(0) @ q.X(1), 2),
            lambda q: [q.X(1), q.SX(0), q.X(1), q.SX(0)],
            True,
        ),
        (lambda q: q.adjoint(q.X(0) @ q.Z(0)), lambda q: [q.X(0), q.Z(0)], True),
        (
            lambda q: q.ctrl(q.X(1) @ q.Z(1), 0),
            lambda q: [q.RY(np.pi / 2, 1), q.CNOT([0, 1]), q.RY(-np.pi / 2, 1), q.CNOT([0, 1])],
            True,
        ),
        (
            lambda q: q.change_op_basis(q.Hadamard(0), q.Z(0)),
            lambda q: [q.Hadamard(0), q.Z(0), q.Hadamard(0)],
            True,
        ),
        (lambda q: q.pow(q.RX(0.3, 0), 2), lambda q: [q.RX(0.6, 0)], False),
        (lambda q: q.exp(q.X(0), -0.15j), lambda q: [q.RX(0.3, 0)], False),
        (lambda q: q.evolve(q.X(0), 0.3), lambda q: [q.RX(0.6, 0)], False),
    ],
    ids=[
        "product",
        "tensor product",
        "phase",
        "power",
        "adjoint",
        "control",
        "change of basis",
        "gate power",
        "exponential",
        "evolution",
    ],
)
def test_operator_arithmetic_gets_the_channels_of_its_gates_written_out(
    qml, arithmetic, gates, noise_moves
) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    model = to_pennylane(_uniform(2), unknown_gates="error", readout=False)
    got = _inserted_channels(qml, model, [arithmetic(qml)])
    expected = _inserted_channels(qml, to_pennylane(_uniform(2), readout=False), gates(qml))
    assert expected
    _assert_same_channels(got, expected)
    moved = "operator arithmetic" in [a.what for a in model.report.approximated]
    assert moved == noise_moves


def _assert_same_channels(got: list, expected: list) -> None:
    assert [wires for wires, _ in got] == [wires for wires, _ in expected]
    for (_, kraus), (_, want) in zip(got, expected, strict=True):
        assert np.abs(kraus - want).max() < 1e-12


def _h_then_rx() -> Profile:
    data = toy(
        device={"name": "h_then_rx", "vendor": "test", "technology": "other", "num_qubits": 1},
        connectivity={"edges": []},
        gates={"h": {"avg_infidelity": 0.01}, "rx": {"avg_infidelity": 0.3}},
    )
    return Profile.model_validate(data)


@pytest.mark.parametrize("unknown_gates", ["typical", "error"])
def test_a_gate_power_gets_the_noise_of_the_gate_it_decomposes_into(qml, unknown_gates) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    model = to_pennylane(_h_then_rx(), unknown_gates=unknown_gates, readout=False)
    got = _noisy_qnode_probs(qml, model, lambda: qml.pow(qml.RX(0.3, 0), 2), 1)
    assert got == pytest.approx([0.66506712, 0.33493288], abs=1e-8)
    assert "typical_noise_used" not in model.report.events


def test_a_controlled_gate_gets_the_noise_of_the_gates_default_mixed_runs(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    @qml.qnode(qml.device("default.mixed", wires=2))
    def circuit():
        qml.ctrl(qml.SX(0), 1)
        return qml.probs(wires=[0, 1])

    (run,), _ = qml.workflow.construct_batch(circuit, level="device")()
    model = to_pennylane(_uniform(2), readout=False)
    got = _inserted_channels(qml, model, [qml.ctrl(qml.SX(0), 1)])
    expected = _inserted_channels(qml, to_pennylane(_uniform(2), readout=False), run.operations)
    _assert_same_channels(got, expected)
    assert "C(SX)" not in model.report.events["typical_noise_used"]


@pytest.mark.parametrize(
    ("gate", "name"),
    [
        (lambda q: q.CRX(0.3, [1, 0]), "CRX"),
        (lambda q: q.ctrl(q.RX(0.3, 0), 1), "CRX"),
        (lambda q: q.Rot(0.1, 0.2, 0.3, 0), "Rot"),
    ],
    ids=["CRX", "controlled RX", "Rot"],
)
def test_named_gates_with_a_decomposition_take_noise_as_one_gate(qml, gate, name) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    model = to_pennylane(_uniform(2), readout=False)
    _inserted_channels(qml, model, [gate(qml)])
    assert dict(model.report.events["typical_noise_used"]) == {name: 1}


def test_uncalibrated_gates_in_a_product_warn_or_raise(qml, manila) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    def apply():
        qml.prod(qml.Hadamard(0), qml.X(0))

    model = to_pennylane(manila, readout=False)
    with pytest.warns(NoiseApproximationWarning, match="h on qubit 0: "):
        _noisy_qnode_probs(qml, model, apply, 1)
    assert dict(model.report.events["typical_noise_used"]) == {"h": 1}
    with pytest.raises(MissingCalibrationError, match="h on qubit 0: "):
        _noisy_qnode_probs(qml, to_pennylane(manila, unknown_gates="error"), apply, 1)


@pytest.mark.parametrize("unknown_gates", ["typical", "error"])
@pytest.mark.parametrize(
    "arithmetic",
    [
        lambda q: q.sum(q.X(0), q.Z(0)),
        lambda q: q.Hamiltonian([0.6, 0.8], [q.X(0), q.Z(0)]),
        lambda q: q.s_prod(2.0, q.X(0)),
        lambda q: q.pow(q.X(0) @ q.S(0), 0.5),
        lambda q: q.pow(q.Hadamard(0), 0.5),
    ],
    ids=["sum", "linear combination", "scaled", "fractional power", "fractional gate power"],
)
def test_arithmetic_without_a_gate_decomposition_raises(qml, arithmetic, unknown_gates) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    model = to_pennylane(_uniform(1), unknown_gates=unknown_gates, readout=False)
    # A tape, because a QNode before PennyLane 0.45 leaves a Sum or an SProd off its tape.
    with pytest.raises(ValueError, match="no decomposition into gates"):
        _inserted_channels(qml, model, [arithmetic(qml)])


def test_an_identity_on_several_wires_gets_each_wires_id_noise(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    model = to_pennylane(_uniform(2), unknown_gates="error", readout=False)
    got = _noisy_qnode_probs(qml, model, lambda: qml.Identity(wires=[0, 1]), 2)
    assert got == pytest.approx([0.81, 0.09, 0.09, 0.01], abs=1e-12)


@pytest.mark.parametrize("level", ["user", "top"])
@pytest.mark.parametrize("unknown_gates", ["typical", "error"])
@pytest.mark.parametrize(
    ("prepare", "expected"),
    [
        (lambda q: q.QubitDensityMatrix(np.diag([1.0, 0.0]), wires=0), [1, 0]),
        (
            lambda q: q.QubitDensityMatrix(np.diag([0.0] * 7 + [1.0]), wires=[0, 1, 2]),
            [0] * 7 + [1],
        ),
        (lambda q: q.BasisState(np.array([1, 1, 1]), wires=[0, 1, 2]), [0] * 7 + [1]),
        (lambda q: q.StatePrep(np.array([0.0, 1.0]), wires=0), [0, 1]),
        (lambda q: q.AmplitudeEmbedding(np.array([0.0, 1.0]), wires=0), [0, 1]),
        (lambda q: q.BasisEmbedding([1], wires=0), [0, 1]),
    ],
    ids=[
        "density matrix",
        "three-qubit density matrix",
        "basis state",
        "state vector",
        "amplitude embedding",
        "basis embedding",
    ],
)
def test_state_preparation_is_ideal(qml, prepare, expected, unknown_gates, level) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    num_wires = len(expected).bit_length() - 1
    model = to_pennylane(_uniform(num_wires), unknown_gates=unknown_gates, readout=False)
    got = _noisy_qnode_probs(qml, model, lambda: prepare(qml), num_wires, level)
    assert got == pytest.approx(expected, abs=1e-12)


# gradients and scale -------------------------------------------------------------------------


@pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
def test_gradient_step_through_noisy_qnode(qml, diff_method) -> None:
    from pennylane import numpy as pnp

    from noisevault.frameworks.pennylane import to_pennylane

    model = to_pennylane(_ion(), layout=[2, 0])

    @qml.qnode(qml.device("default.mixed", wires=2), diff_method=diff_method)
    def circuit(params):
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(params[2], wires=1)
        return qml.expval(qml.Z(0) @ qml.Z(1))

    noisy = qml.add_noise(circuit, model)
    params = pnp.array([0.4, -0.9, 1.3], requires_grad=True)
    grad = qml.grad(noisy)(params)

    step = 1e-5
    finite = [(noisy(params + step * e) - noisy(params - step * e)) / (2 * step) for e in np.eye(3)]
    assert np.asarray(grad) == pytest.approx(np.asarray(finite, dtype=float), abs=1e-7)
    assert abs(float(noisy(params)) - float(circuit(params))) > 1e-4

    stepped = params - 0.1 * grad
    assert float(noisy(stepped)) < float(noisy(params))


def test_layered_circuit_converts_and_runs_quickly(qml, manila) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    rng = np.random.default_rng(7)
    angles = rng.uniform(0, 2 * np.pi, size=(6, 4, 2))
    start = time.perf_counter()
    model = to_pennylane(manila, layout=[1, 2, 3, 4])

    @qml.qnode(qml.device("default.mixed", wires=4))
    def circuit(theta):
        for layer in theta:
            for w in range(4):
                qml.RZ(layer[w, 0], wires=w)
                qml.SX(wires=w)
                qml.RZ(layer[w, 1], wires=w)
            for w in range(3):
                qml.CNOT(wires=[w, w + 1])
        return qml.probs(wires=range(4))

    noisy = qml.add_noise(circuit, model)
    first = np.asarray(noisy(angles))
    first_s = time.perf_counter() - start
    start = time.perf_counter()
    second = np.asarray(noisy(angles + 0.1))
    second_s = time.perf_counter() - start
    print(f"6-layer 4-qubit: first call {first_s:.3f} s, second call {second_s:.3f} s")

    assert first.sum() == pytest.approx(1) and not np.allclose(first, second)
    assert first_s < 10 and second_s < 5


# registry gates PennyLane has only at a fixed angle ------------------------------------------


ANGLES = [0.0, 0.37, np.pi / 2, -1.1, 2.9]


def _matrix(qml, op) -> np.ndarray:
    return qml.matrix(op, wire_order=range(len(op.wires)))  # big-endian, like the registry


@pytest.mark.parametrize("theta", ANGLES)
@pytest.mark.parametrize("phi", ANGLES)
def test_r_is_a_rot_with_the_registry_unitary(qml, theta, phi) -> None:
    from noisevault.frameworks.pennylane import operation_for

    op = operation_for("r")(theta, phi, wires=[0])
    assert op.name == "Rot"
    assert np.allclose(_matrix(qml, op), GATES["r"].unitary(theta, phi), rtol=0, atol=1e-12)


@pytest.mark.parametrize("theta", ANGLES)
def test_rzz_and_zz_are_ising_zz_with_the_registry_unitary(qml, theta) -> None:
    from noisevault.frameworks.pennylane import operation_for

    rzz = operation_for("rzz")(theta, wires=[0, 1])
    assert np.allclose(_matrix(qml, rzz), GATES["rzz"].unitary(theta), rtol=0, atol=1e-12)
    zz = operation_for("zz")(wires=[0, 1])
    assert zz.name == "IsingZZ"
    assert np.allclose(_matrix(qml, zz), GATES["zz"].unitary(), rtol=0, atol=1e-12)


def test_every_buildable_registry_gate_has_the_registry_unitary(qml) -> None:
    from noisevault.frameworks.pennylane import operation_for

    params = (0.37, -1.1, 0.8)
    missing = set()
    for row in GATES.values():
        make = operation_for(row.name)
        if row.unitary is None:
            continue
        if make is None:
            missing.add(row.name)
            continue
        args = (0.0, 0.0) if row.name == "ms" else params[: len(row.params)]
        theirs = _matrix(qml, make(*args, wires=list(range(row.arity))))
        overlap = np.trace(theirs.conj().T @ row.unitary(*args)) / 2**row.arity
        assert abs(abs(overlap) - 1) < 1e-12, row.name
    assert missing == {"cxswap", "swapcx", "czswap"}


@pytest.mark.parametrize(("name", "base"), [("sdg", "S"), ("tdg", "T"), ("sxdg", "SX")])
def test_an_adjoint_registry_gate_builds_the_adjoint_that_gets_its_noise(qml, name, base) -> None:
    from noisevault.frameworks.pennylane import gate_name, operation_for

    op = operation_for(name)(wires=0)
    assert op.name == f"Adjoint({base})"
    assert gate_name(op, {name}) == name


def test_ms_builds_only_at_zero_phases_as_an_ising_xx_that_gets_the_ms_noise(qml) -> None:
    from noisevault.frameworks.pennylane import gate_name, operation_for

    ms = operation_for("ms")(0.0, 0.0, wires=[0, 1])
    assert ms.name == "IsingXX"
    assert np.allclose(_matrix(qml, ms), GATES["ms"].unitary(0.0, 0.0), rtol=0, atol=1e-12)
    assert gate_name(ms, {"ms", "rxx"}) == "ms"
    with pytest.raises(ValueError, match=r"builds only ms\(0, 0\).*Got ms\(0.37, 0.0\)"):
        operation_for("ms")(0.37, 0.0, wires=[0, 1])


def _trapped_ion(*, two_qubit: str = "zz") -> Profile:
    """r and a ZZ native whose noise differs from the typical natives listed first (x, cz)."""
    data = toy(
        gates={
            "rz": {"virtual": True},
            "x": {"avg_infidelity": 4e-2, "duration_ns": 35},
            "r": {"avg_infidelity": 1e-3, "duration_ns": 35},
            "cz": {"avg_infidelity": 9e-2, "duration_ns": 70},
            two_qubit: {"avg_infidelity": 6e-3, "duration_ns": 70},
        },
        idle={"t1_us": 40, "t2_us": 30},
        qubits=[
            {"index": i, "readout": {"p1_given_0": 0.01, "p0_given_1": 0.03}} for i in range(3)
        ],
    )
    return Profile.model_validate(data)


def test_rot_and_ising_zz_at_native_angles_get_the_natives_noise(qml) -> None:
    from noisevault.frameworks.pennylane import operation_for, to_pennylane

    profile = _trapped_ion()
    model = to_pennylane(profile)
    ops = [Op("r", (0,), (np.pi / 2, 0.4)), Op("zz", (0, 1)), Op("r", (1,), (1.3, -2.0))]

    @qml.qnode(qml.device("default.mixed", wires=2))
    def circuit():
        for op in ops:
            operation_for(op.name)(*op.params, wires=list(op.qubits))
        return qml.probs(wires=[0, 1])

    got = np.asarray(qml.add_noise(circuit, model)())
    want = probabilities(profile, ops, 2, readout=True)
    assert _tvd(got, want) < 1e-12
    assert not model.report.events.get("typical_noise_used")


@pytest.mark.parametrize(("angle", "phi0"), [(np.pi / 2, 0.0), (-np.pi / 2 + 2 * np.pi, np.pi)])
def test_ising_xx_at_plus_or_minus_pi_over_2_is_the_native_ms(qml, angle, phi0) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    profile = _trapped_ion(two_qubit="ms")
    model = to_pennylane(profile)

    @qml.qnode(qml.device("default.mixed", wires=2))
    def circuit():
        qml.RX(0.3, wires=0)
        qml.IsingXX(angle, wires=[0, 1])
        return qml.probs(wires=[0, 1])

    got = np.asarray(qml.add_noise(circuit, model)())
    ops = [Op("rx", (0,), (0.3,)), Op("ms", (0, 1), (phi0, 0.0))]
    assert _tvd(got, probabilities(profile, ops, 2)) < 1e-12
    assert dict(model.report.events["typical_noise_used"]) == {"rx": 1}


def test_ising_xx_at_pi_over_2_stays_rxx_on_a_profile_without_ms(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    model = to_pennylane(_trapped_ion(two_qubit="rxx"))

    @qml.qnode(qml.device("default.mixed", wires=2))
    def circuit():
        qml.IsingXX(np.pi / 2, wires=[0, 1])
        return qml.probs(wires=[0, 1])

    qml.add_noise(circuit, model)()
    assert not model.report.events.get("typical_noise_used")


def test_other_angles_keep_their_own_names(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    model = to_pennylane(_trapped_ion())

    @qml.qnode(qml.device("default.mixed", wires=2))
    def circuit():
        qml.Rot(0.3, 1.0, 0.5, wires=0)  # an r followed by a z rotation
        qml.IsingZZ(0.7, wires=[0, 1])
        return qml.probs(wires=[0, 1])

    qml.add_noise(circuit, model)()
    assert dict(model.report.events["typical_noise_used"]) == {"Rot": 1, "rzz": 1}


def test_ising_zz_at_pi_over_2_is_rzz_on_a_profile_without_zz(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    model = to_pennylane(_trapped_ion(two_qubit="rzz"))

    @qml.qnode(qml.device("default.mixed", wires=2))
    def circuit():
        qml.IsingZZ(np.pi / 2 + 2 * np.pi, wires=[0, 1])
        return qml.probs(wires=[0, 1])

    qml.add_noise(circuit, model)()
    assert not model.report.events.get("typical_noise_used")


def _noisy_natives() -> Profile:
    """ms and r much noisier than rxx and the typical h, so each choice shows in the results."""
    data = toy(
        gates={
            "ms": {"avg_infidelity": 0.2, "duration_ns": 70},
            "rxx": {"avg_infidelity": 0.01, "duration_ns": 70},
            "r": {"avg_infidelity": 0.2, "duration_ns": 35},
            "h": {"avg_infidelity": 0.01, "duration_ns": 35},
        },
        qubits=[{"index": 0}, {"index": 1}],
    )
    return Profile.model_validate(data)


def _broadcast_circuit(qml, gate):
    @qml.qnode(qml.device("default.mixed", wires=2))
    def circuit(angle):
        gate(angle)
        return qml.probs(wires=[0, 1])

    return circuit


def test_broadcast_angles_that_select_different_natives_raise(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    model = to_pennylane(_noisy_natives())
    circuit = _broadcast_circuit(qml, lambda a: qml.IsingXX(a, wires=[0, 1]))
    angles = [np.pi / 2, 0.4]
    with pytest.raises(ValueError, match=r"ms, rxx.*qml\.transforms\.broadcast_expand"):
        qml.add_noise(circuit, model)(np.array(angles))

    expanded = qml.add_noise(qml.transforms.broadcast_expand(circuit), model)(np.array(angles))
    each = [np.asarray(qml.add_noise(circuit, model)(a)) for a in angles]
    assert np.asarray(expanded) == pytest.approx(np.array(each), abs=1e-12)


@pytest.mark.parametrize(
    ("profile", "gate", "angles"),
    [
        (_noisy_natives, "IsingXX", [np.pi / 2, -np.pi / 2]),
        (_noisy_natives, "IsingXX", [0.3, 0.4]),
        (partial(_trapped_ion, two_qubit="rxx"), "IsingXX", [np.pi / 2, 0.4]),
        (_noisy_natives, "Rot", [0.2, 0.5]),
        (_noisy_natives, "RX", [0.2, 0.5]),
    ],
    ids=["ms twice", "rxx twice", "rxx without ms", "r twice", "no native at an angle"],
)
def test_broadcasts_whose_angles_share_a_native_match_running_each_angle(
    qml, profile, gate, angles
) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    apply = {
        "IsingXX": lambda a: qml.IsingXX(a, wires=[0, 1]),
        "Rot": lambda a: qml.Rot(a, 1.0, -a, wires=0),
        "RX": lambda a: qml.RX(a, wires=0),
    }[gate]
    model = to_pennylane(profile())
    circuit = _broadcast_circuit(qml, apply)

    batched = np.asarray(qml.add_noise(circuit, model)(np.array(angles)))
    each = [np.asarray(qml.add_noise(circuit, model)(a)) for a in angles]
    assert batched == pytest.approx(np.array(each), abs=1e-12)


def test_check_runs_pennylane_on_quantinuum() -> None:
    require("pennylane")
    import noisevault as nv

    (pennylane,) = nv.load("quantinuum_h1-1").check(frameworks=["pennylane"]).frameworks
    assert pennylane.passed and not pennylane.not_run
    assert {g for c in pennylane.circuits for g in c.gates} == {"r", "zz"}
