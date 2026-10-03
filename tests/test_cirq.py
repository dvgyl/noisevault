from __future__ import annotations

import warnings

import numpy as np
import pytest
from conftest import MANILA_V01, migrated, require, toy

cirq = require("cirq")

import noisevault as nv  # noqa: E402
from noisevault import gates  # noqa: E402
from noisevault.channels import ChannelSpec, superoperator  # noqa: E402
from noisevault.conversion import resolve_op  # noqa: E402
from noisevault.errors import (  # noqa: E402
    DisabledGateError,
    LayoutError,
    MissingCalibrationError,
    NoiseApproximationWarning,
    UnsupportedEffect,
)
from noisevault.frameworks.cirq import (  # noqa: E402
    ECRGate,
    NoiseVaultNoiseModel,
    gate_name,
    to_cirq,
)
from noisevault.profile import Profile  # noqa: E402
from noisevault.reference import Op  # noqa: E402
from noisevault.reference import probabilities as reference  # noqa: E402
from noisevault.report import Report  # noqa: E402

PI = np.pi
ONE_Q = ("id", "x", "y", "h", "sx", "sxdg", "rx", "ry", "r")
TWO_Q = ("cx", "cz", "iswap", "sqrt_iswap", "rzz", "rxx", "ryy", "zz", "ms", "ecr")
# Asymmetric Pauli channel (IX much larger than XI) so an operand swap is visible.
CZ_PAULI = [0.01, 0, 0, 0.002, *[0] * 11]
NONCONTIGUOUS = {0: 3, 1: 4, 2: 1}


def _distinct(**sections) -> Profile:
    """Five qubits where every gate name, pair and qubit has its own noise."""
    defs: dict = {"rz": {"virtual": True}}
    defs |= {n: {"avg_infidelity": 1e-4 * (i + 2), "duration_ns": 35} for i, n in enumerate(ONE_Q)}
    defs |= {n: {"avg_infidelity": 2e-3 * (i + 2), "duration_ns": 300} for i, n in enumerate(TWO_Q)}
    qubits = [
        {
            "index": i,
            "t1_us": 50 + 10 * i,
            "t2_us": 40 + 7 * i,
            "readout": {"p1_given_0": 0.01 * (i + 1), "p0_given_1": 0.02 + 0.005 * i},
        }
        for i in range(5)
    ]
    records = [
        {"gate": "cx", "qubits": q, "avg_infidelity": e}
        for q, e in (([0, 1], 0.011), ([1, 0], 0.013), ([3, 4], 0.017), ([4, 3], 0.019))
    ]
    records += [{"gate": "cz", "qubits": q, "pauli": CZ_PAULI} for q in ([1, 2], [4, 1])]
    data = toy(
        device={
            "name": "distinct",
            "vendor": "test",
            "technology": "superconducting",
            "num_qubits": 5,
        },
        connectivity="all_to_all",
        gates=defs,
        qubits=qubits,
        calibrations=records,
    )
    data.update(sections)
    return Profile.model_validate(data)


def _ion() -> Profile:
    return Profile.uniform(
        "ion",
        technology="trapped_ion",
        num_qubits=4,
        one_qubit_error=2e-3,
        two_qubit_error=1.5e-2,
        readout_error=0.03,
        t1_us=5e4,
        t2_us=400,
        one_qubit_ns=10_000,
        two_qubit_ns=200_000,
    )


def _ops(tree) -> list:
    return list(cirq.flatten_to_ops(tree))


def _superop(ops, physical_of) -> tuple[np.ndarray, list[int]]:
    wires = sorted({physical_of[q] for op in ops for q in op.qubits})
    specs = [
        ChannelSpec("cirq", tuple(physical_of[q] for q in op.qubits), tuple(cirq.kraus(op)))
        for op in ops
    ]
    return superoperator(specs, wires), wires


def probabilities(circuit, model, qubits, *, readout: bool = True) -> np.ndarray:
    """Exact outcome probabilities of ``circuit`` under ``model``, big-endian over ``qubits``.

    The quantum part comes from Cirq's density-matrix simulator; the readout part applies the
    confusion matrices the model attaches to a final measurement of every qubit.
    """
    simulator = cirq.DensityMatrixSimulator(noise=model, dtype=np.complex128)
    rho = simulator.simulate(circuit, qubit_order=qubits).final_density_matrix
    probs = np.real(np.diag(rho)).reshape((2,) * len(qubits))
    if readout:
        (measure,) = _ops(model.noisy_operation(cirq.measure(*qubits, key="m")))
        for (i,), confusion in measure.gate.confusion_map.items():
            probs = np.moveaxis(np.tensordot(confusion.T, probs, axes=([1], [i])), 0, i)
    return probs.reshape(-1)


def _tvd(a, b) -> float:
    return 0.5 * float(np.abs(np.asarray(a) - np.asarray(b)).sum())


# per-gate channels ---------------------------------------------------------------------------

GATE_CASES = [
    (cirq.X, (0,), "x", ()),
    (cirq.X**-1, (1,), "x", ()),
    (cirq.X**0.5, (2,), "sx", ()),
    (cirq.X**-0.5, (0,), "sxdg", ()),
    (cirq.X**0.3, (1,), "rx", (0.3 * PI,)),
    (cirq.rx(0.3), (2,), "rx", (0.3,)),
    (cirq.Y, (1,), "y", ()),
    (cirq.Y**0.25, (2,), "ry", (0.25 * PI,)),
    (cirq.H, (0,), "h", ()),
    (cirq.H**-1, (2,), "h", ()),
    (cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5), (1,), "r", (0.5 * PI, 0.25 * PI)),
    (cirq.I, (2,), "id", ()),
    (cirq.Z**0.3, (0,), "rz", (0.3 * PI,)),
    (cirq.S, (1,), "s", ()),
    (cirq.CNOT, (0, 1), "cx", ()),
    (cirq.CNOT, (1, 0), "cx", ()),
    (cirq.CZ, (1, 2), "cz", ()),
    (cirq.CZ**-1, (2, 1), "cz", ()),
    (cirq.ISWAP, (0, 2), "iswap", ()),
    (cirq.ISWAP**0.5, (2, 0), "sqrt_iswap", ()),
    (cirq.ZZ**0.3, (0, 1), "rzz", (0.3 * PI,)),
    (cirq.ZZ**0.5, (1, 0), "zz", ()),
    (cirq.XX**0.3, (0, 2), "rxx", (0.3 * PI,)),
    (cirq.YY**0.2, (2, 1), "ryy", (0.2 * PI,)),
    (cirq.ms(PI / 4), (0, 1), "ms", (0.0, 0.0)),
    (cirq.ms(-PI / 4), (1, 0), "ms", (PI, 0.0)),
    (cirq.XX**0.5, (0, 2), "ms", (0.0, 0.0)),
    (cirq.ms(0.2), (2, 1), "rxx", (0.4,)),
    (ECRGate(), (0, 1), "ecr", ()),
    (ECRGate(), (2, 0), "ecr", ()),
]


@pytest.mark.parametrize("layout", [None, NONCONTIGUOUS], ids=["identity", "noncontiguous"])
@pytest.mark.parametrize(("gate", "targets", "name", "params"), GATE_CASES, ids=repr)
def test_gate_superoperator_equals_core_channels(gate, targets, name, params, layout) -> None:
    profile = _distinct()
    model = to_cirq(profile, layout=layout, unknown_gates="error")
    qids = [cirq.LineQubit(t) for t in targets]
    physical_of = {cirq.LineQubit(c): p for c, p in (layout or {0: 0, 1: 1, 2: 2}).items()}
    physical = tuple(physical_of[q] for q in qids)

    ops = _ops(model.noisy_operation(gate.on(*qids)))
    got, wires = _superop(ops, physical_of)

    report = Report.start(profile, "test", None)
    core = resolve_op(profile.table, name, physical, unknown_gates="error", report=report)
    ideal = ChannelSpec("unitary", physical, (gates.GATES[name].unitary(*params),))
    expected = superoperator([ideal, *core.channels], wires)
    assert ops[0] == gate.on(*qids)
    assert np.abs(got - expected).max() < 1e-10


def _rotations_only() -> Profile:
    """Calibrated rotations and no fixed-angle gate (x, sx, y, s, zz) of their own."""
    defs = {n: {"avg_infidelity": e} for n, e in (("rx", 3e-3), ("ry", 5e-3), ("rz", 7e-3))}
    defs |= {
        "cx": {"qubits": 2, "avg_infidelity": 0.2},
        "rzz": {"qubits": 2, "avg_infidelity": 0.01},
        "rxx": {"qubits": 2, "avg_infidelity": 0.03},
    }
    return Profile.model_validate(toy(connectivity="all_to_all", gates=defs))


@pytest.mark.parametrize(
    ("gate", "name", "params"),
    [
        (cirq.ZZ**0.5, "rzz", (PI / 2,)),
        (cirq.ms(PI / 4), "rxx", (PI / 2,)),
        (cirq.XX**-0.5, "rxx", (-PI / 2,)),
        (cirq.X, "rx", (PI,)),
        (cirq.X**-0.5, "rx", (-PI / 2,)),
        (cirq.Y, "ry", (PI,)),
        (cirq.S, "rz", (PI / 2,)),
    ],
    ids=repr,
)
def test_fixed_angle_of_a_calibrated_rotation_gets_the_rotation_noise(gate, name, params) -> None:
    profile = _rotations_only()
    model = to_cirq(profile, unknown_gates="error")
    qids = cirq.LineQubit.range(cirq.num_qubits(gate))
    physical = tuple(range(len(qids)))
    got, wires = _superop(
        _ops(model.noisy_operation(gate.on(*qids))), dict(zip(qids, physical, strict=True))
    )

    core = resolve_op(
        profile.table,
        name,
        physical,
        unknown_gates="error",
        report=Report.start(profile, "t", None),
    )
    ideal = ChannelSpec("unitary", physical, (gates.GATES[name].unitary(*params),))
    assert np.abs(got - superoperator([ideal, *core.channels], wires)).max() < 1e-10


@pytest.mark.parametrize(
    ("defined", "exponent", "charged"),
    [
        (("p",), 0.4, "p"),
        (("p", "rz"), 0.4, "p"),
        (("rz",), 0.4, "rz"),
        (("p",), 0.5, "p"),
        (("p", "rz"), 0.5, "p"),
        (("s", "p"), 0.5, "s"),
        (("rz",), -0.5, "rz"),
    ],
)
def test_z_power_takes_the_calibration_of_a_gate_it_equals(defined, exponent, charged) -> None:
    errors = {"p": 0.2, "rz": 0.1, "s": 0.05}
    defs = {"h": {"avg_infidelity": 0.01}} | {n: {"avg_infidelity": errors[n]} for n in defined}
    profile = Profile.model_validate(toy(connectivity="all_to_all", gates=defs))
    q = cirq.LineQubit(0)
    model = to_cirq(profile, unknown_gates="error")
    got, wires = _superop(_ops(model.noisy_operation(cirq.ZPowGate(exponent=exponent)(q))), {q: 0})

    core = resolve_op(
        profile.table, charged, (0,), unknown_gates="error", report=Report.start(profile, "t", None)
    )
    ideal = ChannelSpec("unitary", (0,), (cirq.unitary(cirq.ZPowGate(exponent=exponent)),))
    assert np.abs(got - superoperator([ideal, *core.channels], wires)).max() < 1e-10


def test_check_runs_p_at_an_angle_no_fixed_gate_has() -> None:
    defs = {n: {"avg_infidelity": e} for n, e in (("h", 0.01), ("p", 0.2), ("s", 0.05))}
    defs["cz"] = {"qubits": 2, "avg_infidelity": 0.02}
    (result,) = Profile.model_validate(toy(gates=defs)).check(frameworks=["cirq"]).frameworks
    assert result.passed and not result.not_run
    assert "p" in {g for c in result.circuits for g in c.gates}


@pytest.mark.parametrize(("zz", "angle"), [(False, PI / 2), (True, PI / 4)], ids=["rzz", "rzz+zz"])
def test_check_runs_rzz_at_the_zz_angle_unless_the_profile_defines_zz(zz, angle) -> None:
    profile = _rotations_only()
    if zz:
        data = profile.to_dict()
        data["gates"]["zz"] = {"qubits": 2, "avg_infidelity": 0.05}
        profile = Profile.model_validate(data)
    result = profile.check(frameworks=["cirq"])
    assert {op.params for c in result.circuits for op in c.ops if op.name == "rzz"} == {(angle,)}
    assert result.passed


def test_every_registry_cirq_class_exists() -> None:
    modules = [cirq]
    try:
        import cirq_google

        modules.append(cirq_google)
    except ImportError:
        pass
    named = {i.name: i.cirq for i in gates.GATES.values() if i.cirq}
    missing = {n: c for n, c in named.items() if not any(hasattr(m, c) for m in modules)}
    assert missing == {}


# circuits against the reference simulator ------------------------------------------------------


def _ghz_native(n: int) -> tuple[list, list[Op]]:
    """GHZ from rz, sx and cx only: H = rz(pi/2) sx rz(pi/2) up to a phase."""
    q = cirq.LineQubit.range(n)
    circuit = [cirq.rz(PI / 2)(q[0]), (cirq.X**0.5)(q[0]), cirq.rz(PI / 2)(q[0])]
    ops = [Op("rz", (0,), (PI / 2,)), Op("sx", (0,)), Op("rz", (0,), (PI / 2,))]
    for i in range(n - 1):
        circuit.append(cirq.CNOT(q[i], q[i + 1]))
        ops.append(Op("cx", (i, i + 1)))
    return circuit, ops


def _mirror_native(n: int, pairs: list[tuple[int, int]], seed: int) -> tuple[list, list[Op]]:
    """A random rz/sx/x/cx layer followed by its inverse written in the same gates."""
    rng = np.random.default_rng(seed)
    q = cirq.LineQubit.range(n)
    layer: list[tuple[str, tuple[int, ...], float]] = []
    for i in range(n):
        layer += [("rz", (i,), float(rng.uniform(-PI, PI))), ("sx", (i,), 0.0)]
        if rng.random() < 0.5:
            layer.append(("x", (i,), 0.0))
    layer += [("cx", pair, 0.0) for pair in pairs]

    def emit(name: str, t: tuple[int, ...], angle: float) -> tuple[list, list[Op]]:
        if name == "rz":
            return [cirq.rz(angle)(q[t[0]])], [Op("rz", t, (angle,))]
        gate = {"sx": cirq.X**0.5, "x": cirq.X, "cx": cirq.CNOT}[name]
        return [gate(*(q[i] for i in t))], [Op(name, t)]

    circuit: list = []
    ops: list[Op] = []
    for step in layer:
        c, o = emit(*step)
        circuit += c
        ops += o
    for name, t, angle in reversed(layer):
        undo = (
            [("rz", t, PI), ("sx", t, 0.0), ("rz", t, PI)]
            if name == "sx"
            else [(name, t, -angle if name == "rz" else 0.0)]
        )
        for step in undo:
            c, o = emit(*step)
            circuit += c
            ops += o
    return circuit, ops


def _conformance_cases():
    manila = migrated(MANILA_V01)
    yield "manila-ghz5", manila, *_ghz_native(5), None
    yield "manila-mirror5", manila, *_mirror_native(5, [(1, 0), (2, 3), (1, 2), (4, 3)], 7), None
    yield "manila-ghz3-layout", manila, *_ghz_native(3), {0: 4, 1: 3, 2: 2}
    ion = _ion()
    yield "ion-ghz4", ion, *_ghz_native(4), [3, 0, 2, 1]
    yield "ion-mirror4", ion, *_mirror_native(4, [(0, 3), (2, 1), (1, 3)], 11), None


@pytest.mark.parametrize(
    ("profile", "circuit", "ops", "layout"),
    [case[1:] for case in _conformance_cases()],
    ids=[case[0] for case in _conformance_cases()],
)
def test_exported_model_matches_reference(profile, circuit, ops, layout) -> None:
    n = 1 + max(max(op.qubits) for op in ops)
    qubits = cirq.LineQubit.range(n)
    model = to_cirq(profile, layout=layout, unknown_gates="error")
    kwargs = {"layout": layout, "unknown_gates": "error"}
    for readout in (False, True):
        got = probabilities(cirq.Circuit(circuit), model, qubits, readout=readout)
        want = reference(profile, ops, n, readout=readout, **kwargs)
        assert _tvd(got, want) <= 1e-9
    # terminal measurements carry readout as a channel: the state just before them is the
    # distribution of reported outcomes
    measured = cirq.Circuit(circuit, cirq.measure(*qubits, key="m")).with_noise(model)
    unmeasured = cirq.Circuit(op for op in measured.all_operations() if not cirq.is_measurement(op))
    rho = (
        cirq.DensityMatrixSimulator(dtype=np.complex128)
        .simulate(unmeasured, qubit_order=qubits)
        .final_density_matrix
    )
    assert _tvd(np.real(np.diag(rho)), reference(profile, ops, n, **kwargs)) <= 1e-9


def test_sampled_terminal_measurements_include_readout() -> None:
    """Sampled counts agree with the exact readout within 5 sigma per outcome."""
    profile = _distinct()
    q = cirq.LineQubit.range(2)
    body = [cirq.X(q[0]), cirq.X(q[1]) ** 0.5]
    model = to_cirq(profile, layout={0: 3, 1: 1})
    exact = probabilities(cirq.Circuit(body), model, q)
    without_readout = probabilities(cirq.Circuit(body), model, q, readout=False)
    circuit = cirq.Circuit([*body, cirq.measure(*q, key="m")])
    # the state-vector simulator samples trajectories, one full run per shot
    for simulator, shots in ((cirq.DensityMatrixSimulator, 200_000), (cirq.Simulator, 5_000)):
        result = simulator(noise=model, seed=5).run(circuit, repetitions=shots)
        counts = np.bincount(result.measurements["m"] @ [2, 1], minlength=4) / shots
        sigma = np.sqrt(exact * (1 - exact) / shots)
        assert np.all(np.abs(counts - exact) <= 5 * sigma), simulator.__name__
        assert np.any(np.abs(counts - without_readout) > 5 * sigma)  # the check can fail


def test_mid_circuit_readout_error_leaves_the_true_state() -> None:
    """A misread bit does not flip the qubit: after X the second readout sees |1>."""
    model = to_cirq(_distinct(), layout={0: 4})  # P(1|0) = 0.05, P(0|1) = 0.04
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(q, key="a"), cirq.X(q), cirq.measure(q, key="b"))
    shots = 2000
    result = cirq.DensityMatrixSimulator(noise=model, seed=3).run(circuit, repetitions=shots)
    a, b = result.measurements["a"][:, 0], result.measurements["b"][:, 0]
    assert abs(a.mean() - 0.05) <= 5 * np.sqrt(0.05 * 0.95 / shots)
    # a state-flipping readout would give P(b=1 | a=1) = P(1|0) = 0.05 instead
    assert b[a == 1].mean() > 0.8 and b[a == 0].mean() > 0.8


def test_trajectory_runs_convert_the_circuit_once() -> None:
    model = to_cirq(migrated(MANILA_V01))
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q[0]), cirq.CNOT(*q), cirq.measure(*q, key="m"))
    with pytest.warns(NoiseApproximationWarning):
        cirq.Simulator(noise=model, seed=2).run(circuit, repetitions=200)
    assert model.report.events["typical_noise_used"] == {"h": 1}


def test_measurement_gets_cirq_confusion_matrix_per_qubit() -> None:
    profile = _distinct()
    model = to_cirq(profile, layout=NONCONTIGUOUS)
    q = cirq.LineQubit.range(3)
    original = cirq.measure(q[0], q[2], key="out", invert_mask=(True,))
    (noisy,) = _ops(model.noisy_operation(original))
    # device qubit 3 then device qubit 1; rows are the true outcome, columns the reported one
    assert set(noisy.gate.confusion_map) == {(0,), (1,)}
    assert np.allclose(noisy.gate.confusion_map[(0,)], [[0.96, 0.04], [0.035, 0.965]])
    assert np.allclose(noisy.gate.confusion_map[(1,)], [[0.98, 0.02], [0.025, 0.975]])
    assert noisy.gate.key == "out" and noisy.gate.invert_mask == (True,)
    assert noisy.qubits == original.qubits

    plain = to_cirq(profile, layout=NONCONTIGUOUS, readout=False)
    assert _ops(plain.noisy_operation(original)) == [original]
    assert "readout error (readout=False)" in plain.report.to_dict()["omitted"]


def test_existing_confusion_map_is_refused() -> None:
    model = to_cirq(_distinct())
    q = cirq.LineQubit(0)
    op = cirq.MeasurementGate(1, key="m", confusion_map={(0,): np.eye(2)}).on(q)
    with pytest.raises(ValueError, match="readout=False"):
        model.noisy_operation(op)


def test_unknown_readout_is_reported_and_left_noiseless() -> None:
    profile = Profile.model_validate(toy())
    model = to_cirq(profile)
    op = cirq.measure(cirq.LineQubit(1), key="m")
    assert _ops(model.noisy_operation(op)) == [op]
    assert model.report.to_dict()["unknown"] == ["readout of qubit 1"]


# reset, wait ---------------------------------------------------------------------------------


def _final_rho(model, circuit, initial: np.ndarray) -> np.ndarray:
    simulator = cirq.DensityMatrixSimulator(noise=model, dtype=np.complex128)
    return simulator.simulate(cirq.Circuit(circuit), initial_state=initial).final_density_matrix


def test_reset_gets_the_preparation_error() -> None:
    q = cirq.LineQubit(0)
    excited = np.diag([0.0, 1.0]).astype(complex)
    model = to_cirq(Profile.model_validate(toy(prep={"error": 0.03})))
    assert _final_rho(model, [cirq.reset(q)], excited)[1, 1].real == pytest.approx(0.03, abs=1e-12)

    manila = to_cirq(migrated(MANILA_V01))
    assert _ops(manila.noisy_operation(cirq.reset(q))) == [cirq.reset(q)]
    assert "preparation error of qubit 0" in manila.report.to_dict()["unknown"]


def test_wait_gate_relaxes_with_t1_and_t2() -> None:
    q = cirq.LineQubit(0)
    model = to_cirq(_distinct())  # qubit 0: T1 = 50 us, T2 = 40 us
    wait = cirq.wait(q, nanos=10_000)
    excited = np.diag([0.0, 1.0]).astype(complex)
    plus = np.full((2, 2), 0.5, dtype=complex)
    assert _final_rho(model, [wait], excited)[1, 1].real == pytest.approx(np.exp(-10 / 50), 1e-12)
    assert abs(_final_rho(model, [wait], plus)[0, 1]) == pytest.approx(
        0.5 * np.exp(-10 / 40), 1e-12
    )


def test_wait_without_coherence_times_is_reported() -> None:
    model = to_cirq(Profile.model_validate(toy()))
    wait = cirq.wait(cirq.LineQubit(2), nanos=500)
    assert _ops(model.noisy_operation(wait)) == [wait]
    assert model.report.to_dict()["unknown"] == ["T1 and T2 of qubit 2 (no WaitGate relaxation)"]


def test_wait_clamps_t2_above_2_t1_and_reports_it() -> None:
    model = to_cirq(Profile.model_validate(toy(qubits=[{"index": 1, "t1_us": 10, "t2_us": 100}])))
    plus = np.full((2, 2), 0.5, dtype=complex)
    rho = _final_rho(model, [cirq.wait(cirq.LineQubit(1), nanos=10_000)], plus)
    assert abs(rho[0, 1]) == pytest.approx(0.5 * np.exp(-10 / 20), 1e-12)
    t2 = [(a.what, a.how) for a in model.report.approximated if a.what.startswith("T2")]
    assert t2 == [("T2 of qubit 1", "clamped to 2*T1")]


# qubit mapping -------------------------------------------------------------------------------


def _with_coords() -> Profile:
    qubits = [{"index": i, "t1_us": 30 + 20 * i, "coords": [i // 2, i % 2]} for i in range(4)]
    return Profile.model_validate(
        toy(
            device={
                "name": "grid",
                "vendor": "test",
                "technology": "superconducting",
                "num_qubits": 4,
            },
            connectivity="all_to_all",
            qubits=qubits,
        )
    )


def test_grid_qubits_map_through_profile_coords() -> None:
    profile = _with_coords()
    model = to_cirq(profile)
    op = (cirq.X**0.5).on(cirq.GridQubit(1, 0))
    ops = _ops(model.noisy_operation(op))
    got, _ = _superop(ops, {cirq.GridQubit(1, 0): 2})
    core = resolve_op(
        profile.table, "sx", (2,), unknown_gates="error", report=Report.start(profile, "t", None)
    )
    ideal = ChannelSpec("unitary", (2,), (gates.GATES["sx"].unitary(),))
    assert np.abs(got - superoperator([ideal, *core.channels], [2])).max() < 1e-12

    with pytest.raises(LayoutError, match=r"no qubit at coords \(5, 5\)") as caught:
        model.noisy_operation(cirq.X(cirq.GridQubit(5, 5)))
    assert caught.value.hint == (
        "use the device's coords, for example GridQubit(0.0, 0.0), GridQubit(0.0, 1.0),"
        " GridQubit(1.0, 0.0), or pass layout={cirq.GridQubit(5, 5): <device qubit>, ...}"
        " covering every circuit qubit"
    )


def test_coords_examples_are_qubits_a_grid_qubit_can_map_to() -> None:
    data = _with_coords().to_dict()
    data["qubits"][0]["disabled"] = True
    model = to_cirq(Profile.model_validate(data))
    with pytest.raises(LayoutError, match=r"no qubit at coords \(5, 5\)") as caught:
        model.noisy_operation(cirq.X(cirq.GridQubit(5, 5)))
    assert caught.value.hint == (
        "use the device's coords, for example GridQubit(0.0, 1.0), GridQubit(1.0, 0.0),"
        " GridQubit(1.0, 1.0), or pass layout={cirq.GridQubit(5, 5): <device qubit>, ...}"
        " covering every circuit qubit"
    )


def test_a_disabled_qubit_hint_needs_a_chain_as_wide_as_the_layout() -> None:
    def hint(num_qubits: int, layout: list[int] | None) -> str | None:
        device = {**toy()["device"], "num_qubits": num_qubits}
        data = toy(
            device=device, connectivity="all_to_all", qubits=[{"index": 1, "disabled": True}]
        )
        a, b = cirq.LineQubit.range(2)
        circuit = cirq.Circuit((cirq.X**0.5)(b), cirq.measure(a, key="m"))
        with pytest.raises(LayoutError, match="disabled") as caught:
            sim = cirq.DensityMatrixSimulator(
                noise=to_cirq(Profile.model_validate(data), layout=layout)
            )
            sim.run(circuit)
        return caught.value.hint

    assert hint(3, [0, 1]) == (
        "choose another qubit (profile.suggest_layout(2) proposes a usable chain)"
    )
    assert hint(2, [0, 1]) is None
    assert hint(3, None) is None


def test_default_placement_must_be_injective_within_a_circuit_only() -> None:
    model = to_cirq(_with_coords())
    line, grid = cirq.LineQubit(0), cirq.GridQubit(0, 0)  # both default to device qubit 0
    for q in (line, grid):
        noisy = cirq.Circuit((cirq.X**0.5)(q)).with_noise(model)
        assert noisy.all_qubits() == {q}
    with pytest.raises(LayoutError, match="both map to device qubit 0") as caught:
        cirq.Circuit((cirq.X**0.5)(line), (cirq.X**0.5)(grid)).with_noise(model)
    assert caught.value.hint is None


def _shared_coords(*, second_disabled: bool = False) -> Profile:
    qubits = [
        {"index": 0, "coords": [0, 0]},
        {"index": 1, "coords": [0, 0], "disabled": second_disabled},
        {"index": 2, "coords": [0, 1]},
    ]
    return Profile.model_validate(toy(qubits=qubits))


def test_grid_qubit_at_coords_of_two_enabled_qubits_needs_a_layout() -> None:
    model = to_cirq(_shared_coords())
    with pytest.raises(LayoutError) as caught:
        model.noisy_operation(cirq.X(cirq.GridQubit(0, 0)))
    assert str(caught.value).startswith(
        "test_toy has qubits 0 and 1 at coords (0, 0), so cirq.GridQubit(0, 0) has no single device"
        " qubit"
    )
    assert caught.value.hint == (
        "pass layout={cirq.GridQubit(0, 0): <device qubit>, ...} covering every circuit qubit"
    )
    assert model._qubits.physical([cirq.GridQubit(0, 1)]) == (2,)
    placed = to_cirq(_shared_coords(), layout={cirq.GridQubit(0, 0): 1})
    assert placed._qubits.physical([cirq.GridQubit(0, 0)]) == (1,)


def test_grid_qubit_at_coords_of_an_enabled_and_a_disabled_qubit_takes_the_enabled_one() -> None:
    model = to_cirq(_shared_coords(second_disabled=True))
    assert model._qubits.physical([cirq.GridQubit(0, 0)]) == (0,)


@pytest.mark.parametrize(
    ("layout", "first", "second"),
    [
        ({0: 99, cirq.LineQubit(0): 1}, "0", "cirq.LineQubit(0)"),
        ({cirq.LineQubit(0): 1, 0: 0}, "cirq.LineQubit(0)", "0"),
    ],
)
def test_layout_keys_that_name_one_qubit_raise(layout, first, second) -> None:
    with pytest.raises(LayoutError) as caught:
        to_cirq(Profile.model_validate(toy()), layout=layout)
    assert str(caught.value).startswith(
        f"layout keys {first} and {second} both name cirq.LineQubit(0)"
    )
    assert caught.value.hint == "keep one of the two keys"


_COVERING = ": <device qubit>, ...} covering every circuit qubit"


@pytest.mark.parametrize(
    ("profile", "qubit", "layout", "match", "hint"),
    [
        (
            lambda: Profile.model_validate(toy()),
            cirq.GridQubit(0, 1),
            None,
            "records no qubit",
            "pass layout={cirq.GridQubit(0, 1)" + _COVERING,
        ),
        (
            lambda: Profile.model_validate(toy()),
            cirq.NamedQubit("a"),
            None,
            "needs a layout",
            "pass layout={cirq.NamedQubit('a')" + _COVERING,
        ),
        (lambda: Profile.model_validate(toy()), cirq.LineQubit(7), None, "has qubits 0..2", None),
        (
            lambda: Profile.model_validate(toy()),
            cirq.LineQubit(1),
            {0: 2},
            "no device qubit",
            "map every circuit qubit in layout=",
        ),
        (
            lambda: Profile.model_validate(toy(qubits=[{"index": 1, "disabled": True}])),
            cirq.LineQubit(1),
            None,
            "disabled",
            None,
        ),
    ],
    ids=["grid-no-coords", "named", "out-of-range", "missing-from-layout", "disabled-qubit"],
)
def test_unmappable_qubits_raise_layout_error(profile, qubit, layout, match, hint) -> None:
    model = to_cirq(profile(), layout=layout)
    with pytest.raises(LayoutError, match=match) as caught:
        model.noisy_operation(cirq.X(qubit))
    assert caught.value.hint == hint


def test_explicit_layout_places_named_qubits() -> None:
    profile = _distinct()
    a, b = cirq.NamedQubit("a"), cirq.NamedQubit("b")
    model = to_cirq(profile, layout={a: 4, b: 3})
    ops = _ops(model.noisy_operation(cirq.CNOT(a, b)))
    got, wires = _superop(ops, {a: 4, b: 3})
    core = resolve_op(
        profile.table, "cx", (4, 3), unknown_gates="error", report=Report.start(profile, "t", None)
    )
    assert core.requested == 0.019
    ideal = ChannelSpec("unitary", (4, 3), (gates.GATES["cx"].unitary(),))
    assert np.abs(got - superoperator([ideal, *core.channels], wires)).max() < 1e-12
    with pytest.raises(LayoutError, match="both"):
        to_cirq(profile, layout={a: 4, b: 4})


# unknown and disabled gates, report ----------------------------------------------------------


def test_unknown_gate_gets_typical_noise_with_one_warning() -> None:
    model = to_cirq(_distinct())
    q = cirq.LineQubit.range(2)
    fsim = cirq.FSimGate(0.3, 0.1)
    with pytest.warns(NoiseApproximationWarning, match="fsim") as caught:
        first = _ops(model.noisy_operation(fsim(q[0], q[1])))
        model.noisy_operation(fsim(q[0], q[1]))
    assert len(caught) == 1
    assert len(first) > 1  # typical noise was added
    assert model.report.events["typical_noise_used"] == {"fsim": 2}
    assert [a.what for a in model.report.approximated] == ["gate fsim"]


@pytest.mark.parametrize(
    ("gate", "name"),
    [
        (cirq.CZ**0.5, "cz**0.5"),
        (cirq.FSimGate(0.3, 0.1), "fsim"),
        (cirq.MatrixGate(cirq.unitary(cirq.H)), "matrix"),
    ],
)
def test_non_native_cirq_gates_are_unknown(gate, name) -> None:
    model = to_cirq(_distinct(), unknown_gates="error")
    qids = cirq.LineQubit.range(cirq.num_qubits(gate))
    locus = "qubit 0" if len(qids) == 1 else "qubits 0-1"
    with pytest.raises(MissingCalibrationError, match=rf"^{name.replace('*', '[*]')} on {locus}: "):
        model.noisy_operation(gate.on(*qids))


def test_ms_off_its_native_angle_is_an_xx_rotation() -> None:
    defs = {"rz": {"virtual": True}, "ms": {"qubits": 2, "avg_infidelity": 0.02}}
    model = to_cirq(Profile.model_validate(toy(connectivity="all_to_all", gates=defs)))
    q = cirq.LineQubit.range(2)
    for native in (cirq.ms(PI / 4), cirq.ms(-PI / 4), cirq.XX**0.5, cirq.ms(PI / 4) ** 5):
        model.noisy_operation(native.on(*q))
    assert "typical_noise_used" not in model.report.events
    with pytest.warns(NoiseApproximationWarning, match="rxx"):
        model.noisy_operation(cirq.ms(0.2).on(*q))
    assert model.report.events["typical_noise_used"] == {"rxx": 1}


def test_unknown_gates_option_is_validated_up_front() -> None:
    with pytest.raises(ValueError, match="choose 'typical' or 'error'"):
        to_cirq(_distinct(), unknown_gates="ignore")


def test_swap_must_be_decomposed() -> None:
    model = to_cirq(_distinct())
    with pytest.raises(MissingCalibrationError, match="decompose"):
        model.noisy_operation(cirq.SWAP(*cirq.LineQubit.range(2)))


def test_disabled_gate_raises() -> None:
    profile = _distinct(calibrations=[{"gate": "cx", "qubits": [0, 1], "disabled": True}])
    model = to_cirq(profile)
    with pytest.raises(DisabledGateError, match=r"^cx on qubits 0-1 is disabled in this profile$"):
        model.noisy_operation(cirq.CNOT(*cirq.LineQubit.range(2)))


def _disabling(gate: str, where: str) -> Profile:
    data = toy(
        readout={"p1_given_0": 0.02, "p0_given_1": 0.1},
        prep={"error": 0.03},
        idle={"t1_us": 50, "t2_us": 40},
    )
    if where == "record":
        data["gates"][gate] = {}
        data["calibrations"] = [{"gate": gate, "qubits": [2], "disabled": True}]
    else:
        data["gates"][gate] = {"disabled": True}
    return Profile.model_validate(data)


_MEASURE_X = cirq.PauliMeasurementGate(cirq.DensePauliString("X"), key="p")
_SPECIAL = {
    "reset": lambda a, b: cirq.reset(b),
    "measure": lambda a, b: cirq.measure(a, b, key="m"),
    "pauli_measure": lambda a, b: _MEASURE_X.on(b),
    "delay": lambda a, b: cirq.wait(a, b, nanos=100),
}


@pytest.mark.parametrize("readout", [True, False])
@pytest.mark.parametrize("where", ["record", "definition"])
@pytest.mark.parametrize("kind", list(_SPECIAL))
def test_a_reset_measure_or_wait_the_profile_disables_is_refused(kind, where, readout) -> None:
    gate = "measure" if kind == "pauli_measure" else kind
    layout = [0, 2]
    model = to_cirq(_disabling(gate, where), layout=layout, readout=readout)
    operation = _SPECIAL[kind](*cirq.LineQubit.range(2))
    index = 2 if where == "record" else layout[operation.qubits[0].x]
    message = rf"^{gate} on qubit {index} is disabled in this profile$"
    with pytest.raises(DisabledGateError, match=message):
        model.noisy_operation(operation)


@pytest.mark.parametrize("gate", ["reset", "measure", "delay"])
def test_a_reset_measure_or_wait_where_the_profile_allows_it_keeps_its_noise(gate) -> None:
    q = cirq.LineQubit(0)
    ops = {"reset": cirq.reset(q), "measure": cirq.measure(q), "delay": cirq.wait(q, nanos=100)}
    profile = _disabling(gate, "record")
    data = profile.model_dump(mode="json", exclude_none=True)
    del data["gates"][gate], data["calibrations"]
    allowed = Profile.model_validate(data)
    noisy, plain = (
        to_cirq(p, layout=[0, 2]).noisy_operation(ops[gate]) for p in (profile, allowed)
    )
    assert repr(_ops(noisy)) == repr(_ops(plain))
    assert len(_ops(noisy)) == (1 if gate == "measure" else 2)


def test_circuit_channels_are_kept_and_counted() -> None:
    model = to_cirq(_distinct())
    op = cirq.depolarize(0.1).on(cirq.LineQubit(0))
    assert _ops(model.noisy_operation(op)) == [op]
    assert model.report.events["circuit_channel_kept"] == {"DepolarizingChannel": 1}


def test_report_describes_the_conversion() -> None:
    profile = _distinct(effects=[{"type": "leakage", "gate": "cz", "prob": 1e-4}])
    model = to_cirq(profile, layout=NONCONTIGUOUS)
    assert isinstance(model, NoiseVaultNoiseModel) and isinstance(model, cirq.NoiseModel)
    report = model.report
    assert model.profile is profile
    assert (report.framework, report.framework_version) == ("cirq", cirq.__version__)
    assert (report.profile_id, report.fingerprint) == (profile.id, profile.fingerprint)
    assert report.options == {"layout": NONCONTIGUOUS, "unknown_gates": "typical", "readout": True}
    assert any(e.startswith("readout assignment error") for e in report.exact)
    assert "effect leakage on cz" in report.to_dict()["omitted"]
    assert report.to_dict()["framework"] == "cirq"

    strict = _distinct(effects=[{"type": "leakage", "gate": "cz", "prob": 1e-4, "allow": "exact"}])
    with pytest.raises(UnsupportedEffect):
        to_cirq(strict)


def test_typical_noise_warnings_point_at_the_callers_line() -> None:
    q = cirq.LineQubit.range(2)
    model = migrated(MANILA_V01).to_cirq()
    with pytest.warns(NoiseApproximationWarning) as caught:
        cirq.Circuit(cirq.H(q[0]), cirq.CZ(*q)).with_noise(model)
    assert [w.filename for w in caught] == [__file__] * 2


def test_first_call_just_works_on_a_bundled_profile() -> None:
    profile = nv.load("ibm_manila")
    model = profile.to_cirq()
    q = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.H(q[0]), cirq.CNOT(q[0], q[1]), cirq.CNOT(q[1], q[2]))
    circuit.append(cirq.measure(*q, key="m"))
    shots = 20_000
    with pytest.warns(NoiseApproximationWarning, match="h on qubit 0: "):
        result = cirq.DensityMatrixSimulator(noise=model, seed=1).run(circuit, repetitions=shots)
    counts = np.bincount(result.measurements["m"] @ [4, 2, 1], minlength=8) / shots
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoiseApproximationWarning)
        exact = reference(profile, [Op("h", (0,)), Op("cx", (0, 1)), Op("cx", (1, 2))], 3)
    assert np.all(np.abs(counts - exact) <= 5 * np.sqrt(exact * (1 - exact) / shots))
    assert model.report.events["typical_noise_used"] == {"h": 1}


# Cirq's own gates and edge operations -------------------------------------------------------


def test_phased_xz_is_r_then_a_z_rotation_with_their_noise() -> None:
    """PhasedXZ gets r's noise between its X and Z parts, which an asymmetric channel shows."""
    profile = _distinct(calibrations=[{"gate": "r", "qubits": [0], "pauli": [0.05, 0, 0.01]}])
    model = to_cirq(profile, unknown_gates="error")
    x, z, a = 0.3, 0.4, 0.2
    q = cirq.LineQubit(0)
    gate = cirq.PhasedXZGate(x_exponent=x, z_exponent=z, axis_phase_exponent=a)
    got = probabilities(cirq.Circuit(gate(q), cirq.H(q)), model, [q], readout=False)
    ops = [Op("r", (0,), (PI * x, PI * a)), Op("rz", (0,), (PI * z,)), Op("h", (0,))]
    assert _tvd(got, reference(profile, ops, 1, readout=False)) <= 1e-9
    assert "typical_noise_used" not in model.report.events


@pytest.mark.parametrize("unknown_gates", ["error", "typical"])
@pytest.mark.parametrize(
    ("defined", "z", "z_part"),
    [
        (("p",), 0.4, Op("p", (0,), (PI * 0.4,))),
        (("p", "rz"), 0.4, Op("p", (0,), (PI * 0.4,))),
        (("rz",), 0.4, Op("rz", (0,), (PI * 0.4,))),
        (("s", "p", "rz"), 0.5, Op("s", (0,))),
    ],
)
def test_phased_xz_charges_its_z_part_as_a_z_power_on_its_own(
    defined, z, z_part, unknown_gates
) -> None:
    errors = {"h": 0.01, "r": 0.01, "p": 0.2, "rz": 0.1, "s": 0.05}
    defs = {n: {"avg_infidelity": errors[n]} for n in ("h", "r", *defined)}
    profile = Profile.model_validate(toy(gates=defs))
    x, a = 0.3, 0.2
    q = cirq.LineQubit(0)
    packed = cirq.PhasedXZGate(x_exponent=x, z_exponent=z, axis_phase_exponent=a)
    explicit = [cirq.PhasedXPowGate(phase_exponent=a, exponent=x), cirq.Z**z]
    ops = [Op("r", (0,), (PI * x, PI * a)), z_part, Op("h", (0,))]
    want = reference(profile, ops, 1, readout=False, unknown_gates=unknown_gates)
    for gates_in in ([packed], explicit):
        model = to_cirq(profile, unknown_gates=unknown_gates)
        circuit = cirq.Circuit([g(q) for g in gates_in], cirq.H(q))
        assert _tvd(probabilities(circuit, model, [q], readout=False), want) <= 1e-9
        assert "typical_noise_used" not in model.report.events
    # Like a Z**t of its own, an unresolved Z part may turn out to be a fixed gate.
    import sympy

    unresolved = cirq.PhasedXZGate(
        x_exponent=x, z_exponent=sympy.Symbol("t"), axis_phase_exponent=a
    )
    with pytest.raises(ValueError, match="Resolve the parameters before you add noise"):
        cirq.Circuit(unresolved(q)).with_noise(to_cirq(profile, unknown_gates=unknown_gates))


def test_phased_xz_calibrated_as_its_own_gate_gets_that_calibration() -> None:
    defs = {
        "r": {"avg_infidelity": 2e-2},
        "rz": {"avg_infidelity": 3e-2},
        "phased_xz": {"qubits": 1, "pauli": [0.05, 0, 0.01]},
    }
    profile = Profile.model_validate(toy(gates=defs))
    model = to_cirq(profile, unknown_gates="error")
    q = cirq.LineQubit(0)
    gate = cirq.PhasedXZGate(x_exponent=0.3, z_exponent=0.2, axis_phase_exponent=0.4)
    ops = _ops(model.noisy_operation(gate(q)))
    got, _ = _superop(ops, {q: 0})

    core = resolve_op(
        profile.table,
        "phased_xz",
        (0,),
        unknown_gates="error",
        report=Report.start(profile, "t", None),
    )
    ideal = ChannelSpec("unitary", (0,), (cirq.unitary(gate),))
    assert ops[0] == gate(q)
    assert np.abs(got - superoperator([ideal, *core.channels], [0])).max() < 1e-12


def test_google_gatesets_are_native_on_rainbow() -> None:
    cirq_google = require("cirq_google")
    from noisevault.sources.google import from_cirq_google

    profile = from_cirq_google("rainbow")
    a, b = cirq.GridQubit(5, 3), cirq.GridQubit(5, 4)
    circuit = cirq.Circuit(cirq.H(a), cirq.CNOT(a, b), cirq.SQRT_ISWAP_INV(a, b))
    circuit.append(cirq.measure(a, b, key="m"))
    for gateset in (cirq.SqrtIswapTargetGateset(), cirq_google.SycamoreTargetGateset()):
        compiled = cirq.optimize_for_target_gateset(circuit, gateset=gateset)
        model = to_cirq(profile)
        with warnings.catch_warnings():
            warnings.simplefilter("error", NoiseApproximationWarning)
            noisy = compiled.with_noise(model)
        assert "typical_noise_used" not in model.report.events, type(gateset).__name__
        added = [op for op in noisy.all_operations() if isinstance(op.gate, cirq.KrausChannel)]
        assert len(added) > len(list(compiled.all_operations()))
    model = to_cirq(profile, unknown_gates="error")
    inverse = _ops(model.noisy_operation(cirq.SQRT_ISWAP_INV(a, b)))
    assert inverse[1:] == _ops(model.noisy_operation(cirq.SQRT_ISWAP(a, b)))[1:]


@pytest.mark.parametrize("value", ["none", "exact", 1, None])
def test_readout_option_is_validated_up_front(value) -> None:
    with pytest.raises(TypeError, match="True to add readout error or False"):
        to_cirq(_distinct(), readout=value)


def test_terminal_readout_flips_are_reported_as_part_of_the_state() -> None:
    model = to_cirq(_distinct())
    q = cirq.LineQubit(0)
    cirq.Circuit(cirq.measure(q, key="a"), cirq.X(q)).with_noise(model)
    assert not model.report.approximated
    cirq.Circuit(cirq.X(q), cirq.measure(q, key="b")).with_noise(model)
    assert [a.what for a in model.report.approximated] == ["state after a terminal measurement"]


def test_pauli_measurement_needs_z_basis_or_no_readout() -> None:
    op = cirq.measure_single_paulistring(cirq.X(cirq.LineQubit(0)), key="p")
    with pytest.raises(ValueError, match="Rotate into the Z basis"):
        to_cirq(_distinct()).noisy_operation(op)
    model = to_cirq(_distinct(), readout=False)
    assert _ops(model.noisy_operation(op)) == [op]
    assert not model.report.events


def test_classically_controlled_operations_are_refused_with_a_way_out() -> None:
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q[0], key="a"),
        cirq.X(q[1]).with_classical_controls("a"),
        cirq.measure(q[1], key="b"),
    )
    simulator = cirq.DensityMatrixSimulator(noise=to_cirq(_distinct()))
    with pytest.raises(ValueError, match="classically controlled.*quantum-controlled gate"):
        simulator.run(circuit, repetitions=2)


def test_multi_qubit_identity_is_one_id_per_qubit() -> None:
    model = to_cirq(_distinct(), layout=NONCONTIGUOUS)
    q = cirq.LineQubit.range(3)
    ops = _ops(model.noisy_operation(cirq.IdentityGate(2).on(q[0], q[2])))
    each = [op for qubit in (q[0], q[2]) for op in _ops(model.noisy_operation(cirq.I(qubit)))[1:]]
    assert ops[1:] == each
    assert {op.qubits for op in each} == {(q[0],), (q[2],)}


def test_parameterized_gates_must_be_resolved_when_their_name_depends_on_it() -> None:
    import sympy

    t = sympy.Symbol("t")
    q = cirq.LineQubit.range(2)
    model = to_cirq(migrated(MANILA_V01), unknown_gates="error")
    for op in ((cirq.X**t)(q[0]), (cirq.CZ**t)(*q), cirq.ms(t)(*q), cirq.wait(q[0], nanos=t)):
        with pytest.raises(ValueError, match="Resolve the parameters before you add noise"):
            cirq.Circuit(op).with_noise(model)
    # the simulator resolves first, so X**t at t=1 is the calibrated x
    circuit = cirq.Circuit((cirq.X**t)(q[0]), cirq.measure(q[0], key="m"))
    result = cirq.DensityMatrixSimulator(noise=model, seed=1).run(circuit, {"t": 1}, 200)
    assert result.measurements["m"].mean() > 0.9
    # a name that does not depend on the angle converts before resolution
    assert len(_ops(model.noisy_operation(cirq.rz(t).on(q[0])))) == 1


def test_ecr_gate_has_the_registry_unitary_and_value_equality() -> None:
    a, b = cirq.LineQubit.range(2)
    assert np.allclose(cirq.unitary(ECRGate().on(a, b)), gates.GATES["ecr"].unitary(), atol=0)
    assert ECRGate() == ECRGate() and len({ECRGate(), ECRGate()}) == 1
    assert gate_name(ECRGate()) == "ecr"


def test_check_runs_every_circuit_on_an_eagle_device() -> None:
    (result,) = nv.load("ibm_brisbane").check(frameworks=["cirq"]).frameworks
    assert result.passed and not result.not_run
    assert "ecr" in {g for c in result.circuits for g in c.gates}
