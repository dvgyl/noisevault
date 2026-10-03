import copy
import json
import pickle
import re
import time
import warnings
from itertools import permutations, product

import numpy as np
import pytest
from conftest import MANILA_V01, migrated, require, toy

stim = require("stim")

from noisevault import gates, metrics  # noqa: E402
from noisevault.channels import (  # noqa: E402
    ChannelSpec,
    gate_channels,
    pauli_kraus,
    pauli_twirl,
    superoperator,
    thermal_relaxation_kraus,
)
from noisevault.conversion import resolve_op  # noqa: E402
from noisevault.errors import (  # noqa: E402
    DisabledGateError,
    LayoutError,
    MissingCalibrationError,
    NoiseApproximationWarning,
    NoiseVaultError,
    UnsupportedEffect,
)
from noisevault.frameworks.stim import (  # noqa: E402
    ExistingNoiseError,
    NoiseVaultStimCircuit,
    layout_from_coords,
    sample_with_readout,
    to_stim,
)
from noisevault.profile import Profile  # noqa: E402
from noisevault.reference import _apply  # noqa: E402
from noisevault.report import Report  # noqa: E402

_STIM_TO_ROW = {name: row for row in gates.GATES.values() for name in row.stim}
_PAULI = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1, -1]),
}
_GHZ5 = [("H", (0,)), ("CX", (0, 1)), ("CX", (1, 2)), ("CX", (2, 3)), ("CX", (3, 4))]
_LAYER = [("SQRT_X", (0,)), ("CX", (0, 1)), ("CX", (2, 3)), ("S", (1,)), ("H", (4,))]
_LAYER += [("CX", (1, 2)), ("CX", (3, 4)), ("X", (2,))]
_INVERSE = {"SQRT_X": "SQRT_X_DAG", "S": "S_DAG"}
_MIRROR = _LAYER + [(_INVERSE.get(name, name), q) for name, q in reversed(_LAYER)]


@pytest.fixture(autouse=True)
def _quiet_typical_noise():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoiseApproximationWarning)
        yield


def _manila() -> Profile:
    return migrated(MANILA_V01)


def _readout_profile(p1_given_0: float = 0.02, p0_given_1: float = 0.1, **extra) -> Profile:
    data = toy(readout={"p1_given_0": p1_given_0, "p0_given_1": p0_given_1}, **extra)
    data["gates"] = {"rz": {"virtual": True}, "x": {"virtual": True}, "cz": {"virtual": True}}
    return Profile.model_validate(data)


def _grid(n: int, *, worse_rows: int = 0) -> Profile:
    """An n x n square-grid device with (row, col) coords; the first rows can be noisier."""
    edges = [(r * n + c, r * n + c + 1) for r in range(n) for c in range(n - 1)]
    edges += [(r * n + c, (r + 1) * n + c) for r in range(n - 1) for c in range(n)]
    base = Profile.uniform(
        f"grid{n}",
        technology="superconducting",
        num_qubits=n * n,
        one_qubit_error=1e-3,
        two_qubit_error=4e-3,
        readout_error=1e-2,
        t1_us=100,
        t2_us=80,
        one_qubit_ns=25,
        two_qubit_ns=40,
        connectivity=edges,
    )
    data = base.to_dict()
    data["qubits"] = [{"index": r * n + c, "coords": [r, c]} for r in range(n) for c in range(n)]
    bad = [e for e in edges if e[0] < worse_rows * n]
    data["calibrations"] = [{"gate": "cz", "qubits": list(e), "avg_infidelity": 0.05} for e in bad]
    return Profile.from_dict(data)


def _all_to_all(n: int, gate: str, spec: dict) -> Profile:
    device = {"name": "toy", "vendor": "test", "technology": "superconducting", "num_qubits": n}
    defs = {"rz": {"virtual": True}, gate: spec}
    return Profile.model_validate(toy(device=device, connectivity="all_to_all", gates=defs))


def _program(ops, n: int) -> stim.Circuit:
    lines = [f"{name} {' '.join(map(str, q))}" for name, q in ops]
    return stim.Circuit("\n".join([*lines, f"M {' '.join(map(str, range(n)))}"]))


def _twirled_reference(profile, ops, n, layout, *, asymmetric: bool) -> np.ndarray:
    """Density-matrix probabilities with each gate's resolve_op channel Pauli-twirled."""
    report = Report.start(profile, "test", None)
    rho = np.zeros((2,) * (2 * n), dtype=complex)
    rho[(0,) * (2 * n)] = 1.0
    for name, qubits in ops:
        row = _STIM_TO_ROW[name]
        rho = _apply(rho, [row.unitary()], qubits, n)
        wires = tuple(layout[q] for q in qubits)
        built = resolve_op(profile.table, row.name, wires, unknown_gates="typical", report=report)
        if built.channels:
            rho = _apply(rho, pauli_kraus(pauli_twirl(built.channels, wires)), qubits, n)
    probs = np.real(np.diagonal(rho.reshape(2**n, 2**n))).reshape((2,) * n)
    for c in range(n):
        a, b = profile.table.qubit(layout[c]).readout
        s = (a + b) / 2
        matrix = [[1 - a, b], [a, 1 - b]] if asymmetric else [[1 - s, s], [s, 1 - s]]
        probs = np.moveaxis(np.tensordot(np.array(matrix), probs, axes=([1], [c])), 0, c)
    return probs.reshape(-1)


def _assert_within_5_sigma(bits: np.ndarray, expected: np.ndarray) -> float:
    shots, n = bits.shape
    index = bits.astype(int) @ (1 << np.arange(n)[::-1])
    observed = np.bincount(index, minlength=2**n) / shots
    sigma = np.sqrt(expected * (1 - expected) / shots)
    assert np.all(np.abs(observed - expected) <= 5 * sigma + 5 / shots)
    return 0.5 * float(np.abs(observed - expected).sum())


def _noise_after(out: stim.Circuit, gate: str) -> list[float]:
    items = list(out)
    at = next(i for i, inst in enumerate(items) if inst.name == gate)
    return items[at + 1].gate_args_copy()


# (a) twirled gate noise ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stim_gate", "canonical", "qubits", "layout"),
    [
        ("SQRT_X", "sx", (0,), {0: 3}),
        ("X", "x", (0,), {0: 1}),
        ("H", "h", (0,), {0: 2}),
        ("SQRT_X_DAG", "sxdg", (0,), {0: 0}),
        ("CX", "cx", (0, 1), {0: 2, 1: 1}),
        ("CX", "cx", (1, 0), {0: 3, 1: 4}),
        ("CZ", "cz", (0, 1), {0: 0, 1: 1}),
        ("ISWAP", "iswap", (0, 1), {0: 1, 1: 2}),
    ],
)
def test_gate_noise_is_the_twirl_of_the_shared_channel(stim_gate, canonical, qubits, layout):
    profile = _manila()
    out = to_stim(profile, f"{stim_gate} {' '.join(map(str, qubits))}", layout=layout)
    wires = tuple(layout[q] for q in qubits)
    report = Report.start(profile, "test", None)
    built = resolve_op(profile.table, canonical, wires, unknown_gates="typical", report=report)
    assert _noise_after(out, stim_gate) == pytest.approx(pauli_twirl(built.channels, wires))
    assert list(out)[1].name == f"PAULI_CHANNEL_{len(qubits)}"


def _direct_pauli_probabilities(channels, wires) -> np.ndarray:
    """Pauli error probabilities from the Pauli transfer matrix diagonal (Walsh transform)."""
    labels = ["".join(p) for p in product("IXYZ", repeat=len(wires))]
    d = 2 ** len(wires)
    s = superoperator(channels, wires)

    def op(label):
        m = np.eye(1)
        for c in label:
            m = np.kron(m, _PAULI[c])
        return m

    def apply(m):
        return (s @ m.flatten(order="F")).reshape(d, d, order="F")

    fidelity = {p: np.real(np.trace(op(p).conj().T @ apply(op(p)))) / d for p in labels}

    def commute(p, q):
        return sum(a != "I" and b != "I" and a != b for a, b in zip(p, q, strict=True)) % 2 == 0

    probs = [sum(fidelity[q] * (1 if commute(p, q) else -1) for q in labels) / d**2 for p in labels]
    return np.array(probs[1:])


def test_pauli_channel_2_order_matches_a_direct_pauli_transfer_computation():
    fast, slow = {"index": 0, "t1_us": 2, "t2_us": 1}, {"index": 1, "t1_us": 900, "t2_us": 700}
    profile = Profile.model_validate(toy(qubits=[fast, slow]))
    for order in [(0, 1), (1, 0)]:
        out = to_stim(profile, f"CZ {order[0]} {order[1]}")
        report = Report.start(profile, "test", None)
        built = resolve_op(profile.table, "cz", order, unknown_gates="typical", report=report)
        emitted = np.array(_noise_after(out, "CZ"))
        direct = _direct_pauli_probabilities(built.channels, order)
        assert emitted == pytest.approx(direct, abs=1e-9)
        # the short-T1 qubit carries the X errors: XI when it is the first target, IX otherwise
        xi, ix = emitted[3], emitted[0]
        assert (xi > 10 * ix) if order == (0, 1) else (ix > 10 * xi)


def test_stim_reads_pauli_channel_2_first_letter_on_the_first_target():
    args = [0.0] * 15
    args[3] = 1.0  # XI
    bits = stim.Circuit(f"PAULI_CHANNEL_2({','.join(map(str, args))}) 0 1\nM 0 1")
    assert bits.compile_sampler().sample(4).tolist() == [[True, False]] * 4


_RZZ = {"avg_infidelity": 0.01}
_ZZ = {"avg_infidelity": 0.05}
_MS = {"avg_infidelity": 0.03}
_RXX = {"avg_infidelity": 0.02}
_HALF = (np.pi / 2,)


@pytest.mark.parametrize(
    ("defined", "stim_gate", "canonical", "params"),
    [
        ({"rzz": _RZZ}, "SQRT_ZZ", "rzz", _HALF),
        ({"rzz": _RZZ}, "SQRT_ZZ_DAG", "rzz", (-np.pi / 2,)),
        ({"rzz": _RZZ, "zz": _ZZ}, "SQRT_ZZ", "zz", ()),
        ({"rzz": _RZZ, "zz": _ZZ}, "SQRT_ZZ_DAG", "rzz", (-np.pi / 2,)),
        ({"zz": _ZZ}, "SQRT_ZZ", "zz", ()),
        ({"ms": _MS}, "SQRT_XX", "ms", (0.0, 0.0)),
        ({"ms": _MS, "rxx": _RXX}, "SQRT_XX", "ms", (0.0, 0.0)),
        ({"ms": _MS, "rxx": _RXX}, "SQRT_XX_DAG", "ms", (np.pi, 0.0)),
        ({"rxx": _RXX}, "SQRT_XX", "rxx", _HALF),
        ({"ms": _MS}, "SQRT_YY", "ms", (np.pi / 2, np.pi / 2)),
        ({"ms": _MS}, "SQRT_YY_DAG", "ms", (3 * np.pi / 2, np.pi / 2)),
        ({"ryy": _RXX}, "SQRT_YY", "ryy", _HALF),
    ],
)
def test_fixed_angle_gates_take_the_calibrated_native_they_equal(
    defined, stim_gate, canonical, params
):
    if params is not None:
        u, v = gates.GATES[canonical].unitary(*params), stim.gate_data(stim_gate).unitary_matrix
        assert abs(np.vdot(u, v)) == pytest.approx(4, abs=1e-6)
    # cx is far worse than every native under test, so the typical-noise rule would show.
    natives = {"rz": {"virtual": True}, "cx": {"avg_infidelity": 0.2}, **defined}
    profile = Profile.model_validate(toy(gates=natives))
    out = to_stim(profile, f"{stim_gate} 0 1", unknown_gates="error")
    report = Report.start(profile, "test", None)
    built = resolve_op(profile.table, canonical, (0, 1), unknown_gates="error", report=report)
    assert _noise_after(out, stim_gate) == pytest.approx(pauli_twirl(built.channels, (0, 1)))
    assert not out.report.events.get("typical_noise_used")


def test_sqrt_zz_dag_is_no_zz_gate():
    natives = {"rz": {"virtual": True}, "zz": _ZZ}
    profile = Profile.model_validate(toy(gates=natives))
    with pytest.raises(MissingCalibrationError, match="rzz on qubits"):
        to_stim(profile, "SQRT_ZZ_DAG 0 1", unknown_gates="error")


def test_fixed_angle_gates_without_their_natives_are_named_after_the_rotation():
    out = to_stim(_manila(), "SQRT_XX 0 1\nSQRT_YY 0 1\nSQRT_ZZ_DAG 0 1")
    assert set(out.report.events["typical_noise_used"]) == {"rxx", "ryy", "rzz"}


def test_z_family_gates_are_free_when_rz_is_virtual():
    out = to_stim(_manila(), "S 0\nZ 1\nS_DAG 2\nM 0 1 2", readout="none")
    assert str(out) == "S 0\nZ 1\nS_DAG 2\nM 0 1 2"


def test_typical_noise_warnings_point_at_the_callers_line():
    with pytest.warns(NoiseApproximationWarning) as caught:
        to_stim(_manila(), "H 0\nCZ 0 1")
    assert [w.filename for w in caught] == [__file__] * 2


def test_default_warning_filter_shows_each_typical_noise_cause_once():
    profile = _manila()  # loading resets the warning registries, which would hide repeats
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        for _ in range(3):
            to_stim(profile, "H 0\nCZ 0 1\nH 0")
    assert sorted(str(w.message).split(" on ")[0] for w in caught) == ["cz", "h"]


def test_unknown_gates_get_typical_noise_or_raise():
    profile = _manila()
    with pytest.warns(NoiseApproximationWarning, match="c_xyz"):
        out = to_stim(profile, "C_XYZ 0")
    assert list(out)[1].name == "PAULI_CHANNEL_1"
    assert out.report.events["typical_noise_used"]["c_xyz"] == 1
    with pytest.raises(MissingCalibrationError, match="unknown_gates='typical'"):
        to_stim(profile, "C_XYZ 0", unknown_gates="error")


@pytest.mark.parametrize("tick_ns", [None, 50.0])
def test_report_events_count_every_gate_application(tick_ns):
    circuit = "H 0\nH 0\nH 0\nREPEAT 100 {\n  H 1\n  TICK\n  REPEAT 2 {\n    H 1 2\n  }\n}"
    out = to_stim(_manila(), circuit, tick_ns=tick_ns)
    assert out.report.events["typical_noise_used"]["h"] == 3 + 100 * (1 + 2 * 2)


@pytest.mark.parametrize("unknown_gates", ["typical", "error"])
@pytest.mark.parametrize("gate", ["CXSWAP", "SWAPCX", "CZSWAP", "SWAP"])
def test_gates_needing_two_or_more_entanglers_must_be_decomposed(gate, unknown_gates):
    with pytest.raises(MissingCalibrationError, match=f"^{gate} 0 1 .*decompose") as caught:
        to_stim(_manila(), f"{gate} 0 1", unknown_gates=unknown_gates)
    assert "unknown_gates" not in str(caught.value)
    assert caught.value.hint == "decompose it into the profile's native gates first"


@pytest.mark.parametrize("gate", ["CXSWAP", "SWAPCX", "CZSWAP"])
def test_a_two_entangler_gate_takes_the_profiles_own_calibration_of_it(gate):
    def export(spec: dict, unknown_gates: str = "typical") -> stim.Circuit:
        profile = _all_to_all(2, gate.lower(), spec)
        return to_stim(profile, f"{gate} 0 1", readout="none", unknown_gates=unknown_gates)

    calibrated = export({"avg_infidelity": 0.02})
    assert _noise_after(calibrated, gate) == pytest.approx(metrics.uniform_pauli(0.02, 2))
    assert "typical_noise_used" not in calibrated.report.events
    assert str(export({"virtual": True})) == f"{gate} 0 1"
    with pytest.raises(DisabledGateError, match=f"^{gate} 0 1 .*disabled"):
        export({"disabled": True})
    for unknown_gates in ("typical", "error"):
        with pytest.raises(MissingCalibrationError, match=f"^{gate} 0 1 .*decompose"):
            export({}, unknown_gates)


def test_two_qubit_identity_gets_no_noise():
    assert str(to_stim(_manila(), "II 0 1")) == "II 0 1"


def test_a_two_qubit_identity_the_profile_disables_is_refused():
    def export(spec: dict) -> stim.Circuit:
        profile = _all_to_all(3, "ii", {"qubits": 2, **spec})
        return to_stim(profile, "II 0 1", layout=[2, 0], readout="none")

    for spec in ({}, {"virtual": True}, {"avg_infidelity": 0.02}):
        assert str(export(spec)) == "II 0 1"
    pattern = r"^II 0 1 \(physical qubits 2-0\): ii on qubits 2-0 is disabled in this profile"
    with pytest.raises(DisabledGateError, match=pattern) as caught:
        export({"disabled": True})
    assert caught.value.hint is None


_USABLE_PAIRS = (
    "pass layout= to put 2-qubit gates on pairs with a usable 2-qubit gate"
    " (profile.suggest_layout(2) proposes a chain of such pairs)"
)


def test_the_layout_hint_names_what_the_pair_lacks():
    data = toy(calibrations=[{"gate": "cz", "qubits": [0, 1], "disabled": True}])
    data["gates"]["cx"] = {"disabled": True}
    profile = Profile.model_validate(data)

    def hint(error: type[Exception], gate: str, layout: list[int]) -> str | None:
        where = rf"^{gate} 0 1 \(physical qubits {layout[0]}-{layout[1]}\)"
        with pytest.raises(error, match=where) as caught:
            to_stim(profile, f"{gate} 0 1", layout=layout)
        return caught.value.hint

    assert hint(DisabledGateError, "CZ", [0, 1]) == _USABLE_PAIRS
    assert hint(DisabledGateError, "CX", [1, 2]) is None
    assert hint(MissingCalibrationError, "CZ", [0, 2]) == _connected_pairs(2)
    to_stim(profile, "CZ 0 1", layout=profile.suggest_layout(2))


def _only_cz() -> Profile:
    data = toy()
    data["gates"] = {"h": {"virtual": True}, "cx": {"qubits": 2, "disabled": True}}
    data["gates"]["cz"] = {"qubits": 2, "avg_infidelity": 0.02}
    return Profile.model_validate(data)


@pytest.mark.parametrize("unknown_gates", ["typical", "error"])
def test_no_layout_hint_when_no_pair_runs_the_failing_gate(unknown_gates):
    profile = _only_cz()
    for layout in permutations(range(3), 2):
        with pytest.raises(NoiseVaultError) as caught:
            to_stim(profile, "CX 0 1", layout=list(layout), unknown_gates=unknown_gates)
        assert caught.value.hint is None
    with pytest.raises(DisabledGateError):
        to_stim(profile, "CX 0 1", layout=profile.suggest_layout(2), unknown_gates=unknown_gates)


@pytest.mark.parametrize("unknown_gates", ["typical", "error"])
def test_the_layout_hint_names_a_pair_where_the_failing_gate_runs(unknown_gates):
    enabled = {"gate": "cx", "qubits": [1, 2], "avg_infidelity": 0.01, "disabled": False}
    data = toy(calibrations=[enabled])
    data["gates"]["cx"] = {"qubits": 2, "disabled": True}
    profile = Profile.model_validate(data)
    for layout in ([0, 1], [0, 2], [2, 1]):
        with pytest.raises(NoiseVaultError) as caught:
            to_stim(profile, "CX 0 1", layout=layout, unknown_gates=unknown_gates)
        assert caught.value.hint == "pass layout= to put cx on qubits 1-2, where it runs"
    to_stim(profile, "CX 0 1", layout=[1, 2], unknown_gates=unknown_gates)


def test_the_layout_hint_names_a_direction_where_a_one_way_gate_runs():
    enabled = {"gate": "cx", "avg_infidelity": 0.01, "disabled": False}
    data = toy(calibrations=[{**enabled, "qubits": pair} for pair in ([0, 1], [1, 2])])
    data["gates"] = {"h": {"virtual": True}, "cx": {"qubits": 2, "disabled": True}}
    profile = Profile.model_validate(data)
    with pytest.raises(MissingCalibrationError) as caught:
        to_stim(profile, "CX 1 0", layout=[2, 0])
    assert caught.value.hint == "pass layout= to put cx on qubits 0-1, where it runs"
    to_stim(profile, "CX 1 0", layout=[1, 0])
    with pytest.raises(DisabledGateError):
        to_stim(profile, "CX 1 0", layout=profile.suggest_layout(2))


@pytest.mark.parametrize("unknown_gates", ["typical", "error"])
def test_no_layout_hint_when_no_pair_has_a_usable_2_qubit_gate(unknown_gates):
    data = toy()
    data["gates"] = {"h": {"virtual": True}, "cx": {"avg_infidelity": 0.01, "disabled": True}}
    profile = Profile.model_validate(data)
    with pytest.raises(LayoutError, match="no connected chain of 2 usable qubits"):
        profile.suggest_layout(2)
    for layout in ([0, 2], [0, 1]):
        where = rf"^CX 0 1 \(physical qubits {layout[0]}-{layout[1]}\)"
        with pytest.raises(NoiseVaultError, match=where) as caught:
            to_stim(profile, "CX 0 1", layout=layout, unknown_gates=unknown_gates)
        assert "layout=" not in (caught.value.hint or "")


def test_no_layout_hint_when_no_usable_pair_can_measure():
    gates = {**toy()["gates"], "measure": {}}
    off = [{"gate": "measure", "qubits": [q], "disabled": True} for q in (1, 2)]
    disabled_cz = {"gate": "cz", "qubits": [0, 1], "disabled": True}
    profile = Profile.model_validate(toy(gates=gates, calibrations=[*off, disabled_cz]))
    with pytest.raises(LayoutError, match="can measure"):
        profile.suggest_layout(2)
    with pytest.raises(DisabledGateError) as caught:
        to_stim(profile, "CZ 0 1")
    assert caught.value.hint is None
    with pytest.raises(MissingCalibrationError) as caught:
        to_stim(profile, "CZ 0 2")
    assert caught.value.hint is None


_CONTROLLED_PAULIS = [
    ("MPAD 1\nCX rec[-1] 0", "CX rec[-1] 0", "x"),
    ("MPAD 1\nCY rec[-1] 0", "CY rec[-1] 0", "y"),
    ("MPAD 1\nCZ rec[-1] 0", "CZ rec[-1] 0", "z"),
    ("MPAD 1\nCZ 0 rec[-1]", "CZ 0 rec[-1]", "z"),
    ("MPAD 1\nXCZ 0 rec[-1]", "XCZ 0 rec[-1]", "x"),
    ("MPAD 1\nYCZ 0 rec[-1]", "YCZ 0 rec[-1]", "y"),
    ("CX sweep[0] 0", "CX sweep[0] 0", "x"),
    ("MPAD 1\nREPEAT 2 {\n    CY rec[-1] 0\n}", "CY rec[-1] 0", "y"),
]


@pytest.mark.parametrize("where", ["record", "definition"])
@pytest.mark.parametrize(("circuit", "instruction", "pauli"), _CONTROLLED_PAULIS)
def test_a_classically_controlled_pauli_the_profile_disables_is_refused(
    circuit, instruction, pauli, where
):
    pattern = (
        rf"^{re.escape(instruction)} \(physical qubit 2\): {pauli} on qubit 2 is disabled in this"
        " profile$"
    )
    with pytest.raises(DisabledGateError, match=pattern):
        to_stim(_disabling(pauli, where), circuit, layout=[2], readout="none")
    for other in {"x", "y", "z"} - {pauli}:
        out = to_stim(_disabling(other, where), circuit, layout=[2], readout="none")
        assert str(out) == circuit


_CONNECTED = "pass layout= to put 2-qubit gates on connected pairs"
_FROM_COORDS = "noisevault.stim.layout_from_coords matches the circuit's QUBIT_COORDS"


def _connected_pairs(n: int) -> str:
    return f"{_CONNECTED} (profile.suggest_layout({n}) proposes a layout)"


_NATIVE_OR_TYPICAL = (
    "compile to the profile's native gates, or pass unknown_gates='typical' to use the typical"
    " native gate's noise"
)


def test_the_pair_hints_name_only_layout_tools_that_run():
    star = Profile.model_validate(
        toy(
            device={**toy()["device"], "num_qubits": 4},
            connectivity={"edges": [[0, 1], [0, 2], [0, 3]]},
        )
    )
    with pytest.raises(LayoutError, match="no connected chain of 4"):
        star.suggest_layout(4)
    with pytest.raises(MissingCalibrationError) as caught:
        to_stim(star, "CZ 1 2\nSQRT_X 0 3\nM 0 1 2 3")
    assert caught.value.hint == _CONNECTED
    off = {"gate": "cz", "qubits": [0, 1], "disabled": True}
    star_off = Profile.model_validate({**star.to_dict(), "calibrations": [off]})
    with pytest.raises(DisabledGateError) as caught:
        to_stim(star_off, "CZ 0 1\nSQRT_X 2 3\nM 0 1 2 3")
    assert (
        caught.value.hint == "pass layout= to put 2-qubit gates on pairs with a usable 2-qubit gate"
    )

    coords = [{"index": q, "coords": [0, q]} for q in range(3)]
    line = Profile.model_validate(toy(qubits=coords))
    circuit = "QUBIT_COORDS(0, 0) 0\nQUBIT_COORDS(0, 2) 1\nQUBIT_COORDS(0, 1) 2\nCZ 0 2\nM 0 1 2"
    with pytest.raises(MissingCalibrationError) as caught:
        to_stim(line, circuit)
    tools = f"profile.suggest_layout(3) proposes a layout, and {_FROM_COORDS}"
    assert caught.value.hint == f"{_CONNECTED} ({tools})"
    to_stim(line, circuit, layout=layout_from_coords(circuit, line))


def test_unusable_gate_errors_name_the_stim_instruction_and_the_fix():
    circuit = "H 0\nCX 0 1\nCX 1 2\nM 0 1 2"
    cx_1_2 = r"^CX 1 2 \(physical qubits 1-4\)"
    with pytest.raises(MissingCalibrationError, match=cx_1_2) as caught:
        to_stim(_manila(), circuit, layout={0: 0, 1: 1, 2: 4})
    assert caught.value.hint == _connected_pairs(3)
    with pytest.raises(MissingCalibrationError, match=cx_1_2) as caught:
        to_stim(_manila(), "CX 1 2", layout={1: 1, 2: 4}, unknown_gates="error")
    assert caught.value.hint == _connected_pairs(2)
    to_stim(_manila(), circuit, layout=_manila().suggest_layout(3))
    with pytest.raises(MissingCalibrationError, match="^SQRT_Y 0 .*unknown_gates") as caught:
        to_stim(_manila(), "SQRT_Y 0", unknown_gates="error")
    assert "layout=" not in str(caught.value)
    assert caught.value.hint == _NATIVE_OR_TYPICAL
    to_stim(_manila(), "SQRT_Y 0", unknown_gates="typical")


def test_repeated_targets_in_one_instruction_keep_gate_then_noise_order():
    out = to_stim(_manila(), "SQRT_X 0 0")
    assert [inst.name for inst in out] == ["SQRT_X", "PAULI_CHANNEL_1"] * 2


def test_measurement_feedback_gets_no_gate_noise():
    out = to_stim(_manila(), "M 0\nCX rec[-1] 1", readout="none")
    assert [inst.name for inst in out] == ["M", "CX"]
    assert any(a.what == "classically controlled Paulis" for a in out.report.approximated)


# (b) sampling matches the twirled density-matrix reference ------------------------------------


@pytest.mark.parametrize(
    ("ops", "layout"),
    [(_GHZ5, [0, 1, 2, 3, 4]), (_MIRROR, [4, 3, 2, 1, 0])],
    ids=["ghz5", "mirror"],
)
def test_sampling_matches_twirled_reference_on_manila(ops, layout):
    profile = _manila()
    out = to_stim(profile, _program(ops, 5), layout=layout)
    expected = _twirled_reference(profile, ops, 5, layout, asymmetric=False)
    bits = out.compile_sampler(seed=7).sample(200_000)
    _assert_within_5_sigma(bits, expected)


# (c) REPEAT blocks ---------------------------------------------------------------------------


def test_repeat_blocks_are_kept_with_the_noise_of_their_unrolled_body():
    circuit = stim.Circuit("""
        R 0 1
        REPEAT 3 {
            H 0
            CX 0 1
            REPEAT 2 {
                SQRT_X 1
            }
            MR 1
            DETECTOR(1, 0) rec[-1]
            TICK
        }
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    for tick_ns in (None, 80.0):
        out = to_stim(_manila(), circuit, tick_ns=tick_ns)
        blocks = [item for item in out if isinstance(item, stim.CircuitRepeatBlock)]
        assert [b.repeat_count for b in blocks] == [3]
        assert "PAULI_CHANNEL_2" in str(blocks[0].body_copy())
        assert out.flattened() == to_stim(_manila(), circuit.flattened(), tick_ns=tick_ns)
        assert (out.num_detectors, out.num_observables) == (3, 1)


def test_repeat_whose_first_pass_idles_differently_is_peeled_once():
    circuit = stim.Circuit("H 0\nREPEAT 3 {\n  TICK\n  H 1\n}\nTICK")
    out = to_stim(_manila(), circuit, tick_ns=100.0)
    assert [b.repeat_count for b in out if isinstance(b, stim.CircuitRepeatBlock)] == [2]
    assert out.flattened() == to_stim(_manila(), circuit.flattened(), tick_ns=100.0)


def _idle_twirl(profile: Profile, physical: int, tick_ns: float) -> list[float]:
    q = profile.table.qubit(physical)
    kraus = thermal_relaxation_kraus(q.t1_ns, q.t2_ns, tick_ns)
    return pauli_twirl([ChannelSpec("thermal_relaxation", (physical,), tuple(kraus))], (physical,))


@pytest.mark.parametrize("layout", [None, [2, 3, 4]], ids=["identity", "shifted"])
def test_idle_noise_at_tick_goes_to_qubits_left_idle_in_that_layer(layout):
    profile, circuit = _manila(), "H 0 1\nTICK\nCX 0 1\nTICK\nH 2\nTICK\nH 0"
    physical = layout or [0, 1, 2]
    plain = str(to_stim(profile, circuit, layout=layout)).split("TICK")
    timed = to_stim(profile, circuit, layout=layout, tick_ns=200.0)
    idle = []
    for with_idle, without in zip(str(timed).split("TICK"), plain, strict=True):
        extra = [line for line in with_idle.splitlines() if line not in without.splitlines()]
        idle.append(sorted(int(q) for line in extra for q in line.rsplit(")", 1)[1].split()))
    assert idle == [[2], [2], [0, 1], []]  # nothing after the last TICK
    names = [inst.name for inst in timed]
    emitted = list(timed)[names.index("TICK") - 1].gate_args_copy()
    assert emitted == pytest.approx(_idle_twirl(profile, physical[2], 200.0))
    assert emitted != pytest.approx(_idle_twirl(profile, 2 if layout else 4, 200.0))
    assert "idle noise" in [a.what for a in timed.report.approximated]
    assert any("idle noise (pass tick_ns=" in o for o in to_stim(profile, circuit).report.omitted)


def test_idle_noise_reports_qubits_without_relaxation_data_as_unknown():
    device = {"name": "toy", "vendor": "test", "technology": "superconducting", "num_qubits": 4}
    coherence = [{"t1_us": 50.0}, {"t2_us": 40.0}, {"dephasing_rate_per_s": 300.0}]
    qubits = [{"index": q, **c} for q, c in enumerate(coherence)]
    profile = Profile.model_validate(toy(device=device, qubits=qubits))
    out = to_stim(profile, "TICK\nM 0 1 2 3", readout="none", tick_ns=40.0)
    assert out.report.unknown == ["T1/T2 for idle noise of physical qubit 3"]
    idle = [t.value for inst in out if inst.name == "PAULI_CHANNEL_1" for t in inst.targets_copy()]
    assert sorted(idle) == [0, 1, 2]


def test_measured_and_reset_qubits_are_busy_in_their_tick_layer():
    out = to_stim(_manila(), "M 0\nR 1\nMR 2\nTICK\nH 0 1 2 3", readout="none", tick_ns=200.0)
    before_tick = str(out).split("TICK")[0].splitlines()
    assert before_tick[:3] == ["M 0", "R 1", "MR 2"]
    assert [line.rsplit(" ", 1)[1] for line in before_tick[3:]] == ["3"]


# (d) existing noise --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "noisy", ["DEPOLARIZE1(0.01) 0\nM 0", "H 0\nM(0.01) 0", "MPAD(0.3) 0\nM 0"]
)
def test_existing_noise_raises_with_the_fix(noisy):
    with pytest.raises(ExistingNoiseError, match="^the circuit already has noise") as caught:
        to_stim(_readout_profile(), noisy)
    assert caught.value.hint == (
        "pass existing_noise='strip' to replace it with the profile's noise, or"
        " existing_noise='keep' to add to it"
    )


def test_existing_noise_keep_and_strip():
    profile = _readout_profile()
    kept = to_stim(profile, "DEPOLARIZE1(0.01) 0\nM(0.01) 0", existing_noise="keep")
    assert [inst.name for inst in kept] == ["DEPOLARIZE1", "M"]
    assert list(kept)[1].gate_args_copy() == pytest.approx([0.01 + 0.06 - 2 * 0.01 * 0.06])
    stripped = to_stim(profile, "DEPOLARIZE1(0.01) 0\nM(0.01) 0", existing_noise="strip")
    assert str(stripped) == "M(0.06) 0"
    assert "noise already in the circuit (stripped)" in stripped.report.omitted


def test_kept_noise_targets_are_laid_out_and_idle_like_other_qubits():
    profile = _manila()
    with pytest.raises(LayoutError, match="qubit 7"):
        to_stim(profile, "X_ERROR(0.1) 7\nM 0", existing_noise="keep")
    kept = to_stim(
        profile, "E(0.1) X2\nTICK\nM 0", existing_noise="keep", readout="none", tick_ns=200.0
    )
    assert kept.layout == {0: 0, 2: 2}
    idle = str(kept).split("TICK")[0].splitlines()[1:]
    assert sorted(int(q) for line in idle for q in line.rsplit(")", 1)[1].split()) == [0, 2]
    stripped = to_stim(profile, "X_ERROR(0.1) 7\nM 0", existing_noise="strip")
    assert stripped.layout == {0: 0}


def test_noisy_padding_is_kept_or_stripped_like_other_noise():
    profile = _readout_profile()
    assert str(to_stim(profile, "MPAD(0.3) 0 1\nM 0", existing_noise="keep")).startswith(
        "MPAD(0.3) 0 1\n"
    )
    assert str(to_stim(profile, "MPAD(0.3) 0 1\nM 0", existing_noise="strip")) == (
        "MPAD 0 1\nM(0.06) 0"
    )
    assert str(to_stim(profile, "MPAD 1\nM 0")) == "MPAD 1\nM(0.06) 0"


def test_heralded_noise_cannot_be_stripped_but_can_be_kept():
    circuit = "HERALDED_ERASE(0.1) 0\nM 0"
    with pytest.raises(ExistingNoiseError, match="measurement records") as caught:
        to_stim(_readout_profile(), circuit, existing_noise="strip")
    assert caught.value.hint == "remove it from the circuit or pass existing_noise='keep'"
    out = to_stim(_readout_profile(), circuit, existing_noise="keep", readout="exact")
    assert out.readout_flips.tolist() == [[0.0, 0.0], [0.02, 0.1]]


# (e) readout and reset -----------------------------------------------------------------------


def test_symmetrized_readout_and_reset_errors_in_each_basis():
    profile = _readout_profile(prep={"error": 0.003})
    out = to_stim(profile, "RX 0\nMX 0\nMR 1\nMPP X0*Z1")
    assert str(out).splitlines() == [
        "RX 0",
        "Z_ERROR(0.003) 0",
        "MX(0.06) 0",
        "MR(0.06) 1",
        "X_ERROR(0.003) 1",
        "MPP(0.1128) X0*Z1",  # (1 - 0.88**2) / 2
    ]
    assert "readout error" in [a.what for a in out.report.approximated]


@pytest.mark.parametrize(
    ("circuit", "flips"),
    [
        ("MPP Z0*Z1*Z0", {"MPP(0.06) Z0*Z1*Z0": 0.06}),
        ("MPP Y0*Z1*Y0*Z2*Z2", {"MPP(0.06) Y0*Z1*Y0*Z2*Z2": 0.06}),
        ("MPP Z0*Z0", {"MPP Z0*Z0": 0.0}),
        ("MPP !X0*X0", {"MPP !X0*X0": 1.0}),
        ("MPP X0*Z0*X0*Z0*Z1", {"MPP(0.06) X0*Z0*X0*Z0*Z1": 0.94}),
        ("MPP Z0*Z0 Z1*Z2", {"MPP Z0*Z0": 0.0, "MPP(0.1128) Z1*Z2": 0.1128}),
    ],
)
def test_a_pauli_product_reads_out_only_the_qubits_left_after_reducing_it(circuit, flips):
    out = to_stim(_readout_profile(), circuit)
    assert str(out).splitlines() == list(flips)
    shots = 100_000
    rate = out.compile_sampler(seed=7).sample(shots).mean(axis=0)
    want = np.array(list(flips.values()))
    assert np.all(np.abs(rate - want) <= 5 * np.sqrt(want * (1 - want) / shots) + 5 / shots)


def test_cancelled_pauli_factors_leave_their_qubit_idle():
    out = to_stim(_manila(), "MPP Z0*Z1*Z0\nSPP X2*X2\nTICK", readout="none", tick_ns=200.0)
    before_tick = str(out).split("TICK")[0].splitlines()
    assert before_tick[:2] == ["MPP Z0*Z1*Z0", "SPP X2*X2"]
    assert [line.rsplit(" ", 1)[1] for line in before_tick[2:]] == ["0", "2"]


def test_a_pauli_product_gate_is_noised_on_the_qubits_left_after_reducing_it():
    out = to_stim(_manila(), "SPP Z0*Z1*Z0\nSPP_DAG X2*X2")
    lines = [line.split("(")[0] for line in str(out).splitlines()]
    assert lines == ["SPP Z0*Z1*Z0", "PAULI_CHANNEL_1", "SPP_DAG X2*X2"]
    assert str(out).splitlines()[1].endswith(") 1")
    assert out.report.events["typical_noise_used"] == {"spp": 1}
    placed = "QUBIT_COORDS(0, 0) 0\nQUBIT_COORDS(0, 1) 1\nSPP X0*X0\nCZ 0 1"
    assert layout_from_coords(placed, _grid(3)) == {0: 0, 1: 1}


def _paulis_seen_by_stim(noise: stim.Circuit, n: int) -> dict[str, float]:
    data, partners = " ".join(map(str, range(n))), " ".join(map(str, range(n, 2 * n)))
    pairs = " ".join(f"{q} {q + n}" for q in range(n))
    detectors = "\n".join(f"DETECTOR rec[{i - 2 * n}]" for i in range(2 * n))
    bell = stim.Circuit(f"H {data}\nCX {pairs}")
    probe = bell + noise + bell.inverse() + stim.Circuit(f"M {data} {partners}\n{detectors}")
    seen = {}
    for error in probe.detector_error_model(approximate_disjoint_errors=True):
        if error.type == "error":
            flipped = {t.val for t in error.targets_copy()}
            has_x = [q + n in flipped for q in range(n)]
            has_z = [q in flipped for q in range(n)]
            label = "".join("IXZY"[x + 2 * z] for x, z in zip(has_x, has_z, strict=True))
            seen[label] = error.args_copy()[0]
    return seen


_DISTINCT_3Q = tuple(1e-4 * (k + 1) for k in range(63))
_NO_IDENTITY_3Q = (1 / 63,) * 63


@pytest.mark.parametrize(
    ("n", "metric", "expected"),
    [
        (3, {"avg_infidelity": 0.02}, metrics.uniform_pauli(0.02, 3)),
        (3, {"pauli": _DISTINCT_3Q}, _DISTINCT_3Q),
        (3, {"pauli": _NO_IDENTITY_3Q}, _NO_IDENTITY_3Q),
        (4, {"avg_infidelity": 0.02}, metrics.uniform_pauli(0.02, 4)),
    ],
)
def test_a_calibrated_product_on_three_or_more_qubits_gets_its_whole_pauli_channel(
    n, metric, expected
):
    product = "*".join(f"Z{q}" for q in range(n))
    layout = [(q + 1) % n for q in range(n)]
    profile = _all_to_all(n, "spp", {"qubits": n, **metric})
    out = to_stim(profile, f"SPP {product}", layout=layout, readout="none")
    assert str(out[0]) == f"SPP {product}"
    want = dict(zip(metrics.pauli_labels(n), expected, strict=True))
    assert _paulis_seen_by_stim(out[1:], n) == pytest.approx(want, rel=1e-9)


@pytest.mark.parametrize(
    ("prepare", "measure"),
    [("R", "M"), ("RX", "MX"), ("RY", "MY"), ("MR", "M"), ("MRX", "MX"), ("MRY", "MY")],
)
def test_failed_reset_leaves_the_orthogonal_state_in_every_basis(prepare, measure):
    profile = _readout_profile(prep={"error": 0.2})
    out = to_stim(profile, f"{prepare} 0\n{measure} 0", readout="none")
    shots = 40_000
    ones = out.compile_sampler(seed=2).sample(shots)[:, -1].mean()
    assert abs(ones - 0.2) < 5 * np.sqrt(0.2 * 0.8 / shots)


@pytest.mark.parametrize(
    ("prepare", "measure"),
    [("R", "M"), ("RX", "MX"), ("RY", "MY"), ("MR", "M"), ("MRX", "MX"), ("MRY", "MY")],
)
def test_each_repeated_reset_gets_its_own_preparation_error(prepare, measure):
    profile = _readout_profile(prep={"error": 0.2})
    out = to_stim(profile, f"{prepare} 0\n{prepare} 0\n{measure} 0", readout="none")
    shots = 40_000
    ones = out.compile_sampler(seed=4).sample(shots)[:, -1].mean()
    assert abs(ones - 0.2) < 5 * np.sqrt(0.2 * 0.8 / shots)


def test_a_measure_reset_reads_the_preparation_error_of_the_reset_before_it():
    profile = _readout_profile(prep={"error": 0.003})
    out = to_stim(profile, "MR 0\nMR 0 1")
    assert str(out).splitlines() == [
        "MR(0.06) 0",
        "X_ERROR(0.003) 0",
        "MR(0.06) 0 1",
        "X_ERROR(0.003) 0 1",
    ]


def test_reset_error_is_the_physical_qubits():
    qubits = [{"index": i, "prep": {"error": e}} for i, e in enumerate([0.01, 0.02, 0.03])]
    profile = Profile.model_validate(toy(qubits=qubits))
    out = to_stim(profile, "R 0 1\nRX 2", layout={0: 2, 1: 0, 2: 1})
    assert str(out).splitlines() == [
        "R 0 1",
        "X_ERROR(0.03) 0",
        "X_ERROR(0.01) 1",
        "RX 2",
        "Z_ERROR(0.02) 2",
    ]


def _disabling(gate: str, where: str) -> Profile:
    data = toy(readout={"p1_given_0": 0.02, "p0_given_1": 0.1}, prep={"error": 0.003})
    data["gates"] = {"rz": {"virtual": True}, "x": {"virtual": True}, "cz": {"virtual": True}}
    if where == "record":
        data["gates"][gate] = {}
        data["calibrations"] = [{"gate": gate, "qubits": [2], "disabled": True}]
    else:
        data["gates"][gate] = {"disabled": True}
    return Profile.model_validate(data)


_RESETS = ["R", "RX", "RY", "MR", "MRX", "MRY"]
_MEASURES = ["M", "MX", "MY", "MR", "MRX", "MPP X1*Z0", "MZZ 1 0", "MXX 1 0", "MYY 1 0"]


@pytest.mark.parametrize("readout", ["symmetrize", "none"])
@pytest.mark.parametrize("where", ["record", "definition"])
@pytest.mark.parametrize(
    ("gate", "instruction"),
    [*(("reset", i) for i in _RESETS), *(("measure", i) for i in _MEASURES)],
)
def test_a_reset_or_measurement_the_profile_disables_is_refused(gate, instruction, where, readout):
    if " " not in instruction:
        instruction += " 1 0"
    layout = [2, 0]
    q = 0 if where == "record" else stim.Circuit(instruction)[0].targets_copy()[0].value
    name = instruction.split()[0]
    pattern = (
        rf"^{name} {q} \(physical qubit {layout[q]}\): {gate} on qubit {layout[q]} is disabled"
        " in this profile$"
    )
    with pytest.raises(DisabledGateError, match=pattern):
        to_stim(_disabling(gate, where), instruction, layout=layout, readout=readout)


@pytest.mark.parametrize("gate", ["reset", "measure"])
def test_a_reset_or_measurement_where_the_profile_allows_it_keeps_its_noise(gate):
    profile = _disabling(gate, "record")
    data = profile.model_dump(mode="json", exclude_none=True)
    del data["gates"][gate], data["calibrations"]
    circuit = "R 1\nRX 1\nM 1\nMR 1\nMRY 1\nMPP X1\nMZZ 1 2"
    allowed = Profile.model_validate(data)
    noisy, plain = (str(to_stim(p, circuit, layout=[2, 0, 1])) for p in (profile, allowed))
    assert noisy == plain
    assert "X_ERROR(0.003) 1" in noisy.splitlines()


def test_readout_none_adds_nothing_and_reports_it():
    out = to_stim(_readout_profile(), "X 0\nM 0 1", readout="none")
    assert str(out) == "X 0\nM 0 1"
    assert "readout error (readout='none')" in out.report.omitted


def test_unknown_readout_is_reported_not_zeroed_silently():
    out = to_stim(Profile.model_validate(toy()), "M 0 1")
    assert str(out) == "M 0 1"
    assert out.report.unknown == ["readout error of physical qubits 0 and 1"]


def test_unknown_readout_is_reported_for_exact_readout_too():
    out = to_stim(Profile.model_validate(toy()), "M 0 1", readout="exact")
    assert out.readout_flips.tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert out.report.unknown == ["readout error of physical qubits 0 and 1"]


def test_a_saved_report_names_every_qubit_without_readout():
    device = {"name": "eight", "vendor": "test", "technology": "superconducting"}
    data = toy(
        device={**device, "num_qubits": 8},
        connectivity={"edges": [[q, q + 1] for q in range(7)]},
        qubits=[
            {"index": q, "readout": {"p1_given_0": 0.01, "p0_given_1": 0.02}} for q in (3, 4, 5)
        ],
    )
    out = to_stim(Profile.model_validate(data), "M 0 1 2 3 4 5 6 7")
    again = pickle.loads(pickle.dumps(out))
    for report in (out.report, again.report):
        saved = json.loads(json.dumps(report.to_dict()))
        assert saved["unknown"] == ["readout error of physical qubits 0, 1, 2, 6 and 7"]
        assert (
            "unknown (no noise applied): readout error of physical qubits 0, 1, 2 and 2 more"
        ) in report.summary().splitlines()


def test_sample_with_readout_is_exact_for_asymmetric_readout():
    out = to_stim(_readout_profile(), "X 0\nM 0 !1 2", readout="exact")
    assert str(out) == "X 0\nM 0 !1 2"
    bits = sample_with_readout(out, 200_000, seed=3)
    zeros = 1 - bits.mean(axis=0)
    shots = len(bits)
    # prepared 1 reads 0 with P(0|1); an inverted prepared 0 records 0 with P(1|0)
    for observed, p in zip(zeros, [0.1, 0.02, 0.98], strict=True):
        assert abs(observed - p) < 5 * np.sqrt(p * (1 - p) / shots)
    again = sample_with_readout(out, 1000, seed=3)
    assert np.array_equal(again, sample_with_readout(out, 1000, seed=3))


def test_exact_readout_ghz_matches_asymmetric_reference_on_manila():
    profile = _manila()
    out = to_stim(profile, _program(_GHZ5, 5), readout="exact")
    expected = _twirled_reference(profile, _GHZ5, 5, [0, 1, 2, 3, 4], asymmetric=True)
    symmetric = _twirled_reference(profile, _GHZ5, 5, [0, 1, 2, 3, 4], asymmetric=False)
    assert 0.5 * np.abs(expected - symmetric).sum() > 0.01  # the test separates the two modes
    _assert_within_5_sigma(sample_with_readout(out, 200_000, seed=11), expected)


def _readout_on_qubit_1(p1_given_0: float, p0_given_1: float) -> Profile:
    qubit = {"index": 1, "readout": {"p1_given_0": p1_given_0, "p0_given_1": p0_given_1}}
    return _readout_profile(qubits=[qubit])


@pytest.mark.parametrize(
    ("circuit", "flips"),
    [
        ("MPP Z0*Z1*Z0", [[0.2, 0.3]]),
        ("MPP !Z0*Z1*Z0", [[0.3, 0.2]]),
        ("MPP X0*Z0*X0*Z0*Z1", [[0.3, 0.2]]),
        ("MPP Z0*Z0 !X0*X0", [[0.0, 0.0], [0.0, 0.0]]),
        ("MPP Z0*Z0 Z1 !X1*X2*X2", [[0.0, 0.0], [0.2, 0.3], [0.3, 0.2]]),
    ],
    ids=["Z1", "-Z1", "-I times Z1", "I and -I", "I, Z1 and -X1"],
)
def test_exact_readout_flips_each_reduced_product_by_its_qubit_and_sign(circuit, flips):
    out = to_stim(_readout_on_qubit_1(0.2, 0.3), circuit, readout="exact")
    assert str(out) == circuit
    assert out.readout_flips.tolist() == flips


def test_exact_readout_of_reduced_products_agrees_with_the_results_stim_records():
    p0_given_1 = 0.3
    out = to_stim(
        _readout_on_qubit_1(0.2, p0_given_1),
        "X 1\nMPP Z0*Z1*Z0 X0*Z0*X0*Z0*Z1 !Z0*Z0",
        readout="exact",
    )
    shots = 100_000
    ones = sample_with_readout(out, shots, seed=5).mean(axis=0)
    want = np.array([1 - p0_given_1, p0_given_1, 1.0])
    assert np.all(np.abs(ones - want) <= 5 * np.sqrt(want * (1 - want) / shots))


def test_an_anti_hermitian_product_is_left_for_stim_to_refuse():
    out = to_stim(_readout_profile(), "MPP X0*Z0", readout="exact")
    assert str(out) == "MPP X0*Z0"
    with pytest.raises(ValueError, match="anti-Hermitian"):
        sample_with_readout(out, 1)


@pytest.mark.parametrize(
    ("circuit", "reason"),
    [
        ("M 0\nCX rec[-1] 1\nM 1", "feeds a measurement"),
        ("MPP X0*X1", "multi-qubit product"),
        ("MPP Z0*Z1*Z1*Z2", "multi-qubit product"),
        ("MZZ 0 1", "multi-qubit product"),
    ],
)
def test_exact_readout_refuses_what_it_cannot_do_exactly(circuit, reason):
    with pytest.raises(ValueError, match=reason):
        to_stim(_manila(), circuit, readout="exact")


def test_sample_with_readout_needs_an_exact_export():
    with pytest.raises(ValueError, match="readout='exact'"):
        sample_with_readout(to_stim(_manila(), "M 0"), 10)


# result, options, layout ---------------------------------------------------------------------


def test_result_is_a_stim_circuit_with_report_and_profile():
    profile = _manila()
    out = profile.to_stim(stim.Circuit("H 0\nM 0"), layout={0: 3})
    assert isinstance(out, stim.Circuit) and isinstance(out, NoiseVaultStimCircuit)
    assert out.profile is profile and out.layout == {0: 3}
    assert out.report.framework == "stim" and out.report.options["layout"] == {0: 3}
    assert "Pauli twirl" in out.report.summary()


@pytest.mark.parametrize(
    "clone", [lambda c: pickle.loads(pickle.dumps(c)), copy.deepcopy, copy.copy]
)
def test_pickled_and_copied_results_keep_report_and_exact_readout(clone):
    out = to_stim(_readout_profile(), "X 0\nM 0 1", layout={0: 1, 1: 2}, readout="exact")
    again = clone(out)
    assert isinstance(again, NoiseVaultStimCircuit) and str(again) == str(out)
    assert again.report.summary() == out.report.summary()
    assert (again.profile, again.layout) == (out.profile, {0: 1, 1: 2})
    assert np.array_equal(again.readout_flips, out.readout_flips)
    assert np.array_equal(
        sample_with_readout(again, 50, seed=1), sample_with_readout(out, 50, seed=1)
    )


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"readout": "asym"}, "readout='asym'"),
        ({"existing_noise": "drop"}, "existing_noise"),
        ({"tick_ns": 0}, "tick_ns"),
        ({"unknown_gates": "ideal"}, "unknown_gates"),
    ],
)
def test_bad_options_say_what_to_pass(options, match):
    with pytest.raises(ValueError, match=match):
        to_stim(_manila(), "H 0", **options)


def test_layout_errors_name_the_qubit():
    with pytest.raises(LayoutError, match="qubit 7"):
        to_stim(_manila(), "H 7")


def test_effects_are_omitted_or_refused():
    effect = {"type": "leakage", "gate": "cz", "prob": 1e-4}
    omitted = to_stim(Profile.model_validate(toy(effects=[effect])), "CZ 0 1")
    assert "effect leakage on cz" in omitted.report.omitted
    refused = Profile.model_validate(toy(effects=[{**effect, "allow": "approximate"}]))
    with pytest.raises(UnsupportedEffect):
        to_stim(refused, "CZ 0 1")


def test_layout_from_coords_places_a_surface_code_on_the_best_grid_patch():
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=1)
    profile = _grid(9, worse_rows=3)
    layout = layout_from_coords(circuit, profile)
    edges = set(profile.table.edges())
    pairs = {(t[0].value, t[1].value) for i in circuit.flattened() if i.name == "CX"
             for t in i.target_groups()}  # fmt: skip
    assert all(tuple(sorted((layout[a], layout[b]))) in edges for a, b in pairs)
    assert min(layout.values()) >= 3 * 9  # avoids the noisy top rows
    assert len(set(layout.values())) == len(layout) == 17


def test_layout_from_coords_says_what_to_do_without_coords():
    small = "QUBIT_COORDS(0, 0) 0\nQUBIT_COORDS(0, 9) 1\nH 0\nCZ 0 1\nM 0 1"
    with pytest.raises(LayoutError, match="records no qubit coords") as caught:
        layout_from_coords(small, _manila())
    assert caught.value.hint == "pass layout={stim qubit: physical qubit}"
    to_stim(_manila(), small, layout={0: 0, 1: 1})
    with pytest.raises(LayoutError, match="no rotation or shift") as caught:
        layout_from_coords(small, _grid(4))
    assert caught.value.hint == "pass layout= explicitly"
    to_stim(_grid(4), small, layout={0: 0, 1: 1})
    with pytest.raises(
        LayoutError, match=r"^the circuit gives no 2D QUBIT_COORDS for qubits 0 and 1;"
    ) as caught:
        layout_from_coords("H 0 1", _grid(4))
    assert caught.value.hint == "add QUBIT_COORDS for each qubit or pass layout="
    to_stim(_grid(4), "H 0 1", layout={0: 0, 1: 1})


def test_layout_from_coords_gives_no_layout_hint_for_a_circuit_larger_than_the_device():
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=1)
    for profile, error in ((_manila(), "records no qubit coords"), (_grid(4), "no rotation")):
        with pytest.raises(LayoutError, match=error) as caught:
            layout_from_coords(circuit, profile)
        assert caught.value.hint is None
    with pytest.raises(LayoutError, match="no 2D QUBIT_COORDS") as caught:
        layout_from_coords("H " + " ".join(map(str, range(17))), _grid(4))
    assert caught.value.hint is None


def test_layout_from_coords_refuses_a_match_on_coords_of_two_enabled_qubits():
    five = {"name": "five", "vendor": "test", "technology": "superconducting", "num_qubits": 5}
    data = toy(device=five)
    coords = [[0, 0], [0, 1], [0, 2], [5, 5], [5, 5]]
    data["qubits"] = [{"index": i, "coords": c} for i, c in enumerate(coords)]
    profile = Profile.model_validate(data)
    with pytest.raises(LayoutError) as caught:
        layout_from_coords("QUBIT_COORDS(5, 5) 0\nM 0", profile)
    assert str(caught.value).startswith(
        "test_five has qubits 3 and 4 at coords (5, 5), so the circuit's QUBIT_COORDS have no"
        " single device qubit there"
    )
    assert caught.value.hint == "pass layout={stim qubit: physical qubit}"
    to_stim(profile, "QUBIT_COORDS(5, 5) 0\nM 0", layout={0: 3})
    line = "QUBIT_COORDS(0, 0) 0\nQUBIT_COORDS(0, 1) 1\nQUBIT_COORDS(0, 2) 2\nM 0 1 2"
    assert layout_from_coords(line, profile) == {0: 0, 1: 1, 2: 2}


def _line_of_two(gate: str, disabled_on: tuple[int, ...], where: str) -> Profile:
    two = {"name": "two", "vendor": "test", "technology": "superconducting", "num_qubits": 2}
    data = toy(device=two, connectivity={"edges": [[0, 1]]})
    data["gates"] = {"h": {"virtual": True}, "x": {"virtual": True}, "measure": {}, "reset": {}}
    by_record = where == "record"
    data["gates"][gate]["disabled"] = not by_record
    records = [q for q in (0, 1) if (q in disabled_on) == by_record]
    data["calibrations"] = [{"gate": gate, "qubits": [q], "disabled": by_record} for q in records]
    data["qubits"] = [
        {"index": 0, "coords": [0, 0], "readout": {"p1_given_0": 0.0, "p0_given_1": 0.0}},
        {"index": 1, "coords": [0, 1], "readout": {"p1_given_0": 0.02, "p0_given_1": 0.02}},
    ]
    return Profile.model_validate(data)


_ONE_QUBIT_USES = [
    ("measure", "QUBIT_COORDS(0, 0) 0\nH 0\nM 0"),
    ("measure", "QUBIT_COORDS(0, 0) 0\nMPP X0"),
    ("measure", "QUBIT_COORDS(0, 0) 0\nMR 0"),
    ("reset", "QUBIT_COORDS(0, 0) 0\nMR 0"),
    ("reset", "QUBIT_COORDS(0, 0) 0\nRX 0"),
    ("h", "QUBIT_COORDS(0, 0) 0\nH 0"),
    ("x", "QUBIT_COORDS(0, 0) 0\nCX sweep[0] 0"),
    ("x", "QUBIT_COORDS(0, 0) 0\nMPAD 1\nREPEAT 2 {\n    CX rec[-1] 0\n}"),
]


@pytest.mark.parametrize("where", ["record", "definition"])
@pytest.mark.parametrize(("gate", "circuit"), _ONE_QUBIT_USES)
def test_layout_from_coords_skips_a_qubit_that_disables_an_operation_the_circuit_uses(
    gate, circuit, where
):
    assert layout_from_coords(circuit, _line_of_two(gate, (), where)) == {0: 0}
    profile = _line_of_two(gate, (0,), where)
    layout = layout_from_coords(circuit, profile)
    assert layout == {0: 1}
    to_stim(profile, circuit, layout=layout)
    nowhere = _line_of_two(gate, (0, 1), where)
    with pytest.raises(LayoutError) as caught:
        layout_from_coords(circuit, nowhere)
    assert caught.value.message == (
        "no rotation or shift of the circuit's QUBIT_COORDS fits test_two's qubit coords, puts"
        " every 2-qubit gate on a connected pair and avoids disabled operations"
    )
    assert caught.value.hint is None
    for q in (0, 1):
        with pytest.raises(DisabledGateError):
            to_stim(nowhere, circuit, layout={0: q})
    without_coords = circuit.split("\n", 1)[1]
    with pytest.raises(LayoutError, match="no 2D QUBIT_COORDS") as caught:
        layout_from_coords(without_coords, nowhere)
    assert caught.value.hint is None


def test_layout_from_coords_skips_a_pair_that_disables_the_circuit_s_2_qubit_gate():
    four = {"name": "four", "vendor": "test", "technology": "superconducting", "num_qubits": 4}
    gates = {"cx": {"avg_infidelity": 0.001}, "cz": {"avg_infidelity": 0.002}}
    data = toy(device=four, gates=gates, connectivity={"edges": [[0, 1], [1, 2], [2, 3]]})
    data["calibrations"] = [{"gate": "cx", "qubits": [1, 2], "disabled": True}]
    data["qubits"] = [{"index": q, "coords": [0, q]} for q in range(4)]
    data["qubits"][3]["readout"] = {"p1_given_0": 0.02, "p0_given_1": 0.02}
    profile = Profile.model_validate(data)
    coords = "QUBIT_COORDS(0, 0) 0\nQUBIT_COORDS(0, 1) 1\nQUBIT_COORDS(0, 2) 2\n"
    fan_out = coords + "CX 1 0 1 2"
    layout = layout_from_coords(fan_out, profile)
    assert set(layout.values()) == {1, 2, 3}
    to_stim(profile, fan_out, layout=layout)
    assert set(layout_from_coords(coords + "CZ 1 0 1 2", profile).values()) == {0, 1, 2}


@pytest.mark.parametrize(
    "circuit",
    ["QUBIT_COORDS(0, 0) 1\nCX sweep[0] 1\nM 1", "QUBIT_COORDS(0, 0) 1\nM 1\nCZ rec[-1] 1\nM 1"],
    ids=["sweep", "feedback"],
)
def test_layout_from_coords_places_only_the_qubit_a_classical_control_acts_on(circuit):
    two = {"name": "two", "vendor": "test", "technology": "superconducting", "num_qubits": 2}
    data = toy(device=two, connectivity={"edges": [[0, 1]]})
    data["qubits"] = [{"index": 0, "coords": [0, 0]}, {"index": 1, "coords": [0, 1]}]
    profile = Profile.model_validate(data)
    layout = layout_from_coords(circuit, profile)
    assert list(layout) == [1]
    out = to_stim(profile, circuit, layout=layout, readout="none")
    assert [inst.name for inst in out] == [inst.name for inst in stim.Circuit(circuit)]


@pytest.mark.timing
def test_layout_from_coords_is_fast_for_a_distance_11_code_on_a_32x32_grid():
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=11, rounds=1)
    profile = _grid(32, worse_rows=5)
    start = time.perf_counter()
    layout = layout_from_coords(circuit, profile)
    elapsed = time.perf_counter() - start
    print(f"layout_from_coords d=11 on 32x32: {elapsed:.3f} s")
    assert len(set(layout.values())) == len(layout) == 241
    assert min(layout.values()) >= 5 * 32
    assert elapsed < 0.5


# (f) scale and (g) detector error model ------------------------------------------------------


@pytest.mark.timing
def test_1024_qubits_by_100_layers_converts_and_samples_in_under_5_seconds():
    n, layers = 1024, 100
    lines = [f"R {' '.join(map(str, range(n)))}", "TICK"]
    for layer in range(layers):
        lines += [f"H {' '.join(map(str, range(n)))}", "TICK"]
        pairs = " ".join(f"{i} {i + 1}" for i in range(layer % 2, n - 1, 2))
        lines += [f"CZ {pairs}", "TICK"]
    circuit = stim.Circuit("\n".join([*lines, f"M {' '.join(map(str, range(n)))}"]))
    profile = Profile.uniform(
        "chain1024",
        technology="superconducting",
        num_qubits=n,
        one_qubit_error=1e-3,
        two_qubit_error=5e-3,
        readout_error=1e-2,
        t1_us=100,
        t2_us=80,
        one_qubit_ns=25,
        two_qubit_ns=40,
        connectivity=[(i, i + 1) for i in range(n - 1)],
    )
    start = time.perf_counter()
    out = to_stim(profile, circuit, tick_ns=50.0)
    built = time.perf_counter()
    bits = out.compile_sampler(seed=1).sample(10_000)
    done = time.perf_counter()
    print(f"convert {built - start:.2f} s, sample 10k {done - built:.2f} s")
    assert bits.shape == (10_000, n)
    assert done - start < 5.0


def test_detector_error_model_builds_and_decodes_for_a_noisy_surface_code():
    pymatching = require("pymatching")
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3)
    profile = _grid(7)
    out = to_stim(profile, circuit, layout=layout_from_coords(circuit, profile), tick_ns=50.0)
    dem = out.detector_error_model(decompose_errors=True)
    assert dem.num_detectors == circuit.num_detectors and dem.num_errors > 100
    matching = pymatching.Matching.from_stim_circuit(out)
    detectors, observed = out.compile_detector_sampler(seed=5).sample(
        20_000, separate_observables=True
    )
    decoded = np.mean(matching.decode_batch(detectors)[:, 0] != observed[:, 0])
    raw = np.mean(observed[:, 0])
    assert 0 < decoded < raw / 3


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


def test_idle_noise_reports_the_t2_clamp_like_the_gate_path() -> None:
    gate_profile = _t2_above_2_t1(sx={"avg_infidelity": 1e-3, "duration_ns": 35})
    gate_report = Report.start(gate_profile, "test", None)
    table = gate_profile.table
    gate_report.record_channels(gate_channels(table.gate("sx", (0,)), [table.qubit(0)]))
    out = to_stim(_t2_above_2_t1(), "R 0 1\nTICK\nTICK\nM 0 1", tick_ns=1000.0)
    t2 = [a for a in out.report.approximated if a.what.startswith("T2")]
    assert t2 == gate_report.approximated
