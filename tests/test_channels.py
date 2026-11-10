from __future__ import annotations

import numpy as np
import pytest
from conftest import require, toy

from noisevault import metrics
from noisevault.channels import (
    gate_channels,
    pauli_twirl,
    readout_matrix,
    superoperator,
    thermal_relaxation_kraus,
)
from noisevault.errors import DisabledGateError
from noisevault.profile import Profile
from noisevault.report import Report


def _profile(**sections) -> Profile:
    base = dict(
        idle={"t1_us": 100, "t2_us": 80},
        gates={
            "rz": {"virtual": True},
            "sx": {"avg_infidelity": 3e-4, "duration_ns": 35},
            "x": {"avg_infidelity": 1e-5, "duration_ns": 35},
            "cz": {"avg_infidelity": 8e-3, "duration_ns": 300},
            "ecr": {"pauli": [2e-3] + [0.0] * 14, "duration_ns": 500},
            "reset": {"duration_ns": 1000},
        },
        connectivity="all_to_all",
    )
    base.update(sections)
    return Profile.model_validate(toy(**base))


def _built(profile: Profile, name: str, qubits: tuple[int, ...]):
    table = profile.table
    return gate_channels(table.gate(name, qubits), [table.qubit(q) for q in qubits])


def _avg_infidelity(superop: np.ndarray) -> float:
    d = int(round(np.sqrt(superop.shape[0])))
    f_pro = np.trace(superop).real / d**2
    return 1 - (d * f_pro + 1) / (d + 1)


def _is_trace_preserving(superop: np.ndarray) -> bool:
    d = int(round(np.sqrt(superop.shape[0])))
    vec_identity = np.eye(d).flatten(order="F")
    return np.allclose(vec_identity @ superop, vec_identity, atol=1e-12)


@pytest.mark.parametrize(
    ("name", "qubits"),
    [("sx", (0,)), ("x", (2,)), ("cz", (0, 2)), ("ecr", (1, 0)), ("reset", (1,))],
)
def test_channels_are_cptp(name: str, qubits: tuple[int, ...]) -> None:
    built = _built(_profile(), name, qubits)
    for channel in built.channels:
        d = channel.kraus[0].shape[0]
        assert np.allclose(sum(k.conj().T @ k for k in channel.kraus), np.eye(d), atol=1e-12)
    assert _is_trace_preserving(superoperator(built.channels, qubits))


@pytest.mark.parametrize(
    ("t1", "t2", "duration"),
    [
        (50.0, 70.0, 400.0),
        (120.0, 240.0, 60.0),
        (80.0, 20.0, 1000.0),
        (30.0, 100.0, 300.0),
        (60.0, None, 500.0),
    ],
)
def test_thermal_relaxation_matches_aer(t1: float, t2: float | None, duration: float) -> None:
    noise = require("qiskit_aer.noise")
    qi = require("qiskit.quantum_info")
    aer_t2 = 2 * t1 if t2 is None else min(t2, 2 * t1)  # Aer rejects T2 > 2 T1
    expected = qi.SuperOp(noise.thermal_relaxation_error(t1 * 1e3, aer_t2 * 1e3, duration)).data
    kraus = thermal_relaxation_kraus(t1 * 1e3, None if t2 is None else t2 * 1e3, duration)
    actual = sum(np.kron(k.conj(), k) for k in kraus)
    assert np.allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize(
    ("t1", "t2", "duration", "rate"),
    [(100.0, 80.0, 10_000.0, 1_000.0), (60.0, None, 500.0, 300.0), (None, None, 1e6, 0.3)],
)
def test_dephasing_rate_adds_z_errors_like_a_shorter_t2_in_aer(
    t1: float | None, t2: float | None, duration: float, rate: float
) -> None:
    noise = require("qiskit_aer.noise")
    qi = require("qiskit.quantum_info")
    base_t2 = (2 * t1 if t2 is None else t2) * 1e3 if t1 is not None else np.inf
    # Z errors at a rate scale coherence by exp(-2 rate t), as if 1/T2 grew by 2 rate.
    aer_t2 = 1 / (1 / base_t2 + 2 * rate * 1e-9)
    aer_t1 = np.inf if t1 is None else t1 * 1e3
    expected = qi.SuperOp(noise.thermal_relaxation_error(aer_t1, aer_t2, duration)).data
    kraus = thermal_relaxation_kraus(
        None if t1 is None else t1 * 1e3, None if t2 is None else t2 * 1e3, duration, rate
    )
    actual = sum(np.kron(k.conj(), k) for k in kraus)
    assert np.allclose(actual, expected, atol=1e-12)


def test_two_idle_times_with_a_dephasing_rate_compose_to_one_idle_time() -> None:
    def superop(duration: float) -> np.ndarray:
        kraus = thermal_relaxation_kraus(50e3, 40e3, duration, 1e5)
        return sum(np.kron(k.conj(), k) for k in kraus)

    assert np.allclose(superop(3000.0) @ superop(3000.0), superop(6000.0), atol=1e-12)
    # H, idle, H: P(1) = (1 - coherence) / 2. The old linear rule gave 0.5 for this idle time.
    only_rate = thermal_relaxation_kraus(None, None, 6000.0, 1e5)
    plus = np.full((2, 2), 0.5)
    coherence = 2 * sum(k @ plus @ k.conj().T for k in only_rate)[0, 1].real
    assert (1 - coherence) / 2 == pytest.approx(0.349403, abs=1e-6)


def test_gate_channels_include_the_dephasing_rate() -> None:
    profile = _profile(idle={"dephasing_rate_per_s": 50.0})
    built = _built(profile, "cz", (0, 1))
    p = -np.expm1(-2 * 50.0 * 300e-9) / 2  # Z error on each operand over the 300 ns gate
    process_fidelity = (1 - p) ** 2
    assert built.relaxation == pytest.approx(1 - (4 * process_fidelity + 1) / 5, abs=1e-15)
    measured = _avg_infidelity(superoperator(built.channels, (0, 1)))
    assert measured == pytest.approx(8e-3, abs=1e-12)  # still the stated total error


@pytest.mark.parametrize(("name", "qubits"), [("sx", (0,)), ("cz", (2, 0))])
def test_residual_depolarizing_reaches_the_stated_error(name: str, qubits) -> None:
    profile = _profile()
    built = _built(profile, name, qubits)
    stated = profile.gates[name].avg_infidelity
    assert [c.kind for c in built.channels][0] == "depolarizing"
    measured = _avg_infidelity(superoperator(built.channels, qubits))
    assert measured == pytest.approx(stated, abs=1e-12)
    assert built.achieved == pytest.approx(stated, abs=1e-12) and not built.inexact


def test_relaxation_floor_is_kept_and_recorded() -> None:
    profile = _profile()
    built = _built(profile, "x", (1,))
    measured = _avg_infidelity(superoperator(built.channels, (1,)))
    assert [c.kind for c in built.channels] == ["thermal_relaxation"]
    assert built.inexact and built.requested == 1e-5
    assert built.achieved == pytest.approx(measured, abs=1e-14)
    assert built.achieved == pytest.approx(built.relaxation, abs=1e-14) and built.achieved > 1e-5
    report = Report.start(profile, "test", None)
    report.record_channels(built)
    report.record_channels(built)
    assert len(report.clamped) == 1
    clamp = report.clamped[0]
    assert (clamp.gate, clamp.qubits, clamp.requested) == ("x", (1,), 1e-5)
    assert clamp.achieved == built.achieved


@pytest.mark.parametrize(("name", "qubits", "stated"), [("x", (0,), 0.6), ("cz", (0, 1), 0.79)])
def test_unreachable_stated_error_is_recorded_not_silently_lowered(name, qubits, stated) -> None:
    # 10 us of relaxation at T1 = 10 us leaves too little room for depolarizing noise to add
    profile = _profile(
        idle={"t1_us": 10, "t2_us": 20},
        gates={name: {"avg_infidelity": stated, "duration_ns": 10_000}},
    )
    built = _built(profile, name, qubits)
    measured = _avg_infidelity(superoperator(built.channels, qubits))
    assert built.achieved == pytest.approx(measured, abs=1e-12)
    assert built.relaxation < built.achieved < stated - 1e-3
    report = Report.start(profile, "test", None)
    report.record_channels(built)
    assert [(c.gate, c.qubits, c.requested) for c in report.clamped] == [(name, qubits, stated)]
    assert report.clamped[0].achieved == built.achieved
    assert "less noisy than stated" in report.summary()


def test_t2_above_twice_t1_is_clamped_and_recorded() -> None:
    profile = _profile(qubits=[{"index": 0, "t1_us": 30, "t2_us": 100}])
    built = _built(profile, "sx", (0,))
    assert built.t2_clamped == (0,)
    relaxation = next(c for c in built.channels if c.kind == "thermal_relaxation")
    clamped = thermal_relaxation_kraus(30e3, 60e3, 35)
    expected = sum(np.kron(k.conj(), k) for k in clamped)
    assert np.allclose(superoperator([relaxation], (0,)), expected)
    report = Report.start(profile, "test", None)
    report.record_channels(built)
    assert report.approximated[0].what == "T2 of qubit 0"


def test_pauli_spec_is_the_whole_channel() -> None:
    profile = _profile()
    built = _built(profile, "ecr", (1, 2))
    assert [c.kind for c in built.channels] == ["pauli"]
    superop = superoperator(built.channels, (1, 2))
    rho = np.zeros((4, 4))
    rho[0, 0] = 1
    out = (superop @ rho.flatten(order="F")).reshape(4, 4, order="F")
    # the only error is IX: X on the second operand (qubit 2), so |00> -> |01>
    assert out[1, 1].real == pytest.approx(2e-3) and out[2, 2].real == pytest.approx(0.0)
    assert built.achieved == pytest.approx(metrics.avg_from_pauli([2e-3] + [0.0] * 14))


def test_superoperator_respects_wire_order() -> None:
    built = _built(_profile(), "ecr", (1, 2))
    forward = superoperator(built.channels, (1, 2))
    backward = superoperator(built.channels, (2, 1))
    swap = np.eye(4)[[0, 2, 1, 3]]
    conj = np.kron(swap, swap)
    assert np.allclose(backward, conj @ forward @ conj)


def test_uncalibrated_gate_gets_relaxation_only() -> None:
    built = _built(_profile(), "reset", (0,))
    assert [c.kind for c in built.channels] == ["thermal_relaxation"]
    assert built.requested is None and not built.inexact


def test_ideal_and_disabled_gates() -> None:
    profile = _profile(calibrations=[{"gate": "cz", "qubits": [0, 1], "disabled": True}])
    assert _built(profile, "rz", (0,)).channels == ()
    with pytest.raises(DisabledGateError, match=r"^cz on qubits 0-1 is disabled in this profile$"):
        _built(profile, "cz", (0, 1))


def test_pauli_twirl() -> None:
    profile = _profile()
    pauli = _built(profile, "ecr", (0, 1))
    assert pauli_twirl(pauli.channels, (0, 1)) == pytest.approx([2e-3] + [0.0] * 14, abs=1e-15)
    thermal = _built(profile, "cz", (0, 1))
    twirled = pauli_twirl(thermal.channels, (0, 1))
    assert metrics.avg_from_pauli(twirled) == pytest.approx(thermal.achieved, abs=1e-12)
    one = _built(profile, "sx", (0,))
    twirled_1q = pauli_twirl(one.channels, (0,))
    assert twirled_1q[0] == pytest.approx(twirled_1q[1])  # relaxation twirls to px = py
    assert metrics.avg_from_pauli(twirled_1q) == pytest.approx(one.achieved, abs=1e-12)


def test_readout_matrix_columns_are_prepared_states() -> None:
    table = _profile(readout={"p1_given_0": 0.02, "p0_given_1": 0.05}).table
    m = readout_matrix(table.qubit(0))
    assert np.allclose(m @ [1, 0], [0.98, 0.02]) and np.allclose(m @ [0, 1], [0.05, 0.95])
    assert readout_matrix(_profile().table.qubit(0)) is None
