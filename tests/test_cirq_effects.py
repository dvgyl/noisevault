"""Leakage, atom loss and erasure in the Cirq export, against analytic results."""

import warnings

import numpy as np
import pytest
from conftest import require

cirq = require("cirq")

from noisevault.errors import NoiseApproximationWarning, UnsupportedEffect  # noqa: E402
from noisevault.frameworks.cirq import HeraldedErasure  # noqa: E402
from noisevault.profile import Profile  # noqa: E402

q0, q1, q2 = cirq.LineQubit.range(3)


def _atoms(*effects: dict, noise: float = 0.0, **gates) -> Profile:
    return Profile.uniform(
        "atoms",
        technology="neutral_atom",
        num_qubits=4,
        one_qubit_error=noise,
        two_qubit_error=noise,
        effects=effects,
        **gates,
    )


def _within_5_sigma(rate: float, expected: float, shots: int) -> None:
    sigma = np.sqrt(max(expected * (1 - expected), 1e-12) / shots)
    assert abs(rate - expected) < 5 * sigma + 1e-12, (rate, expected)


def test_an_unheralded_erasure_leaves_the_exact_mixed_state():
    model = _atoms({"type": "erasure", "gate": "x", "prob": 0.3}).to_cirq()
    rho = cirq.DensityMatrixSimulator(noise=model).simulate(cirq.Circuit(cirq.X(q0)))
    # X|0> = |1>; an erasure with probability p leaves (1 - p)|1><1| + p I/2.
    expected = np.diag([0.15, 0.85])
    np.testing.assert_allclose(rho.final_density_matrix, expected, atol=1e-6)


@pytest.mark.parametrize("simulator", [cirq.DensityMatrixSimulator, cirq.Simulator])
def test_a_heralded_erasure_heralds_with_its_prob_and_randomizes_the_qubit(simulator):
    model = _atoms({"type": "erasure", "gate": "x", "prob": 0.25, "heralded": True}).to_cirq()
    circuit = cirq.Circuit(cirq.X(q0), cirq.measure(q0, key="m"))
    shots = 600
    result = simulator(noise=model, seed=5).run(circuit, repetitions=shots)
    herald = result.measurements["herald_erasure_m0_q0"][:, 0].astype(bool)
    bit = result.measurements["m"][:, 0].astype(bool)
    _within_5_sigma(herald.mean(), 0.25, shots)
    assert bit[~herald].all()
    _within_5_sigma(bit[herald].mean(), 0.5, int(herald.sum()))


def test_each_qubit_of_a_two_qubit_gate_gets_its_own_herald():
    model = _atoms({"type": "erasure", "gate": "cz", "prob": 0.1, "heralded": True}).to_cirq()
    noisy = cirq.Circuit(cirq.CZ(q0, q1), cirq.CZ(q1, q2)).with_noise(model)
    heralds = [op.gate for op in noisy.all_operations() if isinstance(op.gate, HeraldedErasure)]
    assert [(h.p, h.key) for h in heralds] == [
        (0.1, "herald_erasure_m0_q0"),
        (0.1, "herald_erasure_m0_q1"),
        (0.1, "herald_erasure_m1_q1"),
        (0.1, "herald_erasure_m1_q2"),
    ]


def test_two_heralds_on_one_place_get_distinct_keys():
    model = _atoms(
        {"type": "erasure", "gate": "x", "prob": 0.1, "heralded": True},
        {"type": "erasure", "gate": "x", "prob": 0.2, "heralded": True},
    ).to_cirq()
    noisy = cirq.Circuit(cirq.X(q0)).with_noise(model)
    assert cirq.measurement_key_names(noisy) == {"herald_erasure_m0_q0", "herald_erasure_m0_q0_1"}


def test_readout_loss_acts_before_the_readout_flip_and_the_measurement():
    profile = _atoms(
        {"type": "atom_loss", "on": "readout", "prob": 0.2, "heralded": True}, readout_error=0.1
    )
    noisy = cirq.Circuit(cirq.measure(q0, key="m")).with_noise(profile.to_cirq())
    ops = list(noisy.all_operations())
    assert isinstance(ops[0].gate, HeraldedErasure) and ops[0].gate.key == "herald_atom_loss_m0_q0"
    assert cirq.is_measurement(ops[-1]) and ops[-1].gate.key == "m"
    assert profile.to_cirq(readout=False).noisy_operation(cirq.measure(q0)) == cirq.measure(q0)


def test_a_wait_gate_turns_an_idle_rate_into_a_probability():
    model = _atoms({"type": "erasure", "on": "idle", "rate_per_s": 2e3}).to_cirq()
    wait = cirq.WaitGate(cirq.Duration(micros=100)).on(q0)
    rho = cirq.DensityMatrixSimulator(noise=model).simulate(cirq.Circuit(cirq.X(q0), wait))
    p = 1 - np.exp(-2e3 * 100e-6)
    np.testing.assert_allclose(rho.final_density_matrix, np.diag([p / 2, 1 - p / 2]), atol=1e-6)


def test_cirq_and_stim_agree_on_the_measured_flip_rate():
    stim = require("stim")
    profile = _atoms({"type": "leakage", "gate": "cz", "prob": 0.08}, noise=2e-3)
    circuit = cirq.Circuit(cirq.X(q0), cirq.CZ(q0, q1), cirq.measure(q0, q1, key="m"))
    rho = cirq.DensityMatrixSimulator(noise=profile.to_cirq(readout=False)).simulate(circuit[:-1])
    probs = np.real(np.diag(rho.final_density_matrix))
    noisy = profile.to_stim(stim.Circuit("X 0\nCZ 0 1\nM 0 1"), readout="none")
    flips = noisy.compile_sampler(seed=6).sample(400_000)
    # The Pauli twirl keeps the flip rates of each qubit in Z, so both give the same marginals.
    _within_5_sigma(flips[:, 0].mean(), probs[2] + probs[3], 400_000)
    _within_5_sigma(flips[:, 1].mean(), probs[1] + probs[3], 400_000)


def test_the_report_states_how_each_effect_is_treated():
    model = _atoms(
        {"type": "erasure", "gate": "cz", "prob": 0.01, "heralded": True},
        {"type": "atom_loss", "on": "readout", "prob": 0.01},
        {"type": "erasure", "on": "idle", "rate_per_s": 10.0},
        {"type": "crosstalk_zz", "on": "idle", "strength_hz": 1e3},
    ).to_cirq(readout=False)
    lines = model.report.summary().splitlines()
    (exact,) = [line for line in lines if line.startswith("exact")]
    assert "effect erasure on cz: a heralded erasure (the qubit becomes I/2)" in exact
    assert "effect erasure on idle: an erasure (the qubit becomes I/2)" in exact
    (omitted,) = [line for line in lines if line.startswith("omitted")]
    assert "effect atom_loss on readout (the cirq export adds no readout error)" in omitted
    assert "effect crosstalk_zz on idle," in omitted


def test_approximated_atom_loss_does_not_meet_allow_exact():
    effect = {"type": "atom_loss", "gate": "cz", "prob": 0.01, "allow": "exact"}
    with pytest.raises(UnsupportedEffect, match="the cirq export approximates it as an erasure"):
        _atoms(effect).to_cirq()


def test_leakage_counted_in_a_locus_record_adds_nothing_there():
    data = _atoms({"type": "leakage", "gate": "cz", "prob": 0.05}, noise=1e-3).to_dict()
    data["calibrations"] = [{"gate": "cz", "qubits": [0, 1], "includes": ["leakage"]}]
    model = Profile.from_dict(data).to_cirq()
    noisy = cirq.Circuit(cirq.CZ(q0, q1), cirq.CZ(q1, q2)).with_noise(model)
    depolarized = [
        op.qubits for op in noisy.all_operations() if isinstance(op.gate, cirq.DepolarizingChannel)
    ]
    assert depolarized == [(q1,), (q2,)]
    assert "loci whose record lists leakage" in model.report.summary()


def test_herald_keys_skip_the_circuits_own_keys():
    model = _atoms({"type": "erasure", "gate": "x", "prob": 0.1, "heralded": True}).to_cirq()
    circuit = cirq.Circuit(cirq.X(q0), cirq.measure(q0, key="herald_erasure_m0_q0"))
    keys = cirq.measurement_key_names(circuit.with_noise(model))
    assert keys == {"herald_erasure_m0_q0", "herald_erasure_m0_q0_1"}


def test_run_refuses_a_measurement_only_circuit_that_it_would_simulate_one_time():
    profile = _atoms({"type": "erasure", "on": "readout", "prob": 0.5, "heralded": True})
    circuit = cirq.Circuit(cirq.measure(q0, key="m"))
    model = profile.to_cirq()
    noisy = circuit.with_noise(model)  # no error here, and the next call reuses the result
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kept = cirq.Simulator(noise=model, seed=1).simulate(circuit)
    assert "herald_erasure_m0_q0" in kept.measurements
    for simulator in (cirq.Simulator, cirq.DensityMatrixSimulator):
        for _ in range(2):  # the second call reuses the noisy circuit and must refuse again
            with pytest.raises(UnsupportedEffect, match="with_noise"):
                simulator(noise=model, seed=1).run(circuit, repetitions=5)
    result = cirq.Simulator(seed=1).run(noisy, repetitions=2000)
    _within_5_sigma(result.measurements["herald_erasure_m0_q0"].mean(), 0.5, 2000)
    _within_5_sigma(result.measurements["m"].mean(), 0.25, 2000)  # an erasure gives 1 half the time


@pytest.mark.parametrize("gate", ["measure", "reset"])
def test_an_effect_on_measure_or_reset_is_omitted_with_the_target_to_use(gate):
    data = _atoms().to_dict()
    data["gates"][gate] = {}
    data["effects"] = [{"type": "erasure", "gate": gate, "prob": 0.5}]
    model = Profile.from_dict(data).to_cirq()
    assert f"use on: readout or on: idle, not gate: {gate}" in model.report.summary()
    data["effects"][0]["allow"] = "exact"
    with pytest.raises(UnsupportedEffect, match="use on: readout"):
        Profile.from_dict(data).to_cirq()


def test_a_gate_with_typical_noise_takes_the_effects_of_the_typical_gate():
    data = _atoms({"type": "erasure", "gate": "cz", "prob": 0.2, "heralded": True}).to_dict()
    data["gates"] = {"h": data["gates"]["h"], "cz": data["gates"]["cz"]}
    model = Profile.from_dict(data).to_cirq()
    with pytest.warns(NoiseApproximationWarning, match="noise of cz"):
        noisy = cirq.Circuit(cirq.CNOT(q0, q1)).with_noise(model)
    assert cirq.measurement_key_names(noisy) == {"herald_erasure_m0_q0", "herald_erasure_m0_q1"}
