from __future__ import annotations

import functools
import importlib
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from conftest import require, toy
from typer.testing import CliRunner

import noisevault as nv
from noisevault import counts, gates, metrics
from noisevault.check import (
    EXACT_TOLERANCE,
    FRAMEWORKS,
    SIGMAS,
    CheckResult,
    NotRun,
    _Qiskit,
    _Stim,
    build_circuits,
    check,
    plan_circuits,
)
from noisevault.cli import app
from noisevault.errors import LayoutError, NoiseVaultError, install_hint
from noisevault.profile import Profile
from noisevault.reference import Op, _apply
from noisevault.reference import probabilities as reference

pytestmark = pytest.mark.filterwarnings("ignore::noisevault.errors.NoiseApproximationWarning")


@pytest.fixture(autouse=True)
def frameworks() -> None:
    for module in ("qiskit_aer", "cirq", "pennylane", "stim"):
        require(module)


def _relaxing(readout_error: float = 0.01) -> Profile:
    """Gates slow enough next to T1 that relaxation dominates their channel."""
    return Profile.uniform(
        "relaxing",
        technology="superconducting",
        num_qubits=3,
        one_qubit_error=1e-3,
        two_qubit_error=1e-2,
        readout_error=readout_error,
        t1_us=2,
        t2_us=3,
        one_qubit_ns=100,
        two_qubit_ns=300,
        connectivity=[(0, 1), (1, 2)],
    )


def test_every_export_matches_the_reference_on_manila() -> None:
    result = nv.load("ibm_manila").check()
    assert isinstance(result, CheckResult) and result.passed
    assert [f.framework for f in result.frameworks] == ["qiskit", "cirq", "pennylane", "stim"]
    assert not result.skipped
    for f in result.frameworks:
        assert len({c.circuit for c in f.circuits}) == len(result.circuits) >= 3
        assert not f.not_run
        if f.framework != "stim":
            assert all(c.tvd <= EXACT_TOLERANCE for c in f.circuits if not c.sampled)
            assert f.method.startswith("exact")


def test_an_export_that_drops_gate_noise_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from noisevault.frameworks import cirq as nv_cirq

    monkeypatch.setattr(nv_cirq.NoiseVaultNoiseModel, "_noise", lambda self, *a: [])
    result = check(nv.load("ibm_manila"), frameworks=["cirq", "pennylane"])
    by_name = {f.framework: f for f in result.frameworks}
    assert not by_name["cirq"].passed and by_name["cirq"].max_tvd > 1e-3
    assert by_name["pennylane"].passed
    assert not result.passed


def _one_pauli_error(
    mix: str, native: str, error: str, loci: list[list[int]], **connectivity: object
) -> Profile:
    pauli = [0.05 if label == error else 0.0 for label in metrics.PAULI_2Q]
    four = {"name": "blind", "vendor": "test", "technology": "superconducting", "num_qubits": 4}
    data = toy(
        device=four,
        connectivity={"edges": [[0, 1], [1, 2], [2, 3]], **connectivity},
        gates={mix: {"avg_infidelity": 0.0}, native: {"avg_infidelity": 0.0}},
        calibrations=[{"gate": native, "qubits": q, "pauli": pauli} for q in loci],
    )
    return Profile.model_validate(data)


@pytest.mark.parametrize(
    ("mix", "native", "error", "loci", "connectivity"),
    [
        ("h", "cx", "IX", [[0, 1], [1, 2], [2, 3]], {}),
        ("h", "cx", "XX", [[0, 1], [1, 2], [2, 3]], {}),
        ("sx", "cz", "IZ", [[2, 3]], {}),
        ("sx", "ecr", "YY", [[0, 1]], {"edges": [[0, 1], [2, 1], [2, 3]], "directed": True}),
        ("r", "zz", "ZZ", [[2, 3]], {}),
    ],
)
def test_only_the_chain_mirror_finds_an_entangler_error_that_the_export_drops(
    monkeypatch: pytest.MonkeyPatch, mix, native, error, loci, connectivity
) -> None:
    from noisevault.frameworks import cirq as nv_cirq

    profile = _one_pauli_error(mix, native, error, loci, **connectivity)
    (good,) = check(profile, frameworks=["cirq"], layout=[0, 1, 2, 3]).frameworks
    assert good.passed
    resolve = nv_cirq.resolve_op

    def without_native(table, name, qubits, **options):
        built = resolve(table, name, qubits, **options)
        return replace(built, channels=()) if name == native else built

    monkeypatch.setattr(nv_cirq, "resolve_op", without_native)
    (cirq,) = check(profile, frameworks=["cirq"], layout=[0, 1, 2, 3]).frameworks
    assert [c.circuit for c in cirq.circuits if not c.passed] == ["chain_mirror"]


def test_a_hardware_run_leaves_out_only_the_chain_mirror() -> None:
    profile = nv.load("ibm_kingston")
    chain, checked = plan_circuits(profile, None, purpose="check")
    assert plan_circuits(profile, None, purpose="run") == (
        chain,
        tuple(c for c in checked if c.name != "chain_mirror"),
    )
    assert [c.name for c in checked] == [
        "ghz_chain",
        "chain_mirror",
        "mirror",
        "single_qubit",
        "readout",
    ]


def test_pennylane_checks_an_ms_native_against_the_reference() -> None:
    natives = {"h": {"avg_infidelity": 0.01}, "ms": {"avg_infidelity": 0.02}}
    result = check(Profile.model_validate(toy(gates=natives)), frameworks=["pennylane"])
    (pennylane,) = result.frameworks
    assert pennylane.not_run == ()
    assert {c.circuit for c in pennylane.circuits} == {c.name for c in result.circuits}
    assert "ms" in next(c for c in pennylane.circuits if c.circuit == "ghz_chain").gates
    assert pennylane.passed and pennylane.max_tvd <= EXACT_TOLERANCE


def test_every_framework_runs_a_swap_native() -> None:
    errors = {"h": 0.01, "cx": 0.02, "swap": 0.03}
    natives = {name: {"avg_infidelity": error} for name, error in errors.items()}
    result = check(Profile.model_validate(toy(gates=natives)), layout=[0, 1])
    assert result.passed and not result.skipped
    for f in result.frameworks:
        assert f.not_run == (), f.framework
        pair = next(c for c in f.circuits if c.circuit == "two_qubit_natives")
        assert "swap" in pair.gates, f.framework


@pytest.mark.parametrize("framework", ["qiskit", "pennylane"])
def test_an_export_without_readout_error_fails(monkeypatch, framework) -> None:
    module = importlib.import_module(f"noisevault.frameworks.{framework}")
    export = getattr(module, f"to_{framework}")
    monkeypatch.setattr(module, f"to_{framework}", functools.partial(export, readout=False))
    (result,) = check(nv.load("ibm_manila"), frameworks=[framework]).frameworks
    assert not result.passed and result.max_tvd > 1e-3


def test_stim_is_compared_with_the_twirled_reference() -> None:
    profile, shots = _relaxing(), 50_000
    result = check(profile, frameworks=["stim"], shots=shots, seed=3)
    (stim,) = result.frameworks
    assert stim.passed and len({c.circuit for c in stim.circuits}) == len(result.circuits)
    # ghz_chain and readout run again with the export's default symmetrized readout
    assert len(stim.circuits) == len(result.circuits) + 2
    # On this profile the same samples are far outside 5 sigma of the untwirled channel.
    chain = [result.layout[i] for i in range(len(result.layout))]
    ghz = next(c for c in result.circuits if c.name == "ghz_chain")
    sampled = _Stim(profile, chain).run(ghz, shots, 3)
    full = reference(profile, ghz.ops, ghz.num_qubits, layout=chain, readout=True)
    sigma = np.sqrt(full * (1 - full) / shots)
    assert np.any(np.abs(sampled - full) > SIGMAS * sigma + SIGMAS / shots)


def test_stim_without_its_readout_flips_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from noisevault.frameworks import stim as nv_stim

    def perfect_readout(circuit, shots, *, seed=None):
        return circuit.compile_sampler(seed=seed).sample(shots)

    monkeypatch.setattr(nv_stim, "sample_with_readout", perfect_readout)
    (stim,) = check(_relaxing(readout_error=0.05), frameworks=["stim"]).frameworks
    assert not stim.passed


def test_stim_without_its_default_symmetrized_readout_fails(monkeypatch) -> None:
    from noisevault.frameworks import stim as nv_stim

    profile = _relaxing(readout_error=0.05)
    (good,) = check(profile, frameworks=["stim"], layout=[0, 1]).frameworks
    assert good.passed
    monkeypatch.setattr(nv_stim._Exporter, "_readout_flip", lambda self, qubits: 0.0)
    (stim,) = check(profile, frameworks=["stim"], layout=[0, 1]).frameworks
    assert not stim.passed


def test_stim_without_its_default_readout_fails_where_every_circuit_ends_uniform(
    monkeypatch,
) -> None:
    from noisevault.frameworks import stim as nv_stim

    four = {"name": "flat", "vendor": "test", "technology": "superconducting", "num_qubits": 4}
    ideal = {"h": {"avg_infidelity": 0.0}, "cz": {"avg_infidelity": 0.0}}
    data = toy(device=four, connectivity={"edges": [[0, 1], [1, 2], [2, 3]]}, gates=ideal)
    profile = Profile.model_validate({**data, "readout": {"error": 0.2}})
    options = {"frameworks": ["stim"], "layout": [0, 1, 2, 3], "shots": 100_000, "seed": 0}
    (good,) = check(profile, **options).frameworks
    assert good.passed
    monkeypatch.setattr(nv_stim._Exporter, "_readout_flip", lambda self, qubits: 0.0)
    (stim,) = check(profile, **options).frameworks
    assert not stim.passed


@pytest.mark.parametrize("framework", ["cirq", "pennylane", "stim"])
def test_an_export_that_puts_rz_before_p_for_a_fixed_phase_gate_fails(
    monkeypatch, framework
) -> None:
    from noisevault.conversion import native_name

    def rz_first(name, defined, rotation=None):
        if gates.GATES[name].family == "z" and name not in defined and "rz" in defined:
            return "rz"
        return native_name(name, defined, rotation)

    one = {"name": "one", "vendor": "test", "technology": "superconducting", "num_qubits": 1}
    errors = {"h": 0.01, "r": 0.01, "p": 0.2, "rz": 0.1}
    natives = {name: {"avg_infidelity": e} for name, e in errors.items()}
    profile = Profile.model_validate(toy(device=one, connectivity={"edges": []}, gates=natives))
    options = {"frameworks": [framework], "shots": 100_000}
    (good,) = check(profile, **options).frameworks
    assert good.passed
    module = importlib.import_module(f"noisevault.frameworks.{framework}")
    monkeypatch.setattr(module, "native_name", rz_first)
    (bad,) = check(profile, **options).frameworks
    assert not bad.passed
    assert [c.passed for c in bad.circuits if c.circuit == "fixed_phase"] == [False]


def test_stim_checks_rzz_at_its_clifford_angle() -> None:
    natives = {
        "h": {"avg_infidelity": 1e-3},
        "rz": {"virtual": True},
        "cx": {"avg_infidelity": 0.2},
        "rzz": {"avg_infidelity": 0.01},
    }
    profile = Profile.model_validate(toy(gates=natives, readout={"error": 0.01}))
    (stim,) = check(profile, frameworks=["stim"], layout=[0, 1]).frameworks
    assert stim.passed and not stim.not_run
    assert "rzz" in next(c for c in stim.circuits if c.circuit == "two_qubit_natives").gates


def test_rxx_and_ms_are_checked_as_different_natives() -> None:
    natives = {
        "h": {"avg_infidelity": 1e-3},
        "rz": {"virtual": True},
        "ms": {"avg_infidelity": 0.05},
        "rxx": {"avg_infidelity": 0.01},
    }
    profile = Profile.model_validate(toy(gates=natives, readout={"error": 0.01}))
    result = check(profile, layout=[0, 1])
    assert result.passed, result
    rxx = next(op for c in result.circuits for op in c.ops if op.name == "rxx")
    assert rxx.params == (np.pi / 4,)  # pi/2 would be the MS gate
    (stim,) = (f for f in result.frameworks if f.framework == "stim")
    assert any("ms" in c.gates for c in stim.circuits)


def _noiseless() -> Profile:
    one_qubit = {
        "name": "quiet",
        "vendor": "test",
        "technology": "superconducting",
        "num_qubits": 1,
    }
    return Profile.model_validate(
        toy(
            device=one_qubit,
            connectivity={"edges": []},
            gates={"h": {"avg_infidelity": 0.0}},
            readout={"error": 0.0},
        )
    )


def test_a_noiseless_profile_passes_stim() -> None:
    (stim,) = check(_noiseless(), frameworks=["stim"], shots=10_000).frameworks
    assert stim.passed and stim.max_tvd < 1e-12
    assert stim.worst.tolerance < 1e-2


def test_stim_adding_noise_to_a_noiseless_profile_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from noisevault.frameworks import stim as nv_stim

    sample = nv_stim.sample_with_readout

    def flipping(circuit, shots, *, seed=None):
        bits = sample(circuit, shots, seed=seed)
        return bits ^ (np.random.default_rng(seed).random(bits.shape) < 0.01)

    monkeypatch.setattr(nv_stim, "sample_with_readout", flipping)
    (stim,) = check(_noiseless(), frameworks=["stim"], shots=10_000).frameworks
    assert not stim.passed


def test_natives_a_framework_lacks_are_explained() -> None:
    result = nv.load("quantinuum_h1-1").check(frameworks=["stim"])
    assert dict(result.skipped) == {
        "stim": "Stim simulates only Clifford gates, and this profile's r gate is not Clifford"
        " at the check angles.",
    }
    result = nv.load("google_weber").check(frameworks=["stim"])
    assert dict(result.skipped) == {
        "stim": "Stim simulates only Clifford gates, and this profile's r gate is not Clifford"
        " at the check angles. Stim has no sqrt_iswap instruction.",
    }
    (stim,) = nv.load("ibm_brisbane").check(frameworks=["stim"]).frameworks
    assert stim.not_run == (
        NotRun("ghz_chain", "Stim has no ecr instruction"),
        NotRun("chain_mirror", "Stim has no ecr instruction"),
        NotRun("mirror", "Stim has no ecr instruction"),
    )


def _ions_with_rotations() -> Profile:
    errors = {"h": 1e-3, "ms": 0.02, "rxx": 0.01, "ryy": 0.01, "zz": 0.01, "rzz": 0.01}
    natives = {name: {"avg_infidelity": error} for name, error in errors.items()}
    return Profile.model_validate(toy(gates=natives, readout={"error": 0.01}))


def test_a_circuit_run_without_some_of_its_gates_names_them() -> None:
    result = check(_ions_with_rotations(), frameworks=["stim"], layout=[0, 1])
    planned = next(c for c in result.circuits if c.name == "two_qubit_natives")
    assert {"rxx", "ryy", "rzz"} <= {op.name for op in planned.ops}
    (stim,) = result.frameworks
    assert stim.passed
    assert stim.not_run == (
        NotRun(
            "two_qubit_natives",
            "; ".join(
                f"Stim simulates only Clifford gates, and this profile's {g} gate is not Clifford"
                " at the check angles"
                for g in ("rxx", "ryy", "rzz")
            ),
            ran_without=("rxx", "ryy", "rzz"),
        ),
    )
    (entry,) = result.to_dict()["frameworks"][0]["not_run"]
    assert entry["circuit"] == "two_qubit_natives"
    assert entry["ran_without"] == ["rxx", "ryy", "rzz"]
    assert (
        "two_qubit_natives ran without rxx, ryy, rzz: Stim simulates only Clifford gates"
        in result.summary()
    )


# Gates that take |0> to a superposition at the check angles.
MIXING = {"h", "sx", "sxdg", "rx", "ry", "r", "u"}


@pytest.mark.parametrize(
    ("ref", "frameworks"),
    [
        ("quantinuum_h1-1", ["stim", "cirq"]),
        ("google_weber", ["pennylane", "cirq"]),
        ("ibm_brisbane", ["stim", "qiskit"]),
    ],
)
def test_entangling_circuits_always_make_a_superposition(ref, frameworks) -> None:
    result = nv.load(ref).check(frameworks=frameworks)
    for circuit in result.circuits:
        if circuit.name not in ("single_qubit", "readout"):
            assert MIXING & {op.name for op in circuit.ops}, circuit
    for f in result.frameworks:
        for c in f.circuits:
            if c.circuit not in ("single_qubit", "readout"):
                assert MIXING & set(c.gates), (f.framework, c)


def test_cirq_readout_must_come_before_the_measurement(monkeypatch) -> None:
    from noisevault.frameworks import cirq as nv_cirq

    measure = nv_cirq.NoiseVaultNoiseModel._measure

    def after(self, operation, *args, **kwargs):
        tree = measure(self, operation, *args, **kwargs)
        return [operation, *(op for op in tree if op is not operation)]

    monkeypatch.setattr(nv_cirq.NoiseVaultNoiseModel, "_measure", after)
    (cirq,) = check(nv.load("ibm_manila"), frameworks=["cirq"]).frameworks
    assert not cirq.passed and cirq.max_tvd > 1e-3


def test_qiskit_readout_is_also_sampled_through_aer_measurement(monkeypatch) -> None:
    from qiskit_aer.noise import NoiseModel

    from noisevault.frameworks import qiskit as nv_qiskit

    export = nv_qiskit.to_qiskit

    def readout_dropped_at_run(profile, **options):
        sim = export(profile, **options)
        data = sim.noise_model.to_dict()
        data["errors"] = [e for e in data["errors"] if e["type"] != "roerror"]
        run = sim.run

        def run_without_readout(circuits, *args, **kwargs):
            sim.set_options(noise_model=NoiseModel.from_dict(data))
            return run(circuits, *args, **kwargs)

        sim.run = run_without_readout
        return sim

    (good,) = check(nv.load("ibm_manila"), frameworks=["qiskit"]).frameworks
    sampled = [c.circuit for c in good.circuits if c.sampled]
    assert good.passed and sampled == ["ghz_chain", "readout"]
    monkeypatch.setattr(nv_qiskit, "to_qiskit", readout_dropped_at_run)
    (bad,) = check(nv.load("ibm_manila"), frameworks=["qiskit"]).frameworks
    assert all(c.passed for c in bad.circuits if not c.sampled)
    assert [c.passed for c in bad.circuits if c.sampled] == [False, False]


def test_a_missing_framework_is_skipped_with_the_install_command(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "stim", None)
    result = check(nv.load("ibm_manila"), frameworks=["stim"])
    assert result.frameworks == ()
    assert result.skipped == (("stim", f"not installed: {install_hint('stim')}"),)
    assert not result.passed


def _asks_for_an_effect() -> Profile:
    effect = {"type": "atom_loss", "on": "readout", "prob": 1e-3, "allow": "exact"}
    return Profile.model_validate(toy(effects=[effect]))


def _refused(framework: str) -> str:
    start = "the export refused this profile: effect atom_loss on readout asks for allow='exact',"
    if framework in ("stim",):
        return (
            f"{start} but the {framework} export approximates it as an erasure (the qubit becomes"
            " I/2) (the atom comes back at once; on the device the trap stays empty); set allow"
            " to 'approximate' to accept the approximation"
        )
    return (
        f"{start} but {framework} export does not model effects yet; set allow to 'omit' to"
        " export without the effect"
    )


def test_every_export_that_refuses_the_profile_is_skipped_with_its_reason() -> None:
    result = check(_asks_for_an_effect())
    assert result.frameworks == ()
    assert result.skipped == tuple((name, _refused(name)) for name in FRAMEWORKS)


def test_nv_check_names_every_export_that_refuses_the_profile(tmp_path: Path) -> None:
    path = _asks_for_an_effect().save(tmp_path / "effect.json")
    result = CliRunner().invoke(app, ["check", str(path)], env={"COLUMNS": "200"})
    lines = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout).splitlines()
    assert [line.split() for line in lines[3:7]] == [[name, "skipped"] for name in FRAMEWORKS]
    assert lines[7:] == [f"{name}: {_refused(name)}" for name in FRAMEWORKS]
    assert result.stderr == "error: no framework could run the check\n"
    assert result.exit_code == 1


@pytest.mark.parametrize(
    "ref", ["ibm_manila", "ibm_brisbane", "quantinuum_h1-1", "google_rainbow", "uniform"]
)
def test_mirror_circuits_undo_themselves(ref: str) -> None:
    profile = (
        Profile.uniform(
            "u", technology="neutral_atom", num_qubits=4, one_qubit_error=1e-3, two_qubit_error=1e-2
        )
        if ref == "uniform"
        else nv.load(ref)
    )
    circuits = build_circuits(profile, list(profile.suggest_layout(3).values()))
    mirror = next(c for c in circuits if c.name == "mirror")
    n = mirror.num_qubits
    rho = np.zeros((2,) * (2 * n), dtype=complex)
    rho[(0,) * (2 * n)] = 1
    for op in mirror.ops:
        rho = _apply(rho, [gates.GATES[op.name].unitary(*op.params)], op.qubits, n)
    assert abs(rho[(0,) * (2 * n)]) == pytest.approx(1, abs=1e-9)
    assert len({op.name for op in mirror.ops}) >= 2


def test_directed_natives_run_in_their_calibrated_direction() -> None:
    profile = nv.load("ibm_brisbane")
    (qiskit,) = check(profile, frameworks=["qiskit"]).frameworks
    assert qiskit.passed and "ecr" in qiskit.circuits[0].gates


def test_a_layout_with_unconnected_neighbors_says_how_to_fix_it() -> None:
    manila = nv.load("ibm_manila")
    with pytest.raises(LayoutError) as info:
        check(manila, layout=[0, 2])
    default = check(manila, frameworks=["cirq"])
    assert default.passed, default
    chain = [*default.layout.values()]
    assert info.value.hint == f"pass a layout whose neighbors share one, such as layout={chain}"


def _disabled(profile: Profile, *qubits: int) -> Profile:
    data = profile.model_dump(mode="json", exclude_none=True)
    data["qubits"] = [{"index": q, "disabled": True} for q in qubits]
    return Profile.model_validate(data)


def test_the_default_chain_leaves_out_disabled_qubits() -> None:
    uniform = Profile.uniform(
        "u",
        technology="superconducting",
        num_qubits=3,
        one_qubit_error=0.01,
        two_qubit_error=0.02,
        readout_error=0.03,
    )
    result = check(_disabled(uniform, 0))
    assert result.layout == {0: 1, 1: 2}
    assert result.passed, result
    assert [f.framework for f in result.frameworks] == ["qiskit", "cirq", "pennylane", "stim"]


@pytest.mark.parametrize(("num_qubits", "layout"), [(4, {0: 2, 1: 3}), (2, {0: 0})])
def test_the_default_chain_is_the_longest_the_enabled_qubits_connect(num_qubits, layout) -> None:
    line = toy(
        device={"name": "line", "technology": "superconducting", "num_qubits": num_qubits},
        connectivity={"edges": [[i, i + 1] for i in range(num_qubits - 1)]},
        readout={"error": 0.01},
    )
    result = check(_disabled(Profile.model_validate(line), 1), frameworks=["cirq"])
    assert result.layout == layout
    assert result.passed, result


def test_the_default_chain_leaves_out_a_link_only_a_custom_gate_calibrates() -> None:
    gates = {
        "rz": {"virtual": True},
        "sx": {"avg_infidelity": 1e-3},
        "cz": {"avg_infidelity": 1e-2},
        "custom": {"qubits": 2},
    }
    custom = [{"gate": "custom", "qubits": [1, 2], "avg_infidelity": 2e-2}]
    data = toy(gates=gates, calibrations=custom, connectivity={"edges": [[0, 1]]})
    profile = Profile.model_validate(data)
    assert profile.suggest_layout(3) == {0: 0, 1: 1, 2: 2}
    with pytest.raises(
        LayoutError, match="qubits 1 and 2 share no calibrated 2-qubit native"
    ) as info:
        check(profile, layout=[0, 1, 2])
    assert info.value.hint == "pass a layout whose neighbors share one, such as layout=[0, 1]"
    assert check(profile, frameworks=["cirq"], layout=[0, 1]).passed
    result = check(profile, frameworks=["cirq"])
    assert result.layout == {0: 0, 1: 1}
    assert result.passed, result
    assert any("cz" in c.gates for c in result.frameworks[0].circuits)


_NO_ONE_QUBIT = "use a profile that calibrates a 1-qubit native gate with a known unitary"


def test_a_layout_with_an_unusable_pair_names_a_layout_that_runs_or_the_calibration() -> None:
    gates = {"sx": {"avg_infidelity": 1e-3}, "cz": {"qubits": 2}, "custom": {"qubits": 2}}
    custom = [{"gate": "custom", "qubits": [0, 1], "avg_infidelity": 2e-2}]
    profile = Profile.model_validate(toy(gates=gates, calibrations=custom))
    with pytest.raises(LayoutError, match="qubits 0 and 1 share no calibrated") as info:
        check(profile, layout=[0, 1])
    assert info.value.hint == "pass a layout whose neighbors share one, such as layout=[0]"
    assert check(profile, frameworks=["cirq"], layout=[0]).passed
    gates["sx"] = {}
    profile = Profile.model_validate(toy(gates=gates, calibrations=custom))
    with pytest.raises(LayoutError, match="qubits 0 and 1 share no calibrated") as info:
        check(profile, layout=[0, 1])
    assert info.value.hint == _NO_ONE_QUBIT


def test_the_default_chain_routes_around_a_link_only_a_custom_gate_calibrates() -> None:
    gates = {"h": {"avg_infidelity": 1e-3}, "cz": {"qubits": 2}, "custom": {"qubits": 2}}
    calibrations = [
        {"gate": "custom", "qubits": [0, 1], "avg_infidelity": 1e-3},
        *({"gate": "cz", "qubits": [q, q + 1], "avg_infidelity": 1e-2} for q in (1, 2, 3)),
    ]
    line = toy(
        device={"name": "line", "technology": "superconducting", "num_qubits": 5},
        connectivity={"edges": [[q, q + 1] for q in range(4)]},
        gates=gates,
        calibrations=calibrations,
        readout={"error": 0.01},
    )
    result = check(Profile.model_validate(line), frameworks=["cirq"])
    assert result.layout == {0: 1, 1: 2, 2: 3, 3: 4}
    assert result.passed, result


def _measure_off(*qubits: int) -> Profile:
    gates = {
        "h": {"avg_infidelity": 0.001},
        "x": {"avg_infidelity": 0.001},
        "cx": {"avg_infidelity": 0.01},
        "measure": {},
    }
    off = [{"gate": "measure", "qubits": [q], "disabled": True} for q in qubits]
    return Profile.model_validate(toy(gates=gates, calibrations=off, readout={"error": 0.01}))


def test_the_default_chain_leaves_out_a_qubit_that_cannot_measure() -> None:
    result = check(_measure_off(0))
    assert result.layout == {0: 1, 1: 2}
    assert result.passed, result
    assert [f.framework for f in result.frameworks] == ["qiskit", "cirq", "pennylane", "stim"]


def test_a_layout_with_a_qubit_that_cannot_measure_names_the_qubits_that_can() -> None:
    with pytest.raises(LayoutError) as info:
        check(_measure_off(1), layout=[0, 1, 2])
    assert info.value.message == (
        "test_toy disables measure on qubit 1, but every check circuit measures all its qubits"
    )
    assert info.value.hint == "use qubits that can measure, such as layout=[0]"
    assert check(_measure_off(1), frameworks=["cirq"], layout=[0]).passed
    with pytest.raises(LayoutError) as info:
        check(_measure_off(0, 1, 2), layout=[0])
    assert info.value.hint is None


def test_a_measure_hint_names_only_qubits_with_a_calibrated_one_qubit_native() -> None:
    sx = [{"gate": "sx", "qubits": [2], "avg_infidelity": 1e-3}]
    off = [{"gate": "measure", "qubits": [1], "disabled": True}]
    gates = {"rz": {"virtual": True}, "sx": {}, "cz": {"avg_infidelity": 1e-2}, "measure": {}}
    profile = Profile.model_validate(toy(gates=gates, calibrations=[*sx, *off]))
    with pytest.raises(LayoutError, match="disables measure on qubit 1") as info:
        check(profile, layout=[1])
    assert info.value.hint == "use qubits that can measure, such as layout=[2]"
    assert check(profile, frameworks=["cirq"], layout=[2]).passed
    profile = Profile.model_validate(toy(gates=gates, calibrations=off))
    with pytest.raises(LayoutError, match="disables measure on qubit 1") as info:
        check(profile, layout=[1])
    assert info.value.hint == _NO_ONE_QUBIT


def test_a_device_where_no_qubit_can_measure_has_no_default_chain() -> None:
    with pytest.raises(
        LayoutError, match=r"^test_toy has only 0 enabled qubits that can measure, not 1$"
    ):
        check(_measure_off(0, 1, 2))


def _sx_calibrated_on(*qubits: int) -> Profile:
    gates = {"rz": {"virtual": True}, "sx": {}, "cz": {"avg_infidelity": 1e-2}}
    sx = [{"gate": "sx", "qubits": [q], "avg_infidelity": 1e-2} for q in qubits]
    return Profile.model_validate(toy(gates=gates, calibrations=sx, readout={"error": 0.01}))


@pytest.mark.parametrize(
    ("calibrated", "layout", "on"),
    [((0, 1), {0: 0, 1: 1}, "on qubits 0-1"), ((2,), {0: 2}, "on qubit 2")],
)
def test_the_default_chain_leaves_out_qubits_no_one_qubit_native_calibrates(
    calibrated, layout, on
) -> None:
    profile = _sx_calibrated_on(*calibrated)
    assert profile.suggest_layout(3) == {0: 0, 1: 1, 2: 2}
    with pytest.raises(NoiseVaultError, match=r"known unitary on qubits 0-1-2, so there") as info:
        check(profile, layout=[0, 1, 2])
    assert info.value.hint == f"pass layout= with other qubits, such as layout={[*layout.values()]}"
    assert check(profile, frameworks=["cirq"], layout=list(layout.values())).passed
    result = check(profile, frameworks=["cirq"])
    assert result.layout == layout
    assert result.passed, result
    assert result.summary().splitlines()[0].endswith(f" circuits {on}")


@pytest.mark.parametrize("h", [{}, {"disabled": True}])
def test_check_and_plan_ask_for_a_calibration_when_no_layout_has_check_circuits(h) -> None:
    device = {"name": "toy", "technology": "superconducting", "num_qubits": 2}
    gates = {"h": h, "cx": {"avg_infidelity": 0.01}}
    profile = Profile.model_validate(toy(device=device, connectivity="all_to_all", gates=gates))
    for layout in ([0], [1], [0, 1], [1, 0]):
        with pytest.raises(NoiseVaultError) as info:
            check(profile, frameworks=["cirq"], layout=layout)
        assert info.value.message.endswith("so there is nothing to check")
        assert info.value.hint == _NO_ONE_QUBIT
    for layout in (None, [0]):
        with pytest.raises(NoiseVaultError) as info:
            counts.plan(profile, layout=layout)
        assert info.value.message.endswith("so there is nothing to run")
        assert info.value.hint == _NO_ONE_QUBIT


def test_a_pair_the_qiskit_export_lacks_is_named_the_way_the_cli_does() -> None:
    runner = _Qiskit(Profile.model_validate(toy(readout={"error": 0.01})), [0, 2])
    assert runner.cannot_express(Op("cz", (0, 1))) == "the Qiskit export has no cz on qubits 0-2"


def test_with_no_calibrated_one_qubit_native_the_one_qubit_chain_has_nothing_to_check(
    tmp_path: Path,
) -> None:
    with pytest.raises(NoiseVaultError) as info:
        check(_sx_calibrated_on())
    assert info.value.message.endswith("on qubit 0, so there is nothing to check")
    assert info.value.hint == _NO_ONE_QUBIT
    path = _sx_calibrated_on().save(tmp_path / "toy.json")
    result = CliRunner().invoke(app, ["check", str(path)])
    assert result.stderr.startswith("error: test_toy has no calibrated native gate")
    assert f"hint: {_NO_ONE_QUBIT}" in result.stderr and "layout=" not in result.stderr


def test_a_device_with_every_qubit_disabled_has_no_default_chain() -> None:
    profile = _disabled(Profile.model_validate(toy()), 0, 1, 2)
    with pytest.raises(LayoutError, match=r"^test_toy has only 0 usable qubits, not 1$"):
        check(profile)


def test_bad_arguments_name_the_choices() -> None:
    profile = nv.load("ibm_manila")
    with pytest.raises(ValueError, match="Choose from qiskit, cirq, pennylane, stim"):
        check(profile, frameworks=["qiskt"])
    with pytest.raises(ValueError, match="positive number of shots"):
        check(profile, shots=0)


def test_result_serializes_and_prints() -> None:
    result = nv.load("ibm_manila").check(frameworks=["cirq"], layout=[2, 1, 0])
    data = json.loads(json.dumps(result.to_dict()))
    assert data["passed"] is True and data["layout"] == {"0": 2, "1": 1, "2": 0}
    (report,) = data["frameworks"][0]["reports"]
    assert report["framework"] == "cirq" and report["approximated"]
    assert "A pass does not measure how well the model matches the hardware." in str(result)


def _line(num_qubits: int, **sections) -> Profile:
    device = {"name": "line", "vendor": "test", "technology": "superconducting"}
    return Profile.model_validate(
        toy(
            device={**device, "num_qubits": num_qubits},
            connectivity={"edges": [[q, q + 1] for q in range(num_qubits - 1)]},
            **sections,
        )
    )


def test_nv_check_json_names_every_qubit_that_an_export_report_names(tmp_path: Path) -> None:
    readout = {"p1_given_0": 0.01, "p0_given_1": 0.02}
    path = _line(8, qubits=[{"index": q, "readout": readout} for q in (3, 4, 5)]).save(
        tmp_path / "eight.json"
    )
    result = CliRunner().invoke(app, ["check", str(path), "--framework", "qiskit", "--json"])
    (report,) = json.loads(result.stdout)["frameworks"][0]["reports"]
    assert report["unknown"] == [
        "readout error of qubits 0, 1, 2, 6 and 7",
        "preparation (reset) error of qubits 0, 1, 2, 3, 4, 5, 6 and 7",
    ]


def test_a_saved_stim_check_names_the_qubits_of_every_circuit_once() -> None:
    result = check(
        _line(4, gates={"z": {"avg_infidelity": 1e-3}}), frameworks=["stim"], layout=[0, 1, 2, 3]
    )
    assert [(c.name, c.num_qubits) for c in result.circuits] == [
        ("single_qubit", 2),
        ("readout", 4),
    ]
    saved = json.loads(json.dumps(result.to_dict()))
    reports = saved["frameworks"][0]["reports"]
    assert [r["options"]["readout"] for r in reports] == ["exact", "symmetrize"]
    for report in reports:
        assert report["unknown"] == ["readout error of physical qubits 0, 1, 2 and 3"]


def _stim_without_the_readout_of_qubit_3(monkeypatch: pytest.MonkeyPatch) -> Profile:
    from noisevault.frameworks import stim as nv_stim

    flip = nv_stim._Exporter._readout_flip

    def without_qubit_3(self, qubits):
        return flip(self, [q for q in qubits if self.physical[q] != 3])

    monkeypatch.setattr(nv_stim._Exporter, "_readout_flip", without_qubit_3)
    return _line(
        4,
        gates={"h": {"avg_infidelity": 1e-3}, "cz": {"avg_infidelity": 1e-2}},
        readout={"error": 0.1},
        qubits=[{"index": 3, "readout": {"error": 0.003}}],
    )


def test_a_failing_sampled_check_names_a_circuit_that_failed(monkeypatch) -> None:
    profile = _stim_without_the_readout_of_qubit_3(monkeypatch)
    worst = [
        check(profile, frameworks=["stim"], seed=seed).frameworks[0].worst for seed in range(4)
    ]
    assert [w.passed for w in worst] == [False] * 4
    assert all(w.circuit == "readout" and w.deviation > w.tolerance for w in worst)


def test_nv_check_prints_the_outcome_that_failed_against_its_tolerance(
    monkeypatch, tmp_path: Path
) -> None:
    path = _stim_without_the_readout_of_qubit_3(monkeypatch).save(tmp_path / "four.json")
    result = CliRunner().invoke(app, ["check", str(path), "--framework", "stim"])
    lines = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout).splitlines()
    assert lines[2].split()[:4] == ["framework", "result", "deviation", "tolerance"]
    assert lines[3].split()[:4] == ["stim", "FAIL", "2.2e-03", "1.9e-03"]
    assert result.exit_code == 1


def test_profile_check_spells_out_its_options() -> None:
    import inspect

    signature = inspect.signature(Profile.check)
    assert list(signature.parameters) == ["self", "frameworks", "layout", "shots", "seed"]
    assert signature.return_annotation == "CheckResult"
    assert list(inspect.signature(Profile.diff).parameters) == ["self", "other", "top"]


def test_a_profile_with_only_rotations_passes_every_framework() -> None:
    natives = {"rz": 2e-3, "rx": 3e-2, "rxx": 4e-2, "ryy": 5e-2, "rzz": 6e-2}
    gates = {name: {"avg_infidelity": error} for name, error in natives.items()}
    profile = Profile.model_validate(toy(gates=gates, readout={"error": 0.01}))
    result = check(profile, layout=[0, 1])
    assert result.passed, result
    assert [f.framework for f in result.frameworks] == ["qiskit", "cirq", "pennylane", "stim"]
    assert all(not f.not_run for f in result.frameworks), result


def test_a_one_qubit_uniform_profile_runs_in_every_framework() -> None:
    profile = Profile.uniform(
        "u",
        technology="superconducting",
        num_qubits=1,
        one_qubit_error=1e-2,
        two_qubit_error=2e-2,
        readout_error=0.02,
    )
    result = check(profile)
    assert result.passed, result
    assert [f.framework for f in result.frameworks] == ["qiskit", "cirq", "pennylane", "stim"]
    assert not result.skipped
    assert {c.num_qubits for c in result.circuits} == {1}


def test_pennylane_checks_an_adjoint_native_with_its_own_noise() -> None:
    require("pennylane")
    data = toy(
        connectivity="all_to_all",
        gates={
            "sx": {"avg_infidelity": 1e-3},
            "sdg": {"avg_infidelity": 2e-2},
            "rz": {"avg_infidelity": 5e-3},
            "cz": {"avg_infidelity": 1e-2},
        },
    )
    [pennylane] = check(Profile.model_validate(data), frameworks=["pennylane"]).frameworks
    assert pennylane.passed
    assert pennylane.not_run == ()
