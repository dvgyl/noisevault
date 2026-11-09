"""Property tests: each framework export against the reference simulator on random circuits.

The tests use the runners, the expected distributions and the tolerances of ``nv check``. For
Stim, the tests also compute the exact distribution of the exported circuit, because samples
cannot show a difference smaller than the sampling error. Profiles are uniform: every qubit has
the same T1, T2 and readout error, and every gate of one arity has the same error and duration.
Thus each export must give the same distribution as the reference, whichever calibration it
charges for a gate.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from math import pi, sqrt
from pathlib import Path

import numpy as np
import pytest
from conftest import examples, require

require("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from noisevault import gates  # noqa: E402
from noisevault.check import (  # noqa: E402
    _RUNNERS,
    EXACT_TOLERANCE,
    FRAMEWORKS,
    Circuit,
    _big_endian,
    _compare,
    _Expected,
    _Runner,
    _Stim,
    _with_readout,
    check,
)
from noisevault.conversion import native_name  # noqa: E402
from noisevault.errors import NoiseApproximationWarning  # noqa: E402
from noisevault.profile import Profile, file_bytes, load_bytes  # noqa: E402
from noisevault.reference import Op, _apply  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::noisevault.errors.NoiseApproximationWarning")

MODULES = {"qiskit": "qiskit_aer", "cirq": "cirq", "pennylane": "pennylane", "stim": "stim"}
EXACT = ("qiskit", "cirq", "pennylane")
SHOTS, SEED = 20_000, 0  # the defaults of check()
DEVICE_QUBITS = 4
MAX_OPS = 10
# Every registry gate that a single calibration can describe. A profile can define each one.
CANDIDATES = tuple(
    info.name
    for info in gates.GATES.values()
    if info.unitary is not None and info.arity <= 2 and not info.multi_entangler
)
ANY_ANGLE = st.integers(-8, 8).map(lambda k: k * pi / 4) | st.floats(-2 * pi, 2 * pi)
# Stim simulates only Clifford gates, and a rotation is Clifford at multiples of pi/2.
CLIFFORD_ANGLE = st.integers(-4, 4).map(lambda k: k * pi / 2)
Setup = tuple[Profile, list[int]]


# generators ---------------------------------------------------------------------------------


def log_uniform(low: float, high: float) -> st.SearchStrategy[float]:
    return st.floats(np.log10(low), np.log10(high)).map(lambda x: float(10**x))


@st.composite
def setups(draw: st.DrawFn) -> Setup:
    """A uniform profile and a chain of 1 to 3 of its qubits, in random order."""
    width = draw(st.integers(1, 3))
    device = draw(st.integers(width, DEVICE_QUBITS))
    t1_us = draw(st.none() | log_uniform(10, 500))
    profile = Profile.uniform(
        "property",
        technology="superconducting",
        num_qubits=device,
        one_qubit_error=draw(log_uniform(1e-5, 1e-2)),
        two_qubit_error=draw(log_uniform(1e-4, 5e-2)),
        t1_us=t1_us,
        t2_us=None if t1_us is None else t1_us * draw(st.floats(0.1, 2.0)),
        one_qubit_ns=draw(st.none() | st.floats(10, 200)),
        two_qubit_ns=draw(st.none() | st.floats(20, 800)),
    )
    data = profile.to_dict()
    readout = draw(st.none() | st.tuples(st.floats(0, 0.1), st.floats(0, 0.1)))
    if readout is not None:
        data["readout"] = {"p1_given_0": readout[0], "p0_given_1": readout[1]}
    if draw(st.booleans()):
        # Each qubit gets its own readout and relaxation, so an export that charges the wrong
        # physical qubit gives other probabilities.
        data["qubits"] = [qubit_record(draw, q, t1_us) for q in range(device)]
    # A profile with fewer natives makes the exports charge the calibrations of other gates.
    data["gates"] = {name: data["gates"][name] for name in draw(natives(data["gates"]))}
    chain = draw(st.permutations(range(device)))[:width]
    return Profile.from_dict(data), chain


def qubit_record(draw: st.DrawFn, index: int, t1_us: float | None) -> dict[str, object]:
    record: dict[str, object] = {
        "index": index,
        "readout": {"p1_given_0": draw(st.floats(0, 0.1)), "p0_given_1": draw(st.floats(0, 0.1))},
    }
    if t1_us is not None:
        own = t1_us * draw(st.floats(0.2, 5.0))
        record |= {"t1_us": own, "t2_us": own * draw(st.floats(0.1, 2.0))}
    return record


@st.composite
def natives(draw: st.DrawFn, defined: dict[str, object]) -> set[str]:
    """A subset of the uniform natives with at least one calibrated gate of each arity."""
    ones = sorted(n for n in defined if gates.GATES[n].arity == 1 and n != "rz")
    twos = sorted(n for n in defined if gates.GATES[n].arity == 2)
    kept = draw(st.sets(st.sampled_from(ones), min_size=1))
    if twos:
        kept |= draw(st.sets(st.sampled_from(twos), min_size=1))
    if draw(st.booleans()):
        kept.add("rz")  # virtual, so the z-family gates are free
    return kept


@st.composite
def circuits(
    draw: st.DrawFn, setup: Setup, runner: _Runner, angle: st.SearchStrategy[float]
) -> Circuit:
    """Random gates on all chain qubits, each one a gate that ``runner`` can express.

    A gate is in the pool when the profile defines it or the gate that it equals. Each export
    then charges a calibration of the profile, never the typical-noise rule.
    """
    profile, chain = setup
    width = len(chain)
    pool = [
        name
        for name in CANDIDATES
        if gates.GATES[name].arity <= width and native_name(name, profile.gates) in profile.gates
    ]
    ops = []
    for _ in range(draw(st.integers(0, MAX_OPS))):
        info = gates.GATES[draw(st.sampled_from(pool))]
        qubits = tuple(draw(st.permutations(range(width)))[: info.arity])
        op = Op(info.name, qubits, tuple(draw(angle) for _ in info.params))
        if runner.cannot_express(op) is None:
            ops.append(op)
    return Circuit("random", width, tuple(ops))


def runner(framework: str, setup: Setup) -> _Runner:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoiseApproximationWarning)
        return _RUNNERS[framework](*setup)


def reloaded(profile: Profile, suffix: str) -> Profile:
    return load_bytes(file_bytes(profile, Path(f"profile{suffix}")))


# exact Stim distribution --------------------------------------------------------------------

_PAULIS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1, -1]).astype(complex),
}
_TWO_QUBIT_LABELS = [a + b for a in "IXYZ" for b in "IXYZ"][1:]  # Stim's argument order
_CLIFFORD_PARTS = np.array([-1, -sqrt(0.5), -0.5, 0, 0.5, sqrt(0.5), 1])


def _snapped(u: np.ndarray) -> np.ndarray:
    """A Clifford unitary at full precision. Stim stores unitaries in single precision."""

    def snap(x: np.ndarray) -> np.ndarray:
        return _CLIFFORD_PARTS[np.abs(x[..., None] - _CLIFFORD_PARTS).argmin(-1)]

    return snap(u.real) + 1j * snap(u.imag)


def _pauli_terms(name: str, args: list[float]) -> list[tuple[str, float]]:
    """The Pauli products of a Stim noise instruction, with their probabilities."""
    if name == "PAULI_CHANNEL_1":
        return list(zip("XYZ", args, strict=True))
    if name == "PAULI_CHANNEL_2":
        return list(zip(_TWO_QUBIT_LABELS, args, strict=True))
    if name in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
        return [(name[0], args[0])]
    if name == "DEPOLARIZE1":
        return [(p, args[0] / 3) for p in "XYZ"]
    if name == "DEPOLARIZE2":
        return [(p, args[0] / 15) for p in _TWO_QUBIT_LABELS]
    raise AssertionError(f"the exported Stim circuit has an unexpected instruction {name}")


def _pauli_channel(terms: list[tuple[str, float]]) -> list[np.ndarray]:
    k = len(terms[0][0])
    kraus = [sqrt(max(0.0, 1 - sum(p for _, p in terms))) * np.eye(2**k)]
    for label, p in terms:
        product = _PAULIS[label[0]]
        for letter in label[1:]:
            product = np.kron(product, _PAULIS[letter])
        kraus.append(sqrt(p) * product)
    return kraus


def stim_probabilities(noisy: object, n: int) -> np.ndarray:
    """Exact outcome probabilities of an exported Stim circuit that measures at the end.

    The gates take Stim's own unitaries, and the noise instructions take Stim's semantics. The
    flips of ``readout_flips`` apply after Stim's measurement, as ``sample_with_readout`` does.
    """
    import stim

    rho = np.zeros((2,) * (2 * n), dtype=complex)
    rho[(0,) * (2 * n)] = 1.0
    measured: list[int] = []
    flips: list[float] = []
    for inst in noisy.flattened():  # type: ignore[attr-defined]
        targets = [t.value for t in inst.targets_copy()]
        args = inst.gate_args_copy()
        if inst.name == "TICK":
            continue
        if inst.name == "M":
            measured += targets
            flips += [args[0] if args else 0.0] * len(targets)
            continue
        assert not measured, f"{inst} comes after a measurement"
        data = stim.gate_data(inst.name)
        if data.is_unitary:
            kraus = [_snapped(_big_endian(data.unitary_matrix))]
        else:
            kraus = _pauli_channel(_pauli_terms(inst.name, args))
        k = round(np.log2(len(kraus[0])))
        for i in range(0, len(targets), k):
            rho = _apply(rho, kraus, targets[i : i + k], n)
    assert measured == list(range(n))
    probs = np.real(np.diagonal(rho.reshape(2**n, 2**n)))
    exact = getattr(noisy, "readout_flips", None)
    matrices = []
    for q, flip in enumerate(flips):
        matrix = np.array([[1 - flip, flip], [flip, 1 - flip]])
        if exact is not None:
            f0, f1 = exact[q]
            matrix = np.array([[1 - f0, f1], [f0, 1 - f1]]) @ matrix
        matrices.append(matrix)
    return _with_readout(probs, matrices)


def stim_export(stim_runner: _Stim, circuit: Circuit, readout: str) -> object:
    """The Stim export of ``circuit`` that the check's Stim runner samples."""
    lines = [
        f"{stim_runner._instruction(op)} {' '.join(map(str, op.qubits))}" for op in circuit.ops
    ]
    lines.append("M " + " ".join(map(str, range(circuit.num_qubits))))
    layout = dict(enumerate(stim_runner.chain))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoiseApproximationWarning)
        return stim_runner._to_stim(
            stim_runner.profile, "\n".join(lines), layout=layout, readout=readout
        )


# 1. exports against the reference -----------------------------------------------------------


@settings(max_examples=examples(0.6))
@given(data=st.data())
def _exact_export_matches_reference(framework: str, data: st.DataObject) -> None:
    setup = data.draw(setups(), label="setup")
    framework_runner = runner(framework, setup)
    circuit = data.draw(circuits(setup, framework_runner, ANY_ANGLE), label="circuit")
    got = framework_runner.run(circuit, SHOTS, SEED)
    want = _Expected(*setup)(circuit, twirled=False)
    result = _compare(circuit, got, want, SHOTS, sampled=False)
    assert result.passed, f"TVD {result.tvd:.3g} > {EXACT_TOLERANCE:g}: {got} != {want}"


@pytest.mark.parametrize("framework", EXACT)
def test_exact_exports_match_the_reference(framework: str) -> None:
    require(MODULES[framework])
    _exact_export_matches_reference(framework)


def test_the_pennylane_runner_refuses_ms_at_other_phases() -> None:
    """Regression: the runner said it could express ms(0, pi/4), then raised ValueError."""
    require("pennylane")
    profile = Profile.uniform(
        "ms", technology="trapped_ion", num_qubits=2, one_qubit_error=1e-3, two_qubit_error=1e-2
    )
    pennylane_runner = runner("pennylane", (profile, [0, 1]))
    assert pennylane_runner.cannot_express(Op("ms", (0, 1), (0.0, pi / 4)))
    assert pennylane_runner.cannot_express(Op("ms", (0, 1), (0.0, 0.0))) is None


@settings(max_examples=examples(0.6))
@given(data=st.data())
def _stim_export_matches_twirled_reference(data: st.DataObject) -> None:
    setup = data.draw(setups(), label="setup")
    stim_runner = runner("stim", setup)
    assert isinstance(stim_runner, _Stim)
    circuit = data.draw(circuits(setup, stim_runner, CLIFFORD_ANGLE), label="circuit")
    expected = _Expected(*setup)
    for readout, symmetrized in (("exact", False), ("symmetrize", True)):
        got = stim_probabilities(stim_export(stim_runner, circuit, readout), circuit.num_qubits)
        want = expected(circuit, twirled=True, symmetrized=symmetrized)
        exact = _compare(circuit, got, want, SHOTS, sampled=False)
        assert exact.passed, f"readout={readout}: TVD {exact.tvd:.3g}: {got} != {want}"
    sampled = _compare(
        circuit,
        stim_runner.run(circuit, SHOTS, SEED),
        expected(circuit, twirled=True),
        SHOTS,
        sampled=True,
    )
    assert sampled.passed, f"Stim samples off by {sampled.deviation:.3g} > {sampled.tolerance:.3g}"


def test_stim_export_matches_the_twirled_reference() -> None:
    require("stim")
    _stim_export_matches_twirled_reference()


def test_stim_export_keeps_every_digit_of_its_arguments() -> None:
    """Regression: Stim's text has 6 significant digits, so M(0.01953125) became M(0.0195312)."""
    require("stim")
    from noisevault.frameworks.stim import to_stim

    error = 0.0123456789012
    profile = Profile.uniform(
        "digits", technology="superconducting", num_qubits=1, one_qubit_error=1e-3
    )
    profile = Profile.from_dict({**profile.to_dict(), "readout": {"error": error}})
    measured = to_stim(profile, "M 0")
    (flip,) = measured[0].gate_args_copy()
    assert flip == pytest.approx(error, rel=1e-12)
    text = f"QUBIT_COORDS({error}, 2) 0\nX_ERROR({error}) 0\nM 0"
    kept = to_stim(profile, text, existing_noise="keep", readout="none")
    assert [inst.gate_args_copy() for inst in kept][:2] == [[error, 2.0], [error]]


# 2. deterministic reports -------------------------------------------------------------------


@settings(max_examples=examples(0.2))
@given(data=st.data())
def _reports_are_deterministic(framework: str, data: st.DataObject) -> None:
    profile, chain = data.draw(setups(), label="setup")
    first = runner(framework, (profile, chain))
    angle = CLIFFORD_ANGLE if framework == "stim" else ANY_ANGLE
    circuit = data.draw(circuits((profile, chain), first, angle), label="circuit")
    builds: list[Callable[[], _Runner]] = [
        lambda: first,
        lambda: runner(framework, (profile, chain)),
        lambda: runner(framework, (reloaded(profile, ".json"), chain)),
        lambda: runner(framework, (reloaded(profile, ".json.gz"), chain)),
    ]
    outcomes = []
    for build in builds:
        built = build()
        result = built.run(circuit, SHOTS, SEED)
        reports = built.reports()
        outcomes.append(
            (result.tolist(), [r.to_dict() for r in reports], [r.summary() for r in reports])
        )
    assert all(outcome == outcomes[0] for outcome in outcomes[1:])


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_reports_are_deterministic(framework: str) -> None:
    require(MODULES[framework])
    _reports_are_deterministic(framework)


# 3. the public check ------------------------------------------------------------------------


@settings(max_examples=examples(0.1))
@given(setup=setups())
def _check_passes_on_uniform_profiles(setup: Setup) -> None:
    profile = setup[0]
    result = check(profile)
    assert result.passed, result.summary()
    # A framework can skip a profile whose natives it cannot express, such as Cirq on a profile
    # whose only 1-qubit native is u. An export must never refuse a uniform profile.
    refused = [why for _, why in result.skipped if why.startswith(("not installed", "the export"))]
    assert not refused, result.summary()
    again = check(reloaded(profile, ".json"))
    assert again.to_dict() == result.to_dict()
    assert again.summary() == result.summary()


def test_check_passes_on_uniform_profiles() -> None:
    for module in MODULES.values():
        require(module)
    _check_passes_on_uniform_profiles()


def test_nv_check_gives_the_same_numbers_every_time() -> None:
    # Cirq's default density-matrix simulator joins split states in set order, so the last bit
    # of a result changed between runs. An empty circuit gives one split state for each qubit.
    require(MODULES["cirq"])
    profile = Profile.uniform(
        "repeat",
        technology="superconducting",
        num_qubits=4,
        one_qubit_error=1e-3,
        two_qubit_error=1e-2,
    )
    profile = Profile.from_dict(
        {**profile.to_dict(), "readout": {"p1_given_0": 0.046875, "p0_given_1": 0.0}}
    )
    empty = Circuit("empty", 3, ())
    results = {
        tuple(runner("cirq", (profile, [0, 1, 2])).run(empty, SHOTS, SEED)) for _ in range(8)
    }
    assert len(results) == 1
    assert len({json.dumps(check(profile).to_dict()) for _ in range(3)}) == 1
