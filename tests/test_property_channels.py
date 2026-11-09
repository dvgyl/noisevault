"""Property tests for the metric conversions in ``metrics`` and the channels in ``channels``.

Superoperators use column stacking, as in ``channels``. The tests compute fidelities from the
superoperator trace, not from ``average_gate_fidelity``, so the two calculations check each other.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import toy
from hypothesis import assume, given
from hypothesis import strategies as st

from noisevault import metrics
from noisevault.channels import (
    ChannelSpec,
    GateChannels,
    average_gate_fidelity,
    depolarizing_kraus,
    gate_channels,
    pauli_kraus,
    pauli_twirl,
    readout_matrix,
    superoperator,
    thermal_relaxation_kraus,
)
from noisevault.profile import Profile
from noisevault.table import GateNoise, QubitNoise

ATOL = 1e-9
ARITIES = st.sampled_from([1, 2])
FACTORS = st.floats(0.0, 4.0)
UNIT = st.floats(0.0, 1.0)


def _superop(kraus: list[np.ndarray]) -> np.ndarray:
    return np.sum([np.kron(np.conj(k), k) for k in kraus], axis=0)


def _infidelity(superop: np.ndarray) -> float:
    """Average infidelity to the identity, from the process fidelity tr(S) / d^2."""
    d = round(math.sqrt(superop.shape[0]))
    return 1.0 - (d * np.trace(superop).real / d**2 + 1.0) / (d + 1.0)


def _assert_cptp(kraus: list[np.ndarray] | tuple[np.ndarray, ...]) -> None:
    d = kraus[0].shape[0]
    assert np.allclose(sum(k.conj().T @ k for k in kraus), np.eye(d), atol=ATOL)


def _choi_is_positive(superop: np.ndarray) -> bool:
    d = round(math.sqrt(superop.shape[0]))
    choi = superop.reshape(d, d, d, d).transpose(3, 1, 2, 0).reshape(d * d, d * d)
    return bool(np.linalg.eigvalsh((choi + choi.conj().T) / 2).min() > -ATOL)


@st.composite
def avg_infidelities(draw: st.DrawFn, num_qubits: int) -> float:
    return draw(st.floats(0.0, metrics.max_avg_infidelity(num_qubits)))


@st.composite
def pauli_vectors(draw: st.DrawFn, num_qubits: int, max_total: float = 1.0) -> list[float]:
    """Non-identity Pauli probabilities with a sum of at most ``max_total``."""
    weights = draw(st.lists(UNIT, min_size=4**num_qubits - 1, max_size=4**num_qubits - 1))
    total = draw(st.floats(0.0, max_total))
    norm = math.fsum(weights)
    return [0.0] * len(weights) if norm == 0 else [total * w / norm for w in weights]


@st.composite
def arity_and_pauli(draw: st.DrawFn, max_total: float = 1.0) -> tuple[int, list[float]]:
    n = draw(ARITIES)
    return n, draw(pauli_vectors(n, max_total))


# metrics ------------------------------------------------------------------------------------


@given(n=st.integers(1, 3), data=st.data())
def test_process_and_avg_infidelity_round_trip(n: int, data: st.DataObject) -> None:
    r = data.draw(avg_infidelities(n))
    e_pro = metrics.process_from_avg(r, n)
    assert 0.0 <= e_pro <= 1.0 + 1e-12
    assert metrics.avg_from_process(e_pro, n) == pytest.approx(r, abs=1e-15)


@given(n=st.integers(1, 3), data=st.data())
def test_depolarizing_and_avg_infidelity_round_trip(n: int, data: st.DataObject) -> None:
    r = data.draw(avg_infidelities(n))
    lam = metrics.depolarizing_from_avg(r, n)
    assert 0.0 <= lam <= metrics.max_depolarizing_param(n) + 1e-12
    assert metrics.avg_from_depolarizing(lam, n) == pytest.approx(r, abs=1e-15)


@given(
    n=ARITIES,
    kind=st.sampled_from(["avg_infidelity", "process_infidelity", "depolarizing_param"]),
    share=UNIT,
)
def test_every_metric_in_its_range_converts_to_an_avg_infidelity_in_range(
    n: int, kind: metrics.MetricKind, share: float
) -> None:
    upper = {
        "avg_infidelity": metrics.max_avg_infidelity(n),
        "process_infidelity": 1.0,
        "depolarizing_param": metrics.max_depolarizing_param(n),
    }[kind]
    value = share * upper
    metrics.check_metric(kind, value, n)
    r = metrics.to_avg_infidelity(kind, value, n)
    assert 0.0 <= r <= metrics.max_avg_infidelity(n) + 1e-12


@given(
    n=ARITIES,
    kind=st.sampled_from(["avg_infidelity", "process_infidelity", "depolarizing_param"]),
    excess=st.floats(1e-9, 10.0),
    below=st.booleans(),
)
def test_check_metric_refuses_values_outside_the_physical_range(
    n: int, kind: metrics.MetricKind, excess: float, below: bool
) -> None:
    upper = {
        "avg_infidelity": metrics.max_avg_infidelity(n),
        "process_infidelity": 1.0,
        "depolarizing_param": metrics.max_depolarizing_param(n),
    }[kind]
    value = -excess if below else upper + 1e-11 + excess
    with pytest.raises(ValueError):
        metrics.check_metric(kind, value, n)


@given(case=arity_and_pauli())
def test_valid_pauli_vectors_pass_and_convert_like_their_channel(
    case: tuple[int, list[float]],
) -> None:
    n, pauli = case
    metrics.check_metric("pauli", pauli, n)
    kraus = pauli_kraus(pauli)
    _assert_cptp(kraus)
    assert metrics.avg_from_pauli(pauli) == pytest.approx(_infidelity(_superop(kraus)), abs=ATOL)


@given(
    case=arity_and_pauli(),
    index=st.integers(0, 14),
    bad=st.sampled_from([math.nan, -math.inf, math.inf, -1e-3]),
)
def test_check_metric_refuses_a_pauli_vector_with_a_bad_entry(
    case: tuple[int, list[float]], index: int, bad: float
) -> None:
    n, pauli = case
    pauli[index % len(pauli)] = bad
    with pytest.raises(ValueError):
        metrics.check_metric("pauli", pauli, n)


def test_check_metric_refuses_a_nan_pauli_entry() -> None:
    with pytest.raises(ValueError):
        metrics.check_metric("pauli", [math.nan, 0.0, 0.0], 1)


@given(n=ARITIES, data=st.data())
def test_uniform_pauli_is_the_depolarizing_channel(n: int, data: st.DataObject) -> None:
    r = data.draw(avg_infidelities(n))
    pauli = metrics.uniform_pauli(r, n)
    assert metrics.avg_from_pauli(pauli) == pytest.approx(r, abs=1e-12)
    assert np.allclose(_superop(pauli_kraus(pauli)), _superop(depolarizing_kraus(r, n)), atol=ATOL)


@given(pauli=pauli_vectors(2))
def test_swapping_a_two_qubit_pauli_vector_conjugates_its_channel_by_swap(
    pauli: list[float],
) -> None:
    swap = np.eye(4)[[0, 2, 1, 3]]
    swapped = metrics.swap_pauli_2q(pauli)
    assert metrics.swap_pauli_2q(swapped) == tuple(pauli)
    expected = [swap @ k @ swap for k in pauli_kraus(pauli)]
    assert np.allclose(_superop(pauli_kraus(swapped)), _superop(expected), atol=ATOL)


@given(length=st.integers(0, 300))
def test_pauli_arity_accepts_only_full_pauli_vectors(length: int) -> None:
    if length in (3, 15, 63, 255):
        assert 4 ** metrics.pauli_arity(length) - 1 == length
    else:
        with pytest.raises(ValueError):
            metrics.pauli_arity(length)


@given(n=ARITIES, data=st.data(), a=FACTORS, b=FACTORS)
def test_scaling_an_avg_infidelity_composes_and_stays_in_range(
    n: int, data: st.DataObject, a: float, b: float
) -> None:
    r = data.draw(avg_infidelities(n))
    assume(metrics.unscalable("avg_infidelity", r, n) is None)
    scaled = metrics.scale_avg_infidelity(r, n, a)
    assert 0.0 <= scaled <= metrics.max_avg_infidelity(n)
    assert metrics.scale_avg_infidelity(r, n, 1.0) == pytest.approx(r, rel=1e-12, abs=1e-15)
    assert metrics.scale_avg_infidelity(r, n, 0.0) == 0.0
    # A scaled value can round to full depolarization, which never scales again.
    assume(metrics.unscalable("avg_infidelity", scaled, n) is None)
    twice = metrics.scale_avg_infidelity(scaled, n, b)
    assert twice == pytest.approx(metrics.scale_avg_infidelity(r, n, a * b), rel=1e-9, abs=1e-12)


def test_full_depolarization_does_not_scale() -> None:
    assert metrics.scale_avg_infidelity(0.5, 1, 0.0) == 0.5


@given(n=ARITIES, data=st.data(), k=st.integers(1, 4))
def test_scaling_an_avg_infidelity_by_k_repeats_the_depolarizing_channel(
    n: int, data: st.DataObject, k: int
) -> None:
    r = data.draw(avg_infidelities(n))
    assume(metrics.unscalable("avg_infidelity", r, n) is None)
    one = _superop(depolarizing_kraus(r, n))
    repeated = np.linalg.matrix_power(one, k)
    assert metrics.scale_avg_infidelity(r, n, k) == pytest.approx(_infidelity(repeated), abs=ATOL)


@given(case=arity_and_pauli(max_total=0.3), k=st.integers(1, 3))
def test_scaling_a_pauli_vector_by_k_repeats_its_channel(
    case: tuple[int, list[float]], k: int
) -> None:
    n, pauli = case
    assume(metrics.unscalable("pauli", pauli, n) is None)
    spec = ChannelSpec("pauli", tuple(range(n)), tuple(pauli_kraus(pauli)))
    expected = pauli_twirl([spec] * k, range(n))
    assert np.allclose(metrics.scale_pauli(pauli, k), expected, atol=ATOL)


@given(case=arity_and_pauli(max_total=0.3), a=FACTORS)
def test_a_scaled_pauli_vector_is_a_valid_pauli_vector(
    case: tuple[int, list[float]], a: float
) -> None:
    n, pauli = case
    scaled = metrics.scale_pauli(pauli, a)
    metrics.check_metric("pauli", scaled, n)
    if metrics.unscalable("pauli", pauli, n) is not None:
        assert scaled == tuple(pauli)


@given(a=UNIT, b=UNIT, factor=FACTORS)
def test_a_scaled_readout_pair_gives_a_column_stochastic_matrix(
    a: float, b: float, factor: float
) -> None:
    scaled = metrics.scale_readout((a, b), factor)
    _assert_readout_matrix(readout_matrix(_qubit(readout=scaled)), scaled)
    if metrics.readout_unscalable((a, b)) is not None:
        assert scaled == (a, b)


@given(a=UNIT, b=UNIT, k=st.integers(0, 4))
def test_scaling_a_readout_pair_by_k_repeats_the_confusion_matrix(
    a: float, b: float, k: int
) -> None:
    assume(metrics.readout_unscalable((a, b)) is None)
    once = readout_matrix(_qubit(readout=(a, b)))
    scaled = readout_matrix(_qubit(readout=metrics.scale_readout((a, b), k)))
    assert once is not None and scaled is not None
    assert np.allclose(scaled, np.linalg.matrix_power(once, k), atol=1e-12)


def _assert_readout_matrix(matrix: np.ndarray | None, pair: tuple[float, float]) -> None:
    """Column j is the distribution of the measured bit for prepared state j."""
    a, b = pair
    assert matrix is not None
    assert (matrix >= 0).all()
    assert np.allclose(matrix.sum(axis=0), 1.0, atol=1e-12)
    assert np.allclose(matrix[:, 0], [1.0 - a, a], atol=1e-15)
    assert np.allclose(matrix[:, 1], [b, 1.0 - b], atol=1e-15)


# channels -----------------------------------------------------------------------------------

TIMES_NS = st.one_of(st.none(), st.floats(1.0, 1e7))
RATES = st.one_of(st.none(), st.floats(0.0, 1e9))
SWAP = np.eye(4)[[0, 2, 1, 3]]


def _qubit(readout: tuple[float, float] | None = None) -> QubitNoise:
    return QubitNoise(0, None, None, None, readout, None, False)


def _relaxation_factors(
    t1: float | None, t2: float | None, rate: float | None, duration: float
) -> tuple[float, float]:
    """Closed form: the excited population decays by exp(-t/T1), the coherence by exp(-t/T2).

    T2 is at most 2 T1, a missing T2 is 2 T1, and a dephasing rate multiplies the coherence by
    1 - 2 p, with p = rate * t capped at 1/2.
    """
    if t2 is None:
        t2 = None if t1 is None else 2.0 * t1
    elif t1 is not None:
        t2 = min(t2, 2.0 * t1)
    population = 1.0 if t1 is None else math.exp(-duration / t1)
    coherence = 1.0 if t2 is None else math.exp(-duration / t2)
    flip = min((rate or 0.0) * duration * 1e-9, 0.5)
    return population, coherence * (1.0 - 2.0 * flip)


@st.composite
def density_matrices(draw: st.DrawFn, num_qubits: int) -> np.ndarray:
    """A random state with complex coherences, so a transposed convention gives other results."""
    d = 2**num_qubits
    parts = st.floats(-1.0, 1.0)
    entries = draw(st.lists(parts, min_size=2 * d * d, max_size=2 * d * d))
    a = np.array(entries[: d * d]).reshape(d, d) + 1j * np.array(entries[d * d :]).reshape(d, d)
    rho = a @ a.conj().T + 1e-3 * np.eye(d)
    return rho / np.trace(rho)


@given(
    t1=TIMES_NS,
    t2=TIMES_NS,
    duration=st.floats(0.0, 1e7),
    rate=RATES,
    rho=density_matrices(1),
)
def test_thermal_relaxation_is_cptp_and_matches_the_closed_form(
    t1: float | None, t2: float | None, duration: float, rate: float | None, rho: np.ndarray
) -> None:
    kraus = thermal_relaxation_kraus(t1, t2, duration, rate)
    _assert_cptp(kraus)
    assert _choi_is_positive(_superop(kraus))

    population, coherence = _relaxation_factors(t1, t2, rate, duration)
    out = sum(k @ rho @ k.conj().T for k in kraus)
    assert out[1, 1].real == pytest.approx(population * rho[1, 1].real, rel=1e-9, abs=1e-15)
    expected = coherence * rho[0, 1]
    assert abs(out[0, 1] - expected) <= 1e-12 * abs(rho[0, 1]) + 1e-6 * abs(expected)


def test_relaxation_keeps_a_small_coherence_long_after_t1() -> None:
    kraus = thermal_relaxation_kraus(1.0, None, 38.0)
    out = sum(k @ np.full((2, 2), 0.5) @ k.conj().T for k in kraus)
    assert out[0, 1].real == pytest.approx(0.5 * math.exp(-19.0), rel=1e-9)


@given(t1=st.floats(1.0, 1e6), t2=TIMES_NS, short=st.floats(0.0, 1e6), extra=st.floats(0.0, 1e6))
def test_relaxation_error_grows_with_the_duration(
    t1: float, t2: float | None, short: float, extra: float
) -> None:
    before = average_gate_fidelity(thermal_relaxation_kraus(t1, t2, short))
    after = average_gate_fidelity(thermal_relaxation_kraus(t1, t2, short + extra))
    assert after <= before + 1e-12


@given(n=ARITIES, data=st.data())
def test_depolarizing_channel_is_cptp_with_the_stated_infidelity(
    n: int, data: st.DataObject
) -> None:
    r = data.draw(avg_infidelities(n))
    kraus = depolarizing_kraus(r, n)
    _assert_cptp(kraus)
    assert _infidelity(_superop(kraus)) == pytest.approx(r, abs=ATOL)
    assert 1.0 - average_gate_fidelity(kraus) == pytest.approx(r, abs=ATOL)


@given(case=arity_and_pauli())
def test_twirling_a_pauli_channel_gives_back_its_pauli_vector(
    case: tuple[int, list[float]],
) -> None:
    n, pauli = case
    spec = ChannelSpec("pauli", tuple(range(n)), tuple(pauli_kraus(pauli)))
    assert np.allclose(pauli_twirl([spec], range(n)), pauli, atol=ATOL)


@st.composite
def two_qubit_profiles(draw: st.DrawFn) -> Profile:
    """Two connected qubits. Qubit 1 can have its own T1, T2 and dephasing rate."""

    def idle() -> dict[str, float]:
        t1_us = draw(st.floats(1.0, 1000.0))
        values = {"t1_us": t1_us}
        if draw(st.booleans()):
            values["t2_us"] = draw(st.floats(0.5, 2.5)) * t1_us
        if draw(st.booleans()):
            values["dephasing_rate_per_s"] = draw(st.floats(0.0, 1e4))
        return values

    def calibrated(max_error: float, max_duration: float) -> dict[str, float]:
        return {
            "avg_infidelity": draw(st.floats(0.0, max_error)),
            "duration_ns": draw(st.floats(0.0, max_duration)),
        }

    qubits = [{"index": 1, **idle()}] if draw(st.booleans()) else []
    data = toy(
        device={"name": "toy", "vendor": "test", "technology": "superconducting", "num_qubits": 2},
        connectivity={"edges": [[0, 1]]},
        idle=idle(),
        readout={"p1_given_0": draw(UNIT), "p0_given_1": draw(UNIT)},
        qubits=qubits,
        gates={
            "rz": {"virtual": True},
            "sx": calibrated(0.3, 2000.0),
            "cz": calibrated(0.7, 20000.0),
            "ecr": {"pauli": draw(pauli_vectors(2)), "duration_ns": 500},
        },
    )
    return Profile.model_validate(data)


def _built(profile: Profile, name: str, wires: tuple[int, ...]) -> GateChannels:
    table = profile.table
    found = table.gate(name, wires)
    assert isinstance(found, GateNoise)
    return gate_channels(found, [table.qubit(q) for q in wires])


def _reachable(qubits: list[QubitNoise], duration: float) -> tuple[float, float]:
    """Average infidelity of relaxation alone, and of full depolarizing noise then relaxation.

    The process fidelity of one relaxing qubit is (1 + 2 c + p) / 4, with p the population and c
    the coherence factor. Depolarizing strength lambda moves the process fidelity linearly to
    1/d^2, because relaxation of the maximally mixed state keeps the trace.
    """
    d = 2 ** len(qubits)
    f_pro = 1.0
    for q in qubits:
        p, c = _relaxation_factors(q.t1_ns, q.t2_ns, q.dephasing_rate_per_s, duration)
        f_pro *= (1.0 + 2.0 * c + p) / 4.0
    lam = metrics.max_depolarizing_param(len(qubits))
    f_full = (1.0 - lam) * f_pro + lam / d**2

    def infidelity(f: float) -> float:
        return 1.0 - (d * f + 1.0) / (d + 1.0)

    return infidelity(f_pro), infidelity(f_full)


@given(
    profile=two_qubit_profiles(),
    gate=st.sampled_from([("sx", (0,)), ("sx", (1,)), ("cz", (0, 1)), ("cz", (1, 0))]),
)
def test_gate_channels_are_cptp_and_match_the_stated_error_when_reachable(
    profile: Profile, gate: tuple[str, tuple[int, ...]]
) -> None:
    name, wires = gate
    built = _built(profile, name, wires)
    for channel in built.channels:
        _assert_cptp(channel.kraus)
    total = superoperator(built.channels, wires)
    assert _choi_is_positive(total)
    vec_identity = np.eye(2 ** len(wires)).flatten(order="F")
    assert np.allclose(vec_identity @ total, vec_identity, atol=ATOL)
    assert built.achieved == pytest.approx(_infidelity(total), abs=ATOL)
    assert metrics.avg_from_pauli(pauli_twirl(built.channels, wires)) == pytest.approx(
        built.achieved, abs=ATOL
    )

    requested = profile.gates[name].avg_infidelity
    assert requested is not None and built.requested == requested
    qubits = [profile.table.qubit(q) for q in wires]
    low, high = _reachable(qubits, built.gate.duration_ns or 0.0)
    assert built.relaxation == pytest.approx(low, abs=ATOL)
    if low + 1e-9 < requested < high - 1e-9:
        assert not built.inexact
        assert built.achieved == pytest.approx(requested, abs=1e-12)
    elif requested < low - 1e-9 or requested > high + 1e-9:
        assert built.inexact
        assert built.achieved == pytest.approx(low if requested < low else high, abs=ATOL)


def _on_wires(
    kraus: np.ndarray, channel_wires: tuple[int, ...], wires: tuple[int, ...]
) -> np.ndarray:
    """``kraus`` as an operator on two qubits ordered as ``wires``, big-endian."""
    if len(channel_wires) == 2:
        return kraus if channel_wires == wires else SWAP @ kraus @ SWAP
    eye = np.eye(2)
    return np.kron(kraus, eye) if channel_wires[0] == wires[0] else np.kron(eye, kraus)


@given(
    profile=two_qubit_profiles(),
    names=st.lists(st.sampled_from(["sx", "cz", "ecr"]), min_size=1, max_size=3),
    reverse=st.booleans(),
    rho=density_matrices(2),
)
def test_superoperator_acts_like_its_kraus_operators(
    profile: Profile, names: list[str], reverse: bool, rho: np.ndarray
) -> None:
    wires = (1, 0) if reverse else (0, 1)
    channels: list[ChannelSpec] = []
    for i, name in enumerate(names):
        gate_wires = (i % 2,) if name == "sx" else (wires if i % 2 else wires[::-1])
        channels.extend(_built(profile, name, gate_wires).channels)
    expected = rho
    for channel in channels:
        ops = [_on_wires(k, channel.wires, wires) for k in channel.kraus]
        expected = sum(k @ expected @ k.conj().T for k in ops)
    out = superoperator(channels, wires) @ rho.flatten(order="F")
    assert np.allclose(out.reshape(4, 4, order="F"), expected, atol=ATOL)


@given(profile=two_qubit_profiles(), qubit=st.sampled_from([0, 1]))
def test_profile_readout_matrices_are_column_stochastic(profile: Profile, qubit: int) -> None:
    noise = profile.table.qubit(qubit)
    assert noise.readout is not None
    _assert_readout_matrix(readout_matrix(noise), noise.readout)
