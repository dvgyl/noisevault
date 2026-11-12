from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median

import numpy as np
import pytest

import noisevault as nv
from noisevault.compare import NoEstimate
from noisevault.counts import PlannedCircuit, simulate
from noisevault.errors import SourceUnavailable
from noisevault.profile import Profile
from noisevault.reference import Op, probabilities
from noisevault.suite import role, suite
from study import models

RUN_AT = datetime(2026, 4, 16, 9, 30, tzinfo=UTC)
NATIVES = ("rz", "sx", "x", "cz")


def device(num_qubits: int = 3, **changes: float) -> Profile:
    """A line of qubits with the gates of an IBM device and the same values on every qubit."""
    values = {
        "one_qubit_error": 4e-3,
        "two_qubit_error": 3e-2,
        "readout_error": 3e-2,
        "t1_us": 100.0,
        "t2_us": 80.0,
        "one_qubit_ns": 30.0,
        "two_qubit_ns": 60.0,
    } | changes
    line = [(i, i + 1) for i in range(num_qubits - 1)]
    profile = Profile.uniform(
        "line", technology="superconducting", num_qubits=num_qubits, connectivity=line, **values
    )
    data = profile.to_dict()
    data["gates"] = {k: v for k, v in data["gates"].items() if k in NATIVES}
    return Profile.model_validate(data)


def uneven(num_qubits: int = 3) -> Profile:
    """``device`` with other values on each qubit and each pair."""
    data = device(num_qubits).to_dict()
    data["qubits"] = [
        {
            "index": q,
            "t1_us": 60.0 + 30 * q,
            "t2_us": 50.0 + 20 * q,
            "readout": {"p1_given_0": 0.01 + 0.01 * q, "p0_given_1": 0.05 - 0.01 * q},
        }
        for q in range(num_qubits)
    ]
    data["calibrations"] = [
        {"gate": "cz", "qubits": [q, q + 1], "avg_infidelity": 0.02 + 0.01 * q}
        for q in range(num_qubits - 1)
    ] + [{"gate": "sx", "qubits": [q], "avg_infidelity": 1e-3 * (q + 1)} for q in range(num_qubits)]
    return Profile.model_validate(data)


def _readouts(profile: Profile) -> list[tuple[float, float]]:
    table = profile.table
    return [table.qubit(q).readout for q in range(table.num_qubits)]


def _with_readouts(profile: Profile, pairs: list[tuple[float, float]]) -> Profile:
    data = profile.to_dict()
    data["qubits"] = [
        {"index": q, "readout": {"p1_given_0": a, "p0_given_1": b}}
        for q, (a, b) in enumerate(pairs)
    ]
    return Profile.model_validate(data)


def test_floored_replaces_only_exact_zeros_by_half_the_smallest_value() -> None:
    profile = _with_readouts(device(), [(0.0, 0.04), (0.006, 0.0), (0.02, 0.03)])

    lifted = models.floored(profile)

    assert models.zero_readouts(profile) == 2
    assert models.zero_readouts(lifted) == 0
    assert _readouts(lifted) == [(0.003, 0.04), (0.006, 0.003), (0.02, 0.03)]
    assert models.floored(lifted) is lifted


def test_floored_keeps_the_factors_of_the_profile() -> None:
    profile = _with_readouts(device(), [(0.0, 0.04), (0.006, 0.01), (0.02, 0.03)])
    scaled = models.with_factors(profile, 2.0, 1.5)

    lifted = models.floored(scaled)

    assert lifted.unmodeled_error == scaled.unmodeled_error
    assert lifted.uncorrected().table.qubit(0).readout == (0.003, 0.04)


def test_ideal_gives_each_deep_circuit_one_result() -> None:
    profile = uneven()
    ideal = models.ideal(profile)
    for circuit in suite(profile):
        if circuit.name.startswith("deep_"):
            probs = probabilities(ideal, circuit.ops, 3, layout=circuit.qubits)
            assert probs.max() == pytest.approx(1.0, abs=1e-12)


def test_generic_is_the_same_for_every_device() -> None:
    a, b = models.generic(uneven()), models.generic(device(t1_us=20.0))

    assert a.table.gate("cz", (0, 1)).avg_infidelity == b.table.gate("cz", (1, 2)).avg_infidelity
    assert a.table.gate("sx", (2,)).spec.depolarizing_param == models.GENERIC_1Q
    assert a.table.gate("cz", (1, 2)).spec.depolarizing_param == models.GENERIC_2Q
    assert _readouts(a) == [(models.GENERIC_READOUT, models.GENERIC_READOUT)] * 3
    assert a.table.qubit(1).relaxation_unknown


def test_medians_gives_every_qubit_and_every_pair_the_same_values() -> None:
    profile = uneven(4)

    found = models.medians(profile)

    table = found.table
    assert {table.qubit(q).t1_ns for q in range(4)} == {median([60.0, 90.0, 120.0, 150.0]) * 1000}
    assert set(_readouts(found)) == {(0.025, 0.035)}
    assert {table.gate("cz", p).avg_infidelity for p in [(0, 1), (1, 2), (2, 3)]} == {0.03}
    assert {table.gate("sx", (q,)).avg_infidelity for q in range(4)} == {0.0025}
    assert found.calibration_fingerprint != profile.calibration_fingerprint


def test_medians_keeps_a_virtual_default_and_scales_only_calibrated_records() -> None:
    data = uneven().to_dict()
    data["calibrations"] += [
        {"gate": "rz", "qubits": [0], "virtual": False, "avg_infidelity": 1e-4},
        {"gate": "rz", "qubits": [1], "virtual": False, "avg_infidelity": 3e-4},
    ]
    profile = Profile.model_validate(data)

    found = models.medians(profile)

    assert found.gates["rz"].virtual
    assert found.table.gate("rz", (2,)).state == "ideal"
    errors = [found.table.gate("rz", (q,)).avg_infidelity for q in (0, 1)]
    assert errors == pytest.approx([2e-4, 2e-4])


def test_symmetric_keeps_the_mean_of_the_two_readout_errors() -> None:
    profile = uneven()

    found = models.symmetric(profile)

    for (a, b), (c, d) in zip(_readouts(profile), _readouts(found), strict=True):
        assert c == d == pytest.approx((a + b) / 2)
    assert found.table.qubit(1).t1_ns == profile.table.qubit(1).t1_ns


def test_readout_measured_recovers_the_readout_errors() -> None:
    truth = _with_readouts(device(t1_us=1e6, t2_us=1e6), [(0.01, 0.06), (0.04, 0.02), (0.1, 0.1)])
    shots = 20_000
    counts = simulate(truth, suite(truth), shots=shots, seed=3, run_at=RUN_AT)

    found = models.readout_measured(device(), counts.select(["readout", "readout_1"]))

    for (a, b), (x, y) in zip(_readouts(truth), _readouts(found), strict=True):
        # readout_1 adds the error of one x gate to P(0|1).
        assert abs(x - a) < 4 * math.sqrt(a * (1 - a) / shots)
        assert abs(y - b - 2e-3) < 4 * math.sqrt(b * (1 - b) / shots) + 2e-3
    assert found.table.qubit(0).readout != found.table.qubit(1).readout


def test_a_measured_readout_is_never_0() -> None:
    truth = _with_readouts(device(), [(0.0, 0.0)] * 3)
    counts = simulate(truth, suite(truth), shots=200, seed=1, run_at=RUN_AT)

    found = models.readout_measured(truth, counts)

    assert min(min(pair) for pair in _readouts(found)) >= 0.5 / 201


def test_gate_factor_recovers_a_known_factor() -> None:
    profile = device()
    truth = models.with_factors(profile, 1.7)
    names = [
        c.name for c in suite(profile) if c.name.startswith("deep_") and role(c.name) == "train"
    ]
    chosen = [c for c in suite(profile) if c.name in names]
    counts = simulate(truth, chosen, shots=2000, seed=5, run_at=RUN_AT)

    found = models.gate_factor(profile, counts, names)

    assert found == pytest.approx(1.7, rel=0.08)


def test_fit_factor_finds_the_maximum_in_few_simulations() -> None:
    profile = device(2)
    circuit = next(c for c in suite(profile) if c.name.startswith("qubit0"))
    counts = simulate(models.with_factors(profile, 3.0), [circuit], shots=4000, seed=1)
    calls: list[float] = []

    def build(factor: float) -> Profile:
        calls.append(factor)
        return models.with_factors(profile, factor)

    found = models.fit_factor(build, counts, [circuit.name])

    assert len(calls) <= 30
    assert min(calls) >= 0.05 and max(calls) <= 20
    assert found == pytest.approx(3.0, rel=0.25)
    assert models.fit_factor(build, counts, [circuit.name]) == found


def test_fit_factor_finds_the_maximum_past_a_flat_likelihood_below_the_relaxation_floor() -> None:
    profile = Profile.uniform(
        "one",
        technology="superconducting",
        num_qubits=1,
        one_qubit_error=0.001,
        readout_error=0.01,
        t1_us=10.0,
        t2_us=10.0,
        one_qubit_ns=100.0,
    )
    circuit = PlannedCircuit(name="xs", qubits=(0,), ops=tuple(Op("x", (0,)) for _ in range(20)))
    counts = simulate(models.with_factors(profile, 10.0), [circuit], shots=100_000, seed=1)

    found = models.gate_factor(profile, counts, ["xs"])

    assert found == pytest.approx(10.0, rel=0.2)


def test_fit_factor_takes_the_grid_point_nearest_1_when_the_likelihood_is_flat() -> None:
    profile = device(1)
    circuit = next(c for c in suite(profile) if c.name.startswith("qubit0"))
    counts = simulate(profile, [circuit], shots=10, seed=1)

    found = models.fit_factor(lambda f: profile, counts, [circuit.name])

    assert found == pytest.approx(1.0, rel=0.05)


def _all_loci(profile: Profile) -> dict[tuple[int, ...], float]:
    return {(q,): 1.0 for q in range(profile.device.num_qubits)} | {
        tuple(edge): 1.0 for edge in profile.connectivity.edges
    }


@pytest.mark.parametrize("factor", [0.3, 2.5])
def test_scaling_every_locus_equals_the_unmodeled_gate_factor(factor: float) -> None:
    data = uneven().to_dict()
    # Other metric kinds, a relaxation floor above the stated error, and a pauli error.
    data["calibrations"] += [
        {"gate": "x", "qubits": [0], "process_infidelity": 2e-3},
        {"gate": "x", "qubits": [1], "avg_infidelity": 1e-5},
        {"gate": "x", "qubits": [2], "pauli": [1e-3, 2e-3, 3e-3]},
        {"gate": "cz", "qubits": [2, 1], "duration_ns": 90.0},
    ]
    profile = Profile.model_validate(data)
    circuit = [
        Op("sx", (0,)),
        Op("x", (0,)),
        Op("x", (1,)),
        Op("x", (2,)),
        Op("cz", (0, 1)),
        Op("sx", (2,)),
        Op("cz", (2, 1)),
        Op("rz", (1,), (0.4,)),
        Op("sx", (1,)),
        Op("delay", (0,), (500.0,)),
        Op("cz", (1, 0)),
        Op("x", (2,)),
    ]
    loci = {locus: factor for locus in _all_loci(profile)}

    scaled = models.scale_loci(profile, loci)
    expected = models.with_factors(profile, factor)

    for ops in (circuit, next(c.ops for c in suite(profile) if c.name.startswith("deep_"))):
        a = probabilities(scaled, ops, 3, unknown_gates="error")
        b = probabilities(expected, ops, 3, unknown_gates="error")
        assert np.max(np.abs(a - b)) < 1e-9
    assert scaled.unmodeled_error is None


def test_scale_loci_changes_only_the_named_loci() -> None:
    profile = uneven()

    scaled = models.scale_loci(profile, {(0, 1): 2.0, (2,): 3.0})

    before, after = profile.table, scaled.table
    assert after.gate("cz", (1, 0)).avg_infidelity > before.gate("cz", (1, 0)).avg_infidelity
    assert after.gate("cz", (1, 2)).avg_infidelity == before.gate("cz", (1, 2)).avg_infidelity
    assert after.gate("sx", (2,)).avg_infidelity > before.gate("sx", (2,)).avg_infidelity
    assert after.gate("sx", (0,)).avg_infidelity == before.gate("sx", (0,)).avg_infidelity


def test_locus_recovers_a_factor_that_differs_between_two_pairs() -> None:
    profile = device()
    truth = models.scale_loci(profile, {(0, 1): 3.0, (1, 2): 1.0})
    circuits = [c for c in suite(profile) if role(c.name) == "train"]
    counts = simulate(truth, circuits, shots=3000, seed=2, run_at=RUN_AT)

    _, factors, rows = models.locus(profile, counts)

    assert factors[(0, 1)] == pytest.approx(3.0, rel=0.25)
    assert factors[(1, 2)] == pytest.approx(1.0, rel=0.25)
    assert all(f == pytest.approx(1.0, rel=0.3) for q, f in factors.items() if len(q) == 1)
    assert [row["qubits"] for row in rows["pairs"]] == [[0, 1], [1, 2]]


def _ibm(profile: Profile, when: datetime, **readout: float) -> Profile:
    data = profile.to_dict()
    data["device"] |= {"name": "ibm_line", "vendor": "ibm", "calibrated_at": when.isoformat()}
    if readout:
        data["readout"] = readout
    return Profile.model_validate(data)


class FakeSource:
    """IBM's public endpoint with a fixed list of calibrations."""

    def __init__(self, calibrations: list[Profile]) -> None:
        self.calibrations = calibrations
        self.calls: list[datetime] = []

    def __call__(self, device: str, at: datetime) -> Profile:
        self.calls.append(at)
        older = [p for p in self.calibrations if p.device.calibrated_at < at]
        if not older:
            raise SourceUnavailable(f"IBM's public endpoint has no {device} calibration older than")
        return max(older, key=lambda p: p.device.calibrated_at)


def test_stale_pulls_the_calibration_of_that_time_and_caches_it(tmp_path: Path) -> None:
    now = _ibm(device(), RUN_AT - timedelta(hours=30))
    week = _ibm(device(), RUN_AT - timedelta(days=9), error=0.05)
    source = FakeSource([week, now])

    first = models.stale(now, RUN_AT, 7, tmp_path, source)
    again = models.stale(now, RUN_AT, 7, tmp_path, lambda device, at: pytest.fail("no cache"))

    assert first is not None and again is not None
    assert first.calibration_fingerprint == week.calibration_fingerprint
    assert again.calibration_fingerprint == week.calibration_fingerprint
    assert source.calls == [RUN_AT - timedelta(days=7)]
    assert models.stale(now, RUN_AT, 1, tmp_path, source) is None
    assert models.stale(now, RUN_AT, 30, tmp_path, source) is None
    assert models.stale(now, RUN_AT, 30, tmp_path, lambda d, a: pytest.fail("no cache")) is None


def test_stale_gives_none_for_another_vendor_and_raises_when_the_source_fails() -> None:
    def broken(device: str, at: datetime) -> Profile:
        raise SourceUnavailable("could not reach IBM's public endpoint (offline)")

    assert models.stale(device(), RUN_AT, 7, None, broken) is None
    with pytest.raises(SourceUnavailable):
        models.stale(_ibm(device(), RUN_AT), RUN_AT, 7, None, broken)


def test_bundled_gives_the_bundled_profile_unless_it_is_c() -> None:
    kingston = nv.load("ibm_kingston@2026-04-15")
    other = kingston.model_copy(
        update={"device": {**kingston.device.model_dump(mode="json"), "calibrated_at": RUN_AT}}
    )
    other = other.model_copy(update={"readout": {"error": 0.02}})

    assert models.bundled(kingston) is None
    found = models.bundled(other)
    assert found is not None
    assert found.calibration_fingerprint == kingston.calibration_fingerprint
    assert models.bundled(device()) is None


def test_two_factor_uses_1_for_a_factor_that_the_fit_does_not_identify() -> None:
    profile = device(2, readout_error=0.0)
    truth = models.with_factors(profile, 2.0)
    circuits = [c for c in suite(profile) if c.name.startswith("deep_") and c.name.endswith("s0")]
    counts = simulate(truth, circuits, shots=1000, seed=1, run_at=RUN_AT)

    fitted, comparison = models.two_factor(profile, counts)

    assert isinstance(comparison.readout, NoEstimate)
    unmodeled = fitted.unmodeled_error
    assert unmodeled is not None and unmodeled.readout is None
    assert unmodeled.gates is not None and unmodeled.gates.factor == pytest.approx(2.0, rel=0.3)


def test_averaged_takes_the_median_of_each_value_over_the_calibrations() -> None:
    def with_values(t1: float, p10: float, cz: float) -> Profile:
        data = Profile.uniform(
            "pair",
            technology="superconducting",
            num_qubits=2,
            one_qubit_error=1e-3,
            two_qubit_error=1e-2,
            readout_error=1e-2,
            t1_us=100.0,
            t2_us=80.0,
        ).to_dict()
        data["gates"] = {k: v for k, v in data["gates"].items() if k in ("sx", "cz")}
        data["qubits"] = [
            {"index": 0, "t1_us": t1, "readout": {"p1_given_0": p10, "p0_given_1": 0.03}}
        ]
        data["calibrations"] = [{"gate": "cz", "qubits": [0, 1], "avg_infidelity": cz}]
        return Profile.model_validate(data)

    newest = with_values(50.0, 0.0, 0.004)
    others = [with_values(200.0, 0.02, 0.010), with_values(120.0, 0.01, 0.020)]

    found = models.averaged(newest, [newest, *others])

    qubit = found.table.qubit(0)
    assert qubit.t1_ns == pytest.approx(120_000.0)
    assert qubit.readout == pytest.approx((0.01, 0.03))
    assert found.table.gate("cz", (0, 1)).avg_infidelity == pytest.approx(0.010)
    assert found.table.qubit(1).t1_ns == newest.table.qubit(1).t1_ns
    assert models.averaged(newest, []).fingerprint == newest.fingerprint
    twice = models.averaged(newest, [newest, others[0], others[0]])
    assert twice.table.gate("cz", (0, 1)).avg_infidelity == pytest.approx(0.010)


def test_averaged_takes_the_values_from_the_given_calibrations_only() -> None:
    def with_values(t1: float, cz: float) -> Profile:
        data = uneven(2).to_dict()
        data["qubits"][0]["t1_us"] = t1
        data["calibrations"] = [{"gate": "cz", "qubits": [0, 1], "avg_infidelity": cz}]
        return Profile.model_validate(data)

    structure = with_values(50.0, 0.004)
    week = [with_values(200.0, 0.010), with_values(120.0, 0.020)]

    found = models.averaged(structure, week)

    assert found.table.qubit(0).t1_ns == pytest.approx(160_000.0)
    assert found.table.gate("cz", (0, 1)).avg_infidelity == pytest.approx(0.015)
    assert (
        found.table.gate("cz", (0, 1)).duration_ns == structure.table.gate("cz", (0, 1)).duration_ns
    )


def test_pulled_calls_the_public_source_with_the_time_as_a_keyword(monkeypatch, tmp_path) -> None:
    asked = []

    def public(device: str, *, at: datetime | None = None) -> Profile:
        asked.append((device, at))
        raise SourceUnavailable("IBM's public endpoint has no ibm_x calibration older than this")

    monkeypatch.setattr("noisevault.sources.ibm_public.pull", public)
    when = datetime(2026, 4, 16, tzinfo=UTC)

    assert models.pulled("ibm_x", when, tmp_path / "cache") is None
    assert asked == [("ibm_x", when)]
