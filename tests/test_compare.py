from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from itertools import cycle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import noisevault as nv
from noisevault import compare as fit
from noisevault.compare import CircuitScore, Comparison, NoEstimate, compare
from noisevault.counts import MeasuredCounts, PlannedCircuit, load_counts, plan, simulate
from noisevault.errors import CountsError, NoiseVaultError, UnsupportedEffect
from noisevault.metrics import scale_readout
from noisevault.profile import ErrorFactor, Profile
from noisevault.reference import Op
from noisevault.reference import probabilities as reference

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "kingston-simulated.counts.json"
KINGSTON = "ibm_kingston@2026-04-15"
TRUE_GATE, TRUE_READOUT = 1.8, 1.3
EXAMPLE_SEED = 1
RUN_AT = datetime(2026, 4, 16, 9, 30, tzinfo=UTC)
LATER = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
slow = pytest.mark.skipif(
    os.environ.get("NOISEVAULT_SLOW") != "1", reason="set NOISEVAULT_SLOW=1 to run (3 minutes)"
)


def scaled(profile: Profile, gate: float | None = None, readout: float | None = None) -> Profile:
    factors = {k: {"factor": v} for k, v in (("gates", gate), ("readout", readout)) if v}
    return profile.model_copy(update={"unmodeled_error": factors})


def example_counts() -> MeasuredCounts:
    kingston = nv.load(KINGSTON)
    truth = scaled(kingston, TRUE_GATE, TRUE_READOUT)
    return simulate(truth, plan(kingston), shots=4000, seed=EXAMPLE_SEED, run_at=RUN_AT)


def written(
    profile: Profile, circuits: Sequence[tuple[str, Sequence[Op], dict[str, int]]]
) -> MeasuredCounts:
    return MeasuredCounts.model_validate(
        {
            "nv_counts": "1.0",
            "source": "simulated",
            "profile": {"id": profile.id, "fingerprint": profile.calibration_fingerprint},
            "backend": profile.device.name,
            "run_at": LATER.isoformat(),
            "bit_order": "clbit0_left",
            "execution": {
                "client": "hand-written",
                "transpiled": False,
                "gate_twirling": False,
                "measure_twirling": False,
                "dynamical_decoupling": False,
                "init_qubits": True,
            },
            "circuits": [
                {
                    "name": name,
                    "qubits": list(range(len(next(iter(counts))))),
                    "ops": [[op.name, list(op.qubits), list(op.params)] for op in ops],
                    "shots": sum(counts.values()),
                    "counts": counts,
                }
                for name, ops, counts in circuits
            ],
        }
    )


def rebound(counts: MeasuredCounts, profile: Profile) -> MeasuredCounts:
    binding = {"id": profile.id, "fingerprint": profile.calibration_fingerprint}
    return counts.model_copy(update={"profile": binding})


def moved(counts: MeasuredCounts, circuit: str, source: str, target: str, n: int) -> MeasuredCounts:
    data = counts.to_dict()
    for entry in data["circuits"]:
        if entry["name"] == circuit:
            entry["counts"][source] -= n
            entry["counts"][target] = entry["counts"].get(target, 0) + n
    return MeasuredCounts.model_validate(data)


def covers(result: ErrorFactor | NoEstimate, truth: float) -> bool:
    if not isinstance(result, ErrorFactor):
        return False
    return (result.low or 0) <= truth <= (result.high or math.inf)


def toy() -> Profile:
    return Profile.uniform(
        "toy", technology="superconducting", num_qubits=1, one_qubit_error=0.001, readout_error=0.01
    )


def one_qubit(name: str, gates: dict[str, Any], **sections: Any) -> Profile:
    return Profile.model_validate(
        {
            "noisevault": "1.0",
            "device": {"name": name, "technology": "superconducting", "num_qubits": 1},
            "connectivity": "all_to_all",
            "gates": gates,
            **sections,
        }
    )


XX = (Op("x", (0,)), Op("x", (0,)))


def test_the_example_counts_give_back_the_factors_they_were_simulated_with() -> None:
    result = compare(nv.load(KINGSTON), load_counts(EXAMPLE))
    assert covers(result.gates, TRUE_GATE), result.gates
    assert covers(result.readout, TRUE_READOUT), result.readout
    assert result.gates.high / result.gates.low < 1.8


def test_the_example_counts_file_is_current() -> None:
    expected = example_counts()
    assert load_counts(EXAMPLE).sha256 == expected.sha256, (
        f"{EXAMPLE.relative_to(ROOT)} is stale; regenerate it with python tests/test_compare.py"
    )


@slow
def test_each_interval_covers_its_truth_on_the_plan() -> None:
    kingston = nv.load(KINGSTON)
    truth, circuits = scaled(kingston, TRUE_GATE, TRUE_READOUT), plan(kingston)
    hits = {"gate": 0, "readout": 0}
    for seed in range(1000, 1100):
        result = compare(kingston, simulate(truth, circuits, shots=4000, seed=seed, run_at=RUN_AT))
        hits["gate"] += covers(result.gates, TRUE_GATE)
        hits["readout"] += covers(result.readout, TRUE_READOUT)
    assert 88 <= hits["gate"] <= 100 and 88 <= hits["readout"] <= 100, hits


def sparse_counts(errors: int) -> tuple[Profile, MeasuredCounts]:
    profile = one_qubit("sparse", {"x": {"avg_infidelity": 0.0}}, readout={"error": 0.00055})
    circuits = {c.name: c.ops for c in plan(profile)}
    assert list(circuits) == ["single_qubit", "readout"]
    half = errors // 2
    return profile, written(
        profile,
        [
            ("single_qubit", circuits["single_qubit"], {"0": 2000 - half, "1": half}),
            ("readout", circuits["readout"], {"0": 2000 - errors + half, "1": errors - half}),
        ],
    )


def test_sparse_counts_with_no_error_keep_an_upper_end_the_chi_square_cutoff_misses() -> None:
    result = compare(*sparse_counts(0))
    assert isinstance(result.readout, ErrorFactor) and result.readout.bound == "lower"
    assert result.readout.high > 1.3


def test_sparse_counts_cover_the_true_factor_over_every_error_count() -> None:
    p = scale_readout((0.00055, 0.00055), 1.0)[0]
    coverage = 0.0
    for errors in range(80):
        weight = math.comb(4000, errors) * p**errors * (1 - p) ** (4000 - errors)
        coverage += weight * covers(compare(*sparse_counts(errors)).readout, 1.0)
    assert coverage >= 0.95


def dense(errors: int) -> tuple[Profile, MeasuredCounts]:
    profile = Profile.uniform(
        "dense", technology="superconducting", num_qubits=1, one_qubit_error=0.0, readout_error=0.1
    )
    shots = 1_000_000
    return profile, written(profile, [("readout", (), {"0": shots - errors, "1": errors})])


def test_dense_counts_refine_the_maximum_between_grid_points() -> None:
    result = compare(*dense(100698))
    assert isinstance(result.readout, ErrorFactor)
    assert abs(result.readout.factor - 1.007832) < 1e-4
    assert result.readout.low < result.readout.factor < result.readout.high
    assert result.readout.describe() == "x1.008 (95% interval 1.001 to 1.015)"
    assert result.dof == 0 and result.dispersion == 1


def test_dense_counts_cover_the_true_factor_without_wide_intervals() -> None:
    rng = np.random.default_rng(2024)
    hits = sum(
        covers(compare(*dense(int(rng.binomial(1_000_000, 0.1)))).readout, 1.0) for _ in range(200)
    )
    assert 180 <= hits <= 199


LONG_GATE = 1.13


def long_circuit(seed: int, shots: int = 1_000_000) -> tuple[Profile, MeasuredCounts]:
    profile = Profile.uniform(
        "long", technology="superconducting", num_qubits=1, one_qubit_error=0.001
    )
    circuit = PlannedCircuit(name="x1000", qubits=(0,), ops=(Op("x", (0,)),) * 1000)
    truth = scaled(profile, LONG_GATE)
    return profile, simulate(truth, [circuit], shots=shots, seed=seed, run_at=LATER)


def test_a_long_circuit_gets_nodes_until_its_interval_covers_the_truth() -> None:
    result = compare(*long_circuit(0))
    assert covers(result.gates, LONG_GATE), result.gates


@slow
def test_each_interval_covers_its_truth_on_a_long_circuit() -> None:
    hits = sum(covers(compare(*long_circuit(seed)).gates, LONG_GATE) for seed in range(100))
    assert hits >= 93, hits


def test_nodes_that_one_comparison_adds_leave_the_next_unchanged() -> None:
    first = compare(*long_circuit(0)).to_dict()
    compare(*long_circuit(1, shots=10_000_000))
    assert compare(*long_circuit(0)).to_dict() == first


def two_peaks(
    seed: int,
    shots: int = 1_000_000,
    last: tuple[float, float] = (7 * math.pi / 4, 0.0),
    truth: float = 1.0,
) -> tuple[Profile, MeasuredCounts]:
    profile = one_qubit("peaks", {"r": {"pauli": [0.04875, 0.00375, 0.00125]}})
    angles = [(math.pi / 2, math.pi / 2), (math.pi / 4, math.pi / 2)]
    angles += [(3 * math.pi / 4, math.pi / 2), last]
    circuit = PlannedCircuit(name="r4", qubits=(0,), ops=tuple(Op("r", (0,), a) for a in angles))
    simulated = scaled(profile, truth) if truth != 1 else profile
    return profile, simulate(simulated, [circuit], shots=shots, seed=seed, run_at=LATER)


EQUAL_ENDS = (3.1401960091961363, math.pi / 4)


def test_an_interval_spans_every_range_the_test_accepts() -> None:
    result = compare(*two_peaks(0))
    assert covers(result.gates, 1.0) and result.gates.high > 8, result.gates


def test_of_two_equal_peaks_the_estimate_is_the_one_nearest_factor_1() -> None:
    result = compare(*two_peaks(5))
    assert isinstance(result.gates, ErrorFactor) and result.gates.high > 8
    assert 0.9 < result.gates.factor < 1.1, result.gates


@pytest.mark.parametrize("shots", [1_000_000, 20_000_000])
def test_intervals_with_two_peaks_cover_the_true_factor(shots: int) -> None:
    hits = sum(covers(compare(*two_peaks(seed, shots)).gates, 1.0) for seed in range(100))
    assert hits >= 93, hits


def test_an_interval_holds_a_peak_narrower_than_the_grid() -> None:
    result = compare(*two_peaks(10, 20_000_000))
    assert covers(result.gates, 8.470897) and covers(result.gates, 1.0), result.gates


OTHER_PEAK = (
    (0.6752639144059001, 2.4641956201773256),
    (5.506796082841274, 5.765717338121322),
    (1.4669746608300045, 0.4946132930675121),
    (2.282016992462106, 0),
)


def test_each_drawn_maximum_starts_at_every_peak_the_fit_keeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    found = []
    draw_max = fit._Fit.draw_max

    def recorded(self: Any, draws: np.ndarray, windows: Any = ()) -> Any:
        best = draw_max(self, draws, windows)
        if not windows:
            found.append((self.surface, draws, best.top))
        return best

    monkeypatch.setattr(fit._Fit, "draw_max", recorded)
    profile, counts = two_peaks(0)
    other = tuple(Op("r", (0,), angles) for angles in OTHER_PEAK)
    r4 = ("r4", counts.circuits[0].ops, {"0": 487757, "1": 512243})
    result = compare(profile, written(profile, [r4, ("other", other, {"0": 516284, "1": 483716})]))
    ((surface, draws, top),) = found
    above = surface.loglik(draws).max(axis=(0, 1)) - top
    assert above.max() <= 2 * fit.ASCENT_TOL, (int(above.argmax()), above.max())
    assert result.p_value is not None and result.p_value < 0.01, result.p_value


def zero_error_loops(seed: int) -> tuple[Profile, MeasuredCounts]:
    profile = one_qubit("zloop2", {"r": {"pauli": [0, 0, 0.1]}})
    steps = [(0.01, [1] * 10 + [-1] * 10), (0.015, [1] * 5 + [-1] * 5)]
    circuits = [
        PlannedCircuit(
            name=f"r{len(signs)}",
            qubits=(0,),
            ops=tuple(Op("r", (0,), (angle * sign, math.pi / 2)) for sign in signs),
        )
        for angle, signs in steps
    ]
    run_at = datetime(2026, 10, 1, tzinfo=UTC)
    return profile, simulate(profile, circuits, shots=10_000, seed=seed, run_at=run_at)


def test_an_interval_holds_a_peak_that_only_the_calibrated_test_accepts() -> None:
    profile, counts = zero_error_loops(389)
    assert {c.name: dict(c.counts) for c in counts.circuits} == {
        "r20": {"0": 9990, "1": 10},
        "r10": {"0": 9990, "1": 10},
    }
    result = compare(profile, counts)
    assert covers(result.gates, 0.404558) and covers(result.gates, 1.0), result.gates


def test_a_flat_likelihood_below_the_relaxation_floor_stays_outside_the_interval() -> None:
    profile = Profile.uniform(
        "relaxed",
        technology="superconducting",
        num_qubits=1,
        one_qubit_error=0.0014,
        readout_error=0.025,
        t1_us=80,
        t2_us=150,
        one_qubit_ns=40,
    )
    counts = simulate(
        scaled(profile, 14, 1.5), plan(profile), shots=1_000_000, seed=2, run_at=LATER
    )
    result = compare(profile, counts)
    assert isinstance(result.gates, ErrorFactor) and result.gates.bound is None, result.gates
    assert 13.8 < result.gates.low < 14 < result.gates.high < 14.2, result.gates


def test_a_gate_response_equal_at_both_ends_of_the_range_still_moves() -> None:
    profile, counts = two_peaks(0, last=EQUAL_ENDS, truth=5.0)
    assert dict(counts.circuits[0].counts) == {"0": 529106, "1": 470894}
    result = compare(profile, counts)
    assert covers(result.gates, 5.0), result.gates
    assert result.p_value is None and result.deviance < 1e-3


def test_intervals_cover_a_gate_response_equal_at_both_ends() -> None:
    hits = sum(
        covers(compare(*two_peaks(seed, last=EQUAL_ENDS, truth=5.0)).gates, 5.0)
        for seed in range(100)
    )
    assert hits >= 93, hits


def dense_pair() -> Profile:
    return one_qubit(
        "dense-pair", {"x": {"avg_infidelity": 0.0}}, readout={"p1_given_0": 0.3, "p0_given_1": 0}
    )


def test_the_maximum_follows_the_shots_to_the_exact_factor() -> None:
    profile = dense_pair()
    counts = {"0": 3490151819, "1": 1509848181}
    result = compare(
        profile, written(profile, [("readout1", (), counts), ("readout2", (), counts)])
    )
    assert abs(result.readout.factor - 1.0079) < 1e-5, result.readout
    assert result.deviance < 0.01 and result.dof == 1 and result.p_value >= 0.01
    assert "within shot noise on every circuit" in str(result)


def test_p_values_on_dense_counts_spread_over_the_whole_range() -> None:
    profile, rng, shots = dense_pair(), np.random.default_rng(7), 5_000_000_000
    p_values = []
    for _ in range(40):
        ones = [int(n) for n in rng.binomial(shots, 0.3, size=2)]
        circuits = [(f"readout{i}", (), {"0": shots - n, "1": n}) for i, n in enumerate(ones)]
        p_values.append(compare(profile, written(profile, circuits)).p_value)
    low, high = np.percentile(p_values, [25, 75])
    assert high - low > 0.3, (low, high)


MOST_SHOTS = 10_000_000_000


def test_the_search_follows_a_ridge_to_a_maximum_far_from_the_grid_peak() -> None:
    profile = one_qubit(
        "ridge", {"x": {"avg_infidelity": 0.01}}, readout={"p1_given_0": 0.3, "p0_given_1": 0}
    )
    empty = {"0": 669337, "1": 3330663}
    circuits = [("xx", XX, {"0": 654400, "1": 3345600}), ("e1", (), empty), ("e2", (), empty)]
    result = compare(profile, written(profile, circuits))
    assert result.deviance < 1e-3, result.deviance
    assert result.gates.factor == pytest.approx(1.13002, rel=1e-3), result.gates
    assert "within shot noise on every circuit" in str(result)


def orthogonal() -> Profile:
    return Profile.model_validate(
        {
            "noisevault": "1.0",
            "device": {"name": "orthogonal", "technology": "superconducting", "num_qubits": 2},
            "connectivity": "all_to_all",
            "gates": {"x": {"avg_infidelity": 0.01}},
            "qubits": [{"index": 1, "readout": {"p1_given_0": 0.3, "p0_given_1": 0}}],
        }
    )


def orthogonal_counts(
    circuits: Sequence[tuple[str, Sequence[Op], dict[str, int]]],
) -> MeasuredCounts:
    data = written(orthogonal(), circuits).to_dict()
    for entry in data["circuits"][1:]:
        entry["qubits"] = [1]
    return MeasuredCounts.model_validate(data)


def test_a_held_factor_gets_the_other_factor_refitted_to_the_likelihood_tolerance() -> None:
    profile = orthogonal()
    xx = ("xx", XX, {"0": 9_776_842_646, "1": 223_157_354})
    empty = {"0": 1_673_342_748, "1": 8_326_657_252}
    counts = orthogonal_counts([xx, ("empty1", (), empty), ("empty2", (), empty)])
    gates = compare(profile, counts).gates
    alone = compare(profile, written(profile, [xx])).gates
    half = math.log(alone.high / alone.low) / 2
    for end, expected in ((gates.low, alone.low), (gates.high, alone.high)):
        assert abs(math.log(end / expected)) < 0.05 * half, (gates, alone)


def test_an_interval_narrower_than_any_fixed_step_keeps_its_width() -> None:
    zeros = round(MOST_SHOTS * 0.7**3)
    counts = {"0": zeros, "1": MOST_SHOTS - zeros}
    profile = dense_pair()
    circuits = [(f"readout{i}", (), counts) for i in range(1000)]
    result = compare(profile, written(profile, circuits)).readout
    frequency = zeros / MOST_SHOTS

    def lr(factor: float) -> float:
        zero = 0.7**factor
        terms = zeros * math.log(frequency / zero) + (MOST_SHOTS - zeros) * math.log(
            (1 - frequency) / (1 - zero)
        )
        return 2 * 1000 * terms

    assert 3.6 < lr(result.low) < 5 and 3.6 < lr(result.high) < 5, result


def test_nodes_follow_the_shots_below_any_fixed_gap() -> None:
    profile, ops = one_qubit("gap", {"x": {"avg_infidelity": 0.2}}), (Op("x", (0,)),) * 3

    def probabilities(factor: float) -> np.ndarray:
        return reference(scaled(profile, factor), ops, 1, unknown_gates="error")

    ones = round(MOST_SHOTS * probabilities(1.73663)[1])
    observed = np.array([MOST_SHOTS - ones, ones])
    circuits = [(f"x{i}", ops, {"0": MOST_SHOTS - ones, "1": ones}) for i in range(50)]
    result = compare(profile, written(profile, circuits)).gates

    def lr(factor: float) -> float:
        return 100 * float(
            np.sum(observed * np.log(observed / (MOST_SHOTS * probabilities(factor))))
        )

    assert 3.6 < lr(result.low) < 5 and 3.6 < lr(result.high) < 5, result
    assert lr(result.factor) < 0.04, result


def test_a_flat_gate_axis_does_not_stop_the_readout_search() -> None:
    zeros = 6_980_294_000
    xx = ("xx", XX, {"0": 3911, "1": 89})
    counts = orthogonal_counts([xx, ("empty", (), {"0": zeros, "1": MOST_SHOTS - zeros})])
    result = compare(orthogonal(), counts)
    exact = math.log(zeros / MOST_SHOTS) / math.log(0.7)
    assert result.readout.factor == pytest.approx(exact, rel=1e-6), result.readout
    assert result.deviance < 0.01, result.deviance


def narrow_ridge(shots: int) -> tuple[Profile, MeasuredCounts]:
    profile = one_qubit("narrow", {"x": {"avg_infidelity": 0.01}}, readout={"error": 0.01})
    ones = round(shots * reference(profile, XX, 1, unknown_gates="error")[1])
    xx = ("xx", XX, {"0": shots - ones, "1": ones})
    return profile, written(profile, [xx, ("readout", (), {"0": 3960, "1": 40})])


def test_every_drawn_maximum_is_the_likelihood_at_its_factors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    found = []
    draw_max = fit._Fit.draw_max

    def recorded(self: Any, draws: np.ndarray, windows: Any = ()) -> Any:
        best = draw_max(self, draws, windows)
        found.append((self.surface, draws, best))
        return best

    monkeypatch.setattr(fit._Fit, "draw_max", recorded)
    compare(*narrow_ridge(100_000))
    for surface, draws, (top, gate, readout) in found:
        reached = [
            surface.loglik_at([g], [r], draws[:, k])[0]
            for k, (g, r) in enumerate(zip(gate, readout, strict=True))
        ]
        np.testing.assert_allclose(top, reached, rtol=1e-12)


def test_a_ridge_narrower_than_the_grid_keeps_the_readout_interval_of_its_readout_circuit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    found = []
    draw_max = fit._Fit.draw_max

    def recorded(self: Any, draws: np.ndarray, windows: Any = ()) -> Any:
        best = draw_max(self, draws, windows)
        if not windows:
            found.append(fit._saturated(draws, self.surface.slices) - best[0])
        return best

    monkeypatch.setattr(fit._Fit, "draw_max", recorded)
    profile, counts = narrow_ridge(10_000_000)
    joint = compare(profile, counts).readout
    (gap,) = found
    assert -1e-6 <= gap.min() and gap.max() <= 2 * fit.ASCENT_TOL, (gap.min(), gap.max())
    alone = compare(profile, written(profile, [("readout", (), {"0": 3960, "1": 40})])).readout
    half = math.log(alone.high / alone.low) / 2
    for end, expected in ((joint.low, alone.low), (joint.high, alone.high)):
        assert abs(math.log(end / expected)) < 0.1 * half, (joint, alone)


def test_unequal_shots_leave_both_factors_to_their_intervals() -> None:
    profile, counts = narrow_ridge(MOST_SHOTS)
    circuits = counts.circuits
    observed = np.concatenate([c.vector() for c in circuits]).astype(float)
    center = fit._exact(profile, circuits, 1.0, 1.0)
    shots = np.array([MOST_SHOTS, 4000.0])
    info = fit._fisher(profile, circuits, shots, center, 1.0, 1.0)
    eigen = np.linalg.eigvalsh(info)
    assert eigen[0] < 1e-6 * eigen[1]
    surface = fit._Surface.cached(profile, circuits)
    assert fit._identify(surface, observed, info) == {"gate": None, "readout": None}
    assert fit._rank(info) == 2


SPARSE_SHOTS, SPARSE_ERROR = 4000, 0.00075


def one_binomial(ones: int) -> tuple[Profile, MeasuredCounts]:
    profile = one_qubit(
        "sparse-binomial", {"x": {"avg_infidelity": 0.0}}, readout={"error": SPARSE_ERROR}
    )
    return profile, written(profile, [("readout", (), {"0": SPARSE_SHOTS - ones, "1": ones})])


def misread(factor: float) -> float:
    return scale_readout((SPARSE_ERROR, SPARSE_ERROR), factor)[0]


def binomial(ones: int, p: float) -> float:
    return math.comb(SPARSE_SHOTS, ones) * p**ones * (1 - p) ** (SPARSE_SHOTS - ones)


@pytest.mark.parametrize("truth", [0.5, 1.0, 2.0, 5.0])
def test_one_sparse_binomial_covers_the_true_factor_over_every_error_count(truth: float) -> None:
    weights = {ones: binomial(ones, misread(truth)) for ones in range(60)}
    coverage = sum(
        weight * covers(compare(*one_binomial(ones)).readout, truth)
        for ones, weight in weights.items()
        if weight > 1e-12
    )
    assert coverage >= 0.95, coverage


def test_a_sparse_interval_reaches_every_factor_the_exact_binomial_test_accepts() -> None:
    def lr(ones: int, factor: float) -> float:
        def loglik(p: float) -> float:
            return ones * math.log(p) + (SPARSE_SHOTS - ones) * math.log1p(-p)

        lowest, highest = (misread(f) for f in fit.FACTOR_RANGE)
        return 2 * (
            loglik(min(max(ones / SPARSE_SHOTS, lowest), highest)) - loglik(misread(factor))
        )

    def tail(factor: float) -> float:
        observed = lr(8, factor)
        return sum(
            binomial(k, misread(factor)) for k in range(60) if lr(k, factor) >= observed - 1e-9
        )

    result = compare(*one_binomial(8)).readout
    accepted = [f for f in np.geomspace(0.5, 8, 81) if tail(f) > 1 - fit.LEVEL]
    slack = fit.END_TOL * math.log(result.high / result.low) / 2
    assert math.log(result.low / min(accepted)) <= slack, (result, min(accepted))
    assert math.log(max(accepted) / result.high) <= slack, (result, max(accepted))


def test_two_runs_of_one_plan_build_the_surface_once(monkeypatch: pytest.MonkeyPatch) -> None:
    built = []
    build = fit._Surface.build.__func__

    def counting(cls: type, base: Profile, circuits: Sequence[PlannedCircuit]) -> Any:
        built.append(circuits)
        return build(cls, base, circuits)

    monkeypatch.setattr(fit._Surface, "build", classmethod(counting))
    monkeypatch.setattr(fit._Surface, "_built", type(fit._Surface._built)())
    profile = toy()
    circuits = [PlannedCircuit(name="xx", qubits=(0,), ops=XX)]
    for seed in (1, 2):
        compare(profile, simulate(profile, circuits, shots=4000, seed=seed, run_at=LATER))
    assert len(built) == 1


def test_the_surface_is_the_reference_at_every_node_and_a_distribution_between() -> None:
    kingston = nv.load(KINGSTON)
    circuits = plan(kingston)
    surface = fit._Surface.cached(kingston, circuits)
    readouts = cycle((surface.readout[0], 0.0, surface.readout[-1]))
    for node, readout in zip(surface.gate_nodes, readouts, strict=False):
        truth = scaled(kingston, math.exp(node), math.exp(readout))
        expected = np.concatenate(
            [
                reference(truth, c.ops, len(c.qubits), layout=c.qubits, unknown_gates="error")
                for c in circuits
            ]
        )
        assert np.abs(surface.probs_at([node], [readout])[0] - expected).max() < 1e-12
    between = surface.probs_at(
        (surface.gate_nodes[:-1] + surface.gate_nodes[1:]) / 2,
        np.zeros(len(surface.gate_nodes) - 1),
    )
    assert between.min() >= 0
    for part in surface.slices:
        assert np.abs(between[:, part].sum(axis=1) - 1).max() < 1e-12


def floor_counts(zero: int, one: int) -> tuple[Profile, MeasuredCounts]:
    profile = one_qubit(
        "idle",
        {"id": {"avg_infidelity": 0.001, "duration_ns": 100}},
        idle={"t1_us": 100, "t2_us": 200},
    )
    one_id, two_ids = (Op("id", (0,)),), (Op("id", (0,)), Op("id", (0,)))
    return profile, written(
        profile,
        [("one_id", one_id, {"0": zero, "1": one}), ("two_ids", two_ids, {"0": zero + one})],
    )


def test_zero_probability_cells_give_finite_likelihoods_in_the_fit_and_every_draw() -> None:
    profile, counts = floor_counts(4000, 0)
    (floor,) = fit._floor_factors(profile, counts.circuits)
    result = compare(profile, counts)
    assert math.isfinite(result.deviance) and result.dof == 1 and result.p_value is not None
    assert "NaN" not in json.dumps(result.to_dict())
    assert result.gates.factor == pytest.approx(floor) and result.gates.bound == "lower"
    above = compare(*floor_counts(3999, 1))
    assert above.gates.factor > floor * 1.01
    assert "NaN" not in json.dumps(above.to_dict())


@pytest.mark.parametrize(
    ("ref", "dof"),
    [("ibm_kingston@2026-04-15", 38), ("ibm_fez@2025-02-26", 30), ("ibm_boston@2026-04-17", 26)],
)
def test_degrees_of_freedom_count_the_outcomes_the_profile_supports(ref: str, dof: int) -> None:
    profile = nv.load(ref)
    run_at = max(RUN_AT, profile.device.calibrated_at)
    result = compare(profile, simulate(profile, plan(profile), shots=4000, seed=3, run_at=run_at))
    assert result.dof == dof


def test_a_readout_only_file_has_no_degrees_of_freedom_and_is_not_tested() -> None:
    profile = toy()
    result = compare(profile, written(profile, [("readout", (), {"0": 3960, "1": 40})]))
    assert (result.dof, result.dispersion, result.p_value) == (0, 1, None)
    assert "fit             not testable (no degrees of freedom left after fitting)" in str(result)


def test_a_flat_gate_axis_keeps_the_readout_estimate_and_an_upper_gate_bound() -> None:
    kingston = nv.load(KINGSTON)
    truth = scaled(kingston, 0.05, 1.0)
    result = compare(kingston, simulate(truth, plan(kingston), shots=4000, seed=5, run_at=RUN_AT))
    assert covers(result.readout, 1.0), result.readout
    assert isinstance(result.gates, ErrorFactor) and result.gates.bound == "lower"
    assert result.gates.low is None and result.gates.high < 1
    lowest_floor = min(fit._floor_factors(kingston, plan(kingston)))
    assert result.gates.factor == pytest.approx(lowest_floor)


def test_a_gate_axis_flat_at_the_estimate_is_not_a_collinearity() -> None:
    gates = {"rz": {"virtual": True}, "sx": {"avg_infidelity": 1e-4, "duration_ns": 100}}
    idle = {"t1_us": 111, "t2_us": 222}
    profile = one_qubit("slow", gates, readout={"error": 0.01}, idle=idle)
    circuits = plan(profile)
    result = compare(profile, simulate(profile, circuits, shots=4000, seed=0, run_at=LATER))
    assert isinstance(result.gates, ErrorFactor) and result.gates.bound == "lower"
    assert result.gates.factor == pytest.approx(1.0)
    assert covers(result.readout, 1.0), result.readout
    (floor,) = fit._floor_factors(profile, circuits)
    assert 2 < floor < 4
    surface = fit._Surface.cached(profile, circuits)
    expected = np.concatenate(
        [
            reference(scaled(profile, floor, 1.0), c.ops, 1, layout=c.qubits, unknown_gates="error")
            for c in circuits
        ]
    )
    assert np.abs(surface.probs_at([math.log(floor)], [0.0])[0] - expected).max() < 1e-12


def test_a_gate_that_barely_moves_the_only_circuit_leaves_the_readout_interval() -> None:
    readout = {"p1_given_0": 0.005, "p0_given_1": 0.07}
    profile = one_qubit("faint", {"x": {"pauli": [1e-6, 1e-5, 2e-3]}}, readout=readout)
    ops = (Op("x", (0,)),)
    ones = round(100_000 * reference(profile, ops, 1, unknown_gates="error")[1])
    result = compare(profile, written(profile, [("x", ops, {"0": 100_000 - ones, "1": ones})]))
    assert result.gates == NoEstimate("the counts do not constrain the gate factor")
    assert covers(result.readout, 1.0) and result.readout.high / result.readout.low < 1.1


def test_a_probability_change_below_any_fixed_cutoff_still_moves_the_gate_factor() -> None:
    profile = one_qubit("tiny", {"ry": {"pauli": [0, 0, 0.01]}})
    ops = (Op("ry", (0,), (7e-5,)),) * 2
    counts = written(profile, [(f"c{k}", ops, {"0": 9_999_999_959, "1": 41}) for k in range(10)])
    result = compare(profile, counts)
    assert isinstance(result.gates, ErrorFactor) and result.gates.low > 1, result.gates


def test_a_change_within_the_rounding_error_does_not_move_a_circuit() -> None:
    profile = one_qubit("dephased", {"ry": {"pauli": [0, 0, 0]}, "rz": {"pauli": [0, 0, 0.02]}})
    ops = (Op("ry", (0,), (1.0,)), *(Op("rz", (0,), (0.3,)),) * 5)
    result = compare(profile, written(profile, [("rz5", ops, {"0": 7702, "1": 2298})]))
    assert result.gates == NoEstimate("no circuit's outcomes move with gate error")


def test_x_then_x_counts_identify_neither_factor() -> None:
    profile = toy()
    counts = simulate(
        profile, [PlannedCircuit(name="xx", qubits=(0,), ops=XX)], shots=4000, seed=2, run_at=LATER
    )
    result = compare(profile, counts)
    same = NoEstimate("gate error and readout error move these counts the same way")
    assert (result.gates, result.readout) == (same, same)
    center = fit._exact(profile, counts.circuits, 1.0, 1.0)
    eigen = np.linalg.eigvalsh(fit._fisher(profile, counts.circuits, [4000], center, 1.0, 1.0))
    assert eigen[0] < fit.SEPARABLE * eigen[1]
    assert (result.dof, result.p_value) == (0, None)
    assert "not testable (no degrees of freedom left after fitting)" in str(result)


def test_readout_counts_with_no_observed_error_give_only_an_upper_end() -> None:
    profile = toy()
    result = compare(profile, written(profile, [("readout", (), {"0": 4000})]))
    assert isinstance(result.readout, ErrorFactor)
    assert (result.readout.bound, result.readout.low) == ("lower", None)
    assert result.readout.high is not None


def test_notes_name_what_the_factors_cannot_scale_or_charge() -> None:
    profile = one_qubit("odd", {"x": {"pauli": [0.1, 0.0, 0.1]}}, readout={"error": 0.01})
    profile = profile.model_copy(
        update={"device": {**profile.to_dict()["device"], "num_qubits": 3}}
    )
    ops = (Op("x", (0,)), Op("x", (1,)), Op("x", (2,)), Op("delay", (0,), (100.0,)))
    counts = {"111": 3400, "011": 200, "101": 200, "110": 200}
    result = compare(profile, written(profile, [("three_x", ops, counts)]))
    assert result.notes == (
        "x on qubits 0, 1 and 2 is not scaled (it has a negative Pauli-Lindblad rate)",
        "delays on qubit 0 add no idle error (no T1 or T2 stated)",
    )
    assert result.gates == NoEstimate("no circuit's outcomes move with gate error")
    lines = str(result).split("\n")
    note = lines.index("note            x on qubits 0, 1 and 2 is not scaled")
    assert lines[note + 1 : note + 3] == [
        "                (it has a negative Pauli-Lindblad rate)",
        "                delays on qubit 0 add no idle error (no T1 or T2 stated)",
    ]


def test_a_saved_comparison_names_every_qubit_and_its_summary_shortens_the_list() -> None:
    profile = one_qubit("odd", {"x": {"pauli": [0.1, 0.0, 0.1]}}, readout={"error": 0.01})
    profile = profile.model_copy(
        update={"device": {**profile.to_dict()["device"], "num_qubits": 6}}
    )
    ops = (*(Op("x", (q,)) for q in range(6)), *(Op("delay", (q,), (100.0,)) for q in range(6)))
    result = compare(profile, written(profile, [("six_x", ops, {"111111": 4000})]))
    assert result.to_dict()["notes"] == [
        "x on qubits 0, 1, 2, 3, 4 and 5 is not scaled (it has a negative Pauli-Lindblad rate)",
        "delays on qubits 0, 1, 2, 3, 4 and 5 add no idle error (no T1 or T2 stated)",
    ]
    assert {type(note) for note in result.to_dict()["notes"]} == {str}
    lines = str(result).split("\n")
    note = lines.index("note            x on qubits 0, 1, 2 and 3 more is not scaled")
    assert lines[note + 1 : note + 4] == [
        "                (it has a negative Pauli-Lindblad rate)",
        "                delays on qubits 0, 1, 2 and 3 more add no idle error",
        "                (no T1 or T2 stated)",
    ]


def excess_readout(seed: int) -> MeasuredCounts:
    kingston = nv.load(KINGSTON)
    data = kingston.to_dict()
    record = next(q for q in data["qubits"] if q["index"] == 150)
    a, b = kingston.table.qubit(150).readout
    record["readout"] = {"p1_given_0": 3 * a, "p0_given_1": 3 * b}
    truth = Profile.from_dict(data)
    return rebound(simulate(truth, plan(kingston), shots=4000, seed=seed, run_at=RUN_AT), kingston)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_excess_readout_error_on_one_qubit_is_a_poor_fit(seed: int) -> None:
    result = compare(nv.load(KINGSTON), excess_readout(seed))
    assert result.p_value < 0.01
    assert "no one pair of factors fits every circuit" in str(result)


def test_counts_simulated_from_the_factors_fit_within_shot_noise() -> None:
    result = compare(nv.load(KINGSTON), load_counts(EXAMPLE))
    assert result.p_value >= 0.01


def fez_with_impossible_shots(n: int) -> tuple[Profile, MeasuredCounts]:
    fez = nv.load("ibm_fez@2025-02-26")
    run_at = datetime(2025, 2, 27, 11, 42, tzinfo=UTC)
    counts = simulate(fez, plan(fez), shots=4000, seed=7, run_at=run_at)
    return fez, moved(counts, "readout", "0000", "1000", n)


def test_shots_the_profile_rules_out_set_p_to_zero_and_count_in_the_tvd() -> None:
    fez, counts = fez_with_impossible_shots(9)
    result = compare(fez, counts)
    assert result.p_value == 0
    assert {s.name: s.impossible for s in result.circuits}["readout"] == 9
    stated = np.concatenate(
        [
            reference(fez, c.ops, len(c.qubits), layout=c.qubits, unknown_gates="error")
            for c in counts.circuits
        ]
    )
    for c, score, start in zip(
        counts.circuits, result.circuits, np.cumsum([0, 16, 8, 4]), strict=True
    ):
        part = stated[start : start + 2 ** len(c.qubits)]
        assert score.tvd_profile == pytest.approx(0.5 * np.abs(c.vector() / c.shots - part).sum())
    assert "qubit 136 has P(1|0) = 0, and 9 readout shots read it as 1" in str(result)
    assert result.fitted_profile().unmodeled_error.fit.impossible_shots == 9


def test_counts_the_profile_rules_out_entirely_identify_neither_factor() -> None:
    fez, counts = fez_with_impossible_shots(0)
    data = counts.to_dict()
    data["circuits"] = [
        {**c, "counts": {"1000": 4000}} for c in data["circuits"] if c["name"] == "readout"
    ]
    result = compare(fez, MeasuredCounts.model_validate(data))
    assert result.gates == result.readout == NoEstimate("the profile rules out every shot")
    assert result.p_value == 0 and result.impossible_shots == 4000


def test_shots_that_leave_the_fit_carry_no_information() -> None:
    profile = Profile.model_validate(
        {
            "noisevault": "1.0",
            "device": {"name": "pair", "technology": "superconducting", "num_qubits": 2},
            "connectivity": "all_to_all",
            "gates": {"x": {"avg_infidelity": 0.01}},
            "qubits": [
                {"index": 0, "readout": {"error": 0.01}},
                {"index": 1, "readout": {"error": 0.0}},
            ],
        }
    )
    xx = ("xx", XX, {"0": 3862, "1": 138})
    result = compare(profile, written(profile, [xx, ("blank", (), {"01": 4000})]))
    same = NoEstimate("gate error and readout error move these counts the same way")
    assert (result.gates, result.readout) == (same, same)
    assert (result.impossible_shots, result.dof) == (4000, 0)


def refused(profile: Profile, counts: MeasuredCounts) -> str:
    with pytest.raises(CountsError) as caught:
        compare(profile, counts)
    message = str(caught.value)
    assert "\n" not in message
    return message


def test_counts_planned_from_another_calibration_are_refused() -> None:
    kingston = nv.load(KINGSTON)
    data = kingston.to_dict()
    data["qubits"][0]["t1_us"] = data["qubits"][0].get("t1_us", 100) + 1
    edited = Profile.from_dict(data)
    message = refused(edited, load_counts(EXAMPLE))
    assert f"planned from nv:{kingston.fingerprint[:12]}" in message
    assert f"nv list` to find nv:{kingston.fingerprint[:12]}" in message


def test_counts_from_another_backend_are_refused() -> None:
    counts = load_counts(EXAMPLE).model_copy(update={"backend": "ibm_fez"})
    assert "ran on ibm_fez, but the profile describes ibm_kingston" in refused(
        nv.load(KINGSTON), counts
    )


def test_counts_that_ran_before_their_calibration_are_refused() -> None:
    counts = load_counts(EXAMPLE).model_copy(update={"run_at": "2026-04-01T00:00:00Z"})
    assert "ran at 2026-04-01 00:00Z, before the calibration" in refused(nv.load(KINGSTON), counts)


def test_an_op_the_profile_does_not_calibrate_on_its_qubits_is_refused() -> None:
    kingston = nv.load(KINGSTON)
    data = load_counts(EXAMPLE).to_dict()
    data["circuits"][0]["ops"].append(["cz", [0, 2], []])
    message = refused(kingston, MeasuredCounts.model_validate(data))
    assert message.startswith("circuit ghz_chain: cz on qubits 148-150: ")


@pytest.mark.parametrize(
    ("name", "hint"),
    [
        ("measure", "run the circuits on qubits that the profile can measure"),
        ("delay", "nv compare scores only ops the profile calibrates on their qubits"),
    ],
)
def test_a_measurement_or_delay_the_profile_disables_is_refused(name: str, hint: str) -> None:
    profile = one_qubit("off", {"x": {"avg_infidelity": 0.01}, name: {"disabled": True}})
    wait = ("wait", (Op("x", (0,)), Op("delay", (0,), (100.0,))), {"0": 40, "1": 3960})
    message = refused(profile, written(profile, [wait]))
    assert message == f"circuit wait: {name} on qubit 0 is disabled in this profile; {hint}"


def test_profile_compare_is_compare_and_import_noisevault_leaves_the_fit_unloaded() -> None:
    profile = scaled(toy(), 1.5, 1.2)
    counts = written(toy(), [("readout", (), {"0": 3960, "1": 40})])
    assert profile.compare(counts).to_dict() == compare(profile, counts).to_dict()
    probe = "import sys, noisevault; print('noisevault.compare' in sys.modules)"
    loaded = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert loaded.stdout.strip() == "False"


def test_the_same_inputs_give_the_same_bytes(tmp_path: Path) -> None:
    kingston, counts = nv.load(KINGSTON), load_counts(EXAMPLE)
    first, second = compare(kingston, counts), compare(kingston, counts)
    assert first.to_dict() == second.to_dict()
    a = first.fitted_profile().save(tmp_path / "a.json").read_bytes()
    b = second.fitted_profile().save(tmp_path / "b.json").read_bytes()
    assert a == b
    fitted = Profile.load(tmp_path / "a.json")
    again = compare(fitted, counts)
    assert (again.gates, again.readout) == (first.gates, first.readout)
    assert fitted.uncorrected().fingerprint == counts.profile.fingerprint


def test_a_gate_factor_that_is_not_identified_has_nothing_to_save() -> None:
    profile = toy()
    xx = PlannedCircuit(name="xx", qubits=(0,), ops=XX)
    neither = compare(profile, simulate(profile, [xx], shots=4000, seed=2, run_at=LATER))
    with pytest.raises(
        NoiseVaultError,
        match="^gate and readout factors are not identified, so there is nothing to save$",
    ):
        neither.fitted_profile()
    readout_only = compare(profile, written(profile, [("readout", (), {"0": 3960, "1": 40})]))
    with pytest.raises(
        NoiseVaultError, match="^the gate factor is not identified, so there is nothing to save$"
    ):
        readout_only.fitted_profile()


def test_a_readout_factor_that_is_not_identified_is_left_out_of_the_fitted_profile() -> None:
    profile = Profile.uniform(
        "quiet", technology="superconducting", num_qubits=1, one_qubit_error=0.002
    )
    circuits = [PlannedCircuit(name="mirror", qubits=(0,), ops=(Op("sx", (0,)),) * 4)]
    result = compare(profile, simulate(profile, circuits, shots=4000, seed=4, run_at=LATER))
    assert result.readout == NoEstimate("the measured qubits state no readout error")
    unmodeled = result.fitted_profile().unmodeled_error
    assert unmodeled.readout is None and unmodeled.gates.factor == pytest.approx(
        result.gates.factor, rel=1e-5
    )


@pytest.mark.parametrize(
    ("estimate", "words"),
    [
        (ErrorFactor(factor=1.84, low=1.54, high=2.12), "x1.84 (95% interval 1.54 to 2.12)"),
        (
            ErrorFactor(factor=0.096, high=0.311, bound="lower"),
            "x0.096 (95% interval, at most 0.311)",
        ),
        (ErrorFactor(factor=20.0, low=11.2, bound="upper"), "x20 (95% interval, at least 11.2)"),
        (
            ErrorFactor(factor=1.00783, low=1.00109, high=1.0146),
            "x1.008 (95% interval 1.001 to 1.015)",
        ),
        (ErrorFactor(factor=1.0, low=1.0, high=1.0004), "x1 (95% interval 1 to 1.0004)"),
        (ErrorFactor(factor=1.0004), "x1"),
    ],
)
def test_an_estimate_reads_with_the_fewest_digits_that_tell_its_values_apart(
    estimate: ErrorFactor, words: str
) -> None:
    assert estimate.describe() == words


def layout(result: Comparison) -> list[str]:
    lines = result.summary(counts_file="run.counts.json").split("\n")
    assert all(len(line) <= 80 for line in lines), max(lines, key=len)
    assert lines[3] == "" and lines[4].startswith("circuit ")
    table_end = lines.index("", 4)
    rows = lines[5:table_end]
    assert len(rows) == len(result.circuits)
    impossible = result.impossible_shots > 0
    assert lines[4].endswith("  impossible shots") == impossible
    for row in rows:
        fitted, noise = (float(v) for v in row.split()[3:5])
        assert row.endswith("  beyond noise") == (not impossible and fitted > noise), row
    block = lines[table_end + 1 :]
    block = block[: block.index("")] if "" in block else block
    labels = {"gate errors", "readout errors", "fit", "next", "note", ""}
    assert all(line[:16].rstrip() in labels and line[16] != " " for line in block), block
    fitted = isinstance(result.gates, ErrorFactor) or isinstance(result.readout, ErrorFactor)
    assert (lines[-3:] == list(fit.NOTE)) == fitted
    return lines


def test_a_good_fit_reads_as_designed() -> None:
    lines = layout(compare(nv.load(KINGSTON), load_counts(EXAMPLE)))
    short = nv.load(KINGSTON).short_fingerprint
    assert lines[0] == f"ibm_kingston@2026-04-15 {short} on qubits 148-149-150-151"
    assert lines[1].startswith("counts run.counts.json, simulated, sha256:")
    assert lines[2] == "run 2026-04-16 09:30Z, 26 h after calibration"
    assert lines[4] == "circuit       shots  profile TVD  fitted TVD  noise TVD 95%"
    assert any(line.startswith("gate errors     x") for line in lines)
    assert any(line.startswith("readout errors  x") for line in lines)
    assert any(line.startswith("fit             within shot noise") for line in lines)


def test_a_poor_fit_puts_the_verdict_first_and_names_the_circuits_below() -> None:
    result = compare(nv.load(KINGSTON), excess_readout(0))
    lines = layout(result)
    flagged = [row.split()[0] for row in lines[5:9] if row.endswith("  beyond noise")]
    verdict = lines.index(f"fit             beyond shot noise (p = {result.p_value:.2g})")
    assert flagged and lines[verdict + 1 : verdict + 3] == [
        f"                on {', '.join(flagged[:-1])} and {flagged[-1]}"
        if len(flagged) > 1
        else f"                on {flagged[0]}",
        "                no one pair of factors fits every circuit",
    ]


def poor(result: Comparison, scores: Sequence[tuple[str, float, float]]) -> Comparison:
    rows = tuple(
        CircuitScore(name, (0, 1), 4000, 0.05, fitted, noise, 0) for name, fitted, noise in scores
    )
    return replace(result, circuits=rows, p_value=0.0025)


def test_the_flag_follows_the_printed_values() -> None:
    result = poor(
        compare(nv.load(KINGSTON), load_counts(EXAMPLE)),
        [("tied", 0.00412, 0.00408), ("over", 0.0050, 0.0041), ("under", 0.0031, 0.0040)],
    )
    lines = layout(result)
    assert lines[5] == "tied      4000       0.0500      0.0041         0.0041"
    assert lines[6].endswith("0.0050         0.0041  beyond noise")
    assert "                on over" in lines


def test_a_node_profile_built_without_revalidation_equals_the_validated_one() -> None:
    kingston = nv.load(KINGSTON)
    built, validated = (
        fit._scaled_without_revalidation(kingston, 1.37, 0.8),
        scaled(kingston, 1.37, 0.8),
    )
    assert built.fingerprint == validated.fingerprint
    assert built.to_dict() == validated.to_dict()
    for c in plan(kingston):
        for q in c.qubits:
            assert built.table.qubit(q) == validated.table.qubit(q)
        for op in c.ops:
            if op.name != "delay":
                qubits = [c.qubits[i] for i in op.qubits]
                assert built.table.gate(op.name, qubits) == validated.table.gate(op.name, qubits)


def test_a_ruled_out_fit_adds_the_impossible_column_and_names_the_qubit() -> None:
    lines = layout(compare(*fez_with_impossible_shots(9)))
    assert (
        lines[4] == "circuit       shots  profile TVD  fitted TVD  noise TVD 95%  impossible shots"
    )
    assert "fit             ruled out (p = 0)" in lines
    assert "                the fit uses the other 15991 shots" in lines


def test_factors_the_counts_do_not_identify_print_no_note() -> None:
    profile = toy()
    counts = simulate(
        profile, [PlannedCircuit(name="xx", qubits=(0,), ops=XX)], shots=4000, seed=2, run_at=LATER
    )
    lines = layout(compare(profile, counts))
    assert lines[0] == f"toy nv:{profile.fingerprint[:12]} on qubit 0"
    assert lines[2] == "run 2026-10-01 12:00Z, calibration age unknown (the profile has no date)"
    assert lines[-6:] == [
        "gate errors     not identified",
        "readout errors  not identified",
        "                gate error and readout error move these counts the same way",
        "fit             not testable (no degrees of freedom left after fitting)",
        "next            add a circuit with no gates, which only readout error moves.",
        "                The circuits from noisevault.counts.plan() include one.",
    ]


def test_a_saved_interval_keeps_the_digits_that_tell_its_values_apart() -> None:
    profile = one_qubit("precise", {"x": {"avg_infidelity": 0.2}})
    x = (Op("x", (0,)),)
    counts = written(
        profile, [(f"x{i}", x, {"0": 2012082172, "1": 7987917828}) for i in range(300)]
    )
    result = compare(profile, counts)
    assert isinstance(result.gates, ErrorFactor)
    full = (result.gates.factor, result.gates.low, result.gates.high)
    saved = result.fitted_profile().unmodeled_error.gates
    assert (saved.factor, saved.low, saved.high) == tuple(float(f"{v:.7g}") for v in full)
    assert saved.low < saved.factor < saved.high
    apart = ErrorFactor(factor=1.84123456, low=1.54123456, high=2.12123456)
    saved = replace(result, gates=apart).fitted_profile().unmodeled_error.gates
    assert (saved.factor, saved.low, saved.high) == (1.84123, 1.54123, 2.12123)


def fit_lines(result: Comparison) -> list[str]:
    lines = layout(result)
    start = next(i for i, line in enumerate(lines) if line.startswith("fit "))
    return [line[16:] for line in lines[start : lines.index("", start)]]


UNREACHED = "no factors from 0.05 to 20 give the measured frequencies"


def test_a_fit_that_misses_the_frequencies_is_tested_with_no_degrees_of_freedom() -> None:
    profile = toy()
    held = compare(profile, written(profile, [("readout", (), {"0": 3200, "1": 800})]))
    assert isinstance(held.readout, ErrorFactor) and held.readout.bound == "upper"
    assert (held.dof, held.p_value) == (0, pytest.approx(1 / (1 + fit.RESAMPLES)))
    assert fit_lines(held) == ["beyond shot noise (p = 0.0025)", "on readout", UNREACHED]
    near = compare(profile, written(profile, [("readout", (), {"0": 3999, "1": 1})]))
    assert isinstance(near.readout, ErrorFactor) and near.readout.bound == "lower"
    assert near.dof == 0 and near.deviance < fit.CHI2_95 and near.p_value is not None
    p = f"p = {near.p_value:.2g}"
    assert fit_lines(near) == [f"within shot noise on every circuit ({p})", UNREACHED]
    peaks, counts = two_peaks(0, last=EQUAL_ENDS)
    beyond = {"0": 542129, "1": 457871}
    inside = compare(peaks, written(peaks, [("r4", counts.circuits[0].ops, beyond)]))
    assert isinstance(inside.gates, ErrorFactor) and inside.gates.bound is None
    assert (inside.dof, inside.p_value) == (0, pytest.approx(1 / (1 + fit.RESAMPLES)))
    assert fit_lines(inside) == ["beyond shot noise (p = 0.0025)", "on r4", UNREACHED]


def one_sparse_qubit_beside(trivial: int) -> tuple[Profile, MeasuredCounts]:
    profile = Profile.model_validate(
        {
            "noisevault": "1.0",
            "device": {"name": "sparse-pair", "technology": "superconducting", "num_qubits": 2},
            "connectivity": "all_to_all",
            "gates": {"x": {"avg_infidelity": 0.0}},
            "qubits": [{"index": 1, "readout": {"error": 0.00075}}],
        }
    )
    sparse = [("sparse", (), {"0": 3992, "1": 8})]
    data = written(profile, sparse + [(f"exact{i}", (), {"0": 4000}) for i in range(trivial)])
    data = data.to_dict()
    data["circuits"][0]["qubits"] = [1]
    return profile, MeasuredCounts.model_validate(data)


def test_circuits_with_one_possible_count_leave_the_exact_interval_unchanged() -> None:
    alone = compare(*one_sparse_qubit_beside(0))
    beside = compare(*one_sparse_qubit_beside(64))
    assert isinstance(alone.readout, ErrorFactor) and alone.readout.bound is None
    assert beside.readout == alone.readout


def test_exact_enumeration_lists_each_combination_of_counts_once_with_its_probability() -> None:
    probs = np.array([0.9, 0.1, 1.0, 0.0, 0.7, 0.3])
    slices = (slice(0, 2), slice(2, 4), slice(4, 6))
    outcomes = fit._enumerated(probs, slices, np.array([3, 5, 3]))
    assert outcomes is not None
    listed = dict(zip(map(tuple, outcomes.counts.T.astype(int)), outcomes.weights, strict=True))

    def binomial(k: int, q: float) -> float:
        return math.comb(3, k) * q**k * (1 - q) ** (3 - k)

    expected = {
        (3 - a, a, 5, 0, 3 - b, b): binomial(a, 0.1) * binomial(b, 0.3)
        for a in range(4)
        for b in range(4)
    }
    assert len(listed) == outcomes.counts.shape[1] == len(expected)
    assert listed == pytest.approx(expected, rel=1e-12)


def with_effect(allow: str) -> Profile:
    effect = {"type": "coherent_overrotation", "gate": "x", "angle_rad": 0.4, "allow": allow}
    return one_qubit(
        "effect", {"x": {"avg_infidelity": 0.01}}, readout={"error": 0.01}, effects=[effect]
    )


@pytest.mark.parametrize("allow", ["exact", "approximate"])
def test_an_effect_the_reference_cannot_leave_out_refuses_the_comparison(allow: str) -> None:
    profile = with_effect(allow)
    counts = written(profile, [("x", (Op("x", (0,)),), {"0": 20, "1": 980})])
    with pytest.raises(UnsupportedEffect) as caught:
        compare(profile, counts)
    assert str(caught.value) == (
        f"effect coherent_overrotation on x asks for allow='{allow}', but the reference"
        " simulator does not model effects yet; set allow to 'omit' to leave the effect out"
    )


def test_an_omitted_effect_is_named_in_the_notes_and_the_summary() -> None:
    profile = with_effect("omit")
    result = compare(profile, written(profile, [("x", (Op("x", (0,)),), {"0": 20, "1": 980})]))
    note = (
        "the reference simulator leaves out effect coherent_overrotation on x, because the"
        " profile sets allow to 'omit'"
    )
    assert result.notes == (note,)
    assert result.to_dict()["notes"] == [note]
    lines = layout(result)
    start = next(i for i, line in enumerate(lines) if line.startswith("note "))
    assert " ".join(line[16:] for line in lines[start : start + 2]) == note


if __name__ == "__main__":
    example_counts().save(EXAMPLE)
    print(f"wrote {EXAMPLE.relative_to(ROOT)}")
