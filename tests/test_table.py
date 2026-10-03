from __future__ import annotations

import time
import tracemalloc
from dataclasses import replace

import pytest
from conftest import toy

import noisevault as nv
from noisevault import metrics
from noisevault.profile import Profile
from noisevault.table import GateNoise, Unavailable

ECR = {"qubits": 2, "avg_infidelity": 7e-3, "duration_ns": 400}


def table_of(**sections):
    return Profile.model_validate(toy(**sections)).table


def test_record_overrides_definition_field_by_field() -> None:
    table = table_of(calibrations=[{"gate": "cz", "qubits": [0, 1], "avg_infidelity": 0.02}])
    found = table.gate("cz", (0, 1))
    assert (found.state, found.origin, found.avg_infidelity) == ("calibrated", "record", 0.02)
    assert found.duration_ns == 70  # inherited from the definition
    default = table.gate("cz", (1, 2))
    assert (default.origin, default.avg_infidelity) == ("default", 1e-2)


def test_metric_group_is_replaced_whole() -> None:
    pauli = [1e-3] * 15
    table = table_of(calibrations=[{"gate": "cz", "qubits": [0, 1], "pauli": pauli}])
    found = table.gate("cz", (0, 1))
    assert found.spec.avg_infidelity is None
    assert found.pauli == tuple(pauli)
    assert found.avg_infidelity == pytest.approx(metrics.avg_from_pauli(pauli))


def test_symmetric_gate_uses_reversed_record_with_permuted_pauli() -> None:
    pauli = [0.0] * 15
    pauli[metrics.PAULI_2Q.index("IX")] = 3e-3  # X on the second operand, qubit 1
    table = table_of(calibrations=[{"gate": "cz", "qubits": [0, 1], "pauli": pauli}])
    found = table.gate("cz", (1, 0))
    assert found.origin == "reversed_record"
    assert found.pauli[metrics.PAULI_2Q.index("XI")] == 3e-3  # still on qubit 1, now first
    assert found.pauli[metrics.PAULI_2Q.index("IX")] == 0.0
    assert found.spec.pauli == found.pauli


def test_directed_gate_is_not_reversed() -> None:
    gates = {**toy()["gates"], "ecr": ECR}
    table = table_of(
        gates=gates,
        connectivity={"edges": [[1, 0], [1, 2]], "directed": True},
        calibrations=[{"gate": "ecr", "qubits": [1, 0], "avg_infidelity": 6e-3}],
    )
    assert table.gate("ecr", (1, 0)).avg_infidelity == 6e-3
    assert isinstance(table.gate("ecr", (0, 1)), Unavailable)
    assert table.gate("ecr", (1, 2)).origin == "default"
    assert isinstance(table.gate("ecr", (2, 1)), Unavailable)


def test_connectivity_decides_where_defaults_apply() -> None:
    undirected = table_of()
    assert undirected.gate("cz", (1, 0)).origin == "default"
    assert isinstance(undirected.gate("cz", (0, 2)), Unavailable)
    everywhere = table_of(connectivity="all_to_all")
    assert everywhere.gate("cz", (0, 2)).origin == "default"


def test_disabled_record_terminates_lookup() -> None:
    table = table_of(
        calibrations=[
            {"gate": "cz", "qubits": [0, 1], "disabled": True},
            {"gate": "cz", "qubits": [1, 0], "avg_infidelity": 1e-2},
        ]
    )
    assert table.gate("cz", (0, 1)).state == "disabled"
    assert not table.allowed("cz", (0, 1))
    assert table.allowed("cz", (1, 0))


def test_gate_states() -> None:
    gates = {**toy()["gates"], "reset": {"duration_ns": 1000}}
    table = table_of(gates=gates)
    assert table.gate("rz", (0,)).state == "ideal"
    assert table.gate("sx", (0,)).state == "calibrated"
    assert table.gate("reset", (0,)).state == "uncalibrated"
    assert isinstance(table.gate("h", (0,)), Unavailable)


def test_z_family_is_free_only_through_a_virtual_rz() -> None:
    table = table_of()
    assert table.gate("s", (1,)).state == "ideal"
    assert table.gate("t", (1,)).state == "ideal"
    assert table.gate("u1", (1,)).state == "ideal"
    calibrated_rz = table_of(
        calibrations=[{"gate": "rz", "qubits": [1], "virtual": False, "avg_infidelity": 1e-4}]
    )
    assert calibrated_rz.gate("s", (0,)).state == "ideal"
    assert isinstance(calibrated_rz.gate("s", (1,)), Unavailable)


def test_defined_z_family_gates_without_their_own_metric_are_free_through_a_virtual_rz() -> None:
    table = table_of(
        gates={
            "rz": {"virtual": True},
            "z": {},
            "s": {"disabled": True},
            "t": {"avg_infidelity": 1e-4},
            "x": {"avg_infidelity": 0.01},
        },
        calibrations=[{"gate": "z", "qubits": [2], "avg_infidelity": 2e-3}],
    )
    free = table.gate("z", (0,))
    assert (free.state, free.avg_infidelity) == ("ideal", None)
    calibrated = table.gate("z", (2,))
    assert (calibrated.state, calibrated.avg_infidelity) == ("calibrated", 2e-3)
    assert table.gate("s", (0,)).state == "disabled"
    assert table.gate("t", (0,)).avg_infidelity == 1e-4


def test_rz_lookup_without_an_rz_definition_is_undefined() -> None:
    found = table_of(gates={"sx": {"avg_infidelity": 1e-3}}).gate("rz", (0,))
    assert isinstance(found, Unavailable) and found.kind == "undefined"


def test_bad_targets_and_disabled_qubits_are_unavailable() -> None:
    table = table_of(qubits=[{"index": 2, "disabled": True}])
    assert "acts on 2" in table.gate("cz", (0,)).reason
    assert "outside" in table.gate("sx", (5,)).reason
    assert "disabled" in table.gate("cz", (1, 2)).reason
    assert table.qubit(2).disabled


def test_reasons_name_qubits_the_way_the_cli_does() -> None:
    table = table_of()
    assert (
        table.gate("cz", (1, 1)).reason == "cz acts on qubits 1-1, but its qubits must be distinct"
    )
    assert table.gate("cz", (0, 2)).reason == (
        "cz has no calibration on qubits 0-2, and the connectivity does not allow cz there"
    )
    no_native = "no calibrated 2-qubit native gate is usable on qubits 0-2"
    assert table.typical(2, (0, 2)).reason == no_native


def test_unavailable_says_why_as_data() -> None:
    table = table_of(qubits=[{"index": 2, "disabled": True}])
    kinds = {
        ("h", (0,)): "undefined",
        ("cz", (0, 2)): "qubit_disabled",
        ("cz", (0,)): "bad_target",
        ("cz", (1, 1)): "bad_target",
        ("sx", (5,)): "bad_target",
    }
    for (name, qubits), kind in kinds.items():
        assert table.gate(name, qubits).kind == kind, (name, qubits)
    assert table_of().gate("cz", (0, 2)).kind == "not_connected"
    assert table_of(gates={"rz": {"virtual": True}}).typical(1, (0,)).kind == "no_native"


def test_qubit_merge_rules() -> None:
    table = table_of(
        idle={"t1_us": 100, "t2_us": 80},
        readout={"p1_given_0": 0.01, "p0_given_1": 0.03},
        prep={"error": 1e-3},
        qubits=[{"index": 1, "t1_us": 40, "readout": {"error": 0.05}}],
    )
    q0, q1 = table.qubit(0), table.qubit(1)
    assert (q0.t1_ns, q0.t2_ns, q0.readout, q0.prep_error) == (1e5, 8e4, (0.01, 0.03), 1e-3)
    assert (q1.t1_ns, q1.t2_ns) == (4e4, 8e4)  # t1 overridden, t2 inherited
    assert q1.readout == (0.05, 0.05)  # readout replaced as a whole
    unknown = table_of().qubit(0)
    assert unknown.readout is None and unknown.prep_error is None and unknown.t1_ns is None


def test_dephasing_rate_merges_like_the_other_idle_fields() -> None:
    table = table_of(
        idle={"t1_us": 100, "dephasing_rate_per_s": 0.3},
        qubits=[{"index": 1, "dephasing_rate_per_s": 2.0}, {"index": 2, "t1_us": 50}],
    )
    assert [table.qubit(q).dephasing_rate_per_s for q in range(3)] == [0.3, 2.0, 0.3]
    assert table_of().qubit(0).dephasing_rate_per_s is None


def test_typical_prefers_most_records_then_name() -> None:
    gates = {**toy()["gates"], "x": {"avg_infidelity": 5e-4}, "id": {"avg_infidelity": 1e-4}}
    record = {"gate": "x", "qubits": [0], "avg_infidelity": 2e-4}
    table = table_of(gates=gates, calibrations=[record])
    assert table.typical(1, (1,)).gate == "x"  # most records
    tied = table_of(gates=gates)
    assert tied.typical(1, (1,)).gate == "sx"  # tie: alphabetical, after any real gate
    assert tied.typical(2, (0, 1)).gate == "cz"


def test_typical_takes_the_identity_only_when_no_real_gate_fits() -> None:
    gates = {**toy()["gates"], "id": {"avg_infidelity": 1e-4}}
    records = [{"gate": "id", "qubits": [q], "avg_infidelity": 1e-4} for q in range(3)]
    table = table_of(gates=gates, calibrations=records)
    assert table.typical(1, (0,)).gate == "sx"  # id has more records, but is an idle slot
    only_id = table_of(gates={**gates, "sx": {"disabled": True}}, calibrations=records)
    assert only_id.typical(1, (0,)).gate == "id"
    for q in range(3):
        assert nv.load("ibm_fez").table.typical(1, (q,)).gate == "sx"


def test_typical_skips_disabled_candidates_and_can_reverse_a_directed_record() -> None:
    gates = {**toy()["gates"], "ecr": ECR}
    records = [
        {"gate": "cz", "qubits": [0, 1], "disabled": True},
        {"gate": "cz", "qubits": [1, 0], "disabled": True},
        {"gate": "cz", "qubits": [2, 1], "avg_infidelity": 1e-2},
        {"gate": "ecr", "qubits": [1, 0], "avg_infidelity": 6e-3},
        {"gate": "ecr", "qubits": [2, 1], "avg_infidelity": 8e-3},
    ]
    table = table_of(
        gates=gates,
        connectivity={"edges": [[1, 0], [2, 1]], "directed": True},
        calibrations=records,
    )
    chosen = table.typical(2, (0, 1))  # cz has most records but is disabled here
    assert (chosen.gate, chosen.origin, chosen.avg_infidelity) == ("ecr", "reversed_record", 6e-3)
    other = table.typical(2, (1, 2))
    assert (other.gate, other.origin) == ("cz", "reversed_record")


def test_typical_ignores_uncalibrated_and_virtual_gates() -> None:
    table = table_of(gates={"rz": {"virtual": True}, "sx": {}, "cz": {"avg_infidelity": 1e-2}})
    assert isinstance(table.typical(1, (0,)), Unavailable)
    assert table.natives(1) == ("sx",)


def test_all_to_all_is_never_expanded() -> None:
    profile = Profile.uniform(
        "big",
        technology="neutral_atom",
        num_qubits=2_000,
        one_qubit_error=1e-3,
        two_qubit_error=5e-3,
    )
    tracemalloc.start()
    start = time.perf_counter()
    table = profile.table
    assert table.gate("cz", (0, 1_999)).avg_infidelity == 5e-3
    assert table.typical(2, (123, 1567)).state == "calibrated"
    assert table.allowed("cz", (1_998, 3))
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 5_000_000 and elapsed < 1.0
    assert isinstance(table.gate("cz", (0, 1)), GateNoise)


def test_disabled_gates_carry_no_error_number() -> None:
    table = table_of(calibrations=[{"gate": "cz", "qubits": [0, 1], "disabled": True}])
    found = table.gate("cz", (0, 1))
    assert found.state == "disabled" and found.avg_infidelity is None and found.pauli is None


def test_typical_does_not_reverse_a_gate_disabled_on_the_pair() -> None:
    gates = {"rz": {"virtual": True}, "sx": {"avg_infidelity": 1e-3}, "ecr": ECR}
    table = table_of(
        gates=gates,
        connectivity={"edges": [[0, 1], [1, 0]], "directed": True},
        calibrations=[
            {"gate": "ecr", "qubits": [0, 1], "disabled": True},
            {"gate": "ecr", "qubits": [1, 0], "avg_infidelity": 6e-3},
        ],
    )
    assert isinstance(table.typical(2, (0, 1)), Unavailable)


def test_listed_pairs_are_connectivity_and_recorded_pairs_low_first() -> None:
    cz_on = [{"gate": "cz", "qubits": [3, 2], "avg_infidelity": 0.02}]
    four = {"name": "toy", "vendor": "test", "technology": "superconducting", "num_qubits": 4}
    directed = {"edges": [[1, 0], [1, 2]], "directed": True}
    table = table_of(device=four, connectivity=directed, calibrations=cz_on)
    assert table.listed_pairs() == [(0, 1), (1, 2), (2, 3)]
    assert table_of(device=four, connectivity="all_to_all", calibrations=cz_on).listed_pairs() == [
        (2, 3)
    ]


@pytest.mark.parametrize(
    ("t1_us", "t2_us", "clamped"),
    [(10, 20.001, True), (10, 20, False), (10, 5, False), (None, 50, False), (10, None, False)],
)
def test_t2_is_clamped_only_above_twice_t1(t1_us, t2_us, clamped) -> None:
    table = table_of(qubits=[{"index": 0, "t1_us": t1_us, "t2_us": t2_us}])
    assert table.qubit(0).t2_clamped is clamped


QUBIT_FIELDS = {
    "idle": {"t1_us": 100, "t2_us": 80, "dephasing_rate_per_s": 0.3},
    "readout": {"p1_given_0": 0.01, "p0_given_1": 0.03},
    "prep": {"error": 1e-3},
}


@pytest.mark.parametrize("factor", [0, 0.5, 1.84, 20])
def test_a_gate_factor_scales_each_gate_error_and_nothing_else(factor: float) -> None:
    pauli = [2e-3, 1e-3, 3e-3]
    sx_on_1 = [{"gate": "sx", "qubits": [1], "pauli": pauli}]
    stated = table_of(calibrations=sx_on_1, **QUBIT_FIELDS)
    table = table_of(
        calibrations=sx_on_1, unmodeled_error={"gates": {"factor": factor}}, **QUBIT_FIELDS
    )
    cz = table.gate("cz", (0, 1))
    assert cz.avg_infidelity == metrics.scale_avg_infidelity(1e-2, 2, factor)
    assert cz.spec == stated.gate("cz", (0, 1)).spec
    sx = table.gate("sx", (1,))
    assert sx.pauli == metrics.scale_pauli(pauli, factor)
    assert sx.avg_infidelity == metrics.avg_from_pauli(sx.pauli)
    assert sx.spec.pauli == tuple(pauli)
    assert [table.qubit(q) for q in range(3)] == [stated.qubit(q) for q in range(3)]


def test_a_readout_factor_scales_each_readout_pair_and_names_the_pairs_it_cannot() -> None:
    kingston = nv.load("ibm_kingston@2026-04-15")
    table = kingston.model_copy(update={"unmodeled_error": {"readout": {"factor": 1.58}}}).table
    for q in (146, 148, 149):
        stated = kingston.table.qubit(q)
        assert table.qubit(q).readout == metrics.scale_readout(stated.readout, 1.58)
        assert table.qubit(q) == replace(stated, readout=table.qubit(q).readout)
    assert sum(kingston.table.qubit(146).readout) >= 1
    assert table.qubit(146).readout == kingston.table.qubit(146).readout
    assert table.qubit(149).readout != kingston.table.qubit(149).readout
    assert table.gate("cz", (148, 149)) == kingston.table.gate("cz", (148, 149))
    assert tuple(p.full for p in table.unscaled()) == (
        "readout of qubit 146 is not scaled (no better than chance)",
    )
    assert kingston.table.unscaled() == ()
    toy_table = table_of(unmodeled_error={"readout": {"factor": 1.58}}, **QUBIT_FIELDS)
    scaled = metrics.scale_readout((0.01, 0.03), 1.58)
    assert toy_table.qubit(0) == replace(table_of(**QUBIT_FIELDS).qubit(0), readout=scaled)


def test_unscaled_names_every_qubit_and_its_short_form_counts_the_rest() -> None:
    cusco = nv.load("ibm_cusco")
    table = cusco.model_copy(update={"unmodeled_error": {"readout": {"factor": 2.0}}}).table
    (phrase,) = table.unscaled()
    every = (
        "0, 14, 18, 37, 39, 52, 56, 57, 71, 75, 76, 78, 90, 94, 95, 96, 97, 101, 109, 113, 114,"
        " 115, 116, 117, 118, 119 and 120"
    )
    assert phrase.full == f"readout of qubits {every} is not scaled (no better than chance)"
    assert phrase.short == (
        "readout of qubits 0, 14, 18 and 24 more is not scaled (no better than chance)"
    )


@pytest.mark.parametrize("factor", [0, 0.5, 1.58, 1e3])
def test_a_qubit_with_zero_readout_error_keeps_it_under_any_factor(factor: float) -> None:
    data = Profile.uniform(
        "u",
        technology="trapped_ion",
        num_qubits=2,
        one_qubit_error=1e-3,
        two_qubit_error=1e-2,
        readout_error=0.02,
    ).to_dict()
    data["qubits"] = [{"index": 0, "readout": {"error": 0}}]
    data["unmodeled_error"] = {"readout": {"factor": factor}}
    table = Profile.from_dict(data).table
    assert table.qubit(0).readout == (0, 0)
    assert table.qubit(1).readout == metrics.scale_readout((0.02, 0.02), factor)
    assert table.unscaled() == ()


@pytest.mark.parametrize("factor", [0, 0.5, 2, 1e3])
def test_a_gate_error_with_no_valid_power_stays_as_stated_and_is_named(factor: float) -> None:
    pauli = [0.0] * 15
    for label in ("IX", "IZ", "XI"):
        pauli[metrics.PAULI_2Q.index(label)] = 0.1
    assert metrics.unscalable("pauli", pauli, 2) == "it has a negative Pauli-Lindblad rate"
    records = [
        {"gate": "cz", "qubits": [0, 1], "pauli": pauli},
        {"gate": "sx", "qubits": [2], "avg_infidelity": 0.5},
        {"gate": "sx", "qubits": [0], "pauli": [0.5, 0, 0]},
    ]
    table = table_of(calibrations=records, unmodeled_error={"gates": {"factor": factor}})
    assert table.gate("cz", (0, 1)).pauli == tuple(pauli)
    assert table.gate("cz", (1, 0)).pauli == metrics.swap_pauli_2q(pauli)
    assert table.gate("sx", (2,)).avg_infidelity == 0.5
    assert table.gate("sx", (0,)).pauli == (0.5, 0, 0)
    assert table.gate("cz", (1, 2)).avg_infidelity == metrics.scale_avg_infidelity(1e-2, 2, factor)
    assert tuple(p.full for p in table.unscaled()) == (
        "cz on qubits 0-1 is not scaled (it has a negative Pauli-Lindblad rate)",
        "sx on qubits 0 and 2 is not scaled (at or past full depolarization)",
    )
    gates = {**toy()["gates"], "cz": {"avg_infidelity": 0.75}}
    default = table_of(gates=gates, unmodeled_error={"gates": {"factor": factor}})
    assert tuple(p.full for p in default.unscaled()) == (
        "default cz error is not scaled (at or past full depolarization)",
    )


def test_unscaled_leaves_out_disabled_qubits_and_their_records() -> None:
    chance = {"p1_given_0": 0.6, "p0_given_1": 0.5}
    table = table_of(
        readout={"error": 0.02},
        qubits=[{"index": 0, "disabled": True, "readout": chance}, {"index": 2, "readout": chance}],
        calibrations=[{"gate": "sx", "qubits": [0], "avg_infidelity": 0.5}],
        unmodeled_error={"gates": {"factor": 2.0}, "readout": {"factor": 2.0}},
    )
    assert tuple(p.full for p in table.unscaled()) == (
        "readout of qubit 2 is not scaled (no better than chance)",
    )


def test_unscaled_names_a_record_only_when_the_record_leaves_its_gate_enabled() -> None:
    gates = {**toy()["gates"], "cz": {"avg_infidelity": 1e-2, "disabled": True}}
    records = [
        {"gate": "sx", "qubits": [1], "avg_infidelity": 0.6, "disabled": True},
        {"gate": "cz", "qubits": [0, 1], "avg_infidelity": 0.78, "disabled": False},
        {"gate": "cz", "qubits": [1, 2], "avg_infidelity": 0.78},
    ]
    table = table_of(gates=gates, calibrations=records, unmodeled_error={"gates": {"factor": 2.0}})
    assert tuple(p.full for p in table.unscaled()) == (
        "cz on qubits 0-1 is not scaled (at or past full depolarization)",
    )


@pytest.mark.timing
def test_unscaled_is_fast_on_a_fitted_156_qubit_profile() -> None:
    kingston = nv.load("ibm_kingston@2026-04-15")
    factors = {"gates": {"factor": 1.8}, "readout": {"factor": 1.3}}
    table = kingston.model_copy(update={"unmodeled_error": factors}).table
    phrases = table.unscaled()
    start = time.perf_counter()
    for _ in range(10):
        assert table.unscaled() == phrases
    elapsed = time.perf_counter() - start
    assert elapsed < 0.02
