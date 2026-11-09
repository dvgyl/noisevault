from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import pytest
from conftest import toy

import noisevault as nv
from noisevault.diff import describe_delta, diff, fmt_metric
from noisevault.profile import Profile


def _profile(**changes: Any) -> Profile:
    """A 3-qubit toy with per-qubit T1 and readout and per-pair cz records."""
    data = toy(
        device={
            "name": "toy",
            "vendor": "test",
            "technology": "superconducting",
            "num_qubits": 3,
            "calibrated_at": "2025-01-01T00:00:00Z",
        },
        idle={"t1_us": 100, "t2_us": 80},
        readout={"error": 0.01},
        qubits=[{"index": 0, "t1_us": 100}, {"index": 1, "t1_us": 120}, {"index": 2, "t1_us": 90}],
        calibrations=[
            {"gate": "cz", "qubits": [0, 1], "avg_infidelity": 0.01},
            {"gate": "cz", "qubits": [1, 2], "avg_infidelity": 0.02},
        ],
    )
    for key, value in changes.items():
        data[key] = value
    return Profile.model_validate(data)


def _later(**changes: Any) -> Profile:
    device = {
        **toy()["device"],
        "calibrated_at": "2025-01-03T06:00:00Z",
        **changes.pop("device", {}),
    }
    return _profile(device=device, **changes)


def test_medians_compare_device_wide_values() -> None:
    after = _later(
        qubits=[{"index": 0, "t1_us": 50}, {"index": 1, "t1_us": 120}, {"index": 2, "t1_us": 90}],
        calibrations=[
            {"gate": "cz", "qubits": [0, 1], "avg_infidelity": 0.03},
            {"gate": "cz", "qubits": [1, 2], "avg_infidelity": 0.02},
        ],
    )
    result = diff(_profile(), after)
    medians = {c.metric: c for c in result.medians}
    assert (medians["t1_us"].before, medians["t1_us"].after) == (100, 90)
    assert medians["t1_us"].relative == pytest.approx(-0.1) and medians["t1_us"].worse
    assert medians["error_2q"].before == pytest.approx(0.015)
    assert medians["error_2q"].after == pytest.approx(0.025) and medians["error_2q"].worse
    assert medians["readout_error"].relative == 0 and not medians["readout_error"].worse
    assert result.time_delta == timedelta(days=2, hours=6)
    assert not result.warnings and not result.identical


def test_largest_changes_come_first_by_relative_size() -> None:
    after = _later(
        qubits=[{"index": 0, "t1_us": 50}, {"index": 1, "t1_us": 132}, {"index": 2, "t1_us": 90}],
        calibrations=[
            {"gate": "cz", "qubits": [0, 1], "avg_infidelity": 0.011},
            {"gate": "cz", "qubits": [1, 2], "avg_infidelity": 0.08},
        ],
    )
    result = diff(_profile(), after, top=2)
    assert [(c.where, c.metric) for c in result.qubits] == [("0", "t1_us"), ("1", "t1_us")]
    assert result.qubits[0].relative == pytest.approx(-0.5)
    assert [(c.where, c.relative) for c in result.pairs] == [
        ("1-2", pytest.approx(3.0)),
        ("0-1", pytest.approx(0.1)),
    ]
    assert diff(_profile(), after, top=0).qubits == ()


def test_disabled_and_reenabled_gates_and_qubits_are_listed() -> None:
    before = _profile(
        qubits=[{"index": 0, "t1_us": 100}, {"index": 2, "disabled": True}],
    )
    after = _later(
        qubits=[{"index": 0, "t1_us": 100}],
        calibrations=[
            {"gate": "cz", "qubits": [0, 1], "disabled": True},
            {"gate": "cz", "qubits": [1, 2], "avg_infidelity": 0.02},
        ],
    )
    result = diff(before, after)
    assert result.newly_disabled == ("cz 0-1",)
    assert result.reenabled == ("qubit 2",)
    assert diff(after, before).newly_disabled == ("qubit 2",)
    assert diff(after, before).reenabled == ("cz 0-1",)


def _directed(edge: list[int], *, disabled: bool) -> Profile:
    record = {"gate": "ecr", "qubits": edge}
    record |= {"disabled": True} if disabled else {"avg_infidelity": 0.01}
    return Profile.model_validate(
        toy(
            connectivity={"edges": [edge, [1, 2]], "directed": True},
            gates={"rz": {"virtual": True}, "sx": {"avg_infidelity": 1e-3}, "ecr": {}},
            calibrations=[record, {"gate": "ecr", "qubits": [1, 2], "avg_infidelity": 0.01}],
        )
    )


def test_a_directed_gate_that_changed_direction_is_not_re_enabled() -> None:
    before, after = _directed([0, 1], disabled=True), _directed([1, 0], disabled=False)
    assert before.table.gate("ecr", (0, 1)).state == "disabled"
    assert after.table.gate("ecr", (1, 0)).state == "calibrated"
    result = diff(before, after)
    assert result.reenabled == () and result.newly_disabled == ()
    assert diff(before, _directed([0, 1], disabled=False)).reenabled == ("ecr 0-1",)
    assert diff(after, _directed([1, 0], disabled=True)).newly_disabled == ("ecr 1-0",)


def test_a_change_shared_by_every_qubit_is_one_row() -> None:
    def device(one: float, two: float) -> Profile:
        return Profile.uniform(
            "u", technology="trapped_ion", num_qubits=4, one_qubit_error=one, two_qubit_error=two
        )

    result = diff(device(1e-4, 1e-3), device(2e-4, 1e-3), top=5)
    assert [(c.where, c.metric) for c in result.qubits] == [("all", "error_1q")]
    assert result.qubits[0].relative == pytest.approx(1.0) and result.pairs == ()
    assert "all qubits     1q avg infidelity 1.00e-04 -> 2.00e-04 (+100.0%)" in result.summary()


def test_summary_lines_up_the_values_of_every_metric() -> None:
    summary = nv.load("ibm_fez").diff(nv.load("ibm_marrakesh"), top=5).summary()
    rows = [line for line in summary.splitlines() if line.startswith(("  qubit ", "  pair "))]
    labels = {"readout error", "1q avg infidelity", "2q avg infidelity"}
    assert all(any(label in row for row in rows) for label in labels)
    assert len({row.index(" -> ") for row in rows}) == 1, rows


def test_added_and_removed_qubits_and_other_devices_are_flagged() -> None:
    bigger = _later(device={"name": "other", "num_qubits": 5})
    result = diff(_profile(), bigger)
    assert result.qubits_added == (3, 4) and result.qubits_removed == ()
    assert diff(bigger, _profile()).qubits_removed == (3, 4)
    assert "from different devices (test_toy and test_other)" in result.warnings[0]
    assert diff(bigger, _profile()).warnings == (
        "the two profiles are from different devices (test_other and test_toy), so the diff"
        " matches qubits and pairs by index",
    )


def test_an_older_second_calibration_of_one_device_is_flagged_with_its_time() -> None:
    older = _later(device={"calibrated_at": "2024-12-31T06:00:00Z"})
    assert diff(_profile(), older).warnings == (
        "test_toy@2024-12-31 is older than test_toy@2025-01-01. Before and after follow the"
        " argument order, not time",
    )
    same_day = _later(device={"calibrated_at": "2025-01-01T06:00:00Z"})
    assert diff(same_day, _profile()).warnings == (
        "test_toy@2025-01-01T00:00Z is older than test_toy@2025-01-01T06:00Z. Before and after"
        " follow the argument order, not time",
    )
    assert diff(_profile(), same_day).warnings == ()


def test_microseconds_print_with_one_decimal_in_tables() -> None:
    values = (190, 99.98, 48.806, None)
    assert [fmt_metric("t1_us", v) for v in values] == ["190.0", "100.0", "48.8", "-"]
    assert fmt_metric("t2_us", 1e7) == "10000000.0"
    assert fmt_metric("error_1q", 2.29e-4) == "2.29e-04"


def test_identical_profiles_say_so() -> None:
    result = diff(_profile(), _profile())
    assert result.identical and result.time_delta == timedelta(0)
    assert "same physics" in result.summary()


def test_text_and_dict_output() -> None:
    a, b = nv.load("ibm_kyiv"), nv.load("ibm_brisbane")
    result = a.diff(b, top=3)
    data = json.loads(json.dumps(result.to_dict()))
    assert data["before"] == "ibm_kyiv@2025-02-26" and len(data["qubits"]) == 3
    assert {m["metric"] for m in data["medians"]} == {
        "t1_us",
        "t2_us",
        "error_1q",
        "error_2q",
        "readout_error",
    }
    text = str(result)
    assert text.splitlines()[0].startswith("ibm_kyiv@2025-02-26 -> ibm_brisbane@2025-02-26")
    assert (
        "largest changes by qubit:" in text
        and "warning: the two profiles are from different devices" in text
    )


def test_negative_top_is_refused() -> None:
    with pytest.raises(ValueError, match="top=-1"):
        diff(_profile(), _profile(), top=-1)


@pytest.mark.parametrize(
    ("delta", "text"),
    [
        (None, "calibration time unknown"),
        (timedelta(0), "same calibration time"),
        (timedelta(days=1, hours=3), "1 day 3 hours later"),
        (-timedelta(minutes=23), "23 minutes earlier"),
        (timedelta(seconds=20), "under a minute later"),
    ],
)
def test_time_deltas_read_plainly(delta, text) -> None:
    assert describe_delta(delta) == text


def _cx(error_10: float) -> Profile:
    return Profile.model_validate(
        toy(
            connectivity={"edges": [[0, 1], [1, 0]], "directed": True},
            gates={"rz": {"virtual": True}, "sx": {"avg_infidelity": 1e-3}, "cx": {}},
            calibrations=[
                {"gate": "cx", "qubits": [0, 1], "avg_infidelity": 0.01},
                {"gate": "cx", "qubits": [1, 0], "avg_infidelity": error_10},
            ],
        )
    )


def test_each_direction_of_a_directed_gate_is_compared() -> None:
    result = diff(_cx(0.02), _cx(0.2))
    assert [(c.where, c.before, c.after) for c in result.pairs] == [("1-0", 0.02, 0.2)]
    medians = {c.metric: c for c in result.medians}
    assert medians["error_2q"].before == pytest.approx(0.015)
    assert medians["error_2q"].after == pytest.approx(0.105)


def _ions(default: float) -> Profile:
    return Profile.model_validate(
        toy(
            connectivity="all_to_all",
            gates={
                "rz": {"virtual": True},
                "sx": {"avg_infidelity": 1e-3},
                "cz": {"avg_infidelity": default},
            },
            calibrations=[{"gate": "cz", "qubits": [0, 1], "avg_infidelity": 0.02}],
        )
    )


def test_all_to_all_pairs_on_the_device_default_are_compared() -> None:
    result = diff(_ions(0.01), _ions(0.1))
    assert [(c.where, c.before, c.after) for c in result.pairs] == [("default", 0.01, 0.1)]
    medians = {c.metric: c for c in result.medians}
    assert (medians["error_2q"].before, medians["error_2q"].after) == (0.01, 0.1)


def test_changes_from_zero_or_to_a_missing_value_are_listed_first() -> None:
    def device(readout_0: float, t1_0: float | None, t1_1: float) -> Profile:
        qubit_0 = {"index": 0, "readout": {"error": readout_0}}
        qubit_0 |= {} if t1_0 is None else {"t1_us": t1_0}
        return _profile(qubits=[qubit_0, {"index": 1, "t1_us": t1_1}], idle=None)

    result = diff(device(0.0, 50, 100), device(0.02, None, 120))
    rows = [(c.where, c.metric, c.before, c.after) for c in result.qubits]
    assert rows == [
        ("0", "readout_error", 0.0, 0.02),
        ("0", "t1_us", 50, None),
        ("1", "t1_us", 100, 120),
    ]
    assert result.qubits[0].worse and not result.qubits[1].worse


def test_a_disabled_gate_definition_is_listed() -> None:
    def device(disabled: bool) -> Profile:
        cz = {"avg_infidelity": 1e-2} | ({"disabled": True} if disabled else {})
        return Profile.model_validate(
            toy(gates={"rz": {"virtual": True}, "sx": {"avg_infidelity": 1e-3}, "cz": cz})
        )

    assert diff(device(False), device(True)).newly_disabled == ("cz default",)
    assert diff(device(True), device(False)).reenabled == ("cz default",)


def test_a_lost_pair_calibration_is_drift_but_a_disabled_pair_is_availability() -> None:
    def device(*records: dict) -> Profile:
        gates = {"rz": {"virtual": True}, "sx": {"avg_infidelity": 1e-3}, "cz": {}}
        cz12 = {"gate": "cz", "qubits": [1, 2], "avg_infidelity": 0.02}
        return Profile.model_validate(toy(gates=gates, calibrations=[*records, cz12]))

    calibrated = device({"gate": "cz", "qubits": [0, 1], "avg_infidelity": 0.01})
    lost = diff(calibrated, device())
    assert [(c.where, c.before, c.after) for c in lost.pairs] == [("0-1", 0.01, None)]
    disabled = diff(calibrated, device({"gate": "cz", "qubits": [0, 1], "disabled": True}))
    assert disabled.pairs == () and disabled.newly_disabled == ("cz 0-1",)


def _enabled_by_records(*pairs: tuple[int, int]) -> Profile:
    cz = {"avg_infidelity": 0.01, "disabled": True}
    records = [{"gate": "cz", "qubits": list(p), "disabled": False} for p in pairs]
    gates = {"rz": {"virtual": True}, "sx": {"avg_infidelity": 1e-3}, "cz": cz}
    return Profile.model_validate(toy(connectivity="all_to_all", gates=gates, calibrations=records))


def test_removing_an_enabling_record_exposes_the_disabled_default() -> None:
    both, one = _enabled_by_records((0, 1), (0, 2)), _enabled_by_records((0, 2))
    assert diff(both, one).newly_disabled == ("cz 0-1",)
    assert diff(one, both).reenabled == ("cz 0-1",)
    assert diff(both, one).reenabled == () and diff(one, both).newly_disabled == ()


def test_a_symmetric_pair_disabled_by_its_reversed_record_is_listed_once() -> None:
    reversed_off = _profile(
        calibrations=[
            {"gate": "cz", "qubits": [1, 0], "disabled": True},
            {"gate": "cz", "qubits": [1, 2], "avg_infidelity": 0.02},
        ]
    )
    assert diff(_profile(), reversed_off).newly_disabled == ("cz 0-1",)


def test_a_pair_calibrated_outside_the_connectivity_is_compared() -> None:
    def device(error: float) -> Profile:
        record = {"gate": "cz", "qubits": [1, 2], "avg_infidelity": error}
        return _profile(connectivity={"edges": [[0, 1]]}, calibrations=[record])

    result = diff(device(0.02), device(0.2))
    assert [(c.where, c.before, c.after) for c in result.pairs] == [("1-2", 0.02, 0.2)]
    medians = {c.metric: c for c in result.medians}
    assert medians["error_2q"].before == pytest.approx(0.015)
    assert medians["error_2q"].after == pytest.approx(0.105)


def _both_orders(record_01: dict, record_10: dict) -> Profile:
    """All-to-all, cz defaulting to 0.01, and a separate cz record for each order of (0, 1)."""
    gates = {
        "rz": {"virtual": True},
        "sx": {"avg_infidelity": 1e-3},
        "cz": {"avg_infidelity": 0.01},
    }
    records = [
        {"gate": "cz", "qubits": [0, 1], **record_01},
        {"gate": "cz", "qubits": [1, 0], **record_10},
    ]
    return Profile.model_validate(toy(connectivity="all_to_all", gates=gates, calibrations=records))


ON, OFF = {"avg_infidelity": 0.01}, {"disabled": True}


def test_each_order_of_a_symmetric_pair_with_its_own_record_is_compared() -> None:
    calibrated, reverse_off = _both_orders(ON, {"avg_infidelity": 0.02}), _both_orders(ON, OFF)
    assert reverse_off.table.gate("cz", (1, 0)).state == "disabled"
    assert reverse_off.table.gate("cz", (0, 1)).state == "calibrated"
    there, back = diff(calibrated, reverse_off), diff(reverse_off, calibrated)
    assert [(r.newly_disabled, r.reenabled) for r in (there, back)] == [
        (("cz 1-0",), ()),
        ((), ("cz 1-0",)),
    ]
    assert there.pairs == ()

    swapped = diff(_both_orders(ON, OFF), _both_orders(OFF, ON))
    assert (swapped.newly_disabled, swapped.reenabled) == (("cz 0-1",), ("cz 1-0",))

    drifted = diff(calibrated, _both_orders(ON, {"avg_infidelity": 0.05}))
    assert [(c.where, c.before, c.after) for c in drifted.pairs] == [("1-0", 0.02, 0.05)]


def test_both_orders_of_a_symmetric_pair_moving_together_are_one_entry() -> None:
    on, off = _both_orders(ON, ON), _both_orders(OFF, OFF)
    assert diff(on, off).newly_disabled == ("cz 0-1",)
    assert diff(off, on).reenabled == ("cz 0-1",)
    drifted = diff(on, _both_orders({"avg_infidelity": 0.03}, {"avg_infidelity": 0.03}))
    assert [(c.where, c.before, c.after) for c in drifted.pairs] == [("0-1", 0.01, 0.03)]


def test_medians_are_the_device_values_that_diff_compares() -> None:
    profile = _profile()

    medians = profile.medians()
    assert medians == pytest.approx(
        {"t1_us": 100, "t2_us": 80, "error_1q": 1e-3, "error_2q": 0.015, "readout_error": 0.01}
    )
    assert json.loads(json.dumps(medians)) == medians
    shown = {c.metric: c.before for c in diff(profile, _later()).medians}
    assert shown == medians


def test_medians_leave_out_a_disabled_qubit_and_its_pairs() -> None:
    qubits = [
        {"index": 0, "t1_us": 100},
        {"index": 1, "t1_us": 120},
        {"index": 2, "disabled": True},
    ]

    medians = _profile(qubits=qubits).medians()
    assert medians["t1_us"] == pytest.approx(110)
    assert medians["error_2q"] == pytest.approx(0.01)


def test_medians_include_the_unmodeled_error_factors_as_the_exports_do() -> None:
    profile = _profile(unmodeled_error={"gates": {"factor": 2.0}, "readout": {"factor": 2.0}})

    stated = profile.uncorrected().medians()
    scaled = profile.medians()
    assert stated == _profile().medians()
    assert scaled["error_1q"] > 1.9 * stated["error_1q"]
    assert scaled["error_2q"] > 1.9 * stated["error_2q"]
    assert scaled["readout_error"] > 1.9 * stated["readout_error"]
    assert scaled["t1_us"] == stated["t1_us"]
    assert scaled["t2_us"] == stated["t2_us"]


def test_the_readout_median_is_the_median_of_each_qubit_mean_of_both_errors() -> None:
    pairs = [(0.01, 0.05), (0.02, 0.06), (0.10, 0.30)]  # per-qubit means 0.03, 0.04, 0.20
    qubits = [
        {"index": i, "readout": {"p1_given_0": a, "p0_given_1": b}}
        for i, (a, b) in enumerate(pairs)
    ]

    assert _profile(qubits=qubits).medians()["readout_error"] == pytest.approx(0.04)


def test_a_median_is_none_when_no_qubit_or_pair_has_the_value() -> None:
    bare = Profile.uniform("bare", technology="other", num_qubits=1, one_qubit_error=1e-3)

    assert bare.medians() == {
        "t1_us": None,
        "t2_us": None,
        "error_1q": pytest.approx(1e-3),
        "error_2q": None,
        "readout_error": None,
    }
