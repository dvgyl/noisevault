from __future__ import annotations

import json
import warnings

import pytest
from conftest import require, toy

import noisevault as nv
from noisevault.errors import LociText, NoiseApproximationWarning, UnsupportedEffect
from noisevault.profile import Profile
from noisevault.report import HONESTY, Clamp, Report


def _report(**sections) -> tuple[Profile, Report]:
    profile = Profile.model_validate(toy(**sections))
    report = Report.start(profile, "qiskit", "2.5.2", layout={"a": 0}, unknown_gates="typical")
    return profile, report


def test_effects_are_omitted_by_default() -> None:
    profile, report = _report(effects=[{"type": "leakage", "gate": "cz", "prob": 1e-4}])
    report.record_effects(profile.effects)
    assert report.to_dict()["omitted"] == ["effect leakage on cz"]


@pytest.mark.parametrize("allow", ["approximate", "exact"])
def test_effects_that_must_be_modeled_refuse_conversion(allow: str) -> None:
    profile, report = _report(
        effects=[{"type": "atom_loss", "on": "readout", "prob": 1e-3, "allow": allow}]
    )
    with pytest.raises(UnsupportedEffect, match="atom_loss") as caught:
        report.record_effects(profile.effects)
    assert caught.value.hint == "set allow to 'omit' to export without the effect"


def test_warn_once_per_key() -> None:
    _, report = _report()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            report.warn_once("h", "h is not native; typical noise used")
        report.warn_once("ry", "ry is not native; typical noise used")
    assert [w.category for w in caught] == [NoiseApproximationWarning] * 2


def test_report_serializes_and_summarizes() -> None:
    profile, report = _report()
    report.mark_exact("gate errors")
    report.approximate("readout", "symmetrized", "mean of P(1|0) and P(0|1)")
    report.approximate("readout", "symmetrized", "mean of P(1|0) and P(0|1)")
    report.mark_unknown("prep on qubit 0")
    report.count("typical_noise_used", "h", 3)
    report.count("typical_noise_used", "h")
    data = json.loads(json.dumps(report.to_dict()))
    assert data["fingerprint"] == profile.fingerprint
    assert data["options"] == {"layout": {"a": 0}, "unknown_gates": "typical"}
    assert data["events"] == {"typical_noise_used": {"h": 4}}
    assert len(data["approximated"]) == 1
    text = report.summary()
    assert profile.id in text and text.endswith(HONESTY)
    assert "used: h took the typical native gate's noise 4 times" in text


def test_summary_states_each_count_as_a_sentence() -> None:
    _, report = _report()
    report.count("typical_noise_used", "cx", 40)
    report.count("typical_noise_used", "h")
    report.count("reversed_record_used", "cz", 20)
    report.count("circuit_channel_kept", "BitFlipChannel", 2)
    report.count("future_event", "q3", 5)
    used = [line for line in report.summary().splitlines() if line.startswith("used:")]
    assert used == [
        "used: cx took the typical native gate's noise 40 times;"
        " h took the typical native gate's noise once;"
        " cz took the calibration recorded for the opposite qubit order 20 times;"
        " the export kept the circuit's own BitFlipChannel as written, with no noise added,"
        " 2 times;"
        " future event: q3 5 times"
    ]
    assert "=" not in used[0]
    assert report.to_dict()["events"]["reversed_record_used"] == {"cz": 20}


def test_clamps_are_counted_as_one_gate_or_several_gates() -> None:
    _, report = _report()
    report.clamped.append(Clamp("cz", (0, 1), 0.003, 0.0039))
    assert "clamped: 1 gate noisier than stated because" in report.summary()
    report.clamped.append(Clamp("cz", (1, 2), 0.003, 0.0041))
    report.clamped.append(Clamp("sx", (0,), 0.01, 0.002))
    lines = report.summary().splitlines()
    assert (
        "clamped: 2 gates noisier than stated because relaxation alone exceeds the stated error."
        " The largest is cz on qubits 1-2, 0.003 -> 0.0041" in lines
    )
    assert (
        "clamped: 1 gate less noisy than stated because relaxation plus the strongest depolarizing"
        " noise stays below the stated error. The largest is sx on qubit 0, 0.01 -> 0.002" in lines
    )


def test_a_saved_report_names_every_locus_as_plain_text() -> None:
    _, report = _report()
    pairs = [(q, q + 1) for q in range(6)]
    report.omit(LociText("native cz: ", LociText("no metric on ", pairs, ", so none")))
    report.mark_unknown(LociText("readout error of ", [(5,)]))
    saved = report.to_dict()
    assert saved["omitted"] == [
        "native cz: no metric on qubits 0-1, 1-2, 2-3, 3-4, 4-5 and 5-6, so none"
    ]
    assert saved["unknown"] == ["readout error of qubit 5"]
    assert {type(text) for text in saved["omitted"] + saved["unknown"]} == {str}
    lines = report.summary().splitlines()
    assert "omitted: native cz: no metric on qubits 0-1, 1-2, 2-3 and 3 more, so none" in lines
    assert "unknown (no noise applied): readout error of qubit 5" in lines


def test_a_saved_report_names_every_qubit_that_the_unmodeled_factors_leave_as_stated() -> None:
    device = toy()["device"] | {"num_qubits": 6}
    profile = Profile.model_validate(
        toy(device=device, readout={"error": 0.5}, unmodeled_error={"readout": {"factor": 1.3}})
    )
    report = Report.start(profile, "qiskit", "2.5.2")
    stated = "readout errors x1.3; T1, T2 and preparation error are not scaled; readout of"
    assert report.to_dict()["unmodeled_error"] == (
        f"{stated} qubits 0, 1, 2, 3, 4 and 5 is not scaled (no better than chance)"
    )
    assert type(report.to_dict()["unmodeled_error"]) is str
    assert report.summary().splitlines()[1] == (
        f"unmodeled error: {stated} qubits 0, 1, 2 and 3 more is not scaled (no better than chance)"
    )


def test_calibration_qualifiers_of_used_gates_are_reported() -> None:
    from noisevault.channels import gate_channels

    gates = toy()["gates"] | {
        "x": {
            "avg_infidelity": 1e-3,
            "scope": "cycle",
            "includes": ["spam", "leakage"],
            "statistic": "median",
            "assumption": "the vendor number is read as average gate fidelity",
        }
    }
    profile, report = _report(gates=gates)
    table = profile.table
    for name in ("x", "sx"):
        report.record_channels(gate_channels(table.gate(name, (0,)), [table.qubit(0)]))
    data = report.to_dict()
    assert {a["what"] for a in data["approximated"]} == {"x error"}
    text = json.dumps(data["approximated"])
    for expected in ("cycle", "measurement", "leakage", "median", "read as average gate fidelity"):
        assert expected in text
    assert "x error" in report.summary()


def test_included_errors_are_named_in_plain_words() -> None:
    from noisevault.channels import gate_channels

    gates = toy()["gates"] | {
        "x": {"avg_infidelity": 1e-3, "includes": ["spam", "leakage", "1q_dressing"]}
    }
    profile, report = _report(gates=gates)
    report.record_channels(gate_channels(profile.table.gate("x", (0,)), [profile.table.qubit(0)]))
    approximated = [line for line in report.summary().splitlines() if "x error" in line]
    assert approximated == [
        "approximated: x error: the stated error already includes single-qubit gate error"
        " (explicit single-qubit gates in the circuit add their own error on top)",
        "approximated: x error: the stated error already includes leakage"
        " (applied as depolarizing noise, so no population leaves the qubit)",
        "approximated: x error: the stated error already includes state preparation and"
        " measurement error (readout and preparation noise, where applied, add their own error"
        " on top)",
    ]


def test_a_report_states_unmodeled_error_on_its_second_line_only_when_the_profile_has_it() -> None:
    readout = {"error": 0.02}
    base, plain = _report(readout=readout)
    fitted, report = _report(
        readout=readout, unmodeled_error={"gates": {"factor": 2.3}, "readout": {"factor": 1.5}}
    )
    note = "gate errors x2.3; readout errors x1.5; T1, T2 and preparation error are not scaled"
    lines = report.summary().splitlines()
    assert lines[1] == f"unmodeled error: {note}"
    assert report.to_dict()["unmodeled_error"] == note
    assert "unmodeled_error" not in plain.to_dict()
    assert plain.summary() == "\n".join(lines[:1] + lines[2:]).replace(
        fitted.fingerprint[:12], base.fingerprint[:12]
    )
    kingston = nv.load("ibm_kingston@2026-04-15")
    scaled = kingston.model_copy(update={"unmodeled_error": {"readout": {"factor": 1.58}}})
    assert Report.start(scaled, "stim", None).to_dict()["unmodeled_error"] == (
        "readout errors x1.58; T1, T2 and preparation error are not scaled;"
        " readout of qubit 146 is not scaled (no better than chance)"
    )


def _options(**options) -> dict:
    report = Report.start(Profile.model_validate(toy()), "pennylane", None, **options)
    return json.loads(json.dumps(report.to_dict()))["options"]


def test_keys_that_stay_distinct_as_strings_serialize_as_an_object() -> None:
    assert _options(layout={0: 2, 1: 0}) == {"layout": {"0": 2, "1": 0}}
    assert _options(layout={"a": 0, ("b", 1): 1}) == {"layout": {"a": 0, "('b', 1)": 1}}


def test_keys_that_collide_as_strings_serialize_as_pairs() -> None:
    assert _options(layout={0: 0, "0": 1}) == {"layout": [[0, 0], ["0", 1]]}
    nested = _options(extra={"layout": {1: 0, "1": 2}, "names": {"x": 1}})
    assert nested == {"extra": {"layout": [[1, 0], ["1", 2]], "names": {"x": 1}}}


def test_pennylane_report_keeps_integer_and_string_wires_apart() -> None:
    require("pennylane")
    model = Profile.model_validate(toy()).to_pennylane(layout={0: 0, "0": 1})
    data = json.loads(json.dumps(model.report.to_dict()))
    assert data["options"]["layout"] == [[0, 0], ["0", 1]]
