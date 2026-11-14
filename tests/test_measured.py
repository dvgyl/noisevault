"""Circuits that measure a part of their qubits, in any order: the field measured, format 1.1."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

import pytest

import noisevault as nv
from noisevault import CountsError, NoiseVaultError
from noisevault.counts import (
    MeasuredCounts,
    PlannedCircuit,
    load_counts,
    outcome_probabilities,
    simulate,
)
from noisevault.profile import Profile
from noisevault.reference import Op

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "kingston-simulated.counts.json"
RUN_AT = datetime(2026, 4, 16, 9, 30, tzinfo=UTC)
READOUT = 0.1


@cache
def clean() -> Profile:
    """A device with no gate error and a readout error of 0.1 on each qubit."""
    return Profile.uniform(
        "clean",
        technology="superconducting",
        num_qubits=3,
        one_qubit_error=0.0,
        two_qubit_error=0.0,
        readout_error=READOUT,
    )


@cache
def kingston() -> Profile:
    return nv.load("ibm_kingston@2026-04-15")


def routed(measured: tuple[int, ...] | None = (2, 0)) -> PlannedCircuit:
    """x on circuit qubit 0 and cx onto circuit qubit 1: the state is 1 on qubits 0 and 1."""
    ops = (Op("x", (0,)), Op("cx", (0, 1)))
    return PlannedCircuit(name="routed", qubits=(0, 1, 2), ops=ops, measured=measured)


def _file(**circuit: Any) -> dict[str, Any]:
    circuits = [_bell(), _bell(name="other")]
    counts = simulate(kingston(), circuits, shots=100, seed=1, run_at=RUN_AT).to_dict()
    counts["circuits"][0].update(circuit)
    return counts


def _bell(measured: tuple[int, ...] | None = None, name: str = "bell") -> PlannedCircuit:
    ops = (Op("sx", (0,)), Op("sx", (1,)), Op("cz", (0, 1)), Op("sx", (1,)))
    return PlannedCircuit(name=name, qubits=(148, 149, 150), ops=ops, measured=measured)


def _refusal(tmp_path: Path, data: dict[str, Any]) -> CountsError:
    path = tmp_path / "run.counts.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(CountsError) as caught:
        load_counts(path)
    return caught.value


def test_the_forecast_of_a_circuit_sums_over_the_qubit_it_does_not_measure() -> None:
    right, wrong = 1 - READOUT, READOUT
    expected = [right * wrong, right * right, wrong * wrong, wrong * right]
    found = outcome_probabilities(clean(), routed())
    assert found == pytest.approx(expected, abs=1e-12)


def test_the_forecast_of_one_measured_qubit_has_two_outcomes() -> None:
    found = outcome_probabilities(clean(), routed(measured=(1,)))
    assert found == pytest.approx([READOUT, 1 - READOUT], abs=1e-12)


def test_simulate_draws_keys_of_the_measured_qubits_and_the_score_reads_them() -> None:
    counts = simulate(clean(), [routed()], shots=1000, seed=3, run_at=RUN_AT)
    (circuit,) = counts.circuits
    assert {len(key) for key in circuit.counts} == {2}
    assert circuit.counts["01"] > 700
    assert circuit.vector().shape == (4,)
    assert circuit.measured_qubits == (2, 0)
    assert counts.qubits == (2, 0)
    forecast = clean().score(counts)
    assert forecast.ruled_out == 0 and forecast.excess < forecast.excess_noise_95


def test_a_measured_circuit_scores_as_the_same_circuit_with_its_qubits_reordered() -> None:
    swapped = PlannedCircuit(
        name="bell",
        qubits=(149, 148, 150),
        ops=(Op("sx", (1,)), Op("sx", (0,)), Op("cz", (1, 0)), Op("sx", (0,))),
        measured=(0, 1, 2),
    )
    truth = kingston().model_copy(update={"unmodeled_error": {"gates": {"factor": 2.0}}})
    ordered = simulate(truth, [_bell(measured=(1, 0, 2))], shots=4000, seed=5, run_at=RUN_AT)
    plain = MeasuredCounts.model_validate(
        {
            **ordered.to_dict(),
            "circuits": [
                {**swapped.model_dump(mode="json"), "shots": 4000, "counts": c.counts}
                for c in ordered.circuits
            ],
        }
    )
    assert plain.circuits[0].measured is None
    a, b = kingston().score(ordered), kingston().score(plain)
    assert a.log_score == pytest.approx(b.log_score, rel=1e-12)
    first, second = kingston().compare(ordered), kingston().compare(plain)
    assert first.gates.factor == pytest.approx(second.gates.factor, rel=1e-9)
    assert first.readout.factor == pytest.approx(second.readout.factor, rel=1e-9)


def test_compare_fits_counts_of_a_circuit_with_a_qubit_that_it_does_not_measure() -> None:
    truth = kingston().model_copy(update={"unmodeled_error": {"readout": {"factor": 2.0}}})
    circuits = [_bell(measured=(0,)), _bell(measured=(2, 1), name="b")]
    counts = simulate(truth, circuits, shots=20000, seed=2, run_at=RUN_AT)
    found = kingston().compare(counts)
    assert [len(c.qubits) for c in found.circuits] == [3, 3]
    assert found.readout.low < 2.0 < found.readout.high


def test_a_file_of_format_1_0_keeps_its_bytes() -> None:
    counts = load_counts(EXAMPLE)
    assert counts.nv_counts == "1.0"
    assert counts.file_bytes(EXAMPLE) == EXAMPLE.read_bytes()


def test_a_file_of_format_1_1_saves_and_loads_the_same_run(tmp_path: Path) -> None:
    counts = simulate(clean(), [routed()], shots=200, seed=1, run_at=RUN_AT)
    path = counts.save(tmp_path / "run.counts.json")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["nv_counts"] == "1.1"
    assert saved["circuits"][0]["measured"] == [2, 0]
    again = load_counts(path)
    assert again == counts and again.sha256 == counts.sha256


def test_a_file_of_format_1_1_with_no_measured_field_is_format_1_0() -> None:
    data = _file()
    data["nv_counts"] = "1.1"
    assert MeasuredCounts.model_validate(data).nv_counts == "1.0"


def test_measured_with_every_qubit_in_order_is_no_measured_field() -> None:
    circuit = _bell(measured=(0, 1, 2))
    assert circuit.measured is None
    assert "measured" not in circuit.model_dump()


def test_qiskit_bit_order_turns_a_key_of_the_measured_qubits(tmp_path: Path) -> None:
    data = _file(measured=[2, 0], counts={"01": 99, "11": 1})
    data["nv_counts"], data["bit_order"] = "1.1", "qiskit"
    path = tmp_path / "run.counts.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_counts(path).circuits[0].counts == {"10": 99, "11": 1}


@pytest.mark.parametrize(
    ("circuit", "message", "hint"),
    [
        (
            {"measured": []},
            "circuits[0].measured: measured lists no circuit qubit",
            "list the measured circuit qubits, or remove measured to measure every qubit",
        ),
        (
            {"measured": [1, 1]},
            "circuits[0].measured: measured lists circuit qubit 1 twice",
            "the device measures each circuit qubit at most once",
        ),
        (
            {"measured": [3]},
            "circuits[0].measured: measured lists circuit qubit 3, but the circuit has 3 qubits",
            "measured lists circuit qubits, which are indices into qubits",
        ),
        (
            {"measured": [0, 2], "counts": {"000": 100}},
            'circuits[0].counts: key "000" has 3 bits, but the circuit measures 2 qubits',
            "write each key in 0 and 1, one bit per circuit qubit",
        ),
    ],
    ids=["empty", "repeated", "outside", "key width"],
)
def test_each_refusal_of_measured_names_the_field(
    tmp_path: Path, circuit: dict[str, Any], message: str, hint: str | None
) -> None:
    data = _file(**circuit)
    data["nv_counts"] = "1.1"
    error = _refusal(tmp_path, data)
    assert error.message == f"{tmp_path / 'run.counts.json'}: {message}"
    assert error.hint == hint


def test_a_file_of_format_1_0_cannot_have_measured(tmp_path: Path) -> None:
    error = _refusal(tmp_path, _file(measured=[1, 0], counts={"00": 100}))
    assert error.message.endswith('a circuit has measured, which counts format "1.0" does not have')
    assert error.hint == 'set "nv_counts" to "1.1"'


def test_the_reader_names_both_formats_that_it_reads(tmp_path: Path) -> None:
    data = _file()
    data["nv_counts"] = "2.0"
    assert _refusal(tmp_path, data).message.endswith('reads only "1.0" and "1.1"')


def test_the_stabilizer_forecast_refuses_a_circuit_with_measured() -> None:
    pytest.importorskip("stim")
    from noisevault.wide import noisy_circuit

    with pytest.raises(NoiseVaultError, match="sets measured"):
        noisy_circuit(kingston(), _bell(measured=(1, 0)))


def test_the_probe_report_reads_the_bits_in_the_order_of_measured() -> None:
    probe = PlannedCircuit(
        name="axis_d2",
        qubits=(148, 149),
        ops=(Op("x", (0,)), Op("sx", (1,)), Op("sx", (1,)), Op("sx", (1,)), Op("sx", (1,))),
        measured=(1, 0),
    )
    counts = simulate(kingston(), [probe, _bell()], shots=2000, seed=4, run_at=RUN_AT)
    report = kingston().probe_report(counts)
    assert report is not None
    (circuit,) = report.circuits
    assert circuit.ideal_result == "01"
    assert [(q.qubit, q.ideal_bit) for q in circuit.wrong_bits] == [(149, "0"), (148, "1")]
    assert circuit.measured == pytest.approx(circuit.forecast, abs=0.05)


def test_a_disabled_measure_on_a_qubit_that_the_circuit_does_not_measure_is_no_refusal() -> None:
    data = kingston().to_dict()
    data["gates"]["measure"] = {}
    off = {"gate": "measure", "qubits": [150], "disabled": True}
    data["calibrations"] = [*data.get("calibrations", []), off]
    profile = Profile.model_validate(data)
    circuits = [_bell(measured=(0,)), _bell(measured=(1, 0), name="b")]
    counts = simulate(profile, circuits, shots=4000, seed=2, run_at=RUN_AT)
    assert profile.score(counts).ruled_out == 0
    assert profile.compare(counts).gates.factor > 0
    with pytest.raises(NoiseVaultError, match="measure on qubit 150 is disabled"):
        outcome_probabilities(profile, _bell(measured=(2,)))
