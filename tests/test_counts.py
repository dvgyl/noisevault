from __future__ import annotations

import copy
import gzip
import json
from collections.abc import Callable
from datetime import UTC, datetime
from functools import cache, reduce
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest
from conftest import deeper_than_the_parser_takes, require, toy
from pydantic import ValidationError

import noisevault as nv
from noisevault import (
    CountsError,
    DisabledGateError,
    LayoutError,
    NoiseVaultError,
    UnsupportedEffect,
)
from noisevault.counts import (
    SAMPLER_V2_OPTIONS,
    MeasuredCircuit,
    MeasuredCounts,
    PlannedCircuit,
    load_counts,
    plan,
    simulate,
)
from noisevault.profile import Profile
from noisevault.reference import Op, probabilities

KINGSTON = "ibm_kingston@2026-04-15"
RUN_AT = datetime(2026, 4, 16, 9, 30, 2, tzinfo=UTC)
FITTED = {"gates": {"factor": 1.8}, "readout": {"factor": 0.0}}


@cache
def kingston() -> Profile:
    return nv.load(KINGSTON)


def _run(bit_order: str = "clbit0_left") -> dict[str, Any]:
    bell = {"00": 46, "01": 3, "10": 5, "11": 46}
    readout = {"000": 97, "100": 2, "010": 1}
    if bit_order == "qiskit":
        bell = {k[::-1]: v for k, v in bell.items()}
        readout = {k[::-1]: v for k, v in readout.items()}
    return {
        "nv_counts": "1.0",
        "source": "hardware",
        "profile": {"id": "ibm_kingston", "fingerprint": kingston().fingerprint},
        "backend": "ibm_kingston",
        "run_at": "2026-04-16T09:30:02Z",
        "bit_order": bit_order,
        "execution": {
            "client": "qiskit-ibm-runtime 0.49.0 SamplerV2, qiskit 2.5.2",
            "transpiled": False,
            "gate_twirling": False,
            "measure_twirling": False,
            "dynamical_decoupling": False,
            "init_qubits": True,
            "job_ids": ["d3f0c1"],
            "options": {
                "twirling": {"enable_gates": False, "enable_measure": False},
                "dynamical_decoupling": {"enable": False},
                "execution": {"init_qubits": True, "meas_type": "classified"},
            },
        },
        "circuits": [
            {
                "name": "bell",
                "qubits": [148, 149],
                "ops": [
                    ["sx", [0], []],
                    ["delay", [1], [32.0]],
                    ["cz", [0, 1], []],
                    ["sx", [0], []],
                    ["delay", [1], [32]],
                ],
                "shots": 100,
                "counts": bell,
            },
            {
                "name": "readout",
                "qubits": [148, 149, 150],
                "ops": [],
                "shots": 100,
                "counts": readout,
            },
        ],
    }


def _write(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data))
    return path


def test_both_bit_orders_load_to_one_model_with_one_hash_and_one_file(tmp_path: Path) -> None:
    qiskit = _write(tmp_path / "qiskit.counts.json", _run("qiskit"))
    left = _run("clbit0_left")
    left["circuits"][1]["counts"]["111"] = 0
    left = _write(tmp_path / "left.counts.json", left)

    a, b = load_counts(qiskit), load_counts(left)

    assert a == b
    assert a.sha256 == b.sha256
    assert a.bit_order == "clbit0_left"
    assert dict(a.circuits[1].counts) == {"000": 97, "010": 1, "100": 2}
    assert a.save(tmp_path / "a.json").read_bytes() == b.save(tmp_path / "b.json").read_bytes()


def test_a_saved_file_loads_back_equal_and_gzip_is_accepted(tmp_path: Path) -> None:
    counts = load_counts(_write(tmp_path / "run.counts.json", _run("qiskit")))
    saved = counts.save(tmp_path / "saved.counts.json")
    packed = tmp_path / "saved.counts.json.gz"
    packed.write_bytes(gzip.compress(saved.read_bytes()))

    assert load_counts(saved) == counts
    assert load_counts(packed).sha256 == counts.sha256


def _set(*path: str | int, value: Any) -> Callable[[dict[str, Any]], None]:
    def change(data: dict[str, Any]) -> None:
        reduce(lambda node, key: node[key], path[:-1], data)[path[-1]] = value

    return change


def _drop(key: str) -> Callable[[dict[str, Any]], None]:
    return lambda data: data.pop(key)


def _too_many_outcomes(data: dict[str, Any]) -> None:
    wide = {"qubits": list(range(10)), "ops": [], "shots": 1, "counts": {"0" * 10: 1}}
    data["circuits"] = [{"name": f"c{i}", **wide} for i in range(5)]


def _shots(circuit: int, shots: int) -> Callable[[dict[str, Any]], None]:
    return lambda data: data["circuits"][circuit].update(shots=shots, counts={"000": shots})


class _Refusal(NamedTuple):
    change: Callable[[dict[str, Any]], None]
    words: str
    hint: str | None


_REFUSALS: dict[str, _Refusal] = {
    "no nv_counts": _Refusal(
        _drop("nv_counts"),
        "is not a counts file (it has no nv_counts key)",
        'give a counts file, which holds "nv_counts": "1.0"',
    ),
    "another format": _Refusal(
        _set("nv_counts", value="2.0"),
        'is counts format "2.0", and this NoiseVault reads only "1.0"',
        "upgrade NoiseVault to read a newer format",
    ),
    "gate twirling on": _Refusal(
        _set("execution", "gate_twirling", value=True),
        "execution: gate_twirling is true, so the device ran random Pauli gates that the ops do"
        " not list",
        "run the circuits again without gate twirling",
    ),
    "twirling recorded on": _Refusal(
        _set("execution", "options", "twirling", "enable_gates", value=True),
        "execution: options.twirling.enable_gates is true, so the device ran random Pauli gates",
        "SamplerV2 option twirling.enable_gates set to False",
    ),
    "init_qubits false": _Refusal(
        _set("execution", "init_qubits", value=False),
        "execution: init_qubits is false, so a shot may start where the previous one ended,"
        " not from 0",
        "with every qubit reset before each shot",
    ),
    "init_qubits not a bool": _Refusal(
        _set("execution", "init_qubits", value=1),
        "execution: init_qubits is 1. Give true or false",
        None,
    ),
    "kerneled": _Refusal(
        _set("execution", "options", "execution", "meas_type", value="kerneled"),
        'options.execution.meas_type is "kerneled", so the job returned IQ data, not bitstrings',
        'execution.meas_type set to "classified"',
    ),
    "counts short of shots": _Refusal(
        _set("circuits", 0, "counts", "11", value=45),
        "circuits[0]: counts sum to 99, but shots is 100",
        "add up to shots",
    ),
    "more shots than a counts file accepts": _Refusal(
        _shots(1, 10**10 + 1),
        "circuits[1].shots: 10000000001 is more than 10000000000, the most shots that a counts"
        " file accepts for one circuit",
        "split the shots over several circuits with the same qubits and ops",
    ),
    "key of the wrong length": _Refusal(
        _set("circuits", 0, "counts", value={"00": 99, "011": 1}),
        'circuits[0].counts: key "011" has 3 bits, but the circuit measures 2 qubits',
        "write each key in 0 and 1, one bit per circuit qubit",
    ),
    "key not of bits": _Refusal(
        _set("circuits", 0, "counts", value={"00": 99, "0x": 1}),
        'circuits[0].counts: key "0x" holds characters other than 0 and 1',
        "write each key in 0 and 1",
    ),
    "op on an unmeasured qubit": _Refusal(
        _set("circuits", 0, "ops", 2, value=["cz", [1, 2], []]),
        "circuits[0]: ops[2] acts on circuit qubit 2, but the circuit measures only circuit"
        " qubits 0 and 1",
        "circuit qubit i is qubits[i]",
    ),
    "op on one qubit twice": _Refusal(
        _set("circuits", 0, "ops", 2, value=["cz", [1, 1], []]),
        "circuits[0].ops[2]: cz acts on qubits 1-1, but its qubits must be distinct",
        None,
    ),
    "measure in ops": _Refusal(
        _set("circuits", 0, "ops", 3, value=["measure", [0], []]),
        "circuits[0].ops[3]: ops cannot hold measure, because the device measures every"
        " circuit qubit",
        "remove the measure",
    ),
    "gate outside the registry": _Refusal(
        _set("circuits", 0, "ops", 2, value=["sycamore", [0, 1], []]),
        'circuits[0].ops[2]: "sycamore" is not a gate NoiseVault can simulate',
        "use a gate name from the registry, such as sx, cz or rz, or delay",
    ),
    "gate name with accents": _Refusal(
        _set("circuits", 0, "ops", 2, value=["résumé", [0, 1], []]),
        'circuits[0].ops[2]: "résumé" is not a gate NoiseVault can simulate',
        None,
    ),
    "misspelled gate": _Refusal(
        _set("circuits", 0, "ops", 2, value=["iswapp", [0, 1], []]),
        '"iswapp" is not a gate NoiseVault can simulate',
        "did you mean 'iswap'?",
    ),
    "not an op": _Refusal(
        _set("circuits", 0, "ops", 0, value=["sx", 0]),
        'circuits[0].ops[0]: ["sx", 0] is not an op',
        '["rz", [0], [1.5708]]',
    ),
    "more than 4096 outcomes": _Refusal(
        _too_many_outcomes,
        "the circuits have 5120 outcomes in all, more than the 4096 nv compare holds",
        "split the circuits over several counts files",
    ),
    "repeated circuit name": _Refusal(
        _set("circuits", 1, "name", value="bell"),
        'circuits[1] repeats the name "bell"',
        "own name",
    ),
    "delay of NaN": _Refusal(
        _set("circuits", 0, "ops", 1, value=["delay", [1], [float("nan")]]),
        "circuits[0].ops[1]: delay duration nan is not a finite number",
        "nanoseconds, 0 or more",
    ),
    "delay of infinity": _Refusal(
        _set("circuits", 0, "ops", 1, value=["delay", [1], [float("inf")]]),
        "circuits[0].ops[1]: delay duration inf is not a finite number",
        "nanoseconds, 0 or more",
    ),
    "negative delay": _Refusal(
        _set("circuits", 0, "ops", 1, value=["delay", [1], [-5]]),
        "circuits[0].ops[1]: delay duration -5 ns is negative",
        "nanoseconds, 0 or more",
    ),
    "delay on two qubits": _Refusal(
        _set("circuits", 0, "ops", 1, value=["delay", [0, 1], [32.0]]),
        "circuits[0].ops[1]: a delay idles one qubit, not 2",
        "one delay for each qubit",
    ),
    "delay without a duration": _Refusal(
        _set("circuits", 0, "ops", 1, value=["delay", [1], []]),
        "circuits[0].ops[1]: a delay takes one duration, not 0",
        "nanoseconds, 0 or more",
    ),
    "parameter too large for a float": _Refusal(
        _set("circuits", 0, "ops", 0, value=["rz", [0], [10**400]]),
        "circuits[0].ops[0]: rz parameter inf is not a finite number",
        None,
    ),
    "short fingerprint": _Refusal(
        _set("profile", "fingerprint", value="609c845ed934"),
        "profile.fingerprint: not a full fingerprint. Give all 64 hex digits",
        None,
    ),
    "unknown key": _Refusal(
        _set("circuits", 0, "count", value={}),
        "circuits[0].count: not a counts format 1.0 key",
        None,
    ),
    "run_at as a number": _Refusal(
        _set("run_at", value=20260930),
        "run_at: 20260930 is not an ISO 8601 time with a timezone. Give a time such as"
        " 2026-09-30T08:00:00Z",
        None,
    ),
    "run_at as a string of digits": _Refusal(
        _set("run_at", value="20260930"),
        'run_at: "20260930" is not an ISO 8601 time with a timezone',
        None,
    ),
    "surrogate in backend": _Refusal(
        _set("backend", value="ibm_\ud800"),
        "backend: the string holds the unpaired surrogate \\ud800, which UTF-8 cannot encode."
        " Remove the surrogate or write the whole character",
        None,
    ),
    "surrogate in profile id": _Refusal(
        _set("profile", "id", value="\ud800"),
        "profile.id: the string holds the unpaired surrogate \\ud800",
        None,
    ),
    "surrogate in client": _Refusal(
        _set("execution", "client", value="q\udfff"),
        "execution.client: the string holds the unpaired surrogate \\udfff",
        None,
    ),
    "surrogate in a job id": _Refusal(
        _set("execution", "job_ids", 0, value="\ud800"),
        "execution.job_ids[0]: the string holds the unpaired surrogate \\ud800",
        None,
    ),
    "surrogate in a circuit name": _Refusal(
        _set("circuits", 1, "name", value="\ud800"),
        "circuits[1].name: the string holds the unpaired surrogate \\ud800",
        None,
    ),
    "surrogate in an option key": _Refusal(
        _set("execution", "options", "\ud800", value=1),
        "execution.options: the key '\\ud800' holds the unpaired surrogate \\ud800",
        None,
    ),
    "surrogate in a counts key": _Refusal(
        _set("circuits", 0, "counts", value={"00": 99, "\ud800": 1}),
        'circuits[0].counts: key "\\ud800" holds characters other than 0 and 1',
        "write each key in 0 and 1",
    ),
    "surrogate in an op name": _Refusal(
        _set("circuits", 0, "ops", 2, value=["\ud800", [0, 1], []]),
        'circuits[0].ops[2]: "\\ud800" is not a gate NoiseVault can simulate',
        "use a gate name from the registry",
    ),
    "surrogate in an option value": _Refusal(
        _set("execution", "options", "note", value="\ud800"),
        "execution.options: note: the string holds the unpaired surrogate \\ud800",
        None,
    ),
}


@pytest.mark.parametrize("case", list(_REFUSALS))
def test_each_refusal_is_one_counts_error_line_that_starts_with_the_path(
    tmp_path: Path, case: str
) -> None:
    change, words, hint = _REFUSALS[case]
    data = _run()
    change(data)
    path = _write(tmp_path / "run.counts.json", data)

    with pytest.raises(CountsError) as caught:
        load_counts(path)

    text = str(caught.value)
    assert text.startswith(f"{path}")
    assert "\n" not in text and "problems)" not in text
    assert words in caught.value.message
    if hint is not None:
        assert hint in caught.value.hint


def test_a_file_that_is_not_counts_is_named_with_what_to_give(tmp_path: Path) -> None:
    damaged = tmp_path / "run.counts.json"
    damaged.write_text("{not json")
    cut = tmp_path / "cut.counts.json"
    cut.write_text('{"nv_counts": "1.0", "source": "hardw')
    notes = tmp_path / "notes.txt"
    notes.write_text("not counts\n")
    profile = tmp_path / "toy.json"
    profile.write_text(json.dumps(toy()))
    wrong = tmp_path / "run.json.gz"
    wrong.write_bytes(b"\x1f\x8b\x08\x00 cut")

    errors = {}
    for path in (damaged, cut, notes, profile, wrong):
        with pytest.raises(CountsError) as caught:
            load_counts(path)
        errors[path] = caught.value

    assert errors[damaged].message.startswith(f"{damaged} is not JSON (expecting property name")
    assert errors[damaged].hint == "the file is damaged or truncated. Save the counts again"
    assert errors[cut].message == (
        f"{cut} is not JSON (unterminated string starting at line 1, column 32)"
    )
    assert errors[notes].message == f"{notes} is not JSON (expecting value at line 1, column 1)"
    assert errors[notes].hint == "give a counts file (.json or .json.gz)"
    assert errors[profile].message == f"{profile} is a profile, not a counts file"
    assert errors[profile].hint == "nv compare takes the profile first and the counts file second"
    assert errors[wrong].message.startswith(f"{wrong} is a damaged gzip file (")
    assert errors[wrong].hint == "the file is damaged or truncated. Save the counts again"
    assert all("\n" not in str(e) for e in errors.values())


def test_an_outcome_counted_twice_is_refused_before_the_second_count_replaces_the_first(
    tmp_path: Path,
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "compare" / "toy-xx.counts.json"
    text = fixture.read_text()
    path = tmp_path / "twice.counts.json"
    path.write_text(text.replace('"counts": {"0": 3950,', '"counts": {"0": 1, "1": 50, "0": 3950,'))

    with pytest.raises(CountsError) as caught:
        load_counts(path)

    assert (caught.value.message, caught.value.hint) == (
        f"{path} has the key circuits[0].counts['0'] twice",
        "the file is damaged. Save the counts again, or keep one of the two keys",
    )


@pytest.mark.filterwarnings("ignore::pydantic.PydanticDeprecatedSince20")
@pytest.mark.parametrize(
    ("method", "kind"), [("model_validate_json", "json_invalid"), ("parse_raw", "value_error")]
)
def test_a_model_json_method_refuses_an_outcome_counted_twice(method: str, kind: str) -> None:
    text = (Path(__file__).parent / "fixtures" / "compare" / "toy-xx.counts.json").read_text()
    read = getattr(MeasuredCounts, method)

    with pytest.raises(ValidationError) as caught:
        read(text.replace('"counts": {"0": 3950,', '"counts": {"0": 1, "1": 50, "0": 3950,'))

    message = "the key circuits[0].counts['0'] appears twice"
    assert [(e["type"], e["msg"].endswith(message)) for e in caught.value.errors()] == [
        (kind, True)
    ]
    assert read(text) == MeasuredCounts.model_validate(json.loads(text))


def test_a_file_nested_deeper_than_the_parser_takes_is_one_counts_error_line(
    tmp_path: Path,
) -> None:
    nested = deeper_than_the_parser_takes()
    deep = tmp_path / "deep.counts.json"
    deep.write_text(nested)

    with pytest.raises(CountsError) as caught:
        load_counts(deep)

    depth = len(nested) // 2
    assert caught.value.message == (
        f"{deep} is not JSON (nested {depth} levels deep at line 1, column {depth})"
    )
    assert caught.value.hint == "the file is damaged or truncated. Save the counts again"


def test_options_nested_past_64_levels_is_one_counts_error_line(tmp_path: Path) -> None:
    nested: Any = 0
    for _ in range(599):
        nested = [nested]
    data = _run()
    data["execution"]["options"] = {"deep": nested}
    path = _write(tmp_path / "run.counts.json", data)

    with pytest.raises(CountsError) as caught:
        load_counts(path)

    assert caught.value.message == f"{path}: execution.options: nested more than 64 levels deep"


def test_a_file_with_several_problems_names_the_first_and_counts_them(tmp_path: Path) -> None:
    data = _run()
    _set("execution", "gate_twirling", value=True)(data)
    _set("circuits", 0, "shots", value=99)(data)

    with pytest.raises(CountsError) as caught:
        load_counts(_write(tmp_path / "run.counts.json", data))

    assert caught.value.message.endswith(
        "execution: gate_twirling is true, so the device ran random Pauli gates that the ops do"
        " not list (1 of 2 problems)"
    )


def test_the_sampler_options_name_real_qiskit_ibm_runtime_options() -> None:
    options_module = require("qiskit_ibm_runtime.options")
    options = options_module.SamplerOptions()
    for path, value in SAMPLER_V2_OPTIONS.items():
        *parents, leaf = path.split(".")
        setattr(reduce(getattr, parents, options), leaf, value)
        assert reduce(getattr, path.split("."), options) == value


def test_vector_is_a_fresh_copy_that_leaves_the_model_and_its_hash_alone(tmp_path: Path) -> None:
    counts = load_counts(_write(tmp_path / "run.counts.json", _run()))
    before = counts.sha256
    bell = counts.circuits[0]

    vector = bell.vector()
    vector[:] = 0

    assert vector is not bell.vector()
    assert bell.vector().tolist() == [46, 3, 5, 46]
    assert counts.sha256 == before == MeasuredCounts.model_validate(counts.to_dict()).sha256


def test_vector_puts_circuit_qubit_0_first_as_the_reference_does() -> None:
    device = Profile.uniform(
        "ideal", technology="superconducting", num_qubits=2, one_qubit_error=0, two_qubit_error=0
    )
    circuit = PlannedCircuit(name="x0", qubits=(0, 1), ops=(Op("x", (0,)),))

    counts = simulate(device, [circuit], shots=10, seed=0, run_at=RUN_AT)

    probs = probabilities(device, circuit.ops, 2, layout=circuit.qubits)
    assert dict(counts.circuits[0].counts) == {"10": 10}
    assert np.argmax(counts.circuits[0].vector()) == np.argmax(probs) == 2


def test_a_counts_model_revalidates_after_its_hash_is_read(tmp_path: Path) -> None:
    counts = load_counts(_write(tmp_path / "run.counts.json", _run()))
    before = counts.sha256

    again = MeasuredCounts.model_validate(counts)

    assert again == counts and again.sha256 == before


def test_an_update_through_model_copy_is_validated_and_hashed_afresh(tmp_path: Path) -> None:
    counts = load_counts(_write(tmp_path / "run.counts.json", _run()))
    before = counts.sha256

    changed = counts.model_copy(update={"source": "simulated"})

    assert changed.sha256 != before
    assert changed.sha256 == MeasuredCounts.model_validate(changed.to_dict()).sha256
    with pytest.raises(ValidationError):
        counts.model_copy(update={"bit_order": "msb"})


def _timeline(profile: Profile, circuit: PlannedCircuit) -> list[float]:
    clock = [0.0] * len(circuit.qubits)
    for op in circuit.ops:
        if op.name == "delay":
            clock[op.qubits[0]] += op.params[0]
            continue
        starts = {clock[q] for q in op.qubits}
        assert len(starts) == 1, f"{circuit.name}: {op} starts with its qubits out of step"
        found = profile.table.gate(op.name, [circuit.qubits[q] for q in op.qubits])
        for q in op.qubits:
            clock[q] += found.duration_ns or 0.0
    return clock


def test_plan_runs_the_check_circuits_with_every_qubit_busy_or_delayed_to_the_end() -> None:
    profile = kingston()
    expected = profile.check(frameworks=[])
    checked = [c for c in expected.circuits if c.name != "chain_mirror"]

    planned = plan(profile)

    assert [c.name for c in planned] == [c.name for c in checked]
    for circuit, check_circuit in zip(planned, checked, strict=True):
        assert circuit.qubits == tuple(expected.layout[i] for i in range(check_circuit.num_qubits))
        assert tuple(op for op in circuit.ops if op.name != "delay") == check_circuit.ops
        clock = _timeline(profile, circuit)
        assert len(set(clock)) == 1, f"{circuit.name} ends at {clock}"
    mirror = next(c for c in planned if c.name == "mirror")
    idle: dict[int, float] = {}
    for op in mirror.ops:
        if op.name == "delay":
            idle[op.qubits[0]] = idle.get(op.qubits[0], 0.0) + op.params[0]
    assert idle == {0: 136.0, 2: 136.0}


def test_plan_writes_no_delay_for_a_profile_without_durations() -> None:
    planned = plan(nv.load("quantinuum_h1-1"))

    assert [c.name for c in planned] == ["ghz_chain", "mirror", "single_qubit", "readout"]
    assert all(op.name != "delay" for c in planned for op in c.ops)


def test_a_fitted_profile_plans_the_run_of_its_calibration() -> None:
    fitted = kingston().model_copy(update={"unmodeled_error": FITTED})

    assert plan(fitted) == plan(kingston())


def test_simulate_is_deterministic_and_binds_to_the_calibration(tmp_path: Path) -> None:
    fitted = kingston().model_copy(update={"unmodeled_error": FITTED})
    circuits = plan(kingston())

    first = simulate(fitted, circuits, shots=2000, seed=7, run_at=RUN_AT)
    second = simulate(fitted, circuits, shots=2000, seed=7, run_at=RUN_AT)
    stated = simulate(kingston(), circuits, shots=2000, seed=7, run_at=RUN_AT)

    assert first == second and first.sha256 == second.sha256
    assert (
        first.save(tmp_path / "a.json").read_bytes()
        == second.save(tmp_path / "b.json").read_bytes()
    )
    assert first.source == "simulated" and first.backend == "ibm_kingston"
    assert first.profile.fingerprint == kingston().fingerprint != fitted.fingerprint
    assert first.run_at == RUN_AT
    assert first.execution.options["unmodeled_error"] == fitted.to_dict()["unmodeled_error"]
    assert [c.counts for c in first.circuits] != [c.counts for c in stated.circuits]
    assert first.qubits == (148, 149, 150, 151)
    assert load_counts(first.save(tmp_path / "run.json")) == first


def test_simulate_without_run_at_stamps_the_current_utc_time() -> None:
    circuit = PlannedCircuit(name="readout", qubits=(0,), ops=())
    device = Profile.uniform("one", technology="superconducting", num_qubits=1, one_qubit_error=0)

    before = datetime.now(UTC)
    counts = simulate(device, [circuit], shots=1, seed=0)

    assert counts.run_at.tzinfo is not None
    assert before <= counts.run_at <= datetime.now(UTC)


@pytest.mark.parametrize("ref", [KINGSTON, "quantinuum_h1-1"])
def test_loaded_circuits_key_a_cache_as_the_planned_ones_do(tmp_path: Path, ref: str) -> None:
    profile = nv.load(ref)
    circuits = plan(profile)
    counts = simulate(profile, circuits, shots=100, seed=1, run_at=RUN_AT)

    loaded = load_counts(counts.save(tmp_path / "run.json"))

    assert all(isinstance(op, Op) for c in loaded.circuits for op in c.ops)
    assert {(c.qubits, c.ops) for c in loaded.circuits} == {(c.qubits, c.ops) for c in circuits}


def test_plan_refuses_a_profile_with_no_calibrated_native_on_the_chain() -> None:
    gates = {"rz": {"virtual": True}, "sx": {}}
    profile = Profile.model_validate(toy(gates=gates, calibrations=[]))
    calibrate = "use a profile that calibrates a 1-qubit native gate with a known unitary"
    with pytest.raises(NoiseVaultError) as info:
        plan(profile)
    assert info.value.message.endswith("on qubit 0, so there is nothing to run")
    assert info.value.hint == calibrate
    with pytest.raises(NoiseVaultError) as info:
        plan(profile, layout=[1])
    assert info.value.message.endswith("on qubit 1, so there is nothing to run")
    assert info.value.hint == calibrate


@pytest.mark.parametrize("allow", ["exact", "approximate"])
def test_simulate_refuses_an_effect_that_the_profile_does_not_let_it_leave_out(allow: str) -> None:
    effect = {"type": "atom_loss", "on": "readout", "prob": 1e-3, "allow": allow}
    profile = Profile.model_validate(toy(effects=[effect]))
    circuit = PlannedCircuit(name="sx", qubits=(0,), ops=(Op("sx", (0,)),))
    with pytest.raises(UnsupportedEffect) as caught:
        simulate(profile, [circuit], shots=100, seed=0, run_at=RUN_AT)
    assert caught.value.message == (
        f"circuit 'sx': effect atom_loss on readout asks for allow='{allow}', but the reference"
        " simulator does not model effects yet"
    )
    assert caught.value.hint == "set allow to 'omit' to leave the effect out"


@pytest.mark.parametrize("name", ["measure", "delay"])
def test_simulate_refuses_an_op_that_the_profile_disables(name: str) -> None:
    data = toy()
    data["gates"][name] = {"disabled": True}
    ops = (Op("sx", (0,)), Op("delay", (0,), (100.0,)))
    circuit = PlannedCircuit(name="sx", qubits=(1,), ops=ops)
    with pytest.raises(DisabledGateError) as caught:
        simulate(Profile.model_validate(data), [circuit], shots=100, seed=0, run_at=RUN_AT)
    assert caught.value.message == f"circuit 'sx': {name} on qubit 1 is disabled in this profile"
    assert caught.value.hint is None


def test_plan_measures_no_qubit_whose_measurement_the_profile_disables() -> None:
    gates = {**toy()["gates"], "measure": {}}
    off = [{"gate": "measure", "qubits": [0], "disabled": True}]
    profile = Profile.model_validate(toy(gates=gates, calibrations=off))
    assert {c.qubits for c in plan(profile)} == {(1, 2)}
    with pytest.raises(LayoutError, match=r"^test_toy disables measure on qubit 0, but"):
        plan(profile, layout=[0, 1])


def _delay_off(ref: str, qubit: int) -> Profile:
    data = nv.load(ref).to_dict()
    data["gates"]["delay"] = {}
    off = {"gate": "delay", "qubits": [qubit], "disabled": True}
    data["calibrations"] = [*data.get("calibrations", []), off]
    return Profile.model_validate(data)


@pytest.mark.parametrize("qubit", [148, 151])
def test_plan_writes_no_delay_on_a_qubit_where_the_profile_disables_delay(qubit: int) -> None:
    profile = _delay_off(KINGSTON, qubit)
    with pytest.raises(LayoutError) as info:
        plan(profile, layout=[148, 149, 150, 151])
    assert info.value.message == (
        f"circuit 'ghz_chain': qubit {qubit} waits 136 ns, but ibm_kingston disables delay on"
        " that qubit"
    )
    assert info.value.hint == "pass layout= with qubits that allow delay"
    chain = [0, 1, 2, 3]
    assert plan(_delay_off("quantinuum_h1-1", 0), chain) == plan(nv.load("quantinuum_h1-1"), chain)


def test_simulate_refuses_a_shot_count_that_is_not_positive() -> None:
    with pytest.raises(ValueError, match="give a positive number of shots"):
        simulate(kingston(), plan(kingston()), shots=0, seed=0)


def test_a_circuit_of_the_most_shots_loads_simulates_and_one_more_shot_is_refused(
    tmp_path: Path,
) -> None:
    data = _run()
    _shots(1, 10**10)(data)
    one = plan(kingston())[:1]

    loaded = load_counts(_write(tmp_path / "run.counts.json", data))
    drawn = simulate(kingston(), one, shots=10**10, seed=0, run_at=RUN_AT)

    assert loaded.circuits[1].vector()[0] == 10**10
    assert drawn.circuits[0].vector().sum() == 10**10
    with pytest.raises(ValueError, match="give at most 10000000000 shots"):
        simulate(kingston(), one, shots=10**10 + 1, seed=0, run_at=RUN_AT)


def test_the_example_in_the_counts_format_page_loads(tmp_path: Path) -> None:
    page = Path(__file__).parents[1] / "docs" / "counts-format.md"
    example = page.read_text(encoding="utf-8").split("```json\n", 1)[1].split("```", 1)[0]

    counts = load_counts(_write(tmp_path / "example.counts.json", json.loads(example)))

    assert counts.profile.fingerprint == kingston().fingerprint
    assert dict(counts.circuits[1].counts) == {"000": 3923, "001": 34, "010": 39, "100": 4}


def test_circuit_models_given_with_qiskit_bit_order_are_read_in_that_order() -> None:
    data = _run("qiskit")
    circuits = [MeasuredCircuit.model_validate(c) for c in data["circuits"]]

    given = MeasuredCounts.model_validate({**data, "circuits": circuits})

    assert given == MeasuredCounts.model_validate(_run("clbit0_left"))


def test_validating_a_run_leaves_the_given_dict_unchanged() -> None:
    data = _run("qiskit")
    kept = copy.deepcopy(data)

    MeasuredCounts.model_validate(data)

    assert data == kept
