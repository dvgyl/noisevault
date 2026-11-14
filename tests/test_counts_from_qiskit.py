"""Counts of the user's own Qiskit circuits: counts_from_qiskit, from_ibm_job and nv pull --job."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from functools import cache
from math import pi
from typing import Any

import pytest

pytest.importorskip("qiskit")

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import U3Gate

import noisevault as nv
from noisevault import CountsError
from noisevault.counts import MeasuredCounts, simulate
from noisevault.frameworks.qiskit import planned_circuit
from noisevault.profile import Profile

RUN_AT = datetime(2026, 4, 16, 9, 30, 2, tzinfo=UTC)
DEVICE_QUBITS = 156


@cache
def kingston() -> Profile:
    return nv.load("ibm_kingston@2026-04-15")


@cache
def device() -> Profile:
    """The device: the same calibration, with 1.8 times the gate errors and 1.3 times the
    readout errors."""
    factors = {"gates": {"factor": 1.8}, "readout": {"factor": 1.3}}
    return kingston().model_copy(update={"unmodeled_error": factors})


def ghz(chain: list[int], measured: list[int], name: str = "ghz") -> QuantumCircuit:
    qc = QuantumCircuit(DEVICE_QUBITS, len(measured), name=name)
    qc.rz(pi / 2, chain[0])
    qc.sx(chain[0])
    qc.rz(pi / 2, chain[0])
    for a, b in zip(chain, chain[1:], strict=False):
        qc.rz(pi / 2, b)
        qc.sx(b)
        qc.rz(pi / 2, b)
        qc.cz(a, b)
        qc.rz(pi / 2, b)
        qc.sx(b)
        qc.rz(pi / 2, b)
    for i, q in enumerate(measured):
        qc.measure(q, i)
    return qc


def mirror(qubits: list[int], depth: int, name: str = "mirror") -> QuantumCircuit:
    qc = QuantumCircuit(DEVICE_QUBITS, len(qubits), name=name)
    pairs = list(zip(qubits, qubits[1:], strict=False))
    for _ in range(depth):
        for q in qubits:
            qc.sx(q)
        for a, b in pairs:
            qc.cz(a, b)
    for _ in range(depth):
        for a, b in reversed(pairs):
            qc.cz(a, b)
        for q in qubits:
            for _ in range(3):
                qc.sx(q)
    qc.measure(qubits[::-1], range(len(qubits)))
    return qc


def device_counts(circuits: list[QuantumCircuit], shots: int = 4000) -> list[dict[str, int]]:
    """Counts of the device as Qiskit gives them, classical bit 0 rightmost."""
    drawn = simulate(device(), [planned_circuit(c) for c in circuits], shots=shots, seed=11)
    return [{key[::-1]: n for key, n in c.counts.items()} for c in drawn.circuits]


def one(circuit: QuantumCircuit, counts: dict[str, int], **run: Any) -> MeasuredCounts:
    return nv.counts_from_qiskit(circuit, counts, profile=kingston(), run_at=RUN_AT, **run)


def test_the_circuit_qubits_are_the_used_device_qubits_in_ascending_order() -> None:
    qc = ghz([150, 148, 149, 151], [151, 148])
    qc.global_phase = 0.3
    qc.barrier()
    planned = planned_circuit(qc)
    assert planned.qubits == (148, 149, 150, 151)
    assert planned.measured == (3, 0)
    assert planned.measured_qubits == (151, 148)
    assert {q for op in planned.ops for q in op.qubits} == {0, 1, 2, 3}


def test_a_routing_qubit_gives_the_marginal_of_the_measured_qubits() -> None:
    qc = ghz([148, 149, 150], [148, 150])
    planned = planned_circuit(qc)
    assert planned.qubits == (148, 149, 150) and planned.measured == (0, 2)
    counts = one(qc, {"00": 480, "11": 520})
    assert counts.nv_counts == "1.1"
    assert counts.circuits[0].counts == {"00": 480, "11": 520}
    forecast = kingston().score(counts)
    probs = forecast.circuits[0].probabilities
    assert probs.shape == (4,)
    assert probs[0] + probs[3] > 0.9


def test_a_classical_bit_order_that_differs_from_the_qubit_order_sets_the_key_order() -> None:
    qc = QuantumCircuit(DEVICE_QUBITS, 2, name="flip")
    qc.x(148)
    qc.measure(148, 1)
    qc.measure(150, 0)
    planned = planned_circuit(qc)
    assert planned.qubits == (148, 150) and planned.measured == (1, 0)
    # Qiskit "10": classical bit 1 (qubit 148) read 1, classical bit 0 (qubit 150) read 0.
    counts = one(qc, {"10": 95, "00": 5})
    assert counts.circuits[0].counts == {"01": 95, "00": 5}
    assert counts.qubits == (150, 148)


def test_spaces_between_registers_are_removed() -> None:
    qc = QuantumCircuit(QuantumRegister(DEVICE_QUBITS, "q"), ClassicalRegister(1, "a"))
    qc.add_register(ClassicalRegister(1, "b"))
    qc.x(148)
    qc.measure(148, 0)
    qc.measure(149, 1)
    counts = one(qc, {"0 1": 90, "1 1": 10})
    assert counts.circuits[0].counts == {"10": 90, "11": 10}
    assert counts.nv_counts == "1.0"


def test_a_classical_bit_that_no_measurement_writes_drops_out_of_the_key() -> None:
    qc = QuantumCircuit(DEVICE_QUBITS, 3, name="gap")
    qc.x(148)
    qc.measure(148, 0)
    qc.measure(149, 2)
    counts = one(qc, {"001": 98, "101": 2})
    assert counts.circuits[0].counts == {"10": 98, "11": 2}
    with pytest.raises(CountsError) as caught:
        one(qc, {"011": 100})
    assert caught.value.message == (
        "circuit gap: key '011' has classical bit 1 set to 1, but no measurement writes that bit"
    )


def test_delays_keep_their_time_in_ns_and_a_delay_on_an_unused_qubit_drops_out() -> None:
    qc = QuantumCircuit(DEVICE_QUBITS, 1, name="wait")
    qc.x(148)
    qc.delay(0.5, 148, unit="us")
    qc.delay(100, 148, unit="dt")
    qc.delay(300, 10, unit="ns")
    qc.measure(148, 0)
    qc.delay(50, 148, unit="ns")
    planned = planned_circuit(qc, dt=2e-9)
    assert planned.qubits == (148,)
    assert [op.name for op in planned.ops] == ["x", "delay", "delay"]
    assert [op.params for op in planned.ops[1:]] == [(500.0,), (pytest.approx(200.0),)]


def _circuit(
    change: Callable[[QuantumCircuit], None], qubits: int = DEVICE_QUBITS
) -> QuantumCircuit:
    qc = QuantumCircuit(qubits, 2, name="mine")
    qc.sx(148)
    change(qc)
    return qc


def _measure_then_gate(qc: QuantumCircuit) -> None:
    qc.measure(148, 0)
    qc.sx(148)


def _reset(qc: QuantumCircuit) -> None:
    qc.reset(149)
    qc.measure(148, 0)


def _condition(qc: QuantumCircuit) -> None:
    qc.measure(148, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(149)


def _foreign_gate(qc: QuantumCircuit) -> None:
    qc.append(U3Gate(0.1, 0.2, 0.3), [148])
    qc.measure(148, 0)


def _measured_twice(qc: QuantumCircuit) -> None:
    qc.measure(148, 0)
    qc.measure(148, 1)


def _clbit_twice(qc: QuantumCircuit) -> None:
    qc.measure(148, 0)
    qc.measure(149, 0)


def _unbound(qc: QuantumCircuit) -> None:
    qc.rz(Parameter("theta"), 148)
    qc.measure(148, 0)


def _dt_delay(qc: QuantumCircuit) -> None:
    qc.delay(16, 148, unit="dt")
    qc.measure(148, 0)


def _no_measurement(qc: QuantumCircuit) -> None:
    qc.x(149)


def _too_many_qubits(qc: QuantumCircuit) -> None:
    for q in range(257):
        qc.x(q)
    qc.measure(0, 0)


_TRANSPILE = (
    "transpile the circuit to the basis gates of the device, such as transpile(circuit,"
    " backend), and give the circuit that the device ran"
)
_BIND = "bind the parameters first, such as circuit.assign_parameters(values)"
REFUSALS = {
    "mid-circuit measurement": (
        _measure_then_gate,
        "sx on qubit 148 comes after a measurement of that qubit (a mid-circuit measurement)",
        "measure each qubit after its last gate",
    ),
    "reset": (
        _reset,
        "reset on qubit 149, but each shot of a counts file starts with each qubit in 0",
        "remove the reset",
    ),
    "classical condition": (
        _condition,
        "if_else is a classical condition, and a counts file holds no classical control",
        "give a circuit with no if_test, for_loop, while_loop or switch",
    ),
    "gate outside the tool": (
        _foreign_gate,
        "u3 is not a gate that NoiseVault can simulate",
        _TRANSPILE,
    ),
    "qubit measured twice": (
        _measured_twice,
        "qubit 148 is measured twice",
        "measure each qubit once, after its last gate",
    ),
    "classical bit written twice": (
        _clbit_twice,
        "classical bit 0 is written twice",
        "measure each qubit into its own classical bit",
    ),
    "unbound parameter": (_unbound, "rz on qubit 148 has the unbound parameter theta", _BIND),
    "delay in dt with no dt": (
        _dt_delay,
        "the delay on qubit 148 lasts 16 dt, but the circuit does not say how long dt is",
        "give dt=, the sample time of the device in seconds, such as backend.dt",
    ),
    "no measurement": (
        _no_measurement,
        "the circuit measures no qubit",
        "measure the qubits after their last gate, such as circuit.measure_all()",
    ),
}


@pytest.mark.parametrize("case", REFUSALS)
def test_each_circuit_that_a_counts_file_cannot_hold_is_refused_with_a_hint(case: str) -> None:
    change, message, hint = REFUSALS[case]
    with pytest.raises(CountsError) as caught:
        planned_circuit(_circuit(change))
    assert caught.value.message == f"circuit mine: {message}"
    assert caught.value.hint == hint


def test_a_circuit_on_more_qubits_than_the_format_holds_is_refused() -> None:
    with pytest.raises(CountsError) as caught:
        planned_circuit(_circuit(_too_many_qubits, qubits=300))
    assert caught.value.message == (
        "circuit mine: the circuit uses 257 qubits, more than the 256 that a counts file holds"
        " for one circuit"
    )


def test_a_key_of_the_wrong_width_is_refused() -> None:
    qc = ghz([148, 149], [148, 149], name="pair")
    with pytest.raises(CountsError) as caught:
        one(qc, {"000": 10})
    assert caught.value.message == (
        "circuit pair: key '000' is not 2 bits of 0 or 1, one for each classical bit of the circuit"
    )
    assert "join_data().get_counts()" in caught.value.hint


def test_one_counts_dict_for_each_circuit() -> None:
    qc = ghz([148, 149], [148, 149])
    with pytest.raises(CountsError, match="2 circuits and 1 counts dict"):
        nv.counts_from_qiskit([qc, qc], [{"00": 1}], profile=kingston(), run_at=RUN_AT)


def test_a_run_flag_that_breaks_a_run_rule_gives_the_error_of_the_execution_model() -> None:
    qc = ghz([148, 149], [148, 149])
    with pytest.raises(CountsError) as caught:
        one(qc, {"00": 10}, gate_twirling=True)
    assert caught.value.message == (
        "execution: gate_twirling is true, so the device ran random Pauli gates that the ops do"
        " not list"
    )
    assert caught.value.hint == "run the circuits again without gate twirling"
    with pytest.raises(CountsError, match="options.twirling.enable_measure is true"):
        one(qc, {"00": 10}, options={"twirling": {"enable_measure": True}})


def test_the_score_of_the_true_profile_is_inside_shot_noise_and_of_a_wrong_profile_is_not() -> None:
    circuits = [ghz([148, 149, 150, 151], [151, 148, 149]), mirror([148, 149, 150], 4)]
    counts = nv.counts_from_qiskit(
        circuits, device_counts(circuits), profile=kingston(), run_at=RUN_AT, source="simulated"
    )
    assert [c.measured for c in counts.circuits] == [(3, 0, 1), (2, 1, 0)]
    truth, calibration = device().score(counts), kingston().score(counts)
    assert truth.excess < truth.excess_noise_95 and truth.p_value > 0.05
    assert calibration.excess > calibration.excess_noise_95 and calibration.p_value < 0.05
    summary = calibration.summary()
    assert "largest excess mirror" in summary


# IBM jobs -----------------------------------------------------------------------------------


class FakeBackend:
    def __init__(self, name: str = "ibm_kingston", dt: float | None = 2e-9 / 9) -> None:
        self.name = name
        self.dt = dt


class FakeData:
    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def get_counts(self) -> dict[str, int]:
        return self._counts


class FakePubResult:
    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def join_data(self) -> FakeData:
        return FakeData(self._counts)


class FakeJob:
    """A qiskit-ibm-runtime RuntimeJobV2 with the attributes that from_ibm_job reads."""

    def __init__(
        self,
        pubs: list[Any],
        counts: list[dict[str, int]],
        *,
        program: str = "sampler",
        options: dict[str, Any] | None = None,
        backend: FakeBackend | None = None,
    ) -> None:
        self.primitive_id = program
        self.inputs = {"pubs": pubs, "options": options or {"default_shots": 4000}}
        self._counts = counts
        self._backend = backend or FakeBackend()

    def job_id(self) -> str:
        return "d1job"

    def backend(self) -> FakeBackend:
        return self._backend

    def result(self) -> list[FakePubResult]:
        return [FakePubResult(c) for c in self._counts]

    def metrics(self) -> dict[str, Any]:
        return {"timestamps": {"running": "2026-04-16T09:30:02.5", "finished": "x"}}


@pytest.fixture
def pulls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, datetime | None]]:
    """IBM's public endpoint, answered with the bundled calibration of ibm_kingston."""
    from noisevault.sources import ibm_public

    seen: list[tuple[str, datetime | None]] = []

    def pull(device: str, at: datetime | None = None) -> Profile:
        seen.append((device, at))
        return kingston()

    monkeypatch.setattr(ibm_public, "pull", pull)
    return seen


def test_the_user_story_scores_the_counts_of_a_job_against_its_calibration(pulls, vault) -> None:
    pytest.importorskip("qiskit_ibm_runtime")
    circuits = [ghz([148, 149, 150, 151], [151, 148, 149]), mirror([148, 149, 150], 4)]
    job = FakeJob([(circuits[0],), (circuits[1], None, 4000)], device_counts(circuits))
    profile, counts = nv.from_ibm_job(job)
    assert pulls == [("ibm_kingston", datetime(2026, 4, 16, 9, 30, 2, 500000, tzinfo=UTC))]
    assert profile.fingerprint == kingston().fingerprint
    assert counts.profile.fingerprint == kingston().calibration_fingerprint
    assert counts.run_at == pulls[0][1] and counts.execution.job_ids == ("d1job",)
    assert counts.execution.client.startswith("qiskit-ibm-runtime ")
    assert counts.execution.options == {"default_shots": 4000}
    (saved,) = vault.glob("*.json.gz")
    assert nv.load(str(saved)).fingerprint == profile.fingerprint
    forecast = profile.score(counts)
    assert forecast.excess > forecast.excess_noise_95


def test_a_job_that_is_not_a_sampler_job_is_refused(pulls) -> None:
    job = FakeJob([], [], program="estimator")
    with pytest.raises(CountsError) as caught:
        nv.from_ibm_job(job)
    assert caught.value.message == "job d1job ran the IBM program 'estimator', not 'sampler'"
    assert caught.value.hint == "give a job of SamplerV2, whose result holds counts"
    assert pulls == []


def test_a_pub_with_parameter_values_is_refused(pulls) -> None:
    qc = ghz([148, 149], [148, 149])
    job = FakeJob([(qc,), (qc, [[0.1]], 100)], [{"00": 1}, {"00": 1}])
    with pytest.raises(CountsError) as caught:
        nv.from_ibm_job(job)
    assert caught.value.message.startswith("job d1job: pub 1 gives parameter values")
    assert caught.value.hint.startswith("bind the parameters in the circuit")


def test_a_job_with_twirling_gives_the_error_of_the_execution_model(pulls) -> None:
    pytest.importorskip("qiskit_ibm_runtime")
    qc = ghz([148, 149], [148, 149])
    options = {"twirling": {"enable_gates": True}}
    with pytest.raises(CountsError) as caught:
        nv.from_ibm_job(FakeJob([(qc,)], [{"00": 1}], options=options))
    assert caught.value.message.startswith("execution: options.twirling.enable_gates is true")
