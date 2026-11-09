"""scripts/run_on_braket.py on the free Braket LocalEmulator, with synthetic device properties.

No test opens an AWS account. ``Hardware`` puts a LocalEmulator behind the AwsDevice interface,
so the tests take the paid path (cost check, job file, collection) without a paid device.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from conftest import require

import noisevault as nv
from noisevault import gates
from noisevault.counts import PlannedCircuit, load_counts, plan
from noisevault.errors import NoiseVaultError, install_hint
from noisevault.reference import Op

require("braket.emulation.local_emulator")
from braket.emulation.local_emulator import LocalEmulator  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_on_braket.py"
FIXTURES = ROOT / "tests" / "fixtures" / "braket"
GARNET = "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet"
CEPHEUS = "arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q"
IBEX = "arn:aws:braket:eu-north-1::device/qpu/aqt/Ibex-Q1"
FORTE = "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1"
DEVICES = {
    GARNET: "iqm_garnet_run.json",
    CEPHEUS: "rigetti_cepheus_run.json",
    IBEX: "aqt_ibex_run.json",
    FORTE: "ionq_forte_run.json",
}


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_on_braket", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = _load_script()


@dataclass
class Hardware:
    """A LocalEmulator behind the AwsDevice interface. It keeps every task that it creates."""

    emulator: Any
    tasks: dict[str, Any] = field(default_factory=dict)
    refuse_from: int | None = None

    def run(self, specification: Any, shots: int) -> Any:
        if self.refuse_from is not None and len(self.tasks) >= self.refuse_from:
            raise RuntimeError("ServiceQuotaExceededException")
        task = self.emulator.run(specification, shots=shots)
        self.tasks[task.id] = task
        return task


def _properties(arn: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / DEVICES[arn]).read_text(encoding="utf-8"))
    return data


def _target(arn: str, properties: dict[str, Any] | None = None) -> tuple[Any, Hardware]:
    properties = properties or _properties(arn)
    hardware = Hardware(LocalEmulator.from_json(json.dumps(properties)))
    name = script.device_name(arn)
    return script.Target(arn, name, hardware, properties), hardware


@dataclass
class Run:
    output: Path
    pending: Path
    hardware: Hardware
    target: Any

    def __call__(self, **kwargs: Any) -> Any:
        options: dict[str, Any] = {
            "shots": 400,
            "output": self.output,
            "pending": self.pending,
            "open_target": lambda: self.target,
            "open_task": self.hardware.tasks.__getitem__,
            "price": None,
            "confirm_cost": Decimal("1000"),
        }
        return script.run(**(options | kwargs))


def _run(tmp_path: Path, arn: str = GARNET) -> Run:
    target, hardware = _target(arn)
    output = tmp_path / "dev.counts.json"
    return Run(output, tmp_path / "dev.job.json", hardware, target)


@pytest.fixture
def garnet(tmp_path: Path) -> Iterator[Run]:
    yield _run(tmp_path)


# The cost --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arn", "shots", "program_set", "dollars"),
    [
        (GARNET, 4000, True, "23.50"),
        (CEPHEUS, 4000, True, "7.10"),
        (IBEX, 4000, True, "376.30"),
        (FORTE, 4000, False, "1281.20"),
        (FORTE, 1000, False, "321.20"),
    ],
)
def test_one_run_costs_the_task_price_per_task_and_the_shot_price_per_shot(
    arn: str, shots: int, program_set: bool, dollars: str
) -> None:
    target, _ = _target(arn)
    batch = script.prepare(target, shots, script.PRICES[arn])
    assert len(batch.planned) == 4
    assert batch.program_set is program_set
    assert batch.cost == Decimal(dollars)


def test_a_program_set_over_the_device_limits_becomes_one_task_per_circuit() -> None:
    properties = _properties(GARNET)
    properties["action"]["braket.ir.openqasm.program_set"]["maximumTotalShots"] = 1000
    target, _ = _target(GARNET, properties)
    batch = script.prepare(target, 4000, script.PRICES[GARNET])
    assert batch.program_set is False
    assert batch.cost == 4 * Decimal("0.30") + 16000 * Decimal("0.00145")


def test_without_confirm_cost_the_script_shows_the_cost_and_submits_nothing(
    garnet: Run, capsys: pytest.CaptureFixture[str]
) -> None:
    assert garnet(confirm_cost=None) is None
    out = capsys.readouterr().out
    assert "about $2.62 ($0.30 a task, $0.00145 a shot)" in out
    assert out.endswith(
        "nothing submitted. To submit, run the same command with --confirm-cost 2.62\n"
    )
    assert not garnet.hardware.tasks
    assert not garnet.pending.exists()


def test_a_confirmed_cost_below_the_estimate_submits_nothing(garnet: Run) -> None:
    with pytest.raises(NoiseVaultError) as caught:
        garnet(confirm_cost=Decimal("2.61"))
    assert caught.value.message == (
        "the run costs about $2.62, more than --confirm-cost 2.61. The script submitted nothing"
    )
    assert caught.value.hint == "give --confirm-cost 2.62 to accept the cost"
    assert not garnet.hardware.tasks
    assert not garnet.pending.exists()


def test_a_device_without_a_known_price_needs_both_prices(tmp_path: Path) -> None:
    other = "arn:aws:braket:us-east-1::device/qpu/ionq/Tempo-1"
    target, hardware = _target(FORTE)
    run = Run(tmp_path / "t.counts.json", tmp_path / "t.job.json", hardware, target)
    run.target = script.Target(other, script.device_name(other), hardware, target.properties)
    with pytest.raises(NoiseVaultError, match="the script has no price for " + other):
        run(confirm_cost=None)
    price = script.Price(Decimal("0.30"), Decimal("0.03"))
    assert run(confirm_cost=None, price=price) is None
    assert script.device_name(other) == "ionq_tempo-1"


# The plan --------------------------------------------------------------------------------------


@pytest.mark.parametrize("arn", list(DEVICES))
def test_each_planned_gate_runs_as_one_native_gate_in_a_verbatim_box(arn: str) -> None:
    target, _ = _target(arn)
    batch = script.prepare(target, 100, script.PRICES[arn])
    natives = {g.lower() for g in target.properties["paradigm"]["nativeGateSet"]}
    for planned, built in zip(batch.planned, batch.circuits, strict=True):
        source = built.to_ir("OPENQASM").source
        assert "#pragma braket verbatim\nbox{" in source
        assert "qubit[" not in source
        body = source.split("box{\n", 1)[1].split("}", 1)[0]
        assert {
            statement.split("(")[0].split(" ")[0] for statement in body.split(";\n") if statement
        } <= natives
        assert [m for m in source.split("\n") if "measure" in m] == [
            f"b[{i}] = measure ${q};" for i, q in enumerate(planned.qubits)
        ]
        gates_ran = [
            i
            for i in built.instructions
            if i.operator.name not in ("StartVerbatimBox", "EndVerbatimBox", "Measure")
        ]
        assert len(gates_ran) == len(planned.ops)
        for op, instruction in zip(planned.ops, gates_ran, strict=True):
            assert [int(q) for q in instruction.target] == [planned.qubits[q] for q in op.qubits]
            assert script._same_up_to_phase(
                instruction.operator.to_matrix(), gates.unitary(op.name, op.params)
            )


@pytest.mark.parametrize(
    ("op", "natives", "braket"),
    [
        (Op("r", (0,), (math.pi / 2, 0.3)), ["gpi", "gpi2", "ms"], "GPi2"),
        (Op("r", (0,), (math.pi, 0.3)), ["gpi", "gpi2", "ms"], "GPi"),
        (Op("r", (0,), (0.7, 0.3)), ["gpi", "gpi2", "ms"], None),
        (Op("r", (0,), (0.7, 0.3)), ["prx", "cz"], "PRx"),
        (Op("rx", (0,), (0.7,)), ["prx", "cz"], "PRx"),
        (Op("rx", (0,), (0.7,)), ["rx", "rz", "cz"], "Rx"),
        (Op("cx", (0, 1), ()), ["cnot"], "CNot"),
        (Op("ms", (0, 1), (0.2, 0.4)), ["ms"], "MS"),
        (Op("rxx", (0, 1), (0.9,)), ["xx"], "XX"),
        (Op("rzz", (0, 1), (0.9,)), ["zz"], "ZZ"),
        (Op("cz", (0, 1), ()), ["iswap"], None),
        (Op("sx", (0,), ()), ["prx"], None),
    ],
)
def test_a_native_gate_is_chosen_only_when_its_matrix_matches(
    op: Op, natives: list[str], braket: str | None
) -> None:
    gate = script.native_gate(op, natives)
    assert (None if gate is None else type(gate).__name__) == braket


def test_a_gate_that_is_not_native_is_refused_before_anything_is_saved(garnet: Run) -> None:
    properties = _properties(GARNET)
    properties["paradigm"]["nativeGateSet"] = ["cz", "gpi"]
    garnet.target = script.Target(GARNET, "iqm_garnet", garnet.hardware, properties)
    with pytest.raises(NoiseVaultError) as caught:
        garnet()
    assert caught.value.message == (
        "circuit ghz_chain: r(1.5708, 0.785398) on qubit 1 is not one native gate of iqm_garnet,"
        " whose native gates are cz, gpi"
    )
    assert not garnet.pending.exists()


def test_a_device_without_verbatim_compilation_is_refused(garnet: Run) -> None:
    properties = _properties(GARNET)
    properties["action"]["braket.ir.openqasm.program"]["supportedPragmas"] = []
    garnet.target = script.Target(GARNET, "iqm_garnet", garnet.hardware, properties)
    with pytest.raises(NoiseVaultError, match="does not list verbatim compilation"):
        garnet()


@pytest.mark.parametrize("arn", [IBEX, FORTE])
def test_the_plan_holds_no_delay_and_the_profile_says_why(arn: str) -> None:
    name = script.device_name(arn)
    stated = nv.from_braket(_properties(arn), device=name)
    assert any(op.name == "delay" for c in plan(stated) for op in c.ops)
    profile = script.calibration(_properties(arn), name)
    assert not any(op.name == "delay" for c in plan(profile) for op in c.ops)
    assert all("duration_ns" not in g for g in profile.to_dict()["gates"].values())
    assert any("removed the gate durations" in note for note in profile.provenance.notes)
    assert profile.to_dict()["readout"] == stated.to_dict()["readout"]


def test_a_calibration_without_durations_is_kept_as_imported() -> None:
    stated = nv.from_braket(_properties(GARNET), device="iqm_garnet")
    assert script.calibration(_properties(GARNET), "iqm_garnet") == stated


def test_a_braket_key_lists_the_measured_qubits_in_their_order() -> None:
    circuit = PlannedCircuit(name="c", qubits=(4, 2, 7), ops=())
    found = {"100": 3, "011": 5}
    assert script._clbit0_left(circuit, found, [2, 7, 4]) == {"010": 3, "101": 5}
    with pytest.raises(
        NoiseVaultError, match="Braket measured qubits 2-4, not the planned qubits 2-3"
    ):
        script._clbit0_left(PlannedCircuit(name="c", qubits=(2, 3), ops=()), {"00": 1}, [2, 4])


# The run ---------------------------------------------------------------------------------------


def test_a_run_saves_the_counts_and_the_calibration_that_nv_compare_scores(
    garnet: Run, capsys: pytest.CaptureFixture[str]
) -> None:
    measured = garnet()
    profile_path = garnet.output.with_name("dev.profile.json")
    assert load_counts(garnet.output) == measured
    profile = nv.load(str(profile_path))
    assert (measured.backend, measured.profile.id) == ("iqm_garnet", "iqm_garnet")
    assert measured.profile.fingerprint == profile.calibration_fingerprint
    assert measured.source == "simulated"
    assert [(c.name, c.qubits, c.ops) for c in measured.circuits] == [
        (c.name, c.qubits, c.ops) for c in plan(profile)
    ]
    assert all(c.shots == 400 for c in measured.circuits)
    (task_id,) = garnet.hardware.tasks
    assert measured.execution.job_ids == (task_id,)
    assert measured.execution.options["program_set"] is True
    assert measured.execution.client.startswith("amazon-braket-sdk ")
    assert not garnet.pending.exists()
    result = profile.compare(measured)
    assert result.p_value is not None and result.p_value > 0.001
    out = capsys.readouterr().out
    assert f"next            nv compare {profile_path} {garnet.output}\n" in out


def test_a_device_without_program_sets_runs_one_task_per_circuit(tmp_path: Path) -> None:
    run = _run(tmp_path, FORTE)
    measured = run(shots=100)
    assert len(run.hardware.tasks) == 4
    assert measured.execution.job_ids == tuple(run.hardware.tasks)
    assert measured.execution.options["program_set"] is False


def test_the_job_file_holds_the_plan_and_each_task_before_the_wait(
    garnet: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[dict[str, Any]]] = []

    def look(tasks: Any, submitted: Any) -> Any:
        seen.append([json.loads(x) for x in garnet.pending.read_text().splitlines()])
        raise KeyboardInterrupt

    monkeypatch.setattr(script, "collect", look)
    with pytest.raises(NoiseVaultError) as caught:
        garnet()
    assert caught.value.message == "stopped before the script saved the counts"
    assert str(caught.value.hint).startswith("run the same command again to collect them.")
    first, task = seen[0]
    assert first["device"] == GARNET and first["program_set"] is True and first["shots"] == 400
    assert [c["name"] for c in first["planned"]] == [
        "ghz_chain",
        "mirror",
        "single_qubit",
        "readout",
    ]
    assert len(first["programs"]) == 4
    assert task == {
        "task": next(iter(garnet.hardware.tasks)),
        "circuits": [c["name"] for c in first["planned"]],
    }
    assert garnet.pending.exists()


def test_the_same_command_collects_a_stopped_wait_and_submits_nothing(
    garnet: Run, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    collect = script.collect
    monkeypatch.setattr(script, "collect", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(NoiseVaultError):
        garnet()
    monkeypatch.setattr(script, "collect", collect)
    capsys.readouterr()
    measured = garnet(confirm_cost=None)
    assert measured is not None
    assert len(garnet.hardware.tasks) == 1
    assert capsys.readouterr().out.startswith(f"collecting task {measured.execution.job_ids[0]}")
    assert not garnet.pending.exists()


def test_a_refused_first_task_leaves_no_job_file(garnet: Run) -> None:
    garnet.hardware.refuse_from = 0
    with pytest.raises(NoiseVaultError) as caught:
        garnet()
    assert caught.value.message == "Braket refused the task (ServiceQuotaExceededException)"
    assert not garnet.pending.exists()


def test_tasks_accepted_before_a_refusal_are_collected_by_the_same_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = _run(tmp_path, FORTE)
    run.hardware.refuse_from = 2
    with pytest.raises(NoiseVaultError) as caught:
        run(shots=100)
    assert caught.value.message.startswith("Braket accepted 2 tasks (")
    assert caught.value.message.endswith(
        "), and then the submission stopped (ServiceQuotaExceededException)"
    )
    run.hardware.refuse_from = None
    measured = run(shots=100)
    assert [c.name for c in measured.circuits] == ["ghz_chain", "mirror"]
    assert len(run.hardware.tasks) == 2
    err = capsys.readouterr().err
    assert "warning: the counts leave out single_qubit, readout" in err


def test_an_existing_counts_file_keeps_the_job_file_for_a_new_name(garnet: Run) -> None:
    garnet.output.write_text("another program's file")
    with pytest.raises(NoiseVaultError) as caught:
        garnet()
    assert caught.value.message == f"could not save the counts, because {garnet.output} exists"
    assert garnet.output.read_text() == "another program's file"
    assert garnet.pending.exists()
    other = garnet.output.with_name("new.counts.json")
    measured = garnet(output=other, collect_only=True)
    assert measured is not None and other.exists()


def test_an_empty_job_file_is_damaged(garnet: Run) -> None:
    garnet.pending.write_bytes(b"")
    with pytest.raises(NoiseVaultError, match="is damaged"):
        garnet()


def test_a_damaged_job_file_names_its_tasks(garnet: Run, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(script, "collect", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(NoiseVaultError):
        garnet()
    (task_id,) = garnet.hardware.tasks
    lines = garnet.pending.read_text().splitlines()
    garnet.pending.write_text("{broken\n" + lines[1] + "\n")
    with pytest.raises(NoiseVaultError) as caught:
        garnet()
    assert caught.value.message.startswith(f"{garnet.pending} is damaged")
    assert f"find the tasks {task_id} in the Amazon Braket console" in str(caught.value.hint)


def test_tasks_on_another_device_are_refused(garnet: Run, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(script, "collect", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(NoiseVaultError):
        garnet()
    target, _ = _target(CEPHEUS)
    garnet.target = target
    with pytest.raises(NoiseVaultError) as caught:
        garnet()
    assert caught.value.message == f"{garnet.pending} records tasks on {GARNET}, not on {CEPHEUS}"


def test_a_job_id_that_cannot_be_saved_is_kept_in_a_new_job_file(
    garnet: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    append = script.Owned.append

    def full_disk(self: Any, data: bytes) -> None:
        if self.path == garnet.pending and b'"task"' in data:
            raise OSError(28, "No space left on device")
        append(self, data)

    monkeypatch.setattr(script.Owned, "append", full_disk)
    with pytest.raises(NoiseVaultError) as caught:
        garnet()
    (task_id,) = garnet.hardware.tasks
    kept = garnet.output.with_name(f"dev.{task_id}.job.json")
    assert caught.value.message == (
        f"could not save task {task_id} to {garnet.pending} ([Errno 28] No space left on device)."
        f" The script saved the tasks to {kept} and submitted no more tasks"
    )
    assert caught.value.hint == f"run the same command with --collect={kept} to collect them"
    monkeypatch.undo()
    garnet.pending.unlink()
    measured = garnet(pending=kept, collect_only=True)
    assert measured is not None and measured.execution.job_ids == (task_id,)
    assert not kept.exists()


def test_a_newer_calibration_from_before_the_run_gives_a_warning(
    garnet: Run, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(script, "collect", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(NoiseVaultError):
        garnet()
    monkeypatch.undo()
    properties = _properties(GARNET)
    properties["standardized"]["oneQubitProperties"]["1"]["T1"]["value"] = 12.0
    garnet.target = script.Target(GARNET, "iqm_garnet", garnet.hardware, properties)
    capsys.readouterr()
    measured = garnet()
    err = capsys.readouterr().err
    assert "warning: iqm_garnet has a calibration from 2026-10-20 06:00Z, before the run" in err
    assert measured.profile.fingerprint != script.calibration(properties, "iqm_garnet").fingerprint


# The command line ------------------------------------------------------------------------------


def _main(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = script.main(list(args))
    out, err = capsys.readouterr()
    return code, out, err


def test_a_local_run_writes_simulated_counts_with_no_aws_account(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "f.counts.json"
    properties = FIXTURES / DEVICES[IBEX]
    code, stdout, stderr = _main(
        capsys, "aqt_ibex-q1", "--local", str(properties), "--shots", "200", "-o", str(out)
    )
    assert code == 0, stderr
    assert "cost            none on the local emulator, about $19.10 ($0.30 a task" in stdout
    counts = load_counts(out)
    assert counts.source == "simulated"
    assert counts.execution.options["emulator"] == "LocalEmulator"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["f.counts.json", "f.profile.json"]


def test_a_local_job_file_left_by_a_killed_run_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "f.counts.json"
    profile = script.calibration(_properties(GARNET), "iqm_garnet")
    first = {
        "device": GARNET,
        "local": "garnet.json",
        "profile": profile.to_dict(),
        "planned": [c.model_dump(mode="json") for c in plan(profile)],
        "shots": 100,
        "program_set": True,
        "programs": [],
    }
    (tmp_path / "f.job.json").write_text(json.dumps(first) + "\n")
    local = str(FIXTURES / DEVICES[GARNET])
    code, _, err = _main(capsys, "iqm_garnet", "--local", local, "-o", str(out))
    assert code == 1
    assert "records a run on the local emulator, which the script cannot collect later" in err


def test_a_missing_braket_extra_gives_the_install_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "braket.aws", None)
    code, out, err = _main(capsys, "iqm_garnet", "-o", "x.json")
    assert (code, out) == (1, "")
    assert err == (
        f"error: run_on_braket.py needs amazon-braket-sdk\nhint: {install_hint('braket')}\n"
    )


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["ibm_fez"], "error: 'ibm_fez' is not a device that the script knows"),
        (
            ["iqm_garnet", "--task-price", "0.3"],
            "error: give --task-price and --shot-price together",
        ),
        (["iqm_garnet", "--collect", "none.job.json"], "error: no job file none.job.json"),
    ],
)
def test_command_line_mistakes_stop_before_any_account_opens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    args: list[str],
    message: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(script, "open_aws", lambda arn: pytest.fail("the account opened"))
    code, _, err = _main(capsys, *args, "-o", "x.counts.json")
    assert code == 1
    assert err.splitlines()[0] == message


@pytest.mark.parametrize("amount", ["-1", "nan", "twenty", "inf"])
def test_confirm_cost_takes_only_an_amount_in_dollars(amount: str) -> None:
    with pytest.raises(SystemExit):
        script._parser().parse_args(["iqm_garnet", "-o", "x", "--confirm-cost", amount])
    assert script._dollars("$23.50") == Decimal("23.50")


def test_the_known_devices_have_the_names_the_docs_give() -> None:
    assert [script.device_name(arn) for arn in script.PRICES] == [
        "iqm_garnet",
        "rigetti_cepheus-1-108q",
        "aqt_ibex-q1",
        "ionq_forte-1",
    ]
    docs = (ROOT / "docs" / "recipes.md").read_text(encoding="utf-8")
    assert all(f"`{script.device_name(arn)}`" in docs for arn in script.PRICES)
    assert np.isclose(float(script.PRICES[GARNET].shot), 0.00145)
