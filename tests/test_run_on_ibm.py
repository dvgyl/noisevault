from __future__ import annotations

import contextlib
import copy
import dataclasses
import errno
import functools
import gzip
import importlib.util
import io
import json
import os
import re
import shlex
import subprocess
import sys
import time
import tomllib
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from conftest import require
from typer.testing import CliRunner

import noisevault as nv
from noisevault import compat
from noisevault.cli import app
from noisevault.counts import SAMPLER_V2_OPTIONS, PlannedCircuit, load_counts, plan
from noisevault.errors import REPOSITORY, NoiseVaultError, install_hint
from noisevault.profile import Profile

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_on_ibm.py"
SHOTS = 4000
FEZ = "nv:06404cefa54f"
FEZ_PLAN = """\
ibm_fez@2025-02-26 nv:06404cefa54f on qubits 136-143-142-141

circuit       qubits           gates  duration
ghz_chain     136-143-142-141     11    300 ns
mirror        136-143-142         37    600 ns
single_qubit  136-143             12    144 ns
readout       136-143-142-141      0      0 ns

shots           4000 per circuit, 4 circuits in one job
usage           about 6.1 s of QPU time (IBM's estimate)
counts file     fez.counts.json
"""
WAITING = (
    "waiting for it to run\nCtrl-C stops waiting. Run the same command again to collect the job\n"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_on_ibm", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = _load_script()


def _fez() -> Any:
    fake_provider = require("qiskit_ibm_runtime.fake_provider")
    aer = require("qiskit_aer")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fake = fake_provider.FakeFez()
    # Without a sim, each local run deep-copies the fake and rebuilds its 156-qubit noise model.
    fake.sim = aer.AerSimulator()
    return fake


def _fractional_fez() -> Any:
    converter = require("qiskit_ibm_runtime.utils.backend_converter")
    fake = _fez()
    configuration = copy.deepcopy(fake.configuration())
    configuration.basis_gates = [*configuration.basis_gates, "rx", "rzz"]
    fake._target = converter.convert_to_target(
        configuration, fake.properties(), include_fractional_gates=True
    )
    return fake


@functools.cache
def _fez_profile() -> Profile:
    return nv.from_qiskit_backend(_fez())


def _pinned_snapshot() -> Profile:
    profile = _fez_profile()
    if profile.short_fingerprint != FEZ:
        pytest.skip(f"FakeFez here is {profile.short_fingerprint}; the pinned numbers are {FEZ}'s")
    return profile


@contextlib.contextmanager
def _five_hours_behind_utc() -> Iterator[None]:
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setenv("TZ", "NVT+5")
            time.tzset()
            yield
    finally:
        time.tzset()


@dataclass
class Submission:
    sampler: Any
    pubs: list[Any]
    job: Any


def _recording_sampler(
    monkeypatch: pytest.MonkeyPatch,
    *,
    lose_the_first_wait: bool = False,
    during_the_first_wait: Callable[[], None] | None = None,
) -> list[Submission]:
    require("qiskit_ibm_runtime")
    submissions: list[Submission] = []

    class Recording(compat.ibm_sampler()):
        def run(self, pubs: Any, **options: Any) -> Any:
            job = super().run(pubs, **options)
            if lose_the_first_wait and not submissions:

                def lost(*args: Any, **kwargs: Any) -> Any:
                    raise ConnectionError("network is unreachable")

                job.result = lost
            if during_the_first_wait is not None and not submissions:
                result = job.result

                def wait(*args: Any, **kwargs: Any) -> Any:
                    during_the_first_wait()
                    return result(*args, **kwargs)

                job.result = wait
            submissions.append(Submission(self, list(pubs), job))
            return job

    monkeypatch.setattr(compat, "ibm_sampler", lambda: Recording)
    return submissions


def _no_job(job_id: str) -> Any:
    raise AssertionError(f"the script opened job {job_id} instead of submitting one")


def _hinted(err: str) -> list[str]:
    hint = err.rpartition("hint: run the same command with ")[2]
    return shlex.split(re.split(" to collect | and a new -o ", hint, maxsplit=1)[0])


def _new_job(pending: Path) -> str:
    return (
        f"To submit a new job, delete {pending}. Then run the command without --collect and"
        " --job-id"
    )


@dataclass
class LocalRun:
    output: Path
    printed: str
    warned: list[str]
    submissions: list[Submission]
    calibrated_at: list[datetime | None]
    fractional: list[bool]
    before: datetime
    after: datetime
    fake: Any


@pytest.fixture(scope="module")
def local_run(tmp_path_factory: pytest.TempPathFactory) -> LocalRun:
    folder = tmp_path_factory.mktemp("run_on_ibm")
    with pytest.MonkeyPatch.context() as monkeypatch, _five_hours_behind_utc():
        monkeypatch.setenv("NOISEVAULT_HOME", str(folder / "nv_home"))
        submissions = _recording_sampler(monkeypatch)
        fake = _fez()
        calibrated_at: list[datetime | None] = []
        fractional: list[bool] = []

        def calibration(at: datetime | None) -> Profile:
            calibrated_at.append(at)
            return nv.from_qiskit_backend(fake)

        def open_backend(flag: bool) -> Any:
            fractional.append(flag)
            return fake

        output = folder / "fez.counts.json"
        printed = io.StringIO()
        before = datetime.now(UTC)
        with contextlib.redirect_stdout(printed), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            script.run(
                device="ibm_fez",
                shots=SHOTS,
                output=output,
                pending=folder / "fez.job.json",
                calibration=calibration,
                open_backend=open_backend,
                open_job=_no_job,
                confirm=None,
            )
        after = datetime.now(UTC)
    return LocalRun(
        output,
        printed.getvalue(),
        [str(w.message) for w in caught],
        submissions,
        calibrated_at,
        fractional,
        before,
        after,
        fake,
    )


def test_a_missing_ibm_extra_gives_the_install_command(tmp_path: Path) -> None:
    hide = "import sys; sys.modules['qiskit_ibm_runtime'] = None; import runpy;"
    run = f"sys.argv = sys.argv[1:]; runpy.run_path({str(SCRIPT)!r}, run_name='__main__')"
    out = tmp_path / "x.counts.json"
    result = subprocess.run(
        [sys.executable, "-c", hide + run, "run_on_ibm.py", "ibm_fez", "-o", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode, result.stdout) == (1, "")
    assert result.stderr == (
        f"error: run_on_ibm.py needs qiskit-ibm-runtime\nhint: {install_hint('ibm')}\n"
    )


def test_the_counts_file_binds_to_the_calibration_and_lists_the_planned_circuits(
    local_run: LocalRun,
) -> None:
    counts = load_counts(local_run.output)
    profile = _fez_profile()
    assert (counts.source, counts.backend) == ("simulated", "ibm_fez")
    assert (counts.profile.id, counts.profile.fingerprint) == ("ibm_fez", profile.fingerprint)
    (submission,) = local_run.submissions
    assert counts.execution.job_ids == (submission.job.job_id(),)
    assert [(c.name, c.qubits, c.ops) for c in counts.circuits] == [
        (c.name, c.qubits, c.ops) for c in plan(profile)
    ]
    assert all(c.shots == SHOTS for c in counts.circuits)
    assert local_run.calibrated_at == [None, counts.run_at]
    assert local_run.fractional == [False]
    assert not (local_run.output.parent / "fez.job.json").exists()


def test_run_at_is_when_the_job_started_in_utc(local_run: LocalRun) -> None:
    run_at = load_counts(local_run.output).run_at
    assert local_run.before <= run_at <= local_run.after


def _instructions(qc: Any) -> list[tuple[Any, ...]]:
    out = []
    for instruction in qc.data:
        operation = instruction.operation
        params = tuple(float(p) for p in operation.params)
        if operation.name == "delay":
            params = (operation.params[0], operation.unit)
        out.append(
            (
                operation.name,
                tuple(qc.find_bit(q).index for q in instruction.qubits),
                params,
                tuple(qc.find_bit(c).index for c in instruction.clbits),
            )
        )
    return out


def test_the_submitted_circuits_are_the_planned_ops_then_the_measurements(
    local_run: LocalRun,
) -> None:
    (submission,) = local_run.submissions
    target = local_run.fake.target
    dt_ns = target.dt * 1e9
    planned = plan(_fez_profile())
    assert len(submission.pubs) == len(planned)
    for circuit, qc in zip(planned, submission.pubs, strict=True):
        expected = []
        for op in circuit.ops:
            qubits = tuple(circuit.qubits[q] for q in op.qubits)
            if op.name == "delay":
                expected.append(("delay", qubits, (round(op.params[0] / dt_ns), "dt"), ()))
            else:
                expected.append((op.name, qubits, op.params, ()))
        expected.append(("barrier", circuit.qubits, (), ()))
        expected += [("measure", (q,), (), (i,)) for i, q in enumerate(circuit.qubits)]
        assert qc.num_qubits == target.num_qubits
        assert (circuit.name, _instructions(qc)) == (circuit.name, expected)


def test_the_job_runs_the_server_side_sampler_program() -> None:
    # qiskit-ibm-runtime 0.52 gave the top-level SamplerV2 name to a client-side sampler, which
    # runs the "executor" program with other pre- and post-processing.
    require("qiskit_ibm_runtime")
    assert compat.ibm_sampler()._program_id() == "sampler"


def test_one_job_carries_every_option_the_counts_format_checks(local_run: LocalRun) -> None:
    (submission,) = local_run.submissions
    recorded = load_counts(local_run.output).execution.options
    paths = {
        **SAMPLER_V2_OPTIONS,
        "default_shots": SHOTS,
        "execution.rep_delay": local_run.fake.default_rep_delay,
    }
    for path, value in paths.items():
        submitted, saved = submission.sampler.options, recorded
        for part in path.split("."):
            submitted, saved = getattr(submitted, part), saved[part]
        assert (path, submitted, saved) == (path, value, value)
    timing = {"execution": {"scheduler_timing": True}}
    assert submission.sampler.options.experimental == recorded["experimental"] == timing
    assert any("have no effect in local testing mode" in w for w in local_run.warned)


_PEP_723 = re.compile(r"(?m)^# /// script$\s(?P<content>(^#(| .*)$\s)+)^# ///$")


def test_the_usage_names_the_script_the_same_way_for_uv_and_a_clone() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, check=True
    )
    assert result.stdout.startswith(
        "usage: run_on_ibm.py [-h] [--shots N] -o FILE [--collect JOB_FILE]\n"
        "                     [--job-id JOB_ID] [--yes] [--suite]\n"
        "                     DEVICE\n"
    )


def test_uv_runs_the_script_from_a_url_with_noisevault_from_the_repository() -> None:
    block = _PEP_723.search(SCRIPT.read_text(encoding="utf-8"))
    assert block, "scripts/run_on_ibm.py needs a # /// script block for uv run"
    lines = block["content"].splitlines(keepends=True)
    metadata = tomllib.loads("".join(line[2:] if line.startswith("# ") else "\n" for line in lines))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert metadata == {
        "requires-python": project["requires-python"],
        "dependencies": [f"noisevault[ibm] @ git+{REPOSITORY}"],
    }


def test_the_run_ends_with_its_files_and_a_command_that_scores_them(local_run: LocalRun) -> None:
    counts = load_counts(local_run.output)
    profile = _fez_profile()
    (job_id,) = counts.execution.job_ids
    ref = f"ibm_fez@{profile.device.calibrated_at:%Y-%m-%d}"
    assert local_run.printed.split("\n\n")[-2:] == [
        f"submitted job {job_id} to ibm_fez\n{WAITING.rstrip()}",
        f"run             {counts.run_at:%Y-%m-%d %H:%MZ}, job {job_id}\n"
        f"counts file     {local_run.output}\n"
        f"calibration     {ref} {profile.short_fingerprint}\n"
        f"next            nv compare {ref} {local_run.output}\n",
    ]
    result = CliRunner().invoke(app, ["compare", ref, str(local_run.output)], env={"COLUMNS": "80"})
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    header = " ".join(line.strip() for line in lines[: lines.index("")])
    assert header.split()[1] == profile.short_fingerprint
    assert f" counts {local_run.output}, simulated, sha256:" in header


def test_the_summary_shows_the_circuits_and_the_usage_and_a_no_submits_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    submissions = _recording_sampler(monkeypatch)
    profile = _pinned_snapshot()
    fake = _fez()
    monkeypatch.chdir(tmp_path)
    prompts: list[str] = []

    def decline(prompt: str) -> bool:
        prompts.append(prompt)
        return False

    measured = script.run(
        device="ibm_fez",
        shots=SHOTS,
        output=Path("fez.counts.json"),
        pending=Path("fez.job.json"),
        calibration=lambda at: profile,
        open_backend=lambda fractional: fake,
        open_job=_no_job,
        confirm=decline,
    )
    assert measured is None
    assert capsys.readouterr().out == FEZ_PLAN + "\nnothing submitted\n"
    assert prompts == ["Submit the job to ibm_fez? [y/N] "]
    assert submissions == []
    assert list(tmp_path.iterdir()) == []


def _submitting(folder: Path, confirm: Callable[[str], bool]) -> Callable[[], Any]:
    profile, fake = _fez_profile(), _fez()
    return lambda: script.run(
        device="ibm_fez",
        shots=SHOTS,
        output=folder / "fez.counts.json",
        pending=folder / "fez.job.json",
        calibration=lambda at: profile,
        open_backend=lambda fractional: fake,
        open_job=_no_job,
        confirm=confirm,
    )


def test_a_job_file_that_appears_before_the_script_submits_is_kept_and_nothing_is_submitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submissions = _recording_sampler(monkeypatch)
    pending = tmp_path / "fez.job.json"

    def another_run_submits_first(prompt: str) -> bool:
        pending.write_text("another run\n")
        return True

    with pytest.raises(NoiseVaultError) as info:
        _submitting(tmp_path, another_run_submits_first)()
    assert (info.value.message, info.value.hint) == (
        f"{pending} exists, so the script did not submit a job",
        f"run the same command with --collect={pending} and a new -o file name. The script never"
        " replaces a file",
    )
    assert submissions == []
    assert sorted(tmp_path.iterdir()) == [pending]
    assert pending.read_text() == "another run\n"


def test_a_failed_submission_leaves_no_job_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refused(backend: Any, batch: Any) -> Any:
        raise NoiseVaultError("submitting the job failed (403 Forbidden)")

    monkeypatch.setattr(script, "submit", refused)
    with pytest.raises(NoiseVaultError, match=re.escape("(403 Forbidden)")):
        _submitting(tmp_path, lambda prompt: True)()
    assert list(tmp_path.iterdir()) == []


def test_a_job_file_that_another_run_changes_in_place_during_a_failed_submission_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pending = tmp_path / "fez.job.json"
    inodes: list[int] = []

    def another_run_writes_the_job_file_and_ibm_refuses(backend: Any, batch: Any) -> Any:
        inodes.append(pending.stat().st_ino)
        pending.write_text("another run\n")
        inodes.append(pending.stat().st_ino)
        raise NoiseVaultError("submitting the job failed (403 Forbidden)")

    monkeypatch.setattr(script, "submit", another_run_writes_the_job_file_and_ibm_refuses)
    with pytest.raises(NoiseVaultError, match=re.escape("(403 Forbidden)")):
        _submitting(tmp_path, lambda prompt: True)()
    assert len(set(inodes)) == 1
    assert list(tmp_path.iterdir()) == [pending]
    assert pending.read_text() == "another run\n"


def test_the_counts_keep_circuit_qubit_0_first() -> None:
    fake = _fez()
    profile = _fez_profile()
    flip = PlannedCircuit(name="flip", qubits=(136, 143), ops=(("x", [0], []),))
    qc = script.isa_circuit(flip, profile, fake.target).circuit
    batch = script.Batch(
        profile=profile,
        planned=(flip,),
        circuits=(qc,),
        durations_ns=(0.0,),
        shots=100,
        options=script.sampler_options(100, fake.default_rep_delay),
        usage_s=0.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        job = script.submit(fake, batch)
        submitted = script.Submitted(job.job_id(), profile, (flip,), batch.options)
        ran = script.collect(job, submitted)
    counts = script.counts_file(profile, submitted, ran)
    assert dict(counts.circuits[0].counts) == {"10": 100}


@pytest.mark.parametrize(
    ("circuit", "message"),
    [
        (
            PlannedCircuit(name="far", qubits=(136, 0), ops=(("cz", [0, 1], []),)),
            "circuit far: the backend does not support cz on qubits 136-0",
        ),
        (
            PlannedCircuit(name="odd", qubits=(136,), ops=(("delay", [0], [10.0]),)),
            "circuit odd: the 10 ns delay on qubit 136 is 2.5 dt, and the backend times delays in"
            " whole dt of 4 ns",
        ),
        (
            PlannedCircuit(name="blip", qubits=(136,), ops=(("delay", [0], [4.0]),)),
            "circuit blip: the 4 ns delay on qubit 136 is 1 dt, shorter than the backend's minimum"
            " of 2 dt",
        ),
    ],
)
def test_ops_the_backend_would_not_run_as_planned_are_refused(
    circuit: PlannedCircuit, message: str
) -> None:
    fake = _fez()
    with pytest.raises(NoiseVaultError) as info:
        script.isa_circuit(circuit, _fez_profile(), fake.target)
    assert info.value.message == message


def test_a_gate_whose_backend_duration_changed_is_refused() -> None:
    target_module = require("qiskit.transpiler")
    profile = _pinned_snapshot()
    fake = _fez()
    error = fake.target["sx"][(136,)].error
    fake.target.update_instruction_properties(
        "sx", (136,), target_module.InstructionProperties(duration=28e-9, error=error)
    )
    with pytest.raises(NoiseVaultError) as info:
        script.isa_circuit(plan(profile)[0], profile, fake.target)
    assert info.value.message == (
        "circuit ghz_chain: sx on qubit 136 lasts 6 dt (24 ns) in the calibration and 7 dt"
        " (28 ns) on the backend, so the planned delays would not fill the gaps"
    )
    assert info.value.hint == "run the script again to plan from the current calibration"


@pytest.mark.parametrize(
    ("grid", "circuit", "message"),
    [
        (
            {"pulse_alignment": 8},
            "ghz_chain",
            "circuit ghz_chain: cz on qubits 136-143 would start at 6 dt, off the backend's grid"
            " of 8 dt",
        ),
        (
            {"acquire_alignment": 16},
            "single_qubit",
            "circuit single_qubit: the measurements would start at 36 dt, off the backend's grid"
            " of 16 dt",
        ),
    ],
)
def test_an_op_or_measurement_off_the_backend_grid_is_refused(
    grid: dict[str, int], circuit: str, message: str
) -> None:
    profile = _pinned_snapshot()
    fake = _fez()
    for name, value in grid.items():
        setattr(fake.target, name, value)
    (planned,) = (c for c in plan(profile) if c.name == circuit)
    with pytest.raises(NoiseVaultError) as info:
        script.isa_circuit(planned, profile, fake.target)
    assert info.value.message == message


def _edited(profile: Profile, edit: Callable[[dict[str, Any]], None]) -> Profile:
    data = profile.to_dict()
    edit(data)
    return Profile.model_validate(data)


def _lower_sx_error(data: dict[str, Any]) -> None:
    record = next(r for r in data["calibrations"] if (r["gate"], r["qubits"]) == ("sx", [136]))
    record["avg_infidelity"] = 1e-4


def _submitted_and_ran(profile: Profile) -> tuple[Any, Any]:
    planned = plan(profile)
    submitted = script.Submitted("job-1", profile, planned, script.sampler_options(SHOTS, 2.5e-4))
    ran = script.Ran(
        job_id="job-1",
        source="simulated",
        run_at=datetime(2026, 4, 16, 9, 30, tzinfo=UTC),
        counts=tuple({"0" * len(c.qubits): SHOTS} for c in planned),
        timing={},
    )
    return submitted, ran


def test_the_counts_bind_to_the_calibration_in_effect_when_the_job_ran() -> None:
    planned = _fez_profile()
    latest = _edited(planned, _lower_sx_error)
    submitted, ran = _submitted_and_ran(planned)
    assert script.bind(submitted, ran, lambda at: latest) == script.Binding(latest)


def test_a_changed_duration_binds_the_counts_to_the_plan_and_warns() -> None:
    planned = _pinned_snapshot()

    def slower_sx(data: dict[str, Any]) -> None:
        data["gates"]["sx"]["duration_ns"] = 28.0

    latest = _edited(planned, slower_sx)
    submitted, ran = _submitted_and_ran(planned)
    assert script.bind(submitted, ran, lambda at: latest) == script.Binding(
        planned,
        f"IBM recalibrated ibm_fez before the job ran ({latest.short_fingerprint}), and sx on"
        " qubit 136 now takes 28 ns, not 24 ns, so the submitted delays may not match the"
        f" timeline that ran. The counts bind to the planned calibration {FEZ}",
        "run the script again for counts that match one calibration",
    )


def test_a_gate_disabled_when_the_job_ran_binds_the_counts_to_the_plan() -> None:
    planned = _fez_profile()

    def disable_cz(data: dict[str, Any]) -> None:
        data["calibrations"] = [
            {"gate": "cz", "qubits": [136, 143], "disabled": True}
            if (r["gate"], r["qubits"]) == ("cz", [136, 143])
            else r
            for r in data["calibrations"]
        ]

    submitted, ran = _submitted_and_ran(planned)
    binding = script.bind(submitted, ran, lambda at: _edited(planned, disable_cz))
    assert binding.profile is planned
    assert binding.warning is not None
    assert "nv compare would refuse the counts against it (circuit ghz_chain: " in binding.warning


def test_a_calibration_older_than_the_plan_binds_the_counts_to_the_plan() -> None:
    planned = _fez_profile()

    def a_day_earlier(data: dict[str, Any]) -> None:
        _lower_sx_error(data)
        data["device"]["calibrated_at"] = "2025-02-25T20:16:25Z"

    latest = _edited(planned, a_day_earlier)
    submitted, ran = _submitted_and_ran(planned)
    assert script.bind(submitted, ran, lambda at: latest) == script.Binding(
        planned,
        f"the calibration IBM returned for the time the job ran, {latest.short_fingerprint}, is"
        f" older than the planned {planned.short_fingerprint}. The counts bind to the planned"
        " calibration",
    )


def test_a_failed_pull_at_run_time_binds_the_counts_to_the_plan() -> None:
    planned = _fez_profile()
    submitted, ran = _submitted_and_ran(planned)

    def calibration(at: datetime | None) -> Profile:
        raise nv.SourceUnavailable("IBM did not return the calibration of ibm_fez (timeout)")

    assert script.bind(submitted, ran, calibration) == script.Binding(
        planned,
        "could not pull the calibration in effect when the job ran (IBM did not return the"
        " calibration of ibm_fez (timeout)). The counts bind to the planned calibration"
        f" {planned.short_fingerprint}",
    )


def _save_job_file(submitted: Any, path: Path, *, with_id: bool = True) -> None:
    data = {
        "job_id": submitted.job_id,
        "profile": submitted.profile.to_dict(),
        "planned": [circuit.model_dump(mode="json") for circuit in submitted.planned],
        "options": submitted.options,
    }
    if not with_id:
        del data["job_id"]
    path.write_text(json.dumps(data) + "\n")


def _collecting_job_1(
    folder: Path,
    monkeypatch: pytest.MonkeyPatch,
    timing: dict[str, Any] | None = None,
    profile: Profile | None = None,
) -> Callable[[], Any]:
    profile = profile or _fez_profile()
    submitted, ran = _submitted_and_ran(profile)
    _save_job_file(submitted, folder / "fez.job.json")
    monkeypatch.setattr(
        script, "collect", lambda *args: dataclasses.replace(ran, timing=timing or {})
    )

    def plan_again(fractional: bool) -> Any:
        raise AssertionError("the script planned a new job instead of collecting job-1")

    return lambda: script.run(
        device="ibm_fez",
        shots=SHOTS,
        output=folder / "fez.counts.json",
        pending=folder / "fez.job.json",
        calibration=lambda at: profile,
        open_backend=plan_again,
        open_job=lambda job_id: job_id,
        confirm=None,
    )


@dataclass
class Disk:
    folder: Path
    bytes_left: int | None = None


def _disk(monkeypatch: pytest.MonkeyPatch, folder: Path) -> Disk:
    disk = Disk(folder)
    real_open, real_write = os.open, os.write
    on_disk: set[int] = set()

    def open_(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        here = Path(path).parent == disk.folder
        if here and disk.bytes_left is not None and flags & os.O_CREAT:
            raise OSError(errno.ENOSPC, "No space left on device", str(path))
        fd = real_open(path, flags, *args, **kwargs)
        if here:
            on_disk.add(fd)
        return fd

    def write(fd: int, data: Any) -> int:
        if disk.bytes_left is None or fd not in on_disk:
            return real_write(fd, data)
        if disk.bytes_left == 0:
            raise OSError(errno.ENOSPC, "No space left on device")
        written = real_write(fd, data[: disk.bytes_left])
        disk.bytes_left -= written
        return written

    monkeypatch.setattr(os, "open", open_)
    monkeypatch.setattr(os, "write", write)
    return disk


def test_a_disk_that_refuses_the_counts_leaves_the_job_to_collect_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _collecting_job_1(tmp_path, monkeypatch)
    _disk(monkeypatch, tmp_path).bytes_left = 0
    pending = tmp_path / "fez.job.json"
    with pytest.raises(NoiseVaultError) as info:
        run()
    assert info.value.message == (
        "could not collect the counts of job job-1 ([Errno 28] No space left on device:"
        f" '{tmp_path / 'fez.counts.json'}')"
    )
    assert info.value.hint == f"run the same command again to collect them. {_new_job(pending)}"
    assert sorted(tmp_path.iterdir()) == [pending]


def test_a_gz_counts_file_is_gzip_with_no_time_stamp_that_load_counts_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _fez_profile()
    submitted, ran = _submitted_and_ran(profile)
    folder = tmp_path / "counts"
    folder.mkdir()
    output, pending = folder / "fez.counts.json.gz", folder / "fez.job.json"
    _save_job_file(submitted, pending)
    monkeypatch.setattr(script, "collect", lambda *args: ran)
    measured = script.run(
        device="ibm_fez",
        shots=SHOTS,
        output=output,
        pending=pending,
        calibration=lambda at: profile,
        open_backend=lambda fractional: None,
        open_job=lambda job_id: job_id,
        confirm=None,
    )
    data = output.read_bytes()
    assert (data[:2], data[4:8]) == (b"\x1f\x8b", bytes(4))
    assert json.loads(gzip.decompress(data)) == measured.to_dict()
    assert load_counts(output).to_dict() == measured.to_dict()
    assert measured.save(folder / "saved.counts.json.gz").read_bytes() == data


@pytest.mark.parametrize("name", ["fez.timing.json", "fez.profile.json"])
def test_a_file_that_appears_beside_the_counts_while_the_job_waits_is_kept(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _edited(_fez_profile(), _lower_sx_error)
    submitted, ran = _submitted_and_ran(profile)
    folder = tmp_path / "counts"
    folder.mkdir()
    pending, other = folder / "fez.job.json", folder / name
    _save_job_file(submitted, pending)

    def another_experiment(*args: Any) -> Any:
        other.write_text("another experiment\n")
        return dataclasses.replace(ran, timing={"ghz_chain": {"start": 0}})

    monkeypatch.setattr(script, "collect", another_experiment)
    with pytest.raises(NoiseVaultError) as info:
        script.run(
            device="ibm_fez",
            shots=SHOTS,
            output=folder / "fez.counts.json",
            pending=pending,
            calibration=lambda at: profile,
            open_backend=lambda fractional: None,
            open_job=lambda job_id: job_id,
            confirm=None,
        )
    assert info.value.message == f"could not save the counts of job job-1, because {other} exists"
    assert sorted(folder.iterdir()) == sorted([other, pending])
    assert other.read_text() == "another experiment\n"


def _when_the_script_creates(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    *,
    before: Callable[[], None] | None = None,
    after: Callable[[], None] | None = None,
    written: Callable[[], None] | None = None,
) -> None:
    real_open, real_write = os.open, os.write
    created: list[int] = []

    def open_(file: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if Path(file) != path or not flags & os.O_EXCL:
            return real_open(file, flags, *args, **kwargs)
        if before is not None:
            before()
        fd = real_open(file, flags, *args, **kwargs)
        created.append(fd)
        if after is not None:
            after()
        return fd

    def write(fd: int, data: Any) -> int:
        count = real_write(fd, data)
        if written is not None and fd in created:
            created.remove(fd)
            written()
        return count

    monkeypatch.setattr(os, "open", open_)
    monkeypatch.setattr(os, "write", write)


def _put(path: Path, text: str) -> None:
    other = path.with_name(".another-program.tmp")
    other.write_text(text)
    os.replace(other, path)


@pytest.mark.parametrize(
    ("created", "moment", "put"),
    [
        ("fez.counts.json", "after", _put),
        ("fez.counts.json", "after", Path.write_text),
        ("fez.counts.json", "written", _put),
        ("fez.counts.json", "written", Path.write_text),
        ("fez.timing.json", "before", _put),
    ],
    ids=[
        "replaced after the create",
        "written in place after the create",
        "replaced after the write",
        "written in place after the write",
        "before the create",
    ],
)
def test_a_counts_file_that_another_program_puts_in_place_is_kept_with_the_job(
    created: str,
    moment: str,
    put: Callable[[Path, str], Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _collecting_job_1(tmp_path, monkeypatch, timing={"ghz_chain": {"start": 0}})
    output, pending = tmp_path / "fez.counts.json", tmp_path / "fez.job.json"
    record = pending.read_bytes()
    others = sorted({output, tmp_path / created})

    def another_program() -> None:
        for path in others:
            put(path, "another experiment\n")

    _when_the_script_creates(monkeypatch, tmp_path / created, **{moment: another_program})
    with pytest.raises(NoiseVaultError) as info:
        run()
    assert info.value.message == (
        f"could not save the counts of job job-1, because {tmp_path / created} exists"
    )
    assert sorted(tmp_path.iterdir()) == sorted([*others, pending])
    assert [path.read_text() for path in others] == ["another experiment\n"] * len(others)
    assert pending.read_bytes() == record


def test_a_defect_after_the_job_ran_raises_instead_of_asking_to_collect_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _collecting_job_1(tmp_path, monkeypatch)

    def defect(*args: Any) -> Any:
        raise TypeError("a defect in the script")

    monkeypatch.setattr(script, "bind", defect)
    with pytest.raises(TypeError, match="a defect in the script"):
        run()


def test_a_job_file_that_another_collect_removed_first_does_not_stop_this_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _collecting_job_1(tmp_path, monkeypatch)
    pending = tmp_path / "fez.job.json"
    collect = script.collect

    def the_other_collect_ends_first(*args: Any) -> Any:
        pending.unlink()
        return collect(*args)

    monkeypatch.setattr(script, "collect", the_other_collect_ends_first)
    measured = run()
    assert measured is not None
    assert load_counts(tmp_path / "fez.counts.json").execution.job_ids == ("job-1",)
    assert not pending.exists()


@pytest.mark.parametrize("saved", ["as a new file", "in place"])
def test_a_job_file_that_a_new_run_saved_while_the_job_waited_is_kept(
    saved: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _collecting_job_1(tmp_path, monkeypatch)
    pending = tmp_path / "fez.job.json"
    job_2 = dataclasses.replace(_submitted_and_ran(_fez_profile())[0], job_id="job-2")
    collect = script.collect
    inodes = [pending.stat().st_ino]

    def a_new_run_saves_job_2(*args: Any) -> Any:
        if saved == "as a new file":
            pending.unlink()
        _save_job_file(job_2, pending)
        inodes.append(pending.stat().st_ino)
        return collect(*args)

    monkeypatch.setattr(script, "collect", a_new_run_saves_job_2)
    assert run() is not None
    assert (inodes[0] == inodes[1]) == (saved == "in place")
    assert load_counts(tmp_path / "fez.counts.json").execution.job_ids == ("job-1",)
    assert json.loads(pending.read_text())["job_id"] == "job-2"


def test_a_link_that_replaces_the_job_file_while_the_job_waits_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _collecting_job_1(tmp_path, monkeypatch)
    pending, saved = tmp_path / "fez.job.json", tmp_path / "saved.job.json"
    collect = script.collect

    def another_program_links_the_job_file_to_a_new_name(*args: Any) -> Any:
        pending.rename(saved)
        pending.symlink_to(saved.name)
        return collect(*args)

    monkeypatch.setattr(script, "collect", another_program_links_the_job_file_to_a_new_name)
    assert run() is not None
    assert load_counts(tmp_path / "fez.counts.json").execution.job_ids == ("job-1",)
    assert pending.is_symlink()
    assert os.readlink(pending) == saved.name


def test_the_saved_files_hold_the_bytes_that_save_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "counts"
    folder.mkdir()
    timing = {"ghz_chain": {"start": 0}}
    profile = _edited(_fez_profile(), _lower_sx_error)
    measured = _collecting_job_1(folder, monkeypatch, timing, profile)()
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == {
        "fez.counts.json": measured.save(tmp_path / "fez.counts.json").read_bytes(),
        "fez.timing.json": (json.dumps(timing, indent=1) + "\n").encode(),
        "fez.profile.json": profile.save(tmp_path / "fez.profile.json").read_bytes(),
    }


@dataclass
class Account:
    pulled_at: list[datetime | None]
    opened: list[bool]
    jobs: list[str]


def _account(
    monkeypatch: pytest.MonkeyPatch, fake: Any, submissions: list[Submission] | None = None
) -> Account:
    runtime = require("qiskit_ibm_runtime")
    snapshot = fake.properties()
    seen = Account([], [], [])

    class Calibrated:
        name = "ibm_fez"
        processor_type = fake.processor_type

        def properties(self, refresh: bool = False, datetime: datetime | None = None) -> Any:
            seen.pulled_at.append(datetime)
            return snapshot

    class Service:
        def __init__(self, **options: Any) -> None:
            pass

        def backend(self, name: str, **options: Any) -> Any:
            if "use_fractional_gates" not in options:
                return Calibrated()
            seen.opened.append(options["use_fractional_gates"])
            return fake

        def job(self, job_id: str) -> Any:
            seen.jobs.append(job_id)
            return next(s.job for s in submissions or () if s.job.job_id() == job_id)

    monkeypatch.setattr(runtime, "QiskitRuntimeService", Service)
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    return seen


def test_the_script_runs_end_to_end_through_the_account_with_fractional_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen = _account(monkeypatch, _fractional_fez())
    output = tmp_path / "fez.counts.json"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert script.main(["ibm_fez", "--yes", "-o", str(output)]) == 0
    counts = load_counts(output)
    assert seen.pulled_at == [None, counts.run_at]
    assert seen.opened == [True]
    assert "rx" in {op.name for circuit in counts.circuits for op in circuit.ops}
    assert counts.profile.fingerprint != _fez_profile().fingerprint
    label, *command, ref, path = capsys.readouterr().out.rstrip("\n").splitlines()[-1].split()
    assert (label, command, path) == ("next", ["nv", "compare"], str(output))
    assert nv.load(ref).fingerprint == counts.profile.fingerprint


def test_running_the_same_command_again_collects_a_job_whose_wait_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    submissions = _recording_sampler(monkeypatch, lose_the_first_wait=True)
    _account(monkeypatch, _fractional_fez(), submissions)
    output, pending = tmp_path / "fez.counts.json", tmp_path / "fez.job.json"
    command = ["ibm_fez", "--yes", "-o", str(output)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert script.main(command) == 1
        (submission,) = submissions
        job_id = submission.job.job_id()
        assert capsys.readouterr().err == (
            f"error: could not collect the counts of job {job_id} (network is unreachable)\n"
            f"hint: run the same command again to collect them. {_new_job(pending)}\n"
        )
        assert (pending.exists(), output.exists()) == (True, False)
        del submission.job.result
        assert script.main(command) == 0
    assert len(submissions) == 1
    assert load_counts(output).execution.job_ids == (job_id,)
    assert not pending.exists()
    printed = capsys.readouterr().out
    assert printed.startswith(f"collecting job {job_id}, submitted earlier for {output}\n{WAITING}")


def _job_to_collect_on_fez(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], output: Path
) -> tuple[Account, Submission]:
    output.parent.mkdir()
    submissions = _recording_sampler(monkeypatch, lose_the_first_wait=True)
    seen = _account(monkeypatch, _fractional_fez(), submissions)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert script.main(["ibm_fez", "--yes", "-o", str(output)]) == 1
    (submission,) = submissions
    del submission.job.result
    capsys.readouterr()
    return seen, submission


def test_a_command_for_another_device_stops_before_it_opens_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "counts"
    output, pending = folder / "fez.counts.json", folder / "fez.job.json"
    seen, _ = _job_to_collect_on_fez(monkeypatch, capsys, output)
    record = pending.read_bytes()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert script.main(["ibm_kingston", "--yes", "-o", str(output)]) == 1
    assert capsys.readouterr().err == (
        f"error: {pending} records a job for ibm_fez, not for ibm_kingston\n"
        "hint: run the command with DEVICE ibm_fez to collect the job\n"
    )
    assert seen.jobs == []
    assert sorted(folder.iterdir()) == [pending]
    assert pending.read_bytes() == record


def _on_hardware(job: Any, device: str, program: str = "sampler") -> Any:
    runtime = require("qiskit_ibm_runtime")

    class HardwareJob(runtime.RuntimeJobV2):
        primitive_id = program

        def __init__(self) -> None:
            pass

        def job_id(self) -> str:
            return job.job_id()

        def result(self, *args: Any, **kwargs: Any) -> Any:
            return job.result(*args, **kwargs)

        def metrics(self) -> dict[str, Any]:
            return job.metrics()

        def backend(self, timeout: float | None = None) -> Any:
            return dataclasses.make_dataclass("Backend", [("name", str)])(device)

    return HardwareJob()


def test_a_job_that_ran_on_another_device_gives_no_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "counts"
    output, pending = folder / "fez.counts.json", folder / "fez.job.json"
    _, submission = _job_to_collect_on_fez(monkeypatch, capsys, output)
    job_id = submission.job.job_id()
    record = pending.read_bytes()
    service = require("qiskit_ibm_runtime").QiskitRuntimeService
    monkeypatch.setattr(
        service, "job", lambda self, job_id: _on_hardware(submission.job, "ibm_kingston")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert script.main(["ibm_fez", "--yes", "-o", str(output)]) == 1
    assert capsys.readouterr().err == (
        f"error: job {job_id} ran on ibm_kingston, but {pending} records a job for ibm_fez\n"
        f"hint: find the job for ibm_fez in your IBM Quantum account. {_new_job(pending)}\n"
    )
    assert sorted(folder.iterdir()) == [pending]
    assert pending.read_bytes() == record


def test_a_job_that_ran_another_ibm_program_gives_no_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "counts"
    output, pending = folder / "fez.counts.json", folder / "fez.job.json"
    _, submission = _job_to_collect_on_fez(monkeypatch, capsys, output)
    job_id = submission.job.job_id()
    record = pending.read_bytes()
    service = require("qiskit_ibm_runtime").QiskitRuntimeService
    monkeypatch.setattr(
        service, "job", lambda self, job_id: _on_hardware(submission.job, "ibm_fez", "executor")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert script.main(["ibm_fez", "--yes", "-o", str(output)]) == 1
    assert capsys.readouterr().err == (
        f"error: job {job_id} ran the IBM program 'executor', not 'sampler', so the options of"
        " the run are not proved\n"
        "hint: look at the options and the circuits of the job in your IBM Quantum account."
        f" {_new_job(pending)}\n"
    )
    assert sorted(folder.iterdir()) == [pending]
    assert pending.read_bytes() == record
    monkeypatch.setattr(
        service, "job", lambda self, job_id: _on_hardware(submission.job, "ibm_fez")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert script.main(["ibm_fez", "--yes", "-o", str(output)]) == 0
    assert output.exists()


@pytest.mark.parametrize("step", ["collect again", "submit a new job"])
@pytest.mark.parametrize("failing", ["open", "wait"])
def test_each_step_of_the_hint_after_a_failed_collect_works_with_collect_given(
    failing: str,
    step: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    submissions = _recording_sampler(monkeypatch, lose_the_first_wait=True)
    seen = _account(monkeypatch, _fractional_fez(), submissions)
    service = require("qiskit_ibm_runtime").QiskitRuntimeService
    folder = tmp_path / "counts"
    folder.mkdir()
    output, pending = folder / "fez.counts.json", folder / "fez.job.json"
    command = ["ibm_fez", "--yes", "-o", str(output)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert script.main(command) == 1
        (submission,) = submissions
        job_id = submission.job.job_id()
        capsys.readouterr()
        if failing == "open":
            del submission.job.result
            job = service.job

            def offline(self: Any, job_id: str) -> Any:
                raise ConnectionError("network is unreachable")

            monkeypatch.setattr(service, "job", offline)
        collect = [*command, "--collect", str(pending), "--job-id", job_id]
        assert script.main(collect) == 1
        problem, hint = {
            "open": (f"could not open job {job_id}", f"collect job {job_id}"),
            "wait": (f"could not collect the counts of job {job_id}", "collect them"),
        }[failing]
        assert capsys.readouterr().err == (
            f"error: {problem} (network is unreachable)\n"
            f"hint: run the same command again to {hint}. {_new_job(pending)}\n"
        )
        assert sorted(folder.iterdir()) == [pending]
        if failing == "open":
            monkeypatch.setattr(service, "job", job)
        else:
            del submission.job.result
        if step == "collect again":
            assert script.main(collect) == 0
        else:
            pending.unlink()
            assert script.main(command) == 0
    collected = submissions[-1].job.job_id()
    assert (len(submissions), collected == job_id) == (
        (1, True) if step == "collect again" else (2, False)
    )
    assert load_counts(output).execution.job_ids == (collected,)
    opened = {"open": 0, "wait": 1}[failing] + (step == "collect again")
    assert seen.jobs == [job_id] * opened
    assert sorted(folder.iterdir()) == [output]


@pytest.mark.parametrize("collect_to", ["another file", "the same file after a move"])
def test_a_file_that_appears_at_the_output_while_the_job_waits_is_kept_with_the_job(
    collect_to: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    folder = tmp_path / "my counts"
    folder.mkdir()
    output, pending = folder / "fez.counts.json", folder / "fez.job.json"
    submissions = _recording_sampler(
        monkeypatch, during_the_first_wait=lambda: output.write_text("another experiment\n")
    )
    _account(monkeypatch, _fractional_fez(), submissions)
    command = ["ibm_fez", "--yes", "-o", str(output)]
    hint = (
        f"hint: run the same command with --collect='{pending}' and a new -o file name. The"
        " script never replaces a file\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert script.main(command) == 1
        (submission,) = submissions
        job_id = submission.job.job_id()
        assert capsys.readouterr().err == (
            f"error: could not save the counts of job {job_id}, because {output} exists\n{hint}"
        )
        assert sorted(folder.iterdir()) == [output, pending]
        assert output.read_text() == "another experiment\n"
        assert script.main(command) == 1
        err = capsys.readouterr().err
        assert err == f"error: {output} exists\n{hint}"
        del submission.job.result
        if collect_to == "another file":
            saved = folder / "fez-2.counts.json"
            assert script.main([*command[:-1], str(saved), *_hinted(err)]) == 0
            assert output.read_text() == "another experiment\n"
        else:
            output.rename(folder / "other.counts.json")
            saved = output
            assert script.main(command) == 0
    assert len(submissions) == 1
    assert load_counts(saved).execution.job_ids == (job_id,)
    assert not pending.exists()


@pytest.mark.parametrize("collect", [True, False], ids=["--collect", "the same command again"])
def test_collect_submits_nothing_when_another_collect_removes_the_job_file_first(
    collect: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    submissions = _recording_sampler(monkeypatch)
    _account(monkeypatch, _fractional_fez(), submissions)
    folder = tmp_path / "counts"
    folder.mkdir()
    pending = folder / "fez.job.json"
    _save_job_file(_submitted_and_ran(_fez_profile())[0], pending)
    service = script._service

    def the_other_collect_ends_while_the_account_opens() -> Any:
        pending.unlink()
        return service()

    monkeypatch.setattr(script, "_service", the_other_collect_ends_while_the_account_opens)
    output = folder / ("fez-2.counts.json" if collect else "fez.counts.json")
    command = ["ibm_fez", "--yes", "-o", str(output)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        code = script.main([*command, "--collect", str(pending)] if collect else command)
    assert (code, submissions) == (1, [])
    assert capsys.readouterr().err == (
        f"error: no job file {pending}\n"
        "hint: another run collected the job or deleted the job file. The script did not submit"
        " a job\n"
    )
    assert list(folder.iterdir()) == []


@pytest.mark.parametrize("change", ["replaced", "moved", "changed in place"])
def test_a_job_file_that_changes_while_the_job_is_submitted_keeps_the_job_in_a_new_job_file(
    change: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    submissions = _recording_sampler(monkeypatch, lose_the_first_wait=True)
    _account(monkeypatch, _fractional_fez(), submissions)
    folder = tmp_path / "my counts"
    folder.mkdir()
    output, pending, moved = (
        folder / "fez.counts.json",
        folder / "fez.job.json",
        tmp_path / "moved.json",
    )
    submit = script.submit
    planned: list[bytes] = []
    inodes: list[int] = []

    def another_program_changes_the_job_file(*args: Any) -> Any:
        job = submit(*args)
        planned.append(pending.read_bytes())
        inodes.append(pending.stat().st_ino)
        if change == "changed in place":
            pending.write_text("another run\n")
            inodes.append(pending.stat().st_ino)
            return job
        if change == "moved":
            pending.rename(moved)
        _put(pending, "another run\n")
        return job

    monkeypatch.setattr(script, "submit", another_program_changes_the_job_file)
    command = ["ibm_fez", "--yes", "-o", str(output)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert script.main(command) == 1
        assert len(set(inodes)) == 1
        (submission,) = submissions
        job_id = submission.job.job_id()
        kept = folder / f"fez.{job_id}.job.json"
        err = capsys.readouterr().err
        assert err == (
            f"error: {pending} changed while the script submitted job {job_id}. The script saved"
            f" job {job_id} to {kept}\n"
            f"hint: run the same command with --collect='{kept}' to collect the job\n"
        )
        assert pending.read_text() == "another run\n"
        assert kept.read_bytes() == planned[0] + script._line({"job_id": job_id})
        if change == "moved":
            assert moved.read_bytes() == planned[0]
        del submission.job.result
        assert script.main([*command, *_hinted(err)]) == 0
    assert len(submissions) == 1
    assert load_counts(output).execution.job_ids == (job_id,)
    assert not kept.exists()
    assert pending.read_text() == "another run\n"


def test_each_printed_command_runs_for_file_names_that_start_with_a_hyphen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    submissions = _recording_sampler(monkeypatch, lose_the_first_wait=True)
    _account(monkeypatch, _fractional_fez(), submissions)
    monkeypatch.chdir(tmp_path)
    pending = Path("-run.job.json")
    submit = script.submit

    def another_program_changes_the_job_file(*args: Any) -> Any:
        job = submit(*args)
        _put(pending, "another run\n")
        return job

    monkeypatch.setattr(script, "submit", another_program_changes_the_job_file)
    command = ["ibm_fez", "--yes", "-o", "./-run.counts.json"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert script.main(command) == 1
        (submission,) = submissions
        job_id = submission.job.job_id()
        kept = f"-run.{job_id}.job.json"
        err = capsys.readouterr().err
        assert err == (
            f"error: {pending} changed while the script submitted job {job_id}. The script saved"
            f" job {job_id} to {kept}\n"
            f"hint: run the same command with --collect={kept} to collect the job\n"
        )
        Path("-run.counts.json").write_text("another experiment\n")
        assert script.main([*command, *_hinted(err)]) == 1
        err = capsys.readouterr().err
        assert err == (
            "error: -run.counts.json exists\n"
            f"hint: run the same command with --collect={kept} and a new -o file name. The script"
            " never replaces a file\n"
        )
        del submission.job.result
        assert script.main(["ibm_fez", "--yes", "-o", "./-run-2.counts.json", *_hinted(err)]) == 0
    label, printed = capsys.readouterr().out.rstrip("\n").splitlines()[-1].split(maxsplit=1)
    nv_, compare, *args = shlex.split(printed)
    assert (label, nv_, compare, args[0], args[2:]) == (
        "next",
        "nv",
        "compare",
        "--",
        ["-run-2.counts.json"],
    )
    result = CliRunner().invoke(app, [compare, *args], env={"COLUMNS": "80"})
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    header = " ".join(line.strip() for line in lines[: lines.index("")])
    assert " counts -run-2.counts.json, simulated, sha256:" in header


def test_a_job_file_that_changes_on_a_full_disk_while_the_job_is_submitted_names_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    submissions = _recording_sampler(monkeypatch)
    _account(monkeypatch, _fractional_fez(), submissions)
    folder = tmp_path / "counts"
    folder.mkdir()
    output, pending = folder / "fez.counts.json", folder / "fez.job.json"
    disk = _disk(monkeypatch, folder)
    submit = script.submit

    def another_program_replaces_the_job_file_and_fills_the_disk(*args: Any) -> Any:
        job = submit(*args)
        _put(pending, "another run\n")
        disk.bytes_left = 0
        return job

    monkeypatch.setattr(script, "submit", another_program_replaces_the_job_file_and_fills_the_disk)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert script.main(["ibm_fez", "--yes", "-o", str(output)]) == 1
    (submission,) = submissions
    job_id = submission.job.job_id()
    kept = folder / f"fez.{job_id}.job.json"
    assert capsys.readouterr().err == (
        f"error: {pending} changed while the script submitted job {job_id}, and the script could"
        f" not save job {job_id} to {kept} ([Errno 28] No space left on device: '{kept}')\n"
        f"hint: find job {job_id} in your IBM Quantum account. The script did not save the planned"
        " circuits of the job\n"
    )
    assert sorted(folder.iterdir()) == [pending]
    assert pending.read_text() == "another run\n"


@pytest.mark.parametrize(
    "fault", ["replaced", "changed in place", "emptied in place", "not cut back"]
)
def test_a_failed_job_id_write_that_leaves_no_job_file_to_collect_keeps_the_job_in_a_new_job_file(
    fault: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    submissions = _recording_sampler(monkeypatch)
    _account(monkeypatch, _fractional_fez(), submissions)
    folder = tmp_path / "my counts"
    folder.mkdir()
    output, pending = folder / "fez.counts.json", folder / "fez.job.json"
    another_run = tmp_path / "another.job.json"
    _save_job_file(_submitted_and_ran(_fez_profile())[0], another_run)
    in_place = {"changed in place": another_run.read_bytes(), "emptied in place": b""}
    disk = _disk(monkeypatch, folder)
    submit, write = script.submit, os.write
    planned: list[bytes] = []
    inodes: list[int] = []

    def submit_and_fill_the_disk(*args: Any) -> Any:
        job = submit(*args)
        planned.append(pending.read_bytes())
        disk.bytes_left = 5
        return job

    def another_run_changes_the_job_file_on_the_full_disk(fd: int, data: Any) -> int:
        if disk.bytes_left != 0:
            return write(fd, data)
        if fault == "replaced":
            _put(pending, another_run.read_text())
        if fault in in_place:
            inodes.append(pending.stat().st_ino)
            pending.write_bytes(in_place[fault])
            inodes.append(pending.stat().st_ino)
        try:
            return write(fd, data)
        finally:
            disk.bytes_left = None

    def the_cut_fails(fd: int, length: int) -> None:
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(script, "submit", submit_and_fill_the_disk)
    monkeypatch.setattr(os, "write", another_run_changes_the_job_file_on_the_full_disk)
    if fault == "not cut back":
        monkeypatch.setattr(os, "ftruncate", the_cut_fails)
    command = ["ibm_fez", "--yes", "-o", str(output)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert script.main(command) == 1
        assert len(set(inodes)) == (fault in in_place)
        (submission,) = submissions
        job_id = submission.job.job_id()
        kept = folder / f"fez.{job_id}.job.json"
        job_line = script._line({"job_id": job_id})
        changed = f"{pending} changed while the script submitted job {job_id}"
        problem, left = {
            "replaced": (changed, another_run.read_bytes()),
            "changed in place": (changed, another_run.read_bytes()),
            "emptied in place": (changed, b""),
            "not cut back": (
                f"could not save job {job_id} to {pending} ([Errno 28] No space left on device)",
                planned[0] + job_line[:5],
            ),
        }[fault]
        err = capsys.readouterr().err
        assert err == (
            f"error: {problem}. The script saved job {job_id} to {kept}\n"
            f"hint: run the same command with --collect='{kept}' to collect the job\n"
        )
        assert kept.read_bytes() == planned[0] + job_line
        assert pending.read_bytes() == left
        assert script.main([*command, *_hinted(err)]) == 0
    assert len(submissions) == 1
    assert load_counts(output).execution.job_ids == (job_id,)
    assert not kept.exists()
    assert pending.read_bytes() == left


def test_a_job_file_that_another_run_saves_in_place_between_two_job_id_writes_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    submissions = _recording_sampler(monkeypatch)
    _account(monkeypatch, _fractional_fez(), submissions)
    folder = tmp_path / "my counts"
    folder.mkdir()
    output, pending = folder / "fez.counts.json", folder / "fez.job.json"
    submit, write = script.submit, os.write
    planned: list[bytes] = []
    another_run: list[bytes] = []
    inodes: list[int] = []

    def submit_and_plan_another_run(*args: Any) -> Any:
        job = submit(*args)
        planned.append(pending.read_bytes())
        another_id = "x" * len(job.job_id())
        assert another_id != job.job_id()
        another_run.append(planned[0] + script._line({"job_id": another_id}))
        return job

    def five_bytes_then_another_run_saves_its_job(fd: int, data: Any) -> int:
        if not another_run or os.fstat(fd).st_ino != pending.stat().st_ino:
            return write(fd, data)
        inodes.append(pending.stat().st_ino)
        written = write(fd, data[:5])
        pending.write_bytes(another_run.pop())
        inodes.append(pending.stat().st_ino)
        return written

    monkeypatch.setattr(script, "submit", submit_and_plan_another_run)
    monkeypatch.setattr(os, "write", five_bytes_then_another_run_saves_its_job)
    command = ["ibm_fez", "--yes", "-o", str(output)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert script.main(command) == 1
        assert len(inodes) == 2
        assert len(set(inodes)) == 1
        (submission,) = submissions
        job_id = submission.job.job_id()
        kept = folder / f"fez.{job_id}.job.json"
        saved = planned[0] + script._line({"job_id": "x" * len(job_id)})
        err = capsys.readouterr().err
        assert err == (
            f"error: {pending} changed while the script submitted job {job_id}. The script saved"
            f" job {job_id} to {kept}\n"
            f"hint: run the same command with --collect='{kept}' to collect the job\n"
        )
        assert pending.read_bytes() == saved
        assert kept.read_bytes() == planned[0] + script._line({"job_id": job_id})
        assert script.main([*command, *_hinted(err)]) == 0
    assert len(submissions) == 1
    assert load_counts(output).execution.job_ids == (job_id,)
    assert not kept.exists()
    assert pending.read_bytes() == saved


def test_a_job_file_that_the_script_cannot_read_gives_an_error_and_submits_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    submissions = _recording_sampler(monkeypatch)
    _account(monkeypatch, _fractional_fez(), submissions)
    output, pending = tmp_path / "fez.counts.json", tmp_path / "fez.job.json"
    _save_job_file(_submitted_and_ran(_fez_profile())[0], pending)
    real_open = os.open

    def open_(file: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if Path(file) == pending and not flags & os.O_CREAT:
            raise PermissionError(errno.EACCES, "Permission denied", str(file))
        return real_open(file, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", open_)
    assert script.main(["ibm_fez", "--yes", "-o", str(output)]) == 1
    assert capsys.readouterr().err == (
        f"error: could not read {pending} ([Errno 13] Permission denied: '{pending}')\n"
        "hint: make the job file readable. Then run the same command again\n"
    )
    assert submissions == []
    assert sorted(tmp_path.iterdir()) == [pending]


def test_a_job_file_that_the_script_cannot_delete_after_it_saves_the_counts_gives_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = _collecting_job_1(tmp_path, monkeypatch)
    output, pending = tmp_path / "fez.counts.json", tmp_path / "fez.job.json"
    unlink = Path.unlink

    def refused(path: Path, *args: Any, **kwargs: Any) -> None:
        if path == pending:
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refused)
    assert run() is not None
    printed = capsys.readouterr()
    assert printed.err == (
        f"warning: could not delete {pending} ([Errno 13] Permission denied: '{pending}'). The"
        " script saved the counts, so you can delete the job file\n"
    )
    label, *command, _, path = printed.out.rstrip("\n").splitlines()[-1].split()
    assert (label, command, path) == ("next", ["nv", "compare"], str(output))
    assert load_counts(output).execution.job_ids == ("job-1",)
    assert pending.exists()


def test_a_disk_that_fills_when_the_job_is_submitted_keeps_the_job_for_the_next_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    submissions = _recording_sampler(monkeypatch)
    _account(monkeypatch, _fractional_fez(), submissions)
    folder = tmp_path / "counts"
    folder.mkdir()
    output, pending = folder / "fez.counts.json", folder / "fez.job.json"
    disk = _disk(monkeypatch, folder)
    submit = script.submit

    def submit_and_fill_the_disk(*args: Any) -> Any:
        job = submit(*args)
        disk.bytes_left = 5
        return job

    monkeypatch.setattr(script, "submit", submit_and_fill_the_disk)
    command = ["ibm_fez", "--yes", "-o", str(output)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert script.main(command) == 1
        (submission,) = submissions
        job_id = submission.job.job_id()
        err = capsys.readouterr().err
        assert err == (
            f"error: could not save job {job_id} to {pending} ([Errno 28] No space left on"
            " device)\n"
            f"hint: run the same command with --job-id {job_id} to collect the job\n"
        )
        assert sorted(folder.iterdir()) == [pending]
        disk.bytes_left = None
        assert script.main([*command, *_hinted(err)]) == 0
    assert len(submissions) == 1
    assert load_counts(output).execution.job_ids == (job_id,)
    assert not pending.exists()


@pytest.mark.parametrize(
    ("with_id", "args", "message", "hint"),
    [
        (
            False,
            [],
            "{pending} has no job id",
            "give --job-id the job id that the script printed or that your IBM Quantum account"
            " shows. To submit a new job, delete {pending}. Then run the command without --collect"
            " and --job-id",
        ),
        (
            True,
            ["--job-id", "job-2"],
            "{pending} records job job-1, not job job-2",
            "run the same command without --job-id to collect job job-1",
        ),
    ],
    ids=["no job id", "another job id"],
)
def test_a_job_file_without_the_given_job_id_stops_before_the_job_opens(
    with_id: bool,
    args: list[str],
    message: str,
    hint: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    submissions = _recording_sampler(monkeypatch)
    _account(monkeypatch, _fractional_fez(), submissions)
    output, pending = tmp_path / "fez.counts.json", tmp_path / "fez.job.json"
    _save_job_file(_submitted_and_ran(_fez_profile())[0], pending, with_id=with_id)
    record = pending.read_bytes()
    assert script.main(["ibm_fez", "--yes", "-o", str(output), *args]) == 1
    assert capsys.readouterr().err == (
        f"error: {message.format(pending=pending)}\nhint: {hint.format(pending=pending)}\n"
    )
    assert submissions == []
    assert sorted(tmp_path.iterdir()) == [pending]
    assert pending.read_bytes() == record


def _with(**changes: Any) -> Callable[[dict[str, Any]], str]:
    return lambda data: json.dumps({**data, **changes}) + "\n"


def _plan_line_cut_short(data: dict[str, Any]) -> str:
    return '{"profile": {\n' + json.dumps({"job_id": data["job_id"]}) + "\n"


def _profile_format_twice(data: dict[str, Any]) -> str:
    plan = json.dumps({k: v for k, v in data.items() if k != "job_id"})
    plan = plan.replace('"profile": {', '"profile": {"noisevault": "0.1", ', 1)
    return plan + "\n" + json.dumps({"job_id": data["job_id"]}) + "\n"


def _job_id_twice(data: dict[str, Any]) -> str:
    plan = json.dumps({k: v for k, v in data.items() if k != "job_id"})
    return plan + '\n{"job_id": "job-1", "job_id": "job-2"}\n'


def _second_circuit_named_ghz_chain(data: dict[str, Any]) -> str:
    data["planned"][1]["name"] = "ghz_chain"
    return json.dumps(data) + "\n"


def _ghz_chain_on_qubit_156(data: dict[str, Any]) -> str:
    data["planned"][0]["qubits"][0] = 156
    return json.dumps(data) + "\n"


def _ghz_chain_with_cx(data: dict[str, Any]) -> str:
    ops = data["planned"][0]["ops"]
    ops[[op[0] for op in ops].index("cz")][0] = "cx"
    return json.dumps(data) + "\n"


_TOO_MANY_OUTCOMES = [{"name": f"wide_{i}", "qubits": list(range(10)), "ops": []} for i in range(5)]


@pytest.mark.parametrize(
    ("damage", "problem", "named"),
    [
        (_with(options=5), "Input should be a valid dictionary", "job-1"),
        (
            _with(options={"twirling": {"enable_gates": True}}),
            "options.twirling.enable_gates is true",
            "job-1",
        ),
        (_plan_line_cut_short, "Expecting property name", "job-1"),
        (_profile_format_twice, "(the key profile.noisevault appears twice)", "job-1"),
        (_job_id_twice, "(the key job_id appears twice)", None),
        (_with(job_id=5), "Input should be a valid string", None),
        (_with(job_id=""), "the job id is empty", None),
        (_second_circuit_named_ghz_chain, 'circuits[1] repeats the name "ghz_chain"', "job-1"),
        (_with(planned=_TOO_MANY_OUTCOMES), "the circuits have 5120 outcomes in all", "job-1"),
        (
            _ghz_chain_on_qubit_156,
            "circuit ghz_chain measures qubit 156, but ibm_fez has qubits 0..155",
            "job-1",
        ),
        (
            _ghz_chain_with_cx,
            "circuit ghz_chain: cx on qubits 136-143: cx is not defined in this profile",
            "job-1",
        ),
    ],
    ids=[
        "options not an object",
        "options against the format",
        "plan line cut short",
        "profile key twice",
        "job id twice",
        "job id a number",
        "no job id",
        "two circuits with one name",
        "too many outcomes",
        "a qubit the device does not have",
        "a gate the profile does not define",
    ],
)
def test_a_damaged_job_file_stops_before_the_calibration_pull_and_names_its_job(
    damage: Callable[[dict[str, Any]], str],
    problem: str,
    named: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    submissions = _recording_sampler(monkeypatch)
    seen = _account(monkeypatch, _fractional_fez(), submissions)
    output, pending = tmp_path / "fez.counts.json", tmp_path / "fez.job.json"
    _save_job_file(_submitted_and_ran(_fez_profile())[0], pending)
    pending.write_text(damage(json.loads(pending.read_text())))
    record = pending.read_bytes()
    assert script.main(["ibm_fez", "--yes", "-o", str(output)]) == 1
    printed = capsys.readouterr().err
    hint = (
        "the script cannot collect a job from the damaged job file. Look for the job in your IBM"
        " Quantum account"
        if named is None
        else f"the script cannot collect job {named} from the damaged job file. Find job {named}"
        " in your IBM Quantum account"
    )
    assert printed.startswith(f"error: {pending} is damaged (")
    assert problem in printed
    assert printed.endswith(f")\nhint: {hint}. {_new_job(pending)}\n")
    assert (submissions, seen.pulled_at, seen.opened, seen.jobs) == ([], [], [], [])
    assert sorted(tmp_path.iterdir()) == [pending]
    assert pending.read_bytes() == record


def test_a_backend_without_the_planned_fractional_gate_refuses_the_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen = _account(monkeypatch, _fez())
    assert script.main(["ibm_fez", "--yes", "-o", str(tmp_path / "fez.counts.json")]) == 1
    assert seen.opened == [True]
    assert capsys.readouterr().err == (
        "error: circuit ghz_chain: the backend does not support rx on qubit 136\n"
    )


@pytest.mark.parametrize(
    ("failing", "reason"),
    [("target", "'network is unreachable'"), ("backend", "'No backend matches the criteria.'")],
)
def test_an_ibm_failure_while_the_script_opens_the_backend_asks_to_run_again(
    failing: str,
    reason: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    api = require("qiskit_ibm_runtime.api.exceptions")
    providers = require("qiskit.providers.exceptions")
    submissions = _recording_sampler(monkeypatch)
    _account(monkeypatch, _fractional_fez())
    service = require("qiskit_ibm_runtime").QiskitRuntimeService
    calibrated = service.backend

    class Offline:
        @property
        def target(self) -> Any:
            raise api.RequestsApiError("network is unreachable")

    def backend(self: Any, name: str, **options: Any) -> Any:
        if "use_fractional_gates" not in options:
            return calibrated(self, name, **options)
        if failing == "target":
            return Offline()
        raise providers.QiskitBackendNotFoundError("No backend matches the criteria.")

    monkeypatch.setattr(service, "backend", backend)
    folder = tmp_path / "counts"
    folder.mkdir()
    assert script.main(["ibm_fez", "--yes", "-o", str(folder / "fez.counts.json")]) == 1
    assert capsys.readouterr().err == (
        f"error: could not open ibm_fez through your IBM Quantum account ({reason})\n"
        "hint: run the same command again. The script did not submit a job\n"
    )
    assert submissions == []
    assert list(folder.iterdir()) == []


def test_without_a_terminal_the_script_asks_for_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    submissions = _recording_sampler(monkeypatch)
    _account(monkeypatch, _fractional_fez())
    monkeypatch.setattr(sys, "stdin", io.StringIO("y\n"))
    folder = tmp_path / "counts"
    folder.mkdir()
    assert script.main(["ibm_fez", "-o", str(folder / "fez.counts.json")]) == 1
    assert capsys.readouterr().err == (
        "error: cannot ask before submitting, because standard input is not a terminal\n"
        "hint: give --yes to submit without asking\n"
    )
    assert submissions == []
    assert list(folder.iterdir()) == []


def test_started_at_reads_ibm_timestamps_as_utc() -> None:
    with _five_hours_behind_utc():
        for stamp in ("2026-04-16T09:30:02.5Z", "2026-04-16T09:30:02.5"):
            assert script.started_at({"timestamps": {"running": stamp}}) == datetime(
                2026, 4, 16, 9, 30, 2, 500000, tzinfo=UTC
            )


def test_a_missing_account_gives_the_setup_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = require("qiskit_ibm_runtime")
    missing = runtime.accounts.AccountNotFoundError("Unable to find account.")

    def no_account(**options: Any) -> None:
        raise missing

    monkeypatch.setattr(runtime, "QiskitRuntimeService", no_account)
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    assert script.main(["ibm_fez", "-o", str(tmp_path / "fez.counts.json")]) == 1
    assert capsys.readouterr() == (
        "",
        f"error: could not open your IBM Quantum account ({missing})\n"
        f"hint: {script.ACCOUNT_SETUP}\n",
    )
    assert list(tmp_path.iterdir()) == []


NEW_NAME = "give -o a new file name. The script never replaces a file"


@pytest.mark.parametrize(
    ("existing", "args", "message", "hint"),
    [
        ("fez.counts.json", ["-o", "fez.counts.json"], "fez.counts.json exists", NEW_NAME),
        ("fez.timing.json", ["-o", "fez.counts.json"], "fez.timing.json exists", NEW_NAME),
        ("fez.profile.json", ["-o", "fez.counts.json"], "fez.profile.json exists", NEW_NAME),
        (
            "fez.counts.json",
            ["-o", "nowhere/fez.counts.json"],
            "no folder nowhere",
            "create it, or give -o another path",
        ),
        (
            "fez.counts.json",
            ["-o", "fez-2.counts.json", "--collect", "fez-2.job.json"],
            "no job file fez-2.job.json",
            "give --collect the .job.json file that the script saved beside the counts file",
        ),
        (
            "fez.counts.json",
            ["-o", "fez-2.counts.json", "--job-id", "job-1"],
            "no job file fez-2.job.json",
            "give --collect the .job.json file that the script saved beside the counts file",
        ),
    ],
)
def test_an_output_the_run_could_not_write_is_refused_before_the_account_opens(
    existing: str,
    args: list[str],
    message: str,
    hint: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = require("qiskit_ibm_runtime")

    def account(**options: Any) -> None:
        raise AssertionError("the script opened the IBM account")

    monkeypatch.setattr(runtime, "QiskitRuntimeService", account)
    monkeypatch.chdir(tmp_path)
    (tmp_path / existing).write_text("paid for\n")
    assert script.main(["ibm_fez", *args]) == 1
    assert capsys.readouterr() == ("", f"error: {message}\nhint: {hint}\n")
    assert sorted(tmp_path.iterdir()) == [tmp_path / existing]
    assert (tmp_path / existing).read_text() == "paid for\n"


def test_a_qubit_that_the_backend_cannot_measure_is_refused() -> None:
    profile = _fez_profile()
    circuit = plan(profile)[0]
    target = _fez().target
    unmeasured = circuit.qubits[-1]

    class NoMeasure:
        """The fake's target, except that it cannot measure one qubit."""

        def instruction_supported(self, name: str, qargs: Any = None, **kwargs: Any) -> bool:
            if (name, qargs) == ("measure", (unmeasured,)):
                return False
            return bool(target.instruction_supported(name, qargs, **kwargs))

        def __getattr__(self, name: str) -> Any:
            return getattr(target, name)

        def __getitem__(self, name: str) -> Any:
            return target[name]

    with pytest.raises(NoiseVaultError) as caught:
        script.isa_circuit(circuit, profile, NoMeasure())
    assert caught.value.message == (
        f"circuit {circuit.name}: the backend does not support measure on qubit {unmeasured}"
    )
    script.isa_circuit(circuit, profile, target)
