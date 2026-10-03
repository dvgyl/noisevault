# /// script
# requires-python = ">=3.11"
# dependencies = ["noisevault[ibm] @ git+https://github.com/dvgyl/noisevault"]
# ///
r"""Run the nv compare circuits on an IBM device and save the counts for nv compare.

    uv run \
      https://raw.githubusercontent.com/dvgyl/noisevault/main/scripts/run_on_ibm.py \
      ibm_kingston --shots 4000 -o kingston-0416.counts.json

uv installs the dependencies listed at the top of this file. In a clone with the ``ibm`` extra,
run ``python scripts/run_on_ibm.py`` with the same arguments.

The script pulls the device's calibration through your IBM Quantum account and plans the
circuits with ``noisevault.counts.plan``. It checks every op and delay against the backend, shows
IBM's usage estimate, and asks before it submits one SamplerV2 job. When the job is done, the
script writes a counts file and binds the counts to the calibration in effect at that time. Then
the script prints the ``nv compare`` command to run next. If the wait for the job stops, run
the same command again. The script then collects the submitted job and does not submit another.

The script needs an IBM Quantum account. Save the account with
``QiskitRuntimeService.save_account``, or put an API key in ``IBM_QUANTUM_TOKEN``.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import json
import os
import shlex
import sys
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import noisevault as nv  # noqa: E402
from noisevault import catalog, gates  # noqa: E402
from noisevault.compare import _bind  # noqa: E402
from noisevault.counts import (  # noqa: E402
    _RUN_RULES,
    COUNTS_FORMAT,
    SAMPLER_V2_OPTIONS,
    MeasuredCounts,
    PlannedCircuit,
    _duration,
    plan,
)
from noisevault.errors import (  # noqa: E402
    CountsError,
    NoiseVaultError,
    install_hint,
    parse_json,
    qubit_loci,
)
from noisevault.profile import (  # noqa: E402
    Profile,
    _readable_json,
    exact_ref,
    file_bytes,
    ref_on_day,
)
from noisevault.sources import ibm_account  # noqa: E402

DEFAULT_SHOTS = 4000
FRACTIONAL_GATES = frozenset({"rx", "rzz"})
CLBITS = "meas"
ACCOUNT_SETUP = (
    "set IBM_QUANTUM_TOKEN to your IBM Quantum API key, or save the key once with"
    " QiskitRuntimeService.save_account(token=...)"
)
LABEL = 16
SUB_JOB_OVERHEAD_S = 2.0
BINARY = getattr(os, "O_BINARY", 0)

Calibration = Callable[[datetime | None], Profile]


@dataclass(frozen=True)
class Batch:
    profile: Profile
    planned: tuple[PlannedCircuit, ...]
    circuits: tuple[Any, ...]
    durations_ns: tuple[float, ...]
    shots: int
    options: dict[str, Any]
    usage_s: float


@dataclass(frozen=True)
class IsaCircuit:
    circuit: Any
    measure_at_ns: float


@dataclass(frozen=True)
class Submitted:
    job_id: str
    profile: Profile
    planned: tuple[PlannedCircuit, ...]
    options: dict[str, Any]


@dataclass(frozen=True)
class Binding:
    profile: Profile
    warning: str | None = None
    hint: str | None = None


@dataclass(frozen=True)
class Ran:
    job_id: str
    source: str
    run_at: datetime
    counts: tuple[dict[str, int], ...]
    timing: dict[str, Any]


@dataclass
class Owned:
    path: Path
    fd: int
    data: bytes = b""
    start: int = field(init=False)

    def __post_init__(self) -> None:
        self.start = len(self.data)

    def _held(self) -> int | None:
        try:
            if not os.path.samestat(os.lstat(self.path), os.fstat(self.fd)):
                return None
            os.lseek(self.fd, 0, os.SEEK_SET)
            content = b""
            while len(content) <= len(self.data) and (
                chunk := os.read(self.fd, len(self.data) + 1 - len(content))
            ):
                content += chunk
        except OSError:
            return None
        if self.start <= len(content) <= len(self.data) and self.data.startswith(content):
            return len(content)
        return None

    def holds(self) -> bool:
        return self._held() is not None

    def check(self) -> int:
        held = self._held()
        if held is None:
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(self.path))
        return held

    def append(self, data: bytes) -> None:
        held = self.check()
        self.data += data
        while held < len(self.data):
            os.write(self.fd, memoryview(self.data)[held:])
            held = self.check()
        self.start = len(self.data)

    def cut(self, size: int) -> None:
        self.check()
        os.ftruncate(self.fd, size)
        self.data = self.data[:size]
        self.start = size

    def drop(self) -> None:
        ours = self.holds()
        os.close(self.fd)
        if ours:
            self.path.unlink(missing_ok=True)


def prepare(
    calibration: Calibration, open_backend: Callable[[bool], Any], shots: int
) -> tuple[Any, Batch]:
    profile = calibration(None)
    planned = plan(profile)
    fractional = any(op.name in FRACTIONAL_GATES for c in planned for op in c.ops)
    try:
        backend = open_backend(fractional)
        target = backend.target
        rep_delay = backend.configuration().default_rep_delay
    except Exception as exc:
        raise NoiseVaultError(
            f"could not open {profile.device.name} through your IBM Quantum account"
            f" ({_reason(exc)})",
            hint="run the same command again. The script did not submit a job",
        ) from None
    built = [isa_circuit(circuit, profile, target) for circuit in planned]
    circuits = tuple(b.circuit for b in built)
    return backend, Batch(
        profile=profile,
        planned=planned,
        circuits=circuits,
        durations_ns=tuple(b.measure_at_ns for b in built),
        shots=shots,
        options=sampler_options(shots, rep_delay),
        usage_s=usage_seconds(planned, circuits, target, rep_delay, shots),
    )


def isa_circuit(circuit: PlannedCircuit, profile: Profile, target: Any) -> IsaCircuit:
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.circuit import Delay

    def refuse(message: str, hint: str | None = None) -> NoiseVaultError:
        return NoiseVaultError(f"circuit {circuit.name}: {message}", hint=hint)

    dt_ns = target.dt * 1e9
    qc = QuantumCircuit(
        QuantumRegister(target.num_qubits, "q"),
        ClassicalRegister(len(circuit.qubits), CLBITS),
        name=circuit.name,
    )
    free = [0.0] * len(circuit.qubits)
    for op in circuit.ops:
        qubits = tuple(circuit.qubits[q] for q in op.qubits)
        name = "delay" if op.name == "delay" else gates.GATES[op.name].qiskit
        if name is None or not target.instruction_supported(name, qubits):
            raise refuse(f"the backend does not support {op.name} on {qubit_loci(qubits)}")
        if op.name == "delay":
            length = op.params[0] / dt_ns
            if _whole(length) is None:
                raise refuse(
                    f"the {op.params[0]:g} ns delay on {qubit_loci(qubits)} is {length:g} dt, and"
                    f" the backend times delays in whole dt of {dt_ns:g} ns"
                )
            if round(length) < target.min_length:
                raise refuse(
                    f"the {op.params[0]:g} ns delay on {qubit_loci(qubits)} is {round(length)} dt,"
                    f" shorter than the backend's minimum of {target.min_length} dt"
                )
            instruction = Delay(round(length), unit="dt")
        else:
            calibrated = float(_duration(profile, op, circuit.qubits)) / dt_ns
            length = _seconds(target, name, qubits) / target.dt
            if abs(length - calibrated) > 1e-6:
                raise refuse(
                    f"{op.name} on {qubit_loci(qubits)} lasts {_dt(calibrated, dt_ns)} in the"
                    f" calibration and {_dt(length, dt_ns)} on the backend, so the planned delays"
                    " would not fill the gaps",
                    hint="run the script again to plan from the current calibration",
                )
            operation = target.operation_from_name(name)
            instruction = operation.base_class(*op.params) if op.params else operation
        start = max(free[q] for q in op.qubits)
        if _whole(start / target.pulse_alignment) is None:
            raise refuse(
                f"{op.name} on {qubit_loci(qubits)} would start at {start:g} dt, off the backend's"
                f" grid of {target.pulse_alignment} dt"
            )
        qc.append(instruction, list(qubits))
        for q in op.qubits:
            free[q] = start + length
    end = max(free)
    if _whole(end / target.acquire_alignment) is None:
        raise refuse(
            f"the measurements would start at {end:g} dt, off the backend's grid of"
            f" {target.acquire_alignment} dt"
        )
    qc.barrier(list(circuit.qubits))
    for i, q in enumerate(circuit.qubits):
        qc.measure(q, i)
    return IsaCircuit(qc, end * dt_ns)


def sampler_options(shots: int, rep_delay: float) -> dict[str, Any]:
    """Every option the job sets.

    The client does not send an unset option, so the server uses its own default for that option.
    """
    options: dict[str, Any] = {"default_shots": shots}
    for path, value in {**SAMPLER_V2_OPTIONS, "execution.rep_delay": rep_delay}.items():
        *parents, leaf = path.split(".")
        node = options
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    options["experimental"] = {"execution": {"scheduler_timing": True}}
    return options


def usage_seconds(
    planned: Sequence[PlannedCircuit],
    circuits: Sequence[Any],
    target: Any,
    rep_delay: float,
    shots: int,
) -> float:
    """IBM's estimate, from https://quantum.cloud.ibm.com/docs/en/guides/estimate-job-run-time"""
    total = SUB_JOB_OVERHEAD_S
    for circuit, qc in zip(planned, circuits, strict=True):
        reset = max(_seconds(target, "reset", (q,)) for q in circuit.qubits)
        total += (qc.estimate_duration(target, unit="s") + reset + rep_delay) * shots
    return total


def submit(backend: Any, batch: Batch) -> Any:
    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=backend, options=batch.options)
    try:
        return sampler.run(list(batch.circuits))
    except Exception as exc:
        raise NoiseVaultError(
            f"submitting the job failed ({_reason(exc)})",
            hint="check your IBM Quantum account for a new job before you run the script again",
        ) from None


def collect(job: Any, submitted: Submitted) -> Ran:
    from qiskit_ibm_runtime import RuntimeJobV2

    result = job.result()
    timing = {}
    for circuit, pub in zip(submitted.planned, result, strict=True):
        compiled = pub.metadata.get("compilation", {}).get("scheduler_timing", {})
        if "timing" in compiled:
            timing[circuit.name] = compiled["timing"]
    return Ran(
        job_id=submitted.job_id,
        source="hardware" if isinstance(job, RuntimeJobV2) else "simulated",
        run_at=started_at(job.metrics()),
        counts=tuple(getattr(pub.data, CLBITS).get_counts() for pub in result),
        timing=timing,
    )


def started_at(metrics: dict[str, Any]) -> datetime:
    """When the job started running, in UTC. IBM sends an ISO 8601 string, which the runtime
    reads as UTC when it has no offset. Local testing mode gives a naive datetime in local time,
    which ``astimezone`` reads as local time."""
    stamp = metrics["timestamps"]["running"]
    if isinstance(stamp, str):
        stamp = datetime.fromisoformat(stamp)
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
    return stamp.astimezone(UTC)


def bind(submitted: Submitted, ran: Ran, calibration: Calibration) -> Binding:
    planned = submitted.profile
    try:
        latest = calibration(ran.run_at)
    except Exception as exc:
        return Binding(
            planned,
            f"could not pull the calibration in effect when the job ran ({_reason(exc)}). The"
            f" counts bind to the planned calibration {planned.short_fingerprint}",
        )
    before, after = planned.device.calibrated_at, latest.device.calibrated_at
    if before is not None and after is not None and after < before:
        return Binding(
            planned,
            f"the calibration IBM returned for the time the job ran, {latest.short_fingerprint},"
            f" is older than the planned {planned.short_fingerprint}. The counts bind to the"
            " planned calibration",
        )
    problem = _unlike_the_plan(latest, submitted, ran)
    if problem is None:
        return Binding(latest)
    return Binding(
        planned,
        f"IBM recalibrated {planned.device.name} before the job ran"
        f" ({latest.short_fingerprint}), and {problem}. The counts bind to the planned"
        f" calibration {planned.short_fingerprint}",
        "run the script again for counts that match one calibration",
    )


def _unlike_the_plan(latest: Profile, submitted: Submitted, ran: Ran) -> str | None:
    try:
        _bind(latest, counts_file(latest, submitted, ran))
    except CountsError as exc:
        return f"nv compare would refuse the counts against it ({exc.message})"
    for circuit in submitted.planned:
        for op in circuit.ops:
            if op.name == "delay":
                continue
            before = _duration(submitted.profile, op, circuit.qubits)
            after = _duration(latest, op, circuit.qubits)
            if before != after:
                qubits = tuple(circuit.qubits[q] for q in op.qubits)
                return (
                    f"{op.name} on {qubit_loci(qubits)} now takes {float(after):g} ns, not"
                    f" {float(before):g} ns, so the submitted delays may not match the timeline"
                    " that ran"
                )
    return None


def counts_file(profile: Profile, submitted: Submitted, ran: Ran) -> MeasuredCounts:
    return MeasuredCounts.model_validate(
        {
            "nv_counts": COUNTS_FORMAT,
            "source": ran.source,
            "profile": {"id": profile.id, "fingerprint": profile.calibration_fingerprint},
            "backend": profile.device.name,
            "run_at": ran.run_at,
            "bit_order": "qiskit",
            "execution": _execution(submitted.options, ran.job_id),
            "circuits": [
                {
                    "name": circuit.name,
                    "qubits": circuit.qubits,
                    "ops": circuit.ops,
                    "shots": sum(counts.values()),
                    "counts": counts,
                }
                for circuit, counts in zip(submitted.planned, ran.counts, strict=True)
            ],
        }
    )


def _execution(options: Any, job_id: Any) -> dict[str, Any]:
    import qiskit
    import qiskit_ibm_runtime

    runtime, framework = qiskit_ibm_runtime.__version__, qiskit.__version__
    return {
        "client": f"qiskit-ibm-runtime {runtime} SamplerV2, qiskit {framework}",
        **{rule.flag: rule.required for rule in _RUN_RULES if rule.flag},
        "job_ids": [] if job_id is None else [job_id],
        "options": options,
    }


def catalog_ref(profile: Profile) -> str | None:
    when = profile.device.calibrated_at
    if when is not None:
        for ref in (ref_on_day(profile.id, when), exact_ref(profile.id, when)):
            try:
                if catalog.resolve(ref).fingerprint == profile.fingerprint:
                    return ref
            except NoiseVaultError:
                continue
    return None


def run(
    *,
    shots: int,
    output: Path,
    pending: Path,
    calibration: Calibration,
    open_backend: Callable[[bool], Any],
    open_job: Callable[[str], Any],
    confirm: Callable[[str], bool] | None,
    collect_only: bool = False,
    job_id: str | None = None,
) -> MeasuredCounts | None:
    if collect_only or pending.exists():
        record, submitted, job = _resume(pending, job_id, open_job)
        print(f"collecting job {submitted.job_id}, submitted earlier for {output}")
    else:
        backend, batch = prepare(calibration, open_backend, shots)
        print(summary(batch, output))
        print()
        device = batch.profile.device.name
        if confirm is not None and not confirm(f"Submit the job to {device}? [y/N] "):
            print("nothing submitted")
            return None
        plan = _line(
            {
                "profile": batch.profile.to_dict(),
                "planned": [circuit.model_dump(mode="json") for circuit in batch.planned],
                "options": batch.options,
            }
        )
        try:
            record = _create({pending: plan})[pending]
        except FileExistsError:
            raise NoiseVaultError(
                f"{pending} exists, so the script did not submit a job",
                hint=_new_name_hint(pending),
            ) from None
        except OSError as exc:
            raise NoiseVaultError(
                f"could not save {pending} ({_reason(exc)})",
                hint="run the same command again. The script did not submit a job",
            ) from None
        try:
            job = submit(backend, batch)
        except BaseException:
            record.drop()
            raise
        submitted = Submitted(job.job_id(), batch.profile, batch.planned, batch.options)
        job_line = _line({"job_id": submitted.job_id})
        kept = _beside(output, f".{submitted.job_id}.job.json")
        changed = f"{pending} changed while the script submitted job {submitted.job_id}"
        try:
            record.append(job_line)
        except BaseException as exc:
            failed = f"could not save job {submitted.job_id} to {pending} ({_reason(exc)})"
            with contextlib.suppress(OSError):
                record.cut(len(plan))
            ours = record.holds()
            with contextlib.suppress(OSError):
                os.close(record.fd)
            if ours and record.data == plan:
                raise NoiseVaultError(
                    failed,
                    hint=f"run the same command with --job-id {submitted.job_id} to collect the"
                    " job",
                ) from None
            _save_elsewhere(failed if ours else changed, kept, submitted.job_id, plan + job_line)
        if not record.holds():
            os.close(record.fd)
            _save_elsewhere(changed, kept, submitted.job_id, plan + job_line)
        print(f"submitted job {submitted.job_id} to {device}")
    print("waiting for it to run")
    print("Ctrl-C stops waiting. Run the same command again to collect the job")
    again = f"run the same command again to collect them. {_new_job(pending)}"
    uncollected = f"could not collect the counts of job {submitted.job_id}"
    try:
        try:
            ran = collect(job, submitted)
        except Exception as exc:
            raise NoiseVaultError(f"{uncollected} ({_reason(exc)})", hint=again) from None
        binding = bind(submitted, ran, calibration)
        bound = binding.profile
        measured = counts_file(bound, submitted, ran)
        _, timing, profile_file = _files(output)
        files = {output: (_readable_json(measured.to_dict()) + "\n").encode("utf-8")}
        if ran.timing:
            files[timing] = (json.dumps(ran.timing, indent=1) + "\n").encode("utf-8")
        ref = catalog_ref(bound)
        if ref is None:
            ref = str(profile_file)
            files[profile_file] = file_bytes(bound, profile_file)
        try:
            for owned in _create(files).values():
                os.close(owned.fd)
        except FileExistsError as exc:
            raise NoiseVaultError(
                f"could not save the counts of job {submitted.job_id}, because {exc.filename}"
                " exists",
                hint=_new_name_hint(pending),
            ) from None
        except OSError as exc:
            raise NoiseVaultError(f"{uncollected} ({_reason(exc)})", hint=again) from None
    except BaseException as exc:
        with contextlib.suppress(OSError):
            os.close(record.fd)
        if isinstance(exc, KeyboardInterrupt):
            raise NoiseVaultError(
                f"stopped before the script saved the counts of job {submitted.job_id}", hint=again
            ) from None
        raise
    try:
        record.drop()
    except OSError as exc:
        print(
            f"warning: could not delete {pending} ({_reason(exc)}). The script saved the"
            " counts, so you can delete the job file",
            file=sys.stderr,
        )
    if binding.warning:
        print(f"warning: {binding.warning}", file=sys.stderr)
        if binding.hint:
            print(f"hint: {binding.hint}", file=sys.stderr)
    elif bound.fingerprint != submitted.profile.fingerprint:
        print(
            f"IBM recalibrated {bound.device.name} before the job ran. The counts bind to the new"
            " calibration, which keeps every planned gate duration"
        )
    lines = [
        ("run", f"{ran.run_at:%Y-%m-%d %H:%MZ}, job {ran.job_id}"),
        ("counts file", str(output)),
        (
            "calibration",
            f"{ref_on_day(bound.id, bound.device.calibrated_at)} {bound.short_fingerprint}",
        ),
    ]
    if ran.timing:
        lines.append(("timing", f"{timing}, how IBM scheduled each circuit"))
    lines.append(("next", catalog.shell_command(["nv", "compare"], [ref, str(output)])))
    print()
    print("\n".join(f"{label:<{LABEL}}{value}" for label, value in lines))
    return measured


def _resume(
    pending: Path, job_id: str | None, open_job: Callable[[str], Any]
) -> tuple[Owned, Submitted, Any]:
    try:
        fd = os.open(pending, os.O_RDONLY | BINARY)
    except FileNotFoundError:
        raise NoiseVaultError(
            f"no job file {pending}",
            hint="another run collected the job or deleted the job file. The script did not submit"
            " a job",
        ) from None
    except OSError as exc:
        raise NoiseVaultError(
            f"could not read {pending} ({_reason(exc)})",
            hint="make the job file readable. Then run the same command again",
        ) from None
    try:
        data, submitted = _submitted(pending, fd, job_id)
        try:
            return Owned(pending, fd, data), submitted, open_job(submitted.job_id)
        except Exception as exc:
            raise NoiseVaultError(
                f"could not open job {submitted.job_id} ({_reason(exc)})",
                hint=f"run the same command again to collect job {submitted.job_id}."
                f" {_new_job(pending)}",
            ) from None
    except BaseException:
        os.close(fd)
        raise


def _save_elsewhere(problem: str, kept: Path, job_id: str, data: bytes) -> NoReturn:
    try:
        os.close(_create({kept: data})[kept].fd)
    except OSError as exc:
        raise NoiseVaultError(
            f"{problem}, and the script could not save job {job_id} to {kept} ({_reason(exc)})",
            hint=f"find job {job_id} in your IBM Quantum account. The script did not save the"
            " planned circuits of the job",
        ) from None
    raise NoiseVaultError(
        f"{problem}. The script saved job {job_id} to {kept}",
        hint=f"run the same command with --collect={shlex.quote(str(kept))} to collect the job",
    )


def _submitted(pending: Path, fd: int, job_id: str | None) -> tuple[bytes, Submitted]:
    lines: list[bytes] = []
    try:
        with open(fd, "rb", closefd=False) as handle:
            lines = handle.readlines()
        data: dict[str, Any] = {}
        for line in lines:
            data.update(parse_json(line))
        profile = Profile.model_validate(data["profile"])
        planned = tuple(PlannedCircuit.model_validate(c) for c in data["planned"])
        options = data["options"]
        recorded = data.get("job_id")
        counts = tuple({"0" * len(c.qubits): 1} for c in planned)
        ran = Ran(recorded, "hardware", datetime.now(UTC), counts, {})
        _bind(profile, counts_file(profile, Submitted(recorded, profile, planned, options), ran))
        if recorded == "":
            raise ValueError("the job id is empty")
    except Exception as exc:
        named = _job_id_in(lines)
        raise NoiseVaultError(
            f"{pending} is damaged ({_reason(exc)})",
            hint="the script cannot collect a job from the damaged job file. Look for the job in"
            f" your IBM Quantum account. {_new_job(pending)}"
            if named is None
            else f"the script cannot collect job {named} from the damaged job file. Find job"
            f" {named} in your IBM Quantum account. {_new_job(pending)}",
        ) from None
    if recorded is None and job_id is None:
        raise NoiseVaultError(
            f"{pending} has no job id",
            hint="give --job-id the job id that the script printed or that your IBM Quantum account"
            f" shows. {_new_job(pending)}",
        )
    if recorded is not None and job_id is not None and recorded != job_id:
        raise NoiseVaultError(
            f"{pending} records job {recorded}, not job {job_id}",
            hint=f"run the same command without --job-id to collect job {recorded}",
        )
    return b"".join(lines), Submitted(recorded or job_id, profile, planned, options)


def _job_id_in(lines: list[bytes]) -> str | None:
    found = None
    for line in lines:
        with contextlib.suppress(ValueError):
            entry = parse_json(line)
            if isinstance(entry, dict) and isinstance(entry.get("job_id"), str) and entry["job_id"]:
                found = entry["job_id"]
    return found


def summary(batch: Batch, output: Path) -> str:
    profile = batch.profile
    chain = tuple(dict.fromkeys(q for c in batch.planned for q in c.qubits))
    rows = [("circuit", "qubits", "gates", "duration")] + [
        (
            c.name,
            "-".join(map(str, c.qubits)),
            str(sum(op.name != "delay" for op in c.ops)),
            f"{ns:.0f} ns",
        )
        for c, ns in zip(batch.planned, batch.durations_ns, strict=True)
    ]
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    table = [
        f"{a:<{widths[0]}}  {b:<{widths[1]}}  {c:>{widths[2]}}  {d:>{widths[3]}}"
        for a, b, c, d in rows
    ]
    labels = [
        ("shots", f"{batch.shots} per circuit, {len(batch.planned)} circuits in one job"),
        ("usage", f"about {batch.usage_s:.1f} s of QPU time (IBM's estimate)"),
        ("counts file", str(output)),
    ]
    return "\n".join(
        [
            f"{ref_on_day(profile.id, profile.device.calibrated_at)}"
            f" {profile.short_fingerprint} on {qubit_loci(chain)}",
            "",
            *table,
            "",
            *(f"{label:<{LABEL}}{value}" for label, value in labels),
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        import qiskit_ibm_runtime  # noqa: F401
    except ImportError:
        return _fail("run_on_ibm.py needs qiskit-ibm-runtime", install_hint("ibm"))
    pending = args.collect or _beside(args.output, ".job.json")
    collect_only = args.collect is not None or args.job_id is not None or pending.exists()
    try:
        if collect_only and not pending.exists():
            raise NoiseVaultError(
                f"no job file {pending}",
                hint="give --collect the .job.json file that the script saved beside the counts"
                " file",
            )
        _writable(args.output, pending)
        service = _service()
        measured = run(
            shots=args.shots,
            output=args.output,
            pending=pending,
            calibration=lambda at: nv.pull(args.device, source="ibm-account", at=at),
            open_backend=lambda fractional: service.backend(
                args.device, use_fractional_gates=fractional
            ),
            open_job=service.job,
            confirm=None if args.yes else _ask,
            collect_only=collect_only,
            job_id=args.job_id,
        )
    except NoiseVaultError as exc:
        return _fail(exc.message, exc.hint)
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
        return 130
    return 0 if measured is not None else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_on_ibm.py",
        description="Run the nv compare circuits on an IBM device and save the counts.",
    )
    parser.add_argument("device", metavar="DEVICE", help="the IBM device, such as ibm_kingston")
    parser.add_argument(
        "--shots",
        metavar="N",
        type=_positive,
        default=DEFAULT_SHOTS,
        help=f"shots per circuit (default {DEFAULT_SHOTS})",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        type=Path,
        required=True,
        help="the counts file to write, which must not exist yet",
    )
    parser.add_argument(
        "--collect",
        metavar="JOB_FILE",
        type=Path,
        help="collect the job that JOB_FILE records instead of submitting a new job",
    )
    parser.add_argument(
        "--job-id",
        metavar="JOB_ID",
        help="collect job JOB_ID when the job file has no job id",
    )
    parser.add_argument("--yes", action="store_true", help="submit without asking")
    return parser


def _positive(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive whole number")
    return value


def _writable(output: Path, pending: Path) -> None:
    for path in _files(output):
        if path.exists():
            raise NoiseVaultError(f"{path} exists", hint=_new_name_hint(pending))
    folder = output.parent
    if not folder.is_dir():
        raise NoiseVaultError(f"no folder {folder}", hint="create it, or give -o another path")
    if not os.access(folder, os.W_OK):
        raise NoiseVaultError(f"cannot write to {folder}", hint="give -o a folder you can write to")


def _service() -> Any:
    try:
        return ibm_account._service()
    except NoiseVaultError as exc:
        raise NoiseVaultError(exc.message, hint=ACCOUNT_SETUP) from None


def _ask(prompt: str) -> bool:
    if not sys.stdin.isatty():
        raise NoiseVaultError(
            "cannot ask before submitting, because standard input is not a terminal",
            hint="give --yes to submit without asking",
        )
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def _fail(message: str, hint: str | None = None) -> int:
    print(f"error: {message}", file=sys.stderr)
    if hint:
        print(f"hint: {hint}", file=sys.stderr)
    return 1


def _reason(exc: BaseException) -> str:
    return exc.message if isinstance(exc, NoiseVaultError) else str(exc) or type(exc).__name__


def _warning_line(message: Warning | str, *_: Any, **__: Any) -> None:
    print(f"warning: {message}", file=sys.stderr)


def _seconds(target: Any, name: str, qubits: tuple[int, ...]) -> float:
    props = target[name].get(qubits) if name in target.operation_names else None
    return 0.0 if props is None or props.duration is None else props.duration


def _whole(value: float) -> int | None:
    n = round(value)
    return n if abs(value - n) <= 1e-6 else None


def _dt(length: float, dt_ns: float) -> str:
    return f"{length:g} dt ({length * dt_ns:g} ns)"


def _new_name_hint(pending: Path) -> str:
    if pending.exists():
        return (
            f"run the same command with --collect={shlex.quote(str(pending))} and a new -o file"
            " name. The script never replaces a file"
        )
    return "give -o a new file name. The script never replaces a file"


def _new_job(pending: Path) -> str:
    return (
        f"To submit a new job, delete {pending}. Then run the command without --collect and"
        " --job-id"
    )


def _files(output: Path) -> tuple[Path, Path, Path]:
    return output, _beside(output, ".timing.json"), _beside(output, ".profile.json")


def _create(files: dict[Path, bytes]) -> dict[Path, Owned]:
    """``os.link`` would make each full file appear at once, but exFAT does not support hard
    links."""
    created: dict[Path, Owned] = {}
    try:
        for path in files:
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | BINARY, 0o666)
            created[path] = Owned(path, fd)
        for path, data in files.items():
            created[path].append(data)
        for owned in created.values():
            owned.check()
    except BaseException:
        for owned in created.values():
            with contextlib.suppress(OSError):
                owned.drop()
        raise
    return created


def _line(data: dict[str, Any]) -> bytes:
    return (json.dumps(data) + "\n").encode("utf-8")


def _beside(output: Path, suffix: str) -> Path:
    stem = output.name.removesuffix(".gz").removesuffix(".json").removesuffix(".counts")
    return output.with_name(stem + suffix)


if __name__ == "__main__":
    warnings.showwarning = _warning_line
    sys.exit(main())
