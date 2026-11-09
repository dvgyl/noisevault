# /// script
# requires-python = ">=3.11"
# dependencies = ["noisevault[braket] @ git+https://github.com/dvgyl/noisevault"]
# ///
r"""Run the nv compare circuits on an Amazon Braket device and save the counts for nv compare.

    uv run \
      https://raw.githubusercontent.com/dvgyl/noisevault/main/scripts/run_on_braket.py \
      iqm_garnet -o garnet-1020.counts.json

uv installs the dependencies listed at the top of this file. In a clone with the ``braket``
extra, run ``python scripts/run_on_braket.py`` with the same arguments.

The script reads the device properties through your AWS account and plans the circuits with
``noisevault.counts.plan``. Each circuit runs in a verbatim box, so the device runs the native
gates as planned, on the planned physical qubits. The script shows the cost in US dollars and
then stops. To submit, run the same command with ``--confirm-cost`` and the dollar amount that
you accept. When the device supports a program set, all circuits go in one task. Otherwise each
circuit is one task. The script saves the task ids before it waits. If the wait stops, run the
same command again. The script then collects the saved tasks and submits nothing.

Braket has no gate-level delay op. Thus the script plans from the calibration without its gate
durations, and the plan holds no delay. The counts bind to that calibration, which the script
saves beside the counts file. Idle error during the gates of other qubits goes into the gate
factor that ``nv compare`` fits.

``--local PROPERTIES`` runs the circuits on the free Braket ``LocalEmulator`` of a saved
properties file, with no AWS account. The emulator refuses a gate that is not native and a pair
that is not connected, as the device would.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shlex
import sys
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import noisevault as nv  # noqa: E402
from noisevault import gates  # noqa: E402
from noisevault._jobfile import BINARY, Owned, beside, create, line  # noqa: E402
from noisevault.compare import _bind  # noqa: E402
from noisevault.counts import (  # noqa: E402
    _RUN_RULES,
    COUNTS_FORMAT,
    MeasuredCounts,
    PlannedCircuit,
    plan,
)
from noisevault.errors import (  # noqa: E402
    NoiseVaultError,
    install_hint,
    parse_json,
    qubit_loci,
)
from noisevault.profile import Profile, file_bytes, ref_on_day  # noqa: E402

DEFAULT_SHOTS = 4000
LABEL = 16
PROGRAM_SET = "braket.ir.openqasm.program_set"
OPENQASM = "braket.ir.openqasm.program"
PRICING = "https://aws.amazon.com/braket/pricing/"
CONSOLE = "the Amazon Braket console"


@dataclass(frozen=True)
class Price:
    task: Decimal
    shot: Decimal


# Amazon Braket on-demand prices in US dollars, from PRICING, read in October 2026.
PRICES = {
    "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet": Price(Decimal("0.30"), Decimal("0.00145")),
    "arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q": Price(
        Decimal("0.30"), Decimal("0.000425")
    ),
    "arn:aws:braket:eu-north-1::device/qpu/aqt/Ibex-Q1": Price(Decimal("0.30"), Decimal("0.0235")),
    "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1": Price(Decimal("0.30"), Decimal("0.08")),
}

# A NoiseVault gate and the Braket gates that can run it as one native gate. Each entry gives
# the Braket gate's parameters from the NoiseVault parameters. The script checks the matrices.
NATIVE: Mapping[str, tuple[tuple[str, Callable[..., tuple[float, ...]]], ...]] = {
    "r": (
        ("prx", lambda theta, phi: (theta, phi)),
        ("gpi2", lambda theta, phi: (phi,)),
        ("gpi", lambda theta, phi: (phi,)),
    ),
    "rx": (("rx", lambda theta: (theta,)), ("prx", lambda theta: (theta, 0.0))),
    "ry": (("ry", lambda theta: (theta,)), ("prx", lambda theta: (theta, math.pi / 2))),
    "rz": (("rz", lambda theta: (theta,)),),
    "x": (("x", lambda: ()),),
    "y": (("y", lambda: ()),),
    "z": (("z", lambda: ()),),
    "h": (("h", lambda: ()),),
    "s": (("s", lambda: ()),),
    "sdg": (("si", lambda: ()),),
    "sx": (("v", lambda: ()),),
    "sxdg": (("vi", lambda: ()),),
    "cz": (("cz", lambda: ()),),
    "cx": (("cnot", lambda: ()),),
    "iswap": (("iswap", lambda: ()),),
    "rxx": (("xx", lambda theta: (theta,)),),
    "ryy": (("yy", lambda theta: (theta,)),),
    "rzz": (("zz", lambda theta: (theta,)),),
    "ms": (("ms", lambda phi0, phi1: (phi0, phi1, math.pi / 2)),),
}


@dataclass(frozen=True)
class Target:
    """The device that runs the circuits: an AwsDevice, or a LocalEmulator for --local."""

    arn: str
    name: str
    device: Any
    properties: dict[str, Any]
    local: Path | None = None


@dataclass(frozen=True)
class Batch:
    profile: Profile
    planned: tuple[PlannedCircuit, ...]
    circuits: tuple[Any, ...]
    shots: int
    program_set: bool
    cost: Decimal
    price: Price


@dataclass(frozen=True)
class Submitted:
    arn: str
    local: bool
    profile: Profile
    planned: tuple[PlannedCircuit, ...]
    shots: int
    program_set: bool
    tasks: tuple[tuple[str, tuple[str, ...]], ...]
    ended: bool = False
    unconfirmed: str | None = None


@dataclass(frozen=True)
class Ran:
    source: str
    run_at: datetime
    counts: dict[str, dict[str, int]]
    task_ids: tuple[str, ...]


def device_name(arn: str) -> str:
    """The profile name of a device, such as iqm_garnet for .../qpu/iqm/Garnet."""
    *_, provider, device = arn.split("/")
    return f"{provider}_{device}".lower()


def arn_of(device: str) -> str:
    if device.startswith("arn:aws:braket:"):
        return device
    for arn in PRICES:
        if device_name(arn) == device:
            return arn
    known = ", ".join(device_name(arn) for arn in PRICES)
    raise NoiseVaultError(
        f"{device!r} is not a device that the script knows",
        hint=f"give a device ARN or one of {known}",
    )


def calibration(properties: Mapping[str, Any], name: str) -> Profile:
    """The device's calibration without gate durations.

    Braket runs no gate-level delay, so a plan with delays cannot run as planned. With no gate
    durations, ``plan`` writes no delay.
    """
    profile = nv.from_braket(dict(properties), device=name)
    data = profile.to_dict()
    removed = False
    for record in (*data.get("gates", {}).values(), *data.get("calibrations", [])):
        if record.get("duration_ns") is not None:
            del record["duration_ns"]
            removed = True
    if not removed:
        return profile
    provenance = data["provenance"]
    provenance["notes"] = [
        *provenance.get("notes", []),
        "run_on_braket.py removed the gate durations, because Braket runs no gate-level delay."
        " The planned circuits thus hold no delay, and the gate factor of nv compare includes"
        " idle error",
    ]
    return Profile.model_validate(data)


def native_gate(op: Any, natives: Sequence[str]) -> Any:
    """The one Braket gate with the matrix of ``op``, from the device's native gates."""
    from braket.circuits import gates as braket_gates

    classes = {
        "prx": braket_gates.PRx,
        "gpi": braket_gates.GPi,
        "gpi2": braket_gates.GPi2,
        "rx": braket_gates.Rx,
        "ry": braket_gates.Ry,
        "rz": braket_gates.Rz,
        "x": braket_gates.X,
        "y": braket_gates.Y,
        "z": braket_gates.Z,
        "h": braket_gates.H,
        "s": braket_gates.S,
        "si": braket_gates.Si,
        "v": braket_gates.V,
        "vi": braket_gates.Vi,
        "cz": braket_gates.CZ,
        "cnot": braket_gates.CNot,
        "iswap": braket_gates.ISwap,
        "xx": braket_gates.XX,
        "yy": braket_gates.YY,
        "zz": braket_gates.ZZ,
        "ms": braket_gates.MS,
    }
    wanted = gates.unitary(op.name, op.params)
    for name, args in NATIVE.get(op.name, ()):
        if name in natives:
            gate = classes[name](*args(*op.params))
            if _same_up_to_phase(gate.to_matrix(), wanted):
                return gate
    return None


def braket_circuit(circuit: PlannedCircuit, natives: Sequence[str], device: str) -> Any:
    """The circuit in one verbatim box on its physical qubits, then a measurement of each qubit.

    The readout circuit has an empty verbatim box, which keeps its qubits physical too.
    """
    from braket.circuits import Circuit, Instruction
    from braket.circuits.compiler_directives import EndVerbatimBox, StartVerbatimBox

    built = Circuit().add_instruction(Instruction(StartVerbatimBox()))
    for op in circuit.ops:
        qubits = tuple(circuit.qubits[q] for q in op.qubits)
        if op.name == "delay":
            raise NoiseVaultError(
                f"circuit {circuit.name}: the plan holds a delay on {qubit_loci(qubits)}, and"
                " Braket runs no gate-level delay"
            )
        gate = native_gate(op, natives)
        if gate is None:
            params = f"({', '.join(f'{p:g}' for p in op.params)})" if op.params else ""
            raise NoiseVaultError(
                f"circuit {circuit.name}: {op.name}{params} on {qubit_loci(qubits)} is not one"
                f" native gate of {device}, whose native gates are {', '.join(natives)}",
                hint="the counts must list the gates that the device ran, so the script runs"
                " only native gates",
            )
        built.add_instruction(Instruction(gate, qubits))
    built.add_instruction(Instruction(EndVerbatimBox()))
    built.measure(list(circuit.qubits))
    source = built.to_ir("OPENQASM").source
    if "qubit[" in source:
        raise NoiseVaultError(
            f"circuit {circuit.name}: the Braket SDK did not write physical qubits",
            hint="install another amazon-braket-sdk version",
        )
    return built


def estimate(price: Price, circuits: int, shots: int, program_set: bool) -> Decimal:
    tasks = 1 if program_set else circuits
    return tasks * price.task + circuits * shots * price.shot


def prepare(target: Target, shots: int, price: Price) -> Batch:
    properties = target.properties
    profile = calibration(properties, target.name)
    planned = plan(profile)
    action = properties.get("action") or {}
    pragmas = (action.get(OPENQASM) or {}).get("supportedPragmas") or []
    if "verbatim" not in pragmas:
        raise NoiseVaultError(
            f"{target.name} does not list verbatim compilation for OpenQASM programs",
            hint="run on a device that runs verbatim boxes, so the counts list the gates that ran",
        )
    natives = [g.lower() for g in (properties.get("paradigm") or {}).get("nativeGateSet") or []]
    circuits = tuple(braket_circuit(c, natives, target.name) for c in planned)
    low, high = (properties.get("service") or {}).get("shotsRange") or (1, math.inf)
    if not low <= shots <= high:
        raise NoiseVaultError(
            f"{target.name} runs {low} to {high} shots per task, not {shots}",
            hint="give --shots a number in that range",
        )
    limits = action.get(PROGRAM_SET) or {}
    program_set = bool(limits) and (
        len(circuits) <= limits.get("maximumExecutables", 0)
        and len(circuits) * shots <= limits.get("maximumTotalShots", 0)
    )
    return Batch(
        profile=profile,
        planned=planned,
        circuits=circuits,
        shots=shots,
        program_set=program_set,
        cost=estimate(price, len(circuits), shots, program_set),
        price=price,
    )


def submit(target: Target, batch: Batch, record: Callable[[Any, tuple[str, ...]], None]) -> None:
    """Create the tasks. ``record`` saves each task before the next task is created."""
    if batch.program_set:
        from braket.program_sets import ProgramSet

        programs = ProgramSet(list(batch.circuits), shots_per_executable=batch.shots)
        task = target.device.run(programs, shots=programs.total_shots)
        record(task, tuple(c.name for c in batch.planned))
        return
    for circuit, built in zip(batch.planned, batch.circuits, strict=True):
        task = target.device.run(built, shots=batch.shots)
        record(task, (circuit.name,))


def collect(tasks: Sequence[Any], submitted: Submitted) -> Ran:
    from braket.aws import AwsQuantumTask

    counts: dict[str, dict[str, int]] = {}
    ended = []
    planned = {c.name: c for c in submitted.planned}
    for task, (task_id, names) in zip(tasks, submitted.tasks, strict=True):
        result = task.result()
        if result is None:
            raise NoiseVaultError(f"task {task_id} gave no result ({task.state()})")
        if submitted.program_set:
            entries = [(e.entries[0].counts, e.entries[0].measured_qubits) for e in result.entries]
        else:
            entries = [(result.measurement_counts, result.measured_qubits)]
        for name, (found, measured) in zip(names, entries, strict=True):
            counts[name] = _clbit0_left(planned[name], found, measured)
        metadata = result.task_metadata
        ended.append(_utc(metadata.endedAt or metadata.createdAt))
    hardware = all(isinstance(task, AwsQuantumTask) for task in tasks)
    return Ran(
        source="hardware" if hardware and not submitted.local else "simulated",
        run_at=min(ended),
        counts=counts,
        task_ids=tuple(task_id for task_id, _ in submitted.tasks),
    )


def counts_file(profile: Profile, submitted: Submitted, ran: Ran, client: str) -> MeasuredCounts:
    circuits = [c for c in submitted.planned if c.name in ran.counts]
    return MeasuredCounts.model_validate(
        {
            "nv_counts": COUNTS_FORMAT,
            "source": ran.source,
            "profile": {"id": profile.id, "fingerprint": profile.calibration_fingerprint},
            "backend": profile.device.name,
            "run_at": ran.run_at,
            "bit_order": "clbit0_left",
            "execution": {
                "client": client,
                **{rule.flag: rule.required for rule in _RUN_RULES if rule.flag},
                "job_ids": list(ran.task_ids),
                "options": {
                    "device": submitted.arn,
                    "emulator": "LocalEmulator" if submitted.local else None,
                    "shots": submitted.shots,
                    "verbatim": True,
                    "program_set": submitted.program_set,
                },
            },
            "circuits": [
                {
                    "name": c.name,
                    "qubits": c.qubits,
                    "ops": c.ops,
                    "shots": sum(ran.counts[c.name].values()),
                    "counts": ran.counts[c.name],
                }
                for c in circuits
            ],
        }
    )


def recalibrated(target: Target, submitted: Submitted, run_at: datetime) -> str | None:
    """A warning when the device has a new calibration from before the run, or None.

    Braket gives no calibration history, so the counts always bind to the planned calibration.
    """
    if submitted.local:
        return None
    try:
        latest = calibration(target.properties, target.name)
    except NoiseVaultError:
        return None
    planned = submitted.profile
    when = latest.device.calibrated_at
    if latest.calibration_fingerprint == planned.calibration_fingerprint or when is None:
        return None
    if when > run_at:
        return None
    return (
        f"{target.name} has a calibration from {when:%Y-%m-%d %H:%MZ}, before the run, that is"
        f" not the planned calibration {planned.short_fingerprint}. Braket gives no calibration"
        " history, so the counts bind to the planned calibration"
    )


def run(
    *,
    shots: int,
    output: Path,
    pending: Path,
    open_target: Callable[[], Target],
    open_task: Callable[[str], Any],
    price: Price | None,
    confirm_cost: Decimal | None,
    collect_only: bool = False,
    partial: bool = False,
) -> MeasuredCounts | None:
    """Plan, show the cost, submit and collect. Returns None when the script submits nothing."""
    tasks: list[Any]
    if collect_only or pending.exists():
        record, submitted = _resume(pending, partial)
        try:
            target = open_target()
        except BaseException:
            os.close(record.fd)
            raise
        if target.arn != submitted.arn:
            os.close(record.fd)
            raise NoiseVaultError(
                f"{pending} records tasks on {submitted.arn}, not on {target.arn}",
                hint=f"run the command with {device_name(submitted.arn)} to collect them",
            )
        try:
            tasks = [open_task(task_id) for task_id, _ in submitted.tasks]
        except Exception as exc:
            os.close(record.fd)
            raise NoiseVaultError(
                f"could not open the tasks of {pending} ({_reason(exc)})",
                hint=f"run the same command again. {_new_run(pending)}",
            ) from None
        print(f"collecting {_tasks(submitted.tasks)} that {pending} records")
    else:
        target = open_target()
        if price is None:
            price = _price(target.arn)
        batch = prepare(target, shots, price)
        print(summary(batch, target, output))
        print()
        if target.local is None:
            if confirm_cost is None:
                print(
                    f"nothing submitted. To submit, run the same command with --confirm-cost"
                    f" {batch.cost:.2f}"
                )
                return None
            if confirm_cost < batch.cost:
                raise NoiseVaultError(
                    f"the run costs about ${batch.cost:.2f}, more than --confirm-cost"
                    f" {confirm_cost}. The script submitted nothing",
                    hint=f"give --confirm-cost {batch.cost:.2f} to accept the cost",
                )
        plan_line = line(
            {
                "device": target.arn,
                "local": None if target.local is None else str(target.local),
                "profile": batch.profile.to_dict(),
                "planned": [c.model_dump(mode="json") for c in batch.planned],
                "shots": batch.shots,
                "program_set": batch.program_set,
                "programs": [c.to_ir("OPENQASM").source for c in batch.circuits],
            }
        )
        try:
            record = create({pending: plan_line})[pending]
        except FileExistsError:
            raise NoiseVaultError(
                f"{pending} exists, so the script submitted nothing",
                hint="run the same command again to collect the tasks that it records."
                f" {_new_run(pending)}",
            ) from None
        except OSError as exc:
            raise NoiseVaultError(
                f"could not save {pending} ({_reason(exc)})",
                hint="run the same command again. The script submitted nothing",
            ) from None
        created: list[tuple[str, tuple[str, ...]]] = []
        tasks = []

        def keep(task: Any, names: tuple[str, ...]) -> None:
            created.append((task.id, names))
            tasks.append(task)
            try:
                record.append(line({"task": task.id, "circuits": list(names)}))
            except BaseException as exc:
                _keep_elsewhere(output, record, created, exc)

        try:
            submit(target, batch, keep)
        except NoiseVaultError:
            raise
        except BaseException as exc:
            refused = _refused(exc)
            if target.local is not None or (refused and not created):
                with contextlib.suppress(OSError):
                    record.drop()
                if isinstance(exc, KeyboardInterrupt):
                    raise
                raise NoiseVaultError(
                    f"Braket refused the task ({_reason(exc)})",
                    hint="the script submitted nothing. Run the same command again",
                ) from None
            _end(record, created, None if refused else _reason(exc))
            os.close(record.fd)
            maybe = "" if refused else ". Braket may also have accepted the next task"
            if not created:
                raise NoiseVaultError(
                    f"Braket did not confirm the task ({_reason(exc)}). Braket may have accepted"
                    " it",
                    hint=f"look for a new task in {CONSOLE}. If there is none, delete {pending}"
                    " and run the command again. The script submits nothing while the job file"
                    " exists",
                ) from None
            raise NoiseVaultError(
                f"Braket accepted {_tasks(created)}, and then the submission stopped"
                f" ({_reason(exc)}){maybe}",
                hint="run the same command again to collect the counts of the accepted tasks."
                f" {_new_run(pending)}",
            ) from None
        _end(record, created)
        submitted = Submitted(
            target.arn,
            target.local is not None,
            batch.profile,
            batch.planned,
            batch.shots,
            batch.program_set,
            tuple(created),
        )
        where = target.name if target.local is None else "the local emulator"
        print(f"submitted {_tasks(created)} to {where}")
    if target.local is None:
        print("waiting for the tasks to run")
        print("Ctrl-C stops waiting. Run the same command again to collect the tasks")
    again = f"run the same command again to collect them. {_new_run(pending)}"
    try:
        try:
            ran = collect(tasks, submitted)
        except NoiseVaultError as exc:
            raise NoiseVaultError(exc.message, hint=again) from None
        except Exception as exc:
            raise NoiseVaultError(
                f"could not collect the counts of {_tasks(submitted.tasks)} ({_reason(exc)})",
                hint=again,
            ) from None
        profile = submitted.profile
        measured = counts_file(profile, submitted, ran, _client(submitted.local))
        _, profile_path = _files(output)
        files = {
            output: measured.file_bytes(output),
            profile_path: file_bytes(profile, profile_path),
        }
        try:
            for owned in create(files).values():
                os.close(owned.fd)
        except FileExistsError as exc:
            raise NoiseVaultError(
                f"could not save the counts, because {exc.filename} exists",
                hint=f"run the same command with --collect={shlex.quote(str(pending))} and a new"
                " -o file name. The script never replaces a file",
            ) from None
        except OSError as exc:
            raise NoiseVaultError(
                f"could not save the counts ({_reason(exc)})", hint=again
            ) from None
    except BaseException as exc:
        with contextlib.suppress(OSError):
            if submitted.local:
                record.drop()
            else:
                os.close(record.fd)
        if isinstance(exc, KeyboardInterrupt):
            raise NoiseVaultError(
                "stopped before the script saved the counts", hint=again
            ) from None
        raise
    try:
        record.drop()
    except OSError as exc:
        print(
            f"warning: could not delete {pending} ({_reason(exc)}). The script saved the counts,"
            " so you can delete the job file",
            file=sys.stderr,
        )
    warning = recalibrated(target, submitted, ran.run_at)
    if warning:
        print(f"warning: {warning}", file=sys.stderr)
    if submitted.unconfirmed is not None:
        print(
            f"warning: Braket did not answer the request after {_tasks(submitted.tasks)}"
            f" ({submitted.unconfirmed}), so Braket may have accepted one more task. Look for it in"
            f" {CONSOLE}. The counts leave it out",
            file=sys.stderr,
        )
    missing = [c.name for c in submitted.planned if c.name not in ran.counts]
    if missing:
        print(
            f"warning: the counts leave out {', '.join(missing)}, because Braket did not accept"
            " a task for each circuit",
            file=sys.stderr,
        )
    lines = [
        ("run", f"{ran.run_at:%Y-%m-%d %H:%MZ}, {_tasks(submitted.tasks)}"),
        ("counts file", str(output)),
        ("calibration", f"{profile_path} {profile.short_fingerprint}"),
        ("next", shlex.join(["nv", "compare", str(profile_path), str(output)])),
    ]
    print()
    print("\n".join(f"{label:<{LABEL}}{value}" for label, value in lines))
    return measured


def summary(batch: Batch, target: Target, output: Path) -> str:
    profile = batch.profile
    chain = tuple(dict.fromkeys(q for c in batch.planned for q in c.qubits))
    rows = [("circuit", "qubits", "gates")] + [
        (c.name, "-".join(map(str, c.qubits)), str(len(c.ops))) for c in batch.planned
    ]
    widths = [max(len(row[i]) for row in rows) for i in range(3)]
    table = [f"{a:<{widths[0]}}  {b:<{widths[1]}}  {c:>{widths[2]}}" for a, b, c in rows]
    n = len(batch.planned)
    tasks = "1 program set task" if batch.program_set else f"{n} tasks, one per circuit"
    cost = f"about ${batch.cost:.2f} (${batch.price.task} a task, ${batch.price.shot} a shot)"
    if target.local is not None:
        cost = f"none on the local emulator, {cost} on {target.name}"
    when = profile.device.calibrated_at
    calibrated = "no date" if when is None else ref_on_day(profile.id, when)
    labels = [
        ("shots", f"{batch.shots} per circuit, {n} circuits in {tasks}"),
        ("cost", cost),
        ("counts file", str(output)),
    ]
    return "\n".join(
        [
            f"{calibrated} {profile.short_fingerprint} on {qubit_loci(chain)}",
            "",
            *table,
            "",
            *(f"{label:<{LABEL}}{value}" for label, value in labels),
        ]
    )


def _keep_elsewhere(
    output: Path, record: Owned, created: list[tuple[str, tuple[str, ...]]], exc: BaseException
) -> NoReturn:
    first = record.data.split(b"\n", 1)[0] + b"\n"
    data = first + b"".join(
        line({"task": task_id, "circuits": list(names)}) for task_id, names in created
    )
    data += line({"end": len(created)})
    kept = beside(output, f".{created[-1][0].rsplit('/', 1)[-1]}.job.json")
    problem = f"could not save task {created[-1][0]} to {record.path} ({_reason(exc)})"
    with contextlib.suppress(OSError):
        os.close(record.fd)
    try:
        os.close(create({kept: data})[kept].fd)
    except OSError as again:
        ids = ", ".join(task_id for task_id, _ in created)
        raise NoiseVaultError(
            f"{problem}, and the script could not save the tasks to {kept} ({_reason(again)})",
            hint=f"find the tasks {ids} in {CONSOLE}",
        ) from None
    raise NoiseVaultError(
        f"{problem}. The script saved the tasks to {kept} and submitted no more tasks",
        hint=f"run the same command with --collect={shlex.quote(str(kept))} to collect them",
    )


def _end(
    record: Owned, created: Sequence[tuple[str, tuple[str, ...]]], unconfirmed: str | None = None
) -> None:
    """Mark the end of the submission, so a collector knows that no task follows.

    ``unconfirmed`` is why Braket did not answer the last request, which can have created a
    task that the job file does not record.
    """
    size = len(record.data)
    end: dict[str, Any] = {"end": len(created)}
    if unconfirmed is not None:
        end["unconfirmed"] = unconfirmed
    try:
        record.append(line(end))
    except OSError as exc:
        with contextlib.suppress(OSError):
            record.cut(size)
        print(
            f"warning: could not mark the end of the submission in {record.path}"
            f" ({_reason(exc)}). To collect the tasks, add --partial",
            file=sys.stderr,
        )


def _refused(exc: BaseException) -> bool:
    """True when Braket surely created no task: the SDK refused the request before it sent it,
    or the service answered with an error. A timeout or a lost connection is not a refusal."""
    try:
        from botocore.exceptions import ClientError
    except ImportError:
        return isinstance(exc, ValueError | TypeError)
    return isinstance(exc, ClientError | ValueError | TypeError)


def _resume(pending: Path, partial: bool = False) -> tuple[Owned, Submitted]:
    try:
        fd = os.open(pending, os.O_RDONLY | BINARY)
    except FileNotFoundError:
        raise NoiseVaultError(
            f"no job file {pending}",
            hint="another run collected the tasks or deleted the job file. The script submitted"
            " nothing",
        ) from None
    except OSError as exc:
        raise NoiseVaultError(f"could not read {pending} ({_reason(exc)})") from None
    try:
        with open(fd, "rb", closefd=False) as handle:
            lines = handle.readlines()
        try:
            submitted = _submitted(lines, partial)
        except Exception as exc:
            ids = _task_ids_in(lines)
            raise NoiseVaultError(
                f"{pending} is damaged ({_reason(exc)})",
                hint=f"find the tasks {', '.join(ids)} in {CONSOLE}. {_new_run(pending)}"
                if ids
                else f"look for the tasks in {CONSOLE}. {_new_run(pending)}",
            ) from None
        if submitted.local:
            raise NoiseVaultError(
                f"{pending} records a run on the local emulator, which the script cannot collect"
                " later",
                hint=f"delete {pending}. Then run the same command again",
            )
        if not submitted.tasks:
            raise NoiseVaultError(
                f"{pending} records no task",
                hint=f"look for a new task in {CONSOLE}. {_new_run(pending)}",
            )
        expected = 1 if submitted.program_set else len(submitted.planned)
        if len(submitted.tasks) < expected and not submitted.ended and not partial:
            raise NoiseVaultError(
                f"{pending} records {len(submitted.tasks)} of {expected} tasks and no end of the"
                " submission, so another run may still submit tasks",
                hint="wait until that run ends. If no run submits tasks now, run the same command"
                f" with --partial to collect the tasks that {pending} records",
            )
    except BaseException:
        os.close(fd)
        raise
    return Owned(pending, fd, b"".join(lines)), submitted


def _submitted(lines: list[bytes], partial: bool = False) -> Submitted:
    """The submission that a job file records. With ``partial``, a last line that a failed write
    cut off is left out."""
    if partial and len(lines) > 1 and not lines[-1].endswith(b"\n"):
        lines = lines[:-1]
    first = parse_json(lines[0])
    profile = Profile.model_validate(first["profile"])
    planned = tuple(PlannedCircuit.model_validate(c) for c in first["planned"])
    names = {c.name for c in planned}
    tasks: list[tuple[str, tuple[str, ...]]] = []
    ended = False
    unconfirmed = None
    for raw in lines[1:]:
        entry = parse_json(raw)
        if ended:
            raise ValueError("a line follows the end of the submission")
        if "end" in entry and set(entry) <= {"end", "unconfirmed"}:
            if entry["end"] != len(tasks):
                raise ValueError(f"the end line counts {entry['end']!r} tasks, not {len(tasks)}")
            unconfirmed = entry.get("unconfirmed")
            if unconfirmed is not None and not isinstance(unconfirmed, str):
                raise ValueError(f"unconfirmed is {unconfirmed!r}")
            ended = True
            continue
        task_id, circuits = entry["task"], tuple(entry["circuits"])
        if not isinstance(task_id, str) or not task_id or not set(circuits) <= names:
            raise ValueError(f"the task line {raw.decode('utf-8', 'replace').strip()} is not valid")
        tasks.append((task_id, circuits))
    shots = first["shots"]
    if not isinstance(shots, int) or isinstance(shots, bool) or shots < 1:
        raise ValueError(f"shots is {shots!r}")
    probe = Ran(
        "hardware", datetime.now(UTC), {c.name: {"0" * len(c.qubits): 1} for c in planned}, ()
    )
    submitted = Submitted(
        first["device"], first["local"] is not None, profile, planned, shots,
        bool(first["program_set"]), tuple(tasks), ended, unconfirmed,
    )  # fmt: skip
    _bind(profile, counts_file(profile, submitted, probe, "check"))
    return submitted


def _task_ids_in(lines: list[bytes]) -> list[str]:
    found = []
    for raw in lines[1:]:
        with contextlib.suppress(ValueError):
            entry = parse_json(raw)
            if isinstance(entry, dict) and isinstance(entry.get("task"), str):
                found.append(entry["task"])
    return found


def _clbit0_left(
    circuit: PlannedCircuit, found: Mapping[str, int], measured: Sequence[int]
) -> dict[str, int]:
    """Braket keys hold one bit per measured qubit, in the order of ``measured``."""
    if sorted(measured) != sorted(circuit.qubits):
        raise NoiseVaultError(
            f"circuit {circuit.name}: Braket measured {qubit_loci(tuple(measured))}, not the"
            f" planned {qubit_loci(circuit.qubits)}"
        )
    position = {q: i for i, q in enumerate(measured)}
    counts: dict[str, int] = {}
    for key, n in found.items():
        reordered = "".join(key[position[q]] for q in circuit.qubits)
        counts[reordered] = counts.get(reordered, 0) + int(n)
    return counts


def _same_up_to_phase(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and abs(abs(np.vdot(a, b)) - len(a)) < 1e-9


def _utc(stamp: str | datetime) -> datetime:
    when = stamp if isinstance(stamp, datetime) else datetime.fromisoformat(stamp)
    return (when if when.tzinfo else when.replace(tzinfo=UTC)).astimezone(UTC)


def _client(local: bool) -> str:
    from braket._sdk import __version__

    return f"amazon-braket-sdk {__version__}" + (" LocalEmulator" if local else "")


def _price(arn: str) -> Price:
    try:
        return PRICES[arn]
    except KeyError:
        raise NoiseVaultError(
            f"the script has no price for {arn}",
            hint=f"give --task-price and --shot-price in US dollars from {PRICING}",
        ) from None


def _tasks(tasks: Sequence[tuple[str, tuple[str, ...]]]) -> str:
    ids = [task_id for task_id, _ in tasks]
    return f"task {ids[0]}" if len(ids) == 1 else f"{len(ids)} tasks ({', '.join(ids)})"


def _new_run(pending: Path) -> str:
    return f"To submit a new run, delete {pending}. Then run the command without --collect"


def _files(output: Path) -> tuple[Path, Path]:
    return output, beside(output, ".profile.json")


def _reason(exc: BaseException) -> str:
    return exc.message if isinstance(exc, NoiseVaultError) else str(exc) or type(exc).__name__


def open_aws(arn: str) -> Target:
    from braket.aws import AwsDevice

    try:
        device = AwsDevice(arn)
        properties = json.loads(device.properties.json())
    except Exception as exc:
        raise NoiseVaultError(
            f"could not open {arn} through your AWS account ({_reason(exc)})",
            hint="set up AWS credentials for Amazon Braket. The script submitted nothing",
        ) from None
    return Target(arn, device_name(arn), device, properties)


def open_local(arn: str, path: Path) -> Target:
    from braket.emulation.local_emulator import LocalEmulator

    try:
        text = path.read_text(encoding="utf-8")
        device = LocalEmulator.from_json(text)
    except OSError as exc:
        raise NoiseVaultError(f"could not read {path} ({_reason(exc)})") from None
    except Exception as exc:
        raise NoiseVaultError(
            f"{path} is not a Braket device properties file ({_reason(exc)})",
            hint="save the file with AwsDevice(arn).properties.json()",
        ) from None
    return Target(arn, device_name(arn), device, json.loads(text), path)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        import braket.aws  # noqa: F401
    except ImportError:
        return _fail("run_on_braket.py needs amazon-braket-sdk", install_hint("braket"))
    pending = args.collect or beside(args.output, ".job.json")
    collect_only = args.collect is not None or pending.exists()
    if (args.task_price is None) != (args.shot_price is None):
        return _fail("give --task-price and --shot-price together")
    price = None if args.task_price is None else Price(args.task_price, args.shot_price)
    try:
        arn = arn_of(args.device)
        if collect_only and not pending.exists():
            raise NoiseVaultError(
                f"no job file {pending}",
                hint="give --collect the .job.json file that the script saved beside the counts"
                " file",
            )
        for path in _files(args.output):
            if path.exists():
                raise NoiseVaultError(
                    f"{path} exists",
                    hint="give -o a new file name. The script never replaces a file",
                )
        folder = args.output.parent
        if not folder.is_dir() or not os.access(folder, os.W_OK):
            raise NoiseVaultError(
                f"cannot write to {folder}", hint="give -o a folder you can write to"
            )
        measured = run(
            shots=args.shots,
            output=args.output,
            pending=pending,
            open_target=(
                (lambda: open_aws(arn))
                if args.local is None
                else (lambda: open_local(arn, args.local))
            ),
            open_task=_open_task,
            price=price,
            confirm_cost=args.confirm_cost,
            collect_only=collect_only,
            partial=args.partial,
        )
    except NoiseVaultError as exc:
        return _fail(exc.message, exc.hint)
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
        return 130
    del measured
    return 0


def _open_task(task_id: str) -> Any:
    from braket.aws import AwsQuantumTask

    return AwsQuantumTask(task_id)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_on_braket.py",
        description="Run the nv compare circuits on an Amazon Braket device and save the counts.",
    )
    known = ", ".join(device_name(arn) for arn in PRICES)
    parser.add_argument("device", metavar="DEVICE", help=f"a device ARN, or one of {known}")
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        type=Path,
        required=True,
        help="the counts file to write, which must not exist yet",
    )
    parser.add_argument(
        "--shots",
        metavar="N",
        type=_positive,
        default=DEFAULT_SHOTS,
        help=f"shots per circuit (default {DEFAULT_SHOTS})",
    )
    parser.add_argument(
        "--confirm-cost",
        metavar="USD",
        type=_dollars,
        help="submit if the run costs at most USD. Without it, the script shows the cost and stops",
    )
    parser.add_argument(
        "--local",
        metavar="PROPERTIES",
        type=Path,
        help="run on the free Braket LocalEmulator of a saved device properties file",
    )
    parser.add_argument(
        "--collect",
        metavar="JOB_FILE",
        type=Path,
        help="collect the tasks that JOB_FILE records instead of submitting new tasks",
    )
    parser.add_argument(
        "--partial",
        action="store_true",
        help="collect the tasks of a job file whose submission stopped before its end",
    )
    parser.add_argument("--task-price", metavar="USD", type=_dollars, help="the price per task")
    parser.add_argument("--shot-price", metavar="USD", type=_dollars, help="the price per shot")
    return parser


def _positive(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive whole number")
    return value


def _dollars(text: str) -> Decimal:
    try:
        value = Decimal(text.removeprefix("$"))
    except InvalidOperation:
        value = Decimal("NaN")
    if not value.is_finite() or value < 0:
        raise argparse.ArgumentTypeError(f"{text!r} is not an amount in US dollars, such as 23.50")
    return value


def _fail(message: str, hint: str | None = None) -> int:
    print(f"error: {message}", file=sys.stderr)
    if hint:
        print(f"hint: {hint}", file=sys.stderr)
    return 1


def _warning_line(message: Warning | str, *_: Any, **__: Any) -> None:
    print(f"warning: {message}", file=sys.stderr)


if __name__ == "__main__":
    warnings.showwarning = _warning_line
    sys.exit(main())
