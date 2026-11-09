r"""Run the weekly nv compare campaign on IBM devices with one command.

    python scripts/ibm_campaign.py ibm_kingston ibm_fez

Run the script in a clone with the ``ibm`` extra. For each device, the script does these steps:

1. It pulls the current calibration through your IBM Quantum account.
2. It picks two chains: the best chain, which ``nv check`` and ``scripts/run_on_ibm.py`` use,
   and the median chain, the chain with the median error among all chains of the same length.
3. It reads the remaining QPU time of your instance. It submits a job only when IBM's estimate
   for the job, plus a reserve, fits in that time.
4. It submits one job per chain with the logic of ``scripts/run_on_ibm.py``. Then it waits for
   each job and saves the counts and the calibration that the counts bind to.

The files go in ``campaign/<device>/<date>/``. The date is the UTC day of the first run in that
ISO week. A rerun in the same week uses the same folder. It collects a submitted job and does
not submit a second job for a chain. A rerun also collects jobs from earlier weeks.

Each week folder holds these files:

    calibration.json        the calibration that the script planned from
    chains.json             the best and median chains, with their summed error
    best.counts.json        the counts of the best chain, for nv compare
    best.profile.json       the calibration that those counts bind to
    best.timing.json        how IBM scheduled each circuit, when IBM sends it
    best.job.json           the submitted job, until the script collects it
    best.usage.json         IBM's estimate of the QPU time of that job
    median.*                the same files for the median chain

The script needs an IBM Quantum account. Save the account with
``QiskitRuntimeService.save_account``, or put an API key in ``IBM_QUANTUM_TOKEN``.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import signal
import socket
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_on_ibm as runner  # noqa: E402

import noisevault as nv  # noqa: E402
from noisevault import check, layout  # noqa: E402
from noisevault.counts import PlannedCircuit, plan  # noqa: E402
from noisevault.errors import NoiseVaultError, install_hint  # noqa: E402
from noisevault.profile import Profile, file_bytes  # noqa: E402

CHAINS = ("best", "median")
DEFAULT_ROOT = Path("campaign")
DEFAULT_RESERVE_S = 20.0
CALIBRATION = "calibration.json"
CHAINS_FILE = "chains.json"
LOCK = ".lock"
UNKNOWN_USAGE_S = 10.0
FINAL_STATES = frozenset({"DONE", "ERROR", "CANCELLED"})


@dataclass(frozen=True)
class Account:
    """The four things the campaign asks of an IBM Quantum account. Tests give fakes."""

    pull: Callable[[str, datetime | None], Profile]
    open_backend: Callable[[str, bool], Any]
    open_job: Callable[[str], Any]
    remaining_s: Callable[[], float | None]


@dataclass
class Budget:
    """IBM's estimate of each job, checked against the remaining QPU time before it submits.

    The usage that IBM reports does not include jobs that have not run yet. So the check also
    takes off ``reserved_s``, the estimates of the jobs in the campaign folder that have not run.
    """

    remaining_s: Callable[[], float | None] | None
    reserve_s: float

    def approve(self, usage_s: float, reserved_s: float) -> None:
        if self.remaining_s is None:
            return
        try:
            left = self.remaining_s()
        except Exception as exc:
            left, reason = None, f" ({runner._reason(exc)})"
        else:
            reason = ""
        if left is None:
            raise NoiseVaultError(
                f"could not read the remaining QPU time of your IBM Quantum instance{reason}, so"
                " the campaign did not submit a job",
                hint="run the campaign again. On a plan with no time limit, give --no-budget-check",
            )
        free = left - reserved_s - self.reserve_s
        if usage_s > free:
            raise NoiseVaultError(
                f"the job needs about {usage_s:.1f} s of QPU time, but only {max(free, 0):.1f} s"
                f" is free ({left:.1f} s left, {reserved_s:.1f} s for jobs that have not run yet,"
                f" {self.reserve_s:g} s reserve), so the campaign did not submit it",
                hint="run the campaign again when IBM's usage window gives more time, or give a"
                " smaller --reserve",
            )


@dataclass(frozen=True)
class Chain:
    qubits: tuple[int, ...]
    error: float


@dataclass(frozen=True)
class Status:
    device: str
    week: str
    chain: str
    state: str  # collected, submitted, done, error
    detail: str = ""


@dataclass
class Campaign:
    root: Path
    account: Account
    budget: Budget
    shots: int = runner.DEFAULT_SHOTS
    wait: bool = True
    statuses: list[Status] = field(default_factory=list)

    def run(self, devices: Sequence[str], today: date) -> list[Status]:
        locked: list[str] = []
        try:
            for device in devices:
                try:
                    _lock(self.root / device)
                except NoiseVaultError as exc:
                    self.statuses.append(Status(device, "-", "-", "error", _message(exc)))
                    continue
                locked.append(device)
            submitted: list[Path] = []
            for device in locked:
                submitted += self._submit(device, today)
            for device in locked:
                for job in self._pending(device):
                    if self.wait:
                        self._collect(device, job)
                    else:
                        now = job in submitted
                        self._record(device, job, "submitted", "now" if now else "earlier")
        finally:
            for device in locked:
                _unlock(self.root / device)
        return self.statuses

    def _submit(self, device: str, today: date) -> list[Path]:
        folder = week_folder(self.root / device, today)
        missing = [kind for kind in CHAINS if not _counts(folder, kind).exists()]
        for kind in CHAINS:
            if kind not in missing:
                self.statuses.append(Status(device, folder.name, kind, "done"))
        todo = [kind for kind in missing if not _job_files(folder, kind)]
        if not todo:
            return []
        _section(f"{device}: plan the week of {folder.name}")
        try:
            pulled = self.account.pull(device, None)
            chains = _chains_for_week(folder, pulled, self._refusal(device, pulled))
        except NoiseVaultError as exc:
            for kind in todo:
                self.statuses.append(Status(device, folder.name, kind, "error", _message(exc)))
            return []

        def calibration(at: datetime | None) -> Profile:
            return pulled if at is None else self.account.pull(device, at)

        submitted = []
        for kind in todo:
            _section(f"{device}: submit the {kind} chain {_loci(chains[kind].qubits)}")
            try:
                runner.run(
                    device=device,
                    shots=self.shots,
                    output=_counts(folder, kind),
                    pending=_job(folder, kind),
                    calibration=calibration,
                    open_backend=lambda fractional: self.account.open_backend(device, fractional),
                    open_job=self.account.open_job,
                    confirm=None,
                    layout=chains[kind].qubits,
                    approve=functools.partial(self._approve, folder, kind),
                    wait=False,
                    keep_profile=True,
                )
            except NoiseVaultError as exc:
                self.statuses.append(Status(device, folder.name, kind, "error", _message(exc)))
                continue
            submitted.append(_job(folder, kind))
        return submitted

    def _refusal(self, device: str, pulled: Profile) -> Callable[[tuple[int, ...]], str | None]:
        """Why the backend cannot run a chain, from the checks that ``run_on_ibm.py`` makes
        before it submits, or None. Nothing is submitted."""
        backends: dict[bool, Any] = {}
        failed: list[BaseException] = []

        def open_backend(fractional: bool) -> Any:
            if fractional not in backends:
                try:
                    backend = self.account.open_backend(device, fractional)
                    # IBM loads these over the network. A failure here is the account's too.
                    backend.target  # noqa: B018
                    backend.configuration()
                    backends[fractional] = backend
                except Exception as exc:
                    failed.append(exc)
                    raise
            return backends[fractional]

        def refusal(path: tuple[int, ...]) -> str | None:
            try:
                runner.prepare(lambda at: pulled, open_backend, self.shots, path)
            except NoiseVaultError as exc:
                if failed:  # the account failed, not the chain, so stop the search
                    raise NoiseVaultError(
                        f"could not open {device} through your IBM Quantum account"
                        f" ({runner._reason(failed[0])})",
                        hint="run the campaign again. The campaign did not submit a job",
                    ) from None
                return exc.message
            return None

        return refusal

    def _approve(self, folder: Path, kind: str, batch: runner.Batch) -> None:
        """Submit only a chain with no counts and no job, within the budget. Save the estimate,
        so that the budget of a later run counts the job until it has run."""
        if _counts(folder, kind).exists() or _job_files(folder, kind):
            raise NoiseVaultError(
                f"the {kind} chain of the week of {folder.name} has counts or a job already, so"
                " the campaign did not submit another job"
            )
        self.budget.approve(batch.usage_s, self._reserved_s())
        usage = {"usage_s": batch.usage_s}
        _write(_usage(folder, kind), (json.dumps(usage) + "\n").encode("utf-8"))

    def _reserved_s(self) -> float:
        """The estimates of the jobs in the campaign folder that have not run yet. A job whose
        status IBM does not give counts as not run."""
        total = 0.0
        for job in self.root.glob("*/*/*.job.json"):
            if job.name.startswith("._") or not _is_week(job.parent):
                continue
            if self._finished(job):
                continue
            usage = _usage(job.parent, job.name.split(".")[0])
            try:
                total += float(json.loads(usage.read_text(encoding="utf-8"))["usage_s"])
            except (OSError, ValueError, KeyError, TypeError):
                total += UNKNOWN_USAGE_S
        return total

    def _finished(self, job: Path) -> bool:
        try:
            job_id = runner._job_id_in(job.read_bytes().splitlines(keepends=True))
            if job_id is None:
                return False
            status = self.account.open_job(job_id).status()
        except Exception:
            return False
        return str(getattr(status, "name", status)).upper() in FINAL_STATES

    def _pending(self, device: str) -> Iterator[Path]:
        folder = self.root / device
        if not folder.is_dir():
            return
        for week in sorted(p for p in folder.iterdir() if _is_week(p)):
            for kind in CHAINS:
                yield from _job_files(week, kind)

    def _collect(self, device: str, job: Path) -> None:
        kind = job.name.split(".")[0]
        _section(f"{device}: collect the {kind} chain of the week of {job.parent.name}")
        try:
            measured = runner.run(
                device=device,
                shots=self.shots,
                output=_counts(job.parent, kind),
                pending=job,
                calibration=lambda at: self.account.pull(device, at),
                open_backend=lambda fractional: self.account.open_backend(device, fractional),
                open_job=self.account.open_job,
                confirm=None,
                collect_only=True,
                keep_profile=True,
            )
        except NoiseVaultError as exc:
            # run_on_ibm.run turns Ctrl-C during the wait into this error. Stop the campaign too.
            if exc.message.startswith("stopped before the script saved"):
                raise KeyboardInterrupt from None
            self._record(device, job, "error", _message(exc))
            return
        assert measured is not None
        self._record(device, job, "collected", f"job {measured.execution.job_ids[0]}")

    def _record(self, device: str, job: Path, state: str, detail: str) -> None:
        kind = job.name.split(".")[0]
        self.statuses.append(Status(device, job.parent.name, kind, state, detail))


def _lock(device_folder: Path) -> None:
    """Take the lock of one device, so that two campaign runs cannot submit the same chain.

    A lock that a stopped process on this computer left behind is taken over."""
    device_folder.mkdir(parents=True, exist_ok=True)
    path = device_folder / LOCK
    owner = f"{socket.gethostname()} {os.getpid()}\n".encode()
    for _ in range(2):
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        except FileExistsError:
            if _stale(path):
                path.unlink(missing_ok=True)
                continue
            raise NoiseVaultError(
                f"{path} exists, so another campaign run may be at work on {device_folder.name}",
                hint=f"if no other run is at work, delete {path}",
            ) from None
        with os.fdopen(fd, "wb") as handle:
            handle.write(owner)
        return
    raise NoiseVaultError(f"could not take the lock {path}", hint="run the campaign again")


def _stale(path: Path) -> bool:
    try:
        host, pid = path.read_text(encoding="utf-8").split()
        if host != socket.gethostname():
            return False
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return True
    except (OSError, ValueError):
        return False
    return False


def _unlock(device_folder: Path) -> None:
    (device_folder / LOCK).unlink(missing_ok=True)


def week_folder(device_folder: Path, today: date) -> Path:
    """The folder of ``today``'s ISO week: an existing folder of that week, else ``today``."""
    week = today.isocalendar()[:2]
    if device_folder.is_dir():
        for found in sorted(p for p in device_folder.iterdir() if _is_week(p)):
            if date.fromisoformat(found.name).isocalendar()[:2] == week:
                return found
    return device_folder / today.isoformat()


def pick_chains(
    profile: Profile, refusal: Callable[[tuple[int, ...]], str | None] = lambda path: None
) -> dict[str, Chain]:
    """The best chain, as ``plan`` picks it, and the median chain of the same length.

    The candidates are the connected chains of that length, other than the best chain, on which
    ``plan`` gives the same circuits as on the best chain: the same names, widths and gates,
    with other delays. The median chain is the candidate at the median of the summed error that
    ``suggest_layout`` minimizes. ``refusal`` gives the reason why the backend cannot run a
    chain, or None. When the backend refuses the median candidate, the next candidate outward
    from the median is the median chain.
    """
    best, candidates = eligible_chains(profile)
    middle = (len(candidates) - 1) // 2
    reason = None
    for i in sorted(range(len(candidates)), key=lambda i: (abs(i - middle), i)):
        error, path = candidates[i]
        reason = refusal(path)
        if reason is None:
            return {"best": best, "median": Chain(path, error)}
    raise NoiseVaultError(
        f"the backend refuses all {len(candidates)} candidates for the median chain of"
        f" {profile.id} (the last because {reason})"
    )


def eligible_chains(profile: Profile) -> tuple[Chain, list[tuple[float, tuple[int, ...]]]]:
    """The best chain, and every other chain with the circuits of the best chain, with its summed
    error, lowest error first."""
    best = plan(profile)
    shape = _shape(best)
    qubits = max((c.qubits for c in best), key=len)
    candidates = []
    for error, found in _ranked_chains(profile, len(qubits)):
        if set(found) == set(qubits):
            continue
        for path in (found, found[::-1]):
            try:
                planned = plan(profile, path)
            except NoiseVaultError:  # plan refuses the chain, so it is not a candidate
                continue
            if _shape(planned) == shape:
                candidates.append((error, path))
                break
    if not candidates:
        raise NoiseVaultError(
            f"{profile.id} has no chain of {len(qubits)} qubits other than the best chain with"
            " the circuits of the best chain"
        )
    return Chain(qubits, _error(profile, qubits)), candidates


def _shape(planned: Sequence[PlannedCircuit]) -> list[tuple[Any, ...]]:
    return [
        (
            c.name,
            len(c.qubits),
            [(op.name, op.qubits, op.params) for op in c.ops if op.name != "delay"],
        )
        for c in planned
    ]


def _ranked_chains(profile: Profile, n: int) -> list[tuple[float, tuple[int, ...]]]:
    table = profile.table
    if table.all_to_all:
        raise NoiseVaultError(
            f"{profile.id} connects every pair of qubits, so the median chain is not defined"
        )
    # A qubit without a usable 1-qubit gate or without a readout error adds no error to the sum,
    # so a chain through it would rank too low. IBM disables every 1-qubit gate on some qubits
    # but keeps their readout and their 2-qubit gates calibrated.
    usable = []
    for q in range(table.num_qubits):
        if table.qubit(q).disabled or not layout.can_measure(table, q):
            continue
        cost = layout._qubit_cost(table, q, ())
        if cost.no_gate == 0 and cost.no_readout == 0:
            usable.append(q)
    neighbors = {
        q: [
            b
            for b in nbs
            if check._usable_pair(profile, min(q, b), max(q, b))
            and layout._edge_cost(table, min(q, b), max(q, b)) is not None
        ]
        for q, nbs in layout._neighbors(table, usable).items()
    }
    found: list[tuple[float, tuple[int, ...]]] = []

    def grow(path: tuple[int, ...]) -> None:
        if len(path) == n:
            if path[0] < path[-1] or n == 1:
                found.append((_error(profile, path), path))
            return
        for nb in neighbors[path[-1]]:
            if nb not in path:
                grow((*path, nb))

    for q in usable:
        grow((q,))
    return sorted(found)


def _error(profile: Profile, path: Sequence[int]) -> float:
    """The summed error of ``suggest_layout``: 1-qubit gate, readout and 2-qubit gate."""
    table = profile.table
    total = sum(layout._qubit_cost(table, q, ()).error for q in path)
    for a, b in zip(path, path[1:], strict=False):
        edge = layout._edge_cost(table, min(a, b), max(a, b))
        total += 0.0 if edge is None else edge.error
    return total


def _chains_for_week(
    folder: Path, pulled: Profile, refusal: Callable[[tuple[int, ...]], str | None]
) -> dict[str, Chain]:
    """The chains of the week. The first run of the week picks them, and reruns keep them."""
    saved = folder / CHAINS_FILE
    if saved.exists():
        data = json.loads(saved.read_text(encoding="utf-8"))
        return {k: Chain(tuple(data[k]["qubits"]), data[k]["error"]) for k in CHAINS}
    chains = pick_chains(pulled, refusal)
    folder.mkdir(parents=True, exist_ok=True)
    record = {
        "calibration": pulled.calibration_fingerprint,
        **{k: {"qubits": list(c.qubits), "error": c.error} for k, c in chains.items()},
    }
    _write_new(folder / CALIBRATION, file_bytes(pulled, folder / CALIBRATION))
    _write_new(saved, (json.dumps(record, indent=1) + "\n").encode("utf-8"))
    return chains


def _write_new(path: Path, data: bytes) -> None:
    if not path.exists():
        _write(path, data)


def _write(path: Path, data: bytes) -> None:
    """Write the whole file, then move it into place, so a stopped run leaves no half file."""
    part = path.with_name(f"{path.name}.part")
    part.write_bytes(data)
    os.replace(part, path)


def _counts(folder: Path, kind: str) -> Path:
    return folder / f"{kind}.counts.json"


def _job(folder: Path, kind: str) -> Path:
    return folder / f"{kind}.job.json"


def _job_files(folder: Path, kind: str) -> list[Path]:
    """The job file of a chain, and the copies ``run_on_ibm.py`` saves under another name when
    the job file changed while it submitted."""
    copies = sorted(p for p in folder.glob(f"{kind}.*.job.json") if not p.name.startswith("._"))
    return [p for p in (_job(folder, kind), *copies) if p.exists()]


def _usage(folder: Path, kind: str) -> Path:
    return folder / f"{kind}.usage.json"


def _is_week(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        return date.fromisoformat(path.name).isoformat() == path.name
    except ValueError:
        return False


def _loci(qubits: Sequence[int]) -> str:
    return "-".join(map(str, qubits))


def _message(exc: NoiseVaultError) -> str:
    return f"{exc.message}. Hint: {exc.hint}" if exc.hint else exc.message


def _section(title: str) -> None:
    print(f"\n== {title}", flush=True)


def table(statuses: Sequence[Status]) -> str:
    rows = [("device", "week", "chain", "state")] + [
        (s.device, s.week, s.chain, s.state) for s in statuses
    ]
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)) for row in rows]
    lines = [line.rstrip() for line in lines]
    for s in statuses:
        if s.state == "error":
            lines.append(f"error: {s.device} {s.week} {s.chain}: {s.detail}")
    return "\n".join(lines)


def ibm_account() -> Account:
    service = runner._service()

    def remaining_s() -> float | None:
        usage = service.usage()
        if usage.get("usage_limit_reached"):
            return 0.0
        left = usage.get("usage_remaining_seconds")
        return None if left is None else float(left)

    return Account(
        pull=lambda device, at: nv.pull(device, source="ibm-account", at=at),
        open_backend=lambda device, fractional: service.backend(
            device, use_fractional_gates=fractional
        ),
        open_job=service.job,
        remaining_s=remaining_s,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        import qiskit_ibm_runtime  # noqa: F401
    except ImportError:
        return runner._fail("ibm_campaign.py needs qiskit-ibm-runtime", install_hint("ibm"))
    try:
        account = ibm_account()
    except NoiseVaultError as exc:
        return runner._fail(exc.message, exc.hint)
    campaign = Campaign(
        root=args.root,
        account=account,
        budget=Budget(None if args.no_budget_check else account.remaining_s, args.reserve),
        shots=args.shots,
        wait=not args.no_wait,
    )
    signal.signal(signal.SIGTERM, _interrupt)
    try:
        statuses = campaign.run(args.devices, datetime.now(UTC).date())
    except KeyboardInterrupt:
        print("stopped. Run the same command again to collect the submitted jobs", file=sys.stderr)
        return 130
    print()
    print(table(statuses))
    return 1 if any(s.state == "error" for s in statuses) else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ibm_campaign.py",
        description="Run this week's nv compare jobs on IBM devices and collect earlier jobs.",
    )
    parser.add_argument("devices", metavar="DEVICE", nargs="+", help="IBM devices, such as ibm_fez")
    parser.add_argument(
        "--root",
        metavar="FOLDER",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"the campaign folder (default {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--shots",
        metavar="N",
        type=runner._positive,
        default=runner.DEFAULT_SHOTS,
        help=f"shots per circuit (default {runner.DEFAULT_SHOTS})",
    )
    parser.add_argument(
        "--reserve",
        metavar="SECONDS",
        type=_seconds,
        default=DEFAULT_RESERVE_S,
        help=f"QPU seconds to keep free (default {DEFAULT_RESERVE_S:g})",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="submit and stop. A rerun collects the jobs",
    )
    parser.add_argument(
        "--no-budget-check",
        action="store_true",
        help="submit without reading the remaining QPU time",
    )
    return parser


def _interrupt(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt


def _seconds(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        value = -1.0
    if not value >= 0:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number of seconds of 0 or more")
    return value


if __name__ == "__main__":
    sys.exit(main())
