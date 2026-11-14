"""An IBM Quantum job: when it ran, its circuits, its counts and its options.

:func:`from_ibm_job` reads a SamplerV2 job of qiskit-ibm-runtime and gives the calibration in
effect when the job ran, and a counts file of the job. :func:`job_start` opens a job by its id
through the saved IBM Quantum account.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import CountsError, SourceUnavailable

if TYPE_CHECKING:
    from ..counts import MeasuredCounts
    from ..profile import Profile

PROGRAM = "sampler"  # the program id of the server-side SamplerV2


def started_at(metrics: Mapping[str, Any]) -> datetime:
    """When the job started running, in UTC. IBM sends an ISO 8601 string, which the runtime
    reads as UTC when it has no offset. Local testing mode gives a naive datetime in local time,
    which ``astimezone`` reads as local time."""
    stamp = metrics["timestamps"]["running"]
    if stamp is None:
        raise ValueError("the job has no start time")
    if isinstance(stamp, str):
        stamp = datetime.fromisoformat(stamp)
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
    return stamp.astimezone(UTC)


def execution(options: Any, job_ids: Sequence[str]) -> dict[str, Any]:
    """The ``execution`` block of a counts file for a SamplerV2 job with these options."""
    import qiskit
    import qiskit_ibm_runtime

    from ..counts import _RUN_RULES

    runtime, framework = qiskit_ibm_runtime.__version__, qiskit.__version__
    return {
        "client": f"qiskit-ibm-runtime {runtime} SamplerV2, qiskit {framework}",
        **{rule.flag: rule.required for rule in _RUN_RULES if rule.flag},
        "job_ids": list(job_ids),
        "options": options,
    }


def job_start(job_id: str) -> tuple[str, datetime]:
    """The device of job ``job_id`` and the time that the job started running, through the
    saved IBM Quantum account."""
    from .ibm_account import _service

    service = _service()
    try:
        job = service.job(job_id)
        device = job.backend().name
        metrics = job.metrics()
    except Exception as exc:  # the runtime raises several types for "not found" and network
        raise SourceUnavailable(
            f"your IBM account cannot open job {job_id} ({exc})",
            hint="check the job id, and that the saved account is the one that ran the job",
        ) from None
    try:
        return device, started_at(metrics)
    except (KeyError, TypeError, ValueError):
        raise SourceUnavailable(
            f"job {job_id} has no start time, so it did not run yet",
            hint="wait until the job ran, then pull again",
        ) from None


def from_ibm_job(job: Any, *, source: str = "ibm") -> tuple[Profile, MeasuredCounts]:
    """The calibration in effect when a SamplerV2 job ran, and the counts of the job.

    The function reads the device, the circuits of the job, the counts, the start time, the job
    id and the options from ``job``. Then it pulls the calibration of the device at the start
    time from ``source`` (``"ibm"``, with no account, or ``"ibm-account"``) and saves it in the
    vault, as ``nv.pull`` does. The counts bind to that calibration. Raises CountsError for a job
    that a counts file cannot hold.
    """
    from ..catalog import pull
    from ..frameworks.qiskit import counts_from_qiskit

    job_id = job.job_id()
    program = getattr(job, "primitive_id", None)
    if program != PROGRAM:
        raise CountsError(
            f"job {job_id} ran the IBM program {program!r}, not {PROGRAM!r}",
            hint="give a job of SamplerV2, whose result holds counts",
        )
    backend = job.backend()
    inputs = job.inputs
    circuits = [_circuit(pub, i, job_id) for i, pub in enumerate(inputs["pubs"])]
    counts = [pub.join_data().get_counts() for pub in job.result()]
    run_at = started_at(job.metrics())
    profile = pull(backend.name, at=run_at, source=source)
    block = execution(dict(inputs.get("options") or {}), [job_id])
    measured = counts_from_qiskit(
        circuits,
        counts,
        profile=profile,
        run_at=run_at,
        backend=backend.name,
        job_ids=block["job_ids"],
        client=block["client"],
        options=block["options"],
        dt=getattr(backend, "dt", None),
    )
    return profile, measured


def _circuit(pub: Any, index: int, job_id: str) -> Any:
    if hasattr(pub, "circuit"):
        circuit, values = pub.circuit, pub.parameter_values
    else:
        circuit, *rest = pub
        values = rest[0] if rest else None
    if _has_values(values):
        raise CountsError(
            f"job {job_id}: pub {index} gives parameter values, and a counts file holds one"
            " circuit with fixed parameters for each pub",
            hint="bind the parameters in the circuit, such as circuit.assign_parameters(values),"
            " and run a pub with no parameter values",
        )
    return circuit


def _has_values(values: Any) -> bool:
    if values is None:
        return False
    if hasattr(values, "num_parameters"):
        return values.num_parameters > 0
    if isinstance(values, Mapping):
        return len(values) > 0
    return np.size(values) > 0
