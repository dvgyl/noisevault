"""NoiseVault: real device noise, pinned and portable.

Importing this package loads no quantum framework. Each export imports its framework when you
use the export.
"""

from __future__ import annotations

__version__ = "0.4.0"

import importlib
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .catalog import ProfileInfo, load, profiles, pull
from .derived import typical
from .errors import (
    AmbiguousRef,
    CountsError,
    DisabledGateError,
    FingerprintMismatch,
    LayoutError,
    MigrationWarning,
    MissingCalibrationError,
    NoiseApproximationWarning,
    NoiseVaultError,
    NoiseVaultWarning,
    ProfileNotFound,
    SourceDataError,
    SourceUnavailable,
    UnsupportedEffect,
)
from .profile import DeviceMedians, Profile, json_schema
from .report import Report

if TYPE_CHECKING:
    from datetime import date, datetime

    from .counts import MeasuredCounts
    from .series import History
    from .sources.hf_archive import ArchiveSpan

_LAZY_MODULES = {"stim": "noisevault.frameworks.stim", "events": "noisevault.events"}


def from_qiskit_backend(backend: Any) -> Profile:
    """A profile from a Qiskit BackendV2 (its Target), such as a qiskit-ibm-runtime fake backend."""
    from .sources.qiskit_backend import from_qiskit_backend as convert

    return convert(backend)


def from_ibm_csv(path: str | Path, *, device: str, calibrated_at: Any) -> Profile:
    """A profile from an IBM Quantum calibration CSV download."""
    from .sources.ibm_csv import from_ibm_csv as convert

    return convert(path, device=device, calibrated_at=calibrated_at)


def from_braket(path_or_dict: str | Path | dict[str, Any], *, device: str | None = None) -> Profile:
    """A profile from saved Amazon Braket standardized device properties.

    Braket properties do not name the device, so pass ``device`` (for example ``"garnet"``) to
    name the profile. By default, the name is the file name without its suffix.
    """
    from .sources.braket import from_braket as convert

    return convert(path_or_dict, device=device)


def from_iqm(
    metrics: str | Path | dict[str, Any],
    *,
    architecture: str | Path | dict[str, Any] | None = None,
    device: str | None = None,
) -> Profile:
    """A profile from a saved IQM Resonance quality metric set.

    Save ``IQMClient.get_quality_metric_set().model_dump_json()``. Pass the dynamic quantum
    architecture of the same calibration set as ``architecture`` to disable the qubits and pairs
    that IQM did not calibrate. ``device`` names the profile, such as ``"garnet"``. By default,
    the name is the file name without its suffix.
    """
    from .sources.iqm import from_iqm as convert

    return convert(metrics, architecture=architecture, device=device)


def from_rigetti(
    path_or_dict: str | Path | dict[str, Any], *, device: str | None = None
) -> Profile:
    """A profile from a saved Rigetti QCS instruction set architecture (ISA).

    Save ``qcs_sdk.qpu.isa.get_instruction_set_architecture(id).json()``. ``device`` names the
    profile. By default, the name is the ISA's ``name``, such as ``"Ankaa-3"``.
    """
    from .sources.rigetti import from_rigetti as convert

    return convert(path_or_dict, device=device)


def from_cirq_google(processor_id: str) -> Profile:
    """A profile from a calibration cirq_google ships (rainbow, weber, willow_pink)."""
    from .sources.google import from_cirq_google as convert

    return convert(processor_id)


def from_calibration_archive(
    path: str | Path, device: str, *, at: str | date | datetime | None = None
) -> Profile:
    """A profile from a local copy of the dataset phanerozoic/qiskit-calibration-drift.

    Each property takes its newest calibration at or before ``at``. With no ``at``, each property
    takes its newest calibration.
    Install ``noisevault[hf]`` for pyarrow.
    """
    from .sources.hf_archive import from_calibration_archive as convert

    return convert(path, device, at=at)


def calibration_archive_devices(path: str | Path) -> dict[str, ArchiveSpan]:
    """Each device in a local copy of phanerozoic/qiskit-calibration-drift, with its ``at`` range.

    ``first`` is the earliest ``at`` accepted, and ``last`` is the device's newest calibration.
    """
    from .sources.hf_archive import calibration_archive_devices as devices

    return devices(path)


def history(
    device: str,
    start: str | date | datetime,
    end: str | date | datetime,
    *,
    step: timedelta = timedelta(days=1),
    source: str = "ibm",
    archive: str | Path | None = None,
) -> History:
    """The calibrations of ``device`` from ``start`` to ``end``, and a table of their values.

    ``source="ibm"`` pulls from IBM's public endpoint at each ``step``. ``source="offline"`` reads
    each profile that loads offline, with no network. ``source="archive"`` reads a local copy of
    the IBM calibration archive at ``archive`` at each ``step``. The result has ``rows``,
    ``to_csv(path)``, ``to_dict()`` and ``summary()``.
    """
    from .series import history as collect

    return collect(device, start, end, step=step, source=source, archive=archive)


def counts_from_qiskit(
    circuits: Any,
    counts: Any,
    *,
    profile: Profile,
    run_at: datetime,
    backend: str | None = None,
    job_ids: Sequence[str] = (),
    source: str = "hardware",
    client: str | None = None,
    options: Mapping[str, Any] | None = None,
    dt: float | None = None,
    gate_twirling: bool = False,
    measure_twirling: bool = False,
    dynamical_decoupling: bool = False,
    init_qubits: bool = True,
) -> MeasuredCounts:
    """A counts file for Qiskit circuits that a device ran, and their Qiskit counts.

    Give the circuits as the device ran them: native gates on the physical qubits of the device,
    with each measurement after the last gate on its qubit. ``counts`` holds one dict of Qiskit
    bit strings for each circuit. ``profile`` is the calibration that the counts bind to, and
    ``run_at`` is when the device started the first circuit. ``dt`` is the sample time of the
    device in seconds, for delays in ``dt``. The run flags state how the device ran the circuits.
    """
    from .frameworks.qiskit import counts_from_qiskit as convert

    return convert(
        circuits,
        counts,
        profile=profile,
        run_at=run_at,
        backend=backend,
        job_ids=job_ids,
        source=source,
        client=client,
        options=options,
        dt=dt,
        gate_twirling=gate_twirling,
        measure_twirling=measure_twirling,
        dynamical_decoupling=dynamical_decoupling,
        init_qubits=init_qubits,
    )


def from_ibm_job(job: Any, *, source: str = "ibm") -> tuple[Profile, MeasuredCounts]:
    """The calibration in effect when a SamplerV2 job of IBM Quantum ran, and its counts.

    The calibration comes from ``source`` (``"ibm"`` with no account, or ``"ibm-account"``) and
    goes into the vault, as ``nv.pull`` saves it.
    """
    from .sources.ibm_job import from_ibm_job as read

    return read(job, source=source)


def __getattr__(name: str) -> Any:
    if name in _LAZY_MODULES:
        return importlib.import_module(_LAZY_MODULES[name])
    raise AttributeError(f"module 'noisevault' has no attribute {name!r}")


__all__ = [
    "AmbiguousRef",
    "CountsError",
    "DeviceMedians",
    "DisabledGateError",
    "FingerprintMismatch",
    "LayoutError",
    "MigrationWarning",
    "MissingCalibrationError",
    "NoiseApproximationWarning",
    "NoiseVaultError",
    "NoiseVaultWarning",
    "Profile",
    "ProfileInfo",
    "ProfileNotFound",
    "Report",
    "SourceDataError",
    "SourceUnavailable",
    "UnsupportedEffect",
    "__version__",
    "calibration_archive_devices",
    "counts_from_qiskit",
    "from_braket",
    "from_calibration_archive",
    "from_cirq_google",
    "from_ibm_csv",
    "from_ibm_job",
    "from_iqm",
    "from_qiskit_backend",
    "from_rigetti",
    "history",
    "json_schema",
    "load",
    "profiles",
    "pull",
    "typical",
]
