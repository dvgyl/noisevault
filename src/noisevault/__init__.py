"""NoiseVault: real device noise, pinned and portable.

Importing this package loads no quantum framework. Each export imports its framework when you
use the export.
"""

from __future__ import annotations

__version__ = "0.3.0"

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .catalog import ProfileInfo, load, profiles, pull
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
from .profile import Profile, json_schema
from .report import Report

if TYPE_CHECKING:
    from datetime import date, datetime

    from .sources.hf_archive import ArchiveSpan

_LAZY_MODULES = {"stim": "noisevault.frameworks.stim"}


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


def __getattr__(name: str) -> Any:
    if name in _LAZY_MODULES:
        return importlib.import_module(_LAZY_MODULES[name])
    raise AttributeError(f"module 'noisevault' has no attribute {name!r}")


__all__ = [
    "AmbiguousRef",
    "CountsError",
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
    "from_braket",
    "from_calibration_archive",
    "from_cirq_google",
    "from_ibm_csv",
    "from_qiskit_backend",
    "json_schema",
    "load",
    "profiles",
    "pull",
]
