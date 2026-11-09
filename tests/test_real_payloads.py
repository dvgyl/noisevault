"""Real saved IQM and Rigetti payloads import, keep their pinned values and pass nv check.

Each real payload is one folder (see the README.md in each real/ folder):

- ``tests/fixtures/iqm/real/<name>/`` with ``metrics.json`` and optionally ``architecture.json``
- ``tests/fixtures/rigetti/real/<name>/`` with ``isa.json``

Each folder can also hold ``expect.json``, with values that the profile must have. A payload
that the vendor's terms do not let the repository publish goes in a private folder.
NOISEVAULT_REAL_PAYLOADS names that folder, which has the same ``iqm/`` and ``rigetti/``
subfolders. With no real payload, the tests skip.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import noisevault as nv
from noisevault.sources.iqm import from_iqm
from noisevault.sources.rigetti import from_rigetti

FIXTURES = Path(__file__).parent / "fixtures"
ENV = "NOISEVAULT_REAL_PAYLOADS"
_REQUIRED = {"iqm": "metrics.json", "rigetti": "isa.json"}


@dataclass(frozen=True)
class Payload:
    vendor: str
    folder: Path
    origin: str = "repo"

    @property
    def id(self) -> str:
        return f"{self.origin}:{self.vendor}/{self.folder.name}"

    def load(self) -> nv.Profile:
        required = self.folder / _REQUIRED[self.vendor]
        assert required.is_file(), f"{self.folder} has no {_REQUIRED[self.vendor]}"
        return _IMPORTERS[self.vendor](self.folder)

    def expected(self) -> dict[str, Any]:
        path = self.folder / "expect.json"
        return json.loads(path.read_text()) if path.exists() else {}


def _iqm(folder: Path) -> nv.Profile:
    architecture = folder / "architecture.json"
    return from_iqm(
        folder / "metrics.json",
        architecture=architecture if architecture.exists() else None,
        device=folder.name,
    )


def _rigetti(folder: Path) -> nv.Profile:
    return from_rigetti(folder / "isa.json")


_IMPORTERS: dict[str, Callable[[Path], nv.Profile]] = {"iqm": _iqm, "rigetti": _rigetti}


def _actual(profile: nv.Profile) -> dict[str, Any]:
    """The values that expect.json can pin, by key."""
    calibrated_at = profile.device.calibrated_at
    return {
        "id": profile.id,
        "num_qubits": profile.device.num_qubits,
        "calibrated_at": calibrated_at.isoformat() if calibrated_at else None,
        "gates": sorted(profile.gates),
        "calibrated": sorted({r.gate for r in profile.calibrations if r.avg_infidelity}),
        "fingerprint": profile.fingerprint,
        "artifact_hash": profile.artifact_hash,
    }


def find_payloads(bases: list[tuple[str, Path, str]]) -> list[Payload]:
    """Every payload folder in each (vendor, base folder, origin).

    Folders whose names start with ``.`` or ``_`` are not payloads. A folder without the
    vendor's file is still a payload, so its test fails instead of skipping in silence.
    """
    found = []
    for vendor, base, origin in bases:
        if base.is_dir():
            for folder in sorted(p for p in base.iterdir() if p.is_dir()):
                if not folder.name.startswith((".", "_")):
                    found.append(Payload(vendor, folder, origin))
    return found


def _bases() -> list[tuple[str, Path, str]]:
    bases = [(vendor, FIXTURES / vendor / "real", "repo") for vendor in _REQUIRED]
    private = os.environ.get(ENV)
    if private:
        bases += [(vendor, Path(private) / vendor, "private") for vendor in _REQUIRED]
    return bases


PAYLOADS = find_payloads(_bases())


def _cases() -> list[Any]:
    if PAYLOADS:
        return [pytest.param(p, id=p.id) for p in PAYLOADS]
    return [pytest.param(None, marks=pytest.mark.skip(reason=f"no real payloads; see {ENV}"))]


@pytest.mark.parametrize("payload", _cases())
def test_a_real_payload_imports_with_data(payload: Payload) -> None:
    profile = payload.load()
    assert _actual(profile)["calibrated"], "no gate has a calibrated record"
    assert any(q.readout is not None for q in profile.qubits), "no qubit has a readout error"


@pytest.mark.parametrize("payload", _cases())
def test_a_real_payload_keeps_its_expected_values(payload: Payload) -> None:
    expected = payload.expected()
    actual = _actual(payload.load())
    unknown = set(expected) - set(actual)
    assert not unknown, f"expect.json has unknown keys {sorted(unknown)}"
    assert {key: actual[key] for key in expected} == expected


@pytest.mark.parametrize("payload", _cases())
def test_a_real_payload_passes_nv_check(payload: Payload) -> None:
    result = payload.load().check()
    if not result.frameworks:
        pytest.skip("no framework is installed, so nv check runs nothing")
    assert result.passed, result.summary()


def test_the_private_folder_exists() -> None:
    private = os.environ.get(ENV)
    if private is None:
        pytest.skip(f"{ENV} is not set")
    assert Path(private).is_dir(), f"{ENV} names {private}, which is not a folder"


def test_the_layout_finds_payload_folders(tmp_path: Path) -> None:
    for folder in ("iqm/garnet-2026-11-01", "rigetti/ankaa-3", "iqm/_notes"):
        (tmp_path / folder).mkdir(parents=True)
    (tmp_path / "iqm" / "README.md").write_text("")
    bases = [("iqm", tmp_path / "iqm", "repo"), ("rigetti", tmp_path / "rigetti", "private")]
    assert [p.id for p in find_payloads(bases)] == [
        "repo:iqm/garnet-2026-11-01",
        "private:rigetti/ankaa-3",
    ]


def test_a_payload_folder_without_its_file_fails_its_test(tmp_path: Path) -> None:
    (tmp_path / "ankaa-3").mkdir()
    (tmp_path / "ankaa-3" / "ankaa.json").write_text("{}")
    (payload,) = find_payloads([("rigetti", tmp_path, "repo")])
    with pytest.raises(AssertionError, match="has no isa.json"):
        test_a_real_payload_imports_with_data(payload)


def test_the_synthetic_fixtures_pass_the_real_payload_checks(tmp_path: Path) -> None:
    """The checks accept a payload of the documented shape and detect a changed value."""
    iqm, rigetti = tmp_path / "iqm" / "synthetic", tmp_path / "rigetti" / "synthetic"
    iqm.mkdir(parents=True)
    rigetti.mkdir(parents=True)
    (iqm / "metrics.json").write_bytes((FIXTURES / "iqm" / "metrics.json").read_bytes())
    (rigetti / "isa.json").write_bytes((FIXTURES / "rigetti" / "cepheus_isa.json").read_bytes())
    payloads = find_payloads([("iqm", iqm.parent, "repo"), ("rigetti", rigetti.parent, "repo")])
    assert len(payloads) == 2
    for payload in payloads:
        (payload.folder / "expect.json").write_text(json.dumps(_actual(payload.load())))
        test_a_real_payload_imports_with_data(payload)
        test_a_real_payload_keeps_its_expected_values(payload)
    (iqm / "expect.json").write_text(json.dumps({"fingerprint": "0" * 64}))
    with pytest.raises(AssertionError):
        test_a_real_payload_keeps_its_expected_values(payloads[0])
