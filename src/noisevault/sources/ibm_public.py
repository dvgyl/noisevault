"""IBM Quantum's public calibration endpoint: current and past calibrations, no account needed."""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from dataclasses import replace
from datetime import date, datetime
from typing import Annotated

from pydantic import AfterValidator, BaseModel, TypeAdapter

from .. import __version__
from ..errors import SourceUnavailable, did_you_mean
from ..profile import Profile, iso_z
from . import OFFLINE_HINT, OLDER_HINT, Origin, checked_by, download, read_reply
from .qiskit_backend import (
    as_utc,
    calibration_from_properties,
    now_utc,
    processor_name,
    sha256_bytes,
    to_profile,
)

BASE_URL = "https://quantum.cloud.ibm.com/api/v1/public/backends"
TIMEOUT_S = 30.0
_SENDER = "IBM's public endpoint"
_TRY_LATER = "try again later, or pull through your IBM account with source='ibm-account'"


_Time = Annotated[str, AfterValidator(checked_by(as_utc))]


class _Parameter(BaseModel):
    name: str
    value: float | None = None
    unit: str | None = None
    date: _Time | None = None


class _Gate(BaseModel):
    gate: str
    qubits: list[int]
    parameters: list[_Parameter] | None = None


class _Properties(BaseModel):
    qubits: list[list[_Parameter]] | None = None
    gates: list[_Gate] | None = None
    last_update_date: _Time | None = None


class _Device(BaseModel):
    name: str


class _ProcessorType(BaseModel):
    family: str | None = None


class _Configuration(BaseModel):
    processor_type: _ProcessorType | None = None


_PROPERTIES = TypeAdapter(_Properties)
_LISTING = TypeAdapter(list[_Device])
_CONFIGURATION = TypeAdapter(_Configuration)


def pull(device: str, *, at: str | date | datetime | None = None) -> Profile:
    """The calibration of ``device`` now, or the newest one older than ``at``.

    ``at`` is a datetime or ISO 8601 string. A date or a naive time is UTC.
    """
    name = device.strip().lower()
    url = properties_url(name, at)
    try:
        raw = fetch(url)
    except _NotFound:
        raise _not_found(name, at) from None
    props = read_reply(raw, url, _PROPERTIES, sender=_SENDER, hint=_TRY_LATER)
    if not props.get("qubits"):
        raise SourceUnavailable(f"IBM's public endpoint returned no qubit data for {name} ({url})")
    origin = Origin(f"{_SENDER} ({url})", hint=OLDER_HINT)
    cal = replace(
        calibration_from_properties(props, origin=origin), name=name, processor=_processor(name)
    )
    return to_profile(
        cal,
        {
            "data_kind": "measured",
            "source_kind": "public_api",
            "source": "IBM Quantum public calibration endpoint",
            "source_url": url,
            "attribution": "IBM Quantum",
            "redistributable": "unknown",
            "retrieved_at": now_utc(),
            "source_hash": sha256_bytes(raw),
        },
        origin=origin,
    )


def properties_url(name: str, at: str | date | datetime | None = None) -> str:
    url = f"{BASE_URL}/{urllib.parse.quote(name)}/properties"
    if at is None:
        return url
    stamp = iso_z(as_utc(at))
    return f"{url}?{urllib.parse.urlencode({'updated_before': stamp})}"


class _NotFound(Exception):
    pass


def fetch(url: str) -> bytes:
    """GET ``url``: its body, _NotFound on HTTP 404, SourceUnavailable on any other failure."""
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": f"noisevault/{__version__}"}
    )
    try:
        return download(request, TIMEOUT_S)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise _NotFound(url) from None
        raise SourceUnavailable(
            f"IBM's public endpoint answered HTTP {exc.code} for {url}", hint=_TRY_LATER
        ) from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise SourceUnavailable(
            f"could not reach IBM's public endpoint ({reason})", hint=OFFLINE_HINT
        ) from None


def listed_devices() -> list[str]:
    """Names of the devices the public endpoint lists right now."""
    listing = read_reply(fetch(BASE_URL), BASE_URL, _LISTING, sender=_SENDER, hint=_TRY_LATER)
    return sorted(entry["name"] for entry in listing)


def _processor(name: str) -> str | None:
    """The processor type from the public configuration, or None if NoiseVault cannot read it."""
    url = f"{BASE_URL}/{urllib.parse.quote(name)}/configuration"
    try:
        config = read_reply(fetch(url), url, _CONFIGURATION, sender=_SENDER, hint=_TRY_LATER)
    except (_NotFound, SourceUnavailable):
        return None
    return processor_name(config.get("processor_type"))


def _not_found(name: str, at: str | date | datetime | None) -> SourceUnavailable:
    try:
        listed = listed_devices()
    except (_NotFound, SourceUnavailable):
        listed = None
    if at is not None and listed is not None and name in listed:
        return SourceUnavailable(
            f"IBM's public endpoint has no {name} calibration older than {at}",
            hint="pick a later date, or pull through your IBM account with source='ibm-account'",
        )
    from ..catalog import bundled_profiles as bundled

    known = f" (the endpoint lists {', '.join(listed)})" if listed else ""
    guess = did_you_mean(name, listed or [])
    if any(info.id == name for info in bundled()):
        hint = f"{name} is bundled, so nv.load({name!r}) loads it offline"
    else:
        hint = "if your IBM account can see it, pull with source='ibm-account'"
    retired = "if not, the device may be retired" if guess else "the device may be retired"
    return SourceUnavailable(
        f"{name} is not listed on the public endpoint{known}; {guess}{retired}", hint=hint
    )


def bundled_profiles() -> list[Profile]:
    """None, because IBM's terms for this data are not an open license."""
    return []
