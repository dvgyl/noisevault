"""IonQ's public characterization endpoint (API v0.4): dated device medians, no account needed.

IonQ publishes one record per characterization with device-wide medians only: 1Q, 2Q and SPAM
fidelity, gate and readout times, T1 and T2. Every qubit and pair of the profile gets those
medians. The importer writes each interpretation of an unstated metric into the profile. IonQ's
EULA restricts redistribution, so NoiseVault has no bundled IonQ profiles. Users pull IonQ
profiles onto their own machines.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, Field, TypeAdapter

from .. import __version__
from ..errors import SourceUnavailable, did_you_mean, plural
from ..profile import Profile, iso_z
from . import OFFLINE_HINT, OLDER_HINT, Origin, checked_by, download, read_reply

API = "https://api.ionq.co/v0.4"
_TIMEOUT_S = 30.0
_PAGE = 10  # the largest page the endpoint serves
_MAX_PROBES = 30  # one-record reads per pull before giving up
_SENDER = "IonQ's API"
_TRY_LATER = "try again later"
# IonQ's history holds medians no trapped-ion device produces (1Q 0.69, 2Q 0.75, SPAM 0.73)
# next to normal ones, and chance-level SPAM placeholders (0.5, 0.501) from before SPAM was
# measured. The importer reads medians below these floors or above 1 as corrupt, not as data.
_FLOORS = {"1q": 0.99, "2q": 0.9, "spam": 0.9}
_NO_FIDELITIES = "without 1Q/2Q fidelities"
_IMPLAUSIBLE = "with implausible medians on "
# IonQ native two-qubit gate -> canonical name. IonQ's zz takes any angle, like rzz.
_TWO_QUBIT = {"zz": "rzz", "ms": "ms"}
_FIDELITY_ASSUMPTION = "IonQ does not state the fidelity metric; read as average gate fidelity"


class _Backend(BaseModel):
    backend: str
    qubits: int | None = None
    supported_native_gates: list[str] = []


class _Statistic(BaseModel):
    median: float | None = None
    stderr: float | None = None


class _Fidelities(BaseModel):
    one: _Statistic | None = Field(None, alias="1q")
    two: _Statistic | None = Field(None, alias="2q")
    spam: _Statistic | None = None


class _Timing(BaseModel):
    one: float | None = Field(None, alias="1q")
    two: float | None = Field(None, alias="2q")
    readout: float | None = None
    t1: float | None = None
    t2: float | None = None


class _Record(BaseModel):
    id: str
    date: Annotated[str, AfterValidator(checked_by(datetime.fromisoformat))]
    backend: str
    qubits: int | None = None
    connectivity: list[Annotated[list[int], Field(min_length=2, max_length=2)]] | None = None
    fidelity: _Fidelities | None = None
    timing: _Timing | None = None


class _Characterizations(BaseModel):
    characterizations: list[_Record] | None = None
    pages: int | None = None


_BACKENDS = TypeAdapter(list[_Backend])
_CHARACTERIZATIONS = TypeAdapter(_Characterizations)


def bundled_profiles() -> list[Profile]:
    """None: IonQ's EULA restricts redistribution, so use :func:`pull` instead."""
    return []


def pull(device: str, *, at: str | date | datetime | None = None) -> Profile:
    """The newest usable characterization of ``device`` at or before ``at`` (default: now).

    ``device`` can be ``ionq_forte-1``, ``forte-1`` or ``qpu.forte-1``. ``at`` is a datetime,
    a date or an ISO 8601 string. ``pull`` reads a date or a naive time as UTC. ``pull`` skips
    a record without 1Q and 2Q fidelities (IonQ backfills some) or with implausible ones (see
    :data:`_FLOORS`), and uses the next older record. The provenance notes give the reason for
    each skip.
    """
    backend = backend_name(device)
    listing = _listing(backend)
    record, raw, url, skipped = _newest_usable(backend, at)
    return to_profile(
        record,
        listing,
        source_url=url,
        source_hash="sha256:" + hashlib.sha256(raw).hexdigest(),
        skipped=skipped,
    )


def backend_name(device: str) -> str:
    """``ionq_forte-1``, ``forte-1`` or ``qpu.forte-1`` -> ``qpu.forte-1``."""
    name = device.strip().lower().removeprefix("ionq_").removeprefix("ionq.")
    return name if name.startswith("qpu.") else f"qpu.{name}"


def to_profile(
    record: Mapping[str, Any],
    listing: Mapping[str, Any],
    *,
    source_url: str,
    source_hash: str,
    skipped: Sequence[str] = (),
) -> Profile:
    """A profile from one characterization record and the backend's entry in the listing.

    ``skipped`` holds the reason that the importer skipped each newer record.
    """
    backend = record["backend"]
    name = backend.removeprefix("qpu.")
    origin = Origin(f"{_SENDER} ({source_url})", hint=OLDER_HINT)
    num_qubits = record.get("qubits")
    if num_qubits is None:
        num_qubits = listing.get("qubits")
    if num_qubits is None:
        raise origin.refuse(
            f"record {record['id']} gives no qubit count, and neither does the backend listing"
        )
    fidelity = record.get("fidelity") or {}
    timing = record.get("timing") or {}
    reason = _rejection(record)
    if reason is not None:
        raise SourceUnavailable(
            f"IonQ published {backend} record {record.get('id')} {reason}",
            hint="use a record from another date",
        )
    one, two = _fidelity(fidelity, "1q"), _fidelity(fidelity, "2q")
    assert one is not None and two is not None  # _rejection() checked both
    seconds = {key: _seconds(timing, key, origin) for key in ("1q", "2q", "readout", "t1", "t2")}
    natives = [g for g in listing.get("supported_native_gates", ()) if g in _TWO_QUBIT]
    if not natives:
        raise SourceUnavailable(
            f"IonQ lists no known two-qubit native gate for {backend}"
            f" ({listing.get('supported_native_gates')}). NoiseVault expects one of"
            f" {', '.join(_TWO_QUBIT)}"
        )
    two_qubit = _TWO_QUBIT[natives[0]]
    two_assumption = _FIDELITY_ASSUMPTION
    if two_qubit == "rzz":
        two_assumption += "; IonQ's native zz gate takes any angle, and this error is used for all"
    notes = [
        "IonQ publishes device medians only: every qubit and pair gets the same values",
        "SPAM fidelity combines state preparation and measurement. NoiseVault uses 1 - fidelity"
        " as a symmetric readout error and leaves prep unknown",
        "t1, t2 and gate times are device-level values that IonQ gives with no measurement"
        " statement. The values are the same on every record of a backend, so NoiseVault reads"
        " the values as nominal",
        "NoiseVault drops a stderr of 0, which IonQ reports on every record as a placeholder",
        f"the two-qubit native ({natives[0]} -> {two_qubit}) comes from IonQ's current backend"
        " listing, also for older records",
    ]
    if record.get("qubits") is None:
        notes.append(
            "the record gives no qubit count, so NoiseVault uses the qubit count of the current"
            f" listing ({num_qubits})"
        )
    notes += _skip_notes(skipped)
    spam = _fidelity(fidelity, "spam")
    readout: dict[str, float] | None = None
    if spam is not None:
        readout = {"error": _clean(1 - spam[0]), **_ns("duration_ns", seconds["readout"])}
    elif (median := _median(fidelity, "spam")) is not None:
        reason = (
            "above 1 (corrupt)"
            if median > 1
            else f"below {_FLOORS['spam']} (a chance-level placeholder or corrupt)"
        )
        notes.append(
            f"SPAM fidelity median {median} is {reason}, so NoiseVault leaves readout error unknown"
        )
    idle = {
        f"{key}_us": _clean(seconds[key] * 1e6) for key in ("t1", "t2") if seconds[key] is not None
    }
    return origin.profile(
        {
            "noisevault": "1.0",
            "device": {
                "vendor": "ionq",
                "name": name,
                "technology": "trapped_ion",
                "num_qubits": num_qubits,
                "processor": name.split("-")[0].capitalize(),
                "calibrated_at": record["date"],
            },
            "connectivity": _connectivity(record.get("connectivity"), num_qubits),
            "gates": {
                "rz": {"virtual": True},
                "r": _gate(one, seconds["1q"], _FIDELITY_ASSUMPTION),
                two_qubit: _gate(two, seconds["2q"], two_assumption),
            },
            "readout": readout,
            "idle": idle or None,
            "provenance": {
                "data_kind": "measured",
                "source_kind": "public_api",
                "source": f"IonQ characterization API v0.4, {backend} record {record['id']}",
                "source_url": source_url,
                "license": "IonQ EULA (not an open license)",
                "attribution": "IonQ",
                "redistributable": "no",
                "retrieved_at": datetime.now(UTC),
                "source_hash": source_hash,
                "tool": f"noisevault {__version__}",
                "notes": notes,
            },
        }
    )


def _gate(
    fidelity: tuple[float, float | None], seconds: float | None, assumption: str
) -> dict[str, Any]:
    value, stderr = fidelity
    return {
        "avg_infidelity": _clean(1 - value),
        "stderr": stderr,
        **_ns("duration_ns", seconds),
        "method": "vendor",
        "statistic": "median",
        "assumption": assumption,
    }


def _rejection(record: Mapping[str, Any]) -> str | None:
    """Why ``record`` cannot pin a profile, or None when its 1Q and 2Q medians are usable."""
    fidelity = record.get("fidelity") or {}
    if _median(fidelity, "1q") is None or _median(fidelity, "2q") is None:
        return _NO_FIDELITIES
    one, two = _fidelity(fidelity, "1q"), _fidelity(fidelity, "2q")
    if one is not None and two is not None and one[0] >= two[0]:
        return None
    medians = f"1Q {_median(fidelity, '1q')}, 2Q {_median(fidelity, '2q')}"
    return f"{_IMPLAUSIBLE}{str(record.get('date'))[:10]} ({medians})"


def _median(fidelity: Mapping[str, Any], key: str) -> float | None:
    """The median IonQ gives for ``key``, or None for its null."""
    entry = fidelity.get(key)
    median = entry.get("median") if isinstance(entry, Mapping) else None
    ok = isinstance(median, int | float) and not isinstance(median, bool)
    return float(median) if ok else None


def _fidelity(fidelity: Mapping[str, Any], key: str) -> tuple[float, float | None] | None:
    """(median, stderr) when the median is plausible data, or None for nulls and placeholders."""
    median = _median(fidelity, key)
    if median is None or not _FLOORS[key] <= median <= 1:
        return None
    stderr = fidelity[key].get("stderr")
    return median, float(stderr) if stderr else None


def _skip_notes(skipped: Sequence[str]) -> list[str]:
    missing = sum(reason == _NO_FIDELITIES for reason in skipped)
    implausible = [reason for reason in skipped if reason != _NO_FIDELITIES]
    notes = [f"skipped {plural(missing, 'newer record')} {_NO_FIDELITIES}"] if missing else []
    if implausible:
        dated = [reason.removeprefix(_IMPLAUSIBLE) for reason in implausible]
        shown = "; ".join(dated[:3]) + ("; ..." if len(dated) > 3 else "")
        notes.append(
            f"skipped {plural(len(implausible), 'newer record')} with implausible fidelity medians"
            f" (1Q below {_FLOORS['1q']}, 2Q below {_FLOORS['2q']}, a median above 1, or a 1Q error"
            f" above the 2Q error), read as corrupt: {shown}"
        )
    return notes


def _seconds(timing: Mapping[str, Any], key: str, origin: Origin) -> float | None:
    """``timing[key]``, or None for its null; ``origin`` refuses a time that is not positive."""
    seconds = timing.get(key)
    if seconds is not None and not (math.isfinite(seconds) and seconds > 0):
        raise origin.refuse(f"timing.{key} is {seconds!r}, not a positive number of seconds")
    return seconds


def _ns(key: str, seconds: float | None) -> dict[str, float]:
    return {} if seconds is None else {key: _clean(seconds * 1e9)}


def _clean(value: float) -> float:
    """Drop binary round-off from decimal inputs: 1 - 0.9998 -> 0.0002, 0.00013 s -> 130000 ns."""
    return float(f"{value:.12g}")


def _connectivity(pairs: Any, num_qubits: int) -> str | dict[str, Any]:
    if not pairs:
        return "all_to_all"
    edges = {tuple(sorted(pair)) for pair in pairs}
    if edges == set(itertools.combinations(range(num_qubits), 2)):
        return "all_to_all"
    return {"edges": sorted(edges), "directed": False}


# network ------------------------------------------------------------------------------------


def _listing(backend: str) -> Mapping[str, Any]:
    url = f"{API}/backends"
    entries = read_reply(_get(url), url, _BACKENDS, sender=_SENDER, hint=_TRY_LATER)
    for entry in entries:
        if entry["backend"] == backend:
            return entry
    qpus = sorted(e["backend"] for e in entries if e["backend"].startswith("qpu."))
    if not qpus:
        raise SourceUnavailable(
            f"IonQ has no backend {backend!r}. The IonQ listing ({url}) names no QPU",
            hint=_TRY_LATER,
        )
    raise SourceUnavailable(
        f"IonQ has no backend {backend!r}: {did_you_mean(backend, qpus)}IonQ lists"
        f" {', '.join(qpus)}"
    )


def _newest_usable(
    backend: str, at: str | date | datetime | None
) -> tuple[dict[str, Any], bytes, str, list[str]]:
    """The newest record at or before ``at`` with plausible 1Q and 2Q fidelities of its own.

    Returns the record, the bytes and URL of its one-record page, and the :func:`_rejection` of
    each newer record that the search skipped. A page of several records fills the null fields of
    a record from the next newer record on the page (seen on qpu.aria-1). So the search judges
    each record on the page that holds only that record. The profile cites that page.
    """
    end = None if at is None else iso_z(_utc(at))
    skipped: list[str] = []
    probes = 0
    page = 1
    while True:
        page_url = _page_url(backend, limit=_PAGE, end=end, page=page)
        body = read_reply(
            _get(page_url), page_url, _CHARACTERIZATIONS, sender=_SENDER, hint=_TRY_LATER
        )
        records = body.get("characterizations") or []
        for listed in records:
            # a record that fails on the filled page also fails alone, so skip it without a probe
            reason = _rejection(listed)
            if reason is None and probes < _MAX_PROBES:
                probes += 1
                url = _page_url(backend, limit=1, end=listed["date"])
                raw = _get(url)
                body_alone = read_reply(
                    raw, url, _CHARACTERIZATIONS, sender=_SENDER, hint=_TRY_LATER
                )
                alone = (body_alone.get("characterizations") or [{}])[0]
                reason = _rejection(alone) if alone.get("id") == listed["id"] else _NO_FIDELITIES
                if reason is None:
                    return alone, raw, url, skipped
            skipped.append(reason or _NO_FIDELITIES)
        if probes >= _MAX_PROBES:
            raise SourceUnavailable(
                f"none of the {len(skipped)} newest {backend} characterizations{_before(at)} has"
                " plausible 1Q and 2Q fidelities of its own",
                hint="pass an earlier at= to search older records",
            )
        if not records or page >= int(body.get("pages") or 0):
            raise SourceUnavailable(
                f"IonQ publishes no {backend} characterization{_before(at)} with plausible 1Q and"
                f" 2Q fidelities of its own ({plural(len(skipped), 'record')} checked)",
                hint="use another backend" if at is None else "pass a later at= or none",
            )
        page += 1


def _before(at: str | date | datetime | None) -> str:
    return "" if at is None else f" at or before {at}"


def _page_url(backend: str, *, limit: int, end: str | None = None, page: int = 1) -> str:
    query = {
        "limit": limit,
        **({"end": end} if end else {}),
        **({"page": page} if page > 1 else {}),
    }
    return f"{API}/backends/{backend}/characterizations?{urllib.parse.urlencode(query)}"


def _utc(at: str | date | datetime) -> datetime:
    if isinstance(at, str):
        at = datetime.fromisoformat(at.strip().replace("Z", "+00:00"))
    if not isinstance(at, datetime):
        at = datetime(at.year, at.month, at.day)
    return at.replace(tzinfo=UTC) if at.tzinfo is None else at.astimezone(UTC)


def _get(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": f"noisevault/{__version__}"}
    )
    try:
        return download(request, _TIMEOUT_S)
    except urllib.error.HTTPError as exc:
        raise SourceUnavailable(
            f"IonQ's API answered HTTP {exc.code} for {url}", hint=_TRY_LATER
        ) from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise SourceUnavailable(
            f"could not reach IonQ's API ({reason})", hint=OFFLINE_HINT
        ) from None
