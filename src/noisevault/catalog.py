"""Where profiles live (the local vault, then the bundled set) and how refs resolve.

The vault is ``$NOISEVAULT_HOME/profiles``, by default ``~/.noisevault/profiles``. Loading never
uses the network. Only :func:`pull` uses the network.
"""

from __future__ import annotations

import difflib
import hashlib
import importlib
import json
import os
import re
import shlex
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from stat import S_ISREG
from typing import Any, Literal, NamedTuple

from pydantic import ValidationError

from .errors import (
    AmbiguousRef,
    FingerprintMismatch,
    NoiseVaultError,
    NoiseVaultWarning,
    ProfileNotFound,
    SourceUnavailable,
    did_you_mean,
    parse_json,
)
from .profile import (
    Held,
    Profile,
    Ref,
    _FileInTheWay,
    canonical_json,
    drop,
    exact_ref,
    file_bytes,
    iso_z,
    load_bytes,
    load_file,
    parse_ref,
    publish,
    take,
    write_atomically,
    write_new,
)
from .sources.qiskit_backend import as_utc

_PULL_SOURCES = {
    "ibm": "noisevault.sources.ibm_public",
    "ibm-account": "noisevault.sources.ibm_account",
    "ionq": "noisevault.sources.ionq",
}

# The ProfileInfo fields that an index entry carries. The listing ignores all other fields.
_ENTRY_TYPES: dict[str, Any] = {
    "id": str,
    "calibrated_at": str | None,
    "technology": str,
    "vendor": str | None,
    "num_qubits": int,
    "fingerprint": str,
    "data_kind": str,
    "license": str | None,
    "processor": str | None,
    "source_kind": str | None,
    "redistributable": str,
}


class _BadArgument(NoiseVaultError, ValueError): ...


@dataclass(frozen=True)
class ProfileInfo:
    id: str
    calibrated_at: datetime | None
    technology: str
    vendor: str | None
    num_qubits: int
    fingerprint: str
    location: Literal["vault", "bundled"]
    path: Path | Traversable
    data_kind: str = "unknown"
    license: str | None = None
    processor: str | None = None
    source_kind: str | None = None
    redistributable: str = "unknown"

    @property
    def ref(self) -> str:
        return exact_ref(self.id, self.calibrated_at)

    def load(self) -> Profile:
        return load_bytes(self.path.read_bytes())

    @classmethod
    def of(
        cls, profile: Profile, location: Literal["vault", "bundled"], path: Path | Traversable
    ) -> ProfileInfo:
        return cls.from_entry(index_entry(profile), location, path)

    @classmethod
    def from_entry(
        cls, entry: dict[str, Any], location: Literal["vault", "bundled"], path: Path | Traversable
    ) -> ProfileInfo:
        """From a line of a catalog index (see :func:`index_entry`). Raises ValueError if bad."""
        fields = {
            "calibrated_at": None,
            "vendor": None,
            "data_kind": "unknown",
            "license": None,
            "processor": None,
            "source_kind": None,
        }
        fields |= {key: entry[key] for key in _ENTRY_TYPES if key in entry}
        for key, kind in _ENTRY_TYPES.items():
            value = fields.get(key)
            if isinstance(value, bool) or not isinstance(value, kind):
                raise ValueError(f"index entry {key} is {value!r}")
        stamp = fields.pop("calibrated_at")
        when = datetime.fromisoformat(stamp) if stamp else None
        if when is not None:
            if when.tzinfo is None:
                raise ValueError(f"index entry time {stamp} has no time zone")
            when = when.astimezone(UTC)
        if not re.fullmatch(r"[0-9a-f]{64}", fields["fingerprint"]):
            raise ValueError(f"index entry fingerprint {fields['fingerprint']!r} is not sha256 hex")
        return cls(**fields, calibrated_at=when, location=location, path=path)


def index_entry(profile: Profile) -> dict[str, Any]:
    """What a catalog index records about a profile, so listing never parses the file."""
    dev, prov = profile.device, profile.provenance
    return {
        "id": profile.id,
        "date": dev.calibrated_at.date().isoformat() if dev.calibrated_at else None,
        "calibrated_at": iso_z(dev.calibrated_at) if dev.calibrated_at else None,
        "vendor": dev.vendor,
        "technology": dev.technology,
        "num_qubits": dev.num_qubits,
        "processor": dev.processor,
        "data_kind": prov.data_kind,
        "source_kind": prov.source_kind,
        "license": prov.license,
        "redistributable": prov.redistributable,
        "fingerprint": profile.fingerprint,
    }


def vault_dir() -> Path:
    home = os.environ.get("NOISEVAULT_HOME")
    return (Path(home) if home else Path.home() / ".noisevault") / "profiles"


def bundled_dir() -> Traversable:
    return files("noisevault") / "data" / "profiles"


def vault_path(profile: Profile) -> Path:
    """Default vault file for a profile: ``<id>@<UTC timestamp>.json.gz``.

    Microseconds appear only when nonzero, so whole-second names match older vault files.
    """
    when = profile.device.calibrated_at
    if when is None:
        stamp = "undated"
    else:
        stamp = when.strftime("%Y-%m-%dT%H%M%S.%fZ" if when.microsecond else "%Y-%m-%dT%H%M%SZ")
    return vault_dir() / f"{profile.id}@{stamp}.json.gz"


def bundled_profiles() -> list[ProfileInfo]:
    index = parse_json((bundled_dir() / "index.json").read_bytes())
    return [
        ProfileInfo.from_entry(entry, "bundled", bundled_dir() / entry["file"])
        for entry in index["profiles"]
    ]


_VAULT_INDEX = ".index.json"
_INDEX_VERSION = 1


def vault_profiles(*, reread: bool = False) -> list[ProfileInfo]:
    folder = vault_dir()
    cached = {} if reread else _read_vault_index(folder)
    index: dict[str, dict[str, Any]] = {}
    out = []
    for path in sorted(folder.glob("*.json*")):
        if path.name.startswith("."):  # the index cache and in-flight temporary files
            continue
        try:
            entry, info = _vault_entry(path, cached.get(path.name))
        except Exception as exc:  # one damaged or odd entry must not hide the rest
            warnings.warn(_skipped(path, exc), NoiseVaultWarning, stacklevel=2)
            continue
        index[path.name] = entry
        out.append(info)
    if index != cached:
        _write_vault_index(folder, index)
    return out


def _vault_entry(path: Path, cached: Any) -> tuple[dict[str, Any], ProfileInfo]:
    """The index entry and listing of one vault file, from the cache while the file is unchanged."""
    stat = path.stat()
    if not S_ISREG(stat.st_mode):  # opening a pipe would block the listing
        raise OSError("not a regular file")
    # chmod and chown move only the change time, and either can make the file unreadable.
    signature = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    if isinstance(cached, dict) and cached.get("signature") == signature:
        try:
            return cached, ProfileInfo.from_entry(cached, "vault", path)
        except ValueError:
            pass
    entry = {"signature": signature, **index_entry(load_file(path))}
    return entry, ProfileInfo.from_entry(entry, "vault", path)


def shell_command(words: Sequence[str], operands: Sequence[str]) -> str:
    end = ["--"] if any(operand.startswith("-") for operand in operands) else []
    return shlex.join([*words, *end, *operands])


def _skipped(path: Path, exc: Exception) -> str:
    """One line that names a vault entry that the listing skips, and why."""
    try:
        dangling = path.readlink() if path.is_symlink() and not path.exists() else None
    except OSError:  # an unreachable target fails these checks as it failed the read
        dangling = None
    if dangling is not None:
        why = f"it links to {dangling}, which does not exist. Remove the link"
    elif isinstance(exc, ValidationError):
        n = exc.error_count()
        why = (
            f"not a valid profile ({n} problem{'s' * (n != 1)})."
            f" Run {shell_command(['nv', 'validate'], [str(path)])}"
        )
    elif isinstance(exc, OSError):
        why = exc.strerror or str(exc)
    else:
        why = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    return f"skipped {path}: {why}"


def _read_vault_index(folder: Path) -> dict[str, dict[str, Any]]:
    path = folder / _VAULT_INDEX
    try:
        if not S_ISREG(path.stat().st_mode):
            return {}
        data = parse_json(path.read_bytes())
        if not isinstance(data, dict) or data.get("version") != _INDEX_VERSION:
            return {}
        entries = data.get("entries")
        if not isinstance(entries, dict) or data.get("digest") != _digest(entries):
            return {}
    except (OSError, ValueError):
        return {}
    return entries


def _write_vault_index(folder: Path, entries: dict[str, dict[str, Any]]) -> None:
    """Best effort and atomic: a read-only vault or a concurrent writer only loses the cache."""
    data = {"version": _INDEX_VERSION, "digest": _digest(entries), "entries": entries}
    text = json.dumps(data, sort_keys=True)
    try:
        write_atomically(folder / _VAULT_INDEX, text.encode("utf-8"))
    except OSError:
        pass


def _digest(entries: dict[str, dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_json(entries).encode("utf-8")).hexdigest()


def profiles(*, technology: str | None = None, vendor: str | None = None) -> list[ProfileInfo]:
    """Every known profile (a vault copy hides an identical bundled one), by id then date."""
    found = [
        info
        for info in _known()
        if (technology is None or info.technology == technology)
        and (vendor is None or info.vendor == vendor)
    ]
    return sorted(found, key=lambda i: (i.id, _epoch(i.calibrated_at), i.location))


def resolve(ref: str | Ref, *, expect: str | None = None) -> ProfileInfo:
    """The single catalog profile that a ``Ref`` names. Raises ProfileNotFound or AmbiguousRef.

    ``expect`` keeps only the profiles with that fingerprint. Of profiles with the same id and
    calibration time, a vault copy shadows a bundled one.
    """
    if isinstance(ref, str):
        parsed = parse_ref_preferring_id(ref)
        if isinstance(parsed, Path):
            raise ValueError(f"{ref!r} is a path, not a catalog ref")
        ref = parsed
    return _pick(ref, _known(), expect)


def _known(*, reread: bool = False) -> list[ProfileInfo]:
    return _dedupe(vault_profiles(reread=reread) + bundled_profiles())


def _pick(ref: Ref, known: list[ProfileInfo], expect: str | None) -> ProfileInfo:
    candidates = [info for info in known if info.id == ref.id]
    if not candidates:
        close = difflib.get_close_matches(ref.id, sorted({i.id for i in known}), n=3)
        if close:
            guesses = ", ".join(f"'{guess}'" for guess in close)
            raise ProfileNotFound(f"no profile with id {ref.id!r}; did you mean {guesses}?")
        raise ProfileNotFound(
            f"no profile with id {ref.id!r}",
            hint="run nv list to see every profile you can load offline",
        )
    if ref.timestamp is None and ref.date is None:
        newest = max(_epoch(i.calibrated_at) for i in candidates)
        matches = [i for i in candidates if _epoch(i.calibrated_at) == newest]
    else:
        matches = [i for i in candidates if _ref_time_matches(ref, i.calibrated_at)]
    if not matches:
        raise _no_calibration(ref, candidates)
    if expect is not None:
        want = _expect_prefix(expect)
        pinned = [i for i in matches if i.fingerprint.startswith(want)]
        if not pinned:
            raise _mismatch(_loaded(ref, _unshadowed(matches)), expect, ref.id, known)
        matches = pinned
    matches = _unshadowed(matches)
    if len(matches) > 1:
        listing = "; ".join(f"{i.ref} ({i.location}, nv:{i.fingerprint[:12]})" for i in matches)
        if len({i.calibrated_at for i in matches}) > 1:
            raise AmbiguousRef(
                f"{ref.id} matches {len(matches)} profiles: {listing}", hint="use a full timestamp"
            )
        files = ", ".join(str(i.path) for i in matches)
        raise AmbiguousRef(
            f"{ref.id} matches {len(matches)} profiles that share a calibration time: {listing}",
            hint=f"pass expect='nv:...' or load one of their files: {files}",
        )
    return matches[0]


def parse_ref_preferring_id(ref: str | Path) -> Path | Ref:
    parsed = parse_ref(ref)
    text = ref.strip() if isinstance(ref, str) else ""
    if isinstance(parsed, Path) and parsed.is_dir() and text == parsed.name and "@" not in text:
        return Ref(text.lower())
    return parsed


def load(ref: str | Path, *, expect: str | None = None) -> Profile:
    """Load a profile by path, ``id``, ``id@YYYY-MM-DD`` or ``id@<timestamp>``, offline.

    ``expect`` pins the fingerprint (full hex, ``sha256:<hex>`` or ``nv:<12 hex>``). A different
    profile raises FingerprintMismatch. Loading a catalog ref checks the profile that it reads
    against the calibration that the ref selects and against the pin. Another process can save
    over the file, or the index can describe the file incorrectly. If the file does not match,
    the load reads every vault file again and resolves the ref again.
    """
    named = parse_ref_preferring_id(ref)
    if isinstance(named, Path):
        if not named.exists():
            raise ProfileNotFound(f"no file {named}")
        path, profile = named, load_file(named)
    else:
        info = resolve(named, expect=expect)
        profile = info.load()
        if not (_holds(info, profile) and _pinned(profile, expect)):
            info = _pick(named, _known(reread=True), expect)
            profile = info.load()
            if not _holds(info, profile):
                raise _changed(named, profile, info)
        path = info.path
    if expect is not None:
        _check_expect(profile, expect, path)
    return profile


def _ref_time_matches(ref: Ref, when: datetime | None) -> bool:
    if ref.timestamp is not None:
        return when == ref.timestamp
    if ref.date is not None:
        return when is not None and when.date() == ref.date
    return True


def _holds(info: ProfileInfo, profile: Profile) -> bool:
    return (profile.id, profile.device.calibrated_at) == (info.id, info.calibrated_at)


def _pinned(profile: Profile, expect: str | None) -> bool:
    return expect is None or profile.fingerprint.startswith(_expect_prefix(expect))


def _changed(ref: Ref, profile: Profile, info: ProfileInfo) -> ProfileNotFound:
    held = exact_ref(profile.id, profile.device.calibrated_at)
    return ProfileNotFound(
        f"{info.path} holds {held}, not {info.ref}, because the file changed during this load",
        hint=f"load {_said(ref)} again",
    )


class Pulled(NamedTuple):
    profile: Profile
    path: Path
    written: bool  # False when the vault already held this calibration


def pull(
    device: str,
    *,
    at: str | date | datetime | None = None,
    source: str | None = None,
    output: str | Path | None = None,
) -> Profile:
    """Download a calibration from a live source and save it (to the vault, or to ``output``).

    Pulling a calibration that the vault already holds (same id and fingerprint) writes nothing,
    so repeated pulls leave one file per calibration. A pull that imports the same calibration
    differently (for example, after a NoiseVault upgrade) replaces the older file and warns. A
    pull to the vault never replaces a file of another calibration. A pull to ``output``
    replaces the file that you name, whichever calibration the file holds.
    """
    return pull_and_save(device, at=at, source=source, output=output).profile


def pull_and_save(
    device: str,
    *,
    at: str | date | datetime | None = None,
    source: str | None = None,
    output: str | Path | None = None,
) -> Pulled:
    source = source.strip().lower() if source else _default_source(device)
    if source not in _PULL_SOURCES:
        raise _unknown_source(source)
    when = None if at is None else as_utc(at)
    module = importlib.import_module(_PULL_SOURCES[source])
    profile = module.pull(device, at=when)
    if output is not None:
        return Pulled(profile, profile.save(output), written=True)
    path = _earlier_path(profile) or vault_path(profile)
    return Pulled(profile, path, _save_in_vault(profile, path))


def _earlier_path(profile: Profile) -> Path | None:
    same_id = [i for i in vault_profiles() if i.id == profile.id]
    held = next((i for i in same_id if i.fingerprint == profile.fingerprint), None)
    old = next((i for i in same_id if i.calibrated_at == profile.device.calibrated_at), None)
    earlier = held or old
    return None if earlier is None else Path(str(earlier.path))


def _save_in_vault(profile: Profile, path: Path) -> bool:
    data = file_bytes(profile, path)
    moved: list[Held] = []
    replaced: list[str] = []
    try:
        written: bool | None = None
        while written is None:
            written = _try_to_save(profile, path, data, moved, replaced)
    except BaseException as exc:
        error = _roll_back(moved, path, exc)
        if error is None:
            raise
        raise error from exc
    for held in moved:
        drop(held)
    for fingerprint in replaced:
        warnings.warn(
            f"replaced {path.name} (nv:{fingerprint[:12]}) with this pull"
            f" ({profile.short_fingerprint}), which imports the same calibration differently."
            " Update any expect= pins",
            NoiseVaultWarning,
            stacklevel=4,
        )
    return written


def _try_to_save(
    profile: Profile, path: Path, data: bytes, moved: list[Held], replaced: list[str]
) -> bool | None:
    there = _same_calibration_at(path, profile)
    if there is None:
        return True if write_new(path, data) else None
    raw, fingerprint = there
    if fingerprint == profile.fingerprint:
        return False
    taken = take(path)
    if taken is None:
        return None
    moved.append(taken)
    if taken.data == raw:
        replaced.append(fingerprint)
    elif _put_back(taken, path):
        moved.pop()
    else:
        raise _FileInTheWay(f"{path} changed during the pull")
    return None


def _roll_back(moved: list[Held], path: Path, exc: BaseException) -> NoiseVaultError | None:
    kept = [held.path.name for held in moved if not _put_back(held, path)]
    if not kept or not isinstance(exc, Exception):
        return None
    names = ", ".join(kept)
    reason = exc.message if isinstance(exc, NoiseVaultError) else str(exc)
    kind = _FileInTheWay if isinstance(exc, FileExistsError) else _NotSaved
    return kind(
        f"{reason}. The pull saved nothing and moved the file that was there to {names}",
        hint=f"move {names} out of {path.parent}, then pull again",
    )


class _NotSaved(NoiseVaultError, OSError): ...


def _put_back(held: Held, path: Path) -> bool:
    try:
        if not publish(held, path):
            return False
    except OSError:
        return False
    drop(held)
    return True


def _same_calibration_at(path: Path, profile: Profile) -> tuple[bytes, str] | None:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return None
    try:
        if not S_ISREG(path.stat().st_mode):
            raise OSError("not a regular file")
        raw = path.read_bytes()
        there = load_bytes(raw)
    except Exception:
        raise _FileInTheWay(
            f"{path} exists but is not readable",
            hint=f"make the file readable or move the file out of {path.parent}, then pull again",
        ) from None
    if (there.id, there.device.calibrated_at) != (profile.id, profile.device.calibrated_at):
        raise _FileInTheWay(
            f"{path} already holds {exact_ref(there.id, there.device.calibrated_at)}, another"
            " calibration",
            hint=f"move that file out of {path.parent} and pull again",
        )
    return raw, there.fingerprint


_DEFAULT_SOURCES = {"ibm_": "ibm", "ionq": "ionq"}  # id prefix -> source that pulls it


def _pull_source(device: str) -> str | None:
    return next((s for p, s in _DEFAULT_SOURCES.items() if device.lower().startswith(p)), None)


def _default_source(device: str) -> str:
    source = _pull_source(device)
    if source is not None:
        return source
    if any(i.id == device for i in bundled_profiles()):
        hint = f"{device} is bundled, so nv.load({device!r}) loads it offline"
    else:
        hint = (
            "nv pull takes IBM devices (ibm_fez) and IonQ devices (ionq_forte-1), and nv list"
            " shows every profile you can load offline"
        )
    raise SourceUnavailable(f"no source pulls {device!r}", hint=hint)


def _unknown_source(source: str) -> _BadArgument:
    choices = ", ".join(_PULL_SOURCES)
    guess = did_you_mean(source, _PULL_SOURCES)
    offline = sorted({i.vendor for i in bundled_profiles() if i.vendor} - _PULL_SOURCES.keys())
    vendor = next((v for v in offline if did_you_mean(source, [v])), None)
    if vendor and not guess:
        return _BadArgument(
            f"unknown source {source!r}. No source serves {vendor} devices",
            hint=f"run nv list --vendor {vendor} to see the {vendor} profiles you can load offline",
        )
    return _BadArgument(f"unknown source {source!r}; {guess}choose one of {choices}")


def _expect_prefix(expect: str) -> str:
    """The fingerprint hex an ``expect`` value pins: 64 digits, or 12 after ``nv:``."""
    want = expect.strip().lower().removeprefix("sha256:")
    short = want.startswith("nv:")
    want = want.removeprefix("nv:")
    if not re.fullmatch(r"[0-9a-f]{12}" if short else r"[0-9a-f]{64}", want):
        raise _BadArgument(
            f"expect={expect!r} is not a fingerprint",
            hint="pass a full sha256 fingerprint or nv:<12 hex>",
        )
    return want


def _check_expect(profile: Profile, expect: str, path: Path | Traversable) -> None:
    if not _pinned(profile, expect):
        held = exact_ref(profile.id, profile.device.calibrated_at)
        loaded = f"{path} holds {held} ({profile.short_fingerprint})"
        raise _mismatch(loaded, expect, profile.id, profiles())


def _said(ref: Ref) -> str:
    return f"{ref.id}@{ref.date}" if ref.date else exact_ref(ref.id, ref.timestamp)


def _loaded(ref: Ref, held: list[ProfileInfo]) -> str:
    said = _said(ref)
    newest = said if ref.date or ref.timestamp else held[0].ref
    found = " or ".join(f"nv:{i.fingerprint[:12]}" for i in held)
    return f"{said} is {found}" if newest == said else f"{said} loads {newest} ({found})"


def _mismatch(
    loaded: str, expect: str, device: str, known: list[ProfileInfo]
) -> FingerprintMismatch:
    want = _expect_prefix(expect)
    missed = f"{loaded}, not the expected {expect}"
    match = next((i for i in known if i.fingerprint.startswith(want)), None)
    if match is not None:
        where = match.ref if match.calibrated_at else str(match.path)
        call = f"nv.load({where!r}, expect={expect!r})"
        return FingerprintMismatch(missed, hint=f"{call} loads the profile with that fingerprint")
    hint = "ask whoever pinned that fingerprint for the profile file"
    if _pull_source(device):
        hint += (
            ". If the source still serves that calibration, run"
            f" nv pull {device} --at <a time it was in effect>"
        )
    return FingerprintMismatch(f"{missed}, and no profile you have has that fingerprint", hint=hint)


def _no_calibration(ref: Ref, candidates: list[ProfileInfo]) -> ProfileNotFound:
    have = ", ".join(sorted({i.ref for i in candidates}))
    if ref.timestamp is not None:
        return ProfileNotFound(
            f"no {ref.id} profile calibrated at {iso_z(ref.timestamp)}. You have {have}"
        )
    fetch = (
        f"run nv pull {ref.id} --at {ref.date}T23:59:59Z to download the calibration in effect at"
        " the end of that day. Then load the ref that nv pull prints"
    )
    return ProfileNotFound(
        f"no {ref.id} profile calibrated on {ref.date} UTC. You have {have}",
        hint=fetch if _pull_source(ref.id) else None,
    )


def _unshadowed(infos: list[ProfileInfo]) -> list[ProfileInfo]:
    shadowed = {(i.id, _epoch(i.calibrated_at)) for i in infos if i.location == "vault"}
    return [
        i for i in infos if i.location == "vault" or (i.id, _epoch(i.calibrated_at)) not in shadowed
    ]


def _dedupe(infos: list[ProfileInfo]) -> list[ProfileInfo]:
    """Keep the first of profiles with the same id, time and fingerprint (vault wins)."""
    seen: set[tuple[str, float, str]] = set()
    out = []
    for info in infos:
        key = (info.id, _epoch(info.calibrated_at), info.fingerprint)
        if key not in seen:
            seen.add(key)
            out.append(info)
    return out


def _epoch(when: datetime | None) -> float:
    return when.timestamp() if when else float("-inf")
