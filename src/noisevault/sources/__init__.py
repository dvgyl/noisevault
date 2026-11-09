"""Calibration sources, each imported only when used."""

from __future__ import annotations

import csv
import io
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from email.message import Message
from typing import IO, Any

from pydantic import TypeAdapter, ValidationError

from ..errors import (
    DuplicateKeyError,
    SourceDataError,
    SourceUnavailable,
    json_path,
    mib,
    parse_json,
    qubit_loci,
    unreadable,
)
from ..profile import Profile

OFFLINE_HINT = (
    "check the network connection, or run nv list to see every profile you can load offline"
)
OLDER_HINT = "pass an earlier at= to use an older calibration"
_TRY_LATER = "try again later"

DOWNLOAD_LIMIT = 64 * 2**20
"""Most bytes that one download from a vendor can have."""


class _SameHostHttps(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: Message,
        newurl: str,
    ) -> urllib.request.Request | None:
        old, new = urllib.parse.urlsplit(req.full_url), urllib.parse.urlsplit(newurl)
        if new.scheme != "https" or (new.hostname, new.port) != (old.hostname, old.port):
            fp.close()
            raise SourceUnavailable(
                f"refused a redirect from {req.full_url} to {newurl}, which is not https on"
                f" {old.hostname}",
                hint=_TRY_LATER,
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_SameHostHttps)


def download(request: urllib.request.Request, timeout: float) -> bytes:
    """The body of ``request``. Raises what ``urlopen`` raises, and SourceUnavailable for a body
    over DOWNLOAD_LIMIT or for a redirect to plain http or to another host.
    """
    body = bytearray()
    with _OPENER.open(request, timeout=timeout) as response:
        while chunk := response.read(2**16):
            body += chunk
            if len(body) > DOWNLOAD_LIMIT:
                raise SourceUnavailable(
                    f"{request.full_url} sent more than {mib(DOWNLOAD_LIMIT)}, the download limit",
                    hint=_TRY_LATER,
                )
    return bytes(body)


@dataclass(frozen=True)
class Origin:
    """How the errors about a source's data name the source, and the next step they give."""

    name: str
    hint: str | None = None

    def refuse(self, problem: str) -> SourceDataError:
        return SourceDataError(f"{self.name}: {problem}", hint=self.hint)

    def profile(self, data: Mapping[str, Any]) -> Profile:
        """The profile of ``data``. A value that the format refuses raises a SourceDataError that
        names the value.
        """
        try:
            return Profile.model_validate(data)
        except ValidationError as exc:
            raise self.refuse(_first_problem(exc, data)) from None


def _first_problem(exc: ValidationError, data: Mapping[str, Any]) -> str:
    """The problem of the first field, from its deepest error. For a union, the deepest error is in
    the member that the value is for.
    """
    errors = exc.errors()
    field = errors[0]["loc"][:1]
    error = max((e for e in errors if e["loc"][:1] == field), key=lambda e: len(e["loc"]))
    message = error["msg"].removeprefix("Value error, ").split("\n")[0]
    if error["type"] != "value_error" and isinstance(error["input"], int | float | str):
        message += f", got {error['input']!r}"
    where = _where(error["loc"], data)
    return f"{where}: {message}" if where else message


_RECORD_NAMES: Mapping[str, Callable[[Mapping[str, Any]], str]] = {
    "qubits": lambda record: f"qubit {record['index']}",
    "calibrations": lambda record: f"{record['gate']} on {qubit_loci(record['qubits'])}",
}


def _where(loc: Sequence[str | int], data: Mapping[str, Any]) -> str:
    """Where ``loc`` points in ``data``, such as ``readout.error of qubit 3``. A key that ``data``
    does not have is pydantic's name for a union member, and the result does not include it.
    """
    keys: list[str | int] = []
    node: Any = data
    for key in loc:
        if (isinstance(node, Mapping) and key in node) or (
            isinstance(node, list | tuple) and isinstance(key, int)
        ):
            keys.append(key)
            node = node[key]
    if len(keys) > 1 and keys[0] in _RECORD_NAMES:
        record = _RECORD_NAMES[keys[0]](data[keys[0]][keys[1]])
        rest = json_path(keys[2:])
        return f"{rest} of {record}" if rest else record
    return json_path(keys)


def checked_by(check: Callable[[str], object]) -> Callable[[str], str]:
    def validate(value: str) -> str:
        check(value)
        return value

    return validate


def read_reply(raw: bytes, url: str, shape: TypeAdapter[Any], *, sender: str, hint: str) -> Any:
    """``raw`` parsed as JSON of ``shape``. If ``raw`` is not, raise SourceUnavailable that names
    ``url``.
    """
    try:
        data = parse_json(raw)
    except DuplicateKeyError as exc:
        raise SourceUnavailable(
            f"{sender} answered {url} with JSON that has the key {json_path(exc.path)} twice",
            hint=hint,
        ) from None
    except ValueError:
        raise SourceUnavailable(
            f"{sender} answered {url} with something other than JSON", hint=hint
        ) from None
    try:
        shape.validate_python(data, strict=True)
    except ValidationError as exc:
        loc = exc.errors()[0]["loc"]
        where = f" at {json_path(loc)}" if loc else ""
        raise SourceUnavailable(
            f"{sender} answered {url} with JSON of the wrong shape{where}", hint=hint
        ) from None
    return data


def source_json(raw: bytes, source: str, hint: str | None = None) -> Any:
    try:
        return parse_json(raw)
    except ValueError as exc:
        raise SourceDataError(unreadable(source, exc), hint=hint) from None


def source_text(raw: bytes, source: str, hint: str | None = None) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SourceDataError(unreadable(source, exc), hint=hint) from None


def csv_by_line(text: str, source: str) -> list[tuple[int, list[str]]]:
    lines = []
    for number, line in enumerate(io.StringIO(text, newline=""), start=1):
        try:
            [cells] = csv.reader([line], strict=True)
        except csv.Error as exc:
            raise SourceDataError(f"{source} line {number} is not valid CSV: {exc}") from None
        lines.append((number, cells))
    return lines
