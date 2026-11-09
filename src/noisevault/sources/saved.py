"""Read a vendor payload that the user saved, from a JSON file or from the parsed dict."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from ..errors import SourceDataError, json_path
from . import Origin, source_json


@dataclass(frozen=True)
class Saved:
    """A saved payload: its data, where it came from and the SHA-256 of what was read."""

    data: Any
    source: str
    hashed: str
    sha256: str
    stem: str | None

    @property
    def origin(self) -> Origin:
        return Origin(self.source, hint=f"correct that value in {self.source}")


def read_saved(path_or_dict: str | Path | Mapping[str, Any], *, save: str) -> Saved:
    """``path_or_dict`` as JSON data. ``save`` says how to save the payload, for errors."""
    if isinstance(path_or_dict, Mapping):
        data = dict(path_or_dict)
        source = "the dict passed in"
        problem = _key_problem(data)
        if problem is not None:
            raise SourceDataError(f"{source}: {problem}", hint=save)
        try:
            raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        except (TypeError, ValueError) as exc:
            raise SourceDataError(f"{source} is not JSON data ({exc})", hint=save) from None
        hashed, stem = "the canonical JSON of the dict passed in", None
    else:
        path = Path(path_or_dict)
        raw = path.read_bytes()
        source = path.name
        data = source_json(raw, source, save)
        hashed, stem = f"the bytes of {path.name}", path.stem
    return Saved(data, source, hashed, "sha256:" + hashlib.sha256(raw).hexdigest(), stem)


def shape(saved: Saved, model: type[BaseModel], *, save: str) -> None:
    """Raise a SourceDataError that names the first value of ``saved`` that ``model`` refuses."""
    try:
        model.model_validate(saved.data)
    except ValidationError as exc:
        error = exc.errors()[0]
        where = json_path(error["loc"]) or "the top level"
        if error["type"] == "missing":
            *parents, last = error["loc"]
            problem = f"{json_path(parents) or 'the top level'} has no {last!r}"
        elif error["type"] in ("dict_type", "model_type"):
            problem = f"{where} is not a JSON object"
        elif error["type"] == "list_type":
            problem = f"{where} is not a JSON array"
        else:
            message = error["msg"].removeprefix("Value error, ")
            problem = f"{where} is {error['input']!r}: {message[:1].lower()}{message[1:]}"
        raise SourceDataError(f"{saved.source}: {problem}", hint=save) from None


def utc_time(value: str, what: str, notes: list[str], origin: Origin) -> datetime:
    """``value`` as an aware time. A time with no zone is UTC, and a note says so."""
    try:
        when = datetime.fromisoformat(value)
    except ValueError:
        raise origin.refuse(f"{what} is {value!r}, not an ISO 8601 time") from None
    if when.tzinfo is None:
        notes.append(f"{what} had no time zone, so NoiseVault reads it as UTC")
        when = when.replace(tzinfo=UTC)
    return when


def clean(value: float) -> float:
    """Drop binary round-off from decimal inputs: 1 - 0.9991 -> 0.0009."""
    return float(f"{value:.12g}")


def _key_problem(node: Any, path: tuple[str | int, ...] = ()) -> str | None:
    """Where ``node`` holds a key that is not a string, or None."""
    if isinstance(node, Mapping):
        for key in node:
            if not isinstance(key, str):
                where = f"a key of {json_path(path)}" if path else "a top-level key"
                return f"{where} is {key!r}, not a string"
        children: Any = node.items()
    elif isinstance(node, list | tuple):
        children = enumerate(node)
    else:
        return None
    found = (_key_problem(value, (*path, key)) for key, value in children)
    return next(filter(None, found), None)
