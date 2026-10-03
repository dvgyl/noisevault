from __future__ import annotations

import difflib
import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any, NoReturn


class NoiseVaultError(Exception):
    """Base class for every expected NoiseVault failure.

    ``hint`` is the next step, if there is one. ``str()`` gives the message and then the hint,
    and ``message`` gives the message alone.
    """

    def __init__(self, *args: object, hint: str | None = None) -> None:
        super().__init__(*args)
        self.hint = hint

    @property
    def message(self) -> str:
        return super().__str__()

    def __str__(self) -> str:
        return f"{self.message}; {self.hint}" if self.hint else self.message


class LayoutError(NoiseVaultError, ValueError):
    """A circuit-to-device qubit mapping is incomplete, not injective, or uses a bad qubit."""


class DisabledGateError(NoiseVaultError):
    """The profile marks this gate on these qubits as disabled."""


class MissingCalibrationError(NoiseVaultError):
    """No calibration exists for an operation and the caller asked for an error."""


class UnsupportedEffect(NoiseVaultError):
    """A profile effect needs a treatment that the export framework does not support."""


class AmbiguousRef(NoiseVaultError, LookupError):
    """A ref matches more than one profile."""


class ProfileNotFound(NoiseVaultError, LookupError):
    """No profile matches a ref."""


class FingerprintMismatch(NoiseVaultError, ValueError):
    """A loaded profile does not carry the fingerprint the caller expected."""


class SourceUnavailable(NoiseVaultError):
    """A calibration source is not reachable or is not installed."""


class SourceDataError(NoiseVaultError, ValueError):
    """A source importer cannot read the calibration data that it received."""


class CountsError(NoiseVaultError, ValueError):
    """A counts file is not valid, or ``nv compare`` cannot score its run.

    The message is one line that starts with the file or the field, and ``hint`` says what to do.
    """


class DuplicateKeyError(ValueError):
    """A JSON object has the same key twice. ``path`` ends with that key."""

    def __init__(self, path: tuple[str | int, ...]) -> None:
        super().__init__(f"the key {json_path(path)} appears twice")
        self.path = path


def json_path(keys: Sequence[str | int]) -> str:
    return "".join(
        f".{key}" if isinstance(key, str) and key.isidentifier() else f"[{key!r}]" for key in keys
    ).removeprefix(".")


def parse_json(raw: bytes) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=_unique)
    except _Repeat:
        pass
    except RecursionError:
        _too_deep(raw)
    try:
        tree = json.loads(raw, object_pairs_hook=_Marked)
    except RecursionError:
        _too_deep(raw)
    raise DuplicateKeyError(_repeated(tree))


def _too_deep(raw: bytes) -> NoReturn:
    text = raw.decode(json.detect_encoding(raw), "surrogatepass")
    depth = deepest = at = 0
    for token in _STRING_OR_BRACKET.finditer(text):
        if token[0] in ("[", "{"):
            depth += 1
            if depth > deepest:
                deepest, at = depth, token.start()
        elif token[0] in ("]", "}"):
            depth -= 1
    raise json.JSONDecodeError(f"nested {deepest} levels deep", text, at)


class _Repeat(Exception):
    pass


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj = dict(pairs)
    if len(obj) < len(pairs):
        raise _Repeat
    return obj


class _Marked(dict[str, Any]):
    def __init__(self, pairs: list[tuple[str, Any]]) -> None:
        super().__init__(pairs)
        counts = Counter(k for k, _ in pairs)
        self.repeated = next((k for k, n in counts.items() if n > 1), None)


def _repeated(tree: Any) -> tuple[str | int, ...]:
    parents: list[tuple[int, str | int]] = [(0, "")]
    stack: list[tuple[Any, int]] = [(tree, 0)]
    while True:
        node, at = stack.pop()
        if isinstance(node, _Marked) and node.repeated is not None:
            break
        if isinstance(node, dict):
            children: list[tuple[str | int, Any]] = list(node.items())
        elif isinstance(node, list):
            children = list(enumerate(node))
        else:
            continue
        for key, child in reversed(children):
            parents.append((at, key))
            stack.append((child, len(parents) - 1))
    path: list[str | int] = [node.repeated]
    while at:
        at, key = parents[at]
        path.append(key)
    return tuple(reversed(path))


_STRING_OR_BRACKET = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"|[\[\]{}]')
_DIGIT_LIMIT = re.compile(r"\((\d+) digits\).* has (\d+) digits")


def unreadable(source: str, exc: Exception) -> str:
    if isinstance(exc, UnicodeDecodeError):
        line = exc.object.count(b"\n", 0, exc.start) + 1
        return f"{source} is not UTF-8 text (byte {exc.object[exc.start]:#04x} on line {line})"
    if isinstance(exc, json.JSONDecodeError):
        where = f"{exc.msg.removesuffix(' at')} at line {exc.lineno}, column {exc.colno}"
        return f"{source} is not JSON ({_lower_first(where)})"
    if isinstance(exc, DuplicateKeyError):
        return f"{source} has the key {json_path(exc.path)} twice"
    if isinstance(exc, ValueError):
        digits = _DIGIT_LIMIT.search(str(exc))
        reason = (
            f"a number has {digits[2]} digits, over the {digits[1]}-digit limit"
            if digits
            else str(exc)
        )
        return f"{source} is not JSON ({_lower_first(reason)})"
    return f"{source} is a damaged gzip file ({_lower_first(str(exc))})"


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def did_you_mean(given: str, choices: Iterable[str]) -> str:
    close = difflib.get_close_matches(given, list(choices), n=1)
    return f"did you mean '{close[0]}'? " if close else ""


def plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def joined(words: Sequence[str], conjunction: str = "and") -> str:
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} {conjunction} {words[-1]}"


def qubit_loci(*loci: Sequence[int], limit: int | None = 4) -> str:
    labels = ["-".join(map(str, locus)) for locus in loci]
    if limit is not None and len(labels) > limit:
        labels = [*labels[: limit - 1], f"{len(labels) - limit + 1} more"]
    single = len(loci) == 1 and len(loci[0]) == 1
    return f"qubit {joined(labels)}" if single else f"qubits {joined(labels)}"


_Loci = tuple[tuple[int, ...], ...]


class LociText(str):
    parts: tuple[str | _Loci, ...]

    def __new__(cls, *parts: str | Iterable[Sequence[int]]) -> LociText:
        flat: list[str | _Loci] = []
        for part in parts:
            if isinstance(part, LociText):
                flat += part.parts
            elif isinstance(part, str):
                flat.append(part)
            else:
                flat.append(tuple(tuple(locus) for locus in part))
        text = super().__new__(cls, _render(flat, None))
        text.parts = tuple(flat)
        return text

    @property
    def short(self) -> str:
        return _render(self.parts, 4)

    def join(self, texts: Iterable[str]) -> LociText:
        parts: list[str] = []
        for text in texts:
            parts += [self, text] if parts else [text]
        return LociText(*parts)


def _render(parts: Iterable[str | _Loci], limit: int | None) -> str:
    return "".join(
        part if isinstance(part, str) else qubit_loci(*part, limit=limit) for part in parts
    )


class NoiseVaultWarning(UserWarning):
    """Base class for NoiseVault warnings."""


class NoiseApproximationWarning(NoiseVaultWarning):
    """An export used an approximation that the caller must know about."""


class MigrationWarning(NoiseVaultWarning):
    """NoiseVault upgraded a file in an older format in memory."""


REPOSITORY = "https://github.com/dvgyl/noisevault"


def install_hint(extra: str) -> str:
    """The pip command that adds an optional extra, for example ``install_hint("cirq")``."""
    return f'pip install "noisevault[{extra}] @ git+{REPOSITORY}"'
