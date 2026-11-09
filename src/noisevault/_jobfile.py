"""Files that the run scripts create and never replace.

``scripts/run_on_ibm.py`` and ``scripts/run_on_braket.py`` save a job file before they submit
work that costs time or money. They write the counts files beside it. A script changes or
deletes a file only while that file is still the file that the script created and still holds
the bytes that the script wrote.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BINARY = getattr(os, "O_BINARY", 0)


@dataclass
class Owned:
    """A file that this process created, with the bytes that it wrote."""

    path: Path
    fd: int
    data: bytes = b""
    start: int = field(init=False)

    def __post_init__(self) -> None:
        self.start = len(self.data)

    def _held(self) -> int | None:
        try:
            if not os.path.samestat(os.lstat(self.path), os.fstat(self.fd)):
                return None
            os.lseek(self.fd, 0, os.SEEK_SET)
            content = b""
            while len(content) <= len(self.data) and (
                chunk := os.read(self.fd, len(self.data) + 1 - len(content))
            ):
                content += chunk
        except OSError:
            return None
        if self.start <= len(content) <= len(self.data) and self.data.startswith(content):
            return len(content)
        return None

    def holds(self) -> bool:
        return self._held() is not None

    def check(self) -> int:
        held = self._held()
        if held is None:
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(self.path))
        return held

    def append(self, data: bytes) -> None:
        held = self.check()
        self.data += data
        while held < len(self.data):
            os.write(self.fd, memoryview(self.data)[held:])
            held = self.check()
        self.start = len(self.data)

    def cut(self, size: int) -> None:
        self.check()
        os.ftruncate(self.fd, size)
        self.data = self.data[:size]
        self.start = size

    def drop(self) -> None:
        ours = self.holds()
        os.close(self.fd)
        if ours:
            self.path.unlink(missing_ok=True)


def create(files: dict[Path, bytes]) -> dict[Path, Owned]:
    """Create every file, or none. A file that exists stops the call with FileExistsError.

    ``os.link`` would make each full file appear at once, but exFAT does not support hard links.
    """
    created: dict[Path, Owned] = {}
    try:
        for path in files:
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | BINARY, 0o666)
            created[path] = Owned(path, fd)
        for path, data in files.items():
            created[path].append(data)
        for owned in created.values():
            owned.check()
    except BaseException:
        for owned in created.values():
            with contextlib.suppress(OSError):
                owned.drop()
        raise
    return created


def line(data: dict[str, Any]) -> bytes:
    """One JSON line of a job file."""
    return (json.dumps(data) + "\n").encode("utf-8")


def beside(output: Path, suffix: str) -> Path:
    """The file beside a counts file, such as ``run.job.json`` for ``run.counts.json``."""
    stem = output.name.removesuffix(".gz").removesuffix(".json").removesuffix(".counts")
    return output.with_name(stem + suffix)
