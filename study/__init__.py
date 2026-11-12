"""The realism study: scripts that score NoiseVault profiles on counts from real devices."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
CONSTRAINTS = Path(".github") / "study-constraints.txt"


@cache
def code_version() -> str:
    """The SHA-256 of the source files of NoiseVault and of the study, and of the library
    versions in ``.github/study-constraints.txt``.

    A result of the study depends on these files, so a kept result is used again only when
    they are the same.
    """
    digest = hashlib.sha256()
    files = [*(_ROOT / "src" / "noisevault").rglob("*.py"), *(_ROOT / "study").glob("*.py")]
    constraints = _ROOT / CONSTRAINTS
    if constraints.exists():
        files.append(constraints)
    for path in sorted(p for p in files if not p.name.startswith("._")):
        digest.update(path.relative_to(_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def kept(
    rows: Path | None,
    kind: str,
    inputs: list[Path],
    make: Callable[[], Any],
    *,
    names: str = "",
    keep: Callable[[Any], bool] | None = None,
) -> Any:
    """The result of ``make``, from the folder ``rows`` when it holds this result already.

    A result is kept for its ``kind``, the bytes of its ``inputs``, the code version and
    ``names``: the values that the result takes from the file names, such as the chain kind.
    ``names`` holds derived values only, never a path, because a path can hold account data.
    The result must be JSON data. ``rows`` None keeps nothing. A failure of ``make`` is not
    kept, and a result for which ``keep`` gives False is not kept either.
    """
    if rows is None:
        return make()
    digest = hashlib.sha256(f"{kind}\n{code_version()}\n{names}\n".encode())
    for path in inputs:
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    saved = rows / f"{kind}-{digest.hexdigest()[:40]}.json"
    if saved.exists():
        return json.loads(saved.read_text(encoding="utf-8"))
    result = make()
    if keep is not None and not keep(result):
        return result
    rows.mkdir(parents=True, exist_ok=True)
    partial = saved.with_suffix(".part")
    partial.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    partial.replace(saved)
    return result
