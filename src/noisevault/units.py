"""Time-unit aliases accepted in hand-written profiles, normalized before validation.

Saved profiles only use the canonical spellings: ``duration_ns``, ``t1_us``, ``t2_us``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import MAX_EMAX, MIN_EMIN, Context, Decimal
from typing import Any

_NS_PER = {"ns": 1, "us": 1_000, "ms": 1_000_000, "s": 1_000_000_000}
# A fixed context, so a caller's decimal precision or traps cannot change a loaded value.
_EXACT = Context(prec=40, Emax=MAX_EMAX, Emin=MIN_EMIN, traps=[])


@dataclass(frozen=True)
class TimeField:
    stem: str
    unit: str
    strictly_positive: bool

    @property
    def canonical(self) -> str:
        return f"{self.stem}_{self.unit}"

    def spellings(self) -> tuple[str, ...]:
        return tuple(f"{self.stem}_{unit}" for unit in _NS_PER)


DURATION = TimeField("duration", "ns", strictly_positive=False)
T1 = TimeField("t1", "us", strictly_positive=True)
T2 = TimeField("t2", "us", strictly_positive=True)


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Move the decimal point of the shortest repr of ``value``, then round one time.

    Thus 1.001 us gives 1001.0 ns, the value that the decimal text states. A float product
    gives 1000.9999999999999, and the fingerprint then differs from a file that says 1001 ns.
    """
    scale = _EXACT.divide(Decimal(_NS_PER[from_unit]), Decimal(_NS_PER[to_unit]))
    return float(_EXACT.multiply(Decimal(repr(value)), scale))


def normalize_times(data: Any, fields: tuple[TimeField, ...]) -> Any:
    """Return ``data`` with every alias of ``fields`` rewritten to its canonical key.

    Raises ValueError for two spellings of one field, a non-numeric or nonfinite value, a
    negative duration, or a nonpositive coherence time. Non-dict input passes through so the
    model validator reports it.
    """
    if not isinstance(data, dict):
        return data
    out = dict(data)
    for field in fields:
        for key in field.spellings():
            if key in out and out[key] is None:
                del out[key]
        given = [key for key in field.spellings() if key in out]
        if not given:
            continue
        if len(given) > 1:
            raise ValueError(f"{field.stem} appears twice ({' and '.join(given)}). Keep one")
        key = given[0]
        value = out.pop(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{key} must be a number, got {value!r}")
        try:
            number = float(value)
        except OverflowError:
            # pydantic reports only ValueError as a validation error
            raise ValueError(
                f"{key} must be finite, got an integer too large for a float"
            ) from None
        if not math.isfinite(number):
            raise ValueError(f"{key} must be finite, got {value!r}")
        if field.strictly_positive and number <= 0:
            raise ValueError(f"{key} must be positive, got {value!r}")
        if number < 0:
            raise ValueError(f"{key} must not be negative, got {value!r}")
        out[field.canonical] = convert(number, key.rsplit("_", 1)[1], field.unit)
    return out
