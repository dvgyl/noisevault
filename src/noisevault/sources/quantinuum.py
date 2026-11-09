"""Quantinuum H-series machines from Quantinuum's published hardware-specification data.

Quantinuum publishes the benchmarking data behind its product data sheets under Apache-2.0 at
https://github.com/Quantinuum/quantinuum-hardware-specifications. The repository also has
``qtm_spec``, the analysis code that turns the data into the published numbers. This module
repeats the point estimates of that analysis (``qtm_spec.combined_analysis.extract_parameters``):

- 1Q error: the randomized-benchmarking decay pooled over all gate zones. The decay becomes
  the average infidelity per native U1q gate (one per single-qubit Clifford), plus leakage per
  gate / 2.
- 2Q error: the same for two-qubit Cliffords. The decay becomes the average infidelity per
  native ZZ (ZZMax) gate with 1.5 ZZ gates per Clifford, plus leakage per gate / 4.
- SPAM: the fraction of wrong outcomes for each prepared state, averaged over qubits.
- memory error per qubit at depth 1: informational only, kept in ``benchmarks``.

The emulator parameters (``p1``, ``p2``, ...) are fault probabilities of a different model.
This module does not use them.
"""

from __future__ import annotations

import hashlib
import re
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from ..errors import SourceDataError, SourceUnavailable, plural
from ..profile import Profile
from . import OFFLINE_HINT, Origin, csv_by_line, download, source_json, source_text

REPOSITORY = "https://github.com/Quantinuum/quantinuum-hardware-specifications"
COMMIT = "59e68bb55bd616694dc8fa37a435e2a68fe1cb6b"  # pinned so a rebuild gives the same bundle
_RAW = f"https://raw.githubusercontent.com/Quantinuum/quantinuum-hardware-specifications/{COMMIT}"
_SPEC_SHEET = f"download notebooks/Spec sheet parameters.csv from {REPOSITORY}"

# Every dataset at COMMIT and the machine's qubit count on that date. The data has a key per gate
# zone, not per qubit, so the counts come from Quantinuum's announcements: H1 machines have 20
# qubits in five zones since June 2022 (H1-2's 2022 data still has the three zones of the
# 12-qubit design), H2-1 went from 32 to 56 qubits in May 2024, and REIMEI is an H1-class
# machine with 20 qubits.
QUBITS: Mapping[tuple[str, str], int] = {
    ("H1-1", "2022_06_09"): 20,
    ("H1-1", "2023_01_20"): 20,
    ("H1-1", "2023_07_17"): 20,
    ("H1-1", "2024_04_10"): 20,
    ("H1-1", "2025_05_02"): 20,
    ("H1-2", "2022_06_09"): 12,
    ("H1-2", "2023_08_21"): 20,
    ("H2-1", "2023_03_10"): 32,
    ("H2-1", "2024_05_20"): 56,
    ("H2-1", "2025_04_30"): 56,
    ("H2-2", "2024_12_06"): 56,
    ("H2-2", "2025_05_29"): 56,
    ("H2-2", "2025_08_28"): 56,
    ("REIMEI", "2025_01_26"): 20,
    ("REIMEI", "2025_06_18"): 20,
}
_PROCESSOR = {"H1": "System Model H1", "H2": "System Model H2", "REIMEI": "System Model H1"}
FILES = ("SQ_RB", "TQ_RB", "SPAM", "Memory_RB")  # the order source_hash covers
_LOWERCASE_SPAM_DATES = {"2022_06_09"}  # these datasets name the file spam.json
_TWO_QUBIT_CLIFFORD_ZZ = 1.5  # ZZ gates per two-qubit Clifford, as qtm_spec converts
_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class Estimate:
    value: float
    stderr: float | None = None


@dataclass(frozen=True)
class SpecValues:
    """The data-sheet quantities of one machine on one date, as ``qtm_spec`` defines them."""

    machine: str
    date: str  # YYYY_MM_DD, the repository's folder name
    one_qubit: Estimate  # average infidelity per U1q gate
    two_qubit: Estimate  # average infidelity per ZZ gate
    one_qubit_leakage: Estimate | None  # leakage per gate, or None if not measured
    two_qubit_leakage: Estimate | None
    spam: tuple[float, float] | Estimate  # (P(1|0), P(0|1)) per prepared state, or combined
    memory: Estimate | None


# public entry points ------------------------------------------------------------------------


def bundled_profiles() -> list[Profile]:
    """The newest dataset of every machine at the pinned commit (downloads about 40 MB)."""
    return [from_repository(machine) for machine in sorted({m for m, _ in QUBITS})]


def from_repository(machine: str, date: str | None = None) -> Profile:
    """Download one dataset (the machine's newest when ``date`` is None) and build its profile."""
    machine = _machine(machine)
    date = _date(date) if date else max(d for m, d in QUBITS if m == machine)
    if (machine, date) not in QUBITS:
        known = ", ".join(sorted(d for m, d in QUBITS if m == machine))
        raise ValueError(
            f"no {machine} dataset dated {date!r} at the pinned commit. Known dates: {known}"
        )
    files = {name: _get(f"{_RAW}/{_file_path(machine, date, name)}") for name in FILES}
    return from_data(machine, date, files)


def from_data(machine: str, date: str, files: Mapping[str, bytes]) -> Profile:
    """A profile from the raw dataset files (``FILES`` keys, bytes as downloaded)."""
    machine = _machine(machine)
    data = {name: _json(files[name], _file_path(machine, date, name)) for name in FILES}
    values = spec_values(machine, date, data)
    digest = hashlib.sha256(b"".join(files[name] for name in FILES)).hexdigest()
    return to_profile(
        values,
        source=f"Quantinuum hardware specifications, data/{machine}/{date} at {COMMIT[:12]}",
        source_url=f"{REPOSITORY}/tree/{COMMIT}/data/{machine}/{date}",
        source_hash=f"sha256:{digest}",
        extra={
            "commit": COMMIT,
            "files": {name: "sha256:" + hashlib.sha256(files[name]).hexdigest() for name in FILES},
        },
        notes=(
            f"source_hash covers {', '.join(n + '.json' for n in FILES)} concatenated in that"
            " order",
            "values repeat the point estimates of qtm_spec.combined_analysis.extract_parameters"
            " (decays pooled over all gate zones, so statistic is mean); stderr is left out"
            " because qtm_spec estimates it with an unseeded bootstrap",
        ),
    )


def from_spec_csv(
    path: str | Path, *, machine: str, date: str, num_qubits: int | None = None
) -> Profile:
    """A profile from one row of the repository's ``Spec sheet parameters.csv`` format.

    Cells read like ``2.15(8)E-03`` (value and uncertainty in the last digit). The CSV gives
    only the combined SPAM error, so readout is symmetric.
    """
    machine, date = _machine(machine), _date(date)
    raw = Path(path).read_bytes()
    lines = csv_by_line(source_text(raw, str(path), _SPEC_SHEET), str(path))
    headers = lines[0][1] if lines else []
    for position, header in enumerate(headers, start=1):
        first = headers.index(header) + 1
        if first != position:
            raise SourceDataError(
                f"{path}: columns {first} and {position} are both {header!r}",
                hint="delete one of them",
            )
    if missing := [repr(column) for column in ("Date", "Machine") if column not in headers]:
        raise SourceDataError(f"{path} has no {' and no '.join(missing)} column", hint=_SPEC_SHEET)
    for line, cells in lines[1:]:
        if cells and len(cells) != len(headers):
            raise SourceDataError(
                f"{path} line {line} has {plural(len(cells), 'cell')},"
                f" but the header has {len(headers)} columns"
            )
    records = [dict(zip(headers, cells, strict=True)) for _, cells in lines[1:] if cells]
    rows = [row for row in records if row["Machine"] == machine]
    dated = [row for row in rows if row["Date"] == date]
    if len(dated) != 1:
        known = ", ".join(sorted({row["Date"] for row in rows})) or "none"
        raise SourceDataError(
            f"{path} has {len(dated)} rows for {machine} dated {date}, not one. The file has"
            f" these dates (YYYY_MM_DD) for {machine}: {known}"
        )
    rows = dated
    values = _csv_values(machine, date, rows[0])
    return to_profile(
        values,
        num_qubits=num_qubits,
        hint=f"correct that value in {path}",
        source=f"Quantinuum spec sheet parameters CSV ({Path(path).name}), {machine} {date}",
        source_url=None,
        source_hash="sha256:" + hashlib.sha256(raw).hexdigest(),
        extra={},
        notes=(
            "values and stderr are the CSV's rounded spec-sheet figures. The CSV's 'Transport 1Q"
            " error' column holds the memory error (qtm_spec writes Memory_RB there)",
        ),
    )


def _date(date: str) -> str:
    """The repository's ``YYYY_MM_DD``. This function also accepts ISO ``YYYY-MM-DD``."""
    return date.strip().replace("-", "_")


def _json(raw: bytes, where: str) -> dict[str, Any]:
    doc = source_json(raw, where)
    if not isinstance(doc, dict):
        raise SourceDataError(f"{where} is not a JSON object")
    return doc


def _key(node: Mapping[str, Any], key: str, where: str) -> Any:
    if not isinstance(node, Mapping):
        raise SourceDataError(f"{where} is not a JSON object")
    if key not in node:
        raise SourceDataError(f"{where} has no {key!r}")
    return node[key]


def _object(node: Any, where: str, of: str) -> Mapping[str, Any]:
    if not isinstance(node, Mapping):
        raise SourceDataError(f"{where} is not a JSON object")
    if not node:
        raise SourceDataError(f"{where} has no {of}")
    return node


# the qtm_spec analysis ----------------------------------------------------------------------


def spec_values(machine: str, date: str, data: Mapping[str, Any]) -> SpecValues:
    """The data-sheet quantities from parsed dataset files, keyed as in ``FILES``."""
    where = {name: _file_path(machine, date, name) for name in FILES}
    one, one_leak = _rb(data["SQ_RB"], where["SQ_RB"], num_qubits=1)
    two, two_leak = _rb(data["TQ_RB"], where["TQ_RB"], num_qubits=2)
    memory, _ = _rb(data["Memory_RB"], where["Memory_RB"], num_qubits=1)
    return SpecValues(
        machine=machine,
        date=date,
        one_qubit=Estimate(one),
        two_qubit=Estimate(two),
        one_qubit_leakage=None if one_leak is None else Estimate(one_leak),
        two_qubit_leakage=None if two_leak is None else Estimate(two_leak),
        spam=_spam(data["SPAM"], where["SPAM"]),
        memory=Estimate(memory),
    )


def _rb(data: Mapping[str, Any], where: str, *, num_qubits: int) -> tuple[float, float | None]:
    d = 2**num_qubits
    per_clifford = _TWO_QUBIT_CLIFFORD_ZZ if num_qubits == 2 else 1.0
    shots, survival = _shots(data, where), _zones(data, "survival", where)
    leak_zones = _zones(data, "leakage_postselect", where) if "leakage_postselect" in data else {}
    lengths = list(dict.fromkeys(m for zone in (survival | leak_zones).values() for m in zone))
    rate = decay_rate(*_pooled(survival, lengths, shots), asymptote=1 / d)
    error = 1 - ((d - 1) * rate ** (1 / per_clifford) + 1) / d
    if "leakage_postselect" not in data:
        return error, None
    leak_rate = decay_rate(*_pooled(leak_zones, lengths, shots), asymptote=0.0)
    leakage = (1 - leak_rate) / per_clifford
    return error + leakage / d, leakage


def _zones(data: Mapping[str, Any], curve: str, where: str) -> dict[str, Mapping[str, Any]]:
    zones = {}
    for zone, by_length in _object(_key(data, curve, where), f"{where}: {curve}", "zones").items():
        at = f"{where}: {curve}[{zone!r}]"
        for m in _object(by_length, at, "sequence lengths"):
            if not m.isdecimal():
                raise SourceDataError(f"{at} has the sequence length {m!r}, not a whole number")
        zones[at] = by_length
    return zones


def _pooled(
    zones: Mapping[str, Mapping[str, Any]], lengths: list[str], shots: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sequence lengths and mean survival over every zone and repetition at each length."""
    means = []
    for m in lengths:
        found = [
            _count(n, shots, f"{at}[{m!r}][{rep!r}]")
            for at, by_length in zones.items()
            for rep, n in _object(_key(by_length, m, at), f"{at}[{m!r}]", "counts").items()
        ]
        means.append(np.mean(found) / shots)
    return np.array([int(m) for m in lengths], dtype=float), np.array(means)


def _shots(data: Mapping[str, Any], where: str) -> int:
    shots = _key(data, "shots", where)
    if not (_whole(shots) and shots > 0):
        raise SourceDataError(f"{where}: shots is {shots!r}, not a positive whole number")
    return int(shots)


def _count(n: Any, shots: int, where: str) -> int:
    if not (_whole(n) and 0 <= n <= shots):
        raise SourceDataError(f"{where} is {n!r}, not a whole number of shots from 0 to {shots}")
    return int(n)


def _whole(value: Any) -> bool:
    if isinstance(value, float):
        return value.is_integer()
    return isinstance(value, int) and not isinstance(value, bool)


def decay_rate(lengths: np.ndarray, means: np.ndarray, *, asymptote: float) -> float:
    """``r`` of the least-squares fit ``means = A r**m + asymptote`` with A and r in [0, 1].

    qtm_spec does this fit with scipy's bounded ``curve_fit``. For a fixed r the best A is linear,
    so the fit becomes a one-dimensional search over r. The search uses a log-spaced grid toward
    1, then a golden-section refinement around the best grid point.
    """
    y = means - asymptote

    def sse(r: float) -> float:
        powers = r**lengths
        norm = float(powers @ powers)
        amplitude = min(max(float(powers @ y) / norm, 0.0), 1.0) if norm > 0 else 0.0
        return float(np.sum((amplitude * powers - y) ** 2))

    grid = np.append(1 - np.logspace(0, -10, 2001), 1.0)
    best = int(np.argmin([sse(r) for r in grid]))
    lo, hi = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
    ratio = (np.sqrt(5) - 1) / 2
    for _ in range(200):
        a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
        if sse(a) <= sse(b):
            hi = b
        else:
            lo = a
    return float((lo + hi) / 2)


def _spam(data: Mapping[str, Any], where: str) -> tuple[float, float]:
    """(P(1|0), P(0|1)): wrong outcomes for each prepared state, averaged over qubits."""
    shots = _shots(data, where)
    survival = _object(_key(data, "survival", where), f"{where}: survival", "qubits")
    rows = [(f"{where}: survival[{qubit!r}]", row) for qubit, row in survival.items()]
    wrong = []
    for state in ("0", "1"):
        found = [_count(_key(row, state, at), shots, f"{at}[{state!r}]") for at, row in rows]
        wrong.append(1 - float(np.mean(found)) / shots)
    # counts over shots are short decimals, so drop the binary round-off of 1 - x
    return float(f"{wrong[0]:.12g}"), float(f"{wrong[1]:.12g}")


_CELL = re.compile(r"^(\d+)(?:\.(\d+))?\((\d+)\)E([+-]?\d+)$")


def parse_cell(cell: str) -> Estimate | None:
    """``'2.15(8)E-03'`` -> Estimate(2.15e-3, 8e-5), and an empty cell -> None."""
    cell = cell.strip()
    if not cell:
        return None
    match = _CELL.match(cell)
    if match is None:
        raise SourceDataError(f"spec-sheet cell {cell!r} is not in a form such as 2.15(8)E-03")
    whole, frac, unc, exp = match.groups()
    scale = 10.0 ** int(exp)
    digits = len(frac or "")
    return Estimate(float(f"{whole}.{frac or '0'}") * scale, int(unc) * 10.0**-digits * scale)


def _csv_values(machine: str, date: str, row: Mapping[str, str]) -> SpecValues:
    def first(*columns: str) -> Estimate:
        for column in columns:
            found = parse_cell(row.get(column) or "")
            if found is not None:
                return found
        raise SourceDataError(f"{machine} {date}: the CSV row has none of {', '.join(columns)}")

    spam = parse_cell(row.get("SPAM error") or "")
    if spam is None:
        raise SourceDataError(f"{machine} {date}: the CSV row has no SPAM error")
    # leakage is part of the error only in the leakage-corrected columns
    corrected = {n: parse_cell(row.get(f"{n} error") or "") is not None for n in ("1Q", "2Q")}
    return SpecValues(
        machine=machine,
        date=date,
        one_qubit=first("1Q error", "1Q error (legacy)"),
        two_qubit=first("2Q error", "2Q error (legacy)"),
        one_qubit_leakage=parse_cell(row.get("1Q leakage") or "") if corrected["1Q"] else None,
        two_qubit_leakage=parse_cell(row.get("2Q leakage") or "") if corrected["2Q"] else None,
        spam=spam,
        memory=parse_cell(row.get("Transport 1Q error") or ""),
    )


# the profile --------------------------------------------------------------------------------


def to_profile(
    values: SpecValues,
    *,
    source: str,
    source_url: str | None,
    source_hash: str,
    extra: dict[str, Any],
    notes: tuple[str, ...],
    num_qubits: int | None = None,
    hint: str | None = None,
) -> Profile:
    """The profile of ``values``. A value that the format refuses raises a SourceDataError that
    names ``source``, with ``hint`` as its next step.
    """
    machine, date = values.machine, values.date
    qubits = num_qubits or QUBITS.get((machine, date))
    if qubits is None:
        raise SourceDataError(
            f"the qubit count of {machine} on {date} is unknown", hint="pass num_qubits="
        )
    leak1, leak2 = values.one_qubit_leakage, values.two_qubit_leakage
    gates = {
        "rz": {"virtual": True},
        "r": _gate(
            values.one_qubit,
            includes=["leakage"] if leak1 else None,
            assumption="Quantinuum 1Q randomized-benchmarking error: average infidelity per"
            " single-qubit Clifford, one U1q gate each (Z rotations are virtual)"
            + (", plus leakage per gate / 2 as qtm_spec reports it" if leak1 else ""),
        ),
        "zz": _gate(
            values.two_qubit,
            includes=["1q_dressing", "leakage"] if leak2 else ["1q_dressing"],
            assumption="Quantinuum 2Q randomized-benchmarking error: average infidelity per"
            " ZZMax gate, from the two-qubit Clifford decay at 1.5 ZZ gates per Clifford, so the"
            " single-qubit gates inside each Clifford are included"
            + (", plus leakage per gate / 4 as qtm_spec reports it" if leak2 else ""),
        ),
    }
    if isinstance(values.spam, Estimate):
        readout: dict[str, float] = {"error": values.spam.value}
        spam_note = "readout is Quantinuum's combined SPAM error, the same for both states"
    else:
        readout = {"p1_given_0": values.spam[0], "p0_given_1": values.spam[1]}
        spam_note = "readout is Quantinuum's combined SPAM error for each prepared state"
    effects = [
        {"type": "leakage", "gate": gate, "prob": leak.value}
        for gate, leak in (("r", leak1), ("zz", leak2))
        if leak is not None
    ]
    benchmarks = {}
    if values.memory is not None:
        benchmarks["memory_error_depth1"] = {
            "value": values.memory.value,
            **({"stderr": values.memory.stderr} if values.memory.stderr is not None else {}),
            "convention": "avg_infidelity",
            "per": "qubit per depth-1 circuit",
            "method": "rb",
        }
    family = "REIMEI" if machine == "REIMEI" else machine[:2]
    return Origin(source, hint=hint).profile(
        {
            "noisevault": "1.0",
            "device": {
                "vendor": "quantinuum",
                "name": machine,
                "technology": "trapped_ion",
                "num_qubits": qubits,
                "processor": _PROCESSOR[family],
                "calibrated_at": datetime.strptime(date, "%Y_%m_%d").replace(tzinfo=UTC),
            },
            "connectivity": "all_to_all",
            "gates": gates,
            "readout": readout,
            "effects": effects,
            "benchmarks": benchmarks,
            "provenance": {
                "data_kind": "measured",
                "source_kind": "published_data",
                "source": source,
                "source_url": source_url,
                "license": "Apache-2.0",
                "attribution": "Quantinuum",
                "redistributable": "yes",
                "retrieved_at": datetime.now(UTC),
                "source_hash": source_hash,
                "tool": f"noisevault {__version__}",
                "notes": [
                    *notes,
                    f"{spam_note}; preparation error is inside it, so prep is left unknown",
                    "calibrated_at is the dataset date; the data gives no time of day",
                    "leakage effects record the leakage rate per gate; the gate errors already"
                    " include leakage / d, so the effects are informational",
                ],
                "extra": extra,
            },
        }
    )


def _gate(estimate: Estimate, *, includes: list[str] | None, assumption: str) -> dict[str, Any]:
    return {
        "avg_infidelity": estimate.value,
        "stderr": estimate.stderr,
        "method": "rb",
        "statistic": "mean",
        "includes": includes,
        "assumption": assumption,
    }


# network ------------------------------------------------------------------------------------


def _machine(name: str) -> str:
    wanted = name.strip().upper().removeprefix("QUANTINUUM_").removeprefix("QUANTINUUM ")
    if wanted not in {m for m, _ in QUBITS}:
        known = ", ".join(sorted({m for m, _ in QUBITS}))
        raise ValueError(f"unknown Quantinuum machine {name!r}. Known machines: {known}")
    return wanted


def _file_path(machine: str, date: str, name: str) -> str:
    stem = "spam" if name == "SPAM" and date in _LOWERCASE_SPAM_DATES else name
    return f"data/{machine}/{date}/{stem}.json"


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": f"noisevault/{__version__}"})
    try:
        return download(request, _TIMEOUT_S)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise SourceUnavailable(f"could not download {url} ({reason})", hint=OFFLINE_HINT) from None
