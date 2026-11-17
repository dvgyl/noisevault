"""Convert a calibration history of IBM devices in four tables to the one table of the archive.

    python -m study.archive_tables FOLDER --out history.parquet [--devices NAME ...]

The input is a calibration history of IBM devices in four tables, collected through the public
API. ``FOLDER`` holds the tables as parquet files. The script reads ``qubit_properties.parquet``
and ``gate_properties.parquet``. It does not read ``backends.parquet`` and
``general_properties.parquet``. The output has the columns of ``study.archive._COLUMNS`` and a
``unit`` column, so that ``study.archive``, ``study.archive_steps``, ``study.archive_zeros`` and
``noisevault.sources.hf_archive`` read it. The script writes the report text to standard output.

The rules of the conversion:

1. Qubit rows. The script keeps the names of ``QUBIT_NAMES``. ``property`` is the name,
   ``qubit_a`` is the qubit and ``qubit_b`` is null. The tables give T1 and T2 in microseconds
   with the unit ``us`` or ``µs``. The output gives both with the unit ``us``. Another unit of
   T1 or T2 stops the script.
2. Gate rows. The script keeps the ``gate_error`` rows of the gate types of ``GATE_TYPES``.
   ``property`` is ``<gate_type>_gate_error``. ``qubit_a`` and ``qubit_b`` are the qubits of the
   gate in their order, and ``qubit_b`` is null for ``sx``. A gate with the wrong number of
   qubits stops the script.
3. ``calibrated_time`` is ``date``. The tables hold no time of observation, so
   ``observed_time`` is ``date`` too. ``is_failure_ceiling`` is false, because the tables do not
   hold this mark. ``study.archive`` takes a gate error of 0.5 or more as a failure value.
4. The script drops a row with a ``date`` before ``EARLIEST`` and each row of a device whose
   name ends with an underscore. The report counts the rows of each rule.
5. The output has one row for each key: device, property, ``qubit_a``, ``qubit_b`` and
   ``calibrated_time``. Rows with the same key and the same value are repeats, and the script
   keeps one of them. When rows with the same key have different values, the script keeps the
   last row in file order. The report counts these keys.
6. The output is in the order of device, property, ``qubit_a``, ``qubit_b`` and
   ``calibrated_time``.

The rows of one device are in many row groups of each table, so the script reads one row group
at a time. It removes the repeats of each row group before it keeps the rows. Then it removes
the repeats of each device and writes the device. The kept rows of all devices stay in memory.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pyarrow as pa

QUBIT_TABLE = "qubit_properties.parquet"
GATE_TABLE = "gate_properties.parquet"
QUBIT_NAMES = ("T1", "T2", "prob_meas0_prep1", "prob_meas1_prep0", "readout_error")
GATE_TYPES = ("sx", "cx", "ecr", "cz", "rzz")
ONE_QUBIT_GATES = ("sx",)
GATE_NAME = "gate_error"
MICROSECONDS = ("us", "µs")
COHERENCE = ("T1", "T2")
EARLIEST = datetime(2015, 1, 1, tzinfo=UTC)
# Sorted, so that the order of the codes is the order of the names.
PROPERTIES = tuple(sorted((*QUBIT_NAMES, *(f"{g}_{GATE_NAME}" for g in GATE_TYPES))))
UNITS = tuple("us" if p in COHERENCE else "" for p in PROPERTIES)
_NO_QUBIT = -1


def schema() -> pa.Schema:
    """The columns of the output, with the types that ``study.archive`` and
    ``noisevault.sources.hf_archive`` read."""
    import pyarrow as pa

    stamp = pa.timestamp("us", tz="UTC")
    return pa.schema(
        [
            ("backend", pa.string()),
            ("property", pa.string()),
            ("qubit_a", pa.int64()),
            ("qubit_b", pa.float64()),
            ("value", pa.float64()),
            ("unit", pa.string()),
            ("calibrated_time", stamp),
            ("observed_time", stamp),
            ("is_failure_ceiling", pa.bool_()),
        ]
    )


@dataclass(frozen=True)
class Part:
    """Rows as arrays. ``device`` indexes the device names of the conversion and ``prop`` indexes
    ``PROPERTIES``. ``qubit_b`` is ``_NO_QUBIT`` for a value of one qubit. ``order`` increases
    in file order. ``conflict`` marks a row whose key had rows with different values."""

    device: np.ndarray
    prop: np.ndarray
    qubit_a: np.ndarray
    qubit_b: np.ndarray
    time_us: np.ndarray
    value: np.ndarray
    order: np.ndarray
    conflict: np.ndarray

    def take(self, index: np.ndarray | slice) -> Part:
        return Part(*(getattr(self, name)[index] for name in self.__dataclass_fields__))

    @staticmethod
    def join(parts: Sequence[Part]) -> Part:
        names = Part.__dataclass_fields__
        return Part(*(np.concatenate([getattr(p, name) for p in parts]) for name in names))


@dataclass
class Counts:
    """The numbers of the report. Each dictionary has one entry for each device name."""

    file_rows: dict[str, int] = field(default_factory=dict)
    kept_names: dict[str, int] = field(default_factory=dict)
    early: dict[str, int] = field(default_factory=dict)
    underscore: dict[str, int] = field(default_factory=dict)
    rows_out: dict[str, int] = field(default_factory=dict)
    conflicts: dict[str, int] = field(default_factory=dict)
    first: dict[str, datetime] = field(default_factory=dict)
    last: dict[str, datetime] = field(default_factory=dict)
    seconds: float = 0.0

    def add(self, counter: dict[str, int], names: list[str], codes: np.ndarray) -> None:
        for code, count in enumerate(np.bincount(codes, minlength=len(names)).tolist()):
            if count:
                counter[names[code]] = counter.get(names[code], 0) + count


def unique_rows(part: Part) -> Part:
    """One row for each key, in the order of the key. The row is the last row of the key in file
    order. ``conflict`` marks a key with two or more different values or a marked row."""
    if part.value.size == 0:
        return part
    order = np.lexsort(
        (part.order, part.time_us, part.qubit_b, part.qubit_a, part.prop, part.device)
    )
    rows = part.take(order)
    same = (
        (rows.device[1:] == rows.device[:-1])
        & (rows.prop[1:] == rows.prop[:-1])
        & (rows.qubit_a[1:] == rows.qubit_a[:-1])
        & (rows.qubit_b[1:] == rows.qubit_b[:-1])
        & (rows.time_us[1:] == rows.time_us[:-1])
    )
    last = np.append(~same, True)
    key = np.cumsum(np.concatenate(([True], ~same))) - 1
    # Bit patterns, so that two NaN values of the same key are one value.
    bits = rows.value.view(np.int64)
    differs = same & (bits[1:] != bits[:-1])
    mixed = np.zeros(int(key[-1]) + 1 if key.size else 0, dtype=bool)
    mixed[key[1:][differs]] = True
    mixed[key[rows.conflict]] = True
    return replace(rows.take(last), conflict=mixed)


def _groups(path: Path, column: str, wanted: Sequence[str]) -> Iterator[int]:
    """The row groups of ``path`` whose statistics of ``column`` allow a value of ``wanted``."""
    import pyarrow.parquet as pq

    meta = pq.ParquetFile(path).metadata
    index = meta.schema.to_arrow_schema().get_field_index(column)
    for group in range(meta.num_row_groups):
        stats = meta.row_group(group).column(index).statistics
        if stats is None or not stats.has_min_max:
            yield group
        elif any(stats.min <= w <= stats.max for w in wanted):
            yield group


class _Names:
    """Device names and their codes, in the order that the tables give them."""

    def __init__(self) -> None:
        self.names: list[str] = []
        self._codes: dict[str, int] = {}

    def codes(self, column: pa.ChunkedArray) -> np.ndarray:
        import pyarrow.compute as pc

        encoded = pc.dictionary_encode(column).combine_chunks()
        local = encoded.dictionary.to_pylist()
        for name in local:
            if name not in self._codes:
                self._codes[name] = len(self.names)
                self.names.append(name)
        recode = np.array([self._codes[n] for n in local], dtype=np.int16)
        return recode[encoded.indices.to_numpy()] if local else np.zeros(0, dtype=np.int16)


def _prop_codes(column: pa.ChunkedArray, suffix: str = "") -> np.ndarray:
    import pyarrow.compute as pc

    encoded = pc.dictionary_encode(column).combine_chunks()
    recode = np.array(
        [PROPERTIES.index(n + suffix) for n in encoded.dictionary.to_pylist()], dtype=np.int8
    )
    return recode[encoded.indices.to_numpy()] if recode.size else np.zeros(0, dtype=np.int8)


def _keep(
    table: pa.Table,
    wanted: Any,
    first_row: int,
    names: _Names,
    counts: Counts,
    devices: Sequence[str] | None,
) -> tuple[pa.Table, np.ndarray, np.ndarray]:
    """The rows of ``wanted`` that rule 4 keeps, their device codes and their file order."""
    import pyarrow as pa
    import pyarrow.compute as pc

    if devices is not None:
        wanted = pc.and_(wanted, pc.is_in(table["backend_name"], pa.array(list(devices))))
    mask = wanted.to_numpy(zero_copy_only=False)
    order = first_row + np.flatnonzero(mask)
    table = table.filter(wanted)
    device = names.codes(table["backend_name"])
    counts.add(counts.kept_names, names.names, device)
    stray = np.array([n.endswith("_") for n in names.names], dtype=bool)[device]
    early = pc.less(table["date"], pa.scalar(EARLIEST, pa.timestamp("us", tz="UTC")))
    early = early.to_numpy(zero_copy_only=False) & ~stray
    counts.add(counts.underscore, names.names, device[stray])
    counts.add(counts.early, names.names, device[early])
    kept = ~(stray | early)
    return table.filter(pa.array(kept)), device[kept], order[kept]


def _offsets(source: Any) -> np.ndarray:
    """The file row of the first row of each row group of the parquet file ``source``."""
    sizes = [source.metadata.row_group(g).num_rows for g in range(source.num_row_groups)]
    return np.concatenate(([0], np.cumsum(sizes, dtype=np.int64)))


def _micros(column: pa.ChunkedArray) -> np.ndarray:
    import pyarrow as pa
    import pyarrow.compute as pc

    return pc.cast(pc.cast(column, pa.timestamp("us", tz="UTC")), pa.int64()).to_numpy()


def qubit_parts(
    path: Path, names: _Names, counts: Counts, devices: Sequence[str] | None = None
) -> Iterator[Part]:
    """The kept rows of each row group of the qubit table, with one row for each key."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    source = pq.ParquetFile(path)
    offsets = _offsets(source)
    columns = ["backend_name", "qubit", "date", "name", "unit", "value"]
    for group in _groups(path, "name", QUBIT_NAMES):
        table = source.read_row_group(group, columns=columns)
        wanted = pc.is_in(table["name"], pa.array(QUBIT_NAMES))
        table, device, order = _keep(table, wanted, int(offsets[group]), names, counts, devices)
        coherence = pc.is_in(table["name"], pa.array(COHERENCE))
        units = pc.is_in(table["unit"], pa.array(MICROSECONDS))
        wrong = pc.and_(coherence, pc.invert(units))
        if pc.any(wrong).as_py():
            unit = pc.filter(table["unit"], wrong)[0].as_py()
            raise ValueError(f"{path.name}: a T1 or T2 row has the unit {unit!r}, not us or µs")
        size = table.num_rows
        yield unique_rows(
            Part(
                device,
                _prop_codes(table["name"]),
                table["qubit"].to_numpy().astype(np.int32),
                np.full(size, _NO_QUBIT, dtype=np.int32),
                _micros(table["date"]),
                table["value"].to_numpy(),
                order,
                np.zeros(size, dtype=bool),
            )
        )


def gate_parts(
    path: Path,
    names: _Names,
    counts: Counts,
    devices: Sequence[str] | None = None,
    first_row: int = 0,
) -> Iterator[Part]:
    """The kept rows of each row group of the gate table, with one row for each key.
    ``first_row`` is added to the file order of each row."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    source = pq.ParquetFile(path)
    offsets = first_row + _offsets(source)
    columns = ["backend_name", "gate_type", "qubits", "date", "name", "value"]
    named = set(_groups(path, "name", (GATE_NAME,)))
    for group in _groups(path, "gate_type", GATE_TYPES):
        if group not in named:
            continue
        table = source.read_row_group(group, columns=columns)
        wanted = pc.and_(
            pc.equal(table["name"], GATE_NAME), pc.is_in(table["gate_type"], pa.array(GATE_TYPES))
        )
        table, device, order = _keep(table, wanted, int(offsets[group]), names, counts, devices)
        qubits = table["qubits"].combine_chunks()
        if qubits.null_count:
            raise ValueError(f"{path.name}: a gate_error row has no qubits")
        length = pc.list_value_length(qubits).to_numpy()
        one = pc.is_in(table["gate_type"], pa.array(ONE_QUBIT_GATES)).to_numpy(zero_copy_only=False)
        if np.any(length != np.where(one, 1, 2)):
            raise ValueError(f"{path.name}: a gate_error row has the wrong number of qubits")
        flat = pc.list_flatten(qubits).to_numpy().astype(np.int32)
        start = np.cumsum(length) - length
        second = flat[np.minimum(start + 1, max(flat.size - 1, 0))] if flat.size else flat
        yield unique_rows(
            Part(
                device,
                _prop_codes(table["gate_type"], f"_{GATE_NAME}"),
                flat[start],
                np.where(one, _NO_QUBIT, second).astype(np.int32),
                _micros(table["date"]),
                table["value"].to_numpy(),
                order,
                np.zeros(table.num_rows, dtype=bool),
            )
        )


def table_of(name: str, rows: Part) -> pa.Table:
    """The output rows of the device ``name``."""
    import pyarrow as pa

    size = rows.value.size
    stamp = pa.array(rows.time_us, type=pa.timestamp("us", tz="UTC"))
    codes = pa.array(rows.prop.astype(np.int32))
    return pa.Table.from_arrays(
        [
            pa.DictionaryArray.from_arrays(
                pa.array(np.zeros(size, dtype=np.int32)), pa.array([name])
            ).cast(pa.string()),
            pa.DictionaryArray.from_arrays(codes, pa.array(PROPERTIES)).cast(pa.string()),
            pa.array(rows.qubit_a.astype(np.int64)),
            pa.array(rows.qubit_b.astype(np.float64), mask=rows.qubit_b == _NO_QUBIT),
            pa.array(rows.value, type=pa.float64()),
            pa.DictionaryArray.from_arrays(codes, pa.array(UNITS)).cast(pa.string()),
            stamp,
            stamp,
            pa.array(np.zeros(size, dtype=bool)),
        ],
        schema=schema(),
    )


def convert(folder: Path, out: Path, devices: Sequence[str] | None = None) -> Counts:
    """Convert the tables in ``folder`` to the one table at ``out``, and count each rule."""
    import pyarrow.parquet as pq

    began = time.monotonic()
    counts = Counts()
    names = _Names()
    qubit_path, gate_path = folder / QUBIT_TABLE, folder / GATE_TABLE
    qubit_rows = pq.ParquetFile(qubit_path).metadata.num_rows
    counts.file_rows = {
        QUBIT_TABLE: qubit_rows,
        GATE_TABLE: pq.ParquetFile(gate_path).metadata.num_rows,
    }
    held: dict[int, list[Part]] = {}
    for part in chain(
        qubit_parts(qubit_path, names, counts, devices),
        gate_parts(gate_path, names, counts, devices, qubit_rows),
    ):
        bounds = np.flatnonzero(np.diff(part.device)) + 1
        for piece in np.split(np.arange(part.value.size), bounds):
            if piece.size:
                held.setdefault(int(part.device[piece[0]]), []).append(part.take(piece))
    out.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(out, schema()) as writer:
        for code in sorted(held, key=lambda c: names.names[c]):
            name = names.names[code]
            rows = unique_rows(Part.join(held.pop(code)))
            writer.write_table(table_of(name, rows), row_group_size=1_000_000)
            counts.rows_out[name] = int(rows.value.size)
            counts.conflicts[name] = int(np.count_nonzero(rows.conflict))
            counts.first[name] = _datetime(int(rows.time_us.min()))
            counts.last[name] = _datetime(int(rows.time_us.max()))
    counts.seconds = time.monotonic() - began
    return counts


def _datetime(micros: int) -> datetime:
    return datetime.fromtimestamp(micros / 1e6, tz=UTC)


def report(counts: Counts) -> str:
    """The numbers of each rule and of each device as text."""
    kept = sum(counts.kept_names.values())
    early, stray = sum(counts.early.values()), sum(counts.underscore.values())
    out = sum(counts.rows_out.values())
    lines = [
        *(f"rows in {name}: {rows}" for name, rows in counts.file_rows.items()),
        f"rows with a kept name: {kept}",
        f"rows dropped with a date before {EARLIEST:%Y-%m-%d}: {early}",
        f"rows dropped for a device name that ends with an underscore: {stray}",
        f"rows removed by rule 5, repeats and older values of a key: {kept - early - stray - out}",
        f"keys with different values: {sum(counts.conflicts.values())}",
        f"rows out: {out}",
        f"seconds: {counts.seconds:.0f}",
        "",
        "device, rows with a kept name, rows out, keys with different values,"
        " rows before the earliest date, first date, last date",
    ]
    for name in sorted(counts.rows_out):
        lines.append(
            f"{name}, {counts.kept_names.get(name, 0)}, {counts.rows_out[name]},"
            f" {counts.conflicts[name]}, {counts.early.get(name, 0)},"
            f" {counts.first[name]:%Y-%m-%d %H:%M:%S}, {counts.last[name]:%Y-%m-%d %H:%M:%S}"
        )
    for name in sorted(counts.underscore):
        lines.append(f"{name}: dropped, {counts.underscore[name]} rows")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("folder", type=Path, help="the folder of the four parquet tables")
    parser.add_argument("--out", type=Path, required=True, help="the output parquet file")
    parser.add_argument("--devices", nargs="+", help="convert only these devices")
    args = parser.parse_args(argv)
    missing = [t for t in (QUBIT_TABLE, GATE_TABLE) if not (args.folder / t).is_file()]
    if missing:
        print(f"error: {args.folder} has no {' or '.join(missing)}", file=sys.stderr)
        return 2
    print(report(convert(args.folder, args.out, args.devices)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
