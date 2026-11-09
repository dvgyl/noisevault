"""Property tests of the profile loader: round trips, fingerprint stability and refusals.

The loaders have two documented error contracts. ``Profile.model_validate`` raises pydantic's
``ValidationError``. ``load_bytes``, ``load_file`` and ``nv.load`` on a path raise
``ValueError`` subclasses for JSON and profile errors (``ValidationError``, ``JSONDecodeError``,
``DuplicateKeyError``, ``UnicodeDecodeError``) and the gzip errors for a damaged gzip file.
``read_json_file`` turns each JSON and gzip error into a ``NoiseVaultError``.
"""

from __future__ import annotations

import copy
import decimal
import gzip
import json
import random
import tempfile
import zlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from conftest import deeper_than_the_parser_takes, examples, toy
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

import noisevault as nv
from noisevault.errors import DuplicateKeyError, NoiseVaultError
from noisevault.profile import Profile, file_bytes, load_bytes, load_file, read_json_file

Json = Any
JsonPath = tuple[str | int, ...]

# The errors that the byte and file loaders may raise for a bad input. A bare ValueError, a
# TypeError, a KeyError, an OverflowError or a RecursionError is a defect.
LOADER_ERRORS: tuple[type[BaseException], ...] = (
    ValidationError,
    json.JSONDecodeError,
    DuplicateKeyError,
    UnicodeDecodeError,
    NoiseVaultError,
    EOFError,
    zlib.error,
    gzip.BadGzipFile,
)

ONE_QUBIT_GATES = ("sx", "x", "h")
TWO_QUBIT_GATES = ("cz", "ecr", "cx")
CUSTOM_GATE = "my_gate"  # not in the registry, so its definition states its arity
TECHNOLOGIES = ("superconducting", "trapped_ion", "neutral_atom", "spin", "photonic", "other")
OFFSETS = (UTC, timezone(timedelta(hours=-3)), timezone(timedelta(hours=5, minutes=30)))
# Powers of ten from each unit to nanoseconds.
NS_EXPONENT = {"ns": 0, "us": 3, "ms": 6, "s": 9}


# valid profiles -----------------------------------------------------------------------------


def _share(upper: float) -> st.SearchStrategy[float]:
    return st.floats(min_value=0.0, max_value=upper)


_positive = st.floats(min_value=0.0, max_value=1e7, exclude_min=True)
_probability = _share(1.0)


def _metric(arity: int) -> st.SearchStrategy[dict[str, Json]]:
    """No metric, or one metric inside its physical range for a gate on ``arity`` qubits."""
    d = 2**arity
    size = d * d - 1
    return st.one_of(
        st.just({}),
        _share(d / (d + 1)).map(lambda v: {"avg_infidelity": v}),
        _share(1.0).map(lambda v: {"process_infidelity": v}),
        _share(d * d / (d * d - 1)).map(lambda v: {"depolarizing_param": v}),
        st.lists(_share(1 / size), min_size=size, max_size=size).map(lambda p: {"pauli": p}),
    )


@st.composite
def _spec(draw: st.DrawFn, arity: int) -> dict[str, Json]:
    spec = draw(_metric(arity))
    if draw(st.booleans()):
        spec["duration_ns"] = draw(_share(1e7))
    if draw(st.booleans()):
        spec["method"] = draw(st.sampled_from(["rb", "irb", "xeb"]))
    return spec


@st.composite
def _readout(draw: st.DrawFn) -> dict[str, Json]:
    if draw(st.booleans()):
        out: dict[str, Json] = {"error": draw(_probability)}
    else:
        out = {"p1_given_0": draw(_probability), "p0_given_1": draw(_probability)}
    if draw(st.booleans()):
        out["duration_ns"] = draw(_share(1e7))
    return out


@st.composite
def _coherence(draw: st.DrawFn) -> dict[str, Json]:
    out: dict[str, Json] = {}
    for key in ("t1_us", "t2_us"):
        if draw(st.booleans()):
            out[key] = draw(_positive)
    return out


@st.composite
def _connectivity(draw: st.DrawFn, n: int) -> Json:
    if draw(st.booleans()):
        return "all_to_all"
    directed = draw(st.booleans())
    pairs = [(a, b) for a in range(n) for b in range(n) if (a < b or (directed and a != b))]
    chosen = draw(st.lists(st.sampled_from(pairs), unique=True)) if pairs else []
    if not directed:  # an undirected edge may be written in either order
        chosen = [(b, a) if draw(st.booleans()) else (a, b) for a, b in chosen]
    return {"edges": [list(edge) for edge in chosen], "directed": directed}


@st.composite
def _gates(draw: st.DrawFn, n: int) -> tuple[dict[str, Json], dict[str, int]]:
    """Gate definitions, and the arity of each gate that a record can calibrate."""
    names = list(ONE_QUBIT_GATES) + (list(TWO_QUBIT_GATES) if n >= 2 else [])
    chosen = draw(st.lists(st.sampled_from(names), unique=True))
    arity = {name: 1 if name in ONE_QUBIT_GATES else 2 for name in chosen}
    if draw(st.booleans()):
        arity[CUSTOM_GATE] = draw(st.integers(1, min(n, 2)))
    defs: dict[str, Json] = {"rz": {"virtual": True}}
    for name, k in arity.items():
        defs[name] = draw(_spec(k))
        if name == CUSTOM_GATE or draw(st.booleans()):  # a restated default is canonicalized
            defs[name]["qubits"] = k
    return defs, arity


@st.composite
def _qubit_records(draw: st.DrawFn, n: int) -> list[dict[str, Json]]:
    records = []
    for index in draw(st.lists(st.integers(0, n - 1), unique=True)):
        record: dict[str, Json] = {"index": index, **draw(_coherence())}
        if draw(st.booleans()):
            record["readout"] = draw(_readout())
        if draw(st.booleans()):
            record["disabled"] = True
        records.append(record)
    return records


@st.composite
def _calibrations(draw: st.DrawFn, n: int, arity: dict[str, int]) -> list[dict[str, Json]]:
    loci = [
        (name, qubits)
        for name, k in sorted(arity.items())
        for qubits in (
            [(q,) for q in range(n)]
            if k == 1
            else [(a, b) for a in range(n) for b in range(n) if a != b]
        )
    ]
    if not loci:
        return []
    records = []
    for name, qubits in draw(st.lists(st.sampled_from(loci), unique=True, max_size=6)):
        record = {"gate": name, "qubits": list(qubits), **draw(_spec(arity[name]))}
        if draw(st.booleans()):
            record["disabled"] = True
        records.append(record)
    return records


@st.composite
def valid_profiles(draw: st.DrawFn) -> dict[str, Json]:
    """A valid profile dict with every section the loader reads, in canonical units."""
    n = draw(st.integers(1, 5))
    device: dict[str, Json] = {
        "name": draw(st.sampled_from(["toy", "chip-7", "q.5", "dev_a"])),
        "vendor": "test",
        "technology": draw(st.sampled_from(TECHNOLOGIES)),
        "num_qubits": n,
    }
    if draw(st.booleans()):
        when = st.datetimes(
            datetime(2000, 1, 1), datetime(2100, 1, 1), timezones=st.sampled_from(OFFSETS)
        )
        device["calibrated_at"] = draw(when).isoformat()
    defs, arity = draw(_gates(n))
    data: dict[str, Json] = {
        "noisevault": "1.0",
        "device": device,
        "connectivity": draw(_connectivity(n)),
        "gates": defs,
        "qubits": draw(_qubit_records(n)),
        "calibrations": draw(_calibrations(n, arity)),
    }
    if draw(st.booleans()):
        data["readout"] = draw(_readout())
    if draw(st.booleans()):
        data["prep"] = {"error": draw(_probability)}
    if draw(st.booleans()):
        idle = draw(_coherence())
        if draw(st.booleans()):
            idle["t2_kind"] = draw(st.sampled_from(["echo", "ramsey", "cpmg"]))
        data["idle"] = idle
    if arity and draw(st.booleans()):
        gate = draw(st.sampled_from(sorted(arity)))
        data["effects"] = [{"type": "leakage", "gate": gate, "prob": draw(_probability)}]
    if draw(st.booleans()):
        data["unmodeled_error"] = {"gates": {"factor": draw(_share(10.0))}}
    if draw(st.booleans()):
        data["provenance"] = {"source": "hypothesis", "notes": ["generated"]}
    return data


# JSON text with a chosen spelling ----------------------------------------------------------


@dataclass(frozen=True)
class Raw:
    """JSON text that the writer copies as it is, such as a number in one exact spelling."""

    text: str


def write_json(
    value: Json,
    *,
    spell: Callable[[float], str] = repr,
    order: random.Random | None = None,
    repeat: JsonPath | None = None,
    path: JsonPath = (),
) -> str:
    """JSON text of ``value``.

    ``spell`` writes each float. ``order`` shuffles the keys of each object. ``repeat`` is the
    path of one key that the text writes two times.
    """

    def inner(item: Json, key: str | int) -> str:
        return write_json(item, spell=spell, order=order, repeat=repeat, path=(*path, key))

    if isinstance(value, Raw):
        return value.text
    if isinstance(value, float):
        return spell(value)
    if isinstance(value, dict):
        items = list(value.items())
        if order is not None:
            order.shuffle(items)
        parts = [f"{json.dumps(k)}: {inner(v, k)}" for k, v in items]
        if repeat is not None and repeat[:-1] == path:
            parts.insert(0, f"{json.dumps(repeat[-1])}: {inner(value[repeat[-1]], repeat[-1])}")
        return "{" + ", ".join(parts) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(inner(v, i) for i, v in enumerate(value)) + "]"
    return json.dumps(value, ensure_ascii=False)


def spellings(value: float) -> list[str]:
    """Other JSON spellings of the same float, such as 0.001, 1e-3, 1.0E-03 and 0.0010."""
    exact = Decimal(repr(value))
    sign, digits, exponent = exact.as_tuple()
    assert isinstance(exponent, int)
    mantissa = "".join(map(str, digits))
    point = len(mantissa) - 1 + exponent
    head = ("-" if sign else "") + mantissa[0]
    tail = mantissa[1:] or "0"
    found = {
        repr(value),
        repr(value).upper(),
        format(exact, "f"),
        f"{head}.{tail}e{point}",
        f"{head}.{tail}0E{'+' if point >= 0 else '-'}{abs(point):02d}",
        f"{('-' if sign else '')}{mantissa}e{exponent}",
    }
    if "." in format(exact, "f"):
        found.add(format(exact, "f") + "000")
    if value == 0:
        found |= {"0", "-0", "0.0", "-0.0", "0e5", "-0.000E-7"}
    return sorted(found)


def _respelled(seed: random.Random) -> Callable[[float], str]:
    return lambda value: seed.choice(spellings(value))


# paths into JSON data --------------------------------------------------------------------


def leaves(value: Json, path: JsonPath = ()) -> Iterator[tuple[JsonPath, Json]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, (*path, key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from leaves(item, (*path, i))
    else:
        yield path, value


def objects(value: Json, path: JsonPath = ()) -> Iterator[tuple[JsonPath, dict[str, Json]]]:
    if isinstance(value, dict):
        yield path, value
        for key, item in value.items():
            yield from objects(item, (*path, key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from objects(item, (*path, i))


def put(data: Json, path: JsonPath, value: Json) -> None:
    for key in path[:-1]:
        data = data[key]
    data[path[-1]] = value


# file helpers ----------------------------------------------------------------------------


@contextmanager
def scratch() -> Iterator[Path]:
    # Hypothesis runs many examples in one test call, so a function-scoped tmp_path is shared.
    with tempfile.TemporaryDirectory() as folder:
        yield Path(folder)


def refused_by_loaders(raw: bytes) -> list[BaseException]:
    """The error of each byte or file loader for ``raw``. Each loader must refuse ``raw``."""
    errors: list[BaseException] = []
    with scratch() as folder:
        path = folder / "p.json"
        path.write_bytes(raw)
        for read in (load_bytes, lambda _: load_file(path), lambda _: nv.load(path)):
            with pytest.raises(Exception) as caught:
                read(raw)
            assert isinstance(caught.value, LOADER_ERRORS), repr(caught.value)
            errors.append(caught.value)
    return errors


# 1. save and load ------------------------------------------------------------------------


@given(valid_profiles())
def test_saved_bytes_load_back_as_the_same_profile(data: dict[str, Json]) -> None:
    profile = Profile.model_validate(data)
    for name in ("p.json", "p.json.gz"):
        loaded = load_bytes(file_bytes(profile, Path(name)))
        assert loaded == profile
        assert loaded.fingerprint == profile.fingerprint
        assert loaded.artifact_hash == profile.artifact_hash
        assert file_bytes(loaded, Path(name)) == file_bytes(profile, Path(name))


@settings(max_examples=examples(0.3))
@given(valid_profiles())
def test_a_saved_file_loads_back_through_every_loader(data: dict[str, Json]) -> None:
    profile = Profile.model_validate(data)
    with scratch() as folder:
        for name in ("p.json", "p.json.gz"):
            path = profile.save(folder / name)
            for loaded in (load_file(path), nv.load(path), Profile.load(path)):
                assert loaded == profile
                assert loaded.fingerprint == profile.fingerprint


@given(valid_profiles())
def test_validating_the_saved_dict_gives_the_same_profile(data: dict[str, Json]) -> None:
    profile = Profile.model_validate(data)
    again = Profile.model_validate(profile.to_dict())
    assert again == profile and again.fingerprint == profile.fingerprint


# 2. fingerprint stability ----------------------------------------------------------------


@given(valid_profiles(), st.randoms(use_true_random=False))
def test_fingerprint_ignores_key_order_and_number_spelling(
    data: dict[str, Json], seed: random.Random
) -> None:
    profile = Profile.model_validate(data)
    shuffled = load_bytes(write_json(data, spell=_respelled(seed), order=seed).encode())
    assert shuffled == profile
    assert shuffled.fingerprint == profile.fingerprint
    assert shuffled.artifact_hash == profile.artifact_hash
    # The saved file keeps the order of the gates, so only a new spelling keeps the bytes.
    respelled = load_bytes(write_json(data, spell=_respelled(seed)).encode())
    assert respelled.to_json() == profile.to_json()


@pytest.mark.parametrize(
    "value", [0.0, 1e-3, 35.0, 2.5e-4, 123456.789, 5e-324, 1.7976931348623157e308]
)
def test_each_spelling_reads_as_the_same_float(value: float) -> None:
    for text in spellings(value):
        # A spelling without a point or an exponent reads as an int, which a number field takes.
        assert float(json.loads(text)) == value, text


@pytest.mark.parametrize("zero", ["-0.0", "-0", "-0.0e3"])
def test_negative_zero_and_zero_give_one_fingerprint(zero: str) -> None:
    data = toy()
    data["gates"]["sx"] = {"avg_infidelity": Raw(zero), "duration_ns": Raw(zero)}
    data["readout"] = {"error": Raw(zero)}
    data["idle"] = {"dephasing_rate_per_s": Raw(zero)}
    signed = load_bytes(write_json(data).encode())
    for section in (data["gates"]["sx"], data["readout"], data["idle"]):
        for key in section:
            section[key] = 0.0
    unsigned = Profile.model_validate(data)
    assert signed == unsigned
    assert signed.fingerprint == unsigned.fingerprint
    assert signed.to_json() == unsigned.to_json()


# 3. duplicates ---------------------------------------------------------------------------


@given(valid_profiles(), st.data())
def test_a_key_written_twice_is_refused_with_its_path(
    data: dict[str, Json], draw: st.DataObject
) -> None:
    where, obj = draw.draw(st.sampled_from([(p, o) for p, o in objects(data) if o]))
    key = draw.draw(st.sampled_from(sorted(obj)))
    raw = write_json(data, repeat=(*where, key)).encode()

    for error in refused_by_loaders(raw):
        assert isinstance(error, DuplicateKeyError)
        assert error.path == (*where, key)
    with scratch() as folder:
        path = folder / "p.json"
        path.write_bytes(raw)
        with pytest.raises(NoiseVaultError, match="twice"):
            read_json_file(path, "profile")


@given(valid_profiles(), st.data())
def test_a_record_written_twice_is_refused(data: dict[str, Json], draw: st.DataObject) -> None:
    sections = [key for key in ("qubits", "calibrations") if data[key]]
    edges = [] if data["connectivity"] == "all_to_all" else data["connectivity"]["edges"]
    if edges:
        sections.append("edges")
    if not sections:
        data["qubits"] = [{"index": 0}]
        sections = ["qubits"]
    section = draw.draw(st.sampled_from(sections))
    if section == "edges":
        a, b = draw.draw(st.sampled_from(edges))
        undirected = not data["connectivity"]["directed"]
        data["connectivity"]["edges"].append(
            [b, a] if undirected and draw.draw(st.booleans()) else [a, b]
        )
        message = "listed twice"
    else:
        first = draw.draw(st.sampled_from(data[section]))
        twin = {k: first[k] for k in ("index", "gate", "qubits") if k in first}
        data[section].append(twin)
        message = "listed twice" if section == "qubits" else "a second record"

    with pytest.raises(ValidationError, match=message):
        Profile.model_validate(data)
    for error in refused_by_loaders(json.dumps(data).encode()):
        assert isinstance(error, ValidationError)


@given(valid_profiles(), st.sampled_from(["us", "ms", "s"]))
def test_two_spellings_of_one_time_are_refused(data: dict[str, Json], unit: str) -> None:
    data["gates"]["rz"] = {"virtual": True, "duration_ns": 0, f"duration_{unit}": 0}
    with pytest.raises(ValidationError, match="duration appears twice"):
        Profile.model_validate(data)


# 4. unit aliases -------------------------------------------------------------------------

# A time with at most 9 significant digits, so each unit writes it exactly in decimal.
_decimal_time = st.builds(
    lambda digits, exponent: Decimal(digits).scaleb(exponent),
    st.integers(1, 10**9),
    st.integers(-6, 6),
)


def _time_fields(data: dict[str, Json]) -> list[tuple[JsonPath, str]]:
    """Path and stem of every canonical time field in ``data``."""
    stems = {"duration_ns": "duration", "t1_us": "t1", "t2_us": "t2"}
    return [(path, stems[path[-1]]) for path, _ in leaves(data) if path[-1] in stems]


@given(valid_profiles(), st.data())
def test_a_time_in_any_unit_gives_the_same_profile(
    data: dict[str, Json], draw: st.DataObject
) -> None:
    canonical = copy.deepcopy(data)
    aliased = copy.deepcopy(data)
    for path, stem in _time_fields(data):
        given_in = "ns" if stem == "duration" else "us"
        value = draw.draw(_decimal_time)
        unit = draw.draw(st.sampled_from(sorted(NS_EXPONENT)))
        shift = NS_EXPONENT[given_in] - NS_EXPONENT[unit]
        put(canonical, path, Raw(str(value)))
        parent = aliased
        for key in path[:-1]:
            parent = parent[key]
        del parent[path[-1]]
        parent[f"{stem}_{unit}"] = Raw(str(value.scaleb(shift)))

    expected = load_bytes(write_json(canonical).encode())
    got = load_bytes(write_json(aliased).encode())
    assert got == expected
    assert got.fingerprint == expected.fingerprint


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ('"duration_us": 1.001', '"duration_ns": 1001'),
        ('"duration_ms": 1.3E-8', '"duration_ns": 0.013'),
        ('"duration_s": 1.1E-11', '"duration_ns": 0.011'),
    ],
)
def test_a_decimal_duration_converts_exactly(alias: str, canonical: str) -> None:
    def profile(duration: str) -> Profile:
        data = toy()
        data["gates"]["sx"] = {"avg_infidelity": 1e-3, "duration_ns": Raw("@")}
        return load_bytes(write_json(data).replace('"duration_ns": @', duration).encode())

    assert profile(alias).gates["sx"].duration_ns == profile(canonical).gates["sx"].duration_ns
    assert profile(alias).fingerprint == profile(canonical).fingerprint


def test_a_decimal_coherence_time_converts_exactly() -> None:
    def fingerprint(idle: dict[str, Json]) -> str:
        return Profile.model_validate(toy(idle=idle)).fingerprint

    assert fingerprint({"t1_ms": 1.001, "t2_ns": 1009}) == fingerprint(
        {"t1_us": 1001, "t2_us": 1.009}
    )


# 5. almost-valid profiles ----------------------------------------------------------------

# Replacements of each JSON type by a value of another type. None is not here: an optional
# field reads null as absent.
_WRONG_TYPE: dict[type, list[Json]] = {
    bool: ["true", 1, [True]],
    int: ["1", 1.5, True, [1]],
    float: ["0.001", True, [0.5], {"value": 0.5}],
    str: [0, True, ["text"], {}],
}


def _drop_required(data: dict[str, Json], draw: st.DataObject) -> None:
    path = draw.draw(
        st.sampled_from(
            [("noisevault",), ("device",), ("connectivity",), ("gates",)]
            + [("device", key) for key in ("name", "technology", "num_qubits")]
        )
    )
    parent = data
    for key in path[:-1]:
        parent = parent[key]
    del parent[path[-1]]


def _wrong_type(data: dict[str, Json], draw: st.DataObject) -> None:
    path, value = draw.draw(st.sampled_from(list(leaves(data))))
    put(data, path, draw.draw(st.sampled_from(_WRONG_TYPE[type(value)])))


def _out_of_range(data: dict[str, Json], draw: st.DataObject) -> None:
    gate = draw.draw(st.sampled_from(sorted(data["gates"])))
    choice = draw.draw(
        st.sampled_from(
            [
                ("gates", gate, "avg_infidelity", 1.5),
                ("gates", gate, "process_infidelity", 1.0001),
                ("gates", gate, "depolarizing_param", 1.5),
                ("gates", gate, "duration_ns", -1e-9),
                ("readout", "error", 1.0000001),
                ("prep", "error", -0.25),
                ("idle", "t1_us", 0),
                ("idle", "t2_us", -3.0),
                ("device", "num_qubits", 0),
            ]
        )
    )
    *path, value = choice
    if path[0] in ("readout", "prep", "idle"):
        data[path[0]] = {}
    if path[0] == "gates":  # one metric at a time, so the range is the only fault
        data["gates"][gate] = {"virtual": True} if gate == "rz" else {}
    put(data, tuple(path), value)


def _nonfinite(data: dict[str, Json], draw: st.DataObject) -> None:
    floats = [path for path, value in leaves(data) if isinstance(value, float)]
    token = draw.draw(st.sampled_from(["NaN", "Infinity", "-Infinity", "1e999", "-1e999"]))
    path = draw.draw(st.sampled_from(floats)) if floats else ("gates", "rz", "duration_ns")
    put(data, path, Raw(token))


def _wrong_pauli_length(data: dict[str, Json], draw: st.DataObject) -> None:
    gate = draw.draw(st.sampled_from(sorted(data["gates"])))
    size = draw.draw(st.sampled_from([0, 1, 2, 4, 14, 16]))
    data["gates"][gate] = {"pauli": [0.0] * size}


def _unknown_qubit(data: dict[str, Json], draw: st.DataObject) -> None:
    n = data["device"]["num_qubits"]
    where = draw.draw(st.sampled_from(["record", "calibration", "edge", "effect"]))
    outside = draw.draw(st.integers(n, n + 3))
    if where == "record":
        data["qubits"] = [{"index": outside}]
    elif where == "calibration":
        data["calibrations"] = [{"gate": "rz", "qubits": [outside], "disabled": True}]
    elif where == "edge":
        data["connectivity"] = {"edges": [[0, outside]]}
    else:
        data["effects"] = [{"type": "crosstalk_zz", "on": "idle", "qubits": [0, outside]}]


def _bad_edge(data: dict[str, Json], draw: st.DataObject) -> None:
    edge = draw.draw(st.sampled_from([[0, 0], [0], [0, 1, 2], [-1, 0], [], "0-1", [0.0, 1]]))
    data["connectivity"] = {"edges": [edge]}


def _extra_key(data: dict[str, Json], draw: st.DataObject) -> None:
    where, obj = draw.draw(
        st.sampled_from([(p, o) for p, o in objects(data) if p[:1] != ("provenance",)])
    )
    obj["colour"] = "blue"


def _deep_nesting(data: dict[str, Json], draw: st.DataObject) -> None:
    depth = draw.draw(st.integers(65, 400))
    nested: Json = 0
    for _ in range(depth):
        nested = [nested]
    if draw.draw(st.booleans()):
        data["extensions"] = {"deep": nested}
    else:
        path, _ = draw.draw(st.sampled_from(list(leaves(data))))
        put(data, path, nested)


def _huge_number(data: dict[str, Json], draw: st.DataObject) -> None:
    floats = [path for path, value in leaves(data) if isinstance(value, float)]
    path = draw.draw(st.sampled_from(floats)) if floats else ("gates", "rz", "duration_ns")
    put(data, path, Raw("1" + "0" * draw.draw(st.integers(309, 1000))))


def _time_out_of_range(data: dict[str, Json], draw: st.DataObject) -> None:
    stamp = draw.draw(st.sampled_from(["0001-01-01T00:30:00+01:00", "9999-12-31T23:59:59-00:01"]))
    data["device"]["calibrated_at"] = stamp


def _unknown_gate(data: dict[str, Json], draw: st.DataObject) -> None:
    data["calibrations"] = [{"gate": "nope", "qubits": [0], "avg_infidelity": 0.01}]


def _two_metrics(data: dict[str, Json], draw: st.DataObject) -> None:
    gate = draw.draw(st.sampled_from(sorted(data["gates"])))
    data["gates"][gate] = {"avg_infidelity": 0.01, "process_infidelity": 0.01}


def _old_format_marker(data: dict[str, Json], draw: st.DataObject) -> None:
    data["schema_version"] = "0.1"


MUTATIONS = [
    _drop_required,
    _wrong_type,
    _out_of_range,
    _nonfinite,
    _wrong_pauli_length,
    _unknown_qubit,
    _bad_edge,
    _extra_key,
    _deep_nesting,
    _huge_number,
    _time_out_of_range,
    _unknown_gate,
    _two_metrics,
    _old_format_marker,
]


@given(valid_profiles(), st.sampled_from(MUTATIONS), st.data())
def test_an_almost_valid_profile_is_refused_with_a_loader_error(
    data: dict[str, Json],
    mutate: Callable[[dict[str, Json], st.DataObject], None],
    draw: st.DataObject,
) -> None:
    mutate(data, draw)
    refused_by_loaders(write_json(data).encode())


@given(valid_profiles(), st.sampled_from(MUTATIONS), st.data())
def test_model_validate_refuses_an_almost_valid_dict_with_a_validation_error(
    data: dict[str, Json],
    mutate: Callable[[dict[str, Json], st.DataObject], None],
    draw: st.DataObject,
) -> None:
    mutate(data, draw)
    parsed = json.loads(write_json(data))
    with pytest.raises(ValidationError):
        Profile.model_validate(parsed)


@given(valid_profiles(), st.data())
def test_a_damaged_file_is_refused_by_every_loader(
    data: dict[str, Json], draw: st.DataObject
) -> None:
    text = write_json(data).encode()
    damage = draw.draw(st.sampled_from(["cut", "bad utf-8", "gzip cut", "gzip flip", "top level"]))
    if damage == "cut":
        raw = text[: draw.draw(st.integers(0, len(text) - 1))]
    elif damage == "bad utf-8":
        at = draw.draw(st.sampled_from([i for i in range(len(text)) if text[i : i + 1] == b'"']))
        bad = draw.draw(st.sampled_from([b"\xff", b"\xc3", b"\xc0\xaf", b"\x80"]))
        raw = text[: at + 1] + bad + text[at + 1 :]
    elif damage == "gzip cut":
        packed = gzip.compress(text)
        raw = packed[: draw.draw(st.integers(2, len(packed) - 1))]
    elif damage == "gzip flip":
        packed = bytearray(gzip.compress(text))
        # A change in the data can leave the text the same, so change the CRC-32 or the size.
        at = draw.draw(st.integers(len(packed) - 8, len(packed) - 1))
        packed[at] ^= draw.draw(st.integers(1, 255))
        raw = bytes(packed)
    else:
        raw = write_json(draw.draw(st.sampled_from([[data], "profile", 1, None, True]))).encode()

    errors = refused_by_loaders(raw)
    if damage == "top level":
        assert all(isinstance(e, ValidationError) for e in errors)
        return
    with scratch() as folder:
        path = folder / "p.json"
        path.write_bytes(raw)
        with pytest.raises(NoiseVaultError):
            read_json_file(path, "profile")


def test_an_encoded_surrogate_is_refused_by_the_profile_validator() -> None:
    # json.loads reads these bytes with "surrogatepass", so the JSON reader accepts them.
    raw = write_json(toy()).encode().replace(b'"toy"', b'"t\xed\xa0\x80y"')
    for error in refused_by_loaders(raw):
        assert isinstance(error, ValidationError)
        assert "unpaired surrogate" in str(error)


@pytest.mark.parametrize(
    "free", [{"a\nb": float("nan")}, {"x": {"a\rb": [1, float("inf")]}}, {"k\nk": "\ud800"}]
)
def test_a_free_form_key_with_a_line_break_gives_a_one_line_error(free: dict[str, Json]) -> None:
    with pytest.raises(ValidationError) as caught:
        Profile.model_validate(toy(extensions=free))
    [error] = caught.value.errors()
    assert "\n" not in error["msg"] and "\r" not in error["msg"], error["msg"]


def test_a_file_nested_past_the_parser_is_refused_by_every_loader() -> None:
    raw = write_json(toy(extensions={"deep": Raw(deeper_than_the_parser_takes())})).encode()
    for error in refused_by_loaders(raw):
        assert isinstance(error, json.JSONDecodeError)


# regressions found by the properties ------------------------------------------------------


@pytest.mark.parametrize("where", ["gates", "idle", "qubits", "readout"])
def test_an_integer_too_large_for_a_float_is_a_validation_error(where: str) -> None:
    huge = 10**400
    sections = {
        "gates": {"gates": {**toy()["gates"], "sx": {"duration_ns": huge}}},
        "idle": {"idle": {"t1_us": huge}},
        "qubits": {"qubits": [{"index": 0, "t2_ms": huge}]},
        "readout": {"readout": {"error": 0.01, "duration_us": huge}},
    }
    with pytest.raises(ValidationError, match="must be finite"):
        Profile.model_validate(toy(**sections[where]))


@pytest.mark.parametrize("stamp", ["0001-01-01T00:30:00+01:00", "9999-12-31T23:59:59-00:01"])
def test_a_time_outside_the_utc_calendar_is_a_validation_error(stamp: str) -> None:
    with pytest.raises(ValidationError, match="calibrated_at"):
        Profile.model_validate(toy(device={**toy()["device"], "calibrated_at": stamp}))


def test_a_pauli_entry_too_large_to_sum_is_a_validation_error() -> None:
    gates = {**toy()["gates"], "sx": {"pauli": [1e308, 1e308, 1e308]}}
    with pytest.raises(ValidationError, match="<= 1"):
        Profile.model_validate(toy(gates=gates))


def test_a_gate_on_more_qubits_than_the_device_has_is_a_validation_error() -> None:
    # Before, the metric check computed 2**qubits and did not stop.
    gates = {**toy()["gates"], "big": {"qubits": 10**20, "avg_infidelity": 0.01}}
    with pytest.raises(ValidationError, match="the device has 3"):
        Profile.model_validate(toy(gates=gates))


def test_a_unit_alias_converts_the_same_in_any_decimal_context() -> None:
    data = toy(gates={**toy()["gates"], "sx": {"avg_infidelity": 1e-3, "duration_us": 1.001}})
    expected = Profile.model_validate(data)
    with decimal.localcontext() as context:
        context.prec = 3
        context.traps[decimal.Inexact] = True
        assert Profile.model_validate(data) == expected
    assert expected.gates["sx"].duration_ns == 1001.0
