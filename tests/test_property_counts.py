"""Property tests for counts format 1.0: ``load_counts``, ``save``, ``plan`` and ``simulate``.

Each rule that these tests check comes from docs/counts-format.md or from the models in
``noisevault.counts``. A refusal is one ``CountsError`` line that starts with the path.
"""

from __future__ import annotations

import copy
import gzip
import json
import re
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta, timezone
from functools import cache
from pathlib import Path
from random import Random
from typing import Any

import numpy as np
import pytest
from conftest import deeper_than_the_parser_takes, examples, toy
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from noisevault import gates
from noisevault.counts import (
    SAMPLER_V2_OPTIONS,
    MeasuredCounts,
    PlannedCircuit,
    load_counts,
    plan,
    simulate,
)
from noisevault.errors import CountsError
from noisevault.profile import Profile
from noisevault.reference import Op, probabilities

GATES = {name: info for name, info in gates.GATES.items() if info.unitary is not None}
OP_NAMES = frozenset({*GATES, "delay"})
MAX_WIDTH = 4
MAX_SHOTS = 10**10
FLAGS = {
    "transpiled": False,
    "gate_twirling": False,
    "measure_twirling": False,
    "dynamical_decoupling": False,
    "init_qubits": True,
}
FIELDS = {
    "top": ("nv_counts", "source", "profile", "backend", "run_at", "bit_order", "execution"),
    "profile": ("id", "fingerprint"),
    "execution": ("client", *FLAGS, "job_ids", "options"),
    "circuit": ("name", "qubits", "ops", "shots", "counts"),
}
REQUIRED = {**FIELDS, "top": (*FIELDS["top"], "circuits"), "execution": ("client", *FLAGS)}
RUN_AT = datetime(2026, 4, 16, 9, 30, 2, tzinfo=UTC)

TEXT = st.text(st.characters(exclude_categories=["Cs"]), max_size=8)
FINITE = st.floats(allow_nan=False, allow_infinity=False)
ANGLES = FINITE | st.integers(-(2**53), 2**53)
DURATIONS = st.floats(min_value=0.0, allow_infinity=False) | st.integers(0, 2**53) | st.just(-0.0)
FINGERPRINTS = st.text("0123456789abcdef", min_size=64, max_size=64)
OFFSETS = st.integers(-(23 * 60 + 59), 23 * 60 + 59).map(lambda m: timezone(timedelta(minutes=m)))
TIMES = st.datetimes(datetime(1900, 1, 2), datetime(2200, 1, 1), timezones=OFFSETS)
OPTION_VALUES = st.recursive(
    st.none() | st.booleans() | st.integers(-(10**20), 10**20) | FINITE | TEXT,
    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(TEXT, inner, max_size=3),
    max_leaves=6,
)
# Any JSON value, NaN and infinity included, because Python's json module reads them.
WILD = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats() | st.text(max_size=8),
    lambda inner: (
        st.lists(inner, max_size=3) | st.dictionaries(st.text(max_size=8), inner, max_size=3)
    ),
    max_leaves=6,
)


def _key(index: int, width: int, bit_order: str) -> str:
    """The key of outcome ``index`` in ``bit_order``. Circuit qubit 0 is the top bit of index."""
    key = format(index, f"0{width}b")
    return key[::-1] if bit_order == "qiskit" else key


@st.composite
def ops(draw: st.DrawFn, width: int) -> list[Any]:
    fits = sorted(n for n in OP_NAMES if n == "delay" or GATES[n].arity <= width)
    name = draw(st.sampled_from(fits))
    if name == "delay":
        return ["delay", [draw(st.integers(0, width - 1))], [draw(DURATIONS)]]
    info = GATES[name]
    qubits = list(draw(st.permutations(range(width))))[: info.arity]
    return [name, qubits, [draw(ANGLES) for _ in info.params]]


@st.composite
def circuits(draw: st.DrawFn, name: str, bit_order: str) -> dict[str, Any]:
    width = draw(st.integers(1, MAX_WIDTH))
    outcomes = draw(st.lists(st.integers(0, 2**width - 1), min_size=1, unique=True))
    # The first outcome has at least one shot, because shots is at least 1.
    tallies = [draw(st.integers(1, 10**9)), *(draw(st.integers(0, 10**9)) for _ in outcomes[1:])]
    qubits = draw(st.lists(st.integers(0, 1000), min_size=width, max_size=width, unique=True))
    return {
        "name": name,
        "qubits": qubits,
        "ops": draw(st.lists(ops(width), max_size=6)),
        "shots": sum(tallies),
        "counts": {_key(i, width, bit_order): k for i, k in zip(outcomes, tallies, strict=True)},
    }


def _nest(path: str, value: Any) -> dict[str, Any]:
    head, _, rest = path.partition(".")
    return {head: _nest(rest, value) if rest else value}


def _required_options() -> dict[str, Any]:
    options: dict[str, Any] = {}
    for path, value in SAMPLER_V2_OPTIONS.items():
        head, _, rest = path.partition(".")
        options.setdefault(head, {}).update(_nest(rest, value))
    return options


@st.composite
def executions(draw: st.DrawFn) -> dict[str, Any]:
    reserved = {path.partition(".")[0] for path in SAMPLER_V2_OPTIONS}
    keys = TEXT.filter(lambda key: key not in reserved)
    options = draw(st.dictionaries(keys, OPTION_VALUES, max_size=3))
    if draw(st.booleans()):
        options.update(_required_options())
    execution: dict[str, Any] = {"client": draw(TEXT), **FLAGS}
    if draw(st.booleans()):
        execution["job_ids"] = draw(st.lists(TEXT, max_size=3))
    if draw(st.booleans()):
        execution["options"] = options
    return execution


@st.composite
def counts_files(draw: st.DrawFn) -> dict[str, Any]:
    bit_order = draw(st.sampled_from(["clbit0_left", "qiskit"]))
    names = draw(st.lists(TEXT, min_size=1, max_size=3, unique=True))
    run_at = draw(TIMES).isoformat()
    return {
        "nv_counts": "1.0",
        "source": draw(st.sampled_from(["hardware", "simulated"])),
        "profile": {"id": draw(TEXT), "fingerprint": draw(FINGERPRINTS)},
        "backend": draw(TEXT),
        "run_at": run_at.replace("+00:00", "Z") if draw(st.booleans()) else run_at,
        "bit_order": bit_order,
        "execution": draw(executions()),
        "circuits": [draw(circuits(name, bit_order)) for name in names],
    }


@contextmanager
def _file(raw: bytes, name: str = "run.counts.json") -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / name
        path.write_bytes(raw)
        yield path


def _bytes(data: Any) -> bytes:
    return json.dumps(data).encode("utf-8")


def _load(raw: bytes) -> MeasuredCounts:
    with _file(raw) as path:
        return load_counts(path)


def _refusal(raw: bytes, name: str = "run.counts.json") -> CountsError:
    with _file(raw, name) as path:
        with pytest.raises(CountsError) as caught:
            load_counts(path)
    text = str(caught.value)
    assert text.startswith(str(path)), text
    assert "\n" not in text and "\r" not in text, text
    return caught.value


def _number(value: float, rng: Random) -> str:
    """One JSON spelling of ``value``. Each spelling names the same number."""
    forms = [repr(value), f"{value:.17e}", f"{value:.17E}"]
    if value.is_integer() and abs(value) < 2**53:
        forms.append("-0" if str(value) == "-0.0" else str(int(value)))
    return rng.choice(forms)


def _spelled(value: Any, rng: Random, path: tuple[str | int, ...] = ()) -> str:
    """JSON text of ``value`` with shuffled keys and a random spelling of each op parameter."""
    if isinstance(value, dict):
        items = list(value.items())
        rng.shuffle(items)
        pairs = (f"{json.dumps(k)}: {_spelled(v, rng, (*path, k))}" for k, v in items)
        return "{" + ", ".join(pairs) + "}"
    if isinstance(value, list):
        inner = (_spelled(v, rng, (*path, i)) for i, v in enumerate(value))
        return "[" + ", ".join(inner) + "]"
    is_param = len(path) == 6 and path[0] == "circuits" and path[2] == "ops" and path[4] == 2
    if is_param and isinstance(value, int | float) and not isinstance(value, bool):
        return _number(float(value), rng)
    return json.dumps(value)


def _flipped(data: dict[str, Any]) -> dict[str, Any]:
    """The same run in the other bit order."""
    other = copy.deepcopy(data)
    other["bit_order"] = "qiskit" if data["bit_order"] == "clbit0_left" else "clbit0_left"
    for circuit in other["circuits"]:
        circuit["counts"] = {key[::-1]: tally for key, tally in circuit["counts"].items()}
    return other


@given(counts_files(), st.booleans())
def test_a_valid_file_loads_and_its_saved_form_loads_back_equal(
    data: dict[str, Any], packed: bool
) -> None:
    raw = _bytes(data)
    name = "run.counts.json.gz" if packed else "run.counts.json"
    with _file(gzip.compress(raw) if packed else raw, name) as path:
        loaded = load_counts(path)
        saved = loaded.save(path.with_name("saved.counts.json"))
        again = load_counts(saved)
        twice = again.save(path.with_name("twice.counts.json"))

        assert again == loaded and again.sha256 == loaded.sha256
        assert twice.read_bytes() == saved.read_bytes()
    assert MeasuredCounts.model_validate(data) == loaded


@given(counts_files(), st.randoms(use_true_random=False))
def test_key_order_and_number_spelling_leave_the_loaded_file_unchanged(
    data: dict[str, Any], rng: Random
) -> None:
    plain = _load(_bytes(data))
    spelled = _load(_spelled(data, rng).encode("utf-8"))

    assert spelled == plain
    assert spelled.sha256 == plain.sha256


@given(counts_files())
def test_both_bit_orders_of_one_run_load_to_one_file(data: dict[str, Any]) -> None:
    assert _load(_bytes(_flipped(data))).sha256 == _load(_bytes(data)).sha256


@given(counts_files())
def test_a_loaded_file_holds_the_run_in_canonical_form(data: dict[str, Any]) -> None:
    loaded = _load(_bytes(data))

    assert loaded.bit_order == "clbit0_left"
    for written, circuit in zip(data["circuits"], loaded.circuits, strict=True):
        width = len(circuit.qubits)
        reverse = data["bit_order"] == "qiskit"
        expected = {(k[::-1] if reverse else k): v for k, v in written["counts"].items() if v}
        assert dict(circuit.counts) == expected
        assert list(circuit.counts) == sorted(circuit.counts)
        assert all(len(key) == width and set(key) <= {"0", "1"} for key in circuit.counts)
        assert sum(circuit.counts.values()) == circuit.shots == written["shots"]
        vector = circuit.vector()
        assert vector.sum() == circuit.shots
        assert all(vector[int(key, 2)] == tally for key, tally in circuit.counts.items())
        assert circuit.qubits == tuple(written["qubits"])
        wire = [(name, tuple(q), tuple(float(p) for p in ps)) for name, q, ps in written["ops"]]
        assert [(op.name, op.qubits, op.params) for op in circuit.ops] == wire


# Almost-valid files. Each mutation breaks one rule of docs/counts-format.md in a valid file.

Mutation = Callable[[dict[str, Any], st.DataObject], Any]
BAD_COUNTS = (
    st.integers(max_value=-1)
    | st.integers(10**30, 10**60)
    | st.floats()
    | st.booleans()
    | st.none()
    | st.text(max_size=4)
    | st.lists(st.integers(0, 9), max_size=2)
)
NOT_INTS = st.floats() | st.booleans() | st.none() | st.text(max_size=4) | st.just([1])


def _circuit(data: dict[str, Any], draw: st.DataObject) -> dict[str, Any]:
    circuit: dict[str, Any] = draw.draw(st.sampled_from(data["circuits"]))
    return circuit


def _outcome(data: dict[str, Any], draw: st.DataObject) -> tuple[dict[str, Any], str]:
    circuit = _circuit(data, draw)
    return circuit, draw.draw(st.sampled_from(sorted(circuit["counts"])))


def _key_of_wrong_width(data: dict[str, Any], draw: st.DataObject) -> Any:
    circuit, key = _outcome(data, draw)
    wrong = key + draw.draw(st.sampled_from("01")) if draw.draw(st.booleans()) else key[:-1]
    circuit["counts"][wrong] = circuit["counts"].pop(key)
    return data


def _key_with_a_foreign_character(data: dict[str, Any], draw: st.DataObject) -> Any:
    circuit, key = _outcome(data, draw)
    at = draw.draw(st.integers(0, len(key) - 1))
    char = draw.draw(st.characters().filter(lambda c: c not in "01"))
    circuit["counts"][key[:at] + char + key[at + 1 :]] = circuit["counts"].pop(key)
    return data


def _bad_count(data: dict[str, Any], draw: st.DataObject) -> Any:
    circuit, key = _outcome(data, draw)
    circuit["counts"][key] = draw.draw(BAD_COUNTS)
    return data


def _counts_off_shots(data: dict[str, Any], draw: st.DataObject) -> Any:
    circuit = _circuit(data, draw)
    shift = draw.draw(st.integers(-5, 5).filter(bool))
    circuit["shots"] = circuit["shots"] + shift
    return data


def _bad_shots(data: dict[str, Any], draw: st.DataObject) -> Any:
    circuit = _circuit(data, draw)
    over = MAX_SHOTS + draw.draw(st.integers(1, 10**12))
    shots = draw.draw(st.integers(max_value=0) | NOT_INTS | st.just(over))
    circuit["shots"] = shots
    if shots == over:
        circuit["counts"] = {next(iter(circuit["counts"])): over}
    return data


def _bad_qubits(data: dict[str, Any], draw: st.DataObject) -> Any:
    circuit = _circuit(data, draw)
    qubits = circuit["qubits"]
    repeated = [*qubits[:-1], qubits[0]] if len(qubits) > 1 else [*qubits, qubits[0]]
    choices = [[], repeated, list(range(11)), [-1], [True], [0.0], ["0"], None, 0]
    circuit["qubits"] = draw.draw(st.sampled_from(choices))
    return data


@st.composite
def bad_ops(draw: st.DrawFn, width: int) -> Any:
    gate = draw(st.sampled_from(sorted(GATES)))
    info = GATES[gate]
    arity = draw(st.integers(0, 4).filter(lambda n: n != info.arity))
    params = [0.5] * len(info.params)
    wrong_params = [0.5] * draw(st.integers(0, 4).filter(lambda n: n != len(info.params)))
    with_params = draw(st.sampled_from(sorted(n for n, i in GATES.items() if i.params)))
    nonfinite = draw(st.sampled_from([float("nan"), float("inf"), -float("inf"), 10**400]))
    nonfinite_params = [nonfinite] * len(GATES[with_params].params)
    unknown = draw(TEXT.filter(lambda name: name not in OP_NAMES))
    qubit = draw(st.sampled_from([-1, True, 0.0, "0", None]))
    choices = [
        [gate, list(range(arity)), params],
        [gate, [0] * info.arity, params] if info.arity > 1 else [gate, [], params],
        [gate, list(range(info.arity)), wrong_params],
        [with_params, list(range(GATES[with_params].arity)), nonfinite_params],
        [unknown, [0], []],
        ["measure", [0], []],
        ["reset", [0], []],
        [gate.upper(), list(range(info.arity)), params] if gate.upper() != gate else [gate],
        ["rz", [width + draw(st.integers(0, 5))], [0.5]],
        ["sx", [qubit], []],
        ["rz", [0], [draw(st.sampled_from([True, "1", None, [0.5]]))]],
        ["delay", [0], [draw(st.floats(max_value=-1e-300) | st.just(float("nan")))]],
        ["delay", [0], [float("inf")]],
        ["delay", [0, 1], [1.0]],
        ["delay", [], [1.0]],
        ["delay", [0], []],
        ["delay", [0], [1.0, 2.0]],
        0,
        "sx",
        [],
        ["sx", [0]],
        ["sx", [0], [], []],
        ["sx", 0, []],
        ["sx", [0], 0],
        {"name": "sx", "qubits": [0], "params": []},
    ]
    return draw(st.sampled_from(choices))


def _bad_op(data: dict[str, Any], draw: st.DataObject) -> Any:
    circuit = _circuit(data, draw)
    at = draw.draw(st.integers(0, len(circuit["ops"])))
    circuit["ops"].insert(at, draw.draw(bad_ops(len(circuit["qubits"]))))
    return data


def _bad_flag(data: dict[str, Any], draw: st.DataObject) -> Any:
    flag = draw.draw(st.sampled_from(sorted(FLAGS)))
    data["execution"][flag] = draw.draw(
        st.sampled_from([not FLAGS[flag], 0, 1, "false", "true", None, [], 0.0])
    )
    return data


def _contradicting_option(data: dict[str, Any], draw: st.DataObject) -> Any:
    path, required = draw.draw(st.sampled_from(sorted(SAMPLER_V2_OPTIONS.items())))
    value = draw.draw(WILD.filter(lambda v: type(v) is not type(required) or v != required))
    options = data["execution"].setdefault("options", {})
    head, _, rest = path.partition(".")
    options.setdefault(head, {}).update(_nest(rest, value))
    return data


def _bad_execution_value(data: dict[str, Any], draw: st.DataObject) -> Any:
    field, value = draw.draw(
        st.sampled_from(
            [
                ("client", None),
                ("client", 1),
                ("client", ["q"]),
                ("job_ids", "job"),
                ("job_ids", [1]),
                ("job_ids", {"a": "b"}),
                ("options", []),
                ("options", "x"),
                ("options", None),
            ]
        )
    )
    data["execution"][field] = value
    return data


def _region(data: dict[str, Any], draw: st.DataObject) -> tuple[str, dict[str, Any]]:
    region = draw.draw(st.sampled_from(sorted(FIELDS)))
    found = {"top": data, "profile": data["profile"], "execution": data["execution"]}
    return region, found.get(region) or _circuit(data, draw)


def _missing_key(data: dict[str, Any], draw: st.DataObject) -> Any:
    region, node = _region(data, draw)
    node.pop(draw.draw(st.sampled_from(REQUIRED[region])))
    return data


def _unknown_key(data: dict[str, Any], draw: st.DataObject) -> Any:
    region, node = _region(data, draw)
    known = {*FIELDS[region], "circuits"}
    node[draw.draw(st.text(max_size=8).filter(lambda k: k not in known))] = draw.draw(WILD)
    return data


def _bad_literal(data: dict[str, Any], draw: st.DataObject) -> Any:
    field, allowed = draw.draw(
        st.sampled_from(
            [
                ("nv_counts", {"1.0"}),
                ("source", {"hardware", "simulated"}),
                ("bit_order", {"qiskit", "clbit0_left"}),
            ]
        )
    )
    data[field] = draw.draw(
        (st.text(max_size=8) | WILD).filter(lambda v: not isinstance(v, str) or v not in allowed)
    )
    return data


def _bad_fingerprint(data: dict[str, Any], draw: st.DataObject) -> Any:
    full = re.compile(r"[0-9a-f]{64}")
    wrong = st.text(max_size=70).filter(lambda t: not full.fullmatch(t))
    data["profile"]["fingerprint"] = draw.draw(wrong | NOT_INTS | st.integers())
    return data


def _bad_run_at(data: dict[str, Any], draw: st.DataObject) -> Any:
    naive = draw.draw(st.datetimes(datetime(1900, 1, 1), datetime(2200, 1, 1))).isoformat()
    data["run_at"] = draw.draw(
        st.sampled_from([naive, "20260930", "2026-02-30T00:00:00Z", "yesterday", "", None])
        | st.integers()
        | st.floats()
    )
    return data


def _bad_circuits(data: dict[str, Any], draw: st.DataObject) -> Any:
    first = data["circuits"][0]
    wide = {"qubits": list(range(10)), "ops": [], "shots": 1, "counts": {"0" * 10: 1}}
    data["circuits"] = draw.draw(
        st.sampled_from(
            [
                [],
                {},
                None,
                "circuits",
                [*data["circuits"], copy.deepcopy(first)],
                [{"name": f"c{i}", **wide} for i in range(5)],
                [*data["circuits"], None],
                [*data["circuits"], []],
            ]
        )
    )
    return data


def _section_of_wrong_type(data: dict[str, Any], draw: st.DataObject) -> Any:
    field = draw.draw(st.sampled_from(["profile", "execution", "circuits", "counts", "ops"]))
    value = draw.draw(st.sampled_from([None, 1, "x", True, [[]]]))
    if field in ("counts", "ops"):
        _circuit(data, draw)[field] = value
    else:
        data[field] = value
    return data


def _wrong_top_level(data: dict[str, Any], draw: st.DataObject) -> Any:
    return draw.draw(st.sampled_from([[data], "nv_counts", 1.0, None, True, {}]))


def _repeated_key(data: dict[str, Any], draw: st.DataObject) -> Any:
    marker = "\0repeat"
    circuit = _circuit(data, draw)
    node = draw.draw(
        st.sampled_from([data, data["profile"], data["execution"], circuit, circuit["counts"]])
    )
    node[marker] = draw.draw(WILD)
    key = draw.draw(st.sampled_from(sorted(k for k in node if k != marker)))
    return _bytes(data).replace(json.dumps(marker).encode(), json.dumps(key).encode(), 1)


def _not_utf8(data: dict[str, Any], draw: st.DataObject) -> Any:
    raw = _bytes(data)
    at = draw.draw(st.integers(0, len(raw)))
    bad = draw.draw(st.sampled_from([b"\xff", b"\x80", b"\xc0\x80", b"\xed\xa0\x80", b"\xe2\x82"]))
    return raw[:at] + bad + raw[at:]


def _truncated(data: dict[str, Any], draw: st.DataObject) -> Any:
    raw = _bytes(data)
    if draw.draw(st.booleans()):
        packed = gzip.compress(raw)
        return packed[: draw.draw(st.integers(0, len(packed) - 1))]
    return raw[: draw.draw(st.integers(0, len(raw) - 1))]


def _blank(data: dict[str, Any], draw: st.DataObject) -> Any:
    return draw.draw(st.sampled_from([b"", b" ", b"\n", b"\xef\xbb\xbf", b"null", b"\x1f\x8b"]))


@cache
def _too_deep() -> int:
    return len(deeper_than_the_parser_takes()) // 2


def _parser_limit() -> int:
    """The deepest array nesting that ``json.loads`` reads from this place in the stack."""
    low, high = _too_deep() // 2, _too_deep()
    while low < high:
        middle = (low + high + 1) // 2
        try:
            json.loads("[" * middle + "]" * middle)
            low = middle
        except RecursionError:
            high = middle - 1
    return low


def _deep(data: dict[str, Any], draw: st.DataObject) -> Any:
    marker = "\0deep"
    path = draw.draw(st.sampled_from(list(_paths(data))))
    _put(data, path, marker)
    # The stack here is not the stack of load_counts, so try a range of depths near the limit.
    limit = _parser_limit()
    depth = draw.draw(st.sampled_from([65, 1000]) | st.integers(limit - 25, limit + 1))
    nested = ("[" * depth + "]" * depth).encode()
    return _bytes(data).replace(json.dumps(marker).encode(), nested)


MUTATIONS: dict[str, Mutation] = {
    "key of wrong width": _key_of_wrong_width,
    "key with a character other than 0 and 1": _key_with_a_foreign_character,
    "count not a non-negative integer": _bad_count,
    "counts that do not add up to shots": _counts_off_shots,
    "shots not a count": _bad_shots,
    "qubits not distinct physical qubits": _bad_qubits,
    "op that breaks the op rules": _bad_op,
    "execution flag other than the required value": _bad_flag,
    "SamplerV2 option other than the required value": _contradicting_option,
    "execution field of the wrong type": _bad_execution_value,
    "required key missing": _missing_key,
    "unknown key": _unknown_key,
    "version, source or bit order not allowed": _bad_literal,
    "fingerprint not 64 hex digits": _bad_fingerprint,
    "run_at not a time with a timezone": _bad_run_at,
    "circuits not a valid list": _bad_circuits,
    "section of the wrong type": _section_of_wrong_type,
    "top level not an object": _wrong_top_level,
    "key twice in one object": _repeated_key,
    "bytes that are not UTF-8": _not_utf8,
    "truncated file": _truncated,
    "empty file": _blank,
    "deep nesting": _deep,
}


@pytest.mark.parametrize("mutation", sorted(MUTATIONS))
@settings(max_examples=examples(0.5))
@given(data=counts_files(), draw=st.data())
def test_an_almost_valid_file_is_refused_with_one_counts_error_line(
    mutation: str, data: dict[str, Any], draw: st.DataObject
) -> None:
    changed = MUTATIONS[mutation](copy.deepcopy(data), draw)

    _refusal(changed if isinstance(changed, bytes) else _bytes(changed))


def _paths(value: Any, path: tuple[str | int, ...] = ()) -> Iterator[tuple[str | int, ...]]:
    if path:
        yield path
    if isinstance(value, dict | list):
        for key, child in value.items() if isinstance(value, dict) else enumerate(value):
            yield from _paths(child, (*path, key))


def _put(data: Any, path: tuple[str | int, ...], value: Any) -> None:
    for key in path[:-1]:
        data = data[key]
    data[path[-1]] = value


@given(data=counts_files(), draw=st.data())
def test_any_value_anywhere_is_read_or_refused_with_one_counts_error_line(
    data: dict[str, Any], draw: st.DataObject
) -> None:
    changed = copy.deepcopy(data)
    path = draw.draw(st.sampled_from(list(_paths(changed))))
    value = draw.draw(WILD)
    _put(changed, path, value)
    raw = _bytes(changed)

    try:
        validated: MeasuredCounts | None = MeasuredCounts.model_validate(changed)
    except ValidationError:
        validated = None
    if validated is None:
        _refusal(raw)
    else:
        assert _load(raw) == validated


@given(st.binary(max_size=64), st.booleans())
def test_arbitrary_bytes_are_one_counts_error_line(raw: bytes, packed: bool) -> None:
    _refusal(gzip.compress(raw) if packed else raw)


@given(
    st.fixed_dictionaries(
        {"nv_counts": st.just("1.0")}, optional={key: WILD for key in REQUIRED["top"][1:]}
    )
    | WILD
)
def test_arbitrary_json_is_one_counts_error_line(data: Any) -> None:
    _refusal(_bytes(data))


# Regression tests for defects that the properties found.


def _minimal(**changes: Any) -> dict[str, Any]:
    circuit = {"name": "c", "qubits": [0, 1], "ops": [], "shots": 3, "counts": {"00": 3}}
    data = {
        "nv_counts": "1.0",
        "source": "hardware",
        "profile": {"id": "toy", "fingerprint": "0" * 64},
        "backend": "toy",
        "run_at": "2026-04-16T09:30:02Z",
        "bit_order": "clbit0_left",
        "execution": {"client": "c", **FLAGS},
        "circuits": [circuit],
    }
    for key, value in changes.items():
        (circuit if key in FIELDS["circuit"] else data)[key] = value
    return data


def test_a_parameter_of_minus_zero_gives_the_hash_of_zero() -> None:
    spellings = ("-0.0", "-0", "0", "0.0", "-0e5")
    text = json.dumps(_minimal(ops=[["rz", [0], ["P"]], ["delay", [1], ["P"]]]))

    hashes = {_load(text.replace('"P"', zero).encode()).sha256 for zero in spellings}

    assert len(hashes) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"a\nb": 1},
        {"counts": {"0\n": -1, "00": 3}},
        {"counts": {"0\r1": "x", "00": 3}},
    ],
)
def test_a_key_with_a_line_break_gives_a_one_line_error(changes: dict[str, Any]) -> None:
    _refusal(_bytes(_minimal(**changes)))


@pytest.mark.parametrize(
    ("key", "named"),
    [("011", '"011" has 3 bits'), ("0x", '"0x" holds characters other than 0 and 1')],
)
def test_a_refused_qiskit_key_is_named_as_the_file_writes_it(key: str, named: str) -> None:
    data = _minimal(bit_order="qiskit", counts={"01": 2, key: 1})

    assert named in _refusal(_bytes(data)).message


@pytest.mark.parametrize("value", [-1, True, "bad"])
def test_a_refused_qiskit_count_is_named_with_its_key_as_the_file_writes_it(value: Any) -> None:
    data = _minimal(bit_order="qiskit", counts={"01": value, "00": 3})

    assert "counts['01']" in _refusal(_bytes(data)).message


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("twirling", "enable_gates"), 0),
        (("twirling", "enable_measure"), 0.0),
        (("dynamical_decoupling", "enable"), -0.0),
        (("execution", "init_qubits"), 1),
    ],
)
def test_a_sampler_option_with_a_number_for_a_flag_is_refused(
    path: tuple[str, str], value: Any
) -> None:
    data = _minimal()
    data["execution"]["options"] = {path[0]: {path[1]: value}}

    assert f"options.{'.'.join(path)}" in _refusal(_bytes(data)).message


@pytest.mark.parametrize("run_at", ["9999-12-31T23:59:59-05:00", "0001-01-01T00:00:00+05:00"])
def test_a_run_at_that_leaves_the_datetime_range_in_utc_is_one_counts_error_line(
    run_at: str,
) -> None:
    _refusal(_bytes(_minimal(run_at=run_at)))


def test_an_options_key_with_a_line_break_gives_a_one_line_error() -> None:
    data = _minimal()
    data["execution"]["options"] = {"a\nb": float("nan")}

    _refusal(_bytes(data))


@pytest.mark.parametrize("field", ["transpiled", "init_qubits"])
def test_a_flag_nested_as_deep_as_the_parser_takes_is_one_counts_error_line(field: str) -> None:
    text = json.dumps(_minimal())
    flag = f'"{field}": {json.dumps(FLAGS[field])}'
    limit = _parser_limit()
    # The parser and the encoder have different stack headroom, so try depths near the limit.
    for depth in range(limit - 20, limit + 1):
        nested = f'"{field}": ' + "[" * depth + "]" * depth
        _refusal(text.replace(flag, nested).encode())


# plan and simulate on the small toy profile from conftest.

TOY_PAIRS = [(0, 1), (1, 0), (1, 2), (2, 1)]


@cache
def _toy() -> Profile:
    return Profile.model_validate(toy())


@st.composite
def toy_circuits(draw: st.DrawFn) -> PlannedCircuit:
    qubits = draw(st.sampled_from([(0,), (1,), (2,), *TOY_PAIRS]))
    names = ["sx", "rz", "delay", *(["cz"] if len(qubits) == 2 else [])]
    ops = []
    for name in draw(st.lists(st.sampled_from(names), max_size=8)):
        if name == "cz":
            ops.append(Op("cz", (0, 1)))
            continue
        qubit = draw(st.integers(0, len(qubits) - 1))
        param = draw(st.floats(0, 1e4)) if name != "sx" else None
        ops.append(Op(name, (qubit,), () if param is None else (param,)))
    return PlannedCircuit(name="random", qubits=qubits, ops=tuple(ops))


@settings(max_examples=examples(0.3))
@given(
    circuit=toy_circuits(),
    shots=st.integers(1, 10**6),
    seed=st.integers(0, 2**32 - 1),
)
def test_simulated_counts_add_up_to_shots_and_load_back_equal(
    circuit: PlannedCircuit, shots: int, seed: int
) -> None:
    profile = _toy()
    planned = (*plan(profile), circuit)
    n = len(circuit.qubits)

    counts = simulate(profile, planned, shots=shots, seed=seed, run_at=RUN_AT)
    probs = probabilities(profile, circuit.ops, n, layout=circuit.qubits, readout=True)

    assert np.all(probs >= -1e-12) and probs.sum() == pytest.approx(1.0, abs=1e-9)
    for wanted, measured in zip(planned, counts.circuits, strict=True):
        assert (measured.name, measured.qubits, measured.ops) == (
            wanted.name,
            wanted.qubits,
            wanted.ops,
        )
        assert sum(measured.counts.values()) == measured.shots == shots
    assert counts == simulate(profile, planned, shots=shots, seed=seed, run_at=RUN_AT)
    with _file(b"") as path:
        assert load_counts(counts.save(path)) == counts
