from __future__ import annotations

import gzip
import json
import pickle
import sys

import pytest
from conftest import deeper_than_the_parser_takes

import noisevault as nv
from noisevault.errors import (
    DuplicateKeyError,
    FingerprintMismatch,
    LayoutError,
    LociText,
    NoiseVaultError,
    did_you_mean,
    parse_json,
    qubit_loci,
    unreadable,
)


def test_an_error_keeps_its_next_step_apart_and_prints_both() -> None:
    error = FingerprintMismatch("ibm_fez is nv:06404cefa54f", hint="load ibm_fez@2025-02-26")
    assert (error.message, error.hint) == ("ibm_fez is nv:06404cefa54f", "load ibm_fez@2025-02-26")
    assert str(error) == "ibm_fez is nv:06404cefa54f; load ibm_fez@2025-02-26"
    assert isinstance(error, ValueError)


def test_an_error_without_a_hint_prints_its_message_alone() -> None:
    error = LayoutError("qubit 9 is not on the device")
    assert error.hint is None and str(error) == error.message == "qubit 9 is not on the device"


def test_a_hint_survives_pickling() -> None:
    copy = pickle.loads(pickle.dumps(NoiseVaultError("no source", hint="pull it again")))
    assert (copy.message, copy.hint, str(copy)) == (
        "no source",
        "pull it again",
        "no source; pull it again",
    )


def test_did_you_mean_quotes_the_closest_choice_or_says_nothing() -> None:
    assert did_you_mean("ibm_fezz", ["ibm_fez", "ibm_kyiv"]) == "did you mean 'ibm_fez'? "
    assert did_you_mean("zzz", ["ibm_fez"]) == ""


def test_unreadable_source_data_is_caught_as_a_noisevault_error_or_a_value_error() -> None:
    assert issubclass(nv.SourceDataError, NoiseVaultError)
    assert issubclass(nv.SourceDataError, ValueError)
    assert "SourceDataError" in nv.__all__


def _raised(read, raw: bytes) -> Exception:
    with pytest.raises((ValueError, EOFError)) as caught:
        read(raw)
    return caught.value


def test_a_number_too_long_to_read_is_named_in_words_not_python_advice() -> None:
    long_number = _raised(json.loads, b'{"n": ' + b"1" * 5000 + b"}")
    limit = sys.get_int_max_str_digits()
    assert unreadable("big.json", long_number) == (
        f"big.json is not JSON (a number has 5000 digits, over the {limit}-digit limit)"
    )


@pytest.mark.parametrize(
    ("raw", "path", "message"),
    [
        (b'{"a": 1, "a": 1}', ("a",), "dup.json has the key a twice"),
        (
            b'[0, {"x": [{"0-1": 1, "0-1": 2}]}]',
            (1, "x", 0, "0-1"),
            "dup.json has the key [1].x[0]['0-1'] twice",
        ),
        (b'{"a": {"b": 1, "b": 2}, "a": 3}', ("a",), "dup.json has the key a twice"),
        (
            b'{"k": [{}], "r": {"z": 1, "y": 2, "z": 3}}',
            ("r", "z"),
            "dup.json has the key r.z twice",
        ),
        (
            b'{"a": {"x": 1, "x": 2}, "b": {"y": 1, "y": 2}}',
            ("a", "x"),
            "dup.json has the key a.x twice",
        ),
    ],
    ids=["same value", "inside arrays", "lost first value", "after a sibling", "first of two"],
)
def test_a_key_twice_in_one_object_is_refused_with_its_path(
    raw: bytes, path: tuple[str | int, ...], message: str
) -> None:
    with pytest.raises(DuplicateKeyError) as caught:
        parse_json(raw)
    assert caught.value.path == path
    assert unreadable("dup.json", caught.value) == message


def test_a_key_twice_before_nesting_deeper_than_the_parser_takes_gives_the_depth() -> None:
    nested = deeper_than_the_parser_takes()
    with pytest.raises(json.JSONDecodeError) as caught:
        parse_json(('[{"a": 1, "a": 2}, ' + nested + "]").encode())
    assert caught.value.msg == f"nested {len(nested) // 2 + 1} levels deep"


def test_a_damaged_gzip_file_gives_its_reason_in_lower_case() -> None:
    cut = _raised(gzip.decompress, gzip.compress(b"{}")[:12])
    message = unreadable("cut.json.gz", cut)
    assert message.startswith("cut.json.gz is a damaged gzip file (")
    reason = message.removeprefix("cut.json.gz is a damaged gzip file (")
    assert reason[0].islower() and reason.lower() == f"{str(cut).lower()})"


def test_qubit_loci_names_four_loci_and_counts_the_rest_unless_the_limit_is_none() -> None:
    five = [(q,) for q in range(5)]
    assert qubit_loci(*five[:4]) == "qubits 0, 1, 2 and 3"
    assert qubit_loci(*five) == "qubits 0, 1, 2 and 2 more"
    assert qubit_loci(*five, limit=None) == "qubits 0, 1, 2, 3 and 4"
    assert qubit_loci((7,), limit=None) == "qubit 7"
    assert qubit_loci((0, 1), limit=None) == "qubits 0-1"


def test_joined_loci_text_names_every_locus_and_its_short_form_counts_the_rest() -> None:
    six = [(q,) for q in range(6)]
    text = LociText("; ").join(["gate errors x2", LociText("readout of ", six, " is not scaled")])
    assert text.full == "gate errors x2; readout of qubits 0, 1, 2, 3, 4 and 5 is not scaled"
    assert text.short == "gate errors x2; readout of qubits 0, 1, 2 and 3 more is not scaled"
    assert LociText("; ").join([]).full == ""


def test_loci_text_used_as_plain_text_fails_or_names_no_list() -> None:
    text = LociText("readout of ", [(q,) for q in range(6)])
    with pytest.raises(TypeError):
        "; ".join([text])
    with pytest.raises(TypeError):
        json.dumps(text)
    assert "0, 1, 2, 3, 4 and 5" not in f"{text}"
