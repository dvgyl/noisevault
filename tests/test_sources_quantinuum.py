"""Quantinuum profiles from the published hardware-specification data."""

from __future__ import annotations

import contextlib
import hashlib
import json
import urllib.error
import urllib.request
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import deeper_than_the_parser_takes

from noisevault import Profile
from noisevault.errors import SourceDataError, SourceUnavailable
from noisevault.reference import Op, probabilities
from noisevault.sources import quantinuum

FIXTURE = Path(__file__).parent / "fixtures" / "quantinuum"
DATASET = FIXTURE / "H2-2" / "2024_12_06"
CSV = FIXTURE / "spec_sheet_parameters.csv"


def _files() -> dict[str, bytes]:
    return {name: (DATASET / f"{name}.json").read_bytes() for name in quantinuum.FILES}


@pytest.fixture(scope="module")
def h2_2():
    return quantinuum.from_data("H2-2", "2024_12_06", _files())


def test_raw_data_reproduces_the_published_spec_sheet_row(h2_2) -> None:
    # The repository's CSV row for H2-2 on 2024-12-06: 1Q 8(2)E-05, 1Q leakage 1.2(3)E-05,
    # 2Q 1.4(1)E-03, 2Q leakage 4.3(6)E-04, memory 5.3(4)E-04, SPAM 1.33(9)E-03; its emulator
    # CSV gives p_meas_0 0.0009 and p_meas_1 0.0018 for the same SPAM data.
    gates = h2_2.gates
    assert gates["r"].avg_infidelity == pytest.approx(8e-5, abs=0.5e-5)
    assert gates["zz"].avg_infidelity == pytest.approx(1.4e-3, abs=0.05e-3)
    leakage = {e.gate: e.prob for e in h2_2.effects if e.type == "leakage"}
    assert leakage["r"] == pytest.approx(1.2e-5, abs=0.05e-5)
    assert leakage["zz"] == pytest.approx(4.3e-4, abs=0.05e-4)
    assert h2_2.benchmarks["memory_error_depth1"]["value"] == pytest.approx(5.3e-4, abs=0.05e-4)
    readout = h2_2.readout
    assert (readout.p1_given_0 + readout.p0_given_1) / 2 == pytest.approx(1.33e-3, abs=0.005e-3)
    assert readout.p1_given_0 == pytest.approx(0.0009, abs=0.5e-4)
    assert readout.p0_given_1 == pytest.approx(0.0018, abs=0.5e-4)


def test_profile_states_what_each_number_means(h2_2) -> None:
    assert h2_2.id == "quantinuum_h2-2"
    assert h2_2.device.num_qubits == 56
    assert h2_2.device.technology == "trapped_ion"
    assert h2_2.device.calibrated_at.isoformat() == "2024-12-06T00:00:00+00:00"
    assert h2_2.connectivity == "all_to_all"
    assert h2_2.gates["rz"].virtual
    one, two = h2_2.gates["r"], h2_2.gates["zz"]
    assert (one.method, one.statistic, one.includes) == ("rb", "mean", ("leakage",))
    assert (two.method, two.statistic, two.includes) == ("rb", "mean", ("1q_dressing", "leakage"))
    assert "1.5 ZZ gates per Clifford" in two.assumption
    assert h2_2.prep is None  # combined SPAM holds the preparation error
    assert any("combined SPAM" in note for note in h2_2.provenance.notes)
    assert all(effect.allow == "omit" for effect in h2_2.effects)


def test_provenance_pins_the_exact_source(h2_2) -> None:
    prov = h2_2.provenance
    joined = b"".join(_files()[name] for name in quantinuum.FILES)
    assert prov.source_hash == "sha256:" + hashlib.sha256(joined).hexdigest()
    assert prov.source_url.endswith(f"/tree/{quantinuum.COMMIT}/data/H2-2/2024_12_06")
    assert (prov.data_kind, prov.source_kind) == ("measured", "published_data")
    assert (prov.license, prov.attribution, prov.redistributable) == (
        "Apache-2.0",
        "Quantinuum",
        "yes",
    )


def test_spec_csv_import_agrees_with_the_raw_data(h2_2) -> None:
    from_csv = quantinuum.from_spec_csv(CSV, machine="quantinuum_h2-2", date="2024_12_06")
    for gate in ("r", "zz"):
        spec = from_csv.gates[gate]
        assert spec.stderr is not None
        assert abs(spec.avg_infidelity - h2_2.gates[gate].avg_infidelity) <= spec.stderr
    assert from_csv.readout.error == pytest.approx(1.33e-3)
    assert (
        from_csv.provenance.source_hash == "sha256:" + hashlib.sha256(CSV.read_bytes()).hexdigest()
    )


def test_legacy_csv_rows_have_no_leakage_correction() -> None:
    profile = quantinuum.from_spec_csv(CSV, machine="H1-1", date="2022_06_09")
    assert profile.gates["zz"].avg_infidelity == pytest.approx(2.40e-3)
    assert profile.gates["zz"].includes == ("1q_dressing",)
    assert profile.effects == ()


def test_profile_converts_on_its_natives_without_approximation(h2_2) -> None:
    ops = [Op("r", (0,), (np.pi / 2, 0.0)), Op("zz", (0, 1)), Op("rz", (1,), (0.4,))]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a typical-noise fallback would warn
        noisy = probabilities(h2_2, ops, 2)
    perfect = Profile.uniform(
        "perfect", technology="trapped_ion", num_qubits=2, one_qubit_error=0, two_qubit_error=0
    )
    tvd = 0.5 * np.abs(noisy - probabilities(perfect, ops, 2)).sum()
    assert 1e-5 < tvd < 1e-2


def test_decay_rate_recovers_a_known_decay() -> None:
    lengths = np.array([2, 32, 128, 512, 2048], dtype=float)
    means = 0.73 * 0.99871**lengths + 0.25
    assert quantinuum.decay_rate(lengths, means, asymptote=0.25) == pytest.approx(0.99871, abs=1e-9)


@pytest.mark.parametrize(
    ("cell", "value", "stderr"),
    [("2.15(8)E-03", 2.15e-3, 8e-5), ("0(2)E-06", 0.0, 2e-6), ("1.33(9)E-03", 1.33e-3, 9e-5)],
)
def test_spec_sheet_cells_parse_value_and_uncertainty(cell: str, value: float, stderr: float):
    parsed = quantinuum.parse_cell(cell)
    assert parsed.value == pytest.approx(value, abs=1e-15)
    assert parsed.stderr == pytest.approx(stderr)


def test_bad_inputs_say_what_is_known() -> None:
    assert quantinuum.parse_cell(" ") is None
    with pytest.raises(SourceDataError, match="2.15"):
        quantinuum.parse_cell("0.002")
    with pytest.raises(ValueError, match="H1-1, H1-2, H2-1, H2-2, REIMEI"):
        quantinuum.from_repository("H3-1")
    with pytest.raises(ValueError, match="Known dates: 2024_12_06, 2025_05_29, 2025_08_28"):
        quantinuum.from_repository("H2-2", "2025_01_01")
    with pytest.raises(
        SourceDataError, match=r"these dates \(YYYY_MM_DD\) for H2-1: 2023_03_10, 2024_05_20$"
    ):
        quantinuum.from_spec_csv(CSV, machine="H2-1", date="2025_04_30")


def test_spec_csv_accepts_iso_dates() -> None:
    iso = quantinuum.from_spec_csv(CSV, machine="H2-2", date="2024-12-06")
    underscored = quantinuum.from_spec_csv(CSV, machine="H2-2", date="2024_12_06")
    assert iso.fingerprint == underscored.fingerprint


def test_spec_csv_with_a_repeated_column_names_both(tmp_path: Path) -> None:
    header, *rows = CSV.read_text(encoding="utf-8").splitlines()
    path = tmp_path / "twice.csv"
    lines = [f"{header},2Q error", *(f"{row},9.9(1)E-02" for row in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(SourceDataError) as info:
        quantinuum.from_spec_csv(path, machine="H2-2", date="2024_12_06")
    assert info.value.message == f"{path}: columns 6 and 13 are both '2Q error'"
    assert info.value.hint == "delete one of them"


def test_spec_csv_of_a_date_with_no_known_qubit_count_asks_for_it(tmp_path: Path) -> None:
    header, *rows = CSV.read_text(encoding="utf-8").splitlines()
    [row] = [r for r in rows if r.startswith("2024_12_06,H2-2,")]
    path = tmp_path / "later.csv"
    path.write_text(f"{header}\n{row.replace('2024_12_06', '2030_01_01')}\n", encoding="utf-8")
    with pytest.raises(SourceDataError) as info:
        quantinuum.from_spec_csv(path, machine="H2-2", date="2030_01_01")
    assert info.value.message == "the qubit count of H2-2 on 2030_01_01 is unknown"
    assert info.value.hint == "pass num_qubits="
    assert quantinuum.from_spec_csv(path, machine="H2-2", date="2030_01_01", num_qubits=56)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (",1.33(9)E-03", ",", "H2-2 2024_12_06: the CSV row has no SPAM error"),
        (
            ",8(2)E-05,7(2)E-05,",
            ",,,",
            "H2-2 2024_12_06: the CSV row has none of 1Q error, 1Q error (legacy)",
        ),
    ],
    ids=["no-spam", "no-1q"],
)
def test_spec_csv_row_missing_a_value_names_it(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    header, *rows = CSV.read_text(encoding="utf-8").splitlines()
    [row] = [r for r in rows if r.startswith("2024_12_06,H2-2,")]
    assert old in row
    path = tmp_path / "damaged.csv"
    path.write_text(f"{header}\n{row.replace(old, new)}\n", encoding="utf-8")
    with pytest.raises(SourceDataError) as info:
        quantinuum.from_spec_csv(path, machine="H2-2", date="2024_12_06")
    assert str(info.value) == message


_SPEC_SHEET_HINT = (
    "download notebooks/Spec sheet parameters.csv from"
    " https://github.com/Quantinuum/quantinuum-hardware-specifications"
)


_SPEC_COLUMNS = CSV.read_text(encoding="utf-8").splitlines()[0].split(",")


def _spec_csv_without(tmp_path: Path, column: str) -> Path:
    rows = [line.split(",") for line in CSV.read_text(encoding="utf-8").splitlines()]
    cut = rows[0].index(column)
    path = tmp_path / "other.csv"
    path.write_text("\n".join(",".join(r[:cut] + r[cut + 1 :]) for r in rows), encoding="utf-8")
    return path


@pytest.mark.parametrize("column", ["Date", "Machine"])
def test_spec_csv_without_the_row_key_columns_names_the_missing_one(
    tmp_path: Path, column: str
) -> None:
    path = _spec_csv_without(tmp_path, column)
    with pytest.raises(SourceDataError) as info:
        quantinuum.from_spec_csv(path, machine="H2-2", date="2024_12_06")
    assert str(info.value) == f"{path} has no {column!r} column; {_SPEC_SHEET_HINT}"


@pytest.mark.parametrize("column", _SPEC_COLUMNS)
def test_a_spec_csv_missing_any_column_imports_or_raises_source_data_error(
    tmp_path: Path, column: str
) -> None:
    path = _spec_csv_without(tmp_path, column)
    with contextlib.suppress(SourceDataError):
        quantinuum.from_spec_csv(path, machine="H2-2", date="2024_12_06")


def test_a_spec_csv_cell_cannot_run_on_to_the_next_line(tmp_path: Path) -> None:
    header, first, *rest = CSV.read_text(encoding="utf-8").splitlines()
    path = tmp_path / "spanning.csv"
    path.write_text("\n".join([header, first.replace(",", ',"x\ny",', 1), *rest]) + "\n")
    with pytest.raises(SourceDataError) as info:
        quantinuum.from_spec_csv(path, machine="H2-2", date="2024_12_06")
    assert str(info.value) == f"{path} line 2 is not valid CSV: unexpected end of data"


def test_spec_csv_that_is_not_utf8_names_the_line_and_the_byte(tmp_path: Path) -> None:
    path = tmp_path / "resaved.csv"
    path.write_bytes(CSV.read_bytes().replace(b"H1-2", b"H1\xad2", 1))
    with pytest.raises(SourceDataError) as info:
        quantinuum.from_spec_csv(path, machine="H2-2", date="2024_12_06")
    assert str(info.value) == (
        f"{path} is not UTF-8 text (byte 0xad on line 3); {_SPEC_SHEET_HINT}"
    )


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("1.33(9)E-03\n", "1.33(9)E-03,\n", "line 10 has 13 cells, but the header has 12 columns"),
        (
            ",,1.33(9)E-03\n",
            ",1.33(9)E-03\n",
            "line 10 has 11 cells, but the header has 12 columns",
        ),
        (",2.8(1)E-03\n", ',"2.8(1)E-03\n', "line 2 is not valid CSV: unexpected end of data"),
    ],
    ids=["extra-field", "missing-field", "unclosed-quote"],
)
def test_spec_csv_with_a_damaged_record_names_its_line(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    text = CSV.read_text(encoding="utf-8")
    assert text.count(old) == 1
    path = tmp_path / "damaged.csv"
    path.write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(SourceDataError) as info:
        quantinuum.from_spec_csv(path, machine="H2-2", date="2024_12_06")
    assert str(info.value) == f"{path} {message}"


def test_spec_csv_with_carriage_return_line_ends_reads_the_same_row(tmp_path: Path) -> None:
    path = tmp_path / "mac.csv"
    path.write_bytes(CSV.read_bytes().replace(b"\n", b"\r"))
    mac, unix = (
        quantinuum.from_spec_csv(p, machine="H2-2", date="2024_12_06") for p in (path, CSV)
    )
    assert mac.fingerprint == unix.fingerprint


_DATA = "data/H2-2/2024_12_06"


@pytest.mark.parametrize(
    ("name", "raw", "message"),
    [
        (
            "SQ_RB",
            b"<html><body>Sign in to this network</body></html>",
            f"{_DATA}/SQ_RB.json is not JSON (expecting value at line 1, column 1)",
        ),
        (
            "SPAM",
            b'{"shots": 10000,\n "survival": {"0": {"0": 9\xb5}}}',
            f"{_DATA}/SPAM.json is not UTF-8 text (byte 0xb5 on line 2)",
        ),
        ("TQ_RB", b"[]", f"{_DATA}/TQ_RB.json is not a JSON object"),
    ],
    ids=["html", "latin-1", "array"],
)
def test_dataset_file_that_is_not_a_json_object_names_the_file(
    name: str, raw: bytes, message: str
) -> None:
    with pytest.raises(SourceDataError) as info:
        quantinuum.from_data("H2-2", "2024_12_06", {**_files(), name: raw})
    assert str(info.value) == message


def test_dataset_file_nested_deeper_than_the_parser_takes_names_the_file() -> None:
    nested = deeper_than_the_parser_takes()
    with pytest.raises(SourceDataError) as info:
        quantinuum.from_data("H2-2", "2024_12_06", {**_files(), "SQ_RB": nested.encode()})
    depth = len(nested) // 2
    assert str(info.value) == (
        f"{_DATA}/SQ_RB.json is not JSON (nested {depth} levels deep at line 1, column {depth})"
    )


@pytest.mark.parametrize(
    ("name", "path", "message"),
    [
        ("SQ_RB", ("shots",), f"{_DATA}/SQ_RB.json has no 'shots'"),
        ("Memory_RB", ("survival",), f"{_DATA}/Memory_RB.json has no 'survival'"),
        (
            "TQ_RB",
            ("survival", "(2, 3)", "32"),
            f"{_DATA}/TQ_RB.json: survival['(2, 3)'] has no '32'",
        ),
        (
            "TQ_RB",
            ("leakage_postselect", "(4, 5)", "128"),
            f"{_DATA}/TQ_RB.json: leakage_postselect['(4, 5)'] has no '128'",
        ),
        ("SPAM", ("survival", "3", "1"), f"{_DATA}/SPAM.json: survival['3'] has no '1'"),
    ],
    ids=["shots", "survival", "sequence-length", "leakage-length", "spam-state"],
)
def test_dataset_file_missing_a_value_names_where(
    name: str, path: tuple[str, ...], message: str
) -> None:
    with pytest.raises(SourceDataError) as info:
        quantinuum.from_data("H2-2", "2024_12_06", _files_without(name, path))
    assert str(info.value) == message


def _at(node: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        node = node[key]
    return node


def _files_without(name: str, path: tuple[str, ...]) -> dict[str, bytes]:
    doc = json.loads(_files()[name])
    del _at(doc, path[:-1])[path[-1]]
    return {**_files(), name: json.dumps(doc).encode()}


def _files_with(name: str, path: tuple[str, ...], value: Any) -> dict[str, bytes]:
    doc = json.loads(_files()[name])
    _at(doc, path[:-1])[path[-1]] = value
    return {**_files(), name: json.dumps(doc).encode()}


_SHOTS = "not a positive whole number"


@pytest.mark.parametrize(
    ("name", "path", "value", "message"),
    [
        (
            "SQ_RB",
            ("shots",),
            1,
            f"{_DATA}/SQ_RB.json: survival['0']['256']['3'] is 97,"
            " not a whole number of shots from 0 to 1",
        ),
        ("TQ_RB", ("shots",), 0, f"{_DATA}/TQ_RB.json: shots is 0, {_SHOTS}"),
        ("Memory_RB", ("shots",), True, f"{_DATA}/Memory_RB.json: shots is True, {_SHOTS}"),
        ("SPAM", ("shots",), "10000", f"{_DATA}/SPAM.json: shots is '10000', {_SHOTS}"),
        (
            "TQ_RB",
            ("leakage_postselect", "(4, 5)", "128", "2"),
            101,
            f"{_DATA}/TQ_RB.json: leakage_postselect['(4, 5)']['128']['2'] is 101,"
            " not a whole number of shots from 0 to 100",
        ),
        (
            "SQ_RB",
            ("survival", "7", "1024", "0"),
            97.5,
            f"{_DATA}/SQ_RB.json: survival['7']['1024']['0'] is 97.5,"
            " not a whole number of shots from 0 to 100",
        ),
        (
            "Memory_RB",
            ("survival", "55", "16", "7"),
            float("nan"),
            f"{_DATA}/Memory_RB.json: survival['55']['16']['7'] is nan,"
            " not a whole number of shots from 0 to 40",
        ),
        (
            "SPAM",
            ("survival", "3", "1"),
            -1.0,
            f"{_DATA}/SPAM.json: survival['3']['1'] is -1.0,"
            " not a whole number of shots from 0 to 10000",
        ),
    ],
    ids=[
        "counts-above-shots",
        "no-shots",
        "boolean-shots",
        "text-shots",
        "leakage-above-shots",
        "fractional-count",
        "nan-count",
        "negative-count",
    ],
)
def test_dataset_file_with_an_impossible_count_names_where(
    name: str, path: tuple[str, ...], value: Any, message: str
) -> None:
    with pytest.raises(SourceDataError) as info:
        quantinuum.from_data("H2-2", "2024_12_06", _files_with(name, path, value))
    assert str(info.value) == message


@pytest.mark.parametrize(
    ("name", "path", "value", "message"),
    [
        (
            "SQ_RB",
            ("survival", "0"),
            {},
            f"{_DATA}/SQ_RB.json: survival['0'] has no sequence lengths",
        ),
        ("TQ_RB", ("survival",), {}, f"{_DATA}/TQ_RB.json: survival has no zones"),
        (
            "Memory_RB",
            ("survival", "55", "64"),
            {},
            f"{_DATA}/Memory_RB.json: survival['55']['64'] has no counts",
        ),
        ("SPAM", ("survival",), {}, f"{_DATA}/SPAM.json: survival has no qubits"),
        (
            "SQ_RB",
            ("survival", "0"),
            [],
            f"{_DATA}/SQ_RB.json: survival['0'] is not a JSON object",
        ),
        ("SPAM", ("survival", "3"), 5, f"{_DATA}/SPAM.json: survival['3'] is not a JSON object"),
        (
            "SQ_RB",
            ("survival", "7", "4096"),
            {"0": 50},
            f"{_DATA}/SQ_RB.json: survival['0'] has no '4096'",
        ),
        (
            "TQ_RB",
            ("leakage_postselect", "(0, 1)", "x"),
            {"0": 50},
            f"{_DATA}/TQ_RB.json: leakage_postselect['(0, 1)'] has the sequence length 'x',"
            " not a whole number",
        ),
    ],
    ids=[
        "no-lengths",
        "no-zones",
        "no-counts",
        "no-qubits",
        "zone-array",
        "spam-row-number",
        "extra-length",
        "length-not-a-number",
    ],
)
def test_dataset_file_with_a_misshapen_map_names_where(
    name: str, path: tuple[str, ...], value: Any, message: str
) -> None:
    with pytest.raises(SourceDataError) as info:
        quantinuum.from_data("H2-2", "2024_12_06", _files_with(name, path, value))
    assert str(info.value) == message


def test_rb_curves_measured_at_different_sequence_lengths_name_the_gap() -> None:
    doc = json.loads(_files()["SQ_RB"])
    for zone in doc["leakage_postselect"].values():
        del zone["1024"]
    damaged = {**_files(), "SQ_RB": json.dumps(doc).encode()}
    with pytest.raises(SourceDataError) as info:
        quantinuum.from_data("H2-2", "2024_12_06", damaged)
    assert str(info.value) == f"{_DATA}/SQ_RB.json: leakage_postselect['0'] has no '1024'"


def _sampled_key_paths(node: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[str, ...]]:
    if isinstance(node, dict):
        numbered = [key for key in node if key.isdigit()]
        for key, value in node.items():
            if key not in numbered[1:-1]:
                yield (*path, key)
                yield from _sampled_key_paths(value, (*path, key))


@pytest.mark.parametrize(
    ("name", "path"),
    [
        pytest.param(name, path, id=f"{name}:{'.'.join(path)}")
        for name in quantinuum.FILES
        for path in _sampled_key_paths(json.loads(_files()[name]))
    ],
)
def test_a_dataset_file_missing_any_key_imports_or_raises_source_data_error(
    name: str, path: tuple[str, ...]
) -> None:
    with contextlib.suppress(SourceDataError):
        quantinuum.from_data("H2-2", "2024_12_06", _files_without(name, path))


@pytest.mark.parametrize(
    ("name", "path"),
    [
        pytest.param(name, path, id=f"{name}:{'.'.join(path)}")
        for name in quantinuum.FILES
        for doc in [json.loads(_files()[name])]
        for path in _sampled_key_paths(doc)
        if path[0] != "sequence_info" and isinstance(_at(doc, path), dict)
    ],
)
def test_emptying_any_object_the_import_reads_names_it(name: str, path: tuple[str, ...]) -> None:
    where = f"{_DATA}/{name}.json: {path[0]}" + "".join(f"[{key!r}]" for key in path[1:])
    with pytest.raises(SourceDataError) as info:
        quantinuum.from_data("H2-2", "2024_12_06", _files_with(name, path, {}))
    assert str(info.value).startswith(f"{where} has no ")


def test_a_failed_download_says_what_loads_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    requested = []

    def offline(handler: object, request: urllib.request.Request) -> None:
        requested.append(request.full_url)
        raise urllib.error.URLError("timed out")

    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", offline)
    with pytest.raises(SourceUnavailable) as info:
        quantinuum.from_repository("H2-2")
    assert info.value.message == f"could not download {requested[0]} (timed out)"
    assert info.value.hint == (
        "check the network connection, or run nv list to see every profile you can load offline"
    )


def test_bundle_takes_the_newest_dataset_of_every_machine(monkeypatch) -> None:
    fetched = []
    monkeypatch.setattr(quantinuum, "from_repository", lambda m: fetched.append(m) or m)
    assert quantinuum.bundled_profiles() == ["H1-1", "H1-2", "H2-1", "H2-2", "REIMEI"]
    newest = {m: max(d for mm, d in quantinuum.QUBITS if mm == m) for m in fetched}
    assert newest == {
        "H1-1": "2025_05_02",
        "H1-2": "2023_08_21",
        "H2-1": "2025_04_30",
        "H2-2": "2025_08_28",
        "REIMEI": "2025_06_18",
    }


@pytest.mark.network
def test_download_matches_the_fixture(h2_2) -> None:
    live = quantinuum.from_repository("H2-2", "2024_12_06")
    assert live.fingerprint == h2_2.fingerprint
    assert live.provenance.source_hash == h2_2.provenance.source_hash


def test_a_csv_value_a_profile_cannot_hold_names_the_row_and_the_value(tmp_path: Path) -> None:
    text = CSV.read_text(encoding="utf-8")
    assert text.count("2.8(1)E-03") == 1
    path = tmp_path / "spec.csv"
    path.write_text(text.replace("2.8(1)E-03", "1.5(1)E+00"), encoding="utf-8")
    with pytest.raises(SourceDataError) as info:
        quantinuum.from_spec_csv(path, machine="H1-1", date="2022_06_09")
    assert (info.value.message, info.value.hint) == (
        "Quantinuum spec sheet parameters CSV (spec.csv), H1-1 2022_06_09: readout.error: Input"
        " should be less than or equal to 1, got 1.5",
        f"correct that value in {path}",
    )
