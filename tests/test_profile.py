from __future__ import annotations

import copy
import errno
import json
import os
import pickle
import random
import re
import shutil
import stat
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from conftest import deeper_than_the_parser_takes, toy
from pydantic import BaseModel, TypeAdapter, ValidationError

import noisevault as nv
from noisevault import profile as profile_module
from noisevault.catalog import bundled_profiles, vault_path
from noisevault.profile import (
    ErrorFactor,
    Profile,
    Ref,
    UnmodeledError,
    json_schema,
    load_file,
    parse_ref,
    profile_id,
    read_json_file,
    unmodeled_note,
)

TWO_EDGES = {"edges": [[0, 1], [1, 2]]}


def _invalid(data: dict, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        Profile.model_validate(data)


def test_toy_profile_is_valid() -> None:
    profile = Profile.model_validate(toy())
    assert profile.id == "test_toy"


@pytest.mark.parametrize(
    ("section", "value", "match"),
    [
        ("color", "blue", "Extra inputs"),
        ("device", {"name": "t", "technology": "other", "num_qubits": 3, "x": 1}, "Extra"),
        ("device", {"name": "a@b", "technology": "superconducting", "num_qubits": 3}, "free of @"),
        ("device", {"name": "t", "technology": "superconducting", "num_qubits": 0}, "greater"),
        ("connectivity", {"edges": [[0, 3]]}, "outside 0..2"),
        ("connectivity", {"edges": [[1, 1]]}, "itself"),
        ("connectivity", {"edges": [[0, 1], [1, 0]]}, "listed twice"),
        ("gates", {"sx": {"avg_infidelity": 1e-3, "colour": 1}}, "Extra"),
        ("gates", {"sx": {"avg_infidelity": 1e-3, "process_infidelity": 1e-3}}, "one error metric"),
        ("gates", {"sx": {"avg_infidelity": 0.7}}, r"avg_infidelity of a 1-qubit gate"),
        ("gates", {"cz": {"avg_infidelity": 0.81}}, r"avg_infidelity of a 2-qubit gate"),
        ("gates", {"sx": {"depolarizing_param": 1.4}}, "depolarizing_param"),
        ("gates", {"sx": {"process_infidelity": 1.2}}, "process_infidelity"),
        ("gates", {"sx": {"pauli": [0.1, -0.1, 0.0]}}, ">= 0"),
        ("gates", {"sx": {"pauli": [0.5, 0.5, 0.5]}}, "sum to"),
        ("gates", {"cz": {"pauli": [0.01, 0.01, 0.01]}}, "needs 15 entries"),
        ("gates", {"rz": {"virtual": True, "avg_infidelity": 0.0}}, "virtual gate"),
        ("gates", {"rz": {"virtual": True, "duration_ns": 10}}, "virtual gate"),
        ("gates", {"fsim": {"avg_infidelity": 1e-3}}, "state its arity"),
        ("gates", {"cz": {"qubits": 1, "avg_infidelity": 1e-3}}, "acts on 2"),
        ("readout", {"p1_given_0": 0.01}, "both p1_given_0 and p0_given_1"),
        ("readout", {"error": 0.01, "p1_given_0": 0.01, "p0_given_1": 0.02}, "not both"),
        ("readout", {"error": 1.5}, "less than or equal"),
        ("prep", {"error": -0.1}, "greater than or equal"),
        ("idle", {"t1_us": 0}, "must be positive"),
        ("idle", {"t2_us": -5}, "must be positive"),
        ("idle", {"t1_us": float("nan")}, "finite"),
        ("idle", {"t1_us": float("inf")}, "finite"),
        ("idle", {"t1_us": 100, "t1_ms": 0.1}, "appears twice"),
        ("idle", {"t1_us": "100"}, "must be a number"),
        ("qubits", [{"index": 3}], "outside 0..2"),
        ("qubits", [{"index": 1}, {"index": 1}], "listed twice"),
        ("effects", [{"type": "leakage", "gate": "cz", "prob": 1e-4, "allow": "maybe"}], "allow"),
        ("effects", [{"type": "heating", "gate": "cz"}], "type"),
        ("effects", [{"type": "leakage", "prob": 1e-4}], "exactly one of gate or on"),
        ("effects", [{"type": "leakage", "gate": "ms", "prob": 1e-4}], "not defined"),
        ("effects", [{"type": "crosstalk_zz", "on": "idle", "qubits": [0, 5]}], "outside"),
        ("provenance", {"source_hash": "sha256:abc"}, "pattern"),
        ("provenance", {"redistributable": "maybe"}, "redistributable"),
        ("noisevault", "2.0", "noisevault"),
    ],
)
def test_section_rules(section: str, value, match: str) -> None:
    data = toy(**{section: value})
    if section == "gates":
        data["gates"] = {**toy()["gates"], **value}
    _invalid(data, match)


@pytest.mark.parametrize(
    ("record", "match"),
    [
        (
            {"gate": "ecr", "qubits": [0, 1], "avg_infidelity": 1e-2},
            r"calibrations \(ecr on qubits 0-1\): gate 'ecr' is not defined in gates",
        ),
        ({"gate": "cz", "qubits": [0], "avg_infidelity": 1e-2}, "acts on 2 qubits"),
        ({"gate": "cz", "qubits": [1, 1], "avg_infidelity": 1e-2}, "distinct"),
        ({"gate": "cz", "qubits": [1, 3], "avg_infidelity": 1e-2}, "outside 0..2"),
        ({"gate": "cz", "qubits": [0, 1], "avg_infidelity": 0.9}, "2-qubit gate"),
        ({"gate": "rz", "qubits": [0], "avg_infidelity": 1e-4}, "virtual gate"),
        ({"gate": "sx", "qubits": [0], "duration_us": 0.1, "duration_ns": 100}, "appears twice"),
        ({"gate": "sx", "qubits": [0], "duration_ns": -1}, "must not be negative"),
    ],
)
def test_calibration_record_rules(record: dict, match: str) -> None:
    _invalid(toy(calibrations=[record]), match)


def test_duplicate_calibration_records_are_rejected() -> None:
    record = {"gate": "cz", "qubits": [0, 1], "avg_infidelity": 1e-2}
    _invalid(toy(calibrations=[record, {**record, "avg_infidelity": 2e-2}]), "second record")


def test_reversed_records_of_a_symmetric_gate_are_distinct_loci() -> None:
    records = [
        {"gate": "cz", "qubits": [0, 1], "avg_infidelity": 1e-2},
        {"gate": "cz", "qubits": [1, 0], "avg_infidelity": 1e-2},
    ]
    Profile.model_validate(toy(calibrations=records))


def test_a_record_may_calibrate_a_virtual_gate_when_it_says_so() -> None:
    record = {"gate": "rz", "qubits": [0], "virtual": False, "avg_infidelity": 1e-4}
    Profile.model_validate(toy(calibrations=[record]))


def test_all_validation_issues_are_reported_together() -> None:
    data = toy(calibrations=[{"gate": "ecr", "qubits": [0, 1]}, {"gate": "cz", "qubits": [0]}])
    with pytest.raises(ValidationError) as info:
        Profile.model_validate(data)
    text = str(info.value)
    assert "ecr" in text and "acts on 2 qubits" in text


@pytest.mark.parametrize(
    ("given", "canonical", "value"),
    [
        ({"duration_us": 0.25}, "duration_ns", 250.0),
        ({"duration_ms": 0.002}, "duration_ns", 2000.0),
        ({"duration_s": 1e-6}, "duration_ns", 1000.0),
        ({"duration_ns": 40}, "duration_ns", 40.0),
    ],
)
def test_duration_aliases_normalize(given: dict, canonical: str, value: float) -> None:
    data = toy()
    data["gates"]["sx"] = {"avg_infidelity": 1e-3, **given}
    spec = Profile.model_validate(data).gates["sx"]
    assert spec.duration_ns == pytest.approx(value, rel=1e-15)


@pytest.mark.parametrize(
    ("given", "t1_us", "t2_us"),
    [
        ({"t1_ms": 0.2, "t2_ns": 150_000}, 200.0, 150.0),
        ({"t1_s": 1.0, "t2_s": 0.5}, 1e6, 5e5),
        ({"t1_ns": 40}, 0.04, None),
    ],
)
def test_coherence_aliases_normalize_in_idle_and_qubits(given: dict, t1_us, t2_us) -> None:
    profile = Profile.model_validate(toy(idle=given, qubits=[{"index": 2, **given}]))
    for holder in (profile.idle, profile.qubits[0]):
        assert holder.t1_us == pytest.approx(t1_us, rel=1e-15)
        assert holder.t2_us == (None if t2_us is None else pytest.approx(t2_us, rel=1e-15))


def test_saved_files_use_canonical_units(tmp_path: Path) -> None:
    data = toy(idle={"t1_ms": 0.2}, readout={"error": 0.01, "duration_us": 1.5})
    path = Profile.model_validate(data).save(tmp_path / "p.json")
    saved = json.loads(path.read_text())
    assert saved["idle"] == {"t1_us": 200.0}
    assert saved["readout"] == {"error": 0.01, "duration_ns": 1500.0}


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
def test_round_trip(tmp_path: Path, suffix: str) -> None:
    profile = Profile.model_validate(
        toy(
            calibrations=[{"gate": "cz", "qubits": [1, 2], "pauli": [1e-3] * 15, "method": "irb"}],
            qubits=[{"index": 0, "t1_us": 90, "readout": {"p1_given_0": 0.01, "p0_given_1": 0.02}}],
            effects=[{"type": "leakage", "gate": "cz", "prob": 1e-4}],
            device={
                "name": "toy",
                "vendor": "test",
                "technology": "superconducting",
                "num_qubits": 3,
                "calibrated_at": "2025-02-26T10:12:00+01:00",
            },
        )
    )
    loaded = load_file(profile.save(tmp_path / f"p{suffix}"))
    assert loaded == profile
    assert loaded.fingerprint == profile.fingerprint
    assert loaded.artifact_hash == profile.artifact_hash
    assert loaded.device.calibrated_at == datetime(2025, 2, 26, 9, 12, tzinfo=UTC)


TIME_FIELDS = ["device.calibrated_at", "provenance.retrieved_at", "unmodeled_error.fit.run_at"]


def _timed(where: str, value: Any) -> dict:
    if where == "device.calibrated_at":
        return toy(device={**toy()["device"], "calibrated_at": value})
    if where == "provenance.retrieved_at":
        return toy(provenance={"retrieved_at": value})
    base = Profile.model_validate(toy(readout={"error": 0.01}))
    return {**base.to_dict(), "unmodeled_error": _block(base, run_at=value)}


@pytest.mark.parametrize(
    ("value", "shown"),
    [
        (20260930, "20260930"),
        (20260930.5, "20260930.5"),
        ("20260930", '"20260930"'),
        ("-1", '"-1"'),
        (True, "true"),
    ],
)
@pytest.mark.parametrize("where", TIME_FIELDS)
def test_a_time_field_refuses_a_number_or_a_string_of_digits(
    where: str, value: Any, shown: str
) -> None:
    with pytest.raises(ValidationError) as caught:
        Profile.model_validate(_timed(where, value))
    assert [(e["loc"], e["msg"]) for e in caught.value.errors()] == [
        (
            tuple(where.split(".")),
            f"Value error, {shown} is not an ISO 8601 time with a timezone."
            " Give a time such as 2026-09-30T08:00:00Z",
        )
    ]


@pytest.mark.parametrize(
    "value",
    ["2026-09-30T05:00:00-03:00", "2026-09-30 08:00Z", datetime(2026, 9, 30, 8, tzinfo=UTC)],
)
@pytest.mark.parametrize("where", TIME_FIELDS)
def test_a_time_field_takes_an_iso_time_with_a_timezone_or_a_datetime(
    where: str, value: Any
) -> None:
    saved = Profile.model_validate(_timed(where, value)).to_dict()
    for key in where.split("."):
        saved = saved[key]
    assert saved == "2026-09-30T08:00:00Z"


def test_a_file_nested_deeper_than_the_parser_takes_is_not_json(tmp_path: Path) -> None:
    nested = deeper_than_the_parser_takes()
    data = toy(extensions={"deep": 0})
    data["device"]["vendor"] = 'say "[[" here'
    text = json.dumps(data, indent=1).replace('"deep": 0', '"deep": ' + nested)
    path = tmp_path / "deep.json"
    path.write_text(text)
    rows = text.splitlines()
    (line,) = [n for n, row in enumerate(rows, 1) if '"deep"' in row]
    innermost = rows[line - 1].index("[") + len(nested) // 2

    with pytest.raises(json.JSONDecodeError) as caught:
        load_file(path)

    assert (caught.value.msg, caught.value.lineno, caught.value.colno) == (
        f"nested {len(nested) // 2 + 2} levels deep",
        line,
        innermost,
    )


def test_a_gate_defined_twice_is_refused_before_the_second_replaces_the_first(
    tmp_path: Path,
) -> None:
    data = toy(device={**toy()["device"], "num_qubits": 1}, connectivity={"edges": []})
    data["gates"] = {"x": {"avg_infidelity": 0.2}}
    text = json.dumps(data).replace('"gates": {', '"gates": {"x": {"avg_infidelity": 0.01}, ')
    path = tmp_path / "twice.json"
    path.write_text(text)

    with pytest.raises(ValueError, match=r"^the key gates\.x appears twice$"):
        load_file(path)
    with pytest.raises(nv.NoiseVaultError) as caught:
        read_json_file(path, "profile")
    assert (caught.value.message, caught.value.hint) == (
        f"{path} has the key gates.x twice",
        "the file is damaged. Pull or export the profile again, or keep one of the two keys",
    )


@pytest.mark.filterwarnings("ignore::pydantic.PydanticDeprecatedSince20")
@pytest.mark.parametrize(
    ("method", "kind"), [("model_validate_json", "json_invalid"), ("parse_raw", "value_error")]
)
def test_a_model_json_method_refuses_a_gate_defined_twice(method: str, kind: str) -> None:
    data = toy(device={**toy()["device"], "num_qubits": 1}, connectivity={"edges": []})
    data["gates"] = {"x": {"avg_infidelity": 0.2}}
    text = json.dumps(data)
    read = getattr(Profile, method)

    with pytest.raises(ValidationError) as caught:
        read(text.replace('"gates": {', '"gates": {"x": {"avg_infidelity": 0.01}, '))

    assert [
        (e["type"], e["msg"].endswith("the key gates.x appears twice"))
        for e in caught.value.errors()
    ] == [(kind, True)]
    assert read(text) == read(text.encode()) == Profile.model_validate(data)
    with pytest.raises(ValidationError):
        read(text[:-1])


def test_gzip_output_is_reproducible(tmp_path: Path) -> None:
    profile = Profile.model_validate(toy())
    assert (
        profile.save(tmp_path / "a.json.gz").read_bytes()
        == profile.save(tmp_path / "b.json.gz").read_bytes()
    )


@pytest.mark.parametrize("name", ["p.json", "p.json.gz"])
def test_a_failed_save_leaves_the_existing_file_whole(tmp_path: Path, name: str) -> None:
    resource = pytest.importorskip("resource")
    old = Profile.model_validate(toy())
    new = Profile.model_validate(toy(readout={"error": 0.05}))
    target = old.save(tmp_path / "out" / name)
    before = target.read_bytes()
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    resource.setrlimit(resource.RLIMIT_FSIZE, (20, hard))
    try:
        with pytest.raises(OSError, match="File too large"):
            new.save(target)
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
    assert target.read_bytes() == before
    assert list(target.parent.iterdir()) == [target]
    assert load_file(new.save(target)) == new


def test_an_interrupted_save_leaves_the_existing_file_whole(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = Profile.model_validate(toy()).save(tmp_path / "p.json")
    before = target.read_bytes()

    def interrupt(src: Path, dst: Path) -> None:
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", interrupt)
        with pytest.raises(KeyboardInterrupt):
            Profile.model_validate(toy(readout={"error": 0.05})).save(target)
    assert target.read_bytes() == before
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize(
    ("failing", "error"), [("write", "Bad file descriptor"), ("replace", "Input/output error")]
)
def test_a_failed_save_never_deletes_a_file_that_replaced_its_hidden_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failing: str, error: str
) -> None:
    path = tmp_path / "noise.json"
    theirs: list[Path] = []

    def other_writer(hidden: Any) -> None:
        other = tmp_path / "theirs"
        other.write_bytes(b"theirs")
        os.rename(other, hidden)
        theirs.append(Path(hidden))

    if failing == "write":
        real_open = os.open

        def read_only_then_replaced(file: Any, flags: int, *args: Any, **kwargs: Any) -> int:
            if not Path(file).name.startswith(f".{path.name}."):
                return real_open(file, flags, *args, **kwargs)
            fd = real_open(file, flags & ~os.O_WRONLY, *args, **kwargs)
            other_writer(file)
            return fd

        monkeypatch.setattr(os, "open", read_only_then_replaced)
    else:

        def replaced_then_failing(src: Any, dst: Any) -> None:
            other_writer(src)
            raise OSError(errno.EIO, os.strerror(errno.EIO))

        monkeypatch.setattr(os, "replace", replaced_then_failing)
    with pytest.raises(OSError, match=error):
        Profile.model_validate(toy()).save(path)
    assert theirs[0].read_bytes() == b"theirs"
    assert not path.exists()


def test_a_save_never_publishes_a_file_that_replaced_its_hidden_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = Profile.model_validate(toy()).save(tmp_path / "p.json")
    before = target.read_bytes()
    real_copymode = shutil.copymode
    theirs: list[Path] = []

    def copymode_after_another_writer(src: Any, dst: Any) -> None:
        other = tmp_path / "theirs"
        other.write_bytes(b"theirs")
        os.rename(other, dst)
        theirs.append(Path(dst))
        real_copymode(src, dst)

    monkeypatch.setattr(shutil, "copymode", copymode_after_another_writer)
    with pytest.raises(FileExistsError) as info:
        Profile.model_validate(toy(readout={"error": 0.05})).save(target)
    assert isinstance(info.value, nv.NoiseVaultError)
    assert (info.value.message, info.value.hint) == (
        f"the save wrote nothing to {target}, because another file replaced the hidden copy"
        f" {theirs[0].name}",
        "save again",
    )
    assert (target.read_bytes(), theirs[0].read_bytes()) == (before, b"theirs")
    assert sorted(tmp_path.iterdir()) == sorted([target, theirs[0]])


@pytest.fixture
def umask(request: pytest.FixtureRequest) -> Iterator[int]:
    caller = os.umask(request.param)
    yield request.param
    os.umask(caller)


@pytest.mark.parametrize(
    ("umask", "mode"), [(0o022, 0o644), (0o077, 0o600)], indirect=["umask"], ids=["022", "077"]
)
def test_a_new_file_gets_the_mode_the_umask_allows(tmp_path: Path, umask: int, mode: int) -> None:
    fresh = Profile.model_validate(toy()).save(tmp_path / "fresh.json.gz")
    assert oct(stat.S_IMODE(fresh.stat().st_mode)) == oct(mode)


@pytest.mark.parametrize(
    ("umask", "mode"),
    [(0o022, 0o600), (0o077, 0o640)],
    indirect=["umask"],
    ids=["600-under-022", "640-under-077"],
)
def test_replacing_a_file_keeps_its_mode(tmp_path: Path, umask: int, mode: int) -> None:
    profile = Profile.model_validate(toy())
    shared = profile.save(tmp_path / "shared.json")
    shared.chmod(mode)
    profile.save(shared)
    assert oct(stat.S_IMODE(shared.stat().st_mode)) == oct(mode)


@pytest.mark.parametrize(
    ("umask", "mode"),
    [(0o022, 0o600), (0o022, 0o640)],
    indirect=["umask"],
    ids=["600-under-022", "640-under-022"],
)
def test_a_replacement_in_progress_is_never_more_open_than_the_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, umask: int, mode: int
) -> None:
    profile = Profile.model_validate(toy())
    private = profile.save(tmp_path / "private.json")
    private.chmod(mode)
    seen: list[tuple[int, int]] = []

    def look(fd: int) -> None:
        info = os.fstat(fd)
        seen.append((stat.S_IMODE(info.st_mode), info.st_size))

    def open_and_look(fd: int, *args, **kwargs):
        handle = open(fd, *args, **kwargs)
        write = handle.write

        def write_and_look(data: bytes) -> int:
            count = write(data)
            handle.flush()
            look(fd)
            return count

        look(fd)
        handle.write = write_and_look
        return handle

    with monkeypatch.context() as patch:
        patch.setattr(nv.profile, "open", open_and_look, raising=False)
        profile.save(private)
    assert seen and seen[-1][1] == private.stat().st_size
    assert [oct(bits) for bits, _ in seen if bits & ~mode] == []


@pytest.mark.parametrize("umask", [0o077], indirect=True, ids=["077"])
@pytest.mark.parametrize("savers", [1, 4], ids=["1-saver", "4-savers"])
def test_saving_leaves_the_process_umask_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, umask: int, savers: int
) -> None:
    set_umask, rename = os.umask, os.replace
    seen = []

    def read_umask() -> None:
        mask = set_umask(0)
        set_umask(mask)
        seen.append(mask)

    together = threading.Barrier(savers, action=read_umask, timeout=10)

    def set_umask_together(mask: int) -> int:
        previous = set_umask(mask)
        together.wait()
        return previous

    def rename_together(src: Path, dst: Path) -> None:
        together.wait()
        rename(src, dst)

    paths = [tmp_path / f"p{i}.json" for i in range(savers)]
    with monkeypatch.context() as patch, ThreadPoolExecutor(savers) as pool:
        patch.setattr(os, "umask", set_umask_together)
        patch.setattr(os, "replace", rename_together)
        list(pool.map(Profile.model_validate(toy()).save, paths))
    assert {p.name: oct(stat.S_IMODE(p.stat().st_mode)) for p in paths} == {
        p.name: "0o600" for p in paths
    }
    read_umask()
    assert {oct(mask) for mask in seen} == {oct(umask)}


def _shuffled(value):
    if isinstance(value, dict):
        items = list(value.items())
        random.Random(len(items)).shuffle(items)
        return {k: _shuffled(v) for k, v in items}
    if isinstance(value, list):
        return [_shuffled(v) for v in value]
    return value


def test_fingerprint_ignores_key_order_provenance_and_extensions() -> None:
    data = toy(calibrations=[{"gate": "cz", "qubits": [0, 1], "avg_infidelity": 0.02}])
    base = Profile.model_validate(data)
    shuffled = Profile.model_validate(_shuffled(data))
    relabeled = Profile.model_validate(
        {**data, "provenance": {"source": "elsewhere", "notes": ["x"]}, "extensions": {"a": 1}}
    )
    assert shuffled.fingerprint == base.fingerprint == relabeled.fingerprint
    assert relabeled.artifact_hash != base.artifact_hash


def _rich_toy() -> dict:
    return toy(
        qubits=[{"index": 0, "t1_us": 100}, {"index": 2, "t1_us": 80}, {"index": 1, "t1_us": 90}],
        calibrations=[
            {"gate": "cz", "qubits": [1, 2], "avg_infidelity": 0.03},
            {"gate": "sx", "qubits": [2], "avg_infidelity": 0.002, "includes": ["spam", "leakage"]},
            {"gate": "cz", "qubits": [0, 1], "avg_infidelity": 0.02},
            {"gate": "sx", "qubits": [0], "avg_infidelity": 0.001},
        ],
        effects=[
            {"type": "leakage", "gate": "cz", "prob": 1e-4},
            {"type": "crosstalk_zz", "on": "idle", "qubits": [0, 1], "strength_hz": 5.0},
        ],
    )


@pytest.mark.parametrize(
    "rewrite",
    [
        lambda d: d["calibrations"].reverse(),
        lambda d: d["qubits"].reverse(),
        lambda d: d["effects"].reverse(),
        lambda d: d["connectivity"]["edges"].reverse(),
        lambda d: d["connectivity"].update(edges=[[1, 0], [2, 1]]),
        lambda d: d["calibrations"][1].update(includes=["leakage", "spam", "spam"]),
        lambda d: d["gates"]["cz"].update(symmetric=True, qubits=2),
        lambda d: d["gates"]["sx"].update(qubits=1),
    ],
    ids=[
        "records",
        "qubits",
        "effects",
        "edges",
        "edge-orientation",
        "includes",
        "cz-defaults",
        "sx-arity",
    ],
)
def test_fingerprint_and_saved_form_ignore_order_and_restated_defaults(rewrite) -> None:
    data = _rich_toy()
    base = Profile.model_validate(data)
    rewrite(data)
    other = Profile.model_validate(data)
    assert other.fingerprint == base.fingerprint
    assert other.to_json() == base.to_json()


def test_direction_of_a_directed_edge_is_physics() -> None:
    data = toy(connectivity={"edges": [[0, 1], [1, 2]], "directed": True})
    base = Profile.model_validate(data).fingerprint
    data["connectivity"]["edges"] = [[1, 0], [1, 2]]
    assert Profile.model_validate(data).fingerprint != base


def test_a_default_that_differs_from_the_registry_is_kept() -> None:
    profile = Profile.model_validate(toy(gates={**toy()["gates"], "cx": {"symmetric": True}}))
    assert profile.gates["cx"].symmetric is True and profile.table.symmetric("cx")


@pytest.mark.parametrize(
    "typo",
    [
        lambda d: d["gates"]["sx"].update(avg_infidelity="1e-3"),
        lambda d: d["gates"]["rz"].update(virtual="true"),
        lambda d: d["gates"]["cz"].update(qubits=2.0),
        lambda d: d["device"].update(num_qubits=3.0),
        lambda d: d["device"].update(num_qubits=True),
        lambda d: d["connectivity"].update(edges=[[0, 1], [True, 2]]),
        lambda d: d["connectivity"].update(directed=0),
        lambda d: d.update(calibrations=[{"gate": "cz", "qubits": [True, 0]}]),
        lambda d: d.update(calibrations=[{"gate": "cz", "qubits": ["0", "1"]}]),
        lambda d: d.update(qubits=[{"index": True}]),
        lambda d: d.update(readout={"error": "0.01"}),
    ],
)
def test_wrong_json_types_are_errors_not_coerced(typo) -> None:
    data = toy()
    typo(data)
    with pytest.raises(ValidationError):
        Profile.model_validate(data)


def test_ints_are_accepted_where_numbers_are_expected() -> None:
    profile = Profile.model_validate(toy(readout={"error": 0}, idle={"t1_us": 100}))
    assert profile.readout.error == 0.0 and profile.idle.t1_us == 100.0


def test_a_profile_cannot_be_changed_in_place() -> None:
    profile = Profile.model_validate(
        toy(benchmarks={"eplg": {"value": 3e-3, "layers": [1, 2]}}, extensions={"x": {"a": 1}})
    )
    for mutate in (
        lambda: profile.gates.__setitem__("sx", profile.gates["cz"]),
        lambda: profile.gates.pop("sx"),
        lambda: profile.benchmarks["eplg"].__setitem__("value", 1.0),
        lambda: profile.extensions.update(y=1),
        lambda: profile.provenance.extra.setdefault("k", 1),
    ):
        with pytest.raises(TypeError, match="immutable"):
            mutate()
    assert profile.benchmarks["eplg"]["layers"] == (1, 2)
    assert profile.to_dict()["benchmarks"] == {"eplg": {"value": 3e-3, "layers": [1, 2]}}


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("where", ["benchmarks", "extensions", "provenance.extra"])
def test_free_form_data_refuses_nonfinite_numbers(tmp_path: Path, where: str, bad: str) -> None:
    free = {"eplg": {"runs": [1, {"value": 2.5}]}}
    section, _, key = where.partition(".")
    data = toy(**{section: {key: free} if key else free})
    path = tmp_path / "p.json"
    path.write_text(json.dumps(data).replace("2.5", bad))
    with pytest.raises(ValidationError) as caught:
        load_file(path)
    message = str(caught.value)
    assert f"\n{where}\n" in message
    assert f"eplg.runs[1].value: {float(bad)} is not a finite number" in message
    path.write_text(json.dumps(data))
    saved = load_file(path).to_dict()[section]
    assert (saved[key] if key else saved) == free


@pytest.mark.parametrize(
    ("free", "message"),
    [
        ({"runs": {0: 0.8, "0": 0.9}}, "runs: the key 0 is not a string"),
        ({"runs": [{"seeds": {1, 2}}]}, r"runs\[0\]\.seeds: a set is not JSON data"),
        ({"when": datetime(2025, 1, 1, tzinfo=UTC)}, "when: a datetime is not JSON data"),
        ({"raw": b"\x00"}, "raw: a bytes is not JSON data"),
        ({b"runs": 0.8, "runs": 0.9}, "the key b'runs' is not a string"),
        ({0: 0.8}, "the key 0 is not a string"),
    ],
)
@pytest.mark.parametrize("where", ["benchmarks", "extensions", "provenance.extra"])
def test_free_form_data_must_be_json_data(where: str, free: dict, message: str) -> None:
    section, _, key = where.partition(".")
    data = toy(**{section: {key: free} if key else free})
    with pytest.raises(ValidationError, match=message):
        Profile.from_dict(data)


def _nested(levels: int, shape: str) -> Any:
    value: Any = 0
    for _ in range(levels):
        value = {"a": value} if shape == "object" else [value]
    return value


@pytest.mark.parametrize(("levels", "shape"), [(65, "array"), (65, "object"), (600, "array")])
@pytest.mark.parametrize("where", ["benchmarks", "extensions", "provenance.extra"])
def test_free_form_data_nested_past_64_levels_is_refused(
    where: str, levels: int, shape: str
) -> None:
    section, _, key = where.partition(".")
    free = {"deep": _nested(levels - 1, shape)}
    data = toy()
    data[section] = {key: free} if key else free
    with pytest.raises(ValidationError) as caught:
        Profile.from_dict(data)
    assert [(e["loc"], e["msg"]) for e in caught.value.errors()] == [
        (tuple(where.split(".")), "Value error, nested more than 64 levels deep")
    ]


@pytest.mark.parametrize("shape", ["array", "object"])
def test_free_form_data_64_levels_deep_saves_and_loads(tmp_path: Path, shape: str) -> None:
    free = {"deep": _nested(63, shape)}
    profile = Profile.from_dict(toy(extensions=free))
    loaded = load_file(profile.save(tmp_path / "deep.json"))
    assert loaded.to_dict()["extensions"] == free


def test_a_gate_key_must_be_a_string() -> None:
    gates = {b"x": {"avg_infidelity": 0.2}, "x": {"avg_infidelity": 0.03}}
    with pytest.raises(ValidationError, match="the key b'x' is not a string"):
        Profile.from_dict(toy(gates=gates))


def test_free_form_data_is_frozen_at_every_level() -> None:
    from types import MappingProxyType

    data = toy()
    data["benchmarks"] = {"eplg": MappingProxyType({"layers": [1, (2, [3]), {"k": [4]}]})}
    profile = Profile.from_dict(data)
    fingerprint = profile.fingerprint
    layers = profile.benchmarks["eplg"]["layers"]
    assert layers == (1, (2, (3,)), {"k": (4,)})
    for mutate in (lambda: layers[2].update(k=5), lambda: profile.benchmarks["eplg"].pop("x")):
        with pytest.raises(TypeError, match="immutable"):
            mutate()
    assert Profile.from_dict(profile.to_dict()).fingerprint == fingerprint


def _every_text_field() -> dict:
    return toy(
        device={**toy()["device"], "processor": "Falcon"},
        gates={
            **toy()["gates"],
            "cz": {"avg_infidelity": 1e-2, "assumption": "the device median"},
            "my_gate": {"qubits": 1, "avg_infidelity": 1e-3},
        },
        qubits=[{"index": 0, "label": "Q0"}],
        calibrations=[{"gate": "cz", "qubits": [0, 1], "avg_infidelity": 0.02}],
        effects=[{"type": "leakage", "gate": "cz", "prob": 1e-4}],
        benchmarks={"eplg": {"method": "layer", "runs": ["a"]}},
        provenance={
            "source": "s",
            "source_url": "https://example.org",
            "license": "CC-BY-4.0",
            "attribution": "a",
            "tool": "t",
            "derived_from": "d",
            "notes": ["n"],
            "extra": {"k": "v"},
        },
        extensions={"x": {"y": ["z"]}},
    )


def _one_surrogate(value: Any, where: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, str):
        yield where, value + "\ud800"
    elif isinstance(value, dict):
        for key, item in value.items():
            renamed = {(k + "\ud800" if k == key else k): v for k, v in value.items()}
            yield f"{where}.{key} (key)", renamed
            for at, changed in _one_surrogate(item, f"{where}.{key}"):
                yield at, {**value, key: changed}
    elif isinstance(value, list):
        for i, item in enumerate(value):
            for at, changed in _one_surrogate(item, f"{where}[{i}]"):
                yield at, [*value[:i], changed, *value[i + 1 :]]


def test_no_string_or_key_with_an_unpaired_surrogate_validates() -> None:
    data = _every_text_field()
    Profile.model_validate(data)
    not_refused = {}
    for where, changed in _one_surrogate(data):
        try:
            Profile.model_validate(changed)
        except ValidationError:
            continue
        except UnicodeEncodeError:
            not_refused[where] = "UnicodeEncodeError"
        else:
            not_refused[where] = "validates"
    assert not_refused == {}


@pytest.mark.parametrize(
    ("section", "loc", "message"),
    [
        (
            {"provenance": {"notes": ["\ud800"]}},
            ("provenance", "notes", 0),
            "the string holds the unpaired surrogate \\ud800",
        ),
        (
            {"extensions": {"x": ["a\ud800"]}},
            ("extensions",),
            "x[0]: the string holds the unpaired surrogate \\ud800",
        ),
        (
            {"extensions": {"x": {"\udfffb": 1}}},
            ("extensions",),
            "x: the key '\\udfffb' holds the unpaired surrogate \\udfff",
        ),
    ],
)
def test_a_file_with_an_escaped_unpaired_surrogate_is_refused_where_it_stands(
    tmp_path: Path, section: dict, loc: tuple, message: str
) -> None:
    path = tmp_path / "p.json"
    path.write_text(json.dumps(toy(**section)))
    with pytest.raises(ValidationError) as caught:
        load_file(path)
    assert [(e["loc"], e["msg"]) for e in caught.value.errors()] == [
        (
            loc,
            f"Value error, {message}, which UTF-8 cannot encode."
            " Remove the surrogate or write the whole character",
        )
    ]


CACHED = ("fingerprint", "artifact_hash", "calibration_fingerprint", "table")


class _Holder(BaseModel):
    profile: Profile


REVALIDATIONS = {
    "model_validate": Profile.model_validate,
    "model field": lambda profile: _Holder(profile=profile).profile,
    "TypeAdapter": lambda profile: TypeAdapter(list[Profile]).validate_python([profile])[0],
    "dict(profile)": lambda profile: Profile(**dict(profile)),
}


@pytest.mark.parametrize("member", CACHED)
@pytest.mark.parametrize("revalidate", REVALIDATIONS.values(), ids=REVALIDATIONS.keys())
def test_a_profile_revalidates_after_a_cached_member_is_read(member, revalidate) -> None:
    profile = Profile.model_validate(toy())
    getattr(profile, member)
    again = revalidate(profile)
    assert again == profile
    assert again.fingerprint == profile.fingerprint


def test_a_profile_that_skipped_validation_is_checked_when_nested() -> None:
    profile = Profile.model_validate(toy())
    unchecked = Profile.model_construct(**{**dict(profile), "noisevault": "0.1"})
    with pytest.raises(ValidationError, match="noisevault"):
        _Holder(profile=unchecked)


CLONES = {
    "copy": copy.copy,
    "deepcopy": copy.deepcopy,
    "pickle": lambda profile: pickle.loads(pickle.dumps(profile)),
    "model_copy": Profile.model_copy,
    "model_copy deep": lambda profile: profile.model_copy(deep=True),
}


@pytest.mark.parametrize("clone", CLONES.values(), ids=CLONES.keys())
def test_a_clone_of_a_used_profile_owns_its_table_and_revalidates(clone) -> None:
    profile = Profile.model_validate(toy(benchmarks={"eplg": {"value": 3e-3}}))
    for member in CACHED:
        getattr(profile, member)
    twin = clone(profile)
    assert twin == profile
    assert twin.fingerprint == profile.fingerprint
    assert twin.table.profile is twin
    assert Profile.model_validate(twin) == profile


def test_hashes_and_table_are_computed_once(monkeypatch) -> None:
    profile = Profile.model_validate(toy())
    hashed = []
    sha256 = profile_module._sha256
    monkeypatch.setattr(profile_module, "_sha256", lambda data: hashed.append(data) or sha256(data))
    reads = {
        (profile.fingerprint, profile.artifact_hash, profile.calibration_fingerprint)
        for _ in range(3)
    }
    assert len(reads) == 1
    assert len(hashed) == 3
    assert profile.table is profile.table


@pytest.mark.parametrize("member", CACHED)
def test_cached_members_cannot_be_assigned(member) -> None:
    profile = Profile.model_validate(toy())
    value = getattr(profile, member)
    with pytest.raises(ValidationError, match="frozen"):
        setattr(profile, member, "0" * 64)
    assert getattr(profile, member) is value


def test_model_copy_with_update_is_validated() -> None:
    profile = Profile.model_validate(toy())
    with pytest.raises(ValidationError, match="not defined in gates"):
        profile.model_copy(update={"calibrations": [{"gate": "nope", "qubits": [0]}]})
    changed = profile.model_copy(update={"calibrations": [{"gate": "sx", "qubits": [0]}]})
    assert changed.calibrations[0].gate == "sx"


@pytest.mark.parametrize(
    ("section", "change", "match"),
    [
        ("idle", {"t1_us": -10}, "t1_us must be positive"),
        ("readout", {"error": -0.1}, "error\n.*greater than or equal to 0"),
        ("prep", {"error": 2.0}, "error\n.*less than or equal to 1"),
    ],
)
def test_nested_models_are_validated_inside_a_profile(section, change, match) -> None:
    profile = Profile.model_validate(
        toy(idle={"t1_us": 100}, readout={"error": 0.01}, prep={"error": 1e-3})
    )
    invalid = getattr(profile, section).model_copy(update=change)
    with pytest.raises(ValidationError, match=match):
        profile.model_copy(update={section: invalid})
    with pytest.raises(ValidationError, match=match):
        Profile(**{**dict(profile), section: invalid})


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["gates"]["sx"].update(avg_infidelity=1.000001e-3),
        lambda d: d["gates"]["cz"].update(duration_ns=71),
        lambda d: d["connectivity"].update(directed=True),
        lambda d: d.update(readout={"error": 0.01}),
        lambda d: d["device"].update(num_qubits=4),
    ],
)
def test_fingerprint_changes_with_the_physics(change) -> None:
    data = toy()
    before = Profile.model_validate(data).fingerprint
    change(data)
    assert Profile.model_validate(data).fingerprint != before


def test_fingerprint_is_the_sha256_of_canonical_physics() -> None:
    import hashlib

    profile = Profile.model_validate(toy())
    physics = {k: v for k, v in profile.to_dict().items() if k not in ("provenance", "extensions")}
    text = json.dumps(physics, sort_keys=True, separators=(",", ":"))
    assert profile.fingerprint == hashlib.sha256(text.encode()).hexdigest()
    assert profile.short_fingerprint == "nv:" + profile.fingerprint[:12]


def test_model_copy_recomputes_hashes() -> None:
    profile = Profile.model_validate(toy())
    old = profile.fingerprint
    readout = Profile.model_validate(toy(readout={"error": 0.02})).readout
    changed = profile.model_copy(update={"readout": readout})
    assert changed.fingerprint != old
    assert changed.table.qubit(0).readout == (0.02, 0.02)


@pytest.mark.parametrize(
    ("vendor", "name", "expected"),
    [
        ("ibm", "ibm_fez", "ibm_fez"),
        ("google", "willow_pink", "google_willow_pink"),
        ("quantinuum", "H2-1", "quantinuum_h2-1"),
        ("ionq", "forte-1", "ionq_forte-1"),
        (None, "toy-ion-20", "toy-ion-20"),
    ],
)
def test_profile_id(vendor, name, expected) -> None:
    assert profile_id(vendor, name) == expected


@pytest.mark.parametrize("name", ["Aria(1)", "t:x", "a,b"])
def test_profile_id_must_be_loadable_as_a_ref(name: str) -> None:
    with pytest.raises(ValidationError, match="profile id"):
        Profile.uniform(
            name,
            technology="trapped_ion",
            num_qubits=2,
            one_qubit_error=1e-3,
            two_qubit_error=1e-2,
        )


def _named(vendor: str | None, name: str) -> Profile:
    return Profile.model_validate(
        toy(device={"vendor": vendor, "name": name, "technology": "other", "num_qubits": 3})
    )


@pytest.mark.parametrize(
    ("vendor", "name", "ident", "suffix", "rename"),
    [
        (None, "noise.json", "noise.json", ".json", "noise_json"),
        (None, "Noise.JSON.GZ", "noise.json.gz", ".gz", "noise.json_gz"),
        ("acme", "chip.gz", "acme_chip.gz", ".gz", "acme_chip_gz"),
    ],
)
def test_a_profile_id_that_a_ref_reads_as_a_file_is_refused_with_a_name_that_loads(
    vendor: str | None, name: str, ident: str, suffix: str, rename: str
) -> None:
    with pytest.raises(ValidationError) as info:
        _named(vendor, name)
    assert (
        f"device: the profile id {ident!r} (from vendor and name) ends in {suffix}, so nv.load"
        " and the nv commands read the id as a file path. Give the device another name, such as"
        f" {rename}"
    ) in str(info.value)
    renamed = _named(vendor, rename)
    renamed.save(vault_path(renamed))
    assert nv.load(renamed.id) == renamed


@pytest.mark.parametrize("name", ["noise.jsonl", "noise.json.v2", "gz", "noise_gz"])
def test_a_profile_id_with_json_or_gz_inside_loads_by_its_ref(name: str) -> None:
    profile = _named(None, name)
    profile.save(vault_path(profile))
    assert nv.load(profile.id) == profile


def test_parse_ref(tmp_path: Path) -> None:
    assert parse_ref("ibm_fez") == Ref("ibm_fez")
    assert parse_ref("IBM_Fez@2025-02-26") == Ref("ibm_fez", date=date(2025, 2, 26))
    assert parse_ref("ibm_fez@2025-02-26T09:12:00Z") == Ref(
        "ibm_fez", timestamp=datetime(2025, 2, 26, 9, 12, tzinfo=UTC)
    )
    assert parse_ref("ibm_fez@2025-02-26T10:12:00+01:00").timestamp == datetime(
        2025, 2, 26, 9, 12, tzinfo=UTC
    )
    assert parse_ref("profiles/fez.json") == Path("profiles/fez.json")
    assert parse_ref("fez.json.gz") == Path("fez.json.gz")
    for bad in ("ibm fez", "ibm_fez@yesterday", "ibm_fez@2025-02-26T09:12:00"):
        with pytest.raises(ValueError):
            parse_ref(bad)


@pytest.mark.parametrize(
    ("ref", "message"),
    [
        ("ibm_fez@", "after @ give a date (2025-02-26) or a timestamp with timezone"),
        ("ibm_fez@  ", "after @ give a date"),
        ("ibm_fez@2025-13-40", "2025-13-40 is not a calendar date. Give one such as 2025-02-26"),
    ],
)
def test_an_incomplete_or_impossible_date_is_refused(ref: str, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        parse_ref(ref)
    with pytest.raises(ValueError, match=re.escape(message)):
        nv.load(ref)


def test_uniform_profile() -> None:
    profile = Profile.uniform(
        "toy-atoms",
        technology="neutral_atom",
        num_qubits=100,
        one_qubit_error=1e-3,
        two_qubit_error=5e-3,
        readout_error=0.01,
        t1_us=1e6,
        two_qubit_ns=250,
    )
    table = profile.table
    assert table.gate("h", (3,)).avg_infidelity == 1e-3
    assert table.gate("cz", (0, 99)).avg_infidelity == 5e-3
    assert table.gate("cz", (0, 99)).duration_ns == 250
    assert table.gate("s", (0,)).state == "ideal"
    assert table.qubit(7).readout == (0.01, 0.01)
    assert table.qubit(7).t1_ns == 1e9
    assert profile.provenance.data_kind == "hypothetical"
    assert "swap" not in profile.gates


@pytest.mark.parametrize("two", [{"two_qubit_error": 2e-2}, {}])
def test_a_one_qubit_uniform_profile_defines_only_one_qubit_gates(two: dict) -> None:
    profile = Profile.uniform(
        "u", technology="superconducting", num_qubits=1, one_qubit_error=1e-2, **two
    )
    table = profile.table
    assert {table.arity(name) for name in profile.gates} == {1}
    assert table.gate("h", (0,)).avg_infidelity == 1e-2
    assert table.edges() == []


def test_a_uniform_profile_on_two_qubits_needs_two_qubit_error() -> None:
    with pytest.raises(ValueError, match="a 2-qubit device needs two_qubit_error"):
        Profile.uniform("u", technology="superconducting", num_qubits=2, one_qubit_error=1e-2)


@pytest.mark.parametrize("coherence", [{"t1_us": 0}, {"t2_us": 0}])
def test_uniform_rejects_zero_coherence_times(coherence: dict) -> None:
    with pytest.raises(ValidationError, match="must be positive"):
        Profile.uniform(
            "u",
            technology="trapped_ion",
            num_qubits=2,
            one_qubit_error=1e-3,
            two_qubit_error=1e-2,
            **coherence,
        )


def test_json_schema_describes_the_format() -> None:
    schema = json_schema()
    assert {"noisevault", "device", "gates", "calibrations"} <= set(schema["properties"])
    assert schema["required"] == ["noisevault", "device", "connectivity", "gates"]


def test_citation_names_source_and_full_fingerprint() -> None:
    profile = Profile.model_validate(
        toy(provenance={"attribution": "Test Lab", "source": "hand entry", "license": "CC0-1.0"})
    )
    assert profile.fingerprint in profile.citation()
    bib = profile.citation("bibtex")
    assert bib.startswith("@misc{nv_test_toy") and "Test Lab" in bib and profile.fingerprint in bib


def test_citation_names_the_noisevault_version_and_the_ref_to_load() -> None:
    data = toy()
    undated = Profile.model_validate(data)
    assert f"NoiseVault {nv.__version__} profile test_toy, fingerprint" in undated.citation()
    data["device"]["calibrated_at"] = "2025-02-26T09:12:00Z"
    dated = Profile.model_validate(data)
    ref = f"NoiseVault {nv.__version__} profile test_toy@2025-02-26T09:12:00Z"
    assert f"{ref}, fingerprint sha256:{dated.fingerprint}." in dated.citation()
    bib = ref.replace("_", "\\_")
    assert f"howpublished = {{{bib}, sha256:{dated.fingerprint}}}" in dated.citation("bibtex")


def test_undated_citation_states_no_year() -> None:
    data = toy()
    bib = Profile.model_validate(data).citation("bibtex")
    assert bib.startswith("@misc{nv_test_toy,") and "year" not in bib and "undated" not in bib
    data["device"]["calibrated_at"] = "2025-02-26T09:12:00Z"
    dated = Profile.model_validate(data).citation("bibtex")
    assert "year = {2025}" in dated and dated.startswith("@misc{nv_test_toy_2025_02_26,")


def _summary_line(profile: Profile, gate: str) -> str:
    lines = [" ".join(line.split()) for line in profile.summary().splitlines()]
    return next(line for line in lines if line.split()[0] == gate)


def test_summary_shows_a_noisy_override_of_a_virtual_gate() -> None:
    data = toy(
        calibrations=[{"gate": "rz", "qubits": [0], "virtual": False, "avg_infidelity": 2e-3}]
    )
    data["device"]["num_qubits"] = 1
    data["connectivity"] = "all_to_all"
    del data["gates"]["cz"]
    assert _summary_line(Profile.from_dict(data), "rz") == "rz 1q avg infidelity 0.002 everywhere"


@pytest.mark.parametrize(
    ("gates", "calibrations", "gate", "line"),
    [
        (
            {},
            [{"gate": "rz", "qubits": [0], "virtual": False, "avg_infidelity": 2e-3}],
            "rz",
            "rz 1q median avg infidelity 0.002 over 1 locus, virtual on 2 loci",
        ),
        (
            {"cz": {"avg_infidelity": 1e-2, "disabled": True}},
            [{"gate": "cz", "qubits": [0, 1], "disabled": False}],
            "cz",
            "cz 2q median avg infidelity 0.01 over 1 locus, disabled on 1 locus",
        ),
        (
            {"cz": {"avg_infidelity": 1e-2, "disabled": True}},
            [],
            "cz",
            "cz 2q disabled",
        ),
        (
            {},
            [{"gate": "sx", "qubits": [2], "disabled": True}],
            "sx",
            "sx 1q median avg infidelity 0.001 over 2 loci, disabled on 1 locus",
        ),
        ({}, [], "rz", "rz 1q virtual"),
        ({}, [], "cz", "cz 2q avg infidelity 0.01 everywhere"),
        (
            {},
            [
                {"gate": "cz", "qubits": [0, 1], "avg_infidelity": 1e-2},
                {"gate": "cz", "qubits": [1, 0], "disabled": True},
            ],
            "cz",
            "cz 2q median avg infidelity 0.01 over 2 loci, disabled on 1 locus",
        ),
        (
            {},
            [
                {"gate": "cz", "qubits": [0, 1], "avg_infidelity": 1e-2},
                {"gate": "cz", "qubits": [1, 0], "avg_infidelity": 1e-2},
            ],
            "cz",
            "cz 2q avg infidelity 0.01 everywhere",
        ),
    ],
)
def test_summary_describes_each_gate_as_resolved(
    gates: dict, calibrations: list, gate: str, line: str
) -> None:
    data = toy(calibrations=calibrations)
    data["gates"].update(gates)
    assert _summary_line(Profile.from_dict(data), gate) == line


@pytest.mark.parametrize(
    ("reverse", "line"),
    [
        ({"disabled": True}, "cz 2q median avg infidelity 0.01 over 1 locus, disabled on 1 locus"),
        ({"avg_infidelity": 0.2}, "cz 2q median avg infidelity 0.105 over 2 loci"),
        ({"avg_infidelity": 0.01}, "cz 2q avg infidelity 0.01 everywhere"),
        ({}, "cz 2q avg infidelity 0.01 everywhere"),
    ],
)
def test_summary_keeps_a_reverse_order_that_resolves_differently(reverse: dict, line: str) -> None:
    calibrations = [{"gate": "cz", "qubits": [0, 1], "avg_infidelity": 1e-2}]
    if reverse:
        calibrations.append({"gate": "cz", "qubits": [1, 0], **reverse})
    data = toy(connectivity="all_to_all", calibrations=calibrations)
    data["device"]["num_qubits"] = 2
    assert _summary_line(Profile.from_dict(data), "cz") == line


@pytest.mark.parametrize(
    ("qubits", "lines"),
    [
        (
            [
                {"index": 0, "t1_us": 300, "readout": {"p1_given_0": 0.008, "p0_given_1": 0.016}},
                {"index": 1, "t1_us": 280, "readout": {"p1_given_0": 0.007, "p0_given_1": 0.013}},
                {"index": 2, "t1_us": 260, "readout": {"p1_given_0": 0.01, "p0_given_1": 0.018}},
                {"index": 3, "t1_us": 240, "readout": {"error": 0.016}, "disabled": True},
            ],
            ["median T1 280.0 us", "median readout error 0.012"],
        ),
        (
            [{"index": 3, "t1_us": 240, "readout": {"error": 0.016}, "disabled": True}],
            ["readout unknown"],
        ),
    ],
)
def test_summary_medians_leave_out_a_disabled_qubit(qubits: list, lines: list) -> None:
    data = toy(qubits=qubits)
    data["device"]["num_qubits"] = 4
    summary = Profile.from_dict(data).summary().splitlines()
    assert [line.strip() for line in summary if "T1" in line or "readout" in line] == lines


COUNTS = "sha256:3fa1c2d4e5b6" + "0" * 52
FITTED = {
    "gates": {"factor": 1.84, "low": 1.54, "high": 2.12},
    "readout": {"factor": 1.58, "low": 1.32, "high": 1.84},
    "fit": {
        "counts": COUNTS,
        "source": "hardware",
        "qubits": [148, 149, 150, 151],
        "run_at": "2026-04-16T09:30:02Z",
        "calibration": "609c845ed934" + "0" * 52,
        "p_value": 0.41,
        "impossible_shots": 0,
    },
}
FITTED_LINES = (
    "gate errors x1.84 (95% interval 1.54 to 2.12)",
    "readout errors x1.58 (95% interval 1.32 to 1.84)",
    "fitted to hardware counts sha256:3fa1c2d4e5b6",
    "on qubits 148-149-150-151, run 2026-04-16 (p = 0.41)",
    "T1, T2 and preparation error are not scaled",
)


def _block(base: Profile, **fit) -> dict:
    bound = {"qubits": [0, 1, 2], "calibration": base.fingerprint}
    return {**FITTED, "fit": {**FITTED["fit"], **bound, **fit}}


def _fitted(**fit) -> tuple[Profile, Profile]:
    base = Profile.model_validate(toy(readout={"error": 0.01}))
    return base, base.model_copy(update={"unmodeled_error": _block(base, **fit)})


def _first_error(data: dict) -> str:
    with pytest.raises(ValidationError) as caught:
        Profile.model_validate(data)
    return caught.value.errors()[0]["msg"]


def test_no_bundled_profile_sets_unmodeled_error() -> None:
    for info in bundled_profiles():
        assert "unmodeled_error" not in info.load().to_dict(), info.ref


def test_a_factor_is_fingerprinted_and_uncorrected_gives_back_the_calibration() -> None:
    base = Profile.model_validate(toy())
    what_if = base.model_copy(update={"unmodeled_error": {"gates": {"factor": 2.3}}})
    assert what_if.to_dict()["unmodeled_error"] == {"gates": {"factor": 2.3}}
    assert what_if.fingerprint != base.fingerprint
    assert what_if.uncorrected().fingerprint == base.fingerprint
    assert what_if.calibration_fingerprint == base.calibration_fingerprint == base.fingerprint
    assert what_if.uncorrected().to_dict() == base.to_dict()
    assert base.uncorrected() is base


@pytest.mark.parametrize("p_value", [0.41, None])
def test_a_fitted_profile_saves_its_block_between_effects_and_benchmarks(
    tmp_path: Path, p_value: float | None
) -> None:
    base = Profile.model_validate(
        toy(
            readout={"error": 0.01},
            effects=[{"type": "leakage", "gate": "cz", "prob": 1e-4}],
            benchmarks={"eplg": 3e-3},
        )
    )
    block = _block(base, p_value=p_value)
    fitted = base.model_copy(update={"unmodeled_error": block})
    saved = json.loads(fitted.save(tmp_path / "fitted.json").read_text())
    assert list(saved)[-4:] == ["effects", "unmodeled_error", "benchmarks", "provenance"]
    assert saved["unmodeled_error"] == block
    loaded = load_file(tmp_path / "fitted.json")
    assert loaded == fitted and loaded.fingerprint == fitted.fingerprint


def test_a_calibration_edit_under_a_fit_is_refused_in_one_line() -> None:
    base, fitted = _fitted()
    data = fitted.to_dict()
    data["gates"]["sx"]["avg_infidelity"] = 2e-3
    edited = Profile.model_validate({k: v for k, v in data.items() if k != "unmodeled_error"})
    assert _first_error(data) == (
        f"Value error, unmodeled_error.fit.calibration: fitted to calibration"
        f" {base.short_fingerprint}, but this profile's calibration is"
        f" {edited.short_fingerprint}. Drop unmodeled_error or refit with nv compare"
    )
    relabeled = {**fitted.to_dict(), "provenance": {"source": "elsewhere"}, "extensions": {"a": 1}}
    assert Profile.model_validate(relabeled).unmodeled_error == fitted.unmodeled_error


@pytest.mark.parametrize(
    "factor",
    [
        {"factor": 2.3},
        {"factor": 0},
        {"factor": 1.84, "low": 1.54, "high": 2.12},
        {"factor": 0.096, "high": 0.311, "bound": "lower"},
        {"factor": 20.0, "low": 11.2, "bound": "upper"},
    ],
)
def test_an_error_factor_may_be_alone_or_carry_its_interval(factor: dict) -> None:
    assert ErrorFactor.model_validate(factor).model_dump(exclude_none=True) == factor


@pytest.mark.parametrize(
    ("factor", "message"),
    [
        ({"factor": 1.5, "low": 1.2}, 'give low and high together, or set bound "lower"'),
        ({"factor": 1.5, "high": 2.0}, 'give low and high together, or set bound "lower"'),
        ({"factor": 0.1, "low": 0.05, "high": 0.3, "bound": "lower"}, 'bound "lower" needs high'),
        ({"factor": 0.1, "bound": "lower"}, 'bound "lower" needs high and no low'),
        ({"factor": 20.0, "low": 11.2, "high": 20.0, "bound": "upper"}, 'bound "upper" needs low'),
        ({"factor": 2.5, "low": 1.54, "high": 2.12}, "factor 2.5 is above high 2.12"),
        ({"factor": 1.0, "low": 1.54, "high": 2.12}, "factor 1.0 is below low 1.54"),
        ({"factor": 0.5, "high": 0.311, "bound": "lower"}, "factor 0.5 is above high 0.311"),
        ({"factor": 5.0, "low": 11.2, "bound": "upper"}, "factor 5.0 is below low 11.2"),
        ({"factor": -1.0}, "greater than or equal to 0"),
        ({"factor": "2"}, "valid number"),
    ],
)
def test_an_error_factor_with_an_illegal_interval_is_refused(factor: dict, message: str) -> None:
    _invalid(toy(unmodeled_error={"gates": factor}), re.escape(message))


@pytest.mark.parametrize(
    ("factors", "fitted", "message"),
    [
        ({}, False, "give a gates factor, a readout factor or both"),
        ({}, True, "give a gates factor, a readout factor or both"),
        ({"gates": {"factor": 2.0}}, True, "gates: a fitted factor states its interval"),
        (
            {"readout": {"factor": 1.58, "low": 1.32, "high": 1.84}},
            False,
            "readout: an interval needs the fit it came from. Add fit, or drop low, high and bound",
        ),
    ],
)
def test_an_unmodeled_block_holds_a_factor_and_a_fit_holds_intervals(
    factors: dict, fitted: bool, message: str
) -> None:
    base = Profile.model_validate(toy(readout={"error": 0.01}))
    block = {**factors, "fit": _block(base)["fit"]} if fitted else factors
    _invalid(toy(readout={"error": 0.01}, unmodeled_error=block), re.escape(message))


@pytest.mark.parametrize("field", list(FITTED["fit"]))
def test_a_fit_states_every_field(field: str) -> None:
    block = _block(Profile.model_validate(toy(readout={"error": 0.01})))
    del block["fit"][field]
    _invalid(
        toy(readout={"error": 0.01}, unmodeled_error=block), f"fit\\.{field}\n.*Field required"
    )


@pytest.mark.parametrize(
    ("fit", "message"),
    [
        ({"impossible_shots": 9}, "impossible_shots is 9, so p_value must be 0"),
        ({"impossible_shots": 9, "p_value": None}, "impossible_shots is 9, so p_value must be 0"),
        ({"qubits": [0, 1, 1]}, "qubit 1 is listed twice"),
        ({"qubits": []}, "at least 1 item"),
        ({"qubits": [0, 3]}, "unmodeled_error.fit.qubits: qubit 3 is outside 0..2"),
        ({"counts": "sha256:3fa1"}, "pattern"),
        ({"calibration": "nv:609c845ed934"}, "pattern"),
        ({"source": "measured"}, "'hardware' or 'simulated'"),
        ({"run_at": "2026-04-16T09:30:02"}, "timezone"),
        ({"p_value": 1.5}, "less than or equal to 1"),
        ({"impossible_shots": -1}, "greater than or equal to 0"),
    ],
)
def test_a_fit_that_contradicts_itself_or_the_device_is_refused(fit: dict, message: str) -> None:
    block = _block(Profile.model_validate(toy(readout={"error": 0.01})), **fit)
    _invalid(toy(readout={"error": 0.01}, unmodeled_error=block), re.escape(message))


@pytest.mark.parametrize("fit", [{"p_value": None}, {"p_value": 0, "impossible_shots": 9}])
def test_a_fit_may_be_untestable_or_ruled_out(fit: dict) -> None:
    _, fitted = _fitted(**fit)
    assert fitted.unmodeled_error.fit.p_value == fit["p_value"]


def test_a_readout_factor_needs_readout_data() -> None:
    block = {"readout": {"factor": 1.5}}
    assert _first_error(toy(unmodeled_error=block)) == (
        "Value error, unmodeled_error.readout: the profile states no readout error,"
        " so the factor scales nothing"
    )
    qubit = [{"index": 1, "readout": {"p1_given_0": 0.01, "p0_given_1": 0.02}}]
    Profile.model_validate(toy(qubits=qubit, unmodeled_error=block))


def test_unmodeled_lines_read_as_the_show_row() -> None:
    assert UnmodeledError.model_validate(FITTED).lines() == FITTED_LINES


@pytest.mark.parametrize(
    ("change", "line"),
    [
        (
            {"gates": {"factor": 0.096, "high": 0.311, "bound": "lower"}},
            (0, "gate errors x0.096 (95% interval, at most 0.311)"),
        ),
        (
            {"readout": {"factor": 20.0, "low": 11.2, "bound": "upper"}},
            (1, "readout errors x20 (95% interval, at least 11.2)"),
        ),
        ({"source": "simulated"}, (2, "fitted to simulated counts sha256:3fa1c2d4e5b6")),
        (
            {"p_value": 0.003},
            (3, "on qubits 148-149-150-151, run 2026-04-16 (p = 0.003, a poor fit)"),
        ),
        ({"p_value": 0.01}, (3, "on qubits 148-149-150-151, run 2026-04-16 (p = 0.01)")),
        ({"p_value": None}, (3, "on qubits 148-149-150-151, run 2026-04-16 (fit not testable)")),
        ({"qubits": [0]}, (3, "on qubit 0, run 2026-04-16 (p = 0.41)")),
        (
            {"run_at": "2026-04-16T23:30:00-02:00"},
            (3, "on qubits 148-149-150-151, run 2026-04-17 (p = 0.41)"),
        ),
    ],
)
def test_unmodeled_lines_name_the_bound_the_source_and_the_fit(
    change: dict, line: tuple[int, str]
) -> None:
    axes = {k: v for k, v in change.items() if k in ("gates", "readout")}
    fit = {k: v for k, v in change.items() if k not in axes}
    block = {**FITTED, **axes, "fit": {**FITTED["fit"], **fit}}
    expected = list(FITTED_LINES)
    expected[line[0]] = line[1]
    assert UnmodeledError.model_validate(block).lines() == tuple(expected)


def test_unmodeled_lines_count_the_shots_the_profile_rules_out() -> None:
    block = {**FITTED, "fit": {**FITTED["fit"], "p_value": 0, "impossible_shots": 9}}
    assert UnmodeledError.model_validate(block).lines()[3:] == (
        "on qubits 148-149-150-151, run 2026-04-16 (p = 0, a poor fit)",
        "the profile ruled out 9 shots",
        "T1, T2 and preparation error are not scaled",
    )
    block["fit"]["impossible_shots"] = 1
    assert "the profile ruled out 1 shot" in UnmodeledError.model_validate(block).lines()


def test_a_hand_written_factor_reads_alone() -> None:
    assert UnmodeledError(gates={"factor": 2.3}).lines() == (
        "gate errors x2.3",
        "T1, T2 and preparation error are not scaled",
    )


def test_summary_states_the_calibration_and_adds_one_line_for_the_factors() -> None:
    base = Profile.model_validate(toy(readout={"error": 0.01}, idle={"t1_us": 100}))
    what_if = base.model_copy(
        update={"unmodeled_error": {"gates": {"factor": 2.3}, "readout": {"factor": 1.5}}}
    )
    assert unmodeled_note(base) == ()
    assert what_if.summary() == (
        base.summary().replace(base.short_fingerprint, what_if.short_fingerprint)
        + "\n  unmodeled error: gate errors x2.3; readout errors x1.5;"
        " T1, T2 and preparation error are not scaled"
    )


def test_summary_shortens_a_long_qubit_list_that_the_unmodeled_note_keeps_whole() -> None:
    device = toy()["device"] | {"num_qubits": 6}
    profile = Profile.model_validate(
        toy(device=device, readout={"error": 0.5}, unmodeled_error={"readout": {"factor": 1.3}})
    )
    chance = "is not scaled (no better than chance)"
    assert unmodeled_note(profile)[-1].full == f"readout of qubits 0, 1, 2, 3, 4 and 5 {chance}"
    assert profile.summary().endswith(f"; readout of qubits 0, 1, 2 and 3 more {chance}")


@pytest.mark.parametrize("source", ["hardware", "simulated"])
def test_citation_states_the_factors_and_the_counts_they_were_fitted_to(source: str) -> None:
    _, fitted = _fitted(source=source)
    clause = (
        "; unmodeled-error factors (gate error rates x1.84, readout error rates x1.58)"
        f" fitted to {source} counts {COUNTS}"
    )
    assert fitted.citation().endswith(f"fingerprint sha256:{fitted.fingerprint}{clause}.")
    assert f", sha256:{fitted.fingerprint}{clause}}},\n" in fitted.citation("bibtex")


def test_citation_of_a_hand_written_factor_states_the_factor() -> None:
    what_if = Profile.model_validate(toy(unmodeled_error={"gates": {"factor": 2.3}}))
    assert what_if.citation().endswith(
        f"sha256:{what_if.fingerprint}; unmodeled-error factors (gate error rates x2.3)."
    )
