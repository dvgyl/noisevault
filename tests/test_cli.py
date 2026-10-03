from __future__ import annotations

import gzip
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from conftest import MANILA_V01, deeper_than_the_parser_takes, migrated, require, toy
from typer.testing import CliRunner, Result

import noisevault as nv
from noisevault.cli import _PACKAGES, app
from noisevault.compare import NOTE as COMPARE_NOTE
from noisevault.counts import PlannedCircuit, plan, simulate
from noisevault.errors import REPOSITORY, SourceUnavailable, install_hint
from noisevault.profile import Profile

runner = CliRunner()


def _unstyled(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)  # typer forces color on CI runners


def test_validate_leads_with_the_ref_and_fingerprint_and_reports_the_migration() -> None:
    result = runner.invoke(app, ["validate", str(MANILA_V01)], env={"COLUMNS": "80"})
    assert result.exit_code == 0, result.output
    fingerprint = migrated(MANILA_V01).short_fingerprint
    assert result.stdout == (
        f"ok: ibm_manila@2024-05-27 {fingerprint}, 5 qubits, 6 gates, 28 records\n"
    )
    assert "warning: upgraded a NoiseVault 0.1 file" in result.stderr


def test_validate_strict_fails_on_warnings() -> None:
    result = runner.invoke(app, ["validate", "--strict", str(MANILA_V01)])
    assert result.exit_code == 1


def test_validate_lists_every_error(tmp_path: Path) -> None:
    data = toy(calibrations=[{"gate": "ecr", "qubits": [0, 1]}, {"gate": "cz", "qubits": [0]}])
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 1
    errors = [line for line in result.stderr.splitlines() if line.startswith("error: ")]
    assert len(errors) == 2 and any("not defined in gates" in line for line in errors)
    assert "Traceback" not in result.output


def test_validate_missing_file_is_a_friendly_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate", str(tmp_path / "nope.json")])
    assert result.exit_code == 1 and "error: no file" in result.stderr


@pytest.mark.parametrize("damage", ["truncated", "corrupt", "directory"])
def test_validate_unreadable_input_is_a_friendly_error(tmp_path: Path, damage: str) -> None:
    packed = gzip.compress(json.dumps(toy()).encode())
    path = tmp_path / "bad.json.gz"
    if damage == "truncated":
        path.write_bytes(packed[:5])
    elif damage == "corrupt":
        path.write_bytes(packed[:10] + b"\xff" * 40)
    else:
        path.mkdir()
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 1
    if damage == "directory":
        assert result.stderr == (
            f"error: {path} is a folder\nhint: give a profile file (.json or .json.gz)\n"
        )
    else:
        assert result.stderr.startswith(f"error: {path} is a damaged gzip file (")
        assert result.stderr.endswith(
            ")\nhint: the file is damaged or truncated. Pull or export the profile again\n"
        )
    assert "Traceback" not in result.output and "Aborted" not in result.output


def test_a_file_that_is_not_utf8_names_the_byte_and_the_line(tmp_path: Path) -> None:
    latin = b'{\n"id": "caf\xe9"}'
    profile = tmp_path / "latin.json"
    profile.write_bytes(latin)
    counts = tmp_path / "latin.counts.json"
    counts.write_bytes(latin)
    for args in (["show", str(profile)], ["validate", str(profile)]):
        result = runner.invoke(app, args)
        assert result.exit_code == 1
        assert result.stderr == (
            f"error: {profile} is not UTF-8 text (byte 0xe9 on line 2)\n"
            "hint: the file is damaged or truncated. Pull or export the profile again\n"
        )
    result = runner.invoke(app, ["compare", "ibm_fez@2025-02-26", str(counts)])
    assert result.exit_code == 1
    assert result.stderr == (
        f"error: {counts} is not UTF-8 text (byte 0xe9 on line 2)\n"
        "hint: the file is damaged or truncated. Save the counts again\n"
    )


def test_validate_notes_t2_clamps(tmp_path: Path) -> None:
    path = tmp_path / "t2.json"
    path.write_text(json.dumps(toy(idle={"t1_us": 50, "t2_us": 150})))
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 0 and "T2 exceeds 2*T1" in result.stderr


def test_validate_names_the_qubits_of_a_per_cycle_record_in_words(tmp_path: Path) -> None:
    records = [
        {"gate": "cz", "qubits": [0, 1], "avg_infidelity": 0.02, "scope": "cycle"},
        {"gate": "sx", "qubits": [2], "avg_infidelity": 0.002, "scope": "cycle"},
    ]
    path = tmp_path / "cycle.json"
    path.write_text(json.dumps(toy(calibrations=records)))
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 0, result.output
    assert result.stderr.splitlines() == [
        "warning: cz on qubits 0-1: error is per cycle, not per gate",
        "warning: sx on qubit 2: error is per cycle, not per gate",
    ]


def test_validate_says_what_an_unknown_or_a_missing_key_means(tmp_path: Path) -> None:
    data = toy(unknown_field=1)
    del data["device"]
    path = tmp_path / "keys.json"
    path.write_text(json.dumps(data))
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 1
    assert result.stderr.splitlines() == [
        "error: device: missing. Format 1.0 requires this key",
        "error: unknown_field: not a format 1.0 key. Put your own data under the top-level"
        " extensions key",
    ]


# profile display ------------------------------------------------------------------------------


def test_profile_prints_as_one_line() -> None:
    profile = nv.load("ibm_fez")
    expected = f"<Profile ibm_fez@2025-02-26 superconducting 156q {profile.short_fingerprint}>"
    assert repr(profile) == str(profile) == expected
    undated = Profile.model_validate(toy())
    assert (
        repr(undated)
        == f"<Profile test_toy@undated superconducting 3q {undated.short_fingerprint}>"
    )


# global behavior ------------------------------------------------------------------------------


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and result.stdout == f"noisevault {nv.__version__}\n"


@pytest.mark.parametrize("columns", [80, 120])
@pytest.mark.parametrize(
    "args",
    [
        ["list"],
        ["show", "ibm_fez", "--qubits", "0,1,2"],
        ["show", "google_rainbow"],
        ["diff", "ibm_kyiv", "ibm_brisbane"],
        ["check", "ibm_manila", "--framework", "cirq,stim"],
        ["compare", "ibm_kingston@2026-04-15", "examples/kingston-simulated.counts.json"],
    ],
    ids=["list", "show-qubits", "show-notes", "diff", "check", "compare"],
)
def test_output_fits_the_terminal(args: list[str], columns: int, monkeypatch) -> None:
    if args[0] == "check":
        require("cirq"), require("stim")
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    result = runner.invoke(app, args, env={"COLUMNS": str(columns)})
    assert result.exit_code == 0, result.output
    assert max(len(line) for line in result.stdout.splitlines()) <= columns


@pytest.mark.parametrize("columns", [80, 120])
def test_list_keeps_every_cell_whole_with_a_pulled_profile(monkeypatch, columns: int) -> None:
    _serve_ionq(monkeypatch)
    assert runner.invoke(app, ["pull", "ionq_forte-1"]).exit_code == 0
    result = runner.invoke(app, ["list"], env={"COLUMNS": str(columns)})
    lines = result.stdout.splitlines()
    assert max(map(len, lines)) <= columns
    group = lines.index("trapped ion")
    first = lines[group + 1].split()
    assert first == [
        "*",
        "ionq_forte-1",
        "2026-09-27",
        "4",
        "Forte",
        "public",
        "API",
        "IonQ",
        "EULA",
    ]
    assert lines[group + 2].split()[:3] == ["quantinuum_h1-1", "2025-05-02", "20"]
    assert "* in your vault" in result.stdout


def test_diff_lists_disabled_gates_without_splitting_an_item(tmp_path: Path) -> None:
    data = nv.load("ibm_fez").model_dump(mode="json", exclude_none=True)
    data["device"]["calibrated_at"] = "2026-10-01T00:00:00Z"
    for record in data["calibrations"]:
        if {27, 28, 71, 72, 129, 130, 153, 154} & set(record["qubits"]):
            record["disabled"] = True
    path = tmp_path / "fez_now.json"
    after = Profile.model_validate(data)
    after.save(path)
    expected = set(nv.load("ibm_fez").diff(after).newly_disabled)
    out = runner.invoke(app, ["diff", "ibm_fez", str(path)], env={"COLUMNS": "80"}).stdout
    lines = out.splitlines()
    section = lines[lines.index("newly disabled") + 1 :]
    shown, gate = set(), None
    for line in section:
        if not line.startswith("  "):
            break
        tokens = line.replace(",", " ").split()
        if not re.fullmatch(r"\d+(-\d+)?", tokens[0]):
            gate, tokens = tokens[0], tokens[1:]
        shown |= {f"{gate} {locus}" for locus in tokens}
    assert len(expected) > 20 and shown == expected


def test_a_profile_id_that_names_a_folder_here_still_loads(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ibm_manila").mkdir()
    result = runner.invoke(app, ["show", "ibm_manila"])
    assert result.exit_code == 0 and result.stdout.startswith("ibm_manila@2024-05-27")


_OUTPUTS = [
    ["list"],
    ["show", "ibm_fez", "--qubits", "0,1,2"],
    ["show", "google_weber"],
    ["diff", "ibm_kyiv", "ibm_brisbane"],
    ["doctor"],
    ["validate", str(MANILA_V01)],
]


@pytest.mark.parametrize("color", [{}, {"NO_COLOR": "1"}, {"FORCE_COLOR": "1"}])
@pytest.mark.parametrize("columns", [40, 60, 80, 120])
def test_no_output_line_ends_in_spaces(columns: int, color: dict[str, str]) -> None:
    for args in _OUTPUTS:
        out = runner.invoke(app, args, env={"COLUMNS": str(columns), **color}).stdout
        plain = _unstyled(out)
        padded = [line for line in plain.splitlines() if line != line.rstrip()]
        assert not padded, (args, padded[:3])


@pytest.mark.parametrize(
    "args",
    [
        ["list"],
        ["list", "--vendor", "google", "--tech", "trapped_ion"],
        ["show", "nosuch"],
        ["show", "ibm_fez@2025-03-01"],
        ["show", "ibm_fez", "--top"],
        ["pull", "google_weber"],
        ["pull", "ibm_fez", "--source", "google"],
        ["diff", "ibm_fez"],
    ],
)
def test_output_writes_commands_without_backticks(args: list[str]) -> None:
    result = runner.invoke(app, args, env={"COLUMNS": "200"}, prog_name="nv")
    output = _unstyled(result.output)
    assert "nv " in output and "`" not in output, output


def test_no_color_removes_color_codes() -> None:
    args = ["diff", "ibm_kyiv", "ibm_brisbane"]
    colored = runner.invoke(app, args, env={"FORCE_COLOR": "1"}).stdout
    plain = runner.invoke(app, args, env={"FORCE_COLOR": "1", "NO_COLOR": "1"}).stdout
    color = re.compile(r"\x1b\[[0-9;]*3[0-7]m")
    assert color.search(colored) and not color.search(plain)


@pytest.mark.parametrize(
    ("args", "error", "hint"),
    [
        (["show", "ibm_fezz"], "did you mean 'ibm_fez'?", None),
        (
            ["show", "ibm_fez@2020-01-01"],
            "no ibm_fez profile calibrated on 2020-01-01 UTC."
            " You have ibm_fez@2025-02-26T20:16:25Z",
            "run nv pull ibm_fez --at 2020-01-01T23:59:59Z to download the calibration in effect"
            " at the end of that day. Then load the ref that nv pull prints",
        ),
        (["show", "quantinuum_h1-1@2020-01-01"], "You have quantinuum_h1-1@2025-05-02", None),
        (
            ["show", "nosuch"],
            "no profile with id 'nosuch'",
            "run nv list to see every profile you can load offline",
        ),
        (
            ["show", "missing.json"],
            "no file missing.json",
            "check the path, or give a profile id such as ibm_fez",
        ),
        (["cite", "ibm fez"], "is not a profile id", None),
        (
            ["check", "ibm_manila", "--framework", "qiskt"],
            "'qiskt': did you mean 'qiskit'? give one or more of qiskit",
            None,
        ),
        (["list", "--tech", "photonics"], "choose from superconducting, trapped_ion", None),
        (["show", "ibm_fez", "--qubits", "0,200"], "has qubits 0 to 155", None),
        (["check", "ibm_manila", "--framework", ""], "--framework '': give one or more", None),
        (["check", "ibm_manila", "--framework", ","], "--framework ',': give one or more", None),
        (
            ["show", "./"],
            "./ is a folder",
            "give a profile file (.json or .json.gz) or a profile id such as ibm_fez",
        ),
        (
            ["list", "--vendor", "google", "--tech", "trapped_ion"],
            "no google trapped_ion profile yet",
            "run nv list to see them all",
        ),
    ],
)
def test_expected_failures_print_an_error_and_no_traceback(args, error, hint) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    first, *rest = result.stderr.splitlines()
    assert first.startswith("error: ") and error in first
    assert rest == ([f"hint: {hint}"] if hint else [])
    assert "Traceback" not in result.output and result.stdout == ""


# list and show --------------------------------------------------------------------------------


def test_list_groups_profiles_by_technology() -> None:
    result = runner.invoke(app, ["list"], env={"COLUMNS": "80"})
    lines = [line.rstrip() for line in result.stdout.splitlines()]
    assert lines[0].split() == ["id", "date", "qubits", "processor", "source"]
    row = next(line for line in lines if line.strip().startswith("ibm_fez "))
    assert row.split() == ["ibm_fez", "2025-02-26", "156", "Heron", "r2", "SDK"]
    groups = [line for line in lines[1:-2] if not line.startswith((" ", "*"))]
    assert groups == ["superconducting", "trapped ion"]
    assert lines.index("superconducting") < lines.index(row) < lines.index("trapped ion")
    assert lines[-2:] == [
        f"{len(nv.profiles())} profiles, all Apache-2.0. See one with nv show <id>.",
        'Load one in Python with nv.load("<id>").',
    ]


def test_list_filters_ignore_case_and_dashes() -> None:
    for args in (["--tech", "Trapped-Ion"], ["--vendor", "Quantinuum"]):
        result = runner.invoke(app, ["list", *args, "--json"])
        assert result.exit_code == 0 and {r["vendor"] for r in json.loads(result.stdout)} == {
            "quantinuum"
        }


def test_list_filters_and_prints_json() -> None:
    result = runner.invoke(app, ["list", "--tech", "trapped_ion", "--json"])
    rows = json.loads(result.stdout)
    assert rows and {r["technology"] for r in rows} == {"trapped_ion"}
    assert {"id", "date", "num_qubits", "processor", "source_kind", "license"} <= set(rows[0])
    vendor = json.loads(runner.invoke(app, ["list", "--vendor", "google", "--json"]).stdout)
    assert {r["id"] for r in vendor} == {"google_rainbow", "google_weber"}


@pytest.mark.parametrize("columns", [60, 72, 80, 120])
def test_list_keeps_every_row_on_one_line(columns: int) -> None:
    infos = nv.profiles()
    result = runner.invoke(app, ["list"], env={"COLUMNS": str(columns)})
    lines = result.stdout.splitlines()
    assert max(map(len, lines)) <= columns
    assert len(lines) == 1 + len({i.technology for i in infos}) + len(infos) + 2
    rows = {tuple(line.split()[:3]) for line in lines if line.startswith("  ")}
    assert rows == {(i.id, i.calibrated_at.date().isoformat(), str(i.num_qubits)) for i in infos}
    header = ["id", "date", "qubits", "processor"] + (["source"] if columns >= 72 else [])
    assert lines[0].split() == header


@pytest.mark.parametrize(
    ("columns", "header"),
    [
        (120, ["id", "date", "qubits", "processor", "source", "license"]),
        (80, ["id", "date", "qubits", "processor", "license"]),
        (60, ["id", "date", "qubits"]),
        (30, ["id", "date", "qubits"]),
    ],
)
def test_list_drops_columns_and_never_cuts_an_id_or_a_date(
    vault: Path, columns: int, header: list[str]
) -> None:
    _manila_in_the_vault(
        "2024-06-03T10:00:00Z", license="AWS Customer Agreement (not an open license)"
    )
    out = runner.invoke(app, ["list"], env={"COLUMNS": str(columns)}).stdout
    lines = out.splitlines()
    assert lines[0].split() == header and "\u2026" not in out
    rows = {tuple(line.lstrip("* ").split()[:3]) for line in lines if line[:2] in ("  ", "* ")}
    assert rows - {("in", "your", "vault")} == {
        (i.id, i.calibrated_at.date().isoformat(), str(i.num_qubits)) for i in nv.profiles()
    }
    licenses = (
        f"{len(nv.profiles())} profiles, Apache-2.0 except ibm_manila@2024-06-03"
        " (AWS Customer Agreement)."
    )
    assert (licenses in " ".join(out.split())) == ("license" not in header)
    if columns >= 60:
        assert max(map(len, lines)) <= columns


def test_show_prints_a_card() -> None:
    profile = nv.load("ibm_fez")
    out = runner.invoke(app, ["show", "ibm_fez"]).stdout
    assert out.startswith(f"ibm_fez@2025-02-26T20:16:25Z  {profile.short_fingerprint}\n")
    for text in (
        "IBM, Heron r2, superconducting, 156 qubits",
        "176 edges, 1 to 3 neighbors per qubit",
        "rz (1q)",
        "virtual",
        "176 (7 disabled)",
        "median T1 144.9 us",
        "Apache-2.0, redistribution allowed",
        profile.fingerprint,
    ):
        assert text in out
    cz = next(line for line in out.splitlines() if "cz (2q)" in line)
    assert cz.split()[-8:] == ["cz", "(2q)", "3.82e-03", "84", "ns", "176", "(7", "disabled)"]


def test_show_qubits_and_json() -> None:
    data = json.loads(runner.invoke(app, ["show", "ibm_fez", "--qubits", "0,5", "--json"]).stdout)
    assert [q["qubit"] for q in data["qubits"]] == [0, 5]
    table = nv.load("ibm_fez").table
    assert data["qubits"][1]["t1_us"] == pytest.approx(table.qubit(5).t1_ns / 1000)
    assert data["connectivity"] == {
        "kind": "edges",
        "edges": 176,
        "directed": False,
        "min_degree": 1,
        "max_degree": 3,
    }
    cz = next(n for n in data["natives"] if n["gate"] == "cz")
    assert cz["records"] == 176 and cz["disabled"] == 7
    text = runner.invoke(app, ["show", "ibm_fez", "--qubits", "0,5"]).stdout
    assert re.search(r"^\s+5\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+ \(sx\)", text, re.M)
    assert data["qubits"][1]["gate_1q"] == "sx"


def test_show_qubits_fits_60_columns_and_shows_state_only_when_a_qubit_has_one(
    tmp_path: Path,
) -> None:
    result = runner.invoke(app, ["show", "ibm_fez", "--qubits", "0,1,87"], env={"COLUMNS": "60"})
    table = result.stdout.splitlines()[-5:]
    assert [row.split()[0] for row in table[2:]] == ["0", "1", "87"]
    assert table[1].split()[-1] == "infidelity" and "\u2026" not in result.stdout
    assert max(map(len, table)) <= 60
    data = toy(qubits=[{"index": 1, "disabled": True}])
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(data))
    out = runner.invoke(app, ["show", str(path), "--qubits", "0,1"], env={"COLUMNS": "80"}).stdout
    rows = out.splitlines()[-3:]
    assert rows[0].split()[-1] == "state" and rows[2].split()[-1] == "disabled"


def test_show_qubits_keeps_the_state_column_for_a_label_alone() -> None:
    weber = nv.load("google_weber").table
    assert not any(weber.qubit(q).disabled for q in (0, 1, 2))
    args = ["show", "google_weber", "--qubits", "0,1,2"]
    table = runner.invoke(app, args, env={"COLUMNS": "100"}).stdout.splitlines()[-4:]
    assert [row.split()[-1] for row in table] == ["state", "q(0,5)", "q(0,6)", "q(1,4)"]


@pytest.mark.parametrize("ref", ["ibm_fez", "ibm_kyiv", "ibm_manila"])
def test_show_qubits_names_a_real_gate_not_the_identity(ref: str) -> None:
    text = runner.invoke(app, ["show", ref, "--qubits", "0,1,2,3,4"]).stdout
    rows = text[text.index("1q avg infidelity") :].splitlines()[1:]
    assert len(rows) == 5 and all(row.endswith("(sx)") for row in map(str.rstrip, rows))


def test_show_all_to_all_and_notes() -> None:
    out = runner.invoke(app, ["show", "quantinuum_h1-1"], env={"COLUMNS": "200"}).stdout
    assert "all-to-all" in out and "device-wide" in out
    assert "T1 and T2 unknown" in out and re.search(r"not modeled\s+leakage on r\s", out)


def test_show_and_diff_name_the_error_metric(tmp_path: Path) -> None:
    lines = runner.invoke(app, ["show", "ibm_fez"], env={"COLUMNS": "80"}).stdout.splitlines()
    assert max(map(len, lines)) <= 80
    natives = next(i for i, line in enumerate(lines) if line.startswith("natives"))
    assert [line.split() for line in lines[natives : natives + 2]] == [
        ["natives", "median", "avg", "median"],
        ["gate", "infidelity", "duration", "records"],
    ]
    reset = next(line for line in lines if "reset (1q)" in line)
    assert reset.split() == ["reset", "(1q)", "-", "1.58", "us", "156"]
    gates = {"rz": {"virtual": True}, "sx": {"avg_infidelity": 1e-3}, "cz": {"duration_ns": 70}}
    calibrations = [{"gate": "cz", "qubits": [0, 1], "avg_infidelity": 0.02}]
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(toy(gates=gates, calibrations=calibrations)))
    out = runner.invoke(app, ["show", str(path)], env={"COLUMNS": "120"}).stdout
    cz = next(line for line in out.splitlines() if "cz (2q)" in line)
    assert cz.split()[2:] == ["2.00e-02", "70", "ns", "1", "(1", "uncalibrated)"]
    drift = runner.invoke(app, ["diff", "ibm_kyiv", "ibm_brisbane"], env={"COLUMNS": "80"}).stdout
    medians = [line.split("  ")[0] for line in drift.splitlines()[1:7]]
    assert medians == [
        "device median",
        "T1 (us)",
        "T2 (us)",
        "1q avg infidelity",
        "2q avg infidelity",
        "readout error",
    ]
    assert re.search(r"^\d+-\d+ +2q avg infidelity ", drift, re.M)


@pytest.mark.parametrize("ref", [i.id for i in nv.catalog.bundled_profiles()])
def test_show_keeps_each_native_on_one_line_down_to_66_columns(ref: str) -> None:
    lines = runner.invoke(app, ["show", ref], env={"COLUMNS": "66"}).stdout.splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith("natives"))
    last = next(i for i, line in enumerate(lines) if line.startswith("coherence"))
    assert last - first == 2 + len(nv.load(ref).gates)
    assert max(len(line) for line in lines[first:last]) <= 66


def test_show_says_where_the_data_came_from_in_plain_words(tmp_path: Path) -> None:
    def provenance(ref: str) -> str:
        out = runner.invoke(app, ["show", ref], env={"COLUMNS": "200"}).stdout
        return next(line for line in out.splitlines() if line.startswith("provenance")).rstrip()

    assert provenance("ibm_fez") == (
        "provenance    measured, SDK snapshot, qiskit-ibm-runtime 0.49.0 FakeFez"
    )
    assert provenance("quantinuum_h1-1") == (
        "provenance    measured, published data, Quantinuum hardware specifications,"
        " data/H1-1/2025_05_02 at 59e68bb55bd6"
    )
    uniform = Profile.uniform(
        "u", technology="trapped_ion", num_qubits=2, one_qubit_error=1e-4, two_qubit_error=1e-3
    ).save(tmp_path / "u.json")
    assert provenance(str(uniform)) == "provenance    hypothetical, written by hand"


def test_show_lists_what_the_model_leaves_out_before_provenance_and_notes() -> None:
    out = runner.invoke(app, ["show", "google_weber"], env={"COLUMNS": "120"}).stdout
    labels = [line.split("  ")[0] for line in out.splitlines()[1:] if line[:1].isalpha()]
    assert labels == [
        "device",
        "connectivity",
        "natives",
        "coherence",
        "readout",
        "not modeled",
        "provenance",
        "license",
        "attribution",
        "fingerprint",
        "assumptions",
        "notes",
    ]


def test_show_writes_schema_tokens_as_words_and_json_keeps_them(tmp_path: Path) -> None:
    def rows(ref: str) -> dict[str, str]:
        out = runner.invoke(app, ["show", ref], env={"COLUMNS": "200"}).stdout
        return {
            line.split("  ")[0]: line.split("  ", 1)[1].strip()
            for line in out.splitlines()[1:]
            if line[:1].isalpha()
        }

    h1 = rows("quantinuum_h1-1")
    assert h1["device"] == "Quantinuum, System Model H1, trapped ion, 20 qubits"
    assert h1["license"] == "Apache-2.0, redistribution allowed"
    weber = rows("google_weber")
    assert weber["device"] == "Google, Sycamore, superconducting, 53 qubits"
    assert weber["not modeled"] == "coherent over-rotation on sqrt_iswap (86 records)"
    data = toy(
        provenance={
            "data_kind": "vendor_model",
            "source_kind": "public_api",
            "source": "IonQ API",
            "license": "IonQ EULA",
            "redistributable": "no",
        }
    )
    data["device"]["vendor"] = "ionq"
    path = tmp_path / "ionq.json"
    path.write_text(json.dumps(data))
    pulled = rows(str(path))
    assert pulled["device"] == "IonQ, superconducting, 3 qubits"
    assert pulled["provenance"] == "vendor model, public API, IonQ API"
    assert pulled["license"] == "IonQ EULA, redistribution not allowed"
    del data["provenance"]["license"], data["provenance"]["redistributable"]
    path.write_text(json.dumps(data))
    assert rows(str(path))["license"] == "unknown, redistribution unknown"
    shown = json.loads(runner.invoke(app, ["show", "google_weber", "--json"]).stdout)
    assert (shown["vendor"], shown["technology"]) == ("google", "superconducting")
    assert shown["effects"][0] == "coherent_overrotation on sqrt_iswap (86 records)"
    trapped = json.loads(runner.invoke(app, ["show", "quantinuum_h1-1", "--json"]).stdout)
    assert trapped["technology"] == "trapped_ion"
    listed = runner.invoke(app, ["list", "--tech", "trapped_ion", "--json"]).stdout
    assert {r["technology"] for r in json.loads(listed)} == {"trapped_ion"}


def test_show_lists_natives_one_qubit_then_two_qubit_then_reset(tmp_path: Path) -> None:
    natives = json.loads(runner.invoke(app, ["show", "ibm_fez", "--json"]).stdout)["natives"]
    assert [n["gate"] for n in natives] == ["id", "rz", "sx", "x", "cz", "reset"]
    gates = {
        "reset": {"duration_ns": 1500},
        "rzz": {"avg_infidelity": 3e-3},
        "x": {"avg_infidelity": 2e-4},
        "cz": {"avg_infidelity": 3e-3},
        "rz": {"virtual": True},
        "sx": {"avg_infidelity": 2e-4},
        "id": {"avg_infidelity": 2e-4},
        "rx": {"avg_infidelity": 2e-4},
    }
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(toy(gates=gates)))
    out = runner.invoke(app, ["show", str(path)], env={"COLUMNS": "80"}).stdout
    shown = [line.split()[0] for line in out.splitlines() if "(1q)" in line or "(2q)" in line]
    assert shown == ["id", "rx", "rz", "sx", "x", "cz", "rzz", "reset"]


def test_show_states_the_calibration_and_adds_a_row_for_unmodeled_error(tmp_path: Path) -> None:
    kingston = nv.load("ibm_kingston@2026-04-15")

    def fitted(p_value: float) -> tuple[Profile, str]:
        fit = {
            "counts": "sha256:3fa1c2d4e5b6" + "0" * 52,
            "source": "hardware",
            "qubits": [148, 149, 150, 151],
            "run_at": "2026-04-16T09:30:02Z",
            "calibration": kingston.fingerprint,
            "p_value": p_value,
            "impossible_shots": 0,
        }
        block = {
            "gates": {"factor": 1.84, "low": 1.54, "high": 2.12},
            "readout": {"factor": 1.58, "low": 1.32, "high": 1.84},
            "fit": fit,
        }
        profile = kingston.model_copy(update={"unmodeled_error": block})
        return profile, str(profile.save(tmp_path / f"fitted-{p_value}.json"))

    def show(*args: str) -> Result:
        return runner.invoke(app, ["show", *args], env={"COLUMNS": "80"})

    profile, path = fitted(0.41)
    result = show(path)
    lines = _unstyled(result.stdout).splitlines()
    row = lines.index("unmodeled     gate errors x1.84 (95% interval 1.54 to 2.12)")
    assert lines[row - 1].startswith("readout       median error")
    assert lines[row : row + 6] == [
        "unmodeled     gate errors x1.84 (95% interval 1.54 to 2.12)",
        "              readout errors x1.58 (95% interval 1.32 to 1.84)",
        "              fitted to hardware counts sha256:3fa1c2d4e5b6",
        "              on qubits 148-149-150-151, run 2026-04-16 (p = 0.41)",
        "              T1, T2 and preparation error are not scaled",
        "              readout of qubit 146 is not scaled (no better than chance)",
    ]
    stated = (
        _unstyled(show("ibm_kingston@2026-04-15").stdout)
        .replace(kingston.fingerprint, profile.fingerprint)
        .replace(kingston.short_fingerprint, profile.short_fingerprint)
    )
    assert lines[:row] + lines[row + 6 :] == stated.splitlines()
    assert result.stderr == ""
    qubits = ["--qubits", "146,149"]
    assert (
        show(path, *qubits).stdout.splitlines()[-3:]
        == (show("ibm_kingston@2026-04-15", *qubits).stdout.splitlines()[-3:])
    )
    data = json.loads(show(path, "--json").stdout)
    assert data["unmodeled_error"] == profile.unmodeled_error.model_dump(mode="json")
    assert data["unmodeled_note"] == [line[14:] for line in lines[row : row + 6]]
    plain = json.loads(show("ibm_kingston@2026-04-15", "--json").stdout)
    assert (plain["unmodeled_error"], plain["unmodeled_note"]) == (None, [])
    assert show(fitted(0.01)[1]).stderr == ""
    poor = show(fitted(0.003)[1])
    assert "on qubits 148-149-150-151, run 2026-04-16 (p = 0.003, a poor fit)" in poor.stdout
    assert poor.stderr == (
        "warning: the unmodeled-error factors are a poor fit to their counts (p = 0.003)."
        " No one pair of factors fits every circuit\n"
    )


def test_show_shortens_a_long_qubit_list_that_its_json_keeps_whole(tmp_path: Path) -> None:
    device = toy()["device"] | {"num_qubits": 6}
    path = tmp_path / "chance.json"
    unmodeled = {"readout": {"factor": 1.3}}
    path.write_text(
        json.dumps(toy(device=device, readout={"error": 0.5}, unmodeled_error=unmodeled))
    )
    out = _unstyled(runner.invoke(app, ["show", str(path)], env={"COLUMNS": "100"}).stdout)
    assert "readout of qubits 0, 1, 2 and 3 more is not scaled (no better than chance)" in out
    data = json.loads(runner.invoke(app, ["show", str(path), "--json"]).stdout)
    assert data["unmodeled_note"][-1] == (
        "readout of qubits 0, 1, 2, 3, 4 and 5 is not scaled (no better than chance)"
    )


def test_show_qubits_prints_microseconds_with_one_decimal() -> None:
    out = runner.invoke(app, ["show", "ibm_fez", "--qubits", "0,5"], env={"COLUMNS": "80"}).stdout
    rows = out.splitlines()[-2:]
    assert [row.split()[:3] for row in rows] == [["0", "48.8", "42.4"], ["5", "190.0", "208.7"]]
    assert "coherence     median T1 144.9 us, median T2 88.0 us" in out


@pytest.mark.parametrize(
    ("idle", "coherence"),
    [
        ({"t1_us": 300, "t2_us": 87.95}, "median T1 300.0 us, median T2 88.0 us"),
        ({"t1_us": 1e8, "t2_us": 1e6}, "median T1 100000000.0 us, median T2 1000000.0 us"),
        ({"t1_us": 50}, "median T1 50.0 us, T2 unknown"),
        ({"t2_us": 40}, "T1 unknown, median T2 40.0 us"),
    ],
    ids=["superconducting", "trapped_ion", "no_t2", "no_t1"],
)
def test_show_prints_coherence_in_microseconds_with_one_decimal(
    tmp_path: Path, idle: dict[str, float], coherence: str
) -> None:
    path = tmp_path / "idle.json"
    path.write_text(json.dumps(toy(idle=idle)))
    out = runner.invoke(app, ["show", str(path)], env={"COLUMNS": "80"}).stdout
    assert next(line for line in out.splitlines() if line.startswith("coherence")) == (
        f"coherence     {coherence}"
    )


def _manila_in_the_vault(calibrated_at: str, **provenance: str) -> Profile:
    data = nv.load("ibm_manila").model_dump(mode="json", exclude_none=True)
    data["device"]["calibrated_at"] = calibrated_at
    data["provenance"].update(provenance)
    data["qubits"][0]["t1_us"] *= 1.1
    profile = Profile.model_validate(data)
    profile.save(nv.catalog.vault_path(profile))
    return profile


def test_show_says_which_calibration_a_bare_id_picked(vault: Path) -> None:
    bundled = nv.load("ibm_manila")
    assert "calibrations" not in runner.invoke(app, ["show", "ibm_manila"]).stdout
    newer = _manila_in_the_vault("2024-06-03T10:00:00Z")
    lines = runner.invoke(app, ["show", "ibm_manila"], env={"COLUMNS": "80"}).stdout.splitlines()
    assert lines[0] == f"ibm_manila@2024-06-03T10:00:00Z  {newer.short_fingerprint}"
    assert [line.rstrip() for line in lines[2:4]] == [
        "calibrations  newest of the 2 you have",
        "              the bundled one is ibm_manila@2024-05-27T18:27:23Z",
    ]
    assert nv.load("ibm_manila@2024-05-27T18:27:23Z").fingerprint == bundled.fingerprint
    for ref in ("ibm_manila@2024-05-27", "ibm_manila@2024-06-03T10:00:00Z"):
        assert "calibrations" not in runner.invoke(app, ["show", ref]).stdout


def test_show_counts_the_calibrations_when_the_bundled_one_is_the_newest(vault: Path) -> None:
    _manila_in_the_vault("2024-01-02T10:00:00Z")
    lines = runner.invoke(app, ["show", "ibm_manila"], env={"COLUMNS": "80"}).stdout.splitlines()
    assert lines[0].startswith("ibm_manila@2024-05-27T18:27:23Z")
    assert lines[2].rstrip() == "calibrations  newest of the 2 you have"
    assert lines[3].startswith("connectivity")


def test_show_says_when_a_vault_copy_replaces_the_bundled_calibration(vault: Path) -> None:
    bundled = nv.load("ibm_manila")
    mine = _manila_in_the_vault("2024-05-27T18:27:23Z")
    lines = runner.invoke(app, ["show", "ibm_manila"], env={"COLUMNS": "80"}).stdout.splitlines()
    assert lines[0] == f"ibm_manila@2024-05-27T18:27:23Z  {mine.short_fingerprint}"
    assert lines[2:4] == [
        "calibrations  newest of the 2 you have",
        f"              your vault copy replaces the bundled one ({bundled.short_fingerprint})",
    ]


def test_show_counts_the_calibrations_when_the_id_names_a_folder_here(
    vault: Path, tmp_path: Path, monkeypatch
) -> None:
    _manila_in_the_vault("2024-06-03T10:00:00Z")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ibm_manila").mkdir()
    lines = runner.invoke(app, ["show", "ibm_manila"], env={"COLUMNS": "80"}).stdout.splitlines()
    assert lines[2] == "calibrations  newest of the 2 you have"


def test_list_names_an_unknown_license_once_when_no_profile_states_one(vault: Path) -> None:
    for when in ("2024-06-03T10:00:00Z", "2024-07-01T10:00:00Z"):
        data = nv.load("ibm_manila").model_dump(mode="json", exclude_none=True)
        data["device"].update(calibrated_at=when, vendor="acme")
        data["provenance"].pop("license")
        profile = Profile.model_validate(data)
        profile.save(nv.catalog.vault_path(profile))
    out = runner.invoke(app, ["list", "--vendor", "acme"], env={"COLUMNS": "30"}).stdout
    assert "2 profiles, license unknown. See one with nv show <id>." in " ".join(out.split())


def test_list_says_unknown_where_a_profile_states_no_license(vault: Path) -> None:
    data = nv.load("ibm_manila").model_dump(mode="json", exclude_none=True)
    data["device"]["calibrated_at"] = "2024-06-03T10:00:00Z"
    data["provenance"].pop("license")
    data["provenance"]["redistributable"] = "unknown"
    profile = Profile.model_validate(data)
    profile.save(nv.catalog.vault_path(profile))
    lines = runner.invoke(app, ["list"], env={"COLUMNS": "120"}).stdout.splitlines()
    row = next(line for line in lines if line.startswith("* ibm_manila "))
    assert row.split() == [
        "*",
        "ibm_manila",
        "2024-06-03",
        "5",
        "Falcon",
        "r5.11",
        "SDK",
        "unknown",
    ]
    card = runner.invoke(app, ["show", "ibm_manila"], env={"COLUMNS": "80"}).stdout
    assert re.search(r"^license +unknown, redistribution unknown$", card, re.M)


def test_list_keeps_one_ids_calibrations_together_newest_first(vault: Path) -> None:
    _manila_in_the_vault("2024-06-03T10:00:00Z")
    lines = runner.invoke(app, ["list"], env={"COLUMNS": "80"}).stdout.splitlines()
    at = [i for i, line in enumerate(lines) if " ibm_manila " in f" {line}"]
    assert [lines[i].split()[:3] for i in at] == [
        ["*", "ibm_manila", "2024-06-03"],
        ["ibm_manila", "2024-05-27", "5"],
    ]
    assert at[1] == at[0] + 1 and lines[at[0] - 1].split()[0] == "ibm_kyiv"


def test_list_and_show_call_a_user_file_imported(vault: Path) -> None:
    _manila_in_the_vault("2024-06-03T10:00:00Z", source_kind="user_file")
    lines = runner.invoke(app, ["list"], env={"COLUMNS": "120"}).stdout.splitlines()
    row = next(line for line in lines if line.startswith("* ibm_manila "))
    assert row.split() == ["*", "ibm_manila", "2024-06-03", "5", "Falcon", "r5.11", "imported"]
    card = runner.invoke(app, ["show", "ibm_manila"], env={"COLUMNS": "200"}).stdout
    assert re.search(r"^provenance +measured, imported file, ", card, re.M), card


def test_check_says_which_calibration_a_bare_id_picked(vault: Path) -> None:
    require("cirq")
    newer = _manila_in_the_vault("2024-06-03T10:00:00Z")
    out = runner.invoke(app, ["check", "ibm_manila", "--framework", "cirq"]).stdout
    newest = f"ibm_manila@2024-06-03T10:00:00Z (newest of 2) {newer.short_fingerprint} on qubits "
    assert out.startswith(newest)
    dated = runner.invoke(app, ["check", "ibm_manila@2024-06-03", "--framework", "cirq"]).stdout
    assert dated.startswith(f"ibm_manila@2024-06-03T10:00:00Z {newer.short_fingerprint} on qubits ")


# pull ---------------------------------------------------------------------------------------


def test_pull_saves_to_the_vault_and_prints_the_card(monkeypatch, vault: Path) -> None:
    from noisevault.sources import ibm_public

    pulled = nv.load("ibm_fez")
    monkeypatch.setattr(ibm_public, "pull", lambda device, at=None: pulled)
    result = runner.invoke(app, ["pull", "ibm_fez"])
    assert result.exit_code == 0, result.output
    (saved,) = vault.glob("*.json.gz")
    assert f"saved: {saved}" in result.stdout and "ibm_fez@2025-02-26" in result.stdout
    assert nv.load(str(saved)).fingerprint == pulled.fingerprint
    assert result.stdout.splitlines()[-1] == f"saved: {saved}"


def test_pull_of_a_calibration_already_in_the_vault_says_nothing_was_written(
    monkeypatch, vault: Path
) -> None:
    from noisevault.sources import ibm_public

    monkeypatch.setattr(ibm_public, "pull", lambda device, at=None: nv.load("ibm_fez"))
    runner.invoke(app, ["pull", "ibm_fez"])
    (saved,) = vault.glob("*.json.gz")
    written = saved.stat().st_mtime_ns
    again = runner.invoke(app, ["pull", "ibm_fez"])
    assert again.exit_code == 0, again.output
    assert again.stdout.splitlines()[-1] == f"already saved: {saved}"
    assert saved.stat().st_mtime_ns == written


def _serve_ionq(monkeypatch: pytest.MonkeyPatch) -> None:
    """IonQ's public endpoint, answered from the recorded responses in the test fixture."""
    from noisevault.sources import ionq

    path = Path(__file__).parent / "fixtures" / "ionq" / "responses.json"
    bodies = {url: json.dumps(body).encode() for url, body in json.loads(path.read_text()).items()}
    monkeypatch.setattr(ionq, "_get", bodies.__getitem__)


def test_pull_at_works_for_every_source_its_help_names(monkeypatch) -> None:
    from typer.main import get_command

    (at,) = [p for p in get_command(app).commands["pull"].params if p.name == "at"]
    assert all(source in at.help for source in ("IBM public", "IBM account", "IonQ"))
    _serve_ionq(monkeypatch)
    result = runner.invoke(app, ["pull", "ionq_forte-1", "--at", "2026-09-01"])
    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("ionq_forte-1@2026-09-01")  # without --at: 2026-09-27


def test_pull_to_a_file(monkeypatch, tmp_path: Path) -> None:
    from noisevault.sources import ionq

    monkeypatch.setattr(ionq, "pull", lambda device, at=None: nv.load("quantinuum_h1-1"))
    target = tmp_path / "forte.json"
    result = runner.invoke(app, ["pull", "ionq_forte-1", "-o", str(target)])
    assert result.exit_code == 0 and target.exists() and f"saved: {target}" in result.stdout


def test_pull_network_failure_gives_one_piece_of_advice(monkeypatch) -> None:
    import urllib.error
    import urllib.request

    def offline(*args, **kwargs):
        raise urllib.error.URLError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    result = runner.invoke(app, ["pull", "ibm_fez", "--at", "2025-01-01"])
    assert result.exit_code == 1
    assert result.stderr == (
        "error: could not reach IBM's public endpoint (timed out)\n"
        "hint: check the network connection, or run nv list to see every profile you can load"
        " offline\n"
    )
    assert "Traceback" not in result.output


_IBM_PROPERTIES = "https://quantum.cloud.ibm.com/api/v1/public/backends/ibm_fez/properties"
_IBM_AGAIN = "try again later, or pull through your IBM account with --source ibm-account"


@pytest.mark.parametrize(
    ("device", "body", "error", "hint"),
    [
        (
            "ibm_fez",
            b"<!DOCTYPE html><html><body>502 Bad Gateway</body></html>",
            f"IBM's public endpoint answered {_IBM_PROPERTIES} with something other than JSON",
            _IBM_AGAIN,
        ),
        (
            "ibm_fez",
            b"[]",
            f"IBM's public endpoint answered {_IBM_PROPERTIES} with JSON of the wrong shape",
            _IBM_AGAIN,
        ),
        (
            "ionq_forte-1",
            b'"maintenance"',
            "IonQ's API answered https://api.ionq.co/v0.4/backends with JSON of the wrong shape",
            "try again later",
        ),
    ],
    ids=["ibm-html", "ibm-list", "ionq-string"],
)
def test_pull_of_a_reply_that_is_not_calibration_data_names_it_and_says_to_try_again(
    monkeypatch, device: str, body: bytes, error: str, hint: str
) -> None:
    from noisevault.sources import ibm_public, ionq

    monkeypatch.setattr(ibm_public, "fetch", lambda url: body)
    monkeypatch.setattr(ionq, "_get", lambda url: body)
    result = runner.invoke(app, ["pull", device])
    assert (result.exit_code, result.stdout) == (1, "")
    assert result.stderr == f"error: {error}\nhint: {hint}\n"


_FIXTURES = Path(__file__).parent / "fixtures"
_IONQ_LISTING = "https://api.ionq.co/v0.4/backends"
_IONQ_CHOSEN = (
    "https://api.ionq.co/v0.4/backends/qpu.forte-1/characterizations?limit=1"
    "&end=2026-09-27T00%3A00%3A00Z"
)
_OLDER = "pass an earlier --at to use an older calibration"


def _ibm_qubit_0(name: str, **change: Any) -> dict[str, Any]:
    props = json.loads((_FIXTURES / "ibm" / "manila_properties.json").read_bytes())
    next(p for p in props["qubits"][0] if p["name"] == name).update(change)
    return {_IBM_PROPERTIES: props}


def _ionq_without_qubit_counts() -> dict[str, Any]:
    replies = json.loads((_FIXTURES / "ionq" / "responses.json").read_text())
    del replies[_IONQ_LISTING][0]["qubits"]
    del replies[_IONQ_CHOSEN]["characterizations"][0]["qubits"]
    return replies


@pytest.mark.parametrize(
    ("device", "replies", "error", "hint"),
    [
        (
            "ibm_fez",
            _ibm_qubit_0("T1", unit="min"),
            f"IBM's public endpoint ({_IBM_PROPERTIES}): T1 of qubit 0 has the unknown time unit"
            " 'min', not one of ns, us, µs, ms or s",
            _OLDER,
        ),
        (
            "ibm_fez",
            _ibm_qubit_0("prob_meas0_prep1", value=1.5),
            f"IBM's public endpoint ({_IBM_PROPERTIES}): readout.p0_given_1 of qubit 0: Input"
            " should be less than or equal to 1, got 1.5",
            _OLDER,
        ),
        (
            "ionq_forte-1",
            _ionq_without_qubit_counts(),
            f"IonQ's API ({_IONQ_CHOSEN}): record 00000000-0000-4000-8000-000000000003 gives no"
            " qubit count, and neither does the backend listing",
            _OLDER,
        ),
        (
            "ionq_forte-1",
            {_IONQ_LISTING: []},
            f"IonQ has no backend 'qpu.forte-1'. The IonQ listing ({_IONQ_LISTING}) names no QPU",
            "try again later",
        ),
    ],
    ids=["ibm-unit", "ibm-range", "ionq-no-qubit-count", "ionq-no-qpu"],
)
def test_pull_of_vendor_data_it_cannot_use_names_the_value_and_a_step_that_works(
    monkeypatch, device: str, replies: dict[str, Any], error: str, hint: str
) -> None:
    from noisevault.sources import ibm_public, ionq

    def reply(url: str) -> bytes:
        if url not in replies:
            raise ibm_public._NotFound(url)
        return json.dumps(replies[url]).encode()

    monkeypatch.setattr(ibm_public, "fetch", reply)
    monkeypatch.setattr(ionq, "_get", reply)
    result = runner.invoke(app, ["pull", device])
    assert (result.exit_code, result.stdout) == (1, "")
    assert _unstyled(result.stderr) == f"error: {error}\nhint: {hint}\n"


def test_pull_errors_name_flags_not_python_arguments(monkeypatch) -> None:
    from noisevault.sources import ibm_account, ibm_public

    def no_account(device, at=None):
        raise SourceUnavailable(
            "could not open your IBM Quantum account (no token)", hint=ibm_account.SETUP
        )

    def not_listed(url):
        raise ibm_public._NotFound(url)

    monkeypatch.setattr(ibm_account, "pull", no_account)
    result = runner.invoke(app, ["pull", "ibm_fez", "--source", "ibm-account"])
    assert result.exit_code == 1
    assert result.stderr.endswith(
        "\nhint: set IBM_QUANTUM_TOKEN to your IBM Quantum API key, or pull without an account"
        " with --source ibm\n"
    )
    assert "source=" not in result.stderr
    monkeypatch.setattr(ibm_public, "fetch", not_listed)
    result = runner.invoke(app, ["pull", "ibm_fez"])
    assert result.stderr.endswith(
        "\nhint: ibm_fez is bundled, so nv show ibm_fez loads it offline\n"
    )


def test_pull_checks_the_output_folder_before_fetching(monkeypatch, tmp_path: Path) -> None:
    from noisevault.sources import ibm_public

    calls = []
    monkeypatch.setattr(ibm_public, "pull", lambda device, at=None: calls.append(device))
    result = runner.invoke(app, ["pull", "ibm_fez", "-o", str(tmp_path / "no" / "x.json")])
    assert result.exit_code == 1 and calls == []
    assert f"the folder {tmp_path / 'no'} does not exist" in result.stderr
    result = runner.invoke(app, ["pull", "ibm_fez", "-o", str(tmp_path)])
    assert result.exit_code == 1 and calls == []
    assert result.stderr == (
        f"error: -o {tmp_path} is a folder\nhint: give a file name such as {tmp_path / 'x.json'}\n"
    )


def test_pull_of_an_unsupported_vendor_says_what_pull_takes() -> None:
    result = runner.invoke(app, ["pull", "rigetti_ankaa-3"])
    assert result.exit_code == 1
    assert result.stderr == (
        "error: no source pulls 'rigetti_ankaa-3'\n"
        "hint: nv pull takes IBM devices (ibm_fez) and IonQ devices (ionq_forte-1), and nv list"
        " shows every profile you can load offline\n"
    )


# diff, check, cite, schema, doctor --------------------------------------------------------------


def test_diff_prints_tables_and_json() -> None:
    result = runner.invoke(app, ["diff", "ibm_kyiv", "ibm_brisbane", "--top", "2"])
    assert result.exit_code == 0
    out = result.stdout
    assert out.startswith(
        "ibm_kyiv@2025-02-26T16:02Z -> ibm_brisbane@2025-02-26T19:33Z  (3 hours later)\n"
    )
    assert "largest changes by qubit" in out and "largest changes by pair" in out
    assert "warning: the two profiles are from different devices" in result.stderr
    data = json.loads(runner.invoke(app, ["diff", "ibm_kyiv", "ibm_brisbane", "--json"]).stdout)
    assert data == nv.load("ibm_kyiv").diff(nv.load("ibm_brisbane")).to_dict()


def test_every_device_median_leaves_out_a_disabled_qubit(tmp_path: Path) -> None:
    def device(disabled_t1_us: float) -> Profile:
        qubits = [
            {"index": 0, "t1_us": 300, "readout": {"error": 0.012}},
            {"index": 1, "t1_us": 280, "readout": {"error": 0.01}},
            {"index": 2, "t1_us": 260, "readout": {"error": 0.014}},
            {"index": 3, "t1_us": disabled_t1_us, "readout": {"error": 0.2}, "disabled": True},
        ]
        data = toy(qubits=qubits)
        data["device"]["num_qubits"] = 4
        return Profile.from_dict(data)

    profile = device(240)
    before, after = profile.save(tmp_path / "a.json"), device(1000).save(tmp_path / "b.json")
    shown = json.loads(runner.invoke(app, ["show", str(before), "--json"]).stdout)
    assert (shown["median_t1_us"], shown["median_readout_error"]) == (280, 0.012)
    summary = profile.summary().splitlines()
    assert summary[-2:] == ["  median T1 280.0 us", "  median readout error 0.012"]
    drift = json.loads(runner.invoke(app, ["diff", str(before), str(after), "--json"]).stdout)
    medians = {m["metric"]: (m["before"], m["after"]) for m in drift["medians"]}
    assert medians["t1_us"] == (280, 280) and medians["readout_error"] == (0.012, 0.012)


def test_diff_of_identical_profiles() -> None:
    result = runner.invoke(app, ["diff", "ibm_fez", "ibm_fez@2025-02-26"])
    assert result.exit_code == 0 and "No change" in result.stdout


def test_diff_of_two_devices_on_one_day_shows_the_times_and_only_the_device_warning() -> None:
    title = "ibm_fez@2025-02-26T20:16Z -> ibm_marrakesh@2025-02-26T19:52Z"
    wide = runner.invoke(app, ["diff", "ibm_fez", "ibm_marrakesh"], env={"COLUMNS": "100"})
    assert wide.stdout.splitlines()[0] == f"{title}  (23 minutes earlier)"
    narrow = runner.invoke(app, ["diff", "ibm_fez", "ibm_marrakesh"], env={"COLUMNS": "80"})
    assert narrow.stdout.splitlines()[:2] == [title, "  (23 minutes earlier)"]
    for result in (wide, narrow):
        assert result.stderr == (
            "warning: the two profiles are from different devices (ibm_fez and ibm_marrakesh), so"
            " the diff matches qubits and pairs by index\n"
        )


def test_check_prints_a_table_per_framework() -> None:
    require("cirq"), require("pennylane")
    result = runner.invoke(app, ["check", "ibm_manila", "--framework", "cirq,pennylane"])
    assert result.exit_code == 0, result.output
    rows = {line.split()[0]: line.split() for line in result.stdout.splitlines() if line}
    assert rows["cirq"][1] == "pass" and rows["pennylane"][1] == "pass"
    assert "qiskit" not in rows
    assert "A pass does not measure how well the model matches the hardware" in " ".join(
        result.stdout.split()
    )


def test_check_failure_exits_1(monkeypatch) -> None:
    require("cirq")
    from noisevault.frameworks import cirq as nv_cirq

    monkeypatch.setattr(nv_cirq.NoiseVaultNoiseModel, "_noise", lambda self, *a: [])
    result = runner.invoke(app, ["check", "ibm_manila", "--framework", "cirq"])
    assert result.exit_code == 1 and "FAIL" in result.stdout
    assert "A pass means" not in result.stdout


def test_check_json_and_skips() -> None:
    require("cirq")
    result = runner.invoke(app, ["check", "quantinuum_h1-1", "--framework", "stim,cirq", "--json"])
    data = json.loads(result.stdout)
    assert [f["framework"] for f in data["frameworks"]] == ["cirq"]
    assert data["skipped"][0]["framework"] == "stim"
    assert result.exit_code == 0


def test_check_counts_a_reduced_circuit_apart_and_names_its_missing_gates(tmp_path) -> None:
    require("stim")
    errors = {"h": 1e-3, "ms": 0.02, "rxx": 0.01, "ryy": 0.01, "zz": 0.01, "rzz": 0.01}
    natives = {name: {"avg_infidelity": error} for name, error in errors.items()}
    path = tmp_path / "ions.json"
    path.write_text(json.dumps(toy(gates=natives, readout={"error": 0.01})))
    result = runner.invoke(app, ["check", str(path), "--framework", "stim"])
    assert result.exit_code == 0, result.output
    row = next(line for line in result.stdout.splitlines() if line.startswith("stim "))
    assert "5 of 6, 1 reduced" in row
    assert (
        "stim: two_qubit_natives ran without rxx, ryy, rzz: Stim simulates only Clifford gates,"
        " and this profile's rxx gate is not Clifford" in " ".join(result.stdout.split())
    )
    data = json.loads(
        runner.invoke(app, ["check", str(path), "--framework", "stim", "--json"]).stdout
    )
    (entry,) = data["frameworks"][0]["not_run"]
    assert entry["ran_without"] == ["rxx", "ryy", "rzz"]


def _uvx(extra: str, command: str) -> str:
    return (
        f'Or, with uv and no install: uvx --from "noisevault[{extra}] @ git+{REPOSITORY}" {command}'
    )


@pytest.mark.parametrize(
    ("args", "missing", "uvx"),
    [
        (
            ["ibm_manila"],
            "cirq,pennylane,stim",
            _uvx("qiskit,cirq,pennylane,stim", "nv check ibm_manila"),
        ),
        (
            ["ibm_manila", "--framework", "stim,qiskit"],
            "stim",
            _uvx("stim,qiskit", "nv check --framework stim,qiskit ibm_manila"),
        ),
    ],
    ids=["every_framework", "named_frameworks"],
)
def test_check_without_some_frameworks_gives_a_pip_and_a_whole_uvx_command(
    monkeypatch, args: list[str], missing: str, uvx: str
) -> None:
    require("qiskit")
    for module in ("cirq", "pennylane", "stim"):
        monkeypatch.setitem(sys.modules, module, None)
    result = runner.invoke(app, ["check", *args], env={"COLUMNS": "80"})
    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    pip = lines.index(f"To add the missing frameworks: {install_hint(missing)}")
    assert lines[pip + 1] == uvx
    assert result.stdout.count("pip install") == result.stdout.count("uvx") == 1
    rows = {line.split()[0]: line for line in lines if line}
    assert rows["stim"].split()[1:] == ["not", "installed"]


@pytest.mark.parametrize(
    ("name", "given", "printed"),
    [
        ("my profiles/manila.json", ["my profiles/manila.json"], "'my profiles/manila.json'"),
        ("-manila.json", ["--", "-manila.json"], "-- -manila.json"),
    ],
    ids=["a space", "a leading hyphen"],
)
def test_the_uvx_check_command_runs_as_printed(
    monkeypatch, tmp_path: Path, name: str, given: list[str], printed: str
) -> None:
    require("qiskit")
    monkeypatch.setitem(sys.modules, "stim", None)
    monkeypatch.chdir(tmp_path)
    path = Path(name)
    path.parent.mkdir(exist_ok=True)
    nv.load("ibm_manila").save(path)
    args = ["check", "--framework", "qiskit,stim", *given]
    result = runner.invoke(app, args, env={"COLUMNS": "80"})
    assert result.exit_code == 0, result.output
    uvx = _uvx("qiskit,stim", f"nv check --framework qiskit,stim {printed}")
    assert uvx in result.stdout.splitlines()
    words = shlex.split(uvx.partition(": ")[2])
    assert words[:4] == ["uvx", "--from", f"noisevault[qiskit,stim] @ git+{REPOSITORY}", "nv"]
    followed = runner.invoke(app, words[4:], env={"COLUMNS": "80"})
    assert followed.exit_code == 0, followed.output
    assert followed.stdout.splitlines()[0] == result.stdout.splitlines()[0]


def test_check_names_a_one_qubit_layout_in_the_singular(tmp_path: Path) -> None:
    require("stim")
    one_qubit = {"rz": {"virtual": True}, "sx": {"avg_infidelity": 1e-3, "duration_ns": 35}}
    data = toy(gates=one_qubit, connectivity={"edges": []})
    data["device"]["num_qubits"] = 1
    path = tmp_path / "one.json"
    path.write_text(json.dumps(data))
    result = runner.invoke(app, ["check", str(path), "--framework", "stim"])
    assert result.exit_code == 0, result.output
    short = Profile.from_dict(data).short_fingerprint
    assert result.stdout.splitlines()[0] == f"test_toy {short} on qubit 0"


def test_check_keeps_each_row_whole_at_60_columns_and_names_a_skip_once() -> None:
    require("qiskit"), require("stim")
    args = ["check", "quantinuum_h1-1", "--framework", "qiskit,stim"]
    result = runner.invoke(app, args, env={"COLUMNS": "60"})
    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert max(map(len, lines)) <= 60
    header = next(line for line in lines if line.startswith("framework"))
    assert header.split() == ["framework", "result", "deviation", "circuits", "method"]
    qiskit = next(line for line in lines if line.startswith("qiskit ")).split()
    assert qiskit[:2] + qiskit[3:] == [
        "qiskit",
        "pass",
        "5",
        "of",
        "5",
        "exact",
        "+",
        "20000",
        "shots",
    ]
    assert re.fullmatch(r"\d\.\de-\d\d", qiskit[2])
    assert next(line for line in lines if line.startswith("stim ")).split() == ["stim", "skipped"]
    text = " ".join(result.stdout.split())
    assert "stim skipped:" not in text
    assert (
        "stim: Stim simulates only Clifford gates, and this profile's r gate is not Clifford at"
        " the check angles." in text
    )
    wide = runner.invoke(app, args, env={"COLUMNS": "80"}).stdout.splitlines()
    assert next(line for line in wide if line.startswith("framework")).split() == [
        "framework",
        "result",
        "deviation",
        "tolerance",
        "circuits",
        "method",
    ]


def test_cite() -> None:
    profile = nv.load("ibm_fez")
    text = runner.invoke(app, ["cite", "ibm_fez"]).stdout
    assert text == (
        "IBM Quantum, via qiskit-ibm-runtime. Calibration of ibm_fez, 2025-02-26T20:16:25Z."
        f" qiskit-ibm-runtime 0.49.0 FakeFez. NoiseVault {nv.__version__} profile"
        f" ibm_fez@2025-02-26T20:16:25Z, fingerprint sha256:{profile.fingerprint}.\n"
    )
    bibtex = runner.invoke(app, ["cite", "ibm_fez", "--bibtex"]).stdout
    assert bibtex == (
        "@misc{nv_ibm_fez_2025_02_26,\n"
        "  title = {{Calibrated noise of ibm\\_fez at 2025-02-26T20:16:25Z}},\n"
        "  author = {{IBM Quantum}},\n"
        "  year = {2025},\n"
        f"  howpublished = {{NoiseVault {nv.__version__} profile"
        f" ibm\\_fez@2025-02-26T20:16:25Z, sha256:{profile.fingerprint}}},\n"
        "  note = {Retrieved via qiskit-ibm-runtime."
        " Source: qiskit-ibm-runtime 0.49.0 FakeFez; license Apache-2.0}\n"
        "}\n"
    )


def test_cite_names_a_ref_that_loads_the_cited_calibration(vault: Path) -> None:
    bundled = nv.load("ibm_manila")
    _manila_in_the_vault("2024-06-03T10:00:00Z")
    for style in ([], ["--bibtex"]):
        text = runner.invoke(app, ["cite", "ibm_manila@2024-05-27", *style]).stdout
        ref = re.search(rf"NoiseVault {re.escape(nv.__version__)} profile (\S+),", text)
        assert ref, text
        assert nv.load(ref[1].replace("\\_", "_")).fingerprint == bundled.fingerprint


def _special_characters_profile(tmp_path: Path) -> Path:
    data = toy(
        provenance={
            "attribution": "R&D Lab (via my_tool #2)",
            "source": "50% of run_7 {raw}",
            "license": "CC0-1.0",
        }
    )
    data["device"]["calibrated_at"] = "2025-02-26T09:12:00Z"
    path = tmp_path / "special.json"
    path.write_text(json.dumps(data))
    return path


def test_cite_bibtex_escapes_latex_specials(tmp_path: Path) -> None:
    path = _special_characters_profile(tmp_path)
    fingerprint = nv.load(str(path)).fingerprint
    bibtex = runner.invoke(app, ["cite", str(path), "--bibtex"]).stdout
    assert bibtex == (
        "@misc{nv_test_toy_2025_02_26,\n"
        "  title = {{Calibrated noise of toy at 2025-02-26T09:12:00Z}},\n"
        "  author = {{R\\&D Lab}},\n"
        "  year = {2025},\n"
        f"  howpublished = {{NoiseVault {nv.__version__} profile"
        f" test\\_toy@2025-02-26T09:12:00Z, sha256:{fingerprint}}},\n"
        "  note = {Retrieved via my\\_tool \\#2."
        " Source: 50\\% of run\\_7 \\{raw\\}; license CC0-1.0}\n"
        "}\n"
    )


def _latex_command() -> list[list[str]] | None:
    if shutil.which("pdflatex") and shutil.which("bibtex"):
        return [["pdflatex", "-interaction=nonstopmode", "main.tex"], ["bibtex", "main"]] + [
            ["pdflatex", "-interaction=nonstopmode", "main.tex"]
        ] * 2
    if shutil.which("tectonic"):
        return [["tectonic", "-X", "compile", "--keep-intermediates", "main.tex"]]
    return None


def test_cite_bibtex_compiles_with_latex(tmp_path: Path) -> None:
    commands = _latex_command()
    if commands is None:
        pytest.skip("neither pdflatex with bibtex nor tectonic is installed")
    refs = [
        runner.invoke(app, ["cite", ref, "--bibtex"]).stdout
        for ref in ("ibm_fez", "google_rainbow", str(_special_characters_profile(tmp_path)))
    ]
    (tmp_path / "refs.bib").write_text("\n".join(refs))
    (tmp_path / "main.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\n\\nocite{*}\n"
        "\\bibliographystyle{plain}\n\\bibliography{refs}\n\\end{document}\n"
    )
    for command in commands:
        done = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
        assert done.returncode == 0, done.stdout + done.stderr
    bbl = (tmp_path / "main.bbl").read_text()
    # unbraced, "IBM Quantum, via qiskit-ibm-runtime" prints as "via qiskit-ibm-runtime IBM Quantum"
    for author in ("IBM Quantum", "R\\&D Lab"):
        assert f"\n{{{author}}}.\n" in bbl
    assert "2025-02-26T20:16:25Z" in bbl  # the braced title keeps its capitals


def test_schema_is_the_format_json_schema() -> None:
    result = runner.invoke(app, ["schema"])
    assert result.exit_code == 0 and json.loads(result.stdout) == nv.json_schema()


def test_doctor_lists_frameworks_and_the_vault(vault: Path) -> None:
    for module in ("qiskit", "cirq", "pennylane", "stim"):
        require(module)
    out = runner.invoke(app, ["doctor"]).stdout
    for package in ("qiskit", "cirq-core", "pennylane", "stim"):
        line = next(line for line in out.splitlines() if line.split()[:1] == [package])
        assert line.split()[1] == version(package)
    assert f"vault: {vault} (0 profiles)" in out
    assert f"bundled profiles: {len(nv.catalog.bundled_profiles())}" in out


def test_doctor_gives_a_whole_install_command_for_missing_packages(monkeypatch) -> None:
    from importlib.metadata import PackageNotFoundError

    import noisevault.cli as cli

    def without_stim(package: str) -> str:
        if package == "stim":
            raise PackageNotFoundError(package)
        return "1.0"

    monkeypatch.setattr(cli, "version", without_stim)
    lines = runner.invoke(app, ["doctor"], env={"COLUMNS": "80"}).stdout.splitlines()
    assert lines[-2:] == [f"To add the missing packages: {install_hint('stim')}", _TRY]


def test_check_with_no_framework_installed_is_one_error_and_one_install_command(
    monkeypatch,
) -> None:
    for module in ("qiskit", "cirq", "pennylane", "stim"):
        monkeypatch.setitem(sys.modules, module, None)
    result = runner.invoke(app, ["check", "ibm_manila"], env={"COLUMNS": "80"})
    assert result.exit_code == 1 and result.stdout == ""
    assert result.stderr.splitlines() == [
        "error: nv check needs a framework to check, and none is installed",
        f"hint: {install_hint('qiskit')}",
        "      (or cirq, pennylane or stim, or several, as in noisevault[qiskit,stim])",
        f'      or, with uv and no install: uvx --from "noisevault[qiskit] @ git+{REPOSITORY}"'
        " nv check ibm_manila",
    ]


@pytest.mark.parametrize("named", ["cirq", "cirq,Cirq, cirq"])
def test_check_of_a_named_framework_that_is_not_installed_installs_that_one(
    monkeypatch, named: str
) -> None:
    monkeypatch.setitem(sys.modules, "cirq", None)
    result = runner.invoke(app, ["check", "ibm_manila", "--framework", named])
    assert result.exit_code == 1 and result.stdout == ""
    assert result.stderr.splitlines() == [
        "error: nv check needs a framework to check, and cirq is not installed",
        f"hint: {install_hint('cirq')}",
        f'      or, with uv and no install: uvx --from "noisevault[cirq] @ git+{REPOSITORY}"'
        " nv check --framework cirq ibm_manila",
    ]


def test_check_states_what_a_pass_means_only_after_a_pass() -> None:
    require("stim")
    result = runner.invoke(app, ["check", "quantinuum_h1-1", "--framework", "stim"])
    assert result.exit_code == 1 and "A pass means" not in result.stdout
    assert "stim: Stim simulates only Clifford gates" in " ".join(result.stdout.split())
    assert "skipped:" not in result.stdout
    assert result.stderr == "error: no framework could run the check\n"


# bad input never prints a traceback -----------------------------------------------------------

_PROFILE_FILE = "give a profile file (.json or .json.gz)"
_DAMAGED_FILE = "the file is damaged or truncated. Pull or export the profile again"
_TOY_XX_COUNTS = Path(__file__).parent / "fixtures/compare/toy-xx.counts.json"
_NESTED_64 = "[" * 64 + "]" * 64


class _Damage(NamedTuple):
    text: str
    error: str
    hint: str | None
    listed: str | None = None


_DAMAGED = {
    "not-json": _Damage("{not json", "is not JSON (expecting property name", _DAMAGED_FILE),
    "empty": _Damage("", "is not JSON (expecting value", _DAMAGED_FILE),
    "cut-string": _Damage(
        '{"noisevault": "1.',
        "is not JSON (unterminated string starting at line 1, column 16)",
        _DAMAGED_FILE,
    ),
    "too-deep": _Damage(deeper_than_the_parser_takes(), "is not JSON (nested ", _DAMAGED_FILE),
    "legacy-empty": _Damage(
        '{"schema_version": "0.1"}',
        "not a valid NoiseVault 0.1 file: provider",
        "fix that field or pull the device again",
    ),
    "legacy-null-gates": _Damage(
        json.dumps({**json.loads(MANILA_V01.read_text()), "gates": None}),
        "gates should be a list, not null",
        "fix that field or pull the device again",
    ),
    "invalid": _Damage(
        json.dumps(toy(gates=None)),
        "is not a valid profile (1 problem)",
        "run nv validate {path} to list them",
        listed="gates: Input should be a valid dictionary",
    ),
    "extensions-too-deep": _Damage(
        json.dumps(toy(extensions={"deep": json.loads(_NESTED_64)})),
        "is not a valid profile (1 problem)",
        "run nv validate {path} to list them",
        listed="extensions: nested more than 64 levels deep",
    ),
}
_COMMANDS = {
    "show": lambda f: ["show", f],
    "cite": lambda f: ["cite", f],
    "check": lambda f: ["check", f, "--framework", "stim"],
    "diff-before": lambda f: ["diff", f, "ibm_manila"],
    "diff-after": lambda f: ["diff", "ibm_manila", f],
    "validate": lambda f: ["validate", f],
    "compare": lambda f: ["compare", f, str(_TOY_XX_COUNTS)],
}


@pytest.mark.parametrize("command", list(_COMMANDS))
@pytest.mark.parametrize("damage", list(_DAMAGED))
def test_every_command_names_a_damaged_file_and_what_is_wrong(
    tmp_path: Path, command: str, damage: str
) -> None:
    text, error, hint, listed = _DAMAGED[damage]
    path = tmp_path / f"{damage}.json"
    path.write_text(text)
    result = runner.invoke(app, _COMMANDS[command](str(path)), env={"COLUMNS": "80"})
    assert result.exit_code == 1 and result.stdout == ""
    assert "Traceback" not in result.output
    if command == "validate" and listed:
        assert result.stderr == f"error: {listed}\n"
        return
    first, *rest = result.stderr.splitlines()
    assert first.startswith(f"error: {path}") and error in first
    assert rest == ([f"hint: {hint.format(path=path)}"] if hint else [])


@pytest.mark.parametrize(
    ("given", "printed"),
    [("my run/toy.json", "'my run/toy.json'"), ("./-toy.json", "-- -toy.json")],
    ids=["a space", "a leading hyphen"],
)
@pytest.mark.parametrize("command", [c for c in _COMMANDS if c != "validate"])
def test_the_validate_hint_runs_as_printed(
    tmp_path: Path, monkeypatch, command: str, given: str, printed: str
) -> None:
    monkeypatch.chdir(tmp_path)
    path = Path(given)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(toy(gates=None)))
    result = runner.invoke(app, _COMMANDS[command](given), env={"COLUMNS": "80"})
    hint = result.stderr.splitlines()[-1]
    assert hint == f"hint: run nv validate {printed} to list them"
    nv_, *args = shlex.split(hint.removeprefix("hint: run ").removesuffix(" to list them"))
    followed = runner.invoke(app, args)
    assert (nv_, followed.exit_code, followed.stdout) == ("nv", 1, "")
    assert followed.stderr == "error: gates: Input should be a valid dictionary\n"


@pytest.mark.parametrize("command", list(_COMMANDS))
def test_every_command_says_a_counts_file_is_not_a_profile(tmp_path: Path, command: str) -> None:
    path = tmp_path / "run.counts.json"
    shutil.copy(_TOY_XX_COUNTS, path)
    result = runner.invoke(app, _COMMANDS[command](str(path)), env={"COLUMNS": "80"})
    assert (result.exit_code, result.stdout) == (1, "")
    hint = {
        "validate": _PROFILE_FILE,
        "compare": "nv compare takes the profile first and the counts file second",
    }.get(command, f"{_PROFILE_FILE} or a profile id such as ibm_fez")
    assert result.stderr == f"error: {path} is a counts file, not a profile\nhint: {hint}\n"


_COUNTS_DAMAGED_FILE = "the file is damaged or truncated. Save the counts again"
_DAMAGED_COUNTS = {
    "not-json": _Damage("{not json", "is not JSON (expecting property name", _COUNTS_DAMAGED_FILE),
    "empty": _Damage("", "is not JSON (expecting value", _COUNTS_DAMAGED_FILE),
    "cut-string": _Damage(
        '{"nv_counts": "1.',
        "is not JSON (unterminated string starting at line 1, column 15)",
        _COUNTS_DAMAGED_FILE,
    ),
    "too-deep": _Damage(
        deeper_than_the_parser_takes(), "is not JSON (nested ", _COUNTS_DAMAGED_FILE
    ),
    "profile": _Damage(
        json.dumps(toy()),
        "is a profile, not a counts file",
        "nv compare takes the profile first and the counts file second",
    ),
    "no-key": _Damage(
        '{"schema_version": "0.1"}',
        "is not a counts file (it has no nv_counts key)",
        'give a counts file, which holds "nv_counts": "1.0"',
    ),
    "invalid": _Damage(
        _TOY_XX_COUNTS.read_text().replace('"shots": 4000', '"shots": 3999'),
        ": circuits[0]: counts sum to 4000, but shots is 3999",
        "give the count of every outcome, so they add up to shots",
    ),
    "options-too-deep": _Damage(
        _TOY_XX_COUNTS.read_text().replace('"options": {', f'"options": {{"deep": {_NESTED_64}, '),
        ": execution.options: nested more than 64 levels deep",
        None,
    ),
}


@pytest.mark.parametrize("damage", list(_DAMAGED_COUNTS))
def test_compare_names_a_damaged_counts_file_and_what_is_wrong(tmp_path: Path, damage: str) -> None:
    text, error, hint, _ = _DAMAGED_COUNTS[damage]
    path = tmp_path / f"{damage}.json"
    path.write_text(text)
    toy_xx = Path(__file__).parent / "fixtures/compare/toy.json"
    result = runner.invoke(app, ["compare", str(toy_xx), str(path)], env={"COLUMNS": "80"})
    assert result.exit_code == 1 and result.stdout == ""
    assert "Traceback" not in result.output
    first, *rest = result.stderr.splitlines()
    assert first.startswith(f"error: {path}") and error in first
    assert rest == ([f"hint: {hint}"] if hint else [])


def test_a_cut_profile_file_is_called_damaged_and_another_file_type_is_named(
    tmp_path: Path,
) -> None:
    text = json.dumps(toy())[:40]
    cut, packed = tmp_path / "cut.json", tmp_path / "cut.json.gz"
    cut.write_text(text)
    packed.write_bytes(gzip.compress(text.encode()))
    for path in (cut, packed):
        result = runner.invoke(app, ["show", str(path)], env={"COLUMNS": "200"})
        assert result.exit_code == 1 and result.stdout == ""
        assert result.stderr.startswith(f"error: {path} is not JSON (")
        assert result.stderr.endswith(f")\nhint: {_DAMAGED_FILE}\n")
    notes = tmp_path / "notes.txt"
    notes.write_text("not a profile\n")
    result = runner.invoke(app, ["show", str(notes)], env={"COLUMNS": "200"})
    assert result.stderr == (
        f"error: {notes} is not JSON (expecting value at line 1, column 1)\nhint: {_PROFILE_FILE}\n"
    )


@pytest.mark.parametrize(
    ("args", "error"),
    [
        (["show", "ibm_fez@"], "after @ give a date (2025-02-26)"),
        (["show", "ibm_fez@2025-13-40"], "2025-13-40 is not a calendar date. Give one such as"),
        (["diff", "ibm_fez", "ibm_fez@"], "after @ give a date (2025-02-26)"),
        (["list", "--tech", "superconductin"], "did you mean 'superconducting'?"),
        (["list", "--vendor", "ibmm"], "did you mean 'ibm'?"),
        (["check", "ibm_manila", "--framework", "qiskt"], "did you mean 'qiskit'?"),
        (["validate", "missing.json"], "error: no file missing.json\nhint: check the path\n"),
    ],
)
def test_a_typo_or_bad_date_gets_one_error_with_the_way_out(args: list[str], error: str) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 1 and result.stdout == ""
    assert error in result.stderr and "Traceback" not in result.output
    assert len(result.stderr.splitlines()) == 1 + result.stderr.count("\nhint: ")


def test_pull_names_the_at_flag_for_a_bad_date() -> None:
    result = runner.invoke(app, ["pull", "ibm_fez", "--at", "2025-13-40"])
    assert result.exit_code == 1
    assert result.stderr.startswith("error: --at '2025-13-40' is not an ISO 8601 date")


# show assumptions -----------------------------------------------------------------------------


def test_show_states_the_assumption_behind_each_overriding_record(tmp_path: Path) -> None:
    gates = {
        "rz": {"virtual": True},
        "sx": {"avg_infidelity": 1e-3},
        "cz": {"avg_infidelity": 0.01, "assumption": "Default measured with isolated RB"},
    }
    calibrations = [
        {"gate": "cz", "qubits": [0, 1], "avg_infidelity": 0.02, "assumption": "Inferred from XEB"},
        {"gate": "cz", "qubits": [1, 2], "avg_infidelity": 0.03},
    ]
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(toy(gates=gates, calibrations=calibrations)))
    data = json.loads(runner.invoke(app, ["show", str(path), "--json"]).stdout)
    assert data["assumptions"] == [
        "cz: Default measured with isolated RB",
        "cz 0-1: Inferred from XEB",
    ]
    out = runner.invoke(app, ["show", str(path)], env={"COLUMNS": "120"}).stdout
    assert re.search(r"^assumptions\s+cz: Default measured with isolated RB\s*$", out, re.M)
    assert re.search(r"^\s+cz 0-1: Inferred from XEB\s*$", out, re.M)


def test_show_natives_as_resolved_not_as_defined(tmp_path: Path) -> None:
    gates = {
        "rz": {"virtual": True},
        "sx": {"avg_infidelity": 1e-3},
        "cz": {"avg_infidelity": 1e-2, "disabled": True},
    }
    calibrations = [{"gate": "rz", "qubits": [0], "virtual": False, "avg_infidelity": 2e-3}]
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(toy(gates=gates, calibrations=calibrations)))
    natives = json.loads(runner.invoke(app, ["show", str(path), "--json"]).stdout)["natives"]
    rz, _, cz = natives
    assert rz["virtual"] is False and rz["median_avg_infidelity"] == 2e-3
    assert rz["loci"] == {"calibrated": 1, "ideal": 2, "uncalibrated": 0, "disabled": 0}
    assert cz["median_avg_infidelity"] is None and cz["disabled"] == 2
    out = runner.invoke(app, ["show", str(path)], env={"COLUMNS": "120"}).stdout
    rows = {line.split()[0]: line.split()[1:] for line in out.splitlines() if "(1q)" in line}
    rows |= {line.split()[0]: line.split()[1:] for line in out.splitlines() if "(2q)" in line}
    assert rows["rz"] == ["(1q)", "2.00e-03", "-", "1", "(2", "virtual)"]
    assert rows["cz"] == ["(2q)", "-", "-", "device-wide", "(2", "disabled)"]


def test_show_counts_a_disabled_reverse_order_of_a_symmetric_gate(tmp_path: Path) -> None:
    calibrations = [
        {"gate": "cz", "qubits": [0, 1], "avg_infidelity": 1e-2},
        {"gate": "cz", "qubits": [1, 0], "disabled": True},
    ]
    data = toy(connectivity="all_to_all", calibrations=calibrations)
    data["device"]["num_qubits"] = 2
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(data))
    natives = json.loads(runner.invoke(app, ["show", str(path), "--json"]).stdout)["natives"]
    cz = next(native for native in natives if native["gate"] == "cz")
    assert cz["loci"] == {"calibrated": 1, "ideal": 0, "uncalibrated": 0, "disabled": 1}


def test_show_leaves_a_disabled_locus_out_of_the_median_duration(tmp_path: Path) -> None:
    calibrations = [{"gate": "cz", "qubits": [0, 1], "disabled": True, "duration_ns": 500}]
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(toy(calibrations=calibrations)))
    natives = json.loads(runner.invoke(app, ["show", str(path), "--json"]).stdout)["natives"]
    cz = next(native for native in natives if native["gate"] == "cz")
    assert (cz["median_duration_ns"], cz["disabled"]) == (70, 1)


def test_show_groups_a_record_assumption_shared_by_many_loci(tmp_path: Path) -> None:
    calibrations = [
        {"gate": "cz", "qubits": list(pair), "avg_infidelity": 0.02, "assumption": "From XEB"}
        for pair in ([0, 1], [1, 2], [2, 3], [3, 4])
    ]
    data = toy(calibrations=calibrations)
    data["device"]["num_qubits"] = 5
    data["connectivity"] = {"edges": [[0, 1], [1, 2], [2, 3], [3, 4]]}
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(data))
    shown = json.loads(runner.invoke(app, ["show", str(path), "--json"]).stdout)
    assert shown["assumptions"] == ["cz (4 records): From XEB"]


# doctor ---------------------------------------------------------------------------------------


def _doctor_without(monkeypatch: pytest.MonkeyPatch, *absent: str) -> str:
    from importlib.metadata import PackageNotFoundError

    import noisevault.cli as cli

    def installed(package: str) -> str:
        if package in absent:
            raise PackageNotFoundError(package)
        return "1.0"

    monkeypatch.setattr(cli, "version", installed)
    return runner.invoke(app, ["doctor"], env={"COLUMNS": "200"}).stdout


def test_doctor_installs_pymatching_by_name_because_no_extra_has_it(monkeypatch) -> None:
    out = _doctor_without(monkeypatch, "pymatching")
    assert re.search(r"^pymatching +not installed$", out, re.M)
    *_, bundled, advice = out.splitlines()
    assert bundled.startswith("bundled profiles: ") and advice == _PYMATCHING


_PYMATCHING = "To add pymatching: pip install pymatching"


def _install(extra: str) -> str:
    return f"To add the missing packages: {install_hint(extra)}"


_TRY = _uvx("qiskit,cirq,pennylane,stim", "nv check ibm_fez")


@pytest.mark.parametrize(
    ("absent", "advice"),
    [
        ((), []),
        (("stim",), [_install("stim"), _TRY]),
        (
            ("cirq-core", "cirq-google", "pennylane", "stim"),
            [_install("cirq,google,pennylane,stim"), _TRY],
        ),
        (("pyarrow",), [_install("hf")]),
        (("cirq-google", "stim", "pymatching"), [_install("google,stim"), _TRY, _PYMATCHING]),
        (("qiskit-ibm-runtime", "cirq-google", "pyarrow"), [_install("google,hf,ibm")]),
        (tuple(_PACKAGES), [_install("all"), _TRY, _PYMATCHING]),
    ],
    ids=[
        "nothing",
        "stim",
        "only_qiskit",
        "only_hf",
        "google_and_stim",
        "no_check_framework",
        "everything",
    ],
)
def test_doctor_installs_what_is_missing_and_tries_nv_check_with_every_framework(
    monkeypatch, absent, advice
) -> None:
    lines = _doctor_without(monkeypatch, *absent).splitlines()
    end = next(i for i, line in enumerate(lines) if line.startswith("bundled profiles: "))
    assert lines[end + 1 :] == advice


def test_each_extra_doctor_names_installs_its_package() -> None:
    import tomllib

    extras = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]

    def requirements(extra: str) -> set[str]:
        found = set()
        for item in extras[extra]:
            nested = re.fullmatch(r"noisevault\[(.+)\]", item)
            if nested:
                found |= {r for name in nested[1].split(",") for r in requirements(name)}
            else:
                found.add(re.split(r"[<>=!~ ;\[]", item)[0])
        return found

    for package, extra in _PACKAGES.items():
        if extra is not None:
            assert package in requirements(extra), (package, extra)
        assert package in requirements("dev")


def test_doctor_reports_a_package_of_every_extra_that_all_installs() -> None:
    import tomllib

    from noisevault.cli import _EXTRAS

    extras = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    (everything,) = extras["all"]
    assert _EXTRAS == sorted(re.fullmatch(r"noisevault\[(.+)\]", everything)[1].split(","))


# diff and list with two calibrations on one day -----------------------------------------------


def _same_day(vault: Path) -> tuple[str, str]:
    """Two H1-1 calibrations on 2025-05-02 in the vault, beside the bundled one of that day."""
    base = nv.load("quantinuum_h1-1").model_dump(mode="json", exclude_none=True)

    def save(when: str, zz: float, qubits: list, calibrations: list) -> str:
        data = {**base, "device": {**base["device"], "calibrated_at": when}}
        data["gates"] = {**base["gates"], "zz": {**base["gates"]["zz"], "avg_infidelity": zz}}
        data["qubits"], data["calibrations"] = qubits, calibrations
        profile = Profile.model_validate(data)
        profile.save(nv.catalog.vault_path(profile))
        return f"quantinuum_h1-1@{when}"

    before = save(
        "2025-05-02T09:15:00Z",
        9.8e-4,
        [{"index": 4, "t1_us": 1.0e7}],
        [{"gate": "r", "qubits": [5], "avg_infidelity": 0.0}],
    )
    after = save(
        "2025-05-02T14:30:00Z",
        1.2e-3,
        [{"index": 3, "t1_us": 5.0e6}],
        [
            {"gate": "r", "qubits": [5], "avg_infidelity": 4e-5},
            {"gate": "zz", "qubits": [0, 1], "avg_infidelity": 3e-3},
        ],
    )
    return before, after


def test_diff_says_new_or_gone_and_colors_every_change(vault: Path) -> None:
    before, after = _same_day(vault)
    env = {"COLUMNS": "80", "FORCE_COLOR": "1"}
    out = runner.invoke(app, ["diff", before, after], env=env).stdout
    plain = _unstyled(out)
    lines = plain.splitlines()
    assert lines[0] == "quantinuum_h1-1 2025-05-02T09:15Z -> 2025-05-02T14:30Z  (5 hours later)"
    rows = {tuple(line.split()[:3]): line.split()[-1] for line in lines if line[:1].isalnum()}
    assert rows[("3", "T1", "(us)")] == "new"
    assert rows[("4", "T1", "(us)")] == "gone"
    assert re.search(r"^3 +T1 \(us\) +- +5000000\.0 +new$", plain, re.M)
    assert re.search(r"^4 +T1 \(us\) +10000000\.0 +- +gone$", plain, re.M)
    assert rows[("default", "2q", "avg")] == "+22.4%"
    assert re.search(
        r"^5 +1q avg infidelity +0\.00e\+00 +4\.00e-05 +\x1b\[31m-\x1b\[0m$", out, re.M
    )
    assert max(map(len, lines)) <= 80


def test_diff_of_one_device_on_one_day_names_the_times_in_its_age_warning(vault: Path) -> None:
    before, after = _same_day(vault)
    result = runner.invoke(app, ["diff", after, before], env={"COLUMNS": "80"})
    assert result.stdout.splitlines()[0] == (
        "quantinuum_h1-1 2025-05-02T14:30Z -> 2025-05-02T09:15Z  (5 hours earlier)"
    )
    assert result.stderr == (
        "warning: quantinuum_h1-1@2025-05-02T09:15Z is older than"
        " quantinuum_h1-1@2025-05-02T14:30Z. Before and after follow the argument order, not time\n"
    )


def test_list_shows_the_time_only_where_a_date_is_shared(vault: Path) -> None:
    _same_day(vault)
    out = runner.invoke(app, ["list"], env={"COLUMNS": "80"}).stdout
    lines = out.splitlines()
    at = [i for i, line in enumerate(lines) if "quantinuum_h1-1 " in line]
    rows = [(lines[i].lstrip("* ").split()[1:3], lines[i + 1].split()) for i in at]
    assert rows == [
        (["2025-05-02", "20"], ["14:30", "UTC"]),
        (["2025-05-02", "20"], ["09:15", "UTC"]),
        (["2025-05-02", "20"], ["00:00", "UTC"]),
    ]
    h1_2 = next(i for i, line in enumerate(lines) if "quantinuum_h1-2 " in line)
    assert lines[h1_2].split()[1] == "2023-08-21" and "quantinuum_h2-1" in lines[h1_2 + 1]
    fez = next(line for line in lines if "ibm_fez" in line)
    assert fez.split()[:3] == ["ibm_fez", "2025-02-26", "156"]
    assert max(map(len, lines)) <= 80


def test_diff_lists_added_qubits_as_ranges() -> None:
    out = runner.invoke(app, ["diff", "ibm_manila", "ibm_fez"], env={"COLUMNS": "80"}).stdout
    assert "qubits added: 5 to 155" in out.splitlines()


@pytest.mark.parametrize(
    ("args", "error"),
    [
        (["diff", "ibm_fez"], "error: missing argument 'AFTER'\nhint: run nv diff --help\n"),
        (["shwo", "ibm_fez"], "error: no such command 'shwo'; did you mean 'show'?\n"),
        (
            ["show", "ibm_fez", "--qubit", "1"],
            "error: no such option '--qubit'; did you mean '--qubits'?\n",
        ),
        (["show", "ibm_fez", "--jsno"], "error: no such option '--jsno'; did you mean '--json'?\n"),
        (
            ["show", "ibm_fez", "--qubits"],
            "error: option '--qubits' requires an argument\n"
            "hint: give qubit indices, for example --qubits 0,1,2\n",
        ),
        (
            ["list", "--tech"],
            "error: option '--tech' requires an argument\n"
            "hint: give a technology, for example --tech trapped_ion\n",
        ),
        (
            ["pull", "ibm_fez", "-o"],
            "error: option '-o' requires an argument\n"
            "hint: give a file name, for example -o fez.json\n",
        ),
    ],
)
def test_a_usage_mistake_is_an_error_and_at_most_one_hint(args: list[str], error: str) -> None:
    result = runner.invoke(app, args, env={"COLUMNS": "80"}, prog_name="nv")
    assert result.exit_code == 2 and result.stdout == ""
    assert result.stderr == error


def test_every_ref_argument_is_described_the_same_way() -> None:
    def help_text(command: str) -> str:
        result = runner.invoke(app, [command, "--help"], env={"COLUMNS": "200"}, prog_name="nv")
        return _unstyled(result.output)

    for command in ("show", "check", "cite", "compare"):
        assert "Profile id (ibm_fez), id@date, or a file path." in help_text(command), command
    diff = help_text("diff")
    assert "First profile id (ibm_fez), id@date, or a file path." in diff
    assert "--top" in diff and "<int range>" not in diff


@pytest.mark.parametrize(
    ("command", "arguments"),
    [
        ("show", "REF"),
        ("check", "REF"),
        ("cite", "REF"),
        ("pull", "DEVICE"),
        ("diff", "BEFORE AFTER"),
        ("validate", "FILE"),
        ("compare", "REF COUNTS"),
    ],
)
def test_usage_lines_name_arguments_in_capitals_and_options_by_their_value(
    command: str, arguments: str
) -> None:
    result = runner.invoke(app, [command, "--help"], env={"COLUMNS": "80"}, prog_name="nv")
    text = _unstyled(result.output)
    assert re.search(rf"^ Usage: nv {command} \[OPTIONS\] {arguments} *$", text, re.M), text
    assert "<str>" not in text and "<path>" not in text and "{" not in text
    if command == "show":
        assert re.search(
            r"^│ --qubits +LIST +Also list these qubits, for example 0,1,2\.", text, re.M
        )
    if command == "pull":
        assert re.search(r"^│ --output +-o +FILE +Save here instead of the vault\.", text, re.M)
    if command == "compare":
        assert re.search(r"^│ --output +-o +FILE +Save the profile with the fitted", text, re.M)


def test_nv_alone_prints_the_help_and_no_error() -> None:
    result = runner.invoke(app, [], env={"COLUMNS": "80"}, prog_name="nv")
    assert result.exit_code == 0, result.output
    text = _unstyled(result.output)
    assert "Usage: nv" in text and "list" in text
    assert "error:" not in text


def test_help_ends_with_the_commands_to_start_with() -> None:
    start = [
        " Start with:",
        "   nv list             the bundled devices, offline",
        "   nv show ibm_fez     one device's calibration",
        "   nv check ibm_fez    each export against the reference",
    ]
    for args in ([], ["--help"]):
        result = runner.invoke(app, args, env={"COLUMNS": "60"}, prog_name="nv")
        assert result.exit_code == 0, result.output
        lines = _unstyled(result.output).rstrip().splitlines()
        assert lines[-4:] == start and lines[-5] == ""
        assert max(map(len, lines)) <= 60
    sub = runner.invoke(app, ["show", "--help"], env={"COLUMNS": "80"}, prog_name="nv")
    assert "Start with" not in sub.output


def test_an_unexpected_failure_is_one_error_and_a_hint_unless_debugging(monkeypatch) -> None:
    def broken(ref: str) -> None:
        raise KeyError("provider")

    monkeypatch.setattr(nv.catalog, "load", broken)
    result = runner.invoke(app, ["show", "ibm_fez"])
    assert result.exit_code == 1
    assert result.stderr.splitlines() == [
        "error: unexpected KeyError: 'provider'",
        "hint: report this bug at https://github.com/dvgyl/noisevault/issues"
        " (NOISEVAULT_DEBUG=1 shows the traceback)",
    ]
    debug = runner.invoke(app, ["show", "ibm_fez"], env={"NOISEVAULT_DEBUG": "1"})
    assert isinstance(debug.exception, KeyError)


ROOT = Path(__file__).resolve().parents[1]
KINGSTON = "ibm_kingston@2026-04-15"
EXAMPLE = "examples/kingston-simulated.counts.json"
FITS = ROOT / "tests" / "fixtures" / "compare"


class _Pinned(NamedTuple):
    ref: str
    counts: str
    cwd: Path


_PINNED = {
    "good-fit": _Pinned(KINGSTON, EXAMPLE, ROOT),
    "poor-fit": _Pinned(KINGSTON, "kingston-excess-readout.counts.json", FITS),
    "ruled-out": _Pinned("ibm_fez@2025-02-26", "fez-impossible-readout.counts.json", FITS),
    "not-identified": _Pinned("toy.json", "toy-xx.counts.json", FITS),
}


def _pinned_inputs() -> dict[str, Any]:
    from test_compare import LATER, XX, excess_readout, fez_with_impossible_shots
    from test_compare import toy as one_qubit_toy

    device = one_qubit_toy()
    xx = PlannedCircuit(name="xx", qubits=(0,), ops=XX)
    return {
        "kingston-excess-readout.counts.json": excess_readout(0),
        "fez-impossible-readout.counts.json": fez_with_impossible_shots(9)[1],
        "toy.json": device,
        "toy-xx.counts.json": simulate(device, [xx], shots=4000, seed=2, run_at=LATER),
    }


def _compare(
    args: list[str], cwd: Path, monkeypatch: pytest.MonkeyPatch, columns: int = 80
) -> Result:
    monkeypatch.chdir(cwd)
    return runner.invoke(app, ["compare", *args], env={"COLUMNS": str(columns)})


def _layout(lines: list[str]) -> None:
    assert all(len(line) <= 80 and line == line.rstrip() for line in lines), lines
    assert all(lines[:3]) and lines[3] == "" and lines[4].startswith("circuit ")
    end = lines.index("", 4)
    impossible = lines[4].endswith("  impossible shots")
    for row in lines[5:end]:
        fitted, noise = (float(value) for value in row.split()[3:5])
        assert row.endswith("  beyond noise") == (not impossible and fitted > noise), row
    block = lines[end + 1 :]
    block = block[: block.index("")] if "" in block else block
    labels = {"gate errors", "readout errors", "fit", "next", "note", ""}
    assert all(line[:16].rstrip() in labels and line[16] != " " for line in block), block
    factors = [line[16:] for line in block if line[:16].rstrip().endswith("errors")]
    fitted_any = any(value.startswith("x") for value in factors)
    assert (lines[-4:] == ["", *COMPARE_NOTE]) == fitted_any


@pytest.mark.parametrize("name", list(_PINNED))
def test_each_compare_output_is_the_pinned_text(name: str, monkeypatch) -> None:
    ref, counts, cwd = _PINNED[name]
    result = _compare([ref, counts], cwd, monkeypatch)
    assert result.exit_code == 0, result.output
    text = _unstyled(result.stdout)
    assert text == (FITS / f"{name}.txt").read_text(encoding="utf-8"), (
        f"tests/fixtures/compare/{name}.txt is stale; regenerate it with python tests/test_cli.py"
    )
    lines = text.splitlines()
    _layout(lines)
    assert lines[4].endswith("  impossible shots") == (name == "ruled-out")
    assert any(line.endswith("  beyond noise") for line in lines) == (name == "poor-fit")
    assert (COMPARE_NOTE[0] in lines) == (name != "not-identified")


def test_the_pinned_compare_inputs_are_current(tmp_path: Path) -> None:
    for name, made in _pinned_inputs().items():
        assert made.save(tmp_path / name).read_bytes() == (FITS / name).read_bytes(), (
            f"tests/fixtures/compare/{name} is stale; regenerate it with python tests/test_cli.py"
        )


_NARROW_HEADS = {
    ("poor-fit", 60): "circuit       fitted TVD  noise TVD 95%",
    ("poor-fit", 72): "circuit       profile TVD  fitted TVD  noise TVD 95%",
    ("ruled-out", 60): "circuit       fitted TVD  noise TVD 95%  impossible shots",
    ("ruled-out", 72): "circuit       profile TVD  fitted TVD  noise TVD 95%  impossible shots",
}


@pytest.mark.parametrize("columns", [60, 72])
@pytest.mark.parametrize("name", list(_PINNED))
def test_a_narrow_terminal_gets_compare_lines_that_fit_and_break_between_words(
    name: str, columns: int, monkeypatch
) -> None:
    ref, counts, cwd = _PINNED[name]
    result = _compare([ref, counts], cwd, monkeypatch, columns)
    assert result.exit_code == 0, result.output
    lines = _unstyled(result.stdout).splitlines()
    wide = (FITS / f"{name}.txt").read_text(encoding="utf-8").splitlines()
    assert max(map(len, lines)) <= columns, max(lines, key=len)
    assert set(" ".join(lines).split()) <= set(" ".join(wide).split())
    assert all(line.count("(") == line.count(")") for line in lines), lines
    head = lines.index(next(line for line in lines if line.startswith("circuit ")))
    wide_head = wide.index(next(line for line in wide if line.startswith("circuit ")))
    assert lines[head] == _NARROW_HEADS.get((name, columns), wide[wide_head])
    flagged = [row.split()[0] for row in lines[head:] if row.endswith("  beyond noise")]
    assert flagged == [row.split()[0] for row in wide if row.endswith("  beyond noise")]
    block = lines[lines.index("", head) + 1 :]
    block = block[: block.index("")] if "" in block else block
    labels = {"gate errors", "readout errors", "fit", "next", "note", ""}
    assert all(line[:16].rstrip() in labels and line[16] != " " for line in block), block


def test_compare_prints_the_ref_the_table_header_and_each_label_in_bold(monkeypatch) -> None:
    monkeypatch.chdir(FITS)
    args = ["compare", *_PINNED["poor-fit"][:2]]
    styled = runner.invoke(app, args, env={"COLUMNS": "80", "FORCE_COLOR": "1"}).stdout
    lines = _unstyled(styled).splitlines()
    bold = re.findall(r"\x1b\[1m(.*?)\x1b\[0m", styled)
    assert bold == [KINGSTON, lines[4], "gate errors", "readout errors", "fit"]
    assert all(line == line.rstrip() for line in lines)


def _pair(folder: Path) -> tuple[str, str]:
    profile = Profile.uniform(
        "pair",
        technology="superconducting",
        num_qubits=2,
        one_qubit_error=0.01,
        two_qubit_error=0.03,
        readout_error=0.02,
    )
    truth = profile.model_copy(
        update={"unmodeled_error": {"gates": {"factor": 1.5}, "readout": {"factor": 1.2}}}
    )
    counts = simulate(truth, plan(profile), shots=4000, seed=1, run_at=_RUN_AT)
    profile.save(folder / "pair.json")
    counts.save(folder / "pair.counts.json")
    return "pair.json", "pair.counts.json"


_RUN_AT = datetime(2026, 10, 1, 12, tzinfo=UTC)


def _never_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fit(self: Profile, counts: Any) -> None:
        pytest.fail("nv compare fitted before refusing -o")

    monkeypatch.setattr(Profile, "compare", fit)


@pytest.mark.parametrize(
    ("output", "what"),
    [
        ("pair.counts.json", "counts file"),
        ("./pair.counts.json", "counts file"),
        ("link.json", "counts file"),
        ("hard.json", "counts file"),
        ("pair.json", "profile file"),
        ("profile-link.json", "profile file"),
    ],
)
def test_compare_o_refuses_a_file_it_reads_before_fitting(
    tmp_path: Path, monkeypatch, output: str, what: str
) -> None:
    profile, counts = _pair(tmp_path)
    (tmp_path / "link.json").symlink_to(counts)
    os.link(tmp_path / counts, tmp_path / "hard.json")
    (tmp_path / "profile-link.json").symlink_to(profile)
    before = {name: (tmp_path / name).read_bytes() for name in (profile, counts)}
    _never_fit(monkeypatch)
    result = _compare([profile, counts, "-o", output], tmp_path, monkeypatch)
    assert result.exit_code == 1 and result.stdout == ""
    assert result.stderr.splitlines() == [
        f"error: -o {Path(output)} is the {what} nv compare reads",
        "hint: save the fitted profile elsewhere, such as pair-fitted.json",
    ]
    assert {name: (tmp_path / name).read_bytes() for name in before} == before


def _vault_ref_backed_elsewhere(vault: Path, folder: Path) -> Path:
    vault.mkdir(parents=True)
    copy = folder / "kingston-copy.json.gz"
    nv.load(KINGSTON).save(copy)
    (vault / f"{nv.catalog.vault_path(nv.load(KINGSTON)).name}").symlink_to(copy)
    return copy


@pytest.mark.parametrize(
    "where", ["bundled-file", "bundled-new", "vault-new", "vault-dotdot", "vault-backing"]
)
def test_compare_o_refuses_the_profiles_noisevault_holds(
    tmp_path: Path, vault: Path, monkeypatch, where: str
) -> None:
    bundled = Path(str(nv.catalog.bundled_dir()))
    read = "is the profile file nv compare reads"
    targets = {
        "bundled-file": (bundled / f"{KINGSTON}.json.gz", read),
        "bundled-new": (bundled / "kingston-fitted.json", "is in NoiseVault's bundled profiles"),
        "vault-new": (vault / "kingston-fitted.json", "is in your vault"),
        "vault-dotdot": (vault / ".." / "profiles" / "fitted.json", "is in your vault"),
        "vault-backing": (tmp_path / "kingston-copy.json.gz", read),
    }
    output, error = targets[where]
    if where == "vault-backing":
        _vault_ref_backed_elsewhere(vault, tmp_path)
    vault.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / EXAMPLE, tmp_path / "run.counts.json")
    before = output.read_bytes() if output.exists() else None
    _never_fit(monkeypatch)
    result = _compare([KINGSTON, "run.counts.json", "-o", str(output)], tmp_path, monkeypatch)
    assert result.exit_code == 1 and result.stdout == ""
    assert result.stderr.startswith(f"error: -o {output} {error}\nhint: ")
    assert (output.read_bytes() if output.exists() else None) == before


def test_compare_exit_codes(tmp_path: Path, monkeypatch) -> None:
    edited = nv.load(KINGSTON).to_dict()
    edited["qubits"][0]["t1_us"] = edited["qubits"][0].get("t1_us", 100) + 1
    Profile.from_dict(edited).save(tmp_path / "edited.json")
    shutil.copy(ROOT / EXAMPLE, tmp_path / "run.counts.json")
    refused = {
        ("ibm_fez", "run.counts.json"): [
            "error: these counts ran on ibm_kingston, but the profile describes ibm_fez",
            "hint: give the profile the counts were planned from",
        ],
        ("edited.json", "run.counts.json"): [
            "error: these counts were planned from nv:609c845ed934, but this profile's calibration"
            f" is nv:{_calibration_short(tmp_path / 'edited.json')}",
            "hint: run nv list to find nv:609c845ed934",
        ],
    }
    for args, stderr in refused.items():
        result = _compare(list(args), tmp_path, monkeypatch)
        assert (result.exit_code, result.stdout, result.stderr.splitlines()) == (1, "", stderr)
    usage = runner.invoke(app, ["compare", KINGSTON], prog_name="nv")
    assert (usage.exit_code, usage.stderr) == (
        2,
        "error: missing argument 'COUNTS'\nhint: run nv compare --help\n",
    )
    ref, counts, cwd = _PINNED["not-identified"]
    unsaved = _compare([ref, counts, "-o", str(tmp_path / "fitted.json")], cwd, monkeypatch)
    assert unsaved.exit_code == 1
    assert _unstyled(unsaved.stdout) == (FITS / "not-identified.txt").read_text(encoding="utf-8")
    assert (
        unsaved.stderr
        == "error: gate and readout factors are not identified, so there is nothing to save\n"
    )
    assert not (tmp_path / "fitted.json").exists()


@pytest.mark.parametrize("packed", [False, True])
def test_compare_says_which_argument_comes_first_when_they_are_swapped(
    tmp_path: Path, monkeypatch, packed: bool
) -> None:
    profile, counts = _pair(tmp_path)
    if packed:
        packed_counts = tmp_path / "pair.counts.json.gz"
        packed_counts.write_bytes(gzip.compress((tmp_path / counts).read_bytes()))
        counts = packed_counts.name
    swapped = _compare([counts, profile], tmp_path, monkeypatch)
    assert (swapped.exit_code, swapped.stdout) == (1, "")
    assert swapped.stderr.splitlines() == [
        f"error: {counts} is a counts file, not a profile",
        "hint: nv compare takes the profile first and the counts file second",
    ]
    shown = runner.invoke(app, ["show", counts])
    assert shown.stderr.splitlines() == [
        f"error: {counts} is a counts file, not a profile",
        "hint: give a profile file (.json or .json.gz) or a profile id such as ibm_fez",
    ]


def _calibration_short(path: Path) -> str:
    return Profile.load(path).uncorrected().short_fingerprint.removeprefix("nv:")


@pytest.mark.parametrize("damage", ["missing", "folder", "unreadable"])
def test_compare_names_a_counts_file_it_cannot_read(tmp_path: Path, monkeypatch, damage) -> None:
    profile, _ = _pair(tmp_path)
    path = tmp_path / "run.counts.json"
    expected = {
        "missing": [f"error: no file {path}", "hint: check the path"],
        "folder": [f"error: {path} is a folder", "hint: give a counts file (.json or .json.gz)"],
        "unreadable": [
            f"error: cannot read {path}: Permission denied",
            "hint: check the file and its permissions",
        ],
    }
    if damage == "folder":
        path.mkdir()
    if damage == "unreadable":
        if os.name == "nt" or os.geteuid() == 0:
            pytest.skip("chmod 000 does not stop this user from reading")
        path.write_text("{}")
        path.chmod(0)
    result = _compare([profile, str(path)], tmp_path, monkeypatch)
    assert (result.exit_code, result.stdout) == (1, "")
    assert result.stderr.splitlines() == expected[damage]


def test_compare_o_saves_the_fitted_profile_the_same_way_twice(tmp_path: Path, monkeypatch) -> None:
    profile, counts = _pair(tmp_path)
    first = _compare([profile, counts, "-o", "a.json"], tmp_path, monkeypatch)
    second = _compare([profile, counts, "-o", "b.json"], tmp_path, monkeypatch)
    assert first.exit_code == second.exit_code == 0, first.output
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    fitted = Profile.load(tmp_path / "a.json")
    assert _unstyled(first.stdout).endswith(
        f"\n\nsaved: a.json, pair {fitted.short_fingerprint} with the fitted factors\n"
    )
    assert fitted.uncorrected().fingerprint == Profile.load(tmp_path / profile).fingerprint
    require("cirq")
    checked = runner.invoke(app, ["check", "a.json", "--framework", "cirq"])
    assert checked.exit_code == 0, checked.output


def _strict_json(text: str) -> Any:
    def refuse(constant: str) -> None:
        raise AssertionError(f"{constant} is not JSON")

    return json.loads(text, parse_constant=refuse)


def test_compare_json_says_written_only_after_the_save(tmp_path: Path, monkeypatch) -> None:
    profile, counts = _pair(tmp_path)
    saved = _compare([profile, counts, "--json", "-o", "fitted.json"], tmp_path, monkeypatch)
    assert saved.exit_code == 0 and saved.stderr == ""
    data = _strict_json(saved.stdout)
    fitted = Profile.load(tmp_path / "fitted.json")
    assert data["written"] == {"path": "fitted.json", "fingerprint": fitted.fingerprint}
    assert data["gates"]["factor"] == pytest.approx(fitted.unmodeled_error.gates.factor, rel=1e-5)

    ref, xx, cwd = _PINNED["not-identified"]
    target = tmp_path / "unsaved.json"
    unsaved = _compare([ref, xx, "--json", "-o", str(target)], cwd, monkeypatch)
    assert unsaved.exit_code == 1 and not target.exists()
    assert "written" not in _strict_json(unsaved.stdout)
    assert (
        unsaved.stderr
        == "error: gate and readout factors are not identified, so there is nothing to save\n"
    )


def test_compare_json_without_written_when_the_folder_turns_read_only(
    tmp_path: Path, monkeypatch
) -> None:
    if os.name == "nt" or os.geteuid() == 0:
        pytest.skip("a read-only folder does not stop this user from writing")
    profile, counts = _pair(tmp_path)
    folder = tmp_path / "out"
    folder.mkdir()
    fit = Profile.compare

    def fit_then_lock(self: Profile, measured: Any) -> Any:
        result = fit(self, measured)
        folder.chmod(0o555)
        return result

    monkeypatch.setattr(Profile, "compare", fit_then_lock)
    try:
        result = _compare([profile, counts, "--json", "-o", "out/f.json"], tmp_path, monkeypatch)
    finally:
        folder.chmod(0o755)
    assert result.exit_code == 1 and list(folder.iterdir()) == []
    assert "written" not in _strict_json(result.stdout)
    assert result.stderr.splitlines() == [
        "error: cannot write out/f.json: Permission denied",
        "hint: choose another folder",
    ]


def test_json_never_prints_a_value_json_cannot_hold(monkeypatch) -> None:
    from noisevault import cli

    card = cli.card
    monkeypatch.setattr(cli, "card", lambda profile: {**card(profile), "median_t1_us": math.nan})
    result = runner.invoke(app, ["show", "ibm_manila", "--json"])
    assert result.exit_code == 1 and result.stdout == ""
    assert result.stderr.startswith("error: Out of range float values are not JSON compliant")


if __name__ == "__main__":
    import tempfile

    os.environ["NOISEVAULT_HOME"] = tempfile.mkdtemp()
    FITS.mkdir(parents=True, exist_ok=True)
    for name, made in _pinned_inputs().items():
        made.save(FITS / name)
        print(f"wrote tests/fixtures/compare/{name}")
    for name, (ref, counts, cwd) in _PINNED.items():
        os.chdir(cwd)
        result = runner.invoke(app, ["compare", ref, counts], env={"COLUMNS": "80"})
        assert result.exit_code == 0, result.output
        (FITS / f"{name}.txt").write_text(_unstyled(result.stdout), encoding="utf-8")
        print(f"wrote tests/fixtures/compare/{name}.txt")
