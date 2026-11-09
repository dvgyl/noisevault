"""The examples and the code in docs/ run as written, and docs/schema matches the code.

Each example and each Python block in docs/ runs alone in a fresh interpreter with an empty
vault. The ``# `` comment lines after a block's last print show its output and must appear in
what it prints. A missing optional package skips the run, unless NOISEVAULT_REQUIRE_ALL=1.
Tests that need the network run only with NOISEVAULT_NETWORK=1. A Python block preceded by a
``<!-- not-run: reason -->`` comment is skipped.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

import pytest
from typer.testing import CliRunner

import noisevault as nv
from noisevault.cli import app
from noisevault.diff import METRICS

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
DOCS = ROOT / "docs"
SCHEMA = DOCS / "schema" / "profile-1.0.json"
HAS_MITIQ = find_spec("mitiq") is not None and find_spec("ply") is not None
HAS_MITIQ_CIRQ = find_spec("mitiq") is not None and find_spec("cirq") is not None
TUTORIALS = DOCS / "tutorials"
# optional packages that the "all" extra does not install, so NOISEVAULT_REQUIRE_ALL never demands
OUTSIDE_ALL = {"mitiq", "ply"}

# example -> text its output must contain
LOCAL_EXAMPLES = {
    "quickstart.py": "GHZ success (000 or 111)",
    "pennylane_gradient.py": "step 30",
    "cirq_google.py": "linear XEB fidelity",
    "custom_device.py": "approximated: effect atom_loss on readout",
    "qec_surface_code.py": "d=5: logical error",
    "neutral_atom_erasure.py": "logical error, heralds used:",
}
OTHER_EXAMPLES = {"compare_devices.py", "drift.py", "mitigation_zne.py", "mitiq_zne_pec.py"}

FENCE = re.compile(r"^```(\w*)[^\n]*\n(.*?)^```", re.S | re.M)
NOT_RUN = re.compile(r"<!--\s*not-run:[^>]*-->\s*$")
INDENTED_FENCE = re.compile(r"^ *```(\w*)\n(.*?)^ *```", re.S | re.M)


def run(
    tmp_path: Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    environment = {
        **os.environ,
        "NOISEVAULT_HOME": str(tmp_path / "nv_home"),
        "MPLBACKEND": "Agg",
        **(env or {}),
    }
    return subprocess.run(
        [sys.executable, *args],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )


def skip_if_optional_missing(result: subprocess.CompletedProcess) -> None:
    """Skip when the run failed only because an optional package is not installed."""
    missing = re.search(r"ModuleNotFoundError: No module named '(\w+)", result.stderr)
    if result.returncode == 0 or missing is None or missing.group(1) == "noisevault":
        return
    if os.environ.get("NOISEVAULT_REQUIRE_ALL") == "1" and missing.group(1) not in OUTSIDE_ALL:
        pytest.fail(f"{missing.group(1)} must be installed when NOISEVAULT_REQUIRE_ALL=1")
    pytest.skip(f"{missing.group(1)} is not installed")


def check(result: subprocess.CompletedProcess, expected: str) -> None:
    skip_if_optional_missing(result)
    assert result.returncode == 0, result.stderr[-3000:]
    assert expected in result.stdout, result.stdout[-3000:]


def test_every_example_is_listed() -> None:
    found = {path.name for path in EXAMPLES.glob("*.py")}
    assert found == set(LOCAL_EXAMPLES) | OTHER_EXAMPLES


@pytest.mark.parametrize("name", sorted(LOCAL_EXAMPLES))
def test_example_runs(name: str, tmp_path: Path) -> None:
    check(run(tmp_path, str(EXAMPLES / name)), LOCAL_EXAMPLES[name])


def test_compare_devices_writes_the_figure(tmp_path: Path) -> None:
    out = tmp_path / "compare.svg"
    check(run(tmp_path, str(EXAMPLES / "compare_devices.py"), str(out)), "toy-atoms")
    assert out.read_text(encoding="utf-8").lstrip().startswith("<?xml")


def test_compare_devices_writes_to_the_current_folder_by_default(tmp_path: Path) -> None:
    check(run(tmp_path, str(EXAMPLES / "compare_devices.py")), "wrote compare_devices.svg")
    assert (tmp_path / "compare_devices.svg").is_file()
    assert not (ROOT / "assets" / "compare_devices.svg").exists()


def test_drift_exits_cleanly_offline(tmp_path: Path) -> None:
    closed_port = "http://127.0.0.1:9"
    result = run(
        tmp_path,
        str(EXAMPLES / "drift.py"),
        env={"HTTPS_PROXY": closed_port, "https_proxy": closed_port},
    )
    check(result, "skipped: could not reach IBM's public endpoint")


@pytest.mark.network
def test_drift_pulls_and_diffs(tmp_path: Path) -> None:
    check(run(tmp_path, str(EXAMPLES / "drift.py")), "device median")


@pytest.mark.skipif(not HAS_MITIQ, reason="mitiq and ply are not installed (Python <= 3.12)")
def test_mitigation_zne_runs(tmp_path: Path) -> None:
    check(run(tmp_path, str(EXAMPLES / "mitigation_zne.py")), "ZNE       P(000)")


@pytest.mark.skipif(HAS_MITIQ, reason="mitiq is installed")
def test_mitigation_zne_says_how_to_install_mitiq(tmp_path: Path) -> None:
    result = run(tmp_path, str(EXAMPLES / "mitigation_zne.py"))
    assert result.returncode == 1
    assert "pip install mitiq ply" in result.stdout


@pytest.mark.skipif(not HAS_MITIQ_CIRQ, reason="mitiq or cirq is not installed")
def test_mitiq_zne_pec_prints_the_values_the_tutorial_shows(tmp_path: Path) -> None:
    result = run(tmp_path, str(EXAMPLES / "mitiq_zne_pec.py"), "200")
    check(result, "PEC    <ZZ>")
    shown = next(
        body
        for language, body in FENCE.findall((TUTORIALS / "mitiq.md").read_text(encoding="utf-8"))
        if language == "text" and "ideal  <ZZ>" in body
    )
    shown_lines, printed = shown.splitlines(), result.stdout.splitlines()
    assert [line for line in shown_lines if not line.startswith("PEC    <ZZ>")] == [
        line for line in printed if not line.startswith("PEC    <ZZ>")
    ]
    # The tutorial shows a run with more samples than this quick run, so the PEC values agree
    # only within their standard errors.
    pec = re.compile(r"PEC    <ZZ> = ([-+.\d]+).*standard error ([.\d]+)")
    shown_pec = [m for m in map(pec.match, shown_lines) if m]
    printed_pec = [m for m in map(pec.match, printed) if m]
    assert len(shown_pec) == len(printed_pec) == 2
    for shown_match, printed_match in zip(shown_pec, printed_pec, strict=True):
        (shown_value, shown_error), (value, error) = (
            map(float, match.groups()) for match in (shown_match, printed_match)
        )
        assert abs(shown_value - value) <= 4 * math.hypot(shown_error, error)


@pytest.mark.skipif(HAS_MITIQ_CIRQ, reason="mitiq and cirq are installed")
def test_mitiq_zne_pec_says_how_to_install_mitiq(tmp_path: Path) -> None:
    result = run(tmp_path, str(EXAMPLES / "mitiq_zne_pec.py"))
    skip_if_optional_missing(result)
    assert result.returncode == 1
    assert "pip install mitiq" in result.stdout


def test_each_block_of_the_mitiq_tutorial_is_part_of_its_example() -> None:
    example = (EXAMPLES / "mitiq_zne_pec.py").read_text(encoding="utf-8")
    blocks = fenced(TUTORIALS / "mitiq.md", "python", runnable_only=False)
    assert len(blocks) >= 4
    for block in blocks:
        assert block.code in example, f"mitiq.md:{block.line} is not in examples/mitiq_zne_pec.py"


@dataclass(frozen=True)
class Block:
    doc: str
    line: int
    code: str

    @property
    def expected(self) -> list[str]:
        """Output lines a ``# `` comment after the last print shows; text before ``...`` only."""
        lines = self.code.splitlines()
        prints = [i for i, line in enumerate(lines) if "print(" in line]
        if not prints:
            return []
        shown = [line[2:] for line in lines[prints[-1] + 1 :] if line.startswith("# ")]
        return [text.split("...", 1)[0].rstrip() for text in shown if not text.startswith("...")]


def fenced(path: Path, language: str, *, runnable_only: bool = True) -> list[Block]:
    text = path.read_text(encoding="utf-8")
    blocks = []
    for match in FENCE.finditer(text):
        before = text[: match.start()].rstrip().rsplit("\n", 1)[-1]
        skipped = runnable_only and NOT_RUN.search(before)
        if match.group(1) == language and not skipped:
            line = text.count("\n", 0, match.start()) + 1
            blocks.append(Block(path.relative_to(DOCS).as_posix(), line, match.group(2)))
    return blocks


DOC_PAGES = sorted(DOCS.rglob("*.md"))
DOC_BLOCKS = [block for path in DOC_PAGES for block in fenced(path, "python")]


def test_docs_have_code() -> None:
    assert len({block.doc for block in DOC_BLOCKS}) >= 5, DOC_BLOCKS


@pytest.mark.parametrize("block", DOC_BLOCKS, ids=lambda block: f"{block.doc}:{block.line}")
def test_doc_python_block_runs_and_prints_what_it_shows(block: Block, tmp_path: Path) -> None:
    script = tmp_path / "doc.py"
    script.write_text(block.code, encoding="utf-8")
    result = run(tmp_path, str(script))
    skip_if_optional_missing(result)
    assert result.returncode == 0, result.stderr[-3000:]
    for line in block.expected:
        assert line in result.stdout, f"{line!r} not in output:\n{result.stdout[-3000:]}"


@pytest.mark.parametrize("path", DOC_PAGES, ids=lambda path: str(path.relative_to(DOCS)))
def test_doc_profiles_are_valid(path: Path) -> None:
    for block in fenced(path, "json"):
        if '"noisevault"' in block.code:
            nv.Profile.from_dict(json.loads(block.code))


def test_pin_and_cite_commands_cite_the_calibration_the_code_pins() -> None:
    text = (DOCS / "recipes.md").read_text(encoding="utf-8")
    start = text.index("## Pin and cite")
    blocks = INDENTED_FENCE.findall(text[start : text.index("\n## ", start + 1)])
    code = "".join(body for language, body in blocks if language == "python")
    ref, expect = re.search(r'nv\.load\("([^"]+)", expect="([^"]+)"\)', code).groups()
    commands = [
        line.split()[1:]
        for language, body in blocks
        if language == "bash"
        for line in body.splitlines()
        if line.strip().startswith("nv ")
    ]
    assert [args[:2] for args in commands] == [["cite", ref]] * len(commands)
    fingerprint = nv.load(ref, expect=expect).fingerprint
    for args in commands:
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
        assert fingerprint in result.output


def test_the_drift_recipe_shows_the_labels_nv_diff_prints() -> None:
    text = (DOCS / "recipes.md").read_text(encoding="utf-8")
    start = text.index("## Compare drift")
    shown = next(
        body for language, body in FENCE.findall(text[start:]) if body.startswith("ibm_fez")
    )
    title, *lines = shown.splitlines()
    assert re.fullmatch(r"ibm_fez \d{4}-\d\d-\d\d -> \d{4}-\d\d-\d\d  \(\d+ days later\)", title), (
        title
    )
    labels = {label for label, _ in METRICS.values()}
    medians = lines[1 : lines.index("")]
    qubits = lines[lines.index("largest changes by qubit") + 2 : -1]
    assert {re.split(r"  +", row)[0] for row in medians} <= labels
    assert {re.split(r"  +", row)[1] for row in qubits} <= labels
    assert len(medians) == len(METRICS) and qubits


def test_schema_file_is_current() -> None:
    expected = json.dumps(nv.json_schema(), indent=2) + "\n"
    assert SCHEMA.read_text(encoding="utf-8") == expected, (
        "docs/schema/profile-1.0.json is stale; regenerate it with:"
        " nv schema > docs/schema/profile-1.0.json"
    )
