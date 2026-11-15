from __future__ import annotations

import ast
import html
import importlib.util
import io
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from conftest import close_text
from rich.console import Console
from rich.terminal_theme import TerminalTheme
from typer.testing import CliRunner

from noisevault import catalog, cli
from noisevault.check import NOTE
from noisevault.errors import REPOSITORY

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
ASSETS = ROOT / "assets"
# optional packages that the "all" extra does not install, so NOISEVAULT_REQUIRE_ALL never demands
OUTSIDE_ALL = {"mitiq", "ply"}
FRAMEWORKS = ("qiskit", "cirq", "pennylane", "stim")

FENCE = re.compile(r"^```(\w*)[^\n]*\n(.*?)^```", re.S | re.M)
NOT_RUN = re.compile(r"<!--\s*not-run:[^>]*-->\s*$")
LINK = re.compile(r"\]\(([^)\s]+)\)|(?:src|srcset|href)=\"([^\"]+)\"")
HEADING = re.compile(r"^#{1,6} +(.+?) *$", re.M)
FRAMEWORK_VERSION = re.compile(rf"\b((?:{'|'.join(FRAMEWORKS)})[\w-]*) \d+(?:\.\d+)+[\w.+]*")


@dataclass(frozen=True)
class Block:
    line: int
    code: str

    @property
    def expected(self) -> list[str]:
        """Output lines the ``# `` comments after the first print show; text before ``...`` only."""
        lines = self.code.splitlines()
        prints = [i for i, line in enumerate(lines) if "print(" in line]
        if not prints:
            return []
        shown = [line[2:] for line in lines[prints[0] + 1 :] if line.startswith("# ")]
        return [text.split("...", 1)[0].rstrip() for text in shown if not text.startswith("...")]


def python_blocks(text: str) -> list[Block]:
    blocks = []
    for match in FENCE.finditer(text):
        before = text[: match.start()].rstrip().rsplit("\n", 1)[-1]
        if match.group(1) == "python" and not NOT_RUN.search(before):
            blocks.append(Block(text.count("\n", 0, match.start()) + 1, match.group(2)))
    return blocks


def anchors(markdown: str) -> set[str]:
    """GitHub's heading ids: lowercase, punctuation dropped, spaces to hyphens."""
    prose = FENCE.sub("", markdown)
    return {
        re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        for heading in HEADING.findall(prose)
    }


def local_targets(markdown: str) -> list[str]:
    prose = FENCE.sub("", markdown)
    targets = [a or b for a, b in LINK.findall(prose)]
    return [t for t in targets if not re.match(r"[a-z]+:", t)]


TEXT = README.read_text(encoding="utf-8")
BLOCKS = python_blocks(TEXT)


def run(tmp_path: Path, script: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "NOISEVAULT_HOME": str(tmp_path / "nv_home"), "MPLBACKEND": "Agg"}
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def skip_if_optional_missing(result: subprocess.CompletedProcess) -> None:
    missing = re.search(r"ModuleNotFoundError: No module named '(\w+)", result.stderr)
    if result.returncode == 0 or missing is None or missing.group(1) == "noisevault":
        return
    if os.environ.get("NOISEVAULT_REQUIRE_ALL") == "1" and missing.group(1) not in OUTSIDE_ALL:
        pytest.fail(f"{missing.group(1)} must be installed when NOISEVAULT_REQUIRE_ALL=1")
    pytest.skip(f"{missing.group(1)} is not installed")


def test_readme_has_python_blocks() -> None:
    assert len(BLOCKS) >= 2, BLOCKS


@pytest.mark.parametrize("block", BLOCKS, ids=lambda block: f"README.md:{block.line}")
def test_python_block_runs_and_prints_what_it_shows(block: Block, tmp_path: Path) -> None:
    script = tmp_path / "readme.py"
    script.write_text(block.code, encoding="utf-8")
    result = run(tmp_path, script)
    skip_if_optional_missing(result)
    assert result.returncode == 0, result.stderr[-3000:]
    output = unversioned(result.stdout)
    position = 0
    for line in block.expected:
        position = find_shown(output, unversioned(line), position)
        assert position >= 0, f"{line!r} not in output, or out of order:\n{result.stdout[-3000:]}"


def literal(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, TypeError, SyntaxError):
        return None


def unversioned(text: str) -> str:
    return FRAMEWORK_VERSION.sub(r"\1 *", text)


def find_shown(output: str, shown: str, start: int) -> int:
    """Where the first match of ``shown`` at or after ``start`` ends in ``output``, or -1.

    A shown dict matches an output line holding an equal dict in any order: for the same seed,
    Aer returns the same counts on Linux and macOS but in a different order.
    """
    found = output.find(shown, start)
    if found >= 0:
        return found + len(shown)
    expected = literal(shown)
    if not isinstance(expected, dict):
        return -1
    end = start
    for line in output[start:].splitlines(keepends=True):
        end += len(line)
        if literal(line.strip()) == expected:
            return end
    return -1


@pytest.mark.parametrize("target", sorted(set(local_targets(TEXT))))
def test_relative_link_resolves(target: str) -> None:
    path, _, anchor = target.partition("#")
    file = (ROOT / path) if path else README
    assert file.exists(), f"README links to {path}, which does not exist"
    if anchor:
        assert file.suffix == ".md", f"{target}: anchors are checked only in Markdown files"
        found = anchors(file.read_text(encoding="utf-8"))
        assert anchor in found, f"{target}: no heading with id {anchor!r} in {file.name}"


# bundled-profiles table ---------------------------------------------------------------------

# vendor -> (name, where its bundled data comes from); a new bundled vendor needs a row here
VENDORS: dict[str, tuple[str, str]] = {
    "ibm": (
        "IBM",
        "[qiskit-ibm-runtime](https://github.com/Qiskit/qiskit-ibm-runtime) fake backends",
    ),
    "quantinuum": (
        "Quantinuum",
        "[hardware-specifications](https://github.com/Quantinuum/quantinuum-hardware-specifications)"
        " repository",
    ),
    "google": (
        "Google",
        "[cirq-google](https://github.com/quantumlib/Cirq/tree/main/cirq-google) calibrations",
    ),
}


def bundled_profiles() -> list[catalog.ProfileInfo]:
    return [info for info in catalog.profiles() if info.location == "bundled"]


def bundled_table() -> str:
    """The README's table of bundled profiles, one row per vendor, from the bundled index."""
    bundled = bundled_profiles()
    rows = ["| Vendor | Technology | Devices | Calibrated | Source and license |"]
    rows.append("| --- | --- | --- | --- | --- |")
    by_vendor = sorted({info.vendor for info in bundled}, key=lambda v: list(VENDORS).index(v))
    for vendor in by_vendor:
        group = sorted((i for i in bundled if i.vendor == vendor), key=lambda i: i.id)
        dates = sorted(i.calibrated_at.date().isoformat() for i in group)
        span = dates[0] if dates[0] == dates[-1] else f"{dates[0]} to {dates[-1]}"
        name, source = VENDORS[vendor]
        devices = ", ".join(f"`{i.id}`" for i in group)
        technology = ", ".join(sorted({i.technology.replace("_", " ") for i in group}))
        licenses = ", ".join(sorted({i.license for i in group}))
        rows.append(
            f"| {name} ({len(group)}) | {technology} | {devices} | {span} | {source}, {licenses} |"
        )
    return "\n".join(rows)


def test_bundled_table_matches_the_catalog() -> None:
    assert bundled_table() in TEXT, (
        "the bundled-profiles table in README.md is stale; regenerate it with"
        " python tests/test_readme.py"
    )


def test_the_nav_link_counts_the_bundled_devices() -> None:
    devices = len({info.id for info in bundled_profiles()})
    assert f"[{devices} devices](#what-ships)" in TEXT


def test_the_python_versions_shown_are_the_ones_pyproject_declares() -> None:
    classifiers = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "classifiers"
    ]
    minors = sorted(
        int(c.rsplit(".", 1)[1])
        for c in classifiers
        if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", c)
    )
    oldest, newest = f"3.{minors[0]}", f"3.{minors[-1]}"
    assert f"/badge/python-{oldest}%20to%20{newest}-" in TEXT
    assert f"Python {oldest} to {newest}" in TEXT


# terminal screenshots -----------------------------------------------------------------------

# asset -> the nv command it shows
SHOTS: dict[str, tuple[str, ...]] = {"cli-show.svg": ("show", "ibm_fez")}
COLUMNS = 80
PAD = 26
FONT = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"
THEME = TerminalTheme(
    (21, 27, 35),
    (230, 237, 243),
    [
        (72, 79, 88),
        (255, 123, 114),
        (63, 185, 80),
        (210, 153, 34),
        (88, 166, 255),
        (188, 140, 255),
        (57, 197, 207),
        (177, 186, 196),
    ],
)
MUTED = "#9198a1"
FIELDS = ("styles", "lines", "backgrounds", "matrix", "terminal_width", "terminal_height")


def record(args: tuple[str, ...]) -> Console:
    """Run ``nv ARGS`` at a fixed width and return the console that recorded stdout."""
    console = Console(
        record=True,
        width=COLUMNS,
        file=io.StringIO(),
        force_terminal=True,
        color_system="truecolor",
        highlight=False,
    )
    console.print(f"[{MUTED}]$[/] nv {' '.join(args)}")
    real = cli.Console

    def factory(*a: Any, stderr: bool = False, **kw: Any) -> Console:
        return real(*a, stderr=stderr, **kw) if stderr else console

    with mock.patch.object(cli, "Console", factory):
        result = CliRunner().invoke(cli.app, list(args))
    assert result.exit_code == 0, result.output
    return console


def shown_text(console: Console) -> str:
    text = console.export_text(clear=False)
    return "\n".join(line.rstrip() for line in text.rstrip().splitlines())


def terminal_svg(args: tuple[str, ...]) -> str:
    """A dark terminal card with the command and its real output, for an <img> on GitHub.

    Rich positions every run of text with textLength, so columns line up in whatever
    monospace font the viewer has. The plain output goes in <desc> for screen readers and so
    the test can check it is current.
    """
    console = record(args)
    text = shown_text(console)
    fmt = "\x00".join(f"{{{field}}}" for field in FIELDS)
    svg = console.export_svg(theme=THEME, code_format=fmt, unique_id="nv")
    parts = dict(zip(FIELDS, svg.split("\x00"), strict=True))
    rows = text.count("\n") + 1
    width = float(parts["terminal_width"]) + 2 * PAD
    height = rows * 24.4 + 2 * PAD - 4
    title = f"Output of nv {' '.join(args)}"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'width="{width:.0f}" height="{height:.0f}" role="img" aria-label="{title}">\n'
        f"<title>{title}</title>\n<desc>{html.escape(text)}</desc>\n"
        f"<style>\n.nv-matrix {{ font-family: {FONT}; font-size: 20px; }}\n"
        f"{parts['styles']}\n</style>\n"
        f"<defs>\n{parts['lines']}\n</defs>\n"
        f'<rect x="0.5" y="0.5" width="{width - 1:.0f}" height="{height - 1:.0f}" rx="10" '
        f'fill="#151b23" stroke="#3d444d"/>\n'
        f'<g transform="translate({PAD} {PAD - 4})">\n{parts["backgrounds"]}\n'
        f'<g class="nv-matrix">\n{parts["matrix"]}\n</g>\n</g>\n</svg>\n'
    )


def svg_desc(path: Path) -> str:
    match = re.search(r"<desc>(.*?)</desc>", path.read_text(encoding="utf-8"), re.S)
    assert match, f"{path.name} has no <desc>"
    return html.unescape(match.group(1))


@pytest.mark.parametrize("name", sorted(SHOTS))
def test_terminal_screenshot_shows_current_output(name: str) -> None:
    current = shown_text(record(SHOTS[name]))
    assert svg_desc(ASSETS / name) == current, (
        f"assets/{name} is stale; regenerate it with python tests/test_readme.py"
    )


def test_the_command_above_the_screenshot_is_the_one_it_shows() -> None:
    above = re.search(r"```bash\n([^\n]+)\n```\s*<img src=\"assets/cli-show\.svg\"", TEXT)
    assert above, "README.md needs a bash block directly above the cli-show.svg screenshot"
    assert above[1].endswith(f" nv {' '.join(SHOTS['cli-show.svg'])}"), above[1]


def test_the_pin_example_cites_the_calibration_its_pull_saves() -> None:
    saved = re.search(r"^nv pull \S+ .*# saves (\S+)@(\d{4}-\d\d-\d\d)T", TEXT, re.M)
    cited = re.search(r"^nv cite (\S+)@(\S+) ", TEXT, re.M)
    compared = re.search(r"^nv diff \S+ (\S+)@(\S+)$", TEXT, re.M)
    assert saved and cited and compared, "README.md's pin example must say what nv pull saves"
    assert cited.groups() == compared.groups() == saved.groups()


def test_hero_images_match_their_grid() -> None:
    build = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_hero.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr


CHECK = ("check", "ibm_fez")
CHECK_BLOCK = re.compile(r"^```text\n\$ nv check ibm_fez\n.*?^```\n", re.S | re.M)
ROUNDOFF = 1e-12


def check_output() -> str:
    """``nv check ibm_fez`` through its last result row, as the README shows it."""
    lines = shown_text(record(CHECK)).splitlines()
    last_row = max(i for i, line in enumerate(lines) if line.split(" ", 1)[0] in FRAMEWORKS)
    return "\n".join(lines[: last_row + 1])


def same_check(shown: str, current: str) -> bool:
    shown_lines, current_lines = shown.splitlines(), current.splitlines()
    return len(shown_lines) == len(current_lines) and all(
        same_line(a, b) for a, b in zip(shown_lines, current_lines, strict=True)
    )


def same_line(shown: str, current: str) -> bool:
    """Exact rows differ in roundoff between library versions. Sampled rows differ between
    platforms for the same seed (Stim's sampling depends on the CPU's SIMD width).
    """
    if shown == current:
        return True
    a, b = shown.split(), current.split()
    if a[:1] != b[:1] or a[0] not in FRAMEWORKS or a[:2] + a[4:] != b[:2] + b[4:]:
        return False
    try:
        pairs = [(float(x), float(y)) for x, y in zip(a[2:4], b[2:4], strict=True)]
    except ValueError:
        return False
    return all(max(x, y) < ROUNDOFF or max(x, y) <= 10 * min(x, y) for x, y in pairs)


def test_the_readme_says_what_a_pass_means_in_the_words_nv_check_prints() -> None:
    assert NOTE in " ".join(TEXT.split())


def test_check_output_matches_a_current_run() -> None:
    """The README shows ``nv check ibm_fez`` through its last result row."""
    missing = [name for name in FRAMEWORKS if importlib.util.find_spec(name) is None]
    if missing and os.environ.get("NOISEVAULT_REQUIRE_ALL") == "1":
        pytest.fail(f"{missing} must be installed when NOISEVAULT_REQUIRE_ALL=1")
    if missing:
        pytest.skip(f"nv check skips {missing}, so its output differs from the README")
    block = CHECK_BLOCK.search(TEXT)
    assert block, "README.md has no ```text block starting with $ nv check ibm_fez"
    shown = block.group(0).removeprefix("```text\n").removesuffix("```\n").rstrip("\n")
    current = check_output()
    assert same_check(shown, current), (
        "README.md must show this nv check output; regenerate it with"
        f" python tests/test_readme.py:\n{current}"
    )


COMPARE_BLOCK = re.compile(
    r"^```bash\ncurl -O (?P<url>\S+)\n```\s*^```text\n\$ nv (?P<command>compare [^\n]+)\n.*?^```\n",
    re.S | re.M,
)
EXAMPLE_COUNTS = "examples/kingston-simulated.counts.json"


def compare_block() -> re.Match[str]:
    block = COMPARE_BLOCK.search(TEXT)
    assert block, "README.md needs a curl -O block, then a ```text block starting with $ nv compare"
    return block


def compare_output(command: str, url: str) -> str:
    owner_and_name = REPOSITORY.removeprefix("https://github.com/")
    assert url == f"https://raw.githubusercontent.com/{owner_and_name}/main/{EXAMPLE_COUNTS}", url
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as folder:
        shutil.copy(ROOT / EXAMPLE_COUNTS, Path(folder) / url.rsplit("/", 1)[1])
        os.chdir(folder)
        try:
            result = CliRunner().invoke(
                cli.app, shlex.split(command), env={"COLUMNS": str(COLUMNS)}
            )
        finally:
            os.chdir(cwd)
    assert result.exit_code == 0, result.output
    return f"$ nv {command}\n" + re.sub(r"\x1b\[[0-9;]*m", "", result.stdout).rstrip("\n")


def test_compare_output_matches_a_run_on_the_downloaded_example() -> None:
    block = compare_block()
    shown = block.group(0)[block.group(0).index("```text\n") + 8 :].removesuffix("```\n")
    current = compare_output(block["command"], block["url"])
    assert close_text(shown, current), (
        "README.md must show this nv compare output; regenerate it with"
        f" python tests/test_readme.py:\n{current}"
    )


if __name__ == "__main__":
    os.environ["NOISEVAULT_HOME"] = tempfile.mkdtemp()  # an empty vault, as in the tests
    for name, args in SHOTS.items():
        (ASSETS / name).write_text(terminal_svg(args), encoding="utf-8")
        print(f"wrote assets/{name}")
    text = CHECK_BLOCK.sub(lambda _: f"```text\n{check_output()}\n```\n", TEXT, count=1)
    block = COMPARE_BLOCK.search(text)
    assert block, "README.md has no nv compare block to regenerate"
    start = block.start() + block.group(0).index("```text\n")
    shown = f"```text\n{compare_output(block['command'], block['url'])}\n```\n"
    text = text[:start] + shown + text[block.end() :]
    README.write_text(text, encoding="utf-8")
    print("wrote the nv check and nv compare output in README.md")
    print("\nbundled-profiles table for README.md:\n")
    print(bundled_table())
