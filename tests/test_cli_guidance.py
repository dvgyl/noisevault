from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from typer.testing import CliRunner

import noisevault as nv
from noisevault.catalog import vault_path
from noisevault.cli import app
from noisevault.errors import install_hint
from noisevault.sources import ibm_public, ionq

runner = CliRunner()


def _error_and_hint(args: list[str], code: int = 1) -> str:
    result = runner.invoke(app, args, env={"COLUMNS": "80"}, prog_name="nv")
    assert result.exit_code == code and result.stdout == ""
    first, *rest = result.stderr.splitlines()
    assert first.startswith("error: ") and len(rest) <= 1, result.stderr
    assert all(line.startswith("hint: ") for line in rest), result.stderr
    return result.stderr


def test_pull_of_a_mistyped_ibm_device_names_the_close_one(monkeypatch) -> None:
    listing = json.dumps([{"name": "ibm_fez"}, {"name": "ibm_kingston"}]).encode()

    def fetch(url: str) -> bytes:
        if url == ibm_public.BASE_URL:
            return listing
        raise ibm_public._NotFound(url)

    monkeypatch.setattr(ibm_public, "fetch", fetch)
    assert "did you mean 'ibm_fez'?" in _error_and_hint(["pull", "ibm_fezz"])


def test_pull_of_a_mistyped_ionq_device_names_the_close_one(monkeypatch) -> None:
    backends = [{"backend": "qpu.forte-1"}, {"backend": "qpu.aria-1"}, {"backend": "simulator"}]
    monkeypatch.setattr(ionq, "_get", lambda url: json.dumps(backends).encode())
    with pytest.raises(nv.SourceUnavailable, match=r"did you mean 'qpu\.forte-1'\?"):
        ionq.pull("ionq_forte-11")


def _refuse(handler: object, request: urllib.request.Request) -> None:
    raise urllib.error.URLError("timed out")


def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", _refuse)


def _without_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", None)


def _only_the_listing(url: str) -> bytes:
    if url == ibm_public.BASE_URL:
        return json.dumps([{"name": "ibm_fez"}]).encode()
    raise ibm_public._NotFound(url)


def _unlisted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ibm_public, "fetch", _only_the_listing)


@pytest.mark.parametrize(
    ("setup", "args", "error", "hint"),
    [
        (
            _offline,
            ["pull", "ibm_fez"],
            "could not reach IBM's public endpoint (timed out)",
            "check the network connection, or run nv list to see every profile you can load"
            " offline",
        ),
        (
            _offline,
            ["pull", "ionq_forte-1"],
            "could not reach IonQ's API (timed out)",
            "check the network connection, or run nv list to see every profile you can load"
            " offline",
        ),
        (
            _without_runtime,
            ["pull", "ibm_fez", "--source", "ibm-account"],
            "--source ibm-account needs qiskit-ibm-runtime",
            install_hint("ibm"),
        ),
        (
            _unlisted,
            ["pull", "ibm_torino"],
            "ibm_torino is not listed on the public endpoint (the endpoint lists ibm_fez);"
            " the device may be retired",
            "ibm_torino is bundled, so nv show ibm_torino loads it offline",
        ),
    ],
)
def test_a_failed_pull_puts_its_next_step_on_a_hint_line(
    monkeypatch: pytest.MonkeyPatch, setup, args: list[str], error: str, hint: str
) -> None:
    setup(monkeypatch)
    assert _error_and_hint(args) == f"error: {error}\nhint: {hint}\n"


_NO_GOOGLE_SOURCE = (
    "error: unknown source '{source}'. No source serves google devices\n"
    "hint: run nv list --vendor google to see the google profiles you can load offline\n"
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "ibmm",
            "unknown source 'ibmm'; did you mean 'ibm'? choose one of ibm, ibm-account, ionq",
        ),
        ("IBM-Account", None),
        ("googel", _NO_GOOGLE_SOURCE.format(source="googel")),
        ("google", _NO_GOOGLE_SOURCE.format(source="google")),
        ("xyz", "unknown source 'xyz'; choose one of ibm, ibm-account, ionq\n"),
    ],
)
def test_pull_with_a_mistyped_source_says_what_to_type(monkeypatch, source, expected) -> None:
    monkeypatch.setattr(
        "noisevault.sources.ibm_account.pull",
        lambda device, at=None: (_ for _ in ()).throw(nv.SourceUnavailable("reached the account")),
    )
    error = _error_and_hint(["pull", "ibm_fez", "--source", source])
    assert (expected or "reached the account") in error


def _same_time_pair(vault: Path) -> list[Path]:
    manila = nv.load("ibm_manila").to_dict()
    paths = []
    for name, error in (("first", 5e-4), ("second", 6e-4)):
        manila["gates"]["sx"]["avg_infidelity"] = error
        paths.append(nv.Profile.model_validate(manila).save(vault / f"{name}.json.gz"))
    return paths


def test_an_ambiguous_ref_on_the_command_line_says_what_to_type(vault: Path) -> None:
    vault.mkdir(parents=True)
    first, second = _same_time_pair(vault)
    error = _error_and_hint(["show", "ibm_manila@2024-05-27T18:27:23Z"])
    assert "expect=" not in error
    assert f"\nhint: give one of their files: {first}, {second}\n" in error
    assert runner.invoke(app, ["show", str(first)]).exit_code == 0


def test_an_ambiguous_ref_in_python_keeps_the_python_fix(vault: Path) -> None:
    vault.mkdir(parents=True)
    _same_time_pair(vault)
    with pytest.raises(nv.AmbiguousRef, match="expect='nv:...'"):
        nv.load("ibm_manila@2024-05-27T18:27:23Z")


@pytest.mark.parametrize(
    ("args", "error"),
    [
        (["--bogus"], "error: no such option '--bogus'\nhint: run nv --help\n"),
        (["--vresion"], "error: no such option '--vresion'; did you mean '--version'?\n"),
        (["--bogus", "list"], "error: no such option '--bogus'\nhint: run nv --help\n"),
    ],
)
def test_a_mistyped_top_level_option_is_one_error_and_at_most_one_hint(
    args: list[str], error: str
) -> None:
    assert _error_and_hint(args, code=2) == error


def test_list_and_doctor_name_a_dangling_vault_link_in_one_line(vault: Path) -> None:
    toy = nv.load("ibm_manila")
    toy.save(vault_path(toy))
    (vault / "gone.json.gz").symlink_to(vault.parent / "moved.json.gz")
    expected = (
        f"warning: skipped {vault / 'gone.json.gz'}: it links to"
        f" {vault.parent / 'moved.json.gz'}, which does not exist. Remove the link\n"
    )
    listed = runner.invoke(app, ["list"], env={"COLUMNS": "200"})
    assert listed.exit_code == 0 and "* ibm_manila" in listed.stdout
    assert listed.stderr == expected
    doctor = runner.invoke(app, ["doctor"], env={"COLUMNS": "200"})
    assert doctor.exit_code == 0 and f"vault: {vault} (1 profile)" in doctor.stdout
    assert doctor.stderr == expected
