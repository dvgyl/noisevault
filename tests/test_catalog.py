from __future__ import annotations

import errno
import gzip
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import warnings
import zlib
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from conftest import MANILA_V01, deeper_than_the_parser_takes, migrated, require, toy
from typer.testing import CliRunner

import noisevault as nv
from noisevault import catalog
from noisevault.catalog import bundled_profiles, vault_dir, vault_path
from noisevault.cli import app
from noisevault.errors import AmbiguousRef, FingerprintMismatch, ProfileNotFound, SourceUnavailable
from noisevault.profile import Profile

FRAMEWORKS = {
    "qiskit",
    "qiskit_aer",
    "qiskit_ibm_runtime",
    "cirq",
    "cirq_google",
    "pennylane",
    "stim",
}


def _dated(stamp: str | None, error: float = 1e-3) -> Profile:
    data = toy(
        device={
            "name": "toy",
            "vendor": "test",
            "technology": "superconducting",
            "num_qubits": 3,
            "calibrated_at": stamp,
        }
    )
    data["gates"]["sx"]["avg_infidelity"] = error
    return Profile.model_validate(data)


def test_bundled_manila_loads_by_id_and_date() -> None:
    by_id = nv.load("ibm_manila")
    assert by_id.id == "ibm_manila"
    assert nv.load("ibm_manila@2024-05-27").fingerprint == by_id.fingerprint
    assert nv.load("ibm_manila@2024-05-27T18:27:23Z").fingerprint == by_id.fingerprint
    old = migrated(MANILA_V01)  # the 0.1 snapshot of the same calibration keeps its numbers
    for record in old.calibrations:
        now = by_id.table.gate(record.gate, record.qubits)
        assert now.avg_infidelity == record.avg_infidelity
        assert now.duration_ns == pytest.approx(record.duration_ns, rel=1e-11)
    assert [q.readout.pair for q in by_id.qubits] == [q.readout.pair for q in old.qubits]
    [info] = [i for i in bundled_profiles() if i.id == "ibm_manila"]
    assert info.fingerprint == by_id.fingerprint and info.location == "bundled"


def test_load_by_path(tmp_path: Path) -> None:
    path = _dated("2025-01-01T00:00:00Z").save(tmp_path / "mine.json.gz")
    assert nv.load(path).id == "test_toy"
    assert nv.load(str(path)).id == "test_toy"
    with pytest.raises(ProfileNotFound):
        nv.load(tmp_path / "missing.json")


def test_expect_pins_the_fingerprint() -> None:
    profile = nv.load("ibm_manila")
    assert nv.load("ibm_manila", expect=profile.short_fingerprint) == profile
    assert nv.load("ibm_manila", expect="sha256:" + profile.fingerprint) == profile
    other = "0" * 64
    with pytest.raises(FingerprintMismatch, match=profile.short_fingerprint):
        nv.load("ibm_manila", expect=other)
    with pytest.raises(FingerprintMismatch):
        nv.load("ibm_manila", expect="nv:000000000000")
    with pytest.raises(ValueError) as info:
        nv.load("ibm_manila", expect="nv:abc")
    assert isinstance(info.value, nv.NoiseVaultError)
    assert (info.value.message, info.value.hint) == (
        "expect='nv:abc' is not a fingerprint",
        "pass a full sha256 fingerprint or nv:<12 hex>",
    )


def test_vault_refs_newest_date_and_ambiguity(vault: Path) -> None:
    for stamp, error in (
        ("2025-01-01T08:00:00Z", 1e-3),
        ("2025-01-01T20:00:00Z", 2e-3),
        ("2025-02-01T08:00:00Z", 3e-3),
    ):
        _dated(stamp, error).save(vault_path(_dated(stamp, error)))
    assert vault_dir() == vault
    assert nv.load("test_toy").gates["sx"].avg_infidelity == 3e-3
    assert nv.load("test_toy@2025-02-01").gates["sx"].avg_infidelity == 3e-3
    assert nv.load("test_toy@2025-01-01T20:00:00Z").gates["sx"].avg_infidelity == 2e-3
    with pytest.raises(AmbiguousRef) as info:
        nv.load("test_toy@2025-01-01")
    assert "2025-01-01T08:00:00Z" in info.value.message
    assert "2025-01-01T20:00:00Z" in info.value.message
    assert info.value.hint == "use a full timestamp"
    with pytest.raises(ProfileNotFound, match="2024-12-31 UTC. You have test_toy@2025-01-01T08"):
        nv.load("test_toy@2024-12-31")


def test_a_ref_date_that_misses_says_it_names_the_calibration_day() -> None:
    misses = {
        "ibm_fez@2025-03-01": (
            "no ibm_fez profile calibrated on 2025-03-01 UTC."
            " You have ibm_fez@2025-02-26T20:16:25Z",
            "run nv pull ibm_fez --at 2025-03-01T23:59:59Z to download the calibration in effect"
            " at the end of that day. Then load the ref that nv pull prints",
        ),
        "quantinuum_h2-1@2020-01-01": (
            "no quantinuum_h2-1 profile calibrated on 2020-01-01 UTC."
            " You have quantinuum_h2-1@2025-04-30T00:00:00Z",
            None,
        ),
        "ibm_fez@2025-02-26T00:00:00Z": (
            "no ibm_fez profile calibrated at 2025-02-26T00:00:00Z."
            " You have ibm_fez@2025-02-26T20:16:25Z",
            None,
        ),
    }
    for ref, (message, hint) in misses.items():
        with pytest.raises(ProfileNotFound) as info:
            nv.load(ref)
        assert (info.value.message, info.value.hint) == (message, hint)
        assert str(info.value) == (f"{message}; {hint}" if hint else message)


def test_an_unknown_id_says_where_to_see_the_ids() -> None:
    with pytest.raises(ProfileNotFound) as info:
        nv.load("xyzzy")
    assert info.value.message == "no profile with id 'xyzzy'"
    assert info.value.hint == "run nv list to see every profile you can load offline"


def test_unknown_id_suggests_close_matches() -> None:
    with pytest.raises(ProfileNotFound) as info:
        nv.load("ibm_manilla")
    assert info.value.message == (
        "no profile with id 'ibm_manilla'; did you mean 'ibm_manila', 'ibm_miami'?"
    )


def test_an_id_that_also_names_a_folder_here_loads_the_id(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ibm_manila").mkdir()
    assert nv.load("ibm_manila") == nv.load("ibm_manila@2024-05-27")
    assert nv.catalog.resolve("ibm_manila").location == "bundled"


def test_pulling_a_bundled_device_no_source_serves_says_how_to_load_it() -> None:
    with pytest.raises(SourceUnavailable) as info:
        nv.pull("google_weber")
    assert info.value.message == "no source pulls 'google_weber'"
    assert info.value.hint == "google_weber is bundled, so nv.load('google_weber') loads it offline"
    assert str(info.value).endswith(f"; {info.value.hint}")


def test_pulling_from_a_vendor_no_source_serves_says_how_to_list_its_profiles() -> None:
    with pytest.raises(ValueError) as info:
        nv.pull("ibm_fez", source="google")
    assert isinstance(info.value, nv.NoiseVaultError)
    assert (info.value.message, info.value.hint) == (
        "unknown source 'google'. No source serves google devices",
        "run nv list --vendor google to see the google profiles you can load offline",
    )


def test_vault_copy_of_a_bundled_profile_is_not_ambiguous(vault: Path) -> None:
    manila = nv.load("ibm_manila")
    manila.save(vault_path(manila))
    assert nv.load("ibm_manila@2024-05-27") == manila
    infos = [i for i in nv.profiles() if i.id == "ibm_manila"]
    assert [i.location for i in infos] == ["vault"]


def test_profiles_filters(vault: Path) -> None:
    _dated("2025-01-01T00:00:00Z").save(vault / "toy.json")
    assert {i.id for i in nv.profiles()} >= {"ibm_manila", "test_toy"}
    assert {i.id for i in nv.profiles(vendor="test")} == {"test_toy"}
    ions = nv.profiles(technology="trapped_ion")
    assert {i.technology for i in ions} <= {"trapped_ion"}
    assert not {"ibm_manila", "test_toy"} & {i.id for i in ions}


def test_unreadable_vault_file_is_skipped_with_a_warning(vault: Path) -> None:
    vault.mkdir(parents=True)
    (vault / "broken.json").write_text("{}")
    with pytest.warns(UserWarning, match="broken.json"):
        assert "ibm_manila" in {i.id for i in nv.profiles()}


def test_import_loads_no_framework() -> None:
    code = (
        "import sys, noisevault as nv; nv.load('ibm_manila').summary();"
        f" print(sorted(m for m in sys.modules if m.split('.')[0] in {sorted(FRAMEWORKS)!r}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_stim_helpers_are_reachable_as_nv_stim() -> None:
    require("stim")
    import noisevault.frameworks.stim as stim_module

    assert nv.stim is stim_module


def _changed_manila(error: float) -> Profile:
    data = nv.load("ibm_manila").to_dict()
    data["gates"]["sx"] = {**data["gates"]["sx"], "avg_infidelity": error}
    return Profile.model_validate(data)


def test_vault_profile_shadows_a_bundled_one_of_the_same_time(vault: Path) -> None:
    bundled = nv.load("ibm_manila")
    mine = _changed_manila(5e-4)
    mine.save(vault_path(mine))
    for ref in ("ibm_manila", "ibm_manila@2024-05-27", "ibm_manila@2024-05-27T18:27:23Z"):
        assert nv.load(ref).fingerprint == mine.fingerprint, ref
    assert nv.load("ibm_manila", expect=bundled.short_fingerprint) == bundled
    with pytest.raises(FingerprintMismatch) as info:
        nv.load("ibm_manila", expect="nv:000000000000")
    assert info.value.message == (
        f"ibm_manila loads ibm_manila@2024-05-27T18:27:23Z ({mine.short_fingerprint}), not the"
        " expected nv:000000000000, and no profile you have has that fingerprint"
    )


def test_a_vault_profile_shadows_only_a_bundled_one_of_its_id_and_time() -> None:
    (manila,) = [i for i in bundled_profiles() if i.id == "ibm_manila"]
    mine, lima = replace(manila, location="vault"), replace(manila, id="ibm_lima")
    undated = replace(manila, calibrated_at=None)
    assert catalog._unshadowed([mine, manila, lima, undated]) == [mine, lima, undated]


def _later_manila() -> Profile:
    data = _changed_manila(5e-4).to_dict()
    data["device"]["calibrated_at"] = "2024-06-03T10:00:00Z"
    return Profile.model_validate(data)


def _load_as_hint_says(hint: str | None) -> Profile:
    assert hint is not None
    target = re.search(r"load[ (]'?([^\s',]+)", hint)
    pin = re.search(r"expect='([^']+)'", hint)
    assert target is not None, hint
    return nv.load(target.group(1), expect=pin.group(1) if pin else None)


def _save_vault_copy_and_later_manila() -> Profile:
    data = nv.load("ibm_manila").to_dict()
    data["qubits"][0]["t1_us"] = 99.0
    mine = Profile.model_validate(data)
    later = _later_manila()
    for profile in (mine, later):
        profile.save(vault_path(profile))
    return mine


def test_following_a_mismatch_hint_loads_the_pinned_profile_past_a_vault_copy(
    vault: Path,
) -> None:
    bundled = nv.load("ibm_manila")
    mine = _save_vault_copy_and_later_manila()
    ref, pin = "ibm_manila@2024-05-27T18:27:23Z", bundled.short_fingerprint
    assert nv.load(ref).fingerprint == mine.fingerprint
    with pytest.raises(FingerprintMismatch) as info:
        nv.load("ibm_manila", expect=pin)
    assert _load_as_hint_says(info.value.hint).fingerprint == bundled.fingerprint
    hint = f"nv.load('{ref}', expect='{pin}') loads the profile with that fingerprint"
    assert info.value.hint == hint
    assert str(info.value) == f"{info.value.message}; {hint}"


def test_a_mismatch_hint_names_the_file_of_an_undated_profile(vault: Path) -> None:
    undated, dated = _dated(None), _dated("2025-01-01T00:00:00Z")
    path = undated.save(vault_path(undated))
    dated.save(vault_path(dated))
    pin = undated.short_fingerprint
    with pytest.raises(FingerprintMismatch) as info:
        nv.load("test_toy@2025-01-01", expect=pin)
    assert _load_as_hint_says(info.value.hint).fingerprint == undated.fingerprint
    assert info.value.hint == (
        f"nv.load('{path}', expect='{pin}') loads the profile with that fingerprint"
    )


def test_a_missed_date_lists_each_ref_once_past_a_vault_copy(vault: Path) -> None:
    _save_vault_copy_and_later_manila()
    error = (
        "no ibm_manila profile calibrated on 2025-01-01 UTC."
        " You have ibm_manila@2024-05-27T18:27:23Z, ibm_manila@2024-06-03T10:00:00Z"
    )
    with pytest.raises(ProfileNotFound) as info:
        nv.load("ibm_manila@2025-01-01")
    assert info.value.message == error
    result = CliRunner().invoke(app, ["show", "ibm_manila@2025-01-01"])
    assert result.exit_code == 1
    assert result.stderr.splitlines()[0] == f"error: {error}"


def test_a_fingerprint_mismatch_names_the_calibration_that_has_the_pin(vault: Path) -> None:
    bundled = nv.load("ibm_manila")
    later = _later_manila()
    path = later.save(vault_path(later))
    pin, held = bundled.short_fingerprint, later.short_fingerprint
    heads = {
        "ibm_manila": f"ibm_manila loads ibm_manila@2024-06-03T10:00:00Z ({held})",
        "ibm_manila@2024-06-03": f"ibm_manila@2024-06-03 is {held}",
        str(path): f"{path} holds ibm_manila@2024-06-03T10:00:00Z ({held})",
    }
    hint = (
        f"nv.load('ibm_manila@2024-05-27T18:27:23Z', expect='{pin}') loads the profile with that"
        " fingerprint"
    )
    for ref, head in heads.items():
        with pytest.raises(FingerprintMismatch) as info:
            nv.load(ref, expect=pin)
        assert info.value.message == f"{head}, not the expected {pin}"
        assert info.value.hint == hint and str(info.value).endswith(f"; {hint}")
    assert nv.load("ibm_manila@2024-05-27T18:27:23Z", expect=pin) == bundled


def test_a_fingerprint_no_profile_has_says_how_to_get_the_file() -> None:
    nowhere, held = "nv:000000000000", nv.load("ibm_manila").short_fingerprint
    with pytest.raises(FingerprintMismatch) as info:
        nv.load("ibm_manila", expect=nowhere)
    assert info.value.message == (
        f"ibm_manila loads ibm_manila@2024-05-27T18:27:23Z ({held}), not the expected"
        f" {nowhere}, and no profile you have has that fingerprint"
    )
    assert info.value.hint == (
        "ask whoever pinned that fingerprint for the profile file. If the source still serves"
        " that calibration, run nv pull ibm_manila --at <a time it was in effect>"
    )
    with pytest.raises(FingerprintMismatch) as info:
        nv.load("quantinuum_h2-1", expect=nowhere)
    assert info.value.hint == "ask whoever pinned that fingerprint for the profile file"


def test_same_time_profiles_in_the_vault_are_told_apart_by_expect(vault: Path) -> None:
    first, second = _changed_manila(5e-4), _changed_manila(6e-4)
    first.save(vault / "first.json.gz")
    second.save(vault / "second.json.gz")
    with pytest.raises(AmbiguousRef) as info:
        nv.load("ibm_manila@2024-05-27T18:27:23Z")
    assert info.value.message.startswith("ibm_manila matches 2 profiles that share a calibration")
    assert info.value.hint == (
        "pass expect='nv:...' or load one of their files:"
        f" {vault / 'first.json.gz'}, {vault / 'second.json.gz'}"
    )
    assert nv.load("ibm_manila", expect=second.fingerprint) == second


def test_profiles_are_sorted_by_id_then_date(vault: Path) -> None:
    for stamp in ("2025-02-01T08:00:00Z", "2025-01-01T08:00:00Z"):
        _dated(stamp).save(vault_path(_dated(stamp)))
    infos = nv.profiles()
    assert [i.id for i in infos] == sorted(i.id for i in infos)
    toys = [i.ref for i in infos if i.id == "test_toy"]
    assert toys == ["test_toy@2025-01-01T08:00:00Z", "test_toy@2025-02-01T08:00:00Z"]


def test_vault_listing_parses_each_file_once(vault: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import noisevault.catalog as catalog

    first = _dated("2025-01-01T00:00:00Z")
    path = first.save(vault_path(first))
    parsed: list[Path] = []
    real = catalog.load_file
    monkeypatch.setattr(catalog, "load_file", lambda p: parsed.append(p) or real(p))
    assert [i.fingerprint for i in catalog.vault_profiles()] == [first.fingerprint]
    assert [i.fingerprint for i in catalog.vault_profiles()] == [first.fingerprint]
    assert parsed == [path]  # the second listing came from the index
    changed = _dated("2025-01-01T00:00:00Z", error=4.25e-3)  # another size, too
    changed.save(path)
    assert [i.fingerprint for i in catalog.vault_profiles()] == [changed.fingerprint]
    assert parsed == [path, path]


def test_listing_carries_data_kind_and_license() -> None:
    [info] = [i for i in nv.profiles() if i.id == "ibm_manila"]
    assert (info.data_kind, info.license) == ("measured", "Apache-2.0")


def test_readme_fingerprint_pin_loads() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    pins = re.findall(r'nv\.load\("([^"]+)", expect="(nv:[0-9a-f]{12})"\)', readme)
    assert pins, "README no longer shows a pinned load"
    for ref, expect in pins:
        assert nv.load(ref, expect=expect).short_fingerprint == expect


def test_dot_files_in_the_vault_are_not_profiles(vault: Path) -> None:
    vault.mkdir(parents=True)
    (vault / ".index.json.4242.tmp").write_text("{\x01")  # a cache write cut short
    (vault / ".ibm_x@2025.json.gz.4242.tmp.json.gz").write_bytes(b"\x1f\x8b")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert "ibm_manila" in {i.id for i in nv.profiles()}


def _serve(monkeypatch: pytest.MonkeyPatch, profile: Profile) -> None:
    from noisevault.sources import ibm_public

    monkeypatch.setattr(ibm_public, "pull", lambda device, at=None: profile)


def _serve_manila_calibrated_at(monkeypatch: pytest.MonkeyPatch, *stamps: str) -> None:
    from noisevault.sources import ibm_public

    def pull(device: str, at: datetime | None = None) -> Profile:
        older = [s for s in stamps if at is None or datetime.fromisoformat(s) < at]
        data = nv.load("ibm_manila").to_dict()
        data["device"]["calibrated_at"] = max(older, key=datetime.fromisoformat)
        return Profile.model_validate(data)

    monkeypatch.setattr(ibm_public, "pull", pull)


@pytest.mark.parametrize(
    ("history", "printed"),
    [
        (("2024-05-30T04:56:23Z", "2024-06-01T19:00:00Z"), "ibm_manila@2024-06-01T19:00:00Z"),
        (("2024-05-30T04:56:23Z",), "ibm_manila@2024-05-30T04:56:23Z"),
    ],
    ids=["calibrated-that-day", "calibrated-before"],
)
def test_following_the_hint_of_a_missed_date_ends_with_a_profile_that_loads(
    monkeypatch: pytest.MonkeyPatch, history: tuple[str, ...], printed: str
) -> None:
    _serve_manila_calibrated_at(monkeypatch, *history)
    runner = CliRunner()
    asked = ["show", "ibm_manila@2024-06-01"]
    missed = runner.invoke(app, asked)
    assert missed.exit_code == 1
    hint = missed.stderr.splitlines()[-1]
    pull = re.search(r"run nv (pull \S+ --at \S+)", hint)
    assert pull is not None, hint
    assert runner.invoke(app, pull.group(1).split()).stdout.split()[0] == printed
    assert hint.endswith(". Then load the ref that nv pull prints")
    assert runner.invoke(app, ["show", printed]).exit_code == 0
    assert runner.invoke(app, asked).exit_code == (0 if "2024-06-01" in printed else 1)


def _pull_quietly(**options) -> catalog.Pulled:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return catalog.pull_and_save("ibm_toy", source="ibm", **options)


def test_pulls_a_fraction_of_a_second_apart_keep_both_calibrations(
    monkeypatch: pytest.MonkeyPatch, vault: Path
) -> None:
    first = _dated("2024-05-27T18:27:23.100000Z", 1e-3)
    second = _dated("2024-05-27T18:27:23.900000Z", 2e-3)
    for profile in (first, second):
        _serve(monkeypatch, profile)
        assert _pull_quietly().written
    assert len(list(vault.glob("*.json.gz"))) == 2
    assert nv.load("test_toy@2024-05-27T18:27:23.1Z") == first
    assert nv.load("test_toy@2024-05-27T18:27:23.9Z") == second


def test_vault_files_named_by_whole_seconds_still_load_and_are_reused(
    monkeypatch: pytest.MonkeyPatch, vault: Path
) -> None:
    held = _dated("2024-05-27T18:27:23.100000Z", 1e-3)
    legacy = held.save(vault / "test_toy@2024-05-27T182723Z.json.gz")
    assert [i.ref for i in nv.profiles() if i.id == "test_toy"] == [
        "test_toy@2024-05-27T18:27:23.100000Z"
    ]
    _serve(monkeypatch, held)
    assert _pull_quietly() == (held, legacy, False)
    reconverted = _dated("2024-05-27T18:27:23.100000Z", 1.5e-3)
    _serve(monkeypatch, reconverted)
    with pytest.warns(UserWarning, match="imports the same calibration differently"):
        pulled = catalog.pull_and_save("ibm_toy", source="ibm")
    assert pulled.path == legacy and list(vault.glob("*.json.gz")) == [legacy]
    assert nv.load(legacy) == reconverted


def test_a_pull_never_replaces_a_vault_file_of_another_calibration(
    monkeypatch: pytest.MonkeyPatch, vault: Path
) -> None:
    held = _dated("2024-05-27T18:27:23.100000Z", 1e-3)
    legacy = held.save(vault / "test_toy@2024-05-27T182723Z.json.gz")
    _serve(monkeypatch, _dated("2024-05-27T18:27:23Z", 2e-3))
    with pytest.raises(FileExistsError) as info:
        _pull_quietly()
    assert isinstance(info.value, nv.NoiseVaultError)
    error = f"{legacy} already holds test_toy@2024-05-27T18:27:23.100000Z, another calibration"
    hint = f"move that file out of {vault} and pull again"
    assert (info.value.message, info.value.hint) == (error, hint)
    result = CliRunner().invoke(app, ["pull", "ibm_toy"])
    assert result.exit_code == 1 and result.stderr == f"error: {error}\nhint: {hint}\n"
    assert nv.load(legacy) == held


def test_a_failed_pull_to_a_file_leaves_the_existing_file_whole(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resource = pytest.importorskip("resource")
    target = _dated("2025-01-01T00:00:00Z").save(tmp_path / "existing.json.gz")
    before = target.read_bytes()
    newer = _dated("2025-02-01T00:00:00Z")
    _serve(monkeypatch, newer)
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    resource.setrlimit(resource.RLIMIT_FSIZE, (20, hard))
    try:
        with pytest.raises(OSError, match="File too large"):
            _pull_quietly(output=target)
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
    assert target.read_bytes() == before
    assert list(tmp_path.iterdir()) == [target]
    assert _pull_quietly(output=target) == (newer, target, True)
    assert nv.load(target) == newer and target.read_bytes()[:2] == b"\x1f\x8b"


@pytest.mark.parametrize(
    "damage",
    [
        lambda entry: [],
        lambda entry: "stale",
        lambda entry: {k: v for k, v in entry.items() if k != "id"},
        lambda entry: {**entry, "calibrated_at": 5},
        lambda entry: {**entry, "id": []},
        lambda entry: {**entry, "num_qubits": "3"},
        lambda entry: {**entry, "num_qubits": True},
        lambda entry: {**entry, "fingerprint": None},
        lambda entry: {**entry, "fingerprint": "nv:1234"},
        lambda entry: {**entry, "vendor": 7},
        lambda entry: {**entry, "license": ["MIT"]},
        lambda entry: {**entry, "calibrated_at": "2025-01-01T00:00:00"},
    ],
    ids=[
        "list",
        "string",
        "missing id",
        "bad timestamp",
        "list id",
        "string qubits",
        "bool qubits",
        "no fingerprint",
        "short fingerprint",
        "int vendor",
        "list license",
        "naive timestamp",
    ],
)
def test_a_damaged_index_entry_is_rebuilt_from_its_file(vault: Path, damage) -> None:
    profile = _dated("2025-01-01T00:00:00Z")
    path = profile.save(vault_path(profile))
    catalog.vault_profiles()
    index = vault / ".index.json"
    data = json.loads(index.read_text())
    entry = data["entries"][path.name]
    index.write_text(json.dumps({**data, "entries": {path.name: damage(entry)}}))
    assert catalog.vault_profiles() == [catalog.ProfileInfo.of(profile, "vault", path)]
    assert nv.load("test_toy") == profile
    assert nv.load("ibm_manila").id == "ibm_manila"
    assert _indexed(vault)[path.name] == entry


def _indexed(vault: Path) -> dict[str, dict[str, object]]:
    return json.loads((vault / ".index.json").read_text())["entries"]


def _misindex(vault: Path, path: Path, **wrong: object) -> dict[str, object]:
    catalog.vault_profiles()
    index = vault / ".index.json"
    data = json.loads(index.read_text())
    entry = data["entries"][path.name]
    data["entries"][path.name] = {**entry, **wrong}
    index.write_text(json.dumps(data))
    return entry


@pytest.mark.parametrize(
    "foreign",
    [
        lambda data, entries: entries,
        lambda data, entries: {
            **data,
            "version": 999,
            "digest": catalog._digest(entries),
            "entries": entries,
        },
        lambda data, entries: {**data, "entries": entries},
    ],
    ids=["no version", "unknown version", "stale digest"],
)
def test_an_index_this_noisevault_did_not_write_is_ignored_and_rewritten(
    monkeypatch: pytest.MonkeyPatch, vault: Path, foreign
) -> None:
    profile = _dated("2025-01-01T00:00:00Z")
    path = profile.save(vault_path(profile))
    catalog.vault_profiles()
    index = vault / ".index.json"
    data = json.loads(index.read_text())
    entry = data["entries"][path.name]
    index.write_text(json.dumps(foreign(data, {path.name: {**entry, "id": "ibm_mistyped"}})))
    writes: list[Path] = []
    real = catalog.write_atomically
    monkeypatch.setattr(catalog, "write_atomically", lambda p, d: writes.append(p) or real(p, d))
    assert [i.id for i in catalog.vault_profiles()] == ["test_toy"]
    data = json.loads(index.read_text())
    assert data["entries"] == {path.name: entry}
    canonical = json.dumps(data["entries"], sort_keys=True, separators=(",", ":"))
    assert (data["version"], data["digest"]) == (
        catalog._INDEX_VERSION,
        hashlib.sha256(canonical.encode()).hexdigest(),
    )
    assert [i.id for i in catalog.vault_profiles()] == ["test_toy"]
    assert writes == [index]


def test_an_index_with_a_key_twice_is_ignored_and_rewritten(vault: Path) -> None:
    profile = _dated("2025-01-01T00:00:00Z")
    path = profile.save(vault_path(profile))
    catalog.vault_profiles()
    index = vault / ".index.json"
    data = json.loads(index.read_text())
    entry = data["entries"][path.name]
    mistyped = {path.name: {**entry, "id": "ibm_mistyped"}}
    text = json.dumps({**data, "digest": catalog._digest(mistyped), "entries": mistyped})
    index.write_text(text.replace('"id": "ibm_mistyped"', '"id": "test_toy", "id": "ibm_mistyped"'))
    assert [i.id for i in catalog.vault_profiles()] == ["test_toy"]
    assert json.loads(index.read_text())["entries"] == {path.name: entry}


def test_a_bundled_index_with_a_key_twice_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "index.json").write_text('{"profiles": [], "profiles": []}')
    monkeypatch.setattr(catalog, "bundled_dir", lambda: tmp_path)
    with pytest.raises(ValueError, match=r"^the key profiles appears twice$"):
        bundled_profiles()


def test_what_an_index_entry_records_changes_only_with_the_index_version() -> None:
    profile = Profile.model_validate(
        toy(
            device={
                "name": "toy",
                "vendor": "test",
                "technology": "superconducting",
                "num_qubits": 3,
                "processor": "Falcon r5.11",
                "calibrated_at": "2025-01-01T00:00:00Z",
            },
            provenance={
                "data_kind": "measured",
                "source_kind": "package_snapshot",
                "license": "Apache-2.0",
                "attribution": "test",
                "redistributable": "yes",
            },
        )
    )
    assert (catalog._INDEX_VERSION, catalog.index_entry(profile)) == (
        1,
        {
            "id": "test_toy",
            "date": "2025-01-01",
            "calibrated_at": "2025-01-01T00:00:00Z",
            "vendor": "test",
            "technology": "superconducting",
            "num_qubits": 3,
            "processor": "Falcon r5.11",
            "data_kind": "measured",
            "source_kind": "package_snapshot",
            "license": "Apache-2.0",
            "redistributable": "yes",
            "fingerprint": profile.fingerprint,
        },
    )


@pytest.mark.parametrize(
    "damaged",
    [
        '"num_qubits": 1e309',
        '"num_qubits": NaN',
        '"num_qubits": -Infinity',
        '"num_qubits": "\\ud800"',
        '"\\udfff": 5',
        "deep nesting",
    ],
    ids=["1e309", "NaN", "-Infinity", "lone surrogate", "lone surrogate key", "deep nesting"],
)
def test_an_index_that_cannot_be_hashed_is_ignored_and_rewritten(vault: Path, damaged: str) -> None:
    if damaged == "deep nesting":
        damaged = '"num_qubits": ' + deeper_than_the_parser_takes()
    manila = nv.load("ibm_manila")
    manila.save(vault / "current.json.gz")
    catalog.vault_profiles()
    index = vault / ".index.json"
    written = index.read_text()
    index.write_text(written.replace('"num_qubits": 5', damaged, 1))
    runner = CliRunner()
    for args in (["list"], ["show", "ibm_manila"]):
        result = runner.invoke(app, args)
        assert (result.exit_code, result.stderr) == (0, ""), args
    assert index.read_text() == written


def test_a_pipe_named_like_the_index_does_not_block_the_listing(vault: Path) -> None:
    profile = _dated("2025-01-01T00:00:00Z")
    profile.save(vault_path(profile))
    pipe = vault / ".index.json"
    os.mkfifo(pipe)
    listed: list[set[str]] = []
    reader = threading.Thread(target=lambda: listed.append({i.id for i in nv.profiles()}))
    reader.start()
    reader.join(timeout=5)
    blocked = reader.is_alive()
    if blocked:
        os.close(os.open(pipe, os.O_WRONLY | os.O_NONBLOCK))
        reader.join()
    assert not blocked, "listing the vault waited on the pipe"
    assert "test_toy" in listed[0]
    assert pipe.is_file()


def test_an_index_entry_time_is_read_as_utc() -> None:
    (manila,) = [i for i in bundled_profiles() if i.id == "ibm_manila"]
    entry = {**catalog.index_entry(manila.load()), "calibrated_at": "2024-05-28T01:27:23+07:00"}
    info = catalog.ProfileInfo.from_entry(entry, "bundled", manila.path)
    assert info.ref == "ibm_manila@2024-05-27T18:27:23Z"
    assert info.calibrated_at.date().isoformat() == "2024-05-27"


def test_an_index_time_with_an_offset_does_not_bypass_utc_day_matching(vault: Path) -> None:
    manila = nv.load("ibm_manila")
    path = manila.save(vault_path(manila))
    entry = _misindex(vault, path, calibrated_at="2024-05-28T01:27:23+07:00")
    for expect in (None, manila.short_fingerprint):
        with pytest.raises(ProfileNotFound) as info:
            nv.load("ibm_manila@2024-05-28", expect=expect)
        assert info.value.message == (
            "no ibm_manila profile calibrated on 2024-05-28 UTC."
            " You have ibm_manila@2024-05-27T18:27:23Z"
        )
    assert _indexed(vault)[path.name] == entry


def test_a_misindexed_id_hides_a_vault_profile_from_neither_load_nor_pull(
    monkeypatch: pytest.MonkeyPatch, vault: Path
) -> None:
    later = _later_manila()
    path = later.save(vault_path(later))
    entry = _misindex(vault, path, id="ibm_mistyped")
    assert nv.load("ibm_manila") == later
    _serve(monkeypatch, later)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert catalog.pull_and_save("ibm_manila", source="ibm") == (later, path, False)
    assert _indexed(vault)[path.name] == entry


def test_a_stale_index_date_does_not_bypass_dated_ref_matching(vault: Path) -> None:
    manila = nv.load("ibm_manila")
    path = manila.save(vault_path(manila))
    entry = _misindex(vault, path, calibrated_at="2024-05-28T18:27:23Z")
    with pytest.raises(ProfileNotFound) as info:
        nv.load("ibm_manila@2024-05-28")
    assert info.value.message == (
        "no ibm_manila profile calibrated on 2024-05-28 UTC."
        " You have ibm_manila@2024-05-27T18:27:23Z"
    )
    with pytest.raises(ProfileNotFound):
        nv.load("ibm_manila@2024-05-28", expect=manila.short_fingerprint)
    assert _indexed(vault)[path.name] == entry


def test_a_stale_index_date_does_not_hide_a_vault_profile(vault: Path) -> None:
    profile = _dated("2025-01-01T00:00:00Z")
    path = profile.save(vault_path(profile))
    entry = _misindex(vault, path, calibrated_at="2025-01-02T00:00:00Z")
    assert nv.load("test_toy@2025-01-01") == profile
    assert _indexed(vault)[path.name] == entry


def test_a_stale_index_time_does_not_make_a_ref_ambiguous(vault: Path) -> None:
    morning = _dated("2025-01-01T08:00:00Z")
    evening = _dated("2025-01-01T20:00:00Z", error=2e-3)
    path = morning.save(vault_path(morning))
    evening.save(vault_path(evening))
    _misindex(vault, path, calibrated_at="2025-01-01T20:00:00Z")
    assert nv.load("test_toy@2025-01-01T20:00:00Z") == evening


def test_a_stale_index_fingerprint_does_not_refuse_a_correct_pin(vault: Path) -> None:
    profile = _dated("2025-01-01T00:00:00Z")
    path = profile.save(vault_path(profile))
    _misindex(vault, path, fingerprint="ab" * 32)
    assert nv.load("test_toy", expect=profile.short_fingerprint) == profile
    assert [i.fingerprint for i in nv.profiles() if i.id == "test_toy"] == [profile.fingerprint]


def test_a_load_corrects_the_stale_index_fields_nv_list_shows(vault: Path) -> None:
    profile = _dated("2025-01-01T00:00:00Z")
    path = profile.save(vault_path(profile))
    entry = _misindex(vault, path, vendor="ionq", num_qubits=27, license="CC0-1.0")
    assert nv.load("test_toy") == profile
    assert [i for i in nv.profiles() if i.id == "test_toy"] == [
        catalog.ProfileInfo.of(profile, "vault", path)
    ]
    assert _indexed(vault)[path.name] == entry


@pytest.mark.parametrize("wrong", ["calibrated_at", "fingerprint"])
def test_a_pull_does_not_trust_a_stale_index_entry(
    monkeypatch: pytest.MonkeyPatch, vault: Path, wrong: str
) -> None:
    first = _dated("2025-01-01T00:00:00Z")
    path = first.save(vault_path(first))
    second = _dated("2025-01-02T00:00:00Z", error=2e-3)
    served = {"calibrated_at": "2025-01-02T00:00:00Z", "fingerprint": second.fingerprint}
    _misindex(vault, path, **{wrong: served[wrong]})
    _serve(monkeypatch, second)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pulled = catalog.pull_and_save("ibm_toy", source="ibm")
    assert (pulled.path, pulled.written) == (vault_path(second), True)
    assert nv.load("test_toy@2025-01-01") == first
    assert nv.load("test_toy@2025-01-02") == second


def _saved_while_loading(
    monkeypatch: pytest.MonkeyPatch, path: Path, saved: Profile, then: Profile | None = None
) -> None:
    real = catalog.ProfileInfo.load

    def load(self: catalog.ProfileInfo) -> Profile:
        if self.path != path:
            return real(self)
        saved.save(path)
        try:
            return real(self)
        finally:
            if then is not None:
                then.save(path)

    monkeypatch.setattr(catalog.ProfileInfo, "load", load)


def _january_manila() -> Profile:
    data = nv.load("ibm_manila").to_dict()
    data["device"]["calibrated_at"] = "2025-01-01T00:00:00Z"
    data["qubits"][0]["t1_us"] = 99.0
    return Profile.model_validate(data)


@pytest.mark.parametrize(
    ("ref", "pinned", "saved"),
    [
        ("ibm_manila@2024-05-27", False, _january_manila),
        ("ibm_manila@2024-05-27T18:27:23Z", False, _january_manila),
        ("ibm_manila@2024-05-27", True, _january_manila),
        ("ibm_manila@2024-05-27", True, lambda: _changed_manila(5e-4)),
    ],
    ids=["date", "timestamp", "pinned date", "pinned, converted again"],
)
def test_a_load_answers_its_ref_and_pin_when_the_file_it_resolved_is_saved_over(
    monkeypatch: pytest.MonkeyPatch, vault: Path, ref: str, pinned: bool, saved
) -> None:
    may = nv.load("ibm_manila")
    current = may.save(vault / "current.json.gz")
    catalog.vault_profiles()
    _saved_while_loading(monkeypatch, current, saved())
    loaded = nv.load(ref, expect=may.short_fingerprint if pinned else None)
    assert (loaded.device.calibrated_at.isoformat(), loaded.qubits[0].t1_us) == (
        "2024-05-27T18:27:23+00:00",
        may.qubits[0].t1_us,
    )
    assert loaded == may


def test_a_vault_file_saved_over_after_it_resolved_leaves_its_day_not_found(
    monkeypatch: pytest.MonkeyPatch, vault: Path
) -> None:
    first = _dated("2025-01-01T00:00:00Z")
    later = _dated("2025-02-01T00:00:00Z", error=2e-3)
    path = first.save(vault / "current.json.gz")
    catalog.vault_profiles()
    _saved_while_loading(monkeypatch, path, later)
    with pytest.raises(ProfileNotFound) as info:
        nv.load("test_toy@2025-01-01")
    assert info.value.message == (
        "no test_toy profile calibrated on 2025-01-01 UTC. You have test_toy@2025-02-01T00:00:00Z"
    )


def test_an_undated_load_answers_the_newest_profile_when_the_file_it_resolved_is_saved_over(
    monkeypatch: pytest.MonkeyPatch, vault: Path
) -> None:
    may, june = nv.load("ibm_manila"), _later_manila()
    current = _january_manila().save(vault / "current.json.gz")
    june.save(vault_path(june))
    catalog.vault_profiles()
    _saved_while_loading(monkeypatch, current, may)
    loaded = nv.load("ibm_manila")
    assert loaded.device.calibrated_at.isoformat() == "2024-06-03T10:00:00+00:00"
    assert loaded == june


@pytest.mark.parametrize("ref", ["test_toy@2025-01-01", "test_toy"])
def test_a_file_that_changes_under_both_reads_of_a_load_is_not_found(
    monkeypatch: pytest.MonkeyPatch, vault: Path, ref: str
) -> None:
    first = _dated("2025-01-01T00:00:00Z")
    later = _dated("2025-02-01T00:00:00Z", error=2e-3)
    path = first.save(vault / "current.json.gz")
    catalog.vault_profiles()
    _saved_while_loading(monkeypatch, path, later, then=first)
    with pytest.raises(ProfileNotFound) as info:
        nv.load(ref)
    assert (info.value.message, info.value.hint) == (
        f"{path} holds test_toy@2025-02-01T00:00:00Z, not test_toy@2025-01-01T00:00:00Z,"
        " because the file changed during this load",
        f"load {ref} again",
    )


def test_a_load_rereads_the_files_when_the_index_misdescribes_an_unchanged_one(
    vault: Path,
) -> None:
    may = nv.load("ibm_manila")
    current = _january_manila().save(vault / "current.json.gz")
    catalog.vault_profiles()
    entry = _indexed(vault)[current.name]
    claimed = {**entry, **catalog.index_entry(may)}
    catalog._write_vault_index(vault, {current.name: claimed})
    assert nv.load("ibm_manila@2024-05-27") == may
    assert _indexed(vault) == {current.name: entry}


@pytest.mark.parametrize(("ref", "picked"), [("test_toy@2025-01-03", 2), ("test_toy", -1)])
def test_a_load_through_the_index_reads_only_the_file_it_resolves(
    monkeypatch: pytest.MonkeyPatch, vault: Path, ref: str, picked: int
) -> None:
    days = [_dated(f"2025-01-0{day}T00:00:00Z", error=day * 1e-4) for day in range(1, 6)]
    for profile in days:
        profile.save(vault_path(profile))
    catalog.vault_profiles()
    reads: list[Path] = []
    real = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: reads.append(self) or real(self))
    assert nv.load(ref) == days[picked]
    indexes = (vault / ".index.json", catalog.bundled_dir() / "index.json")
    assert [p for p in reads if p not in indexes] == [vault_path(days[picked])]


def test_odd_vault_entries_are_skipped_with_one_line_each(vault: Path) -> None:
    profile = _dated("2025-01-01T00:00:00Z")
    profile.save(vault_path(profile))
    (vault / "gone.json.gz").symlink_to(vault.parent / "moved_away.json.gz")
    (vault / "folder.json").mkdir()
    os.mkfifo(vault / "pipe.json")
    (vault / "empty.json").write_text("{}")
    with pytest.warns(nv.NoiseVaultWarning) as caught:
        found = {i.id for i in nv.profiles()}
        assert nv.load("test_toy") == profile
    assert {"test_toy", "ibm_manila"} <= found
    messages = sorted({str(w.message) for w in caught})
    assert [m.split(":")[0] for m in messages] == [
        f"skipped {vault / name}"
        for name in ("empty.json", "folder.json", "gone.json.gz", "pipe.json")
    ]
    assert all("\n" not in m for m in messages)
    assert f"links to {vault.parent / 'moved_away.json.gz'}, which does not exist" in messages[2]
    assert messages[0].endswith(f". Run nv validate {vault / 'empty.json'}")


def test_the_command_for_a_skipped_entry_runs_as_printed_for_a_name_with_a_space(
    vault: Path,
) -> None:
    vault.mkdir(parents=True)
    damaged = vault / "my run.json"
    damaged.write_text("{}")
    with pytest.warns(nv.NoiseVaultWarning) as caught:
        nv.profiles()
    (message,) = {str(w.message) for w in caught}
    assert message.endswith(f". Run nv validate '{damaged}'")
    nv_, *args = shlex.split(message.rpartition(". Run ")[2])
    result = CliRunner().invoke(app, args)
    assert (nv_, result.exit_code, result.stdout) == ("nv", 1, "")
    assert result.stderr.startswith("error: noisevault: missing. Format 1.0 requires this key\n")


def test_the_command_for_a_skipped_entry_runs_as_printed_for_a_path_that_starts_with_a_hyphen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NOISEVAULT_HOME", "-nv")
    Path("-nv/profiles").mkdir(parents=True)
    Path("-nv/profiles/empty.json").write_text("{}")
    with pytest.warns(nv.NoiseVaultWarning) as caught:
        nv.profiles()
    (message,) = {str(w.message) for w in caught}
    assert message.endswith(". Run nv validate -- -nv/profiles/empty.json")
    nv_, *args = shlex.split(message.rpartition(". Run ")[2])
    result = CliRunner().invoke(app, args)
    assert (nv_, result.exit_code, result.stdout) == ("nv", 1, "")
    assert result.stderr.startswith("error: noisevault: missing. Format 1.0 requires this key\n")


def test_a_link_to_an_unreadable_file_is_skipped_and_the_rest_still_list(
    monkeypatch: pytest.MonkeyPatch, vault: Path
) -> None:
    profile = _dated("2025-01-01T00:00:00Z")
    profile.save(vault_path(profile))
    hidden = vault.parent / "locked" / "toy.json.gz"
    hidden.parent.mkdir()
    profile.save(hidden)
    link = vault / "locked.json.gz"
    link.symlink_to(hidden)
    real_stat = Path.stat

    def denied(self: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if self == link and follow_symlinks:
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(self))
        return real_stat(self, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", denied)
    with pytest.warns(nv.NoiseVaultWarning) as caught:
        assert nv.load("ibm_manila").id == "ibm_manila"
        assert nv.load("test_toy") == profile
    assert {str(w.message) for w in caught} == {f"skipped {link}: {os.strerror(errno.EACCES)}"}

    result = CliRunner().invoke(app, ["list"], env={"COLUMNS": "120"})
    assert result.exit_code == 0, result.output
    assert "test_toy" in result.stdout and "ibm_manila" in result.stdout
    assert f"warning: skipped {link}: {os.strerror(errno.EACCES)}" in result.stderr


def _lock(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    """What ``chmod a-r`` does to an indexed file, also as root: the inode change time moves
    (size and modification time stay) and reading it fails."""
    before = path.stat().st_ctime_ns
    deadline = time.monotonic() + 5
    while path.stat().st_ctime_ns == before:  # file systems with a coarse clock need a retry
        assert time.monotonic() < deadline, "chmod never moved the change time"
        time.sleep(0.01)
        path.chmod(path.stat().st_mode)
    real_read, real_open = Path.read_bytes, Path.open

    def refuse(self: Path) -> None:
        if self == path:
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(self))

    def denied_read(self: Path) -> bytes:
        refuse(self)
        return real_read(self)

    def denied_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        refuse(self)
        return real_open(self, *args, **kwargs)

    # The loaders read a file through one of these two methods.
    monkeypatch.setattr(Path, "read_bytes", denied_read)
    monkeypatch.setattr(Path, "open", denied_open)


def test_an_indexed_file_made_unreadable_is_skipped_and_the_rest_still_list(
    monkeypatch: pytest.MonkeyPatch, vault: Path
) -> None:
    profile = _dated("2025-01-01T00:00:00Z")
    path = profile.save(vault_path(profile))
    catalog.vault_profiles()
    _lock(monkeypatch, path)
    skipped = f"skipped {path}: {os.strerror(errno.EACCES)}"
    with pytest.warns(nv.NoiseVaultWarning) as caught:
        found = {i.id for i in nv.profiles()}
    assert "ibm_manila" in found and "test_toy" not in found
    assert {str(w.message) for w in caught} == {skipped}

    result = CliRunner().invoke(app, ["list"], env={"COLUMNS": "120"})
    assert result.exit_code == 0, result.output
    assert "ibm_manila" in result.stdout and "test_toy" not in result.stdout
    assert f"warning: {skipped}" in result.stderr


def test_a_pull_onto_an_unreadable_vault_file_does_not_call_it_saved(
    monkeypatch: pytest.MonkeyPatch, vault: Path
) -> None:
    profile = _dated("2025-01-01T00:00:00Z")
    path = profile.save(vault_path(profile))
    before = path.read_bytes()
    catalog.vault_profiles()
    _serve(monkeypatch, profile)
    _lock(monkeypatch, path)
    with pytest.warns(nv.NoiseVaultWarning, match="skipped"):
        with pytest.raises(FileExistsError) as info:
            catalog.pull_and_save("ibm_toy", source="ibm")
    assert isinstance(info.value, nv.NoiseVaultError)
    assert (info.value.message, info.value.hint) == (
        f"{path} exists but is not readable",
        f"make the file readable or move the file out of {vault}, then pull again",
    )
    monkeypatch.undo()
    assert path.read_bytes() == before


def test_listing_reads_no_profile_once_the_vault_is_indexed(
    monkeypatch: pytest.MonkeyPatch, vault: Path
) -> None:
    profile = _dated("2025-01-01T00:00:00Z")
    profile.save(vault_path(profile))
    catalog.vault_profiles()
    manila = nv.load("ibm_manila")
    expected = {
        "test_toy": (None, None, "unknown"),
        "ibm_manila": ("Falcon r5.11", "package_snapshot", "yes"),
    }
    assert manila.device.processor == "Falcon r5.11"

    def no_parse(*args: object) -> None:
        raise AssertionError("listing parsed a profile file")

    monkeypatch.setattr(catalog, "load_file", no_parse)
    monkeypatch.setattr(catalog, "load_bytes", no_parse)
    runner = CliRunner()
    assert runner.invoke(app, ["list"]).exit_code == 0
    result = runner.invoke(app, ["list", "--json"])
    assert result.exit_code == 0, result.output
    rows = {row["id"]: row for row in json.loads(result.stdout)}
    for name, values in expected.items():
        row = rows[name]
        assert (row["processor"], row["source_kind"], row["redistributable"]) == values
    assert rows["ibm_manila"]["calibrated_at"] == "2024-05-27T18:27:23Z"


def test_vault_files_have_one_gzip_header_on_every_python(
    monkeypatch: pytest.MonkeyPatch, vault: Path
) -> None:
    """Before Python 3.13, gzip.compress(mtime=0) let zlib write its platform's OS byte."""
    monkeypatch.setattr(
        gzip, "compress", lambda data, level=9, *, mtime=None: zlib.compress(data, level, wbits=31)
    )
    profile = _dated("2025-01-01T00:00:00Z")
    raw = profile.save(vault_path(profile)).read_bytes()
    assert raw[:10] == b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"  # no time, no name, OS 255
    assert nv.load("test_toy") == profile


def test_concurrent_pulls_of_one_calibration_both_succeed_and_one_writes(
    monkeypatch: pytest.MonkeyPatch, vault: Path
) -> None:
    profile = _dated("2025-01-01T00:00:00Z")
    _serve(monkeypatch, profile)
    vault.mkdir(parents=True)
    both_listed = threading.Barrier(2, timeout=10)
    real_listing = catalog.vault_profiles
    failures: list[BaseException] = []
    written: list[bool] = []

    def pull() -> None:
        try:
            written.append(catalog.pull_and_save("ibm_toy", source="ibm").written)
        except BaseException as exc:
            failures.append(exc)

    threads = [threading.Thread(target=pull) for _ in range(2)]
    listing = set(threads)

    def list_then_wait_for_the_other_pull(*, reread: bool = False) -> list[catalog.ProfileInfo]:
        listed = real_listing(reread=reread)
        if threading.current_thread() in listing:
            listing.discard(threading.current_thread())
            both_listed.wait()
        return listed

    monkeypatch.setattr(catalog, "vault_profiles", list_then_wait_for_the_other_pull)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == []
    assert sorted(written) == [False, True]
    assert [p.name for p in vault.iterdir() if p.name != ".index.json"] == [
        vault_path(profile).name
    ]
    assert nv.load("test_toy") == profile


@pytest.fixture(params=["hard links", "no hard links", "exFAT"])
def links(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.param == "hard links":
        return

    def no_link(*args: object, **kwargs: object) -> None:
        raise OSError(errno.ENOTSUP, os.strerror(errno.ENOTSUP))

    monkeypatch.setattr(os, "link", no_link)
    if request.param == "exFAT":
        for name in ("fstat", "lstat"):
            monkeypatch.setattr(os, name, _numbered_at_first_write(getattr(os, name)))


def _numbered_at_first_write(real: Callable[..., os.stat_result]) -> Callable[..., os.stat_result]:
    def numbered(*args: Any, **kwargs: Any) -> os.stat_result:
        result = real(*args, **kwargs)
        if result.st_size:
            return result
        fields = list(result)
        fields[1] = -1 - result.st_ino
        return os.stat_result(fields)

    return numbered


def _saved_while_pulling(monkeypatch: pytest.MonkeyPatch, save: Callable[[Path], object]) -> Path:
    manila = nv.load("ibm_manila")
    _serve(monkeypatch, manila)
    path = vault_path(manila)
    real_open = os.open
    pending = [save]

    def open_after_the_other_writer(file: str | Path, *args: Any, **kwargs: Any) -> int:
        if pending and Path(file).name.startswith(f".{path.name}."):
            pending.pop()(path)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(os, "open", open_after_the_other_writer)
    return path


@pytest.mark.usefixtures("links")
def test_a_pull_never_replaces_a_calibration_saved_while_it_writes(
    monkeypatch: pytest.MonkeyPatch, vault: Path
) -> None:
    later = _later_manila()
    path = _saved_while_pulling(monkeypatch, later.save)
    with pytest.raises(FileExistsError) as info:
        catalog.pull_and_save("ibm_manila", source="ibm")
    assert isinstance(info.value, nv.NoiseVaultError)
    assert (info.value.message, info.value.hint) == (
        f"{path} already holds ibm_manila@2024-06-03T10:00:00Z, another calibration",
        f"move that file out of {vault} and pull again",
    )
    assert nv.load(path) == later
    assert [p.name for p in vault.iterdir() if p.name != ".index.json"] == [path.name]


@pytest.mark.usefixtures("links")
def test_a_pull_writes_nothing_when_the_same_calibration_is_saved_while_it_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manila = nv.load("ibm_manila")
    path = _saved_while_pulling(monkeypatch, manila.save)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert catalog.pull_and_save("ibm_manila", source="ibm") == (manila, path, False)
    assert nv.load(path) == manila


@pytest.mark.usefixtures("links")
def test_a_pull_replaces_an_other_import_of_its_calibration_saved_while_it_writes_and_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manila, other = nv.load("ibm_manila"), _changed_manila(5e-4)
    path = _saved_while_pulling(monkeypatch, other.save)
    replaced = rf"replaced {re.escape(path.name)} \({other.short_fingerprint}\) with this pull"
    with pytest.warns(nv.NoiseVaultWarning, match=replaced):
        assert catalog.pull_and_save("ibm_manila", source="ibm") == (manila, path, True)
    assert nv.load(path) == manila


@pytest.mark.usefixtures("links")
def test_a_pull_refuses_an_unreadable_file_saved_while_it_writes(
    monkeypatch: pytest.MonkeyPatch, vault: Path
) -> None:
    path = _saved_while_pulling(monkeypatch, lambda path: path.write_bytes(b"not a profile"))
    with pytest.raises(FileExistsError) as info:
        catalog.pull_and_save("ibm_manila", source="ibm")
    assert (info.value.message, info.value.hint) == (
        f"{path} exists but is not readable",
        f"make the file readable or move the file out of {vault}, then pull again",
    )
    assert path.read_bytes() == b"not a profile"


def _saved_when_the_pull_moves(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    before: Callable[[Path], object],
    after: Callable[[Path], object] = lambda path: None,
) -> None:
    pending = [(before, after)]
    for name in ("rename", "replace"):
        real = getattr(os, name)

        def moving(src: Any, dst: Any, *args: Any, real: Any = real, **kwargs: Any) -> None:
            if pending and path in (Path(src), Path(dst)):
                first, then = pending.pop()
                first(path)
                real(src, dst, *args, **kwargs)
                then(path)
            else:
                real(src, dst, *args, **kwargs)

        monkeypatch.setattr(os, name, moving)


@pytest.mark.usefixtures("links")
@pytest.mark.parametrize("moment", ["after the listing", "as the pull moves the file"])
def test_a_pull_never_replaces_another_calibration_saved_over_the_import_it_replaces(
    monkeypatch: pytest.MonkeyPatch, vault: Path, moment: str
) -> None:
    manila, later = nv.load("ibm_manila"), _later_manila()
    _serve(monkeypatch, manila)
    path = _changed_manila(5e-4).save(vault_path(manila))
    if moment == "after the listing":
        listing = catalog.vault_profiles

        def list_then_save(*, reread: bool = False) -> list[catalog.ProfileInfo]:
            listed = listing(reread=reread)
            later.save(path)
            return listed

        monkeypatch.setattr(catalog, "vault_profiles", list_then_save)
    else:
        _saved_when_the_pull_moves(monkeypatch, path, later.save)
    with warnings.catch_warnings(record=True) as caught, pytest.raises(FileExistsError) as info:
        warnings.simplefilter("always")
        catalog.pull_and_save("ibm_manila", source="ibm")
    assert (info.value.message, info.value.hint, caught) == (
        f"{path} already holds ibm_manila@2024-06-03T10:00:00Z, another calibration",
        f"move that file out of {vault} and pull again",
        [],
    )
    assert nv.load(path) == later
    assert [p.name for p in vault.iterdir() if p.name != ".index.json"] == [path.name]


@pytest.mark.usefixtures("links")
def test_the_replace_warning_names_the_import_that_the_pull_replaced(
    monkeypatch: pytest.MonkeyPatch, vault: Path
) -> None:
    manila, saved = nv.load("ibm_manila"), _changed_manila(7e-4)
    _serve(monkeypatch, manila)
    path = _changed_manila(5e-4).save(vault_path(manila))
    _saved_when_the_pull_moves(monkeypatch, path, saved.save)
    with pytest.warns(nv.NoiseVaultWarning) as caught:
        assert catalog.pull_and_save("ibm_manila", source="ibm") == (manila, path, True)
    assert [str(w.message) for w in caught] == [
        f"replaced {path.name} ({saved.short_fingerprint}) with this pull"
        f" ({manila.short_fingerprint}), which imports the same calibration differently."
        " Update any expect= pins"
    ]
    assert nv.load(path) == manila
    assert [p.name for p in vault.iterdir() if p.name != ".index.json"] == [path.name]


@pytest.mark.usefixtures("links")
def test_a_pull_keeps_every_file_when_its_vault_path_changes_twice(
    monkeypatch: pytest.MonkeyPatch, vault: Path
) -> None:
    manila, later, third = nv.load("ibm_manila"), _later_manila(), _changed_manila(7e-4)
    _serve(monkeypatch, manila)
    path = _changed_manila(5e-4).save(vault_path(manila))
    _saved_when_the_pull_moves(monkeypatch, path, later.save, third.save)
    with pytest.raises(FileExistsError) as info:
        catalog.pull_and_save("ibm_manila", source="ibm")
    [moved] = [p for p in vault.iterdir() if p.name.startswith(f".{path.name}.")]
    assert (info.value.message, info.value.hint) == (
        f"{path} changed during the pull. The pull saved nothing and moved the file that was"
        f" there to {moved.name}",
        f"move {moved.name} out of {vault}, then pull again",
    )
    assert (nv.load(moved), nv.load(path)) == (later, third)


def _moved_aside(vault: Path, path: Path) -> list[Path]:
    return [p for p in vault.iterdir() if p.name.startswith(f".{path.name}.")]


@pytest.mark.parametrize(
    ("links", "put_back"),
    [("hard links", True), ("no hard links", False), ("exFAT", False)],
    indirect=["links"],
)
def test_a_pull_that_cannot_write_its_replacement_keeps_the_import_it_moved(
    monkeypatch: pytest.MonkeyPatch, vault: Path, links: None, put_back: bool
) -> None:
    manila, old = nv.load("ibm_manila"), _changed_manila(5e-4)
    _serve(monkeypatch, manila)
    path = old.save(vault_path(manila))
    real_open = os.open
    full: list[bool] = []

    def no_space_once_moved(file: str | Path, *args: Any, **kwargs: Any) -> int:
        if full and Path(file).parent == vault:
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
        return real_open(file, *args, **kwargs)

    _saved_when_the_pull_moves(monkeypatch, path, lambda path: None, lambda path: full.append(True))
    monkeypatch.setattr(os, "open", no_space_once_moved)
    with pytest.raises(OSError) as info:
        catalog.pull_and_save("ibm_manila", source="ibm")
    if put_back:
        assert (str(info.value), _moved_aside(vault, path)) == (
            "[Errno 28] No space left on device",
            [],
        )
        assert nv.load(path) == old
    else:
        [moved] = _moved_aside(vault, path)
        assert isinstance(info.value, nv.NoiseVaultError)
        assert (info.value.message, info.value.hint, path.exists()) == (
            "[Errno 28] No space left on device. The pull saved nothing and moved the file that"
            f" was there to {moved.name}",
            f"move {moved.name} out of {vault}, then pull again",
            False,
        )
        assert nv.load(moved) == old


@pytest.mark.usefixtures("links")
def test_a_pull_keeps_the_import_it_moved_when_another_calibration_takes_its_path(
    monkeypatch: pytest.MonkeyPatch, vault: Path
) -> None:
    manila, old, later = nv.load("ibm_manila"), _changed_manila(5e-4), _later_manila()
    _serve(monkeypatch, manila)
    path = old.save(vault_path(manila))
    _saved_when_the_pull_moves(monkeypatch, path, lambda path: None, later.save)
    with pytest.raises(FileExistsError) as info:
        catalog.pull_and_save("ibm_manila", source="ibm")
    [moved] = _moved_aside(vault, path)
    assert (info.value.message, info.value.hint) == (
        f"{path} already holds ibm_manila@2024-06-03T10:00:00Z, another calibration. The pull"
        f" saved nothing and moved the file that was there to {moved.name}",
        f"move {moved.name} out of {vault}, then pull again",
    )
    assert (nv.load(moved), nv.load(path)) == (old, later)


def _changed_after_the_listing(
    monkeypatch: pytest.MonkeyPatch, change: Callable[[], object]
) -> None:
    listing = catalog.vault_profiles

    def list_then_change(*, reread: bool = False) -> list[catalog.ProfileInfo]:
        listed = listing(reread=reread)
        change()
        return listed

    monkeypatch.setattr(catalog, "vault_profiles", list_then_change)


def test_a_pull_reads_the_listed_copy_of_its_profile_again_before_it_reports_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manila, later = nv.load("ibm_manila"), _later_manila()
    _serve(monkeypatch, manila)
    path = manila.save(vault_path(manila))
    _changed_after_the_listing(monkeypatch, lambda: later.save(path))
    with pytest.raises(FileExistsError) as info:
        catalog.pull_and_save("ibm_manila", source="ibm")
    assert info.value.message == (
        f"{path} already holds ibm_manila@2024-06-03T10:00:00Z, another calibration"
    )
    assert nv.load(path) == later


def test_a_pull_saves_its_profile_again_when_the_listed_copy_is_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manila = nv.load("ibm_manila")
    _serve(monkeypatch, manila)
    path = manila.save(vault_path(manila))
    _changed_after_the_listing(monkeypatch, path.unlink)
    assert catalog.pull_and_save("ibm_manila", source="ibm") == (manila, path, True)
    assert nv.load(path) == manila


@pytest.mark.parametrize("links", ["no hard links", "exFAT"], indirect=True)
def test_a_pull_without_hard_links_never_reports_a_file_that_replaced_its_vault_file(
    monkeypatch: pytest.MonkeyPatch, vault: Path, links: None
) -> None:
    manila, later = nv.load("ibm_manila"), _later_manila()
    _serve(monkeypatch, manila)
    path = vault_path(manila)
    real_open = os.open
    pending = [later.save]

    def replaced_once_created(file: str | Path, *args: Any, **kwargs: Any) -> int:
        fd = real_open(file, *args, **kwargs)
        if pending and Path(file) == path:
            pending.pop()(path)
        return fd

    monkeypatch.setattr(os, "open", replaced_once_created)
    with pytest.raises(FileExistsError) as info:
        catalog.pull_and_save("ibm_manila", source="ibm")
    assert info.value.message == (
        f"{path} already holds ibm_manila@2024-06-03T10:00:00Z, another calibration"
    )
    assert nv.load(path) == later
    assert [p.name for p in vault.iterdir() if p.name != ".index.json"] == [path.name]


@pytest.mark.parametrize(
    ("links", "error"),
    [
        ("hard links", "{path} already holds ibm_manila@2024-06-03T10:00:00Z, another calibration"),
        ("no hard links", None),
        ("exFAT", None),
    ],
    indirect=["links"],
)
def test_a_pull_never_reports_or_deletes_a_file_that_replaced_its_hidden_copy(
    monkeypatch: pytest.MonkeyPatch, links: None, error: str | None
) -> None:
    manila, later = nv.load("ibm_manila"), _later_manila()
    _serve(monkeypatch, manila)
    path = vault_path(manila)
    real_link = os.link
    copies: list[Path] = []

    def link_after_another_writer(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        if not copies:
            copies.append(Path(src))
            os.replace(later.save(Path(src).with_name("other.json")), src)
        real_link(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "link", link_after_another_writer)
    if error is None:
        assert catalog.pull_and_save("ibm_manila", source="ibm") == (manila, path, True)
        assert nv.load(path) == manila
    else:
        with pytest.raises(FileExistsError) as info:
            catalog.pull_and_save("ibm_manila", source="ibm")
        assert info.value.message == error.format(path=path)
    assert nv.load(copies[0]) == later


@pytest.mark.parametrize(
    ("links", "hidden"),
    [("hard links", True), ("no hard links", False)],
    ids=["hidden copy", "vault file"],
    indirect=["links"],
)
def test_a_failed_vault_write_leaves_no_file_and_the_next_pull_saves(
    monkeypatch: pytest.MonkeyPatch, vault: Path, links: None, hidden: bool
) -> None:
    manila = nv.load("ibm_manila")
    _serve(monkeypatch, manila)
    path = vault_path(manila)
    failing = f".{path.name}." if hidden else path.name
    real_open = os.open

    def unwritable(file: str | Path, flags: int, *args: Any, **kwargs: Any) -> int:
        if Path(file).name.startswith(failing):
            flags &= ~os.O_WRONLY
        return real_open(file, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", unwritable)
    with pytest.raises(OSError, match="Bad file descriptor"):
        catalog.pull_and_save("ibm_manila", source="ibm")
    assert [p.name for p in vault.iterdir() if p.name != ".index.json"] == []
    monkeypatch.setattr(os, "open", real_open)
    assert catalog.pull_and_save("ibm_manila", source="ibm") == (manila, path, True)


@pytest.mark.parametrize("links", ["no hard links", "exFAT"], indirect=True)
def test_a_vault_write_that_fails_part_way_leaves_no_file(
    monkeypatch: pytest.MonkeyPatch, vault: Path, links: None
) -> None:
    resource = pytest.importorskip("resource")
    _serve(monkeypatch, nv.load("ibm_manila"))
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    real_link = os.link

    def link_with_a_size_limit(*args: Any, **kwargs: Any) -> None:
        resource.setrlimit(resource.RLIMIT_FSIZE, (20, hard))
        real_link(*args, **kwargs)

    monkeypatch.setattr(os, "link", link_with_a_size_limit)
    try:
        with pytest.raises(OSError, match="File too large"):
            catalog.pull_and_save("ibm_manila", source="ibm")
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
    assert [p.name for p in vault.iterdir() if p.name != ".index.json"] == []


@pytest.mark.parametrize("source", ["ibm", "ionq"])
def test_an_impossible_date_is_refused_before_any_request(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    def no_request(*args: object, **kwargs: object) -> None:
        raise AssertionError("the source was asked before the date was checked")

    for module in ("noisevault.sources.ibm_public", "noisevault.sources.ionq"):
        monkeypatch.setattr(f"{module}.pull", no_request)
    with pytest.raises(ValueError, match="not an ISO 8601 date"):
        catalog.pull_and_save("ibm_fez", at="2025-02-30", source=source)
