"""Shared test helpers.

Optional frameworks are imported through ``require``: a missing one skips the test, unless
NOISEVAULT_REQUIRE_ALL=1, which turns the skip into a failure so a full environment cannot
silently skip coverage.
"""

from __future__ import annotations

import copy
import gc
import importlib
import importlib.util
import json
import os
import warnings
from collections.abc import Iterator
from importlib.metadata import version
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from packaging.version import Version

from noisevault.errors import MigrationWarning
from noisevault.profile import Profile, load_file

# Property tests run under one Hypothesis profile. HYPOTHESIS_PROFILE selects it. Without it, CI
# (where the CI variable is set) uses "ci" and a local run uses "dev". "ci" is derandomized, so a
# CI failure is not flaky and repeats locally with HYPOTHESIS_PROFILE=ci. "deep" is for a long
# local search: HYPOTHESIS_PROFILE=deep pytest tests/test_property_*.py
HYPOTHESIS_PROFILES = {"ci": 25, "dev": 100, "deep": 2000}
# A core install without the dev extra has no Hypothesis and runs the example tests only.
# NOISEVAULT_REQUIRE_ALL=1 collects the property tests anyway, so their import error shows.
collect_ignore_glob: list[str] = []
if importlib.util.find_spec("hypothesis") is None:
    if os.environ.get("NOISEVAULT_REQUIRE_ALL") != "1":
        collect_ignore_glob.append("test_property_*.py")
else:
    from hypothesis import settings

    for _name, _count in HYPOTHESIS_PROFILES.items():
        settings.register_profile(
            _name, max_examples=_count, deadline=None, derandomize=_name == "ci", print_blob=True
        )
    settings.load_profile(
        os.environ.get("HYPOTHESIS_PROFILE") or ("ci" if os.environ.get("CI") else "dev")
    )


def examples(share: float) -> int:
    """A share of the active profile's examples, for a property whose examples are slow."""
    from hypothesis import settings

    return max(1, round(settings().max_examples * share))


V01 = Path(__file__).parent / "fixtures" / "v01"
MANILA_V01 = V01 / "fake_manila_2024-05-27T15-27-23-03-00.json"

_TOY: dict[str, Any] = {
    "noisevault": "1.0",
    "device": {"name": "toy", "vendor": "test", "technology": "superconducting", "num_qubits": 3},
    "connectivity": {"edges": [[0, 1], [1, 2]]},
    "gates": {
        "rz": {"virtual": True},
        "sx": {"avg_infidelity": 1e-3, "duration_ns": 35},
        "cz": {"avg_infidelity": 1e-2, "duration_ns": 70},
    },
}


def require(module: str) -> ModuleType:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        if os.environ.get("NOISEVAULT_REQUIRE_ALL") == "1":
            pytest.fail(f"{module} must be installed when NOISEVAULT_REQUIRE_ALL=1 ({exc})")
        pytest.skip(f"{module} is not installed", allow_module_level=True)


# Fake backends newer than the qiskit-ibm-runtime floor, with the release that added them.
FAKES_ADDED_IN = {"FakeKingston": "0.47"}


def needs_runtime(release: str) -> None:
    """Skip when the installed qiskit-ibm-runtime predates `release`; the extra allows it."""
    if Version(version("qiskit-ibm-runtime")) < Version(release):
        pytest.skip(f"needs qiskit-ibm-runtime {release} or later")


def toy(**sections: Any) -> dict[str, Any]:
    """A small valid profile dict; keyword arguments replace top-level sections."""
    data = copy.deepcopy(_TOY)
    data.update(copy.deepcopy(sections))
    return data


def migrated(path: Path) -> Profile:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MigrationWarning)
        return load_file(path)


def deeper_than_the_parser_takes() -> str:
    """JSON arrays nested past the depth ``json.loads`` can parse in this interpreter.

    Python 3.11 bounds that depth by the recursion limit, 3.12 and 3.13 by a fixed C limit, and
    3.14 by the free stack, so the depth is found by doubling rather than fixed.
    """
    depth = 1024
    while depth <= 2**23:
        text = "[" * depth + "]" * depth
        try:
            json.loads(text)
        except RecursionError:
            return text
        depth *= 2
    pytest.skip("json.loads parses arrays nested 8 million deep here")


@pytest.fixture(autouse=True)
def vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test gets an empty private vault, never the user's ~/.noisevault."""
    home = tmp_path / "nv_home"
    monkeypatch.setenv("NOISEVAULT_HOME", str(home))
    return home / "profiles"


@pytest.fixture(autouse=True)
def _no_gc_pauses_in_time_budgets(request: pytest.FixtureRequest) -> Iterator[None]:
    if "timing" not in request.keywords:
        yield
        return
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: a scale test that still runs by default")
    config.addinivalue_line("markers", "timing: a time budget, checked only in a serial run")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Tests marked ``network`` reach live endpoints, so they run only with NOISEVAULT_NETWORK=1.

    Time budgets (``timing``) are only meaningful in a serial run; parallel workers share the CPU.
    """
    if os.environ.get("PYTEST_XDIST_WORKER"):
        serial_only = pytest.mark.skip(reason="timing budget is checked in a serial run")
        for item in items:
            if "timing" in item.keywords:
                item.add_marker(serial_only)
    if os.environ.get("NOISEVAULT_NETWORK") == "1":
        return
    skip = pytest.mark.skip(reason="set NOISEVAULT_NETWORK=1 to run network tests")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
