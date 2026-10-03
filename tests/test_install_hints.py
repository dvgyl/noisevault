"""Each framework export names the pip command for its extra when the framework is missing."""

from __future__ import annotations

import importlib
import re
import sys

import pytest
from conftest import require

import noisevault as nv
from noisevault.errors import install_hint

# (export module, a module it needs, the extra that installs it)
GUARDS = [
    ("noisevault.frameworks.cirq", "cirq", "cirq"),
    ("noisevault.frameworks.pennylane", "pennylane", "pennylane"),
    ("noisevault.frameworks.stim", "stim", "stim"),
    ("noisevault.frameworks.qiskit", "qiskit", "qiskit"),
    ("noisevault.frameworks.qiskit", "qiskit_aer", "qiskit"),
]


def _without(monkeypatch: pytest.MonkeyPatch, export: str, missing: str) -> None:
    """Make ``missing`` unimportable and ``export`` import afresh; monkeypatch restores both."""
    require(missing)
    monkeypatch.setitem(sys.modules, missing, None)
    monkeypatch.delitem(sys.modules, export, raising=False)


@pytest.mark.parametrize(("export", "missing", "extra"), GUARDS)
def test_missing_framework_names_the_install_command(
    monkeypatch: pytest.MonkeyPatch, export: str, missing: str, extra: str
) -> None:
    _without(monkeypatch, export, missing)
    with pytest.raises(ImportError, match=re.escape(install_hint(extra))):
        importlib.import_module(export)


def test_install_hint_is_a_pip_command_for_the_extra() -> None:
    assert install_hint("qiskit") == (
        'pip install "noisevault[qiskit] @ git+https://github.com/dvgyl/noisevault"'
    )


@pytest.mark.parametrize(
    ("method", "export", "missing", "extra"),
    [
        ("to_qiskit", "noisevault.frameworks.qiskit", "qiskit_aer", "qiskit"),
        ("to_cirq", "noisevault.frameworks.cirq", "cirq", "cirq"),
        ("to_pennylane", "noisevault.frameworks.pennylane", "pennylane", "pennylane"),
    ],
)
def test_profile_export_methods_raise_the_hint(
    monkeypatch: pytest.MonkeyPatch, method: str, export: str, missing: str, extra: str
) -> None:
    profile = nv.load("ibm_manila")
    _without(monkeypatch, export, missing)
    with pytest.raises(ImportError, match=re.escape(install_hint(extra))):
        getattr(profile, method)()
