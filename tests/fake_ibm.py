"""A fake IBM Quantum account for the campaign scripts: fake backends run each job locally."""

from __future__ import annotations

import copy
import functools
import importlib.util
import sys
import warnings
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from conftest import require

import noisevault as nv
from noisevault.profile import Profile

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_PROFILES: dict[tuple[str, str, float], Profile] = {}


@functools.cache
def _simulator() -> Any:
    aer = require("qiskit_aer")
    noise = require("qiskit_aer.noise")
    model = noise.NoiseModel()
    model.add_all_qubit_quantum_error(noise.depolarizing_error(0.004, 1), ["sx", "x"])
    model.add_all_qubit_quantum_error(noise.depolarizing_error(0.02, 2), ["cz"])
    model.add_all_qubit_readout_error(noise.ReadoutError([[0.97, 0.03], [0.05, 0.95]]))
    return aer.AerSimulator(noise_model=model)


def noisy_fake(cls_name: str = "FakeFez") -> Any:
    """A fake backend whose simulator adds a small, fast noise model."""
    fake_provider = require("qiskit_ibm_runtime.fake_provider")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fake = getattr(fake_provider, cls_name)()
    # Without a sim, each local run deep-copies the fake and rebuilds its full noise model.
    fake.sim = _simulator()
    return fake


class FakeIBM:
    """Fake backends by device name, a job store, and a QPU-time counter."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, backends: dict[str, Any]) -> None:
        runtime = require("qiskit_ibm_runtime")
        self.backends = backends
        self.remaining: float | None = 600.0
        self.jobs: dict[str, Any] = {}
        self.submitted: list[str] = []
        self.opened: list[str] = []
        self.states: dict[str, str] = {}
        self.scales = dict.fromkeys(backends, 1.0)
        record = self

        class Recording(runtime.SamplerV2):  # type: ignore[misc, name-defined]
            def run(self, pubs: Any, **options: Any) -> Any:
                job = super().run(pubs, **options)
                record.jobs[job.job_id()] = job
                record.submitted.append(job.job_id())
                record.states[job.job_id()] = "QUEUED"
                return job

        monkeypatch.setattr(runtime, "SamplerV2", Recording)

    def account(self, campaign: ModuleType) -> Any:
        return campaign.Account(
            pull=self.pull,
            open_backend=lambda device, fractional: self.backends[device],
            open_job=self.open_job,
            remaining_s=lambda: self.remaining,
        )

    def pull(self, device: str, at: datetime | None) -> Profile:
        fake = self.backends[device]
        key = (type(fake).__name__, str(fake.properties().last_update_date), self.scales[device])
        if key not in _PROFILES:
            _PROFILES[key] = nv.from_qiskit_backend(fake)
        return _PROFILES[key]

    def open_job(self, job_id: str) -> Any:
        self.opened.append(job_id)
        return _Job(self, self.jobs[job_id])

    def recalibrate(self, device: str, factor: float, when: datetime) -> None:
        """Scale every gate and readout error of ``device`` and date the calibration ``when``."""
        fake = self.backends[device]
        props = copy.deepcopy(fake._props_dict)
        props["last_update_date"] = when.isoformat()
        for gate in props["gates"]:
            for param in gate["parameters"]:
                if param["name"] == "gate_error":
                    param["value"] = min(param["value"] * factor, 0.5)
        for qubit in props["qubits"]:
            for param in qubit:
                if param["name"] in ("readout_error", "prob_meas0_prep1", "prob_meas1_prep0"):
                    param["value"] = min(param["value"] * factor, 0.5)
        fake._props_dict = props
        self.scales[device] *= factor


class _Job:
    """A local job whose ``status()`` is the state the test sets. A new job is "QUEUED"."""

    def __init__(self, ibm: FakeIBM, job: Any) -> None:
        self._ibm, self._job = ibm, job

    def status(self) -> str:
        return self._ibm.states[self._job.job_id()]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._job, name)
