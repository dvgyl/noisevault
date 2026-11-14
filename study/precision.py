"""The precision of the fitted gate factor, from the Fisher information of a profile.

    python -m study.precision fisher [--profile ibm_kingston@2026-04-15]
    python -m study.precision budget [--profile quantinuum_h1-1@2025-05-02]

Both commands take the device to be the profile with a true gate factor and a readout factor of
1. :func:`noisevault.design.information` gives the Fisher information of one shot of a circuit
on the log gate factor and the log readout factor. The information of a run is the sum over its
circuits of the shots times this matrix. The inverse of the information is the covariance of the
two log factors. Thus the square root of its first diagonal entry is the relative standard error
of the gate factor, with the readout factor also unknown.

``fisher`` compares two sets of circuits with the same total shots, ``SHOTS``:

- the fit circuits: the ``train`` circuits of :func:`noisevault.suite.suite`;
- the short circuits: the circuits of :func:`noisevault.counts.plan`, which ``nv check`` uses.

Each circuit of a set gets the same shots. The output gives the relative standard error of each
set for each true gate factor of ``GATE_FACTORS``, and their ratio. A ratio r means that the
short circuits need r squared times the shots for the same error. The output also gives the
error of the short circuits with ``SHORT_SHOTS`` shots for each circuit, and the median stated
error of each 1-qubit gate of the profile.

``budget`` gives designs for a fixed amount of money with :func:`noisevault.design.design`, for
each price and amount of ``CASES`` and each assumed gate factor of ``GATE_FACTORS``. The prices
are the public prices of some devices, in USD. The default profile is an ion-trap profile. The
last part gives the error of the short circuits for some of the same prices and amounts. Each
short circuit then gets the largest number of shots that the amount pays.
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

import noisevault as nv
from noisevault.counts import PlannedCircuit, plan
from noisevault.design import Price, design, information
from noisevault.profile import Profile
from noisevault.suite import SUITE_VERSION, role, suite

FISHER_PROFILE = "ibm_kingston@2026-04-15"
BUDGET_PROFILE = "quantinuum_h1-1@2025-05-02"
GATE_FACTORS = (1.0, 2.0)
SHOTS = 36_000
SHORT_SHOTS = 4_000


class Case(NamedTuple):
    """A price and an amount of money."""

    name: str
    price: Price
    budget: float

    def label(self, *, tasks: bool = False) -> str:
        task = ", one task" if tasks and self.price.one_task else ""
        return f"{self.name}{task}, {self.budget:g} USD"


IONQ = Price(0.30, 0.08)
AQT = Price(0.30, 0.0235, one_task=True)
CASES = (
    Case("IonQ Forte price", IONQ, 321.0),
    Case("IonQ Forte price", IONQ, 160.0),
    Case("AQT IBEX price", AQT, 212.0),
    Case("AQT IBEX price", AQT, 100.0),
    Case("IQM Garnet price", Price(0.30, 0.00145, one_task=True), 35.0),
    Case("Rigetti Cepheus price", Price(0.30, 0.000425, one_task=True), 10.5),
)
SHORT_CASES = (CASES[1], CASES[0], CASES[3])


def run_information(
    profile: Profile, circuits: Sequence[PlannedCircuit], gate_factor: float
) -> np.ndarray:
    """The Fisher information of one shot of each circuit together, when the device has the
    true gate factor ``gate_factor``."""
    matrices = (information(profile, c, gate_factor=gate_factor) for c in circuits)
    return sum(matrices, np.zeros((2, 2)))


def relative_errors(info: np.ndarray, shots: int) -> tuple[float, float]:
    """The relative standard errors of the gate factor and of the readout factor, when each
    circuit of the information ``info`` of :func:`run_information` gets ``shots`` shots."""
    covariance = np.linalg.inv(shots * info)
    return math.sqrt(covariance[0, 0]), math.sqrt(covariance[1, 1])


def stated_errors(profile: Profile) -> dict[str, tuple[str, float]]:
    """For each 1-qubit gate with a stated error on a qubit: the kind of the error and its
    median over the qubits."""
    found: dict[str, tuple[str, list[float]]] = {}
    for record in profile.calibrations:
        if len(record.qubits) == 1 and record.metric is not None:
            kind, value = record.metric
            found.setdefault(record.gate, (kind, []))[1].append(float(value))
    return {gate: (kind, statistics.median(v)) for gate, (kind, v) in sorted(found.items())}


def short_shots(case: Case, circuits: int) -> int:
    """The largest number of shots for each of ``circuits`` circuits that ``case`` pays."""
    fixed = case.price.cost(circuits, 0)
    each = case.price.cost(circuits, 1) - fixed
    return max(0, math.floor((case.budget - fixed) / each + 1e-9))


def _fisher(profile: Profile) -> list[str]:
    fit = [c for c in suite(profile) if role(c.name) == "train"]
    short = list(plan(profile))
    lines = [
        "Relative standard error of the gate factor from the Fisher information of the profile.",
        f"Profile: {profile.id} calibrated {profile.device.calibrated_at}"
        f" | circuit set version {SUITE_VERSION}",
        f"Fit circuits: {', '.join(c.name for c in fit)}",
        f"Short circuits: {', '.join(c.name for c in short)}",
    ]
    for factor in GATE_FACTORS:
        on_fit, on_short = (run_information(profile, c, factor) for c in (fit, short))
        deep = relative_errors(on_fit, SHOTS // len(fit))[0]
        same = relative_errors(on_short, SHOTS // len(short))[0]
        few = relative_errors(on_short, SHORT_SHOTS)[0]
        lines += [
            f"true gate factor {factor}: {len(fit)} fit circuits, {SHOTS} shots: {deep:.4f}"
            f" | {len(short)} short circuits, {SHOTS} shots: {same:.4f}"
            f" | ratio {same / deep:.2f} | ratio squared {(same / deep) ** 2:.1f}",
            f"    {len(short)} short circuits with {SHORT_SHOTS} shots each: {few:.4f}",
        ]
    for gate, (kind, value) in stated_errors(profile).items():
        lines.append(f"median stated {gate} error of the profile ({kind}): {value!r}")
    return lines


def _budget(profile: Profile) -> list[str]:
    circuits = suite(profile)
    lines = [f"suite of {profile.id}: {len(circuits)} circuits"]
    for case in CASES:
        for factor in GATE_FACTORS:
            found = design(
                profile, circuits, budget=case.budget, price=case.price, gate_factor=factor
            )
            lines += [f"== {case.label()}, assumed gate factor {factor}", found.summary()]
    short = list(plan(profile))
    lines += [
        "",
        f"== The short circuits for the same money (profile {profile.id})",
        f"circuits: {', '.join(c.name for c in short)}",
    ]
    on_short = {factor: run_information(profile, short, factor) for factor in GATE_FACTORS}
    for case in SHORT_CASES:
        shots = short_shots(case, len(short))
        for factor in GATE_FACTORS:
            error = relative_errors(on_short[factor], shots)[0]
            lines.append(
                f"{case.label(tasks=True)}, {len(short)} circuits, {shots} shots each,"
                f" true gate factor {factor}: standard error of the gate factor {100 * error:.1f}%"
            )
    return lines


def report(command: str, profile: Profile) -> str:
    """The text of ``command``, ``fisher`` or ``budget``, for ``profile``."""
    return "\n".join(_fisher(profile) if command == "fisher" else _budget(profile))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, default, text in (
        ("fisher", FISHER_PROFILE, "the fit circuits against the short circuits"),
        ("budget", BUDGET_PROFILE, "designs for a fixed amount of money"),
    ):
        command = commands.add_parser(name, help=text)
        command.add_argument("--profile", default=default, help=f"the profile (default {default})")
    args = parser.parse_args(argv)
    print(report(args.command, nv.load(args.profile)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
