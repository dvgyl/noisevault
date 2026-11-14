"""The probability of the right result of each circuit with a systematic over-rotation.

    python -m study.overrotation [--profile ibm_kingston@2026-04-15]

The script takes the circuits of :func:`noisevault.suite.suite` for the profile, with the seed 0,
on the qubits of the ``readout`` circuit. It then computes the state of each circuit with no
noise, with one error only: each ``sx`` gate turns by pi / 2 + e in place of pi / 2, and each
``x`` gate turns by pi + e_x in place of pi. Both are rotations about the x axis. Each other gate
is exact, and a ``delay`` does nothing.

The right result of a circuit is the bit string with the highest probability when e = 0. For
each e of ``TURNS``, the output gives the probability of the right result in two cases: e_x = 0,
and e_x = e. The output also gives the number of ``sx``, ``x`` and 2-qubit gates of each circuit.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence

import numpy as np

import noisevault as nv
from noisevault import gates
from noisevault.counts import PlannedCircuit
from noisevault.profile import Profile
from noisevault.suite import SUITE_VERSION, suite

PROFILE = "ibm_kingston@2026-04-15"
TURNS = (0.0, 0.002, 0.005, 0.01, 0.02)
CASES = {"x error = 0": 0.0, "x error = e": 1.0}


def rx(theta: float) -> np.ndarray:
    """The rotation by ``theta`` about the x axis."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def probabilities(circuit: PlannedCircuit, e: float, e_x: float) -> np.ndarray:
    """The probability of each bit string of ``circuit``, circuit qubit 0 first, with no noise
    and the turn errors ``e`` of ``sx`` and ``e_x`` of ``x``."""
    n = len(circuit.qubits)
    state = np.zeros((2,) * n, dtype=complex)
    state[(0,) * n] = 1
    turned = {"sx": rx(np.pi / 2 + e), "x": rx(np.pi + e_x)}
    for op in circuit.ops:
        if op.name == "delay":
            continue
        u = turned.get(op.name)
        if u is None:
            u = gates.unitary(op.name, op.params)
        k = len(op.qubits)
        u = u.reshape((2,) * (2 * k))
        state = np.tensordot(u, state, axes=(list(range(k, 2 * k)), list(op.qubits)))
        state = np.moveaxis(state, list(range(k)), list(op.qubits))
    return np.abs(state.reshape(-1)) ** 2


def right_results(circuit: PlannedCircuit, x_share: float) -> list[float]:
    """The probability of the right result for each e of ``TURNS``, with e_x = x_share * e."""
    right = int(np.argmax(probabilities(circuit, 0.0, 0.0)))
    return [float(probabilities(circuit, e, x_share * e)[right]) for e in TURNS]


def chain_circuits(profile: Profile) -> list[PlannedCircuit]:
    """The circuits of the suite whose qubits are all qubits of the ``readout`` circuit."""
    circuits = suite(profile)
    chain = set(next(c for c in circuits if c.name == "readout").qubits)
    return [c for c in circuits if set(c.qubits) <= chain]


def report(profile: Profile) -> str:
    """The probability of the right result of each circuit of :func:`chain_circuits`, as text."""
    circuits = chain_circuits(profile)
    used = Counter(
        op.name for c in circuits for op in c.ops if len(op.qubits) == 1 and op.name != "delay"
    )
    lines = [
        f"profile: {profile.id}, calibrated {profile.device.calibrated_at}; circuit set version"
        f" {SUITE_VERSION}, seed 0, {len(circuits)} circuits on the qubits of the readout circuit",
        "1-qubit gates used: " + ", ".join(f"{name} {n}" for name, n in used.most_common()),
    ]
    for label, x_share in CASES.items():
        lines += [
            "",
            f"{'circuit':18s} {'sx':>5s} {'x':>5s} {'2q':>5s} "
            + " ".join(f"e={e:<6g}" for e in TURNS)
            + f"   (probability of the right result; {label})",
        ]
        for c in circuits:
            names = Counter(op.name for op in c.ops)
            two = sum(1 for op in c.ops if len(op.qubits) == 2)
            row = " ".join(f"{p:8.3f}" for p in right_results(c, x_share))
            lines.append(f"{c.name:18s} {names['sx']:5d} {names['x']:5d} {two:5d} {row}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--profile", default=PROFILE, help=f"the profile (default {PROFILE})")
    args = parser.parse_args(argv)
    print(report(nv.load(args.profile)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
