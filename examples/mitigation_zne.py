"""Zero-noise extrapolation with Mitiq, against the calibrated noise of IBM Fez.

Without noise, a mirror circuit (layers of gates, then their inverse) returns |000> with
probability 1. Mitiq folds gates to amplify the noise, runs each folded circuit on the NoiseVault
simulator and extrapolates back to zero noise.

Mitiq supports Python up to 3.12. Mitiq's Qiskit conversion imports ply.

Needs:
    pip install "noisevault[qiskit] @ git+https://github.com/dvgyl/noisevault"
    pip install mitiq ply
"""

import sys

try:
    import ply  # noqa: F401
    from mitiq import zne
except ImportError:
    print("this example needs Mitiq: pip install mitiq ply (Mitiq supports Python up to 3.12)")
    sys.exit(1)

import numpy as np
from qiskit import QuantumCircuit, transpile

import noisevault as nv

SHOTS = 20_000
fez = nv.load("ibm_fez")
sim = fez.to_qiskit()
layout = list(fez.suggest_layout(3).values())

rng = np.random.default_rng(3)
half = QuantumCircuit(3)
for _ in range(6):
    for q in range(3):
        half.rz(rng.uniform(0, 2 * np.pi), q)
        half.sx(q)
    half.cz(0, 1)
    half.cz(1, 2)
mirror = half.compose(half.inverse())


def p000(circuit: QuantumCircuit) -> float:
    """Return the probability of 000 on Fez.

    Optimization level 0 translates the folded inverses and does not cancel them.
    """
    measured = circuit.copy()
    measured.measure_all()
    native = transpile(measured, sim, initial_layout=layout, optimization_level=0)
    counts = sim.run(native, shots=SHOTS, seed_simulator=7).result().get_counts()
    return counts.get("000", 0) / SHOTS


factory = zne.inference.RichardsonFactory(scale_factors=[1, 2, 3])
mitigated = zne.execute_with_zne(mirror, p000, factory=factory, scale_noise=zne.scaling.fold_global)

print(f"mirror circuit: {mirror.size()} gates on physical qubits {layout}")
print("ideal     P(000) = 1.000")
print(f"noisy     P(000) = {p000(mirror):.3f}")
print(f"ZNE       P(000) = {mitigated:.3f}  (Richardson, scale factors 1, 2, 3)")
print(f"noise: {fez.id}@{fez.device.calibrated_at:%Y-%m-%d} {fez.short_fingerprint}")
