"""A random circuit on Google Weber's grid qubits, scored by linear cross-entropy (XEB).

Weber's profile records each qubit's grid coordinates, so cirq.GridQubit(row, col) addresses the
device qubit at that position with no layout argument.

Needs: pip install "noisevault[cirq] @ git+https://github.com/dvgyl/noisevault"
"""

import textwrap

import cirq
import numpy as np

import noisevault as nv

weber = nv.load("google_weber")
model = weber.to_cirq()

# A connected chain of 6 well-calibrated qubits, as GridQubits.
coords = {q.index: q.coords for q in weber.qubits}
qubits = [cirq.GridQubit(*map(int, coords[p])) for p in weber.suggest_layout(6).values()]

circuit = cirq.experiments.random_rotations_between_grid_interaction_layers_circuit(
    qubits, depth=8, seed=7
)
# Compile to Weber's natives (PhasedXZ and sqrt-iSWAP) so every gate has its own calibration.
circuit = cirq.optimize_for_target_gateset(circuit, gateset=cirq.SqrtIswapTargetGateset())


def probabilities(noise=None):
    sim = cirq.DensityMatrixSimulator(noise=noise)
    rho = sim.simulate(circuit, qubit_order=qubits).final_density_matrix
    return np.real(np.diag(rho))


ideal, noisy = probabilities(), probabilities(model)
d = 2 ** len(qubits)
xeb = (d * np.sum(noisy * ideal) - 1) / (d * np.sum(ideal**2) - 1)

two_qubit = sum(1 for op in circuit.all_operations() if len(op.qubits) == 2)
print(f"qubits: {', '.join(f'({q.row},{q.col})' for q in qubits)}")
print(f"{len(circuit)} moments, {two_qubit} two-qubit gates")
print(f"linear XEB fidelity under Weber's 2021-11-03 calibration: {xeb:.3f}")
print()
# The report's approximated and omitted lines say where the simulation differs from the device.
for line in model.report.summary().splitlines():
    if line.startswith(("NoiseVault", "approximated", "omitted", "unknown", "clamped")):
        print(textwrap.fill(line, 88, subsequent_indent="    "))
