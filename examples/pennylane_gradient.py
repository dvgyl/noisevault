"""Train a small variational circuit under the calibrated noise of IBM Fez.

The ansatz uses Fez's native gates (RZ, SX, CZ), so every gate gets its own calibrated noise and
none needs the typical-noise approximation. Gradients flow through the noise channels.

Needs: pip install "noisevault[pennylane] @ git+https://github.com/dvgyl/noisevault"
"""

import textwrap

import pennylane as qml
from pennylane import numpy as pnp

import noisevault as nv

fez = nv.load("ibm_fez")
layout = fez.suggest_layout(3)  # wire i maps to physical qubit layout[i]
model = fez.to_pennylane(layout=layout)
dev = qml.device("default.mixed", wires=3)


def ansatz(params):
    for layer in params:
        for wire, angle in enumerate(layer):
            qml.SX(wire)
            qml.RZ(angle, wire)
            qml.SX(wire)
        qml.CZ([0, 1])
        qml.CZ([1, 2])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))


ideal = qml.QNode(ansatz, dev, diff_method="backprop")
noisy = qml.add_noise(ideal, model)

params = pnp.array([[0.1, 0.4, -0.3], [0.2, -0.5, 0.3]], requires_grad=True)
print(f"physical qubits {list(layout.values())}")
print(f"<Z0 Z2>  ideal {ideal(params):+.4f}   noisy {noisy(params):+.4f}")
grad = qml.grad(noisy)(params)
print(f"noisy gradient, first layer: {pnp.round(grad[0], 4)}")

# Minimize <Z0 Z2> under device noise. The noisy minimum cannot reach -1.
opt = qml.GradientDescentOptimizer(stepsize=0.4)
for step in range(1, 31):
    params = opt.step(noisy, params)
    if step % 10 == 0:
        print(f"step {step:2d}: noisy {noisy(params):+.4f}   ideal {ideal(params):+.4f}")
print()
# The report's approximated and omitted lines say where the simulation differs from the device.
for line in model.report.summary().splitlines():
    if line.startswith(("NoiseVault", "approximated", "omitted", "unknown", "clamped")):
        print(textwrap.fill(line, 88, subsequent_indent="    "))
