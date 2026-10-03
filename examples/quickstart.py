"""Run a GHZ circuit under the calibrated noise of IBM Fez.

Fez has a bundled profile, so the script works offline.

Needs: pip install "noisevault[qiskit] @ git+https://github.com/dvgyl/noisevault"
"""

import textwrap

from qiskit import QuantumCircuit, transpile

import noisevault as nv

fez = nv.load("ibm_fez")
print(fez.summary().splitlines()[0])

sim = fez.to_qiskit()

ghz = QuantumCircuit(3)
ghz.h(0)
ghz.cx(0, 1)
ghz.cx(1, 2)
ghz.measure_all()

# Place the circuit on the best-calibrated chain of 3 qubits, in Fez's native gates.
layout = list(fez.suggest_layout(3).values())
native = transpile(ghz, sim, initial_layout=layout, seed_transpiler=1)
counts = sim.run(native, shots=4000, seed_simulator=1).result().get_counts()

print(f"physical qubits {layout}")
for bits, n in sorted(counts.items(), key=lambda item: -item[1]):
    print(f"  {bits}  {n}")
print(f"GHZ success (000 or 111): {(counts.get('000', 0) + counts.get('111', 0)) / 4000:.3f}")
print()
# The report's approximated and omitted lines say where the simulation differs from the device.
for line in sim.report.summary().splitlines():
    if line.startswith(("NoiseVault", "approximated", "omitted", "unknown", "clamped")):
        print(textwrap.fill(line, 88, subsequent_indent="    "))
