"""Describe hypothetical devices, save and reload them, and run them in Qiskit and Stim.

The numbers below are illustrative, not measurements of any device.

Needs: pip install "noisevault[qiskit,stim] @ git+https://github.com/dvgyl/noisevault"
"""

import tempfile
import textwrap
from pathlib import Path

import stim
from qiskit import QuantumCircuit, transpile

import noisevault as nv

ions = nv.Profile.uniform(
    "toy-ions",
    technology="trapped_ion",
    num_qubits=12,
    one_qubit_error=3e-5,
    two_qubit_error=1e-3,
    readout_error=2e-3,
    t2_us=1e6,  # 1 s of coherence. Ions have no meaningful T1 decay.
    one_qubit_ns=10_000,
    two_qubit_ns=200_000,
)

atoms = nv.Profile.uniform(
    "toy-atoms",
    technology="neutral_atom",
    num_qubits=12,
    one_qubit_error=5e-4,
    two_qubit_error=5e-3,
    readout_error=1e-2,
    t1_us=4e6,
    t2_us=1.5e6,
    one_qubit_ns=500,
    two_qubit_ns=250,
)
# Record that imaging loses atoms. No export models loss yet, so reports list it as omitted.
atoms = atoms.model_copy(update={"effects": [{"type": "atom_loss", "on": "readout", "prob": 5e-3}]})

with tempfile.TemporaryDirectory() as folder:
    for device in (ions, atoms):
        path = device.save(Path(folder) / f"{device.id}.json")
        again = nv.load(path)
        assert again.fingerprint == device.fingerprint
        print(f"saved and reloaded {path.name}: {again.short_fingerprint}")
print()

N = 8
ghz = QuantumCircuit(N)
ghz.h(0)
for q in range(N - 1):
    ghz.cx(q, q + 1)
ghz.measure_all()

ghz_stim = stim.Circuit("H 0\n" + "".join(f"CX {q} {q + 1}\n" for q in range(N - 1)))
ghz_stim.append("M", range(N))

for device in (ions, atoms):
    sim = device.to_qiskit()
    counts = sim.run(transpile(ghz, sim, seed_transpiler=1), shots=4000, seed_simulator=1)
    counts = counts.result().get_counts()
    qiskit_ok = (counts.get("0" * N, 0) + counts.get("1" * N, 0)) / 4000

    noisy = device.to_stim(ghz_stim)
    shots = noisy.compile_sampler(seed=1).sample(100_000)
    stim_ok = (shots.all(axis=1) | ~shots.any(axis=1)).mean()

    print(f"{device.id:<10} GHZ-{N} success  Qiskit {qiskit_ok:.3f}   Stim {stim_ok:.3f}")
    omitted = [line for line in noisy.report.summary().splitlines() if line.startswith("omitted")]
    print(textwrap.fill(omitted[0], 88, initial_indent=" " * 11, subsequent_indent=" " * 15))
