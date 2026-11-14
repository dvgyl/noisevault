# Find your question

## My device has no fake backend. How do I get its noise model?

Pull the calibration from IBM's public endpoint. The pull needs no account:

```bash
nv pull ibm_phoenix
```

<!-- not-run: loads the calibration that nv pull saves -->
```python
import noisevault as nv

sim = nv.load("ibm_phoenix").to_qiskit()
```

The endpoint lists the IBM devices that are online. For a device that the endpoint does not
list, `nv pull` names the devices that it lists. A retired device can be one of the 25 bundled
profiles, which `nv list` shows. For a fake backend that NoiseVault does not bundle, use
`nv.from_qiskit_backend(backend)`. For IonQ, Amazon Braket, IQM, Rigetti and Google devices,
see [Data sources](data-sources.md).

## How do I get the noise of a real device in PennyLane without the Qiskit plugin?

Install the `pennylane` extra. It installs no Qiskit:

```bash
pip install "noisevault[pennylane] @ git+https://github.com/dvgyl/noisevault"
```

```python
import pennylane as qml

import noisevault as nv

fez = nv.load("ibm_fez")
model = fez.to_pennylane(layout=fez.suggest_layout(2))


@qml.qnode(qml.device("default.mixed", wires=2))
def circuit(theta):
    qml.SX(0)
    qml.RZ(theta, 0)
    qml.SX(0)
    qml.CZ([0, 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


noisy = qml.add_noise(circuit, model)
print(f"noisy {noisy(0.3):.4f}, ideal {circuit(0.3):.4f}")
# noisy -0.9393, ideal -0.9553
```

`to_pennylane()` returns a `qml.NoiseModel`, and gradients flow through its channels. PennyLane
circuits have no timing, so they get no idle noise.
[Train a PennyLane circuit under device noise](recipes.md#train-a-pennylane-circuit-under-device-noise)
trains a circuit, and [Frameworks](frameworks.md#pennylane) lists what the export leaves out.

## How do I add the noise of a real device to my Stim circuit, and get a detector error model for a decoder?

```python
import stim

import noisevault as nv

fez = nv.load("ibm_fez")
code = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=3)
noisy = fez.to_stim(code, layout=fez.suggest_layout(code.num_qubits), tick_ns=100)
dem = noisy.detector_error_model()
print(dem.num_detectors, dem.num_errors)
# 8 21
```

Each gate gets the Pauli twirl of its calibrated channel on its physical qubits. Each qubit that
waits in a `TICK` layer gets relaxation for `tick_ns`. For the qubits of a hardware run, give
`layout` as a dict, such as `{0: 136, 1: 143}`. Each key is a Stim qubit, and each value is a
physical qubit. Stim's `CX` is not an IBM native gate, so `CX` gets the noise of `cz` on the same
pair, with a warning.
[Run a QEC memory experiment](recipes.md#run-a-qec-memory-experiment-with-stim-and-pymatching)
decodes with PyMatching, and [Frameworks](frameworks.md#stim) gives the options.

## How do I get the noise model of a past date, and how do I cite it in a paper?

Give the date or the time of your job to `--at`:

```bash
nv pull ibm_fez --at 2026-11-12T14:30Z
```

`nv pull` saves the newest calibration that the endpoint showed at that time. Its first output
line gives the ref and the fingerprint, such as `ibm_fez@2026-11-12T14:09:05Z nv:7a500676dfc0`.
Cite that ref:

```bash
nv cite ibm_fez@2026-11-12T14:09:05Z
nv cite ibm_fez@2026-11-12T14:09:05Z --bibtex
```

In your code, load the profile with `nv.load(REF, expect=FINGERPRINT)`. A pulled IBM profile has
no open license, so publish the fingerprint and not the file.
[Pin and cite a calibration for a paper](recipes.md#pin-and-cite-a-calibration-for-a-paper) gives
the full steps.

## Which error numbers go into a noise model, and what does each stated number mean?

| Stated number | Meaning | Use in the noise model |
| --- | --- | --- |
| Gate error | The average gate infidelity r = 1 - F_avg, usually from randomized benchmarking | The gate's channel, with relaxation included, has the error r. If relaxation alone is larger than r, the gate keeps the relaxation |
| T1 | The time constant of decay from \|1> to \|0> | Amplitude damping during each gate and each stated wait |
| T2 | The time constant of the loss of phase | Dephasing during each gate and each stated wait. A T2 above 2 T1 becomes 2 T1 |
| P(1\|0) | The probability to read 1 when the qubit is in \|0> | The readout matrix |
| P(0\|1) | The probability to read 0 when the qubit is in \|1> | The readout matrix |

A gate error r is not the probability of a depolarizing channel. For one qubit, r = 1e-3 is a
total Pauli error of 1.5e-3, with 5e-4 on each of X, Y and Z:

```python
import noisevault as nv

toy = nv.Profile.uniform("toy", technology="superconducting", num_qubits=1, one_qubit_error=1e-3)
print({pauli: round(p, 6) for pauli, p in toy.pauli_error("sx", [0]).items()})
# {'X': 0.0005, 'Y': 0.0005, 'Z': 0.0005}
```

Qiskit's `depolarizing_error(lambda, 1)` needs lambda = 2e-3 for the same gate. Cirq's
`depolarize(p)` and Stim's `DEPOLARIZE1(p)` need p = 1.5e-3. [Conventions](conventions.md) gives
the equations, how each vendor states its numbers, and how NoiseVault builds each channel.

## Which gates of my circuit get noise, and what does each export leave out?

Each gate gets its channel after the ideal gate, on the physical qubits of the gate. A gate that
the profile does not calibrate gets the noise of the typical native gate, with a warning. The
report of each export says what it did:

```python
import cirq

import noisevault as nv

model = nv.load("ibm_fez").to_cirq()
a, b = cirq.LineQubit.range(2)
circuit = cirq.Circuit(cirq.H(a), cirq.CNOT(a, b), cirq.measure(a, b, key="m"))
cirq.DensityMatrixSimulator(noise=model, seed=1).run(circuit, repetitions=100)
print(model.report.summary())
# NoiseVault ... -> cirq ...: ibm_fez (nv:06404cefa54f)
# ...
# approximated: gate h: noise of the typical 1-qubit native gate ...
# approximated: gate cx: noise of the typical 2-qubit native gate ...
# ...
# omitted: preparation error of the initial state ...
# used: h took the typical native gate's noise once; cx took the typical native gate's noise once
```

`h` and `cx` are not IBM native gates, so each one got the noise of one native gate. On the
device, `cx` is a `cz` with single-qubit gates around it, so this circuit gets less noise than
on the device. Compile the circuit to the native gates first for realistic noise. To refuse a
gate that the profile does not calibrate, pass `unknown_gates="error"`. The Qiskit simulator
raises `CircuitNotNativeError` for a circuit that is not in native gates. Save
`report.to_dict()` next to your results. [Frameworks](frameworks.md) lists what the report of
each export can list.

## My old Qiskit noise code does not run any more. What is the present form?

Each old line below fails with Qiskit 2.1 and 2.5, qiskit-aer 0.17 and qiskit-ibm-runtime 0.40
and 0.49, the versions that NoiseVault supports. Each present form runs with them.

| Old code | Result in the tested versions | Present form |
| --- | --- | --- |
| `from qiskit import IBMQ` | `ImportError` | `nv pull ibm_fez`, with no account |
| `from qiskit_ibm_provider import IBMProvider` | `ModuleNotFoundError` | `nv pull ibm_fez`, with no account |
| `from qiskit import Aer` | `ImportError` | `from qiskit_aer import AerSimulator` |
| `from qiskit.providers.aer.noise import NoiseModel` | `ModuleNotFoundError` | `from qiskit_aer.noise import NoiseModel` |
| `from qiskit.test.mock import FakeMontreal` | `ModuleNotFoundError` | `from qiskit_ibm_runtime.fake_provider import FakeFez` |
| `from qiskit.providers.fake_provider import FakeMontreal` | `ImportError` | `from qiskit_ibm_runtime.fake_provider import FakeFez` |
| `AerSimulator.from_backend(FakeMontreal())` | qiskit-ibm-runtime has no `FakeMontreal` | `sim = nv.load("ibm_fez").to_qiskit()` |
| `execute(circuit, sim, noise_model=noise_model)` | `ImportError` | `sim = AerSimulator(noise_model=noise_model)`, then `sim.run(transpile(circuit, sim))` |
| `sim.run(assemble(circuit))` | `ImportError` | `sim.run(transpile(circuit, sim))` |
| `circuit.snapshot_density_matrix("rho")` | `AttributeError` | `native.save_density_matrix()` after `transpile`, then `sim.run(native, method="density_matrix")` |

`FakeFez` needs the `ibm` extra. This code runs each present form:

```python
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime.fake_provider import FakeFez

import noisevault as nv

bell = QuantumCircuit(2)
bell.h(0)
bell.cx(0, 1)
measured = bell.measure_all(inplace=False)

noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), ["cx"])
sim = nv.load("ibm_fez").to_qiskit()
for backend in (AerSimulator.from_backend(FakeFez()), sim, AerSimulator(noise_model=noise_model)):
    backend.run(transpile(measured, backend, seed_transpiler=1), shots=100).result().get_counts()

native = transpile(bell, sim, initial_layout=[0, 1], seed_transpiler=1)
native.save_density_matrix(qubits=[0, 1])
rho = sim.run(native, method="density_matrix").result().data()["density_matrix"]
print(f"purity {rho.purity().real:.3f}")
# purity 0.977
```

Add `save_density_matrix` after `transpile`. `transpile` cannot translate `save_density_matrix`
to the native gates of a device, also with `AerSimulator.from_backend`. A NoiseVault profile
keeps both readout errors of each qubit, and `from_backend` uses their mean.
[Coming from fake backends](../README.md#coming-from-fake-backends) gives the other
differences.

## How do I make a noise model from my own lab's numbers?

Write your numbers into a profile dict, and give the dict to `nv.Profile.from_dict`. `gates`
holds one value of each gate for the device. `qubits` holds T1, T2 and the two readout errors of
each qubit. `calibrations` holds the error of each gate on each qubit or pair. Then check the
file:

```bash
nv validate lab3.json
nv check lab3.json
```

[Make a profile from your own calibration numbers](recipes.md#make-a-profile-from-your-own-calibration-numbers)
builds a 3-qubit profile from a table of numbers. For a device with one value of each gate,
`nv.Profile.uniform` is shorter, as in
[Describe a hypothetical device](recipes.md#describe-a-hypothetical-device).

## I want the readout errors of my device at the time of my job for readout mitigation.

Pull the calibration of the job time with `nv pull DEVICE --at TIME`. Then read the two readout
errors of each qubit. This example uses the bundled `ibm_fez` profile:

```python
import numpy as np

import noisevault as nv

fez = nv.load("ibm_fez@2025-02-26")
for q in (3, 4):
    p1_given_0, p0_given_1 = fez.table.qubit(q).readout
    matrix = np.array([[1 - p1_given_0, p0_given_1], [p1_given_0, 1 - p0_given_1]])
    print(q, matrix.round(4).tolist())
# 3 [[0.9946, 0.0264], [0.0054, 0.9736]]
# 4 [[1.0, 0.0029], [0.0, 0.9971]]
```

Each column of the matrix is a prepared state, and each row is a measured result, as in
[Conventions](conventions.md#readout). `readout` is `None` for a qubit with no stated readout
error. Know two limits:

- A stated readout error is one reading. On IBM devices, a readout value changes about each 4
  hours (median). The calibration of the job time holds the newest reading before the job.
  [Limits of the data](limitations.md#limits-of-the-data) gives the ages.
- Some stated values are exactly 0. On qubit 4, P(1|0) is 0, so the matrix says that qubit 4
  never reads 1 from 0. A device does read 1 from 0 on some shots. `fez.zero_readouts()` names
  the 11 such qubits of this profile, and `fez.floored()` gives a profile with these values
  replaced. [Scoring](scoring.md#ruled-out) gives the rule.

Each qubit's readout error is independent of the other qubits in every export. NoiseVault does
not mitigate readout errors.

## Do the exports to Qiskit, Cirq, PennyLane and Stim give the same noise?

Yes, for every gate that each framework can express. Run the check:

```bash
nv check ibm_fez
```

`nv check` runs small circuits through each installed export and compares the results with
NoiseVault's own density-matrix reference. Qiskit, Cirq and PennyLane must agree with the
reference exactly, and Stim must agree with the Pauli twirl of the reference within shot noise.
The exports differ where a framework has no means to apply a noise. PennyLane gets no idle
noise, and Stim flips readout results with the mean of the two readout errors by default. A pass
does not measure how well the model matches the hardware.
[Check a conversion](frameworks.md#check-a-conversion) explains each column.

## What does the tool not model?

- Crosstalk. A gate on one pair does not change the qubits near it.
- Leakage. The Stim and Cirq exports apply leakage as an erasure. The qubit comes back at once
  in a random state. The other exports apply no leakage.
- Errors with a direction, such as a rotation that is always 0.3 degrees too large. Each gate
  error becomes random noise, so these errors do not add up as on the device.
- Correlated readout, and preparation error of the initial state.
- Idle noise outside the waits that the circuit states.

[Limitations](limitations.md) gives each limit and the checks of the exports.
