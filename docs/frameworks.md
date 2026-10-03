# Frameworks

Each export turns a profile into an object of the framework's own type. You can use that object
in code that you already have. Each export carries a `.report`. The report lists what the
simulator reproduces exactly, what it approximates, what it leaves out, and what the profile does
not know. Print the report with `.report.summary()`. You can also save `.report.to_dict()` next
to your results.

| Framework | Call | Returns | Simulate with |
| --- | --- | --- | --- |
| Qiskit | `profile.to_qiskit()` | `NoiseVaultSimulator`, an `AerSimulator` | `sim.run(transpile(circuit, sim))` |
| Cirq | `profile.to_cirq()` | `NoiseVaultNoiseModel`, a `cirq.NoiseModel` | `cirq.DensityMatrixSimulator(noise=model)` |
| PennyLane | `profile.to_pennylane()` | `NoiseVaultPennyLaneModel`, a `qml.NoiseModel` | `qml.add_noise(qnode, model)` on `default.mixed` |
| Stim | `profile.to_stim(circuit)` | `NoiseVaultStimCircuit`, a `stim.Circuit` | its own samplers, or `detector_error_model()` |

Each framework installs with the extra of the same name. These extras are `qiskit`, `cirq`,
`pennylane` and `stim`. The `all` extra installs every framework. Importing `noisevault` imports
no framework.

```bash
pip install "noisevault[qiskit] @ git+https://github.com/dvgyl/noisevault"
```

Every export takes `unknown_gates`. With `"typical"` (the default), a gate the profile does not
calibrate gets the noise of the typical native gate of its arity, with a warning. With
`"error"`, the export raises `MissingCalibrationError`. [Conventions](conventions.md) defines that
rule, the channels and the readout matrix.

A gate that the profile does not define takes the calibration of a defined gate that it equals.
`sx` and `x` take `rx`, and `zz` takes `rzz`. The fixed phase gates `z`, `s`, `t` and their
inverses take `p`, which they equal exactly. If the profile does not define `p`, these gates take
`rz`, which they equal up to a global phase. `u1` follows the same order, and `p` itself falls
back to `rz`. Cirq, PennyLane, Stim and the `nv check` reference share this rule. The Qiskit
transpiler first compiles Qiskit circuits to the profile's natives, so for Qiskit the transpiler
picks the gate.

## Scale the errors a calibration leaves out

A profile can set `unmodeled_error`, which holds factors on its gate and readout error rates.
[Profile format](profile-format.md#unmodeled-error) describes the field. Every export applies
the factors, and so does the reference that `nv check` compares with. To see what twice the
gate error does, copy a profile with a factor of 2:

```python
import stim

import noisevault as nv

fez = nv.load("ibm_fez")
doubled = fez.model_copy(update={"unmodeled_error": {"gates": {"factor": 2.0}}})
code = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=3)
noisy = doubled.to_stim(code, layout=doubled.suggest_layout(code.num_qubits))
print(noisy.report.summary().splitlines()[1])
# unmodeled error: gate errors x2; T1, T2 and preparation error are not scaled
```

The report states the factors on its second line, and `report.to_dict()` holds the same text
under `unmodeled_error`. The line also names each error that the factors leave as stated. T1,
T2 and preparation error never scale. Neither does an error with no valid power, such as a
readout pair that is no better than chance. The report of a profile without `unmodeled_error`
has no such line.

## Choose qubits

Profiles number physical qubits from 0. Integer circuit qubits map to the same physical qubit
unless you pass a `layout`. To run a small circuit on good qubits, ask the profile for a chain:

```python
import noisevault as nv

fez = nv.load("ibm_fez")
print(fez.suggest_layout(4))
# {0: 136, 1: 143, 2: 142, 3: 141}
```

`suggest_layout(n)` returns a connected chain of n enabled qubits that can measure, with low
summed gate and readout error. `noisevault.layout.suggest_layout(profile, n, usable_pair=f)`
also does not make qubits `a < b` neighbors in the chain when `f(a, b)` is false.
`suggest_layout` prefers complete qubits. A complete qubit has every single-qubit native, apart
from the identity, that is usable on at least one qubit of the device. The records that
enable or disable a gate decide where it is usable. `suggest_layout` is a starting point, not a
placer. A layout onto a disabled or missing qubit raises `LayoutError` with the fix.

## Qiskit

`to_qiskit()` builds an Aer simulator with a Qiskit `Target`. The `Target` holds every native
gate on every allowed locus, with the error and duration that the simulator applies.
`transpile(circuit, sim)` therefore compiles to the device's natives, routes around disabled
gates and places circuits by noise. The `Target` also holds `measure`, `delay` and `reset` on
every enabled qubit, but not on a qubit where the profile disables them. `transpile` then raises
`TranspilerError` for a circuit that needs one of them there, and `sim.run` raises
`DisabledGateError`. Scheduling adds no `delay` on such a qubit. Each `reset` has the profile's
reset duration, so `scheduling_method="alap"` can schedule circuits with resets. `to_qiskit()`
keys the noise model on physical qubits.

```python
from qiskit import QuantumCircuit, transpile

import noisevault as nv

sim = nv.load("ibm_fez").to_qiskit()
ghz = QuantumCircuit(3)
ghz.h(0)
ghz.cx(0, 1)
ghz.cx(1, 2)
ghz.measure_all()
counts = sim.run(transpile(ghz, sim, seed_transpiler=1), shots=2000).result().get_counts()
print(counts)
print(sim.report.summary())
```

`sim.run` accepts only circuits already in native gates on allowed loci. Any other circuit raises
`CircuitNotNativeError` with the `transpile` call that fixes the circuit. You can inspect
`sim.target`, `sim.noise_model` and `sim.profile`.

What the report can list:

- **Exact.** Gate noise as an Aer `QuantumError` per native and physical locus. Readout as
  P(1|0) and P(0|1) per qubit with Aer's `ReadoutError`. Thermal relaxation during `delay`.
  A bit flip with the preparation error after each `reset`, when the profile has one.
- **Approximated.** `to_qiskit()` clamps T2 values above 2 T1. It exports natives
  with no Qiskit instruction of their own under the gate that contains them. `zz` becomes `rzz`,
  and `ms` becomes `rxx`. The alias gets the native's noise at any angle. Google's `sqrt_iswap`
  is an instruction of its own, but Qiskit's transpiler reaches `sqrt_iswap` only through `cx`.
  Each `cx` takes two `sqrt_iswap`. A general two-qubit block therefore costs six `sqrt_iswap`
  where three are enough. The report states this cost. The initial state is ideal. When the
  profile has a preparation error, the report notes that the initial state is ideal.
- **Omitted.** Idle time outside explicit delays. Transpile with `scheduling_method="alap"` to
  insert delays on idle qubits. Effects. `to_qiskit()` leaves natives that Qiskit cannot target,
  such as Google's `sycamore`, out of the simulator. The report names these natives.
- **Unknown.** Values the profile lacks. The bundled IBM profiles have no preparation error, so
  resets add none. A `delay` on a qubit with no T1, T2 or dephasing rate adds no noise, and the
  report names the qubit.

`readout=False` leaves measurements noiseless. The `Target` then gives `measure` an error of 0,
so `transpile` does not place circuits by a readout error that the simulator does not apply.

Qiskit Aer 0.17.0 to 0.17.2 can stop Python with a segmentation fault or a bus error during
`sim.run`. The fault occurs with the `statevector` method, which Aer selects for small shot
counts and for wide circuits. With this method, Aer converts each gate error to Kraus operators
with the LAPACK routine `zheevx`. For some valid noise channels, `zheevx` does not converge.
A change of a few parts per million in one T1 value can cause or remove the fault. The
`density_matrix` method does not use this conversion. To avoid the fault, run
`sim.run(transpile(ghz, sim), shots=1000, method="density_matrix")`. The `density_matrix` method
needs 16 x 4^n bytes for n qubits, for example 16 MiB for 10 qubits. Aer issue
[#2455](https://github.com/Qiskit/qiskit-aer/issues/2455) records the `zheevx` failure.

On a profile with disabled qubits or gates, transpile with
`initial_layout=list(profile.suggest_layout(n).values())`. Qiskit's `optimization_level=0`
places circuit qubit i on physical qubit i. Levels 1 to 3 do not check that a qubit has the
single-qubit gates a circuit needs. `suggest_layout` does check. It skips a qubit that lacks a
single-qubit native other qubits have, so the chain has the same basis as the rest of the
device. `suggest_layout` skips such a qubit even when the qubit's other gates could make the
missing one. For example, `suggest_layout` skips an IBM qubit without `x`, although `rz` and
`sx` can make `x`. When no chain of n complete qubits exists, `suggest_layout` uses as few
incomplete qubits as it can. It then warns and names those qubits and their missing gates. The
transpiler can fail on those qubits.

## Cirq

`to_cirq()` returns a noise model that follows every gate with its channels as
`cirq.KrausChannel` operations on the same qubits.

```python
import cirq

import noisevault as nv

model = nv.load("quantinuum_h1-1").to_cirq()
a, b = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.PhasedXPowGate(phase_exponent=-0.5, exponent=0.5).on(a),  # the native r gate
    (cirq.ZZ**0.5).on(a, b),  # the native zz gate
    cirq.measure(a, b, key="m"),
)
result = cirq.DensityMatrixSimulator(noise=model, seed=1).run(circuit, repetitions=2000)
print(result.histogram(key="m"))
print(model.report.summary())
```

`LineQubit(i)` is device qubit i. `GridQubit(r, c)` is the qubit whose `coords` are `[r, c]`. Google
profiles record `coords`. If two enabled qubits have the same `coords`, give `layout`. Other qubit
types need `layout={qubit: index, ...}`. Do not use both `i` and `cirq.LineQubit(i)` as keys in one
layout. They name the same qubit, so the model raises `LayoutError`. Gates match by Cirq class and
exponent. `cirq.X**0.5` is `sx`, `cirq.ZZ**0.5` is `zz`, and `cirq.Z**t` is the phase gate
`p`, or `s`, `t` or their inverses at those exponents. The model splits `cirq.PhasedXZGate` into the
`r` gate and `cirq.Z**z`. Each part gets the noise that it would get on its own.

What the report can list:

- **Exact.** Gate noise as Kraus channels after each gate. Readout error, as a `confusion_map`
  on mid-circuit measurements and the equivalent channel before terminal ones. The preparation
  error after each reset. Thermal relaxation during `cirq.WaitGate`.
- **Approximated.** The state after a terminal measurement includes the readout flips. Sampled
  results are exact. To inspect states, build the model with `readout=False`.
- **Omitted.** The preparation error of the initial state, idle time outside `WaitGate`, and the
  profile's leakage effects.

Classically controlled operations raise an error that says how to restructure the circuit. An
operation on a qubit where the profile disables it raises `DisabledGateError`. Measurements,
`cirq.reset` and `cirq.WaitGate` use the profile's `measure`, `reset` and `delay` entries.

## PennyLane

`to_pennylane()` returns a `qml.NoiseModel`. Apply it to a QNode on `default.mixed` with
`qml.add_noise`. Gradients flow through the noise channels.

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
print(noisy(0.3), qml.grad(noisy)(qml.numpy.array(0.3, requires_grad=True)))
print(model.report.summary())
```

Integer wire i maps to device qubit i. Other wire labels need `layout`. A measurement without
wires, such as `qml.probs()`, reads every device wire. The model checks those wires against the
layout and the `measure` entry, as for `wires=`. `qml.add_noise` on a tape cannot see the device.
The model then refuses a measurement without wires if a wire that the circuit does not use can
map to a qubit that cannot measure. A measurement without wires gets readout error only on the
wires that the circuit's operations touch. To give every wire readout error, pass `wires=` to
the measurement.

What the report can list:

- **Exact.** Gate errors. Readout errors, applied before each measurement in its measured
  basis.
- **Approximated.** The initial state is ideal. State preparation with `BasisState`,
  `StatePrep`, `QubitDensityMatrix`, `AmplitudeEmbedding` or `BasisEmbedding` is noiseless.
  Templates that prepare a state with gates, such as `MottonenStatePreparation`, get noise like
  other templates. `qml.add_noise` at its default `level="user"` noises `qml.adjoint` gates and
  templates through their decomposition. To noise `Adjoint(SX)`, `Adjoint(S)` and `Adjoint(T)`
  as the profile's `sxdg`, `sdg` and `tdg`, pass `level="top"`. Operator arithmetic, such as
  `qml.prod`, `@`, `qml.pow`, `qml.exp` or `qml.ctrl`, gets the noise of the gates it
  decomposes into, after the whole operator. An operator with its own gate name, such as
  `qml.CNOT` or `qml.CRX`, gets noise as one gate. The basis rotation
  before a Pauli measurement is ideal. Gates conditioned on mid-circuit measurements get their
  noise whether or not the condition holds. A measurement without wires gets readout error on
  the wires that the circuit's operations and measurements use.
- **Omitted.** Idle time, because PennyLane circuits have no timing. Readout on mid-circuit
  measurements. Readout on observables not measured in one product basis. Readout on
  `qml.classical_shadow` and `qml.shadow_expval`, which pick a random measurement basis for each
  shot after the noise model acts. Effects.

Operator arithmetic with no decomposition into gates, such as `qml.sum`, raises an error. It
has no gate noise, and `default.mixed` cannot run it. Before PennyLane 0.45, a QNode leaves
`qml.sum`, `qml.Hamiltonian` and `qml.s_prod` off its tape, so the circuit runs without them
and the model never gets them.

When readout noise is on, `qml.add_noise` keeps only part of a shot vector's results
(`shots=[100, 200]`). The model therefore raises an error for shot vectors instead of returning
wrong numbers. Run each shot count separately. You can also pass `readout=False`.

With shots, `default.mixed` gives the same shots to Pauli words that commute on each wire. Such
words have the same Pauli letter on each wire that they share, for example
`qml.sample(qml.Z(0))` and `qml.sample(qml.X(1))`. The model gives these measurements one set of
readout operations, so they stay on one tape and their samples stay correlated. Two words that
read one wire in different bases, such as `qml.Z(0)` and `qml.X(0)`, get separate shots. If a
third word commutes with both, `default.mixed` decides which shots that word shares. The model
cannot see that decision, so the model raises an error. To fix the error, wrap the QNode in
`qml.transforms.split_non_commuting` before `qml.add_noise`. `default.mixed` also puts
`qml.probs(op=...)` of an identity or zero observable, such as `qml.I(0) @ qml.I(1)`, in a group
of commuting words. Those probabilities then show the basis of that group, and the model gives
them the readout operations of that group.

The model simplifies each Pauli observable as PennyLane does. The model then selects the measured
basis, the words that share shots and the wires that a measurement reads. The simplification
removes each word whose coefficient is at most 1e-8. For example,
`qml.expval(qml.X(0) + 0 * qml.Y(1))` reads only wire 0, in the X basis. Thus the model checks
the `measure` entry only on wire 0. `qml.probs(op=...)` gives outcomes on every wire of its
observable, so the model checks the `measure` entry on each of those wires.

PennyLane has no operation named after the `r`, `zz` and `ms` natives of trapped-ion profiles.
A `qml.Rot(a, theta, -a)` gets the profile's `r` noise, and `qml.IsingZZ(pi/2)` gets its `zz`
noise. On a profile with an `ms` native, `qml.IsingXX(±pi/2)` and `qml.IsingYY(±pi/2)` get its
`ms` noise. Other angles, and gates the profile has no native for, get typical noise with a
warning, and the report counts each use. A broadcast operation gets one noise channel for all
its elements. The model therefore raises an error when the operation's angles need the noise of
different gates, such as `qml.IsingXX` over `[pi/2, 0.4]`. Expand the broadcast before you add
noise, with `qml.add_noise(qml.transforms.broadcast_expand(qnode), model)`.

The model checks every wire that a circuit uses against the layout and the profile, also with
`readout=False`. The check includes wires that the circuit only measures. A model built by adding
or subtracting noise models checks only the wires its operations or readout reach.

A gate, a measurement or a reset on a qubit where the profile disables it raises
`DisabledGateError`. Measurements, `qml.measure` included, use the profile's `measure` entry.
`qml.measure(0, reset=True)` also uses its `reset` entry. PennyLane has no delay operation, so
the `delay` entry has no effect.

## Stim

`to_stim(circuit)` returns a copy of a Stim circuit with each gate followed by the Pauli twirl
of its channel. The twirl is a `PAULI_CHANNEL_1` or `PAULI_CHANNEL_2`, or on three or more qubits
a chain of `CORRELATED_ERROR` and `ELSE_CORRELATED_ERROR`. Annotations, `REPEAT` blocks,
detectors and observables pass through, so decoders such as PyMatching work on the result.

```python
import stim

import noisevault as nv

fez = nv.load("ibm_fez")
code = stim.Circuit.generated("repetition_code:memory", distance=5, rounds=5)
noisy = fez.to_stim(code, layout=fez.suggest_layout(code.num_qubits))
dem = noisy.detector_error_model()
shots = noisy.compile_detector_sampler(seed=1).sample(10_000)
print(dem.num_errors, shots.mean())
print(noisy.report.summary())
```

`CX` is not a Fez native, so here `CX` gets the noise of `cz` on the same pair, with a warning.
The report also states this substitution. For a grid device,
`noisevault.stim.layout_from_coords(circuit, profile)` places a circuit by matching its
`QUBIT_COORDS` to the profile's qubit coords. If a match uses coords that two enabled qubits
have, the function raises `LayoutError`. Then give `layout=` to `to_stim`.

An instruction on a qubit where the profile disables it raises `DisabledGateError`. Measurements
(`M`, `MX`, `MY`, `MPP`, `MXX` and the others) use the profile's `measure` entry. Resets (`R`,
`RX` and `RY`) use its `reset` entry. `MR`, `MRX` and `MRY` use both entries. Stim has no delay
instruction, so the `delay` entry has no effect.

In `MPP` and `SPP`, the export first reduces each Pauli product. Pauli factors on one qubit
multiply. A qubit whose Pauli factors cancel is neither read out nor busy. A product that reduces
to the identity gets no readout flip.

Options:

- `readout="symmetrize"` (default) flips each result with the mean of P(1|0) and P(0|1).
  `readout="exact"` keeps measurements perfect inside the circuit and stores the asymmetric
  error for `noisevault.stim.sample_with_readout(noisy, shots)`.
  `readout="none"` adds no readout error.
- `tick_ns=` is the duration of one `TICK` layer. Qubits idle in a layer get twirled relaxation
  for that time.
- `existing_noise` decides what happens to noise already in the circuit. `"error"` (default)
  raises an error, `"keep"` keeps the noise and `"strip"` removes the noise.

What the report can list:

- **Exact.** Readout error, only with `readout="exact"` and `sample_with_readout`.
- **Approximated.** Gate noise is the Pauli twirl of each gate's channel. The twirl keeps each
  gate's average fidelity and drops relaxation's bias toward |0>. `detector_error_model()`
  treats the Pauli channel components as independent (Stim's `approximate_disjoint_errors`, on
  by default here). Symmetric readout. Idle noise per `TICK` when you set `tick_ns`.
- **Omitted.** The initial preparation error, idle noise without `tick_ns`, and effects.

Stim simulates Clifford circuits only. Write non-Clifford circuits for one of the other three
frameworks.

## Check a conversion

`nv check REF` runs small circuits through each installed export and compares the results with
NoiseVault's own density-matrix reference simulator. Use `nv check` after you change a profile by
hand. `nv check` samples the Stim export with exact readout. It compares the samples with the
reference after each gate's Pauli twirl. `nv check` also samples two Stim circuits with the default
symmetrized readout. These circuits are the widest circuit and a circuit that only measures.
`nv check` compares those samples with a reference that uses each qubit's mean readout error. The
check needs the measurement-only circuit because readout error leaves a uniform distribution
unchanged. When the profile calibrates `p`, the check also runs a fixed phase gate that the profile
does not define, such as `s`. Every export must charge that gate as `p`.

The `chain_mirror` circuit runs the chain of 2-qubit gates and then its inverse, two times. The
first time, only the first qubit starts in a superposition. The second time, every qubit starts in
a superposition. Thus `chain_mirror` finds 2-qubit errors that the other circuits do not show. An
example is an X error on the target qubit of `cx` after `h`. The circuits of a hardware run, from
`noisevault.counts.plan()`, do not include `chain_mirror`.

The `deviation` and `tolerance` columns show the circuit that is nearest to its tolerance, or
furthest past it. For an exact framework, the deviation is the TVD from the reference, and the
tolerance is 1e-9. A sampled check compares the frequency of each outcome with the reference
probability of that outcome. Each outcome has its own tolerance, which is 5 times the standard
error of its frequency plus 5/shots. The deviation is the difference of the outcome with the
largest difference relative to its tolerance. The tolerance column then shows the tolerance of
that outcome. A sampled circuit passes when every outcome is within its tolerance. Thus a row
shows FAIL only when its deviation is larger than its tolerance. `--json` gives the `deviation`
and the `tolerance` of each circuit. Under each framework, `reports` holds one full export report
for each export configuration that the check ran. Each report names every affected qubit.

A framework that cannot express one of a circuit's gates runs the circuit without that gate. It
skips the circuit when nothing useful remains. The `circuits` column counts only circuits that
ran whole, for example `5 of 6, 1 reduced`. A line under the table names each gate left out and
why, such as `stim: two_qubit_natives ran without rxx, ryy, rzz`. A pass covers only the gates
that ran. `--json` lists the same information under each framework's `not_run`. There,
`ran_without` names the gates that a reduced circuit left out.
