# Conventions

This page defines the numbers in a profile and how NoiseVault turns them into channels. Every
export follows these rules, so a gate on a given qubit gets the same channel in Qiskit, Cirq,
PennyLane and the reference simulator. Stim gets the Pauli twirl of the same channel.

## Error metrics

Vendors report gate errors in different metrics, and the same number means different noise
under each. For an n-qubit gate with d = 2^n:

| Metric | Profile key | Definition |
| --- | --- | --- |
| Average gate infidelity r | `avg_infidelity` | r = 1 - F_avg, with F_avg the fidelity averaged over pure input states. Randomized benchmarking reports r. |
| Process infidelity e | `process_infidelity` | e = 1 - F_pro, with F_pro the entanglement fidelity. |
| Depolarizing parameter lambda | `depolarizing_param` | The channel (1 - lambda) rho + lambda I/d, as in Qiskit's `depolarizing_error(lambda, n)`. |
| Pauli probabilities | `pauli` | p_P for each non-identity Pauli P. Their sum is the total Pauli error. |

These equations convert between the metrics:

- F_avg = (d F_pro + 1) / (d + 1), so e = r (d + 1) / d.
- r = lambda (d - 1) / d.
- For a Pauli channel, e = sum of p_P.
- A depolarizing channel spreads e evenly, so p_P = e / (d^2 - 1) for each of the d^2 - 1
  Paulis.
- Cirq's `depolarize(p)`, Stim's `DEPOLARIZE1(p)` and PennyLane's `DepolarizingChannel(p)` take
  p = e, the total Pauli error.

For one qubit (d = 2), r = 1e-3 gives e = 1.5e-3, lambda = 2e-3 and p_X = p_Y = p_Z = 5e-4. For
two qubits (d = 4), r = 1e-2 gives e = 1.25e-2, lambda = 1.333e-2 and 15 Pauli terms of 8.33e-4
each. The Stim export shows these numbers directly:

```python
import stim

import noisevault as nv

toy = nv.Profile.uniform(
    "toy", technology="superconducting", num_qubits=2, one_qubit_error=1e-3, two_qubit_error=1e-2
)
print(toy.to_stim(stim.Circuit("CZ 0 1\nH 0")))
# CZ 0 1
# PAULI_CHANNEL_2(0.000833333, ... 15 equal terms ...) 0 1
# H 0
# PAULI_CHANNEL_1(0.0005, 0.0005, 0.0005) 0
```

The quantity (d F_avg - 1) / (d - 1) is the randomized-benchmarking decay parameter, not the
process fidelity. If you read one as the other, the error is d^2 / (d^2 - 1) times too high. That
ratio is 4/3 for one qubit.

Three vendor numbers need care:

- IBM's two-qubit error comes from benchmarking layers that include single-qubit Cliffords.
  Importers record `"includes": ["1q_dressing"]`.
- Google's XEB "Pauli error per cycle" is the process error of one cycle. One cycle is a random
  single-qubit gate on each qubit, followed by the entangler. The Google importer subtracts the
  single-qubit part, as Google does to infer a per-gate error, and stores the result as
  `process_infidelity`.
- IonQ publishes fidelities without naming the metric. The IonQ importer reads them as average
  gate fidelity and writes that reading into each gate's `assumption`.

## Channel construction

A gate with a stated average infidelity r, a duration t and qubits with T1 and T2 gets two
channels, applied after the ideal gate:

1. An n-qubit depolarizing channel with the residual strength computed below.
2. Thermal relaxation on each qubit for time t.

Thermal relaxation is zero-temperature amplitude damping with gamma = 1 - exp(-t / T1), then
pure dephasing at rate 1/T2 - 1/(2 T1). A `dephasing_rate_per_s` adds Z errors at that
rate, so the Z error probability is (1 - exp(-2 rate t)) / 2. With no T1, there is no amplitude
damping. With no T2, T2 = 2 T1 (no pure dephasing).

NoiseVault solves for the depolarizing part so that the composed channel has the stated error.
Qiskit Aer uses the same rule for device noise. With F_relax the average gate fidelity of
relaxation alone and r_relax = 1 - F_relax:

lambda = d (r - r_relax) / (d F_relax - 1)

These edge cases change the result, and the report records each:

- **T2 clamp.** A T2 above 2 T1 is unphysical for this model. NoiseVault clamps such a T2 to
  2 T1, and the report lists "T2 of qubit q: clamped to 2*T1".
- **Relaxation floor.** If r_relax already exceeds r, NoiseVault adds no depolarizing noise, and
  the gate keeps the full relaxation. The gate is then noisier than stated.
- **Depolarizing ceiling.** If r is above the highest error that a depolarizing channel and the
  relaxation together can reach, the depolarizing part stops at that maximum. The gate is then
  less noisy than stated. That case needs an error far above any real calibration.

For the floor and the ceiling, the report's `clamped` list records each gate's requested and
achieved error.

A gate with a `pauli` spec gets exactly that Pauli channel and no relaxation, because the spec
is the whole channel. A virtual gate gets no channel.

In this example, a 500 ns gate on qubits with T1 = 10 us cannot have an error as low as 1e-3.
The gate therefore reaches the relaxation floor.

```python
import stim

import noisevault as nv

slow = nv.Profile.uniform(
    "slow",
    technology="superconducting",
    num_qubits=2,
    one_qubit_error=1e-4,
    two_qubit_error=1e-3,
    t1_us=10,
    t2_us=8,
    one_qubit_ns=50,
    two_qubit_ns=500,
)
noisy = slow.to_stim(stim.Circuit("CZ 0 1"))
clamp = noisy.report.clamped[0]
print(
    f"{clamp.gate}{list(clamp.qubits)}: requested {clamp.requested}, achieved {clamp.achieved:.4f}"
)
# cz[0, 1]: requested 0.001, achieved 0.0665
```

## Readout

A profile stores readout as a = P(1 | prepared 0) and b = P(0 | prepared 1). NoiseVault's
confusion matrix has the prepared state in columns:

```
M[measured, prepared] = [[1 - a,     b],
                         [    a, 1 - b]]
```

so measured probabilities are M p. Each framework receives the matrix in its own form:

| Framework | How the export applies readout |
| --- | --- |
| Qiskit Aer | `ReadoutError(M.T)`. Aer's rows are the prepared state. |
| Cirq | A mid-circuit measurement gets `cirq.MeasurementGate(confusion_map={...: M.T})`, whose rows are the true state. A terminal measurement gets the confusion channel immediately before the measurement, because Cirq's sampling of terminal measurements would drop a confusion map. |
| PennyLane | The confusion channel, as a `qml.QubitChannel`, before each measurement in the measured basis. Mid-circuit measurements get no readout error. |
| Stim | With `readout="symmetrize"` (the default), the export writes `M((a + b) / 2)`, a symmetric flip. With `readout="exact"`, the circuit has perfect measurements, and `noisevault.stim.sample_with_readout` then flips each recorded bit with a or b. |

The confusion channel has Kraus operators `K_mp = sqrt(M[m, p]) |m><p|` for m, p in {0, 1}. It
handles any column-stochastic M, including a + b > 1. A generalized amplitude
damping channel (GAD) can also reproduce readout error, but only when a + b <= 1. The two
frameworks define its p parameter in opposite ways. With gamma = a + b:

| Framework | Call that reproduces (a, b) |
| --- | --- |
| PennyLane | `qml.GeneralizedAmplitudeDamping(a + b, a / (a + b), wires=q)` |
| Cirq | `cirq.generalized_amplitude_damp(b / (a + b), a + b)` |

If you use PennyLane's p in Cirq, the call swaps a and b. NoiseVault does not use GAD, but the
difference matters if you build readout noise by hand.

## Preparation and idle time

- A reset gets a bit flip with probability `prep.error` after it, in every framework.
- The initial state is ideal |0...0>. Profiles do not distinguish initial preparation from
  reset, and reports list the initial state as approximated or omitted.
- The exports apply idle noise only where a circuit states a duration. Qiskit `delay`
  instructions and Cirq `WaitGate` get thermal relaxation for their duration. Stim gets
  relaxation on qubits idle in a TICK layer when you pass `tick_ns=`. PennyLane circuits have no
  timing, so they get no idle noise. Qiskit circuits get delays when you transpile with
  `scheduling_method="alap"`.

## Qubit order

Profiles number physical qubits `0..num_qubits-1`. A `layout` maps circuit qubits to physical
qubits. By default, integer label i is physical qubit i. Kraus operators inside NoiseVault are
big-endian (the first qubit is the most significant tensor factor). Each framework then reports
outcomes in its own order:

| Framework | Circuit qubits | Bit order in results |
| --- | --- | --- |
| Qiskit | Transpiled circuits act on physical qubits directly. | Little-endian. Qubit 0 is the rightmost bit of a counts key. |
| Cirq | `LineQubit(i)` is qubit i. `GridQubit(r, c)` is the qubit at coords `[r, c]`. Others need `layout=`. | The order of the qubits in `cirq.measure(...)`, first qubit first. |
| PennyLane | Integer wire i is qubit i. Other wire labels need `layout=`. | In `qml.probs(wires=[...])`, the first wire is the most significant bit. |
| Stim | Stim qubit i is qubit i. `noisevault.stim.layout_from_coords` matches `QUBIT_COORDS` to profile coords. | Measurement record order. |

## Gates the profile does not calibrate

A circuit may use a gate that the profile does not calibrate. The gate can be missing from
`gates`, such as `h` on an IBM device. The gate can also be a native with no error metric, or a
gate on a pair that the connectivity does not allow. The `unknown_gates` option decides what
happens:

- `"typical"` (the default) gives the gate the noise of the typical native gate of its arity on
  the same qubits. The typical native is the calibrated native with the most calibration
  records. NoiseVault breaks ties by name. The identity `id` is an idle slot, so NoiseVault uses
  `id` only when no other native fits. If the profile marks the typical native as disabled on
  those qubits, NoiseVault uses the next one. For a directed two-qubit native, NoiseVault can use
  the record from the reversed pair. The export warns once per gate name with a
  `NoiseApproximationWarning`. It lists the gate under "approximated" in the report and counts
  each use in the report's `typical_noise_used` events.
- `"error"` raises `MissingCalibrationError` instead.

Z-family gates (`z`, `s`, `sdg`, `t`, `tdg`, `p`, `u1`, `rz`) are free when the profile's `rz`
is virtual, so they never need the typical rule. Gates that need several native entanglers are
`swap`, `cxswap`, `swapcx`, `czswap`, `cswap` and `ccx`. Unless the profile calibrates them,
these gates and gates on more than two qubits raise `MissingCalibrationError` with either
`unknown_gates` value. Decompose them into native gates first.

The typical rule keeps an un-transpiled circuit runnable, but its gate count is not the
device's. An `h` on IBM hardware is one `sx` between two virtual `rz` gates, and a `cx` is a
`cz` with single-qubit gates around it. For realistic noise, compile to the profile's native
gates first. In Qiskit, use `transpile(circuit, sim)`. In Cirq, use a target gateset.
