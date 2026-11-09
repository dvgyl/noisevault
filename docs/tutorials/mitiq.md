# Mitiq: ZNE and PEC on real device noise

This tutorial mitigates the errors of a small circuit on the calibrated noise of IBM Fez. It
uses two methods from [Mitiq](https://mitiq.readthedocs.io):

- **Zero-noise extrapolation (ZNE)** runs the circuit at higher noise levels and extrapolates
  the result back to zero noise.
- **Probabilistic error cancellation (PEC)** writes each ideal gate as a linear combination of
  noisy circuits. The combination is the *representation* of the gate. PEC samples circuits
  from the representations and combines their results.

PEC needs a noise model for each gate. Users often guess one number, such as a depolarizing
noise level, for all gates. Here, each representation comes from the profile's error for that
gate on those qubits. At the end, the tutorial compares each mitigated value with the ideal
value.

The full script is [examples/mitiq_zne_pec.py](../../examples/mitiq_zne_pec.py). Each code
block below is a part of that script.

## Install

Mitiq supports Python up to 3.12. This tutorial uses Cirq, so it does not need `ply`:

```bash
pip install "noisevault[cirq] @ git+https://github.com/dvgyl/noisevault" mitiq
```

## 1. Load the noise model

The script pins the bundled IBM Fez calibration by its fingerprint. `suggest_layout(2)` picks
two connected qubits with low error. `readout=False` leaves out readout error, because ZNE and
PEC mitigate gate errors. Mitiq has a separate method for readout error.

<!-- not-run: a part of examples/mitiq_zne_pec.py -->
```python
fez = nv.load("ibm_fez@2025-02-26", expect="nv:06404cefa54f")
a, b = cirq.LineQubit.range(2)
layout = dict(zip((a, b), fez.suggest_layout(2).values(), strict=True))
noise = fez.to_cirq(layout=layout, readout=False)
```

## 2. Build the circuit and the executor

The circuit has four layers of the Fez native gates `rz`, `sx` and `cz`. The observable is
`<ZZ>`, the expectation value of Z on both qubits.

<!-- not-run: a part of examples/mitiq_zne_pec.py -->
```python
rng = np.random.default_rng(5)
circuit = cirq.Circuit()
for _ in range(4):
    circuit += [cirq.rz(rng.uniform(0, 2 * np.pi)).on(q) for q in (a, b)]
    circuit += [(cirq.X**0.5).on(q) for q in (a, b)]
    circuit += cirq.CZ(a, b)
```

Mitiq adds gates that Fez does not have. ZNE adds the inverse of `sx`, and PEC adds `Y`. A
compiler writes these gates in native gates, so the executor does the same. Without this step,
the noise model gives each such gate the noise of a typical gate and prints a warning.

<!-- not-run: a part of examples/mitiq_zne_pec.py -->
```python
def native(circuit: cirq.Circuit) -> cirq.Circuit:
    """Write sxdg as Z sx Z and Y as Z then X, as a compiler does: Fez has neither gate."""

    def compiled(op: cirq.Operation) -> list[cirq.Operation]:
        q = op.qubits[0]
        if op.gate == cirq.X**-0.5:
            return [cirq.Z(q), (cirq.X**0.5).on(q), cirq.Z(q)]
        if op.gate == cirq.Y:
            return [cirq.Z(q), cirq.X(q)]
        return [op]

    return cirq.Circuit(compiled(op) for op in circuit.all_operations())
```

The executor is the function that Mitiq calls to run a circuit. This one runs Cirq's
density-matrix simulator, so each value is exact and has no shot noise. The only random part
of the results is the sampling of PEC itself.

<!-- not-run: a part of examples/mitiq_zne_pec.py -->
```python
ZZ = np.diag([1.0, -1.0, -1.0, 1.0])


def expectation(circuit: cirq.Circuit, noise_model: cirq.NoiseModel | None = noise) -> float:
    """The exact <ZZ> of the final state of the circuit under the noise model."""
    simulator = cirq.DensityMatrixSimulator(noise=noise_model)
    rho = simulator.simulate(native(circuit), qubit_order=[a, b]).final_density_matrix
    return float(np.real(np.trace(rho @ ZZ)))
```

## 3. Get each gate's error from the profile

`fez.pauli_error(gate, qubits)` returns the error that the exports apply after a gate, as Pauli
error probabilities. Character i of each label acts on `qubits[i]`:

```text
>>> fez.pauli_error("sx", [96])
{'X': 6.67e-05, 'Y': 6.67e-05, 'Z': 0.000117}
```

The values above are rounded to three digits.

The Z error is larger than the X and Y errors. The profile's T2 time causes this dephasing.
NoiseVault gets the probabilities from the gate's channel: depolarizing noise, then thermal
relaxation over the gate's duration. A *Pauli twirl* keeps the Pauli part of that channel. The
twirl drops the rest, such as the pull of amplitude damping toward `|0>`.

<!-- not-run: a part of examples/mitiq_zne_pec.py -->
```python
def profile_gate(op: cirq.Operation) -> str:
    """The name of the profile gate that runs op."""
    return "rz" if isinstance(op.gate, cirq.Rz) else {cirq.X**0.5: "sx", cirq.CZ: "cz"}[op.gate]


def pauli_error(op: cirq.Operation) -> dict[str, float]:
    return fez.pauli_error(profile_gate(op), [layout[q] for q in op.qubits])
```

## 4. Build the PEC representations

The script builds two kinds of representation from the same profile.

**Depolarizing.** Mitiq's `represent_operation_with_global_depolarizing_noise` takes a
`noise_level`. This value is the total Pauli error probability of the gate, which is the sum of
the values from `pauli_error`. An IBM `gate_error` is an average infidelity r, a different
number. On k qubits, the total Pauli error is r (d + 1) / d with d = 2^k. For example, it is
1.5 r on one qubit and 1.25 r on two qubits.

<!-- not-run: a part of examples/mitiq_zne_pec.py -->
```python
def depolarizing(op: cirq.Operation) -> pec.OperationRepresentation:
    """The representation that Mitiq builds for depolarizing noise of the total Pauli error."""
    noise_level = sum(pauli_error(op).values())
    return represent_operation_with_global_depolarizing_noise(cirq.Circuit(op), noise_level)
```

**Pauli channel.** This representation inverts each gate's Pauli channel exactly, so it keeps
the larger Z error. A Pauli channel with probabilities p_P changes each Pauli operator Q only by
a factor f_Q. f_Q is the sum of p_P over the P that commute with Q, minus the sum over the P
that do not. The inverse channel divides by f_Q. As a sum of Pauli operations, the inverse has
the coefficients q_P = 4^-k sum over Q of (+1 or -1) / f_Q. The sign is +1 when P and Q
commute. `inverse_coefficients` gives the q_P. Each term of the representation is the noisy
gate, then the Pauli operation P, with the coefficient q_P. The helper `commute` tells if two
Pauli labels commute, and `pauli` gives the gates of a label. Both are in the script.

<!-- not-run: a part of examples/mitiq_zne_pec.py -->
```python
def inverse_coefficients(error: dict[str, float]) -> dict[str, float]:
    """The coefficient of each Pauli operation in the inverse of the Pauli channel error."""
    labels = ["I" * len(next(iter(error))), *error]
    p = np.array([1.0 - sum(error.values()), *error.values()])
    sign = np.array([[1.0 if commute(s, t) else -1.0 for t in labels] for s in labels])
    eigenvalues = sign @ p
    return dict(zip(labels, sign @ (1.0 / eigenvalues) / len(labels), strict=True))


def pauli_channel(op: cirq.Operation) -> pec.OperationRepresentation:
    """The representation that inverts the Pauli channel of the profile exactly."""
    quasi = inverse_coefficients(pauli_error(op))
    noisy = [pec.NoisyOperation(cirq.Circuit(op, pauli(label, op.qubits))) for label in quasi]
    return pec.OperationRepresentation(cirq.Circuit(op), noisy, list(quasi.values()))
```

The script builds one representation for each operation in the circuit. An `rz` gate has no
error, so its representation is the gate alone.

## 5. Run ZNE and PEC

ZNE folds the whole circuit to scale the noise by 1, 3 and 5. Richardson extrapolation then
fits these three points. PEC samples 4000 circuits. Most samples are the circuit with no added
gate, and `force_run_all=False` runs each distinct circuit only once.

<!-- not-run: a part of examples/mitiq_zne_pec.py -->
```python
def run_pec(build) -> tuple[float, float]:
    """The PEC value and its standard error, with one representation per operation."""
    representations = [build(op) for op in dict.fromkeys(circuit.all_operations())]
    value, data = pec.execute_with_pec(
        circuit,
        expectation,
        representations=representations,
        num_samples=SAMPLES,
        force_run_all=False,
        random_state=7,
        full_output=True,
    )
    return value, data["pec_error"]
```

## 6. Compare with the ideal value

`python examples/mitiq_zne_pec.py` prints:

```text
qubits 96 and 103 of ibm_fez, 12 moments
ideal  <ZZ> = +0.4604
noisy  <ZZ> = +0.4476  error 0.0128
ZNE    <ZZ> = +0.4604  error 0.0001  (Richardson, 1, 3, 5)
PEC    <ZZ> = +0.4614  error 0.0010  (depolarizing, 4000 samples, standard error 0.0015)
PEC    <ZZ> = +0.4636  error 0.0032  (Pauli channel, 4000 samples, standard error 0.0014)
noise: ibm_fez@2025-02-26 nv:06404cefa54f
```

- ZNE removes almost all of the error on this circuit.
- Both PEC values are within about two standard errors of the ideal value. The standard error
  comes from the sampling of PEC, and it falls as 1 / sqrt(samples). Another random state gives
  another value within a few standard errors.
- With this little noise, the two representations give the same result within their standard
  errors.

## Limits

- The simulator applies amplitude damping. A Pauli representation cannot invert this part of
  the noise, so PEC can keep a bias from it.
- The simulator also applies noise to the Pauli gates that PEC adds, and the representations
  leave this noise out. A gate gets an added Pauli gate with a probability near its error rate,
  so this bias is near the product of two error rates.
- The noise model adds no noise to idle qubits and no readout error, so the comparison shows
  only gate errors. On a device, use both ZNE or PEC and readout-error mitigation.
- PEC is unbiased only if the representations match the device. A profile is one calibration on
  one date. Use `nv pull` for a newer calibration, and see
  [Unmodeled error](../profile-format.md#unmodeled-error) for the error that a calibration
  leaves out.

## Next steps

- [Mitigate errors with Mitiq](../recipes.md#mitigate-errors-with-mitiq) runs ZNE through
  Qiskit.
- [Frameworks](../frameworks.md#cirq) describes the Cirq noise model and its report.
