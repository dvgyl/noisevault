"""ZNE and PEC from Mitiq on the calibrated noise of IBM Fez, compared with the ideal value.

Zero-noise extrapolation (ZNE) runs the circuit at larger noise and extrapolates back to zero
noise. Probabilistic error cancellation (PEC) needs a representation of each noisy gate: a
linear combination of noisy circuits that equals the ideal gate. This example builds each
representation from the profile's gate errors with Profile.pauli_error, so PEC needs no guessed
noise level.

The circuit runs on two qubits of the bundled ibm_fez profile, on Cirq's density-matrix
simulator, so each expectation value is exact and the only sampling is PEC's own.
docs/tutorials/mitiq.md explains each step.

Needs:
    pip install "noisevault[cirq] @ git+https://github.com/dvgyl/noisevault" mitiq
"""

import sys

try:
    from mitiq import pec, zne
    from mitiq.pec.representations import represent_operation_with_global_depolarizing_noise
except ImportError:
    print("this example needs Mitiq: pip install mitiq")
    sys.exit(1)

import cirq
import numpy as np

import noisevault as nv

# 1. The noise model: two neighbouring qubits of IBM Fez, with no readout error.
fez = nv.load("ibm_fez@2025-02-26", expect="nv:06404cefa54f")
a, b = cirq.LineQubit.range(2)
layout = dict(zip((a, b), fez.suggest_layout(2).values(), strict=True))
noise = fez.to_cirq(layout=layout, readout=False)

# 2. The circuit: four layers of the Fez native gates rz, sx and cz.
rng = np.random.default_rng(5)
circuit = cirq.Circuit()
for _ in range(4):
    circuit += [cirq.rz(rng.uniform(0, 2 * np.pi)).on(q) for q in (a, b)]
    circuit += [(cirq.X**0.5).on(q) for q in (a, b)]
    circuit += cirq.CZ(a, b)


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


ZZ = np.diag([1.0, -1.0, -1.0, 1.0])


def expectation(circuit: cirq.Circuit, noise_model: cirq.NoiseModel | None = noise) -> float:
    """The exact <ZZ> of the final state of the circuit under the noise model."""
    simulator = cirq.DensityMatrixSimulator(noise=noise_model)
    rho = simulator.simulate(native(circuit), qubit_order=[a, b]).final_density_matrix
    return float(np.real(np.trace(rho @ ZZ)))


# 3. A PEC representation of each operation, from the profile's Pauli error of its gate.
def profile_gate(op: cirq.Operation) -> str:
    """The name of the profile gate that runs op."""
    return "rz" if isinstance(op.gate, cirq.Rz) else {cirq.X**0.5: "sx", cirq.CZ: "cz"}[op.gate]


def pauli_error(op: cirq.Operation) -> dict[str, float]:
    return fez.pauli_error(profile_gate(op), [layout[q] for q in op.qubits])


def depolarizing(op: cirq.Operation) -> pec.OperationRepresentation:
    """The representation that Mitiq builds for depolarizing noise of the total Pauli error."""
    noise_level = sum(pauli_error(op).values())
    return represent_operation_with_global_depolarizing_noise(cirq.Circuit(op), noise_level)


PAULI_GATES = {"I": [], "X": [cirq.X], "Y": [cirq.Z, cirq.X], "Z": [cirq.Z]}  # Y is XZ


def pauli(label: str, qubits: tuple[cirq.Qid, ...]) -> list[cirq.Operation]:
    """The gates of the Pauli string label, with character i on qubits[i]."""
    pairs = zip(label, qubits, strict=True)
    return [gate.on(q) for c, q in pairs for gate in PAULI_GATES[c]]


def commute(p: str, q: str) -> bool:
    """True when the Pauli strings p and q commute."""
    return sum(x != "I" and y != "I" and x != y for x, y in zip(p, q, strict=True)) % 2 == 0


def pauli_channel(op: cirq.Operation) -> pec.OperationRepresentation:
    """The representation that inverts the Pauli channel of the profile exactly."""
    error = pauli_error(op)
    labels = ["I" * len(op.qubits), *error]
    p = np.array([1.0 - sum(error.values()), *error.values()])
    sign = np.array([[1.0 if commute(s, t) else -1.0 for t in labels] for s in labels])
    eigenvalues = sign @ p
    quasi = sign @ (1.0 / eigenvalues) / len(labels)
    noisy = [pec.NoisyOperation(cirq.Circuit(op, pauli(label, op.qubits))) for label in labels]
    return pec.OperationRepresentation(cirq.Circuit(op), noisy, list(quasi))


# 4. The ideal, noisy, ZNE and PEC values.
SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 4000  # a test passes fewer


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


ideal = expectation(circuit, noise_model=None)
noisy = expectation(circuit)
factory = zne.inference.RichardsonFactory(scale_factors=[1, 3, 5])
zne_value = zne.execute_with_zne(
    circuit, expectation, factory=factory, scale_noise=zne.scaling.fold_global
)
print(f"qubits {layout[a]} and {layout[b]} of {fez.id}, {len(circuit)} moments")
print(f"ideal  <ZZ> = {ideal:+.4f}")
print(f"noisy  <ZZ> = {noisy:+.4f}  error {abs(noisy - ideal):.4f}")
print(f"ZNE    <ZZ> = {zne_value:+.4f}  error {abs(zne_value - ideal):.4f}  (Richardson, 1, 3, 5)")
for label, build in (("depolarizing", depolarizing), ("Pauli channel", pauli_channel)):
    value, error = run_pec(build)
    print(
        f"PEC    <ZZ> = {value:+.4f}  error {abs(value - ideal):.4f}"
        f"  ({label}, {SAMPLES} samples, standard error {error:.4f})"
    )
print(f"noise: {fez.id}@{fez.device.calibrated_at:%Y-%m-%d} {fez.short_fingerprint}")
