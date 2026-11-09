"""Surface-code memory on a neutral-atom device whose 2-qubit gates convert errors to erasures.

The device heralds each erasure. A decoder that reads the heralds knows where the erasures
occurred, so it fails less often than a decoder that ignores them. The script exports to Stim,
then decodes with PyMatching both ways.

Needs:
    pip install "noisevault[stim] @ git+https://github.com/dvgyl/noisevault"
    pip install pymatching
"""

import numpy as np
import pymatching
import stim

import noisevault as nv

DISTANCE, SHOTS = 3, 10_000
ERASURE = 0.01  # probability that a CX erases each of its qubits

device = nv.Profile.uniform(
    "toy-atoms",
    technology="neutral_atom",
    num_qubits=30,
    one_qubit_error=1e-4,
    two_qubit_error=5e-4,
    readout_error=2e-3,
    # Erasure conversion: most 2-qubit gate failures leave the atom in a state that the
    # device detects. Stim's generated code uses CX, so the effect is on cx.
    effects=[{"type": "erasure", "gate": "cx", "prob": ERASURE, "heralded": True}],
)


def erasure_models(circuit: stim.Circuit) -> tuple[stim.DetectorErrorModel, list]:
    """The error model when no herald fires, and the extra errors of each herald that fires.

    A fired herald fully depolarizes its qubit. A silent herald adds no error. MPAD keeps each
    herald's place in the measurement record, so the detectors stay the same. A tagged
    identity marks each herald's place, because without_noise() moves the other instructions.
    """
    silent = stim.Circuit()
    for inst in circuit.flattened():
        if inst.name != "HERALDED_ERASE":
            silent.append(inst)
            continue
        for target in inst.targets_copy():
            silent.append("MPAD", [0])
            silent.append(stim.CircuitInstruction("I", [target], tag="herald"))
    clean = silent.without_noise()
    extra = []
    for at, inst in enumerate(clean):
        if inst.tag == "herald":
            erase = stim.CircuitInstruction("DEPOLARIZE1", inst.targets_copy(), [0.75])
            alone = clean[:at] + stim.Circuit(str(erase)) + clean[at + 1 :]
            extra.append(alone.detector_error_model(decompose_errors=True))
    base = silent.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    return base, extra


def herald_records(circuit: stim.Circuit) -> list[int]:
    """The index of each HERALDED_ERASE record in the measurement record, in order."""
    found, record = [], 0
    for inst in circuit.flattened():
        if inst.name == "HERALDED_ERASE":
            found += range(record, record + len(inst.targets_copy()))
        record += stim.gate_data(inst.name).produces_measurements * len(inst.targets_copy())
    return found


def logical_errors(noisy: stim.Circuit) -> tuple[float, float]:
    """Logical error rate when the decoder ignores the heralds, and when it uses them."""
    records = noisy.compile_sampler(seed=1).sample(SHOTS)
    detectors, observables = noisy.compile_m2d_converter().convert(
        measurements=records, separate_observables=True
    )
    truth = observables[:, 0]

    # Ignore the heralds: the average error model, where an erasure is depolarizing noise.
    model = noisy.detector_error_model(decompose_errors=True)
    blind = pymatching.Matching.from_detector_error_model(model).decode_batch(detectors)

    # Use the heralds: decode each shot with the error model given its erasures.
    base, extra = erasure_models(noisy)
    fired = records[:, herald_records(noisy)]
    patterns, group = np.unique(fired, axis=0, return_inverse=True)
    aware = np.zeros_like(blind)
    for i, pattern in enumerate(patterns):
        shots = group.ravel() == i
        given = sum((extra[h] for h in np.flatnonzero(pattern)), base)
        aware[shots] = pymatching.Matching.from_detector_error_model(given).decode_batch(
            detectors[shots]
        )
    return np.mean(blind[:, 0] != truth), np.mean(aware[:, 0] != truth)


code = stim.Circuit.generated("surface_code:rotated_memory_z", distance=DISTANCE, rounds=DISTANCE)
noisy = device.to_stim(code)
blind, aware = logical_errors(noisy)
print(f"d={DISTANCE} memory, erasure {ERASURE} per CX qubit, {SHOTS} shots")
print(f"logical error, heralds ignored: {blind:.4f}")
print(f"logical error, heralds used:    {aware:.4f}")

print()
# The report says how the export treated the erasure effect.
for line in noisy.report.summary().splitlines():
    if "effect" in line:
        print(line)
