"""Surface-code memory on a calibrated grid device.

The script exports to Stim, then decodes with PyMatching.

Needs:
    pip install "noisevault[stim] @ git+https://github.com/dvgyl/noisevault"
    pip install pymatching
"""

import textwrap

import numpy as np
import pymatching
import stim

import noisevault as nv

SIDE, SHOTS = 11, 20_000


def grid_device(side: int) -> nv.Profile:
    """A hypothetical square-grid device whose qubits have (row, col) coords."""
    edges = [(r * side + c, r * side + c + 1) for r in range(side) for c in range(side - 1)]
    edges += [(r * side + c, (r + 1) * side + c) for r in range(side - 1) for c in range(side)]
    device = nv.Profile.uniform(
        "grid_121",
        technology="superconducting",
        num_qubits=side * side,
        one_qubit_error=2e-4,
        two_qubit_error=1.5e-3,
        readout_error=5e-3,
        t1_us=100,
        t2_us=80,
        one_qubit_ns=25,
        two_qubit_ns=40,
        connectivity=edges,
    )
    data = device.to_dict()
    data["qubits"] = [{"index": i, "coords": divmod(i, side)} for i in range(side * side)]
    data["prep"] = {"error": 1e-3}  # the surface code resets ancillas every round
    return nv.Profile.from_dict(data)


device = grid_device(SIDE)
for distance in (3, 5):
    code = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=distance, rounds=distance
    )
    noisy = device.to_stim(code, layout=nv.stim.layout_from_coords(code, device), tick_ns=50)
    matching = pymatching.Matching.from_stim_circuit(noisy)
    sampler = noisy.compile_detector_sampler(seed=1)
    detectors, observables = sampler.sample(SHOTS, separate_observables=True)
    failures = np.sum(matching.decode_batch(detectors)[:, 0] != observables[:, 0])
    print(f"d={distance}: logical error per {distance}-round memory = {failures / SHOTS:.4f}")

print()
# The report's approximated and omitted lines say where the simulation differs from the device.
for line in noisy.report.summary().splitlines():
    if line.startswith(("NoiseVault", "approximated", "omitted", "unknown", "clamped")):
        print(textwrap.fill(line, 88, subsequent_indent="    "))
