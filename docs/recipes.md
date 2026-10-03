# Recipes

Short, task-focused guides. The test suite runs every Python block on this page and checks the
output it shows. The drift recipe needs the network, and the Mitiq recipe needs Mitiq. The
[examples](../examples) folder has longer versions.

## Pin and cite a calibration for a paper

A result simulated under device noise is reproducible only if readers can get the same noise.
Pin the calibration by its fingerprint. Publish the calibration and the fingerprint.

1. Choose the calibration. This recipe uses the bundled profile `ibm_fez@2025-02-26`, which
   loads offline and is the same on every computer. `nv list` shows the other bundled profiles.

   To use the newest calibration instead, run `nv pull ibm_fez`. The first line of its output
   is the saved ref and fingerprint, such as `ibm_fez@2026-10-01T20:14:09Z` and
   `nv:d9067e68adc2`. In steps 2 and 3, use that ref and fingerprint in place of
   `ibm_fez@2025-02-26` and `nv:06404cefa54f`.

2. Print the citation. It names the source, the calibration time, the NoiseVault version, the
   ref that loads the calibration, and the full fingerprint:

   ```bash
   nv cite ibm_fez@2025-02-26
   nv cite ibm_fez@2025-02-26 --bibtex
   ```

3. Load the profile by ref and fingerprint in your code. Save the report of the export next to
   your results:

```python
import json

import noisevault as nv

fez = nv.load("ibm_fez@2025-02-26", expect="nv:06404cefa54f")
sim = fez.to_qiskit()
# ... run your circuits ...
with open("noise_manifest.json", "w") as f:
    json.dump(sim.report.to_dict(), f, indent=2)
print(fez.citation())
```

If the profile changes, `expect=` raises `FingerprintMismatch` instead of running with
different noise. The manifest records the profile id, fingerprint, framework versions, options
and everything the export approximated.

If you use a pulled calibration and not a bundled profile, your readers need the profile file
itself. Before you publish the profile file, check its terms with `nv show REF`. You can always
share the fingerprint alone.

## Compare drift between two dates

IBM's public endpoint keeps calibration history. Pull two dates. Then diff the two calibrations:

```bash
nv pull ibm_fez --at 2026-07-03
nv pull ibm_fez --at 2026-10-01
nv diff ibm_fez@2026-07-02 ibm_fez@2026-09-30
```

`nv pull --at` returns the newest calibration before that time. Here the pulls print
`ibm_fez@2026-07-02T23:05:55Z` and `ibm_fez@2026-09-30T23:14:35Z`, so the diff uses those dates.
For other dates, use the dates your pulls print. The diff lists device medians, the qubits and
pairs that changed most, and gates that IBM disabled or re-enabled. The output starts with
these lines:

```
ibm_fez 2026-07-02 -> 2026-09-30  (90 days later)
device median        before     after  change
T1 (us)               119.6       118   -1.3%
T2 (us)               89.44     92.16   +3.0%
1q avg infidelity  3.05e-04  3.16e-04   +3.6%
2q avg infidelity  2.72e-03  2.61e-03   -3.9%
readout error      8.18e-03  1.05e-02  +28.4%

largest changes by qubit
qubit  metric               before     after    change
61     T2 (us)                5.25     98.94  +1784.8%
123    readout error      6.35e-03  1.06e-01  +1565.4%
149    1q avg infidelity  1.40e-03  1.71e-02  +1121.7%
113    T1 (us)               18.95     155.4   +720.0%
30     T1 (us)               17.59     114.1   +548.6%
...
```

The gate-error and coherence medians moved by 4% or less, and the readout median rose 28%.
Single qubits changed by much more. On qubit 61, T2 went from 5.25 us to 98.94 us. A simulation
pinned to one date can differ from the same device three months later. `nv diff --json` gives
the same data for scripts, and `profile.diff(other)` returns it in Python.
[examples/drift.py](../examples/drift.py) diffs today's calibration against the one from 90
days ago.

## Mitigate errors with Mitiq

A NoiseVault simulator is a noisy backend for Mitiq. The executor transpiles each folded circuit
at `optimization_level=0`, so the transpiler translates the inverse gates that folding adds
without cancelling them.

Mitiq is optional and supports Python up to 3.12. Its Qiskit conversion also needs `ply`:

```bash
pip install "noisevault[qiskit] @ git+https://github.com/dvgyl/noisevault" mitiq ply
```

```python
from mitiq import zne
from qiskit import QuantumCircuit, transpile

import noisevault as nv

fez = nv.load("ibm_fez")
sim = fez.to_qiskit()
layout = list(fez.suggest_layout(3).values())


def p000(circuit: QuantumCircuit) -> float:
    measured = circuit.copy()
    measured.measure_all()
    native = transpile(measured, sim, initial_layout=layout, optimization_level=0)
    counts = sim.run(native, shots=20_000, seed_simulator=7).result().get_counts()
    return counts.get("000", 0) / 20_000


half = QuantumCircuit(3)
for _ in range(6):
    for q in range(3):
        half.rz(0.7 * (q + 1), q)
        half.sx(q)
    half.cz(0, 1)
    half.cz(1, 2)
mirror = half.compose(half.inverse())  # ideally returns |000> with probability 1

factory = zne.inference.RichardsonFactory(scale_factors=[1, 2, 3])
mitigated = zne.execute_with_zne(mirror, p000, factory=factory, scale_noise=zne.scaling.fold_global)
print(f"noisy {p000(mirror):.3f}, mitigated {mitigated:.3f}")
# noisy 0.887, mitigated 0.996
```

Richardson extrapolation from scale factors 1, 2 and 3 removes most of the error here.
[examples/mitigation_zne.py](../examples/mitigation_zne.py) runs the same recipe with random
angles.

## Run a QEC memory experiment with Stim and PyMatching

Stim builds the code circuit, NoiseVault adds the device noise, and PyMatching decodes. A
hypothetical square-grid device with `coords` on each qubit lets `layout_from_coords` place the
code by its `QUBIT_COORDS`:

```python
import numpy as np
import pymatching
import stim

import noisevault as nv

side = 7
edges = [(r * side + c, r * side + c + 1) for r in range(side) for c in range(side - 1)]
edges += [(r * side + c, (r + 1) * side + c) for r in range(side - 1) for c in range(side)]
grid = nv.Profile.uniform(
    "grid_49",
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
data = grid.to_dict()
data["qubits"] = [{"index": i, "coords": divmod(i, side)} for i in range(side * side)]
data["prep"] = {"error": 1e-3}
grid = nv.Profile.from_dict(data)

code = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3)
noisy = grid.to_stim(code, layout=nv.stim.layout_from_coords(code, grid), tick_ns=50)
matching = pymatching.Matching.from_stim_circuit(noisy)
detectors, observables = noisy.compile_detector_sampler(seed=1).sample(
    20_000, separate_observables=True
)
failures = np.sum(matching.decode_batch(detectors)[:, 0] != observables[:, 0])
print(f"logical error rate per 3 rounds: {failures / 20_000:.4f}")
```

The CNOTs in Stim's circuit get the grid's `cx` calibration, and `tick_ns=50` adds relaxation on
idle qubits at each `TICK`. To run on a real device's topology, use a profile whose qubits record
`coords`, such as `google_weber`. Then check that `layout_from_coords` finds a placement.
[examples/qec_surface_code.py](../examples/qec_surface_code.py) compares distances 3 and 5.

## Describe a hypothetical device

`Profile.uniform` gives every 1- and 2-qubit registry gate one error per arity. Z-family gates
such as `s` and `t` are free through a virtual `rz`. `Profile.uniform` leaves out `swap`,
`cxswap`, `swapcx` and `czswap`, because each needs several native entanglers. Decompose these
gates first. In every other circuit of 1- and 2-qubit gates, each gate gets its own calibrated noise
and none falls back to typical noise:

```python
import noisevault as nv

ions = nv.Profile.uniform(
    "toy-ions",
    technology="trapped_ion",
    num_qubits=12,
    one_qubit_error=3e-5,
    two_qubit_error=1e-3,
    readout_error=2e-3,
    t2_us=1e6,
    one_qubit_ns=10_000,
    two_qubit_ns=200_000,
)
ions.save("toy-ions.json")
print(nv.load("toy-ions.json").short_fingerprint == ions.short_fingerprint)
```

A neutral-atom device in gate mode loses atoms during imaging. Record the atom loss as an
`atom_loss` effect. No export models effects yet, so the report lists the effect as omitted
rather than dropping it silently:

```python
import stim

import noisevault as nv

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
atoms = atoms.model_copy(update={"effects": [{"type": "atom_loss", "on": "readout", "prob": 5e-3}]})
noisy = atoms.to_stim(stim.Circuit("H 0\nCX 0 1\nM 0 1"))
print([line for line in noisy.report.summary().splitlines() if line.startswith("omitted")])
# ['omitted: effect atom_loss on readout, initial state preparation error ...']
```

To require that the exports model an effect, set `"allow": "approximate"` or `"exact"` on the
effect. Exports then raise `UnsupportedEffect` instead of omitting it.

To give qubits or pairs their own values, edit the saved JSON by hand. See
[Profile format](profile-format.md). Then run `nv validate toy-ions.json`.

## Train a PennyLane circuit under device noise

`qml.add_noise` applies the model to a QNode on `default.mixed`. Write the circuit in the
profile's native gates so every gate gets its own calibration. For IBM, use `RZ`, `SX` and `CZ`:

```python
import pennylane as qml
from pennylane import numpy as pnp

import noisevault as nv

fez = nv.load("ibm_fez")
model = fez.to_pennylane(layout=fez.suggest_layout(2))


@qml.qnode(qml.device("default.mixed", wires=2), diff_method="backprop")
def cost(params):
    for wire in (0, 1):
        qml.SX(wire)
        qml.RZ(params[wire], wire)
        qml.SX(wire)
    qml.CZ([0, 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


noisy = qml.add_noise(cost, model)
params = pnp.array([0.4, -0.2], requires_grad=True)
opt = qml.GradientDescentOptimizer(stepsize=0.3)
for _ in range(20):
    params = opt.step(noisy, params)
print(f"noisy {noisy(params):.4f}, ideal at the same angles {cost(params):.4f}")
```

The noisy minimum stays above -1 because gate and readout errors shrink the expectation value.
[examples/pennylane_gradient.py](../examples/pennylane_gradient.py) trains a 3-qubit version.

## Measure a profile against hardware

`nv check` shows that each export implements the profile's noise model. It does not show how
close that model is to the device. To measure how close the noise model is to the device, run
circuits on the device. Then score the profile on the circuits' counts. `nv compare` fits two
factors, one on every gate error rate and one on every readout error rate. Each factor has a 95%
interval, and a goodness-of-fit test says whether one pair of factors explains every circuit.
[How nv compare fits the factors](limitations.md#how-nv-compare-fits-the-factors) explains the
fit and the tests of its intervals.

On an IBM device, `scripts/run_on_ibm.py` runs the circuits and saves the counts file. It needs
[uv](https://docs.astral.sh/uv/) and an IBM Quantum account. Put your API key in
`IBM_QUANTUM_TOKEN`. You can also save the account once with
`QiskitRuntimeService.save_account(token=...)`. uv reads the script's dependencies from its first
lines and installs them, so the script needs no clone or install:

```bash
uv run https://raw.githubusercontent.com/dvgyl/noisevault/main/scripts/run_on_ibm.py ibm_kingston --shots 4000 -o kingston-0416.counts.json
```

The script pulls the calibration in effect now through your account and plans the circuits from
it. It checks every gate, gate duration and delay against the backend, and it refuses a plan that
the backend would run differently. Before it submits anything, it shows the circuits and IBM's
estimate of the QPU time they use, and asks you to confirm. For the calibration of 2026-04-15 it
shows:

```text
ibm_kingston@2026-04-15 nv:a6bcc38ccc1b on qubits 148-149-150-151

circuit            qubits           gates  duration
ghz_chain          148-149-150-151     11    268 ns
mirror             148-149-150         49    752 ns
single_qubit       148-149             16    256 ns
two_qubit_natives  148-149              6    200 ns
readout            148-149-150-151      0      0 ns

shots           4000 per circuit, 5 circuits in one job
usage           about 7.1 s of QPU time (IBM's estimate)
counts file     kingston-0416.counts.json

Submit the job to ibm_kingston? [y/N]
```

`--yes` submits without asking. The script then submits one job and waits for IBM to run it. The
wait depends on the device's queue. If you press Ctrl-C or the connection drops during the wait,
the job keeps running at IBM. Run the same command again to collect the job's counts instead of
submitting another job. When the job has run, the script saves the counts file and ends with the
command that scores it:

```text
next            nv compare ibm_kingston@2026-04-15 kingston-0416.counts.json
```

The counts bind to the calibration in effect when the job started running. That calibration can
be newer than the one that the script planned from. When the newer calibration still calibrates
every gate in the circuits with the same duration, the counts bind to the newer calibration. The
script then says that IBM recalibrated. When the newer calibration does not, the counts bind to
the planned calibration. A warning then names both fingerprints, says what changed and suggests
running the script again.

On another device, take steps 1 to 3 by hand. Step 4 is the same for every device.

1. Plan the circuits. `plan(profile)` from `noisevault.counts` returns the `nv check` circuits on
   the qubits that `nv check` picks. `plan` leaves out `chain_mirror`, a circuit that only
   `nv check` runs. `plan` schedules each gate as soon as its qubits are free.
   It writes every wait as a `delay`, so the device and the noise model get the same idle time.
2. Run each circuit on the device exactly as planned. Do not transpile the circuits, twirl them
   or add dynamical decoupling. Start each shot in the ground state. Measure every circuit qubit
   after the last operation.
3. Save the counts as a [counts file](counts-format.md). The file records the calibration that
   you planned the circuits from. `nv compare` refuses counts that you planned from another
   calibration.
4. Score the profile. Then save the profile with the fitted factors:

   ```bash
   nv compare ibm_kingston@2026-04-15 kingston-0416.counts.json
   nv compare ibm_kingston@2026-04-15 kingston-0416.counts.json -o kingston-fitted.json
   nv check kingston-fitted.json
   nv cite kingston-fitted.json
   ```

   Every export of `kingston-fitted.json` applies the factors, and `nv check` confirms that each
   export reproduces the noise model of the fitted profile. `nv cite` prints one fingerprint that
   pins the calibration and the factors.

The Python below takes the same steps. In place of a device run, it simulates counts at twice
the calibration's gate error:

```python
import noisevault as nv
from noisevault.counts import load_counts, plan, simulate

kingston = nv.load("ibm_kingston@2026-04-15")
circuits = plan(kingston)
device = kingston.model_copy(update={"unmodeled_error": {"gates": {"factor": 2.0}}})
simulate(device, circuits, shots=4000, seed=1).save("kingston.counts.json")

result = kingston.compare(load_counts("kingston.counts.json"))
result.fitted_profile().save("kingston-fitted.json")
print(result)
# ibm_kingston@2026-04-15 nv:609c845ed934 on qubits 148-149-150-151
# ...
# gate errors     x2... (95% interval ...)
# readout errors  x1... (95% interval ...)
# fit             within shot noise on every circuit ...
```

`print(result)` prints the text that `nv compare` shows, and `result.gates` and `result.readout`
hold each factor with its interval. A fit beyond shot noise has p below 0.01. Such a fit means
that no pair of factors explains every circuit, as when one qubit reads out worse than its
calibration states. `nv compare` still prints the best pair and exits with status 0.

A profile can state a readout error of exactly 0 for a measured qubit. A wrong reading on that qubit
then has probability 0 at every factor. `nv compare` counts the shots that read such a qubit wrong,
reports p = 0 and names the qubit.
[Limitations](limitations.md#what-the-unmodeled-error-factors-absorb) lists what the factors absorb
and what they cannot express.
