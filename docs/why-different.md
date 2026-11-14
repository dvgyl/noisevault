# My device result differs from my simulation

You ran a circuit on an IBM device and on a simulator, for example a fake backend. The two
results differ much. Is the noise model wrong, old or incomplete? This page shows how to measure
the gap for your own circuit, in four steps:

1. Get the calibration that was in effect when your job ran.
2. Score your own counts against that calibration.
3. Read the result.
4. Find the cause.

The steps need your circuits as the device ran them, and the counts of the device. The circuits
must use the native gates of the device, on its physical qubits, as `transpile(circuit, backend)`
gives them.

## 1. Get the calibration of the job time

A fake backend holds one old calibration. The device had another calibration on the day of your
job. `nv pull --job` reads the start time of the job through your IBM Quantum account, and saves
the calibration that was in effect at that time:

```bash
nv pull ibm_kingston --job JOB_ID
```

The command prints the start time that it used. The vault keeps the calibration after the device
retires, so you can score the same counts again later.

With the job object of qiskit-ibm-runtime, one call gives the calibration and the counts:

<!-- not-run: needs a finished SamplerV2 job of IBM Quantum -->
```python
import noisevault as nv
from qiskit_ibm_runtime import QiskitRuntimeService

job = QiskitRuntimeService().job("JOB_ID")
profile, counts = nv.from_ibm_job(job)
print(profile.score(counts).summary())
```

`from_ibm_job` reads a SamplerV2 job with no parameter values. It refuses a job that used gate
twirling, measurement twirling or dynamical decoupling, because the device then ran other gates
than the circuit lists. The SamplerV2 defaults use none of them.

## 2. Score your own counts

With no job object, give your circuits and your counts to `nv.counts_from_qiskit`. This page
has no device, so a second profile plays the device. It has the same calibration, with 1.8
times the gate errors and 1.3 times the readout errors. `simulate` draws its counts.

```python
from datetime import UTC, datetime
from math import pi

from qiskit import QuantumCircuit

import noisevault as nv
from noisevault.counts import simulate
from noisevault.frameworks.qiskit import planned_circuit

profile = nv.load("ibm_kingston@2026-04-15")
when = datetime(2026, 4, 16, 9, 30, tzinfo=UTC)


def h(qc, q):  # a Hadamard gate in the native gates rz and sx
    qc.rz(pi / 2, q)
    qc.sx(q)
    qc.rz(pi / 2, q)


ghz = QuantumCircuit(156, 3, name="ghz")
h(ghz, 148)
for a, b in [(148, 149), (149, 150), (150, 151)]:
    h(ghz, b)
    ghz.cz(a, b)
    h(ghz, b)
ghz.measure([151, 148, 149], [0, 1, 2])  # the circuit does not read qubit 150

mirror = QuantumCircuit(156, 3, name="mirror")
for _ in range(4):
    mirror.sx([148, 149, 150])
    mirror.cz(148, 149)
    mirror.cz(149, 150)
for _ in range(4):
    mirror.cz(149, 150)
    mirror.cz(148, 149)
    for _ in range(3):  # three sx gates undo one sx gate
        mirror.sx([148, 149, 150])
mirror.measure([148, 149, 150], [0, 1, 2])

circuits = [ghz, mirror]

# The device. Qiskit writes classical bit 0 rightmost, so each key turns around.
device = profile.model_copy(
    update={"unmodeled_error": {"gates": {"factor": 1.8}, "readout": {"factor": 1.3}}}
)
run = simulate(device, [planned_circuit(c) for c in circuits], shots=4000, seed=1, run_at=when)
device_counts = [{key[::-1]: n for key, n in c.counts.items()} for c in run.circuits]

counts = nv.counts_from_qiskit(
    circuits, device_counts, profile=profile, run_at=when, source="simulated"
)
counts.save("my.counts.json")
print(profile.score(counts).summary())
# ibm_kingston@2026-04-15 nv:609c845ed934, calibration age 26 h
# counts sha256:..., simulated, run 2026-04-16 09:30Z on ibm_kingston
#   qubits 148-149-150-151
#
# circuit  shots  log score   excess  ruled out
# ghz       4000   -0.86807  0.00321          0
# mirror    4000   -0.59269  0.01704          0
# total     8000   -0.73038  0.01013          0
#
# quadratic score 0.62359
# fit excess 0.01013, shot noise ... at 95% (p = 0.005)
# largest excess mirror (0.01704)
```

The counts file names your Qiskit version. The hash of the counts and the shot noise thus
change with that version, and the example shows `...` for them.

For your own run, give your circuits and the counts of the device. Give the calibration from
step 1 as `profile`, and the start time of the job as `run_at`. A key of Qiskit counts can have spaces
between registers, as in `"01 1"`. The function removes them.

The function keeps the device qubits that have a gate or a measurement, and drops the idle
qubits. A circuit can act on a qubit that it does not measure, such as qubit 150 of `ghz`. The
score then sums over the outcomes of that qubit. Each measurement must come after the last gate
on its qubit. The function refuses a circuit that the counts file cannot hold, such as a circuit
with `reset` or `if_test`. The hint says what to do.

## 3. Read the result

- **`excess`** is how far the profile is from the best possible forecast of the counts. It is 0
  for a perfect profile, apart from shot noise.
- **`shot noise`** is the excess that shot noise alone gives in 95% of the cases, if the profile
  is right. `p` is the probability of an excess this large from shot noise alone.
- **`largest excess`** names the circuit that the profile describes worst.

Here the excess is 0.01013, and shot noise explains only 0.00137 (p = 0.005). So the gap is
real, and the counts have enough shots to show it. The `mirror` circuit has most of the gap.
When the excess is inside the shot noise, the calibration of the job time explains your result.
The gap to your fake backend then most likely comes from the old calibration of the fake backend.

## 4. Find the cause

Compare the calibration of the fake backend with the calibration of your job. The bundled
profile `ibm_kingston` holds the data of the fake backend `FakeKingston`. The second profile is
the one that step 1 saved. Use the date that step 1 printed:

```bash
nv diff ibm_kingston ibm_kingston@2026-04-16
```

`nv diff` lists the qubits and pairs that changed most. A qubit of your circuit in that list is
a cause of the gap.

`nv compare` fits one factor for the gate errors and one factor for the readout errors. The
factors tell how far the errors of the device were from the calibration:

```text
$ nv compare ibm_kingston@2026-04-15 my.counts.json
...
gate errors     x2.01 (95% interval 1.77 to 2.25)
readout errors  x1.19 (95% interval 0.892 to 1.51)
fit             within shot noise on every circuit (p = 0.9)
```

Here the device had about twice the gate errors of the calibration. The intervals contain the
factors that made the counts, 1.8 and 1.3. [Scoring](scoring.md) defines each number of a score,
and [Is my noise model any good?](first-score.md) shows how to check a fitted profile.

## Limits

- **A circuit on more than 10 used qubits has no exact forecast.** The reference simulator holds
  10 qubits. The score names such a circuit and leaves it out.
- **Idle time between gates adds no error.** The score models the idle time of a `delay`
  instruction only. To include the idle error, schedule the circuit before you run it, such as
  `transpile(circuit, backend, scheduling_method="alap")`. Give `dt=backend.dt` for delays in
  units of `dt`.
- **The counts file records the circuits as the device ran them.** Give the transpiled circuits,
  not the circuits before transpilation.
