<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/hero-dark.svg">
    <img src="assets/hero-light.svg" alt="NoiseVault" width="100%">
  </picture>
</p>

<div align="center">

**Calibrated noise from real quantum computers, as one file that works in Qiskit, Cirq, PennyLane and Stim.**

[![CI](https://github.com/dvgyl/noisevault/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/dvgyl/noisevault/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache%202.0-1f1f1f.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11%20to%203.14-1f1f1f.svg)](pyproject.toml)

[Install](#install) · [Quickstart](#quickstart) · [25 devices](#what-ships) · [Docs](#documentation) · [Changelog](CHANGELOG.md)

</div>

<br/>

Load a device by name. Then simulate your circuits under the noise that the device had on a given
day. Pin the fingerprint of the profile, and anyone can rerun your simulations with the same
noise. Each export reports what it reproduces exactly, what it approximates and what it leaves
out. `nv compare` reads counts measured on the device and fits factors that scale the profile's
errors to match the counts.

To try NoiseVault without an install, use [uv](https://docs.astral.sh/uv/):

```bash
uvx --from git+https://github.com/dvgyl/noisevault nv show ibm_fez
```

<img src="assets/cli-show.svg" alt="Terminal output of nv show ibm_fez. The output shows a 156-qubit Heron r2 device and its native gates, with the median average infidelity and duration of each gate. It also shows the median T1, T2 and readout error, the data source, the license and the profile fingerprint." width="830">

## Install

In a virtual environment with Python 3.11 to 3.14, install NoiseVault from GitHub:

```bash
pip install "noisevault[qiskit] @ git+https://github.com/dvgyl/noisevault"
```

For another framework, use `cirq`, `pennylane` or `stim` in place of `qiskit`. For several
frameworks, name them all, as in `noisevault[qiskit,stim]`. For every extra, use `all`. In a uv
project, `uv add` takes the same quoted argument. `nv doctor` lists what is installed.

## Quickstart

Run a GHZ circuit under the calibrated noise of IBM Fez. `ibm_fez` is a bundled profile, so this
example runs offline.

```python
from qiskit import QuantumCircuit, transpile

import noisevault as nv

sim = nv.load("ibm_fez").to_qiskit()

ghz = QuantumCircuit(3)
ghz.h(0)
ghz.cx(0, 1)
ghz.cx(1, 2)
ghz.measure_all()

compiled = transpile(ghz, sim, seed_transpiler=1)
counts = sim.run(compiled, seed_simulator=1).result().get_counts()
print(counts)
# {'111': 510, '011': 8, '101': 6, '100': 4, '001': 1, '110': 6, '010': 2, '000': 487}
print(sim.report.summary())
# NoiseVault 0.2.0 -> qiskit 2.5.2 (qiskit-aer 0.17.2): ibm_fez (nv:06404cefa54f)
# options: unknown_gates='typical', readout=True
# exact: gate noise: channels per exported native and physical locus (Aer QuantumError), ...
# approximated: cz error: the stated error already includes single-qubit gate error ...
# approximated: T2 of qubit 87: clamped to 2*T1 (the stated T2 exceeds 2*T1)
# omitted: idle time outside explicit delays ...
# unknown (no noise applied): preparation (reset) error of qubits 0, 1, 2 and 153 more
# clamped: 82 gates noisier than stated because relaxation alone exceeds the stated error ...
# used: cz took the calibration recorded for the opposite qubit order once
# Calibration-derived models approximate the hardware. They are not a digital twin.
```

The report says what this noise model reproduces, approximates, clamps or leaves out. Save
`sim.report.to_dict()` next to your results.

Because the simulator holds the device's gates, connectivity and errors, `transpile` compiles to
Fez's native gates and places the circuit by noise. To add the idle noise that the report
lists as omitted, also pass `scheduling_method='alap'` to `transpile`.

`to_cirq()`, `to_pennylane()` and `to_stim(circuit)` give the same noise to the other three
frameworks. Each export is the framework's own type and has its own report.
[Frameworks](docs/frameworks.md) has an example for each framework.

## Pin a calibration

A bundled profile holds one calibration. `nv pull` fetches other calibrations from IBM's public
endpoint or from IonQ, with no account. It saves them to your vault, `~/.noisevault/profiles`.
With `--at`, `nv pull` fetches the calibration that was in effect at that date:

```bash
nv pull ibm_fez --at 2025-06-01   # saves ibm_fez@2025-05-31T22:01:04Z
nv cite ibm_fez@2025-05-31        # cite the calibration by the date that nv pull printed
nv diff ibm_fez@2025-02-26 ibm_fez@2025-05-31
```

`nv diff` compares the device medians, lists the qubits and pairs that changed most, and names
the gates that were disabled or re-enabled.

A bare id such as `ibm_fez` loads the newest calibration that you have. After this pull, the
quickstart's `nv.load("ibm_fez")` loads 2025-05-31 instead of the bundled 2025-02-26. To load
the calibration of a given date, add the date to the id, as in `ibm_fez@2025-02-26`.

The fingerprint is a SHA-256 hash of a profile's physics. Pass the fingerprint as `expect` when
you load a profile. If the numbers ever differ, the load fails:

```python
import noisevault as nv

fez = nv.load("ibm_fez@2025-02-26", expect="nv:06404cefa54f")
print(fez)
# <Profile ibm_fez@2025-02-26 superconducting 156q nv:06404cefa54f>
```

A mismatch raises `FingerprintMismatch` with both fingerprints. The
[pin and cite recipe](docs/recipes.md#pin-and-cite-a-calibration-for-a-paper) shows the full
workflow for a paper.

## What ships

These bundled profiles ship inside the package and load offline with `nv.load(id)`. All of them
come from sources with open licenses. [NOTICE](NOTICE) lists each file with its source. `nv list`
shows the bundled profiles with their qubit counts and processors.

| Vendor | Technology | Devices | Calibrated | Source and license |
| --- | --- | --- | --- | --- |
| IBM (18) | superconducting | `ibm_aachen`, `ibm_berlin`, `ibm_boston`, `ibm_brisbane`, `ibm_brussels`, `ibm_cusco`, `ibm_fez`, `ibm_kawasaki`, `ibm_kingston`, `ibm_kyiv`, `ibm_manila`, `ibm_marrakesh`, `ibm_miami`, `ibm_pittsburgh`, `ibm_quebec`, `ibm_sherbrooke`, `ibm_strasbourg`, `ibm_torino` | 2024-05-27 to 2026-04-17 | [qiskit-ibm-runtime](https://github.com/Qiskit/qiskit-ibm-runtime) fake backends, Apache-2.0 |
| Quantinuum (5) | trapped ion | `quantinuum_h1-1`, `quantinuum_h1-2`, `quantinuum_h2-1`, `quantinuum_h2-2`, `quantinuum_reimei` | 2023-08-21 to 2025-08-28 | [hardware-specifications](https://github.com/Quantinuum/quantinuum-hardware-specifications) repository, Apache-2.0 |
| Google (2) | superconducting | `google_rainbow`, `google_weber` | 2021-11-03 to 2021-11-16 | [cirq-google](https://github.com/quantumlib/Cirq/tree/main/cirq-google) calibrations, Apache-2.0 |

You can get more devices and dates in three ways:

- Pull from an IBM Quantum account with `nv pull --source ibm-account`. This source needs the
  `ibm` extra.
- Import a Qiskit backend, an IBM calibration CSV, saved Amazon Braket device properties or a
  cirq-google calibration. You can also import a Hugging Face archive of IBM calibrations or a
  dataset from Quantinuum's repository. IBM's fake backends need the `ibm` extra. A cirq-google
  calibration needs the `google` extra. The archive needs the `hf` extra.
- Use `nv.Profile.uniform(...)` to describe a device that does not exist, as in
  [Describe a hypothetical device](docs/recipes.md#describe-a-hypothetical-device).

[Data sources](docs/data-sources.md) gives each source's fields and license.

## Check a conversion

`nv check` runs small circuits through each installed export and compares the results with
NoiseVault's own density-matrix reference:

```text
$ nv check ibm_fez
ibm_fez@2025-02-26T20:16:25Z nv:06404cefa54f on qubits 136-143-142-141
5 circuits: ghz_chain, chain_mirror, mirror, single_qubit, readout
framework  result  deviation  tolerance  circuits  method
qiskit     pass      3.6e-03    8.8e-03  5 of 5    exact + 20000 shots
cirq       pass      1.1e-15    1.0e-09  5 of 5    exact
pennylane  pass      6.1e-16    1.0e-09  5 of 5    exact
stim       pass      5.8e-04    1.3e-03  5 of 5    20000 shots, 5 sigma
```

A pass means each export implements the same noise model as the NoiseVault reference on these
circuits. A pass does not measure how well the model matches the hardware. Calibration-derived
models approximate the hardware. They are not a digital twin. [Limitations](docs/limitations.md)
lists what no export models.

To run the check on all four frameworks with uv and no install, use this command:

```bash
uvx --from "noisevault[qiskit,cirq,pennylane,stim] @ git+https://github.com/dvgyl/noisevault" nv check ibm_fez
```

## Compare with hardware

`nv check` shows that the exports agree with the reference. `nv compare` measures how far the
reference is from the device. It scores a profile on counts measured on the device. Then it fits
how far the profile's gate errors and readout errors must scale to match the counts.

NoiseVault has not yet been compared with counts from a real device. To see the output, score the
bundled Kingston profile on example counts. A simulation with gate errors x1.8 and readout errors
x1.3 made these counts. First, download the counts file:

```bash
curl -O https://raw.githubusercontent.com/dvgyl/noisevault/main/examples/kingston-simulated.counts.json
```

```text
$ nv compare ibm_kingston@2026-04-15 kingston-simulated.counts.json
ibm_kingston@2026-04-15 nv:609c845ed934 on qubits 148-149-150-151
counts kingston-simulated.counts.json, simulated, sha256:5e343e753c75
run 2026-04-16 09:30Z, 26 h after calibration

circuit       shots  profile TVD  fitted TVD  noise TVD 95%
ghz_chain      4000       0.0217      0.0215         0.0327
mirror         4000       0.0237      0.0050         0.0067
single_qubit   4000       0.0067      0.0011         0.0037
readout        4000       0.0096      0.0036         0.0046

gate errors     x2.16 (95% interval 1.76 to 2.58)
readout errors  x1.34 (95% interval 1.16 to 1.53)
fit             within shot noise on every circuit (p = 0.66)

Factors multiply the profile's error rates, so x2 means about twice the errors.
On these qubits, the factors absorb crosstalk, leakage, coherent error and idle
error beyond T1 and T2. The profile from -o applies the factors to every qubit.
```

Both intervals contain the factors that the simulation used. `nv compare` needs no framework. To
run it with no install, put
`uvx --from git+https://github.com/dvgyl/noisevault` in front of `nv compare`.

`nv compare ... -o fitted.json` saves the profile with the fitted factors. Every export of the
saved profile applies the factors.

To measure an IBM device yourself, put your IBM Quantum API key in `IBM_QUANTUM_TOKEN`. Then use
uv to run the `nv compare` circuits on the device. The script needs no clone or install:

```bash
uv run https://raw.githubusercontent.com/dvgyl/noisevault/main/scripts/run_on_ibm.py ibm_kingston -o kingston.counts.json
```

The script shows IBM's estimate of the QPU time and asks before it submits the job. It also prints
the `nv compare` command for the counts.
[Measure a profile against hardware](docs/recipes.md#measure-a-profile-against-hardware) shows how
to plan and record a run. [Counts format](docs/counts-format.md) describes the counts file.
[Limitations](docs/limitations.md#what-the-unmodeled-error-factors-absorb) lists what the factors
cannot express.

## Documentation

| Page | What it covers |
| --- | --- |
| [Frameworks](docs/frameworks.md) | Each export, its options and what its report can list |
| [Recipes](docs/recipes.md) | Pin and cite, drift, Mitiq, QEC with Stim, hypothetical devices, PennyLane training, hardware comparison |
| [Data sources](docs/data-sources.md) | Bundled, pulled and imported data, with licenses |
| [Profile format](docs/profile-format.md) | Every field of format 1.0, with examples |
| [Counts format](docs/counts-format.md) | The file that records a hardware run for `nv compare` |
| [Conventions](docs/conventions.md) | Error metrics, channel construction, readout and qubit order |
| [Limitations](docs/limitations.md) | What the noise models leave out and what has been checked |
| [JSON Schema](docs/schema/profile-1.0.json) | The schema of format 1.0, also printed by `nv schema` |
| [Examples](examples) | Runnable scripts, from the quickstart to a QEC memory experiment |

## Contributing

Bug reports, new data sources and fixes are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) covers
the development setup, the tests and how to add a source.

## Citing

If you use NoiseVault, cite the software with [CITATION.cff](CITATION.cff), or with
**Cite this repository** on GitHub. Also give the fingerprint of every profile that you used.
`nv cite REF` prints the fingerprint with the source, the calibration time, the NoiseVault version
and the ref that loads the calibration. `nv cite REF --bibtex` prints a BibTeX entry.

## License

Apache License 2.0. See [LICENSE](LICENSE). Bundled calibration data keeps the license and
attribution of its source. [NOTICE](NOTICE) lists every bundled file and where it comes from.

## Acknowledgements

The bundled calibrations come from IBM Quantum through
[qiskit-ibm-runtime](https://github.com/Qiskit/qiskit-ibm-runtime), from Google Quantum AI
through [cirq-google](https://github.com/quantumlib/Cirq), and from Quantinuum's
[hardware-specifications](https://github.com/Quantinuum/quantinuum-hardware-specifications)
repository. NoiseVault uses [Qiskit](https://github.com/Qiskit/qiskit) and
[Qiskit Aer](https://github.com/Qiskit/qiskit-aer), [Cirq](https://github.com/quantumlib/Cirq),
[PennyLane](https://github.com/PennyLaneAI/pennylane), [Stim](https://github.com/quantumlib/Stim)
and [PyMatching](https://github.com/oscarhiggott/PyMatching).
