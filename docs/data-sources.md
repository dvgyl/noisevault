# Data sources

NoiseVault gets calibrations three ways:

- **Bundled.** 25 profiles ship inside the package and load offline with `nv.load`. All come
  from openly licensed sources.
- **Pulled.** `nv pull` (or `nv.pull`) fetches a live calibration from IBM or IonQ and saves it
  to your vault, `~/.noisevault/profiles`. Set `NOISEVAULT_HOME` to move the vault.
- **Imported.** An importer reads data that you already have. The data can be a Qiskit backend,
  an IBM calibration CSV, saved Amazon Braket device properties, a saved IQM Resonance quality
  metric set or a saved Rigetti QCS ISA. The data can also be a calibration that cirq-google
  ships, or a local copy of a published archive of IBM calibrations.

`nv list` shows the vault first, then the bundled set. A vault profile with the same id,
calibration time and fingerprint as a bundled profile hides that bundled profile. When one
device has several calibrations on the same date, `nv list` adds the UTC time under the date. A
ref with the full time, such as `ibm_fez@2025-02-26T20:16:25Z`, loads the calibration at that
time.

## Licensing policy

NoiseVault bundles only data whose license allows redistribution, and records that license in
each profile's `provenance`. Everything else stays on your computer. The data records the rule:

- `provenance.redistributable` is `"yes"`, `"no"` or `"unknown"`.
- The build script refuses to bundle any profile that is not `"yes"`.
- Pulled and imported profiles from IBM's services, the IBM calibration archive, IonQ, Braket
  and your own files are `"no"` or `"unknown"`. NoiseVault writes these profiles only to your
  vault or to the path that you choose.

Check a profile's terms with `nv show REF`, which prints its source, license and
redistributable flag. Before you publish a pulled profile, for example next to a paper, read
the provider's terms. You can always publish the profile's fingerprint (`nv cite REF`), and
others can then check that they use the same calibration.

The bundled data keeps its upstream license and attribution. [NOTICE](../NOTICE) lists every
bundled file and its source.

## Bundled profiles

Every bundled profile loads offline with `nv.load(ID)`. None needs an account, and each one is
a single calibration with no history.

| Source | Profiles | License | Calibration dates | Stored |
| --- | --- | --- | --- | --- |
| qiskit-ibm-runtime 0.49.0 fake backends | 18 IBM devices. The set has Eagle (`ibm_brisbane`, ...), Heron (`ibm_fez`, ...) and Nighthawk (`ibm_berlin`, `ibm_miami`) devices and the 5-qubit Falcon `ibm_manila`. | Apache-2.0 | 2024-05-27 to 2026-04-17 | Per-qubit T1, T2 and readout, per-gate errors and durations of every native. |
| Quantinuum hardware-specifications repository, pinned commit | `quantinuum_h1-1`, `h1-2`, `h2-1`, `h2-2`, `reimei` | Apache-2.0 | 2023-08-21 to 2025-08-28 | Device-wide average infidelity of each native gate, device-wide readout error, and leakage rates as `leakage` effects. No T1 or T2. |
| cirq-google 1.6.1 | `google_rainbow`, `google_weber` | Apache-2.0 | 2021-11-16 (rainbow) and 2021-11-03 (weber) | Per-qubit T1, T2 and readout, per-qubit and per-pair gate errors and durations, and fSim coherent errors as `coherent_overrotation` effects. |

The bundled files total 0.24 MB. Each bundled profile holds one calibration. The vendors have
recalibrated the devices many times since then. To get current numbers, pull a new calibration.

## Pulled sources

`nv pull DEVICE [--at TIME] [--source NAME] [-o FILE]` picks the source from the device id.
If the vault already has the calibration, `nv pull` writes nothing.

### IBM public endpoint (`--source ibm`, the default for `ibm_...`)

| | |
| --- | --- |
| Access | `https://quantum.cloud.ibm.com/api/v1/public/backends/<name>/properties` |
| Account | None |
| History | Yes. `--at 2026-06-01` returns the newest calibration before that time. |
| License | Not an open license. Profiles are `redistributable: "unknown"` and stay local. |
| Stored | Per-qubit T1, T2 and readout, per-gate errors and durations of every native, disabled gates and qubits, the raw response's SHA-256. |

IBM does not document the endpoint, and the endpoint may change or disappear. If the endpoint
fails, the error says so and suggests the account source.

### IBM account (`--source ibm-account`)

| | |
| --- | --- |
| Access | `qiskit-ibm-runtime`, `backend.properties(datetime=...)`. Install `noisevault[ibm]`. |
| Account | Yes. The source uses the account saved with `QiskitRuntimeService.save_account`, or `IBM_QUANTUM_TOKEN` (with `IBM_QUANTUM_INSTANCE` when the key sees several instances). |
| History | Yes, through `--at`. |
| License | Your account's terms. `redistributable: "unknown"`. |
| Stored | The same fields as the public endpoint. |

### IonQ characterizations (`--source ionq`, the default for `ionq...`)

| | |
| --- | --- |
| Access | IonQ API v0.4, `https://api.ionq.co/v0.4/backends/<backend>/characterizations` |
| Account | None |
| History | Yes, through `--at`. The importer skips records without their own 1Q and 2Q fidelities and uses the next older record. The reason goes in the provenance notes. |
| License | IonQ EULA, not an open license. `redistributable: "no"`. |
| Stored | Only device-wide medians of 1Q, 2Q and SPAM fidelity, gate and readout times, T1 and T2. Every qubit and pair gets the same values. |

Device ids are `ionq_forte-1`, `ionq_aria-1` and so on. With `--source ionq`, `forte-1` and
`qpu.forte-1` also work.
IonQ does not say which fidelity metric it reports. The importer reads the fidelity as average
gate fidelity and writes that assumption into each gate's `assumption`, which `nv show` prints.

```bash
nv pull ibm_fez
nv pull ibm_fez --at 2026-06-01
nv pull ionq_forte-1
nv list
```

## Imported sources

| Importer | Reads | Account | History | License of the result | Stored |
| --- | --- | --- | --- | --- | --- |
| `nv.from_qiskit_backend(backend)` | Any Qiskit `BackendV2`, such as a fake backend or one from your account. | Only for account backends | No. It reads the backend's current `Target`. For an older IBM calibration, use `nv pull --at`. | Apache-2.0 for qiskit-ibm-runtime fake backends, else `unknown` | Per-qubit T1, T2 and readout, per-gate errors and durations of every native in the `Target`. |
| `nv.from_ibm_csv(path, device=..., calibrated_at=...)` | The calibration CSV you download from the IBM Quantum platform. Formats from 2023 to 2026. | To download it | One file is one calibration. You give its time as `calibrated_at`. | `unknown` | Per-qubit T1, T2, readout errors and readout length, per-gate errors and durations, and qubits marked not operational. |
| `nv.from_braket(path_or_dict, device=...)` | Braket device properties you saved with `AwsDevice(arn).properties.json()`, or their `standardized` part (v1, v2, v3). | AWS, to save it | No. One file is one calibration. | AWS Customer Agreement, `no` | For v1 and v2, per-qubit T1, T2 and fidelities, and per-pair gate fidelities. For v3, device-level values. |
| `nv.from_iqm(metrics, architecture=..., device=...)` | An IQM Resonance quality metric set that you saved with `iqm-client`, and optionally the dynamic quantum architecture of the same calibration set. | IQM Resonance, to save it | No. One set is one calibration. | IQM Resonance terms of use, `unknown` | Per-qubit T1, T2 and readout errors, per-qubit `prx` and per-pair `cz` fidelities, durations where the set has them, and disabled qubits and pairs from the architecture. See [the mapping](#iqm-resonance). |
| `nv.from_rigetti(path_or_dict, device=...)` | A Rigetti QCS instruction set architecture (ISA) that you saved with `qcs-sdk-python`. | Rigetti QCS, to save it | No. One ISA is one calibration. | Rigetti QCS terms of service, `no` | Per-qubit readout error, `rx` RB fidelity, T1 and T2 where the ISA has them, per-pair `cz` or `iswap` fidelities, and disabled qubits and pairs. See [the mapping](#rigetti-qcs). |
| `nv.from_calibration_archive(path, device, at=...)` | A local copy of the Hugging Face dataset `phanerozoic/qiskit-calibration-drift`, which records each new IBM calibration of `ibm_fez`, `ibm_kingston`, `ibm_marrakesh` and `ibm_torino`. Install `noisevault[hf]`. | None | Yes. Each property takes its newest calibration at or before `at`. | `unknown`. The dataset is CC-BY-4.0, and the numbers are IBM's. | The same fields as a pull. Early dates have fewer gates, and `ibm_torino` has no durations. |
| `nv.from_cirq_google(processor_id)` | The calibration cirq-google ships for `rainbow`, `weber` or `willow_pink`. Install `noisevault[google]` (cirq-google 1.6 or later). | None | No. cirq-google ships one calibration per device. | Apache-2.0 | The same fields as the bundled Google profiles. |
| `noisevault.sources.quantinuum.from_repository(machine, date)` | Any of the 15 datasets in Quantinuum's repository at the pinned commit, such as `("H1-1", "2023_01_20")`. The importer downloads the dataset from GitHub. | None | Yes. Each dataset is one date, and `date=None` gives the newest dataset for the device. | Apache-2.0 | The same fields as the bundled Quantinuum profiles. |
| `nv.Profile.uniform(...)`, or a file you write | A hypothetical device. | None | No | Yours | The values you give. |

```python
from qiskit_ibm_runtime.fake_provider import FakeTorino

import noisevault as nv

torino = nv.from_qiskit_backend(FakeTorino())
print(torino.id, torino.short_fingerprint, torino.provenance.license)
```

`willow_pink` is importable but not bundled. The two-qubit values of `willow_pink` are under
cirq-google's "per cycle" key, but the values reproduce Google's published per-gate CZ error for
that device. While this conflict exists, the importer reads the values as per-gate errors and
records that reading in the gate's `assumption`.

Some qiskit-ibm-runtime fake backends are models, not calibrations of a device. One is
`FakeNighthawk`. The package says that the error values of `FakeNighthawk` are not typical of
Nighthawk. Another is `FakeFractionalBackend`, modeled on `FakeLima`. The properties that these
fake backends ship name the backend (`fake_nighthawk`) instead of a device. NoiseVault therefore
gives their profiles `data_kind` `vendor_model` and a note in `provenance.notes` that quotes the
package. No bundled profile comes from these fake backends.

NoiseVault does not save an imported profile until you call `profile.save(path)`.

### IBM calibration archive on Hugging Face

The dataset [phanerozoic/qiskit-calibration-drift](https://huggingface.co/datasets/phanerozoic/qiskit-calibration-drift)
records IBM calibrations. Its poller reads IBM's backend properties every 30 minutes and adds a
row for each new calibration of a property. The dataset covers `ibm_fez`, `ibm_marrakesh` and
`ibm_torino` from 31 January 2026, and `ibm_kingston` from 16 March 2026. NoiseVault reads a
local copy of its one parquet file and never downloads the file for you.

To get the file, install the `hf` extra and the Hugging Face command line. Then download the
data folder. The file is about 66 MB.

```bash
pip install "noisevault[hf] @ git+https://github.com/dvgyl/noisevault" huggingface_hub
hf download phanerozoic/qiskit-calibration-drift --repo-type dataset --include "data/*"
```

`hf download` prints the snapshot folder it wrote, which ends in `snapshots/<revision>`. The
file is `data/train-00000-of-00001.parquet` inside that folder. A `git clone` without Git LFS
gives a small pointer file instead of the parquet file, and the importer rejects the pointer
file.

<!-- not-run: needs the 66 MB file that hf download writes -->
```python
import noisevault as nv

path = "<snapshot folder>/data/train-00000-of-00001.parquet"
for device, span in nv.calibration_archive_devices(path).items():
    print(device, span.first, span.last)
fez = nv.from_calibration_archive(path, "ibm_fez", at="2026-06-01")
```

`calibration_archive_devices` returns each device with two times. `first` is the earliest `at`
that gives a profile. It is the later of two times: when the archive first recorded the device,
and the device's earliest calibration. `last` is the device's newest calibration.
`from_calibration_archive` takes the newest calibration at or before `at` for each property. A
property is, for example, the T1 of one qubit or the `cz` error of one pair. Without `at`, every
property takes its newest calibration. An `at` before `first` raises `SourceDataError`, because
the archive holds only the calibrations that were still current at its first poll. Each call
reads the file again, and the read takes 2 to 3 seconds for one device.

The dataset's license is CC-BY-4.0, except for its sunspot column `SN`, which is CC-BY-NC-4.0.
The importer never reads `SN`. Each profile credits the dataset and IBM Quantum in `provenance`:

- `license` is `CC-BY-4.0`.
- `attribution` is `IBM Quantum, via phanerozoic/qiskit-calibration-drift`.
- `source` names the dataset and says that NoiseVault converted it. `source_url` links to the
  dataset.
- `source_hash` is the file's SHA-256, which equals the Hub's LFS object id, so the hash
  identifies the revision. When the path contains `snapshots/<revision>`, `extra.revision`
  records the revision and `source_url` links to that revision's file.
- `redistributable` is `unknown`, as for `nv pull`, because the numbers are IBM's.

The archive has these limits:

- Until 8 May 2026, the poller wrote rows with no unit, and those rows give T1 and T2 in
  seconds. The importer converts these values to microseconds. Those rows hold only T1, T2, the
  readout errors and the `sx` and `cz` errors. Rows from a backfill of IBM calibrations,
  calibrated from late January 2026, add the durations and most other gates.
- If the device's newest calibration has a gate that the time you ask for lacks, the profile
  leaves the gate out and `provenance.notes` names it. For example, `ibm_fez` at its `first`
  time has no `xslow`.
- IBM retired `ibm_torino` in April 2026, and the archive has only its rows with no unit.
  Its profiles have the `sx` and `cz` errors and no durations.
- The archive records each calibration when its poller first sees that calibration. The archive
  never records that IBM stopped listing a property, so a profile keeps such a property at its
  last value. At 2026-09-01, `ibm_fez` has `xslow` errors from 29 May 2026 and a T2 on qubit 72
  from 21 October 2025. IBM's own calibration for that time does not have these values.
  `nv pull ibm_fez --at 2026-09-01` and the archive agree on all 1,132 gate records that both
  have. `provenance.notes` names every profile value calibrated more than 7 days before `at`.
  When you give no `at`, the notes name every profile value calibrated more than 7 days before
  the newest calibration. The notes also give the date of the oldest of these values. The notes
  do not name a value that the profile does not use, for example the duration of a disabled gate.
- The importer disables a qubit whose `prob_meas1_prep0` or `prob_meas0_prep1` is 1, and
  `provenance.notes` names that qubit. A gate error of 1 disables that gate, as in a pull.

### IQM Resonance and Rigetti QCS field mapping

This section gives the mapping from each IQM and Rigetti field to profile format 1.0. The
mapping comes from the public SDKs and documentation, not from saved real payloads. The sources
are `iqm-client` 36.0.0 with `iqm-station-control-client` 14.0.0, `pyquil` 4.23.0 and
`qcs-sdk-python` 0.31.3. Each importer writes the interpretations below into the profile, and
`nv show` prints them.

#### IQM Resonance

The importer reads two documents that you save with `iqm-client`:

- The quality metric set, from `IQMClient.get_quality_metric_set()`. Its JSON has
  `observations`, `describes_id` (the calibration set id), `invalid`, `created_timestamp` and
  `end_timestamp`.
- Optionally, the dynamic quantum architecture (DQA), from
  `IQMClient.get_dynamic_quantum_architecture()`. Its JSON has `calibration_set_id`, `qubits`,
  `computational_resonators` and `gates`. Each gate has `implementations` with `loci`,
  `default_implementation` and `override_default_implementation`.

Each observation has `dut_field`, `value`, `unit`, `uncertainty` and `invalid`. The `dut_field`
is a dotted name. A `__` joins the components of a locus, and text after a `:` is a suffix, for
example `metrics.irb.cz.tgss.QB1__QB2.fidelity:par=d2`. The importer reads these names:

| `dut_field` | Unit | Profile field | Qualifiers |
| --- | --- | --- | --- |
| `characterization.model.QB<n>.t1_time` | `s` | `qubits[].t1_us` | |
| `characterization.model.QB<n>.t2_echo_time` | `s` | `qubits[].t2_us` | `t2_kind: "echo"` |
| `characterization.model.QB<n>.t2_time` | `s` | `qubits[].t2_us`, only when the qubit has no `t2_echo_time` | `t2_kind: "ramsey"` |
| `metrics.ssro.<measure>.<impl>.QB<n>.error_0_to_1` | none | `qubits[].readout.p1_given_0` | |
| `metrics.ssro.<measure>.<impl>.QB<n>.error_1_to_0` | none | `qubits[].readout.p0_given_1` | |
| `metrics.ssro.<measure>.<impl>.QB<n>.fidelity` | none | `qubits[].readout.error` = 1 - fidelity, only when the two error values are missing | |
| `metrics.rb.prx.<impl>.QB<n>.fidelity` | none | `calibrations[]` of `r`, `avg_infidelity` = 1 - fidelity | `method: "rb"`, `assumption` |
| `metrics.irb.cz.<impl>.QB<a>__QB<b>.fidelity` | none | `calibrations[]` of `cz`, `avg_infidelity` = 1 - fidelity | `method: "irb"`, `assumption` |
| `gates.<gate>.<impl>.<locus>.duration` | `s` | `duration_ns` of that record, also when the record has no fidelity, or `readout.duration_ns` for `measure` or `measure_fidelity` | |
| `uncertainty` of a `prx` or `cz` fidelity | none | `stderr` of that record | `assumption` |

`<measure>` is `measure` or `measure_fidelity`. The importer converts each time with its `unit`
(`s`, `ms`, `us` or `ns`). A fidelity or a readout error must have the unit `""`, which IQM uses
for a dimensionless value. The importer refuses any other unit.

Other rules:

- **Qubit numbers.** `QB<n>` becomes qubit index n - 1, with `label` `QB<n>`. This is the order
  that `qiskit-iqm` uses. An index that no observation and no DQA entry names becomes
  `disabled: true`.
- **Gate names.** IQM's `prx` is the canonical `r`. IQM applies Z rotations in software, so
  `rz` is `virtual: true`. `cz` stays `cz`.
- **Implementations.** A gate can have several implementations, such as `tgss` and `crf` for
  `cz`. With a DQA, the importer uses only the implementation that the DQA selects for the
  locus (`override_default_implementation`, then `default_implementation`). When that
  implementation has no data or does not list the locus, the gate stays uncalibrated on the
  locus, because data of another implementation describes a different pulse. Without a DQA,
  the importer uses the first implementation in alphabetical order. `provenance.notes` names
  each such choice.
- **Disabled pairs and qubits.** The DQA lists only calibrated loci. There is no flag for a
  disabled locus. With a DQA, the connectivity is the union of the `cz` loci. A pair that has
  data but no DQA locus gets a `cz` record with `disabled: true`. A qubit that is not in the DQA
  `qubits` list becomes `disabled: true`. The importer refuses a DQA locus with the wrong number
  of components, such as a `cz` locus with one qubit. Without a DQA, the connectivity is the set
  of pairs that have a `cz` fidelity.
- **Sentinels.** IQM marks data that automated systems must not use with `invalid: true`. The
  importer skips an invalid observation and names it in `provenance.notes`. It refuses a whole
  set that is invalid. The importer then applies these NoiseVault rules, in this order:
  1. A fidelity of exactly 1, or of 0 or less, is unknown, and a note names it. `iqm-benchmarks`
     says that IQM commonly fills an uncharacterized qubit with exactly 1.0.
  2. A gate fidelity at or below 1/(d+1), with d = 2 for one qubit and d = 4 for two qubits,
     gives a record with `disabled: true`. At 1/(d+1) the gate fully depolarizes, so it cannot
     be useful. The IBM importers use the same rule.
  3. A T1 or T2 that is not a positive number is unknown.

  IQM does not say that `uncertainty` is a standard error. The importer stores it as `stderr`
  of a gate record and writes that reading into the gate's `assumption`. An `uncertainty` of
  `null` gives no `stderr`. The importer does not store the uncertainty of a T1, a T2 or a
  readout value, because the format has no field for it.
- **Suffixes.** A suffix qualifies a metric, for example with the conditions of the
  measurement. The importer does not interpret suffixes. When two observations differ only in
  the suffix, the importer keeps the first one and names the others in `provenance.notes`. This
  is what `ObservationFinder` in `iqm-station-control-client` does.
- **Calibration time.** `device.calibrated_at` is the `end_timestamp` of the metric set, or
  `created_timestamp` when the set has no end time.
- **Not read.** Two-qubit Clifford fidelities (`metrics.rb.clifford...`), `metrics.qndness`,
  `metrics.ghz_state`, `move` gates and loci with a computational resonator. Loci with a
  resonator occur on star devices such as Sirius, and the profile format cannot express a
  resonator. `provenance.notes` names each kind that the importer did not read.

The `assumption` of `r` says that IQM does not state whether its RB fidelity is per `prx` gate
or per Clifford, and that NoiseVault reads it as the average gate fidelity of one `prx`. The
`assumption` of `cz` says that NoiseVault reads the fidelity as the average gate fidelity of one
`cz`. The `method` of each record comes from the observation name, for example `irb` for
`metrics.irb.cz...`.

#### Rigetti QCS

The importer reads the instruction set architecture (ISA) that you save with
`qcs_sdk.qpu.isa.get_instruction_set_architecture(id).json()`. The REST endpoint is
`GET /v1/quantumProcessors/<id>/instructionSetArchitecture`. The ISA has these keys:

- `name`, the processor id, such as `Ankaa-3`.
- `architecture`, with `family`, `nodes` (each with `node_id`) and `edges` (each with
  `node_ids`).
- `instructions` and `benchmarks`. Each is a list of operations with `name`, `sites` and
  `characteristics`. Each site has `node_ids` and `characteristics`. Each characteristic has
  `name`, `value` and `timestamp`, and optionally `error`, `node_ids` and `parameter_values`.

The ISA has no unit field. Fidelities are from 0 to 1, and times are in seconds.

| Operation | Characteristic | Profile field | Qualifiers |
| --- | --- | --- | --- |
| instruction `MEASURE`, one site per qubit | `fRO` | `qubits[].readout.error` = 1 - fRO | |
| benchmark `randomized_benchmark_1q` | `fRB` | `calibrations[]` of `rx`, `avg_infidelity` = 1 - fRB | `method: "rb"`, `measured: "isolated"`, `assumption` |
| benchmark `randomized_benchmark_simultaneous_1q` | `fRB` | the same, only for a qubit with no usable isolated value | `method: "srb"`, `measured: "simultaneous"`, `assumption` |
| instruction `CZ`, one site per pair | `fCZ` | `calibrations[]` of `cz`, `avg_infidelity` = 1 - fCZ | `method: "vendor"`, `assumption` |
| instruction `ISWAP`, one site per pair | `fISWAP` | `calibrations[]` of `iswap`, `avg_infidelity` = 1 - fISWAP | `method: "vendor"`, `assumption` |
| benchmark `FreeInversionRecovery` | `T1` | `qubits[].t1_us` | |
| benchmark `FreeInductionDecay` | `T2` | `qubits[].t2_us` | `t2_kind: "ramsey"` |
| `fRB`, `fCZ` or `fISWAP` | `error` | `stderr` of that record | |
| instruction `RZ` | none | `rz` with `virtual: true`, and a record with `disabled: true` on each live node without an `RZ` site | |

In the simultaneous benchmark, one site lists all qubits, and each `fRB` names its qubit in its
own `node_ids`. `fRO` is the mean of P(0|0) and P(1|1), as pyquil reads it. One number cannot
give the two error probabilities, so the importer stores 1 - fRO as a symmetric
`readout.error`. That is correct for the mean error only. For example, P(1|0) = 0.01 and
P(0|1) = 0.09 give fRO = 0.95 and `error` = 0.05. `provenance.notes` says so. Rigetti does not
publish T1 and T2 for all processors, so both are optional. A T1 or T2 must be more than 0 s
and less than 1 s, and any other value is unknown. A superconducting qubit has no T1 of 1 s or
more, so such a value more probably has a different unit.

Other rules:

- **Qubit numbers.** A `node_id` is the qubit index, with `label` the node id. Rigetti numbers
  nodes with gaps, so an index that is not an `architecture` node becomes `disabled: true`.
- **Disabled qubits and pairs.** The ISA has no `dead` flag. pyquil marks a node dead when it
  has no site of `RX`, `RZ`, `MEASURE` or the wildcard `_`. A site of `I` or `RESET` does not
  count. The importer applies the same rule and sets `disabled: true` on that qubit. The
  connectivity is the `architecture` edges. For each two-qubit gate, an edge with no site for
  that gate gets a record with `disabled: true`. pyquil marks an edge dead only when it has no
  two-qubit site at all, so the importer is stricter per gate.
- **Site checks.** The importer refuses a site of a known instruction with the wrong number of
  nodes, such as an `RX` site on two nodes, as pyquil does. It also refuses a characteristic
  whose `node_ids` are outside its site.
- **Sentinels.** Rigetti reports a dead site as `value` 0.5 with `error` 1.0. A gate site with
  that marker gives a record with `disabled: true`. A `MEASURE`, `T1` or `T2` value with it is
  unknown. The other rules are the same NoiseVault rules as for IQM, in the same order: a
  fidelity of exactly 1, or of 0 or less, is unknown, and then a gate fidelity at or below
  1/(d+1) gives `disabled: true`. When the isolated `fRB` of a qubit is unknown, the importer
  uses the simultaneous one. Rigetti's `error` is the standard error of the value, and the
  importer stores it as `stderr`. A timestamp of `1970-01-01T00:00:00+00:00` is a placeholder,
  and the importer ignores it.
- **Calibration time.** `device.calibrated_at` is the newest timestamp of a characteristic that
  the importer reads. `provenance.notes` gives the oldest timestamp too.
- **Not read.** `RESET` with `fAR`, because the active reset fidelity possibly includes readout
  error. `XY` and `CPHASE`, because they take an angle and no registry gate matches them, as in
  pyquil's `NoiseModel.from_isa`. `I`, which is not a calibrated gate. `provenance.notes` names
  each one that the ISA has.

The `assumption` of `rx` says that Rigetti calibrates `RX` only at angles of plus or minus pi/2
and pi, that Rigetti does not state whether fRB is per gate or per Clifford, and that NoiseVault
reads it as the average gate fidelity of one `rx` at every angle. The `assumption` of each
two-qubit gate says that Rigetti does not state the fidelity metric, and that NoiseVault reads
1 - fidelity as the average gate infidelity.

#### Values that need real payloads

The mapping above is from SDK source code and documentation. `tests/test_real_payloads.py`
checks each real payload in `tests/fixtures/iqm/real/` and `tests/fixtures/rigetti/real/`. The
README in each folder says how to add one. These points need a saved real payload to confirm:

- The exact IQM readout and T2 names on Garnet and Emerald: `measure` or `measure_fidelity`, and
  `t2_time` or `t2_echo_time`.
- The meaning of IQM suffixes such as `:par=d2`, and which observation to keep when two differ
  only in the suffix.
- Whether IQM fills an uncharacterized element with a fidelity of exactly 1.
- Whether Rigetti `T1` and `T2` values are in seconds on every processor, and the name of the
  Ankaa two-qubit instruction (`ISWAP` or other).

## How each source's numbers are read

Each importer turns its source's conventions into explicit fields, so no export has to guess:

- **IBM.** Gate errors are average infidelities from randomized benchmarking. An error at or
  above the physical bound d/(d+1) becomes `disabled: true`. In practice, such an error is
  `gate_error = 1`, IBM's marker for a broken gate. A qubit or gate marked not operational also
  becomes `disabled: true`. `rz` is virtual. Device-wide defaults are medians over working qubits
  and gates. The medians leave out a qubit marked not operational and every gate on that qubit.
  `from_qiskit_backend` disables a gate on every qubit or pair that the backend's `Target` does
  not list for that gate. In a CSV or a pull, a qubit or pair with no published error for a gate
  takes that gate's median. `provenance.notes` names the qubit or pair. Two-qubit errors carry
  `"includes": ["1q_dressing"]`. The importer checks each value before it applies these rules. An
  error or a probability must be a finite number from 0 to 1. A duration must be a finite number
  0 or more. In BackendProperties data, an `operational` flag must be 0 or 1. The importer
  refuses any other value. A T1 or T2 that is not a positive finite number counts as missing, and
  `provenance.notes` names it.
- **Quantinuum.** The importer repeats the analysis code in Quantinuum's repository
  (`qtm_spec`). The importer pools the randomized-benchmarking decays over gate zones and
  converts them to the average infidelity per native gate. The importer adds leakage to that
  infidelity as `qtm_spec` reports it. The emulator parameters (`p1`, `p2`) describe a different
  model, and the importer does not use them. The importer keeps leakage rates as `leakage`
  effects.
- **Google.** The importer stores single-qubit RB Pauli errors as `process_infidelity`. The
  importer removes the single-qubit part from the two-qubit XEB errors per cycle, as Google does
  to infer per-gate errors. The importer records the coherent part of each fSim error as a
  `coherent_overrotation` effect.
- **IonQ.** The importer stores medians with `statistic: "median"` and writes the metric reading
  in `assumption`. SPAM fidelity becomes a symmetric readout error, and preparation stays
  unknown.

## Add a source

A source is one module in `src/noisevault/sources/`. To add one:

1. Write `src/noisevault/sources/<name>.py`. Convert the source's data into a profile dict.
   Build the profile with `Origin(name, hint).profile(data)` from `noisevault.sources`. `name`
   says where the data came from, such as the file name or the URL. `hint` is the next step a
   user can take, such as `f"correct that value in {path.name}"`, or `OLDER_HINT` for a source
   that takes a date. A value the profile format refuses then raises `SourceDataError` with one
   line that names the source, the field and the value. For data the module cannot read, raise
   `origin.refuse(problem)`. Import optional packages inside functions, never at module level,
   so a core install keeps working.
2. Fill `provenance` with `data_kind`, `source_kind`, `source`, `source_url`, `license`,
   `attribution`, `redistributable`, `retrieved_at` and the `source_hash` of the raw bytes.
   Set `redistributable: "yes"` only when the license allows redistribution.
3. Make every interpretation explicit. Use the metric key that matches what the vendor
   measured. Set `method`, `statistic` and `includes`. Add an `assumption` where the vendor does
   not define the meaning. A vendor's sentinel values become `disabled`. Never store a per-cycle
   error as a per-gate error without converting it.
4. Expose `bundled_profiles() -> list[Profile]`. Return `[]` unless the data has an open
   license.
5. For a live source, expose `pull(device, *, at=None) -> Profile`. Register the module in
   `_PULL_SOURCES` in `src/noisevault/catalog.py`. Raise `SourceUnavailable` for every network
   or lookup failure. Read each reply with `read_reply` from `noisevault.sources`.
   `read_reply(raw, url, shape, sender=..., hint=...)` raises `SourceUnavailable` that names the
   URL when the reply is not JSON or does not match `shape`, a pydantic `TypeAdapter`.
6. Pass an error's next step as `hint=`, not in the message. `nv` prints the hint on its own
   `hint:` line.
7. For a bundled source, add the module to `SOURCES` in `scripts/build_catalog.py`. Run
   `python scripts/build_catalog.py`. Commit the regenerated profiles and NOTICE.
8. Add tests in `tests/test_sources_<name>.py` that run on saved fixtures, not the network.

[CONTRIBUTING](../CONTRIBUTING.md) covers the test commands.
