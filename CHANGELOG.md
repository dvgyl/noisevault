# Changelog

This file lists all notable changes to NoiseVault. Versions follow
[Semantic Versioning](https://semver.org).

## Unreleased

### Added

- **Factors for the error a calibration leaves out.** A profile can set `unmodeled_error`, with
  a factor on its gate error rates, a factor on its readout error rates, or both. Every export
  and the `nv check` reference apply the factors. `nv diff` compares the errors after it applies
  the factors. A factor raises each error channel to that power. As a result, a factor of 1
  keeps the stated error, and no factor takes an error out of its physical range. No factor
  changes T1, T2 or preparation error. An error with no valid power, such as a readout pair no
  better than chance, stays as stated.
  - Each export's report states the factors on its second line, which starts with
    `unmodeled error:` and names each error left as stated. `report.to_dict()` holds the same
    text under `unmodeled_error`.
  - `nv show` prints the calibration as stated, then the factors on an `unmodeled` row.
  - A factor and its interval print with three significant digits, or with more when three
    would print two different values alike, as in `x1.008 (95% interval 1.001 to 1.015)`.
  - The field is part of the fingerprint, and `nv cite` names the factors.
    `profile.uncorrected()` returns the profile without the field, with the calibration's
    fingerprint. `profile.calibration_fingerprint` is that fingerprint, the one a counts file
    names. Profiles without the field keep their fingerprints.
  - A NoiseVault install that does not know the field refuses a profile that sets it, because
    format 1.0 forbids unknown keys.

  See [Unmodeled error](docs/profile-format.md#unmodeled-error).
- **IBM calibration history from Hugging Face.** `nv.from_calibration_archive(path, device,
  at=...)` reads a local copy of the Hugging Face dataset `phanerozoic/qiskit-calibration-drift`.
  The function returns the IBM calibration of `ibm_fez`, `ibm_kingston`, `ibm_marrakesh` or
  `ibm_torino` that was in effect at `at`. A provenance note names the values calibrated more
  than 7 days before `at`, or before the newest calibration when you give no `at`. The note
  leaves out values that the profile does not use, such as the duration of a disabled gate. It
  also leaves out each T1 or T2 that the import treats as missing and each archive row with no
  value. The zero errors that `virtual: true` replaces are not in the note. The note names an
  old `rz` value when that value calibrates or disables `rz`. The importer reads the file once.
  Thus the values of a profile and its `source_hash` come from the same file, also when
  another program replaces the file during the import.
  `nv.calibration_archive_devices(path)` gives the times `first` and `last` of each device.
  `first` is the earliest `at` that gives a profile. An earlier `at` raises `nv.SourceDataError`,
  which names that time. Install
  `noisevault[hf]` (pyarrow 14.0.1 or later). `nv doctor` lists pyarrow. See
  [IBM calibration archive on Hugging Face](docs/data-sources.md#ibm-calibration-archive-on-hugging-face).
- **Scoring a profile on counts from a device.** `nv compare REF COUNTS` scores a profile on counts
  that a device measured. It fits one factor on every gate error rate and one factor on every
  readout error rate. Each factor has a 95% interval. `nv compare` also tests whether one pair of
  factors explains every circuit. It exits with status 0 for every fit result. `-o FILE` saves the
  profile with the fitted factors in `unmodeled_error`, and `--json` prints the result as JSON.
  Before the fit, `-o` refuses the counts file, the profile file and any path in the vault or among
  the bundled profiles. In Python, `profile.compare(counts)` returns the same result, and
  `result.fitted_profile()` returns the profile with the factors. `result.summary()` returns the
  text that `nv compare` prints. `result.summary_lines()` returns the same lines, each with the
  number of leading characters that `nv compare` prints in bold, for a program that styles the
  output itself.
  - `noisevault.counts` reads and writes counts files. `load_counts(path)` reads one as a
    `MeasuredCounts`. `plan(profile)` returns the circuits to run, with every wait written as a
    `delay`. `simulate(profile, circuits, shots=..., seed=...)` draws counts from a profile.
    `SAMPLER_V2_OPTIONS` maps each Qiskit Runtime `SamplerV2` option that `load_counts` checks,
    such as `twirling.enable_gates`, to the value that `load_counts` requires. The map is for
    code that runs the circuits through `SamplerV2` itself.
  - `plan()` uses only qubits where the profile allows `measure`. A layout with another qubit
    raises `LayoutError`. When a circuit must wait on a qubit where the profile disables
    `delay`, `plan()` raises `LayoutError`. The hint says to pass `layout=` with qubits that
    allow `delay`.
  - `load_counts` raises `nv.CountsError` for a file it refuses. The message names the file and
    the field, and `hint` says how to correct the file.
  - A circuit in a counts file has at most 10^10 shots. `simulate` refuses more shots.
  - `simulate` and `nv compare` use the reference simulator, which follows the rule of the exports
    for profile effects. The reference simulator leaves out an effect with `allow` set to
    `"omit"`, and `nv compare` names that effect in a note. An example is `the reference simulator
    leaves out effect coherent_overrotation on x, because the profile sets allow to 'omit'`. For
    any other `allow` value, `simulate` and `nv compare` refuse the profile with
    `UnsupportedEffect`. The hint says `set allow to 'omit' to leave the effect out`.
  - The reference simulator refuses a delay or a measurement on a qubit where the profile
    disables that operation, as the exports do. `simulate` names the circuit in the error.
    `nv compare` refuses counts whose circuits use such a delay or measurement.
  - The fit finds the maximum and each interval end to within 0.001 in log-likelihood, at every
    number of shots that a counts file accepts. At each interval end, the fit finds the other
    factor to the same limit. The search for the maximum stops only when every neighboring point
    is less than 0.001 below the best point. Thus a factor that the counts do not constrain cannot
    stop the search early.
  - Each maximum that the fit reports is the likelihood at the reported factors, so no reported
    maximum is above the true maximum. Before, a circuit with many more shots than the others
    could give wrong intervals. For one qubit with a circuit of 10^8 shots and a readout circuit
    of 4000 shots, the readout interval ended at 2.56. It now ends at 1.35.
  - The fit keeps each other peak of the likelihood where the likelihood-ratio statistic is at
    most nine times the interval cutoff. For each drawn set of counts, the fit searches for the
    maximum near every kept peak. Thus a drawn set whose maximum is at a second peak gets that
    maximum. Before, the search was only near the estimate. In an example with two peaks, the
    p-value was 0.012, and it is now 0.0075. The fit also computes the likelihood of each drawn
    set at each kept peak. When the grids are too coarse for a kept peak, the fit climbs from that
    peak in each set. Thus no drawn maximum is below the likelihood at a kept peak. In an example
    where one grid spans the whole range, the p-value was 0.586, and it is now 0.564.
  - Below the lowest relaxation floor, every gate factor gives the same probabilities, so the
    likelihood is flat there. A climb along the gate axis no longer goes below that floor. Thus it
    cannot stop on the flat part. The fit also climbs from the best grid point when an end of the
    factor range is higher than that point. Before, for a factor far below the estimate, a drawn
    maximum could be 5.76 lower than the likelihood at a grid point.
  - The slope of a circuit counts in the Fisher information only when its probabilities 0.1% below
    and 0.1% above the estimate differ. The difference must be more than the rounding error of the
    reference. Before, rounding error could remove a degree of freedom. For one circuit that
    readout error does not move, the fit was not testable. It now has 1 degree of freedom and
    p = 0.733.
  - `nv compare` reports both factors as not identified only when the smallest eigenvalue of the
    Fisher information is below 4.4e-16 times the largest. That limit is the floating-point
    precision of the eigenvalues. Before, the limit was 1e-6. Thus a circuit with 10^10 shots
    beside a circuit with 4000 shots removed the estimates of both factors.
  - A circuit moves with a factor when its outcome probabilities change by more than the
    rounding error of the reference. For a circuit with n operations on q qubits, that limit is
    (n + q + 1) times 2.2e-16. Before, a change of at most 1e-9 in total variation distance did
    not count, and `nv compare` reported the factor as not identified. An example is ten
    circuits of 10^10 shots each, where the probabilities change by 8e-10 over the range. Their
    counts now give a gate factor of at least 8.5.
  - Each interval end gets a second test with the distribution of the likelihood-ratio statistic
    at that end. The test uses the exact probability of each combination of counts when two
    conditions hold. Each circuit has at most two outcomes, and the counts give at most 4096
    likely combinations. Otherwise, the test draws 400 sets of counts. A set whose best factor is
    at an end of the 0.05 to 20 range gets its statistic at that end. In the one-qubit examples of
    the docs, an interval from a few error shots contains the true factor with a probability of
    0.95 or more.
  - `-o` saves each factor and interval end with six significant digits, or with more when six
    would save two different values alike.
  - With no degrees of freedom left, `nv compare` reports the fit as not testable, with two
    exceptions. A factor at an end of the 0.05 to 20 range, or a deviance above 3.84, shows that
    the fit misses the measured frequencies. Such a fit gets the test and a p-value, and the
    verdict says that no factors from 0.05 to 20 give the measured frequencies.
  - When a command that takes a profile gets a counts file, the error says that the file is a
    counts file and not a profile. When the two arguments are in the wrong order, `nv compare`
    names the correct order.
  - `scripts/run_on_ibm.py` runs the `nv compare` circuits on an IBM device through your IBM
    Quantum account. It writes a counts file bound to the calibration in effect when the job ran.
  - `scripts/run_on_ibm.py` never replaces a file that it did not write: the counts file and its
    `.timing.json`, `.profile.json` and `.job.json` files. The script deletes only a file that it
    created. `--collect JOB_FILE` saves a submitted job's counts to a new `-o` and never submits a
    job. If the script cannot save a job id, `--job-id JOB_ID` collects that job. A failed IBM
    request while the script opens the device gives an error and a retry hint, not a traceback.
  - If the job file exists when the script starts, the same command collects that job and never
    submits another. Before it opens a saved job, the script checks the whole saved plan. The
    check includes unique circuit names and the limit on outcomes. The script also checks the
    plan against its profile. A qubit that the device does not have, or a gate that the profile
    does not define, stops the script before it opens the job. A damaged job file gives
    an error, and if the file names a job, the hint names that job. If another program rewrites
    the job file or puts a link in its place, the script keeps that file or link. The script
    writes the job id in parts when one write does not hold all of it. Before each part, the
    script checks the job file again, so it also keeps a file that changes between two parts.
  - In two cases, the script saves the job to `<stem>.<job id>.job.json` and prints the
    `--collect` command. In the first case, another program replaces or rewrites the job file
    while the script submits the job or adds the job id. In the second case, a failed write
    leaves a part of the job id in the job file, and the script cannot remove that part.
  - The commands that the script prints quote each path, and they work for a file name that
    starts with a hyphen. After a failed collection, the hint first says how to collect the job
    again, then how to submit a new job.

  See [Counts format](docs/counts-format.md),
  [Measure a profile against hardware](docs/recipes.md#measure-a-profile-against-hardware) and
  [How nv compare fits the factors](docs/limitations.md#how-nv-compare-fits-the-factors).
- **A filter on the qubit pairs of a suggested layout.**
  `noisevault.layout.suggest_layout(profile, n, usable_pair=f)` takes a function `f(a, b)` of two
  qubits, with `a < b`. The chain then has no neighbors `a` and `b` for which `f(a, b)` is false.
  See [Choose qubits](docs/frameworks.md#choose-qubits).
- [docs/frameworks.md](docs/frameworks.md) names a Qiskit Aer defect that can stop Python with
  a segmentation fault during `sim.run`. The note gives the `density_matrix` method as a
  workaround.

### Changed

- **Messages and help text.** Error messages, warnings, hints and help text now follow one
  writing standard, ASD-STE100 Simplified Technical English. Each concept has one term. Each
  message names qubits in one form: `qubit 3`, `qubits 0-2` for a pair, and
  `qubits 0-1, 1-2 and 2-3` for a list of pairs. Code that matches the text of a message must
  match the new text. Examples:
  - `cz has no calibration on (0, 2) and connectivity does not allow it` is now
    `cz has no calibration on qubits 0-2, and the connectivity does not allow cz there`.
  - `'ibm_fez@2025-13-40': 2025-13-40 is not a calendar date; give one such as 2025-02-26` is now
    `'ibm_fez@2025-13-40': 2025-13-40 is not a calendar date. Give one such as 2025-02-26`.
  - `nv list --help` says `for example trapped_ion` in place of `e.g. trapped_ion`.
- **Profile ids that end in `.json` or `.gz`.** Validation now refuses such a profile id, because
  `nv.load` and the `nv` commands read the id as a file path. The error suggests an id that loads.
- **`nv check --json` and `CheckResult.to_dict()`.** Each framework now gives `reports`, with one
  full export report for each export configuration that the check ran. Each report names every
  affected qubit. NoiseVault removes the `report` key and its `approximated`, `omitted`, `unknown`
  and `summary` keys. It also removes the `FrameworkCheck` fields `approximated`, `omitted`,
  `unknown` and `report`. Each circuit entry and `worst` also give `deviation`.
- **Qubit placement in the Cirq and Stim exports.** A Cirq `layout` with both `i` and
  `cirq.LineQubit(i)` as keys now raises `LayoutError`. Automatic placement in the Cirq export
  and `noisevault.stim.layout_from_coords` raise `LayoutError` when two enabled qubits have the
  coords of a circuit qubit. The error names the qubits, and the hint says to give a layout.
- **PennyLane measurements that share shots.** With shots, two Pauli words can read one wire in
  different bases while a third word commutes with both. The PennyLane noise model now raises
  `NoiseVaultError` for such measurements, because `default.mixed` decides which shots the third
  word shares. The hint says to wrap the QNode in `qml.transforms.split_non_commuting`.
  `qml.probs(op=...)` of an identity or zero observable, such as
  `qml.probs(op=qml.I(0) @ qml.I(1))`, also counts as such a third word.
- The repository moved to https://github.com/dvgyl/noisevault. Install commands, links and the
  bug-report hint use the new address.
- `Report.omitted`, `Report.unknown`, `Report.unmodeled_error` and `Comparison.notes` now hold
  `LociText` values, not strings. Use `.full` to get every qubit and `.short` to get at most four.
  `to_dict()` and the printed summaries do not change.

### Fixed

- **Revalidating a profile after reading its fingerprint.** `Profile.model_validate(profile)`, a
  pydantic model or `TypeAdapter` with a `Profile` field, and `Profile(**dict(profile))` raised
  `Extra inputs are not permitted` after a read of `fingerprint`, `artifact_hash` or `table`.
  These three calls now accept the profile. With pydantic 2.13, an assignment to `fingerprint`,
  `artifact_hash` or `table` replaced the stored value. The assignment now raises
  `ValidationError`, like any other change to a frozen profile.
- **Loading from a vault whose index disagrees with its files.** `nv.load`, the commands that
  take a ref, and `nv pull` used the vault's index, `.index.json`, and did not check the index
  against the files. An index entry with the wrong date let `ibm_manila@2024-06-04` load a
  calibration from 2024-06-03, and `nv pull` then saved over that calibration. A load could also
  return another calibration when a save replaced a file during the load. NoiseVault now uses the
  index only when its version and digest show that NoiseVault wrote it, and otherwise reads the
  files again. `nv.load` checks the profile that it read against the calibration that the ref
  selected and against the `expect=` pin. If the profile does not match, `nv.load` reads the
  vault again. An `.index.json` that is not a regular file no longer stops `nv list`.
- **Importer input that is damaged or incomplete.** Three importers returned wrong values with
  no error. Each now raises `SourceDataError` that names the file and the field or line.
  - `from_ibm_csv` lost every row after a quote left open, so a file could import with no
    two-qubit records. For a row cut short, `from_ibm_csv` read the missing cells as blank, and a
    blank Operational cell enabled a disabled qubit. `from_ibm_csv` now refuses a file with a quote
    left open or a row cut short. A CSV whose lines end in a bare carriage return failed with a
    `csv` module error. Such a CSV now reads like any other CSV.
  - `from_braket` read a time with no `unit` as seconds, so a T1 of 18.5 us became 18,500,000
    us. It read a standardized v1 or v2 two-qubit fidelity with no `fidelityType` as randomized
    benchmarking. On one IQM pair, the misread changed the error from 0.009 to 0.013. Braket's
    schemas require both fields, and `from_braket` now refuses a file that leaves one out. A v3
    fidelity with no type still reads as randomized benchmarking, the only type v3 defines.
  - `noisevault.sources.quantinuum.from_repository` accepted counts that cannot be right. A shot
    count of 1 in H2-2's single-qubit file gave an error of 4.4e-16 instead of 7.85e-5, and an
    empty first zone gave 0.5. The importer now refuses a shot count that is not a positive whole
    number and a count outside 0 to the shot count. It also refuses a sequence length that is
    not a whole number, a zone with no sequence lengths, and an empty or misshapen map.
- **Gate order of a profile from a Qiskit backend.** `nv.from_qiskit_backend` wrote the gates in
  an order that changed from one Python process to the next. As a result, one calibration could
  give different file bytes. The importer now reads the backend's gates in name order. The
  fingerprint and the bundled files do not change.
- **Vendor values and qubit indices of the wrong type or value.** The Braket, IonQ, IBM and
  Hugging Face importers used some values with no check. Such a value could change with no
  error, go missing with no note, or stop the import with an unexpected Python error. A Braket
  T1 of `true` became 1 s. An archive `qubit_b` of 2.5 became qubit 2. A negative IonQ gate time
  gave no gate time. An IBM gate on the wrong number of qubits stopped `nv pull` with a
  `ValueError`. An IBM `rz` gate on qubit 99 of a 5-qubit device gave no error. A Braket
  `updatedAt` of `false` gave the time of the service refresh. Each importer now checks these
  values where it reads them, also in metadata, headers, timestamps and the gates that it skips.
  A bad value raises `nv.SourceDataError`, which names the source, the field and the value. A
  Braket qubit id that is not a whole number, such as `q1`, raises the error. A value of the
  wrong type in an IonQ reply gives the error for a reply of the wrong shape. An IonQ median
  above 1 is corrupt data, and a provenance note says so. A provenance note also counts the
  archive rows with no property or no `calibrated_time`, which the profile does not use. An
  archive device with no such row, or with no `observed_time`, raises the error. Archive rows
  with no `backend` also raise the error. An archive timestamp with no time zone is UTC, as the
  dataset stores it. Before, such a timestamp stopped an import with `at=` with a `TypeError`.
- **Damaged profiles and vendor replies.** Each case below now gives one error that names the
  damaged input. Before, a command printed an unexpected error and asked for a bug report, or
  printed a message such as `Expecting value: line 1 column 1 (char 0)` that named nothing.
  - A profile file nested deeper than the JSON parser takes. A load of such a file raises the
    `json.JSONDecodeError` that any damaged file gives. The error gives the depth and the
    position of the deepest bracket. The commands call the file damaged.
  - Free-form data in `benchmarks`, `extensions` or `provenance.extra` nested more than 256
    levels deep. The format now limits that data to 64 levels, so a profile nested deeper is
    invalid and `nv validate` names the field. A profile nested 65 to 256 levels deep loaded
    before, and NoiseVault now refuses that profile too.
  - An importer's file that is not UTF-8 text, or a Braket or Quantinuum file that is not JSON.
    `from_braket`, `from_ibm_csv` and the Quantinuum importers raised `UnicodeDecodeError`,
    `json.JSONDecodeError` or, for an integer too long to parse, `ValueError`. These importers
    now raise `nv.SourceDataError` that names the file and what is wrong with it, as in
    `saved.json is not JSON (expecting value at line 1, column 7)` or
    `saved.json is not UTF-8 text (byte 0xb5 on line 2)`.
    `noisevault.sources.quantinuum.from_spec_csv` also refuses a quoted cell that continues on
    the next line, and the error names the line, as `from_ibm_csv` does.
  - A reply from IBM's public endpoint or IonQ's API that is not JSON, or is JSON of the wrong
    shape such as `[]` or `"maintenance"`. The error names the URL, and the first wrong field
    when there is one. The hint says to try again later. A damaged IBM device list or
    configuration is the exception. Those replies only add detail, such as the processor name,
    so `nv pull` leaves the detail out and gives no error. Before, some IonQ replies of the wrong
    shape gave a profile from an older record, or one dated 1970, with no error.
  - An IBM time in a unit NoiseVault does not read, such as a T1 in `min`, from `nv pull`,
    `nv.from_qiskit_backend` or `nv.from_calibration_archive`. The error names the time, the
    qubit or gate, and the unit.
  - A vendor value that no profile can hold, such as a readout error of 1.5. `nv pull` and every
    importer raise `nv.SourceDataError`, which names the source and the first such value with
    its qubit or gate. When the source takes a date (`--at`, or `at=` in Python), the hint says
    to pass an earlier one. For an IBM CSV, a Braket file or a Quantinuum spec sheet, the hint
    says to correct the value in the file. Before, `nv pull` said that the profile was not valid
    and to run `nv validate FILE`, but a pull has no file. The importers raised pydantic's
    `ValidationError`. Every IBM path now checks each value before any vendor rule runs. It
    refuses a value that is not finite, such as a gate error of `inf`, and an error or a
    probability outside 0 to 1. It also refuses a negative duration and, in `BackendProperties`
    data, an `operational` flag that is not 0 or 1. Before, a vendor rule for a disabled gate or
    for readout could remove such a value with no error. A gate error of `inf` disabled the gate,
    and an `operational` flag of `nan` kept the qubit enabled.
  - An IonQ record whose qubit count is missing from the record and from IonQ's backend listing.
    The error names the record, and the hint says to pass an earlier `--at`.
  - An IonQ backend listing with no `qpu.` backend in it. The error ended with "it lists" and
    named nothing. The error now says that the listing "names no QPU", and the hint says to try
    again later.
- **IBM parameters with two values.** When IBM calibration data gave one parameter of a qubit or
  gate two different values, `nv pull` and the importers used the last value. They now raise
  `nv.SourceDataError`, which names the parameter, the qubit or gate, and both values. An
  identical repeat counts once.
- **JSON keys that occur two times.** When one JSON object had the same key two times, NoiseVault
  used the second value with no error. Now each JSON input refuses such an object, and the error
  names the key, as in `run.json has the key device.name twice`. The check covers profiles,
  counts files, importer files, vendor replies and the job file of `scripts/run_on_ibm.py`. The
  `model_validate_json()` and `parse_raw()` methods of each NoiseVault model, such as `Profile`
  and `MeasuredCounts`, also refuse such JSON text. They raise the same pydantic
  `ValidationError` as for JSON that is not valid. For a profile or a counts file that a command
  reads, the hint says to keep one of the two keys. NoiseVault ignores a vault index with such a
  key and reads the vault files again.
- **A vault file saved during `nv pull`.** `nv pull` never replaces or deletes a vault file that
  another process saves during the pull, also on exFAT and FAT drives. The pull applies the usual
  vault rules to that file. If another process replaces the hidden copy that `profile.save`
  writes first, `profile.save` writes nothing and raises a `NoiseVaultError`, which is also a
  `FileExistsError`. A pull that moves an older import of its calibration aside deletes that
  import only after the pull saves its own file. Before, a failed save lost both files. If the
  save fails, the pull puts the import back. If another file is at that path, the pull keeps the
  import under a hidden name, and the error names that file. A pull that finds its profile in the
  vault reads that file again before it reports the result.
- **Cirq placement next to a disabled qubit.** When a disabled qubit has the same coords as an
  enabled qubit, `GridQubit` placement now uses the enabled qubit.
- **Correlated PennyLane samples.** With shots, Pauli words that commute on each wire, such as
  `qml.sample(qml.Z(0))` and `qml.sample(qml.X(1))`, now share one set of readout operations and
  one tape. Their samples stay correlated. `qml.probs(op=...)` of an identity or zero observable
  shares shots with such words in `default.mixed`. The model now reads such a measurement in the
  basis of that shot group.
- **PennyLane Pauli words with a zero coefficient.** PennyLane drops a Pauli word with a
  coefficient of at most 1e-8 before it groups measurements. The PennyLane noise model kept such
  words. As a result, `qml.sample(qml.X(0) + 0*qml.Y(1))` and `qml.sample(qml.X(1))` got
  separate shots, and `qml.dot([1, 0], [qml.X(0), qml.Z(0)])` got no readout error. The model now
  drops these words before it picks the measured basis and the shot groups.
- **Disabled measurements, resets and delays.** Every export now refuses a measurement, a reset
  or a delay on a qubit where the profile disables that operation. A disabled gate gets the same
  refusal. Before, some exports ran such an operation with the noise of an allowed operation.
  - The Cirq and Stim exports raise `DisabledGateError` for a reset, a measurement or a Cirq
    `WaitGate` on such a qubit. The Stim `MR`, `MRX` and `MRY` instructions need both `measure`
    and `reset`. Stim has no delay instruction.
  - The Stim export also refuses `II` when the profile disables `ii`. A measurement record or a
    sweep bit can control a Pauli. The Stim export refuses such a Pauli when the profile
    disables `x`, `y` or `z` on its qubit. `noisevault.stim.layout_from_coords` does not choose
    a placement that puts a gate, a measurement or a reset where the profile disables it.
  - The Qiskit `Target` leaves out `measure`, `delay` and `reset` on a qubit where the profile
    disables them. `transpile` then refuses a circuit that needs one of them there, and `sim.run`
    raises `DisabledGateError`.
  - In the PennyLane export, `qml.measure` and terminal measurements on such a qubit raise
    `DisabledGateError`. `qml.measure(wire, reset=True)` also needs `reset`. PennyLane has no
    delay operation.
  - The PennyLane export checks `measure` and applies readout only on the wires that a
    measurement reads. A Pauli observable reads only the wires of its simplified words, so
    `qml.expval(qml.X(0) + 0 * qml.Y(1))` and `qml.expval(qml.X(0) @ qml.I(1))` read only wire
    0. A measurement without wires, such as `qml.probs()`, reads every device wire, also a wire
    that the circuit does not use. `qml.add_noise` on a tape cannot see the device. In that
    case, the export raises `DisabledGateError` when a device wire can map to a qubit that
    cannot measure. The hint says to pass `wires=` to the measurement or to apply
    `qml.add_noise` to the QNode.
  - Each `DisabledGateError` message names qubits in the same form as other messages, as in
    `cx on qubits 0-1 is disabled in this profile`. Before, the message read `cx on (0, 1)`.
- **Qiskit `Target` durations and errors.** The `Target` from the Qiskit export gave `reset` no
  duration, so `transpile` with `scheduling_method="alap"` failed for a circuit with a reset. Each
  `reset` now has its duration from the profile. The `Target` also gave `reset` no error, so
  `transpile` could put a reset on the qubit with the higher preparation error. Each `reset` now
  has the preparation error of its qubit. With `readout=False`, the simulator applied no
  readout error, but the `Target` gave `measure` the readout errors of the profile. As a result,
  `transpile` could place a circuit on a worse qubit. The `Target` now gives `measure` an error of
  0 and keeps its duration.
- **Errors in 2-qubit gates that `nv check` did not find.** `nv check` now also runs a
  `chain_mirror` circuit. The circuit runs the chain of 2-qubit gates and its inverse two times.
  The first time, only the first qubit starts in a superposition. The second time, every qubit
  starts in a superposition. The circuit finds 2-qubit gate errors that the other circuits do not
  show. An example is an X error on the target qubit of `cx` after `h`. `noisevault.counts.plan()`
  and `nv compare` leave out `chain_mirror`, so counts files keep their circuits.
- **Natives that `nv check` did not run.** The Cirq check said that Cirq has no swap gate and did
  not run a `swap` native. The Cirq check now runs `swap`. `operation_for("ms")` in the PennyLane
  export now builds `ms(0, 0)` as `qml.IsingXX(pi/2)`, so `nv check` checks an `ms` native in
  PennyLane. At other phases, `operation_for("ms")` raises `ValueError`. `operation_for("sdg")`,
  `operation_for("tdg")` and `operation_for("sxdg")` returned `None`. They now build PennyLane
  adjoint operations, so `nv check` runs these natives in PennyLane. Before, `nv check` said that
  PennyLane has no such gate. `nv check` applies the PennyLane noise at the top level, so each
  native gets its own noise.
- **Qubits that cannot measure in `nv check`.** `nv check` and `profile.suggest_layout(n)` could
  choose a qubit where the profile disables `measure`. They now choose only qubits that can
  measure. A check layout with another qubit raises `LayoutError`. The default chain of
  `nv check` also avoids a pair that only a custom gate calibrates. Before, such a pair could
  make the check use fewer qubits.
- **Hints that named a step that cannot run.** With `unknown_gates='error'`, an export refused
  a gate with no error metric and said to pass `unknown_gates='typical'`. The hint now gives
  that step only when the typical native gate's noise is usable on those qubits. Otherwise, the
  hint names a native gate that needs an error metric there. When no native gate is usable
  there, the error has no hint.
  - The Qiskit export names `profile.to_cirq()` only when Cirq can run a native of the needed
    size on some qubits. A virtual native with no error metric counts. Before, the Qiskit error
    for such a profile said that `profile.to_cirq()` cannot run one. The Qiskit report names
    `unknown_gates='typical'` only when that option gives the gate noise. Sometimes no native
    has an error metric on any enabled qubit or pair. The Qiskit error then says so, and the
    hint says to calibrate a native that Qiskit provides.
  - A Stim refusal on a 2-qubit gate suggests `layout=` only when the failing gate runs on a
    pair of qubits that can measure. Sometimes that gate does not run in both directions on
    every pair with a usable 2-qubit gate. The hint then names one pair where the gate runs.
    `noisevault.stim.layout_from_coords` gives no layout hint when the circuit has more qubits
    than the device has enabled qubits. It also gives none when no enabled qubit or pair allows
    an operation of the circuit.
  - When `nv check` or `noisevault.counts.plan()` refuses a layout, the hint names a layout that
    has check circuits, as in `use qubits that can measure, such as layout=[0, 1, 2, 3]`. When
    no layout has check circuits, the hint says to use a profile that calibrates a 1-qubit
    native gate with a known unitary.
  - A layout that puts a circuit qubit on a disabled qubit raises `LayoutError`. The hint said
    to choose another qubit, also when the profile has too few usable qubits for the circuit.
    The hint now names `profile.suggest_layout(n)` with the number of circuit qubits as `n`. It
    does so only when that call finds a chain. The Cirq simulator gives the noise model one part
    of the circuit at a time. Thus, with no `layout=`, the Cirq export cannot count the circuit
    qubits, and its error for a disabled qubit has no hint.
  - A Stim refusal on a 2-qubit gate names `profile.suggest_layout(n)` only when that call finds
    a chain. It names `noisevault.stim.layout_from_coords` only when that function places the
    circuit.
  - A Cirq `GridQubit` at coords that the profile does not have gets a hint with example coords.
    The examples now include only coords with one enabled qubit. Sometimes two Cirq qubits map
    to the same device qubit by default. That error now has no hint, because the export cannot
    count the circuit qubits.
  - The Qiskit simulator refuses a circuit with an instruction that the device does not provide.
    The hint said to transpile the circuit first, but `transpile` sometimes could not compile
    the circuit. For example, Qiskit stopped with a Rust panic when the circuit used more qubits
    than the profile has enabled. The simulator now transpiles the circuit with
    `seed_transpiler=0` and checks the result before it gives the hint. When that step fails,
    the error has no hint. The hint now includes `seed_transpiler=0`, so it names the exact step
    that the simulator checked.
  - A Qiskit circuit wider than the device gets a hint for the qubit count. The hint compared the
    qubits that the circuit uses with all device qubits, also the disabled qubits. It now names
    a narrower circuit only when that circuit transpiles for the simulator. Otherwise, it names a
    profile with enough enabled qubits, or it gives no hint.
  - The Qiskit report said to insert idle delays with `scheduling_method='alap'`. That step
    fails when the profile gives no duration for an instruction, such as `measure`. The report
    now names the step only when every instruction in the `Target` has a duration.
- **Braket device names that are not a profile id.** A Braket file name or `device=` can give a
  name that is not a valid profile id. The hint then says to rename the file or to pass `device=`
  with another name.
- **Braket v3 provenance note.** The note said that every qubit and pair gets the same v3
  values, also when some qubits use their own `oneQubitProperties` fidelity. The note now names
  those qubits.
- **Format 0.1 files that mark a gate or qubit not operational with a string.** The upgrade read
  `operational` by truthiness, so `"false"`, `"no"`, `"off"` and `"0"` left the gate or qubit
  enabled. The upgrade now reads the flag as NoiseVault 0.1 did, so these values disable the gate
  or qubit. A value that rule cannot read raises a `ValueError` that names the field.
- **Saving profiles from several threads.** `profile.save(path)` read the umask by setting the
  process umask to 0 and back. A file that another thread created in that moment did not get
  the umask, and two saves at once could leave the umask at 0. `profile.save` no longer changes
  the umask. A new file still gets the mode the umask allows, and a replaced file keeps its mode.
- **A circuit wider than the device on the Qiskit export.** The simulator from `to_qiskit()`
  refused a circuit with more qubits than the device and said to transpile it, but `transpile`
  refuses that circuit too. The hint now names a step that works. For idle extra qubits, the hint
  says to build the circuit on at most the device's qubit count. For a circuit transpiled for a
  larger backend, the hint says to transpile the original circuit. For a circuit that uses more
  qubits than the device has, the hint says to load a profile with enough qubits.
- **Command line.**
  - When IBM had no calibration of a device before the `--at` date,
    `nv pull --source ibm-account` gave advice for retired devices. It now says that IBM returned
    no calibration before that date, and the hint says to pick a later date.
  - When `nv pull --source ibm-account` cannot open a device, the hint now names the devices
    that the account can see. Before, the hint said to list them with
    `QiskitRuntimeService().backends()`.
  - A hint that prints an `nv validate` or `nv check` command now quotes the path, and puts `--`
    before a path that starts with a hyphen. Thus the printed command works for a file name with
    a space or a leading hyphen.
  - `nv show` counted gate loci with no error metric as "without error", a label that read as
    free of error. Exports give those loci the typical native gate's noise. `nv show` now counts
    them as "uncalibrated".
  - `nv diff` warned that the second profile was older than the first even for two different
    devices. For two calibrations of one device on one day, the warning named both by that date,
    as in "ibm_manila@2024-05-27 is older than ibm_manila@2024-05-27". It now warns only when both
    profiles are of one device. When the dates match, the warning gives the times.
  - For a `.json` or `.json.gz` file that is not valid JSON, the hint said to give a profile file
    with one of those names. For a damaged gzip file, the hint said to copy or pull the file
    again. Both hints now say that the file is damaged or truncated and to pull or export the
    profile again. A file with another name gets the hint to give a profile file. For a file that
    is not UTF-8 text, the error said "not JSON (not UTF-8 text)". The error now names the first
    byte that is not UTF-8 and its line, as in `bad.json is not UTF-8 text (byte 0xff on line 2)`.
    For a file cut short inside a string, the error no longer reads
    "Unterminated string starting at at line 1".
  - When a vault copy replaced a bundled calibration, the error for a dated ref with no
    calibration on that day listed the replaced calibration twice. The error now lists each
    calibration once.
  - In a narrow terminal, `nv check` wrapped cells such as "exact + 20000 shots" over several
    lines. It now leaves out the tolerance column, then the circuits column, and keeps each cell
    on one line.
  - A sampled `nv check` passes only when each outcome is within its own tolerance. Each row
    showed the TVD and one tolerance for all outcomes, so a FAIL row could show a TVD below its
    tolerance. The table column is now `deviation`. Each row shows the deviation and the
    tolerance that decide the result.

## 0.2.0 (2026-10-01)

Version 0.2.0 is a rebuild around one hardware-agnostic file format. Code written for 0.1 needs
changes. The 0.1 profile files still load.

### Added

- **Profile format 1.0.** A profile is one JSON file for one calibration of one device. The format
  covers superconducting, trapped-ion, neutral-atom, spin and photonic qubits. Each error metric has
  its own key (`avg_infidelity`, `process_infidelity`, `depolarizing_param`, `pauli`), so no number
  is guessed. Per-qubit and per-gate records override device-wide defaults. Disabled gates and
  qubits, directed connectivity, provenance and license fields are part of the format. See
  [docs/profile-format.md](docs/profile-format.md).
- **Fingerprints.** Every profile has a SHA-256 fingerprint of its physics.
  `nv.load(ref, expect="nv:...")` fails if the profile changed. `nv cite` prints a citation
  with the full fingerprint, the NoiseVault version and the ref that loads the cited
  calibration. On a mismatch, the error names the calibration the ref loaded. If you have the
  pinned calibration, the hint is the `nv.load` call, with the same pin, that loads it. A dated
  ref with no calibration on that UTC day fails with "no ibm_fez profile calibrated on
  2025-03-01 UTC". For a device that `nv pull` serves, the hint is
  `nv pull ibm_fez --at 2025-03-01T23:59:59Z`. The hint also says to load the ref that
  `nv pull` prints.
- **25 bundled profiles that load offline.** The bundled profiles cover 18 IBM devices from
  qiskit-ibm-runtime and 5 Quantinuum devices from Quantinuum's published benchmark data. They
  also cover Google's Rainbow and Weber from cirq-google. All 25 profiles are Apache-2.0 data,
  and the files are byte-identical on Python 3.11 to 3.14.
- **Four exports.** Each export returns the framework's own object with a `.report`. The report
  states what the export reproduces exactly, approximates or omits. The four exports are:
  - `to_qiskit()` returns an `AerSimulator` with a device `Target`, so `transpile` compiles to
    the device's natives and avoids disabled gates.
  - `to_cirq()` returns a `cirq.NoiseModel`. `GridQubit(r, c)` addresses the qubit at those
    coordinates.
  - `to_pennylane()` returns a `qml.NoiseModel` for `qml.add_noise`, with gradients through the
    noise. A broadcast whose angles need different natives raises an error that names the fix,
    `qml.transforms.broadcast_expand`. Operator arithmetic such as `qml.prod` gets the noise of
    the gates it decomposes into.
  - `to_stim(circuit)` returns a noisy copy of a Stim circuit for QEC-size sampling and
    decoding. `CXSWAP`, `SWAPCX` and `CZSWAP` are the registry gates `cxswap`, `swapcx` and
    `czswap`. Each takes the profile's calibration of that gate. If the profile has no
    calibration of the gate, the export raises `MissingCalibrationError` and asks you to
    decompose the gate.
- **One rule for fixed-angle gates in every export.** A gate such as `s`, `sx` or `ms` takes
  the noise of its own native if the profile has that native. Otherwise, the gate takes the
  noise of the rotation that it equals.
- **No typical noise for gates that need a decomposition.** A gate that needs several native
  entanglers, such as `swap` or `ccx`, never takes the typical native gate's noise. A gate on
  more than two qubits also never takes the typical native gate's noise. If the profile does
  not calibrate such a gate, the export raises `MissingCalibrationError` with either
  `unknown_gates` value and asks you to decompose the gate.
- **Readable reports.** Approximation warnings point at your own line of code.
  `report.summary()` states the usage counts as sentences on one line that starts with `used:`,
  such as "cx took the typical native gate's noise 40 times". It names each `includes` item in
  plain words, such as "single-qubit gate error".
- **Live pulls without an account.** `nv pull` reads IBM's public calibration endpoint, with
  history through `--at`, and IonQ's published characterizations. Pulls through an IBM account
  also work. `nv pull` saves pulled profiles to `~/.noisevault/profiles`.
- **Importers.** Five importers read vendor data. The five importers are `from_qiskit_backend`,
  `from_ibm_csv`, `from_braket`, `from_cirq_google`, and the importer of Quantinuum's dated
  datasets. `from_braket` maps each native to the gate with the same matrix.
  `from_qiskit_backend` allows a gate only on the qubits that its `Target` lists, and takes the
  technology from the backend. `from_qiskit_backend` also labels a fake backend whose data is a
  model, such as `FakeNighthawk`, as `vendor_model`. An IBM gate error missing from a CSV or a
  pull takes the device median, and `provenance.notes` names those qubits and pairs.
  `from_ibm_csv` reads IBM's CSV formats from 2023 to 2026 and treats a cell that says
  `undefined` as blank. `from_ibm_csv` refuses a file in an older format and a file with two
  columns for the same value. It also refuses a copy in which a spreadsheet turned
  `partner:value` cells into times. Each refusal is a one-line error that says why. Importers
  raise `SourceDataError` for calibration data they cannot read. `SourceDataError` is both a
  `NoiseVaultError` and a `ValueError`. Its `hint` holds the next step, if there is one.
  `from_qiskit_backend` also raises `SourceDataError` for a backend with no fixed qubit count.
- **Hypothetical devices.** `Profile.uniform` makes a profile for a hypothetical device.
  `profile.suggest_layout(n)` picks a well-calibrated chain of qubits that each have every
  one-qubit native the device has.
- **Command line.** The command line is `nv`, also named `noisevault`. Its commands are `list`,
  `show`, `pull`, `diff`, `check`, `cite`, `validate`, `doctor` and `schema`.
  - A bare `nv` prints the help, which ends with three commands to start with. A usage
    mistake prints one line with the closest match.
  - A failure prints what went wrong on an `error:` line, and the next step, if there is one,
    on a `hint:` line. In Python, a `NoiseVaultError` keeps that step in `hint`. `str(error)`
    ends with that step.
  - `nv show` and `nv diff` label gate errors as average gate infidelity.
  - When you have several calibrations of a device, `nv show` and `nv check` say which one a
    bare id loaded. `nv check` names the calibration it checked on its first line.
  - `nv check` lists missing frameworks with one install command. With no framework installed,
    it prints one error and one install command. When an export refuses the profile,
    `nv check` lists that framework as skipped with the reason and still checks the others.
  - `nv check` counts a circuit that ran with gates removed as reduced. It also samples a
    measurement-only circuit through each framework's own readout. Each row shows the TVD and
    the tolerance of the circuit with the highest ratio of TVD to tolerance. `--json` reports
    that circuit's name, TVD and tolerance under `worst`.
  - `nv list` keeps one device's calibrations together, newest first. It shows the time when
    two calibrations share a date. It never cuts an id or a date.
  - In a narrow terminal, `nv list` leaves out the source column, then the processor column,
    then the license column. Without the license column, the line that counts the profiles
    names the most common license, then each other license with its profiles. When every
    profile has the same license, that line names it and the table has no license column.
  - `nv validate` says what an unknown or a missing key means.
  - `nv` prints commands without backticks, so you can paste them into a shell as they are. No
    output line ends in padding spaces.
  - `nv diff` marks values as new or gone and compares both orders of a symmetric pair.
  - NoiseVault gives a warning and skips a damaged or unreadable file in the vault. The other
    profiles still list and load.
- Documentation in [docs/](docs/) and runnable [examples](examples/).

### Changed

- To install from GitHub with the extra for your framework, run
  `pip install "noisevault[qiskit] @ git+https://github.com/dvgyl/noisevault"`.
  To try the command line without installing, run
  `uvx --from git+https://github.com/dvgyl/noisevault nv show ibm_fez`.
- The core install needs only numpy, pydantic, typer and rich. Qiskit, Cirq, PennyLane and
  Stim are extras, and importing `noisevault` imports none of them.
- Minimum versions are pennylane 0.43.3, stim 1.15, typer 0.27 and rich 13.8. CI installs
  every direct dependency at its declared minimum and runs the tests.
- NoiseVault supports Python 3.11 to 3.14.
- NoiseVault upgrades format 0.1 files in memory and gives a `MigrationWarning`. Save them to
  keep the 1.0 form.

### Removed

- The 0.1 Python API, command line, `snapshots/` folder and experiment scripts.

## 0.1.0 (2026-07-14)

Version 0.1.0 is the first release.

- Schema 0.1 for dated JSON files of IBM device calibrations, with provenance and raw-payload
  hashes.
- Importers for IBM Quantum accounts, Qiskit fake backends and IBM calibration CSV files.
- Exports to Qiskit Aer, Cirq and PennyLane noise models, compared on small benchmark circuits
  with exact density-matrix simulation.
