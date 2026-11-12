# Counts format 1.0

A counts file records what a device measured when it ran a set of circuits. The counts file binds
the run to the profile that the circuits were planned from. The file also binds the run to the
physical qubits that the circuits ran on and to the exact ops that the device executed.
`nv compare` scores a profile on a counts file.

NoiseVault reads a counts file with `load_counts(path)` from `noisevault.counts`. A file may be
gzip-compressed (`.json.gz`). A file can hold at most 64 MiB of JSON, after decompression.
`plan(profile)` gives the circuits to run, and
`simulate(profile, circuits, shots=..., seed=...)` draws counts for them from a profile.

## An example

The example shows a hypothetical run of two circuits on three qubits of ibm_kingston. The example
uses invented counts to show the format.

```json
{
  "nv_counts": "1.0",
  "source": "hardware",
  "profile": {"id": "ibm_kingston",
              "fingerprint": "609c845ed93458cb1f3999275b0925f6c2731a5fadb581c6d2adc6f2c81f13b4"},
  "backend": "ibm_kingston",
  "run_at": "2026-04-16T09:30:02Z",
  "bit_order": "qiskit",
  "execution": {
    "client": "qiskit-ibm-runtime 0.49.0 SamplerV2, qiskit 2.5.2",
    "transpiled": false, "gate_twirling": false, "measure_twirling": false,
    "dynamical_decoupling": false, "init_qubits": true,
    "job_ids": ["example-job-1"],
    "options": {"twirling": {"enable_gates": false, "enable_measure": false},
                "dynamical_decoupling": {"enable": false},
                "execution": {"init_qubits": true, "meas_type": "classified"}}
  },
  "circuits": [
    {"name": "ghz_chain", "qubits": [148, 149, 150],
     "ops": [["sx", [0], []], ["sx", [1], []], ["sx", [2], []], ["cz", [0, 1], []],
             ["delay", [2], [68.0]], ["cz", [1, 2], []], ["sx", [0], []], ["sx", [1], []],
             ["sx", [2], []], ["delay", [0], [68.0]]],
     "shots": 4000,
     "counts": {"000": 978, "100": 19, "010": 1028, "110": 15, "001": 11, "101": 943,
                "011": 8, "111": 998}},
    {"name": "readout", "qubits": [148, 149, 150], "ops": [], "shots": 4000,
     "counts": {"000": 3923, "100": 34, "010": 39, "001": 4}}
  ]
}
```

Circuit qubit i of a circuit is `qubits[i]`, so `ghz_chain` runs on qubits 148, 149 and 150. Ops
name circuit qubits. The `cz` gate takes 68 ns on these qubits, so qubit 150 waits 68 ns for the
first `cz`. Qubit 148 idles 68 ns while the second `cz` runs. Each wait is a `delay` op. After the
last op, the device measures each circuit qubit i into classical bit i. With
`"bit_order": "qiskit"`, the rightmost character of a counts key is classical bit 0, so `"100"`
means that qubit 150 read 1.

## Top-level fields

`load_counts` requires every field.

| Field | Content |
| --- | --- |
| `nv_counts` | The format version, `"1.0"`. |
| `source` | `hardware`, or `simulated` for counts drawn from a profile. |
| `profile` | The `id` and the calibration `fingerprint` of the profile the circuits were planned from. |
| `backend` | The device that ran the circuits, as the profile's `device.name` spells it. |
| `run_at` | When the device started the first circuit, an ISO 8601 time with a timezone. Stored in UTC. |
| `bit_order` | How the counts keys order the classical bits, `qiskit` or `clbit0_left`. |
| `execution` | How the device ran the circuits. |
| `circuits` | The circuits, at least one, each with its ops and counts. |

Unknown keys are errors everywhere except inside `execution.options`. Values are strict. A count
of `"4000"` or `4000.0`, or a flag of `1`, is an error rather than a coercion.

`profile.fingerprint` is the fingerprint of the planning profile without `unmodeled_error`, as 64
hex digits. In Python, the value is `profile.calibration_fingerprint`, the fingerprint of
`profile.uncorrected()`. `nv compare` scores the counts only against a profile with this
calibration fingerprint and this `backend`.

## Execution

Format 1.0 accepts only a run of exactly the listed ops, where each shot starts from a fresh
ground state. Each flag must have the value that starts its row, and `load_counts` refuses a file
with the other value.

| Field | Content |
| --- | --- |
| `client` | The software that ran the circuits, such as `"qiskit-ibm-runtime 0.49.0 SamplerV2, qiskit 2.5.2"`. |
| `transpiled` | `false`. A transpiled run may have executed other ops than the listed ones. |
| `gate_twirling` | `false`. Gate twirling adds random Pauli gates that the ops do not list. |
| `measure_twirling` | `false`. Measurement twirling flips qubits at random before the measurement. The random flips change the readout error. |
| `dynamical_decoupling` | `false`. Dynamical decoupling adds pulses on idle qubits that the ops do not list. |
| `init_qubits` | `true`. Without `init_qubits`, a shot may start where the previous shot ended, not from 0. |
| `job_ids` | The ids of the jobs that ran the circuits. Optional, default `[]`. |
| `options` | The run's submission options, as a JSON object nested at most 64 levels deep. `options` itself is level 1. Optional, default `{}`. |

When `options` holds one of these Qiskit Runtime `SamplerV2` options, the option must have the
value shown, or `load_counts` refuses the file:

| Option | Required value |
| --- | --- |
| `twirling.enable_gates` | `false` |
| `twirling.enable_measure` | `false` |
| `dynamical_decoupling.enable` | `false` |
| `execution.init_qubits` | `true` |
| `execution.meas_type` | `"classified"` |

The required values are the `SamplerV2` defaults in qiskit-ibm-runtime 0.40 and 0.49, so a run
with the default options passes. `SAMPLER_V2_OPTIONS` in `noisevault.counts` holds this table as a
mapping from each option's path to its required value.

A job id links to the IBM Quantum account that ran the job. Therefore, share a counts file only
where you would share its job ids. NoiseVault reads counts files from your disk and never uploads
them. A profile fitted with `nv compare` stores the SHA-256 of the counts file, not the counts or
the job ids.

## Circuits

| Field | Content |
| --- | --- |
| `name` | The circuit's name. Names are unique within a file. |
| `qubits` | The physical qubits, 1 to 256 and distinct. Circuit qubit i is `qubits[i]`. |
| `ops` | The ops the device ran, in order, on circuit qubits. Can be empty. |
| `shots` | The number of shots, at least 1 and at most 10^10 (10000000000). |
| `counts` | The number of shots that gave each outcome. The counts add up to `shots`. |

The device measures every circuit qubit once, after the last op, into the classical bit with the
same index. `simulate` also refuses `shots` above 10^10.

A file has two qubit limits:

| Limit | Applies to |
| --- | --- |
| 256 qubits | Each circuit in the file. A file with a wider circuit does not load. |
| 10 qubits | Each circuit that `nv compare`, `nv score`, `simulate` or `circuit.vector()` reads. These use the exact reference simulator, which holds 2^n outcomes. |

A circuit on 11 to 256 qubits loads, saves, and goes through `select` and `split`. Its counts stay
a dict of bit strings. `nv compare`, `nv score` and `simulate` refuse such a circuit and name it.
Select the circuits of 10 qubits or fewer first. For a wide Clifford circuit, use the stabilizer
forecast of `noisevault.wide` instead. The circuits of 10 qubits or fewer in one file hold at most
4096 outcomes in all, where a circuit on n qubits has 2^n.

### Ops

An op is `[name, [circuit qubits], [parameters]]`, as `nv check --json` writes it, such as
`["rz", [0], [1.5708]]`. The name is `delay`, or a gate in the
[gate registry](profile-format.md#gates) that has a unitary. A gate acts on as many distinct
qubits as the registry says. Its parameters are the registry's, as finite numbers in radians.

`["delay", [q], [ns]]` idles circuit qubit q for ns nanoseconds, 0 or more. A delay idles one
qubit. `measure` and `reset` are not ops.

### Counts keys

A key has one character, `0` or `1`, per circuit qubit. `bit_order` says which end holds
classical bit 0:

| `bit_order` | Classical bit 0 | Key for "circuit qubit 0 read 1, qubit 1 read 0" |
| --- | --- | --- |
| `qiskit` | The rightmost character, as Qiskit prints counts. | `"01"` |
| `clbit0_left` | The leftmost character. | `"10"` |

`load_counts` stores keys as `clbit0_left`, sorted, with zero counts dropped. `circuit.vector()`
gives the counts as an array in the order of the reference simulator's probabilities, with
circuit qubit 0 as the most significant bit. A circuit on more than 10 qubits has no such array.

## Planned circuits

`plan(profile)` returns the circuits `nv check` runs, on the qubits `nv check` picks.
`plan(profile, layout)` uses the qubits of `layout` instead, as `profile.check(layout=...)` does.
`plan` schedules each gate as soon as its qubits are free, using the gate durations the profile
states, and writes every wait as a `delay`. It also pads each qubit with a delay to the end of the
circuit. No qubit then has a gap that the ops do not fill, so any scheduling policy on the device
gives the same timeline. A gate with no stated duration, such as a virtual `rz`, takes 0 ns. A
profile with `unmodeled_error` plans the same circuits as its calibration.

```python
import noisevault as nv
from noisevault.counts import load_counts, plan, simulate

kingston = nv.load("ibm_kingston@2026-04-15")
circuits = plan(kingston)
counts = simulate(kingston, circuits, shots=4000, seed=1)
counts.save("kingston.counts.json")

print([c.name for c in circuits])
print(load_counts("kingston.counts.json") == counts)
# ['ghz_chain', 'mirror', 'single_qubit', 'readout']
# True
```

`simulate` draws the counts from the reference simulator, with the profile's unmodeled-error
factors applied. It binds the counts to the profile's calibration fingerprint and sets `source` to
`simulated`. `run_at` defaults to the current UTC time. The same `seed` and `run_at` give the same
file. The reference simulator does not model effects. It leaves out an effect with `allow` set to
`omit`, and it refuses a profile with an effect that sets any other `allow` value.

## Counts from an IBM device

`scripts/run_on_ibm.py` runs the planned circuits on an IBM device through your IBM Quantum
account. It writes a counts file bound to the calibration in effect when the job ran. The script
also does the following:

- It sets `experimental.execution.scheduler_timing` to `true` in `execution.options`. This option
  asks IBM to return how it scheduled each circuit.
- When IBM returns a schedule, the script writes the schedule to `<stem>.timing.json` beside the
  counts file. The timing file has an entry for each circuit that IBM returned a schedule for.
- If your catalog does not have the calibration that the counts bind to, the script saves that
  calibration to `<stem>.profile.json` beside the counts file. The `nv compare` command that the
  script prints then names this file.
- It saves the submitted job to `<stem>.job.json` beside the counts file. The script deletes the
  job file after it saves the counts. If the wait for the job stops, run the same command
  again.
  The script then collects that job instead of submitting another.

`<stem>` is the counts file name without `.counts.json`. For `kingston-0416.counts.json`, the
files are `kingston-0416.timing.json`, `kingston-0416.profile.json` and `kingston-0416.job.json`.

### The script never replaces a file

The script never replaces a file that it did not write. This rule applies to the counts file and
to the `.timing.json`, `.profile.json` and `.job.json` files beside it.

- Before the script opens your account, it checks the counts, timing and profile files. If one of
  them exists, the script stops. Give `-o` a new file name.
- Before the script submits the job, it creates the job file and writes the planned circuits
  into it. If the job file exists at that time, the script stops and submits nothing. A second
  run of the same command can create the job file while the first run waits for your answer.
- After IBM accepts the job, the script adds the job id to the job file. Before and after it adds
  the job id, the script checks that the job file is still the file that it created. The file
  must also still hold the bytes that the script wrote. If another program replaced, moved or
  deleted the job file, or changed it in place, the script does not change that file. The script
  saves the job to a new job file and stops. See
  [If the script cannot save the job id](#if-the-script-cannot-save-the-job-id).
- After the job runs, the script writes the counts, timing and profile files together. If
  another program created or replaced one of them before the script finished, the script keeps
  none of its own files. It keeps the other program's file and the job file.
- The script deletes a job file only if that file is still the job file that it read or created.
  The file must also still hold the bytes that the script read or wrote. If another run saved a
  new job file at the same path or changed the job file in place, the script keeps that file.
- If the script cannot delete the job file after it saves the counts, it prints a warning. Then
  it prints the saved files and the `nv compare` command. You can delete the job file yourself.

To collect a job to a new counts file, give `--collect` the job file. Then give `-o` a new file
name. For example, run the same command with
`--collect kingston-0416.job.json -o kingston-0416b.counts.json`. The script collects that job and
does not submit another. You can also move the other program's file to a different path. Then run
the same command again.

The script checks for the job file before it opens your account. If the job file exists at that
time, the script collects the job that the file records. The script also collects a job when you
give `--collect` or `--job-id`. When the script collects a job, it never submits a job. If the
script cannot read the job file, or the job file is gone when the script reads it, the script
stops with an error. If a value in the job file is not valid, the script stops before it opens the
job. If the script can read a job id in the damaged file, the hint names that job. The script
cannot collect that job, so find it in your IBM Quantum account.

The job file records the device of the job. If `DEVICE` names another device, the script stops
before it opens the job. If IBM reports that the job ran on another device, the script stops and
writes no counts file. In both cases, the error names the two devices and the script keeps the
job file.

### If the script cannot save the job id

IBM gives the job id after the script submits the job. If the script cannot add the job id to the
job file, the script stops. The error names the job id, and the hint gives the command that
collects the job:

```text
error: could not save job d1h9q8k5x4w0008r7t2g to kingston-0416.job.json ([Errno 28] No space left on device)
hint: run the same command with --job-id d1h9q8k5x4w0008r7t2g to collect the job
```

The job file keeps the planned circuits. Make space on the disk, then run the command in the
hint. The script collects that job and does not submit another.

In two cases, the script saves the planned circuits and the job id to a new job file,
`<stem>.<job id>.job.json`:

- Another program replaced, moved or deleted the job file, or changed it in place, while the
  script submitted the job or added the job id.
- A failed write left a part of the job id in the job file, and the script could not remove that
  part.

The script creates this file only if no file has that name. The hint gives the command that
collects the job:

```text
error: kingston-0416.job.json changed while the script submitted job d1h9q8k5x4w0008r7t2g. The script saved job d1h9q8k5x4w0008r7t2g to kingston-0416.d1h9q8k5x4w0008r7t2g.job.json
hint: run the same command with --collect=kingston-0416.d1h9q8k5x4w0008r7t2g.job.json to collect the job
```

If the script cannot save that file either, the error names the job id. Find the job in your IBM
Quantum account.

### Errors from IBM

If a request to IBM fails while the script opens the device, the script stops before it submits
a job. The error names the device and the reason:

```text
error: could not open ibm_kingston through your IBM Quantum account ('network is unreachable')
hint: run the same command again. The script did not submit a job
```

## Counts from an Amazon Braket device

`scripts/run_on_braket.py` runs the planned circuits on an Amazon Braket device through your AWS
account. It needs the `braket` extra, which installs `amazon-braket-sdk`. The script reads the
device properties, imports them with `nv.from_braket` and plans from the result.

- **Native gates only.** Each circuit runs in one verbatim box on its physical qubits, so Braket
  does not compile or move the gates. Each planned gate must be one native gate of the device
  with the same matrix, up to a global phase. For example, `r` runs as `prx` on IQM and AQT, and
  as `gpi2` or `gpi` on IonQ. The script refuses a plan with another gate. The readout circuit has
  no gate, so it has an empty verbatim box. The box keeps its qubits physical.
- **No delays.** Braket has no gate-level `delay` op. Thus the script removes the gate durations
  from the calibration before it plans, and the plan holds no `delay`. The device still idles
  some qubits while other qubits run gates. The noise model has no idle time, so `nv compare`
  puts that idle error into the gate factor. Braket gives gate durations only in standardized v3
  properties, so the change affects AQT and IonQ. A provenance note in the profile records it.
- **The cost first.** The script estimates the cost from the price per task and the price per
  shot, and shows it in US dollars. Without `--confirm-cost`, it submits nothing. With
  `--confirm-cost USD`, it submits only if the estimate is at most USD. The script knows the
  prices of `iqm_garnet`, `rigetti_cepheus-1-108q`, `aqt_ibex-q1` and `ionq_forte-1` from the
  [Amazon Braket pricing page](https://aws.amazon.com/braket/pricing/). For another device, give
  `--task-price` and `--shot-price`.
- **One task or one task per circuit.** When the device properties list the program set action,
  all circuits go in one task. Otherwise, each circuit is one task, so the run pays the task price
  for each circuit.
- **`--suite`.** The script runs the
  [deep circuit set](recipes.md#run-the-deep-circuit-set) of `noisevault.suite.suite` in place
  of the `nv compare` circuits, with `--shots` for each circuit. The circuit table then also
  gives the role of each circuit.
- **`--budget USD`.** The script runs the circuits and the shots that
  `noisevault.design.design` chooses from the deep circuit set for at most USD. `--budget`
  implies `--suite`. Do not give `--shots` with `--budget`, because the design chooses the shots.
  The design must know the price per task. Thus the script makes two designs. The design for
  one program set task spends at most the cost of `maximumTotalShots` shots, so that it fits in
  the program set limits of the device. The design for one task per circuit pays the task price
  for each circuit. The script takes the design with the smaller predicted error of the gate
  factor. The cost that the script shows is the cost of the tasks that it submits, rounded up to
  the cent, and the cost is never more than USD. If USD does not pay 100 shots of the smallest
  design, the script stops and submits nothing.
- **Saved tasks.** Before the first task, the script creates `<stem>.job.json`, with the plan.
  It adds each task id to the job file before it creates the next task. The script follows the
  rules of [the IBM script](#the-script-never-replaces-a-file): it never replaces a file, and it
  deletes the job file only after it saves the counts. If the wait stops, run the same command
  again. The script then collects the tasks that the job file records and submits nothing. If
  Braket accepts some tasks and refuses the next, the counts file holds the circuits of the
  accepted tasks.
- **No second charge.** If Braket does not answer a request, for example after a timeout, the
  task can exist all the same. The script then keeps the job file and stops. Look for the task in
  the Amazon Braket console before you delete the job file. The script deletes the job file only
  when Braket refuses the first task with an error. When the request without an answer follows
  other tasks, the job file records that. The collector then saves the counts of the recorded
  tasks and prints a warning: look in the console for one more task.
- **The end of the submission.** After the last task, the script adds an end line to the job
  file. A collector refuses a job file that has fewer tasks than circuits and no end line,
  because another run can still add tasks to it. If no run adds tasks, for example after a
  power cut, give `--partial` to collect the recorded tasks.
- **The calibration file.** Braket profiles are not in the catalog, so the script always saves
  the calibration to `<stem>.profile.json`. The printed `nv compare` command names that file.
  Braket gives no calibration history. Thus the counts bind to the calibration that the script
  read before it submitted. When the device has a newer calibration from before the run, the
  script prints a warning.
- **`run_at`.** Braket does not say when a task starts to run. `run_at` is the earliest time at
  which a task ended.
- **`--local PROPERTIES`.** The free Braket `LocalEmulator` runs the circuits from a saved
  properties file, with no AWS account. Its noise model comes from the same file. The counts have
  `source` set to `simulated`. A local run cannot be collected later.

This run of the test file `tests/fixtures/braket/iqm_garnet_run.json` on the local emulator
shows a design for 35 US dollars. The file describes no real device, so the numbers show only
how the script works:

```text
$ python scripts/run_on_braket.py iqm_garnet --local tests/fixtures/braket/iqm_garnet_run.json --budget 35 -o garnet.counts.json
iqm_garnet@2026-10-20 nv:e2977716a282 on qubits 1-2-3-4

circuit       qubits   gates  role
readout       1-2-3-4      0  train
readout_1     1-2-3-4      8  train
deep_d4_s0    1-2-3-4     80  train
deep_d8_s0    1-2-3-4    156  train
deep_d8_s1    1-2-3-4    154  train
deep_d4_s2    1-2-3-4     80  test_near
pair0_d12_s0  1-2        122  train
pair1_d12_s0  2-3        120  train
pair2_d12_s0  3-4        120  train

budget          $35
shots           2659 per circuit, 9 circuits in 1 program set task
cost            none on the local emulator, about $35.00 ($0.30 a task, $0.00145 a shot) on iqm_garnet
gate factor     predicted standard error 1.5%
readout factor  predicted standard error 4.2%
counts file     garnet.counts.json
```

The exact cost is $34.99995. On the device, the script submits with `--confirm-cost 35.00`.

The counts file has `bit_order` set to `clbit0_left`. `execution.job_ids` holds the task ids. A
task ARN contains the id of your AWS account, so share a counts file only where you would share
that id. `execution.options` records the device ARN, the shots per circuit, `verbatim` and
`program_set`.

When the script submits the first task, the Braket SDK saves the results to the default Braket
bucket in your account, `amazon-braket-<region>-<account id>`. The SDK creates that S3 bucket if
it does not exist.

## SHA-256 and canonical form

`counts.sha256` is `sha256:` plus the SHA-256 of the canonical JSON (sorted keys, no whitespace)
of the loaded file. The loaded file has its keys in one order, with zero counts dropped. The
same run therefore has one SHA-256, whether its file says `qiskit` or `clbit0_left`.
`counts.save(path)` writes readable JSON of the loaded form, so the same run always gives the
same bytes. A `.gz` suffix writes gzip with a zero timestamp, as `profile.save(path)` does.
`scripts/run_on_ibm.py` writes its counts file in the same way, so `-o run.counts.json.gz` gives
a gzip file.

## Errors

`load_counts` raises `nv.CountsError`, a `ValueError`, for a file it refuses. The message is one
line that starts with the path and names the field, such as
`run.counts.json: circuits[0]: counts sum to 3999, but shots is 4000`. `hint` says how to fix the
problem. A key that occurs two times in one JSON object is an error, such as two `"0"` keys in
`counts`. `load_counts` raises `OSError` for a file that it cannot read.
