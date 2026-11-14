# The calibration of a device over time, as one table

`nv history` collects the calibrations of one device over a range of dates. It writes their
values as one table, with one row for each value. You can then plot the drift of a qubit or
give the table to a model.

## One command

This command asks IBM's public endpoint for the calibration of `ibm_fez` at the start of each
day from 1 to 3 November 2026. It needs the network and no account:

```bash
nv history ibm_fez --from 2026-11-01 --to 2026-11-03 -o fez.csv
```

```text
ibm_fez from 2026-11-01T00:00:00Z to 2026-11-03T00:00:00Z, every 1 day (ibm)
3 calibrations, 9,055 values
  2026-10-31T23:49:19Z  nv:a75951ec747b
  2026-11-01T23:54:11Z  nv:2b6419ca592e
  2026-11-02T23:26:55Z  nv:dff94053d8ae
saved: fez.csv
```

The first rows of `fez.csv`:

```text
device,calibrated_at,fingerprint,kind,gate,qubit_a,qubit_b,value,unit
ibm_fez,2026-10-31T23:49:19Z,nv:a75951ec747b,t1,,0,,42.0786442417,us
ibm_fez,2026-10-31T23:49:19Z,nv:a75951ec747b,t1,,1,,215.401801498,us
ibm_fez,2026-10-31T23:49:19Z,nv:a75951ec747b,t1,,2,,147.143895332,us
```

Without `-o`, the command writes the table as CSV to standard output. With `--json`, it writes
one JSON value: the range, the list of calibrations, the columns, and the rows. Progress goes to
standard error. The command asks no question, and a failure exits with status 1.

`--step` sets the time between two steps: `1d` (the default), `12h` or `30m`. Each step gives
the calibration in effect at the time of the step. Thus the first calibration can be older
than `--from`, as in the example. Two steps that give the same calibration time give one
calibration in the table.

## The columns

| Column | Value |
| --- | --- |
| `device` | The profile id, such as `ibm_fez`. |
| `calibrated_at` | The calibration time, in UTC, in ISO 8601 form. |
| `fingerprint` | `nv:` and 12 hexadecimal digits. `nv.load(ref, expect=...)` accepts it. |
| `kind` | `t1`, `t2`, `p1_given_0`, `p0_given_1`, `gate_error` or `gate_duration`. |
| `gate` | The gate, such as `sx` or `cz`. Empty for a value of a qubit. |
| `qubit_a` | The qubit, or the first qubit of a gate. |
| `qubit_b` | The second qubit of a two-qubit gate. Empty for one qubit. |
| `value` | The number. |
| `unit` | `us` for T1 and T2, `ns` for a duration, empty for a probability or an error. |

`p1_given_0` is the probability to read 1 when the qubit is in 0. `p0_given_1` is the
probability to read 0 when the qubit is in 1. `gate_error` is the average gate infidelity, as in
the [profile format](profile-format.md).

The table holds the values of each qubit and of each gate on one or two qubits. A value that a
profile gives only for the whole device is not in the table.

The table has no column for the date of each value. A profile keeps only the oldest, median and
newest date of each kind of value, in `provenance.extra["value_times"]`. Thus NoiseVault cannot
give the date of one value.

## Sources

| `--source` | What it reads | Network |
| --- | --- | --- |
| `ibm` (the default) | IBM's public endpoint, one pull for each step. | Yes |
| `offline` | Each profile of the device that loads offline, with a calibration time in the range. These are the profiles that `nv list` shows. `--step` has no effect. | No |
| `archive` | A local copy of the [IBM calibration archive](data-sources.md#ibm-calibration-archive-on-hugging-face). Give the file with `--archive FILE`. One read of the file for each step. | No |

<!-- not-run: needs the 66 MB file that hf download writes -->
```python
import noisevault as nv

path = "data/train-00000-of-00001.parquet"
table = nv.history("ibm_fez", "2026-06-01", "2026-06-30", source="archive", archive=path)
print(table.summary())
```

## Load the table in pandas

The block below runs offline. It reads the bundled profile of `ibm_fez`, makes one column for
each qubit, and prints the T1 of qubit 0:

```python
import noisevault as nv
import pandas as pd

nv.history("ibm_fez", "2025-01-01", "2025-12-31", source="offline").to_csv("fez.csv")

frame = pd.read_csv("fez.csv", parse_dates=["calibrated_at"])
t1 = frame[frame.kind == "t1"].pivot(index="calibrated_at", columns="qubit_a", values="value")
print(t1.shape)
print(round(t1[0].iloc[0], 1))
# (1, 156)
# 48.8
```

`nv.history` gives the same result as the command. The result has `rows`, `to_csv(path)`,
`to_dict()` and `summary()`. `to_csv()` with no path returns the CSV text.

## Limits

- **One pull for each step.** Each step of the `ibm` source makes one pull, and one pull
  sends two requests to the endpoint. The command waits 0.5 seconds between two steps. A range
  has at most 400 steps. For a longer range, use a larger step, such as `--step 7d`.
- **The public endpoint has no data for a retired device.** On 2026-11-14, the endpoint did not
  list `ibm_torino`, at any date. The command then fails at the first step. Use
  `--source offline` for the profiles that you pulled before, or `--source archive`.
- **A value keeps the date of its own calibration.** IBM gives each value its own date. A value
  can be some days older than the calibration time of its row. Read
  [the ages of IBM values](limitations.md#limits-of-the-data) for the typical ages.
- **A failed step stops the command.** The error names the step and its date, and the command
  writes no file. NoiseVault writes no part of a table.
