# Which qubits should my circuit use?

The qubits of a device differ in quality. This page shows three tools that choose qubits from a
profile. They give a chain of good qubits, a list of qubits to avoid, and a choice from several
days of calibrations.

## A chain of good qubits

`suggest_layout(n)` returns a connected chain of `n` qubits with a low sum of errors:

```python
import noisevault as nv

kingston = nv.load("ibm_kingston@2026-04-15")
print(kingston.suggest_layout(4))
# {0: 148, 1: 149, 2: 150, 3: 151}
```

For each qubit, the sum adds the error of its typical 1-qubit gate and its mean readout error.
For each consecutive pair, it adds the error of its typical 2-qubit gate. Two qubits are
consecutive only when the pair has a usable 2-qubit gate. [Choose qubits](frameworks.md#choose-qubits)
gives the other rules. Give the chain to your framework as the layout, for example
`initial_layout=list(kingston.suggest_layout(4).values())` in Qiskit.

## Qubits to avoid

A calibration can state values that show a failure, not noise. `unusable_qubits()` names each
qubit that a circuit should not use, with the reason:

```python
import noisevault as nv

kingston = nv.load("ibm_kingston@2026-04-15")
for qubit, reason in kingston.unusable_qubits().items():
    print(qubit, reason)
# 120 each of its 2-qubit gates is disabled or at or past full depolarization
# 132 its readout is no better than chance for a prepared 1 (P(0|1) = 0.719)
```

A qubit is unusable for one of three reasons:

- The profile disables the qubit.
- Its stated P(1|0) or P(0|1) is 0.5 or more. The qubit then reads a prepared 0, or a
  prepared 1, wrong in at least half of the shots.
- It has 2-qubit gates, and the profile disables each one or gives it an error at or past full
  depolarization. A gate at full depolarization gives a random state.

`suggest_layout` never puts these qubits in the chain. It also does not join two qubits with a
2-qubit gate at or past full depolarization.

## A choice from several days

Give `suggest_layout` the earlier calibrations of the same device as `history`. The search then
ranks qubits by the median of each value over the history and the newest profile. The
connectivity, the disabled marks and the unusable qubits still come from the newest profile.

<!-- not-run: needs the 106 MB archive file that hf download writes -->
```python
from datetime import UTC, datetime, timedelta

import noisevault as nv

archive = "train.parquet"
today = datetime(2026, 9, 1, 12, tzinfo=UTC)
newest = nv.from_calibration_archive(archive, "ibm_kingston", at=today)
week = [
    nv.from_calibration_archive(archive, "ibm_kingston", at=today - timedelta(days=d))
    for d in range(7, 0, -1)
]
print(newest.suggest_layout(4, history=week))
```

[Data sources](data-sources.md) tells how to get the archive. `nv.typical(profiles)` gives the
profile of medians itself. The last profile gives the structure, so give the profiles oldest
first.

## What a stated error is

A stated error is one measurement at one time. A readout error, for example, is a count of
wrong results in a few thousand shots, so the next calibration gives another count. The value
of a qubit also changes between calibrations. Thus the lowest values of a day are low partly by
chance. In the public archive of IBM calibrations, take the lowest tenth of the stated P(1|0)
values of a calibration. The next measured values of the same qubits have a median 2.4 to 3.7
times higher. A chain that takes the lowest values of one calibration thus expects fewer errors
than it gets.

A history does not remove this effect, but it makes the effect smaller. A test on `ibm_kingston`
used 29 dates from June to October 2026 and a chain of 4 qubits. On each date, one chain came
from the calibration of the day, and one chain came from a history of the 7 days before. The
calibration of the next day then gave the errors of each chain:

| | Chain from the day | Chain from the history |
|---|---|---|
| Next-day readout error / stated readout error (median) | 1.14 | 1.03 |
| Next-day 2-qubit gate error / stated error (median) | 1.03 | 1.02 |
| Next-day sum of errors of the chain (median) | 0.0273 | 0.0280 |

The stated errors of the chain from the history are thus closer to the next day. But that
chain did not have fewer errors. Its next-day sum was lower on 10 dates, higher on 13 dates,
and the same on 6 dates, where both choices gave the same chain. Use a history when you want
the stated errors of the chain to be a fair forecast. Do not expect a better chain from it.
