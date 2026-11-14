# Score a noise model on wide circuits and on detection events

`nv score` computes the exact probability of each outcome with the reference simulator. The
reference simulator holds 10 qubits at most. Two kinds of data need more qubits:

- **Wide circuits.** The circuit set of `noisevault.suite` has Clifford circuits on 20, 40 and
  80 qubits. A Clifford circuit with Pauli noise runs in the stabilizer simulator Stim on
  hundreds of qubits.
- **Detection events.** An error-correction experiment records which detectors fire in each
  shot. A decoder reads a decoder model, the detector error model (DEM) that Stim makes from a
  noisy circuit. The noise of the DEM comes from a device calibration. The detection events of
  the experiment show if that noise is correct.

`noisevault.events` scores a profile on both kinds of data with one method. The method needs
Stim: install the `stim` extra. In both uses, an **event** is a bit with the value 1. In a wide
circuit, an event is a wrong bit: a qubit whose measured bit differs from the one ideal result.
In an experiment, an event is a detector that fired.

The measured events are an array of shots by bits. Stim samples the same bits from each
profile. For each profile, the report compares the samples with the measured events in four
ways:

- **Event rate.** The share of shots with an event on each bit, measured and forecast.
- **Bit score.** The score of a shot is the sum over the bits of the log probability of each
  bit, 0 or 1. The probability comes from the forecast rate of the bit. The report gives the
  mean over the shots.
- **Count score.** The log probability of the number of events in a shot, under the forecast
  distribution of that number. The report gives the mean over the shots.
- **No event.** The share of shots with no event.

A higher score is better. Both scores are proper scoring rules, as in [Scoring](scoring.md).

## Score the wide circuits of a run

`score_wide` takes the circuits of a counts file that have more than 10 qubits. This example
makes the counts offline. A second profile with two times the gate errors stands for the device:

```python
from datetime import UTC, datetime

import noisevault as nv
from noisevault.events import score_wide, simulate_wide
from noisevault.suite import suite

calibration = nv.Profile.uniform(
    "line_24",
    technology="superconducting",
    num_qubits=24,
    connectivity=[(q, q + 1) for q in range(23)],
    one_qubit_error=1e-3,
    two_qubit_error=1e-2,
    readout_error=2e-2,
    t1_us=200,
    t2_us=150,
    one_qubit_ns=32,
    two_qubit_ns=68,
)
device = calibration.model_copy(update={"unmodeled_error": {"gates": {"factor": 2.0}}})
circuits = [c for c in suite(calibration) if len(c.qubits) > 10]
run_at = datetime(2026, 11, 14, tzinfo=UTC)
counts = simulate_wide(device, circuits, shots=4000, seed=1, run_at=run_at)

report = score_wide({"calibration": calibration, "device": device}, counts, samples=200_000)
print(report.summary())
# wide circuits: clifford_n20_d5, 4000 shots, forecasts from 200000 samples
# ...
# calibration: too few events, ...
# ...
#   bit score: device is better by ...
```

On one machine, the example prints this text. The numbers depend on the Stim version and on the
machine:

```text
wide circuits: clifford_n20_d5, 4000 shots, forecasts from 200000 samples
measured: 5.042 events per shot, no event in 0.045 of the shots

model        events  ratio  no event  bit score  count score
device        4.989   0.99     0.046  -11.17801     -2.43872
calibration   3.109   0.62     0.171  -11.80315     -2.69735

device: the number of events agrees with the measured number, within chance
calibration: too few events, 0.62 times the measured number (z = -42.2)
  largest differences, forecast against measured: q3 0.190 against 0.311, q6
    0.175 against 0.290, q7 0.209 against 0.325

calibration against device:
  bit score: device is better by 0.62514 per shot than calibration
    (z = 22.78, p = 7.1e-115)
  count score: no verdict, because a forecast has fewer than 100 samples of a
    number of events that the measured shots show; use more samples
    (z = 22.94, p = 1.7e-116)
```

For a real run, load the counts with `noisevault.counts.load_counts` and give the profiles.
`models` can be one profile, a list of profiles or a dict of name to profile. `circuits=` names
the circuits to score. `samples` is the number of Stim samples of each profile for each circuit,
1,000,000 by default.

On the command line, `nv score` gives the same report with `--wide`, after its ranking:

```bash
nv score ibm_fez@2026-11-12 ibm_fez@2026-10-13 run.counts.json --wide
```

Without `--wide`, `nv score` leaves these circuits out and names them, because the samples take
about a minute for each profile. A counts file with only wide circuits gives no ranking, so it
needs `--wide`. Then `nv score` prints only this report, and the lists `models` and `paired` of
`--json` are empty. `--samples N` sets the number of samples, and `--json` gives the report
under the key `wide`.

## Score the detection events of an experiment

`score_detectors` takes three inputs:

1. The profiles.
2. The Stim circuit of the experiment, with its `DETECTOR` instructions and with no noise.
3. The measured detection events: an array of shots by detectors, or a file that Stim can read.

For each profile, `profile.to_stim` adds the noise of the profile to the circuit. Give `layout=`
to put the circuit qubits on the physical qubits of the experiment. Give `tick_ns=` for the time
of one `TICK` layer. This example simulates the device with a second profile:

```python
import stim

import noisevault as nv
from noisevault.events import score_detectors

calibration = nv.Profile.uniform(
    "line_9",
    technology="superconducting",
    num_qubits=9,
    connectivity=[(q, q + 1) for q in range(8)],
    one_qubit_error=1e-3,
    two_qubit_error=1e-2,
    readout_error=2e-2,
    t1_us=200,
    t2_us=150,
    one_qubit_ns=32,
    two_qubit_ns=68,
)
device = calibration.model_copy(update={"unmodeled_error": {"gates": {"factor": 2.0}}})
code = stim.Circuit.generated("repetition_code:memory", distance=5, rounds=5)
detection_events = device.to_stim(code).compile_detector_sampler(seed=1).sample(20_000)

report = score_detectors({"calibration": calibration, "device": device}, code, detection_events)
print(report.summary())
# detection events: 24 detectors, 20000 shots, forecasts from 1000000 samples
# ...
# calibration: too few events, ...
# ...
#   bit score: device is better by ...
```

On one machine, the example prints this text:

```text
detection events: 24 detectors, 20000 shots, forecasts from 1000000 samples
measured: 2.196 events per shot, no event in 0.276 of the shots

model        events  ratio  no event  bit score  count score
device        2.207   1.01     0.274   -7.28779     -1.69683
calibration   1.588   0.72     0.408   -7.40600     -1.76520

device: the number of events agrees with the measured number, within chance
calibration: too few events, 0.72 times the measured number (z = -45.1)
  largest differences, forecast against measured: D14 0.074 against 0.109, D10
    0.074 against 0.108, D5 0.074 against 0.107

calibration against device:
  bit score: device is better by 0.11821 per shot than calibration
    (z = 23.85, p = 1e-125)
  count score: no verdict, because a forecast has fewer than 100 samples of a
    number of events that the measured shots show; use more samples
    (z = 24.52, p = 8.4e-133)
```

The report has one bit for each detector, `D` and its index. `to_dict()` also gives the detector
coordinates of the circuit. For a file, the suffix names the Stim format: `.01`, `.b8`, `.r8`,
`.ptb64`, `.hits` or `.dets`. Give `data_format=` for another suffix. The file must hold the
detection events only, with no observable. A `b8` file of another circuit can have a size that
fits this circuit, so check that the file and the circuit belong together.

`score_detectors` refuses these inputs, with a message that names the problem:

- A circuit with no `DETECTOR` instruction.
- Data with another number of detectors than the circuit.
- A circuit that already has noise. Give `circuit.without_noise()`.
- A gate that a profile does not define, or a 2-qubit gate on a pair with no calibration. Give
  `unknown_gates="typical"` to use the noise of a typical native gate for an undefined gate.

## Read the report

The first line names the data, the number of shots and the number of samples of each forecast.
The `measured:` line gives the mean number of events in a shot and the share of shots with no
event.

The table has one row for each profile, best bit score first:

| Column | Meaning |
| --- | --- |
| `events` | The forecast mean number of events in a shot. |
| `ratio` | `events` divided by the measured mean number of events. |
| `no event` | The forecast share of shots with no event. |
| `bit score` | The mean bit score per shot. |
| `count score` | The mean count score per shot. |

One line for each profile answers the question "too few or too many events?". It compares the
forecast and the measured number of events, with the shot error and the sampling error of both.
The line says "agrees ... within chance" when the difference is not significant at the 5% level.
Otherwise it gives the direction, the ratio and the z value. The line gives no test when the
forecast expects fewer than 25 events in all shots, or fewer than 25 bits with no event. A
normal test is not valid for so few.

The next line names the bits whose rate differs most from the measured rate, at most three. A
bit is on the line only when its difference is significant at the 5% level divided by the number
of bits (a Bonferroni correction). Thus a line of many bits does not show bits by chance. The
test of a bit is Fisher's exact test. It compares the measured shots with an event and the
forecast samples with an event. Thus it is valid for a rare event and for few samples.

With two or more profiles, the report tests the best profile against each other profile. Each
test is a paired test of the score differences of the shots. Both tests add the sampling
variance of both forecasts to the shot variance, so they hold the error of the samples. The
bit score decides `better`. A test gives no verdict in two cases:

- A forecast has too few samples. For the bit score, a bit has fewer than 100 samples of 0 or
  of 1. For the count score, a number of events that the measured shots show has fewer than 100
  samples. Then the estimate of the sampling error is not valid.
- The sampling variance is larger than the shot variance. More shots cannot then make the test
  stronger. Give more samples.

A profile that forecasts far too few events has few samples with many events. Thus its count
score often gives no verdict, as in both examples. The bit score still decides.

`report.to_dict()` gives the same numbers for programs. Its keys are `kind`, `counts`,
`circuits`, `shots`, `level`, `measured`, `models`, `paired` and `bits`. Each entry of `bits`
gives the measured rate, and the forecast rate and its sampling error for each profile.

## Limits

- **The bit score treats the bits as independent.** It uses only the rate of each bit. Thus it
  does not see a correlation between bits, for example two detectors that one error fires
  together. Two profiles with the same rates and different correlations get the same bit score.
- **The count score sees a part of the correlations.** At the same rates, correlated events give
  a wider spread of the number of events in a shot. The count score sees that spread. It does not
  see which bits fire together.
- **The forecast comes from samples.** The sampling error does not become smaller with more
  measured shots. Use at least 20 times more samples than shots. Then each number of events
  that 5 shots show has 100 samples.
- **Stim holds Pauli noise only.** NoiseVault gives Stim the Pauli twirl of each channel. The twirl keeps the error rate of each
  gate and drops the bias of relaxation toward 0. For detection events, the readout error is the
  mean of P(1|0) and P(0|1).
- **The test needs independent shots and fixed profiles.** Make the profiles before you see
  the data, or fit them on other data.

## Samples from another simulator

`score_events(measured, sampled)` gives the same report for samples from any simulator.
`measured` is an array of shots by bits. `sampled` maps a name to the samples of that model: an
array of samples by bits, or an iterable of such arrays.
