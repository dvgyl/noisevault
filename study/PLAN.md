# Analysis plan of the realism study

This file fixes the analysis before the test data is scored. The commit date of each version of
this file is the record. A later change gets a dated entry in [Changes](#changes).

## Question

A calibration is a forecast of the results of a circuit. The study measures three things:

1. How well a noise model from the calibration of a device forecasts counts from that device.
2. How fast the forecast gets worse as the calibration gets older.
3. How much a small number of shots on the device improves the forecast.

## Data

A run is one job on one chain of qubits of one device. The job holds the circuits of
`noisevault.suite.suite` for the calibration that the device had when the job was planned. The
counts bind to the calibration in effect when the job ran.

- IBM devices: `ibm_kingston`, `ibm_fez` and `ibm_marrakesh`, one run each week on the best chain
  and on the median chain, with 2000 shots for each circuit.
- Other vendors: runs through Amazon Braket, with the circuits and shots that the budget allows.

`noisevault.suite.role` gives each circuit one role:

- `train`: a model can fit its parameters on these circuits.
- `test_near`: deep circuits of the same kind as the `train` circuits, from another random seed.
- `test_far`: circuits of another kind. These are the `nv compare` circuits, a periodic circuit
  and a circuit with 1-qubit gates on all qubits at the same time.

No model sees a test circuit before its score is computed.

The runs of the week of 2026-11-09 with the `nv compare` circuits only are exploratory. Their fits
were read before this plan was written. They showed that IBM calibrations can state a readout
error of exactly 0, and that two factors do not fit the counts.

## Models

Each model is a profile, so the reference simulator gives its forecast. C is the calibration that
the counts bind to.

| Name | Model |
| --- | --- |
| `ideal` | No noise |
| `generic` | One fixed model for every device: depolarizing error 0.001 for 1-qubit gates and 0.01 for 2-qubit gates, readout error 0.01, no relaxation |
| `medians` | Each value of C replaced by the median of that value over the device |
| `stale(d)` | The calibration of the same device that was in effect d days before the run, for d = 1, 7, 30, 90, 180 and 365 |
| `bundled` | The bundled profile of the device |
| `symmetric` | C with the two readout errors of each qubit replaced by their mean |
| `calibration` | C as stated |
| `two_factor` | C with one gate factor and one readout factor from `nv compare` on the `train` circuits |
| `readout_measured` | C with the readout errors of each qubit measured on `readout` and `readout_1`, and one gate factor fitted on the other `train` circuits |
| `locus` | `readout_measured` with one factor for each qubit fitted on its `qubit*` circuit, and one factor for each pair fitted on its `pair*` circuits, in place of the one gate factor |

Zero floor: a calibration can state a probability of exactly 0, and a device then gives the
outcome that the model calls impossible. Each model from a calibration is scored two times: as
stated, and with each readout probability of exactly 0 replaced by half of the smallest readout
probability above 0 in the same calibration. The floored form is the form in each comparison. The
number of runs that the stated form rules out is a result of its own.

A measured readout probability is (k + 1/2) / (n + 1) for k wrong results in n shots.

## Scores

- The primary score is the log score: the mean over shots of the natural log of the probability
  that the model gave to the measured outcome. A model that gives a measured outcome probability
  0 is ruled out.
- The second score is the quadratic score: the mean over shots of 2 p(x) - sum of p(y)^2.
- The excess is the log score of the measured frequencies minus the log score of the model. Its
  reference is the excess of counts drawn from the model itself.

`nv score` computes these scores. `test_near` and `test_far` are scored separately.

## Comparisons

Each comparison is the paired test of `noisevault.score.paired` on the test circuits of one run,
at the 5% level, with the log score. The primary comparisons are:

1. `calibration` against `medians`: do the values of each qubit forecast better than device
   medians?
2. `calibration` against `stale(d)` for each d, and against `bundled`: does a newer calibration
   forecast better? The difference in log score as a function of d is the staleness curve. These
   models have no fitted parameter, so this comparison uses every circuit of the run.
3. `calibration` against `symmetric`: do two readout errors for each qubit forecast better than
   their mean?
4. `two_factor` against `calibration`: do two fitted factors forecast better?
5. `readout_measured` against `two_factor`, and `locus` against `readout_measured`.

For each comparison, the result over all runs is the number of runs in which the first model is
better, the second model is better, or neither, for each device. A comparison counts as confirmed
when the sign test over the runs rejects equal chances at the 5% level, with the Holm correction
for the five comparisons.

The adequacy of the best model is its excess on `test_far`, with the p-value of `nv score`.

## Rules

- Every collected run is in the analysis. A run whose counts bind to the planned calibration
  because the device was recalibrated before the job ran is marked, and results are given with
  and without the marked runs.
- A model that cannot simulate a circuit of a run has no score for that run, and the reason is
  recorded.
- The analysis of IBM runs stops with the run of the week of 2027-02-01, or earlier when at least
  8 weeks of runs exist for each device.
- Results that are not in this plan are labelled exploratory.

## Changes

- 2026-11-12: first version.
