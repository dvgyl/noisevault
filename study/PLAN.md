# Analysis plan of the realism study

This file fixes the analysis before a test circuit of a run that counts is scored. The commit
date of each version of this file is the record. The sections from [Question](#question) to
[Rules](#rules) state the plan in force. [Changes](#changes) lists each change with its date and
with what was known at that time. [Freeze](#freeze) names the frozen versions.

## Question

A calibration is a forecast of the results of a circuit. The study measures four things:

1. How well a noise model from the calibration of a device forecasts counts from that device.
2. How fast the forecast gets worse as the calibration gets older.
3. How much a small number of shots on the device improves the forecast.
4. How large the systematic rotation error of a gate is. A calibration does not state this
   error, and a noise model from a calibration cannot forecast a circuit in which it adds up.

## Runs

A run is one job on one chain of 4 qubits of one device. The job holds the circuits of
`noisevault.suite.suite` (circuit set version 2) for the calibration that the device had when
the job was planned, with `fresh` set to the ISO year times 100 plus the ISO week. The circuits
of a job run in a random order that the week and the chain fix. Each circuit has 1250 shots. The
counts bind to the calibration in effect when the job ran.

- The devices are `ibm_kingston`, `ibm_fez` and `ibm_marrakesh`. Each device has two runs each
  week. The best chain is the chain of 4 connected qubits with the lowest sum of the stated
  errors of its 1-qubit gates, readout and 2-qubit gates. The median chain is the chain at the
  median of this sum.
- The runs that count are the runs of the weeks of 2026-11-16 to 2027-01-04. These are 8 weeks,
  and there is no earlier stop.
- A run before the week of 2026-11-16 is a pilot run. A run after the week of 2027-01-04 is a
  late run. A trial run is one more run in a week that has its runs (files `best-trial.*` and
  `median-trial.*`). Pilot runs, late runs and trial runs are in no estimate, and their results
  are exploratory.
- A run of another vendor is secondary.
- A run whose counts bind to the planned calibration because the device was recalibrated before
  the job ran is marked. Results are given with and without the marked runs.

The pilot runs were six runs with the `nv compare` circuits (2026-11-09 and 2026-11-10) and six
runs with version 1 of the deep circuits (2026-11-12). They showed that IBM calibrations can
state a readout error of exactly 0, that two factors do not fit the counts of the short
circuits, and that the 1-qubit circuits of version 1 let a systematic over-rotation add up. The
fits and the test scores of the pilot runs were read before the entries of 2026-11-12 from "the
scores of the pilot runs" on were written.

## Circuits

`noisevault.suite.role` gives each circuit one role:

- `train`: a model can fit its parameters on these circuits. They are the two readout circuits,
  and mirror circuits on the chain, on each pair and on each qubit of the chain.
- `test_near`: mirror circuits on the chain from two other random streams. One stream is the
  same in each week. The other stream is new in each week.
- `test_far`: circuits of another kind. These are the `nv compare` circuits other than
  `readout`, a circuit with 1-qubit gates on all qubits at the same time, mirror circuits on a
  chain of 8 qubits, and one mirror circuit on a chain of 20, 40 and 80 qubits.
- `probe`: circuits in which a systematic error of a gate adds up. These are `axis_d64` and
  `axis_d256`, one gate 64 and 256 times on each qubit of the chain, and `periodic_d*`, a mirror
  circuit on the chain that repeats two layers.

No model sees a test circuit or a probe circuit before its score is computed.

A mirror circuit has layers of random gates, then one random Pauli gate or no gate on each
qubit, then the layers undone in the opposite order. On a device whose 1-qubit gates turn about
one axis, a random virtual Z rotation by a multiple of 90 degrees, or none, comes before each
1-qubit gate. This is the design of randomized mirror circuits (Proctor et al., Nature Physics
18, 75, 2022). The depth of a circuit comes from the calibration with its stated errors: the
first depth at which the predicted polarization is at or below a target. The polarization of a
circuit on n qubits with the probability p of its ideal result is (p - 1/2^n) / (1 - 1/2^n). The
targets are 0.8, 0.6 and 0.4 on the chain, 0.75 and 0.4 on a pair, 0.7 on a qubit, and 0.6 and
0.35 on 8 qubits. The depth of a circuit on 20, 40 or 80 qubits is the first depth at which the
calibration predicts a wrong bit with a mean probability of 0.15 or more over the qubits.

The comparisons use these groups of circuits:

- `all4`: each circuit whose qubits are all qubits of the `readout` circuit of the run, without
  the probe circuits.
- `test4`: the test circuits in `all4`. `test_near` and `test_far` are its two parts.
- `wide`: the circuits on 8 qubits.
- The circuits on 20, 40 and 80 qubits, which only the stabilizer simulator scores.

## Models

Each model is a profile, so the reference simulator gives its forecast. C is the calibration that
the counts bind to.

| Name | Model |
| --- | --- |
| `ideal` | No noise |
| `generic` | One fixed model for every device: depolarizing parameter 0.001 for 1-qubit gates and 0.01 for 2-qubit gates (average infidelities of 0.0005 and 0.0075), readout error 0.01, no relaxation |
| `medians` | Each value of C replaced by the median of that value over the device |
| `stale(d)` | The calibration of the same device that was in effect d days before the run, for d = 1, 7, 30, 90, 180 and 365. When that calibration is C itself, the model is C |
| `bundled` | The bundled profile of the device |
| `symmetric` | C with the two readout errors of each qubit replaced by their mean |
| `calibration` | C as stated |
| `averaged(7)` | Each value of C replaced by the median of that value over the 7 calibrations in effect at the run time and 1 to 6 days before |
| `two_factor` | C with one gate factor and one readout factor from `nv compare` on the `train` circuits |
| `two_factor_before` | C with the two factors that `two_factor` fitted in the run of the same device and the same chain kind 5 to 9 days before |
| `readout_measured` | C with the readout errors of each qubit measured on `readout` and `readout_1`, and one gate factor fitted on the other `train` circuits |
| `locus` | `readout_measured` with one factor for each qubit fitted on its `qubit*` circuit, and one factor for each pair fitted on its `pair*` circuits, in place of the one gate factor |

Zero floor: a calibration can state a probability of exactly 0, and a device then gives the
outcome that the model calls impossible. Each model from a calibration is therefore scored in
its floored form: each readout probability of exactly 0 is replaced by half of the smallest
readout probability above 0 in the same calibration. Each model starts from C as stated, and the
floor applies to the result. The `two_factor` model takes the factors that `nv compare` fits on
C as stated, and the floor applies after the fit. The floored form is the form in each
comparison. Only the calibration C is also scored as stated. The number of runs in which C as
stated is ruled out is a result of its own.

A measured readout probability is (k + 1/2) / (n + 1) for k wrong results in n shots.

A model that cannot simulate a circuit of a group has no score for that group, and the reason
is recorded. A comparison needs the scores of both of its models on its group.

## Scores

- The primary score is the log score: the mean over shots of the natural log of the probability
  that the model gave to the measured outcome. A model that gives a measured outcome probability
  0 is ruled out.
- The second score is the quadratic score: the mean over shots of 2 p(x) - sum of p(y)^2.
- The excess is the log score of the measured frequencies minus the log score of the model. The
  misfit is the excess minus the mean excess of counts drawn from the model itself, for each
  shot. A fitted model has an error from the shots of its fit that this reference does not hold,
  so its misfit is a little too high, and the plan gives no p-value for it.

`nv score` computes these scores. `noisevault.score.paired` gives the test of two models on one
run, at the 5% level. Its verdict is the verdict of the log score. This test is a description.

## Primary comparisons

The five primary comparisons are:

1. `calibration` against `medians` on `all4`: do the values of each qubit forecast better than
   device medians?
2. `calibration` against `stale(30)` on `all4`: does the calibration of the run forecast better
   than the calibration from 30 days before?
3. `calibration` against `symmetric` on `all4`: do two readout errors for each qubit forecast
   better than their mean?
4. `two_factor` against `calibration` on `test4`: do two fitted factors forecast better?
5. `locus` against `two_factor` on `test4`: do measured readout errors and one factor for each
   qubit and each pair forecast better than two factors?

The unit is the device-week: the set of runs of one device in one ISO week. For a comparison and
a run, delta is the log score of the first model minus the log score of the second model, for
each shot, on the circuits of the comparison. It is the difference that
`noisevault.score.paired` gives for the log score. Y is the mean of delta over the two runs of a
device-week. A device-week is complete when it has its run on the best chain and its run on the
median chain, and each run has a finite delta. Only complete device-weeks are in the estimate,
and the summary gives the number of the others.

For a device with n complete device-weeks, n at least 4, m is the mean of Y and s^2 is its
sample variance. The estimate is the mean of m over the devices. Its standard error is the
square root of the sum over the devices of f s^2 / n, divided by the number of devices. The
factor f corrects for weeks that are correlated. The first value of rho is a ratio. Its upper
part is the sum, over the devices and over each pair of weeks of a device that are 7 days apart,
of the product of the two values of Y - m. Its lower part is the sum, over the devices and over
all their weeks, of (Y - m)^2. rho is this value plus (1 + 3 times this value) / n, with the
mean n of the devices, limited to the range from 0 to 0.9. g is the sum for k from 1 to n - 1
of (1 - k / n) rho^k, and f = (1 + 2 g) / (1 - 2 g / (n - 1)). The test is the two-sided t-test
with the sum of n - 1 over the devices as degrees of freedom. The interval of an estimate is
the 95% interval of this t distribution. A standard error of 0 gives no test and no interval.

The five tests form one family with the Holm correction at the 5% level. A comparison with no
test counts with a p-value of 1. A confirmed comparison holds for these devices in the weeks of
the study. It is not a result for devices in general. A confirmed difference below 0.001 nats
for each shot is called small.

The primary result of a comparison uses all runs that count and the log score.

## Strata

Two properties of a run are strata: the chain kind (best or median), and whether the
calibration states a readout probability of exactly 0 on a qubit of the chain. The summary gives
each primary estimate for each chain kind, and for the runs with and without such a zero. In a
stratum, Y of a device-week is the mean of delta over its runs in the stratum, one or two. The difference of delta between the best chain
and the median chain of a device-week is a secondary estimate with the same rule for the
standard error. A confirmed comparison 5 is read with its two parts: `readout_measured` against
`two_factor`, and `locus` against `readout_measured`.

## Secondary results

Each result of this section is descriptive. None is in the family of tests.

- For each primary comparison: the estimate with f = 1, the values m and s of each device, the
  interval in which Y of a new week of the same device is expected, the median standard error
  of delta of a run, the sign test of Y over the device-weeks, the verdict counts of the runs,
  and the estimate without the marked runs, for `test_near` and `test_far`, and with the
  quadratic score.
- `calibration` against `stale(d)` for each other d, and against `bundled`. The staleness curve
  uses the nominal age d as its x value. A point of the curve is the estimate for delta of
  `stale(d)` against `calibration` on `all4`, with its 95% interval and the number of
  device-weeks. The trend is the least-squares slope of delta against log10(d) of each run,
  with the same estimate. Each result is given for each chain kind. The median ages of the
  values of each model are a note.
- `readout_measured` against `two_factor`, and `locus` against `readout_measured`.
- `averaged(7)` against `calibration` on `all4`, with the number of different calibrations in
  each median.
- `two_factor_before` against `calibration` and against `two_factor` on `test4`.
- On the 8-qubit circuits alone: `two_factor` against `calibration`, `locus` against
  `two_factor`, and `calibration` against `medians`, with the number of qubits of the 8-qubit
  chain that are in the chain of the run.
- The misfit of `locus`, `two_factor` and `calibration` on the `test_far` circuits of the chain:
  its median for each device, and the part of the excess that comes from outcomes with fewer
  than 1 expected shot.
- The age of a value against its fitted factor: the Spearman rank correlation between the age
  of the value of each qubit and each pair and the absolute value of ln f of the `locus` model,
  with no p-value, with the number of factors at a limit of the search, and also without them.
- The number of runs with a fitted factor that the fit does not identify.
- The probe circuits. For each qubit and each probe, the measured fraction of wrong bits and
  the forecasts of `calibration` and of `two_factor`. The fraction of wrong bits of the
  `readout` circuit is the baseline. The growth ratio of a qubit is the fraction above the
  baseline at 256 gates, divided by the fraction above the baseline at 64 gates. A systematic
  rotation error gives a ratio near 16 when the fractions are small. An error with no direction
  gives a ratio near 4. For the periodic probe, the summary gives the measured probability of
  the ideal result and the forecasts of `calibration` and of `two_factor`.
- A check of independent shots. For each circuit that both runs of a device-week hold, the two
  sets of counts are compared. On 8 qubits, the outcomes with fewer than 5 expected shots become
  one cell, and the statistic is the Pearson chi-square statistic of the 2 by K table. On a
  circuit of 20, 40 or 80 qubits, the same statistic uses the number of wrong bits, with the
  same rule for a cell. The summary
  gives the ratio of the statistic to its degrees of freedom for each device-week.

## Circuits on 20, 40 and 80 qubits

The exact simulator cannot run these circuits. A stabilizer simulator gives the forecast of a
model, with each noise channel replaced by its Pauli twirl. Three scores apply: the ideal bit
string (right or wrong), the number of wrong bits, and each qubit alone (the sum over the qubits
of the log probability of its bit). The score of each qubit judges each qubit alone, so it does
not judge the correlations between qubits. The verdict of a comparison on these circuits is the
verdict of the score of each qubit. The log score of the number of wrong bits also has a test.
The score of the ideal bit string and the quadratic score of the number of wrong bits are given
as values, with no test. The comparisons are `calibration` against `medians`,
`two_factor` against `calibration`, and `averaged(7)` against `calibration`. The fitted factors
come from the `train` circuits on 4 qubits of the same run.

Each marginal of the forecast uses 4,000,000 samples, or its exact value. The variance of a
test holds the sampling variance of both forecasts. For the score of each qubit, the sampling
variance of one model and one circuit is a^T C a / S. C is the covariance matrix of the
wrong-bit values of the qubits over the S samples, and a holds N (f - q) / (q (1 - q)) for each
qubit, with N shots, the measured fraction f and the forecast q of a wrong bit. The score of the
number of wrong bits uses (N^2 / S) (sum over the cells of f^2 / p - 1). Each model and each
circuit has random samples of its own. A test gives no verdict when the sampling variance is
larger than the shot variance. It also gives no verdict when a qubit of a circuit has fewer
than 100 sampled wrong bits or fewer than 100 sampled right bits, or when a cell of the number
of wrong bits that holds a measured shot has fewer than 100 samples. Both runs of a device-week hold the same circuits on 8,
20, 40 and 80 qubits, so a summary over these circuits uses the mean of the two runs of a
device-week as one value. The size of the twirl approximation is measured on the 8-qubit
circuits, where both forecasts exist, for each model that these comparisons use.

## Rules

- Every run of the weeks that count is in the analysis.
- A run that is in the data two times counts one time.
- Results that are not in this plan are labelled exploratory.
- From the first run that counts, each change of this plan and each change of the analysis code
  that changes a result is marked in [Changes](#changes) as made after test scores were read.

## Freeze

The Git tag `realism-study-v1` marks the frozen versions of this plan, of the circuit set
(`noisevault.suite`), of the campaign script and of the analysis code. The workflows
`ibm-campaign.yml` and `study.yml` check out this tag and install the library versions of
`.github/study-constraints.txt`. Work on other parts of the repository thus does not change a
run or a result of the study. A change of the frozen code gets a new tag and an entry in
[Changes](#changes).

The workflow `study.yml` computes the scores after each weekly campaign and saves them as the
artifact `study-results`. Each signed-in GitHub user can read this artifact, so the scores of a
run are open from the day of the run. For this reason, each change after the first run that
counts is a change after test scores were read.

Known at the freeze: the fits and the test scores of the pilot runs, and the results of one
trial run of circuit set version 2 on `ibm_fez` on 2026-11-12. No run that counts existed.

## Changes

- 2026-11-12: first version.
- 2026-11-12: the age of a calibration. IBM gives a time for each value of a calibration. On
  2026-11-12, the readout values of `ibm_fez` were 51 hours older than the time of the
  calibration, and its 1-qubit gate errors were 151 hours older. The staleness curve therefore
  uses the age of the values at the run time: the median age of the readout values, of the
  1-qubit gate errors and of the 2-qubit gate errors of each model. The age from the time of the
  calibration is also given. This entry was written before a test circuit of the deep circuit
  set was scored.
- 2026-11-12: the family of tests. The first version named more comparisons than the five that
  its Holm correction counted. The plan now names five primary comparisons, one sign test for
  each, and the secondary comparisons that are descriptive. This entry was written before a test
  circuit of the deep circuit set was scored.
- 2026-11-12: wide circuits. From the week of 2026-11-16, a run also holds deep circuits on the
  suggested chain of 8 qubits of the device. This chain can hold the qubits of the `train`
  circuits, and it holds at least 4 other qubits. These circuits are in `test_far`. Three
  secondary comparisons use the wide circuits alone: `two_factor` against `calibration`, `locus`
  against `two_factor`, and `calibration` against `medians`. They test whether a model that was
  fitted on 4 qubits forecasts 8 qubits. No score of a deep run was read before this entry.
- 2026-11-12: the model `averaged(7)` and its secondary comparison. A public archive of IBM
  calibrations from February to November 2026 shows that a value changes about one time each
  day, and that the change does not grow much with the time between two measurements. It also
  shows that the lowest readout errors of a calibration are biased low: for the lowest tenth of
  the P(1|0) values of a calibration, the median of the next measured value of the same qubit
  is 2.4 to 3.7 times higher. A median over a week removes a part of this noise. This entry was
  written before a score of a deep run was read.
- 2026-11-12: one more secondary comparison, the age of a value against its fitted factor. This
  entry was written before a fitted factor of a qubit or a pair was read.
- 2026-11-12: the `generic` model. Its depolarizing error is the depolarizing parameter: 0.001
  for 1-qubit gates and 0.01 for 2-qubit gates. These values are average infidelities of 0.0005
  and 0.0075. No score of a deep run was read before this entry.
- 2026-11-12: the depths of the deep circuit set. From the week of 2026-11-16, `suite` takes the
  depths at which the profile with a gate factor of 2 predicts the ideal result with the
  probabilities 0.8, 0.6 and 0.4 (pairs: 0.8 and 0.55, qubits: 0.85). The runs of the week of
  2026-11-09 used the stated errors and the probabilities 0.9, 0.75 and 0.55 (pairs: 0.85 and
  0.6, qubits: 0.9). The reason is the gate factors of the exploratory runs. No score of a deep
  run was read before this entry.
- 2026-11-12: clarifications. The `readout` circuit is a `train` circuit. Only the calibration C
  is also scored as stated. The verdict of a run in a sign test is the verdict of the log score.
  The adequacy test uses the `locus` model. The relation of the age of a value to its fitted
  factor is a Spearman rank correlation. The summary gives each result also without the marked
  runs. No score of a deep run was read before this entry.
- 2026-11-12: Clifford circuits on 20, 40 and 80 qubits. From the first week in which a run
  holds them, a run has one mirror circuit of Clifford gates on a chain of each of these sizes.
  The exact simulator cannot run them, so a stabilizer simulator gives the forecast of a model,
  with each noise channel replaced by its Pauli twirl. The depth of a circuit is the first depth
  at which the calibration with a gate factor of 2 predicts a wrong bit with a mean probability
  of 0.15 or more over the qubits. These circuits are test circuits. Three scores apply: the
  ideal bit string (right or wrong), the number of wrong bits, and each qubit alone (the sum over
  the qubits of the log probability of its bit). The verdict of a comparison on these circuits
  is the verdict of the third score. The comparisons are secondary: `calibration` against
  `medians`, `two_factor` against `calibration`, and `averaged(7)` against `calibration`. The
  fitted factors come from the `train` circuits on 4 qubits of the same run. The size of the
  twirl approximation is measured on the 8-qubit circuits, where both forecasts exist. No run
  with these circuits existed when this entry was written.
- 2026-11-12: more clarifications. The `two_factor` model takes the factors that `nv compare`
  fits on C as stated, and the zero floor applies after the fit. Each other model starts from C
  as stated too, and the zero floor applies to the result. A run that is in the data two times
  counts one time. No score of a deep run was read before this entry.
- 2026-11-12: pilot runs. The runs before the week of 2026-11-16 are pilot runs. They are not in
  the sign tests of the primary comparisons, and their results are exploratory. The reason is a
  defect of the `qubit*` circuits, found in the fits on the `train` circuits of the six deep runs
  of 2026-11-12. A `qubit*` circuit held only `sx`, `x` and `id` gates. Each of these is a
  rotation about one axis, so a small systematic over-rotation of each gate adds up over the
  circuit. A simulation with no noise shows the effect: an over-rotation of 0.005 rad for each
  `sx` gate lowers the probability of the right result of the `qubit*` circuits of `ibm_kingston`
  from 1 to between 0.35 and 0.66. Such an over-rotation adds about 0.000004 to the error of the
  gate, so a calibration cannot show it. In the six runs, the total variation distance between
  the forecast of the calibration and the counts was above 0.5 for 14 of the 24 `qubit*`
  circuits. A value above 0.5 means that the wrong result was more frequent than the right
  result. The fits of the pilot runs used these circuits, so their fitted factors do not give
  the size of the errors in the other circuits. The circuit set changes before the first run
  that counts, and a later entry gives the change. The five primary comparisons do not change.
  No score of a test circuit was read before this entry.
- 2026-11-12: the scores of the pilot runs. The test scores of the six pilot runs of 2026-11-12
  were read on 2026-11-12, after the entry on pilot runs. In those runs, the `qubit*` circuits
  and the `qubits_*` circuit decide almost each comparison. Each entry below was written after
  that reading, and before the first run that counts.
- 2026-11-12: the runs that count. The primary analysis uses the IBM runs from the week of
  2026-11-16 to the week of 2027-01-04. These are 8 weeks, and there is no earlier stop. A later
  run is exploratory. The family of five tests is for the IBM runs. The runs of other vendors
  are secondary. This entry replaces the rule on the last week.
- 2026-11-12: the unit of the primary analysis. A device-week is the set of runs of one device in
  one ISO week. For a primary comparison and a run, delta is the log score of the first model
  minus the log score of the second model, for each shot, on the circuits of the comparison. It
  is the difference that `noisevault.score.paired` gives for the log score. Y is the mean of
  delta over the two runs of a device-week. A device-week is complete when it has both runs and
  each run has a finite delta. Only complete device-weeks are in the estimate, and the summary
  gives the number of the others. For a device with n complete device-weeks, n at least 4, m is
  the mean of Y and s^2 is its sample variance. The estimate is the mean of m over the devices.
  Its standard error is the square root of the sum over the devices of f s^2 / n, divided by the
  number of devices. The factor f corrects for weeks that are correlated. rho is the pooled
  correlation of Y - m between weeks that are 7 days apart, plus (1 + 3 rho) / n with the mean n
  of the devices, limited to the range from 0 to 0.9. g is the sum for k from 1 to n - 1 of
  (1 - k / n) rho^k, and f = (1 + 2 g) / (1 - 2 g / (n - 1)). The test is the two-sided t-test
  with the sum of n - 1 over the devices as degrees of freedom. The five tests form one family
  with the Holm correction at the 5% level. A confirmed comparison holds for these devices in
  the weeks of the study. It is not a result for devices in general. The summary also gives the
  estimate with f = 1, the values m and s of each device, the interval in which Y of a new week
  of the same device is expected, the median standard error of delta of a run, the sign test of
  Y over the device-weeks, and the verdict counts of the runs. These are descriptions. This
  entry replaces the sign test over runs as the test of a primary comparison.
- 2026-11-12: the circuits of the primary comparisons. A primary comparison uses only the
  circuits whose qubits are all qubits of the `readout` circuit of the run. A `probe` circuit
  (see `noisevault.suite.role`) is in no comparison. The comparisons of two models with no
  fitted parameter use all such circuits (group `all4`). A comparison with a fitted model uses
  the test circuits among them (group `test4`). The 8-qubit circuits and the Clifford circuits
  are in secondary comparisons only. Both runs of a device-week hold the same 8-qubit circuits
  and the same Clifford circuits, so a summary over these circuits uses the mean of the two runs
  of a device-week as one value.
- 2026-11-12: the primary form. The primary result of a comparison uses all runs that count and
  the log score. The results without the marked runs, for each device, for `test_near` and
  `test_far`, and with the quadratic score are descriptions. A confirmed difference below 0.001
  nats for each shot is called small.
- 2026-11-12: strata. Two properties of a run are strata: the chain kind (best or median), and
  whether the calibration states a readout probability of exactly 0 on a qubit of the chain. The
  summary gives each primary estimate for each chain kind, and for the runs with and without
  such a zero. For one chain kind, Y is the delta of the one run. The difference of delta
  between the best chain and the median chain of a device-week is a secondary estimate with the
  same rule for the standard error. A confirmed comparison 5 is read with its two parts:
  `readout_measured` against `two_factor`, and `locus` against `readout_measured`.
- 2026-11-12: the misfit, in place of the adequacy test. A fitted model has an error from the
  shots of its fit, and the reference of `nv score` does not hold this error. The plan therefore
  gives no p-value for the excess of a fitted model. The result is the size of the misfit: the
  excess minus the mean excess of counts drawn from the model, for each shot, on the `test_far`
  circuits of the chain, for `locus`, `two_factor` and `calibration`. For a fitted model this
  value is a little too high. The summary gives its median for each device, and the part of the
  excess that comes from outcomes with fewer than 1 expected shot.
- 2026-11-12: the tests on the Clifford circuits. Each marginal of the stabilizer forecast uses
  4,000,000 samples, or its exact value. The variance of the test of the score of each qubit
  holds the sampling variance of both forecasts. For one model and one circuit this variance is
  (N^2 / S) times the sum over the qubits of (f - q)^2 / (q (1 - q)), with N shots, S samples,
  the measured fraction f and the forecast q of a wrong bit. The score of the number of wrong
  bits uses (N^2 / S) (sum over the cells of f^2 / p - 1). A test gives no verdict when the
  sampling variance is larger than the shot variance. The summary uses one value for a
  device-week. The score of the ideal bit string gives no verdict on 40 and 80 qubits. The
  score of each qubit judges each qubit alone, so it does not judge the correlations between
  qubits. The size of the twirl approximation is given for each model that the Clifford
  comparisons use.
- 2026-11-12: more descriptions. (1) The staleness curve uses the nominal age d as its x value,
  and the three median ages of the values are a note. A point of the curve is the estimate of
  the entry on the unit, for delta of `stale(d)` against `calibration` on `all4`, with its 95%
  interval and the number of device-weeks. The trend is the least-squares slope of delta against
  log10(d) of each device-week, with the same estimate. Each result is given for each chain
  kind. (2) The age of a value against its fitted factor has no p-value. The summary gives the
  number of factors at a limit of the search, and the correlation also without them. (3) A check
  of independent shots: for each circuit that both runs of a device-week hold, the two sets of
  counts are compared. On 8 qubits, the outcomes with fewer than 5 expected shots in the pooled
  counts become one cell, and the statistic is the Pearson chi-square statistic of the 2 by K
  table. On a Clifford circuit, the same statistic uses the number of wrong bits. The summary
  gives the ratio of the statistic to its degrees of freedom for each device-week. (4) For each
  wide comparison, the summary gives the number of qubits of the 8-qubit chain that are in the
  chain of the run. (5) The summary gives the number of different calibrations in each median of
  `averaged(7)`, and the number of runs with a fitted factor that the fit does not identify.
- 2026-11-12: the circuit set, version 2. From the week of 2026-11-16, the circuits of a run
  change in six points. (1) A mirror circuit holds one random Pauli gate, or no gate, on each
  qubit between its two halves. (2) On a device whose 1-qubit gates turn about one axis, a random
  virtual Z rotation by a multiple of 90 degrees, or none, comes before each 1-qubit gate. (3)
  The inverse of a gate is one gate where the device permits: `sx` is undone by
  `rz(pi) sx rz(pi)`. Version 1 used three more `sx` gates. (4) The 1-qubit gates are the
  calibrated natives with no parameter, without the identity: `sx` and `x` on the IBM devices.
  (5) The depths come from the stated errors, not from 2 times the stated errors. A target is a
  polarization: the predicted probability p of the ideal result on n qubits as
  (p - 1/2^n) / (1 - 1/2^n). The targets are 0.8, 0.6 and 0.4 on the chain, 0.75 and 0.4 on a
  pair, 0.7 on a qubit, and 0.6 and 0.35 on 8 qubits. The reason is the pilot: in four of the six
  runs, the stated errors forecast the probability of the ideal result of each chain circuit
  within 0.09. In one run the difference was up to 0.13, and in one run it was up to 0.29. (6)
  Each job also holds three `test_near` circuits from a random stream of its ISO week, and the
  circuits of a job run in a random order that the week and the chain fix. This is the design of
  randomized mirror circuits (Proctor et al., Nature Physics 18, 75, 2022). With these rules, a
  systematic rotation error of a gate does not add up over a circuit, and the two halves of a
  circuit do not cancel it. In a simulation with no noise and an over-rotation of 0.01 rad for
  each `sx` and `x` gate, each mirror circuit with random layers gives its ideal result with a
  probability above 0.95 on the sets that were tested. The periodic circuit repeats two layers,
  so it keeps a part of the effect (0.94 on one set).
- 2026-11-12: probe circuits. A run holds two probe circuits, `axis_d64` and `axis_d256`: `sx`
  64 and 256 times on each qubit of the chain, with the ideal result 0 on each qubit. They are in
  no fit and in no comparison. For each qubit and each probe, the summary gives the measured
  fraction of wrong bits and the forecasts of `calibration` and of `two_factor`. The fraction of
  wrong bits of the `readout` circuit is the baseline. The growth ratio of a qubit is the
  fraction above the baseline at 256 gates, divided by the fraction above the baseline at 64
  gates. A systematic rotation error gives a ratio near 16 when the fractions are small. An
  error with no direction gives a ratio near 4. This result is descriptive.
- 2026-11-12: shots. From the week of 2026-11-16, a run has 1500 shots for each circuit, not
  2000. A job of circuit set version 2 holds 37 circuits. The free time of the IBM plan is 600 s
  in each period of 28 days, and such a period can hold five weekly runs. With 1500 shots, one
  week of runs uses about 100 s. The spread between weeks limits the precision of a primary
  estimate, and the number of shots does not.
- 2026-11-12: trial runs. A trial run is one more run of the circuit set in a week that has its
  runs, with the files `best-trial.*` and `median-trial.*`. It tests a change of the circuits on
  a device before the weekly run. A trial run is in no estimate, and its results are
  exploratory.
- 2026-11-12: the model `two_factor_before`. It is a secondary model: the gate factor and the
  readout factor that `two_factor` fitted in the run of the same device and the same chain kind
  5 to 9 days before, on the calibration C of this run, with the zero floor after. Two secondary
  comparisons on `test4` use it: against `calibration` and against `two_factor`. A fitted model
  knows the state of the device in the seconds of its job. This model shows how much of the gain
  of a fit is left after one week.
- 2026-11-12: the sampling variance of the score of each qubit, corrected. The marginals of one
  set of samples are correlated between qubits. The sampling variance of one model and one
  circuit is therefore a^T C a / S. C is the covariance matrix of the wrong-bit values of the
  qubits over the samples, and a holds N (f - q) / (q (1 - q)) for each qubit. This form
  replaces the sum over the qubits in the entry on the tests on the Clifford circuits.
- 2026-11-12: a model with no score. In the first version, a model that could not simulate one
  circuit of a run had no score for that run. A stale calibration that has no gate on one qubit
  of the 8-qubit chain then left primary comparison 2, which uses only the circuits of the
  4-qubit chain. From now on, a model that cannot simulate a circuit of a group has no score for
  that group only.
- 2026-11-12: the periodic circuit is a probe. In the first version, the periodic circuit was a
  `test_far` circuit. It repeats two layers with the same virtual rotations, so a systematic
  error of a gate adds up in it. A simulation with no noise on two qubits shows this: at 2 times
  the depth, a small systematic error of each `sx` and `x` gate or of each `cz` gate gives about
  4 times the loss of the ideal result in the periodic circuit, and about 2 times the loss in a
  mirror circuit with random layers. A comparison of models must use circuits in which such an
  error does not add up, so the periodic circuit is in no comparison. Its name now gives its
  true number of cycles.
- 2026-11-12: library versions. The weekly campaign and the study workflow install fixed
  versions of the libraries that decide the circuits and the forecasts
  (`.github/study-constraints.txt`), and each campaign run records its versions. The counts file
  of a run holds the circuits that ran, so the analysis never builds a circuit again.
- 2026-11-12: 1250 shots. A run has 1250 shots for each circuit, not 1500. One week of runs
  then uses about 85 s of the free 600 s of a period of 28 days. The period that ends on
  2026-12-07 also holds the pilot runs and a trial run, and 1500 shots would leave almost no
  free time in that week.
- 2026-11-12: text made exact, and defects of the analysis code corrected. A check of the
  analysis code against this plan gave these changes. (1) The text now gives the exact
  form of rho, the level of the interval and the rule for a standard error of 0. (2) In a
  stratum, Y is the mean over the runs of the device-week in the stratum. (3) `stale(d)` is C
  when the calibration of d days before is C itself; the code left such a run out. (4) On the
  circuits of 20, 40 and 80 qubits, only the two tests that hold the sampling variance stay, each
  model and each circuit has samples of its own, and a test needs at least 100 samples of each
  kind for each qubit. (5) The ages of the values of a pair now use the order of the chain; the
  code used the order of the circuits in the counts file, which is random from this week on.
  (6) A row of the analysis that used an outside source that failed is computed again in the
  next run. No run that counts existed when this entry was written.
- 2026-11-12: the sections above this list were written again, so that they state the plan in
  force. The rewriting changes no rule. Each rule comes from an entry of this list.
- 2026-11-12: freeze. The tag `realism-study-v1` marks the frozen versions (see
  [Freeze](#freeze)). Before the freeze, one trial run of circuit set version 2 on `ibm_fez`
  (two jobs, 1250 shots for each circuit) was analyzed, and its results were read. The fitted
  gate factors were x0.94 and x1.04. From 64 to 256 gates, the wrong bits of the `axis` probes
  above the baseline grew 11 to 20 times on 7 of 8 qubits. These results are exploratory.
