# Score a profile as a forecast

A noise model gives a probability to each outcome of a circuit. Thus a noise model is a
probabilistic forecast of the counts that a device measures. `nv score` and `noisevault.score`
measure how well a profile forecasts a counts file. They fit nothing. They use proper scoring
rules, which give the best expected score to the model that is true.

`nv compare` asks a different question. It fits a gate factor and a readout factor to the counts,
and it tests if the fitted model explains every circuit. When no factor pair explains the counts,
`nv compare` reports a poor fit, but it does not say which of two models is closer to the device.
`nv score` answers that question.

## What a score is

For each circuit in the counts file, NoiseVault computes the outcome probabilities of the circuit
with the reference simulator and the profile. The profile's `unmodeled_error` factors apply, as
in every export. Then NoiseVault gives each measured shot a score from the probability of its
outcome. A circuit score is the mean over the shots of the circuit. A total score is the mean
over all shots of all circuits.

The profile does not have to be the calibration that the circuits were planned from. A profile
of another date, of another device or written by hand is valid, if the reference simulator can run
the circuits on the measured qubits with it. `same_calibration` is true when the profile has the
calibration fingerprint that the counts file records. `calibration_age` is the run time minus
`device.calibrated_at`. It is negative for a calibration made after the run.

## The two rules

In the formulas, p(x) is the probability that the profile gives outcome x, and f(x) is the
measured frequency of outcome x in a circuit. A higher score is better for both rules.

- **Log score.** The score of a shot with outcome x is ln p(x). The circuit score is
  `sum_x f(x) ln p(x)`. For three or more outcomes, the log score is the only proper rule, up to
  scale, that depends only on the probability of the outcome that occurred.
- **Quadratic score.** The score of a shot with outcome x is `2 p(x) - sum_y p(y)^2`. The circuit
  score is `2 sum_x f(x) p(x) - sum_y p(y)^2`. This score is 1 minus the Brier score for a
  discrete outcome, so a forecast that gives the measured outcome probability 1 scores 1. The
  quadratic score is always finite.

## Ruled out

A profile can give a measured outcome a probability of exactly 0. Some IBM calibrations state a
readout error of 0 on a qubit, and the device still reads the wrong value on some shots. The log
score of such a shot is minus infinity. `ruled_out` counts these shots, and the log score of the
profile is then `-inf`. The profile is ruled out: one shot is enough to prove that the device does
not follow this model. The quadratic score stays finite, so two ruled-out profiles can still be
compared with the quadratic rule.

## The excess and its shot-noise reference

The saturated log score of a circuit is `sum_x f(x) ln f(x)`. No model gets a higher log score on
the same counts. The excess is the saturated log score minus the log score of the profile:

```text
excess = sum_x f(x) ln( f(x) / p(x) )
```

The excess is the plug-in estimate of the Kullback-Leibler divergence from the device to the
profile, in nats per shot. The total excess is the mean over all shots. A ruled-out profile has
an infinite excess, and the JSON output gives `null` for it.

The excess is more than 0 also when the profile is exactly true, because the measured
frequencies have shot noise. For the reference, NoiseVault draws 200 sets of counts from the
profile itself, with the same shots for each circuit. The seed comes from the SHA-256 of the
counts file, as in `nv compare`. `excess_noise_mean` and `excess_noise_95` are the mean and the
95th percentile of the excess of the drawn sets. The p-value is

```text
p = (1 + number of drawn sets with excess >= measured excess) / 201
```

A ruled-out profile has p = 0. This test is a goodness-of-fit test of a fixed model with no
fitted parameter. A small p-value means that the profile is not the device. It does not say
which other profile is better.

## The paired test

`paired(a, b)` tests whether forecast a or forecast b scores better on the same counts. Both
forecasts must score the same counts file and the same circuits. For each rule and each
outcome x, the test takes the difference d(x) = s_a(x) - s_b(x) of the two scores. For a circuit
c with N_c shots:

```text
mean_c = sum_x f(x) d(x)
var_c  = sum_x f(x) d(x)^2 - mean_c^2
D      = sum_c N_c mean_c
Var(D) = sum_c N_c var_c
z      = D / sqrt(Var(D))
```

`var_c` above uses the measured frequencies. An outcome that no shot showed then adds no
variance. If such an outcome has a large score difference, the test rejects too often. The test
therefore computes `var_c` three times: with the measured frequencies, with the probabilities of
profile a and with the probabilities of profile b. It uses the largest of the three values.

The two-sided p-value comes from the standard normal distribution. The result gives the mean
difference per shot, D / N, with its standard error, z and p for each rule. `better` is `"a"` or
`"b"` when p is below 0.05, and `"neither"` when it is not.

The log rule decides. These cases have no z:

- One profile is ruled out and the other is not. The profile that is not ruled out is better.
- Both profiles are ruled out. The log rule gives no verdict, and the quadratic rule decides.
- One profile gives probability 0 to an outcome that the other profile expects, and no shot
  showed that outcome. One more shot can then change the difference without limit, so the log
  rule gives no test, and the quadratic rule decides.
- The two profiles give the same score to every outcome. The verdict is "no difference".

### When the test is valid

The paired test needs three conditions:

- The shots are independent. A device that drifts during a job gives shots that are not.
- The number of shots is large enough for the normal approximation. The largest of three
  variances makes the test more careful with rare outcomes, but it gives no exact guarantee.
- Both profiles were fixed before you saw the counts. A profile is also fixed when its factors
  were fitted on other shots or on other circuits. A profile whose factors `nv compare` fitted on
  the same counts has an advantage that the test does not correct. Such a profile scores better
  than its true quality, so do not test it on the counts of its fit.

`nv score` tests the best profile against each other profile and makes no correction for the
number of tests. For a planned comparison of two profiles, give only those two.

## Held-out data

Two methods of `MeasuredCounts` give counts that a fit did not see:

- `counts.select(names)` gives the same run with only the named circuits.
- `counts.split(fraction, seed)` divides the shots of each circuit at random, without
  replacement. The first part gets `round(fraction * shots)` shots of each circuit, and the second
  part gets the other shots. The same seed gives the same parts. Each part keeps every circuit,
  so `split` refuses a fraction that leaves a part of a circuit with no shots.

Each result is a new counts file with its own SHA-256, bound to the same calibration as the run.
`nv compare` and `score` accept it. Fit on one part and score on the other:

```python
import noisevault as nv
from noisevault.counts import plan, simulate
from noisevault.score import paired, score

kingston = nv.load("ibm_kingston@2026-04-15")
device = kingston.model_copy(update={"unmodeled_error": {"gates": {"factor": 1.8}}})
counts = simulate(device, plan(kingston), shots=8000, seed=1)

train, test = counts.split(0.5, seed=1)
fitted = kingston.compare(train).fitted_profile()
result = paired(score(fitted, test), score(kingston, test))
print(result.better)
# a
```

The fitted profile is model a, and it scores better than the calibration on the shots that the
fit did not see.

## An example

[`examples/kingston-simulated.counts.json`](../examples/kingston-simulated.counts.json) holds
counts that NoiseVault simulated from the ibm_kingston calibration of 2026-04-15, with gate
errors 1.8 times and readout errors 1.3 times the calibration. The command scores the
calibration and the ibm_fez calibration as forecasts of these counts:

```text
$ nv score ibm_kingston@2026-04-15 ibm_fez kingston-simulated.counts.json
counts kingston-simulated.counts.json, simulated, sha256:efeaf3d1f923
run 2026-04-16 09:30Z on ibm_kingston, qubits 148-149-150-151

model                             age  log score   excess  quadratic  ruled out
ibm_kingston@2026-04-15          26 h   -0.85501  0.00362    0.70980          0
ibm_fez                  other device       -inf      inf    0.70764         52

ibm_fez against ibm_kingston@2026-04-15:
  log score: ibm_fez is ruled out by 52 shots, so ibm_kingston@2026-04-15 is
    better
  quadratic score: ibm_kingston@2026-04-15 is better by 0.00217 per shot than
    ibm_fez (z = 5.99, p = 2.2e-09, standard error 0.00036)
```

The rows put the best profile first. Profiles that are ruled out come last, in the order of
their quadratic score. The ibm_fez profile states P(1|0) = 0 on qubits 148 and 150. In the
`readout` circuit, which has no gates, 52 shots read qubit 148 or 150 as 1. These shots rule out
ibm_fez. The `excess` of ibm_kingston is above its shot noise, because the device of
the example has more error than the calibration states. In a terminal of 120 columns, the table
also shows each profile's fingerprint and the `noise 95%` column, the 95th percentile
of the shot-noise excess. In a narrower terminal, the table leaves out the fingerprint first,
then the noise column, then the age.

With two or more profiles, one line for each other profile gives the paired test against the
best profile. `nv score ... --json` prints all numbers, also the scores of each circuit, as one
JSON object. `nv score` exits with status 0 when it scored every profile.
