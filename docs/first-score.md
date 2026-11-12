# Is my noise model any good?

A profile gives your simulator the noise of a device. This page shows how to find out how far
you can trust it, with three commands. You need a profile and a counts file: the results of
some circuits on the device. To try the commands without a device, download example counts.
NoiseVault simulated them from the `ibm_kingston` calibration of 2026-04-15, with gate errors 1.8
times and readout errors 1.3 times the calibration:

```bash
curl -O https://raw.githubusercontent.com/dvgyl/noisevault/main/examples/kingston-simulated.counts.json
```

[Measure a profile against hardware](recipes.md#measure-a-profile-against-hardware) tells how
to get counts from a device.

## 1. Look at the profile

```text
$ nv show ibm_kingston@2026-04-15
ibm_kingston@2026-04-15T07:15:15Z  nv:609c845ed934
device        IBM, Heron r2, superconducting, 156 qubits
...
readout       median error 9.52e-03 (P(1|0) 8.79e-03, P(0|1) 1.15e-02)
value ages    readout 0.7 h, T1 7.9 h, T2 7.8 h, 1q gates 7.2 h, 2q gates 6.5 h
              older than the calibration time (median)
```

Two lines tell you what the calibration cannot know:

- **`value ages`.** A calibration has one time, but each value has its own. A gate error can be
  hours or days older than the calibration time.
- **A readout error of exactly 0.** When the calibration states such a value, `nv show` names
  the qubits. The device does read these qubits wrong on some shots, so the profile calls
  results impossible that occur. `Profile.floored()` and `nv score --floor` replace these
  values. [Scoring](scoring.md#ruled-out) gives the rule.

## 2. Score the profile on the counts

```text
$ nv score ibm_kingston@2026-04-15 kingston-simulated.counts.json
counts kingston-simulated.counts.json, simulated, sha256:efeaf3d1f923
run 2026-04-16 09:30Z on ibm_kingston, qubits 148-149-150-151

model                        fingerprint   age  log score   excess  noise 95%  quadratic
ibm_kingston@2026-04-15  nv:609c845ed934  26 h   -0.85501  0.00362    0.00140    0.70980
```

Read two columns first:

- **`excess`** says how far the profile is from the best possible forecast of these counts. It
  is 0 for a perfect profile, apart from shot noise.
- **`noise 95%`** is the excess that shot noise alone gives in 95% of the cases, if the profile
  is right.

Here the excess is 0.00362 and shot noise explains only 0.00140. So the profile does not
describe these counts, and the counts have enough shots to show it. A third column, `ruled out`,
appears when the profile gives a measured result the probability 0.

## 3. Find a better profile, and check it

`nv compare` fits one factor for the gate errors and one for the readout errors, and `-o` saves
the profile with these factors:

```text
$ nv compare ibm_kingston@2026-04-15 kingston-simulated.counts.json -o fitted.json
...
gate errors     x2.16 (95% interval 1.77 to 2.59)
readout errors  x1.34 (95% interval 1.16 to 1.53)
fit             within shot noise on every circuit (p = 0.69)

saved: fitted.json, ibm_kingston nv:e378e0aa8065 with the fitted factors
```

Then score the two profiles together. `nv score` takes any number of profiles and tests the best
one against each other one:

```text
$ nv score ibm_kingston@2026-04-15 fitted.json kingston-simulated.counts.json
model                        fingerprint   age  log score   excess  noise 95%  quadratic
fitted.json              nv:e378e0aa8065  26 h   -0.85221  0.00083    0.00150    0.71001
ibm_kingston@2026-04-15  nv:609c845ed934  26 h   -0.85501  0.00362    0.00140    0.70980

ibm_kingston@2026-04-15 against fitted.json:
  log score: fitted.json is better by 0.00280 per shot than
    ibm_kingston@2026-04-15 (z = 4.27, p = 1.9e-05, standard error 0.00065)
```

The excess of the fitted profile is inside the shot noise. But this check used the counts of
the fit, so the fitted profile has an advantage. For a fair check, fit on one part of the shots
and score on the other part. [Held-out data](scoring.md#held-out-data) shows how, in five lines
of Python. In the same way you can score an older calibration, which `nv pull --at` gets, or
the profile of another device against the same counts.

## What a score does not tell you

A score holds for circuits like the ones in the counts file. Two limits matter in practice:

- **Circuits with few gates say little.** The interval of the gate factor above reaches from
  1.77 to 2.59. By a computation with the profile, the
  [deep circuit set](recipes.md#run-the-deep-circuit-set) measures the same factor about 8 times
  more exactly for the same number of shots.
- **A circuit that repeats one gate many times is outside each such model.** A profile states
  how large an error is, not its direction. Small errors with one direction add up in such a
  circuit. On an IBM device, 256 `sx` gates in a row gave the right result in about 1% of the
  shots, and the profile predicted about 70%.
  [Limitations](limitations.md#comparison-with-hardware) has the numbers. When the counts hold
  such probe circuits, `nv score` leaves them out of the score and shows them in a
  [probe report](scoring.md#probe-circuits-and-what-a-score-leaves-out).
