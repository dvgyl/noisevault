# Limitations

NoiseVault builds each noise model from calibration numbers. The noise model approximates the
hardware. It is not a digital twin. Results from it are predictions of a model, not of the device.

## What the model is

Every noise model is Markovian and local. The noise model applies a separate channel after each
gate, on the qubits of that gate. Each measurement gets its own readout error. Nothing depends on
what happened earlier in the circuit. NoiseVault builds the channel for a gate from one error
metric, a duration, T1 and T2. Two gates with the same numbers get the same channel, whatever the
device does physically.

The numbers come from benchmarks, mostly randomized benchmarking. A benchmark average describes
typical circuits, not any specific one. Errors that depend on the circuit (the gate's angle,
neighboring operations, the time since the last calibration) are not in the numbers.

## What every export leaves out

- **Idle noise outside stated durations.** Qubits relax only during Qiskit `delay`, Cirq
  `WaitGate` and Stim `TICK` layers with `tick_ns`. Time a qubit spends waiting while others
  run gates adds no noise unless you schedule the circuit. PennyLane circuits never get idle
  noise.
- **Crosstalk.** A gate on one pair does not disturb its neighbors. No export models always-on ZZ
  coupling, spectator errors or measurement-induced disturbance of other qubits.
- **Leakage.** Population never leaves the qubit subspace. Where a vendor's error includes
  leakage (Quantinuum), the leakage counts toward the gate's error but acts as ordinary
  depolarizing noise.
- **Coherent errors.** Every gate error becomes stochastic noise. On hardware, systematic
  over-rotations add up coherently. In the noise model, they do not add up coherently. The
  profile records Google's fSim coherent errors, but no export applies them.
- **Effects.** Profiles can record `leakage`, `atom_loss`, `erasure`, crosstalk and coherent
  over-rotation. Neither the exports nor the reference simulator model them in this release. When
  an effect sets `allow` to `omit`, the exports and the reference simulator leave it out. Every
  report lists it as omitted, and `nv compare` names it in a note. For any other `allow` value,
  each export, `simulate` and `compare` refuse the profile with an error.
- **Initial-state preparation error.** Circuits start in an ideal |0...0>. Preparation error
  applies only after explicit resets, and only when the profile has it. The bundled IBM
  profiles have no preparation error.
- **Correlated readout.** Each qubit's readout error is independent of the others.

## Approximations inside the exports

- **Relaxation floor.** When relaxation alone exceeds a gate's stated error, the gate keeps the
  relaxation and is noisier than stated. On the bundled `ibm_fez` profile, the relaxation floor
  applies on 82 loci. The 82 loci are `id`, `sx` and `x` on the same 26 qubits, and `cz` on 2
  pairs in both directions. The largest change is `cz[91, 98]`, from 0.0031 to 0.0039. The report
  lists every case. A stated error can be too large for any channel to reach. Such an error is
  far above real calibrations. The report records such an error the same way, with the achieved
  error below the stated error.
- **T2 above 2 T1.** Every export clamps such a T2 to 2 T1.
- **Gates the profile does not calibrate** get the noise of the typical native gate by default.
  An un-transpiled circuit then has the wrong gate count. Transpile to native gates for
  realistic results.
- **Double counting of single-qubit noise.** IBM and Quantinuum two-qubit errors come from
  benchmarks that include single-qubit gates. When a circuit's single-qubit gates also get their
  own noise, part of that error counts twice. The extra error is roughly that of the
  single-qubit gates inside the benchmark's two-qubit layer.
- **Stim** applies the Pauli twirl of each channel. The twirl keeps the average fidelity but drops
  relaxation's bias toward |0>. `detector_error_model()` treats Pauli channel components as
  independent, an O(p^2) change. The Stim export symmetrizes readout by default.
- **Qiskit** exports `zz` and `ms` natives as `rzz` and `rxx`, which then get the native's noise
  at any angle.

## Limits of the data

- **Device medians.** IonQ publishes only device-wide medians, and Quantinuum's published
  benchmarks pool all gate zones. Every qubit and pair of those profiles gets the same values,
  so per-qubit variation is missing.
- **Unstated metrics.** IonQ does not say which fidelity it reports. NoiseVault reads the reported
  number as average gate fidelity and records that assumption in the profile. If the number is a
  process fidelity, the gate errors of the noise model are (d + 1) / d times too high. That ratio
  is 1.5 for one qubit and 1.25 for two.
- **Old bundled profiles.** The bundled profiles hold fixed past calibrations. The Google
  calibrations are from November 2021. IBM's are from May 2024 to April 2026, and Quantinuum's
  are from August 2023 to August 2025. The vendors have recalibrated the devices since then and
  have retired some of them. Pull a current calibration when you need one.
- **Google willow_pink** is importable but not bundled, because the scope of its two-qubit error
  values (per cycle or per gate) is unresolved.
- **IBM does not document its public endpoint.** IBM can change or remove the endpoint without
  notice. The account source (`--source ibm-account`) uses the documented `qiskit-ibm-runtime`
  API.

## What has been checked, and what has not

The test suite checks that circuits simulated with the Qiskit, Cirq and PennyLane exports match
NoiseVault's reference simulator to a total variation distance of 1e-9. It also checks that Stim
agrees with the twirled reference within sampling error. On IBM fake backends, the suite checks
that the Qiskit export matches Aer's `NoiseModel.from_backend`. `nv check` runs a similar
comparison for any profile.

These checks show that the exports implement the noise model consistently. None of them compares
the noise model with outcomes measured on hardware. `nv compare` makes that comparison for counts
you measure. `nv compare` scores a profile on those counts. It also fits how far the profile's
gate and readout error rates must scale to match the device.
[Measure a profile against hardware](recipes.md#measure-a-profile-against-hardware) shows how.
[How nv compare fits the factors](#how-nv-compare-fits-the-factors) explains the fit and the tests
of its intervals.

NoiseVault has not yet been validated against hardware runs. No counts from a real device have
been compared, so treat its predictions as estimates whose error against the device is unknown.

## How nv compare fits the factors

`nv compare` reports a gate factor and a readout factor, each with a 95% interval, and a p-value
for the fit. How far to trust those numbers depends on how the fit computes them and on what the
tests have shown.

### Which counts it accepts

`nv compare` refuses counts that name another device or another calibration fingerprint. It also
refuses counts that ran before the calibration, or that use an op the profile does not calibrate on
the measured qubits. It also refuses counts whose circuits use a delay or a measurement that the
profile disables. The fit starts from the calibration without `unmodeled_error`. The factors it
reports therefore multiply the stated error rates, even when the profile already carries factors.

### The likelihood

The fit is a multinomial maximum likelihood over all circuits at once. One factor acts on every
gate error rate, and one factor acts on every readout error rate. The fit searches each factor
from 0.05 to 20. NoiseVault's reference simulator computes the outcome probabilities on the
profile with the candidate factors set in `unmodeled_error`. The reference simulator thus uses the
same noise model that every export applies. `nv check` tests each export against the same
reference. A factor therefore means the same thing in the fit as in a saved profile.

The reference runs exactly at 25 gate factors, evenly spaced in log factor. It also runs exactly
at each factor where a measured gate's scaled error equals its relaxation floor. Between those
runs, the probabilities are linear in log factor, so they stay between 0 and 1. The fit applies
readout error exactly at every point. The deviance, the fitted TVDs and the counts drawn for the
p-value all come from an exact run at the fitted factors.

The error of the straight line between two runs does not get smaller with more shots, but the
statistical error does. An example is one qubit with 1000 `x` gates and a million shots. There,
the line alone moved the 95% interval off the true factor in 64 of 100 runs. The fit therefore
checks the line near the likelihood maximum, where the likelihood-ratio statistic is at most
nine times the interval cutoff. In each gap between two runs there, the reference also runs at
the midpoint. The check multiplies the shots of each circuit by the chi-square distance between
that run and the line. When the sum is more than 0.04, the midpoint becomes a new run, and the
check repeats on each half. This limit keeps the shift of the estimate below 0.2 standard errors.
The new runs depend only on the profile and the counts, so earlier comparisons never change a
result.

The search for the maximum also follows the shots. It starts on a grid of 193 factors on each
axis and finds every peak of the likelihood on that grid. Around each peak, the search makes its
step 8 times smaller until the next step changes the log-likelihood by less than 0.001. With 5
billion shots in a circuit, the step therefore becomes much smaller than the grid spacing. Thus
the deviance does not grow with the shots because of the search. Steps on both axes at once can
stop before the top of a narrow, curved peak. The search keeps the result of these steps only when
every neighboring grid point is less than 0.001 below it. When the counts do not constrain one
factor, the neighbors on its axis are always that close. Thus one close neighbor does not show a
maximum. When a peak near the maximum fails this check, the search moves along the gate axis. At
each gate factor, it finds the best readout factor. When the best point is
at the edge of the search window, the window becomes 2 times wider. Thus the search can reach a
maximum far from the grid peak. Above about 10^12 shots in total, the rounding error of the
log-likelihood can be more than 0.001. There, the search stops when the computer cannot represent
a smaller step.

### Which factors the counts identify

Gate error and readout error can move counts the same way. The `readout` circuit from `plan()` has
no gates, so only readout error moves it, and that circuit separates the two factors.

`nv compare` reports a factor as not identified in these cases:

- No circuit moves with the factor. The fit compares the outcome probabilities at every exact run
  of the reference. A response can be equal at the two ends of the range and different between
  them. A change counts as a move when it is more than the rounding error of the reference. For a
  circuit with n operations on q qubits, that limit is (n + q + 1) times 2.2e-16. The likelihood
  of the counts then shows how much a small change constrains the factor. An example is ten
  circuits of 10^10 shots each, where the probabilities change by 8e-10 over the range. Their
  counts give a gate factor of at least 8.5.
- Fewer than 100 shots fall in circuits that move with the factor.
- The interval of the factor reaches both ends of the search range.

`nv compare` reports both factors as not identified when the two factors move the counts in the
same direction. That condition holds when the smallest eigenvalue of the Fisher information is
below 4.4e-16 times the largest, the floating-point precision of the eigenvalues. Above that limit,
the counts carry information on both factors, and the interval of each factor shows how well the
counts constrain it. An example is a circuit with 10^10 shots beside a circuit with 4000 shots.
There, the smaller eigenvalue is 2e-7 times the larger, and `nv compare` reports both factors. The
condition also needs both factors to move the counts. Thus the information on each factor must be
at least 1e-6 of the information on the other. Otherwise, the intervals show which factor the
counts constrain. A factor whose interval reaches one end of the range keeps a one-sided interval,
printed with "at most" or "at least".

### The goodness-of-fit test

The deviance is twice the gap between the log-likelihood of the counts' own frequencies and that of
the fitted model. The test compares the deviance with its degrees of freedom. The test counts the
outcomes that the profile can produce at some factor in the range. The degrees of freedom are that
count minus one for each circuit, minus the rank of the Fisher information. That rank is 2 when the
counts determine both factors, and less when they do not. The rank counts the eigenvalues above the
same limit. A circuit whose shots all leave the likelihood adds no degrees of freedom and no Fisher
information.

The p-value ranks the deviance among the deviances of 400 sets of counts drawn from the fitted
model. The smallest possible p is therefore 1/401, about 0.0025. Below 0.01, `nv compare` reports
the fit as beyond shot noise. That result means that no one pair of factors fits every circuit.
With no degrees of freedom left, `nv compare` reports the fit as not testable and gives no
p-value. But the fit can miss the measured frequencies when a factor stops at an end of the
search range. The fit also misses them when the deviance is more than the chi-square value 3.84.
In these two cases, `nv compare` runs the same test and gives a p-value. The verdict then says
that no factors from 0.05 to 20 give the measured frequencies.

The `noise TVD 95%` column comes from the same draws. The column value is the 95th percentile of
the TVD between each drawn set and the model refitted to that set. `nv compare` flags each circuit
whose fitted TVD exceeds that value. `nv compare` seeds the draws from the SHA-256 of the counts,
so the same profile and counts always give the same output.

### The intervals

Each 95% interval holds the factor values that a likelihood-ratio test does not reject at the 5%
level. `nv compare` refits the other factor at each value. The starting cutoff is the chi-square
value 3.84. When the deviance exceeds its degrees of freedom, `nv compare` multiplies the cutoff by
their ratio, the dispersion. As a result, a fit that misses by more than shot noise gets wider
intervals.

The refit of the other factor stops at the same 0.001 in log-likelihood as the search for the
maximum. The search for each end of an interval keeps two values: the last value that the test
accepts and the first value that it rejects. The search stops when the refitted log-likelihood at
these two values differs by less than 0.001. Thus the search makes no interval narrower than the
test allows, at any number of shots.

The chi-square cutoff is a large-sample approximation. It covers too little when a factor depends
on a few error shots. On one qubit with a readout error of 0.00055 and 4000 shots, about two shots
read wrong. In that case, intervals from the chi-square cutoff alone contain the true factor with
probability 0.86. NoiseVault therefore tests each end of an interval again, with the distribution
of the likelihood-ratio statistic at that end. This calibrated test passes the end when the
probability of a statistic at least as large as the observed one is more than 5%. The end moves
outward while it passes, and NoiseVault locates the outermost passing end to within 2% of the
chi-square half-width. A value inside the chi-square cutoff always passes, so this step can only
widen an interval.

When the profile allows at most two outcomes in each circuit, as with one measured qubit, the
counts of each circuit follow a binomial distribution. NoiseVault then lists every count within 10
standard deviations plus 10 of the mean. If the circuits give at most 4096 combinations of these
counts, the calibrated test uses the exact probability of each combination. The calibrated test
treats the probability outside the listed counts as a larger statistic. Otherwise, NoiseVault draws
400 sets of counts from the model at that end, with the same shots per circuit. With few error
shots, one count can carry 5% of the probability. In that case, 400 draws can reject a factor that
the exact probabilities accept, and the result depends on the seed of the draws.

NoiseVault computes the statistic of the observed counts in the same way as the statistic of each
listed or drawn set. Thus equal counts give equal statistics. In the 0.00055 case, the probability
that an interval contains the true factor rises to 0.975. At true factors 0.5, 2 and 5, it is
0.974, 0.973 and 0.968. On one qubit with a readout error of 0.00075 and one circuit of 4000 shots,
the probability is 0.988 at true factor 1. At true factors 0.5, 2 and 5, it is 0.981, 0.963 and
0.963.

The likelihood can have more than one peak. For example, an `r` gate with unequal Pauli errors
can give the same outcome probabilities at two separate gate factors. The 95% interval then runs
from the lowest accepted factor to the highest. It can therefore contain factors between the
peaks that the test rejects. The fit tests every peak with the same check as an interval end. This
includes a peak narrower than the grid step and a peak that only the calibrated test accepts.
The estimate is the highest peak. When two peaks differ by at most 0.02 in log-likelihood, the
counts cannot order them, and the estimate is the peak nearest factor 1.

Most listed and drawn sets have their maximum near the factor that made them. The fit finds those
maxima on a small grid around the estimate and around each factor that it tests. The fit also keeps
each other peak where the likelihood-ratio statistic is at most nine times the interval cutoff. Each
kept peak that no earlier grid holds gets a grid of its own. The same grids give the maximum of the
observed counts. The grid step is a quarter of the interval half-width. For a one-sided interval,
the step is at least 0.0125 in log factor. The grid also holds the two ends of the factor range. A
set with no error shots has its maximum at the lower end, which a grid around a larger estimate does
not reach.

On each grid, the fit starts from the best grid point and builds a quadratic from the nine grid
points around it. The quadratic includes the term that couples the two factors. The fit then
computes the likelihood at the vertex of the quadratic. It keeps that value when the quadratic
predicts it within 0.02 and no neighboring grid point is more than 3.84 lower. Otherwise, the fit
climbs from the best point. At each readout factor, it finds the best gate factor, and it then moves
the readout factor. Each climb steps to a better neighbor, or it computes the vertex of a parabola
through its two neighbors and makes its step smaller. A climb stops when both neighbors are within
0.001 of its best point. It also stops when the parabola predicts the likelihood at the vertex
within 0.001 and no neighbor is more than 3.84 lower. Fits with one factor held climb along the
other factor in the same way. Thus each maximum that the fit reports is the likelihood of the model
at the reported factors. No reported maximum is above the true maximum. The fit keeps the highest
result over all grids. A set whose maximum is at a second peak therefore gets that maximum. In an
example with two peaks, a search near the estimate alone gave p = 0.012. The search near both peaks
gives p = 0.0075, below the poor-fit limit of 0.01.

On a narrow ridge of the likelihood, the quadratic on the grid misses the maximum, and the climb
finds it. An example is one qubit with an `x x` circuit of 10^8 shots and a readout circuit of 4000
shots. Only the readout circuit separates the factors, and alone it gives a readout interval of
0.72 to 1.35. Separate parabolas on each axis of the grid put the upper end at 2.56. With the
climb, the interval is 0.70 to 1.35.

### Outcomes the profile rules out

An outcome can have probability 0 at every factor, as when the profile states a readout error of
exactly 0 for a measured qubit. No pair of factors explains a shot on such an outcome, so those
shots leave the likelihood, and the fit uses the other shots. The shots still count in the TVDs,
and they set p to 0, so the fit reads `ruled out`. When the profile rules out every shot, the
counts identify neither factor.

### What the tests show, and what they do not

A CI job simulates 100 runs of the ibm_kingston plan at 4000 shots per circuit, with gate errors
x1.8 and readout errors x1.3. It requires each interval to contain its true factor in 88 to 100
of the runs. The gate interval contains the true gate factor in 93 runs, and the readout
interval contains the true readout factor in 95. To run the job locally, set
`NOISEVAULT_SLOW=1`. Then run `pytest tests/test_compare.py -k each_interval_covers`.

Four faster tests run with the rest of the suite. In the two one-qubit cases above, the probability
must be at least 0.95. The test checks true factor 1 in the 0.00055 case and true factors 0.5, 1, 2
and 5 in the 0.00075 case. The second test uses one qubit with a readout error of 0.1 and a million
shots. Its interval must contain the true factor in 180 to 199 of 200 runs. A third test uses the
`r` gate with two peaks. Its interval must contain factor 1 in at least 93 of 100 runs, both at a
million shots and at 20 million shots. It contains factor 1 in 99 and 100 runs. A fourth test
uses an `r` gate response that is equal at both ends of the range, with a true factor of 5. Its
interval contains 5 in 99 of 100 runs. The CI job also runs the 1000-gate case above. That
interval must contain the true factor in at least 93 of 100 runs, and it contains the true factor
in 96.

Each of these tests draws its counts from the model it fits, and the two-factor test covers one
device at one pair of true factors. The tests show that the intervals cover factors the model can
express. They do not show how the factors behave when the device differs from the model in a way
that no pair of factors captures. On such counts, the goodness-of-fit test is the check, and an
interval describes the best pair of factors, not the device. No counts from a real device have
been compared yet.

## What the unmodeled-error factors absorb

`nv compare` fits one factor on every gate error rate and one on every readout error rate. The
gate factor absorbs any error beyond the calibration that acts like more gate error. Such error
includes crosstalk, leakage, coherent error, idle error beyond T1 and T2, and drift since the
calibration. Some profiles state no durations or no T1 and T2, such as the bundled Quantinuum
profiles. These profiles also put their idle and transport error into the gate factor.

- **One chain represents the device.** `nv compare` fits the factors on the three or four qubits
  that `nv check` picks, a well-calibrated chain. A saved profile applies them to every qubit, and
  other qubits can have more or less excess error. `fit.qubits` in the profile names the
  measured qubits.
- **One gate factor covers every gate.** The check circuits cannot separate excess error on
  one-qubit gates from excess error on two-qubit gates, so one factor scales both. The fitted
  value averages the two kinds by the error each contributes to the check circuits. Two-qubit
  gates carry 63% of the stated gate error in those circuits on the bundled ibm_kingston, 66% on
  ibm_fez and 94% on quantinuum_h2-1. If the excess is mostly in one kind, a circuit with a
  different mix gets too much or too little error.
- **Short circuits, stochastic error.** On IBM Heron devices, each check circuit runs for less
  than 1 µs before measurement. These circuits have no spectator qubits, no parallel layers and
  no mid-circuit measurement. The factors scale stochastic error, so coherent error that builds up
  over many gates on the device stays stochastic in the model. Whether a factor fitted on these
  circuits predicts the logical error rate of a QEC circuit is untested.
- **Some values never scale.** T1, T2, dephasing, preparation error, gate durations and effects
  keep their stated values. An error with no valid power also stays as stated. Examples are a
  readout pair no better than chance and a Pauli channel with a negative Pauli-Lindblad rate.
  `nv compare` and every report name such an error.

## Out of scope

Analog neutral-atom programs (time-dependent Rydberg Hamiltonians) and continuous-variable
photonics do not fit a gate-on-qubit model. Profiles can carry their parameters in
`extensions`, but no export uses them. See
[How technologies map](profile-format.md#how-technologies-map).
