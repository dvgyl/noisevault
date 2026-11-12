# The realism study

The study measures how well a NoiseVault profile forecasts counts from the device that the
profile describes. [PLAN.md](PLAN.md) fixes the question, the models, the scores and the
comparisons. The plan was written before the test data was scored, and each later change has a
date.

## The scripts

Run each script from the repository root, in an environment with NoiseVault installed.

| Script | Input | Output |
| --- | --- | --- |
| `python -m study.derive FOLDER --out results` | Counts files with the profile file beside each one | `fits.json`, `fits.md`, and the realism table `table.json`, `table.md` |
| `python -m study.analyze FOLDER --out results --cache calibrations` | The same folder | `analysis.json`, `analysis.md`: the score of each model and the paired tests of the plan, for each run of the deep circuit set |
| `python -m study.summarize results/analysis.json --out results` | `analysis.json` | `summary.json`, `summary.md`: the estimates over device-weeks of the plan, with the descriptions and the secondary results |
| `python -m study.archive PARQUET --out results` | The parquet file of the public archive of IBM calibrations | `archive.json`, `archive.md`: how often IBM measures each value again, and how old the values are |

`FOLDER` is a tree of counts files, such as the campaign folder of
`scripts/ibm_campaign.py`. Each counts file `NAME.counts.json` needs the profile file
`NAME.profile.json` in the same folder. The run scripts write both files.

`--rows FOLDER` on `study.derive` and `study.analyze` keeps the result of each counts file. A
later call uses a kept result again when the counts file, the profile file and the source files
of NoiseVault and the study are the same.

`study.analyze` reads older calibrations of IBM devices from IBM's public endpoint. `--offline`
leaves out the models that need them. `--samples N` sets the number of stabilizer samples of
each forecast of a Clifford circuit (default 4,000,000). `study.archive` needs the `hf` extra
and the file `data/train-00000-of-00001.parquet` of the Hugging Face dataset
`phanerozoic/qiskit-calibration-drift`.

## What the outputs hold

The outputs hold derived numbers only: scores, fitted factors, intervals, tests and ages. They
hold no counts, no calibration value, no file name and no job id. The SHA-256 of each counts file
and the fingerprint of each calibration identify the data. A failure gives only the class of the
error and of its first cause, such as `CountsError from MissingCalibrationError`.

### The row of a run in `analysis.json`

| Field | Content |
| --- | --- |
| `circuits` | The number of circuits in each group. `all4` holds each circuit whose qubits are all qubits of the `readout` circuit (the chain), and `test4` the test circuits among them. `test_near` and `test_far` hold circuits of the chain only. `wide` holds the 8-qubit `wide_*` circuits. No group holds a `probe` circuit. |
| `stated` | The readout probabilities of 0 in the calibration as stated, `zero_on_chain` (a 0 on a qubit of the chain), and the ruled-out shots and log score of the calibration as stated. |
| `scores`, `paired` | The scores of each model on each group, and the paired tests of the plan. `winner` is the verdict of the log score. A fitted model has no p-value of its excess. |
| `misfit` | For `locus`, `two_factor` and `calibration` on `test_far`: the excess minus the mean excess of counts drawn from the model, per shot, and the part of the excess from outcomes with fewer than 1 expected shot. |
| `probes` | The measured fraction of wrong bits of each chain qubit in `readout` (the baseline), and for each probe circuit the ideal bit string, its measured and forecast probability, and the measured and forecast fraction of wrong bits of each qubit. |
| `wide_chain_overlap` | The number of qubits of the 8-qubit chain that are in the chain of the run. |
| `clifford` | The scores and tests on the Clifford circuits. The tests of the qubit score and of the wrong-bit score hold the sampling variance of both forecasts, and `sampling_ratio` gives its size. |
| `twirl_check` | For each model of the Clifford comparisons and each 8-qubit circuit: the log score from the exact simulator and from the stabilizer forecast. |
| `fits` | The fitted factors and, for `averaged(7)`, the number of different calibrations in the median. |
| `ages` | The age of the calibration time and of the values of C and of each stale calibration. |

`analysis.json` also holds `shot_checks`: for each device-week with one run on each chain kind,
the Pearson chi-square statistic of each circuit that both runs hold, with its degrees of freedom.

### The summary

Only the IBM runs from the week of 2026-11-16 to the week of 2027-01-04 count. For each primary
comparison, `summary.md` gives the estimate over device-weeks, its standard error, the t-test and
the Holm correction, as `study/estimate.py` computes them. A device needs 4 complete
device-weeks to be in an estimate, so the first weeks of the study give no test. The summary
also gives the descriptions and strata of the plan, the secondary comparisons with the same
estimate, the staleness curve, the misfit, the probe table with the growth ratio of each qubit,
the Clifford values of each device-week, the twirl check and the check of independent shots.
The verdict counts of the pilot runs and of the late runs are exploratory.

## The models

`study/models.py` builds each model of the plan as a profile, so `nv score` and each export can
use it. A model with fitted parameters fits only the `train` circuits of a run.
`noisevault.suite.role` gives the role of each circuit.

## The weekly runs

The workflow `.github/workflows/ibm-campaign.yml` runs the deep circuit set each week on IBM
devices. The workflow `.github/workflows/study.yml` then runs the scripts above on the campaign
folder and publishes their outputs as the artifact `study-results`.
[Run the weekly IBM campaign](../docs/recipes.md#run-the-weekly-ibm-campaign) tells how to set
up the runs with your own IBM Quantum account.

## Your own counts

To score your own profile on your own counts, you do not need the study scripts:

```bash
nv score ibm_kingston@2026-11-09 other-profile.json kingston.counts.json
```

[Scoring](../docs/scoring.md) explains the scores and the paired test.
