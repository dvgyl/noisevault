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
| `python -m study.summarize results/analysis.json --out results` | `analysis.json` | `summary.json`, `summary.md`: the sign tests of the primary comparisons and the points of the staleness curve |
| `python -m study.archive PARQUET --out results` | The parquet file of the public archive of IBM calibrations | `archive.json`, `archive.md`: how often IBM measures each value again, and how old the values are |

`FOLDER` is a tree of counts files, such as the campaign folder of
`scripts/ibm_campaign.py`. Each counts file `NAME.counts.json` needs the profile file
`NAME.profile.json` in the same folder. The run scripts write both files.

`--rows FOLDER` on `study.derive` and `study.analyze` keeps the result of each counts file. A
later call uses a kept result again when the counts file, the profile file and the source files
of NoiseVault and the study are the same.

`study.analyze` reads older calibrations of IBM devices from IBM's public endpoint. `--offline`
leaves out the models that need them. `study.archive` needs the `hf` extra and the file
`data/train-00000-of-00001.parquet` of the Hugging Face dataset
`phanerozoic/qiskit-calibration-drift`.

## What the outputs hold

The outputs hold derived numbers only: scores, fitted factors, intervals, tests and ages. They
hold no counts, no calibration value, no file name and no job id. The SHA-256 of each counts file
and the fingerprint of each calibration identify the data.

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
