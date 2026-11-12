# NoiseVault for scripts and AI agents

This page is for a program that calls NoiseVault: a shell script, a CI job or an AI agent. It
gives the rules that each command follows, and the JSON output of each command.

## Rules of each command

- **The result goes to standard output.** Progress, warnings, hints and errors go to standard
  error. A program can read standard output alone.
- **`--json` gives the result as one JSON value** on standard output, with no color and no
  line wrapping. The text output is for people, and its layout can change between versions.
- **No command asks a question.** A command never waits for input.
- **The exit status tells the outcome:**

  | Status | Meaning |
  | --- | --- |
  | 0 | The command did its work. |
  | 1 | An error, or a check that did not pass (`nv check`, `nv validate --strict`). |
  | 2 | A wrong command line, such as a missing argument. |

- **An error is one line that starts with `error:`** on standard error. A second line that
  starts with `hint:` says what to do, when NoiseVault knows it.
- **A profile has a fingerprint.** `nv:` and 12 hexadecimal digits name one calibration with its
  values. Record the fingerprint of each profile that a result used.
- **Only `nv pull` uses the network.** Each other command works offline.

```text
$ nv show no_such_device --json
error: no profile with id 'no_such_device'
hint: run nv list to see every profile you can load offline
$ echo $?
1
```

## The commands and their JSON output

| Command | What it does | Top-level JSON keys |
| --- | --- | --- |
| `nv list --json` | Lists the profiles that load offline | A list of objects: `ref`, `id`, `date`, `calibrated_at`, `technology`, `vendor`, `num_qubits`, `processor`, `source_kind`, `data_kind`, `license`, `redistributable`, `location`, `fingerprint` |
| `nv show REF --json` | Describes one profile | `ref`, `id`, `calibrated_at`, `vendor`, `processor`, `technology`, `num_qubits`, `connectivity`, `natives`, `median_t1_us`, `median_t2_us`, `median_readout_error`, `median_p1_given_0`, `median_p0_given_1`, `zero_readout_qubits`, `disabled_qubits`, `value_ages`, `provenance`, `fingerprint`, `short_fingerprint`, `assumptions`, `notes`, `effects`, `unmodeled_error`, `unmodeled_note` |
| `nv diff A B --json` | Compares two profiles | `before`, `after`, `before_fingerprint`, `after_fingerprint`, `identical`, `time_delta_seconds`, `medians`, `qubits`, `pairs`, `newly_disabled`, `reenabled`, `pairs_added`, `pairs_removed`, `qubits_added`, `qubits_removed`, `warnings` |
| `nv check REF --json` | Checks each framework export against the reference | `profile_id`, `fingerprint`, `passed`, `layout`, `shots`, `seed`, `circuits`, `frameworks`, `skipped`, `failed`, `note` |
| `nv compare REF COUNTS --json` | Fits a gate factor and a readout factor to counts | `profile_id`, `fingerprint`, `calibration`, `counts`, `source`, `backend`, `run_at`, `calibration_age_hours`, `qubits`, `circuits`, `gates`, `readout`, `deviance`, `dof`, `dispersion`, `p_value`, `resamples`, `impossible_shots`, `ruled_out`, `notes`, `note` |
| `nv score REF... COUNTS --json` | Scores profiles as forecasts of counts, with no fit | `counts`, `source`, `backend`, `run_at`, `qubits`, `models`, `paired`, `floor`, `zero_readout_qubits`, `probes` |
| `nv schema` | Prints the JSON Schema of the profile format | The schema itself |

`nv pull`, `nv cite`, `nv validate` and `nv doctor` print text. [Profile format](profile-format.md)
and [Counts format](counts-format.md) define the two file formats, and [Scoring](scoring.md)
defines each number of `nv score`.

A value that does not exist is `null`. A number that is not finite is also `null`: the log score
of a profile that is ruled out is `null`, and its `ruled_out` is above 0.

`nv score` does not score each circuit of a counts file. Each entry of `models` has `left_out`,
a list of the circuits that the score did not use, each with `name` and `reason`. `probes` holds
the [probe report](scoring.md#probe-circuits-and-what-a-score-leaves-out) with the forecast of
the best profile. It is `null` when the counts hold no probe circuit.

## A check in a script

This script fails when a profile does not describe the counts of a device:

```bash
nv score ibm_kingston@2026-04-15 run.counts.json --json > score.json || exit 1
python - <<'EOF'
import json, sys

best = json.load(open("score.json"))["models"][0]
ruled_out = best["ruled_out"] > 0
beyond_noise = not ruled_out and best["excess"] > best["excess_noise_95"]
print(best["ref"], "ruled out" if ruled_out else f"excess {best['excess']:.5f}")
sys.exit(1 if ruled_out or beyond_noise else 0)
EOF
```

`models` is in the order of the score, best first. `excess_noise_95` is the excess that shot
noise alone gives in 95% of the cases, if the profile is right.

## The same in Python

Each command has a function with the same name and the same result:

```python
import noisevault as nv
from noisevault.counts import plan, simulate

profile = nv.load("ibm_kingston@2026-04-15", expect="nv:609c845ed934")
counts = simulate(profile, plan(profile), shots=2000, seed=1)

forecast = profile.score(counts)
print(forecast.ruled_out, forecast.to_dict()["circuits"][0]["name"])
# 0 ghz_chain
```

`expect=` makes `nv.load` fail when the fingerprint is not the one that you recorded. A result
object has `to_dict()` with the data of `--json`, and `summary()` with the text output.
