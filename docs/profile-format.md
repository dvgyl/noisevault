# Profile format 1.0

A profile is one JSON file that records the calibrated noise of one quantum device at one point
in time. It holds the numbers that a simulator needs. These numbers are gate errors and durations,
coherence times, and readout and preparation errors. The profile also states which qubits and gate
pairs exist. It also records where the numbers came from and under what license.

NoiseVault reads a profile with `nv.load(path_or_ref)` and checks one with `nv validate FILE`.
Every export (`to_qiskit`, `to_cirq`, `to_pennylane`, `to_stim`) starts from the same rules,
described on this page. A file may be gzip-compressed (`.json.gz`).

The JSON Schema in [`schema/profile-1.0.json`](schema/profile-1.0.json) describes the structure of
a saved file. The schema does not express the cross-field rules below (one metric per spec,
records that match their definitions, qubit indices in range). The schema lists only the canonical
unit spellings. `nv validate` checks everything. `nv schema` prints the schema.

## Two examples

A hand-written profile of a hypothetical 8-ion device:

```json
{
  "noisevault": "1.0",
  "device": {"name": "demo-ions", "vendor": "example", "technology": "trapped_ion", "num_qubits": 8},
  "connectivity": "all_to_all",
  "gates": {
    "rz": {"virtual": true},
    "r": {"avg_infidelity": 3e-5, "duration_us": 10},
    "zz": {"avg_infidelity": 1e-3, "duration_us": 200, "includes": ["1q_dressing"]}
  },
  "readout": {"p1_given_0": 0.001, "p0_given_1": 0.003},
  "idle": {"t2_s": 1}
}
```

Every gate of a kind has the same error, so the device-wide definitions in `gates` are complete.
`duration_us` and `t2_s` are hand-writing aliases. NoiseVault stores them as
`duration_ns: 10000.0` and `t2_us: 1000000.0`. There is no T1, so gates get pure dephasing and
no amplitude damping. The profile id is `example_demo-ions`.

A hypothetical 5-qubit superconducting device with per-qubit and per-gate records:

```json
{
  "noisevault": "1.0",
  "device": {"name": "demo5", "vendor": "example", "technology": "superconducting", "num_qubits": 5,
             "calibrated_at": "2026-09-01T08:00:00Z"},
  "connectivity": {"directed": true, "edges": [[1, 0], [1, 2], [3, 2], [3, 4]]},
  "gates": {
    "rz": {"virtual": true},
    "sx": {"avg_infidelity": 2.5e-4, "duration_ns": 32, "method": "rb", "statistic": "median"},
    "x": {"avg_infidelity": 2.5e-4, "duration_ns": 32, "method": "rb", "statistic": "median"},
    "ecr": {"avg_infidelity": 8e-3, "duration_ns": 500, "method": "rb", "statistic": "median"}
  },
  "readout": {"error": 0.02, "duration_ns": 1200},
  "prep": {"error": 0.001},
  "idle": {"t1_us": 150, "t2_us": 110, "t2_kind": "echo"},
  "qubits": [
    {"index": 3, "t1_us": 40, "t2_us": 35, "readout": {"p1_given_0": 0.03, "p0_given_1": 0.06}}
  ],
  "calibrations": [
    {"gate": "ecr", "qubits": [1, 0], "avg_infidelity": 7e-3, "statistic": "individual"},
    {"gate": "ecr", "qubits": [3, 2], "avg_infidelity": 1.2e-2, "duration_ns": 660},
    {"gate": "ecr", "qubits": [3, 4], "disabled": true},
    {"gate": "sx", "qubits": [4], "pauli": [1e-4, 1e-4, 3e-4]}
  ],
  "provenance": {"data_kind": "hypothetical", "source_kind": "hand_written"}
}
```

The definitions in `gates` hold device-wide defaults, here medians. A record in `calibrations`
overrides them on specific qubits. The table shows what each lookup resolves to under the rules
below.

| Lookup | Result | Why |
| --- | --- | --- |
| `ecr` on (1, 0) | 7e-3, 500 ns | The record overrides the metric and keeps the default duration. |
| `ecr` on (1, 2) | 8e-3, 500 ns | No record. The directed edge 1 to 2 allows the default. |
| `ecr` on (0, 1) | unavailable | `ecr` is not symmetric and the edge goes only from 1 to 0. |
| `ecr` on (3, 2) | 1.2e-2, 660 ns | The record overrides both fields. |
| `ecr` on (3, 4) | disabled | Using it raises `DisabledGateError`. Qiskit's transpiler avoids it. |
| `sx` on 4 | Pauli channel, average infidelity 3.3e-4 | A `pauli` spec is the whole channel. |
| `sx` on 3 | 2.5e-4, 32 ns | The default. |
| qubit 3 | T1 40 us, T2 35 us, readout (0.03, 0.06) | The qubit record overrides the defaults. |
| qubit 0 | T1 150 us, T2 110 us, readout (0.02, 0.02) | The defaults. |

## Top-level fields

| Field | Required | Content |
| --- | --- | --- |
| `noisevault` | yes | The format version, `"1.0"`. |
| `device` | yes | Name, vendor, technology, qubit count, processor and calibration time. |
| `connectivity` | yes | `"all_to_all"`, or the coupling edges. |
| `gates` | yes | Gate definitions with device-wide defaults, keyed by gate name. |
| `readout` | no | Default readout error of every qubit. |
| `prep` | no | Default preparation (reset) error of every qubit. |
| `idle` | no | Default T1, T2 and extra dephasing of every qubit. |
| `qubits` | no | Per-qubit records that override `readout`, `prep` and `idle`. |
| `calibrations` | no | Per-gate records that override a definition on specific qubits. |
| `effects` | no | Physics outside the gate-and-readout model, such as leakage and atom loss. |
| `unmodeled_error` | no | Factors on the profile's error rates, for error the calibration leaves out. |
| `benchmarks` | no | Free-form benchmark values, such as layer fidelity. |
| `provenance` | no | Source, license, attribution and retrieval details. |
| `extensions` | no | Free-form vendor data that is not part of the physics. |

Unknown keys are errors everywhere except inside `benchmarks`, `extensions` and
`provenance.extra`. Those three fields hold JSON data only. The allowed values are objects with
string keys, arrays, strings, finite numbers, `true`, `false` and `null`. From Python, a key that
is not a string, a set, or any other object is an error rather than a value converted on save.
Values are strict. `"1"`, `3.0` for an integer field, or `1` for a boolean are errors, not
coercions. The time fields `calibrated_at`, `retrieved_at` and `run_at` take an ISO 8601 time with
a timezone, such as `2026-09-30T08:00:00Z`. In these fields, a number or a string of digits is an
error, not a Unix timestamp.

A key that occurs two times in one JSON object is an error, and the message names that key. The
second value does not replace the first.

## Device

| Field | Content |
| --- | --- |
| `name` | Device name, such as `ibm_fez` or `H1-1`. No `@`, `/` or `\`, and no padding. |
| `vendor` | Vendor, such as `ibm`, `quantinuum` or `google`. Optional. |
| `technology` | One of `superconducting`, `trapped_ion`, `neutral_atom`, `spin`, `photonic`, `other`. |
| `num_qubits` | Number of qubits, at least 1. The format numbers the qubits `0` to `num_qubits - 1`. |
| `processor` | Processor family, such as `Heron r2`. Optional. |
| `calibrated_at` | The time of the calibration, an ISO 8601 time with a timezone. NoiseVault stores the time in UTC. Optional. |

### Profile ids and refs

The profile id is `name` when `name` already starts with `vendor_`, else `vendor_name`.
NoiseVault lowercases the id and writes spaces as `-`. Example ids are `ibm_fez`, `google_weber`
and `quantinuum_h1-1`. With no vendor, the id is the name. An id may use only `a-z`, `0-9`, `_`,
`.` and `-`. An id may not end in `.json` or `.gz`, because a ref with that suffix names a file.

`nv.load` and every CLI command take a ref:

| Ref | Meaning |
| --- | --- |
| `path/to/file.json` | A file. Any ref with a path separator, a `.json` or `.gz` suffix, or an existing file name is a file. |
| `ibm_fez` | The newest profile with that id. |
| `ibm_fez@2025-02-26` | The profile calibrated on that UTC date. |
| `ibm_fez@2025-02-26T20:16:25Z` | The profile calibrated at that time. |

A ref that matches two profiles raises `AmbiguousRef` and lists both. `expect=` pins a
fingerprint (see [Fingerprint and artifact hash](#fingerprint-and-artifact-hash)).

## Connectivity

`"all_to_all"` allows every pair. Otherwise give the edges:

```json
{"edges": [[0, 1], [1, 2]], "directed": false}
```

Undirected edges allow both operand orders. NoiseVault stores undirected edges as `[low, high]`,
sorted. Directed edges (`"directed": true`) allow only the listed order. Self-loops, repeated
edges and indices outside `0..num_qubits-1` are errors.

## Gates

`gates` maps each gate name the device offers to its definition. Use the canonical names of the
gate registry so every framework recognizes the gate:

- `id`, `x`, `y`, `z`, `h`, `s`, `sdg`, `t`, `tdg`, `sx`, `sxdg`, `rx`, `ry`, `rz`, `p`, `u1`,
  `u`, `r`
- `cx`, `cy`, `cz`, `ecr`, `swap`, `cxswap`, `swapcx`, `czswap`, `iswap`, `sqrt_iswap`, `rzz`,
  `rxx`, `ryy`, `zz`, `ms`
- `cswap`, `ccx`
- `measure`, `reset`, `delay`

`zz` is exp(-i pi/4 ZZ), the native entangler of Quantinuum devices. `ms` is IonQ's Molmer-Sorensen
gate with two phases. `r(theta, phi)` is a rotation about an axis in the XY plane. `cxswap`,
`swapcx` and `czswap` are Stim's `CXSWAP`, `SWAPCX` and `CZSWAP`. `cxswap` applies `cx` and then
`swap`, and `swapcx` applies them in the other order. The format allows other names (Google's
`sycamore`, for example), but their definitions must state the arity with `qubits`. The Qiskit
export reports such a gate as omitted. The other exports apply it only when a circuit uses a gate
that they map to that name.

A definition and a calibration record share these fields:

| Field | Content |
| --- | --- |
| `avg_infidelity` | Average gate infidelity, 1 - F_avg. |
| `process_infidelity` | Process (entanglement) infidelity. |
| `depolarizing_param` | The lambda of Qiskit's `depolarizing_error`. |
| `pauli` | Pauli error probabilities, 3 numbers for 1 qubit and 15 for 2 qubits. |
| `duration_ns` | Gate duration in nanoseconds. |
| `virtual` | `true` for a gate done in software, such as a frame change. No error, no duration. |
| `disabled` | `true` when the gate must not be used. |
| `method` | How the error was measured. One of `rb`, `irb`, `srb`, `xeb`, `gst`, `layer`, `model`, `vendor`. |
| `measured` | `isolated` or `simultaneous`. |
| `statistic` | `individual`, `median` or `mean`. |
| `scope` | `gate` or `cycle`. A cycle error includes the surrounding layer. |
| `includes` | What the number already contains. The values are `1q_dressing` (single-qubit gate error), `leakage` and `spam` (state preparation and measurement error). |
| `stderr` | Standard error of the metric. |
| `assumption` | How an importer read a vendor number, in words. `nv show` prints it. |

A definition also has `qubits` and `symmetric`. `qubits` is the arity, and a definition for a
name outside the registry must have `qubits`. `symmetric` states whether a calibration record on
one operand order holds for the other.

### Metric keys name the metric

A spec holds at most one of `avg_infidelity`, `process_infidelity`, `depolarizing_param` and
`pauli`. The key says what the number means, so NoiseVault does not guess. For an n-qubit gate
with d = 2^n, NoiseVault converts with:

- process infidelity = avg_infidelity * (d + 1) / d
- avg_infidelity = depolarizing_param * (d - 1) / d
- the sum of the `pauli` entries is the process infidelity

NoiseVault checks these bounds:

- avg_infidelity is at most d / (d + 1)
- process infidelity is at most 1
- depolarizing_param is at most d^2 / (d^2 - 1)
- the `pauli` entries are non-negative, and their sum is at most 1

Two-qubit `pauli` vectors use Stim's order `IX, IY, IZ, XI, XX, ..., ZZ`, and the
first letter acts on `qubits[0]`. [Conventions](conventions.md) derives these relations.

A number is the total error of the operation. When the profile states T1, T2 and the duration, an
export adds relaxation for the duration. The export then solves for the depolarizing part so the
composed channel has the stated error. A `pauli` spec is the whole channel, and an export adds no
relaxation to it.

The qualifiers (`method` through `assumption`) do not change the noise that an export applies.
They record what the number is, so a reader can judge it. An export's report lists a gate that
the export used under "approximated" if the gate's number has any of these qualifiers:

- `scope: cycle`
- an `includes` list
- a `statistic` of `median` or `mean`
- an `assumption`

`nv validate` warns about calibration records with `scope: cycle`.

### Gate states

Every lookup of a gate on specific qubits resolves to one of four states:

| State | When | What an export does |
| --- | --- | --- |
| `ideal` | `virtual: true` | No noise. Z-family gates (`z`, `s`, `t`, `p`, ...) are also ideal when `rz` is virtual, unless the Z-family gate has its own metric. |
| `calibrated` | A metric is present after merging. | Adds the gate's channels. |
| `uncalibrated` | A native with no metric anywhere. | Uses typical noise or raises (see [Conventions](conventions.md#gates-the-profile-does-not-calibrate)). |
| `disabled` | `disabled: true` after merging. | Raises `DisabledGateError`. |

`virtual: true` together with a metric or a positive duration is an error.

## Calibration records

A record in `calibrations` names a defined gate and the qubits it acts on, and carries any of
the shared fields:

```json
{"gate": "cz", "qubits": [2, 3], "avg_infidelity": 0.02}
```

Merge rules:

- Record fields override the definition field by field. A record with only `duration_ns` keeps
  the definition's metric.
- A record replaces the metric as a whole. A record with `pauli` drops the definition's
  `avg_infidelity`, so a spec never has two metrics.
- `disabled: true` ends the lookup. The gate is unusable on those qubits.

These cases are errors:

- two records for the same gate and qubits
- a record for a gate missing from `gates`
- a record whose qubit count differs from the gate's arity
- repeated qubits within one record
- qubit indices outside the device

### Direction and symmetry

The registry sets the default of `symmetric`. `cz`, `rzz`, `rxx`, `ryy`, `zz`, `ms`, `iswap`,
`swap` and `sqrt_iswap` are symmetric. `cx`, `cy`, `ecr` and unknown names are not symmetric. A
definition may override the default.

The lookup of gate G on qubits (a, b) takes the first of:

1. A record for G on (a, b).
2. If G is symmetric, a record for G on (b, a). NoiseVault swaps the `pauli` labels of that
   record to match.
3. G's definition, if connectivity allows (a, b). Undirected edges allow both orders, directed
   edges allow only the listed order, and `all_to_all` allows every pair.
4. Otherwise G is unavailable on (a, b).

## Qubits, readout, preparation and idle

Device-wide defaults:

```json
"readout": {"p1_given_0": 0.008, "p0_given_1": 0.015, "duration_ns": 1560},
"prep": {"error": 1e-4},
"idle": {"t1_us": 150, "t2_us": 110, "t2_kind": "echo"}
```

- `readout` holds either both `p1_given_0` (P(read 1 | prepared 0)) and `p0_given_1`, or one
  symmetric `error`. `duration_ns` is optional.
- `prep.error` is the probability that a reset leaves the qubit in |1>.
- `idle` holds `t1_us`, `t2_us`, `t2_kind` (`echo`, `ramsey` or `cpmg`) and
  `dephasing_rate_per_s`, an extra Z error with probability rate x time.

A record in `qubits` overrides these defaults for one qubit:

| Field | Content |
| --- | --- |
| `index` | The qubit, `0..num_qubits-1`. Each index appears at most once. |
| `t1_us`, `t2_us`, `t2_kind`, `dephasing_rate_per_s` | Override `idle` field by field. |
| `readout` | Replaces the default readout as a whole. |
| `prep` | Replaces the default preparation error as a whole. |
| `label` | The vendor's name for the qubit, such as `q(0,5)`. |
| `coords` | Position on the device. Cirq maps `GridQubit(r, c)` to the qubit at `[r, c]`. |
| `disabled` | `true` when the qubit must not be used. A layout onto it raises `LayoutError`. |

A missing readout or preparation error stays unknown. Exports apply no noise for it and list it
under "unknown" in the report. It is never silently zero. A T2 above 2 T1 is legal, but exports
clamp it to 2 T1 and report the clamp.

### Units and aliases

Saved files use `duration_ns`, `t1_us` and `t2_us`. A hand-written file may use any of these
spellings, and NoiseVault converts them:

| Field | Accepted spellings |
| --- | --- |
| duration | `duration_ns`, `duration_us`, `duration_ms`, `duration_s` |
| T1 | `t1_ns`, `t1_us`, `t1_ms`, `t1_s` |
| T2 | `t2_ns`, `t2_us`, `t2_ms`, `t2_s` |

Two spellings of one field, a non-finite value, a negative duration and a T1 or T2 of zero or
less are errors.

## Effects

`effects` records physics that the gate-and-readout model does not cover. Format 1.0 allows only
these `type` values:

| `type` | Meaning |
| --- | --- |
| `leakage` | Population leaves the qubit subspace. |
| `atom_loss` | A neutral atom leaves its trap. |
| `erasure` | An error at a known location, such as a lost photon in dual-rail encoding. |
| `crosstalk_measurement` | Measuring one qubit disturbs others. |
| `crosstalk_zz` | An always-on ZZ coupling between qubits. |
| `coherent_overrotation` | A systematic unitary error. |

Each effect names exactly one of `gate` (a defined gate) or `on` (`readout` or `idle`), and may
carry `qubits`, `prob`, `rate_per_s`, `strength_hz`, `angle_rad` and `heralded`.

```json
{"type": "atom_loss", "on": "readout", "prob": 0.005}
```

### Where an effect acts

The target of an effect sets where the effect acts and which number it uses:

| Target | Where the effect acts | Number |
| --- | --- | --- |
| `"gate": name` | After each application of the gate, after the gate's own noise. | `prob`, for each qubit of the gate and each application. |
| `"on": "readout"` | Directly before each measurement of a qubit. | `prob`, for each measured qubit. |
| `"on": "idle"` | After each idle time that an export models, after the relaxation of that time. | `rate_per_s`. An idle time t gives the probability 1 - exp(-`rate_per_s` t). |

Without `qubits`, the effect acts on every qubit. On a gate, `qubits` names one locus, in any
order. On `readout` or `idle`, `qubits` lists the qubits. An effect with `qubits` replaces an
effect of the same `type` and target without `qubits` on those qubits. When several effects
still match one place, all of them act, each with its own probability.

A gate without its own noise takes the noise of the typical native gate (see `unknown_gates` in
[Frameworks](frameworks.md)). Such a gate also takes the gate effects of that native gate, unless
an effect names the gate itself. For example, on a device whose 2-qubit native gate is `cz`, a
`cx` in the circuit takes the noise of `cz` and each effect on `cz`.

`"heralded": true` means that the device reports each event, for example when imaging finds an
empty trap. Without `"heralded": true`, the event is silent.

### How the Stim and Cirq exports model leakage, atom loss and erasure

The Stim and Cirq exports model `leakage`, `atom_loss` and `erasure` with one channel, an
erasure. An erasure replaces the state of the qubit with the maximally mixed state I/2. This
is the same as a random Pauli I, X, Y or Z, each with probability 1/4. For each effect, the
exports add this channel with probability p, where p comes from `prob` or `rate_per_s`.

| Export | Heralded | Not heralded |
| --- | --- | --- |
| Stim | `HERALDED_ERASE(p)` on each qubit. Each target adds one measurement record. The record is 1 when the erasure occurs. | `DEPOLARIZE1(3p/4)` |
| Cirq | A `noisevault.frameworks.cirq.HeraldedErasure` gate with a measurement key. The record is 1 when the erasure occurs. | `cirq.depolarize(3p/4)` |

The Stim export changes every `rec[-k]` target after an added herald, so that detectors,
observables and classically controlled gates read the same measurements as before. Herald
records stay in the samples of `compile_sampler()`. The Cirq key of a herald is
`herald_<type>_m<moment>_q<qubit>`, for example `herald_erasure_m3_q5`, where the qubit is the
device qubit. When the circuit or another herald already uses the key, the herald's key ends in
`_1`, or else `_2`, and so on.

The channel is exact for `erasure`. For the other two types, the report states the
approximation:

- **`atom_loss`.** A lost atom comes back at once in a random state. On the device, the trap
  stays empty, later gates on the site do nothing, and readout of the site gives a fixed result.
- **`leakage`.** Leaked population comes back at once in a random state. On the device, it stays
  outside the qubit until a reset or until it decays, gates do not act on it, and it can disturb
  neighboring qubits.

### What the report says

`allow` states the least faithful treatment that you accept. The order is `omit`, `approximate`,
`exact`, and the default is `omit`. Every export models an effect as well as it can, also with
`allow` set to `omit`. When the export's treatment is less faithful than `allow`, the export
raises `UnsupportedEffect`.

| Effect | Stim | Cirq | Qiskit, PennyLane, reference simulator |
| --- | --- | --- | --- |
| `erasure` on a gate or on `readout` | exact | exact | omitted |
| `erasure` on `idle` | approximated, see below | exact, during `cirq.WaitGate` | omitted |
| `atom_loss`, `leakage` | approximated | approximated | omitted |
| `crosstalk_measurement`, `crosstalk_zz`, `coherent_overrotation` | omitted | omitted | omitted |

The Stim export applies idle effects only to qubits idle in a `TICK` layer with `tick_ns`, as
for idle relaxation. The report also omits an effect in these cases:

- The effect has no `prob` (on a gate or on `readout`) or no `rate_per_s` (on `idle`).
- The effect names the gate `measure`, `reset` or `delay`. Use `"on": "readout"` or
  `"on": "idle"` instead.
- The effect is on `readout`, and the export adds no readout error (`readout="none"` in Stim,
  `readout=False` in Cirq).
- The effect is on `idle`, and the Stim export has no `tick_ns`.

In the Stim and Cirq exports, a `leakage` effect on a gate adds nothing on a locus whose gate
record lists `leakage` in `includes`. The record is the calibration record of that locus, or
else the definition. The stated error of the gate already contains the leakage, as depolarizing
noise. The report lists the effect as approximated, also when the effect has no `prob`. The
bundled Quantinuum profiles are an example. A record without a stated error contains no
leakage, so the effect acts there.

## Unmodeled error

`unmodeled_error` states how far the device's errors exceed the calibration, as factors on the
profile's own error rates. A profile without it gives the noise the calibration states.

```json
"unmodeled_error": {"gates": {"factor": 2.3}}
```

This hand-written factor makes every calibrated gate err about 2.3 times as often. A fitted
block also carries the 95% interval of each factor and a record of the counts that the fit used:

```json
"unmodeled_error": {
  "gates": {"factor": 1.84, "low": 1.54, "high": 2.12},
  "readout": {"factor": 1.58, "low": 1.32, "high": 1.84},
  "fit": {"counts": "sha256:3fa1c2d4e5b6...", "source": "hardware", "qubits": [148, 149, 150, 151],
          "run_at": "2026-04-16T09:30:02Z", "calibration": "609c845ed934...",
          "p_value": 0.41, "impossible_shots": 0}
}
```

This example shortens the `counts` and `calibration` values. A file holds all 64 hex digits of
each.

| Field | Content |
| --- | --- |
| `gates` | The factor on every calibrated gate error, of any arity, `pauli` specs included. |
| `readout` | The factor on `p1_given_0` and `p0_given_1` of every qubit. The profile must state a readout error. |
| `fit` | The counts that the fit used. A hand-written factor has no `fit`. |

The block needs `gates`, `readout` or both. NoiseVault never scales T1, T2,
`dephasing_rate_per_s`, preparation error, durations or effects.

### Error factors

| Field | Content |
| --- | --- |
| `factor` | The factor, 0 or more. 1 keeps the stated errors. |
| `low`, `high` | The 95% interval of a fitted factor. |
| `bound` | `"lower"` or `"upper"` when the interval reached that end of the range the fit searched. Only the other side is a confidence limit. |

An interval has one of three shapes:

| Shape | Fields | Printed as |
| --- | --- | --- |
| Two-sided | `low` and `high` | `x1.84 (95% interval 1.54 to 2.12)` |
| Open below | `high`, and `bound` set to `"lower"` | `x0.096 (95% interval, at most 0.311)` |
| Open above | `low`, and `bound` set to `"upper"` | `x20 (95% interval, at least 11.2)` |

Each value prints with three significant digits, or with more when three would print two
different values alike, as in `x1.008 (95% interval 1.001 to 1.015)`. `nv compare -o` saves each
value with six significant digits, or with more when six would save two different values alike.

The factor lies inside its interval. Every factor in a block with `fit` has an interval, and a
factor in a block without `fit` has none.

### How a factor scales an error

A factor s raises each error channel to the power s. Factor 1 keeps the stated error, and
factor 0 removes it. Powers compose, so scaling by 1.5 and then by 2 equals scaling by 3. For
small errors, x2 means about twice the error. Multiplying a large error by the factor could leave
the physical range, so NoiseVault takes the power instead.

- A `pauli` spec scales through its Pauli fidelities. Each fidelity f becomes f^s. NoiseVault
  reads any other metric as a depolarizing channel, which scales the same way.
- A readout pair scales through its confusion matrix M, which becomes M^s. The ratio of
  `p1_given_0` to `p0_given_1` stays the same.

A few errors have no valid power at some factors. NoiseVault leaves them as stated at every
factor, and `nv show` and every export's report name them:

- A gate error at or past full depolarization.
- A `pauli` spec with a negative Pauli-Lindblad rate.
- A readout pair whose `p1_given_0` and `p0_given_1` sum to 1 or more. Such a pair is no better
  than chance.

A scaled gate error below what the gate's relaxation alone causes keeps the relaxation, as a
stated error does (see [Channel construction](conventions.md#channel-construction)).

### The fit record

| Field | Content |
| --- | --- |
| `counts` | The SHA-256 of the [counts file](counts-format.md#sha-256-and-canonical-form), as `sha256:` and 64 hex digits. |
| `source` | `hardware`, or `simulated` for counts drawn from a profile. |
| `qubits` | The measured qubits, in the order the circuits first use them. |
| `run_at` | When the device started the first circuit, an ISO 8601 time with a timezone. NoiseVault stores the time in UTC. |
| `calibration` | The fingerprint of this profile without `unmodeled_error`, as 64 hex digits. |
| `p_value` | The p-value of the goodness-of-fit test, from 0 to 1. Below 0.01, every printout calls the fit poor. `null` when the fit is not testable (see [The goodness-of-fit test](limitations.md#the-goodness-of-fit-test)). |
| `impossible_shots` | Shots on outcomes the profile gives probability 0. When `impossible_shots` is above 0, `p_value` is 0. |

The fit record requires every field, `p_value` included. The qubits are distinct and inside the
device.

`calibration` binds the factors to the calibration that the fit used. A change to any physics
outside `unmodeled_error`, such as a gate error or a T1, fails validation with "Drop
unmodeled_error or refit with nv compare". Provenance and extensions can change.

### Fingerprint, display and compatibility

`unmodeled_error` is part of the fingerprint, so one fingerprint pins a calibration and its
factors. `profile.uncorrected()` returns the profile without the block, and the fingerprint of
that profile is `fit.calibration`. `nv cite` names the factors and the counts.

`nv show` and `profile.summary()` print the calibration as stated, then the factors on a row of
their own. Every export applies the factors, and its report states them in one line.

A NoiseVault install that does not know this field refuses a profile that sets it, because format
1.0 forbids unknown keys.

## Benchmarks and extensions

`benchmarks` holds named benchmark values as free-form JSON, such as
`{"eplg_100": {"value": 3.2e-3, "convention": "avg_infidelity"}}`. Exports do not read it. It is
part of the fingerprint.

`extensions` holds vendor data that is not physics NoiseVault uses, such as Google's fSim error
angles. The fingerprint does not include `extensions`. Data for out-of-scope technologies goes
here (see [How technologies map](#how-technologies-map)).

`benchmarks`, `extensions` and `provenance.extra` hold JSON nested at most 64 levels deep. The
field's own object is level 1, so `"extensions": {"fsim": {"theta": 0.1}}` is two levels deep. A
profile with deeper data is invalid, and `nv validate` names the field.

## Provenance

| Field | Content |
| --- | --- |
| `data_kind` | `measured`, `vendor_model`, `spec_sheet`, `hypothetical` or `unknown`. |
| `source_kind` | `package_snapshot`, `public_api`, `account_api`, `user_file`, `published_data`, `vendor_sample`, `hand_written`, `derived` or `other`. |
| `source` | The source in words, such as `qiskit-ibm-runtime 0.49.0 FakeFez`. |
| `source_url` | Where the data came from. |
| `license` | The data's license, such as `Apache-2.0`. |
| `attribution` | Who to credit. `citation()` uses it. |
| `redistributable` | `yes`, `no` or `unknown`. Only a profile with `yes` can be a bundled profile. |
| `retrieved_at` | When NoiseVault fetched the data. |
| `source_hash` | `sha256:<hex>` of the raw source bytes. |
| `tool` | The NoiseVault version that wrote the profile. |
| `derived_from` | The profile this one was derived from. |
| `notes` | Free-text notes, one per decision an importer made. |
| `extra` | Free-form source details, such as a commit hash. |

## Fingerprint and artifact hash

A profile has two hashes, both SHA-256 over canonical JSON (sorted keys, no whitespace, floats
as Python `repr`):

- The **fingerprint** covers the physics, which is everything except `provenance` and
  `extensions`. Two profiles with the same fingerprint produce the same noise models under one
  NoiseVault version. `nv:` plus the first 12 hex digits is the short form, such as
  `nv:06404cefa54f`.
- The **artifact hash** covers the whole profile, provenance included.

Canonical form makes the fingerprint independent of how a file is written. NoiseVault sorts the
records and the edges. NoiseVault also drops a `symmetric` or `qubits` value that only restates
the registry default. Re-ordering `calibrations` does not change the fingerprint. Editing
`provenance.notes` changes the artifact hash but not the fingerprint.

Pin a fingerprint when you load, and a different profile fails with an error:

```python
import noisevault as nv

fez = nv.load("ibm_fez", expect="nv:06404cefa54f")
print(fez.fingerprint)
```

`expect` takes the 12-digit short form or the full 64-digit hex. A mismatch raises
`FingerprintMismatch`. A paper should cite the full fingerprint. `nv cite` prints it.

`profile.save(path)` writes canonical JSON. A `.gz` suffix writes gzip with a zero timestamp,
so the same profile always produces the same bytes.

## Files in format 0.1

NoiseVault 0.1 files (`"schema_version": "0.1"`) still load. NoiseVault upgrades them in memory
and emits a `MigrationWarning`. Save the profile to keep the 1.0 form.

## How technologies map

The format describes gates on qubits with Markovian errors. Each technology fits it this way:

- **Superconducting.** Natives such as `sx`, `x`, a virtual `rz`, and `cz`, `ecr` or `cx`.
  Coupling edges, directed for `ecr` and `cx` when the device calibrates only one direction.
  Per-qubit T1, T2 and asymmetric readout. Google's `sqrt_iswap` is a registry gate, and the
  definition of `sycamore` has `"qubits": 2`.
- **Trapped ion.** `"connectivity": "all_to_all"`, natives such as `r`, a virtual `rz`, and `zz`
  or `ms`. Vendors often publish device-wide medians or means, and `statistic` records which one.
  T1 is usually absent, and T2 is long. Leakage goes in `effects`.
- **Neutral atom in gate mode.** Natives such as `rz`, `r` and `cz`, with `all_to_all` or the
  edges within the interaction radius. Atom loss goes in `effects` as `atom_loss`.
- **Spin qubits.** The mapping is like the superconducting one, with coupling edges, per-qubit T1
  and T2, and `dephasing_rate_per_s` for extra dephasing.
- **Photonic dual-rail.** Each qubit is a pair of modes. Gate errors map to the metrics above.
  Photon loss is an `erasure` effect with `"heralded": true`.

Out of scope, and why:

- **Analog neutral-atom computing** (programs as time-dependent Rydberg Hamiltonians). There is
  no gate set, so there are no per-gate errors to record. To pin the vendor's parameters next to
  a gate-mode profile, keep them in `extensions`.
- **Continuous-variable photonics** (squeezed states, homodyne measurement). The modes are not
  qubits, so qubit channels do not describe them. The same `extensions` rule applies.
