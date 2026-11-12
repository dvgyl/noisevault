"""Score the models of the realism study on each run of the deep circuit set.

    python -m study.analyze campaign --out results --cache pulls

The script reads each counts file below the folder that holds a run of the deep circuit set
(a file with the ``readout_1`` circuit or ``deep_*`` circuits), with the profile file beside it.
For each run, it builds the models of ``PLAN.md``, scores each model on the ``test_near``,
``test_far`` and all circuits, and runs the paired test of each primary comparison. It writes
``analysis.json`` and ``analysis.md`` to the output folder.

The exact simulator scores only the circuits of 10 qubits or fewer, so each group holds only
such circuits. The group ``all4`` is each circuit on the qubits of the chain of the run, and
``test4`` is each test circuit among them (see :func:`groups`). No group holds a ``probe``
circuit. The block ``probes`` gives the probe circuits alone. The block ``clifford`` of a row
scores the Clifford circuits on 20 to 80 qubits with the stabilizer forecast of
``study/wide.py``. The block ``twirl_check`` compares the exact forecast and the stabilizer
forecast of each model of the Clifford comparisons on each 8-qubit ``wide_*`` circuit. The
stabilizer samples take a seed from the SHA-256 of the counts and the model name, so a row
repeats.

A fitted model fits only on the ``train`` circuits. A model from a calibration is scored in the
floored form of the plan. The row of a run also gives the ruled-out shots of the calibration as
stated. The stale models pull calibrations from IBM's public endpoint and keep them in the
``--cache`` folder. ``--offline`` skips the stale models. A model that fails for a run has its
reason in the row, and the other models of the run continue.

The output holds derived numbers only: scores, tests, fitted factors, measured readout errors
and the frequency of the ideal result of each deep circuit. It holds no counts, no calibration
value, no file name and no job id. The SHA-256 of each counts file and the fingerprint of each
calibration identify the data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

import noisevault as nv
from noisevault import wide as stabilizer
from noisevault.counts import MeasuredCircuit, MeasuredCounts, load_counts
from noisevault.errors import NoiseVaultError
from noisevault.profile import Profile, iso_z
from noisevault.reference import MAX_QUBITS, outcome_bits
from noisevault.score import Forecast, RuleTest, paired, score
from noisevault.suite import role
from study import ages, kept, models, wide
from study.derive import counts_files, profile_beside
from study.estimate import week_of

FORMAT = "2"
STALE_DAYS = (1, 7, 30, 90, 180, 365)
AVERAGED_DAYS = 7
GROUPS = ("test4", "test_near", "test_far", "wide", "all4")
TEST_GROUPS = ("test_near", "test_far")
FITTED = ("two_factor", "readout_measured", "locus")
# The models of the misfit, on the test_far circuits of the chain.
MISFIT_MODELS = ("locus", "two_factor", "calibration")
# The models that give a forecast of each probe circuit.
PROBE_MODELS = ("calibration", "two_factor")
# The waits after the time of C at which the source is asked for the reply of C. IBM publishes a
# calibration some minutes after its time, so an early request can give the calibration before.
PUBLICATION_WAITS = (
    timedelta(seconds=1),
    timedelta(minutes=15),
    timedelta(minutes=30),
    timedelta(hours=1),
    timedelta(hours=2),
    timedelta(hours=4),
)
DEEP_PREFIXES = ("deep_", "pair", "qubit", "periodic_")
CHAIN_KINDS = ("best", "median")
# The cells of the check of independent shots have at least this many expected shots.
CHECK_MIN_EXPECTED = 5
STALE = tuple(f"stale({d})" for d in STALE_DAYS)
MODELS = (
    "ideal",
    "generic",
    "medians",
    *STALE,
    "bundled",
    "symmetric",
    "averaged(7)",
    "calibration",
    "two_factor",
    "readout_measured",
    "locus",
)
# The five primary comparisons of the plan: model a, model b and the circuit group. Two models
# with no fitted parameter are tested on each circuit of the chain ("all4"). A fitted model is
# tested on the test circuits of the chain ("test4").
PRIMARY: tuple[tuple[str, str, str], ...] = (
    ("calibration", "medians", "all4"),
    ("calibration", "stale(30)", "all4"),
    ("calibration", "symmetric", "all4"),
    ("two_factor", "calibration", "test4"),
    ("locus", "two_factor", "test4"),
)
# The two parts of primary comparison 5.
PARTS_OF_5: tuple[tuple[str, str, str], ...] = (
    ("readout_measured", "two_factor", "test4"),
    ("locus", "readout_measured", "test4"),
)
WIDE_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("two_factor", "calibration", "wide"),
    ("locus", "two_factor", "wide"),
    ("calibration", "medians", "wide"),
)
# The comparisons of the model with the factors of the run 5 to 9 days before. A second pass
# over the runs gives them, because the model needs the fit of another run.
BEFORE_DAYS = (5, 9)
BEFORE_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("two_factor_before", "calibration", "test4"),
    ("two_factor_before", "two_factor", "test4"),
)
SECONDARY: tuple[tuple[str, str, str], ...] = (
    *(("calibration", name, "all4") for name in (*STALE, "bundled") if name != "stale(30)"),
    *PARTS_OF_5,
    ("averaged(7)", "calibration", "all4"),
    *WIDE_COMPARISONS,
    *((a, b, group) for a, b, _ in PRIMARY for group in TEST_GROUPS),
    *BEFORE_COMPARISONS,
)
_READOUT = ("readout", "readout_1")
CLIFFORD_PREFIX = "clifford_"
CLIFFORD_MODELS = (
    "generic",
    "medians",
    "stale(30)",
    "averaged(7)",
    "symmetric",
    "calibration",
    "two_factor",
)
# The secondary comparisons on the Clifford circuits. The verdict is that of the qubit score.
CLIFFORD_COMPARISONS = (
    ("calibration", "medians"),
    ("two_factor", "calibration"),
    ("averaged(7)", "calibration"),
)
# The models of the twirl check: each model that a Clifford comparison uses.
TWIRL_MODELS = tuple(dict.fromkeys(name for pair in CLIFFORD_COMPARISONS for name in pair))
TWIRL_SAMPLES = 1_000_000


def is_deep_run(counts: MeasuredCounts) -> bool:
    return any(c.name == "readout_1" or c.name.startswith("deep_") for c in counts.circuits)


def exact_names(counts: MeasuredCounts) -> list[str]:
    """The names of the circuits that the exact simulator holds: 10 qubits or fewer."""
    return [c.name for c in counts.circuits if len(c.qubits) <= MAX_QUBITS]


def chain_of(counts: MeasuredCounts) -> tuple[int, ...] | None:
    """The qubits of the ``readout`` circuit of a run: the chain of the run."""
    return next((tuple(c.qubits) for c in counts.circuits if c.name == "readout"), None)


def groups(counts: MeasuredCounts) -> dict[str, list[str]]:
    """The circuit names of each group that the plan scores. A group with no circuit is left out.

    Each group holds only circuits of 10 qubits or fewer, and no ``probe`` circuit. ``all4`` is
    each circuit whose qubits are all qubits of the chain, and ``test4`` is each test circuit
    among them. ``test_near`` and ``test_far`` also hold only circuits of the chain. ``wide``
    holds the ``wide_*`` circuits.
    """
    chain = set(chain_of(counts) or ())
    by_name = {c.name: c for c in counts.circuits}
    names = [n for n in exact_names(counts) if role(n) != "probe"]
    on_chain = [n for n in names if chain and set(by_name[n].qubits) <= chain]
    found = {g: [n for n in on_chain if role(n) == g] for g in TEST_GROUPS}
    found["test4"] = [n for n in on_chain if role(n) in TEST_GROUPS]
    found["wide"] = [n for n in names if n.startswith("wide_")]
    found["all4"] = on_chain
    return {g: found[g] for g in GROUPS if found[g]}


def scored_names(named: Mapping[str, Sequence[str]]) -> list[str]:
    """Each circuit of a group, in the order of the groups."""
    return list(dict.fromkeys(n for names in named.values() for n in names))


def score_groups(
    profile: Profile, counts: MeasuredCounts, named: Mapping[str, Sequence[str]]
) -> tuple[dict[str, Forecast], dict[str, str]]:
    """The forecast of ``profile`` for each group of ``named`` that it can simulate, and the
    reason for each other group.

    The first try simulates each circuit of all groups one time, and each group selects from
    the result. When the profile cannot simulate a circuit, each group is scored on its own, so
    a circuit outside a group does not remove the score of that group.
    """
    every = scored_names(named)
    try:
        found = score(profile, counts, circuits=every)
    except NoiseVaultError:
        pass
    else:
        return {g: found.select(names) for g, names in named.items()}, {}
    forecasts: dict[str, Forecast] = {}
    failed: dict[str, str] = {}
    for group, names in named.items():
        try:
            forecasts[group] = score(profile, counts, circuits=names)
        except NoiseVaultError as exc:
            failed[group] = _reason(exc)
    return forecasts, failed


def chain_kind(path: Path) -> str | None:
    """ "best" or "median" from the stem of the counts file, such as ``best-deep``."""
    stem = path.name.split(".")[0]
    kind = stem.split("-")[0]
    return kind if kind in CHAIN_KINDS else None


def is_trial(path: Path) -> bool:
    """True for a trial run: a counts file such as ``best-trial.counts.json``. A trial run is one
    more run of the circuit set in a week that has its runs. It is in no estimate."""
    parts = path.name.split(".")[0].split("-")
    return len(parts) > 1 and parts[1] == "trial"


def build(
    calibration: Profile,
    counts: MeasuredCounts,
    *,
    cache: Path | None,
    offline: bool,
    pull: models.Pull | None = None,
) -> tuple[dict[str, Profile], dict[str, str], dict[str, Any]]:
    """Each model of the plan in the form that the comparisons score, the reason for each model
    that is not built, and the fitted values.

    Each model from a calibration starts from the calibration as stated. The zero floor applies
    to the result, as the plan states.
    """
    train = counts.select([c.name for c in counts.circuits if role(c.name) == "train"])
    built: dict[str, Profile] = {}
    missing: dict[str, str] = {}
    fits: dict[str, Any] = {}

    def attempt(name: str, make: Callable[[], Profile | None], absent: str = "") -> None:
        try:
            found = make()
        except Exception as exc:
            # One model that fails must not stop the other models of the run. The failure is in
            # the output, with its reason.
            missing[name] = _reason(exc)
            return
        if found is None:
            missing[name] = absent
        else:
            built[name] = found

    def two() -> Profile:
        fitted, comparison = models.two_factor(calibration, train)
        row = comparison.to_dict()
        fits["two_factor"] = {
            key: row[key] for key in ("gates", "readout", "p_value", "dispersion")
        }
        return models.floored(fitted)

    def week() -> Profile | None:
        """The median over the calibrations in effect at the run time and 1 to 6 days before.

        A calibration that is in effect at two of these times counts two times.
        """
        if calibration.device.vendor != "ibm":
            return None
        device = calibration.device.name
        pulls = [
            models.pulled(device, counts.run_at - timedelta(days=days), cache, pull)
            for days in range(AVERAGED_DAYS)
        ]
        found = [p for p in pulls if p is not None]
        if not found:
            return None
        different = {p.device.calibrated_at or p.calibration_fingerprint for p in found}
        fits["averaged(7)"] = {"pulls": len(found), "calibrations": len(different)}
        return models.floored(models.averaged(calibration, found))

    def measured() -> Profile:
        rates = models.floored(models.readout_measured(calibration, train))
        others = [c.name for c in train.circuits if c.name not in _READOUT]
        factor = models.gate_factor(rates, train, others)
        chain = next(c.qubits for c in train.circuits if c.name == "readout")
        fits["readout_measured"] = {
            "readout": models.readout_rates(rates, chain),
            "gates": factor,
        }
        return models.with_factors(rates, factor)

    def per_locus() -> Profile:
        fitted, _, rows = models.locus(calibration, train)
        fits["locus"] = rows
        return models.floored(fitted)

    attempt("ideal", lambda: models.ideal(calibration))
    attempt("generic", lambda: models.generic(calibration))
    attempt("medians", lambda: models.floored(models.medians(calibration)))
    for days, name in zip(STALE_DAYS, STALE, strict=True):
        if offline:
            missing[name] = "skipped: --offline"
            continue
        attempt(
            name,
            lambda days=days: _floored(models.stale(calibration, counts.run_at, days, cache, pull)),
            _no_stale(calibration),
        )
    attempt(
        "bundled",
        lambda: _floored(models.bundled(calibration)),
        "NoiseVault bundles no other calibration of this device",
    )
    attempt("symmetric", lambda: models.floored(models.symmetric(calibration)))
    if offline:
        missing["averaged(7)"] = "skipped: --offline"
    else:
        attempt("averaged(7)", week, _no_stale(calibration))
    attempt("calibration", lambda: models.floored(calibration))
    attempt("two_factor", two)
    attempt("readout_measured", measured)
    attempt("locus", per_locus)
    return built, missing, fits


def analyze_run(
    path: Path,
    *,
    cache: Path | None,
    offline: bool,
    pull: models.Pull | None = None,
    properties: ages.Properties | None = None,
    samples: int = wide.SAMPLES,
) -> dict[str, Any] | None:
    """The derived row of one counts file, or None when the file is not a run of the deep set.

    ``samples`` is the number of stabilizer samples of each Clifford forecast.
    """
    counts = load_counts(path)
    if not is_deep_run(counts):
        return None
    calibration = nv.load(str(profile_beside(path)))
    built, missing, fits = build(calibration, counts, cache=cache, offline=offline, pull=pull)
    named = groups(counts)
    every = scored_names(named)
    forecasts: dict[str, dict[str, Forecast]] = {}
    done: dict[str, tuple[dict[str, Forecast], dict[str, str]]] = {}

    def scored(name: str, profile: Profile) -> dict[str, Forecast]:
        """The forecast of each group that ``profile`` can simulate. A model that cannot simulate
        any group has its reason under its name. A model that can simulate a part of the groups
        has the reason of each other group under "name (group)"."""
        if profile.fingerprint not in done:
            done[profile.fingerprint] = score_groups(profile, counts, named)
        found, failed = done[profile.fingerprint]
        if not found and failed:
            missing[name] = next(iter(failed.values()))
        else:
            for group, reason in failed.items():
                missing[f"{name} ({group})"] = reason
        return found

    for name in MODELS:
        if name not in built or not every:
            continue
        found = scored(name, built[name])
        if found:
            forecasts[name] = found

    stated_forecasts = scored("calibration (stated)", calibration.uncorrected()) if every else {}
    narrow = tuple(
        dict.fromkeys(q for c in counts.circuits if len(c.qubits) <= MAX_QUBITS for q in c.qubits)
    )
    chain = chain_of(counts)
    when = calibration.device.calibrated_at
    age = None if when is None else (counts.run_at - when).total_seconds() / 3600
    shots = {c.shots for c in counts.circuits}
    return {
        "device": calibration.device.name,
        "vendor": calibration.device.vendor,
        "technology": calibration.device.technology,
        "run_at": iso_z(counts.run_at),
        "calibrated_at": None if when is None else iso_z(when),
        "calibration_age_hours": age,
        "qubits": list(narrow),
        "chain": chain_kind(path),
        "trial": is_trial(path),
        "shots": shots.pop() if len(shots) == 1 else None,
        "circuits": {g: len(names) for g, names in named.items()},
        "counts_sha256": counts.sha256,
        "calibration": calibration.calibration_fingerprint,
        "marked": _marked(calibration, counts, cache, offline, pull),
        "ages": {} if offline else _ages(calibration, counts, cache, properties, narrow),
        "stated": {
            "zero_readouts": models.zero_readouts(calibration),
            "zero_on_chain": None if chain is None else zero_on(calibration, chain),
            "ruled_out": {g: f.ruled_out for g, f in stated_forecasts.items()},
            "log_score": {g: _finite(f.log_score) for g, f in stated_forecasts.items()},
        },
        "wide_chain_overlap": _overlap(counts, chain),
        "scores": {
            name: {g: _forecast(f, fitted=name in FITTED) for g, f in by_group.items()}
            for name, by_group in forecasts.items()
        },
        "paired": _comparisons(forecasts),
        "misfit": _misfit(forecasts),
        "deep": _deep(counts, forecasts),
        "probes": _probes(counts, built, chain, missing),
        "clifford": _clifford(counts, built, samples),
        "twirl_check": _twirl_check(counts, built, forecasts),
        "fits": fits,
        "failures": dict(sorted(missing.items(), key=lambda item: _order(item[0]))),
    }


def zero_on(calibration: Profile, qubits: Sequence[int]) -> bool:
    """True when the calibration as stated has a readout probability of exactly 0 on one of
    ``qubits``."""
    stated = calibration.uncorrected().table
    return any(value == 0 for q in qubits for value in stated.qubit(q).readout or ())


def analyze(
    root: Path,
    *,
    cache: Path | None = None,
    offline: bool = False,
    pull: models.Pull | None = None,
    properties: ages.Properties | None = None,
    rows_folder: Path | None = None,
    samples: int = wide.SAMPLES,
) -> dict[str, Any]:
    """The analysis of each run of the deep circuit set below ``root``.

    ``rows_folder`` keeps the row of each counts file, so that a later call does not analyze the
    same run again. A counts file with the same counts as an earlier file, for example the same
    run saved as ``.counts.json`` and ``.counts.json.gz``, is a duplicate. Only the first file
    gives a row, and ``duplicates`` gives the number of the other files.

    A second pass opens the two counts files of each device-week and gives the check of
    independent shots (see :func:`shot_check`).
    """
    rows, failures, skipped, duplicates = [], [], 0, 0
    paths: dict[str, Path] = {}
    seen: set[str] = set()
    kind = "analysis-offline" if offline else "analysis"
    if samples != wide.SAMPLES:
        kind += f"-{samples}"
    for path in counts_files(root):

        def run(path: Path = path) -> dict[str, Any] | None:
            return analyze_run(
                path,
                cache=cache,
                offline=offline,
                pull=pull,
                properties=properties,
                samples=samples,
            )

        try:
            digest = load_counts(path).sha256
            if digest in seen:
                duplicates += 1
                continue
            seen.add(digest)
            beside = profile_beside(path)
            row = kept(rows_folder, kind, [path, beside] if beside.exists() else [path], run)
        except NoiseVaultError as exc:
            # A message can hold the path of the file, and a file name can hold an id of the
            # account. So a failure gives only the error type and the hash of the file.
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            failures.append({"error": type(exc).__name__, "file": f"sha256:{digest}"})
            continue
        if row is None:
            skipped += 1
        else:
            rows.append(row)
            paths[row["counts_sha256"]] = path
    rows.sort(key=lambda r: (r["device"], r["run_at"], r["chain"] or "", r["counts_sha256"]))
    for row in rows:
        _add_before(row, rows, paths[row["counts_sha256"]])
    return {
        "nv_study_analysis": FORMAT,
        "rows": rows,
        "skipped": skipped,
        "duplicates": duplicates,
        "failures": failures,
        "shot_checks": _shot_checks(rows, paths),
    }


def device_week(row: Mapping[str, Any]) -> tuple[str, date]:
    """The device and the Monday of the ISO week of the run time in UTC."""
    return row["device"], week_of(_run_time(row).date())


def earlier_run(
    row: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    """The newest run of the same device and the same chain kind from ``BEFORE_DAYS`` before
    ``row``, or None when there is no such run. A trial run is never the earlier run."""
    when = _run_time(row)
    low, high = (timedelta(days=d) for d in BEFORE_DAYS)
    found = [
        r
        for r in rows
        if r["device"] == row["device"]
        and r["chain"] == row["chain"]
        and not r.get("trial")
        and low <= when - _run_time(r) <= high
    ]
    return max(found, key=_run_time, default=None)


def _add_before(row: dict[str, Any], rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Add the model ``two_factor_before`` to ``row``: the gate factor and the readout factor of
    ``two_factor`` in the earlier run (see :func:`earlier_run`), on the calibration C of this
    run, with the zero floor after. ``row["before"]`` gives the earlier run and the factors, or
    the reason for no model. The comparisons of ``BEFORE_COMPARISONS`` go to ``row["paired"]``.
    """
    earlier = earlier_run(row, rows)
    if earlier is None:
        row["before"] = {"missing": "no_earlier_run"}
        return
    fitted = (earlier.get("fits") or {}).get("two_factor")
    own = (row.get("fits") or {}).get("two_factor")
    if not fitted or not own:
        row["before"] = {"missing": "no_two_factor_fit"}
        return
    factors = {k: (fitted.get(k) or {}).get("factor") for k in ("gates", "readout")}
    row["before"] = {
        "run_at": earlier["run_at"],
        "days": (_run_time(row) - _run_time(earlier)).total_seconds() / 86400,
        "factors": factors,
    }
    try:
        counts = load_counts(path)
        calibration = nv.load(str(profile_beside(path)))
        named = groups(counts)
        if "test4" not in named:
            row["before"]["missing"] = "no_test4_circuits"
            return
        mine = {k: (own.get(k) or {}).get("factor") for k in ("gates", "readout")}
        built = {
            "two_factor_before": models.with_factors(calibration, *factors.values()),
            "two_factor": models.with_factors(calibration, *mine.values()),
            "calibration": calibration,
        }
        forecasts = {
            name: {"test4": score(models.floored(profile), counts, circuits=named["test4"])}
            for name, profile in built.items()
        }
    except NoiseVaultError as exc:
        row["before"]["missing"] = _reason(exc)
        return
    row["scores"]["two_factor_before"] = {
        "test4": _forecast(forecasts["two_factor_before"]["test4"], fitted=True)
    }
    row["paired"] += _comparisons(forecasts, BEFORE_COMPARISONS)


def _run_time(row: Mapping[str, Any]) -> datetime:
    return datetime.fromisoformat(str(row["run_at"]).replace("Z", "+00:00")).astimezone(UTC)


def _shot_checks(rows: Sequence[dict[str, Any]], paths: Mapping[str, Path]) -> list[dict[str, Any]]:
    """The check of independent shots of each device-week with one run on each chain kind. The
    check uses the two normal runs, not the trial runs."""
    weeks: dict[tuple[str, date], list[dict[str, Any]]] = {}
    for row in rows:
        if not row.get("trial"):
            weeks.setdefault(device_week(row), []).append(row)
    checks = []
    for (device, week), found in sorted(weeks.items()):
        kinds = sorted(r["chain"] or "" for r in found)
        if kinds != sorted(CHAIN_KINDS):
            continue
        first, second = (load_counts(paths[r["counts_sha256"]]) for r in found)
        try:
            check = shot_check(first, second)
        except NoiseVaultError as exc:
            check = {"circuits": [], "not_the_same": None, "reason": _reason(exc)}
        checks.append({"device": device, "week": week.isoformat(), **check})
    return checks


def shot_check(first: MeasuredCounts, second: MeasuredCounts) -> dict[str, Any]:
    """The Pearson chi-square statistic of the 2 by K table of each circuit that both runs hold.

    Two circuits are the same when they have the same name, the same qubits and the same ops.
    On a Clifford circuit, a cell is a number of wrong bits. On another circuit, a cell is an
    outcome. The cells with fewer than ``CHECK_MIN_EXPECTED`` expected shots in a run, from the
    pooled counts, become one cell. The result gives the statistic, its degrees of freedom K - 1
    and their ratio for each circuit, and the number of circuits with one name in both runs that
    are not the same.
    """
    others = {c.name: c for c in second.circuits}
    circuits, different = [], 0
    for a in first.circuits:
        b = others.get(a.name)
        if b is None:
            continue
        if (a.qubits, a.ops) != (b.qubits, b.ops):
            different += 1
            continue
        if a.name.startswith(CLIFFORD_PREFIX):
            ideal = stabilizer.ideal_bits(a)
            table = np.array([_wrong_cells(c, ideal) for c in (a, b)])
        else:
            keys = sorted(set(a.counts) | set(b.counts))
            table = np.array([[c.counts.get(k, 0) for k in keys] for c in (a, b)], dtype=float)
        statistic, df = chi_square(table)
        circuits.append(
            {
                "name": a.name,
                "statistic": statistic,
                "df": df,
                "ratio": None if df == 0 else statistic / df,
            }
        )
    return {"circuits": circuits, "not_the_same": different}


def chi_square(table: np.ndarray) -> tuple[float, int]:
    """The Pearson chi-square statistic of a 2 by K table and its degrees of freedom.

    The columns with fewer than ``CHECK_MIN_EXPECTED`` expected shots in a row become one column
    first. The expected shots of a cell are the shots of its row times the pooled fraction of its
    column.
    """
    table = np.asarray(table, dtype=float)
    table = table[:, table.sum(axis=0) > 0]
    rows = table.sum(axis=1, keepdims=True)
    expected = rows * table.sum(axis=0) / table.sum()
    rare = (expected < CHECK_MIN_EXPECTED).any(axis=0)
    if rare.any():
        table = np.column_stack([table[:, ~rare], table[:, rare].sum(axis=1)])
    if table.shape[1] < 2:
        return 0.0, 0
    expected = rows * table.sum(axis=0) / table.sum()
    return float(((table - expected) ** 2 / expected).sum()), table.shape[1] - 1


def _wrong_cells(circuit: MeasuredCircuit, ideal: str) -> np.ndarray:
    """The shots of each number of wrong bits, 0 to the number of qubits."""
    keys, times = wide._rows(circuit)
    wrong = (keys != np.array([bit == "1" for bit in ideal])).sum(axis=1)
    return np.bincount(wrong, weights=times, minlength=len(ideal) + 1)


def markdown(data: dict[str, Any]) -> str:
    lines: list[str] = []
    for r in data["rows"]:
        title = f"{r['device']}, {r['chain'] or 'chain'} chain, run {r['run_at']}"
        lines += [f"## {title}", "", "| Model | " + " | ".join(GROUPS) + " |"]
        lines.append("| --- |" + " --- |" * len(GROUPS))
        for name, by_group in r["scores"].items():
            cells = [_score(by_group.get(g)) for g in GROUPS]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        ruled = r["stated"]["ruled_out"].get("all4", 0)
        on_chain = {True: "yes", False: "no", None: "-"}[r["stated"]["zero_on_chain"]]
        lines += [
            "",
            f"Calibration as stated: {r['stated']['zero_readouts']} readout probabilities of 0,"
            f" a 0 on the chain: {on_chain}, {ruled} ruled-out shots on the chain.",
        ]
        for name, reason in r["failures"].items():
            lines.append(f"No {name}: {reason}")
        lines.append("")
    for title, primary in (("Primary comparisons", True), ("Secondary comparisons", False)):
        lines += [f"## {title}", "", "Better is the verdict of the log score.", ""]
        lines.append("| Device | Chain | Run | A | B | Group | Better | z | p |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in data["rows"]:
            for test in r["paired"]:
                if test["primary"] != primary:
                    continue
                log = test["log"]
                lines.append(
                    f"| {r['device']} | {r['chain'] or '-'} | {r['run_at']} | {test['a']} |"
                    f" {test['b']} | {test['group']} | {test['winner'] or 'neither'} |"
                    f" {_number(log['z'], '.2f')} | {_number(log['p_value'], '.3g')} |"
                )
        lines.append("")
    if data.get("duplicates"):
        lines += [f"Duplicate counts files not analyzed: {data['duplicates']}", ""]
    for failure in data["failures"]:
        lines += [f"Failed: {failure['error']} on the counts file {failure['file'][:19]}", ""]
    return "\n".join(lines)


def write(data: dict[str, Any], out: Path) -> None:
    """Write ``analysis.json`` and ``analysis.md`` to the folder ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=1, allow_nan=False) + "\n"
    (out / "analysis.json").write_text(text, encoding="utf-8")
    (out / "analysis.md").write_text(markdown(data), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", type=Path, help="the folder with the counts files")
    parser.add_argument("--out", type=Path, default=Path("results"), help="the output folder")
    parser.add_argument(
        "--cache", type=Path, default=None, help="the folder for pulled calibrations"
    )
    parser.add_argument("--offline", action="store_true", help="skip the stale models")
    parser.add_argument(
        "--rows", type=Path, default=None, help="a folder that keeps the row of each run"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=wide.SAMPLES,
        help="the stabilizer samples of each Clifford forecast",
    )
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        print(f"error: {args.root} is not a folder", file=sys.stderr)
        return 2
    data = analyze(
        args.root,
        cache=args.cache,
        offline=args.offline,
        rows_folder=args.rows,
        samples=args.samples,
    )
    write(data, args.out)
    print(f"wrote {len(data['rows'])} runs and {len(data['failures'])} failures to {args.out}")
    return 0 if data["rows"] and not data["failures"] else 1


def _floored(profile: Profile | None) -> Profile | None:
    return None if profile is None else models.floored(profile)


def _no_stale(calibration: Profile) -> str:
    if calibration.device.vendor != "ibm":
        return "the device is not an IBM device"
    return "the source has no other calibration in effect at that time"


def _marked(
    calibration: Profile,
    counts: MeasuredCounts,
    cache: Path | None,
    offline: bool,
    pull: models.Pull | None,
) -> bool | None:
    """True when the calibration in effect at the run is not C, so the counts bind to the
    planned calibration. None when the script cannot tell."""
    if offline or calibration.device.vendor != "ibm":
        return None
    try:
        found = models.pulled(calibration.device.name, counts.run_at, cache, pull)
    except NoiseVaultError:
        return None
    return None if found is None else not models.same_calibration(found, calibration)


def _ages(
    calibration: Profile,
    counts: MeasuredCounts,
    cache: Path | None,
    properties: ages.Properties | None,
    qubits: Sequence[int] | None = None,
) -> dict[str, Any]:
    """For the calibration C and for each stale calibration, the age of the calibration time and
    of the values of ``qubits`` at the run time, in hours. IBM dates each value, so only an IBM
    device has ages. Without ``qubits``, the ages are of every qubit that ``counts`` measures.

    The entry ``"calibration"`` comes from the reply of C itself, whose ``last_update_date`` is
    the time of C. IBM publishes a calibration some minutes after that time, so the source is
    asked at the time of C plus each wait of ``PUBLICATION_WAITS``, in that order, and the first
    reply of C is used. A reply with a later calibration time stops the search. When no reply
    is of C, the result has no entry ``"calibration"``, and ``"missing"`` gives the reason:
    ``undated``, ``no_reply`` (no reply at all), ``source_failed`` or ``other_calibration``.
    """
    if calibration.device.vendor != "ibm":
        return {}
    device = calibration.device.name
    qubits = counts.qubits if qubits is None else qubits

    def public(name: str, at: datetime) -> Mapping[str, Any] | None:
        return ages.public_properties(name, at, cache)

    fetch = properties or public

    def entry(reply: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "calibration_h": ages.calibration_age(reply, counts.run_at),
            "values": ages.value_ages(reply, counts.run_at, qubits),
            "loci": ages.locus_ages(reply, counts.run_at, qubits),
        }

    found: dict[str, Any] = {}
    missing: dict[str, str] = {}
    when = calibration.device.calibrated_at
    if when is None:
        missing["calibration"] = "undated"
    else:
        reason = "no_reply"
        for wait in PUBLICATION_WAITS:
            try:
                # The source gives the newest published reply before the requested time.
                reply = fetch(device, when + wait)
            except NoiseVaultError:
                reason = "source_failed"
                break
            if reply is None:
                continue
            published = ages.calibration_time(reply)
            if published == when:
                found["calibration"] = entry(reply)
                break
            reason = "other_calibration"
            if published is not None and published > when:
                break
        if "calibration" not in found:
            missing["calibration"] = reason
    for name, days in zip(STALE, STALE_DAYS, strict=True):
        try:
            reply = fetch(device, counts.run_at - timedelta(days=days))
        except NoiseVaultError:
            continue
        if reply is not None:
            found[name] = entry(reply)
    if missing:
        found["missing"] = missing
    return found


def _forecast(forecast: Forecast, *, fitted: bool) -> dict[str, Any]:
    """The scores of one model on one group. A fitted model has no p-value of its excess, because
    the reference of ``nv score`` does not hold the error of the fit."""
    data = forecast.to_dict()
    for key in ("circuits", "counts", *(("p_value",) if fitted else ())):
        data.pop(key)
    return data


def _misfit(forecasts: Mapping[str, Mapping[str, Forecast]]) -> dict[str, Any]:
    """For each model of ``MISFIT_MODELS``, the size of its misfit on the ``test_far`` circuits
    of the chain, per shot.

    ``misfit`` is the excess minus the mean excess of counts drawn from the model. ``rare`` is
    the part of the excess from the outcomes that the model expects fewer than 1 time in the
    shots of their circuit. Both values are None when the model rules out a shot.
    """
    found: dict[str, Any] = {}
    for name in MISFIT_MODELS:
        forecast = forecasts.get(name, {}).get("test_far")
        if forecast is None:
            continue
        names = {c.name for c in forecast.circuits}
        measured = {c.name: c.vector() for c in forecast.counts.circuits if c.name in names}
        rare = 0.0
        for c in forecast.circuits:
            n, p = measured[c.name], c.probabilities
            chosen = (n > 0) & (c.shots * p < 1) & (p > 0)
            rare += float(n[chosen] @ (np.log(n[chosen]) - math.log(c.shots) - np.log(p[chosen])))
        excess = _finite(forecast.excess)
        found[name] = {
            "excess": excess,
            "noise_mean": forecast.excess_noise_mean,
            "misfit": None if excess is None else excess - forecast.excess_noise_mean,
            "rare": None if excess is None else rare / forecast.shots,
        }
    return found


def _overlap(counts: MeasuredCounts, chain: Sequence[int] | None) -> int | None:
    """The number of qubits of the ``wide_*`` circuits that are in the chain of the run."""
    wide_qubits = {q for c in counts.circuits if c.name.startswith("wide_") for q in c.qubits}
    if not wide_qubits or chain is None:
        return None
    return len(wide_qubits & set(chain))


def _probes(
    counts: MeasuredCounts,
    built: Mapping[str, Profile],
    chain: Sequence[int] | None,
    missing: dict[str, str],
) -> dict[str, Any]:
    """The table of the ``probe`` circuits of a run.

    ``baseline`` gives the measured fraction of wrong bits of each chain qubit in the
    ``readout`` circuit. For each probe circuit, the table gives the ideal bit string (the most
    probable result of the ``ideal`` model), the measured and the forecast probability of it,
    and for each qubit the measured and the forecast fraction of wrong bits. The forecasts come
    from each model of ``PROBE_MODELS``.
    """
    by_name = {c.name: c for c in counts.circuits}
    baseline = []
    if "readout" in by_name:
        readout = by_name["readout"]
        wrong = _wrong_fractions(readout, "0" * len(readout.qubits))
        baseline = [{"qubit": q, "measured": w} for q, w in zip(readout.qubits, wrong, strict=True)]
    probes = [c for c in counts.circuits if role(c.name) == "probe" and len(c.qubits) <= MAX_QUBITS]
    result: dict[str, Any] = {"chain": None if chain is None else list(chain)}
    result |= {"baseline": baseline, "circuits": []}
    if not probes or "ideal" not in built:
        return result
    names = [c.name for c in probes]
    probabilities: dict[str, dict[str, np.ndarray]] = {}
    for name in ("ideal", *PROBE_MODELS):
        if name not in built:
            continue
        try:
            found = score(built[name], counts, circuits=names)
        except NoiseVaultError as exc:
            missing[f"{name} (probe)"] = _reason(exc)
            continue
        probabilities[name] = {c.name: c.probabilities for c in found.circuits}
    if "ideal" not in probabilities:
        return result
    for c in probes:
        width = len(c.qubits)
        index = int(np.argmax(probabilities["ideal"][c.name]))
        bits = outcome_bits(index, width)
        wrong = _wrong_fractions(c, bits)
        models_ = {m: p[c.name] for m, p in probabilities.items() if m != "ideal"}
        forecast_wrong = {m: _forecast_wrong(p, bits) for m, p in models_.items()}
        result["circuits"].append(
            {
                "name": c.name,
                "ideal": bits,
                "measured": c.counts.get(bits, 0) / c.shots,
                "forecast": {m: float(p[index]) for m, p in models_.items()},
                "qubits": [
                    {
                        "qubit": q,
                        "measured": wrong[i],
                        "forecast": {m: values[i] for m, values in forecast_wrong.items()},
                    }
                    for i, q in enumerate(c.qubits)
                ],
            }
        )
    return result


def _wrong_fractions(circuit: MeasuredCircuit, ideal: str) -> list[float]:
    """The measured fraction of wrong bits of each circuit qubit."""
    keys, times = wide._rows(circuit)
    wrong = keys != np.array([bit == "1" for bit in ideal])
    return [float(v) for v in times @ wrong / circuit.shots]


def _forecast_wrong(probabilities: np.ndarray, ideal: str) -> list[float]:
    """The forecast fraction of wrong bits of each circuit qubit. Outcome index i has the bits of
    ``outcome_bits(i, n)``, circuit qubit 0 first."""
    width = len(ideal)
    index = np.arange(2**width)
    found = []
    for i, bit in enumerate(ideal):
        ones = (index >> (width - 1 - i)) & 1
        found.append(float(probabilities[ones != int(bit)].sum()))
    return found


def _comparisons(
    forecasts: Mapping[str, Mapping[str, Forecast]],
    secondary: Sequence[tuple[str, str, str]] | None = None,
) -> list[dict[str, Any]]:
    """The paired tests of the primary and the secondary comparisons. With ``secondary``, only
    the tests of those secondary comparisons."""
    tests = []
    chosen = [(True, PRIMARY), (False, SECONDARY)] if secondary is None else [(False, secondary)]
    for primary, comparisons in chosen:
        for a, b, group in comparisons:
            if group not in forecasts.get(a, {}) or group not in forecasts.get(b, {}):
                continue
            test = paired(forecasts[a][group], forecasts[b][group])
            data = test.to_dict()
            tests.append(
                {
                    "a": a,
                    "b": b,
                    "group": group,
                    "primary": primary,
                    # The plan takes the verdict of the log score, in each output.
                    "winner": {"a": a, "b": b, "neither": None}[test.log.better],
                    "level": data["level"],
                    "log": data["log"],
                    "quadratic": data["quadratic"],
                    "better": data["better"],
                }
            )
    return tests


def _deep(
    counts: MeasuredCounts, forecasts: Mapping[str, Mapping[str, Forecast]]
) -> list[dict[str, Any]]:
    """For each deep circuit, the ideal result, its measured frequency and its probability in
    each model that has a score for a group of the circuit. The ideal result is the most
    probable result under the ``ideal`` model."""
    if "ideal" not in forecasts:
        return []
    by_model = {
        name: {c.name: c.probabilities for found in by_group.values() for c in found.circuits}
        for name, by_group in forecasts.items()
    }
    rows = []
    for c in counts.circuits:
        if not c.name.startswith(DEEP_PREFIXES) or c.name not in by_model["ideal"]:
            continue
        index = int(np.argmax(by_model["ideal"][c.name]))
        bits = outcome_bits(index, len(c.qubits))
        rows.append(
            {
                "name": c.name,
                "role": role(c.name),
                "ideal": bits,
                "measured": c.counts.get(bits, 0) / c.shots,
                "models": {
                    name: float(probs[c.name][index])
                    for name, probs in by_model.items()
                    if c.name in probs
                },
            }
        )
    return rows


def _clifford(counts: MeasuredCounts, built: dict[str, Profile], samples: int) -> dict[str, Any]:
    """The stabilizer scores of the models on each Clifford circuit, and the paired tests of
    ``CLIFFORD_COMPARISONS`` on each circuit and on all of them together.

    Each forecast uses ``samples`` stabilizer samples. The tests of the qubit score and of the
    log rule of the wrong-bit score hold the sampling variance of both forecasts, and
    ``sampling_ratio`` gives the sampling variance divided by the shot variance."""
    circuits = [c for c in counts.circuits if c.name.startswith(CLIFFORD_PREFIX)]
    found: dict[str, list[wide.Forecast]] = {}
    failures: dict[str, str] = {}
    if not circuits:
        return {"samples": samples, "circuits": [], "paired": [], "failures": failures}
    if not stabilizer.has_stim():
        failures = {"stabilizer": "ImportError"}
        return {"samples": samples, "circuits": [], "paired": [], "failures": failures}
    for name in CLIFFORD_MODELS:
        if name not in built:
            continue
        seed = wide.seed_of(counts.sha256, name)
        try:
            found[name] = [
                wide.forecast(built[name], c, samples=samples, seed=seed) for c in circuits
            ]
        except Exception as exc:
            # A model that cannot simulate a circuit has no score, as the plan states.
            failures[name] = _reason(exc)
    rows = []
    for i, c in enumerate(circuits):
        ideal = stabilizer.ideal_bits(c)
        success, wrong_fraction = wide.measured(c, ideal)
        rows.append(
            {
                "name": c.name,
                "qubits": len(c.qubits),
                "depth": int(c.name.rsplit("_d", 1)[1]),
                "shots": c.shots,
                "measured": {"success": success, "wrong_fraction": wrong_fraction},
                "models": {
                    name: {
                        "predicted": {
                            "success": forecasts[i].success,
                            "wrong_fraction": float(forecasts[i].qubit.mean()),
                        },
                        "scores": vars(wide.scores(c, forecasts[i])),
                    }
                    for name, forecasts in found.items()
                },
            }
        )
    tests = []
    for a, b in CLIFFORD_COMPARISONS:
        if a not in found or b not in found:
            continue
        parts = [([c], [found[a][i]], [found[b][i]], c.name) for i, c in enumerate(circuits)]
        parts.append((circuits, found[a], found[b], "all"))
        for chosen, fa, fb, label in parts:
            result = wide.compare(chosen, fa, fb)
            tests.append(
                {
                    "a": a,
                    "b": b,
                    "circuits": label,
                    "winner": {"a": a, "b": b, "neither": None}[result.qubit_log.better],
                    "success": {
                        "log": _rule(result.success_log),
                        "quadratic": _rule(result.success_quadratic),
                        "verdict": result.success_verdict,
                    },
                    "wrong": {
                        "log": _rule(result.wrong_log),
                        "quadratic": _rule(result.wrong_quadratic),
                        "sampling_ratio": result.wrong_sampling.ratio,
                    },
                    "qubit": {
                        "log": _rule(result.qubit_log),
                        "sampling_ratio": result.qubit_sampling.ratio,
                    },
                }
            )
    return {"samples": samples, "circuits": rows, "paired": tests, "failures": failures}


def _twirl_check(
    counts: MeasuredCounts, built: dict[str, Profile], forecasts: dict[str, dict[str, Forecast]]
) -> list[dict[str, Any]]:
    """For each model of ``TWIRL_MODELS`` and each ``wide_*`` circuit, the log score from the
    exact simulator and from ``TWIRL_SAMPLES`` stabilizer samples with add-half smoothing over
    the outcomes."""
    by_name = {c.name: c for c in counts.circuits}
    rows = []
    for model in TWIRL_MODELS:
        exact = forecasts.get(model, {}).get("wide")
        if exact is None:
            continue
        rows += _twirl_rows(counts, by_name, built[model], model, exact)
    return rows


def _twirl_rows(
    counts: MeasuredCounts,
    by_name: Mapping[str, MeasuredCircuit],
    profile: Profile,
    model: str,
    exact: Forecast,
) -> list[dict[str, Any]]:
    rows = []
    for found in exact.circuits:
        c = by_name[found.name]
        row: dict[str, Any] = {
            "model": model,
            "name": c.name,
            "exact_log": _finite(found.log_score),
        }
        try:
            bits = stabilizer.sample(
                profile,
                c,
                TWIRL_SAMPLES,
                seed=wide.seed_of(counts.sha256, f"{model} twirl"),
            )
        except Exception as exc:
            # A circuit with a gate outside the Clifford table has no stabilizer forecast.
            rows.append(row | {"stabilizer_log": None, "difference": None, "reason": _reason(exc)})
            continue
        width = len(c.qubits)
        index = bits.astype(np.int64) @ (1 << np.arange(width - 1, -1, -1))
        cells = np.bincount(index, minlength=2**width) + 0.5
        probs = cells / cells.sum()
        measured = c.vector()
        stabilizer_log = float(measured @ np.log(probs)) / c.shots
        difference = None if row["exact_log"] is None else stabilizer_log - row["exact_log"]
        rows.append(row | {"stabilizer_log": stabilizer_log, "difference": difference})
    return rows


def _rule(test: RuleTest) -> dict[str, Any]:
    return {
        "outcome": test.outcome,
        "difference": test.difference,
        "standard_error": test.standard_error,
        "z": test.z,
        "p_value": test.p_value,
        "better": test.better,
    }


def _order(name: str) -> int:
    """The place of a failure in the order of ``MODELS``. A failure of one group, such as
    "stale(30) (wide)", takes the place of its model."""
    model = name.split(" (")[0]
    return MODELS.index(model) if model in MODELS else len(MODELS)


def _reason(exc: Exception) -> str:
    """The type of ``exc`` and, for a validation error, the profile fields that failed.

    The reason holds no message text and no input value, because a message can hold a file
    name, a job id or a calibration value. Of the path of a failed field, only the first part
    is kept, and only when it is a field of the profile format. A later part can be a key that
    the user wrote, such as a key of ``provenance.extra``. When ``exc`` has a cause, the reason
    also gives the type of the first cause, such as "CountsError from MissingCalibrationError".
    """
    name = type(exc).__name__
    cause = exc.__cause__
    while cause is not None and cause.__cause__ is not None:
        cause = cause.__cause__
    if cause is not None:
        name = f"{name} from {type(cause).__name__}"
    if not isinstance(exc, ValidationError):
        return name
    fields = []
    for error in exc.errors(include_url=False, include_context=False, include_input=False):
        first = error["loc"][0] if error["loc"] else None
        fields.append(first if first in Profile.model_fields else "(profile)")
    return f"{name} at {', '.join(dict.fromkeys(fields))}"


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _score(forecast: dict[str, Any] | None) -> str:
    if forecast is None:
        return "-"
    if forecast["log_score"] is None:
        return f"ruled out ({forecast['ruled_out']} shots)"
    return f"{forecast['log_score']:.5f}"


def _number(value: float | None, form: str) -> str:
    return "-" if value is None else format(value, form)


if __name__ == "__main__":
    raise SystemExit(main())
