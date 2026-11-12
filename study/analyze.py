"""Score the models of the realism study on each run of the deep circuit set.

    python -m study.analyze campaign --out results --cache pulls

The script reads each counts file below the folder that holds a run of the deep circuit set
(a file with the ``readout_1`` circuit or ``deep_*`` circuits), with the profile file beside it.
For each run, it builds the models of ``PLAN.md``, scores each model on the ``test_near``,
``test_far`` and all circuits, and runs the paired test of each primary comparison. It writes
``analysis.json`` and ``analysis.md`` to the output folder.

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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

import noisevault as nv
from noisevault.counts import MeasuredCounts, load_counts
from noisevault.errors import NoiseVaultError
from noisevault.profile import Profile, iso_z
from noisevault.reference import outcome_bits
from noisevault.score import Forecast, paired, score
from noisevault.suite import role
from study import ages, kept, models
from study.derive import counts_files, profile_beside

FORMAT = "1"
STALE_DAYS = (1, 7, 30, 90, 180, 365)
AVERAGED_DAYS = 7
GROUPS = ("test", "test_near", "test_far", "wide", "all")
TEST_GROUPS = ("test_near", "test_far")
DEEP_PREFIXES = ("deep_", "pair", "qubit", "periodic_")
CHAIN_KINDS = ("best", "median")
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
# with no fitted parameter are tested on every circuit. A fitted model is tested on "test".
PRIMARY: tuple[tuple[str, str, str], ...] = (
    ("calibration", "medians", "all"),
    ("calibration", "stale(30)", "all"),
    ("calibration", "symmetric", "all"),
    ("two_factor", "calibration", "test"),
    ("locus", "two_factor", "test"),
)
SECONDARY: tuple[tuple[str, str, str], ...] = (
    *(("calibration", name, "all") for name in (*STALE, "bundled") if name != "stale(30)"),
    ("readout_measured", "two_factor", "test"),
    ("locus", "readout_measured", "test"),
    ("averaged(7)", "calibration", "all"),
    ("two_factor", "calibration", "wide"),
    ("locus", "two_factor", "wide"),
    ("calibration", "medians", "wide"),
    *((a, b, group) for a, b, _ in PRIMARY for group in TEST_GROUPS),
)
_READOUT = ("readout", "readout_1")


def is_deep_run(counts: MeasuredCounts) -> bool:
    return any(c.name == "readout_1" or c.name.startswith("deep_") for c in counts.circuits)


def groups(counts: MeasuredCounts) -> dict[str, list[str]]:
    """The circuit names of each group that the plan scores. A group with no circuit is left out."""
    names = [c.name for c in counts.circuits]
    found = {g: [n for n in names if role(n) == g] for g in TEST_GROUPS}
    found["test"] = [n for n in names if role(n) in TEST_GROUPS]
    found["wide"] = [n for n in names if n.startswith("wide_")]
    found["all"] = names
    return {g: found[g] for g in GROUPS if found[g]}


def chain_kind(path: Path) -> str | None:
    """ "best" or "median" from the stem of the counts file, such as ``best-deep``."""
    stem = path.name.split(".")[0]
    kind = stem.split("-")[0]
    return kind if kind in CHAIN_KINDS else None


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
        """The median over the calibrations in effect at the run time and 1 to 6 days before."""
        if calibration.device.vendor != "ibm":
            return None
        device = calibration.device.name
        earlier = [
            models.pulled(device, counts.run_at - timedelta(days=days), cache, pull)
            for days in range(1, AVERAGED_DAYS)
        ]
        others = [p for p in earlier if p is not None]
        return models.floored(models.averaged(calibration, others)) if others else None

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
) -> dict[str, Any] | None:
    """The derived row of one counts file, or None when the file is not a run of the deep set."""
    counts = load_counts(path)
    if not is_deep_run(counts):
        return None
    calibration = nv.load(str(profile_beside(path)))
    built, missing, fits = build(calibration, counts, cache=cache, offline=offline, pull=pull)
    named = groups(counts)
    forecasts: dict[str, dict[str, Forecast]] = {}
    done: dict[str, Forecast] = {}

    def scored(profile: Profile, group: str) -> Forecast:
        """Each circuit is simulated one time for a profile, and each group selects from it."""
        if profile.fingerprint not in done:
            done[profile.fingerprint] = score(profile, counts)
        return done[profile.fingerprint].select(named[group])

    for name in MODELS:
        if name not in built:
            continue
        try:
            forecasts[name] = {g: scored(built[name], g) for g in named}
        except NoiseVaultError as exc:
            missing[name] = _reason(exc)

    try:
        stated_forecasts = {g: scored(calibration.uncorrected(), g) for g in named}
    except NoiseVaultError as exc:
        missing["calibration (stated)"] = _reason(exc)
        stated_forecasts = {}
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
        "qubits": list(counts.qubits),
        "chain": chain_kind(path),
        "shots": shots.pop() if len(shots) == 1 else None,
        "circuits": {g: len(names) for g, names in named.items()},
        "counts_sha256": counts.sha256,
        "calibration": calibration.calibration_fingerprint,
        "marked": _marked(calibration, counts, cache, offline, pull),
        "ages": {} if offline else _ages(calibration, counts, cache, properties),
        "stated": {
            "zero_readouts": models.zero_readouts(calibration),
            "ruled_out": {g: f.ruled_out for g, f in stated_forecasts.items()},
            "log_score": {g: _finite(f.log_score) for g, f in stated_forecasts.items()},
        },
        "scores": {
            name: {g: _forecast(f) for g, f in by_group.items()}
            for name, by_group in forecasts.items()
        },
        "paired": _comparisons(forecasts),
        "deep": _deep(counts, forecasts),
        "fits": fits,
        "failures": dict(sorted(missing.items(), key=lambda item: _order(item[0]))),
    }


def analyze(
    root: Path,
    *,
    cache: Path | None = None,
    offline: bool = False,
    pull: models.Pull | None = None,
    properties: ages.Properties | None = None,
    rows_folder: Path | None = None,
) -> dict[str, Any]:
    """The analysis of each run of the deep circuit set below ``root``.

    ``rows_folder`` keeps the row of each counts file, so that a later call does not analyze the
    same run again. A counts file with the same counts as an earlier file, for example the same
    run saved as ``.counts.json`` and ``.counts.json.gz``, is a duplicate. Only the first file
    gives a row, and ``duplicates`` gives the number of the other files.
    """
    rows, failures, skipped, duplicates = [], [], 0, 0
    seen: set[str] = set()
    kind = "analysis-offline" if offline else "analysis"
    for path in counts_files(root):

        def run(path: Path = path) -> dict[str, Any] | None:
            return analyze_run(path, cache=cache, offline=offline, pull=pull, properties=properties)

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
    rows.sort(key=lambda r: (r["device"], r["run_at"], r["chain"] or "", r["counts_sha256"]))
    return {
        "nv_study_analysis": FORMAT,
        "rows": rows,
        "skipped": skipped,
        "duplicates": duplicates,
        "failures": failures,
    }


def markdown(data: dict[str, Any]) -> str:
    lines: list[str] = []
    for r in data["rows"]:
        title = f"{r['device']}, {r['chain'] or 'chain'} chain, run {r['run_at']}"
        lines += [f"## {title}", "", "| Model | " + " | ".join(GROUPS) + " |"]
        lines.append("| --- |" + " --- |" * len(GROUPS))
        for name, by_group in r["scores"].items():
            cells = [_score(by_group.get(g)) for g in GROUPS]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        ruled = r["stated"]["ruled_out"].get("all", 0)
        lines += [
            "",
            f"Calibration as stated: {r['stated']['zero_readouts']} readout probabilities of 0,"
            f" {ruled} ruled-out shots.",
        ]
        for name, reason in r["failures"].items():
            lines.append(f"No {name}: {reason}")
        lines.append("")
    for title, primary in (("Primary comparisons", True), ("Secondary comparisons", False)):
        lines += [f"## {title}", ""]
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
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        print(f"error: {args.root} is not a folder", file=sys.stderr)
        return 2
    data = analyze(args.root, cache=args.cache, offline=args.offline, rows_folder=args.rows)
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
) -> dict[str, Any]:
    """For the calibration of the run and for each stale calibration, the age of the
    calibration time and of the values of the chain at the run time, in hours. IBM dates each
    value, so only an IBM device has ages."""
    if calibration.device.vendor != "ibm":
        return {}
    device = calibration.device.name

    def public(name: str, at: datetime) -> Mapping[str, Any] | None:
        return ages.public_properties(name, at, cache)

    fetch = properties or public
    found: dict[str, Any] = {}
    for name, days in (("calibration", 0), *zip(STALE, STALE_DAYS, strict=True)):
        try:
            reply = fetch(device, counts.run_at - timedelta(days=days))
        except NoiseVaultError:
            continue
        if reply is not None:
            found[name] = {
                "calibration_h": ages.calibration_age(reply, counts.run_at),
                "values": ages.value_ages(reply, counts.run_at, counts.qubits),
                "loci": ages.locus_ages(reply, counts.run_at, counts.qubits),
            }
    return found


def _forecast(forecast: Forecast) -> dict[str, Any]:
    data = forecast.to_dict()
    for key in ("circuits", "counts"):
        data.pop(key)
    return data


def _comparisons(forecasts: dict[str, dict[str, Forecast]]) -> list[dict[str, Any]]:
    tests = []
    for primary, comparisons in ((True, PRIMARY), (False, SECONDARY)):
        for a, b, group in comparisons:
            if a not in forecasts or b not in forecasts or group not in forecasts[a]:
                continue
            test = paired(forecasts[a][group], forecasts[b][group])
            data = test.to_dict()
            tests.append(
                {
                    "a": a,
                    "b": b,
                    "group": group,
                    "primary": primary,
                    "winner": {"a": a, "b": b, "neither": None}[test.better],
                    "level": data["level"],
                    "log": data["log"],
                    "quadratic": data["quadratic"],
                    "better": data["better"],
                }
            )
    return tests


def _deep(
    counts: MeasuredCounts, forecasts: dict[str, dict[str, Forecast]]
) -> list[dict[str, Any]]:
    """For each deep circuit, the ideal result, its measured frequency and its probability in
    each model. The ideal result is the most probable result under the ``ideal`` model."""
    if "ideal" not in forecasts:
        return []
    by_model = {
        name: {c.name: c.probabilities for c in by_group["all"].circuits}
        for name, by_group in forecasts.items()
    }
    rows = []
    for c in counts.circuits:
        if not c.name.startswith(DEEP_PREFIXES):
            continue
        index = int(np.argmax(by_model["ideal"][c.name]))
        bits = outcome_bits(index, len(c.qubits))
        rows.append(
            {
                "name": c.name,
                "role": role(c.name),
                "ideal": bits,
                "measured": c.counts.get(bits, 0) / c.shots,
                "models": {name: float(probs[c.name][index]) for name, probs in by_model.items()},
            }
        )
    return rows


def _order(name: str) -> int:
    return MODELS.index(name) if name in MODELS else len(MODELS)


def _reason(exc: Exception) -> str:
    message = exc.message if isinstance(exc, NoiseVaultError) else str(exc)
    return f"{type(exc).__name__}: {message}"


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
