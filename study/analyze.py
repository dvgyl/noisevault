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
import json
import math
import sys
from collections.abc import Callable, Sequence
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
from study import models
from study.derive import counts_files, profile_beside

FORMAT = "1"
STALE_DAYS = (1, 7, 30, 90, 180, 365)
GROUPS = ("test_near", "test_far", "all")
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
    "calibration",
    "two_factor",
    "readout_measured",
    "locus",
)
# The primary comparisons of the plan: model a, model b and the circuit groups of the test.
COMPARISONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("calibration", "medians", TEST_GROUPS),
    *(("calibration", name, GROUPS) for name in (*STALE, "bundled")),
    ("calibration", "symmetric", TEST_GROUPS),
    ("two_factor", "calibration", TEST_GROUPS),
    ("readout_measured", "two_factor", TEST_GROUPS),
    ("locus", "readout_measured", TEST_GROUPS),
)
_READOUT = ("readout", "readout_1")


def is_deep_run(counts: MeasuredCounts) -> bool:
    return any(c.name == "readout_1" or c.name.startswith("deep_") for c in counts.circuits)


def groups(counts: MeasuredCounts) -> dict[str, list[str]]:
    """The circuit names of each group that the plan scores. A group with no circuit is left out."""
    names = [c.name for c in counts.circuits]
    found = {g: [n for n in names if role(n) == g] for g in TEST_GROUPS}
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
    that is not built, and the fitted values."""
    train = counts.select([c.name for c in counts.circuits if role(c.name) == "train"])
    base = models.floored(calibration)
    built: dict[str, Profile] = {}
    missing: dict[str, str] = {}
    fits: dict[str, Any] = {}

    def attempt(name: str, make: Callable[[], Profile | None], absent: str = "") -> None:
        try:
            found = make()
        except (NoiseVaultError, ValueError) as exc:
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

    def measured() -> Profile:
        rates = models.readout_measured(base, train)
        others = [c.name for c in train.circuits if c.name not in _READOUT]
        factor = models.gate_factor(rates, train, others)
        chain = next(c.qubits for c in train.circuits if c.name == "readout")
        fits["readout_measured"] = {
            "readout": models.readout_rates(rates, chain),
            "gates": factor,
        }
        return models.with_factors(rates, factor)

    def per_locus() -> Profile:
        fitted, _, rows = models.locus(base, train)
        fits["locus"] = rows
        return fitted

    attempt("ideal", lambda: models.ideal(calibration))
    attempt("generic", lambda: models.generic(calibration))
    attempt("medians", lambda: models.medians(base))
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
    attempt("symmetric", lambda: models.symmetric(base))
    attempt("calibration", lambda: base)
    attempt("two_factor", two)
    attempt("readout_measured", measured)
    attempt("locus", per_locus)
    return built, missing, fits


def analyze_run(
    path: Path, *, cache: Path | None, offline: bool, pull: models.Pull | None = None
) -> dict[str, Any] | None:
    """The derived row of one counts file, or None when the file is not a run of the deep set."""
    counts = load_counts(path)
    if not is_deep_run(counts):
        return None
    calibration = nv.load(str(profile_beside(path)))
    built, missing, fits = build(calibration, counts, cache=cache, offline=offline, pull=pull)
    named = groups(counts)
    forecasts: dict[str, dict[str, Forecast]] = {}
    done: dict[tuple[str, str], Forecast] = {}

    def scored(profile: Profile, group: str) -> Forecast:
        key = (profile.fingerprint, group)
        if key not in done:
            done[key] = score(profile, counts, circuits=named[group])
        return done[key]

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
    root: Path, *, cache: Path | None = None, offline: bool = False, pull: models.Pull | None = None
) -> dict[str, Any]:
    rows, failures, skipped = [], [], 0
    for path in counts_files(root):
        try:
            row = analyze_run(path, cache=cache, offline=offline, pull=pull)
        except NoiseVaultError as exc:
            # The message of a counts or profile error names fields, not values of the account.
            failures.append({"error": type(exc).__name__, "message": exc.message})
            continue
        if row is None:
            skipped += 1
        else:
            rows.append(row)
    rows.sort(key=lambda r: (r["device"], r["run_at"], r["chain"] or "", r["counts_sha256"]))
    return {"nv_study_analysis": FORMAT, "rows": rows, "skipped": skipped, "failures": failures}


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
    lines += ["## Primary comparisons", ""]
    lines.append("| Device | Chain | Run | A | B | Group | Better | z | p |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in data["rows"]:
        for test in r["paired"]:
            log = test["log"]
            lines.append(
                f"| {r['device']} | {r['chain'] or '-'} | {r['run_at']} | {test['a']} |"
                f" {test['b']} | {test['group']} | {test['winner'] or 'neither'} |"
                f" {_number(log['z'], '.2f')} | {_number(log['p_value'], '.3g')} |"
            )
    for failure in data["failures"]:
        lines += ["", f"Failed: {failure['error']}: {failure['message']}"]
    return "\n".join(lines) + "\n"


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
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        print(f"error: {args.root} is not a folder", file=sys.stderr)
        return 2
    data = analyze(args.root, cache=args.cache, offline=args.offline)
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


def _forecast(forecast: Forecast) -> dict[str, Any]:
    data = forecast.to_dict()
    for key in ("circuits", "counts"):
        data.pop(key)
    return data


def _comparisons(forecasts: dict[str, dict[str, Forecast]]) -> list[dict[str, Any]]:
    tests = []
    for a, b, wanted in COMPARISONS:
        if a not in forecasts or b not in forecasts:
            continue
        for group in wanted:
            if group not in forecasts[a]:
                continue
            test = paired(forecasts[a][group], forecasts[b][group])
            data = test.to_dict()
            tests.append(
                {
                    "a": a,
                    "b": b,
                    "group": group,
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
