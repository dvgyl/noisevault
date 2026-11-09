r"""Write a drift report from the folders of the weekly IBM campaign.

    python scripts/drift_report.py

The script reads ``campaign/<device>/<date>/``, which ``scripts/ibm_campaign.py`` writes. For
each device, it does these steps:

1. For each date and chain, it fits the error factors as ``nv compare`` does, on the counts file
   and the calibration that the counts bind to.
2. For each two consecutive dates, it compares the two calibrations as ``nv diff`` does.

The script writes ``drift.json`` and ``drift.md`` in the campaign folder. The JSON file gives the
gate and readout factors with their 95% intervals for each date and chain, and the largest
calibration changes between consecutive dates. The Markdown file shows the same data as tables.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import noisevault as nv  # noqa: E402
from noisevault.compare import compare  # noqa: E402
from noisevault.counts import load_counts  # noqa: E402
from noisevault.diff import diff  # noqa: E402
from noisevault.errors import NoiseVaultError  # noqa: E402
from noisevault.profile import ErrorFactor, Profile, iso_z  # noqa: E402

CHAINS = ("best", "median")
CALIBRATION = "calibration.json"
REPORT_FORMAT = "1.0"
TOP = 5
FIT_KEYS = (
    "profile_id",
    "fingerprint",
    "calibration",
    "counts",
    "source",
    "run_at",
    "calibration_age_hours",
    "qubits",
    "gates",
    "readout",
    "dispersion",
    "p_value",
    "impossible_shots",
)


def report(root: Path, devices: Sequence[str] | None = None) -> dict[str, Any]:
    """The drift report of every device folder under ``root``, or of ``devices`` only."""
    if devices is None:
        devices = sorted(p.name for p in root.iterdir() if p.is_dir() and _weeks(p))
    return {
        "nv_drift": REPORT_FORMAT,
        "generated_at": iso_z(datetime.now(UTC).replace(microsecond=0)),
        "devices": {device: _device(root / device) for device in devices},
    }


def _device(folder: Path) -> dict[str, Any]:
    weeks = _weeks(folder)
    dates = [_date(week) for week in weeks]
    changes = []
    for before, after in zip(weeks, weeks[1:], strict=False):
        changes.append(_change(before, after))
    return {"dates": dates, "changes": changes}


def _date(week: Path) -> dict[str, Any]:
    entry: dict[str, Any] = {"date": week.name, "calibration": None}
    try:
        entry["calibration"] = _load(week / CALIBRATION).calibration_fingerprint
    except NoiseVaultError as exc:
        entry["calibration_error"] = exc.message
    entry["chains"] = {kind: _fit(week, kind) for kind in CHAINS}
    return entry


def _load(path: Path) -> Profile:
    """The profile in ``path``. A missing, unreadable or damaged file raises NoiseVaultError."""
    if not path.exists():
        raise NoiseVaultError(f"no file {path.name}")
    try:
        return nv.load(path)
    except NoiseVaultError as exc:
        raise NoiseVaultError(f"{path.name}: {exc.message}") from None
    except (OSError, ValueError) as exc:
        raise NoiseVaultError(f"could not read {path.name} ({type(exc).__name__})") from None


def _fit(week: Path, kind: str) -> dict[str, Any]:
    counts_file = week / f"{kind}.counts.json"
    profile_file = week / f"{kind}.profile.json"
    if not counts_file.exists():
        state = "submitted" if (week / f"{kind}.job.json").exists() else "missing"
        return {"state": state}
    try:
        if not profile_file.exists():
            raise NoiseVaultError(f"no profile file {profile_file.name} beside the counts file")
        profile = _load(profile_file)
        try:
            counts = load_counts(counts_file)
        except (OSError, ValueError) as exc:
            raise NoiseVaultError(
                f"could not read {counts_file.name} ({type(exc).__name__})"
            ) from None
        fitted = compare(profile, counts).to_dict()
    except NoiseVaultError as exc:
        return {"state": "error", "error": exc.message}
    return {"state": "fitted", **{key: fitted[key] for key in FIT_KEYS}}


def _change(before: Path, after: Path) -> dict[str, Any]:
    span = {"from": before.name, "to": after.name}
    try:
        a, b = (_load(week / CALIBRATION) for week in (before, after))
    except NoiseVaultError as exc:
        return {**span, "error": exc.message}
    found = diff(a, b, top=TOP)
    return {**span, **found.to_dict(), "summary": found.summary()}


def _weeks(folder: Path) -> list[Path]:
    weeks = []
    for path in folder.iterdir():
        try:
            if path.is_dir() and date.fromisoformat(path.name).isoformat() == path.name:
                weeks.append(path)
        except ValueError:
            continue
    return sorted(weeks)


def markdown(data: dict[str, Any]) -> str:
    lines = ["# Drift report", "", f"Generated {data['generated_at']}."]
    for device, found in data["devices"].items():
        lines += ["", f"## {device}", "", "### Error factors", ""]
        lines += [
            "| Date | Chain | Qubits | Gate factor | Readout factor | Fit p |",
            "|---|---|---|---|---|---|",
        ]
        for entry in found["dates"]:
            for kind, fit in entry["chains"].items():
                lines.append(_fit_row(entry["date"], kind, fit))
        lines += ["", "### Calibration changes"]
        if not found["dates"]:
            lines += ["", "No dates yet."]
        elif not found["changes"]:
            lines += ["", "One date only, so there is nothing to compare."]
        for change in found["changes"]:
            title = f"{change['from']} to {change['to']}"
            if "error" in change:
                lines += ["", f"#### {title}", "", f"Not compared: {change['error']}."]
                continue
            lines += ["", f"#### {title}", "", "```text", change["summary"], "```"]
    return "\n".join(lines) + "\n"


def _fit_row(when: str, kind: str, fit: dict[str, Any]) -> str:
    if fit["state"] != "fitted":
        detail = fit.get("error", fit["state"])
        return f"| {when} | {kind} | | {_cell(detail)} | | |"
    qubits = "-".join(map(str, fit["qubits"]))
    p = "" if fit["p_value"] is None else f"{fit['p_value']:.2g}"
    gate, readout = _factor(fit["gates"]), _factor(fit["readout"])
    return f"| {when} | {kind} | {qubits} | {gate} | {readout} | {p} |"


def _factor(found: dict[str, Any]) -> str:
    if "not_identified" in found:
        return _cell(f"not identified: {found['not_identified']}")
    return ErrorFactor.model_validate(found).describe()


def _cell(text: str) -> str:
    for old, new in (("\\", "\\\\"), ("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"), ("|", "\\|")):
        text = text.replace(old, new)
    return text.replace("\n", " ")


def write(data: dict[str, Any], folder: Path) -> tuple[Path, Path]:
    paths = folder / "drift.json", folder / "drift.md"
    texts = json.dumps(data, indent=1) + "\n", markdown(data)
    for path, text in zip(paths, texts, strict=True):
        part = path.with_name(f"{path.name}.part")
        part.write_text(text, encoding="utf-8")
        os.replace(part, path)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.root.is_dir():
        print(f"error: no campaign folder {args.root}", file=sys.stderr)
        print("hint: run scripts/ibm_campaign.py first, or give --root", file=sys.stderr)
        return 1
    missing = [d for d in args.devices if not (args.root / d).is_dir()]
    if missing:
        print(f"error: no folder for {', '.join(missing)} in {args.root}", file=sys.stderr)
        return 1
    data = report(args.root, args.devices or None)
    output = args.output or args.root
    output.mkdir(parents=True, exist_ok=True)
    for path in write(data, output):
        print(f"wrote {path}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drift_report.py",
        description="Write a drift report from the folders of the weekly IBM campaign.",
    )
    parser.add_argument(
        "devices", metavar="DEVICE", nargs="*", help="devices to report (default: every device)"
    )
    parser.add_argument(
        "--root",
        metavar="FOLDER",
        type=Path,
        default=Path("campaign"),
        help="the campaign folder (default campaign)",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FOLDER",
        type=Path,
        help="the folder for drift.json and drift.md (default: the campaign folder)",
    )
    return parser


if __name__ == "__main__":
    sys.exit(main())
