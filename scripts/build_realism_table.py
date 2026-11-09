"""Build the realism table from a folder of counts files.

The table has one row for each counts file. A row gives the device, its vendor and technology,
the date of the run, the gate factor and the readout factor that ``nv compare`` fits, each with
its 95% interval, and the goodness of fit. The script writes the table as JSON and as Markdown.
Run from the repository root:

    python scripts/build_realism_table.py COUNTS_FOLDER -o docs/realism/table
    python scripts/build_realism_table.py --demo          # rebuild docs/realism/demo.*
    python scripts/build_realism_table.py --demo --check  # exit 1 if the demo is out of date

``-o docs/realism/table`` writes ``docs/realism/table.json`` and ``docs/realism/table.md``. For
each counts file, the script uses the profile file beside it (``<stem>.profile.json``, which the
run scripts write) or else the catalog profile with the calibration fingerprint of the counts.

The table holds no counts file name, job id or task id, because a name or an id can hold the id
of an account. The SHA-256 of each counts file identifies it.

The demo simulates counts from bundled profiles with known factors. Every demo row is
"simulated", and the row gives the factors that the simulation used.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import noisevault as nv  # noqa: E402
from noisevault.compare import NOTE, Comparison, FactorResult, NoEstimate  # noqa: E402
from noisevault.counts import MeasuredCounts, load_counts, plan, simulate  # noqa: E402
from noisevault.errors import NoiseVaultError  # noqa: E402
from noisevault.profile import POOR_FIT_P_VALUE, ErrorFactor, Profile, iso_z  # noqa: E402

TABLE_FORMAT = "1"
DEMO = ROOT / "docs" / "realism" / "demo"
DEMO_SHOTS = 4000
# (catalog ref, run time, gate factor, readout factor, seed) of each simulated demo run.
DEMO_RUNS = (
    ("ibm_kingston@2026-04-15", datetime(2026, 4, 16, 9, 30, tzinfo=UTC), 1.6, 1.2, 1),
    ("ibm_kingston@2026-04-15", datetime(2026, 4, 23, 9, 30, tzinfo=UTC), 2.2, 1.3, 2),
    ("quantinuum_h1-1@2025-05-02", datetime(2025, 5, 3, 14, 0, tzinfo=UTC), 1.3, 1.0, 3),
    ("google_weber@2021-11-03", datetime(2021, 11, 4, 8, 0, tzinfo=UTC), 1.1, 1.5, 4),
)
COLUMNS = (
    "Device",
    "Vendor",
    "Technology",
    "Date",
    "Gate factor (95% interval)",
    "Readout factor (95% interval)",
    "Fit",
    "Source",
)


def counts_files(folder: Path) -> list[Path]:
    """The counts files in ``folder``, by name. AppleDouble files (``._*``) are not counts."""
    found = [*folder.glob("*.counts.json"), *folder.glob("*.counts.json.gz")]
    return sorted(p for p in found if not p.name.startswith("._"))


def profile_for(path: Path, counts: MeasuredCounts) -> Profile:
    """The profile file beside the counts, or the catalog profile with their calibration."""
    stem = path.name.removesuffix(".gz").removesuffix(".json").removesuffix(".counts")
    beside = path.with_name(stem + ".profile.json")
    if beside.exists():
        return nv.load(str(beside))
    want = counts.profile
    for info in nv.profiles():
        if info.id != want.id:
            continue
        found = info.load()
        if found.calibration_fingerprint == want.fingerprint:
            return found.uncorrected()
    raise NoiseVaultError(
        f"{path}: no profile {want.id} nv:{want.fingerprint[:12]}",
        hint=f"put the profile file beside the counts as {beside.name}, or pull the calibration",
    )


def row(counts: MeasuredCounts, result: Comparison) -> dict[str, Any]:
    profile = result.profile
    device = profile.device
    simulated_with = None
    options = counts.execution.options
    if counts.source == "simulated" and "unmodeled_error" in options:
        # noisevault.counts.simulate records the factors it used, or null for none.
        block = options["unmodeled_error"] or {}
        simulated_with = {
            axis: (block.get(axis) or {}).get("factor", 1.0) for axis in ("gates", "readout")
        }
    return {
        "device": profile.id,
        "vendor": device.vendor,
        "technology": device.technology,
        "date": f"{counts.run_at:%Y-%m-%d}",
        "run_at": iso_z(counts.run_at),
        "source": counts.source,
        "simulated_with": simulated_with,
        "calibration": {
            "id": profile.id,
            "fingerprint": profile.calibration_fingerprint,
            "calibrated_at": None if device.calibrated_at is None else iso_z(device.calibrated_at),
        },
        "qubits": list(counts.qubits),
        "shots": sum(c.shots for c in counts.circuits),
        "gates": _factor(result.gates),
        "readout": _factor(result.readout),
        "p_value": _rounded(result.p_value),
        "dispersion": _rounded(result.dispersion),
        "deviance": _rounded(result.deviance),
        "dof": result.dof,
        "counts": counts.sha256,
    }


def build(folder: Path) -> dict[str, Any]:
    paths = counts_files(folder)
    if not paths:
        raise NoiseVaultError(
            f"no counts files in {folder}", hint="give a folder with .counts.json files"
        )
    rows = []
    for path in paths:
        counts = load_counts(path)
        result = profile_for(path, counts).compare(counts)
        rows.append(row(counts, result))
    rows.sort(key=lambda r: (r["device"], r["run_at"], r["counts"]))
    return {"realism_table": TABLE_FORMAT, "note": " ".join(NOTE), "rows": rows}


def markdown(table: dict[str, Any], title: str) -> str:
    rows = table["rows"]
    lines = [
        f"# {title}",
        "",
        "Each row scores a profile on counts from one run with `nv compare`. A factor multiplies",
        "the error rates of the profile, so x2 means about twice the errors. A factor of 1 means",
        "that the calibration explains the counts. The fit column gives the p-value of the",
        f"goodness-of-fit test. A p-value below {POOR_FIT_P_VALUE} means that one gate factor and",
        "one readout factor do not explain every circuit.",
        "",
    ]
    if rows and all(r["source"] == "simulated" for r in rows):
        lines += [
            "Every row is simulated. The counts come from a profile with known factors, not from",
            "a device. The source column gives the factors that the simulation used.",
            "",
        ]
    lines += [
        "| " + " | ".join(COLUMNS) + " |",
        "| " + " | ".join("---" for _ in COLUMNS) + " |",
    ]
    for r in rows:
        cells = (
            f"`{r['device']}`",
            r["vendor"] or "unknown",
            r["technology"].replace("_", " "),
            r["date"],
            _cell(r["gates"]),
            _cell(r["readout"]),
            _fit(r["p_value"]),
            _source(r),
        )
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "The JSON file beside this page holds each value with six significant digits."]
    return "\n".join(lines) + "\n"


def demo(folder: Path) -> None:
    """Simulated counts files for the demo, written to ``folder``."""
    for i, (ref, run_at, gate, readout, seed) in enumerate(DEMO_RUNS):
        profile = nv.load(ref)
        truth = profile.model_copy(
            update={"unmodeled_error": {"gates": {"factor": gate}, "readout": {"factor": readout}}}
        )
        counts = simulate(truth, plan(profile), shots=DEMO_SHOTS, seed=seed, run_at=run_at)
        counts.save(folder / f"{i + 1:02d}-{profile.id}-{run_at:%Y%m%d}.counts.json")


def write(table: dict[str, Any], output: Path, title: str) -> dict[Path, bytes]:
    return {
        output.with_suffix(".json"): (json.dumps(table, indent=1) + "\n").encode("utf-8"),
        output.with_suffix(".md"): markdown(table, title).encode("utf-8"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.demo == (args.folder is not None):
        print("error: give a counts folder or --demo", file=sys.stderr)
        return 2
    if args.check and not args.demo:
        print("error: --check works only with --demo", file=sys.stderr)
        return 2
    try:
        if args.demo:
            with tempfile.TemporaryDirectory() as tmp:
                demo(Path(tmp))
                table = build(Path(tmp))
            files = write(table, args.output or DEMO, "Realism table demo")
        else:
            table = build(args.folder)
            output = args.output or args.folder / "realism"
            files = write(table, output, "Realism table")
    except NoiseVaultError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.hint:
            print(f"hint: {exc.hint}", file=sys.stderr)
        return 1
    if args.check:
        stale = [p for p, data in files.items() if not p.exists() or p.read_bytes() != data]
        for path in stale:
            print(f"out of date: {path}", file=sys.stderr)
        if stale:
            print("hint: run python scripts/build_realism_table.py --demo", file=sys.stderr)
        return 1 if stale else 0
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"wrote {path}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_realism_table.py",
        description="Fit nv compare on each counts file and write the realism table.",
    )
    parser.add_argument("folder", metavar="FOLDER", nargs="?", type=Path, help="counts files")
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        type=Path,
        help="write PATH.json and PATH.md (default FOLDER/realism, or docs/realism/demo)",
    )
    parser.add_argument("--demo", action="store_true", help="build the table of simulated runs")
    parser.add_argument("--check", action="store_true", help="exit 1 if the demo is out of date")
    return parser


def _factor(result: FactorResult) -> dict[str, Any]:
    if isinstance(result, NoEstimate):
        return {"not_identified": result.reason}
    return {
        "factor": _rounded(result.factor),
        "low": _rounded(result.low),
        "high": _rounded(result.high),
        "bound": result.bound,
    }


def _rounded(value: float | None) -> float | None:
    """Six significant digits, so a difference in the last bits of a fit does not show."""
    return None if value is None else float(f"{value:.6g}")


def _cell(factor: dict[str, Any]) -> str:
    if "not_identified" in factor:
        return "not identified"
    shown = ErrorFactor.model_validate(factor).describe()
    return shown.replace(" (95% interval, ", " (").replace(" (95% interval ", " (")


def _fit(p_value: float | None) -> str:
    if p_value is None:
        return "no test"
    text = f"p = {p_value:.2g}"
    return text if p_value >= POOR_FIT_P_VALUE else f"{text}, poor"


def _source(r: dict[str, Any]) -> str:
    used = r["simulated_with"]
    if r["source"] != "simulated":
        return r["source"]
    if used is None:
        return "simulated, factors unknown"
    return f"simulated, x{used['gates']:g} gates, x{used['readout']:g} readout"


if __name__ == "__main__":
    sys.exit(main())
