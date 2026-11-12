"""Derive the results of the realism study from a tree of counts files.

    python -m study.derive campaign --out results

The script reads each ``*.counts.json`` file below the folder, with the profile file beside it
(``<stem>.profile.json``, which the run scripts write). It fits the factors of ``nv compare`` on
each file and writes ``fits.json`` and ``fits.md`` to the output folder. It also writes the same
fits as ``table.json`` and ``table.md``, in the format of ``scripts/build_realism_table.py``. For
a run of the deep circuit set, the fit uses only the ``train`` circuits (see ``PLAN.md``).

The output holds derived numbers only: fitted factors, intervals, distances and tests. A row holds
no counts, no calibration value, no file name and no job id. The SHA-256 of each counts file and
the fingerprint of each calibration identify the data. A file that cannot be fitted gives a
failure entry with the type and the message of the error.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import noisevault as nv
from noisevault.counts import MeasuredCounts, load_counts
from noisevault.errors import NoiseVaultError
from noisevault.suite import role

FORMAT = "1"
COUNTS_SUFFIXES = (".counts.json", ".counts.json.gz")
TABLE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_realism_table.py"


def counts_files(root: Path) -> Iterator[Path]:
    """Each counts file below ``root``, in a fixed order. AppleDouble files are not counts."""
    for path in sorted(root.rglob("*")):
        if path.name.startswith("._") or not path.is_file():
            continue
        if path.name.endswith(COUNTS_SUFFIXES):
            yield path


def profile_beside(path: Path) -> Path:
    stem = path.name
    for suffix in COUNTS_SUFFIXES:
        stem = stem.removesuffix(suffix)
    return path.with_name(stem + ".profile.json")


def fit_counts(counts: MeasuredCounts) -> tuple[MeasuredCounts, str]:
    """The counts that a fit can use. A run of the deep circuit set gives only its ``train``
    circuits, so that no fit sees a test circuit."""
    names = [c.name for c in counts.circuits]
    if not any(name == "readout_1" or name.startswith("deep_") for name in names):
        return counts, "all"
    return counts.select([name for name in names if role(name) == "train"]), "train"


def fit(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """The derived row of one counts file, and its row of the realism table."""
    run = load_counts(path)
    counts, fit_on = fit_counts(run)
    profile = nv.load(str(profile_beside(path)))
    result = profile.compare(counts)
    device = profile.device
    row = {
        "vendor": device.vendor,
        "technology": device.technology,
        "calibrated_at": None if device.calibrated_at is None else device.calibrated_at.isoformat(),
        "fit_on": fit_on,
        "run_counts": run.sha256,
        **result.to_dict(),
    }
    table_row = {**_table_script().row(counts, result), "fit_on": fit_on, "run_counts": run.sha256}
    return row, table_row


def derive(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """The fits of each counts file below ``root``, and the realism table of the same fits."""
    rows, table_rows, failures = [], [], []
    for path in counts_files(root):
        try:
            row, table_row = fit(path)
        except NoiseVaultError as exc:
            # The message of a counts or profile error names fields, not values of the account.
            failures.append({"error": type(exc).__name__, "message": exc.message})
            continue
        rows.append(row)
        table_rows.append(table_row)
    rows.sort(key=lambda r: (r["profile_id"], r["run_at"], r["counts"]))
    table_rows.sort(key=lambda r: (r["device"], r["run_at"], r["counts"]))
    script = _table_script()
    table = {
        "realism_table": script.TABLE_FORMAT,
        "note": " ".join(script.NOTE),
        "rows": table_rows,
    }
    return {"nv_study_fits": FORMAT, "rows": rows, "failures": failures}, table


def _table_script() -> Any:
    """``scripts/build_realism_table.py`` as a module, for the format of the realism table."""
    name = "build_realism_table"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, TABLE_SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def markdown(data: dict[str, Any]) -> str:
    lines = [
        "| Device | Run | Age (h) | Qubits | Gate factor | Readout factor | p | Dispersion |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in data["rows"]:
        age = r["calibration_age_hours"]
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {:.2f} |".format(
                r["profile_id"],
                r["run_at"],
                "-" if age is None else f"{age:.1f}",
                "-".join(str(q) for q in r["qubits"]),
                _factor(r["gates"]),
                _factor(r["readout"]),
                "-" if r["p_value"] is None else f"{r['p_value']:.3g}",
                r["dispersion"],
            )
        )
    lines += ["", "| Device | Run | Circuit | Shots | Profile TVD | Fitted TVD | Noise TVD 95% |"]
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in data["rows"]:
        for c in r["circuits"]:
            lines.append(
                f"| {r['profile_id']} | {r['run_at']} | {c['name']} | {c['shots']} |"
                f" {c['tvd_profile']:.4f} | {c['tvd_fitted']:.4f} | {c['shot_noise_95']:.4f} |"
            )
    for failure in data["failures"]:
        lines += ["", f"Failed: {failure['error']}: {failure['message']}"]
    return "\n".join(lines) + "\n"


def _factor(found: dict[str, Any]) -> str:
    if found.get("factor") is None:
        return "not identified"
    low, high = found.get("low"), found.get("high")
    low_text = "..." if low is None else f"{low:.3g}"
    high_text = "..." if high is None else f"{high:.3g}"
    return f"x{found['factor']:.3g} ({low_text} to {high_text})"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", type=Path, help="the folder with the counts files")
    parser.add_argument("--out", type=Path, default=Path("results"), help="the output folder")
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        print(f"error: {args.root} is not a folder", file=sys.stderr)
        return 2
    data, table = derive(args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fits.json").write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    (args.out / "fits.md").write_text(markdown(data), encoding="utf-8")
    if table["rows"]:
        for path, text in _table_script().write(table, args.out / "table", "Realism table").items():
            path.write_bytes(text)
    print(f"wrote {len(data['rows'])} rows and {len(data['failures'])} failures to {args.out}")
    return 0 if data["rows"] and not data["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
