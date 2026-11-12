"""Derive the results of the realism study from a tree of counts files.

    python -m study.derive campaign --out results

The script reads each ``*.counts.json`` file below the folder, with the profile file beside it
(``<stem>.profile.json``, which the run scripts write). It fits the factors of ``nv compare`` on
each file and writes ``fits.json`` and ``fits.md`` to the output folder. It also writes the same
fits as ``table.json`` and ``table.md``, in the format of ``scripts/build_realism_table.py``. For
a run of the deep circuit set, the fit uses only the ``train`` circuits (see ``PLAN.md``).

The realism table leaves out each run of version 1 of the deep circuit set. The 1-qubit circuits
of that version held only gates about one axis, so a small systematic rotation error added up
in them, and the fitted factors do not give the size of the errors in other circuits. The fits
of these runs stay in ``fits.json``, with ``circuit_set`` 1.

The output holds derived numbers only: fitted factors, intervals, distances and tests. A row holds
no counts, no calibration value, no file name and no job id. The SHA-256 of each counts file and
the fingerprint of each calibration identify the data. A file that cannot be fitted gives a
failure entry with the type of the error and the SHA-256 of the file. The entry holds no error
message, because a message can hold a file name.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing
import os
import sys
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import noisevault as nv
from noisevault.counts import MeasuredCounts, load_counts
from noisevault.errors import NoiseVaultError
from noisevault.suite import train_names
from study import kept

FORMAT = "1"
# Version 2 of the deep circuit set has the probe circuits axis_*. A run with a design for a
# budget can leave them out, so the date decides for such a run. No version 1 run exists from
# this date.
VERSION_2_FROM = datetime(2026, 11, 16, tzinfo=UTC)
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
    train = train_names([c.name for c in counts.circuits])
    return (counts, "all") if train is None else (counts.select(train), "train")


def circuit_set(counts: MeasuredCounts) -> int | None:
    """The version of the deep circuit set of a run, or None for a run of the ``nv compare``
    circuits alone."""
    names = [c.name for c in counts.circuits]
    if train_names(names) is None:
        return None
    if any(name.startswith("axis_") for name in names) or counts.run_at >= VERSION_2_FROM:
        return 2
    return 1


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
        "circuit_set": circuit_set(run),
        "run_counts": run.sha256,
        **result.to_dict(),
    }
    table_row = {**_table_script().row(counts, result), "fit_on": fit_on, "run_counts": run.sha256}
    return row, table_row


def _kept_fit(task: tuple[Path, Path | None]) -> tuple[Any, dict[str, str] | None]:
    """The kept or new fit of one counts file, or its failure."""
    path, rows_folder = task
    try:
        beside = profile_beside(path)
        inputs = [path, beside] if beside.exists() else [path]
        return kept(rows_folder, "fit", inputs, lambda: fit(path)), None
    except NoiseVaultError as exc:
        # A message can hold the path of the file, and a file name can hold an id of the
        # account. So a failure gives only the error type and the hash of the file.
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return None, {"error": type(exc).__name__, "file": f"sha256:{digest}"}


def derive(
    root: Path, rows_folder: Path | None = None, jobs: int = 1
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The fits of each counts file below ``root``, and the realism table of the same fits.

    ``rows_folder`` keeps the fit of each counts file, so that a later call does not fit the
    same file again. With ``jobs`` above 1, that many processes fit files at the same time. The
    result does not depend on ``jobs``.
    """
    rows, table_rows, failures = [], [], []
    tasks = [(path, rows_folder) for path in counts_files(root)]
    if jobs > 1 and len(tasks) > 1:
        # A new process that starts from a copy of this one can stop for ever when a library holds
        # a lock at that moment. A process that starts fresh has no such copy.
        fresh = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(jobs, len(tasks)), mp_context=fresh) as pool:
            found = list(pool.map(_kept_fit, tasks))
    else:
        found = [_kept_fit(task) for task in tasks]
    for pair, failure in found:
        if failure is not None:
            failures.append(failure)
            continue
        row, table_row = pair
        rows.append(row)
        if row["circuit_set"] != 1:
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
        lines += ["", f"Failed: {failure['error']} on the counts file {failure['file'][:19]}"]
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
    parser.add_argument(
        "--rows", type=Path, default=None, help="a folder that keeps the fit of each counts file"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="the number of processes that fit files at the same time (default: each CPU)",
    )
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        print(f"error: {args.root} is not a folder", file=sys.stderr)
        return 2
    data, table = derive(args.root, args.rows, args.jobs)
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
