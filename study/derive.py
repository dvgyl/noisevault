"""Derive the results of the realism study from a tree of counts files.

    python -m study.derive campaign --out results

The script reads each ``*.counts.json`` file below the folder, with the profile file beside it
(``<stem>.profile.json``, which the run scripts write). It fits the factors of ``nv compare`` on
each file and writes ``fits.json`` and ``fits.md`` to the output folder.

The output holds derived numbers only: fitted factors, intervals, distances and tests. A row holds
no counts, no calibration value, no file name and no job id. The SHA-256 of each counts file and
the fingerprint of each calibration identify the data. A file that cannot be fitted gives a
failure entry with the type and the message of the error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import noisevault as nv
from noisevault.counts import load_counts
from noisevault.errors import NoiseVaultError

FORMAT = "1"
COUNTS_SUFFIXES = (".counts.json", ".counts.json.gz")


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


def fit(path: Path) -> dict[str, Any]:
    """The derived row of one counts file."""
    counts = load_counts(path)
    profile = nv.load(str(profile_beside(path)))
    row = profile.compare(counts).to_dict()
    device = profile.device
    return {
        "vendor": device.vendor,
        "technology": device.technology,
        "calibrated_at": None if device.calibrated_at is None else device.calibrated_at.isoformat(),
        **row,
    }


def derive(root: Path) -> dict[str, Any]:
    rows, failures = [], []
    for path in counts_files(root):
        try:
            rows.append(fit(path))
        except NoiseVaultError as exc:
            # The message of a counts or profile error names fields, not values of the account.
            failures.append({"error": type(exc).__name__, "message": exc.message})
    rows.sort(key=lambda r: (r["profile_id"], r["run_at"], r["counts"]))
    return {"nv_study_fits": FORMAT, "rows": rows, "failures": failures}


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
    data = derive(args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fits.json").write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    (args.out / "fits.md").write_text(markdown(data), encoding="utf-8")
    print(f"wrote {len(data['rows'])} rows and {len(data['failures'])} failures to {args.out}")
    return 0 if data["rows"] and not data["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
