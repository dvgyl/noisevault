"""Build the project website from site/index.html into _site/index.html.

The page takes its numbers from the bundled profiles, CITATION.cff, the outputs that README.md
documents and the real output of nv show and nv cite. The build stops if a number in the page
text does not match those sources.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import noisevault as nv
from noisevault.profile import gate_stats, qubit_medians

ROOT = Path(__file__).resolve().parents[1]
INDEX = json.loads((Path(nv.__file__).parent / "data" / "profiles" / "index.json").read_text())
NAMES = {"qiskit": "Qiskit", "cirq": "Cirq", "pennylane": "PennyLane", "stim": "Stim"}


def two_qubit_error(profile: nv.Profile) -> float | None:
    errors = [
        gate_stats(profile, name).median_error
        for name in profile.gates
        if profile.table.arity(name) == 2
    ]
    errors = [e for e in errors if e is not None]
    return min(errors) if errors else None


def chip() -> dict:
    """Readout error and layout of every ibm_fez qubit, for the chip map."""
    profile = nv.load("ibm_fez")
    edges = [list(e) for e in profile.connectivity.edges]
    # Heron r2: rows of 16 qubits, each followed by 4 bridge qubits to the next row.
    pos: dict[int, list[int]] = {}
    for i in range(profile.device.num_qubits):
        row, k = divmod(i, 20)
        if k < 16:
            pos[i] = [k, 2 * row]
    for i in range(profile.device.num_qubits):
        row, k = divmod(i, 20)
        if k >= 16:
            above = next(
                b if a == i else a
                for a, b in edges
                if i in (a, b) and (b if a == i else a) // 20 == row
            )
            pos[i] = [pos[above][0], 2 * row + 1]
    return {
        "readout": [
            round((q.readout.p1_given_0 + q.readout.p0_given_1) / 2 * 100, 4)
            for q in profile.qubits
        ],
        "edges": edges,
        "pos": [pos[i] for i in sorted(pos)],
        "fingerprint": profile.short_fingerprint,
        "medianReadout": round(qubit_medians(profile).readout_error * 100, 3),
    }


def fingerprint_change() -> dict:
    """The fingerprint of ibm_fez before and after one cz error grows by 1%."""
    profile = nv.load("ibm_fez")
    data = profile.to_dict()
    record = next(c for c in data["calibrations"] if c["gate"] == "cz" and c.get("avg_infidelity"))
    before = record["avg_infidelity"]
    record["avg_infidelity"] = before * 1.01
    changed = nv.Profile.from_dict(data)
    return {
        "qubits": record["qubits"],
        "before": before,
        "after": record["avg_infidelity"],
        "fingerprint": changed.short_fingerprint,
        "fullBefore": profile.fingerprint.split(":")[-1],
        "fullAfter": changed.fingerprint.split(":")[-1],
    }


def fleet() -> list[dict]:
    rows = []
    for entry in sorted(INDEX["profiles"], key=lambda r: r["id"]):
        profile = nv.load(entry["id"])
        rows.append(
            {
                "id": entry["id"],
                "vendor": entry["id"].split("_")[0],
                "qubits": profile.device.num_qubits,
                "date": entry["date"],
                "twoQ": two_qubit_error(profile),
                "readout": qubit_medians(profile).readout_error,
            }
        )
    return rows


def cli(*args: str) -> list[str]:
    env = {**os.environ, "NO_COLOR": "1", "COLUMNS": "100", "TERM": "dumb"}
    run = subprocess.run(
        [sys.executable, "-c", "from noisevault.cli import app; app()", *args],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return run.stdout.rstrip("\n").split("\n")


def citation() -> dict:
    text = (ROOT / "CITATION.cff").read_text()
    field = lambda key: re.search(rf"^{key}: \"?([^\"\n]+)\"?$", text, re.M).group(1)  # noqa: E731
    family = re.search(r'family-names: "([^"]+)"', text).group(1)
    given = re.search(r'given-names: "([^"]+)"', text).group(1)
    released = date.fromisoformat(field("date-released"))
    bibtex = "\n".join(
        [
            f"@software{{{family}_NoiseVault_{released.year},",
            f"  author  = {{{family}, {given}}},",
            "  title   = {{NoiseVault}},",
            f"  version = {{{field('version')}}},",
            f"  year    = {{{released.year}}},",
            f"  url     = {{{field('repository-code')}}},",
            f"  license = {{{field('license')}}}",
            "}",
        ]
    )
    return {
        "version": field("version"),
        "released": f"{released.day} {released:%B %Y}",
        "bibtex": bibtex,
    }


def readme_results() -> dict:
    """The quickstart counts, nv check table and nv compare fit that README.md documents."""
    readme = (ROOT / "README.md").read_text()
    counts_text = re.search(r"^# (\{'111'.*\})$", readme, re.M).group(1)
    counts = {k: int(v) for k, v in re.findall(r"'([01]{3})': (\d+)", counts_text)}
    check = []
    for name, dev, tol, method in re.findall(
        r"^(qiskit|cirq|pennylane|stim)\s+pass\s+(\S+)\s+(\S+)\s+5 of 5\s+(.+)$", readme, re.M
    ):
        method = re.sub(r"(\d{2})(\d{3})", r"\1,\2", method.strip())
        check.append({"name": NAMES[name], "dev": float(dev), "tol": float(tol), "method": method})
    circuits = [
        {"name": n, "profile": float(a), "fitted": float(b), "noise": float(c)}
        for n, a, b, c in re.findall(
            r"^(\w+)\s+4000\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$", readme, re.M
        )
    ]
    truth = re.search(
        r"gate\s+errors\s+x([\d.]+)\s+and\s+readout\s+errors\s+x([\d.]+)\s+made\s+these\s+counts",
        readme,
    )
    factors = []
    for label, key, true in (
        ("Gate errors", "gate errors", truth.group(1)),
        ("Readout errors", "readout errors", truth.group(2)),
    ):
        est, lo, hi = re.search(
            rf"^{key}\s+x([\d.]+) \(95% interval ([\d.]+) to ([\d.]+)\)$", readme, re.M
        ).groups()
        factors.append(
            {
                "name": label,
                "est": float(est),
                "lo": float(lo),
                "hi": float(hi),
                "truth": float(true),
            }
        )
    if len(counts) != 8 or len(check) != 4 or len(circuits) != 4:
        raise SystemExit(
            "README.md no longer has the quickstart counts, nv check table or nv compare table"
        )
    return {
        "quickstartCounts": counts_text,
        "ghz": counts,
        "check": check,
        "compare": {"circuits": circuits, "factors": factors},
    }


def verify(page: str, data: dict) -> None:
    """Stop if a number in the page text does not come from the data."""
    readout = data["fez"]["readout"]
    change = data["mismatch"]
    flips = bin(int(change["fullBefore"], 16) ^ int(change["fullAfter"], 16)).count("1")
    vendors = {
        v: sum(r["vendor"] == v for r in data["fleet"]) for v in ("ibm", "quantinuum", "google")
    }
    wrong = sum(n for k, n in data["ghz"].items() if k not in ("000", "111"))
    expected = [
        f"{min(readout):.2f}%",
        f"{max(readout):.1f}%",
        f"{len(readout)} qubits",
        f"{wrong} of {sum(data['ghz'].values()):,}",
        f"{len(data['fleet'])} devices",
        f"{flips} of the 256 bits",
        f"{vendors['ibm']} from IBM, {vendors['quantinuum']} from Quantinuum"
        f" and {vendors['google']} from Google",
    ]
    missing = [e for e in expected if e not in page]
    if missing:
        raise SystemExit(f"site/index.html does not match the data: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path, default=ROOT / "_site", help="output directory (default: _site)"
    )
    args = parser.parse_args()
    cite = citation()
    data = {
        "version": cite["version"],
        "bibtex": cite["bibtex"],
        "fez": chip(),
        "mismatch": fingerprint_change(),
        "fleet": fleet(),
        "cli": {"show": cli("show", "ibm_fez"), "cite": cli("cite", "ibm_fez", "--bibtex")},
        **readme_results(),
    }
    template = (ROOT / "site" / "index.html").read_text()
    verify(template, data)
    page = (
        template.replace("__DATA__", json.dumps(data, separators=(",", ":")))
        .replace("__VERSION__", cite["version"])
        .replace("__RELEASED__", cite["released"])
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.html").write_text(page)
    print(f"wrote {args.out / 'index.html'} (NoiseVault {cite['version']}, {len(page) // 1024} KB)")


if __name__ == "__main__":
    main()
