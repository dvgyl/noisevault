"""The models of the realism study (see ``PLAN.md``), each as a profile.

Each function takes C, the calibration that the counts bind to, and gives one model of the
table in the plan. A model keeps the qubits and the gate names of C, so the reference simulator
can run the circuits of the run with it. A fitted model fits its parameters only on the counts
that it receives. The caller gives it the ``train`` circuits only.

A gate factor f scales a gate error as the ``unmodeled_error.gates`` factor does. The
depolarizing parameter lambda of the gate becomes ``1 - (1 - lambda) ** f``. An error at or past
full depolarization does not change, and the relaxation during the gate stays a floor.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from noisevault import catalog, metrics
from noisevault.compare import FACTOR_RANGE, Comparison
from noisevault.counts import MeasuredCircuit, MeasuredCounts
from noisevault.errors import SourceUnavailable
from noisevault.profile import ErrorFactor, Profile, gate_loci, iso_z, qubit_medians
from noisevault.reference import probabilities
from noisevault.table import GateNoise

Pull = Callable[[str, datetime], Profile]
Locus = tuple[int, ...]

GENERIC_1Q = 0.001
GENERIC_2Q = 0.01
GENERIC_READOUT = 0.01
# The search ends when the bracket of log f is this narrow: 0.5% in f.
LOG_TOLERANCE = 0.005
_GOLDEN = (math.sqrt(5) - 1) / 2
_INDEX = "index.json"


def zero_readouts(profile: Profile) -> int:
    """How many readout probabilities of the enabled qubits are exactly 0, as stated."""
    stated = profile.uncorrected().table
    return sum(
        value == 0
        for q in range(stated.num_qubits)
        if not stated.qubit(q).disabled and stated.qubit(q).readout is not None
        for value in stated.qubit(q).readout or ()
    )


def floored(profile: Profile) -> Profile:
    """The zero floor rule of the plan.

    Each stated readout probability of exactly 0 becomes half of the smallest stated readout
    probability above 0. The ``unmodeled_error`` factors stay. The result is ``profile`` itself
    when no probability is 0, or when no probability is above 0.
    """
    data = profile.to_dict()
    readouts = [data["readout"]] if "readout" in data else []
    readouts += [q["readout"] for q in data.get("qubits", ()) if "readout" in q]
    values = [v for r in readouts for v in _pair(r)]
    positive = [v for v in values if v > 0]
    if not positive or all(v > 0 for v in values):
        return profile
    floor = min(positive) / 2

    def lifted(readout: dict[str, Any]) -> dict[str, Any]:
        low, high = (v if v > 0 else floor for v in _pair(readout))
        return _readout(readout, low, high)

    if "readout" in data:
        data["readout"] = lifted(data["readout"])
    for q in data.get("qubits", ()):
        if "readout" in q:
            q["readout"] = lifted(q["readout"])
    return Profile.model_validate(data)


def ideal(profile: Profile) -> Profile:
    """No noise: no gate error, no readout error, no relaxation and no preparation error."""
    return _fixed(profile, 0.0, 0.0, 0.0, kind="avg_infidelity")


def generic(profile: Profile) -> Profile:
    """One fixed model for every device: depolarizing parameter 0.001 for 1-qubit gates and
    0.01 for other gates, readout error 0.01, no relaxation and no preparation error."""
    return _fixed(profile, GENERIC_1Q, GENERIC_2Q, GENERIC_READOUT, kind="depolarizing_param")


def medians(profile: Profile) -> Profile:
    """C with each value replaced by the median of that value over the device.

    A gate gets the median of its average infidelity and of its duration over its loci. A qubit
    gets the median T1, T2, dephasing rate, preparation error, P(1|0) and P(0|1) over the
    enabled qubits. A disabled gate or qubit stays disabled.
    """
    base = profile.uncorrected()
    table = base.table
    data = base.to_dict()
    for name in base.gates:
        loci = [g for g in gate_loci(base, name) if g.state == "calibrated"]
        if not loci:
            continue
        error = median(g.avg_infidelity for g in loci if g.avg_infidelity is not None)
        durations = [g.duration_ns for g in loci if g.duration_ns is not None]
        values: dict[str, Any] = {"avg_infidelity": error}
        if durations:
            values["duration_ns"] = median(durations)
        data["gates"][name] = _metric(data["gates"][name], values)
        data["calibrations"] = [
            _metric(r, values) if r["gate"] == name else r for r in data.get("calibrations", ())
        ]

    middle = qubit_medians(base)
    working = [table.qubit(q) for q in range(table.num_qubits) if not table.qubit(q).disabled]
    idle: dict[str, Any] = {k: v for k, v in data.get("idle", {}).items() if k == "t2_kind"}
    idle |= {"t1_us": middle.t1_us, "t2_us": middle.t2_us}
    rates = [q.dephasing_rate_per_s for q in working if q.dephasing_rate_per_s is not None]
    idle["dephasing_rate_per_s"] = median(rates) if rates else None
    idle = {k: v for k, v in idle.items() if v is not None}
    preps = [q.prep_error for q in working if q.prep_error is not None]
    readout = None
    if middle.p1_given_0 is not None and middle.p0_given_1 is not None:
        readout = (middle.p1_given_0, middle.p0_given_1)
    for key, value in (
        ("idle", idle or None),
        ("prep", {"error": median(preps)} if preps else None),
        ("readout", None if readout is None else _readout(data.get("readout", {}), *readout)),
    ):
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    for q in data.get("qubits", ()):
        for key in ("t1_us", "t2_us", "dephasing_rate_per_s", "prep"):
            q.pop(key, None)
        if "readout" in q:
            if readout is None:
                q.pop("readout")
            else:
                q["readout"] = _readout(q["readout"], *readout)
    return Profile.model_validate(data)


def symmetric(profile: Profile) -> Profile:
    """C with the two readout errors of each qubit replaced by their mean."""
    data = profile.uncorrected().to_dict()

    def mean(readout: dict[str, Any]) -> dict[str, Any]:
        low, high = _pair(readout)
        return _readout(readout, (low + high) / 2, (low + high) / 2)

    if "readout" in data:
        data["readout"] = mean(data["readout"])
    for q in data.get("qubits", ()):
        if "readout" in q:
            q["readout"] = mean(q["readout"])
    return Profile.model_validate(data)


def stale(
    profile: Profile,
    run_at: datetime,
    days: int,
    cache: Path | None,
    pull: Pull | None = None,
) -> Profile | None:
    """The calibration of the device that was in effect ``days`` days before ``run_at``.

    The calibration comes from IBM's public endpoint, or from ``cache`` when an earlier call
    saved it there. The result is None when the device is not an IBM device, when the endpoint
    has no calibration before that time, or when that calibration is C. Any other failure of
    the endpoint raises SourceUnavailable.
    """
    if profile.device.vendor != "ibm":
        return None
    found = pulled(profile.device.name, run_at - timedelta(days=days), cache, pull)
    if found is None or same_calibration(found, profile):
        return None
    return found


def pulled(
    device: str, at: datetime, cache: Path | None, pull: Pull | None = None
) -> Profile | None:
    """The newest calibration of ``device`` before ``at``, or None when the source has none.

    ``cache`` is a folder, or None for no cache. It holds one file for each calibration, named
    by device and calibration time, and an index from each requested time to that file.
    """
    key = f"{device} {iso_z(at)}"
    index = _read_index(cache)
    if key in index:
        name = index[key]
        if name is None:
            return None
        assert cache is not None
        return Profile.load(cache / name)
    if pull is None:
        from noisevault.sources.ibm_public import pull as public

        pull = public
    try:
        found = pull(device, at)
    except SourceUnavailable as exc:
        # The endpoint says "no ... calibration older than" when it has no earlier calibration.
        if "older than" not in exc.message:
            raise
        found = None
    if cache is not None:
        name = None
        if found is not None:
            when = found.device.calibrated_at
            stamp = "undated" if when is None else when.strftime("%Y-%m-%dT%H%M%SZ")
            name = f"{device}@{stamp}.json.gz"
            found.save(cache / name)
        _write_index(cache, {**index, key: name})
    return found


def bundled(profile: Profile) -> Profile | None:
    """The newest bundled profile of the device, or None when NoiseVault bundles none or when
    the bundled profile is C."""
    found = [info for info in catalog.bundled_profiles() if info.id == profile.id]
    if not found:
        return None
    newest = max(found, key=lambda info: (info.calibrated_at is not None, info.calibrated_at))
    loaded = newest.load()
    return None if same_calibration(loaded, profile) else loaded


def same_calibration(a: Profile, b: Profile) -> bool:
    """True when two profiles hold one calibration: the same fingerprint or the same time."""
    if a.calibration_fingerprint == b.calibration_fingerprint:
        return True
    when = a.device.calibrated_at
    return when is not None and when == b.device.calibrated_at


def readout_measured(profile: Profile, counts: MeasuredCounts) -> Profile:
    """C with the readout errors of each qubit of the chain measured on the counts.

    P(1|0) of a qubit is the frequency of 1 on that qubit in ``readout``, and P(0|1) is the
    frequency of 0 on that qubit in ``readout_1``. For k such results in n shots, the value is
    (k + 1/2) / (n + 1), so it is never 0 or 1. The gate errors stay as stated.
    """
    zeros, ones = _circuit(counts, "readout"), _circuit(counts, "readout_1")
    if zeros.qubits != ones.qubits:
        raise ValueError("the readout and readout_1 circuits measure different qubits")
    data = profile.uncorrected().to_dict()
    records = {q["index"]: q for q in data.get("qubits", ())}
    stated = data.get("readout", {})
    for i, q in enumerate(zeros.qubits):
        record = records.setdefault(q, {"index": q})
        low = _rate(zeros, i, "1")
        high = _rate(ones, i, "0")
        record["readout"] = _readout(record.get("readout", stated), low, high)
    data["qubits"] = list(records.values())
    return Profile.model_validate(data)


def readout_rates(profile: Profile, qubits: Sequence[int]) -> list[dict[str, Any]]:
    """P(1|0) and P(0|1) of each of ``qubits``, as derived numbers for the output."""
    table = profile.uncorrected().table
    rows = []
    for q in qubits:
        low, high = table.qubit(q).readout or (None, None)
        rows.append({"qubit": q, "p1_given_0": low, "p0_given_1": high})
    return rows


def with_factors(profile: Profile, gates: float | None, readout: float | None = None) -> Profile:
    """``profile`` with hand-written ``unmodeled_error`` factors. None leaves an error as stated."""
    block = {
        name: {"factor": value}
        for name, value in (("gates", gates), ("readout", readout))
        if value is not None
    }
    return profile.uncorrected().model_copy(update={"unmodeled_error": block or None})


def two_factor(profile: Profile, train: MeasuredCounts) -> tuple[Profile, Comparison]:
    """C with the gate factor and the readout factor that ``nv compare`` fits on ``train``.

    A factor that the fit does not identify is 1, so that error stays as stated. The factors
    have no interval in the result, because the comparison holds the intervals.
    """
    comparison = profile.uncorrected().compare(train)
    gate = comparison.gates.factor if isinstance(comparison.gates, ErrorFactor) else None
    readout = comparison.readout.factor if isinstance(comparison.readout, ErrorFactor) else None
    return with_factors(profile, gate, readout), comparison


def gate_factor(profile: Profile, counts: MeasuredCounts, circuits: Iterable[str]) -> float:
    """The gate factor of maximum likelihood on the named circuits, with the readout fixed.

    The factor scales every gate error as the ``unmodeled_error.gates`` factor does. The search
    covers the range of ``nv compare``, 0.05 to 20.
    """
    return fit_factor(lambda f: with_factors(profile, f), counts, circuits)


def fit_factor(
    build: Callable[[float], Profile], counts: MeasuredCounts, circuits: Iterable[str]
) -> float:
    """The factor f of maximum likelihood, where ``build(f)`` gives the model at f.

    The log-likelihood is the multinomial log-likelihood of the named circuits under exact
    reference simulation. A golden-section search on log f finds the maximum. The search needs
    about 15 simulations of the circuits.
    """
    chosen = [_circuit(counts, name) for name in dict.fromkeys(circuits)]
    if not chosen:
        raise ValueError("fit_factor needs at least one circuit")
    measured = [c.vector() for c in chosen]

    def loglik(x: float) -> float:
        model = build(math.exp(x))
        total = 0.0
        for c, n in zip(chosen, measured, strict=True):
            p = probabilities(
                model, c.ops, len(c.qubits), layout=c.qubits, readout=True, unknown_gates="error"
            )
            seen = n > 0
            if np.any(p[seen] <= 0):
                return -math.inf
            total += float(n[seen] @ np.log(p[seen]))
        return total

    low, high = math.log(FACTOR_RANGE[0]), math.log(FACTOR_RANGE[1])
    a, b = high - _GOLDEN * (high - low), low + _GOLDEN * (high - low)
    fa, fb = loglik(a), loglik(b)
    while high - low > LOG_TOLERANCE:
        # A tie keeps the lower bracket, so that a flat likelihood gives the same result each time.
        if fa >= fb:
            high, b, fb = b, a, fa
            a = high - _GOLDEN * (high - low)
            fa = loglik(a)
        else:
            low, a, fa = a, b, fb
            b = low + _GOLDEN * (high - low)
            fb = loglik(b)
    return math.exp((low + high) / 2)


def scale_loci(profile: Profile, factors: Mapping[Locus, float]) -> Profile:
    """``profile`` with the error of each gate on each locus of ``factors`` scaled by its factor.

    A locus of one qubit scales every 1-qubit gate on that qubit. A locus of two qubits scales
    every 2-qubit gate on that pair, in both directions. The rule is the rule of the
    ``unmodeled_error.gates`` factor. Each scaled locus gets a calibration record with the scaled
    error, and the other fields of that locus stay.
    """
    base = profile
    table = base.table
    data = base.to_dict()
    records = {(r["gate"], tuple(r["qubits"])): r for r in data.get("calibrations", ())}
    for locus, factor in factors.items():
        orders = [tuple(locus)] if len(locus) == 1 else [tuple(locus), tuple(locus)[::-1]]
        for name in base.gates:
            if table.arity(name) != len(locus):
                continue
            for qubits in orders:
                found = table.gate(name, qubits)
                if not isinstance(found, GateNoise) or found.state != "calibrated":
                    continue
                stated = found.spec.metric
                if stated is None:
                    continue
                spec = found.spec.model_dump(mode="json", exclude_none=True)
                spec.pop("qubits", None)
                spec.pop("symmetric", None)
                records[(name, qubits)] = {
                    "gate": name,
                    "qubits": list(qubits),
                    **_metric(spec, _scaled(stated, len(qubits), factor)),
                }
    data["calibrations"] = list(records.values())
    return Profile.model_validate(data)


def locus(
    profile: Profile, counts: MeasuredCounts
) -> tuple[Profile, dict[Locus, float], dict[str, Any]]:
    """``readout_measured``, then one factor for each qubit and one for each pair of the chain.

    The factor of a qubit scales each 1-qubit gate on that qubit and comes from that qubit's
    ``qubit*`` circuit. The factor of a pair scales each 2-qubit gate on that pair and comes from
    that pair's ``pair*`` circuits, with the qubit factors fixed. A locus with no circuit keeps
    factor 1. The result is the model, the factors, and the fitted values for the output.
    """
    base = readout_measured(profile, counts)
    chain = _circuit(counts, "readout").qubits
    names = [c.name for c in counts.circuits]
    factors: dict[Locus, float] = {}
    rows: dict[str, Any] = {"qubits": [], "pairs": []}
    for i, q in enumerate(chain):
        own = [n for n in names if n.startswith(f"qubit{i}_")]
        value = None
        if own:
            value = fit_factor(lambda f, q=q: scale_loci(base, {(q,): f}), counts, own)
            factors[(q,)] = value
        rows["qubits"].append({"qubit": q, "factor": value})
    fixed = dict(factors)
    for i in range(len(chain) - 1):
        pair = (chain[i], chain[i + 1])
        own = [n for n in names if n.startswith(f"pair{i}_")]
        value = None
        if own:
            value = fit_factor(
                lambda f, pair=pair: scale_loci(base, {**fixed, pair: f}), counts, own
            )
            factors[pair] = value
        rows["pairs"].append({"qubits": list(pair), "factor": value})
    return scale_loci(base, factors), factors, rows


def _fixed(profile: Profile, one: float, two: float, readout: float, *, kind: str) -> Profile:
    base = profile.uncorrected()
    table = base.table
    data = base.to_dict()
    for name, spec in data["gates"].items():
        if any(key in spec for key in metrics.METRIC_KEYS):
            data["gates"][name] = _metric(spec, {kind: one if table.arity(name) == 1 else two})
    data["calibrations"] = [
        _metric(r, {kind: one if len(r["qubits"]) == 1 else two})
        if any(key in r for key in metrics.METRIC_KEYS)
        else r
        for r in data.get("calibrations", ())
    ]
    data.pop("idle", None)
    data.pop("prep", None)
    data["readout"] = _readout(data.get("readout", {}), readout, readout)
    for q in data.get("qubits", ()):
        for key in ("t1_us", "t2_us", "t2_kind", "dephasing_rate_per_s", "prep"):
            q.pop(key, None)
        if "readout" in q:
            q["readout"] = _readout(q["readout"], readout, readout)
    return Profile.model_validate(data)


def _metric(spec: Mapping[str, Any], values: Mapping[str, Any]) -> dict[str, Any]:
    """``spec`` with its error metric replaced by the metric in ``values``."""
    kept = {k: v for k, v in spec.items() if k not in metrics.METRIC_KEYS}
    return kept | dict(values)


def _scaled(stated: tuple[Any, Any], num_qubits: int, factor: float) -> dict[str, Any]:
    kind, value = stated
    if kind == "pauli":
        return {"pauli": list(metrics.scale_pauli(value, factor))}
    r = metrics.to_avg_infidelity(kind, value, num_qubits)
    return {"avg_infidelity": metrics.scale_avg_infidelity(r, num_qubits, factor)}


def _pair(readout: Mapping[str, Any]) -> tuple[float, float]:
    if "error" in readout:
        return readout["error"], readout["error"]
    return readout["p1_given_0"], readout["p0_given_1"]


def _readout(readout: Mapping[str, Any], low: float, high: float) -> dict[str, Any]:
    """``readout`` with P(1|0) = ``low`` and P(0|1) = ``high``. Its duration stays."""
    kept = {k: v for k, v in readout.items() if k == "duration_ns"}
    if low == high:
        return kept | {"error": low}
    return kept | {"p1_given_0": low, "p0_given_1": high}


def _circuit(counts: MeasuredCounts, name: str) -> MeasuredCircuit:
    for c in counts.circuits:
        if c.name == name:
            return c
    raise ValueError(f"the counts have no circuit {name!r}")


def _rate(circuit: MeasuredCircuit, bit: int, wrong: str) -> float:
    k = sum(n for key, n in circuit.counts.items() if key[bit] == wrong)
    return (k + 0.5) / (circuit.shots + 1)


def _read_index(cache: Path | None) -> dict[str, str | None]:
    if cache is None or not (cache / _INDEX).is_file():
        return {}
    return json.loads((cache / _INDEX).read_text(encoding="utf-8"))


def _write_index(cache: Path, index: dict[str, str | None]) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    temporary = cache / (_INDEX + ".new")
    temporary.write_text(json.dumps(index, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(cache / _INDEX)
