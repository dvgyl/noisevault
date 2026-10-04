from __future__ import annotations

import copy
import math
import re
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeAlias

import numpy as np

from . import gates, metrics
from .channels import gate_channels
from .conversion import resolve_op
from .errors import (
    CountsError,
    DisabledGateError,
    LayoutError,
    LociText,
    MissingCalibrationError,
    NoiseVaultError,
    joined,
    plural,
    qubit_loci,
)
from .profile import POOR_FIT_P_VALUE, Bound, ErrorFactor, UnmodeledError, iso_z, ref_on_day
from .reference import charged_as, outcome_bits, probabilities
from .report import Report
from .table import GateNoise, refuse_disabled, unscaled_phrases

if TYPE_CHECKING:
    from .counts import MeasuredCounts, PlannedCircuit
    from .profile import Profile

FACTOR_RANGE = (0.05, 20.0)
LEVEL = 0.95
CHI2_95 = 3.841458820694124
GATE_NODES = 25
FINE_POINTS = 193
RESAMPLES = 400
EXACT_OUTCOMES = 4096
WINDOW_POINTS = 33
END_TOL = 0.02
IMPOSSIBLE = 1e-12
MIN_INFORMATIVE_SHOTS = 100
EPS = float(np.finfo(float).eps)
SEPARABLE = 2 * EPS
FLAT = 1e-6
FD_STEP = 1e-3
REFINE_TOL = 0.04
REGION = 9.0
TIE = REFINE_TOL / 2
ASCENT_TOL = 1e-3
NOTE = (
    "Factors multiply the profile's error rates, so x2 means about twice the errors.",
    "On these qubits, the factors absorb crosstalk, leakage, coherent error and idle",
    "error beyond T1 and T2. The profile from -o applies the factors to every qubit.",
)
NEXT_READOUT = (
    "add a circuit with no gates, which only readout error moves.",
    "The circuits from noisevault.counts.plan() include one.",
)

Axis = Literal["gate", "readout"]
_AXES: tuple[Axis, Axis] = ("gate", "readout")
_LO, _HI = math.log(FACTOR_RANGE[0]), math.log(FACTOR_RANGE[1])
_STEP = (_HI - _LO) / (FINE_POINTS - 1)
_MIN_WINDOW = 0.05
_SAME_WAY = "gate error and readout error move these counts the same way"
_UNREACHED = (
    f"no factors from {FACTOR_RANGE[0]:g} to {FACTOR_RANGE[1]:g} give the measured frequencies"
)
_LABEL = 16
_WIDTH = 80
_DROPPABLE = ("shots", "profile TVD")
_WORD = re.compile(r"\([^()]*\)|\S+")
_TVD_DIGITS = 4
_CACHE_SIZE = 4
_CHUNK = 1 << 22

ByPoint: TypeAlias = np.ndarray
ByCell: TypeAlias = np.ndarray
ByColumn: TypeAlias = np.ndarray
ByPointByCell: TypeAlias = np.ndarray
ByPointByDraw: TypeAlias = np.ndarray
ByCellByDraw: TypeAlias = np.ndarray
ByPointByColumn: TypeAlias = np.ndarray
ByGateByReadout: TypeAlias = np.ndarray
ByPointByQubitBit: TypeAlias = np.ndarray
ByPointByMeasuredByPrepared: TypeAlias = np.ndarray


@dataclass(frozen=True)
class NoEstimate:
    reason: str


FactorResult = ErrorFactor | NoEstimate


class SummaryLine(NamedTuple):
    text: str
    bold_end: int


@dataclass(frozen=True)
class CircuitScore:
    name: str
    qubits: tuple[int, ...]
    shots: int
    tvd_profile: float
    tvd_fitted: float
    shot_noise_95: float
    impossible: int


@dataclass(frozen=True)
class Comparison:
    profile: Profile
    counts: MeasuredCounts = field(repr=False)
    circuits: tuple[CircuitScore, ...]
    gates: FactorResult
    readout: FactorResult
    deviance: float
    dof: int
    p_value: float | None
    ruled_out: tuple[str, ...]
    notes: tuple[LociText, ...]

    @property
    def impossible_shots(self) -> int:
        return sum(score.impossible for score in self.circuits)

    @property
    def dispersion(self) -> float:
        """The deviance divided by its degrees of freedom, and at least 1. With no degrees of
        freedom, the dispersion is 1."""
        return _dispersion(self.deviance, self.dof)

    @property
    def calibration_age(self) -> timedelta | None:
        when = self.profile.device.calibrated_at
        return None if when is None else self.counts.run_at - when

    def fitted_profile(self) -> Profile:
        """The calibration with an ``unmodeled_error`` block that records this fit."""
        if isinstance(self.gates, NoEstimate):
            if isinstance(self.readout, NoEstimate):
                raise NoiseVaultError(
                    "gate and readout factors are not identified, so there is nothing to save"
                )
            raise NoiseVaultError("the gate factor is not identified, so there is nothing to save")
        base = self.profile.uncorrected()
        block: dict[str, Any] = {"gates": _saved(self.gates)}
        if isinstance(self.readout, ErrorFactor):
            block["readout"] = _saved(self.readout)
        block["fit"] = {
            "counts": self.counts.sha256,
            "source": self.counts.source,
            "qubits": list(self.counts.qubits),
            "run_at": self.counts.run_at,
            "calibration": base.fingerprint,
            "p_value": self.p_value,
            "impossible_shots": self.impossible_shots,
        }
        return base.model_copy(update={"unmodeled_error": block})

    def summary(self, *, counts_file: str | None = None, width: int = _WIDTH) -> str:
        """The text ``nv compare`` prints in a terminal ``width`` columns wide, 80 at most.

        Lines break between words. Below 80 columns the table drops its shots column, then its
        profile TVD column, until it fits.
        """
        lines = self.summary_lines(counts_file=counts_file, width=width)
        return "\n".join(line.text for line in lines)

    def summary_lines(
        self, *, counts_file: str | None = None, width: int = _WIDTH
    ) -> tuple[SummaryLine, ...]:
        width = min(width, _WIDTH)
        blank = SummaryLine("", 0)
        lines = [
            *self._header(counts_file, width),
            blank,
            *self._table(width),
            blank,
            *self._findings(width),
        ]
        if self._fitted:
            note = NOTE if width == _WIDTH else _wrap(" ".join(NOTE), width)
            lines += [blank, *(SummaryLine(text, 0) for text in note)]
        return tuple(lines)

    __str__ = summary

    def to_dict(self) -> dict[str, Any]:
        age = self.calibration_age
        return {
            "profile_id": self.profile.id,
            "fingerprint": self.profile.fingerprint,
            "calibration": self.profile.calibration_fingerprint,
            "counts": self.counts.sha256,
            "source": self.counts.source,
            "backend": self.counts.backend,
            "run_at": iso_z(self.counts.run_at),
            "calibration_age_hours": None if age is None else age.total_seconds() / 3600,
            "qubits": list(self.counts.qubits),
            "circuits": [
                {
                    "name": s.name,
                    "qubits": list(s.qubits),
                    "shots": s.shots,
                    "tvd_profile": s.tvd_profile,
                    "tvd_fitted": s.tvd_fitted,
                    "shot_noise_95": s.shot_noise_95,
                    "impossible": s.impossible,
                }
                for s in self.circuits
            ],
            "gates": _factor_dict(self.gates),
            "readout": _factor_dict(self.readout),
            "deviance": self.deviance,
            "dof": self.dof,
            "dispersion": self.dispersion,
            "p_value": self.p_value,
            "resamples": RESAMPLES,
            "impossible_shots": self.impossible_shots,
            "ruled_out": list(self.ruled_out),
            "notes": [note.full for note in self.notes],
            "note": " ".join(NOTE) if self._fitted else None,
        }

    @property
    def _fitted(self) -> bool:
        return isinstance(self.gates, ErrorFactor) or isinstance(self.readout, ErrorFactor)

    def _header(self, counts_file: str | None, width: int) -> list[SummaryLine]:
        profile, counts = self.profile, self.counts
        ref = ref_on_day(profile.id, profile.device.calibrated_at)
        digest = counts.sha256.removeprefix("sha256:")[:12]
        source = (
            f"{counts.source} counts sha256:{digest}"
            if counts_file is None
            else f"counts {counts_file}, {counts.source}, sha256:{digest}"
        )
        age = self.calibration_age
        since = (
            "calibration age unknown (the profile has no date)"
            if age is None
            else f"{_duration(age)} after calibration"
        )
        run = counts.run_at.strftime("%Y-%m-%d %H:%MZ")
        first = f"{ref} {profile.short_fingerprint} on {qubit_loci(counts.qubits)}"
        texts = (first, source, f"run {run}, {since}")
        lines = [SummaryLine(part, 0) for text in texts for part in _wrap(text, width, "  ")]
        lines[0] = SummaryLine(lines[0].text, len(ref))
        return lines

    def _table(self, width: int) -> list[SummaryLine]:
        dropped: list[str] = []
        rows = self._table_rows(dropped)
        for column in _DROPPABLE:
            if max(len(row.text) for row in rows) <= width:
                break
            dropped.append(column)
            rows = self._table_rows(dropped)
        return rows

    def _table_rows(self, dropped: Sequence[str]) -> list[SummaryLine]:
        impossible = self.impossible_shots > 0
        cells: dict[str, Callable[[CircuitScore], str]] = {
            "shots": lambda s: str(s.shots),
            "profile TVD": lambda s: f"{s.tvd_profile:.{_TVD_DIGITS}f}",
            "fitted TVD": lambda s: f"{s.tvd_fitted:.{_TVD_DIGITS}f}",
            "noise TVD 95%": lambda s: f"{s.shot_noise_95:.{_TVD_DIGITS}f}",
            **({"impossible shots": lambda s: str(s.impossible)} if impossible else {}),
        }
        columns = {
            title: [cell(s) for s in self.circuits]
            for title, cell in cells.items()
            if title not in dropped
        }
        widths = {title: max(map(len, [title, *values])) for title, values in columns.items()}
        name = max(len("circuit"), *(len(s.name) for s in self.circuits))
        head = "circuit".ljust(name) + "".join(f"  {t:>{widths[t]}}" for t in columns)
        rows = [SummaryLine(head, len(head))]
        for i, s in enumerate(self.circuits):
            row = s.name.ljust(name) + "".join(f"  {v[i]:>{widths[t]}}" for t, v in columns.items())
            if not impossible and _beyond_noise(s):
                row += "  beyond noise"
            rows.append(SummaryLine(row, 0))
        return rows

    def _findings(self, width: int) -> list[SummaryLine]:
        gate, readout = self.gates, self.readout
        gate_reason, readout_reason = _reason(gate), _reason(readout)
        shared = gate_reason is not None and gate_reason == readout_reason
        entries = [
            ("gate errors", [_say(gate), *([gate_reason] if gate_reason and not shared else [])]),
            ("readout errors", [_say(readout), *([readout_reason] if readout_reason else [])]),
            ("fit", self._verdict()),
        ]
        if _SAME_WAY in (gate_reason, readout_reason):
            entries.append(("next", list(NEXT_READOUT)))
        if self.notes:
            entries.append(("note", [note.short for note in self.notes]))
        lines = []
        for label, values in entries:
            for i, value in enumerate(values):
                wrapped = _wrap(value, width - _LABEL)
                shown = label if i == 0 else ""
                lines.append(SummaryLine(f"{shown:<{_LABEL}}{wrapped[0]}", len(shown)))
                lines += [SummaryLine(" " * _LABEL + part, 0) for part in wrapped[1:]]
        return lines

    def _verdict(self) -> list[str]:
        flagged = [s.name for s in self.circuits if _beyond_noise(s)]
        impossible = self.impossible_shots
        if impossible:
            total = sum(s.shots for s in self.circuits)
            lines = [
                "ruled out (p = 0)",
                f"the profile gives {plural(impossible, 'shot')} probability 0",
                *self.ruled_out,
            ]
            if impossible < total:
                lines.append(f"the fit uses the other {total - impossible} shots")
            if flagged:
                lines.append(f"beyond shot noise on {joined(flagged)}")
            return lines
        if self.p_value is None:
            return ["not testable (no degrees of freedom left after fitting)"]
        p = f"p = {self.p_value:.2g}"
        unreached = [_UNREACHED] if self.dof <= 0 else []
        if self.p_value < POOR_FIT_P_VALUE:
            where = [f"on {joined(flagged)}"] if flagged else []
            why = unreached or ["no one pair of factors fits every circuit"]
            return [f"beyond shot noise ({p})", *where, *why]
        scope = "overall" if flagged else "on every circuit"
        return [f"within shot noise {scope} ({p})", *unreached]


def _wrap(text: str, width: int, indent: str = "") -> list[str]:
    """``text`` in lines of at most ``width`` that break between words.

    Each group in parentheses stays on one line. Every line after the first starts with ``indent``.
    """
    lines: list[str] = []
    for word in _WORD.findall(text):
        if lines and len(lines[-1]) + 1 + len(word) <= width:
            lines[-1] += f" {word}"
        else:
            lines.append(f"{indent if lines else ''}{word}")
    return lines or [""]


def compare(profile: Profile, counts: MeasuredCounts) -> Comparison:
    """Score ``profile`` on ``counts`` and fit its gate and readout factors.

    Raises CountsError when the counts come from a plan for a different calibration or ran
    before the calibration. Raises CountsError also when the counts contain an op that the
    profile does not calibrate on its qubits.
    """
    base = _bind(profile, counts)
    circuits = counts.circuits
    surface = _Surface.cached(base, circuits)
    measured = np.concatenate([c.vector() for c in circuits]).astype(float)
    observed = np.where(surface.supported, measured, 0.0)
    shots = np.array([observed[part].sum() for part in surface.slices])

    surface, values, peaks = surface.refined(observed, CHI2_95)
    estimate = _estimate(peaks)
    log_gate, log_readout = estimate.gate, estimate.readout
    gate = None if surface.gate is None else math.exp(log_gate)
    readout = None if surface.readout is None else math.exp(log_readout)
    fitted = _exact(base, circuits, gate, readout)
    info = _fisher(base, circuits, shots, fitted, gate, readout)
    unfitted = _identify(surface, observed, info)
    rank = _rank(info)
    used = [part for part, n in zip(surface.slices, shots, strict=True) if n]
    dof = sum(int(surface.supported[part].sum()) - 1 for part in used) - rank
    deviance = _deviance(observed, fitted, surface.slices)
    dispersion = _dispersion(deviance, dof)
    surface, values, peaks = surface.refined(observed, CHI2_95 * dispersion, values, peaks)

    seed = int(counts.sha256[7:23], 16)
    fit = _Fit(surface, values, observed, estimate, peaks, shots, dispersion, seed)
    results = {axis: unfitted[axis] or fit.interval(axis) for axis in _AXES}

    draws = fit.draw(fitted, np.random.default_rng(seed))
    every = np.column_stack([observed, draws])
    top, draw_gate, draw_readout = fit.draw_max(every)
    surface_deviance = 2 * (_saturated(every, surface.slices) - top)
    refits = surface.probs_at(draw_gate[1:], draw_readout[1:])

    impossible = measured - observed
    if impossible.sum():
        p_value: float | None = 0.0
    elif dof <= 0 and deviance <= CHI2_95 and _LO < log_gate < _HI and _LO < log_readout < _HI:
        p_value = None
    else:
        exceed = np.sum(surface_deviance[1:] >= surface_deviance[0])
        p_value = float((1 + exceed) / (1 + RESAMPLES))

    as_given = _exact(profile, circuits, None, None)
    scores = []
    for c, part, n in zip(circuits, surface.slices, shots, strict=True):
        frequencies = measured[part] / c.shots
        if n:
            noise = 0.5 * np.abs(draws[part].T / n - refits[:, part]).sum(axis=1)
            noise_95 = float(np.percentile(noise, 100 * LEVEL))
        else:
            noise_95 = 0.0
        scores.append(
            CircuitScore(
                name=c.name,
                qubits=tuple(c.qubits),
                shots=c.shots,
                tvd_profile=_tvd(frequencies, as_given[part]),
                tvd_fitted=_tvd(frequencies, fitted[part]),
                shot_noise_95=noise_95,
                impossible=int(impossible[part].sum()),
            )
        )
    return Comparison(
        profile=profile,
        counts=counts,
        circuits=tuple(scores),
        gates=results["gate"],
        readout=results["readout"],
        deviance=deviance,
        dof=dof,
        p_value=p_value,
        ruled_out=_ruled_out(surface, circuits, impossible),
        notes=_notes(base, circuits),
    )


class _Surface:
    _built: OrderedDict[Any, _Surface] = OrderedDict()

    def __init__(
        self,
        gate_nodes: ByPoint,
        before_readout: ByPointByCell,
        readout_pairs: tuple[tuple[tuple[float, float] | None, ...], ...],
        slices: tuple[slice, ...],
        gate: np.ndarray | None,
        readout: np.ndarray | None,
        run: Callable[[float], ByCell],
        rounding: ByCell,
    ) -> None:
        self.gate_nodes = gate_nodes
        self.before_readout = before_readout
        self.readout_pairs = readout_pairs
        self.slices = slices
        self.gate = gate
        self.readout = readout
        self.run = run
        self.rounding = rounding
        reads = _grid(readout)
        peaks = [self.probs_at(np.full(len(reads), x), reads).max(axis=0) for x in gate_nodes]
        self.supported = np.max(peaks, axis=0) > IMPOSSIBLE

    @classmethod
    def cached(cls, base: Profile, circuits: Sequence[PlannedCircuit]) -> _Surface:
        key = (base.fingerprint, tuple((c.qubits, c.ops) for c in circuits))
        if key in cls._built:
            cls._built.move_to_end(key)
            return cls._built[key]
        surface = cls._built[key] = cls.build(base, circuits)
        if len(cls._built) > _CACHE_SIZE:
            cls._built.popitem(last=False)
        return surface

    @classmethod
    def build(cls, base: Profile, circuits: Sequence[PlannedCircuit]) -> _Surface:
        fine = np.linspace(_LO, _HI, FINE_POINTS)
        nodes = np.unique(
            np.concatenate(
                [
                    fine[:: (FINE_POINTS - 1) // (GATE_NODES - 1)],
                    np.log(_floor_factors(base, circuits)),
                ]
            )
        )
        sizes = [2 ** len(c.qubits) for c in circuits]
        starts = np.cumsum([0, *sizes])
        slices = tuple(slice(int(a), int(b)) for a, b in zip(starts[:-1], starts[1:], strict=True))

        runs: dict[float, ByCell] = {}

        def run(log_gate: float) -> ByCell:
            if log_gate not in runs:
                runs[log_gate] = _exact(base, circuits, math.exp(log_gate), None, read=False)
            return runs[log_gate]

        before_readout = np.array([run(x) for x in nodes])
        rounding = _rounding(circuits)
        if not any(_varies(before_readout[:, part], rounding[part]) for part in slices):
            gate, nodes, before_readout = None, np.zeros(1), run(0.0)[None, :]
        else:
            gate = np.unique(np.concatenate([fine, nodes]))
        pairs = tuple(tuple(base.table.qubit(q).readout for q in c.qubits) for c in circuits)
        scales = any(
            pair is not None
            and metrics.scale_readout(pair, FACTOR_RANGE[0])
            != metrics.scale_readout(pair, FACTOR_RANGE[1])
            for per_circuit in pairs
            for pair in per_circuit
        )
        readout = fine if scales else None
        return cls(nodes, before_readout, pairs, slices, gate, readout, run, rounding)

    def axis(self, axis: Axis) -> np.ndarray | None:
        return self.gate if axis == "gate" else self.readout

    def probs_at(self, log_gate: ByPoint, log_readout: ByPoint) -> ByPointByCell:
        unread = self._before_readout_at(np.asarray(log_gate, dtype=float))
        factors, which = np.unique(np.asarray(log_readout, dtype=float), return_inverse=True)
        which = which.ravel()
        matrices = {
            pair: _confusion(pair, factors)[which]
            for pair in {pair for pairs in self.readout_pairs for pair in pairs if pair is not None}
        }
        out = np.empty_like(unread)
        for part, pairs in zip(self.slices, self.readout_pairs, strict=True):
            block = unread[:, part].reshape(-1, *(2,) * len(pairs))
            for i, pair in enumerate(pairs):
                if pair is not None:
                    block = _read(block, i + 1, matrices[pair])
            out[:, part] = block.reshape(len(unread), part.stop - part.start)
        return out

    def loglik_at(
        self, log_gate: ByPoint, log_readout: ByPoint, counts: ByCell | ByCellByDraw
    ) -> ByPoint | ByPointByDraw:
        return _loglik(self.probs_at(log_gate, log_readout), counts)

    def loglik_each(
        self, log_gate: ByColumn, log_readout: ByColumn, counts: ByCellByDraw
    ) -> ByColumn:
        return _paired(self.probs_at(log_gate, log_readout), counts)

    def loglik(self, counts: ByCell | ByCellByDraw, gate: ByPoint | None = None) -> ByGateByReadout:
        gate = _grid(self.gate) if gate is None else gate
        readout = _grid(self.readout)
        points_gate, points_readout = (a.ravel() for a in np.meshgrid(gate, readout, indexing="ij"))
        counts = np.asarray(counts, dtype=float)
        out = np.empty((len(points_gate), *counts.shape[1:]))
        chunk = max(1, _CHUNK // len(counts))
        for start in range(0, len(points_gate), chunk):
            part = slice(start, start + chunk)
            out[part] = self.loglik_at(points_gate[part], points_readout[part], counts)
        return out.reshape(len(gate), len(readout), *counts.shape[1:])

    def refined(
        self,
        counts: ByCell,
        cutoff: float,
        values: ByGateByReadout | None = None,
        peaks: list[_Peak] | None = None,
    ) -> tuple[_Surface, ByGateByReadout, list[_Peak]]:
        surface = self
        values = self.loglik(counts) if values is None else values
        sizes = [part.stop - part.start for part in self.slices]
        weights = np.repeat([counts[part].sum() for part in self.slices], sizes)
        while self.gate is not None:
            peaks = surface.peaks(counts, values) if peaks is None else peaks
            near = _runs(2 * (values.max() - values.max(axis=1)) <= REGION * cutoff)
            region = [(self.gate[a], self.gate[b - 1]) for a, b in near]
            region += [(x.gate, x.gate) for x in _retained(peaks, cutoff)]
            finer = surface._split(region, weights)
            if finer is surface:
                break
            moved = np.any(
                finer._before_readout_at(self.gate) != surface._before_readout_at(self.gate), axis=1
            )
            values = values.copy()
            values[moved] = finer.loglik(counts, self.gate[moved])
            surface, peaks = finer, None
        return surface, values, surface.peaks(counts, values) if peaks is None else peaks

    def _split(self, region: Sequence[tuple[float, float]], weights: ByCell) -> _Surface:
        nodes, unread = list(self.gate_nodes), list(self.before_readout)
        k = 0
        while k < len(nodes) - 1:
            a, b = nodes[k], nodes[k + 1]
            middle = (a + b) / 2
            if a < middle < b and any(a <= hi and b >= lo for lo, hi in region):
                exact = self.run(middle)
                if _distance(exact, (unread[k] + unread[k + 1]) / 2, weights) > REFINE_TOL:
                    nodes.insert(k + 1, middle)
                    unread.insert(k + 1, exact)
                    continue
            k += 1
        if len(nodes) == len(self.gate_nodes):
            return self
        finer = copy.copy(self)
        finer.gate_nodes, finer.before_readout = np.array(nodes), np.array(unread)
        return finer

    def peaks(self, counts: ByCell, values: ByGateByReadout) -> list[_Peak]:
        gate, readout = _grid(self.gate), _grid(self.readout)
        starts: set[tuple[int, int, Axis]] = set()
        for a, b in _crests(values.max(axis=1)):
            i, j = _nearest_one(values[a:b], gate[a:b, None], readout[None, :])
            starts.add((a + i, j, "gate"))
        for a, b in _crests(values.max(axis=0)):
            i, j = _nearest_one(values[:, a:b], gate[:, None], readout[None, a:b])
            starts.add((i, a + j, "readout"))
        climbed = [
            (*self._climb(counts, float(gate[i]), float(readout[j]), float(values[i, j])), along)
            for i, j, along in sorted(starts)
        ]
        top = max(peak.loglik for peak, _, _ in climbed)
        return sorted(
            {
                (
                    peak
                    if settled or 2 * (top - peak.loglik) > REGION * CHI2_95
                    else self._ascend(counts, peak)
                )._replace(along=along)
                for peak, settled, along in climbed
            }
        )

    def _climb(
        self, counts: ByCell, best_gate: float, best_readout: float, best: float
    ) -> tuple[_Peak, bool]:
        for span in (_STEP, _STEP / 8):
            near_gate = _around(self.gate, best_gate, span)
            near_readout = _around(self.readout, best_readout, span)
            g, r = np.meshgrid(near_gate, near_readout, indexing="ij")
            window = self.loglik_at(g.ravel(), r.ravel(), counts).reshape(g.shape)
            a, b = _nearest_one(window, g, r)
            if window[a, b] >= best:
                best_gate, best_readout, best = float(g[a, b]), float(r[a, b]), float(window[a, b])
        return _Peak(best_gate, best_readout, best), _settled(window, a, b, g, r)

    def _ascend(self, counts: ByCell, start: _Peak) -> _Peak:
        def readout_at(gate: float, readout: float) -> tuple[float, float]:
            if self.readout is None:
                return readout, float(self.loglik_at([gate], [readout], counts)[0])
            return _zoom(lambda xs: self.loglik_at(np.full(len(xs), gate), xs, counts), readout)

        if self.gate is None:
            readout, value = readout_at(start.gate, start.readout)
            return _Peak(start.gate, readout, value)
        nuisance = {start.gate: start.readout}

        def profile(gates: ByPoint) -> ByPoint:
            out = []
            for x in gates:
                nuisance[x], value = readout_at(float(x), nuisance.get(x, start.readout))
                out.append(value)
            return np.array(out)

        gate, value = _zoom(profile, start.gate)
        return _Peak(gate, nuisance[gate], value)

    def flat_below(self) -> float:
        moved = np.any(self.before_readout != self.before_readout[0], axis=1)
        return float(self.gate_nodes[np.argmax(moved) - 1]) if moved.any() else _LO

    def moves(self, axis: Axis) -> list[bool]:
        nodes, ends = self.gate_nodes, np.array([_LO, _HI])
        probs = self.probs_at(np.repeat(nodes, 2), np.tile(ends, len(nodes)))
        probs = probs.reshape(len(nodes), 2, -1)
        lines = probs.transpose(1, 0, 2) if axis == "gate" else probs
        return [
            any(_varies(line[:, part], self.rounding[part]) for line in lines)
            for part in self.slices
        ]

    def _before_readout_at(self, log_gate: ByPoint) -> ByPointByCell:
        nodes, unread = self.gate_nodes, self.before_readout
        if len(nodes) == 1:
            return np.repeat(unread, len(log_gate), axis=0)
        x = np.clip(log_gate, nodes[0], nodes[-1])
        k = np.clip(np.searchsorted(nodes, x, side="right") - 1, 0, len(nodes) - 2)
        w = ((x - nodes[k]) / (nodes[k + 1] - nodes[k]))[:, None]
        return (1 - w) * unread[k] + w * unread[k + 1]


class _Fit:
    def __init__(
        self,
        surface: _Surface,
        values: ByGateByReadout,
        observed: np.ndarray,
        estimate: _Peak,
        peaks: Sequence[_Peak],
        shots: np.ndarray,
        dispersion: float,
        seed: int,
    ) -> None:
        self.surface = surface
        self.values = values
        self.observed = observed
        self.at: dict[Axis, float] = {"gate": estimate.gate, "readout": estimate.readout}
        self.best = estimate.loglik
        self.peaks = peaks
        self.shots = shots
        self.dispersion = dispersion
        self.cutoff = CHI2_95 * dispersion
        self.flat_below = surface.flat_below()
        self.seed = seed
        self._wilks: dict[Axis, dict[int, float | None]] = {}
        self._peak_ends: dict[Axis, dict[int, float | None]] = {}
        self._windows: tuple[_Window, ...] = ()
        self._restricted: dict[tuple[Axis, float], tuple[float, float]] = {}

    def interval(self, axis: Axis) -> FactorResult:
        at = self.at[axis]
        ends: dict[int, float | None] = {}
        for side, edge in ((-1, _LO), (1, _HI)):
            wilks = self.wilks(axis)[side]
            if wilks is None:
                ends[side] = None
                continue
            start = next(
                (x for x in self._beyond(axis, wilks, side) if self.accepts(axis, x)), wilks
            )
            ends[side] = self._end(axis, start, side, edge)
        low, high = ends[-1], ends[1]
        if low is None and high is None:
            return NoEstimate(f"the counts do not constrain the {axis} factor")
        bound: Bound | None = "lower" if low is None else "upper" if high is None else None
        return ErrorFactor(
            factor=math.exp(at),
            low=None if low is None else math.exp(low),
            high=None if high is None else math.exp(high),
            bound=bound,
        )

    def _beyond(self, axis: Axis, end: float, side: int) -> list[float]:
        coords = {
            peak.gate if axis == "gate" else peak.readout
            for peak in self.peaks
            if peak.along == axis
        }
        return sorted((x for x in coords if (x - end) * side > 0), key=lambda x: -x * side)

    def _end(self, axis: Axis, wilks: float, side: int, edge: float) -> float | None:
        near = self._peak_ends[axis][side]
        half = abs((wilks if near is None else near) - self.at[axis])
        tol = END_TOL * half

        def beyond(x: float) -> bool:
            return (x - edge) * side >= 0

        inside = edge if beyond(wilks + side * tol) else wilks + side * tol
        if inside == wilks or not self.accepts(axis, inside):
            return wilks
        step = 0.25 * half
        while True:
            if inside == edge:
                return None
            candidate = inside + side * step
            if beyond(candidate):
                if self.accepts(axis, edge):
                    return None
                outside = edge
                break
            if not self.accepts(axis, candidate):
                outside = candidate
                break
            inside, step = candidate, 2 * step
        return _bisect(inside, outside, partial(self.accepts, axis), lambda a, b: abs(b - a) <= tol)

    def wilks(self, axis: Axis) -> dict[int, float | None]:
        if axis not in self._wilks:
            at = self.at[axis]
            ends: dict[int, float | None] = {}

            def within_cutoff(held: float) -> bool:
                return self.lr(axis, held) <= self.cutoff

            def resolved(inside: float, outside: float) -> bool:
                return self.lr(axis, outside) - self.lr(axis, inside) <= 2 * ASCENT_TOL

            line = _grid(self.surface.axis(axis))
            profile = self.values.max(axis=1 if axis == "gate" else 0)
            accepted = line[2 * (self.best - profile) <= self.cutoff]
            peak_ends: dict[int, float | None] = {}
            for side, edge in ((-1, _LO), (1, _HI)):
                if within_cutoff(edge):
                    ends[side] = peak_ends[side] = None
                    continue
                end = peak_ends[side] = _bisect(at, edge, within_cutoff, resolved)
                farther = [*accepted[(accepted - end) * side > 0]]
                farther += [x for x in self._beyond(axis, end, side) if within_cutoff(x)]
                if farther:
                    inside = float(max(farther) if side > 0 else min(farther))
                    outward = line[(line - inside) * side > 0][::side]
                    for outside in outward:
                        if not within_cutoff(outside):
                            end = _bisect(inside, float(outside), within_cutoff, resolved)
                            break
                        inside = float(outside)
                ends[side] = end
            self._wilks[axis], self._peak_ends[axis] = ends, peak_ends
        return self._wilks[axis]

    def lr(self, axis: Axis, held: float) -> float:
        return 2 * (self.best - self.restricted(axis, held)[1])

    def restricted(self, axis: Axis, held: float) -> tuple[float, float]:
        if (axis, held) not in self._restricted:
            self._restricted[axis, held] = self._restrict(axis, held)
        return self._restricted[axis, held]

    def _restrict(self, axis: Axis, held: float) -> tuple[float, float]:
        other = self.surface.axis(_other(axis))
        if other is None:
            return 0.0, float(self._line(axis, held, np.zeros(1), self.observed)[0])

        def f(others: ByPoint) -> ByPoint:
            return self._line(axis, held, others, self.observed)

        center = float(other[int(np.argmax(f(other)))])
        line = np.linspace(max(center - _STEP, _LO), min(center + _STEP, _HI), WINDOW_POINTS)
        values = f(line)[:, None]
        k = np.argmax(values, axis=0)
        vertex = _refine(values, k, line)
        x, top = float(line[k[0]] + vertex.shift[0]), float(values[k[0], 0] + vertex.gain[0])
        value = float(f(np.array([x]))[0])
        if not _at_edge(line, int(k[0])) and abs(value - top) <= ASCENT_TOL:
            return x, value
        return _zoom(f, x)

    def accepts(self, axis: Axis, held: float) -> bool:
        observed = self.lr(axis, held)
        if observed <= self.cutoff:
            return True
        nuisance, _ = self.restricted(axis, held)
        point = _point(axis, held, np.array([nuisance]))
        probs = np.where(self.surface.supported, self.surface.probs_at(*point)[0], 0.0)
        outcomes = _enumerated(probs, self.surface.slices, self.shots) or _Outcomes(
            self.draw(probs, np.random.default_rng(self.seed)),
            np.full(RESAMPLES, 1 / (1 + RESAMPLES)),
            1 / (1 + RESAMPLES),
        )
        windows = self.windows()
        center = tuple(float(x[0]) for x in point)
        if not any(_spans(window, *center) for window in windows):
            windows += (self._window_at(*center),)
        every = np.column_stack([self.observed, outcomes.counts])
        top = self.draw_max(every, windows)[0]
        stats = 2 * (top - self.draw_restricted(every, axis, held, windows))
        extreme = stats[1:] >= stats[0] / self.dispersion - 1e-9
        return bool(outcomes.unlisted + outcomes.weights[extreme].sum() > 1 - LEVEL)

    def draw(self, probs: ByCell, rng: np.random.Generator) -> ByCellByDraw:
        probs = np.where(self.surface.supported, probs, 0.0)
        parts: list[np.ndarray] = []
        for part, shots in zip(self.surface.slices, self.shots, strict=True):
            cell = probs[part]
            if shots:
                parts.append(rng.multinomial(int(shots), cell / cell.sum(), size=RESAMPLES).T)
            else:
                parts.append(np.zeros((len(cell), RESAMPLES)))
        return np.concatenate(parts).astype(float)

    def draw_max(self, draws: ByCellByDraw, windows: Sequence[_Window] = ()) -> _Best:
        windows = windows or self.windows()
        gate_span, readout_span = _spacing(windows[0].gate), _spacing(windows[0].readout)

        def climb(counts: ByCellByDraw, start: _Best) -> _Best:
            def profile(
                readout: ByColumn, which: np.ndarray, gate: ByColumn
            ) -> tuple[ByColumn, ByColumn]:
                return self._free_max("readout", readout, counts[:, which], gate, gate_span)

            top, gate = profile(start.readout, np.arange(len(start.top)), start.gate)
            readout, top, gate = _climb(profile, start.readout, top, gate, readout_span)
            return _Best(top, gate, readout)

        starts = [
            self._polish(draws, [_window_max(part, draws) for part in parts], climb)
            for parts in ((window, *window.edges) for window in windows)
        ]
        for gate, readout, narrow in self._kept(windows):
            top = self.surface.loglik_at([gate], [readout], draws)[0]
            start = _Best(top, np.full(len(top), gate), np.full(len(top), readout))
            if narrow:
                start = _climbed(draws, start, np.flatnonzero(np.isfinite(top)), climb)
            starts.append(start)
        return _highest(starts)[0]

    def _kept(self, windows: Sequence[_Window]) -> list[tuple[float, float, bool]]:
        step = [max(_spacing(_grid_line(window, axis)) for window in windows) for axis in _AXES]
        neighbors = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]) * step
        kept = sorted(_retained(self.peaks, self.cutoff), key=lambda peak: -peak.loglik)
        out: list[tuple[float, float, bool]] = []
        for gate, readout, top, _ in kept:
            if any(abs(gate - g) <= _STEP / 8 and abs(readout - r) <= _STEP / 8 for g, r, _ in out):
                continue
            gates, readouts = np.clip(np.array([gate, readout]) + neighbors, _LO, _HI).T
            lowest = self.surface.loglik_at(gates, readouts, self.observed).min()
            out.append((gate, readout, bool(top - lowest > CHI2_95)))
        return out

    def draw_restricted(
        self, draws: ByCellByDraw, axis: Axis, held: float, windows: Sequence[_Window] = ()
    ) -> ByColumn:
        windows = windows or self.windows()
        free = _other(axis)
        span = _spacing(_grid_line(windows[0], free))

        def climb(counts: ByCellByDraw, start: _Best) -> _Best:
            fixed = np.full(len(start.top), held)
            top, others = self._free_max(axis, fixed, counts, _grid_line(start, free), span)
            return _Best(top, *_point(axis, held, others))

        def found(window: _Window) -> tuple[_Best, _Best, np.ndarray]:
            line = _grid_line(window, free)
            values = self._line(axis, held, line, draws)
            k = np.argmax(values, axis=0)
            top, vertex = values[k, np.arange(len(k))], _refine(values, k, line)
            return (
                _Best(top, *_point(axis, held, line[k])),
                _Best(top + vertex.gain, *_point(axis, held, line[k] + vertex.shift)),
                vertex.settled,
            )

        starts = [
            self._polish(draws, [found(part) for part in (window, *window.edges)], climb)
            for window in windows
        ]
        return _highest(starts)[0].top

    def _polish(
        self,
        draws: ByCellByDraw,
        found: Sequence[tuple[_Best, _Best, np.ndarray]],
        climb: Callable[[ByCellByDraw, _Best], _Best],
    ) -> _Best:
        grids, vertices, settled = zip(*found, strict=True)
        vertex, pick = _highest(vertices)
        value = self.surface.loglik_each(vertex.gate, vertex.readout, draws)
        best, _ = _highest([*grids, vertex._replace(top=value)])
        beaten = np.flatnonzero((pick > 0) & ~settled[0])
        settled = np.array(settled)[pick, np.arange(len(pick))]
        with np.errstate(invalid="ignore"):
            loose = np.flatnonzero(~settled | (np.abs(value - vertex.top) > TIE))
        best = _climbed(draws, best, loose, climb)
        return _highest([best, _climbed(draws, grids[0], beaten, climb)])[0]

    def _free_max(
        self, axis: Axis, held: ByColumn, counts: ByCellByDraw, others: ByColumn, span: float
    ) -> tuple[ByColumn, ByColumn]:
        def f(x: ByColumn, which: np.ndarray, _: ByColumn) -> tuple[ByColumn, ByColumn]:
            point = (held[which], x) if axis == "gate" else (x, held[which])
            return self.surface.loglik_each(*point, counts[:, which]), x

        lo = _LO if axis == "gate" else self.flat_below
        others = np.maximum(others, lo)
        top, _ = f(others, np.arange(len(others)), others)
        others, top, _ = _climb(f, others, top, others, span, lo)
        return top, others

    def windows(self) -> tuple[_Window, ...]:
        if not self._windows:
            windows = [self._window_at(self.at["gate"], self.at["readout"])]
            for peak in _retained(self.peaks, self.cutoff):
                if not any(_spans(window, peak.gate, peak.readout) for window in windows):
                    windows.append(self._window_at(peak.gate, peak.readout))
            self._windows = tuple(windows)
        return self._windows

    def _window_at(self, gate: float, readout: float) -> _Window:
        lines: dict[Axis, np.ndarray] = {}
        for axis, center in (("gate", gate), ("readout", readout)):
            if self.surface.axis(axis) is None:
                lines[axis] = np.zeros(1)
                continue
            self.wilks(axis)
            ends = [e for e in self._peak_ends[axis].values() if e is not None]
            half = max((abs(e - self.at[axis]) for e in ends), default=(_HI - _LO) / 4)
            if len(ends) < 2 or half >= _MIN_WINDOW:
                half = max(half, _MIN_WINDOW)
            lines[axis] = np.linspace(
                max(center - 4 * half, _LO), min(center + 4 * half, _HI), WINDOW_POINTS
            )
        gate_line, readout_line = lines["gate"], lines["readout"]
        gate_ends, readout_ends = _unreached(gate_line), _unreached(readout_line)
        pairs = ((gate_ends, readout_line), (gate_line, readout_ends), (gate_ends, readout_ends))
        edges = tuple(self._grid_window(*pair) for pair in pairs if all(map(len, pair)))
        return self._grid_window(gate_line, readout_line, edges)

    def _grid_window(
        self, gate: ByPoint, readout: ByPoint, edges: tuple[_Window, ...] = ()
    ) -> _Window:
        g, r = (a.ravel() for a in np.meshgrid(gate, readout, indexing="ij"))
        return _Window(gate, readout, self.surface.probs_at(g, r), edges)

    def _line(
        self, axis: Axis, held: float, others: ByPoint, counts: ByCell | ByCellByDraw
    ) -> ByPoint | ByPointByDraw:
        return self.surface.loglik_at(*_point(axis, held, others), counts)


@dataclass(frozen=True)
class _Window:
    gate: ByPoint
    readout: ByPoint
    probs: ByPointByCell
    edges: tuple[_Window, ...] = ()


class _Outcomes(NamedTuple):
    counts: ByCellByDraw
    weights: ByColumn
    unlisted: float


def _enumerated(probs: ByCell, slices: Sequence[slice], shots: np.ndarray) -> _Outcomes | None:
    blocks = []
    for part, n in zip(slices, shots, strict=True):
        block = _binomial(probs[part], int(n))
        if block is None:
            return None
        blocks.append(block)
    sizes = [len(pmf) for _, pmf in blocks]
    stride = math.prod(sizes)
    if stride > EXACT_OUTCOMES:
        return None
    flat, pick = np.arange(stride), []
    for size in sizes:
        stride //= size
        pick.append(flat // stride % size)
    counts = np.concatenate([columns[:, k] for (columns, _), k in zip(blocks, pick, strict=True)])
    weights = np.prod([pmf[k] for (_, pmf), k in zip(blocks, pick, strict=True)], axis=0)
    return _Outcomes(counts, weights, max(0.0, 1 - float(weights.sum())))


def _binomial(cell: ByCell, shots: int) -> tuple[ByCellByDraw, ByColumn] | None:
    used = np.flatnonzero(cell > 0)
    if not shots or len(used) == 1:
        columns = np.zeros((len(cell), 1))
        columns[used[:1]] = shots
        return columns, np.ones(1)
    if len(used) > 2:
        return None
    q = float(cell[used[1]] / cell[used].sum())
    spread = 10 * math.sqrt(shots * q * (1 - q)) + 10
    low, high = max(0, math.floor(shots * q - spread)), min(shots, math.ceil(shots * q + spread))
    if high - low >= EXACT_OUTCOMES:
        return None
    k = np.arange(low, high + 1)
    columns = np.zeros((len(cell), len(k)))
    columns[used[0]], columns[used[1]] = shots - k, k
    choose = [math.lgamma(shots + 1) - math.lgamma(x + 1) - math.lgamma(shots - x + 1) for x in k]
    return columns, np.exp(np.array(choose) + k * math.log(q) + (shots - k) * math.log1p(-q))


def _unreached(line: np.ndarray) -> np.ndarray:
    return np.array([x for x in (_LO, _HI) if len(line) > 1 and not line[0] <= x <= line[-1]])


def _spans(window: _Window, gate: float, readout: float) -> bool:
    return all(
        len(line) == 1 or line[0] <= x <= line[-1]
        for line, x in ((window.gate, gate), (window.readout, readout))
    )


class _Best(NamedTuple):
    top: ByColumn
    gate: ByColumn
    readout: ByColumn


def _highest(found: Sequence[_Best]) -> tuple[_Best, ByColumn]:
    pick = np.argmax([best.top for best in found], axis=0)
    columns = np.arange(len(pick))
    return _Best(*(np.array(field)[pick, columns] for field in zip(*found, strict=True))), pick


def _climbed(
    draws: ByCellByDraw,
    best: _Best,
    loose: np.ndarray,
    climb: Callable[[ByCellByDraw, _Best], _Best],
) -> _Best:
    if len(loose):
        climbed = climb(draws[:, loose], _Best(*(field[loose] for field in best)))
        for field, values in zip(best, climbed, strict=True):
            field[loose] = values
    return best


def _grid_line(window: _Window | _Best, axis: Axis) -> np.ndarray:
    return window.gate if axis == "gate" else window.readout


def _window_max(window: _Window, draws: ByCellByDraw) -> tuple[_Best, _Best, np.ndarray]:
    gate, readout = window.gate, window.readout
    values = _loglik(window.probs, draws).reshape(len(gate), len(readout), -1)
    columns = np.arange(values.shape[-1])
    i, j = np.unravel_index(np.argmax(values.reshape(-1, len(columns)), axis=0), values.shape[:2])
    top = values[i, j, columns]

    def at(di: int, dj: int) -> ByColumn:
        return values[
            np.clip(i + di, 0, len(gate) - 1), np.clip(j + dj, 0, len(readout) - 1), columns
        ]

    on_gate, on_readout = _interior(gate, i), _interior(readout, j)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope_gate = np.where(on_gate, (at(1, 0) - at(-1, 0)) / 2, 0.0)
        slope_readout = np.where(on_readout, (at(0, 1) - at(0, -1)) / 2, 0.0)
        curve_gate = np.where(on_gate, at(1, 0) - 2 * top + at(-1, 0), -1.0)
        curve_readout = np.where(on_readout, at(0, 1) - 2 * top + at(0, -1), -1.0)
        corners = at(1, 1) - at(1, -1) - at(-1, 1) + at(-1, -1)
        mixed = np.where(on_gate & on_readout, corners / 4, 0.0)
        det = curve_gate * curve_readout - mixed**2
        step_gate = (mixed * slope_readout - curve_readout * slope_gate) / det
        step_readout = (mixed * slope_gate - curve_gate * slope_readout) / det
        fits = (curve_gate < 0) & (det > 0) & (np.abs(step_gate) <= 1) & (np.abs(step_readout) <= 1)
        step_gate, step_readout = np.where(fits, step_gate, 0.0), np.where(fits, step_readout, 0.0)
        gain = np.where(fits, (slope_gate * step_gate + slope_readout * step_readout) / 2, 0.0)
    vertex = _Best(
        top + gain,
        gate[i] + step_gate * _spacing(gate),
        readout[j] + step_readout * _spacing(readout),
    )
    drop = np.maximum(
        np.where(on_gate, top - np.minimum(at(1, 0), at(-1, 0)), 0.0),
        np.where(on_readout, top - np.minimum(at(0, 1), at(0, -1)), 0.0),
    )
    settled = fits & (drop <= CHI2_95) & ~_stuck(gate, i) & ~_stuck(readout, j)
    return _Best(top, gate[i], readout[j]), vertex, settled


def _interior(line: np.ndarray, k: ByColumn) -> ByColumn:
    return (len(line) >= 3) & (k > 0) & (k < len(line) - 1)


def _stuck(line: np.ndarray, k: ByColumn) -> ByColumn:
    return (len(line) > 1) & ~_interior(line, k) & (line[k] > _LO) & (line[k] < _HI)


def _climb(
    f: Callable[[ByColumn, np.ndarray, ByColumn], tuple[ByColumn, ByColumn]],
    x: ByColumn,
    top: ByColumn,
    aux: ByColumn,
    span: float,
    lo: float = _LO,
) -> tuple[ByColumn, ByColumn, ByColumn]:
    x, top, aux = x.copy(), top.copy(), aux.copy()
    spans = np.full(len(x), span)
    active = np.flatnonzero(spans > 0)
    while len(active):
        center, value, h = x[active], top[active], spans[active]
        low, high = np.maximum(center - h, lo), np.minimum(center + h, _HI)
        both = np.tile(active, 2)
        sides, aux_sides = f(np.concatenate([low, high]), both, aux[both])
        (f_low, f_high), (aux_low, aux_high) = np.split(sides, 2), np.split(aux_sides, 2)
        up = f_high > f_low
        moved = np.where(up, f_high, f_low) > value
        flat = ~moved & (
            ((low == center) | (value - f_low <= ASCENT_TOL))
            & ((high == center) | (value - f_high <= ASCENT_TOL))
        )
        a, b, da, db = low - center, high - center, f_low - value, f_high - value
        with np.errstate(divide="ignore", invalid="ignore"):
            curve = (db / b - da / a) / (b - a)
            slope = da / a - curve * a
            vertex, predicted = center - slope / (2 * curve), value - slope**2 / (4 * curve)
        fits = np.flatnonzero(~moved & ~flat & (curve < 0) & np.isfinite(predicted))
        f_vertex, aux_vertex = f(vertex[fits], active[fits], aux[active[fits]])
        better = f_vertex > value[fits]
        fine = (np.abs(f_vertex - predicted[fits]) <= ASCENT_TOL) & (
            -np.minimum(da, db)[fits] <= CHI2_95
        )
        side = active[moved]
        x[side] = np.where(up, high, low)[moved]
        top[side] = np.where(up, f_high, f_low)[moved]
        aux[side] = np.where(up, aux_high, aux_low)[moved]
        gained = active[fits][better]
        x[gained] = vertex[fits][better]
        top[gained] = f_vertex[better]
        aux[gained] = aux_vertex[better]
        done = flat.copy()
        done[fits] |= fine
        spans[active[moved]] *= 2
        spans[active[~moved & ~done]] /= 8
        active = active[~done]
    return x, top, aux


def _bind(profile: Profile, counts: MeasuredCounts) -> Profile:
    if counts.backend != profile.device.name:
        raise CountsError(
            f"these counts ran on {counts.backend}, but the profile describes"
            f" {profile.device.name}",
            hint="give the profile the counts were planned from",
        )
    planned, calibration = counts.profile.fingerprint, profile.calibration_fingerprint
    if planned != calibration:
        raise CountsError(
            f"these counts were planned from nv:{planned[:12]}, but this profile's calibration is"
            f" nv:{calibration[:12]}",
            hint=f"run `nv list` to find nv:{planned[:12]}",
        )
    when = profile.device.calibrated_at
    if when is not None and counts.run_at < when:
        raise CountsError(
            f"these counts ran at {counts.run_at:%Y-%m-%d %H:%MZ}, before the calibration they"
            f" were planned from, taken at {when:%Y-%m-%d %H:%MZ}",
            hint="check run_at in the counts file",
        )
    base = profile.uncorrected()
    table = base.table
    report = Report.start(base, "reference", None)
    for c in counts.circuits:
        for q in c.qubits:
            if not 0 <= q < table.num_qubits:
                raise CountsError(
                    f"circuit {c.name} measures qubit {q}, but {profile.id} has qubits"
                    f" 0..{table.num_qubits - 1}"
                )
            if table.qubit(q).disabled:
                raise CountsError(
                    f"circuit {c.name} measures qubit {q}, which {profile.id} marks disabled"
                )
            try:
                refuse_disabled(table.gate("measure", (q,)))
            except DisabledGateError as exc:
                raise CountsError(
                    f"circuit {c.name}: {exc.message}",
                    hint="run the circuits on qubits that the profile can measure",
                ) from None
        try:
            for op in c.ops:
                targets = [c.qubits[q] for q in op.qubits]
                if op.name == "delay":
                    refuse_disabled(table.gate("delay", targets))
                    continue
                resolve_op(
                    table,
                    charged_as(base, op.name, gates.unitary(op.name, op.params)),
                    targets,
                    unknown_gates="error",
                    report=report,
                )
        except (MissingCalibrationError, DisabledGateError, LayoutError) as exc:
            raise CountsError(
                f"circuit {c.name}: {exc.message}",
                hint="nv compare scores only ops the profile calibrates on their qubits",
            ) from None
    return base


def _floor_factors(base: Profile, circuits: Sequence[PlannedCircuit]) -> tuple[float, ...]:
    table, floors = base.table, set()
    for found, infidelity in _calibrated_gates(base, circuits):
        if found.pauli:
            continue
        n = len(found.qubits)
        built = gate_channels(found, [table.qubit(q) for q in found.qubits])
        stated = metrics.depolarizing_from_avg(infidelity, n)
        relaxed = metrics.depolarizing_from_avg(built.relaxation, n)
        if 0 < relaxed < 1 and 0 < stated < 1:
            factor = math.log1p(-relaxed) / math.log1p(-stated)
            if FACTOR_RANGE[0] < factor < FACTOR_RANGE[1]:
                floors.add(factor)
    return tuple(sorted(floors))


def _calibrated_gates(
    base: Profile, circuits: Sequence[PlannedCircuit]
) -> Iterator[tuple[GateNoise, float]]:
    for c in circuits:
        for op in c.ops:
            if op.name == "delay":
                continue
            name = charged_as(base, op.name, gates.unitary(op.name, op.params))
            found = base.table.gate(name, tuple(c.qubits[q] for q in op.qubits))
            if isinstance(found, GateNoise) and found.avg_infidelity is not None:
                yield found, found.avg_infidelity


def _fisher(
    base: Profile,
    circuits: Sequence[PlannedCircuit],
    shots: np.ndarray,
    center: np.ndarray,
    gate: float | None,
    readout: float | None,
) -> np.ndarray:
    step = math.exp(FD_STEP)
    sizes = [2 ** len(c.qubits) for c in circuits]
    ends = np.cumsum(sizes)[:-1]
    rounding = np.split(_rounding(circuits), ends)
    slopes = []
    for i, value in enumerate((gate, readout)):
        if value is None:
            slopes.append(np.zeros_like(center))
            continue
        up, down = [gate, readout], [gate, readout]
        up[i], down[i] = value * step, value / step
        rows = np.stack([_exact(base, circuits, *up), _exact(base, circuits, *down)])
        moves = [
            _varies(part, bound)
            for part, bound in zip(np.split(rows, ends, axis=1), rounding, strict=True)
        ]
        slopes.append(np.repeat(moves, sizes) * (rows[0] - rows[1]) / (2 * FD_STEP))
    jacobian = np.stack(slopes)
    weights = np.repeat(shots, sizes)
    keep = center > IMPOSSIBLE
    return (jacobian[:, keep] * (weights[keep] / center[keep])) @ jacobian[:, keep].T


def _identify(
    surface: _Surface, counts: np.ndarray, info: np.ndarray
) -> dict[Axis, NoEstimate | None]:
    shots = [counts[part].sum() for part in surface.slices]
    if not sum(shots):
        every = NoEstimate("the profile rules out every shot")
        return {"gate": every, "readout": every}
    out: dict[Axis, NoEstimate | None] = {}
    for axis in _AXES:
        if surface.axis(axis) is None:
            out[axis] = NoEstimate(_absent(surface, axis))
            continue
        responding = int(
            sum(n for n, moves in zip(shots, surface.moves(axis), strict=True) if moves)
        )
        if responding < MIN_INFORMATIVE_SHOTS:
            out[axis] = NoEstimate(f"only {plural(responding, 'shot')} respond to {axis} error")
        else:
            out[axis] = None
    if out["gate"] is None and out["readout"] is None:
        diagonal = np.diag(info)
        flat = diagonal <= FLAT * diagonal.max()
        if not flat.any() and _rank(info) < len(info):
            same = NoEstimate(_SAME_WAY)
            out = {"gate": same, "readout": same}
    return out


def _rank(info: np.ndarray) -> int:
    eigen = np.linalg.eigvalsh(info)
    return int(np.sum(eigen > SEPARABLE * eigen[-1])) if eigen[-1] > 0 else 0


def _absent(surface: _Surface, axis: Axis) -> str:
    if axis == "gate":
        return "no circuit's outcomes move with gate error"
    if any(pair and sum(pair) for pairs in surface.readout_pairs for pair in pairs):
        return "the readout error of the measured qubits does not scale"
    return "the measured qubits state no readout error"


def _ruled_out(
    surface: _Surface, circuits: Sequence[PlannedCircuit], impossible: ByCell
) -> tuple[str, ...]:
    found: dict[tuple[int, int], dict[str, int]] = {}
    for c, part, pairs in zip(circuits, surface.slices, surface.readout_pairs, strict=True):
        shots = impossible[part]
        if not shots.any():
            continue
        width = len(c.qubits)
        unread = surface.before_readout[:, part].reshape(-1, *(2,) * width)
        for i, pair in enumerate(pairs):
            if pair is None:
                continue
            marginal = unread.sum(axis=tuple(a for a in range(1, width + 1) if a != i + 1))
            for bit, stated in ((1, pair[0]), (0, pair[1])):
                if stated or marginal[:, bit].max() > IMPOSSIBLE:
                    continue
                cells = [k for k in np.flatnonzero(shots) if outcome_bits(k, width)[i] == str(bit)]
                if cells:
                    per = found.setdefault((c.qubits[i], bit), {})
                    per[c.name] = per.get(c.name, 0) + int(shots[cells].sum())
    phrases = []
    for (qubit, bit), per in sorted(found.items()):
        error = "P(1|0)" if bit else "P(0|1)"
        runs = joined([plural(n, f"{name} shot") for name, n in per.items()])
        phrases.append(f"qubit {qubit} has {error} = 0, and {runs} read it as {bit}")
    return tuple(phrases)


def _notes(base: Profile, circuits: Sequence[PlannedCircuit]) -> tuple[LociText, ...]:
    table = base.table
    gates_left = [
        (found.gate, found.qubits, reason)
        for found, _ in _calibrated_gates(base, circuits)
        if (reason := _why_unscaled(found))
    ]
    measured = sorted({q for c in circuits for q in c.qubits})
    pairs = [(q, table.qubit(q).readout) for q in measured]
    chance = [
        (q, reason)
        for q, pair in pairs
        if pair is not None and (reason := metrics.readout_unscalable(pair))
    ]
    notes = unscaled_phrases(gates_left, chance)
    idle = sorted(
        {
            c.qubits[op.qubits[0]]
            for c in circuits
            for op in c.ops
            if op.name == "delay" and table.qubit(c.qubits[op.qubits[0]]).relaxation_unknown
        }
    )
    if idle:
        on = [(q,) for q in idle]
        notes.append(LociText("delays on ", on, " add no idle error (no T1 or T2 stated)"))
    report = Report.start(base, "reference", None)
    report.record_effects(base.effects)
    if report.omitted:
        left_out = joined([what.full for what in report.omitted])
        notes.append(
            LociText(
                f"the reference simulator leaves out {left_out}, because the profile"
                " sets allow to 'omit'"
            )
        )
    return tuple(notes)


def _why_unscaled(found: GateNoise) -> str | None:
    metric = found.spec.metric
    return None if metric is None else metrics.unscalable(*metric, len(found.qubits))


def _exact(
    base: Profile,
    circuits: Sequence[PlannedCircuit],
    gate: float | None,
    readout: float | None,
    *,
    read: bool = True,
) -> ByCell:
    scaled = _scaled_without_revalidation(base, gate, readout)
    report = Report.start(scaled, "reference", None)
    return np.concatenate(
        [
            probabilities(
                scaled,
                c.ops,
                len(c.qubits),
                layout=c.qubits,
                readout=read,
                unknown_gates="error",
                report=report,
            )
            for c in circuits
        ]
    )


def _scaled_without_revalidation(
    base: Profile, gate: float | None, readout: float | None
) -> Profile:
    if gate is None and readout is None:
        return base
    unmodeled = UnmodeledError(
        gates=None if gate is None else ErrorFactor(factor=gate),
        readout=None if readout is None else ErrorFactor(factor=readout),
    )
    fields = {name: getattr(base, name) for name in type(base).model_fields}
    return type(base).model_construct(**{**fields, "unmodeled_error": unmodeled})


def _paired(probs: ByPointByCell, counts: ByCellByDraw) -> ByColumn:
    zero = probs <= 0
    with np.errstate(divide="ignore"):
        out = np.einsum("kc,ck->k", np.where(zero, 0.0, np.log(probs)), counts)
    out[(zero & (counts.T > 0)).any(axis=1)] = -np.inf
    return out


def _loglik(probs: ByPointByCell, counts: ByCell | ByCellByDraw) -> ByPoint | ByPointByDraw:
    counts = np.asarray(counts, dtype=float)
    zero = probs <= 0
    with np.errstate(divide="ignore"):
        out = np.where(zero, 0.0, np.log(probs)) @ counts
    if zero.any():
        out[(zero.astype(float) @ (counts > 0)) > 0] = -np.inf
    return out


def _dispersion(deviance: float, dof: int) -> float:
    return max(1.0, deviance / dof) if dof > 0 else 1.0


def _deviance(counts: ByCell, probs: ByCell, slices: Sequence[slice]) -> float:
    total = 0.0
    for part in slices:
        n, p = counts[part], probs[part]
        seen = n > 0
        with np.errstate(divide="ignore"):
            total += 2 * float(np.sum(n[seen] * np.log(n[seen] / (n.sum() * p[seen]))))
    return total


def _saturated(counts: ByCellByDraw, slices: Sequence[slice]) -> ByColumn:
    total = np.zeros(counts.shape[1])
    for part in slices:
        n = counts[part]
        with np.errstate(divide="ignore", invalid="ignore"):
            total += np.where(n > 0, n * np.log(n / n.sum(axis=0)), 0.0).sum(axis=0)
    return total


class _Peak(NamedTuple):
    gate: float
    readout: float
    loglik: float
    along: Axis = "gate"


def _retained(peaks: Sequence[_Peak], cutoff: float) -> list[_Peak]:
    top = max(peak.loglik for peak in peaks)
    return [peak for peak in peaks if 2 * (top - peak.loglik) <= REGION * cutoff]


def _estimate(peaks: Sequence[_Peak]) -> _Peak:
    top = max(peak.loglik for peak in peaks)
    tied = [peak for peak in peaks if peak.loglik >= top - TIE]
    return min(tied, key=lambda peak: abs(peak.gate) + abs(peak.readout))


def _crests(profile: ByPoint) -> list[tuple[int, int]]:
    starts = np.flatnonzero(np.concatenate([[True], profile[1:] != profile[:-1]]))
    stops = np.append(starts[1:], len(profile))
    heights = np.concatenate([[-np.inf], profile[starts], [-np.inf]])
    crest = (heights[1:-1] > heights[:-2]) & (heights[1:-1] > heights[2:])
    return [(int(a), int(b)) for a, b, top in zip(starts, stops, crest, strict=True) if top]


def _settled(window: np.ndarray, a: int, b: int, g: np.ndarray, r: np.ndarray) -> bool:
    top = window[a, b]
    near = [
        window[i, j]
        for i, j in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1))
        if 0 <= i < window.shape[0] and 0 <= j < window.shape[1]
    ]
    flat = window.max() - window.min() <= ASCENT_TOL
    return (flat or _inside(window, a, b, g, r)) and top - min(near, default=top) <= ASCENT_TOL


def _zoom(f: Callable[[ByPoint], ByPoint], x: float) -> tuple[float, float]:
    best = float(f(np.array([x]))[0])
    span = _STEP / 8
    while True:
        line = np.unique(
            np.append(np.linspace(max(x - span, _LO), min(x + span, _HI), WINDOW_POINTS), x)
        )
        values = f(line)
        k = int(np.argmax(values))
        if values[k] > best:
            x, best = float(line[k]), float(values[k])
            if _at_edge(line, k):
                span *= 2
                continue
        near = values[max(k - 1, 0) : k + 2]
        finer = 2 * float(np.diff(line).max())
        if best - near.min() <= ASCENT_TOL or finer >= span:
            return x, best
        span = finer


def _at_edge(line: np.ndarray, k: int) -> bool:
    return k in (0, len(line) - 1) and _LO < line[k] < _HI


def _inside(window: np.ndarray, a: int, b: int, g: np.ndarray, r: np.ndarray) -> bool:
    for k, size, at in ((a, window.shape[0], g[a, b]), (b, window.shape[1], r[a, b])):
        if size > 1 and k in (0, size - 1) and _LO < at < _HI:
            return False
    return True


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def _distance(exact: ByCell, line: ByCell, weights: ByCell) -> float:
    total = exact + line
    seen = total > 0
    return float((weights[seen] * (exact - line)[seen] ** 2 / (total[seen] / 2)).sum())


def _nearest_one(values: np.ndarray, gate: np.ndarray, readout: np.ndarray) -> tuple[int, int]:
    top = float(values.max())
    ties = values >= top - max(1e-9, 1e-12 * abs(top))
    distance = np.where(ties, np.abs(gate) + np.abs(readout), np.inf)
    i, j = np.unravel_index(int(np.argmin(distance)), values.shape)
    return int(i), int(j)


class _Vertex(NamedTuple):
    shift: ByColumn
    gain: ByColumn
    settled: ByColumn


def _refine(values: ByPointByColumn, best: ByColumn, line: np.ndarray) -> _Vertex:
    zero = np.zeros(values.shape[1])
    if len(values) < 3:
        return _Vertex(zero, zero, ~_stuck(line, best))
    columns = np.arange(values.shape[1])
    inner = np.clip(best, 1, len(values) - 2)
    left, mid, right = (values[inner + d, columns] for d in (-1, 0, 1))
    curve = 2 * mid - left - right
    usable = (best == inner) & (curve > 0) & np.isfinite(left) & np.isfinite(right)
    with np.errstate(invalid="ignore", divide="ignore"):
        shift = np.where(usable, 0.5 * (right - left) / curve * _spacing(line), 0.0)
        gain = np.where(usable, (right - left) ** 2 / (8 * curve), 0.0)
    resolved = usable & (mid - np.minimum(left, right) <= CHI2_95)
    return _Vertex(shift, gain, np.where(best == inner, resolved, ~_stuck(line, best)))


def _point(axis: Axis, held: float, others: ByPoint) -> tuple[ByPoint, ByPoint]:
    fixed = np.full(len(others), held)
    return (fixed, others) if axis == "gate" else (others, fixed)


def _other(axis: Axis) -> Axis:
    return "readout" if axis == "gate" else "gate"


def _grid(axis: np.ndarray | None) -> np.ndarray:
    return np.zeros(1) if axis is None else axis


def _around(axis: np.ndarray | None, center: float, span: float) -> np.ndarray:
    if axis is None:
        return np.zeros(1)
    line = np.linspace(max(center - span, _LO), min(center + span, _HI), WINDOW_POINTS)
    return np.unique(np.append(line, center))


def _spacing(line: np.ndarray) -> float:
    return float(line[1] - line[0]) if len(line) > 1 else 0.0


def _read(
    block: ByPointByQubitBit, axis: int, matrices: ByPointByMeasuredByPrepared
) -> ByPointByQubitBit:
    m = matrices.reshape(len(matrices), 2, 2, *(1,) * (block.ndim - 2))
    zero, one = np.take(block, 0, axis=axis), np.take(block, 1, axis=axis)
    return np.stack(
        [m[:, 0, 0] * zero + m[:, 0, 1] * one, m[:, 1, 0] * zero + m[:, 1, 1] * one], axis=axis
    )


def _confusion(pair: tuple[float, float], log_factors: ByPoint) -> ByPointByMeasuredByPrepared:
    scaled = [metrics.scale_readout(pair, math.exp(x)) for x in log_factors]
    a, b = np.array(scaled).reshape(-1, 2).T
    return np.stack([np.stack([1 - a, b], -1), np.stack([a, 1 - b], -1)], -2)


def _bisect(
    inside: float,
    outside: float,
    accepts: Callable[[float], bool],
    close: Callable[[float, float], bool],
) -> float:
    while not close(inside, outside):
        middle = (inside + outside) / 2
        if middle in (inside, outside):
            break
        if accepts(middle):
            inside = middle
        else:
            outside = middle
    return inside


def _varies(rows: ByPointByCell, rounding: ByCell) -> bool:
    return bool((np.abs(rows - rows[0]) > rounding).any())


def _rounding(circuits: Sequence[PlannedCircuit]) -> ByCell:
    steps = [(len(c.ops) + len(c.qubits) + 1) * EPS for c in circuits]
    return np.repeat(steps, [2 ** len(c.qubits) for c in circuits])


def _tvd(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def _saved(estimate: ErrorFactor) -> dict[str, Any]:
    values = [v for v in (estimate.factor, estimate.low, estimate.high) if v is not None]
    digits = 6
    while len({f"{v:.{digits}g}" for v in values}) < len(set(values)):
        digits += 1

    def rounded(value: float | None) -> float | None:
        return None if value is None else float(f"{value:.{digits}g}")

    return {
        "factor": rounded(estimate.factor),
        "low": rounded(estimate.low),
        "high": rounded(estimate.high),
        "bound": estimate.bound,
    }


def _factor_dict(result: FactorResult) -> dict[str, Any]:
    if isinstance(result, NoEstimate):
        return {"not_identified": result.reason}
    return result.model_dump()


def _beyond_noise(score: CircuitScore) -> bool:
    return round(score.tvd_fitted, _TVD_DIGITS) > round(score.shot_noise_95, _TVD_DIGITS)


def _say(result: FactorResult) -> str:
    return result.describe() if isinstance(result, ErrorFactor) else "not identified"


def _reason(result: FactorResult) -> str | None:
    return result.reason if isinstance(result, NoEstimate) else None


def _duration(age: timedelta) -> str:
    minutes = age.total_seconds() / 60
    if minutes < 60:
        return f"{round(minutes)} min"
    if minutes < 48 * 60:
        return f"{round(minutes / 60)} h"
    return f"{round(minutes / 1440)} days"
