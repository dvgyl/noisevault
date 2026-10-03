"""Command line: ``noisevault`` (alias ``nv``).

Every command prints plain results on stdout and ``error:``, ``hint:`` and ``warning:`` lines
on stderr. Expected failures exit with status 1 and no traceback. ``--json`` output is for
machines and never contains color. The command line reads NO_COLOR and COLUMNS at each
invocation.
"""

from __future__ import annotations

import errno
import json
import os
import platform
import re
import sys
import warnings
import zlib
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, NamedTuple, NoReturn, get_args

import typer
from pydantic import ValidationError
from rich.console import Console, RenderableType
from rich.table import Table
from rich.text import Text
from typer.core import TyperCommand, TyperGroup
from typer.main import get_click_type

from . import __version__, catalog
from .diff import (
    METRICS,
    describe_delta,
    distinguishing_stamps,
    fmt_error,
    fmt_metric,
    fmt_relative,
    fmt_us,
)
from .errors import (
    REPOSITORY,
    NoiseVaultError,
    ProfileNotFound,
    did_you_mean,
    install_hint,
    joined,
    plural,
    qubit_loci,
)
from .profile import (
    POOR_FIT_P_VALUE,
    Profile,
    Technology,
    exact_ref,
    gate_stats,
    iso_z,
    json_bytes,
    json_schema,
    qubit_medians,
    read_json_file,
    ref_on_day,
    unmodeled_note,
)
from .table import GateNoise

if TYPE_CHECKING:
    from .check import CheckResult, FrameworkCheck
    from .compare import SummaryLine
    from .counts import MeasuredCounts

# Click's UsageError. typer exports only this subclass of UsageError.
_USAGE_ERROR = typer.BadParameter.__mro__[1]
_CLICK_ERRORS = sys.modules[_USAGE_ERROR.__module__]
# A bare `nv` raises this error after it prints the help. Older click has no such class.
_NO_ARGS = getattr(_CLICK_ERRORS, "NoArgsIsHelpError", ())
_STRING = get_click_type(annotation=str, parameter_info=typer.Argument())


class _Command(TyperCommand):
    """A usage line names an argument as REF, not {REF}."""

    def collect_usage_pieces(self, ctx: Any) -> list[str]:
        return [piece.strip("{}") for piece in super().collect_usage_pieces(ctx)]


class _App(typer.Typer):
    def command(self, *args: Any, cls: Any = None, **kwargs: Any) -> Any:
        return super().command(*args, cls=cls or _Command, **kwargs)


class _PlainString(type(_STRING)):
    """A str argument whose help names it by its metavar alone, with no <str> beside it."""

    def get_metavar(self, param: Any, ctx: Any = None) -> str:
        return ""


_PLAIN = _PlainString()


def _argument(metavar: str, text: str) -> Any:
    return typer.Argument(metavar=metavar, click_type=_PLAIN, help=text)


class _Commands(TyperGroup):
    """A usage mistake or an unexpected failure ends in one ``error:`` line, never a box or a
    traceback. NOISEVAULT_DEBUG=1 lets an unexpected failure raise."""

    def parse_args(self, ctx: Any, args: list[str]) -> list[str]:
        try:
            return super().parse_args(ctx, args)
        except _NO_ARGS as exc:
            if help_text := exc.format_message():  # empty when rich has printed it already
                typer.echo(help_text)
            raise typer.Exit() from None
        except _USAGE_ERROR as exc:
            _usage_error(exc)

    def format_help(self, ctx: Any, formatter: Any) -> None:
        # typer 0.27.0 joins epilog= lines into one line, so this method writes the epilog.
        super().format_help(ctx, formatter)
        formatter.write(_START)

    def invoke(self, ctx: Any) -> Any:
        try:
            return super().invoke(ctx)
        except _USAGE_ERROR as exc:
            _usage_error(exc)
        except Exception as exc:
            # Exit and Abort, from typer or click, carry an exit code: they are control flow.
            if hasattr(exc, "exit_code") or os.environ.get("NOISEVAULT_DEBUG"):
                raise
            _fail(
                f"unexpected {type(exc).__name__}: {exc}",
                f"report this bug at {REPOSITORY}/issues (NOISEVAULT_DEBUG=1 shows the traceback)",
            )


_VALUE_HINTS = {
    "--qubits": "qubit indices, for example --qubits 0,1,2",
    "--tech": "a technology, for example --tech trapped_ion",
    "--vendor": "a vendor, for example --vendor ibm",
    "--framework": "one or more of qiskit,cirq,pennylane,stim, for example --framework cirq,stim",
    "--top": "a count, for example --top 10",
    "--at": "a date or time, for example --at 2025-02-26",
    "--source": "ibm, ibm-account or ionq, for example --source ionq",
    "--output": "a file name, for example --output fez.json",
    "-o": "a file name, for example -o fez.json",
}


def _usage_error(exc: Any) -> NoReturn:
    command = exc.ctx.command_path if exc.ctx else "nv"
    if isinstance(exc, _CLICK_ERRORS.NoSuchOption):
        params = exc.ctx.command.get_params(exc.ctx) if exc.ctx else []
        guess = did_you_mean(exc.option_name, [opt for p in params for opt in p.opts]).rstrip()
        message = f"no such option '{exc.option_name}'" + (f"; {guess}" if guess else "")
        _fail(message, None if guess else f"run {command} --help", code=2)
    message = exc.format_message().rstrip(".").replace(". Did you mean", "; did you mean")
    message = message[:1].lower() + message[1:]
    option = getattr(exc, "option_name", None)
    if option in _VALUE_HINTS:
        hint = f"give {_VALUE_HINTS[option]}"
    else:
        hint = None if "did you mean" in message else f"run {command} --help"
    _fail(message, hint, code=2)


_REF_HELP = "Profile id (ibm_fez), id@date, or a file path."
_START = """
 Start with:
   nv list             the bundled devices, offline
   nv show ibm_fez     one device's calibration
   nv check ibm_fez    each export against the reference
"""

app = _App(
    cls=_Commands,
    no_args_is_help=True,
    add_completion=False,
    help="NoiseVault: real device noise, pinned and portable.",
)
out = Console(highlight=False)
err = Console(stderr=True, highlight=False, soft_wrap=True)

# Package doctor reports -> the extra that installs it. pymatching is in no extra: only the QEC
# recipe uses it.
_PACKAGES: dict[str, str | None] = {
    "qiskit": "qiskit",
    "qiskit-aer": "qiskit",
    "qiskit-ibm-runtime": "ibm",
    "cirq-core": "cirq",
    "cirq-google": "google",
    "pennylane": "pennylane",
    "stim": "stim",
    "pyarrow": "hf",
    "pymatching": None,
}
_EXTRAS = sorted({extra for extra in _PACKAGES.values() if extra})
_INSTALL_ALL = install_hint("all")


class _SourceWords(NamedTuple):
    list_label: str
    show_phrase: str


_SOURCE_KINDS = {
    "package_snapshot": _SourceWords("SDK", "SDK snapshot"),
    "public_api": _SourceWords("public API", "public API"),
    "account_api": _SourceWords("account", "account API"),
    "user_file": _SourceWords("imported", "imported file"),
    "published_data": _SourceWords("published", "published data"),
    "vendor_sample": _SourceWords("sample", "vendor sample"),
    "hand_written": _SourceWords("by hand", "written by hand"),
    "derived": _SourceWords("derived", "derived from another profile"),
}
_WORDS = {
    "ibm": "IBM",
    "google": "Google",
    "quantinuum": "Quantinuum",
    "ionq": "IonQ",
    "iqm": "IQM",
    "rigetti": "Rigetti",
    "quera": "QuEra",
    "coherent_overrotation": "coherent over-rotation",
    "crosstalk_zz": "ZZ crosstalk",
    "crosstalk_measurement": "measurement crosstalk",
}
_REDISTRIBUTION = {
    "yes": "redistribution allowed",
    "no": "redistribution not allowed",
    "unknown": "redistribution unknown",
}
_FOREIGN_ERROR_HINTS: tuple[tuple[type[BaseException], str], ...] = (
    (ImportError, f"install the frameworks: {_INSTALL_ALL}"),
    (ValidationError, "run nv validate FILE to list every problem in the file"),
    (FileNotFoundError, "check the path, or give a profile id such as ibm_fez"),
)
_EXPECTED = (
    NoiseVaultError,
    ValidationError,
    OSError,
    ValueError,
    ImportError,
    EOFError,
    zlib.error,
)


def _version(value: bool) -> None:
    if value:
        out.print(f"noisevault {__version__}", markup=False)
        raise typer.Exit()


@app.callback()
def main(
    _: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version, is_eager=True, help="Print the version and exit."
        ),
    ] = False,
) -> None:
    """NoiseVault: real device noise, pinned and portable."""
    global out, err
    out = Console(highlight=False)
    err = Console(stderr=True, highlight=False, soft_wrap=True)


# list ---------------------------------------------------------------------------------------


@app.command("list")
def list_profiles(
    tech: Annotated[
        str | None,
        typer.Option(
            "--tech", metavar="NAME", help="Only this technology, for example trapped_ion."
        ),
    ] = None,
    vendor: Annotated[
        str | None, typer.Option("--vendor", metavar="NAME", help="Only this vendor.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """List the profiles you can load offline: the bundled set and your vault."""
    with _friendly():
        technologies = get_args(Technology)
        if tech is not None:
            tech = tech.lower().replace("-", "_")
        if tech is not None and tech not in technologies:
            guess = did_you_mean(tech, technologies)
            raise ValueError(f"--tech {tech!r}: {guess}choose from {', '.join(technologies)}")
        vendors = sorted({i.vendor for i in catalog.profiles() if i.vendor})
        if vendor is not None:
            vendor = next((v for v in vendors if v.lower() == vendor.lower()), vendor)
        if vendor is not None and vendor not in vendors:
            guess = did_you_mean(vendor, vendors)
            raise ValueError(f"--vendor {vendor!r}: {guess}choose from {', '.join(vendors)}")
        infos = catalog.profiles(technology=tech, vendor=vendor)
        rows = [_list_row(info) for info in infos]
        if as_json:
            _echo_json(rows)
            return
        if not rows:
            wanted = " ".join(x for x in (vendor, tech) if x)
            raise ProfileNotFound(f"no {wanted} profile yet", hint="run nv list to see them all")
        same_day: dict[tuple[str, Any], list[datetime]] = {}
        for info in infos:
            if info.calibrated_at:
                same_day.setdefault((info.id, info.calibrated_at.date()), []).append(
                    info.calibrated_at
                )
        for row, info in zip(rows, infos, strict=True):
            stamp = info.calibrated_at
            day = same_day[info.id, stamp.date()] if stamp else []
            row["when"] = _when(stamp, day) if stamp else "undated"
            row["short_ref"] = info.ref if len(day) > 1 else ref_on_day(info.id, stamp)
            row["shown_license"] = (row["license"] or "unknown").split(" (")[0]
        newest_first = sorted(
            zip(rows, infos, strict=True),
            key=lambda pair: (pair[1].id, -_epoch(pair[1].calibrated_at)),
        )
        rows = [row for row, _ in newest_first]
        licenses = {row["shown_license"] for row in rows}
        shared = licenses.pop() if len(licenses) == 1 and "unknown" not in licenses else None
        columns = ["id", "date", "qubits", "processor", "source"]
        if shared is None:
            columns.append("license")
        table = _fit(lambda shown: _list_table(rows, shown), columns, _LIST_DROPS)
        _emit(table, natural_width=_natural_width(table))
        if any(r["location"] == "vault" for r in rows):
            _emit("* in your vault (nv doctor shows its folder)")
        count = plural(len(rows), "profile")
        if shared:
            count += f", all {shared}" if len(rows) > 1 else f", {shared}"
        elif "license" not in columns:
            count += f", {_licenses(rows)}"
        _emit(f"{count}. See one with nv show <id>.")
        _emit('Load one in Python with nv.load("<id>").')


_LIST_DROPS = ("source", "processor", "license")


def _fit(build: Callable[[list[str]], Table], columns: list[str], drops: Sequence[str]) -> Table:
    table = build(columns)
    for column in drops:
        if _natural_width(table) <= out.width:
            break
        if column in columns:
            columns.remove(column)
            table = build(columns)
    return table


def _natural_width(table: Table) -> int:
    return out.measure(table, options=out.options.update_width(10_000)).maximum


def _licenses(rows: list[dict[str, Any]]) -> str:
    by_license: dict[str, list[str]] = {}
    for row in rows:
        by_license.setdefault(row["shown_license"], []).append(row["short_ref"])
    common, *others = sorted(by_license, key=lambda name: -len(by_license[name]))
    named = {name: "license unknown" if name == "unknown" else name for name in by_license}
    if not others:
        return named[common]
    exceptions = "; ".join(f"{', '.join(by_license[name])} ({named[name]})" for name in others)
    return f"{named[common]} except {exceptions}"


def _list_table(rows: list[dict[str, Any]], columns: list[str]) -> Table:
    table = Table(box=None, pad_edge=False, header_style="bold")
    for column in columns:
        justify = "right" if column == "qubits" else "left"
        table.add_column(column, justify=justify, no_wrap=True)
    for technology in sorted({row["technology"] for row in rows}):
        table.add_row(f"[bold]{_words(technology)}[/bold]")
        for row in (r for r in rows if r["technology"] == technology):
            mark = "* " if row["location"] == "vault" else "  "
            cells = {
                "id": f"{mark}{row['id']}",
                "date": row["when"],
                "qubits": str(row["num_qubits"]),
                "processor": row["processor"] or "-",
                "source": _source_label(row["source_kind"]),
                "license": row["shown_license"],
            }
            table.add_row(*(cells[column] for column in columns))
    return table


def _source_label(kind: str | None) -> str:
    return _SOURCE_KINDS[kind].list_label if kind in _SOURCE_KINDS else kind or "-"


def _when(stamp: datetime, same_day: list[datetime]) -> str:
    day = stamp.date().isoformat()
    if len(same_day) < 2:
        return day
    clock = "%H:%M"
    if len({s.strftime(clock) for s in same_day}) < len(same_day):
        clock = "%H:%M:%S"
    return f"{day}\n{stamp.astimezone(UTC).strftime(clock)} UTC"


def _list_row(info: catalog.ProfileInfo) -> dict[str, Any]:
    return {
        "ref": info.ref,
        "id": info.id,
        "date": info.calibrated_at.date().isoformat() if info.calibrated_at else None,
        "calibrated_at": iso_z(info.calibrated_at) if info.calibrated_at else None,
        "technology": info.technology,
        "vendor": info.vendor,
        "num_qubits": info.num_qubits,
        "processor": info.processor,
        "source_kind": info.source_kind,
        "data_kind": info.data_kind,
        "license": info.license,
        "redistributable": info.redistributable,
        "location": info.location,
        "fingerprint": info.fingerprint,
    }


# show ---------------------------------------------------------------------------------------


@app.command()
def show(
    ref: Annotated[str, _argument("REF", _REF_HELP)],
    qubits: Annotated[
        str | None,
        typer.Option("--qubits", metavar="LIST", help="Also list these qubits, for example 0,1,2."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Show a profile: device, natives, coherence, readout, provenance and fingerprint."""
    with _friendly():
        profile = _load(ref)
        data = card(profile)
        indices = _parse_qubits(qubits, profile) if qubits is not None else None
        if indices is not None:
            stated = profile.uncorrected()
            data["qubits"] = [_qubit_row(stated, q) for q in indices]
        if as_json:
            _echo_json({**data, "unmodeled_note": [n.full for n in data["unmodeled_note"]]})
            return
        _print_card(data, on_hand=_on_hand(ref, profile))
        if indices is not None:
            _print_qubits(data["qubits"])


class _OnHand(NamedTuple):
    count: int
    bundled_note: str | None


def _on_hand(ref: str, profile: Profile) -> _OnHand | None:
    target = catalog.parse_ref_preferring_id(ref)
    if isinstance(target, Path) or target.date or target.timestamp:
        return None
    same = [info for info in catalog.profiles() if info.id == profile.id]
    if len(same) < 2:
        return None
    bundled = next((i for i in same if i.location == "bundled"), None)
    if bundled is None or bundled.fingerprint == profile.fingerprint:
        return _OnHand(len(same), None)
    if bundled.calibrated_at == profile.device.calibrated_at:
        replaced = f"nv:{bundled.fingerprint[:12]}"
        return _OnHand(len(same), f"your vault copy replaces the bundled one ({replaced})")
    return _OnHand(len(same), f"the bundled one is {bundled.ref}")


def card(profile: Profile) -> dict[str, Any]:
    dev, prov, table = profile.device, profile.provenance, profile.table
    stated, unmodeled = profile.uncorrected(), profile.unmodeled_error
    medians = qubit_medians(stated)
    return {
        "ref": exact_ref(profile.id, dev.calibrated_at),
        "id": profile.id,
        "calibrated_at": iso_z(dev.calibrated_at) if dev.calibrated_at else None,
        "vendor": dev.vendor,
        "processor": dev.processor,
        "technology": dev.technology,
        "num_qubits": dev.num_qubits,
        "connectivity": _connectivity(profile),
        "natives": [_native(stated, name) for name in _native_order(profile)],
        "median_t1_us": medians.t1_us,
        "median_t2_us": medians.t2_us,
        "median_readout_error": medians.readout_error,
        "median_p1_given_0": medians.p1_given_0,
        "median_p0_given_1": medians.p0_given_1,
        "disabled_qubits": [i for i in range(dev.num_qubits) if table.qubit(i).disabled],
        "provenance": prov.model_dump(mode="json", exclude_none=True, exclude={"notes", "extra"}),
        "fingerprint": profile.fingerprint,
        "short_fingerprint": profile.short_fingerprint,
        "assumptions": _assumptions(profile),
        "notes": list(prov.notes),
        "effects": _effects(profile),
        "unmodeled_error": unmodeled and unmodeled.model_dump(mode="json"),
        "unmodeled_note": list(unmodeled_note(profile)),
    }


def _native_order(profile: Profile) -> list[str]:
    table = profile.table
    return sorted(profile.gates, key=lambda name: (name == "reset", table.arity(name) or 0, name))


def _words(token: str) -> str:
    return _WORDS.get(token) or token.replace("_", " ")


def _assumptions(profile: Profile) -> list[str]:
    """Each gate's default assumption, then the records that state a different one."""
    overrides: dict[tuple[str, str], list[tuple[int, ...]]] = {}
    for record in profile.calibrations:
        if record.assumption and record.assumption != profile.gates[record.gate].assumption:
            overrides.setdefault((record.gate, record.assumption), []).append(record.qubits)
    lines = []
    for name, spec in profile.gates.items():
        if spec.assumption:
            lines.append(f"{name}: {spec.assumption}")
        for (gate, text), loci in overrides.items():
            if gate == name:
                where = (
                    ", ".join("-".join(map(str, q)) for q in loci)
                    if len(loci) <= 3
                    else f"({len(loci)} records)"
                )
                lines.append(f"{name} {where}: {text}")
    return lines


def _connectivity(profile: Profile) -> dict[str, Any]:
    table = profile.table
    if table.all_to_all:
        return {"kind": "all_to_all"}
    edges = table.edges()
    degree = [0] * table.num_qubits
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    return {
        "kind": "edges",
        "edges": len(edges),
        "directed": profile.connectivity.directed,  # type: ignore[union-attr]
        "min_degree": min(degree),
        "max_degree": max(degree),
    }


def _native(profile: Profile, name: str) -> dict[str, Any]:
    stats = gate_stats(profile, name)
    states = stats.states
    return {
        "gate": name,
        "qubits": profile.table.arity(name),
        "virtual": bool(states) and states["ideal"] == states.total(),
        "median_avg_infidelity": stats.median_error,
        "median_duration_ns": stats.median_duration_ns,
        "records": sum(r.gate == name for r in profile.calibrations),
        "disabled": states["disabled"],
        "loci": {
            state: states[state] for state in ("calibrated", "ideal", "uncalibrated", "disabled")
        },
    }


def _effects(profile: Profile) -> list[str]:
    counts: dict[str, int] = {}
    for effect in profile.effects:
        key = f"{effect.type} on {effect.gate or effect.on}"
        counts[key] = counts.get(key, 0) + 1
    return [f"{key} ({n} records)" if n > 1 else key for key, n in counts.items()]


def _print_card(
    data: dict[str, Any], *, brief: bool = False, on_hand: _OnHand | None = None
) -> None:
    out.print(f"[bold]{data['ref']}[/bold]  {data['short_fingerprint']}", soft_wrap=True)
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column(overflow="fold")
    vendor = data["vendor"]
    device = ", ".join(
        x
        for x in (
            vendor and _words(vendor),
            data["processor"],
            _words(data["technology"]),
            f"{data['num_qubits']} qubits",
        )
        if x
    )
    grid.add_row("device", device)
    if on_hand:
        grid.add_row("calibrations", f"newest of the {on_hand.count} you have")
        if on_hand.bundled_note:
            grid.add_row("", on_hand.bundled_note)
    grid.add_row("connectivity", _describe_connectivity(data["connectivity"]))
    grid.add_row("natives", _natives_table(data["natives"]))
    grid.add_row("coherence", _coherence(data))
    grid.add_row("readout", _readout(data))
    if data["disabled_qubits"]:
        grid.add_row("disabled", "qubits " + ", ".join(map(str, data["disabled_qubits"])))
    _add_lines(grid, "unmodeled", [line.short for line in data["unmodeled_note"]])
    if not brief:
        _add_lines(grid, "not modeled", [_plain_effect(line) for line in data["effects"]])
    prov = data["provenance"]
    grid.add_row("provenance", _provenance(prov))
    redistribution = _REDISTRIBUTION[prov["redistributable"]]
    grid.add_row("license", f"{prov.get('license') or 'unknown'}, {redistribution}")
    if prov.get("attribution"):
        grid.add_row("attribution", prov["attribution"])
    grid.add_row("fingerprint", data["fingerprint"])
    if not brief:
        _add_lines(grid, "assumptions", data["assumptions"])
        _add_lines(grid, "notes", data["notes"])
    _emit(grid)
    fit = data["unmodeled_error"] and data["unmodeled_error"]["fit"]
    if fit and fit["p_value"] is not None and fit["p_value"] < POOR_FIT_P_VALUE:
        err.print(
            "warning: the unmodeled-error factors are a poor fit to their counts"
            f" (p = {fit['p_value']:.2g}). No one pair of factors fits every circuit",
            markup=False,
        )


def _add_lines(grid: Table, label: str, items: list[str]) -> None:
    for i, item in enumerate(items):
        grid.add_row(label if i == 0 else "", item)


def _plain_effect(line: str) -> str:
    kind, _, rest = line.partition(" ")
    return f"{_words(kind)} {rest}"


def _provenance(prov: dict[str, Any]) -> str:
    kind = prov.get("data_kind", "unknown")
    words = ["unknown kind" if kind == "unknown" else _words(kind)]
    if prov.get("source_kind") in _SOURCE_KINDS:
        words.append(_SOURCE_KINDS[prov["source_kind"]].show_phrase)
    if prov.get("source"):
        words.append(prov["source"])
    elif len(words) == 1:
        words.append("source unknown")
    return ", ".join(words)


def _describe_connectivity(conn: dict[str, Any]) -> str:
    if conn["kind"] == "all_to_all":
        return "all-to-all"
    kind = "directed edges" if conn["directed"] else "edges"
    return (
        f"{conn['edges']} {kind}, {conn['min_degree']} to {conn['max_degree']} neighbors per qubit"
    )


def _natives_table(natives: list[dict[str, Any]]) -> Table:
    table = Table(box=None, pad_edge=False, show_edge=False, header_style="italic")
    for column in ("gate", "median avg\ninfidelity", "median\nduration", "records"):
        table.add_column(column, justify="left" if column == "gate" else "right")
    for n in natives:
        arity = f" ({n['qubits']}q)" if n["qubits"] else ""
        if n["virtual"]:
            table.add_row(f"{n['gate']}{arity}", "virtual", "-", "-")
            continue
        records = str(n["records"]) if n["records"] else "device-wide"
        has_error = n["median_avg_infidelity"] is not None
        notes = [
            f"{count} {word}"
            for count, word in (
                (n["loci"]["ideal"], "virtual"),
                (n["loci"]["uncalibrated"] if has_error else 0, "uncalibrated"),
                (n["loci"]["disabled"], "disabled"),
            )
            if count
        ]
        if notes:
            records += f" ({', '.join(notes)})"
        table.add_row(
            f"{n['gate']}{arity}",
            fmt_error(n["median_avg_infidelity"]),
            _duration(n["median_duration_ns"]),
            records,
        )
    return table


def _coherence(data: dict[str, Any]) -> str:
    t1, t2 = data["median_t1_us"], data["median_t2_us"]
    if t1 is None and t2 is None:
        return "T1 and T2 unknown (gates get no relaxation)"
    return ", ".join(
        f"{name} unknown" if value is None else f"median {name} {fmt_us(value)} us"
        for name, value in (("T1", t1), ("T2", t2))
    )


def _readout(data: dict[str, Any]) -> str:
    if data["median_readout_error"] is None:
        return "unknown (no readout error applied)"
    return (
        f"median error {fmt_error(data['median_readout_error'])}"
        f" (P(1|0) {fmt_error(data['median_p1_given_0'])},"
        f" P(0|1) {fmt_error(data['median_p0_given_1'])})"
    )


def _parse_qubits(text: str, profile: Profile) -> list[int]:
    try:
        indices = [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise ValueError(
            f"--qubits {text!r}: give qubit indices separated by commas, for example 0,1,2"
        ) from None
    n = profile.device.num_qubits
    bad = [q for q in indices if not 0 <= q < n]
    if bad or not indices:
        raise ValueError(f"--qubits {text!r}: {profile.id} has qubits 0 to {n - 1}")
    return indices


def _qubit_row(profile: Profile, index: int) -> dict[str, Any]:
    table = profile.table
    q = table.qubit(index)
    one = table.typical(1, (index,))
    record = next((r for r in profile.qubits if r.index == index), None)
    return {
        "qubit": index,
        "t1_us": None if q.t1_ns is None else q.t1_ns / 1000,
        "t2_us": None if q.t2_ns is None else q.t2_ns / 1000,
        "p1_given_0": None if q.readout is None else q.readout[0],
        "p0_given_1": None if q.readout is None else q.readout[1],
        "error_1q": one.avg_infidelity if isinstance(one, GateNoise) else None,
        "gate_1q": one.gate if isinstance(one, GateNoise) else None,
        "disabled": q.disabled,
        "label": record.label if record else None,
    }


def _print_qubits(rows: list[dict[str, Any]]) -> None:
    columns = ["qubit", "T1 (us)", "T2 (us)", "P(1|0)", "P(0|1)", "1q avg infidelity", "state"]
    cells = [
        [
            str(r["qubit"]),
            fmt_us(r["t1_us"]),
            fmt_us(r["t2_us"]),
            fmt_error(r["p1_given_0"]),
            fmt_error(r["p0_given_1"]),
            f"{fmt_error(r['error_1q'])} ({r['gate_1q']})" if r["gate_1q"] else "-",
            "disabled" if r["disabled"] else (r["label"] or ""),
        ]
        for r in rows
    ]
    has_state = any(r["disabled"] or r["label"] for r in rows)
    if not has_state:
        columns.remove("state")
    table = Table(box=None, pad_edge=False, header_style="bold")
    for column in columns:
        table.add_column(column, justify="right" if column != "state" else "left")
    for row in cells:
        table.add_row(*row[: len(columns)])
    out.print()
    _emit(table)


# pull ---------------------------------------------------------------------------------------


@app.command()
def pull(
    device: Annotated[
        str, _argument("DEVICE", "Device to pull, for example ibm_fez or ionq_forte-1.")
    ],
    at: Annotated[
        str | None,
        typer.Option(
            "--at",
            metavar="DATE",
            help="Calibration in effect at this date or time (IBM public, IBM account, IonQ).",
        ),
    ] = None,
    source: Annotated[
        str | None,
        typer.Option(
            "--source",
            metavar="NAME",
            help="ibm, ibm-account or ionq. The default comes from the name.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", metavar="FILE", help="Save here instead of the vault."),
    ] = None,
) -> None:
    """Download a live calibration and save it as a profile (network)."""
    with _friendly():
        if output is not None:
            _check_writable(output)
        with err.status(f"Pulling {device}{f' at {at}' if at else ''}..."):
            pulled = catalog.pull_and_save(device, at=at, source=source, output=output)
        _print_card(card(pulled.profile), brief=True)
        saved = "saved" if pulled.written else "already saved"
        out.print(f"{saved}: {pulled.path}", markup=False, soft_wrap=True)


def _check_writable(output: Path) -> None:
    folder = output.parent
    if output.is_dir():
        raise NoiseVaultError(
            f"-o {output} is a folder", hint=f"give a file name such as {output / 'x.json'}"
        )
    if not folder.is_dir():
        raise NoiseVaultError(
            f"-o {output}: the folder {folder} does not exist", hint="create it first"
        )
    if not os.access(folder, os.W_OK):
        raise NoiseVaultError(
            f"-o {output}: you cannot write to {folder}", hint="choose another folder"
        )


# diff ---------------------------------------------------------------------------------------


@app.command()
def diff(
    before: Annotated[
        str, _argument("BEFORE", "First profile id (ibm_fez), id@date, or a file path.")
    ],
    after: Annotated[
        str, _argument("AFTER", "Second profile id (ibm_kyiv), id@date, or a file path.")
    ],
    top: Annotated[
        int, typer.Option("--top", min=0, metavar="N", help="Qubits and pairs to list.")
    ] = 5,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Show how calibration changed between two profiles."""
    with _friendly():
        first, second = _load(before), _load(after)
        result = first.diff(second, top=top)
        if as_json:
            _echo_json(result.to_dict())
            return
        title = Text.from_markup(_diff_title(first, second))
        delta = f"({describe_delta(result.time_delta)})"
        if title.cell_len + 2 + len(delta) <= out.width:
            _emit(Text.assemble(title, f"  {delta}"))
        else:
            _emit(title)
            _emit(f"  {delta}")
        for warning in result.warnings:
            err.print(f"warning: {warning}", markup=False)
        if result.identical:
            _emit("No change: the two profiles have the same fingerprint.")
            return
        medians = Table(box=None, pad_edge=False, header_style="bold")
        for column in ("device median", "before", "after", "change"):
            medians.add_column(column, justify="left" if column == "device median" else "right")
        for c in result.medians:
            medians.add_row(
                METRICS[c.metric][0],
                fmt_metric(c.metric, c.before),
                fmt_metric(c.metric, c.after),
                _change(c),
            )
        _emit(medians)
        for title, changes in (("qubit", result.qubits), ("pair", result.pairs)):
            if not changes:
                continue
            table = Table(box=None, pad_edge=False, header_style="bold", title_justify="left")
            for column in (title, "metric", "before", "after", "change"):
                table.add_column(
                    column, justify="right" if column in ("before", "after", "change") else "left"
                )
            for c in changes:
                table.add_row(
                    c.where,
                    METRICS[c.metric][0],
                    fmt_metric(c.metric, c.before),
                    fmt_metric(c.metric, c.after),
                    _change(c),
                )
            out.print()
            _emit(f"largest changes by {title}")
            _emit(table)
        for label, items in (
            ("newly disabled", result.newly_disabled),
            ("re-enabled", result.reenabled),
        ):
            if items:
                out.print()
                _emit(label)
                _emit(_loci_table(items))
        for label, qubits in (
            ("qubits added", result.qubits_added),
            ("qubits removed", result.qubits_removed),
        ):
            if qubits:
                _emit(f"{label}: {_runs(qubits)}")


def _runs(indices: Sequence[int]) -> str:
    """``0, 3 to 9, 12``: three or more consecutive indices as one run."""
    runs: list[list[int]] = []
    for i in sorted(indices):
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return ", ".join(
        f"{run[0]} to {run[-1]}" if len(run) > 2 else ", ".join(map(str, run)) for run in runs
    )


def _diff_title(first: Profile, second: Profile) -> str:
    when = distinguishing_stamps(first.device.calibrated_at, second.device.calibrated_at)
    if first.id != second.id:
        refs = (f"{p.id}@{w}" if w else p.id for p, w in zip((first, second), when, strict=True))
        return " -> ".join(f"[bold]{ref}[/bold]" for ref in refs)
    return f"[bold]{first.id}[/bold] {when[0] or 'undated'} -> {when[1] or 'undated'}"


def _loci_table(labels: tuple[str, ...]) -> Table:
    """One row per gate (or "qubit"), its loci folded at item boundaries."""
    by_gate: dict[str, list[str]] = {}
    for label in labels:
        gate, locus = label.split(" ", 1)
        by_gate.setdefault(gate, []).append(locus)
    table = Table(box=None, pad_edge=False, show_header=False)
    table.add_column(no_wrap=True)
    table.add_column()
    for gate, loci in by_gate.items():
        table.add_row(f"  {gate}", ", ".join(loci))
    return table


def _change(change: Any) -> str:
    if change.before is None or change.after is None:
        # "worse" has no meaning when one side has no value, so these changes get no color
        return "-" if change.before == change.after else "new" if change.before is None else "gone"
    text = fmt_relative(change.relative)
    if change.relative == 0:
        return text
    return f"[red]{text}[/red]" if change.worse else f"[green]{text}[/green]"


# check --------------------------------------------------------------------------------------


@app.command()
def check(
    ref: Annotated[str, _argument("REF", _REF_HELP)],
    framework: Annotated[
        str | None,
        typer.Option(
            "--framework",
            metavar="NAMES",
            help="Comma-separated: qiskit,cirq,pennylane,stim (default all).",
        ),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Check that each framework export reproduces the reference model on small circuits."""
    from .check import FRAMEWORKS, NOTE

    with _friendly():
        names = list(FRAMEWORKS)
        if framework is not None:
            names = list(
                dict.fromkeys(n.strip().lower() for n in framework.split(",") if n.strip())
            )
            unknown = [n for n in names if n not in FRAMEWORKS]
            if not names or unknown:
                given = unknown[0] if unknown else framework
                raise ValueError(
                    f"--framework {given!r}: {did_you_mean(given, FRAMEWORKS)}give one or more"
                    f" of {','.join(FRAMEWORKS)}"
                )
        profile = _load(ref)
        parts = []
        for name in names:  # one at a time, so the spinner says which one is running
            with err.status(f"Checking the {name} export..."):
                parts.append(profile.check(frameworks=[name]))
        result = replace(
            parts[0],
            frameworks=tuple(f for part in parts for f in part.frameworks),
            skipped=tuple(s for part in parts for s in part.skipped),
        )
        missing = [n for n, why in result.skipped if why.startswith("not installed")]
        named = ",".join(names) if framework is not None else None
        if as_json:
            _echo_json(result.to_dict())
        elif len(missing) < len(names):
            qubits = qubit_loci([result.layout[i] for i in range(len(result.layout))])
            on_hand = _on_hand(ref, profile)
            newest = f" (newest of {on_hand.count})" if on_hand else ""
            title = Text(exact_ref(profile.id, profile.device.calibrated_at), style="bold")
            _emit(Text.assemble(title, f"{newest} {profile.short_fingerprint} on {qubits}"))
            _emit(f"{len(result.circuits)} circuits: {', '.join(c.name for c in result.circuits)}")
            rows = [_check_row(f, result) for f in result.frameworks]
            rows += [
                {"framework": name, "result": "not installed" if name in missing else "skipped"}
                for name, _ in result.skipped
            ]
            columns = ["framework", "result", "deviation", "tolerance", "circuits", "method"]
            table = _fit(lambda shown: _check_table(rows, shown), columns, _CHECK_DROPS)
            _emit(table, natural_width=_natural_width(table))
            for f in result.frameworks:
                for part in f.not_run:
                    _emit(f"{f.framework}: {part.describe()}")
            for name, reason in result.skipped:
                if name not in missing:
                    _emit(f"{name}: {reason}")
            if missing:
                command = install_hint(",".join(missing))
                out.print(f"To add the missing frameworks: {command}", markup=False, soft_wrap=True)
                uvx = _uvx_hint(",".join(names), _check_command(ref, named))
                out.print(f"Or, with uv and no install: {uvx}", markup=False, soft_wrap=True)
            if any(f.passed for f in result.frameworks):
                _emit(NOTE)
        if len(missing) == len(names):
            _none_installed(missing, ref=ref, framework=named)
        if not result.frameworks:
            raise NoiseVaultError("no framework could run the check")
        if not result.passed:
            raise typer.Exit(1)


_CHECK_DROPS = ("tolerance", "circuits")


def _check_row(f: FrameworkCheck, result: CheckResult) -> dict[str, str]:
    reduced = {n.circuit for n in f.not_run if n.ran_without}
    ran = len({c.circuit for c in f.circuits} - reduced)
    counted = f"{ran} of {len(result.circuits)}"
    kinds = frozenset(c.sampled for c in f.circuits)
    method = {
        frozenset({False}): "exact",
        frozenset({True}): f"{result.shots} shots, 5 sigma",
    }.get(kinds, f"exact + {result.shots} shots")
    return {
        "framework": f.framework,
        "result": "[green]pass[/green]" if f.passed else "[red]FAIL[/red]",
        "deviation": f"{f.worst.deviation:.1e}",
        "tolerance": f"{f.worst.tolerance:.1e}",
        "circuits": f"{counted}, {len(reduced)} reduced" if reduced else counted,
        "method": method,
    }


def _check_table(rows: list[dict[str, str]], columns: list[str]) -> Table:
    table = Table(box=None, pad_edge=False, header_style="bold")
    for column in columns:
        justify = "right" if column in ("deviation", "tolerance") else "left"
        table.add_column(column, justify=justify, no_wrap=True)
    for row in rows:
        table.add_row(*(row.get(column, "") for column in columns))
    return table


def _none_installed(missing: list[str], *, ref: str, framework: str | None) -> NoReturn:
    first, *others = missing
    if framework is None:
        error = "none is installed"
        extra = first
        lines = [
            install_hint(first),
            f"(or {joined(others, 'or')}, or several, as in noisevault[{first},{others[-1]}])",
        ]
    else:
        error = f"{joined(missing)} {'is' if len(missing) == 1 else 'are'} not installed"
        extra = ",".join(missing)
        lines = [install_hint(extra)]
    lines.append(f"or, with uv and no install: {_uvx_hint(extra, _check_command(ref, framework))}")
    _fail(f"nv check needs a framework to check, and {error}", "\n      ".join(lines))


def _check_command(ref: str, framework: str | None) -> str:
    options = [] if framework is None else ["--framework", framework]
    return catalog.shell_command(["nv", "check", *options], [ref])


def _uvx_hint(extra: str, command: str) -> str:
    return f'uvx --from "noisevault[{extra}] @ git+{REPOSITORY}" {command}'


@app.command()
def compare(
    ref: Annotated[str, _argument("REF", _REF_HELP)],
    counts: Annotated[str, _argument("COUNTS", "Counts file: .json or .json.gz, format 1.0.")],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            metavar="FILE",
            help="Save the profile with the fitted factors here.",
        ),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Score a profile on counts from a device and fit its gate and readout factors."""
    with _friendly():
        profile = _load(ref, counts_hint=_PROFILE_FIRST)
        counts_file = Path(counts)
        if output is not None:
            _check_fitted_output(
                output, profile, counts=counts_file, profile_file=_profile_file(ref)
            )
        measured = _read_counts(counts_file)
        with err.status("Fitting the gate and readout factors..."):
            result = profile.compare(measured)
        if not as_json:
            _print_comparison(result.summary_lines(counts_file=counts, width=out.width))
        written: dict[str, str] | None = None
        failure: NoiseVaultError | None = None
        if output is not None:
            try:
                fitted = result.fitted_profile()
                fitted.save(output)
            except NoiseVaultError as exc:
                failure = exc
            except OSError as exc:
                failure = NoiseVaultError(
                    f"cannot write {output}: {exc.strerror or exc}", hint="choose another folder"
                )
            else:
                written = {"path": str(output), "fingerprint": fitted.fingerprint}
        if as_json:
            _echo_json(result.to_dict() | ({"written": written} if written else {}))
        elif written:
            saved = f"saved: {output}, {profile.id} nv:{written['fingerprint'][:12]}"
            out.print()
            out.print(Text(f"{saved} with the fitted factors"), soft_wrap=True)
        if failure is not None:
            raise failure


_COUNTS_FILE = "give a counts file (.json or .json.gz)"


def _profile_file(ref: str) -> Path:
    target = catalog.parse_ref_preferring_id(ref)
    return target if isinstance(target, Path) else Path(str(catalog.resolve(target).path))


def _check_fitted_output(
    output: Path, profile: Profile, *, counts: Path, profile_file: Path
) -> None:
    elsewhere = f"save the fitted profile elsewhere, such as {profile.id}-fitted.json"
    for read, what in ((counts, "counts file"), (profile_file, "profile file")):
        if output.exists() and read.exists() and os.path.samefile(output, read):
            raise NoiseVaultError(f"-o {output} is the {what} nv compare reads", hint=elsewhere)
    vault = f"the vault holds the calibrations that refs load, so {elsewhere}"
    stores = (
        (catalog.vault_dir(), "your vault", vault),
        (Path(str(catalog.bundled_dir())), "NoiseVault's bundled profiles", elsewhere),
    )
    for folder, where, hint in stores:
        if folder.is_dir() and any(
            p.exists() and os.path.samefile(p, folder) for p in output.resolve().parents
        ):
            raise NoiseVaultError(f"-o {output} is in {where}", hint=hint)
    _check_writable(output)


def _read_counts(path: Path) -> MeasuredCounts:
    from .counts import load_counts

    if path.is_dir():
        raise NoiseVaultError(f"{path} is a folder", hint=_COUNTS_FILE)
    try:
        return load_counts(path)
    except FileNotFoundError:
        raise NoiseVaultError(f"no file {path}", hint="check the path") from None
    except OSError as exc:
        raise NoiseVaultError(
            f"cannot read {path}: {exc.strerror or exc}", hint="check the file and its permissions"
        ) from None


def _print_comparison(lines: Sequence[SummaryLine]) -> None:
    for line in lines:
        text = Text(line.text)
        if line.bold_end:
            text.stylize("bold", 0, line.bold_end)
        out.print(text, soft_wrap=True)


# cite, validate, doctor, schema ---------------------------------------------------------------


@app.command()
def cite(
    ref: Annotated[str, _argument("REF", _REF_HELP)],
    bibtex: Annotated[bool, typer.Option("--bibtex", help="Print a BibTeX entry.")] = False,
) -> None:
    """Print a citation for a profile, with its fingerprint."""
    with _friendly():
        typer.echo(_load(ref).citation("bibtex" if bibtex else "text"))


@app.command()
def validate(
    file: Annotated[str, _argument("FILE", "Profile: .json or .json.gz, format 1.0 or 0.1.")],
    strict: Annotated[bool, typer.Option("--strict", help="Treat warnings as errors.")] = False,
) -> None:
    """Check a profile file against the format rules."""
    path = Path(file)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            profile = Profile.from_dict(read_json_file(path, "profile"))
        except FileNotFoundError:
            _fail(f"no file {path}", "check the path")
        except ValidationError as exc:
            if _holds_counts(path):
                _fail(f"{path} is a counts file, not a profile", _PROFILE_FILE)
            for line in _validation_lines(exc):
                err.print(f"error: {line}", markup=False)
            raise typer.Exit(1) from None
        except _UNREADABLE as exc:
            _fail(*_unreadable(path, exc))
        except NoiseVaultError as exc:
            _fail(exc.message, exc.hint)
    notes = [str(w.message) for w in caught] + _soft_issues(profile)
    _emit(
        f"ok: {ref_on_day(profile.id, profile.device.calibrated_at)} {profile.short_fingerprint},"
        f" {plural(profile.device.num_qubits, 'qubit')}, {plural(len(profile.gates), 'gate')},"
        f" {plural(len(profile.calibrations), 'record')}"
    )
    for note in notes:
        err.print(f"warning: {note}", markup=False)
    if strict and notes:
        raise typer.Exit(1)


@app.command()
def doctor() -> None:
    """Show NoiseVault, Python and package versions, and where profiles live."""
    from .check import FRAMEWORKS

    table = Table("component", "version", box=None, pad_edge=False, header_style="bold")
    table.add_row("noisevault", __version__)
    table.add_row("python", platform.python_version())
    missing = []
    for package in _PACKAGES:
        try:
            table.add_row(package, version(package))
        except PackageNotFoundError:
            table.add_row(package, "not installed")
            missing.append(package)
    _emit(table)
    vault = catalog.vault_dir()
    with _friendly():
        count = len(catalog.vault_profiles()) if vault.is_dir() else 0
    out.print(f"vault: {vault} ({plural(count, 'profile')})", markup=False, soft_wrap=True)
    _emit(f"bundled profiles: {len(catalog.bundled_profiles())}")
    extras = sorted({extra for p in missing if (extra := _PACKAGES[p])})
    if extras:
        extra = "all" if extras == _EXTRAS else ",".join(extras)
        out.print(
            f"To add the missing packages: {install_hint(extra)}", markup=False, soft_wrap=True
        )
    if any(name in FRAMEWORKS for name in extras):
        uvx = _uvx_hint(",".join(FRAMEWORKS), "nv check ibm_fez")
        out.print(f"Or, with uv and no install: {uvx}", markup=False, soft_wrap=True)
    loose = [package for package in missing if _PACKAGES[package] is None]
    if loose:
        _emit(f"To add {', '.join(loose)}: pip install {' '.join(loose)}")


@app.command()
def schema() -> None:
    """Print the JSON Schema of profile format 1.0."""
    _echo_json(json_schema())


# shared helpers -----------------------------------------------------------------------------


def _load(ref: str, *, counts_hint: str | None = None) -> Profile:
    target = catalog.parse_ref_preferring_id(ref)
    if not isinstance(target, Path):
        return catalog.load(ref)
    if not target.exists():
        raise FileNotFoundError(errno.ENOENT, "no such file", str(target))
    if target.is_dir():
        raise NoiseVaultError(f"{ref} is a folder", hint=_PROFILE_OR_ID)
    try:
        return Profile.from_dict(read_json_file(target, "profile"))
    except _UNREADABLE as exc:
        if isinstance(exc, ValidationError) and _holds_counts(target):
            raise NoiseVaultError(
                f"{target} is a counts file, not a profile", hint=counts_hint or _PROFILE_OR_ID
            ) from None
        message, hint = _unreadable(target, exc)
        raise NoiseVaultError(message, hint=hint) from None


def _holds_counts(path: Path) -> bool:
    data = json_bytes(path.read_bytes())
    return isinstance(data, dict) and "nv_counts" in data


# What reading a profile file can raise besides FileNotFoundError. ValidationError is a ValueError.
_UNREADABLE = (ValueError, OSError)


_PROFILE_FILE = "give a profile file (.json or .json.gz)"
_PROFILE_OR_ID = f"{_PROFILE_FILE} or a profile id such as ibm_fez"
_PROFILE_FIRST = "nv compare takes the profile first and the counts file second"


class _FileProblem(NamedTuple):
    message: str
    hint: str | None


def _unreadable(path: Path, exc: BaseException) -> _FileProblem:
    if isinstance(exc, ValidationError):
        problems = plural(exc.error_count(), "problem")
        return _FileProblem(
            f"{path} is not a valid profile ({problems})",
            f"run {catalog.shell_command(['nv', 'validate'], [str(path)])} to list them",
        )
    if isinstance(exc, IsADirectoryError):
        return _FileProblem(f"{path} is a folder", _PROFILE_FILE)
    if isinstance(exc, OSError):
        return _FileProblem(
            f"cannot read {path}: {exc.strerror or exc}", "check the file and its permissions"
        )
    if isinstance(exc, NoiseVaultError):
        return _FileProblem(f"{path}: {exc.message}", exc.hint)
    return _FileProblem(f"{path}: {exc}", None)


@contextmanager
def _friendly() -> Iterator[None]:
    """Turn expected failures into ``error:`` and ``hint:`` lines, and warnings into lines."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            yield
        except typer.Exit:
            raise
        except _EXPECTED as exc:
            _report_warnings(caught)
            _error(exc)
        _report_warnings(caught)


def _report_warnings(caught: list[warnings.WarningMessage]) -> None:
    for message in dict.fromkeys(str(w.message) for w in caught):
        err.print(f"warning: {message}", markup=False)
    caught.clear()


def _error(exc: BaseException) -> NoReturn:
    if isinstance(exc, NoiseVaultError):
        _fail(_cli_terms(exc.message), exc.hint and _cli_terms(exc.hint))
    hint = next((h for kind, h in _FOREIGN_ERROR_HINTS if isinstance(exc, kind)), None)
    if isinstance(exc, ValidationError):
        _fail(f"not a valid profile ({plural(exc.error_count(), 'problem')})", hint)
    if isinstance(exc, FileNotFoundError) and exc.filename:
        _fail(f"no file {exc.filename}", hint)
    _fail(_cli_terms(str(exc) or type(exc).__name__), hint)


# Library messages name Python arguments. At the command line, the same choice is a flag.
_CLI_TERMS = (
    (re.compile(r"""source=(['"])([\w-]+)\1 or (['"])([\w-]+)\3"""), r"--source \2 or \4"),
    (re.compile(r"""source=(['"])([\w-]+)\1"""), r"--source \2"),
    (re.compile(r"""nv\.load\((['"])([\w.@:-]+)\1\)"""), r"nv show \2"),
    (re.compile(r"\bat=(?= )"), "--at"),
    (re.compile(r"\bat=(?=['\"])"), "--at "),
    (
        re.compile(r"pass expect='nv:\.\.\.' or load one of their files:"),
        "give one of their files:",
    ),
    (re.compile(r"`(nv [^`]+)`"), r"\1"),
)


def _cli_terms(message: str) -> str:
    for pattern, replacement in _CLI_TERMS:
        message = pattern.sub(replacement, message)
    return message


def _fail(message: str, hint: str | None = None, *, code: int = 1) -> NoReturn:
    err.print(f"error: {message}", markup=False)
    if hint:
        err.print(f"hint: {hint}", markup=False)
    raise typer.Exit(code) from None


def _emit(renderable: RenderableType, *, natural_width: int | None = None) -> None:
    """Print without the trailing spaces rich adds when it pads or wraps a line, so pasted output
    is clean."""
    if isinstance(renderable, str):
        renderable = Text(renderable)
    options = out.options if natural_width is None else out.options.update_width(natural_width)
    for line in out.render_lines(renderable, options, pad=False):
        text = Text.assemble(*((segment.text, segment.style) for segment in line))
        text.rstrip()
        out.print(text, soft_wrap=True)


def _echo_json(data: Any) -> None:
    typer.echo(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))


def _epoch(when: datetime | None) -> float:
    return when.timestamp() if when else float("-inf")


def _duration(ns: float | None) -> str:
    if ns is None:
        return "-"
    if ns >= 1e6:
        return f"{ns / 1e6:.3g} ms"
    return f"{ns / 1e3:.3g} us" if ns >= 1e3 else f"{ns:.3g} ns"


_PLAIN_ERRORS = {
    "extra_forbidden": "not a format 1.0 key. Put your own data under the top-level extensions key",
    "missing": "missing. Format 1.0 requires this key",
}


def _validation_lines(exc: ValidationError) -> list[str]:
    lines = []
    for error in exc.errors():
        where = ".".join(str(part) for part in error["loc"])
        message = _PLAIN_ERRORS.get(error["type"]) or error["msg"].removeprefix("Value error, ")
        for part in message.split("\n"):
            lines.append(f"{where}: {part}" if where else part)
    return lines


def _soft_issues(profile: Profile) -> list[str]:
    """Legal values to note, because they change what an export produces."""
    notes = []
    for i in range(profile.device.num_qubits):
        q = profile.table.qubit(i)
        if q.t2_clamped:
            notes.append(f"qubit {i}: T2 exceeds 2*T1, so exports clamp T2 to 2*T1")
    for record in profile.calibrations:
        if record.scope == "cycle":
            notes.append(
                f"{record.gate} on {qubit_loci(record.qubits)}: error is per cycle, not per gate"
            )
    return notes
