from __future__ import annotations

from collections.abc import Container, Mapping, Sequence
from types import MappingProxyType
from typing import Literal

from . import gates
from .channels import ChannelSpec, GateChannels, gate_channels, thermal_relaxation_kraus
from .errors import LayoutError, MissingCalibrationError, qubit_loci
from .report import Report
from .table import GateNoise, NoiseTable, Unavailable, refuse_disabled

UnknownGates = Literal["typical", "error"]
TYPICAL_FIX = "compile to native gates for realistic gate counts, or pass unknown_gates='error'"


# The gates that each registry gate equals, in the order that a profile without a calibration
# for the registry gate uses their calibrations. A fixed gate equals its rotation at one angle,
# up to global phase. A z-family fixed gate, and u1, equal p exactly, so p comes before rz.
_FALLBACKS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        **dict.fromkeys(("x", "sx", "sxdg"), ("rx",)),
        "y": ("ry",),
        **dict.fromkeys(("z", "s", "sdg", "t", "tdg", "u1"), ("p", "rz")),
        "p": ("rz",),
        "zz": ("rzz",),
    }
)


def native_name(name: str, defined: Container[str], rotation: str | None = None) -> str:
    """The gate whose calibration an operation equal to registry gate ``name`` uses.

    ``defined`` holds a profile's gate names. The profile's own ``name`` comes first. Next comes the
    first defined gate that ``name`` equals. For ``sx``, that gate is ``rx``. For ``s`` and the
    other z-family gates, ``p`` comes first and then ``rz``. ``rotation`` names that gate for a
    native outside this table (``rxx`` or ``ryy`` for an ``ms``). If the profile defines no such
    gate, the operation keeps its own name. A native with parameters gives way to ``rotation``, so
    errors and reports name the gate the circuit wrote.
    """
    if name in defined:
        return name
    equal = _FALLBACKS.get(name) or ((rotation,) if rotation else ())
    found = next((gate for gate in equal if gate in defined), None)
    if found is not None:
        return found
    info = gates.lookup(name)
    return rotation if rotation and info and info.params else name


def resolve_op(
    table: NoiseTable,
    name: str,
    qubits: Sequence[int],
    *,
    unknown_gates: UnknownGates,
    report: Report,
) -> GateChannels:
    """Channels for gate ``name`` on physical ``qubits``, with each decision recorded in ``report``.

    An ideal gate gets no channels and a calibrated gate gets its own channels. A disabled gate
    raises DisabledGateError. A gate has no calibration on ``qubits`` in four cases. The profile
    does not define the gate, or the gate is not a native. The gate has no error metric, or the
    profile does not allow the pair. With ``unknown_gates="error"``, such a gate raises
    MissingCalibrationError. With ``"typical"``, the gate gets the noise of the typical native gate
    of its arity. The report and one warning for each gate name state the substitution. Under either
    setting, a gate with no calibration on ``qubits`` raises MissingCalibrationError if it needs
    several native entanglers or acts on more than two qubits.
    """
    qubits = tuple(qubits)
    noise, why = resolve_gate(table, name, qubits, unknown_gates=unknown_gates)
    if why is not None:
        report.count("typical_noise_used", name)
        report.approximate(
            f"gate {name}", f"noise of the typical {len(qubits)}-qubit native gate", TYPICAL_FIX
        )
        report.warn_once(
            f"typical_noise_used:{name}",
            f"{name} on {qubit_loci(qubits)}: {why}, so the export uses the noise of"
            f" {noise.gate} instead. To fix: {TYPICAL_FIX}",
        )
    return _channels(table, noise, report)


def resolve_gate(
    table: NoiseTable, name: str, qubits: Sequence[int], *, unknown_gates: UnknownGates
) -> tuple[GateNoise, str | None]:
    """The gate noise that resolve_op gives ``name`` on ``qubits``, with no side effects.

    The second item is None when ``name`` has its own noise there. Otherwise it is the reason that
    ``name`` has none, and the first item is the typical native gate. Raises as resolve_op does.
    """
    if unknown_gates not in ("typical", "error"):
        raise ValueError(f"unknown_gates={unknown_gates!r}: choose 'typical' or 'error'")
    qubits = tuple(qubits)
    _check_qubits(table, name, qubits)
    info = gates.lookup(name)
    if info is not None and info.unitary is None:
        raise ValueError(f"{name} is not a unitary gate, and resolve_op handles only gates")

    found = table.gate(name, qubits)
    if isinstance(found, GateNoise) and found.state != "uncalibrated":
        refuse_disabled(found)
        return found, None
    if isinstance(found, Unavailable) and found.kind == "bad_target":
        raise ValueError(found.reason)
    why = found.reason if isinstance(found, Unavailable) else f"{name} has no error metric"
    where = f"{name} on {qubit_loci(qubits)}"
    if (info is not None and info.multi_entangler) or len(qubits) > 2:
        raise MissingCalibrationError(
            f"{where}: {name} needs more than one native entangling gate, so no single"
            " calibration describes it",
            hint="decompose it into the profile's native gates first",
        )
    typical = table.typical(len(qubits), qubits)
    if unknown_gates == "error":
        raise MissingCalibrationError(
            f"{where}: {why}", hint=_error_hint(table, name, qubits, typical)
        )
    if isinstance(typical, Unavailable):
        raise MissingCalibrationError(
            f"{where}: {why}. No calibrated {len(qubits)}-qubit native gate is usable there either"
        )
    return typical, why


def _error_hint(
    table: NoiseTable, name: str, qubits: tuple[int, ...], typical: GateNoise | Unavailable
) -> str | None:
    """A step that gives ``name`` noise on ``qubits``, or None when the profile has none."""
    if isinstance(typical, GateNoise):
        return (
            "compile to the profile's native gates, or pass unknown_gates='typical' to use"
            " the typical native gate's noise"
        )
    usable = [n for n in table.natives(len(qubits)) if table.allowed(n, qubits)]
    where = qubit_loci(qubits)
    if name in usable:
        return f"give {name} an error metric on {where}"
    if usable:
        return (
            f"give {usable[0]} an error metric on {where}, then compile to the profile's native"
            " gates"
        )
    return None


def _check_qubits(table: NoiseTable, name: str, qubits: tuple[int, ...]) -> None:
    if len(set(qubits)) != len(qubits):
        raise LayoutError(f"{name} acts on {qubit_loci(qubits)}, but its qubits must be distinct")
    for q in qubits:
        if not 0 <= q < table.num_qubits:
            raise LayoutError(
                f"{name} acts on physical qubit {q}, but the device has qubits"
                f" 0..{table.num_qubits - 1}"
            )
        if table.qubit(q).disabled:
            raise LayoutError(
                f"{name} acts on physical qubit {q}, which the profile marks disabled"
            )


def _channels(table: NoiseTable, gate: GateNoise, report: Report) -> GateChannels:
    built = gate_channels(gate, [table.qubit(q) for q in gate.qubits])
    report.record_channels(built)
    return built


def idle_channel(
    table: NoiseTable, qubit: int, duration_ns: float, report: Report, *, label: str = "delay"
) -> ChannelSpec | None:
    noise = table.qubit(qubit)
    if noise.relaxation_unknown:
        report.mark_unknown(f"T1 and T2 of qubit {qubit} (no {label} relaxation)")
        return None
    if noise.t2_clamped:
        report.record_t2_clamp(qubit)
    if duration_ns <= 0:
        return None
    kraus = thermal_relaxation_kraus(
        noise.t1_ns, noise.t2_ns, duration_ns, noise.dephasing_rate_per_s
    )
    return ChannelSpec("thermal_relaxation", (qubit,), tuple(kraus))
