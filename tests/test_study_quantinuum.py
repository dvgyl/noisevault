from __future__ import annotations

import json
import math
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pytest

from noisevault import gates
from noisevault.counts import load_counts
from noisevault.profile import Profile
from noisevault.reference import Op, probabilities
from study import quantinuum as qt

# The vendor definitions, from the OpenQASM 2.0 U and CX and from hqslib1_dev.inc and
# qelib1.inc. These matrices do not use the gate registry.


def U(theta: float, phi: float, lam: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array(
        [[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]]
    )


def U1q(theta: float, phi: float) -> np.ndarray:
    return U(theta, phi - math.pi / 2, math.pi / 2 - phi)


def Rz(lam: float) -> np.ndarray:
    return U(0, 0, lam)


CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
I2 = np.eye(2)
ZZ_PAULI = np.diag([1.0, -1.0, -1.0, 1.0])


def first(u: np.ndarray) -> np.ndarray:
    return np.kron(u, I2)


def second(u: np.ndarray) -> np.ndarray:
    return np.kron(I2, u)


def then(*steps: np.ndarray) -> np.ndarray:
    total = np.eye(len(steps[0]), dtype=complex)
    for step in steps:
        total = step @ total
    return total


def ZZ_comment() -> np.ndarray:
    """The decomposition in a comment of hqslib1_dev.inc, where ZZ itself is opaque."""
    return then(
        second(U1q(math.pi / 2, math.pi / 2)),
        CX,
        first(Rz(-math.pi / 2)),
        second(U1q(math.pi / 2, 0)),
        second(U1q(math.pi / 2, -math.pi / 2)),
    )


def RZZ_definition(theta: float) -> np.ndarray:
    return np.diag(np.exp(-0.5j * theta * np.diag(ZZ_PAULI)))


THETA, PHI = 0.7314, -2.0517
VENDOR = {
    "U1q": (U1q(THETA, PHI), (THETA, PHI)),
    "Rz": (Rz(THETA), (THETA,)),
    "rz": (Rz(THETA), (THETA,)),
    "u1": (Rz(THETA), (THETA,)),
    "rx": (U1q(THETA, 0), (THETA,)),
    "ry": (U1q(THETA, math.pi / 2), (THETA,)),
    "x": (U1q(math.pi, 0), ()),
    "y": (U1q(math.pi, math.pi / 2), ()),
    "z": (Rz(math.pi), ()),
    "h": (then(U1q(math.pi / 2, -math.pi / 2), Rz(math.pi)), ()),
    "id": (then(Rz(0), U1q(0, math.pi / 2), Rz(0)), ()),
    "ZZ": (RZZ_definition(math.pi / 2), ()),
    "RZZ": (RZZ_definition(THETA), (THETA,)),
    "rzz": (RZZ_definition(THETA), (THETA,)),
}
QELIB1 = {
    "rx": U(THETA, -math.pi / 2, math.pi / 2),
    "ry": U(THETA, 0, 0),
    "rz": U(0, 0, THETA),
    "x": U(math.pi, 0, math.pi),
    "h": U(math.pi / 2, 0, math.pi),
    "id": U(0, 0, 0),
    "rzz": then(CX, second(U(0, 0, THETA)), CX),
}


def registry_unitary(ops: Sequence[Op], n: int) -> np.ndarray:
    total = np.eye(2**n, dtype=complex)
    for op in ops:
        u = gates.unitary(op.name, op.params)
        if n == 2 and len(op.qubits) == 1:
            u = first(u) if op.qubits == (0,) else second(u)
        elif n == 2 and op.qubits == (1, 0):
            swap = np.eye(4)[[0, 2, 1, 3]]
            u = swap @ u @ swap
        total = u @ total
    return total


def equal_up_to_phase(a: np.ndarray, b: np.ndarray) -> bool:
    overlap = np.trace(a.conj().T @ b) / len(a)
    return bool(np.isclose(abs(overlap), 1.0, atol=1e-12))


def test_every_gate_row_has_a_vendor_definition() -> None:
    assert set(VENDOR) == set(qt.GATES)


@pytest.mark.parametrize("name", sorted(VENDOR))
def test_each_gate_mapping_equals_the_vendor_definition_up_to_a_global_phase(name: str) -> None:
    matrix, params = VENDOR[name]
    gate = qt.GATES[name]
    qubits = tuple(range(gate.arity))
    ops = gate.ops(params, qubits)
    assert all(gates.lookup(op.name) is not None for op in ops)
    assert equal_up_to_phase(registry_unitary(ops, gate.arity), matrix)
    if name in QELIB1:
        assert equal_up_to_phase(matrix, QELIB1[name])


def test_a_quarter_turn_rzz_is_the_native_zz_and_another_angle_is_rzz() -> None:
    assert qt.GATES["RZZ"].ops((math.pi / 2,), (0, 1)) == [Op("zz", (0, 1), ())]
    assert qt.GATES["rzz"].ops((-1.5 * math.pi,), (0, 1)) == [Op("zz", (0, 1), ())]
    assert qt.GATES["RZZ"].ops((0.3,), (0, 1)) == [Op("rzz", (0, 1), (0.3,))]


def test_the_commented_zz_decomposition_differs_from_zz_by_a_z_rotation() -> None:
    assert not equal_up_to_phase(ZZ_comment(), gates.unitary("zz"))
    assert equal_up_to_phase(ZZ_comment(), first(Rz(math.pi)) @ gates.unitary("zz"))


def test_the_cx_of_hqslib1_from_the_registry_zz_is_cx() -> None:
    cx = then(
        second(U1q(-math.pi / 2, math.pi / 2)),
        gates.unitary("zz"),
        first(Rz(-math.pi / 2)),
        second(U1q(math.pi / 2, math.pi)),
        second(Rz(-math.pi / 2)),
    )
    assert equal_up_to_phase(cx, CX)


# the parser -----------------------------------------------------------------------------------

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\ninclude "hqslib1_dev.inc";\n'


def test_the_parser_reads_the_statements_of_the_vendor_files() -> None:
    program = qt.parse_qasm(
        HEADER
        + "qreg q[2];\nqreg a[2];\ncreg c[2];\ncreg l[2];\nbarrier q, a;\n"
        + "order4 a[0], q[0], a[1], q[1];\nif (c==1) ZZ q[0], q[1];\n"
        + "rx(0.5*pi) q[0];\nry(-pi/2) q[1];\nrz(3.5*pi) q[1];\nZZ q[0],q[1];\n"
        + "x a[0];\nh a[0];\nZZ q[0], a[0];\nmeasure a[0] -> l[0];\nmeasure q[0] -> c[0];\n"
        + "measure q[1] -> c[1];\n"
    )
    half = math.pi / 2
    assert program.qregs == {"q": (0, 2), "a": (2, 2)}
    assert program.ops == (
        Op("r", (0,), (half, 0.0)),
        Op("r", (1,), (-half, half)),
        Op("rz", (1,), (3.5 * math.pi,)),
        Op("zz", (0, 1), ()),
        Op("r", (2,), (math.pi, 0.0)),
        Op("r", (2,), (half, -half)),
        Op("rz", (2,), (math.pi,)),
        Op("zz", (0, 2), ()),
    )
    assert program.measured == {2: ("l", 0), 0: ("c", 0), 1: ("c", 1)}


def test_an_if_on_zero_runs_its_gate() -> None:
    program = qt.parse_qasm(HEADER + "qreg q[1];\ncreg c[1];\nif (c==0) x q[0];\n")
    assert program.ops == (Op("r", (0,), (math.pi, 0.0)),)


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("qreg q[2];\ncx q[0], q[1];", "does not know the statement cx"),
        ("qreg q[1];\nsleep(1.0) q[0];", "does not know the statement sleep"),
        ("qreg q[1];\nreset q[0];", "does not know the statement reset"),
        ("qreg q[1];\nU(0,0,0) q[0];", "does not know the statement U"),
        ("qreg q[1];\nrx(__import__('os')) q[0];", "is not an angle"),
        ("qreg q[1];\nrx(x) q[0];", "is not an angle"),
        ("qreg q[2];\nrx(pi) q;", "give one qubit"),
        ("qreg q[1];\nrx(pi) q[1];", "has no qubit 1"),
        ("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\nx q[0];", "after its measurement"),
        ("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\nif (c==1) x q[0];", "cannot know"),
        ("qreg q[1];\ncreg c[1];\nif (c==1) cx q[0], q[0];", "does not know the statement cx"),
        ("qreg q[2];\nZZ q[0], q[0];", "must be distinct"),
        ("qreg q[2];\ncreg c[2];\nmeasure q -> c[0];", "a register into a register"),
        ("qreg q[2];\ncreg c[3];\nmeasure q -> c;", "different sizes"),
        ("qreg q[2];\norder3 q[0], q[1];", "order3 takes 3 qubits"),
        ('include "other.inc";', "does not know other.inc"),
        ("qreg q[1];\ngate g a { x a; }", "does not know the statement gate"),
    ],
)
def test_the_parser_refuses_what_it_does_not_know(body: str, reason: str) -> None:
    with pytest.raises(qt.QasmError, match=reason):
        qt.parse_qasm(HEADER + body + "\n")


def test_the_parser_needs_the_openqasm_line() -> None:
    with pytest.raises(qt.QasmError, match="OPENQASM 2.0"):
        qt.parse_qasm("qreg q[1];")


# the converter --------------------------------------------------------------------------------

# Two qubits run in parallel, each with an ancilla for the leakage check. Shot strings print
# bit 0 on the right. Shot 1: c = 01 (qubit 0 read 1), l = 10 (ancilla of qubit 1 read 1).
SQ_QASM = HEADER + (
    "qreg q[2];\nqreg a[2];\ncreg c[2];\ncreg l[2];\n"
    "rx(pi) q[0];\nry(0.5*pi) q[1];\n"
    "x a[0];\nh a[0];\nZZ q[0], a[0];\nZZ q[0], a[0];\nh a[0];\n"
    "x a[1];\nh a[1];\nZZ q[1], a[1];\nZZ q[1], a[1];\nh a[1];\n"
    "measure a[0] -> l[0];\nmeasure q[0] -> c[0];\nmeasure a[1] -> l[1];\nmeasure q[1] -> c[1];\n"
)
SQ_DOC = {
    "shots": 3,
    "qasm": {"SQ_RB: (2, 1) [7]": SQ_QASM},
    "raw_data": {"SQ_RB (2, 1)": {"c": ["01", "11", "00"], "l": ["10", "00", "00"]}},
}


def test_each_parallel_qubit_becomes_a_circuit_with_the_marginal_counts_of_its_bits() -> None:
    found = qt.circuits(SQ_DOC, "SQ_RB")
    assert [c["name"] for c in found] == ["SQ_RB_q0_m2_s1", "SQ_RB_q1_m2_s1"]
    zero, one = found
    assert zero["qubits"] == [0, 2]
    assert zero["counts"] == {"10": 2, "00": 1}
    assert one["qubits"] == [1, 3]
    assert one["counts"] == {"01": 1, "10": 1, "00": 1}
    assert one["ops"][0] == Op("r", (0,), (math.pi / 2, math.pi / 2))
    assert sum(op.name == "zz" for op in one["ops"]) == 2


def test_postselection_keeps_the_shots_that_pass_the_leakage_check() -> None:
    zero, one = qt.circuits(SQ_DOC, "SQ_RB", leakage="postselect")
    assert zero["qubits"] == [0]
    assert zero["counts"] == {"1": 2, "0": 1}
    assert one["counts"] == {"1": 1, "0": 1}
    assert one["ops"] == [Op("r", (0,), (math.pi / 2, math.pi / 2))]


def test_a_pair_keeps_the_order_of_its_classical_bits() -> None:
    doc = {
        "qasm": {
            "TQ_RB: (8, 0)": HEADER
            + "qreg q[3];\ncreg c[3];\nZZ q[2], q[1];\nx q[0];\n"
            + "measure q[2] -> c[2];\nmeasure q[1] -> c[1];\nmeasure q[0] -> c[0];\n"
        },
        "raw_data": {"TQ_RB (8, 0)": {"c": ["100", "010", "011"]}},
    }
    single, pair = qt.circuits(doc, "TQ_RB")
    assert pair["name"] == "TQ_RB_q1-2_m8_s0"
    assert pair["qubits"] == [1, 2]
    assert pair["ops"] == [Op("zz", (1, 0), ())]
    assert pair["counts"] == {"01": 1, "10": 2}
    assert single["counts"] == {"0": 2, "1": 1}


def test_the_2022_form_with_one_list_of_shots_is_read() -> None:
    doc = {
        "qasm": {"128, 2": HEADER + "qreg q[2];\ncreg cr[2];\nx q[0];\nmeasure q -> cr;\n"},
        "raw data": {"128, 2": ["01", "01", "10"]},
    }
    zero, one = qt.circuits(doc, "SQ_RB")
    assert zero["name"] == "SQ_RB_q0_m128_s2"
    assert zero["counts"] == {"1": 2, "0": 1}
    assert one["name"] == "SQ_RB_q1_m128_s2"
    assert one["counts"] == {"0": 2, "1": 1}


def test_shots_must_name_the_same_circuits_as_the_programs() -> None:
    doc = {**SQ_DOC, "raw_data": {"SQ_RB (2, 2)": SQ_DOC["raw_data"]["SQ_RB (2, 1)"]}}
    with pytest.raises(qt.QasmError, match="do not name the same circuits"):
        qt.circuits(doc, "SQ_RB")


def test_a_file_with_only_summary_counts_is_refused() -> None:
    with pytest.raises(qt.QasmError, match="only summary counts"):
        qt.circuits({"shots": 100, "survival": {}}, "SQ_RB")


def test_too_many_connected_qubits_are_refused() -> None:
    gates_text = "".join(f"ZZ q[0], q[{k}];\n" for k in range(1, 11))
    measures = "".join(f"measure q[{k}] -> c[{k}];\n" for k in range(11))
    doc = {
        "qasm": {"(2, 0)": HEADER + "qreg q[11];\ncreg c[11];\n" + gates_text + measures},
        "raw_data": {"(2, 0)": {"c": ["0" * 11]}},
    }
    with pytest.raises(qt.QasmError, match="connect 11 qubits"):
        qt.circuits(doc, "Memory_RB")


def test_roles_follow_the_parity_of_the_sequence_index() -> None:
    assert qt.role("SQ_RB_q0_m2_s0") == "train"
    assert qt.role("TQ_RB_q2-3_m64_s5") == "test_near"
    assert qt.role("TQ_RB_q2-3_m64_s12") == "train"
    with pytest.raises(ValueError, match="not a circuit name"):
        qt.role("ghz_chain")


# the binding ----------------------------------------------------------------------------------


def test_the_counts_bind_to_the_bundled_profile_of_the_folder_date() -> None:
    profile = qt.bundled_profile("H1-1", date(2025, 5, 2))
    assert profile is not None
    assert profile.id == "quantinuum_h1-1"
    assert qt.bundled_profile("H1-1", date(2024, 4, 10)) is None
    (counts,) = qt.counts_files(
        SQ_DOC,
        "SQ_RB",
        profile,
        run_at=datetime(2025, 5, 2, tzinfo=UTC),
        file="data/H1-1/2025_05_02/SQ_RB.json",
        file_sha256="sha256:" + "0" * 64,
    )
    assert counts.profile.fingerprint == profile.calibration_fingerprint
    assert counts.backend == "H1-1"
    assert counts.source == "hardware"
    assert counts.execution.client == qt.CLIENT
    assert counts.run_at == datetime(2025, 5, 2, tzinfo=UTC)
    assert profile.score(counts).shots == 6


def test_the_command_writes_counts_and_profiles_with_no_local_path(tmp_path: Path) -> None:
    folder = tmp_path / "vendor" / "data" / "H1-1" / "2025_05_02"
    folder.mkdir(parents=True)
    (folder / "SQ_RB.json").write_text(json.dumps(SQ_DOC), encoding="utf-8")
    (folder / "SPAM.json").write_text(json.dumps({"shots": 1, "survival": {}}), encoding="utf-8")
    old = tmp_path / "vendor" / "data" / "H1-1" / "2024_04_10"
    old.mkdir()
    (old / "SQ_RB.json").write_text(json.dumps(SQ_DOC), encoding="utf-8")
    out = tmp_path / "out"
    lines: list[str] = []
    assert qt.main([str(tmp_path / "vendor"), "--out", str(out)]) == 0
    written = qt.convert_folder(tmp_path / "vendor", tmp_path / "again", log=lines.append)
    assert [w.benchmark for w in written] == ["SQ_RB"]
    assert any("skip H1-1 2024_04_10" in line for line in lines)
    assert any("skip data/H1-1/2025_05_02/SPAM.json" in line for line in lines)
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert files == ["H1-1/2025_05_02/SQ_RB.counts.json", "H1-1/2025_05_02/SQ_RB.profile.json"]
    counts = load_counts(out / files[0])
    assert counts.execution.options["file"] == "data/H1-1/2025_05_02/SQ_RB.json"
    for path in out.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert str(tmp_path) not in text
            assert str(Path.home()) not in text


# the variant without 1-qubit gate error -------------------------------------------------------


def tiny_profile() -> Profile:
    return Profile.from_dict(
        {
            "noisevault": "1.0",
            "device": {
                "name": "demo",
                "vendor": "example",
                "technology": "trapped_ion",
                "num_qubits": 2,
            },
            "connectivity": "all_to_all",
            "gates": {
                "rz": {"virtual": True},
                "r": {"avg_infidelity": 3e-5, "stderr": 1e-6, "duration_ns": 10000},
                "sx": {"depolarizing_param": 2e-4},
                "glow": {"qubits": 1, "pauli": [1e-4, 0, 2e-4]},
                "zz": {"avg_infidelity": 1e-3, "includes": ["1q_dressing"]},
            },
            "calibrations": [
                {"gate": "r", "qubits": [1], "avg_infidelity": 5e-5},
                {"gate": "zz", "qubits": [0, 1], "avg_infidelity": 2e-3},
            ],
            "readout": {"p1_given_0": 0.001, "p0_given_1": 0.003},
        }
    )


def test_the_variant_sets_every_one_qubit_gate_error_to_zero() -> None:
    stated = tiny_profile()
    variant = qt.without_one_qubit_gate_error(stated)
    specs = variant.to_dict()["gates"]
    assert specs["r"] == {"avg_infidelity": 0.0, "duration_ns": 10000.0}
    assert specs["sx"] == {"avg_infidelity": 0.0}
    assert specs["glow"] == {"qubits": 1, "avg_infidelity": 0.0}
    assert specs["rz"] == {"virtual": True}
    assert specs["zz"] == stated.to_dict()["gates"]["zz"]
    records = {r["gate"]: r for r in variant.to_dict()["calibrations"]}
    assert records["r"]["avg_infidelity"] == 0.0
    assert records["zz"]["avg_infidelity"] == 2e-3


def test_the_variant_keeps_readout_and_the_device() -> None:
    stated, variant = tiny_profile(), qt.without_one_qubit_gate_error(tiny_profile())
    for key in ("device", "readout", "connectivity"):
        assert variant.to_dict()[key] == stated.to_dict()[key]
    assert variant.calibration_fingerprint != stated.calibration_fingerprint
    assert "every 1-qubit gate error is set to 0" in variant.to_dict()["provenance"]["notes"][-1]


def test_the_variant_gives_a_one_qubit_circuit_only_readout_error() -> None:
    stated = tiny_profile()
    variant = qt.without_one_qubit_gate_error(stated)
    ops = [Op("r", (0,), (math.pi, 0.0))] * 51
    probs = probabilities(variant, ops, 1, layout=(1,), readout=True, unknown_gates="error")
    assert np.allclose(probs, [0.003, 0.997])
    noisy = probabilities(stated, ops, 1, layout=(1,), readout=True, unknown_gates="error")
    assert noisy[0] > 0.003


# the 1-qubit gates in the 2-qubit benchmark ---------------------------------------------------


def rb_doc() -> dict[str, dict[str, object]]:
    # Each step has 1 ZZ gate and 3 charged 1-qubit gates: 1.5 for each qubit. The rz gate is
    # virtual. The first two gates of each circuit do not grow with the length.
    start = "rx(pi) q[0];\nrx(pi) q[1];\n"
    step = "rx(pi/2) q[0];\nry(pi/2) q[0];\nrz(pi/2) q[1];\nry(pi/2) q[1];\nZZ q[0], q[1];\n"
    programs = {
        f"TQ_RB ({length}, 0)": HEADER
        + "qreg q[2];\ncreg c[2];\n"
        + start
        + step * length
        + "measure q -> c;\n"
        for length in (1, 3, 5)
    }
    return {"qasm": programs, "raw_data": {key: {"c": ["00", "00"]} for key in programs}}


def test_the_gate_count_is_the_growth_of_the_one_qubit_gates_for_each_two_qubit_gate() -> None:
    assert qt.dressing_gates(rb_doc()) == pytest.approx(1.5)


def test_the_gate_count_needs_two_sequence_lengths() -> None:
    one = {name: {"TQ_RB (1, 0)": part["TQ_RB (1, 0)"]} for name, part in rb_doc().items()}
    with pytest.raises(qt.QasmError, match="two sequence lengths"):
        qt.dressing_gates(one)


def test_the_gate_count_needs_two_qubit_gates_that_grow_with_the_length() -> None:
    programs = {
        f"TQ_RB ({length}, 0)": HEADER
        + "qreg q[1];\ncreg c[1];\n"
        + "rx(pi) q[0];\n" * length
        + "measure q -> c;\n"
        for length in (1, 3)
    }
    doc = {"qasm": programs, "raw_data": {key: {"c": ["0"]} for key in programs}}
    with pytest.raises(qt.QasmError, match="do not grow"):
        qt.dressing_gates(doc)


def uniform_profile() -> Profile:
    return Profile.from_dict(
        {
            "noisevault": "1.0",
            "device": {
                "name": "demo",
                "vendor": "example",
                "technology": "trapped_ion",
                "num_qubits": 3,
            },
            "connectivity": "all_to_all",
            "gates": {
                "rz": {"virtual": True},
                "r": {"avg_infidelity": 3e-5, "includes": ["leakage"]},
                "zz": {
                    "avg_infidelity": 1e-3,
                    "stderr": 1e-5,
                    "includes": ["1q_dressing", "leakage"],
                },
                "plain": {"qubits": 2, "avg_infidelity": 2e-3},
            },
            "effects": [
                {"type": "leakage", "gate": "r", "prob": 5e-6},
                {"type": "leakage", "gate": "zz", "prob": 2e-4},
            ],
            "readout": {"p1_given_0": 0.001, "p0_given_1": 0.003},
        }
    )


def process_error(profile: Profile, gate: str, qubits: tuple[int, ...]) -> float:
    return sum(profile.pauli_error(gate, qubits, unknown_gates="error").values())


def test_the_net_error_is_the_stated_error_minus_the_one_qubit_gates_of_the_benchmark() -> None:
    net = qt.net_of_one_qubit_dressing(uniform_profile(), 1.25)
    specs = net.to_dict()["gates"]
    # r_net = r - (6 / 5) k (r_a + r_b), from e_net = e - k (e_a + e_b) in process infidelity.
    assert specs["zz"] == {
        "avg_infidelity": pytest.approx(1e-3 - 1.2 * 1.25 * 6e-5),
        "includes": ["leakage"],
    }
    assert specs["r"] == uniform_profile().to_dict()["gates"]["r"]
    assert specs["plain"] == {"qubits": 2, "avg_infidelity": 2e-3}
    assert "1.25" in net.to_dict()["provenance"]["notes"][-1]


def test_the_net_error_and_the_one_qubit_gates_give_the_stated_process_error() -> None:
    stated = uniform_profile()
    for k in (0.5, 1.25, 2.0):
        net = qt.net_of_one_qubit_dressing(stated, k)
        dressed = process_error(net, "zz", (0, 1)) + k * (
            process_error(net, "r", (0,)) + process_error(net, "r", (1,))
        )
        assert dressed == pytest.approx(process_error(stated, "zz", (0, 1)), rel=1e-12)


def test_a_count_of_zero_keeps_the_stated_error_and_a_net_error_is_never_below_zero() -> None:
    stated = uniform_profile()
    same = qt.net_of_one_qubit_dressing(stated, 0.0).to_dict()["gates"]["zz"]
    assert same["avg_infidelity"] == 1e-3
    none = qt.net_of_one_qubit_dressing(stated, 100.0).to_dict()["gates"]["zz"]
    assert none["avg_infidelity"] == 0.0


def test_a_pauli_vector_keeps_its_shape_in_the_net_error() -> None:
    data = uniform_profile().to_dict()
    vector = [1e-4] * 3 + [2e-4] * 12
    data["gates"]["zz"] = {"pauli": vector, "includes": ["1q_dressing"]}
    net = qt.net_of_one_qubit_dressing(Profile.from_dict(data), 1.25).to_dict()["gates"]["zz"]
    # The process infidelity is the sum of the vector: 2.7e-3 - 2 k (3 / 2) 3e-5.
    factor = (2.7e-3 - 2 * 1.25 * 4.5e-5) / 2.7e-3
    assert net == {"pauli": pytest.approx([p * factor for p in vector])}


def test_the_net_error_needs_one_error_for_each_gate_on_the_whole_device() -> None:
    with pytest.raises(ValueError, match="calibration record"):
        qt.net_of_one_qubit_dressing(tiny_profile(), 1.25)


def test_without_leakage_takes_the_leakage_out_of_each_gate_error_that_holds_it() -> None:
    data = qt.without_leakage(uniform_profile()).to_dict()
    assert data["gates"]["r"] == {"avg_infidelity": pytest.approx(3e-5 - 5e-6)}
    assert data["gates"]["zz"] == {
        "avg_infidelity": pytest.approx(1e-3 - 2e-4),
        "includes": ["1q_dressing"],
    }
    assert data["gates"]["plain"] == {"qubits": 2, "avg_infidelity": 2e-3}
    assert "effects" not in data
    assert data["readout"] == uniform_profile().to_dict()["readout"]


def test_without_leakage_keeps_a_gate_error_that_does_not_hold_the_leakage() -> None:
    stated = uniform_profile().to_dict()
    stated["gates"]["r"] = {"avg_infidelity": 3e-5}
    stated["effects"].append({"type": "leakage", "gate": "zz", "prob": 9e-3})
    data = qt.without_leakage(Profile.from_dict(stated)).to_dict()
    assert data["gates"]["r"] == {"avg_infidelity": 3e-5}
    assert data["effects"] == [{"type": "leakage", "gate": "r", "prob": 5e-6, "allow": "omit"}]
    # The two leakage effects of zz hold more than its error, so the error is 0.
    assert data["gates"]["zz"] == {"avg_infidelity": 0.0, "includes": ["1q_dressing"]}
    with pytest.raises(ValueError, match="calibration record"):
        qt.without_leakage(tiny_profile())
