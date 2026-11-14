from __future__ import annotations

from datetime import UTC, datetime

import pytest
from conftest import toy

import noisevault as nv
from noisevault.profile import Profile

_PAULI = [0.001, 0.002, 0.003]


def _pair(**sections) -> Profile:
    data = toy(
        device={"name": "pair", "technology": "superconducting", "num_qubits": 2},
        connectivity={"edges": [[0, 1]]},
        readout={"error": 0.02},
        idle={"t1_us": 100.0, "t2_us": 80.0},
        qubits=[{"index": 0, "readout": {"p1_given_0": 0.0, "p0_given_1": 0.04}}],
        calibrations=[
            {"gate": "sx", "qubits": [1], "pauli": _PAULI, "stderr": 1e-4},
            {"gate": "cz", "qubits": [0, 1], "avg_infidelity": 0.2},
        ],
    )
    return Profile.model_validate({**data, **sections})


def _values(profile: Profile) -> dict:
    table = profile.table
    return {
        "sx0": table.gate("sx", (0,)).avg_infidelity,
        "sx1": table.gate("sx", (1,)).spec.pauli,
        "stderr": table.gate("sx", (1,)).spec.stderr,
        "cz": table.gate("cz", (0, 1)).avg_infidelity,
        "readout0": table.qubit(0).readout,
        "readout1": table.qubit(1).readout,
        "t1": table.qubit(0).t1_ns,
        "durations": (table.gate("sx", (0,)).duration_ns, table.gate("cz", (0, 1)).duration_ns),
    }


@pytest.mark.parametrize(
    ("factors", "changed"),
    [
        ({"one_qubit": 2.0}, {"sx0": 2e-3, "sx1": tuple(2 * p for p in _PAULI), "stderr": 2e-4}),
        ({"two_qubit": 3.0}, {"cz": 0.6}),
        ({"readout": 1.5}, {"readout0": (0.0, 0.06), "readout1": (0.03, 0.03)}),
    ],
)
def test_scaled_multiplies_the_stated_errors_of_one_group(factors, changed) -> None:
    profile = _pair()
    found = profile.scaled(**factors)
    expected = _values(profile) | changed
    for key, value in _values(found).items():
        assert value == pytest.approx(expected[key]), key
    assert found.fingerprint != profile.fingerprint
    assert found.provenance.derived_from == f"sha256:{profile.fingerprint}"
    assert found.provenance.source_kind == "derived"


def test_scaled_states_the_factors_and_keeps_unmodeled_error() -> None:
    profile = _pair(unmodeled_error={"gates": {"factor": 1.2}})
    found = profile.scaled(one_qubit=2, two_qubit=0.5, readout=3)
    assert found.provenance.notes[-1] == (
        "The stated errors are multiplied by 2 for 1-qubit gates, 0.5 for 2-qubit gates and 3"
        " for readout."
    )
    assert found.unmodeled_error == profile.unmodeled_error


def test_scaled_with_factors_of_1_is_the_profile_itself() -> None:
    profile = _pair()
    assert profile.scaled() is profile
    assert profile.scaled(one_qubit=1, two_qubit=1.0, readout=1) is profile


@pytest.mark.parametrize(
    ("factors", "message"),
    [
        (
            {"two_qubit": 4.5},
            r"^two_qubit=4.5 puts cz on qubits 0-1 \(avg_infidelity 0.2\) outside its valid"
            r" range. The largest valid two_qubit factor for this profile is 4$",
        ),
        (
            {"readout": 30},
            r"^readout=30 puts p0_given_1 of qubit 0 \(0.04\) outside its valid range. The"
            r" largest valid readout factor for this profile is 25$",
        ),
        ({"one_qubit": -1}, r"^one_qubit must be a finite number of 0 or more, got -1$"),
    ],
)
def test_scaled_refuses_a_value_outside_its_range(factors, message) -> None:
    with pytest.raises(ValueError, match=message):
        _pair().scaled(**factors)


def test_scaled_up_to_the_largest_valid_factor_works() -> None:
    assert _pair().scaled(two_qubit=4).table.gate("cz", (0, 1)).avg_infidelity == pytest.approx(0.8)


def test_scaled_keeps_failed_values_as_stated() -> None:
    profile = _pair(
        qubits=[{"index": 0, "readout": {"p1_given_0": 0.9, "p0_given_1": 0.04}}],
        calibrations=[{"gate": "cz", "qubits": [0, 1], "avg_infidelity": 0.8}],
    )
    found = profile.scaled(two_qubit=2, readout=2)
    assert found.table.gate("cz", (0, 1)).avg_infidelity == pytest.approx(0.8)
    assert found.table.qubit(0).readout == pytest.approx((0.9, 0.04))
    assert found.table.qubit(1).readout == pytest.approx((0.04, 0.04))
    assert found.provenance.notes[-1].endswith(" 2 failed gates or readouts stay as stated.")


def test_scaled_multiplies_the_stderr_of_a_record_that_takes_the_default_error() -> None:
    gates = {**toy()["gates"], "sx": {"avg_infidelity": 0.01, "duration_ns": 35}}
    profile = _pair(gates=gates, calibrations=[{"gate": "sx", "qubits": [0], "stderr": 0.001}])
    found = profile.scaled(one_qubit=2).table.gate("sx", (0,))
    assert found.avg_infidelity == pytest.approx(0.02)
    assert found.spec.stderr == pytest.approx(0.002)


def test_scaled_refuses_fitted_factors() -> None:
    fitted = nv.load("ibm_fez")
    fit = {
        "counts": "sha256:" + "0" * 64,
        "source": "hardware",
        "qubits": [0, 1, 2],
        "run_at": "2026-01-01T00:00:00Z",
        "calibration": fitted.calibration_fingerprint,
        "p_value": 0.5,
        "impossible_shots": 0,
    }
    profile = fitted.model_copy(
        update={"unmodeled_error": {"gates": {"factor": 1.2, "low": 1.1, "high": 1.3}, "fit": fit}}
    )
    with pytest.raises(ValueError, match=r"Scale profile.uncorrected\(\) instead$"):
        profile.scaled(two_qubit=2)


def _day(when: datetime, t1: float, p10: float, cz: float) -> Profile:
    return _pair(
        device={
            "name": "pair",
            "technology": "superconducting",
            "num_qubits": 2,
            "calibrated_at": when.isoformat(),
        },
        qubits=[{"index": 0, "t1_us": t1, "readout": {"p1_given_0": p10, "p0_given_1": 0.03}}],
        calibrations=[{"gate": "cz", "qubits": [0, 1], "avg_infidelity": cz}],
    )


def test_typical_takes_the_median_of_each_value() -> None:
    days = [
        _day(datetime(2026, 6, 1, tzinfo=UTC), 200.0, 0.02, 0.010),
        _day(datetime(2026, 6, 2, tzinfo=UTC), 120.0, 0.01, 0.020),
        _day(datetime(2026, 6, 3, tzinfo=UTC), 50.0, 0.0, 0.004),
    ]
    found = nv.typical(days)
    assert found.table.qubit(0).t1_ns == pytest.approx(120_000.0)
    assert found.table.qubit(0).readout == pytest.approx((0.01, 0.03))
    assert found.table.gate("cz", (0, 1)).avg_infidelity == pytest.approx(0.010)
    assert found.device.calibrated_at == days[-1].device.calibrated_at
    assert found.provenance.derived_from == f"sha256:{days[-1].fingerprint}"
    assert found.provenance.notes[-1] == (
        "Each T1, T2, readout probability and gate error is the median over 3 profiles,"
        " calibrated 2026-06-01T00:00:00Z to 2026-06-03T00:00:00Z."
    )
    two = nv.typical(days[:2])
    assert two.table.qubit(0).t1_ns == pytest.approx(160_000.0)


def test_typical_keeps_the_structure_of_the_last_profile() -> None:
    off = _pair(qubits=[{"index": 0, "disabled": True}])
    found = nv.typical([_pair(), off])
    assert found.table.qubit(0).disabled
    assert nv.typical([off, _pair()]).table.qubit(0).readout == pytest.approx((0.0, 0.04))


def test_typical_refuses_profiles_of_two_devices() -> None:
    with pytest.raises(
        ValueError, match="^typical needs profiles of one device, got pair and test_toy$"
    ):
        nv.typical([Profile.model_validate(toy()), _pair()])
    with pytest.raises(ValueError, match="at least one profile"):
        nv.typical([])
