"""Leakage, atom loss and erasure in the Stim export, against analytic results."""

import numpy as np
import pytest
from conftest import require

stim = require("stim")

from noisevault.errors import UnsupportedEffect  # noqa: E402
from noisevault.profile import Profile  # noqa: E402

SHOTS = 200_000


def _atoms(*effects: dict, n: int = 30, noise: float = 0.0, **gates) -> Profile:
    return Profile.uniform(
        "atoms",
        technology="neutral_atom",
        num_qubits=n,
        one_qubit_error=noise,
        two_qubit_error=noise,
        effects=effects,
        **gates,
    )


def _within_5_sigma(rate: float, expected: float, shots: int = SHOTS) -> None:
    sigma = np.sqrt(max(expected * (1 - expected), 1e-12) / shots)
    assert abs(rate - expected) < 5 * sigma + 1e-12, (rate, expected)


def _mechanisms(dem: stim.DetectorErrorModel) -> dict[tuple[str, ...], float]:
    """Each set of symptoms with the probability that an odd number of its errors occur."""
    found: dict[tuple[str, ...], float] = {}
    for inst in dem.flattened():
        if inst.type == "error":
            key = tuple(sorted(str(t) for t in inst.targets_copy()))
            p, q = inst.args_copy()[0], found.get(key, 0.0)
            found[key] = p + q - 2 * p * q
    return found


def test_a_heralded_erasure_on_a_gate_heralds_with_its_prob_and_randomizes_the_qubit():
    noisy = _atoms({"type": "erasure", "gate": "x", "prob": 0.2, "heralded": True}).to_stim(
        "X 0\nM 0"
    )
    assert str(noisy) == "X 0\nHERALDED_ERASE(0.2) 0\nM 0"
    herald, bit = noisy.compile_sampler(seed=1).sample(SHOTS).T
    _within_5_sigma(herald.mean(), 0.2)
    assert bit[~herald].all()  # no erasure, no error
    _within_5_sigma(bit[herald].mean(), 0.5, int(herald.sum()))


def test_an_unheralded_erasure_flips_a_z_measurement_with_half_its_prob():
    noisy = _atoms({"type": "erasure", "gate": "x", "prob": 0.2}).to_stim(
        "X 0\nM 0\nDETECTOR rec[-1]"
    )
    assert "DEPOLARIZE1(0.15" in str(noisy)
    assert _mechanisms(noisy.detector_error_model()) == pytest.approx({("D0",): 0.1})


def test_each_qubit_of_a_two_qubit_gate_is_erased_on_its_own():
    noisy = _atoms({"type": "erasure", "gate": "cx", "prob": 0.3, "heralded": True}).to_stim(
        "CX 0 1\nCX 1 2"
    )
    assert "HERALDED_ERASE(0.3) 0 1" in str(noisy)
    first, second = noisy.compile_sampler(seed=2).sample(SHOTS)[:, :2].T
    _within_5_sigma(first.mean(), 0.3)
    _within_5_sigma((first & second).mean(), 0.09)


@pytest.mark.parametrize("heralded", [True, False])
def test_repetition_code_detection_events_match_the_analytic_rates(heralded):
    # Loss at each readout makes that one result random: a flip with probability p/2. A
    # detector over n results flips when an odd number of them flip: (1 - (1 - p)^n) / 2.
    p, distance, rounds = 0.08, 3, 4
    effect = {"type": "atom_loss", "on": "readout", "prob": p, "heralded": heralded}
    code = stim.Circuit.generated("repetition_code:memory", distance=distance, rounds=rounds)
    noisy = _atoms(effect).to_stim(code)  # the profile has no readout error
    events = noisy.compile_detector_sampler(seed=3).sample(SHOTS).mean(axis=0)
    ancillas = distance - 1
    results = [1] * ancillas + [2] * ancillas * (rounds - 1) + [3] * ancillas
    assert len(events) == len(results)
    for rate, n in zip(events, results, strict=True):
        _within_5_sigma(rate, (1 - (1 - p) ** n) / 2)
    heralds = noisy.num_measurements - code.num_measurements
    assert heralds == (code.num_measurements if heralded else 0)


@pytest.mark.parametrize(
    "code",
    [
        stim.Circuit.generated("repetition_code:memory", distance=3, rounds=4),
        stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=5),
        stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=3),
    ],
    ids=["repetition", "surface_z", "surface_x"],
)
def test_heralds_leave_detectors_and_observables_on_the_same_measurements(code):
    def device(heralded: bool) -> Profile:
        return _atoms(
            {"type": "erasure", "gate": "cx", "prob": 0.02, "heralded": heralded},
            {"type": "atom_loss", "on": "readout", "prob": 0.01, "heralded": heralded},
            {"type": "leakage", "on": "idle", "rate_per_s": 1e3, "heralded": heralded},
            noise=1e-3,
            t1_us=1e3,
            t2_us=1e3,
        )

    heralded = device(True).to_stim(code, tick_ns=100)
    silent = device(False).to_stim(code, tick_ns=100)
    assert heralded.num_measurements > silent.num_measurements == code.num_measurements
    # Stim reads HERALDED_ERASE's Paulis as independent, DEPOLARIZE1's exactly: an O(p^2) gap.
    found = _mechanisms(heralded.detector_error_model())
    expected = _mechanisms(silent.detector_error_model())
    assert found.keys() == expected.keys()
    assert found == pytest.approx(expected, abs=4e-4)
    flat = device(True).to_stim(code.flattened(), tick_ns=100)
    assert flat.flattened() == heralded.flattened()


def test_heralds_shift_record_targets_through_nested_loops_feedback_and_padding():
    program = stim.Circuit("""
        R 0 1
        MPAD 0
        REPEAT 3 {
            CX 0 1
            REPEAT 2 {
                X 0
                M 1
                CX rec[-1] 0
                DETECTOR rec[-1] rec[-2]
            }
            M 0
            DETECTOR rec[-1] rec[-4]
        }
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-2] rec[-10]
    """)

    def export(circuit: stim.Circuit, heralded: bool) -> stim.Circuit:
        effect = {"type": "erasure", "gate": "x", "prob": 0.1, "heralded": heralded}
        return _atoms(effect).to_stim(circuit)

    noisy = export(program, True)
    assert noisy.flattened() == export(program.flattened(), True).flattened()
    found = _mechanisms(noisy.detector_error_model())
    expected = _mechanisms(export(program, False).detector_error_model())
    assert found.keys() == expected.keys()
    assert found == pytest.approx(expected, abs=0.01)
    events = [line for line in noisy.report.summary().splitlines() if line.startswith("used")]
    assert events == ["used: the erasure effect added a herald measurement record 6 times"]


def test_kept_heralded_noise_counts_as_a_record_to_shift():
    program = "R 0\nHERALDED_ERASE(0.5) 0\nX 0\nM 0\nDETECTOR rec[-1]\nDETECTOR rec[-2]"
    effect = {"type": "erasure", "gate": "x", "prob": 0.1, "heralded": True}
    noisy = _atoms(effect).to_stim(program, existing_noise="keep")
    assert str(noisy).splitlines()[-2:] == ["DETECTOR rec[-1]", "DETECTOR rec[-3]"]


def test_exact_readout_gives_herald_records_no_flip():
    profile = _atoms(
        {"type": "atom_loss", "on": "readout", "prob": 0.1, "heralded": True}, readout_error=0.02
    )
    noisy = profile.to_stim("M 0 1", readout="exact")
    assert str(noisy) == "HERALDED_ERASE(0.1) 0 1\nM 0 1"
    assert noisy.readout_flips.tolist() == [[0, 0], [0, 0], [0.02, 0.02], [0.02, 0.02]]


def test_effects_act_on_their_qubits_and_a_locus_record_replaces_the_default():
    profile = _atoms(
        {"type": "erasure", "gate": "cx", "prob": 0.1},
        {"type": "erasure", "gate": "cx", "qubits": [1, 0], "prob": 0.3},
        {"type": "erasure", "on": "readout", "qubits": [2], "prob": 0.2},
    )
    noisy = profile.to_stim("CX 0 1\nCX 1 0\nCX 1 2\nM 0 1 2")
    assert noisy == stim.Circuit(f"""
        CX 0 1
        DEPOLARIZE1({0.75 * 0.3!r}) 0 1
        CX 1 0
        DEPOLARIZE1({0.75 * 0.3!r}) 1 0
        CX 1 2
        DEPOLARIZE1({0.75 * 0.1!r}) 1 2
        DEPOLARIZE1({0.75 * 0.2!r}) 2
        M 0 1 2
    """)


def test_an_idle_effect_turns_its_rate_into_a_probability_per_tick():
    profile = _atoms({"type": "erasure", "on": "idle", "rate_per_s": 1e4, "heralded": True})
    noisy = profile.to_stim("H 0 1\nTICK\nH 0\nTICK", tick_ns=1000)
    assert [str(inst) for inst in noisy] == [
        "H 0 1",
        "TICK",
        "H 0",
        "HERALDED_ERASE(0.00995017) 1",
        "TICK",
    ]
    assert noisy[3].gate_args_copy() == pytest.approx([1 - np.exp(-1e4 * 1e-6)], rel=1e-12)


def test_the_report_states_how_each_effect_is_treated():
    profile = _atoms(
        {"type": "erasure", "gate": "cx", "prob": 0.01, "heralded": True},
        {"type": "atom_loss", "on": "readout", "prob": 0.01},
        {"type": "leakage", "on": "idle", "rate_per_s": 10.0},
        {"type": "crosstalk_zz", "on": "idle", "strength_hz": 1e3},
        {"type": "erasure", "gate": "x"},
    )
    lines = profile.to_stim("CX 0 1\nM 0").report.summary().splitlines()
    assert "exact: effect erasure on cx: a heralded erasure (the qubit becomes I/2)" in lines
    assert (
        "approximated: effect atom_loss on readout: an erasure (the qubit becomes I/2) (the atom"
        " comes back at once; on the device the trap stays empty)"
    ) in lines
    (omitted,) = [line for line in lines if line.startswith("omitted")]
    assert "effect leakage on idle (the stim export models no idle time)" in omitted
    assert "effect crosstalk_zz on idle (the stim export does not model crosstalk_zz)" in omitted
    assert "effect erasure on x (the effect gives no prob)" in omitted


@pytest.mark.parametrize(
    ("effect", "options", "message", "hint"),
    [
        (
            {"type": "atom_loss", "on": "readout", "prob": 0.01, "allow": "exact"},
            {},
            "but the stim export approximates it as an erasure",
            "set allow to 'approximate'",
        ),
        (
            {"type": "erasure", "on": "readout", "prob": 0.01, "allow": "approximate"},
            {"readout": "none"},
            "but the stim export adds no readout error",
            "set allow to 'omit'",
        ),
        (
            {"type": "erasure", "on": "idle", "rate_per_s": 1.0, "allow": "exact"},
            {"tick_ns": 50.0},
            "approximates it as an erasure (the qubit becomes I/2) (only where the export",
            "set allow to 'approximate'",
        ),
        (
            {"type": "coherent_overrotation", "gate": "cx", "angle_rad": 0.01, "allow": "exact"},
            {},
            "but the stim export does not model coherent_overrotation",
            "set allow to 'omit'",
        ),
    ],
)
def test_an_effect_that_asks_for_more_than_the_export_gives_is_refused(
    effect, options, message, hint
):
    with pytest.raises(UnsupportedEffect, match="allow=") as caught:
        _atoms(effect).to_stim("CX 0 1\nM 0", **options)
    assert message in caught.value.message
    assert hint in (caught.value.hint or "")


def test_exact_erasure_meets_allow_exact():
    effect = {"type": "erasure", "gate": "cx", "prob": 0.01, "allow": "exact"}
    assert "DEPOLARIZE1" in str(_atoms(effect).to_stim("CX 0 1"))


def test_leakage_already_in_a_gate_error_is_not_added_again():
    data = _atoms({"type": "leakage", "gate": "cx", "prob": 0.01}, noise=1e-3).to_dict()
    data["gates"]["cx"]["includes"] = ["leakage"]
    profile = Profile.from_dict(data)
    noisy = profile.to_stim("CX 0 1")
    assert "DEPOLARIZE1" not in str(noisy) and "ERASE" not in str(noisy)
    assert (
        "approximated: effect leakage on cx: counted in the stated error of cx as depolarizing"
        " noise (the gate's includes lists leakage, so the export does not add the effect again)"
    ) in noisy.report.summary().splitlines()
