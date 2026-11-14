from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from conftest import require

stim = require("stim")

from noisevault import events  # noqa: E402
from noisevault.counts import MeasuredCircuit  # noqa: E402
from noisevault.profile import Profile  # noqa: E402
from study import wide as study  # noqa: E402

RUN_AT = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)
WIDTH = 12


def _factor(profile: Profile, factor: float) -> Profile:
    return profile.model_copy(update={"unmodeled_error": {"gates": {"factor": factor}}})


@pytest.fixture(scope="module")
def line_9() -> Profile:
    return Profile.uniform(
        "line_9",
        technology="superconducting",
        num_qubits=9,
        connectivity=[(q, q + 1) for q in range(8)],
        one_qubit_error=1e-3,
        two_qubit_error=1e-2,
        readout_error=2e-2,
        t1_us=200.0,
        t2_us=150.0,
        one_qubit_ns=32.0,
        two_qubit_ns=68.0,
    )


@pytest.fixture(scope="module")
def code() -> stim.Circuit:
    return stim.Circuit.generated("repetition_code:memory", distance=5, rounds=5)


def _detections(profile: Profile, circuit: stim.Circuit, shots: int, seed: int) -> np.ndarray:
    return profile.to_stim(circuit).compile_detector_sampler(seed=seed).sample(shots)


# the move of the study code ------------------------------------------------------------------


def _draws(n: int, rate: float, seed: int) -> np.ndarray:
    r = np.random.default_rng(seed)
    common = r.random(n) < 0.05
    return (r.random((n, WIDTH)) < rate) | (common[:, None] & (r.random((n, WIDTH)) < 0.5))


def _counts(bits: np.ndarray) -> dict[str, int]:
    rows, times = np.unique(bits.astype(np.uint8), axis=0, return_counts=True)
    return {"".join(map(str, row)): int(k) for row, k in zip(rows, times, strict=True)}


def test_the_package_gives_the_numbers_of_the_study_code_on_a_fixed_case() -> None:
    # The values come from study/wide.py before its method moved into noisevault.events.
    circuit = MeasuredCircuit(
        name="clifford_fixed",
        qubits=tuple(range(WIDTH)),
        ops=(),
        shots=3000,
        counts=_counts(_draws(3000, 0.04, 1)),
    )
    bits_a, bits_b = _draws(20_000, 0.04, 2), _draws(20_000, 0.025, 3)
    a = events.accumulate([bits_a[:7000], bits_a[7000:]], WIDTH)
    b = events.accumulate([bits_b], WIDTH)
    measured = events.circuit_events(circuit, "0" * WIDTH)

    result = events.compare([measured], [a], [b])
    old = study.compare(
        [circuit],
        [study.Forecast(circuit.name, "0" * WIDTH, a)],
        [study.Forecast(circuit.name, "0" * WIDTH, b)],
    )

    exact = {"rel": 1e-12, "abs": 0.0}
    assert a.none == pytest.approx(0.5814959252037398, **exact)
    assert float(a.rate.sum()) == pytest.approx(0.7613619319034047, **exact)
    assert float(a.covariance.sum()) == pytest.approx(2.1202267900000002, **exact)
    assert vars(result.a) == pytest.approx(
        {
            "none_log": -0.6804062355064037,
            "none_quadratic": 0.5126868631819647,
            "count_log": -1.1537574240803292,
            "count_quadratic": 0.42387116682766995,
            "bit_log": -2.9357870428458384,
        },
        **exact,
    )
    assert result.b.bit_log == pytest.approx(-2.96578669315047, **exact)
    assert result.count_log.z == pytest.approx(7.406009749115345, **exact)
    assert result.count_log.better == "neither"
    assert result.bit_log.z == pytest.approx(3.6961267110943288, **exact)
    assert result.bit_log.p_value == pytest.approx(0.00021891374412477934, **exact)
    assert result.bit_log.better == "a"
    assert result.bit_sampling.variance == pytest.approx(123.47005455662966, **exact)
    assert result.bit_sampling.shot_variance == pytest.approx(469.42959195276603, **exact)
    assert result.count_sampling.variance == pytest.approx(53.6092427231064, **exact)
    assert result.count_sampling.reason == "few_samples"
    assert events.scores([measured], [b]).count_log == pytest.approx(-1.2016901712511143, **exact)
    assert not events.few_rate_samples(a)
    assert events.few_count_samples(b, measured, a, b)
    assert old.qubit_log == result.bit_log and old.wrong_log == result.count_log
    assert old.a.qubit_log == result.a.bit_log and old.b.success_log == result.b.none_log
