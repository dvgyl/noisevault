from __future__ import annotations

import math

import numpy as np
import pytest

import noisevault as nv
from noisevault.counts import plan
from noisevault.design import Price
from noisevault.profile import Profile
from study import precision


@pytest.fixture(scope="module")
def pair() -> Profile:
    return Profile.uniform(
        "pair",
        technology="trapped_ion",
        num_qubits=2,
        one_qubit_error=5e-4,
        two_qubit_error=8e-3,
        readout_error=5e-3,
    )


def test_the_relative_errors_are_the_square_roots_of_the_inverse_information() -> None:
    info = np.array([[4.0, 0.0], [0.0, 1.0]])
    assert precision.relative_errors(info, 100) == pytest.approx((0.05, 0.1))


def test_the_short_circuits_of_the_bundled_kingston_profile_give_a_known_error() -> None:
    kingston = nv.load(precision.FISHER_PROFILE)
    info = precision.run_information(kingston, plan(kingston), 1.0)
    assert precision.relative_errors(info, 4000) == pytest.approx((0.1494, 0.0768), abs=5e-5)
    assert precision.stated_errors(kingston)["sx"] == (
        "avg_infidelity",
        pytest.approx(2.3033e-4, abs=1e-8),
    )


def test_a_factor_with_no_information_has_an_infinite_error() -> None:
    # A readout error of 0 gives the readout factor no information. The gate factor then has
    # the error of a fit with one factor: 1 / sqrt(3000 * 2.2130576001).
    info = np.array([[2.2130576001, 0.0], [0.0, 0.0]])
    gate, readout = precision.relative_errors(info, 3000)
    assert gate == pytest.approx(1 / math.sqrt(3000 * 2.2130576001)) and readout == math.inf
    assert gate == pytest.approx(0.01227278, abs=1e-8)
    assert precision.relative_errors(np.zeros((2, 2)), 100) == (math.inf, math.inf)
    assert precision.relative_errors(np.ones((2, 2)), 100) == (math.inf, math.inf)
    # Information of rank one cannot separate the two factors.
    one_direction = np.outer([0.1, 0.3], [0.1, 0.3])
    assert precision.relative_errors(one_direction, 3000) == (math.inf, math.inf)


@pytest.mark.parametrize(
    ("case", "shots"),
    [
        (precision.Case("a", Price(0.30, 0.08), 160.0), 496),
        (precision.Case("b", Price(0.30, 0.0235, one_task=True), 100.0), 1060),
        (precision.Case("c", Price(5.0, 1.0), 10.0), 0),
    ],
)
def test_the_short_shots_are_the_most_that_the_amount_pays(case, shots: int) -> None:
    assert precision.short_shots(case, 4) == shots
    if shots:
        assert case.price.cost(4, shots) <= case.budget < case.price.cost(4, shots + 1)


def test_a_case_names_one_task_only_in_the_part_of_the_short_circuits() -> None:
    case = precision.Case("AQT IBEX price", Price(0.30, 0.0235, one_task=True), 212.0)
    assert case.label() == "AQT IBEX price, 212 USD"
    assert case.label(tasks=True) == "AQT IBEX price, one task, 212 USD"


def test_the_fit_circuits_tell_more_than_the_short_circuits(pair: Profile) -> None:
    lines = precision.report("fisher", pair).splitlines()
    assert lines[1].startswith("Profile: pair calibrated")
    for factor in precision.GATE_FACTORS:
        line = next(x for x in lines if x.startswith(f"true gate factor {factor}:"))
        assert float(line.split("| ratio ")[1].split()[0]) > 1


def test_the_budget_command_gives_a_design_for_each_case_and_factor(pair: Profile) -> None:
    out = precision.report("budget", pair)
    assert out.count("== ") == len(precision.CASES) * len(precision.GATE_FACTORS) + 1
    assert out.count("predicted standard error of the gate factor") == 12
    assert out.count("standard error of the gate factor") == 12 + 6
