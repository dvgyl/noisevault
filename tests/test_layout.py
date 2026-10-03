from __future__ import annotations

import time
import warnings

import numpy as np
import pytest
from conftest import toy

from noisevault.errors import LayoutError, NoiseVaultWarning
from noisevault.layout import normalize_layout, suggest_layout
from noisevault.profile import Profile


def _line(n: int, **sections) -> Profile:
    return Profile.model_validate(
        toy(
            device={"name": "line", "technology": "superconducting", "num_qubits": n},
            connectivity={"edges": [[i, i + 1] for i in range(n - 1)]},
            **sections,
        )
    )


def test_integer_labels_default_to_identity() -> None:
    assert normalize_layout([2, 0], None, _line(3)) == {2: 2, 0: 0}
    assert normalize_layout([np.int64(1)], None, _line(3)) == {1: 1}


def test_mapping_and_sequence_layouts() -> None:
    profile = _line(5)
    assert normalize_layout(["a", "b"], {"a": 4, "b": 1, "unused": 0}, profile) == {"a": 4, "b": 1}
    assert normalize_layout([0, 1], [3, 2], profile) == {0: 3, 1: 2}


_EVERY_QUBIT = "<physical qubit>, ...} covering every circuit qubit"


@pytest.mark.parametrize(
    ("labels", "layout", "match", "hint"),
    [
        (["a"], None, "no integer index", "pass layout={'a': " + _EVERY_QUBIT),
        ([True], None, "no integer index", "pass layout={True: " + _EVERY_QUBIT),
        ([0, 1], {0: 1}, r"no physical qubit for \[1\]", "map every circuit qubit"),
        ([0, 1], {0: 2, 1: 2}, "both 0 and 1 to qubit 2", None),
        ([0], {0: 9}, "qubits 0..4", None),
        ([0], {0: -1}, "qubits 0..4", None),
        ([0], {0: "q3"}, "not a qubit index", None),
        (
            [0],
            {0: 3},
            "disabled",
            "choose another qubit (profile.suggest_layout(1) proposes a usable chain)",
        ),
    ],
)
def test_layout_errors(labels, layout, match, hint) -> None:
    profile = _line(5, qubits=[{"index": 3, "disabled": True}])
    with pytest.raises(LayoutError, match=match) as caught:
        normalize_layout(labels, layout, profile)
    assert caught.value.hint == hint


def test_the_disabled_qubit_hint_needs_a_chain_as_wide_as_the_circuit() -> None:
    profile = _line(5, qubits=[{"index": 2, "disabled": True}])
    chain_of_2 = "choose another qubit (profile.suggest_layout(2) proposes a usable chain)"

    def hint(labels: list[int], layout: list[int], **kwargs: int) -> str | None:
        with pytest.raises(LayoutError, match="disabled") as caught:
            normalize_layout(labels, layout, profile, **kwargs)
        return caught.value.hint

    assert hint([0, 1], [0, 2]) == chain_of_2
    assert hint([0, 1, 2], [0, 1, 2]) is None
    assert hint([0], [2], width=2) == chain_of_2
    assert hint([0], [2], width=3) is None
    assert hint([0], [2], width=0) is None


def test_adapters_can_supply_their_own_integer_rule() -> None:
    class LineQubit:
        def __init__(self, x: int) -> None:
            self.x = x

    qubits = [LineQubit(1), LineQubit(4)]
    mapping = normalize_layout(qubits, None, _line(5), index_of=lambda q: q.x)
    assert list(mapping.values()) == [1, 4]


def _grid(rows: int, cols: int, seed: int = 3) -> Profile:
    rng = np.random.default_rng(seed)
    n = rows * cols
    edges = [(r * cols + c, r * cols + c + 1) for r in range(rows) for c in range(cols - 1)]
    edges += [(r * cols + c, (r + 1) * cols + c) for r in range(rows - 1) for c in range(cols)]
    calibrations = [
        {"gate": "cz", "qubits": list(e), "avg_infidelity": float(rng.uniform(2e-3, 2e-2))}
        for e in edges
    ]
    calibrations += [
        {"gate": "sx", "qubits": [q], "avg_infidelity": float(rng.uniform(1e-4, 1e-3))}
        for q in range(n)
    ]
    qubits = [{"index": q, "readout": {"error": float(rng.uniform(5e-3, 5e-2))}} for q in range(n)]
    return Profile.model_validate(
        toy(
            device={"name": "grid", "technology": "superconducting", "num_qubits": n},
            connectivity={"edges": [list(e) for e in edges]},
            calibrations=calibrations,
            qubits=qubits,
        )
    )


def _is_chain(profile: Profile, layout: dict[int, int]) -> bool:
    edges = {frozenset(e) for e in profile.table.edges()}
    path = [layout[i] for i in range(len(layout))]
    return len(set(path)) == len(path) and all(
        frozenset(pair) in edges for pair in zip(path, path[1:], strict=False)
    )


def test_suggest_layout_is_a_deterministic_connected_chain_of_good_qubits() -> None:
    profile = _grid(4, 4)
    layout = suggest_layout(profile, 5)
    assert sorted(layout) == [0, 1, 2, 3, 4] and _is_chain(profile, layout)
    assert suggest_layout(_grid(4, 4), 5) == layout
    worst = max(range(16), key=lambda q: profile.table.qubit(q).readout[0])
    assert worst not in layout.values()


def test_suggest_layout_avoids_disabled_qubits_and_gates() -> None:
    profile = _line(5, qubits=[{"index": 1, "disabled": True}])
    assert set(suggest_layout(profile, 3).values()) == {2, 3, 4}
    broken = _line(5, calibrations=[{"gate": "cz", "qubits": [2, 3], "disabled": True}])
    with pytest.raises(LayoutError, match="no connected chain"):
        suggest_layout(broken, 4)
    assert set(suggest_layout(broken, 3).values()) == {0, 1, 2}


@pytest.mark.parametrize("where", ["record", "definition"])
def test_suggest_layout_leaves_out_qubits_that_cannot_measure(where) -> None:
    gates = {**toy()["gates"], "measure": {"disabled": where == "definition"}}
    on = [{"gate": "measure", "qubits": [q], "disabled": False} for q in (2, 3, 4)]
    off = [{"gate": "measure", "qubits": [q], "disabled": True} for q in (0, 1)]
    profile = _line(5, gates=gates, calibrations=on if where == "definition" else off)
    assert set(suggest_layout(profile, 3).values()) == {2, 3, 4}
    with pytest.raises(
        LayoutError, match=r"^line has only 3 enabled qubits that can measure, not 4$"
    ):
        suggest_layout(profile, 4)


def test_suggest_layout_leaves_out_pairs_the_caller_rules_out() -> None:
    profile = _line(4)
    assert suggest_layout(profile, 3) == {0: 0, 1: 1, 2: 2}
    assert suggest_layout(profile, 3, usable_pair=lambda a, b: (a, b) != (0, 1)) == {
        0: 1,
        1: 2,
        2: 3,
    }
    with pytest.raises(LayoutError, match="no connected chain of 4 usable qubits"):
        suggest_layout(profile, 4, usable_pair=lambda a, b: (a, b) != (1, 2))


@pytest.mark.timing
def test_suggest_layout_is_fast_at_156_qubits() -> None:
    profile = _grid(12, 13)
    profile.table.typical(1, (0,))  # table built outside the timing
    start = time.perf_counter()
    layout = suggest_layout(profile, 20)
    elapsed = time.perf_counter() - start
    assert _is_chain(profile, layout) and elapsed < 0.5


def test_suggest_layout_on_all_to_all() -> None:
    profile = Profile.uniform(
        "ions", technology="trapped_ion", num_qubits=56, one_qubit_error=3e-5, two_qubit_error=1e-3
    )
    assert len(set(suggest_layout(profile, 10).values())) == 10
    with pytest.raises(LayoutError):
        suggest_layout(profile, 57)


@pytest.mark.timing
def test_suggest_layout_on_all_to_all_takes_the_best_qubits_fast() -> None:
    good = [5, 17, 140]
    data = Profile.uniform(
        "ions",
        technology="trapped_ion",
        num_qubits=156,
        one_qubit_error=1e-3,
        two_qubit_error=1e-2,
        readout_error=0.02,
    ).to_dict()
    data["qubits"] = [{"index": q, "readout": {"error": 0.001}} for q in good]
    data["qubits"].append({"index": 3, "disabled": True})
    profile = Profile.model_validate(data)
    profile.table.typical(1, (0,))  # table built outside the timing
    start = time.perf_counter()
    layout = suggest_layout(profile, 100)
    elapsed = time.perf_counter() - start
    assert sorted(layout) == list(range(100)) and len(set(layout.values())) == 100
    assert set(good) <= set(layout.values()) and 3 not in layout.values()
    assert elapsed < 0.05


def _ions(**sections) -> Profile:
    gates = {"rz": {"virtual": True}, "x": {"avg_infidelity": 1e-3}, "cz": {"avg_infidelity": 1e-2}}
    sections.setdefault("connectivity", "all_to_all")
    return Profile.model_validate(toy(gates=sections.pop("gates", gates), **sections))


def test_suggest_layout_on_all_to_all_skips_a_disabled_pair() -> None:
    profile = _ions(calibrations=[{"gate": "cz", "qubits": [0, 1], "disabled": True}])
    layout = suggest_layout(profile, 2)
    assert profile.table.allowed("cz", (layout[0], layout[1]))


@pytest.mark.parametrize("connectivity", ["all_to_all", {"edges": [[0, 1], [1, 2]]}])
def test_missing_calibration_ranks_after_calibrated_qubits(connectivity) -> None:
    gates = {"rz": {"virtual": True}, "x": {}, "cz": {"avg_infidelity": 1e-2}}
    calibrations = [{"gate": "x", "qubits": [1], "avg_infidelity": 0.5}]
    profile = _ions(gates=gates, calibrations=calibrations, connectivity=connectivity)
    assert suggest_layout(profile, 1) == {0: 1}
    readout = _ions(qubits=[{"index": 2, "readout": {"error": 0.4}}], connectivity=connectivity)
    assert suggest_layout(readout, 1) == {0: 2}


@pytest.mark.parametrize("connectivity", ["all_to_all", {"edges": [[0, 1], [1, 2]]}])
def test_missing_gate_calibration_ranks_after_missing_readout(connectivity) -> None:
    gates = {"rz": {"virtual": True}, "x": {}, "cz": {"avg_infidelity": 1e-2}}
    profile = _ions(
        gates=gates,
        calibrations=[{"gate": "x", "qubits": [1], "avg_infidelity": 1e-3}],
        qubits=[{"index": 0, "readout": {"error": 1e-4}}],
        connectivity=connectivity,
    )
    layout = suggest_layout(profile, 1)
    assert layout == {0: 1}
    assert profile.table.gate("x", (layout[0],)).state == "calibrated"


def test_a_pair_calibrated_outside_the_connectivity_links_the_chain() -> None:
    profile = _ions(
        connectivity={"edges": [[0, 1]]},
        calibrations=[{"gate": "cz", "qubits": [1, 2], "avg_infidelity": 0.02}],
    )
    layout = suggest_layout(profile, 3)
    path = [layout[i] for i in range(3)]
    assert path in ([0, 1, 2], [2, 1, 0])
    assert all(profile.table.allowed("cz", pair) for pair in zip(path, path[1:], strict=False))


_SX_X = {
    "rz": {"virtual": True},
    "sx": {"avg_infidelity": 2e-3},
    "x": {"avg_infidelity": 1e-3},
    "cz": {"avg_infidelity": 1e-2},
}


def test_suggest_layout_skips_a_qubit_without_sx() -> None:
    # Without sx, qubit 0 has only rz and x: its cheap x would otherwise rank it first.
    profile = _ions(gates=_SX_X, calibrations=[{"gate": "sx", "qubits": [0], "disabled": True}])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        layout = suggest_layout(profile, 2)
    assert set(layout.values()) == {1, 2}


@pytest.mark.parametrize(
    ("natives", "off"),
    [
        (("sx", "x"), ["x"]),
        (("sx", "x"), ["sx"]),
        (("sx", "x"), ["rz"]),
        (("rx", "ry"), ["rx"]),
        (("h", "x"), ["x"]),
        (("u", "sx"), ["u"]),
    ],
)
def test_a_qubit_missing_any_native_the_others_have_is_chosen_last(natives, off) -> None:
    # Qubit 0 reads out best, and its remaining gates could make the missing one in some cases.
    gates = {"rz": {"virtual": True}, **{n: {"avg_infidelity": 1e-3} for n in natives}}
    profile = _ions(
        gates={**gates, "cz": {"avg_infidelity": 1e-2}},
        calibrations=[{"gate": g, "qubits": [0], "disabled": True} for g in off],
        qubits=[{"index": 0, "readout": {"error": 1e-4}}],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert suggest_layout(profile, 1)[0] != 0


def test_a_qubit_missing_only_the_identity_is_complete() -> None:
    gates = {"rz": {"virtual": True}, **{n: {"avg_infidelity": 1e-3} for n in ("sx", "x", "id")}}
    profile = _ions(
        gates={**gates, "cz": {"avg_infidelity": 1e-2}},
        calibrations=[{"gate": "id", "qubits": [0], "disabled": True}],
        qubits=[{"index": 0, "readout": {"error": 1e-4}}],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert suggest_layout(profile, 1) == {0: 0}


@pytest.mark.parametrize(
    "x_off",
    [
        {"gates": {**_SX_X, "x": {"avg_infidelity": 1e-3, "disabled": True}}},
        {"calibrations": [{"gate": "x", "qubits": [q], "disabled": True} for q in range(3)]},
    ],
    ids=["by default", "on every qubit"],
)
def test_a_gate_disabled_everywhere_is_not_required(x_off) -> None:
    profile = _ions(**{"gates": _SX_X, **x_off}, qubits=[{"index": 0, "readout": {"error": 1e-4}}])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert suggest_layout(profile, 1) == {0: 0}
        assert set(suggest_layout(profile, 3).values()) == {0, 1, 2}


def test_a_qubit_where_a_record_enables_a_disabled_default_is_complete() -> None:
    profile = _ions(
        gates={**_SX_X, "sx": {"avg_infidelity": 2e-3, "disabled": True}},
        calibrations=[{"gate": "sx", "qubits": [2], "disabled": False}],
        qubits=[{"index": 0, "readout": {"error": 1e-4}}],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert suggest_layout(profile, 1) == {0: 2}
    with pytest.warns(NoiseVaultWarning, match=r"qubit 0 \(sx disabled\)"):
        suggest_layout(profile, 2)


_QUBIT_0_SHORT = {
    "sx re-enabled elsewhere": {
        "gates": {**_SX_X, "sx": {"avg_infidelity": 2e-3, "disabled": True}},
        "calibrations": [{"gate": "sx", "qubits": [q], "disabled": False} for q in (1, 2)],
    },
    "rx off, sx and ry left": {
        "gates": {
            "sx": {"avg_infidelity": 2e-3},
            "ry": {"avg_infidelity": 2e-3},
            "rx": {"avg_infidelity": 1e-3},
            "cz": {"avg_infidelity": 1e-2},
        },
        "calibrations": [{"gate": "rx", "qubits": [0], "disabled": True}],
        "qubits": [{"index": q, "readout": {"error": e}} for q, e in enumerate([1e-3, 0.2, 0.3])],
    },
}


@pytest.mark.parametrize(
    ("case", "missing"), [("sx re-enabled elsewhere", "sx"), ("rx off, sx and ry left", "rx")]
)
def test_suggest_layout_avoids_a_qubit_missing_a_native_other_qubits_have(
    case: str, missing: str
) -> None:
    profile = _ions(**_QUBIT_0_SHORT[case])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        layout = suggest_layout(profile, 2)
    assert set(layout.values()) == {1, 2}
    with pytest.warns(NoiseVaultWarning, match=rf"qubit 0 \({missing} disabled\)"):
        assert set(suggest_layout(profile, 3).values()) == {0, 1, 2}


def test_suggest_layout_falls_back_to_an_incomplete_qubit_and_says_so() -> None:
    profile = _ions(gates=_SX_X, calibrations=[{"gate": "sx", "qubits": [0], "disabled": True}])
    with pytest.warns(NoiseVaultWarning, match=r"qubit 0 .*sx"):
        layout = suggest_layout(profile, 3)
    assert set(layout.values()) == {0, 1, 2}
