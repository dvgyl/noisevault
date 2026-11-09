"""IonQ profiles from the public characterization endpoint.

Offline tests serve ``fixtures/ionq/responses.json`` (made-up numbers in the real response
shape) in place of the network.
"""

from __future__ import annotations

import copy
import hashlib
import json
import urllib.error
import urllib.request
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import deeper_than_the_parser_takes

from noisevault.errors import SourceDataError, SourceUnavailable
from noisevault.reference import Op, probabilities
from noisevault.sources import ionq

RESPONSES = json.loads((Path(__file__).parent / "fixtures" / "ionq" / "responses.json").read_text())


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """Serve the fixture; every URL the module asks for must be one of them."""
    bodies = {url: json.dumps(body).encode() for url, body in RESPONSES.items()}

    def get(url: str) -> bytes:
        assert url in bodies, f"unexpected request {url}"
        return bodies[url]

    monkeypatch.setattr(ionq, "_get", get)
    return bodies


@pytest.mark.parametrize("name", ["ionq_forte-1", "forte-1", "qpu.forte-1", " QPU.Forte-1 "])
def test_device_names_accept_every_spelling(name: str) -> None:
    assert ionq.backend_name(name) == "qpu.forte-1"


def test_newest_record_with_its_own_fidelities_is_chosen(served) -> None:
    # The newest record lacks 1Q; the next lacks 2Q, but on the 10-record page the API has
    # filled it from the newer record. Its own one-record page shows it missing, so skip it.
    profile = ionq.pull("forte-1")
    assert profile.device.calibrated_at.isoformat() == "2026-09-27T00:00:00+00:00"
    assert "skipped 2 newer records without 1Q/2Q fidelities" in profile.provenance.notes
    url = ionq._page_url("qpu.forte-1", limit=1, end="2026-09-27T00:00:00Z")
    assert profile.provenance.source_url == url
    assert profile.provenance.source_hash == "sha256:" + hashlib.sha256(served[url]).hexdigest()


def test_medians_become_device_defaults_with_stated_assumptions(served) -> None:
    profile = ionq.pull("qpu.forte-1")
    assert (profile.id, profile.device.num_qubits) == ("ionq_forte-1", 4)
    assert profile.connectivity == "all_to_all"
    one, two = profile.gates["r"], profile.gates["rzz"]
    assert (one.avg_infidelity, one.duration_ns) == (0.0002, 130_000)
    assert (two.avg_infidelity, two.duration_ns) == (0.0049, 970_000)
    for spec in (one, two):
        assert (spec.statistic, spec.stderr) == ("median", None)  # IonQ's stderr 0 is dropped
        assert "IonQ does not state the fidelity metric" in spec.assumption
    assert (profile.readout.error, profile.readout.duration_ns) == (0.0088, 150_000)
    assert (profile.idle.t1_us, profile.idle.t2_us) == (1e8, 1e6)
    assert any("nominal" in note for note in profile.provenance.notes)
    assert profile.prep is None
    prov = profile.provenance
    assert (prov.source_kind, prov.redistributable, prov.attribution) == (
        "public_api",
        "no",
        "IonQ",
    )


def test_placeholder_spam_is_missing_not_data(served) -> None:
    profile = ionq.pull("ionq_forte-1", at="2026-09-01")
    assert profile.device.calibrated_at.date().isoformat() == "2026-09-01"
    assert profile.readout is None
    assert profile.table.qubit(0).readout is None


def test_ms_backends_and_partial_connectivity(served) -> None:
    profile = ionq.pull("aria-1")
    assert "ms" in profile.gates and "rzz" not in profile.gates
    assert profile.connectivity.edges == ((0, 1), (1, 2))


@pytest.mark.parametrize(
    ("device", "two_qubit", "params"), [("forte-1", "rzz", (0.8,)), ("aria-1", "ms", (0.0, 0.0))]
)
def test_profiles_convert_on_their_natives(served, device: str, two_qubit: str, params) -> None:
    profile = ionq.pull(device)
    ops = [Op("r", (0,), (np.pi / 2, 0.0)), Op(two_qubit, (0, 1), params), Op("rz", (1,), (0.3,))]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert probabilities(profile, ops, 2).sum() == pytest.approx(1)


def test_implausible_medians_are_skipped_and_named(served) -> None:
    # aria-1's newest record has 1Q fidelity 0.79 next to a normal 2Q: corrupt, not data.
    profile = ionq.pull("aria-1")
    assert profile.device.calibrated_at.date().isoformat() == "2026-02-18"
    assert profile.gates["r"].avg_infidelity == pytest.approx(0.0068)
    [note] = [n for n in profile.provenance.notes if "implausible" in n]
    assert (
        "skipped 1 newer record with implausible" in note
        and "corrupt: 2026-03-01 (1Q 0.79, 2Q 0.9861)" in note
    )


@pytest.mark.parametrize(
    ("one", "two", "usable"),
    [(0.9998, 0.995, True), (0.79, 0.995, False), (0.9998, 0.75, False), (0.9706, 0.9882, False)],
)
def test_gate_medians_below_floor_or_inverted_are_rejected(one, two, usable) -> None:
    record = {
        "date": "2025-01-01T00:00:00Z",
        "fidelity": {"1q": {"median": one}, "2q": {"median": two}},
    }
    assert (ionq._rejection(record) is None) == usable


def test_implausible_spam_leaves_readout_unknown(served) -> None:
    body = json.loads(served[ionq._page_url("qpu.forte-1", limit=1, end="2026-09-27T00:00:00Z")])
    record = body["characterizations"][0]
    record["fidelity"]["spam"]["median"] = 0.7261
    listing = RESPONSES["https://api.ionq.co/v0.4/backends"][0]
    profile = ionq.to_profile(record, listing, source_url="u", source_hash="sha256:" + "0" * 64)
    assert profile.readout is None
    assert any("SPAM fidelity median 0.7261" in note for note in profile.provenance.notes)


def test_a_dated_record_keeps_its_own_qubit_count(served) -> None:
    body = json.loads(served[ionq._page_url("qpu.forte-1", limit=1, end="2026-09-27T00:00:00Z")])
    record = body["characterizations"][0]
    grown = {**RESPONSES["https://api.ionq.co/v0.4/backends"][0], "qubits": 8}
    profile = ionq.to_profile(record, grown, source_url="u", source_hash="sha256:" + "0" * 64)
    assert profile.device.num_qubits == record["qubits"] == 4
    assert profile.connectivity == "all_to_all"

    del record["qubits"]
    profile = ionq.to_profile(record, grown, source_url="u", source_hash="sha256:" + "0" * 64)
    assert profile.device.num_qubits == 8
    assert (
        "the record gives no qubit count, so NoiseVault uses the qubit count of the current"
        " listing (8)"
    ) in profile.provenance.notes


@pytest.mark.parametrize(
    ("bad", "problem"),
    [
        ([2, 4], "connectivity edge [2, 4] is outside 0..3"),
        ([2, 2], "connectivity.edges: edge [2, 2] joins a qubit to itself"),
    ],
)
def test_a_bad_edge_is_rejected_even_among_as_many_edges_as_a_complete_graph(
    served, bad: list[int], problem: str
) -> None:
    for url in [u for u in served if "qpu.forte-1/characterizations" in u]:
        body = json.loads(served[url])
        for record in body["characterizations"] or []:
            record["connectivity"] = [bad if e == [2, 3] else e for e in record["connectivity"]]
        served[url] = json.dumps(body).encode()
    with pytest.raises(SourceDataError) as info:
        ionq.pull("forte-1")
    assert (info.value.message, info.value.hint) == (f"IonQ's API ({_CHOSEN}): {problem}", _OLDER)


def test_errors_say_what_to_do(served) -> None:
    with pytest.raises(SourceUnavailable, match="qpu.aria-1, qpu.forte-1, qpu.harmony"):
        ionq.pull("forte-9")
    with pytest.raises(SourceUnavailable) as info:
        ionq.pull("forte-1", at="2025-01-01")
    assert info.value.message == (
        "IonQ publishes no qpu.forte-1 characterization at or before 2025-01-01 with plausible"
        " 1Q and 2Q fidelities of its own (0 records checked)"
    )
    assert info.value.hint == "pass a later at= or none"
    # every harmony record lacks a 1Q fidelity, so no date can help
    with pytest.raises(SourceUnavailable) as info:
        ionq.pull("harmony")
    assert info.value.message == (
        "IonQ publishes no qpu.harmony characterization with plausible 1Q and 2Q fidelities of"
        " its own (2 records checked)"
    )
    assert info.value.hint == "use another backend"


def test_a_search_that_runs_out_of_probes_says_to_search_older(
    served, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ionq, "_MAX_PROBES", 1)
    with pytest.raises(SourceUnavailable) as info:
        ionq.pull("forte-1")
    assert info.value.message == (
        "none of the 4 newest qpu.forte-1 characterizations has plausible 1Q and 2Q fidelities"
        " of its own"
    )
    assert info.value.hint == "pass an earlier at= to search older records"


def test_a_record_without_fidelities_cannot_make_a_profile() -> None:
    record = {"backend": "qpu.forte-1", "id": "r1", "qubits": 4, "fidelity": {}}
    with pytest.raises(SourceUnavailable) as info:
        ionq.to_profile(record, {"qubits": 4}, source_url="u", source_hash="h")
    assert info.value.message == "IonQ published qpu.forte-1 record r1 without 1Q/2Q fidelities"
    assert info.value.hint == "use a record from another date"


def _refuse(handler: object, request: urllib.request.Request) -> None:
    raise urllib.error.URLError("timed out")


def _throttle(handler: object, request: urllib.request.Request) -> None:
    raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {}, None)


@pytest.mark.parametrize(
    ("https_open", "message", "hint"),
    [
        (
            _refuse,
            "could not reach IonQ's API (timed out)",
            "check the network connection, or run nv list to see every profile you can load"
            " offline",
        ),
        (_throttle, f"IonQ's API answered HTTP 429 for {ionq.API}/backends", "try again later"),
    ],
)
def test_a_failed_request_says_what_to_do(
    monkeypatch: pytest.MonkeyPatch, https_open, message: str, hint: str
) -> None:
    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", https_open)
    with pytest.raises(SourceUnavailable) as info:
        ionq.pull("forte-1")
    assert (info.value.message, info.value.hint) == (message, hint)


_NOT_JSON = {
    "html": b"<!DOCTYPE html><html><body>502 Bad Gateway</body></html>",
    "cut short": json.dumps(RESPONSES[f"{ionq.API}/backends"]).encode()[:100],
    "latin-1": "<html>Accès refusé</html>".encode("latin-1"),
}


@pytest.mark.parametrize("body", [*_NOT_JSON, "too deep"])
@pytest.mark.parametrize(
    "url",
    [
        f"{ionq.API}/backends",
        ionq._page_url("qpu.forte-1", limit=10),
        ionq._page_url("qpu.forte-1", limit=1, end="2026-09-28T00:00:00Z"),
    ],
    ids=["listing", "page", "record"],
)
def test_a_reply_that_is_not_json_says_to_try_again(served, url: str, body: str) -> None:
    raw = deeper_than_the_parser_takes().encode() if body == "too deep" else _NOT_JSON[body]
    served[url] = raw
    with pytest.raises(SourceUnavailable) as info:
        ionq.pull("forte-1")
    assert (info.value.message, info.value.hint) == (
        f"IonQ's API answered {url} with something other than JSON",
        "try again later",
    )


_LISTING = f"{ionq.API}/backends"
_PAGE = ionq._page_url("qpu.forte-1", limit=10)
_CHOSEN = ionq._page_url("qpu.forte-1", limit=1, end="2026-09-27T00:00:00Z")


def _edited(url: str, edit: Callable[[Any], object]) -> bytes:
    body = copy.deepcopy(RESPONSES[url])
    edit(body)
    return json.dumps(body).encode()


def _first_record_with(url: str, **fields: Any) -> bytes:
    return _edited(url, lambda body: body["characterizations"][0].update(fields))


_WRONG_SHAPE = {
    "listing as text": (_LISTING, b'"maintenance"', ""),
    "a backend without a name": (
        _LISTING,
        _edited(_LISTING, lambda body: body[0].pop("backend")),
        " at [0].backend",
    ),
    "a qubit count as text in the listing": (
        _LISTING,
        _edited(_LISTING, lambda body: body[0].update(qubits="four")),
        " at [0].qubits",
    ),
    "native gates as text": (
        _LISTING,
        _edited(_LISTING, lambda body: body[0].update(supported_native_gates="zz")),
        " at [0].supported_native_gates",
    ),
    "native gates as null": (
        _LISTING,
        _edited(_LISTING, lambda body: body[0].update(supported_native_gates=None)),
        " at [0].supported_native_gates",
    ),
    "page as a list": (_PAGE, b"[]", ""),
    "records as text": (
        _PAGE,
        _edited(_PAGE, lambda body: body.update(characterizations="x")),
        " at characterizations",
    ),
    "a record without a date": (
        _PAGE,
        _edited(_PAGE, lambda body: body["characterizations"][2].pop("date")),
        " at characterizations[2].date",
    ),
    "fidelity as a list": (
        _PAGE,
        _first_record_with(_PAGE, fidelity=[0.99]),
        " at characterizations[0].fidelity",
    ),
    "a page count as text": (
        _PAGE,
        _edited(_PAGE, lambda body: body.update(pages="four")),
        " at pages",
    ),
    "record as text": (_CHOSEN, b'"maintenance"', ""),
    "an id as a number": (
        _CHOSEN,
        _first_record_with(_CHOSEN, id=3),
        " at characterizations[0].id",
    ),
    "a backend as a number": (
        _CHOSEN,
        _first_record_with(_CHOSEN, backend=1),
        " at characterizations[0].backend",
    ),
    "a date that is not a date": (
        _CHOSEN,
        _first_record_with(_CHOSEN, date="yesterday"),
        " at characterizations[0].date",
    ),
    "a qubit count as text": (
        _CHOSEN,
        _first_record_with(_CHOSEN, qubits="four"),
        " at characterizations[0].qubits",
    ),
    "a pair of three qubits": (
        _CHOSEN,
        _first_record_with(_CHOSEN, connectivity=[[0, 1, 2]]),
        " at characterizations[0].connectivity[0]",
    ),
    "timing as a list": (
        _CHOSEN,
        _first_record_with(_CHOSEN, timing=[1e-4]),
        " at characterizations[0].timing",
    ),
    "a time as text": (
        _CHOSEN,
        _edited(_CHOSEN, lambda body: body["characterizations"][0]["timing"].update({"1q": "1"})),
        " at characterizations[0].timing['1q']",
    ),
    "a median as text": (
        _CHOSEN,
        _edited(
            _CHOSEN, lambda body: body["characterizations"][0]["fidelity"]["2q"].update(median="1")
        ),
        " at characterizations[0].fidelity['2q'].median",
    ),
}


@pytest.mark.parametrize("case", list(_WRONG_SHAPE))
def test_a_reply_of_the_wrong_shape_says_where_and_to_try_again(served, case: str) -> None:
    url, raw, where = _WRONG_SHAPE[case]
    served[url] = raw
    with pytest.raises(SourceUnavailable) as info:
        ionq.pull("forte-1")
    assert (info.value.message, info.value.hint) == (
        f"IonQ's API answered {url} with JSON of the wrong shape{where}",
        "try again later",
    )


_OLDER = "pass an earlier at= to use an older calibration"


@pytest.mark.parametrize(
    "missing",
    [
        pytest.param(dict.pop, id="absent"),
        pytest.param(lambda entry, key: entry.update({key: None}), id="null"),
    ],
)
def test_a_record_and_a_listing_without_a_qubit_count_say_so(served, missing) -> None:
    served[_LISTING] = _edited(_LISTING, lambda body: missing(body[0], "qubits"))
    served[_CHOSEN] = _edited(_CHOSEN, lambda body: missing(body["characterizations"][0], "qubits"))
    with pytest.raises(SourceDataError) as info:
        ionq.pull("forte-1")
    assert (info.value.message, info.value.hint) == (
        f"IonQ's API ({_CHOSEN}): record 00000000-0000-4000-8000-000000000003 gives no qubit"
        " count, and neither does the backend listing",
        _OLDER,
    )


@pytest.mark.parametrize("listing", [[], [RESPONSES[_LISTING][-1]]], ids=["empty", "simulator"])
def test_a_listing_with_no_qpu_says_so(served, listing: list[dict[str, Any]]) -> None:
    assert all(not entry["backend"].startswith("qpu.") for entry in listing)
    served[_LISTING] = json.dumps(listing).encode()
    with pytest.raises(SourceUnavailable) as info:
        ionq.pull("forte-1")
    assert (info.value.message, info.value.hint) == (
        f"IonQ has no backend 'qpu.forte-1'. The IonQ listing ({_LISTING}) names no QPU",
        "try again later",
    )


def test_nothing_is_bundled() -> None:
    assert ionq.bundled_profiles() == []


@pytest.mark.network
def test_live_forte_1() -> None:
    profile = ionq.pull("qpu.forte-1")
    assert profile.device.num_qubits == 36
    assert profile.connectivity == "all_to_all"
    assert profile.gates["rzz"].statistic == "median"
    assert 0 < profile.gates["rzz"].avg_infidelity < 0.1


@pytest.mark.parametrize(("key", "seconds"), [("1q", -0.00013), ("t1", 0)])
def test_a_time_that_is_not_positive_names_the_field(served, key: str, seconds: float) -> None:
    served[_CHOSEN] = _edited(
        _CHOSEN, lambda body: body["characterizations"][0]["timing"].update({key: seconds})
    )
    with pytest.raises(SourceDataError) as info:
        ionq.pull("forte-1")
    assert (info.value.message, info.value.hint) == (
        f"IonQ's API ({_CHOSEN}): timing.{key} is {seconds!r}, not a positive number of seconds",
        _OLDER,
    )


def test_a_qubit_count_of_0_is_refused_not_replaced_by_the_listing(served) -> None:
    served[_CHOSEN] = _first_record_with(_CHOSEN, qubits=0)
    with pytest.raises(SourceDataError) as info:
        ionq.pull("forte-1")
    assert (info.value.message, info.value.hint) == (
        f"IonQ's API ({_CHOSEN}): device.num_qubits: Input should be greater than or equal to 1,"
        " got 0",
        _OLDER,
    )


def test_a_median_above_1_is_read_as_corrupt_and_named(served) -> None:
    served[_CHOSEN] = _edited(
        _CHOSEN, lambda body: body["characterizations"][0]["fidelity"]["1q"].update(median=1.5)
    )
    profile = ionq.pull("forte-1")
    assert profile.device.calibrated_at.date().isoformat() == "2026-09-01"
    [note] = [n for n in profile.provenance.notes if "implausible" in n]
    assert "a median above 1" in note
    assert note.endswith("read as corrupt: 2026-09-27 (1Q 1.5, 2Q 0.9951)")


def test_a_spam_median_above_1_leaves_readout_unknown_and_says_so(served) -> None:
    served[_CHOSEN] = _edited(
        _CHOSEN, lambda body: body["characterizations"][0]["fidelity"]["spam"].update(median=1.5)
    )
    profile = ionq.pull("forte-1")
    assert profile.readout is None
    assert (
        "SPAM fidelity median 1.5 is above 1 (corrupt), so NoiseVault leaves readout error unknown"
        in profile.provenance.notes
    )
