"""ADDRESS false-positive on bare toponyms (TRAU-530).

Recon-confirmed root cause: `GLINER2_ENTITY_MAPPING` mapped the GLiNER label
'city' straight to `ADDRESS`, so a bare toponym with no structural signal
("from cahersiveen to thurles via farranfore") was masked as
[ADDRESS_1]/[ADDRESS_2]/[ADDRESS_3] and the LLM refused to answer. A bare place
name is not personal data; `PRESIDIO_TO_STANDARD` already documents that intent
by deliberately excluding `LOCATION`, and 'city' -> ADDRESS routed around it.

Fix: 'city' maps to the TRANSIENT type `ADDRESS_CITY`, resolved in a new
Step 0.5 of `_select_accepted_detections` (between the TRAU-522 placeholder
overlap filter and the whitelist filter). A city span adjacent to a real
ADDRESS span — separated only by punctuation/whitespace/digits — is promoted
and MERGED into one ADDRESS span; a city with no address anchor is discarded.
Adjacency is the only signal: no keyword tables, no city whitelist, no
structural regex.

`ADDRESS_CITY` is transient by construction and must NEVER reach the output —
it is not in `PRESIDIO_TO_STANDARD`, so the Step 1 whitelist is a second,
independent backstop behind the explicit Step 0.5 discard. See
`test_address_city_never_escapes_selection` — that one is an invariant test,
not a feature test.
"""

from __future__ import annotations

import pytest
from presidio_analyzer import RecognizerResult

from tests.conftest import (
    FakeAnalyzer,
    make_gliner_pipeline,
    pii_mod,
    user_payload,
)

ADDRESS_CITY = pii_mod._ADDRESS_CITY_TYPE

# Minimal whitelist for the unit path. Mirrors the production contract: the
# transient ADDRESS_CITY is deliberately absent.
_STANDARD = {"PERSON": "PERSON", "ADDRESS": "ADDRESS"}


def _dets(text: str, items: list[tuple[str, str]], score: float = 0.9):
    """Build detections by locating each (substring, entity_type) in order,
    advancing a cursor so repeated substrings get distinct offsets."""
    out: list[RecognizerResult] = []
    cursor = 0
    for needle, etype in items:
        i = text.index(needle, cursor)
        out.append(
            RecognizerResult(entity_type=etype, start=i, end=i + len(needle), score=score)
        )
        cursor = i + len(needle)
    return out


def _select(text: str, dets, mapping=None):
    return pii_mod._select_accepted_detections(text, dets, mapping or _STANDARD)


def _spans(results):
    return sorted((r.entity_type, r.start, r.end) for r in results)


def _masked(text: str, results):
    return sorted(text[r.start : r.end] for r in results)


# ---------------------------------------------------------------------------
# Wiring: the label mapping and the transient-type contract
# ---------------------------------------------------------------------------


def test_city_label_maps_to_transient_type_not_address():
    """The root cause, pinned: 'city' must NOT map straight to ADDRESS."""
    assert pii_mod.GLINER2_ENTITY_MAPPING["city"] == ADDRESS_CITY


@pytest.mark.parametrize("label", ["address", "street address", "postal code"])
def test_structural_address_labels_still_map_to_address(label):
    """Only 'city' changes; the structural labels keep their direct mapping."""
    assert pii_mod.GLINER2_ENTITY_MAPPING[label] == "ADDRESS"


def test_transient_type_is_not_a_canonical_output_type():
    """Backstop contract: ADDRESS_CITY absent from the whitelist means the Step 1
    filter drops it even if Step 0.5 ever failed to resolve it."""
    assert ADDRESS_CITY not in pii_mod.Pipeline.PRESIDIO_TO_STANDARD


def test_gliner_detector_emits_transient_type_for_city_label():
    """End of the wire: the real `GLiNER2Detector.detect` (fake model, real
    mapping/offset code) must emit ADDRESS_CITY for a 'city' span."""

    class FakeModel:
        def extract_entities(self, text, labels, include_confidence=True, include_spans=True):
            i = text.find("Zagreb")
            return {
                "entities": {
                    "city": [
                        {"text": "Zagreb", "start": i, "end": i + 6, "confidence": 0.9}
                    ]
                }
            }

    det = pii_mod.GLiNER2Detector()
    det._model = FakeModel()
    results = det.detect("Idem u Zagreb sutra.")

    assert [(r.entity_type, r.start, r.end) for r in results] == [(ADDRESS_CITY, 7, 13)]


# ---------------------------------------------------------------------------
# Gap predicate (Step 0.5 uses its OWN constants, independent of the
# whitespace-only `_ADJACENCY_MAX_WS_GAP` used by `_merge_adjacent_same_type`)
# ---------------------------------------------------------------------------


def test_city_gap_constants_are_independent_of_adjacency_merge():
    """The two predicates are deliberately separate: `_merge_adjacent_same_type`
    allows whitespace ONLY (a comma between two PERSON sub-tokens means two
    distinct people), while address components are routinely comma-separated."""
    assert pii_mod._CITY_ADJACENCY_MAX_GAP != pii_mod._ADJACENCY_MAX_WS_GAP
    for ch in " ,.-0123456789":
        assert ch in pii_mod._CITY_ADJACENCY_ALLOWED_GAP_CHARS
    for ch in "abcXYZčšž":
        assert ch not in pii_mod._CITY_ADJACENCY_ALLOWED_GAP_CHARS


def test_letters_in_gap_block_promotion():
    """'to'/'via'/'and' between spans means separate entities, not one address."""
    text = "Ilica 5 near Zagreb"
    dets = _dets(text, [("Ilica 5", "ADDRESS"), ("Zagreb", ADDRESS_CITY)])
    assert _spans(_select(text, dets)) == [("ADDRESS", 0, 7)]


def test_gap_beyond_max_blocks_promotion():
    """A long all-allowed gap still blocks: distance is its own signal."""
    text = "Ilica 5, 111111111111, Zagreb"  # 14-char gap, all allowed chars
    dets = _dets(text, [("Ilica 5", "ADDRESS"), ("Zagreb", ADDRESS_CITY)])
    gap = text[7 : text.index("Zagreb")]
    assert len(gap) > pii_mod._CITY_ADJACENCY_MAX_GAP
    assert all(c in pii_mod._CITY_ADJACENCY_ALLOWED_GAP_CHARS for c in gap)
    assert _spans(_select(text, dets)) == [("ADDRESS", 0, 7)]


def test_gap_at_exactly_max_is_allowed():
    text = "Ilica 5, 12345678, Zagreb"
    dets = _dets(text, [("Ilica 5", "ADDRESS"), ("Zagreb", ADDRESS_CITY)])
    gap = text[7 : text.index("Zagreb")]
    assert len(gap) == pii_mod._CITY_ADJACENCY_MAX_GAP
    assert _spans(_select(text, dets)) == [("ADDRESS", 0, len(text))]


# ---------------------------------------------------------------------------
# POSITIVE: a real address masks as ONE span
# ---------------------------------------------------------------------------


def test_1_street_number_postal_city_is_one_span():
    """Acceptance 1. GLiNER emits the postal code as its own ADDRESS span."""
    text = "Vukovarska 23, 10000 Zagreb"
    dets = _dets(
        text,
        [("Vukovarska 23", "ADDRESS"), ("10000", "ADDRESS"), ("Zagreb", ADDRESS_CITY)],
    )
    out = _select(text, dets)
    assert _spans(out) == [("ADDRESS", 0, 27)]
    assert _masked(text, out) == ["Vukovarska 23, 10000 Zagreb"]


def test_1b_same_text_when_postal_code_is_not_detected():
    """Acceptance 1, variant. Whether GLiNER emits a separate 'postal code' span
    is NEUTVRĐENO (neural, not statically knowable), so both shapes are pinned:
    with the postal code undetected the gap ', 10000 ' is still all-allowed."""
    text = "Vukovarska 23, 10000 Zagreb"
    dets = _dets(text, [("Vukovarska 23", "ADDRESS"), ("Zagreb", ADDRESS_CITY)])
    out = _select(text, dets)
    assert _spans(out) == [("ADDRESS", 0, 27)]


def test_2_street_number_city_is_one_span():
    """Acceptance 2. The ', ' gap that `_merge_adjacent_same_type` refuses."""
    text = "Ilica 5, Zagreb"
    dets = _dets(text, [("Ilica 5", "ADDRESS"), ("Zagreb", ADDRESS_CITY)])
    out = _select(text, dets)
    assert _spans(out) == [("ADDRESS", 0, 15)]
    assert _masked(text, out) == ["Ilica 5, Zagreb"]


def test_3_chained_city_spans_merge_transitively():
    """Acceptance 3. Two city spans chain off one address anchor — a single
    left-to-right pass over start-sorted spans reaches the fixed point."""
    text = "12 Main Street, Thurles, Co. Tipperary"
    dets = _dets(
        text,
        [
            ("12 Main Street", "ADDRESS"),
            ("Thurles", ADDRESS_CITY),
            ("Co. Tipperary", ADDRESS_CITY),
        ],
    )
    out = _select(text, dets)
    assert _spans(out) == [("ADDRESS", 0, 38)]
    assert _masked(text, out) == ["12 Main Street, Thurles, Co. Tipperary"]


def test_3b_second_city_alone_does_not_survive_a_blocked_chain():
    """Acceptance 3, pessimistic variant. If GLiNER returns 'Tipperary' without
    'Co.', the letters in ', Co. ' break the chain: the anchored part still
    masks whole, the orphaned city is discarded (never masked alone)."""
    text = "12 Main Street, Thurles, Co. Tipperary"
    dets = _dets(
        text,
        [
            ("12 Main Street", "ADDRESS"),
            ("Thurles", ADDRESS_CITY),
            ("Tipperary", ADDRESS_CITY),
        ],
    )
    out = _select(text, dets)
    assert _spans(out) == [("ADDRESS", 0, 23)]


def test_4_address_inside_a_sentence():
    """Acceptance 4."""
    text = "Poslao sam paket na Savsku cestu 32, 10000 Zagreb"
    dets = _dets(
        text,
        [("Savsku cestu 32", "ADDRESS"), ("10000", "ADDRESS"), ("Zagreb", ADDRESS_CITY)],
    )
    out = _select(text, dets)
    assert _spans(out) == [("ADDRESS", 20, 49)]
    assert _masked(text, out) == ["Savsku cestu 32, 10000 Zagreb"]


def test_13_city_left_of_address_also_merges():
    """Acceptance 13 (S2): the predicate is direction-agnostic."""
    text = "Zagreb, Vukovarska 23"
    dets = _dets(text, [("Zagreb", ADDRESS_CITY), ("Vukovarska 23", "ADDRESS")])
    out = _select(text, dets)
    assert _spans(out) == [("ADDRESS", 0, 21)]
    assert _masked(text, out) == ["Zagreb, Vukovarska 23"]


def test_14_digits_in_gap_over_mask_is_accepted_behaviour():
    """Acceptance 14. ACCEPTED OVER-MASKING, not a bug: digits are allowed in the
    gap because a postal code routinely sits between street and city, and we
    cannot tell '10000' (postal) from '47' (anything else) without the structural
    tables this ticket forbids. Merging only ever GROWS a masked span, so the
    worst case is over-masking (fail-safe), never a leak — the same argument
    `_merge_adjacent_same_type` makes for its own widening."""
    text = "Vukovarska 23, 47 Zagreb"
    dets = _dets(text, [("Vukovarska 23", "ADDRESS"), ("Zagreb", ADDRESS_CITY)])
    out = _select(text, dets)
    assert _spans(out) == [("ADDRESS", 0, 24)]


def test_score_of_merged_span_is_max_of_members():
    """S1. Step 5 sorts by score DESC, so the merged span must carry the
    strongest member's score, not the first one's."""
    text = "Ilica 5, Zagreb"
    dets = [
        RecognizerResult(entity_type="ADDRESS", start=0, end=7, score=0.61),
        RecognizerResult(entity_type=ADDRESS_CITY, start=9, end=15, score=0.93),
    ]
    out = _select(text, dets)
    assert len(out) == 1
    assert out[0].score == pytest.approx(0.93)


# ---------------------------------------------------------------------------
# NEGATIVE: bare toponyms mask nothing
# ---------------------------------------------------------------------------


def test_5_the_reported_bug_masks_nothing():
    """Acceptance 5 — the exact reported message. Three bare toponyms, no
    address anchor anywhere: nothing may be masked."""
    text = (
        "help me plan a coordinated bus and train trip from cahersiveen "
        "to thurles via farranfore on thursday 23rd july"
    )
    dets = _dets(
        text,
        [
            ("cahersiveen", ADDRESS_CITY),
            ("thurles", ADDRESS_CITY),
            ("farranfore", ADDRESS_CITY),
        ],
    )
    assert _select(text, dets) == []


@pytest.mark.parametrize(
    "text,city",
    [
        ("I need to check weather today in New York", "New York"),
        ("Koji je najbolji restoran u Zagrebu?", "Zagrebu"),
        ("Let's meet in Dublin next week", "Dublin"),
    ],
    ids=["new-york", "zagreb", "dublin"],
)
def test_6_7_8_standalone_city_is_never_masked(text, city):
    """Acceptance 6-8."""
    dets = _dets(text, [(city, ADDRESS_CITY)])
    assert _select(text, dets) == []


def test_two_adjacent_cities_without_an_address_anchor_are_both_dropped():
    """A comma-separated city pair is still anchor-less — merging cities to each
    other must not manufacture an address out of nothing."""
    text = "Zagreb, Split"
    dets = _dets(text, [("Zagreb", ADDRESS_CITY), ("Split", ADDRESS_CITY)])
    assert _select(text, dets) == []


# ---------------------------------------------------------------------------
# MIXED
# ---------------------------------------------------------------------------


def test_9_city_next_to_a_person_still_passes_through():
    """Acceptance 9. DELIBERATE PRODUCT DECISION, not a bug: a city adjacent to a
    PERSON is NOT promoted. PERSON is not an address anchor, so 'Klanjec' stays
    in the prompt. Do not 'fix' this."""
    text = "Ivan Horvat, Klanjec"
    dets = _dets(text, [("Ivan Horvat", "PERSON"), ("Klanjec", ADDRESS_CITY)])
    out = _select(text, dets)
    assert _spans(out) == [("PERSON", 0, 11)]
    assert _masked(text, out) == ["Ivan Horvat"]


def test_10_person_plus_address_with_city():
    """Acceptance 10. The PERSON must not bridge into the address cluster."""
    text = "Ivan Horvat, Vukovarska 23, Zagreb"
    dets = _dets(
        text,
        [
            ("Ivan Horvat", "PERSON"),
            ("Vukovarska 23", "ADDRESS"),
            ("Zagreb", ADDRESS_CITY),
        ],
    )
    out = _select(text, dets)
    assert _spans(out) == [("ADDRESS", 13, 34), ("PERSON", 0, 11)]
    assert _masked(text, out) == ["Ivan Horvat", "Vukovarska 23, Zagreb"]


def test_person_does_not_bridge_two_address_clusters():
    """A PERSON sitting between an address and a city is not a separator the gap
    predicate can see (it reads raw text), but its letters block the gap anyway."""
    text = "Ilica 5, Ivan Horvat, Zagreb"
    dets = _dets(
        text,
        [("Ilica 5", "ADDRESS"), ("Ivan Horvat", "PERSON"), ("Zagreb", ADDRESS_CITY)],
    )
    out = _select(text, dets)
    assert _spans(out) == [("ADDRESS", 0, 7), ("PERSON", 9, 20)]


# ---------------------------------------------------------------------------
# REGRESSION guards
# ---------------------------------------------------------------------------


def test_12_date_time_still_dropped_by_the_whitelist():
    """Acceptance 12. DATE_TIME was removed from PRESIDIO_TO_STANDARD in v0.9.2
    (date substrings inside credit-card expiries). Step 0.5 must not resurrect
    it — 'thursday 23rd july' stays unmasked."""
    text = "trip to thurles on thursday 23rd july"
    dets = _dets(
        text, [("thurles", ADDRESS_CITY), ("thursday 23rd july", "DATE_TIME")]
    )
    assert _select(text, dets) == []


def test_plain_address_without_any_city_is_unchanged():
    """Regression: the no-city path must behave exactly as before."""
    text = "Ulica kneza Branimira 42"
    dets = _dets(text, [("Ulica kneza Branimira 42", "ADDRESS")])
    out = _select(text, dets)
    assert _spans(out) == [("ADDRESS", 0, 24)]
    assert out[0].score == pytest.approx(0.9)


def test_placeholder_overlap_filter_still_runs_first():
    """TRAU-522 Step 0 must stay ahead of Step 0.5: a city span overlapping an
    existing placeholder is dropped before it can anchor anything."""
    text = "[ADDRESS_1], Zagreb"
    dets = _dets(text, [("[ADDRESS_1]", "ADDRESS"), ("Zagreb", ADDRESS_CITY)])
    # The ADDRESS hit on the placeholder is dropped by Step 0, leaving the city
    # anchor-less -> discarded. Nothing survives, so the placeholder keeps its
    # number and vault parity holds.
    assert _select(text, dets) == []


def test_overlapping_city_and_address_collapse_to_one_span():
    """A city span nested inside an address span must not double-count."""
    text = "Vukovarska 23 Zagreb"
    # Built by hand: the nested city shares offsets with the enclosing address,
    # which the cursor-advancing `_dets` helper cannot express.
    dets = [
        RecognizerResult(entity_type="ADDRESS", start=0, end=20, score=0.9),
        RecognizerResult(entity_type=ADDRESS_CITY, start=14, end=20, score=0.9),
    ]
    out = _select(text, dets)
    assert _spans(out) == [("ADDRESS", 0, 20)]


# ---------------------------------------------------------------------------
# INVARIANT (acceptance 15) — not a feature test
# ---------------------------------------------------------------------------


_INVARIANT_CASES = [
    ("Zagreb", [("Zagreb", ADDRESS_CITY)]),
    ("Ilica 5, Zagreb", [("Ilica 5", "ADDRESS"), ("Zagreb", ADDRESS_CITY)]),
    ("Zagreb, Ilica 5", [("Zagreb", ADDRESS_CITY), ("Ilica 5", "ADDRESS")]),
    ("Zagreb, Split", [("Zagreb", ADDRESS_CITY), ("Split", ADDRESS_CITY)]),
    ("Zagreb Split", [("Zagreb", ADDRESS_CITY), ("Split", ADDRESS_CITY)]),
    ("Ivan Horvat, Zagreb", [("Ivan Horvat", "PERSON"), ("Zagreb", ADDRESS_CITY)]),
    ("[ADDRESS_1], Zagreb", [("[ADDRESS_1]", "ADDRESS"), ("Zagreb", ADDRESS_CITY)]),
    ("Zagreb Zagreb", [("Zagreb", ADDRESS_CITY), ("Zagreb", ADDRESS_CITY)]),
    (
        "Ilica 5, Zagreb, Split",
        [("Ilica 5", "ADDRESS"), ("Zagreb", ADDRESS_CITY), ("Split", ADDRESS_CITY)],
    ),
]


@pytest.mark.parametrize(
    "text,items", _INVARIANT_CASES, ids=[c[0][:20] for c in _INVARIANT_CASES]
)
def test_address_city_never_escapes_selection(text, items):
    """INVARIANT: `_select_accepted_detections` never returns ADDRESS_CITY, under
    any input. Every surviving span is either promoted to ADDRESS or discarded.

    This is the contract that lets the transient type exist at all: a leaked
    ADDRESS_CITY would be looked up in PRESIDIO_TO_STANDARD (pii_filter.py:3608)
    and minted as a literal `[ADDRESS_CITY_1]` placeholder written to the vault —
    silent corruption, not a crash, because `_PLACEHOLDER_RE` happily matches it.
    """
    for det in _select(text, _dets(text, items)):
        assert det.entity_type != ADDRESS_CITY, det


def test_address_city_never_escapes_as_raw_entity_type():
    """The same invariant one layer out: `_build_enriched_detection` copies the
    raw type into `metadata.pii_detections[*].raw_entity_type`
    (pii_filter.py:1070), which leaves the pipeline. Promotion therefore builds a
    NEW RecognizerResult rather than mutating in place, so no downstream consumer
    ever sees the transient type."""
    text = "Ilica 5, Zagreb"
    dets = _dets(text, [("Ilica 5", "ADDRESS"), ("Zagreb", ADDRESS_CITY)])
    for det in _select(text, dets):
        enriched = pii_mod._build_enriched_detection(
            det, text, "ADDRESS", text[det.start : det.end], "[ADDRESS_1]"
        )
        assert enriched["raw_entity_type"] != ADDRESS_CITY
        assert enriched["entity_type"] != ADDRESS_CITY


def test_transient_type_would_be_dropped_even_without_step_0_5():
    """Backstop proof: with Step 0.5 bypassed (no ADDRESS anchor to promote
    against), the Step 1 whitelist alone still removes the transient type. The
    invariant does not rest on a single branch."""
    text = "Zagreb"
    dets = [RecognizerResult(entity_type=ADDRESS_CITY, start=0, end=6, score=0.99)]
    assert _select(text, dets) == []


# ---------------------------------------------------------------------------
# E2E through inlet (the bug as the user hit it)
# ---------------------------------------------------------------------------


async def test_inlet_reported_message_reaches_the_llm_intact():
    """Acceptance 5, end to end: the message the LLM refused to answer must come
    out of `inlet` byte-identical, with no vault mints."""
    text = (
        "help me plan a coordinated bus and train trip from cahersiveen "
        "to thurles via farranfore on thursday 23rd july"
    )
    pipe = make_gliner_pipeline(
        name_spans={
            "cahersiveen": ADDRESS_CITY,
            "thurles": ADDRESS_CITY,
            "farranfore": ADDRESS_CITY,
        }
    )
    pipe.analyzer_hr = FakeAnalyzer({})

    out = await pipe.inlet(
        {"chat_id": "chat-trau-530", "messages": [{"role": "user", "content": text}]},
        user=user_payload(True),
    )

    assert out["messages"][0]["content"] == text
    assert out["metadata"]["pii_detections"] == []
    assert out["metadata"]["pii_placeholder_map"] == {}


async def test_inlet_real_address_with_city_masks_as_one_placeholder():
    """Acceptance 10, end to end: one [ADDRESS_1] covering street + city, plus
    the person — not [ADDRESS_1] [ADDRESS_2]."""
    text = "Ivan Horvat, Vukovarska 23, Zagreb"
    pipe = make_gliner_pipeline(
        name_spans={
            "Ivan Horvat": "PERSON",
            "Vukovarska 23": "ADDRESS",
            "Zagreb": ADDRESS_CITY,
        }
    )
    pipe.analyzer_hr = FakeAnalyzer({})

    out = await pipe.inlet(
        {"chat_id": "chat-trau-530-b", "messages": [{"role": "user", "content": text}]},
        user=user_payload(True),
    )
    content = out["messages"][0]["content"]

    assert content == "[PERSON_1], [ADDRESS_1]"
    assert "[ADDRESS_2]" not in content
    assert out["metadata"]["pii_reverse_map"]["[ADDRESS_1]"] == "Vukovarska 23, Zagreb"


async def test_inlet_city_beside_person_is_not_masked():
    """Acceptance 9, end to end: the deliberate product decision, pinned where a
    future change would actually break it."""
    text = "Ivan Horvat, Klanjec"
    pipe = make_gliner_pipeline(
        name_spans={"Ivan Horvat": "PERSON", "Klanjec": ADDRESS_CITY}
    )
    pipe.analyzer_hr = FakeAnalyzer({})

    out = await pipe.inlet(
        {"chat_id": "chat-trau-530-c", "messages": [{"role": "user", "content": text}]},
        user=user_payload(True),
    )

    assert out["messages"][0]["content"] == "[PERSON_1], Klanjec"
