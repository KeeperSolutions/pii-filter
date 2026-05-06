"""Unit tests for `mask_text` and inlet integration tests for masking.

`mask_text` is exercised directly with synthesized `RecognizerResult` instances
so the tests don't depend on the spaCy model. Inlet integration tests in this
module run a real analyzer pass using the module-scoped `started_pipeline`
fixture defined below.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from presidio_analyzer import RecognizerResult

from pii_filter import CUSTOM_ENTITY_TYPES, Pipeline, mask_text

# ---------------------------------------------------------------------------
# mask_text — unit tests
# ---------------------------------------------------------------------------

# Whitelist mirroring Pipeline.PRESIDIO_TO_STANDARD; kept local to make
# masking tests self-contained and easy to extend with synthetic types.
_STANDARD: dict[str, str] = {
    "PERSON": "PERSON",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "HR_OIB": "HR_OIB",
    "HR_IBAN": "HR_IBAN",
    "CREDIT_CARD": "CREDIT_CARD",
}


def _fresh_state() -> tuple[dict[str, int], dict[str, str], dict[str, str]]:
    return {}, {}, {}


def test_basic_single_entity() -> None:
    text = "Moj OIB je 12345678903"
    dets = [RecognizerResult(entity_type="HR_OIB", start=11, end=22, score=0.85)]
    counters, fwd, rev = _fresh_state()

    masked, enriched = mask_text(text, dets, _STANDARD, counters, fwd, rev)

    assert masked == "Moj OIB je [HR_OIB_1]"
    assert fwd == {"12345678903": "[HR_OIB_1]"}
    assert rev == {"[HR_OIB_1]": "12345678903"}
    assert counters == {"HR_OIB": 1}
    assert len(enriched) == 1
    assert enriched[0]["original"] == "12345678903"
    assert enriched[0]["placeholder"] == "[HR_OIB_1]"
    assert enriched[0]["entity_type"] == "HR_OIB"


def test_dedupe_same_value() -> None:
    """Same original value within one call gets one placeholder, counter at 1."""
    text = "OIB 12345678903 i opet 12345678903"
    dets = [
        RecognizerResult(entity_type="HR_OIB", start=4, end=15, score=0.9),
        RecognizerResult(entity_type="HR_OIB", start=23, end=34, score=0.9),
    ]
    counters, fwd, rev = _fresh_state()

    masked, enriched = mask_text(text, dets, _STANDARD, counters, fwd, rev)

    assert masked == "OIB [HR_OIB_1] i opet [HR_OIB_1]"
    assert counters == {"HR_OIB": 1}
    assert len(fwd) == 1
    assert len(enriched) == 2
    # Both enriched entries share the same placeholder.
    assert {e["placeholder"] for e in enriched} == {"[HR_OIB_1]"}


def test_distinct_values_same_type() -> None:
    text = "Ana Ivić i Marko Marić"
    dets = [
        RecognizerResult(entity_type="PERSON", start=0, end=8, score=0.85),
        RecognizerResult(entity_type="PERSON", start=11, end=22, score=0.85),
    ]
    counters, fwd, rev = _fresh_state()

    masked, enriched = mask_text(text, dets, _STANDARD, counters, fwd, rev)

    assert masked == "[PERSON_1] i [PERSON_2]"
    assert counters == {"PERSON": 2}
    assert fwd == {"Ana Ivić": "[PERSON_1]", "Marko Marić": "[PERSON_2]"}


def test_overlap_score_resolution() -> None:
    """Higher-score detection wins over a lower-score overlapping detection."""
    text = "OIB 12345678903"
    # Both detections cover the same span; HR_OIB has the higher score.
    dets = [
        RecognizerResult(entity_type="HR_OIB", start=4, end=15, score=0.9),
        RecognizerResult(entity_type="PHONE_NUMBER", start=4, end=15, score=0.4),
    ]
    counters, fwd, rev = _fresh_state()

    masked, enriched = mask_text(text, dets, _STANDARD, counters, fwd, rev)

    assert masked == "OIB [HR_OIB_1]"
    assert len(enriched) == 1
    assert enriched[0]["entity_type"] == "HR_OIB"
    # PHONE was discarded entirely.
    assert "PHONE" not in counters


def test_overlap_custom_wins_on_tie() -> None:
    """Equal score tie: a custom-recognizer entity beats a built-in."""
    text = "value 12345678903"
    dets = [
        # PERSON is a built-in; HR_OIB is in CUSTOM_ENTITY_TYPES.
        RecognizerResult(entity_type="PERSON", start=6, end=17, score=0.85),
        RecognizerResult(entity_type="HR_OIB", start=6, end=17, score=0.85),
    ]
    assert "HR_OIB" in CUSTOM_ENTITY_TYPES
    assert "PERSON" not in CUSTOM_ENTITY_TYPES
    counters, fwd, rev = _fresh_state()

    masked, enriched = mask_text(text, dets, _STANDARD, counters, fwd, rev)

    assert masked == "value [HR_OIB_1]"
    assert enriched[0]["entity_type"] == "HR_OIB"


def test_multiple_types() -> None:
    text = "Ivan, OIB 12345678903, IBAN HR1723600001101234565"
    dets = [
        RecognizerResult(entity_type="PERSON", start=0, end=4, score=0.85),
        RecognizerResult(entity_type="HR_OIB", start=10, end=21, score=0.9),
        RecognizerResult(entity_type="HR_IBAN", start=28, end=49, score=0.95),
    ]
    counters, fwd, rev = _fresh_state()

    masked, enriched = mask_text(text, dets, _STANDARD, counters, fwd, rev)

    assert masked == "[PERSON_1], OIB [HR_OIB_1], IBAN [HR_IBAN_1]"
    assert counters == {"PERSON": 1, "HR_OIB": 1, "HR_IBAN": 1}
    assert len(enriched) == 3


def test_non_canonical_filtered() -> None:
    """Detections whose entity_type is not in the whitelist are dropped silently."""
    text = "Microsoft Office is software"
    dets = [
        RecognizerResult(entity_type="ORGANIZATION", start=0, end=16, score=0.85),
    ]
    counters, fwd, rev = _fresh_state()

    masked, enriched = mask_text(text, dets, _STANDARD, counters, fwd, rev)

    assert masked == text
    assert enriched == []
    assert fwd == {}


def test_empty_detections() -> None:
    text = "Nothing sensitive here."
    counters, fwd, rev = _fresh_state()

    masked, enriched = mask_text(text, [], _STANDARD, counters, fwd, rev)

    assert masked == text
    assert enriched == []
    assert fwd == {}


def test_empty_text() -> None:
    counters, fwd, rev = _fresh_state()

    masked, enriched = mask_text("", [], _STANDARD, counters, fwd, rev)

    assert masked == ""
    assert enriched == []


def test_unicode_offsets_preserved() -> None:
    """Croatian non-ASCII chars must not corrupt offsets — Python str indexes
    code points, so as long as Presidio reports code-point offsets (which it
    does), masking is straightforward. Verify roundtrip on `čćšđž` text."""
    text = "Pozdrav čćšđž, Ana Ivić!"
    person_start = text.index("Ana Ivić")
    person_end = person_start + len("Ana Ivić")
    dets = [RecognizerResult(entity_type="PERSON", start=person_start, end=person_end, score=0.9)]
    counters, fwd, rev = _fresh_state()

    masked, enriched = mask_text(text, dets, _STANDARD, counters, fwd, rev)

    assert masked == "Pozdrav čćšđž, [PERSON_1]!"
    assert enriched[0]["original"] == "Ana Ivić"


def test_existing_state_preserved_across_calls() -> None:
    """Calling mask_text twice with shared state: second call reuses placeholders
    from the first and continues counters where they left off."""
    counters, fwd, rev = _fresh_state()

    text1 = "OIB 12345678903"
    dets1 = [RecognizerResult(entity_type="HR_OIB", start=4, end=15, score=0.9)]
    masked1, _ = mask_text(text1, dets1, _STANDARD, counters, fwd, rev)
    assert masked1 == "OIB [HR_OIB_1]"

    # Second call: same OIB reuses placeholder; new OIB gets _2.
    text2 = "Old 12345678903 New 23456789014"
    dets2 = [
        RecognizerResult(entity_type="HR_OIB", start=4, end=15, score=0.9),
        RecognizerResult(entity_type="HR_OIB", start=20, end=31, score=0.9),
    ]
    masked2, _ = mask_text(text2, dets2, _STANDARD, counters, fwd, rev)

    assert masked2 == "Old [HR_OIB_1] New [HR_OIB_2]"
    assert counters == {"HR_OIB": 2}


def test_zero_length_detection_skipped() -> None:
    """Defensive: a buggy recognizer reporting start == end (or start > end)
    must be discarded silently — never allocate a placeholder for the empty
    string nor inject one into the masked text. Hardening for future custom
    recognizers (Task 10 ADDRESS, Task 14 OPF)."""
    text = "Innocuous text"
    dets = [
        RecognizerResult(entity_type="HR_OIB", start=5, end=5, score=0.9),
        # Inverted span (start > end) is also nonsense; same treatment.
        RecognizerResult(entity_type="HR_OIB", start=8, end=4, score=0.9),
    ]
    counters, fwd, rev = _fresh_state()

    masked, enriched = mask_text(text, dets, _STANDARD, counters, fwd, rev)

    assert masked == text
    assert enriched == []
    assert fwd == {}
    assert rev == {}
    assert counters == {}


def test_three_overlapping_only_one_survives() -> None:
    """Three overlapping detections collapse to a single accepted entity."""
    text = "0123456789012"
    dets = [
        RecognizerResult(entity_type="HR_OIB", start=0, end=11, score=0.9),
        RecognizerResult(entity_type="HR_JMBG", start=0, end=13, score=0.95),
        RecognizerResult(entity_type="PERSON", start=2, end=8, score=0.5),
    ]
    standard = dict(_STANDARD)
    standard["HR_JMBG"] = "HR_JMBG"
    counters, fwd, rev = _fresh_state()

    masked, enriched = mask_text(text, dets, standard, counters, fwd, rev)

    # Highest score (HR_JMBG, 0.95) wins.
    assert masked == "[HR_JMBG_1]"
    assert len(enriched) == 1
    assert enriched[0]["entity_type"] == "HR_JMBG"


# ---------------------------------------------------------------------------
# Inlet — masking integration tests (use real analyzer)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(loop_scope="module", scope="module")
async def started_pipeline() -> AsyncIterator[Pipeline]:
    p = Pipeline()
    await p.on_startup()
    yield p
    await p.on_shutdown()


def _oib_check(first10: str) -> int:
    a = 10
    for d in first10:
        a = (a + int(d)) % 10
        if a == 0:
            a = 10
        a = (a * 2) % 11
    return (11 - a) % 10


def _make_oib(first10: str) -> str:
    return f"{first10}{_oib_check(first10)}"


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_masks_user_message(started_pipeline: Pipeline) -> None:
    oib = _make_oib("1234567890")
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": f"My OIB is {oib}"}],
        "metadata": {"chat_id": "abc-1"},
    }
    result = await started_pipeline.inlet(body)

    masked = result["messages"][-1]["content"]
    assert "[HR_OIB_1]" in masked
    assert oib not in masked
    fwd = result["metadata"]["pii_placeholder_map"]
    rev = result["metadata"]["pii_reverse_map"]
    assert fwd[oib] == "[HR_OIB_1]"
    assert rev["[HR_OIB_1]"] == oib


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_skips_when_last_is_assistant(started_pipeline: Pipeline) -> None:
    """If only assistant messages are present, inlet returns body unchanged."""
    body: dict[str, Any] = {
        "messages": [{"role": "assistant", "content": "Hello, how can I help?"}],
        "metadata": {},
    }
    original_content = body["messages"][-1]["content"]
    result = await started_pipeline.inlet(body)

    assert result["messages"][-1]["content"] == original_content
    assert "pii_detections" not in result["metadata"]
    assert "pii_placeholder_map" not in result["metadata"]


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_targets_last_user_message_in_history(started_pipeline: Pipeline) -> None:
    """In multi-turn history, masking applies to the most recent user message."""
    oib = _make_oib("1234567890")
    body: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": "First turn (no PII)"},
            {"role": "assistant", "content": "Got it."},
            {"role": "user", "content": f"Second turn — OIB {oib}"},
        ]
    }
    result = await started_pipeline.inlet(body)

    # First user message should be untouched.
    assert result["messages"][0]["content"] == "First turn (no PII)"
    # Last user message should be masked.
    last = result["messages"][-1]["content"]
    assert oib not in last
    assert "[HR_OIB_1]" in last


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_multimodal_content(started_pipeline: Pipeline) -> None:
    """Text parts are masked; non-text parts (image_url) are untouched."""
    oib = _make_oib("1234567890")
    image_part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,XYZ"}}
    text_part = {"type": "text", "text": f"Look at this — OIB {oib}"}
    body: dict[str, Any] = {"messages": [{"role": "user", "content": [text_part, image_part]}]}
    result = await started_pipeline.inlet(body)

    parts = result["messages"][-1]["content"]
    # Image part untouched.
    assert parts[1] is image_part
    assert parts[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,XYZ"}}
    # Text part masked in place.
    assert oib not in parts[0]["text"]
    assert "[HR_OIB_1]" in parts[0]["text"]


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_preserves_pii_detections_with_placeholder(
    started_pipeline: Pipeline,
) -> None:
    oib = _make_oib("1234567890")
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": f"OIB {oib}"}],
        "metadata": {"chat_id": "abc-2"},
    }
    result = await started_pipeline.inlet(body)

    detections = result["metadata"]["pii_detections"]
    assert detections, "expected at least one detection"
    sample = next(d for d in detections if d["entity_type"] == "HR_OIB")
    assert {
        "entity_type",
        "start",
        "end",
        "score",
        "raw_entity_type",
        "original",
        "placeholder",
    } <= sample.keys()
    assert sample["original"] == oib
    assert sample["placeholder"] == "[HR_OIB_1]"


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_reverse_map_is_inverse_of_forward(started_pipeline: Pipeline) -> None:
    oib = _make_oib("1234567890")
    body: dict[str, Any] = {"messages": [{"role": "user", "content": f"OIB {oib} again {oib}"}]}
    result = await started_pipeline.inlet(body)

    fwd = result["metadata"]["pii_placeholder_map"]
    rev = result["metadata"]["pii_reverse_map"]
    assert {ph: orig for orig, ph in fwd.items()} == rev


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_disabled_valve_skips_masking(started_pipeline: Pipeline) -> None:
    started_pipeline.valves.enabled = False
    try:
        oib = _make_oib("1234567890")
        body: dict[str, Any] = {"messages": [{"role": "user", "content": f"OIB {oib}"}]}
        result = await started_pipeline.inlet(body)
        # Content untouched, no metadata maps.
        assert result["messages"][-1]["content"] == f"OIB {oib}"
        assert "pii_placeholder_map" not in result.get("metadata", {})
    finally:
        started_pipeline.valves.enabled = True


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_masks_grouped_iban(started_pipeline: Pipeline) -> None:
    """End-to-end: an HR IBAN in ISO 13616 4-char grouped form must be
    detected by the recognizer regex, validated by the mod-97 check, and
    masked with a placeholder. This is the user-facing form banking apps
    display, so a miss here is a real PII leak."""
    grouped = "HR12 1001 0051 8630 0016 0"
    body: dict[str, Any] = {"messages": [{"role": "user", "content": f"Moj IBAN je {grouped}."}]}
    result = await started_pipeline.inlet(body)

    masked_text = result["messages"][-1]["content"]
    assert grouped not in masked_text, "raw IBAN leaked into masked text"
    assert "[HR_IBAN_1]" in masked_text

    fwd = result["metadata"]["pii_placeholder_map"]
    assert fwd[grouped] == "[HR_IBAN_1]"


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_does_not_mask_country_names(started_pipeline: Pipeline) -> None:
    """Country names detected as LOCATION must not be masked.

    Real ADDRESS recognition (street + number + postal code) is Task 10 scope.
    Until then LOCATION is intentionally not in PRESIDIO_TO_STANDARD, so
    spaCy/Presidio LOCATION hits are dropped silently before masking and
    country names like 'Hrvatska' / 'Njemačku' stay in the prompt the LLM sees.
    """
    original = "Pišem iz Hrvatske, putujem u Njemačku."
    body: dict[str, Any] = {"messages": [{"role": "user", "content": original}]}
    result = await started_pipeline.inlet(body)

    detections = result["metadata"].get("pii_detections", [])
    address_detections = [d for d in detections if d.get("entity_type") == "ADDRESS"]
    location_detections = [d for d in detections if d.get("entity_type") == "LOCATION"]
    assert address_detections == [], "ADDRESS detection out of Task 4 scope"
    assert location_detections == [], "LOCATION must not appear as canonical entity"

    assert result["messages"][-1]["content"] == original

    fwd = result["metadata"].get("pii_placeholder_map", {})
    assert "Hrvatska" not in fwd
    assert "Hrvatske" not in fwd
    assert "Njemačku" not in fwd


@pytest.mark.asyncio(loop_scope="module")
async def test_analyzer_no_misc_entity_in_raw_results(started_pipeline: Pipeline) -> None:
    """labels_to_ignore=['MISC','O'] must suppress the spaCy MISC label at
    the NER stage so it never reaches Presidio's entity mapper.

    Calling `inlet()` and inspecting `metadata.pii_detections` cannot prove
    this — those results are already filtered through PRESIDIO_TO_STANDARD
    (which doesn't include MISC), so MISC would be dropped regardless of the
    NLP-engine config. Hit the analyzer directly and inspect raw results so
    a regression in the `labels_to_ignore` config would actually fail the
    test instead of silently passing on the whitelist drop.
    """
    assert started_pipeline.analyzer is not None
    text = "Microsoft Office i Hrvatska Pošta su tvrtke u zagrebu."
    raw = started_pipeline.analyzer.analyze(text=text, language="hr")
    misc_or_o = [r for r in raw if r.entity_type in {"MISC", "O"}]
    assert misc_or_o == [], f"expected MISC/O suppressed at NER stage, got {misc_or_o!r}"


# ---------------------------------------------------------------------------
# Task 5 — inlet integration tests against the Redis thread vault
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_thread_consistency_across_requests(
    started_pipeline: Pipeline,
) -> None:
    """Same chat_id + same PII value across two inlet calls reuses the same
    placeholder. This is the core epic acceptance criterion for Task 5."""
    oib = _make_oib("1112223330")
    chat_id = "task5-consistency-thread"

    body1: dict[str, Any] = {
        "chat_id": chat_id,
        "messages": [{"role": "user", "content": f"OIB: {oib}"}],
    }
    result1 = await started_pipeline.inlet(body1)
    masked1 = result1["messages"][-1]["content"]
    fwd1 = result1["metadata"]["pii_placeholder_map"]
    placeholder_first = fwd1[oib]

    body2: dict[str, Any] = {
        "chat_id": chat_id,
        "messages": [{"role": "user", "content": f"Provjera istog OIB-a: {oib}"}],
    }
    result2 = await started_pipeline.inlet(body2)
    masked2 = result2["messages"][-1]["content"]
    fwd2 = result2["metadata"]["pii_placeholder_map"]

    assert placeholder_first in masked1
    assert placeholder_first in masked2
    assert fwd2[oib] == placeholder_first


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_cross_thread_isolation(started_pipeline: Pipeline) -> None:
    """Different chat_ids hold independent counters: same value yields the
    same numeric suffix in each thread but they are separate vault entries."""
    oib = _make_oib("2223334440")
    body_a: dict[str, Any] = {
        "chat_id": "task5-isolation-thread-A",
        "messages": [{"role": "user", "content": f"OIB: {oib}"}],
    }
    body_b: dict[str, Any] = {
        "chat_id": "task5-isolation-thread-B",
        "messages": [{"role": "user", "content": f"OIB: {oib}"}],
    }

    result_a = await started_pipeline.inlet(body_a)
    result_b = await started_pipeline.inlet(body_b)

    placeholder_a = result_a["metadata"]["pii_placeholder_map"][oib]
    placeholder_b = result_b["metadata"]["pii_placeholder_map"][oib]
    assert placeholder_a == "[HR_OIB_1]"
    assert placeholder_b == "[HR_OIB_1]"


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_chat_id_in_body_metadata_fallback(
    started_pipeline: Pipeline,
) -> None:
    """When body has no top-level `chat_id` but `metadata.chat_id` is set,
    the inlet uses the metadata value as the thread key."""
    oib = _make_oib("3334445550")
    chat_id = "task5-metadata-fallback"
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": f"OIB: {oib}"}],
        "metadata": {"chat_id": chat_id},
    }
    result = await started_pipeline.inlet(body)
    placeholder = result["metadata"]["pii_placeholder_map"][oib]

    body2: dict[str, Any] = {
        "chat_id": chat_id,
        "messages": [{"role": "user", "content": f"OIB ponovo: {oib}"}],
    }
    result2 = await started_pipeline.inlet(body2)
    assert result2["metadata"]["pii_placeholder_map"][oib] == placeholder


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_ephemeral_thread_when_chat_id_missing(
    started_pipeline: Pipeline,
) -> None:
    """A request without any chat_id still masks for the single turn but
    cannot share state with future requests (ephemeral thread)."""
    oib = _make_oib("4445556660")
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": f"OIB: {oib}"}],
    }
    result = await started_pipeline.inlet(body)

    fwd = result["metadata"]["pii_placeholder_map"]
    assert oib in fwd
    placeholder = fwd[oib]
    assert placeholder.startswith("[HR_OIB_")
    assert placeholder in result["messages"][-1]["content"]


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_writes_snapshot_to_body_metadata(
    started_pipeline: Pipeline,
) -> None:
    """Forward-compat hinge for Task 6: inlet must populate
    `body.metadata.pii_placeholder_map` and `pii_reverse_map` so the outlet
    can read them without depending on Redis directly."""
    oib = _make_oib("5556667770")
    body: dict[str, Any] = {
        "chat_id": "task5-snapshot",
        "messages": [{"role": "user", "content": f"Moj OIB je {oib}"}],
    }
    result = await started_pipeline.inlet(body)
    metadata = result["metadata"]

    assert "pii_placeholder_map" in metadata
    assert "pii_reverse_map" in metadata
    fwd = metadata["pii_placeholder_map"]
    rev = metadata["pii_reverse_map"]
    assert oib in fwd
    placeholder = fwd[oib]
    assert rev[placeholder] == oib


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_redis_down_block_mode(
    started_pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the vault healthcheck fails and `degradation_mode='block'` the
    inlet must raise, never letting the request through unmasked."""
    assert started_pipeline.vault is not None

    async def _unhealthy() -> bool:
        return False

    monkeypatch.setattr(started_pipeline.vault, "healthcheck", _unhealthy)
    assert started_pipeline.valves.degradation_mode == "block"

    oib = _make_oib("6667778880")
    body: dict[str, Any] = {
        "chat_id": "task5-block-mode",
        "messages": [{"role": "user", "content": f"OIB: {oib}"}],
    }
    with pytest.raises(RuntimeError, match="degradation_mode='block'"):
        await started_pipeline.inlet(body)


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_redis_down_passthrough_mode(
    started_pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With `degradation_mode='passthrough'` and a dead vault the inlet
    falls back to per-request dicts (Task 4 behavior); masking still
    happens and `body.metadata` snapshots are still populated."""
    assert started_pipeline.vault is not None

    async def _unhealthy() -> bool:
        return False

    monkeypatch.setattr(started_pipeline.vault, "healthcheck", _unhealthy)
    monkeypatch.setattr(started_pipeline.valves, "degradation_mode", "passthrough")

    oib = _make_oib("7778889990")
    body: dict[str, Any] = {
        "chat_id": "task5-passthrough-mode",
        "messages": [{"role": "user", "content": f"OIB: {oib}"}],
    }
    result = await started_pipeline.inlet(body)

    fwd = result["metadata"]["pii_placeholder_map"]
    rev = result["metadata"]["pii_reverse_map"]
    assert oib in fwd
    placeholder = fwd[oib]
    assert rev[placeholder] == oib
    assert placeholder in result["messages"][-1]["content"]


# ---------------------------------------------------------------------------
# UserValves stub
# ---------------------------------------------------------------------------


def test_user_valves_default() -> None:
    p = Pipeline()
    assert p.user_valves.pii_masking_enabled is True
