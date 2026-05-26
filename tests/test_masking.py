"""Unit tests for `mask_text` and inlet integration tests for masking.

`mask_text` is exercised directly with synthesized `RecognizerResult` instances
so the tests don't depend on the spaCy model. Inlet integration tests in this
module run a real analyzer pass using the module-scoped `started_pipeline`
fixture defined below.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Generator
from typing import Any

import pytest
import pytest_asyncio
from presidio_analyzer import RecognizerResult

from pii_filter import CUSTOM_ENTITY_TYPES, Pipeline, mask_text, restore_text
from tests.conftest import postgres_binary_missing
from tests.helpers.mock_vault import MockThreadVault


@pytest.fixture(autouse=True, scope="module")
def _swap_vault_to_mock() -> Generator[None, None, None]:
    """Replace `ThreadVault` with `MockThreadVault` for this module so
    `Pipeline.on_startup` wires an in-memory vault that does not require a
    running Postgres process. Real-Postgres integration coverage lives in
    `test_postgres_vault.py` (and post-Task-9, the renamed `test_thread_vault.py`).

    Also injects a dummy `PII_FILTER_POSTGRES_URL` so the `on_startup`
    DSN-empty guard doesn't trip — the mock ignores the value.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("pii_filter.ThreadVault", MockThreadVault)
        mp.setenv("PII_FILTER_POSTGRES_URL", "postgresql://mock-vault-dsn")
        yield


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
# restore_text — unit tests (Task 6)
# ---------------------------------------------------------------------------


def test_restore_text_basic_single_placeholder() -> None:
    text = "Vaš [HR_OIB_1] je validan."
    reverse_map = {"[HR_OIB_1]": "12345678903"}

    restored, restored_keys, hallucinated_keys = restore_text(text, reverse_map)

    assert restored == "Vaš 12345678903 je validan."
    assert restored_keys == ["[HR_OIB_1]"]
    assert hallucinated_keys == []


def test_restore_text_multiple_distinct_placeholders() -> None:
    text = "Bok [PERSON_1], OIB [HR_OIB_1] je tvoj."
    reverse_map = {"[PERSON_1]": "Ivan Horvat", "[HR_OIB_1]": "12345678903"}

    restored, restored_keys, hallucinated_keys = restore_text(text, reverse_map)

    assert restored == "Bok Ivan Horvat, OIB 12345678903 je tvoj."
    assert restored_keys == ["[HR_OIB_1]", "[PERSON_1]"]
    assert hallucinated_keys == []


def test_restore_text_same_placeholder_repeated() -> None:
    """Repeated placeholder substitutes everywhere but is recorded once."""
    text = "[PERSON_1] je rekao [PERSON_1], i opet [PERSON_1]."
    reverse_map = {"[PERSON_1]": "Ana"}

    restored, restored_keys, hallucinated_keys = restore_text(text, reverse_map)

    assert restored == "Ana je rekao Ana, i opet Ana."
    assert restored_keys == ["[PERSON_1]"]
    assert hallucinated_keys == []


def test_restore_text_hallucination_only() -> None:
    """Placeholder that the regex matches but reverse_map cannot resolve
    must remain literally in the text and surface in `hallucinated`."""
    text = "Nepoznati [PERSON_99] u odgovoru."
    reverse_map = {"[PERSON_1]": "Ivan"}

    restored, restored_keys, hallucinated_keys = restore_text(text, reverse_map)

    assert restored == text  # untouched
    assert restored_keys == []
    assert hallucinated_keys == ["[PERSON_99]"]


def test_restore_text_mixed_restored_and_hallucinated() -> None:
    text = "[PERSON_1] zna [HR_OIB_1] ali [PERSON_99] ne zna [HR_OIB_42]."
    reverse_map = {"[PERSON_1]": "Ivan", "[HR_OIB_1]": "12345678903"}

    restored, restored_keys, hallucinated_keys = restore_text(text, reverse_map)

    assert restored == "Ivan zna 12345678903 ali [PERSON_99] ne zna [HR_OIB_42]."
    assert restored_keys == ["[HR_OIB_1]", "[PERSON_1]"]
    assert hallucinated_keys == ["[HR_OIB_42]", "[PERSON_99]"]


def test_restore_text_empty_inputs() -> None:
    """Empty text → empty result; empty map → text unchanged with no records."""
    assert restore_text("", {}) == ("", [], [])
    assert restore_text("", {"[PERSON_1]": "Ivan"}) == ("", [], [])
    assert restore_text("Some text [PERSON_1]", {}) == ("Some text [PERSON_1]", [], [])
    assert restore_text("No placeholders here.", {"[PERSON_1]": "Ivan"}) == (
        "No placeholders here.",
        [],
        [],
    )


def test_restore_text_unicode_originals() -> None:
    """Croatian characters round-trip through restoration without corruption."""
    text = "Zovem se [PERSON_1] iz [PERSON_2]."
    reverse_map = {"[PERSON_1]": "Ana Ivić", "[PERSON_2]": "Đorđe Šljivančanin"}

    restored, restored_keys, _ = restore_text(text, reverse_map)

    assert restored == "Zovem se Ana Ivić iz Đorđe Šljivančanin."
    assert restored_keys == ["[PERSON_1]", "[PERSON_2]"]


def test_restore_text_original_contains_placeholder_shape() -> None:
    """Single-pass `re.sub` must NOT re-restore a placeholder-shaped substring
    that happens to live inside an original value.

    `str.replace` chains would loop again over already-substituted text and
    risk swapping the inner `[DOC_1]` token; `re.sub` with a callable is
    atomic and only runs once over the input.
    """
    text = "See [DOC_1] for [PERSON_1]."
    reverse_map = {
        "[DOC_1]": "the manual at [PERSON_1]",  # original mentions another placeholder shape
        "[PERSON_1]": "Ivan",
    }

    restored, restored_keys, _ = restore_text(text, reverse_map)

    # The first match `[DOC_1]` is replaced wholesale with its original; the
    # second match `[PERSON_1]` (the literal one in the input) is replaced
    # with "Ivan". The placeholder string baked into DOC_1's original is
    # NOT re-scanned, so it stays as the literal `[PERSON_1]` substring.
    assert restored == "See the manual at [PERSON_1] for Ivan."
    assert restored_keys == ["[DOC_1]", "[PERSON_1]"]


# ---------------------------------------------------------------------------
# Inlet — masking integration tests (use real analyzer)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(loop_scope="module", scope="module")
async def started_pipeline() -> AsyncIterator[Pipeline]:
    # Vault construction uses the module-level `_swap_vault_to_mock` autouse
    # fixture, which substitutes `MockThreadVault` for `ThreadVault`
    # so the analyzer-heavy tests in this module never need a real Postgres.
    p = Pipeline()
    # Use default languages (["hr", "en"]) so EN integration tests in this
    # module exercise the English analyzer path when en_core_web_lg is available.
    # Tests that require the EN model guard with `if p.analyzer_en is None:
    # pytest.skip(...)` — that guard still handles environments where the
    # model is not installed.
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
    assert started_pipeline.analyzer_hr is not None
    text = "Microsoft Office i Hrvatska Pošta su tvrtke u zagrebu."
    raw = started_pipeline.analyzer_hr.analyze(text=text, language="hr")
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
async def test_inlet_vault_down_block_mode(
    started_pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the vault healthcheck fails and `degradation_mode='block'` the
    inlet must raise, never letting the request through unmasked.

    Flips `MockThreadVault.force_unhealthy = True` to simulate a downed
    vault — backend-agnostic per the post-Task-9 single-backend contract.
    """
    vault = started_pipeline.vault
    assert vault is not None
    assert isinstance(vault, MockThreadVault)

    monkeypatch.setattr(vault, "force_unhealthy", True)
    assert started_pipeline.valves.degradation_mode == "block"

    oib = _make_oib("6667778880")
    body: dict[str, Any] = {
        "chat_id": "task9-block-mode",
        "messages": [{"role": "user", "content": f"OIB: {oib}"}],
    }
    with pytest.raises(RuntimeError, match="degradation_mode='block'"):
        await started_pipeline.inlet(body)


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_vault_down_passthrough_mode(
    started_pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With `degradation_mode='passthrough'` and a dead vault the inlet
    falls back to per-request dicts (Task 4 behavior); masking still
    happens and `body.metadata` snapshots are still populated.
    """
    vault = started_pipeline.vault
    assert vault is not None
    assert isinstance(vault, MockThreadVault)

    monkeypatch.setattr(vault, "force_unhealthy", True)
    monkeypatch.setattr(started_pipeline.valves, "degradation_mode", "passthrough")

    oib = _make_oib("7778889990")
    body: dict[str, Any] = {
        "chat_id": "task9-passthrough-mode",
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
# Outlet — placeholder restoration integration tests (Task 6)
# ---------------------------------------------------------------------------


def _outlet_body_with_choices(
    content: Any,
    *,
    reverse_map: dict[str, str] | None,
    chat_id: str = "task6-outlet",
    include_metadata: bool = True,
    extra_message_keys: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an OpenAI-shaped non-streaming completion body for outlet tests.

    `reverse_map=None` omits the `pii_reverse_map` key entirely (covers the
    "missing key" no-op path); pass `{}` to cover the "empty map" no-op
    path. Set `include_metadata=False` to omit the metadata key entirely.
    """
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if extra_message_keys:
        message.update(extra_message_keys)
    body: dict[str, Any] = {
        "chat_id": chat_id,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
    }
    if include_metadata:
        metadata: dict[str, Any] = {"chat_id": chat_id}
        if reverse_map is not None:
            metadata["pii_reverse_map"] = reverse_map
        body["metadata"] = metadata
    return body


@pytest.mark.asyncio(loop_scope="module")
async def test_outlet_happy_path_str_content_after_inlet(
    started_pipeline: Pipeline,
) -> None:
    """Full round-trip: inlet masks PII into placeholders, outlet restores
    the originals when the simulated LLM response keeps the placeholders.

    This is the headline acceptance criterion: end-to-end the user must
    see the original PII value in the assistant response.
    """
    oib = _make_oib("8889990010")
    chat_id = "task6-roundtrip-str"

    # Step 1 — inlet masks the user message.
    inlet_body: dict[str, Any] = {
        "chat_id": chat_id,
        "messages": [{"role": "user", "content": f"Moj OIB je {oib}"}],
    }
    inlet_result = await started_pipeline.inlet(inlet_body)
    placeholder = inlet_result["metadata"]["pii_placeholder_map"][oib]
    reverse_map = inlet_result["metadata"]["pii_reverse_map"]
    assert placeholder in inlet_result["messages"][-1]["content"]

    # Step 2 — simulate the LLM echoing the placeholder back, then outlet.
    outlet_body = _outlet_body_with_choices(
        f"Vaš {placeholder} je validan.",
        reverse_map=reverse_map,
        chat_id=chat_id,
    )
    outlet_result = await started_pipeline.outlet(outlet_body)

    final_content = outlet_result["choices"][0]["message"]["content"]
    assert final_content == f"Vaš {oib} je validan."
    assert placeholder not in final_content


@pytest.mark.asyncio(loop_scope="module")
async def test_outlet_happy_path_multimodal_content(
    started_pipeline: Pipeline,
) -> None:
    """Multi-modal `list[dict]` content: every text part is restored
    independently; non-text parts (image_url) are untouched."""
    reverse_map = {"[PERSON_1]": "Ivan", "[HR_OIB_1]": "12345678903"}
    image_part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,XYZ"}}
    parts = [
        {"type": "text", "text": "Pozdrav [PERSON_1]."},
        image_part,
        {"type": "text", "text": "OIB [HR_OIB_1] je validan."},
    ]
    body = _outlet_body_with_choices(parts, reverse_map=reverse_map)

    result = await started_pipeline.outlet(body)
    out_parts = result["choices"][0]["message"]["content"]

    assert out_parts[0]["text"] == "Pozdrav Ivan."
    assert out_parts[1] is image_part
    assert out_parts[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,XYZ"}}
    assert out_parts[2]["text"] == "OIB 12345678903 je validan."


@pytest.mark.asyncio(loop_scope="module")
async def test_outlet_empty_reverse_map_is_noop(
    started_pipeline: Pipeline, caplog: pytest.LogCaptureFixture
) -> None:
    """Empty `pii_reverse_map` triggers the DEBUG no-op path; body is
    returned unchanged and no INFO/WARN log lines are emitted."""
    body = _outlet_body_with_choices("[PERSON_1] je nepoznat.", reverse_map={})
    snapshot_content = body["choices"][0]["message"]["content"]

    with caplog.at_level("DEBUG", logger="pii_filter"):
        result = await started_pipeline.outlet(body)

    assert result is body
    assert result["choices"][0]["message"]["content"] == snapshot_content
    # Empty body metadata triggers vault fallback (post-merge bugfix #2);
    # the per-test mock vault has no entry for this chat_id, so the
    # snapshot returns empty and outlet bails out at the DEBUG no-op path.
    assert any(
        ("vault snapshot empty" in rec.message) or ("pii_reverse_map missing" in rec.message)
        for rec in caplog.records
        if rec.levelname == "DEBUG"
    )
    # No INFO summary, no WARN hallucination line for an empty-map no-op.
    assert not any(
        rec.levelname == "INFO" and "outlet processed" in rec.message for rec in caplog.records
    )


@pytest.mark.asyncio(loop_scope="module")
async def test_outlet_missing_reverse_map_key_is_noop(
    started_pipeline: Pipeline,
) -> None:
    """Metadata exists but lacks `pii_reverse_map` — outlet is a no-op."""
    body = _outlet_body_with_choices("[PERSON_1] je tu.", reverse_map=None)
    # metadata is present but `pii_reverse_map` is omitted.
    assert "pii_reverse_map" not in body["metadata"]

    result = await started_pipeline.outlet(body)

    assert result is body
    assert result["choices"][0]["message"]["content"] == "[PERSON_1] je tu."


@pytest.mark.asyncio(loop_scope="module")
async def test_outlet_missing_metadata_is_noop(started_pipeline: Pipeline) -> None:
    """Body has no `metadata` key at all — outlet is a no-op."""
    body = _outlet_body_with_choices("[PERSON_1] je tu.", reverse_map=None, include_metadata=False)
    assert "metadata" not in body

    result = await started_pipeline.outlet(body)

    assert result is body
    assert result["choices"][0]["message"]["content"] == "[PERSON_1] je tu."


@pytest.mark.asyncio(loop_scope="module")
async def test_outlet_hallucination_only_logs_warn_and_keeps_text(
    started_pipeline: Pipeline, caplog: pytest.LogCaptureFixture
) -> None:
    """Hallucinated placeholder (in text but not in reverse_map) is left
    literally in the response and surfaces as a single WARN line."""
    reverse_map = {"[PERSON_1]": "Ivan"}
    body = _outlet_body_with_choices(
        "Tajanstveni [PERSON_99] u odgovoru.",
        reverse_map=reverse_map,
        chat_id="task6-hallucination",
    )

    with caplog.at_level("WARNING", logger="pii_filter"):
        result = await started_pipeline.outlet(body)

    assert result["choices"][0]["message"]["content"] == "Tajanstveni [PERSON_99] u odgovoru."
    warn_records = [
        r for r in caplog.records if r.levelname == "WARNING" and "hallucinations" in r.message
    ]
    assert len(warn_records) == 1
    rec = warn_records[0]
    assert "task6-hallucination" in rec.getMessage()
    assert "count=1" in rec.getMessage()
    assert "[PERSON_99]" in rec.getMessage()


@pytest.mark.asyncio(loop_scope="module")
async def test_outlet_mixed_restored_and_hallucinated(
    started_pipeline: Pipeline, caplog: pytest.LogCaptureFixture
) -> None:
    """Mixed bag: restored placeholders are substituted; hallucinations
    are left literal; both lists are accounted for in logs."""
    reverse_map = {"[PERSON_1]": "Ivan", "[HR_OIB_1]": "12345678903"}
    body = _outlet_body_with_choices(
        "[PERSON_1] zna [HR_OIB_1] ali [PERSON_99] ne zna [HR_OIB_42].",
        reverse_map=reverse_map,
        chat_id="task6-mixed",
    )

    with caplog.at_level("INFO", logger="pii_filter"):
        result = await started_pipeline.outlet(body)

    assert (
        result["choices"][0]["message"]["content"]
        == "Ivan zna 12345678903 ali [PERSON_99] ne zna [HR_OIB_42]."
    )
    info_recs = [r for r in caplog.records if "outlet processed" in r.message]
    assert info_recs, "expected an INFO summary line"
    assert "placeholders_restored=2" in info_recs[-1].getMessage()
    assert "hallucinations=2" in info_recs[-1].getMessage()
    warn_recs = [r for r in caplog.records if "hallucinations detected" in r.message]
    assert len(warn_recs) == 1
    msg = warn_recs[0].getMessage()
    assert "[HR_OIB_42]" in msg and "[PERSON_99]" in msg


@pytest.mark.asyncio(loop_scope="module")
async def test_outlet_tool_calls_response_is_noop(
    started_pipeline: Pipeline, caplog: pytest.LogCaptureFixture
) -> None:
    """Tool-calling LLM response: `message.content is None`, `tool_calls`
    populated. Outlet skips at DEBUG, body untouched."""
    reverse_map = {"[PERSON_1]": "Ivan"}
    body = _outlet_body_with_choices(
        None,
        reverse_map=reverse_map,
        extra_message_keys={
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "x"}}],
        },
    )

    with caplog.at_level("DEBUG", logger="pii_filter"):
        result = await started_pipeline.outlet(body)

    # Body unchanged; the assistant message still has content=None.
    assert result["choices"][0]["message"]["content"] is None
    assert any(
        "tool_calls" in rec.message or "not str|list" in rec.message
        for rec in caplog.records
        if rec.levelname == "DEBUG"
    )


@pytest.mark.asyncio(loop_scope="module")
async def test_outlet_streaming_delta_chunk_is_noop(
    started_pipeline: Pipeline, caplog: pytest.LogCaptureFixture
) -> None:
    """Streaming chunk shape (`choices[0].delta`, no `message`): outlet
    skips at DEBUG and returns body unchanged."""
    body: dict[str, Any] = {
        "chat_id": "task6-streaming",
        "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": "[PERSON_1] kaže..."}},
        ],
        "metadata": {
            "chat_id": "task6-streaming",
            "pii_reverse_map": {"[PERSON_1]": "Ivan"},
        },
    }
    snapshot = body["choices"][0]["delta"]["content"]

    with caplog.at_level("DEBUG", logger="pii_filter"):
        result = await started_pipeline.outlet(body)

    assert result["choices"][0]["delta"]["content"] == snapshot
    assert any(
        "delta" in rec.message and "streaming" in rec.message
        for rec in caplog.records
        if rec.levelname == "DEBUG"
    )


@pytest.mark.asyncio(loop_scope="module")
async def test_outlet_swallows_unexpected_exception_and_logs_error(
    started_pipeline: Pipeline,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Decision 5 + Task 6 follow-up split: an *unexpected* internal error
    (anything other than KeyError/AttributeError/TypeError) is caught,
    logged at ERROR with `exc_info=True`, and the body is returned
    unchanged. This is the programmer-error path — it should be loud in
    production observability so somebody triages it."""
    import pii_filter as pii_filter_module

    def _boom(*_args: object, **_kwargs: object) -> tuple[str, list[str], list[str]]:
        raise RuntimeError("simulated restore_text failure")

    monkeypatch.setattr(pii_filter_module, "restore_text", _boom)

    reverse_map = {"[PERSON_1]": "Ivan"}
    body = _outlet_body_with_choices(
        "[PERSON_1] je tu.",
        reverse_map=reverse_map,
        chat_id="task6-boom",
    )
    snapshot = body["choices"][0]["message"]["content"]

    with caplog.at_level("ERROR", logger="pii_filter"):
        result = await started_pipeline.outlet(body)

    assert result is body
    # Restoration failed → text untouched (the exception fires on the first
    # text part, no partial mutation possible in this scenario).
    assert result["choices"][0]["message"]["content"] == snapshot
    error_recs = [r for r in caplog.records if r.levelname == "ERROR" and "UNEXPECTED" in r.message]
    assert len(error_recs) == 1
    rec = error_recs[0]
    assert "RuntimeError" in rec.getMessage()
    assert "task6-boom" in rec.getMessage()
    # `exc_info=True` must propagate the traceback to the log record.
    assert rec.exc_info is not None, "ERROR record should carry exc_info for tracebacks"
    assert rec.exc_info[0] is RuntimeError


@pytest.mark.asyncio(loop_scope="module")
async def test_outlet_swallows_expected_exception_and_logs_warn(
    started_pipeline: Pipeline,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Decision 5 + Task 6 follow-up split: the *expected* fail-safe family
    (KeyError, AttributeError, TypeError) is caught, logged at WARN, and
    the body is returned unchanged. These are operational, not bugs — a
    malformed body that slipped past the defensive shape guard should not
    page someone."""

    def _bad_iter(_self: Pipeline, _message: dict[str, Any]) -> Any:
        raise TypeError("simulated malformed body")

    monkeypatch.setattr(Pipeline, "_iter_text_parts", _bad_iter)

    reverse_map = {"[PERSON_1]": "Ivan"}
    body = _outlet_body_with_choices(
        "[PERSON_1] je tu.",
        reverse_map=reverse_map,
        chat_id="task6-expected-fail",
    )
    snapshot = body["choices"][0]["message"]["content"]

    with caplog.at_level("DEBUG", logger="pii_filter"):
        result = await started_pipeline.outlet(body)

    assert result is body
    assert result["choices"][0]["message"]["content"] == snapshot
    warn_recs = [
        r for r in caplog.records if r.levelname == "WARNING" and "expected fail-safe" in r.message
    ]
    assert len(warn_recs) == 1
    msg = warn_recs[0].getMessage()
    assert "TypeError" in msg
    assert "task6-expected-fail" in msg
    # Must NOT have escalated to ERROR for the expected family.
    assert not any(r.levelname == "ERROR" and "UNEXPECTED" in r.message for r in caplog.records)


@pytest.mark.asyncio(loop_scope="module")
async def test_outlet_empty_list_content_logs_specific_debug_message(
    started_pipeline: Pipeline, caplog: pytest.LogCaptureFixture
) -> None:
    """Empty `list[dict]` content (`message.content = []`) hits its own
    dedicated DEBUG line ("empty list") rather than the generic catch-all,
    so log readers can distinguish "multi-modal with no parts" from "type
    other than str|list" (e.g. None for tool_calls)."""
    reverse_map = {"[PERSON_1]": "Ivan"}
    body = _outlet_body_with_choices([], reverse_map=reverse_map)

    with caplog.at_level("DEBUG", logger="pii_filter"):
        result = await started_pipeline.outlet(body)

    # Body unchanged; no-op path.
    assert result is body
    assert result["choices"][0]["message"]["content"] == []
    # The specific empty-list message must be present.
    assert any(
        "message.content is empty list" in r.message
        for r in caplog.records
        if r.levelname == "DEBUG"
    )
    # The generic catch-all must NOT have fired (would be misleading here).
    assert not any("is not str|list" in r.message for r in caplog.records if r.levelname == "DEBUG")


@pytest.mark.asyncio(loop_scope="module")
async def test_outlet_messages_fallback_shape(started_pipeline: Pipeline) -> None:
    """Legacy/test body using `messages[-1]` (no `choices` key) is honored
    when the last message is an assistant message."""
    reverse_map = {"[PERSON_1]": "Ivan", "[HR_OIB_1]": "12345678903"}
    body: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": "OIB je 12345678903"},
            {"role": "assistant", "content": "Bok [PERSON_1], OIB [HR_OIB_1] je tvoj."},
        ],
        "metadata": {"chat_id": "task6-msgs", "pii_reverse_map": reverse_map},
    }

    result = await started_pipeline.outlet(body)

    assert result["messages"][-1]["content"] == "Bok Ivan, OIB 12345678903 je tvoj."


@pytest.mark.asyncio(loop_scope="module")
async def test_outlet_messages_fallback_skips_when_last_is_user(
    started_pipeline: Pipeline,
) -> None:
    """Fallback shape with last message as user role is rejected — outlet
    only restores assistant turns."""
    reverse_map = {"[PERSON_1]": "Ivan"}
    body: dict[str, Any] = {
        "messages": [
            {"role": "assistant", "content": "Bok [PERSON_1]."},
            {"role": "user", "content": "[PERSON_1] te zove."},
        ],
        "metadata": {"chat_id": "task6-user-last", "pii_reverse_map": reverse_map},
    }
    snapshot_user = body["messages"][-1]["content"]
    snapshot_assistant = body["messages"][0]["content"]

    result = await started_pipeline.outlet(body)

    assert result["messages"][-1]["content"] == snapshot_user
    assert result["messages"][0]["content"] == snapshot_assistant


@pytest.mark.asyncio(loop_scope="module")
async def test_outlet_disabled_valve_skips_restoration(
    started_pipeline: Pipeline,
) -> None:
    """When `valves.enabled=False`, outlet returns body unchanged without
    ever consulting metadata."""
    started_pipeline.valves.enabled = False
    try:
        reverse_map = {"[PERSON_1]": "Ivan"}
        body = _outlet_body_with_choices("[PERSON_1] je tu.", reverse_map=reverse_map)
        result = await started_pipeline.outlet(body)
        assert result["choices"][0]["message"]["content"] == "[PERSON_1] je tu."
    finally:
        started_pipeline.valves.enabled = True


@pytest.mark.asyncio(loop_scope="module")
async def test_outlet_chat_id_appears_in_info_log(
    started_pipeline: Pipeline, caplog: pytest.LogCaptureFixture
) -> None:
    """The summary INFO line includes the chat_id and the count fields
    required by spec §2.1.10."""
    reverse_map = {"[PERSON_1]": "Ivan"}
    body = _outlet_body_with_choices(
        "[PERSON_1] je tu.",
        reverse_map=reverse_map,
        chat_id="task6-info-log",
    )

    with caplog.at_level("INFO", logger="pii_filter"):
        await started_pipeline.outlet(body)

    info_recs = [r for r in caplog.records if "outlet processed" in r.message]
    assert info_recs, "expected an INFO summary line"
    msg = info_recs[-1].getMessage()
    assert "chat_id=task6-info-log" in msg
    assert "placeholders_restored=1" in msg
    assert "hallucinations=0" in msg


# ---------------------------------------------------------------------------
# UserValves stub
# ---------------------------------------------------------------------------


def test_user_valves_default() -> None:
    p = Pipeline()
    assert p.user_valves.pii_masking_enabled is True


# ---------------------------------------------------------------------------
# Task 5.1 — inlet integration tests against the Postgres thread vault
# ---------------------------------------------------------------------------
#
# These tests require a `pg_ctl` / `postgres` binary on PATH (provided by
# `pytest-postgresql`). The whole section skips cleanly when the binary is
# absent. The `started_pipeline_postgres` fixture lives in `conftest.py` and
# is module-scoped so spaCy loads once for the whole section.


_postgres_skip = pytest.mark.skipif(
    postgres_binary_missing,
    reason="pg_ctl / postgres binary not on PATH; skipping Postgres-backed tests",
)


@_postgres_skip
@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_postgres_backend_thread_consistency_across_requests(
    started_pipeline_postgres: Pipeline,
) -> None:
    """Same chat_id + same PII value across two inlet calls reuses the same
    placeholder when the Postgres backend is the source of truth — Task 5.1
    AC mirrors the Task 5 Redis equivalent."""
    oib = _make_oib("9990001110")
    chat_id = "task5.1-pg-consistency-thread"

    body1: dict[str, Any] = {
        "chat_id": chat_id,
        "messages": [{"role": "user", "content": f"OIB: {oib}"}],
    }
    result1 = await started_pipeline_postgres.inlet(body1)
    masked1 = result1["messages"][-1]["content"]
    placeholder_first = result1["metadata"]["pii_placeholder_map"][oib]

    body2: dict[str, Any] = {
        "chat_id": chat_id,
        "messages": [{"role": "user", "content": f"Provjera istog OIB-a: {oib}"}],
    }
    result2 = await started_pipeline_postgres.inlet(body2)
    masked2 = result2["messages"][-1]["content"]
    placeholder_second = result2["metadata"]["pii_placeholder_map"][oib]

    assert placeholder_first in masked1
    assert placeholder_first in masked2
    assert placeholder_second == placeholder_first


@_postgres_skip
@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_postgres_backend_cross_thread_isolation(
    started_pipeline_postgres: Pipeline,
) -> None:
    """Different chat_ids hold independent counters: same value yields the
    same numeric suffix in each thread but they are separate Postgres rows."""
    oib = _make_oib("8889990000")
    body_a: dict[str, Any] = {
        "chat_id": "task5.1-pg-isolation-A",
        "messages": [{"role": "user", "content": f"OIB: {oib}"}],
    }
    body_b: dict[str, Any] = {
        "chat_id": "task5.1-pg-isolation-B",
        "messages": [{"role": "user", "content": f"OIB: {oib}"}],
    }

    result_a = await started_pipeline_postgres.inlet(body_a)
    result_b = await started_pipeline_postgres.inlet(body_b)

    placeholder_a = result_a["metadata"]["pii_placeholder_map"][oib]
    placeholder_b = result_b["metadata"]["pii_placeholder_map"][oib]
    assert placeholder_a == "[HR_OIB_1]"
    assert placeholder_b == "[HR_OIB_1]"


@_postgres_skip
@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_postgres_backend_chat_id_in_body_metadata_fallback(
    started_pipeline_postgres: Pipeline,
) -> None:
    """`body["metadata"]["chat_id"]` resolves through the Postgres path the
    same way it does through Redis — the chat_id resolver lives in the
    Pipeline class, not the vault."""
    oib = _make_oib("7778889990")
    chat_id = "task5.1-pg-metadata-fallback"
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": f"OIB: {oib}"}],
        "metadata": {"chat_id": chat_id},
    }
    result = await started_pipeline_postgres.inlet(body)
    placeholder = result["metadata"]["pii_placeholder_map"][oib]

    body2: dict[str, Any] = {
        "chat_id": chat_id,
        "messages": [{"role": "user", "content": f"OIB ponovo: {oib}"}],
    }
    result2 = await started_pipeline_postgres.inlet(body2)
    assert result2["metadata"]["pii_placeholder_map"][oib] == placeholder


@_postgres_skip
@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_postgres_backend_writes_snapshot_to_body_metadata(
    started_pipeline_postgres: Pipeline,
) -> None:
    """Task 6 outlet contract preservation — the SINGLE most important
    Task 5.1 regression guard. After Postgres-backend inlet, both
    `pii_placeholder_map` and `pii_reverse_map` MUST be `dict[str, str]`
    populated identically to what the Redis backend writes."""
    oib = _make_oib("6667778880")
    body: dict[str, Any] = {
        "chat_id": "task5.1-pg-snapshot",
        "messages": [{"role": "user", "content": f"Moj OIB je {oib}"}],
    }
    result = await started_pipeline_postgres.inlet(body)
    metadata = result["metadata"]

    assert "pii_placeholder_map" in metadata
    assert "pii_reverse_map" in metadata
    fwd = metadata["pii_placeholder_map"]
    rev = metadata["pii_reverse_map"]
    assert isinstance(fwd, dict)
    assert isinstance(rev, dict)
    # Both maps are dict[str, str].
    for k, v in fwd.items():
        assert isinstance(k, str) and isinstance(v, str)
    for k, v in rev.items():
        assert isinstance(k, str) and isinstance(v, str)
    # Round-trip: reverse is the inverse of forward.
    assert oib in fwd
    placeholder = fwd[oib]
    assert rev[placeholder] == oib


@_postgres_skip
@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_postgres_backend_with_vault_enabled_false_falls_back_to_per_request(
    started_pipeline_postgres: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bugfix regression: with the ThreadVault wired up but
    `vault_enabled=False`, the inlet must bypass the vault entirely and
    fall back to per-request dicts (Task 4 behavior). Verifies the global
    `vault_enabled` kill switch is the sole gate on the vault path."""
    assert started_pipeline_postgres.vault is not None

    async def _spy_get_placeholder(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("vault.get_placeholder must not be called when vault_enabled=False")

    monkeypatch.setattr(started_pipeline_postgres.vault, "get_placeholder", _spy_get_placeholder)
    monkeypatch.setattr(started_pipeline_postgres.valves, "vault_enabled", False)

    oib = _make_oib("7770001110")
    body: dict[str, Any] = {
        "chat_id": "task5.1-pg-vault-disabled",
        "messages": [{"role": "user", "content": f"OIB: {oib}"}],
    }
    result = await started_pipeline_postgres.inlet(body)

    fwd = result["metadata"]["pii_placeholder_map"]
    rev = result["metadata"]["pii_reverse_map"]
    assert oib in fwd
    placeholder = fwd[oib]
    assert rev[placeholder] == oib
    assert placeholder in result["messages"][-1]["content"]


@_postgres_skip
@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_postgres_backend_with_vault_enabled_true_uses_vault(
    started_pipeline_postgres: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bugfix regression (positive case): with `vault_enabled=True` and
    the ThreadVault wired up, the inlet must call
    `vault.get_placeholder` for every detected entity — proving the
    vault path is reached."""
    assert started_pipeline_postgres.vault is not None
    assert started_pipeline_postgres.valves.vault_enabled is True

    real_get_placeholder = started_pipeline_postgres.vault.get_placeholder
    call_count = 0

    async def _spy_get_placeholder(*args: Any, **kwargs: Any) -> str:
        nonlocal call_count
        call_count += 1
        return await real_get_placeholder(*args, **kwargs)

    monkeypatch.setattr(started_pipeline_postgres.vault, "get_placeholder", _spy_get_placeholder)

    oib = _make_oib("6660001110")
    body: dict[str, Any] = {
        "chat_id": "task5.1-pg-vault-enabled",
        "messages": [{"role": "user", "content": f"OIB: {oib}"}],
    }
    result = await started_pipeline_postgres.inlet(body)

    fwd = result["metadata"]["pii_placeholder_map"]
    assert oib in fwd
    assert call_count >= 1


# ---------------------------------------------------------------------------
# Task 3.1 — Inlet integration tests (deny-list, OIB phone-context)
#
# These tests use the module-scoped `started_pipeline` (mock vault)
# fixture so spaCy loads once for the whole module.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_filters_out_common_english_keywords(
    started_pipeline: Pipeline,
) -> None:
    """PERSON detections whose text is in the default deny-list must be dropped.

    spaCy hr_core_news_lg misclassifies common English/code keywords as PERSON
    at score 0.850 (confirmed per Q2 diagnostic). The deny-list in the default
    Valves must suppress them so they never reach the vault or counter.
    """
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": "Task: please summarize. JSON output expected."}],
        "metadata": {"chat_id": "task3.1-deny-list"},
    }
    result = await started_pipeline.inlet(body)
    detections = result["metadata"].get("pii_detections", [])
    leaked = [d["original"] for d in detections if d["entity_type"] == "PERSON"]
    assert leaked == [], f"Deny-list failed: {leaked!r} leaked as PERSON detections"


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_preserves_real_pii_with_strict_recognizer(
    started_pipeline: Pipeline,
) -> None:
    """Real PII (OIB, email) must still be detected after Task 3.1 changes.
    Regression guard for spec AC 3.5 / 3.6."""
    oib = _make_oib("1234567890")  # "12345678903"
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": f"Moj OIB je {oib} i email ivan@example.com"}],
        "metadata": {"chat_id": "task3.1-real-pii"},
    }
    result = await started_pipeline.inlet(body)
    detections = result["metadata"].get("pii_detections", [])
    assert any(
        d["entity_type"] == "HR_OIB" and d["original"] == oib for d in detections
    ), f"Valid OIB {oib} not detected after 3.1 changes"


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_rejects_phone_as_oib(
    started_pipeline: Pipeline,
) -> None:
    """'15551234567' has a valid OIB checksum (spec Q1 collision) but appears
    immediately after the phone keyword — must not be stored as HR_OIB."""
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": "My phone: 15551234567 please call."}],
        "metadata": {"chat_id": "task3.1-phone-oib"},
    }
    result = await started_pipeline.inlet(body)
    detections = result["metadata"].get("pii_detections", [])
    oib_hits = [
        d for d in detections if d["entity_type"] == "HR_OIB" and d["original"] == "15551234567"
    ]
    assert oib_hits == [], f"Phone number 15551234567 incorrectly classified as HR_OIB: {oib_hits}"


# ===========================================================================
# Task 8.5 — Multi-Turn History Scope Filter tests
# ===========================================================================


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_masks_all_user_messages_in_history(
    started_pipeline: Pipeline,
) -> None:
    """All user messages in a multi-turn history must be masked, not just the last one."""
    oib1 = _make_oib("1111111111")
    oib2 = _make_oib("2222222222")
    oib3 = _make_oib("3333333333")
    body: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": f"OIB: {oib1}"},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": f"Also: {oib2}"},
            {"role": "assistant", "content": "Got it."},
            {"role": "user", "content": f"And: {oib3}"},
        ],
        "metadata": {"chat_id": "task8.5-all-messages"},
    }
    result = await started_pipeline.inlet(body)

    assert oib1 not in result["messages"][0]["content"]
    assert oib2 not in result["messages"][2]["content"]
    assert oib3 not in result["messages"][4]["content"]
    assert "[HR_OIB_" in result["messages"][0]["content"]
    assert "[HR_OIB_" in result["messages"][2]["content"]
    assert "[HR_OIB_" in result["messages"][4]["content"]


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_passes_through_assistant_messages(
    started_pipeline: Pipeline,
) -> None:
    """Assistant messages must pass through inlet unchanged."""
    oib = _make_oib("9876543210")
    assistant_content = f"For reference, the OIB is {oib}."
    body: dict[str, Any] = {
        "messages": [
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": "What was that OIB?"},
        ],
        "metadata": {"chat_id": "task8.5-assistant-passthrough"},
    }
    result = await started_pipeline.inlet(body)

    assert result["messages"][0]["content"] == assistant_content


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_skips_already_masked_message_via_regex(
    started_pipeline: Pipeline,
) -> None:
    """A user message containing a placeholder pattern must be skipped (no re-analysis)."""
    oib = _make_oib("1234509876")
    body: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": "My OIB is [HR_OIB_1] from last time."},
            {"role": "user", "content": f"New message: {oib}"},
        ],
        "metadata": {"chat_id": "task8.5-skip-masked"},
    }
    result = await started_pipeline.inlet(body)

    assert result["messages"][0]["content"] == "My OIB is [HR_OIB_1] from last time."
    assert oib not in result["messages"][1]["content"]
    assert "[HR_OIB_" in result["messages"][1]["content"]


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_processes_oldest_first(
    started_pipeline: Pipeline,
) -> None:
    """First user message gets the lowest counter index; later messages get higher ones."""
    oib1 = _make_oib("1000000001")
    oib2 = _make_oib("2000000002")
    body: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": f"First: {oib1}"},
            {"role": "user", "content": f"Second: {oib2}"},
        ],
        "metadata": {"chat_id": "task8.5-oldest-first"},
    }
    result = await started_pipeline.inlet(body)

    assert "[HR_OIB_1]" in result["messages"][0]["content"]
    assert "[HR_OIB_2]" in result["messages"][1]["content"]


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_cross_message_same_pii_gets_same_placeholder(
    started_pipeline: Pipeline,
) -> None:
    """The same PII value appearing in different history positions yields the same placeholder."""
    oib = _make_oib("5555555555")
    body: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": f"My OIB: {oib}"},
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": f"Confirm OIB: {oib}"},
        ],
        "metadata": {"chat_id": "task8.5-same-pii"},
    }
    result = await started_pipeline.inlet(body)

    assert "[HR_OIB_1]" in result["messages"][0]["content"]
    assert "[HR_OIB_1]" in result["messages"][2]["content"]


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_respects_max_messages_cap(
    started_pipeline: Pipeline,
) -> None:
    """Only the last N user messages are processed; older ones pass through unchanged."""
    oib_old = _make_oib("0000000001")
    oib_new = _make_oib("9999999999")
    body: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": f"Old: {oib_old}"},
            {"role": "assistant", "content": "Response."},
            {"role": "user", "content": f"New: {oib_new}"},
        ],
        "metadata": {"chat_id": "task8.5-cap-test"},
    }
    started_pipeline.valves.multi_turn_history_max_messages = 1
    try:
        result = await started_pipeline.inlet(body)
    finally:
        started_pipeline.valves.multi_turn_history_max_messages = 20

    assert oib_old in result["messages"][0]["content"], "Old message (beyond cap) must be unchanged"
    assert (
        oib_new not in result["messages"][2]["content"]
    ), "New message (within cap) must be masked"


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_cap_zero_disables_history_processing(
    started_pipeline: Pipeline,
) -> None:
    """cap=0 means all history is beyond cap — behaves like Task 4 (only last user message)."""
    oib_early = _make_oib("1111000001")
    oib_late = _make_oib("2222000002")
    body: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": f"Early: {oib_early}"},
            {"role": "user", "content": f"Late: {oib_late}"},
        ],
        "metadata": {"chat_id": "task8.5-cap-zero"},
    }
    started_pipeline.valves.multi_turn_history_max_messages = 0
    try:
        result = await started_pipeline.inlet(body)
    finally:
        started_pipeline.valves.multi_turn_history_max_messages = 20

    assert oib_early in result["messages"][0]["content"], "Early message must be unchanged (cap=0)"
    assert oib_late not in result["messages"][1]["content"], "Late message must be masked"


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_multi_turn_disabled_falls_back_to_task4_behavior(
    started_pipeline: Pipeline,
) -> None:
    """multi_turn_history_scope=False reverts to Task 4: only last user message is masked."""
    oib_early = _make_oib("3333000003")
    oib_late = _make_oib("4444000004")
    body: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": f"Early: {oib_early}"},
            {"role": "user", "content": f"Late: {oib_late}"},
        ],
        "metadata": {"chat_id": "task8.5-scope-false"},
    }
    started_pipeline.valves.multi_turn_history_scope = False
    try:
        result = await started_pipeline.inlet(body)
    finally:
        started_pipeline.valves.multi_turn_history_scope = True

    assert (
        oib_early in result["messages"][0]["content"]
    ), "Early message must be unchanged (scope=False)"
    assert oib_late not in result["messages"][1]["content"], "Late message must be masked"


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_already_masked_message_does_not_call_analyzer(
    started_pipeline: Pipeline,
) -> None:
    """Presidio analyzer must not be invoked for messages that already contain placeholders."""
    from unittest.mock import MagicMock

    already_masked = "Kontaktiraj [PERSON_1] o [HR_OIB_1]."
    fresh = "A fresh message with no PII."
    body: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": already_masked},
            {"role": "user", "content": fresh},
        ],
        "metadata": {"chat_id": "task8.5-mock-analyzer"},
    }
    real_analyzer = started_pipeline.analyzer_hr
    mock_analyzer = MagicMock()
    mock_analyzer.analyze.return_value = []
    started_pipeline.analyzer_hr = mock_analyzer
    try:
        result = await started_pipeline.inlet(body)
    finally:
        started_pipeline.analyzer_hr = real_analyzer

    assert (
        mock_analyzer.analyze.call_count == 1
    ), f"Expected 1 HR analyzer call (fresh message only), got {mock_analyzer.analyze.call_count}"
    assert result["messages"][0]["content"] == already_masked


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_partial_placeholder_message_is_skipped(
    started_pipeline: Pipeline,
) -> None:
    """A message with both a placeholder AND raw PII is treated as already-masked (known limitation)."""
    oib = _make_oib("7777777777")
    mixed_content = f"Moj OIB je {oib}, a [PERSON_1] mi je kolega."
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": mixed_content}],
        "metadata": {"chat_id": "task8.5-partial-skip"},
    }
    result = await started_pipeline.inlet(body)

    assert (
        result["messages"][0]["content"] == mixed_content
    ), "Mixed message (placeholder + raw PII) must be left unchanged as a documented limitation"


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_pii_detections_include_message_index(
    started_pipeline: Pipeline,
) -> None:
    """Each detection record in pii_detections must carry a message_index field."""
    oib0 = _make_oib("1234000001")
    oib2 = _make_oib("5678000002")
    body: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": f"OIB: {oib0}"},
            {"role": "assistant", "content": "Got it."},
            {"role": "user", "content": f"OIB: {oib2}"},
        ],
        "metadata": {"chat_id": "task8.5-message-index"},
    }
    result = await started_pipeline.inlet(body)

    detections = result["metadata"].get("pii_detections", [])
    assert len(detections) >= 2
    for det in detections:
        assert "message_index" in det, f"Detection missing message_index field: {det}"
    indices = {det["message_index"] for det in detections}
    assert 0 in indices
    assert 2 in indices


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_metadata_logger_reports_processed_and_skipped_counts(
    started_pipeline: Pipeline,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The inlet info log line must include messages_processed and messages_skipped_already_masked."""
    import logging

    body: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": "Already masked [HR_OIB_1]."},
            {"role": "user", "content": "Fresh message with no PII."},
        ],
        "metadata": {"chat_id": "task8.5-log-counts"},
    }
    with caplog.at_level(logging.INFO, logger="pii_filter"):
        await started_pipeline.inlet(body)

    log_line = next(
        (r.message for r in caplog.records if "messages_processed=" in r.message),
        None,
    )
    assert log_line is not None, "Expected log line with 'messages_processed=' not found"
    assert "messages_skipped_already_masked=" in log_line
    assert "messages_processed=1" in log_line
    assert "messages_skipped_already_masked=1" in log_line


# --- Integration tests with vault ---


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_multi_turn_with_vault_consistency(
    started_pipeline: Pipeline,
) -> None:
    """Same OIB in message 0 and message 4 of a multi-turn history gets the same placeholder."""
    oib = _make_oib("8888800001")
    body: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": f"Moj OIB je {oib}."},
            {"role": "assistant", "content": "Razumijem."},
            {"role": "user", "content": "Mozesh li potvrditi?"},
            {"role": "assistant", "content": "Da."},
            {"role": "user", "content": f"Potvrdi: {oib}"},
        ],
        "metadata": {"chat_id": "task8.5-vault-consistency"},
    }
    result = await started_pipeline.inlet(body)

    msg0 = result["messages"][0]["content"]
    msg4 = result["messages"][4]["content"]
    assert oib not in msg0
    assert oib not in msg4
    assert "[HR_OIB_1]" in msg0
    assert "[HR_OIB_1]" in msg4


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_edit_previous_message_re_mask_is_consistent(
    started_pipeline: Pipeline,
) -> None:
    """Editing a message (re-sending full history with changed content) preserves the OIB placeholder."""
    oib = _make_oib("1122334455")
    chat_id = "task8.5-edit-history"

    body1: dict[str, Any] = {
        "messages": [{"role": "user", "content": f"OIB je {oib}."}],
        "metadata": {"chat_id": chat_id},
    }
    result1 = await started_pipeline.inlet(body1)
    assert result1["metadata"]["pii_placeholder_map"].get(oib) == "[HR_OIB_1]"

    body2: dict[str, Any] = {
        "messages": [{"role": "user", "content": f"OIB je {oib} i ime mi je Ivan."}],
        "metadata": {"chat_id": chat_id},
    }
    result2 = await started_pipeline.inlet(body2)

    assert oib not in result2["messages"][0]["content"]
    assert "[HR_OIB_1]" in result2["messages"][0]["content"]


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_long_conversation_with_already_masked_history(
    started_pipeline: Pipeline,
) -> None:
    """20-message conversation where 19 are already masked: only last is freshly analyzed."""
    oib_new = _make_oib("6677889900")
    already_masked_msgs: list[dict[str, Any]] = [
        {"role": "user", "content": f"Turn {i}: [HR_OIB_1] and [PERSON_1]."} for i in range(19)
    ]
    already_masked_msgs.append({"role": "user", "content": f"New turn: {oib_new}"})
    body: dict[str, Any] = {
        "messages": already_masked_msgs,
        "metadata": {"chat_id": "task8.5-long-mostly-masked"},
    }
    result = await started_pipeline.inlet(body)

    for i in range(19):
        assert result["messages"][i]["content"] == f"Turn {i}: [HR_OIB_1] and [PERSON_1]."
    assert oib_new not in result["messages"][19]["content"]
    assert "[HR_OIB_" in result["messages"][19]["content"]


# --- Performance test ---


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.benchmark
async def test_inlet_multi_turn_p95_latency_under_200ms(
    started_pipeline: Pipeline,
) -> None:
    """P95 inlet latency must be under 200ms for 20-message conversations (AC 8.5.8).

    19 already-masked messages are skipped via regex pre-check; only the last
    message triggers Presidio -- simulating steady-state production traffic.
    """
    import time

    oib_fresh = _make_oib("9988776655")
    already_masked_msgs: list[dict[str, Any]] = [
        {"role": "user", "content": f"Turn {i}: [HR_OIB_1] and [PERSON_1]."} for i in range(19)
    ]

    latencies_ms: list[float] = []
    for _ in range(100):
        fresh_msg: dict[str, Any] = {"role": "user", "content": f"Fresh: {oib_fresh}"}
        body: dict[str, Any] = {
            "messages": already_masked_msgs + [fresh_msg],
            "metadata": {"chat_id": "task8.5-perf"},
        }
        t0 = time.perf_counter()
        await started_pipeline.inlet(body)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    latencies_ms.sort()
    p95_ms = latencies_ms[94]
    assert p95_ms < 200.0, f"P95 latency {p95_ms:.1f}ms exceeds 200ms target"


# ---------------------------------------------------------------------------
# Task 3.2 — Multi-Language Detection integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_detects_us_ssn_in_english_text(started_pipeline: Pipeline) -> None:
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": "My SSN is 123-45-6789"}],
        "metadata": {"chat_id": "task3.2-us-ssn"},
    }
    result = await started_pipeline.inlet(body)
    content = result["messages"][0]["content"]
    assert "[US_SSN_1]" in content, f"Expected US_SSN placeholder, got: {content!r}"


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_detects_uk_nhs_in_english_text(started_pipeline: Pipeline) -> None:
    if started_pipeline.analyzer_en is None:
        pytest.skip("en_core_web_lg not installed")
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": "My NHS number is 943 476 5919"}],
        "metadata": {"chat_id": "task3.2-uk-nhs"},
    }
    result = await started_pipeline.inlet(body)
    content = result["messages"][0]["content"]
    assert "[UK_NHS_1]" in content, f"Expected UK_NHS placeholder, got: {content!r}"


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_detects_oib_in_english_chat(started_pipeline: Pipeline) -> None:
    oib = _make_oib("1234567893")
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": f"My colleague's OIB is {oib}"}],
        "metadata": {"chat_id": "task3.2-oib-en"},
    }
    result = await started_pipeline.inlet(body)
    content = result["messages"][0]["content"]
    assert (
        "[HR_OIB_1]" in content
    ), f"Expected HR_OIB placeholder (custom recognizer in EN registry), got: {content!r}"


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_detects_person_in_english_text(started_pipeline: Pipeline) -> None:
    if started_pipeline.analyzer_en is None:
        pytest.skip("en_core_web_lg not installed")
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": "My name is John Smith"}],
        "metadata": {"chat_id": "task3.2-person-en"},
    }
    result = await started_pipeline.inlet(body)
    content = result["messages"][0]["content"]
    assert (
        "[PERSON_1]" in content
    ), f"Expected PERSON placeholder via en_core_web_lg NER, got: {content!r}"


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_mixed_hr_en_text(started_pipeline: Pipeline) -> None:
    oib = _make_oib("9876543211")
    text = f"Hi, my OIB is {oib} but my colleague's SSN is 234-56-7890"
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": text}],
        "metadata": {"chat_id": "task3.2-mixed"},
    }
    result = await started_pipeline.inlet(body)
    content = result["messages"][0]["content"]
    assert "[HR_OIB_1]" in content, f"Expected HR_OIB placeholder in mixed text, got: {content!r}"
    assert "[US_SSN_1]" in content, f"Expected US_SSN placeholder in mixed text, got: {content!r}"


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_email_in_english_context(started_pipeline: Pipeline) -> None:
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": "Email me at john@example.com please"}],
        "metadata": {"chat_id": "task3.2-email"},
    }
    result = await started_pipeline.inlet(body)
    content = result["messages"][0]["content"]
    assert "[EMAIL_1]" in content, f"Expected EMAIL placeholder, got: {content!r}"


@pytest.mark.asyncio
async def test_inlet_with_hr_only_languages_skips_en_analyzer() -> None:
    p = Pipeline()
    p.valves.languages = ["hr"]
    await p.on_startup()
    try:
        assert p.analyzer_hr is not None
        assert p.analyzer_en is None, "EN analyzer must be None in HR-only mode"
        oib = _make_oib("1111122222")
        body: dict[str, Any] = {
            "messages": [{"role": "user", "content": f"SSN: 234-56-7890, OIB: {oib}"}],
            "metadata": {"chat_id": "task3.2-hr-only"},
        }
        result = await p.inlet(body)
        content = result["messages"][0]["content"]
        assert "[HR_OIB_1]" in content, f"OIB must still be detected in HR-only mode: {content!r}"
        assert (
            "[US_SSN_1]" in content
        ), f"SSN must be detected via custom recognizer in HR registry: {content!r}"
    finally:
        await p.on_shutdown()


@pytest.mark.asyncio
async def test_inlet_with_en_only_languages_skips_hr_analyzer() -> None:
    p = Pipeline()
    p.valves.languages = ["en"]
    try:
        await p.on_startup()
    except RuntimeError as exc:
        pytest.skip(f"en_core_web_lg not available: {exc}")
    try:
        assert p.analyzer_hr is None, "HR analyzer must be None in EN-only mode"
        assert p.analyzer_en is not None
        oib = _make_oib("2222233333")
        body: dict[str, Any] = {
            "messages": [{"role": "user", "content": f"OIB: {oib}"}],
            "metadata": {"chat_id": "task3.2-en-only"},
        }
        result = await p.inlet(body)
        content = result["messages"][0]["content"]
        assert (
            "[HR_OIB_1]" in content
        ), f"OIB must be detected via custom recognizer duplicated in EN registry: {content!r}"
    finally:
        await p.on_shutdown()


@pytest.mark.benchmark
@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_dual_analyzer_latency_within_budget(started_pipeline: Pipeline) -> None:
    import time

    oib = _make_oib("5555566666")
    already_masked_msgs = [
        {"role": "user", "content": f"[PERSON_{i}] mentioned OIB [HR_OIB_{i}]"}
        for i in range(1, 10)
    ]
    latencies_ms: list[float] = []
    for i in range(20):
        fresh_msg = {"role": "user", "content": f"New message {i}: SSN is 234-56-7890 OIB {oib}"}
        body: dict[str, Any] = {
            "messages": already_masked_msgs + [fresh_msg],
            "metadata": {"chat_id": f"task3.2-latency-{i}"},
        }
        t0 = time.perf_counter()
        await started_pipeline.inlet(body)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    latencies_ms.sort()
    p95_ms = latencies_ms[18]
    assert p95_ms < 1000.0, f"Dual-analyzer P95 latency {p95_ms:.1f}ms exceeds 1000ms budget"


# ---------------------------------------------------------------------------
# Task 3.3 — Integration tests: NER spillover filter + deny-list expansion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inlet_hr_baseline_no_label_word_false_positives(
    started_pipeline: Pipeline,
) -> None:
    """HR text: deny-list catches PERSON false positives on Croatian label words."""
    oib = _make_oib("1234567890")
    body: dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": f"Moj OIB je {oib}, a zovem se Ivan Horvat. Email mi je ivan@firma.hr.",
            }
        ],
        "metadata": {"chat_id": "task3.3-hr-baseline"},
    }
    result = await started_pipeline.inlet(body)
    content = result["messages"][0]["content"]

    detections = result.get("metadata", {}).get("pii_detections", [])
    detected_originals = {d.get("original", "") for d in detections}

    assert (
        "Moj OIB" not in detected_originals
    ), f"'Moj OIB' must not enter vault — caught by deny-list; detections={detected_originals}"
    assert (
        "Email" not in detected_originals
    ), f"'Email' must not enter vault — caught by deny-list; detections={detected_originals}"
    assert "[HR_OIB_1]" in content, f"HR_OIB must be detected: {content!r}"


@pytest.mark.asyncio
async def test_inlet_en_baseline_person_detected(started_pipeline: Pipeline) -> None:
    """EN text: PERSON and US_SSN must both be detected when EN analyzer is active."""
    if started_pipeline.analyzer_en is None:
        pytest.skip("en_core_web_lg not installed — skipping EN baseline integration test")
    body: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": "My name is John Smith and my SSN is 123-45-6789."}
        ],
        "metadata": {"chat_id": "task3.3-en-baseline"},
    }
    result = await started_pipeline.inlet(body)
    content = result["messages"][0]["content"]
    detections = result.get("metadata", {}).get("pii_detections", [])
    entity_types = {d.get("entity_type") for d in detections}
    assert "PERSON" in entity_types, f"PERSON must be detected in EN text; content={content!r}"
    assert "US_SSN" in entity_types, f"US_SSN must be detected in EN text; content={content!r}"


@pytest.mark.asyncio
async def test_inlet_cross_language_oib_in_english_sentence(
    started_pipeline: Pipeline,
) -> None:
    """Cross-language: HR_OIB (regex entity) detected regardless of window language."""
    oib = _make_oib("9876543210")
    body: dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": f"My Croatian colleague's OIB is {oib}.",
            }
        ],
        "metadata": {"chat_id": "task3.3-cross-lang"},
    }
    result = await started_pipeline.inlet(body)
    detections = result.get("metadata", {}).get("pii_detections", [])
    entity_types = {d.get("entity_type") for d in detections}
    assert "HR_OIB" in entity_types, (
        f"HR_OIB (regex entity) must be detected even in EN-dominant text; "
        f"detections={detections}"
    )


@pytest.mark.asyncio
async def test_inlet_logger_includes_ner_spillover_count(
    started_pipeline: Pipeline,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Logger line must include ner_spillover_dropped field (AC 3.3.13)."""
    import logging

    oib = _make_oib("1111122222")
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": f"Moj OIB je {oib}."}],
        "metadata": {"chat_id": "task3.3-logger"},
    }
    with caplog.at_level(logging.INFO, logger="pii_filter"):
        await started_pipeline.inlet(body)

    log_text = "\n".join(caplog.messages)
    assert (
        "ner_spillover_dropped=" in log_text
    ), f"Logger must include ner_spillover_dropped field (AC 3.3.13); log={log_text!r}"


@pytest.mark.asyncio
async def test_inlet_skips_openwebui_background_task_title_generation(
    started_pipeline: Pipeline,
) -> None:
    """OpenWebUI background tasks must skip inlet to avoid false positives
    on embedded chat history."""
    body = {
        "messages": [
            {
                "role": "user",
                "content": "### Task: Generate title for: USER: My OIB is 12345678903",
            }
        ],
        "metadata": {
            "chat_id": "test-bg-task",
            "task": "title_generation",
        },
    }
    result = await started_pipeline.inlet(body)
    # Body returned unchanged — no metadata.pii_detections added
    assert "pii_detections" not in result.get("metadata", {})
    # Content unchanged (no masking applied)
    assert "12345678903" in result["messages"][0]["content"]


@pytest.mark.asyncio
async def test_inlet_processes_normal_user_message_without_task_field(
    started_pipeline: Pipeline,
) -> None:
    """Normal user messages (no metadata.task) must be processed normally."""
    body = {
        "messages": [{"role": "user", "content": "Moj OIB je 12345678903"}],
        "metadata": {
            "chat_id": "test-normal",
            # No "task" key
        },
    }
    result = await started_pipeline.inlet(body)
    # PII detected and masked
    assert "12345678903" not in result["messages"][0]["content"]
    assert "[HR_OIB_1]" in result["messages"][0]["content"]


@pytest.mark.asyncio
async def test_inlet_no_date_detection_in_credit_card_substring(
    started_pipeline: Pipeline,
) -> None:
    """DATE_TIME should NOT be detected as substring of CREDIT_CARD
    or PHONE entities. v0.9.2 removed DATE_TIME from PRESIDIO_TO_STANDARD."""
    body = {
        "messages": [
            {
                "role": "user",
                "content": "My card is 4111-1111-1111-1111 expiring 12/27.",
            }
        ],
        "metadata": {"chat_id": "test-no-date"},
    }
    result = await started_pipeline.inlet(body)
    detections = result["metadata"]["pii_detections"]
    entity_types = {d["entity_type"] for d in detections}
    # CREDIT_CARD detected; DATE absolutely not
    assert "CREDIT_CARD" in entity_types
    assert "DATE" not in entity_types
    assert "DATE_TIME" not in entity_types


@pytest.mark.asyncio
async def test_inlet_phone_detected_alongside_credit_card(
    started_pipeline: Pipeline,
) -> None:
    """PHONE detection must work even when text contains CREDIT_CARD
    sharing digit sequences. v0.9.2 fix."""
    body = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "My card is 4111-1111-1111-1111 and " "call me at +1 202 456 1111 anytime."
                ),
            }
        ],
        "metadata": {"chat_id": "test-phone-cc-overlap"},
    }
    result = await started_pipeline.inlet(body)
    detections = result["metadata"]["pii_detections"]
    entity_types_with_values = {(d["entity_type"], d["original"]) for d in detections}
    assert ("CREDIT_CARD", "4111-1111-1111-1111") in entity_types_with_values
    assert ("PHONE", "+1 202 456 1111") in entity_types_with_values


# ---------------------------------------------------------------------------
# Task 8 — UserValves wiring + Valves.presidio_enabled (integration)
# ---------------------------------------------------------------------------
#
# These tests share the module-scoped `started_pipeline`, so each one is
# careful to restore the toggles to their defaults in a `try/finally` block.


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_user_disabled_full_path(
    started_pipeline: Pipeline, caplog: pytest.LogCaptureFixture
) -> None:
    """End-to-end: user opt-out → raw PII delivered to LLM, no metadata
    written, no vault entries created for this chat_id (AC 8.2, 8.3).
    """
    oib = _make_oib("1234567890")
    chat_id = "ch_user_disabled_full"
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": f"Moj OIB je {oib}"}],
        "metadata": {"chat_id": chat_id},
    }
    try:
        started_pipeline.user_valves = Pipeline.UserValves(pii_masking_enabled=False)
        with caplog.at_level(logging.INFO, logger="pii_filter"):
            result = await started_pipeline.inlet(body, user={"id": "user_42"})

        # Raw OIB still in the body — analyzer did not run.
        assert oib in result["messages"][-1]["content"]
        # No detection metadata added.
        assert "pii_placeholder_map" not in result["metadata"]
        assert "pii_reverse_map" not in result["metadata"]
        assert "pii_detections" not in result["metadata"]
        # Vault not touched — no entries exist for this chat_id.
        assert started_pipeline.vault is not None
        forward, reverse = await started_pipeline.vault.snapshot_for_request(chat_id)
        assert forward == {}
        assert reverse == {}
        # Audit log emitted.
        user_disabled_logs = [r for r in caplog.records if "user_disabled" in r.getMessage()]
        assert len(user_disabled_logs) == 1
        assert f"chat_id={chat_id}" in user_disabled_logs[0].getMessage()
    finally:
        started_pipeline.user_valves = Pipeline.UserValves()


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_presidio_disabled_full_path(
    started_pipeline: Pipeline, caplog: pytest.LogCaptureFixture
) -> None:
    """End-to-end: admin presidio kill → raw PII delivered to LLM, but
    metadata keys are present (empty when no prior vault state) and the
    `presidio_disabled` audit line is emitted (AC 8.8).
    """
    oib = _make_oib("2223334440")
    chat_id = "ch_presidio_disabled_full"
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": f"Moj OIB je {oib}"}],
        "metadata": {"chat_id": chat_id},
    }
    try:
        started_pipeline.valves.presidio_enabled = False
        with caplog.at_level(logging.INFO, logger="pii_filter"):
            result = await started_pipeline.inlet(body, user={"id": "user_42"})

        # Raw OIB still in the body — analyzer skipped.
        assert oib in result["messages"][-1]["content"]
        # Metadata keys present for outlet symmetry, but empty (no prior state).
        assert result["metadata"]["pii_placeholder_map"] == {}
        assert result["metadata"]["pii_reverse_map"] == {}
        # presidio_disabled audit + summary lines emitted.
        messages = [r.getMessage() for r in caplog.records]
        assert any(f"presidio_disabled: chat_id={chat_id}" in m for m in messages)
        assert any("presidio_disabled=True" in m for m in messages)
    finally:
        started_pipeline.valves.presidio_enabled = True


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_presidio_disabled_still_writes_vault_snapshot(
    started_pipeline: Pipeline,
) -> None:
    """Pre-existing vault entries (from an earlier normal-path turn) must
    surface in metadata when `presidio_enabled` is later flipped off, so
    outlet can restore history placeholders (decision #4 / §2.1, AC 8.8).
    """
    oib = _make_oib("3334445550")
    chat_id = "ch_presidio_snapshot_carry"

    # Turn 1: normal path populates vault.
    body_turn1: dict[str, Any] = {
        "messages": [{"role": "user", "content": f"Moj OIB je {oib}"}],
        "metadata": {"chat_id": chat_id},
    }
    result_turn1 = await started_pipeline.inlet(body_turn1)
    assert "[HR_OIB_1]" in result_turn1["messages"][-1]["content"]
    snapshot_after_t1 = result_turn1["metadata"]["pii_placeholder_map"]
    assert snapshot_after_t1[oib] == "[HR_OIB_1]"

    # Turn 2: presidio_disabled — must still surface the same forward/reverse maps.
    try:
        started_pipeline.valves.presidio_enabled = False
        body_turn2: dict[str, Any] = {
            "messages": [{"role": "user", "content": "Followup question."}],
            "metadata": {"chat_id": chat_id},
        }
        result_turn2 = await started_pipeline.inlet(body_turn2)

        assert result_turn2["metadata"]["pii_placeholder_map"][oib] == "[HR_OIB_1]"
        assert result_turn2["metadata"]["pii_reverse_map"]["[HR_OIB_1]"] == oib
        # Followup not masked (analyzer never ran).
        assert result_turn2["messages"][-1]["content"] == "Followup question."
    finally:
        started_pipeline.valves.presidio_enabled = True


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_user_disabled_takes_precedence_over_presidio_disabled(
    started_pipeline: Pipeline, caplog: pytest.LogCaptureFixture
) -> None:
    """When both toggles are off, the user toggle wins (it is first in
    the early-return chain). No vault snapshot is pulled and no
    `presidio_disabled` log is emitted (AC 8.9, decision #8).
    """
    oib = _make_oib("4445556660")
    chat_id = "ch_user_wins_over_presidio"
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": f"Moj OIB je {oib}"}],
        "metadata": {"chat_id": chat_id},
    }
    try:
        started_pipeline.user_valves = Pipeline.UserValves(pii_masking_enabled=False)
        started_pipeline.valves.presidio_enabled = False
        with caplog.at_level(logging.INFO, logger="pii_filter"):
            result = await started_pipeline.inlet(body, user={"id": "user_42"})

        # Body untouched — user opt-out won.
        assert oib in result["messages"][-1]["content"]
        # No metadata written (the user-disabled branch returns before the
        # presidio_disabled branch would have written empty maps).
        assert "pii_placeholder_map" not in result["metadata"]
        # Only the user_disabled log line — no presidio_disabled line.
        messages = [r.getMessage() for r in caplog.records]
        assert any("user_disabled" in m for m in messages)
        assert not any("presidio_disabled: chat_id" in m for m in messages)
    finally:
        started_pipeline.user_valves = Pipeline.UserValves()
        started_pipeline.valves.presidio_enabled = True
