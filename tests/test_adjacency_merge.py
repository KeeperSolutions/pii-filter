"""Adjacency merge of same-type spans separated only by whitespace.

Recon-confirmed: on the FIRST "Robert Plant" GLiNER emits the sub-tokens
'first name' "Robert" + 'last name' "Plant" (two adjacent PERSON spans, NO
enclosing full-name span), while the SECOND occurrence comes back as one
'person' "Robert Plant". Containment-collapse only merges ENCLOSED spans, so the
two adjacent sub-tokens survive as [PERSON_1] [PERSON_2] — one person fragmented
into two inconsistent placeholders. Not a leak (all masked), but it confuses the
LLM and corrupts outlet restore.

Fix: after containment collapse, merge CONSECUTIVE same-type spans whose only gap
is whitespace into one span, so "Robert" + "Plant" -> "Robert Plant" (one
[PERSON_1]), consistent with the full-name occurrence. Merging only ever grows a
masked span — worst case over-masks, never a leak.
"""

from __future__ import annotations

from tests.conftest import (
    FakeAnalyzer,
    FakeGliner,
    RecognizerResult,
    make_gliner_pipeline,
    pii_mod,
    user_payload,
)


def _merge(results, text):
    return pii_mod._merge_adjacent_same_type(results, text)


def _spans(results):
    return sorted((r.entity_type, r.start, r.end) for r in results)


# ---------------------------------------------------------------------------
# Unit: _merge_adjacent_same_type
# ---------------------------------------------------------------------------


def test_merges_two_adjacent_same_type_separated_by_space():
    text = "Robert Plant"
    results = [
        RecognizerResult(entity_type="PERSON", start=0, end=6, score=0.99),   # Robert
        RecognizerResult(entity_type="PERSON", start=7, end=12, score=0.92),  # Plant
    ]
    merged = _merge(results, text)
    assert _spans(merged) == [("PERSON", 0, 12)]
    assert text[merged[0].start : merged[0].end] == "Robert Plant"


def test_merged_span_keeps_max_score():
    text = "Robert Plant"
    results = [
        RecognizerResult(entity_type="PERSON", start=0, end=6, score=0.80),
        RecognizerResult(entity_type="PERSON", start=7, end=12, score=0.95),
    ]
    merged = _merge(results, text)
    assert merged[0].score == 0.95


def test_does_not_merge_across_non_whitespace():
    text = "Robert, Plant"  # comma between -> two distinct
    results = [
        RecognizerResult(entity_type="PERSON", start=0, end=6, score=0.9),
        RecognizerResult(entity_type="PERSON", start=8, end=13, score=0.9),
    ]
    merged = _merge(results, text)
    assert _spans(merged) == [("PERSON", 0, 6), ("PERSON", 8, 13)]


def test_does_not_merge_across_word_gap():
    text = "Robert and Plant"  # "and" between
    results = [
        RecognizerResult(entity_type="PERSON", start=0, end=6, score=0.9),
        RecognizerResult(entity_type="PERSON", start=11, end=16, score=0.9),
    ]
    merged = _merge(results, text)
    assert _spans(merged) == [("PERSON", 0, 6), ("PERSON", 11, 16)]


def test_does_not_merge_different_types():
    text = "Plant Wales"
    results = [
        RecognizerResult(entity_type="PERSON", start=0, end=5, score=0.9),
        RecognizerResult(entity_type="ADDRESS", start=6, end=11, score=0.9),
    ]
    merged = _merge(results, text)
    assert _spans(merged) == [("ADDRESS", 6, 11), ("PERSON", 0, 5)]


def test_merges_three_adjacent_tokens():
    text = "Robert van Berg"
    results = [
        RecognizerResult(entity_type="PERSON", start=0, end=6, score=0.9),
        RecognizerResult(entity_type="PERSON", start=7, end=10, score=0.9),
        RecognizerResult(entity_type="PERSON", start=11, end=15, score=0.9),
    ]
    merged = _merge(results, text)
    assert _spans(merged) == [("PERSON", 0, 15)]


def test_leaves_non_adjacent_same_type_separate():
    text = "Robert works, later Plant arrives"
    results = [
        RecognizerResult(entity_type="PERSON", start=0, end=6, score=0.9),
        RecognizerResult(entity_type="PERSON", start=20, end=25, score=0.9),
    ]
    merged = _merge(results, text)
    assert _spans(merged) == [("PERSON", 0, 6), ("PERSON", 20, 25)]


# ---------------------------------------------------------------------------
# detect(): the exact recon repro (sub-tokens at first mention, full at second)
# ---------------------------------------------------------------------------


class SubtokenGliner(FakeGliner):
    """Reproduces the raw model output for 'Robert Plant ... Robert Plant':
    first mention as first-name + last-name sub-tokens, second as the full name."""

    def detect(self, text):  # bypass parent; emit exact spans
        out = []
        # first mention sub-tokens
        r = text.find("Robert Plant")
        if r != -1:
            out.append(RecognizerResult(entity_type="PERSON", start=r, end=r + 6, score=0.99))       # Robert
            out.append(RecognizerResult(entity_type="PERSON", start=r + 7, end=r + 12, score=0.92))  # Plant
            # second mention full name
            r2 = text.find("Robert Plant", r + 12)
            if r2 != -1:
                out.append(RecognizerResult(entity_type="PERSON", start=r2, end=r2 + 12, score=0.95))
        return out


def test_detect_merges_subtokens_into_full_name():
    det = pii_mod.GLiNER2Detector()
    det._model = None  # unused; override detect via subclass instead
    det = SubtokenGliner(name_spans={})
    text = "Who is Robert Plant and is Robert Plant from Wales?"
    # SubtokenGliner is a bare fake; exercise the module merge as detect would.
    merged = pii_mod._merge_adjacent_same_type(det.detect(text), text)
    spans = _spans(merged)
    # Both occurrences are now single full-name PERSON spans.
    assert ("PERSON", 7, 19) in spans, spans
    assert ("PERSON", 27, 39) in spans, spans
    assert all(text[s:e] == "Robert Plant" for _, s, e in spans), spans


# ---------------------------------------------------------------------------
# Integration: full inlet -> both occurrences the SAME single placeholder
# ---------------------------------------------------------------------------


class InletSubtokenGliner(FakeGliner):
    def detect(self, text):
        import pii_filter as _m

        out = [
            RecognizerResult(entity_type="PERSON", start=m.start(), end=m.end(), score=0.9)
            for m in _m._PLACEHOLDER_RE.finditer(text)
        ]
        r = text.find("Robert Plant")
        if r != -1:
            out.append(RecognizerResult(entity_type="PERSON", start=r, end=r + 6, score=0.99))
            out.append(RecognizerResult(entity_type="PERSON", start=r + 7, end=r + 12, score=0.92))
            r2 = text.find("Robert Plant", r + 12)
            if r2 != -1:
                out.append(RecognizerResult(entity_type="PERSON", start=r2, end=r2 + 12, score=0.95))
        return out


async def test_inlet_robert_plant_single_consistent_placeholder():
    pipe = make_gliner_pipeline(masking_enabled=True, name_spans={})
    pipe._gliner = InletSubtokenGliner(name_spans={})
    pipe.analyzer_hr = FakeAnalyzer({})
    pipe.valves.ner_person_coreference_enabled = True

    out = await pipe.inlet(
        {"chat_id": "chat-robert", "messages": [
            {"role": "user", "content": "Who is Robert Plant and is Robert Plant from Wales?"}
        ]},
        user=user_payload(True),
    )
    content = out["messages"][0]["content"]

    assert "Robert Plant" not in content, content
    # One person, one placeholder, both occurrences identical — no [PERSON_1] [PERSON_2] split.
    assert content.count("[PERSON_1]") == 2, content
    assert "[PERSON_2]" not in content, content
