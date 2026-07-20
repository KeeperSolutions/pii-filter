"""Mask full detected values at ALL occurrences BEFORE the positional splice.

Recon-confirmed leak: GLiNER recall is asymmetric across repeated mentions —
"Ana Kovac ... Ana Kovac" may come back with only the surname "Kovac" at the
first mention and the full "Ana Kovac" at the second. The old per-position
splice masked only detected spans, so the first mention became "Ana [PERSON_1]"
— the first name "Ana" leaked. Fix: once placeholders are minted, mask each
detected value's LITERAL string at every occurrence over the analyzed text
(longest-first, word-boundary, case-sensitive) — so the full "Ana Kovac" is
masked wholesale at the first mention too, where only "Kovac" was detected.

Sub-token safety (the key proof): GLiNER emits 'person'/'first name'/'last name'
for one name, but all map to PERSON (there is NO FIRST_NAME/LAST_NAME type) and
containment-collapse merges the enclosed sub-tokens — so one name is ONE
[PERSON_1], never a [FIRST_NAME_2][PERSON_1] fragment.

These tests drive a REAL GLiNER2Detector with a fake torch model that returns
exact labelled spans, so the production detect() (mapping + collapse + adjacency)
and the inlet mask path both run for real.
"""

from __future__ import annotations

from tests.conftest import FakeAnalyzer, make_gliner_pipeline, pii_mod, user_payload


class LabelledModel:
    """Fake gliner2 model: returns configured (label, start, end) spans verbatim,
    so the real GLiNER2Detector.detect() does the mapping/collapse/adjacency."""

    def __init__(self, spans):
        self._spans = spans  # list of (gliner_label, start, end)

    def extract_entities(self, text, labels, include_confidence=True, include_spans=True):
        ents: dict[str, list[dict]] = {}
        for label, s, e in self._spans:
            ents.setdefault(label, []).append(
                {"text": text[s:e], "start": s, "end": e, "confidence": 0.95}
            )
        return {"entities": ents}


def _pipe_with_spans(spans, *, masking_enabled=True):
    pipe = make_gliner_pipeline(masking_enabled=masking_enabled, name_spans={})
    det = pii_mod.GLiNER2Detector()
    det._model = LabelledModel(spans)
    pipe._gliner = det
    pipe.analyzer_hr = FakeAnalyzer({})  # no Presidio detections
    pipe.valves.ner_person_coreference_enabled = True
    return pipe


def _body(text):
    return {"chat_id": "chat-fullvals", "messages": [{"role": "user", "content": text}]}


# ---------------------------------------------------------------------------
# Main repro: asymmetric recall — first mention only surname, second full
# ---------------------------------------------------------------------------


async def test_asymmetric_recall_first_name_not_leaked():
    text = "Contact Ana Kovac and later Ana Kovac again."
    first = text.index("Ana Kovac")               # 8
    second = text.index("Ana Kovac", first + 1)   # 28
    spans = [
        ("last name", first + 4, first + 9),      # only "Kovac" at the first mention
        ("person", second, second + 9),           # full "Ana Kovac" at the second
        ("first name", second, second + 3),       # "Ana" sub-token (enclosed -> collapses)
    ]
    pipe = _pipe_with_spans(spans)

    out = await pipe.inlet(_body(text), user=user_payload(True))
    content = out["messages"][0]["content"]

    # The whole name is masked at BOTH mentions; "Ana" never leaks.
    assert "Ana Kovac" not in content, content
    assert "Ana" not in content, content
    assert "Kovac" not in content, content
    assert content.count("[PERSON_1]") == 2, content


# ---------------------------------------------------------------------------
# Sub-token safety (the key proof): one name -> one placeholder, no fragments
# ---------------------------------------------------------------------------


async def test_subtokens_collapse_to_single_person_placeholder():
    text = "Meet Ana Kovac today."
    at = text.index("Ana Kovac")  # 5
    spans = [
        ("person", at, at + 9),         # Ana Kovac
        ("first name", at, at + 3),     # Ana
        ("last name", at + 4, at + 9),  # Kovac
    ]
    pipe = _pipe_with_spans(spans)

    out = await pipe.inlet(_body(text), user=user_payload(True))
    content = out["messages"][0]["content"]

    # One name -> exactly one placeholder; no first/last-name fragment placeholders.
    assert content == "Meet [PERSON_1] today.", content
    assert "[PERSON_2]" not in content, content
    assert "FIRST_NAME" not in content and "LAST_NAME" not in content, content


# ---------------------------------------------------------------------------
# Word-boundary safety: a minted value never fires inside a larger word
# ---------------------------------------------------------------------------


async def test_word_boundary_minted_value_not_masked_inside_larger_word():
    text = "Ana and Anamarija are colleagues, Ana again."
    a1 = text.index("Ana")  # 0
    a2 = text.index("Ana again")  # later standalone Ana
    spans = [("first name", a1, a1 + 3), ("first name", a2, a2 + 3)]
    pipe = _pipe_with_spans(spans)

    out = await pipe.inlet(_body(text), user=user_payload(True))
    content = out["messages"][0]["content"]

    assert "Anamarija" in content, content          # untouched (word boundary)
    assert content.count("[PERSON_1]") == 2, content  # both standalone Ana masked


# ---------------------------------------------------------------------------
# Standalone partial elsewhere must NOT leak (why we mask ALL detected surfaces)
# ---------------------------------------------------------------------------


async def test_standalone_partial_surname_also_masked():
    text = "Kovac arrived; then Ana Kovac spoke."
    k = text.index("Kovac")                    # 0 (standalone surname)
    full = text.index("Ana Kovac")             # full name later
    spans = [
        ("last name", k, k + 5),               # standalone "Kovac"
        ("person", full, full + 9),            # "Ana Kovac"
        ("first name", full, full + 3),        # "Ana"
    ]
    pipe = _pipe_with_spans(spans)

    out = await pipe.inlet(_body(text), user=user_payload(True))
    content = out["messages"][0]["content"]

    # Neither the standalone surname nor the full name leaks.
    assert "Kovac" not in content, content
    assert "Ana" not in content, content


# ---------------------------------------------------------------------------
# Regression: ordinary single full detections are unchanged
# ---------------------------------------------------------------------------


async def test_regression_single_full_detection_unchanged():
    text = "Hello John Smith, welcome."
    at = text.index("John Smith")
    spans = [("person", at, at + 10)]
    pipe = _pipe_with_spans(spans)

    out = await pipe.inlet(_body(text), user=user_payload(True))
    content = out["messages"][0]["content"]

    assert content == "Hello [PERSON_1], welcome.", content


async def test_card_omits_subsumed_subvalue():
    """The PII card (metadata) must not list a bare surname that shares a
    placeholder with the full name it is part of — otherwise the user sees both
    "Kovac" and "Ana Kovac" for one masked person."""
    text = "Meet Ana Kovac and Ana Kovac."
    first = text.index("Ana Kovac")
    second = text.index("Ana Kovac", first + 1)
    spans = [
        ("last name", first + 4, first + 9),  # only "Kovac" at the first mention
        ("person", second, second + 9),       # full "Ana Kovac" at the second
        ("first name", second, second + 3),
    ]
    pipe = _pipe_with_spans(spans)

    out = await pipe.inlet(_body(text), user=user_payload(True))
    detections = out["metadata"]["pii_detections"]
    originals = [d["original"] for d in detections]

    assert "Ana Kovac" in originals, originals
    assert "Kovac" not in originals, originals
    # Public card mirrors it: one PERSON span, not two.
    public = out["metadata"]["pii_detections_public"]
    assert len(public) == 1, public


async def test_regression_two_distinct_names_separate_placeholders():
    text = "John Smith met Jane Doe."
    j = text.index("John Smith")
    d = text.index("Jane Doe")
    spans = [("person", j, j + 10), ("person", d, d + 8)]
    pipe = _pipe_with_spans(spans)

    out = await pipe.inlet(_body(text), user=user_payload(True))
    content = out["messages"][0]["content"]

    assert content == "[PERSON_1] met [PERSON_2].", content
