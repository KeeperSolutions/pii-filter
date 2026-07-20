"""Opt-in debug log that dumps the original->placeholder mapping per request.

Gated behind the env flag PII_DEBUG_UNMASK_LOG. OFF by default and MUST stay off
in production/Cloud Run because it logs plaintext PII. When on, every processed
user message emits a `[PII_DEBUG]` log line with the original text, the masked
text, and the value->placeholder pairs — so a real OpenWebUI chat can be watched
live via `docker logs -f ... | grep PII_DEBUG`.
"""

from __future__ import annotations

import logging

import pytest

from tests.conftest import FakeAnalyzer, FakeGliner, make_gliner_pipeline, pii_mod, user_payload

CHAT = "chat-dbglog"
PERSON = "Jimmy Fallon"


class FirstOccurrenceGliner(FakeGliner):
    """Emits each configured name only at its FIRST position (real-model recall
    on a repeated identical entity), plus placeholder re-detection."""

    def detect(self, text):
        import pii_filter as _m
        from tests.conftest import RecognizerResult

        results = [
            RecognizerResult(entity_type="PERSON", start=m.start(), end=m.end(), score=0.9)
            for m in _m._PLACEHOLDER_RE.finditer(text)
        ]
        for needle, etype in self.name_spans.items():
            idx = text.find(needle)
            if idx != -1:
                results.append(
                    RecognizerResult(entity_type=etype, start=idx, end=idx + len(needle), score=0.9)
                )
        return results


def _pipe():
    pipe = make_gliner_pipeline(masking_enabled=True, name_spans={PERSON: "PERSON"})
    pipe._gliner = FirstOccurrenceGliner(name_spans={PERSON: "PERSON"})
    pipe.analyzer_hr = FakeAnalyzer({})
    return pipe


def _body(text):
    return {"chat_id": CHAT, "messages": [{"role": "user", "content": text}]}


async def test_no_debug_log_when_flag_unset(monkeypatch, caplog):
    monkeypatch.delenv("PII_DEBUG_UNMASK_LOG", raising=False)
    pipe = _pipe()
    with caplog.at_level(logging.INFO):
        await pipe.inlet(_body(f"How is {PERSON}?"), user=user_payload(True))
    assert "PII_DEBUG" not in caplog.text


async def test_debug_log_emitted_when_flag_on(monkeypatch, caplog):
    monkeypatch.setenv("PII_DEBUG_UNMASK_LOG", "true")
    pipe = _pipe()
    text = f"How is {PERSON} doing, and who is {PERSON}?"
    with caplog.at_level(logging.INFO):
        await pipe.inlet(_body(text), user=user_payload(True))

    debug_lines = [r.getMessage() for r in caplog.records if "PII_DEBUG" in r.getMessage()]
    assert debug_lines, "expected a [PII_DEBUG] line when the flag is on"
    blob = "\n".join(debug_lines)
    # The mapping must be visible: original value -> its placeholder.
    assert PERSON in blob
    assert "[PERSON_1]" in blob
    # Both the original and the masked form of the message are shown.
    assert text in blob
    assert "How is [PERSON_1] doing, and who is [PERSON_1]?" in blob


async def test_debug_log_is_multiline_with_occurrence_counts(monkeypatch, caplog):
    """Readable format: a multi-line block with labelled original/masked lines
    and a per-value occurrence count (the 'identified in both places' signal)."""
    monkeypatch.setenv("PII_DEBUG_UNMASK_LOG", "true")
    pipe = _pipe()
    text = f"How is {PERSON} doing, and who is {PERSON}?"
    with caplog.at_level(logging.INFO):
        await pipe.inlet(_body(text), user=user_payload(True))

    rec = next(r.getMessage() for r in caplog.records if "PII_DEBUG" in r.getMessage())
    assert "\n" in rec, "debug output should be multi-line"
    # Occurrence count for the repeated value is shown (2 mentions).
    assert "2" in rec and PERSON in rec
    # Labelled original/masked lines present.
    assert "original" in rec.lower() and "masked" in rec.lower()
    # The mapping arrow is present.
    assert "[PERSON_1]" in rec


class _LabelledModel:
    def __init__(self, spans):
        self._spans = spans

    def extract_entities(self, text, labels, include_confidence=True, include_spans=True):
        ents: dict = {}
        for label, s, e in self._spans:
            ents.setdefault(label, []).append(
                {"text": text[s:e], "start": s, "end": e, "confidence": 0.95}
            )
        return {"entities": ents}


async def test_debug_report_omits_subsumed_subvalue(monkeypatch, caplog):
    """When a partial detection ("Kovac") shares a placeholder with the full
    value ("Ana Kovac"), the report must NOT list the subsumed sub-value — it is
    never masked independently (longest-first) and double-counts one entity."""
    monkeypatch.setenv("PII_DEBUG_UNMASK_LOG", "true")
    from tests.conftest import FakeAnalyzer

    text = "Meet Ana Kovac and Ana Kovac."
    first = text.index("Ana Kovac")
    second = text.index("Ana Kovac", first + 1)
    spans = [
        ("last name", first + 4, first + 9),  # only "Kovac" at first mention
        ("person", second, second + 9),       # full "Ana Kovac" at second
        ("first name", second, second + 3),
    ]
    pipe = make_gliner_pipeline(masking_enabled=True, name_spans={})
    det = pii_mod.GLiNER2Detector()
    det._model = _LabelledModel(spans)
    pipe._gliner = det
    pipe.analyzer_hr = FakeAnalyzer({})
    pipe.valves.ner_person_coreference_enabled = True

    with caplog.at_level(logging.INFO):
        await pipe.inlet(_body(text), user=user_payload(True))

    rec = next(r.getMessage() for r in caplog.records if "PII_DEBUG" in r.getMessage())
    # The full value is reported; the subsumed bare surname is not a separate row.
    assert "'Ana Kovac' →" in rec, rec
    assert "'Kovac' →" not in rec, rec
    # Count reflects distinct entities (1), not raw map size (2).
    assert "1 masked" in rec, rec


async def test_debug_log_no_ansi_by_default(monkeypatch, caplog):
    """No color escape codes unless PII_DEBUG_COLOR is enabled — keeps piped/
    grepped logs clean."""
    monkeypatch.setenv("PII_DEBUG_UNMASK_LOG", "true")
    monkeypatch.delenv("PII_DEBUG_COLOR", raising=False)
    pipe = _pipe()
    with caplog.at_level(logging.INFO):
        await pipe.inlet(_body(f"Hi {PERSON} and {PERSON}."), user=user_payload(True))
    rec = next(r.getMessage() for r in caplog.records if "PII_DEBUG" in r.getMessage())
    assert "\033[" not in rec, "no ANSI codes unless PII_DEBUG_COLOR is on"


async def test_debug_log_masking_off_does_not_emit(monkeypatch, caplog):
    """masking OFF short-circuits above the mask path -> nothing to dump."""
    monkeypatch.setenv("PII_DEBUG_UNMASK_LOG", "true")
    pipe = make_gliner_pipeline(masking_enabled=False, name_spans={PERSON: "PERSON"})
    with caplog.at_level(logging.INFO):
        await pipe.inlet(_body(f"How is {PERSON}?"), user=user_payload(False))
    assert "PII_DEBUG" not in caplog.text


async def test_debug_log_emitted_for_remask_only_turn(monkeypatch, caplog):
    """A later turn where the name is ALREADY vaulted produces no new detection
    (the re-mask masks it before NER). The live-watch log must still fire so the
    turn is visible — otherwise it looks like nothing was masked."""
    monkeypatch.setenv("PII_DEBUG_UNMASK_LOG", "true")
    pipe = _pipe()
    # Turn 1 vaults the name (detection path).
    await pipe.inlet(_body(f"Meet {PERSON}."), user=user_payload(True))
    caplog.clear()
    # Turn 2: name recurs but is already vaulted -> re-mask only, no detection.
    with caplog.at_level(logging.INFO):
        await pipe.inlet(_body(f"Again, {PERSON}."), user=user_payload(True))

    debug_lines = [r.getMessage() for r in caplog.records if "PII_DEBUG" in r.getMessage()]
    assert debug_lines, "re-mask-only turn must still emit a [PII_DEBUG] line"
    blob = "\n".join(debug_lines)
    assert PERSON in blob and "[PERSON_1]" in blob


@pytest.mark.parametrize("val", ["0", "false", "no", ""])
async def test_flag_falsey_values_disable(monkeypatch, caplog, val):
    monkeypatch.setenv("PII_DEBUG_UNMASK_LOG", val)
    pipe = _pipe()
    with caplog.at_level(logging.INFO):
        await pipe.inlet(_body(f"Hi {PERSON}."), user=user_payload(True))
    assert "PII_DEBUG" not in caplog.text
