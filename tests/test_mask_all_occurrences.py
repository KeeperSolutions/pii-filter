"""Mask-all-occurrences for values minted THIS turn (TRAU-522 repeat-name leak).

Recon-confirmed root cause: GLiNER emits a repeated identical name only ONCE
(e.g. "Jimmy Fallon ... Jimmy Fallon" -> a single PERSON span at the first
position), and the splice masks only detected positions (no global replace), so
the second literal occurrence leaks to the LLM. The vault re-mask that runs
FIRST is a no-op on a first mention (the value is not vaulted yet; its mapping
is minted only during the splice).

Fix: as the splice mints ``[PERSON_N]`` for a detected surface, accumulate a
``value -> placeholder`` map for this request, then run ONE more re-mask pass
(the exact vault-remask machinery: longest-first, word-boundary, case-sensitive)
over the spliced text so EVERY remaining occurrence of a just-minted value is
masked to the same placeholder. Worst case is over-masking (fail-safe), never a
leak.

The fake GLiNER here reproduces the model's real behavior: it returns only the
FIRST occurrence of each configured name (unlike conftest.FakeGliner, which
returns all — and so would hide this bug).
"""

from __future__ import annotations

from tests.conftest import (
    FakeAnalyzer,
    FakeGliner,
    RecognizerResult,
    make_gliner_pipeline,
    user_payload,
)

CHAT = "chat-maskall"


class FirstOccurrenceGliner(FakeGliner):
    """GLiNER stand-in that emits each configured name only at its FIRST
    position (plus placeholder re-detection, inherited) — mirroring the real
    model's recall on a repeated identical entity."""

    def detect(self, text):
        results = []
        # Re-detect placeholders like the parent (dangerous behavior guard).
        import pii_filter as _m  # loaded by conftest

        for mt in _m._PLACEHOLDER_RE.finditer(text):
            results.append(
                RecognizerResult(entity_type="PERSON", start=mt.start(), end=mt.end(), score=0.9)
            )
        for needle, entity_type in self.name_spans.items():
            idx = text.find(needle)  # FIRST occurrence only
            if idx != -1:
                results.append(
                    RecognizerResult(
                        entity_type=entity_type, start=idx, end=idx + len(needle), score=0.9
                    )
                )
        return results


def _pipe(*, names, masking_enabled=True):
    pipe = make_gliner_pipeline(masking_enabled=masking_enabled, name_spans=names)
    pipe._gliner = FirstOccurrenceGliner(name_spans=names)
    pipe.analyzer_hr = FakeAnalyzer({})  # no Presidio detections
    return pipe


def _body(text):
    return {"chat_id": CHAT, "messages": [{"role": "user", "content": text}]}


# ---------------------------------------------------------------------------
# THE repro: repeated name, both occurrences must be masked
# ---------------------------------------------------------------------------


async def test_repeated_name_both_occurrences_masked():
    """"Jimmy Fallon" twice, GLiNER detects only the first. BOTH must become the
    SAME placeholder; neither leaks."""
    pipe = _pipe(names={"Jimmy Fallon": "PERSON"})

    out = await pipe.inlet(
        _body("How is Jimmy Fallon doing, and who is Jimmy Fallon?"),
        user=user_payload(True),
    )
    content = out["messages"][0]["content"]

    assert "Jimmy Fallon" not in content, content
    assert content.count("[PERSON_1]") == 2, content


async def test_three_occurrences_all_masked():
    """Three mentions, one detection -> all three masked to one placeholder."""
    pipe = _pipe(names={"Ivan Horvat": "PERSON"})

    out = await pipe.inlet(
        _body("Ivan Horvat, pa opet Ivan Horvat, i treći put Ivan Horvat."),
        user=user_payload(True),
    )
    content = out["messages"][0]["content"]

    assert "Ivan Horvat" not in content, content
    assert content.count("[PERSON_1]") == 3, content


# ---------------------------------------------------------------------------
# Safety: word boundary, no over-reach, no cross-value merge
# ---------------------------------------------------------------------------


async def test_word_boundary_minted_value_not_masked_inside_larger_word():
    """A minted "Fallon" must NOT mask the substring inside "Fallonville"."""
    pipe = _pipe(names={"Fallon": "PERSON"})

    out = await pipe.inlet(
        _body("Fallon lives near Fallonville but Fallon works downtown."),
        user=user_payload(True),
    )
    content = out["messages"][0]["content"]

    assert "Fallonville" in content, content  # untouched
    assert content.count("[PERSON_1]") == 2, content  # the two standalone Fallon
    # exactly one un-masked token remains and it is inside Fallonville
    assert content.count("Fallon") == 1, content


async def test_distinct_single_occurrence_names_keep_separate_placeholders():
    """Two different names, one occurrence each: each keeps its own placeholder;
    mask-all does NOT merge distinct values (regression guard)."""
    pipe = _pipe(names={"Jimmy Fallon": "PERSON", "Ana Kovac": "PERSON"})

    out = await pipe.inlet(
        _body("Jimmy Fallon met Ana Kovac yesterday."),
        user=user_payload(True),
    )
    content = out["messages"][0]["content"]

    assert "Jimmy Fallon" not in content and "Ana Kovac" not in content, content
    assert "[PERSON_1]" in content and "[PERSON_2]" in content, content
    assert content.count("[PERSON_1]") == 1 and content.count("[PERSON_2]") == 1, content


async def test_longest_first_when_minted_name_contains_shorter_minted_name():
    """If both "Jimmy Fallon" and a standalone "Jimmy" are minted, the full name
    wins at its position (longest-first) — no partial "[PERSON_x]y Fallon"."""
    pipe = _pipe(names={"Jimmy Fallon": "PERSON", "Jimmy": "PERSON"})

    out = await pipe.inlet(
        _body("Jimmy Fallon and later just Jimmy alone, then Jimmy Fallon again."),
        user=user_payload(True),
    )
    content = out["messages"][0]["content"]

    assert "Jimmy Fallon" not in content, content
    # Both full-name occurrences masked to the full-name placeholder.
    full_ph = "[PERSON_1]"
    assert content.count(full_ph) == 2, content
    # No corrupted splice like "[PERSON_1] Fallon" or "[PERSON_2] Fallon".
    assert "Fallon" not in content, content


# ---------------------------------------------------------------------------
# Regression: vault re-mask path and masking-off untouched
# ---------------------------------------------------------------------------


async def test_prior_turn_vaulted_name_still_remasked_by_vault_path():
    """A name vaulted on a PRIOR turn, recurring this turn, is handled by the
    vault re-mask (runs first) — the new mask-all pass must not disturb that."""
    pipe = _pipe(names={"Jimmy Fallon": "PERSON"})
    # Turn 1 vaults the name.
    await pipe.inlet(_body("Meet Jimmy Fallon."), user=user_payload(True))
    # Turn 2: name recurs twice; vault re-mask masks known occurrences, mask-all
    # covers any freshly-detected ones. All must be [PERSON_1].
    out = await pipe.inlet(
        _body("Jimmy Fallon again, still Jimmy Fallon."), user=user_payload(True)
    )
    content = out["messages"][0]["content"]

    assert "Jimmy Fallon" not in content, content
    assert content.count("[PERSON_1]") == 2, content


async def test_masking_off_short_circuits_no_mask_all():
    """masking OFF: content verbatim, no placeholders, vault untouched."""
    pipe = _pipe(names={"Jimmy Fallon": "PERSON"}, masking_enabled=False)
    original = "Jimmy Fallon and Jimmy Fallon."

    out = await pipe.inlet(_body(original), user=user_payload(False))

    assert out["messages"][0]["content"] == original
