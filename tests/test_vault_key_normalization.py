"""Whitespace-variant vault keys must resolve to ONE placeholder (TRAU-530).

Production repro: the same Dublin address appeared in two turns of one thread,
differing only in a newline, and got two placeholders::

    turn 1  [ADDRESS_2] = "45 Baggot Street Lower,\\nDublin 2"
    turn 2  [ADDRESS_6] = "45 Baggot Street Lower, Dublin 2"

The LLM saw [ADDRESS_2] in the history and [ADDRESS_6] in the new turn with no
way to know they were the same place, so the conversation's semantics broke. The
control case "221B Baker Street, London NW1 6XE" was byte-identical across both
turns and correctly reused one placeholder — which is what pinned the diagnosis.

Root cause: the vault key was the raw detection surface, hashed straight into
`lookup_hash` (the PK) by `BlindIndex.compute` with zero normalization, so two
whitespace variants were two different primary keys.

Fix: `_normalize_vault_key` collapses whitespace for the LOOKUP key only. The
LITERAL surface still drives masking. Those two must never be merged — see
`test_masking_still_uses_the_literal_surface`, which is the leak guard, not a
feature test.
"""

from __future__ import annotations

import pytest

from tests.conftest import (
    FakeAnalyzer,
    FakeGliner,
    make_gliner_pipeline,
    pii_mod,
    user_payload,
)

_normalize = pii_mod._normalize_vault_key

# The production values, verbatim.
ADDR_MULTILINE = "45 Baggot Street Lower,\nDublin 2"
ADDR_SINGLELINE = "45 Baggot Street Lower, Dublin 2"
ADDR_CONTROL = "221B Baker Street, London NW1 6XE"


def _u(text):
    return {"role": "user", "content": text}


def _body(chat_id, messages):
    return {"chat_id": chat_id, "messages": messages}


# ---------------------------------------------------------------------------
# Unit: the normalizer itself
# ---------------------------------------------------------------------------


def test_newline_and_space_normalize_to_the_same_key():
    """The reported bug, at its smallest."""
    assert _normalize(ADDR_MULTILINE) == _normalize(ADDR_SINGLELINE)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("a\nb", "a b"),
        ("a  b", "a b"),
        ("a\t\tb", "a b"),
        ("a\r\nb", "a b"),
        ("  a b  ", "a b"),
        ("\na b\n", "a b"),
        ("a\xa0b", "a b"),  # NBSP — copy-paste from PDF/Word
        ("a b", "a b"),  # already canonical: no-op
    ],
    ids=["lf", "double-space", "tabs", "crlf", "outer", "outer-nl", "nbsp", "noop"],
)
def test_whitespace_collapse_cases(raw, expected):
    assert _normalize(raw) == expected


def test_case_is_preserved():
    """No casefold: the detection path is case-sensitive by design, and folding
    could merge two genuinely distinct values."""
    assert _normalize("Dublin") != _normalize("dublin")
    assert _normalize("  Dublin  ") == "Dublin"


def test_unicode_composition_is_not_normalized():
    """No NFKC: composed vs decomposed diacritics are a separate problem."""
    composed = "Lukić"  # ć
    decomposed = "Lukić"  # c + combining acute
    assert _normalize(composed) != _normalize(decomposed)


def test_internal_non_whitespace_is_untouched():
    """Only whitespace collapses — punctuation and digits are identity."""
    assert _normalize("HR12 3456 7890") == "HR12 3456 7890"
    assert _normalize("45 Baggot Street Lower, Dublin 2") == ADDR_SINGLELINE


# ---------------------------------------------------------------------------
# T1 — the regression test for the reported bug
# ---------------------------------------------------------------------------


async def test_t1_address_whitespace_variants_share_one_placeholder():
    """T1. Turn 1 has the address broken over two lines, turn 2 has it on one.
    Both must resolve to the SAME placeholder."""
    pipe = make_gliner_pipeline(
        name_spans={ADDR_MULTILINE: "ADDRESS", ADDR_SINGLELINE: "ADDRESS"}
    )
    pipe.analyzer_hr = FakeAnalyzer({})
    chat = "chat-t1"

    t1 = await pipe.inlet(
        _body(chat, [_u(f"Moja adresa je {ADDR_MULTILINE}.")]), user=user_payload(True)
    )
    t2 = await pipe.inlet(
        _body(chat, [_u(f"Posalji na {ADDR_SINGLELINE}.")]), user=user_payload(True)
    )

    first = t1["metadata"]["pii_detections"][0]["placeholder"]
    second = t2["metadata"]["pii_detections"][0]["placeholder"]
    assert first == second == "[ADDRESS_1]", (first, second)
    assert "[ADDRESS_2]" not in t2["messages"][0]["content"]


async def test_t1_control_case_byte_identical_still_works():
    """The control from the production report: an address that never varied kept
    reusing one placeholder. Guards against a fix that breaks the working path."""
    pipe = make_gliner_pipeline(name_spans={ADDR_CONTROL: "ADDRESS"})
    pipe.analyzer_hr = FakeAnalyzer({})
    chat = "chat-t1-control"

    t1 = await pipe.inlet(_body(chat, [_u(f"A: {ADDR_CONTROL}")]), user=user_payload(True))
    t2 = await pipe.inlet(_body(chat, [_u(f"B: {ADDR_CONTROL}")]), user=user_payload(True))

    assert t1["metadata"]["pii_detections"][0]["placeholder"] == "[ADDRESS_1]"
    assert t2["metadata"]["pii_detections"][0]["placeholder"] == "[ADDRESS_1]"


# ---------------------------------------------------------------------------
# T2 — not ADDRESS-specific
# ---------------------------------------------------------------------------


async def test_t2_person_whitespace_variants_share_one_placeholder():
    """T2. The key derivation is type-agnostic, so the fix must be too."""
    pipe = make_gliner_pipeline(
        name_spans={"Ivan\nHorvat": "PERSON", "Ivan Horvat": "PERSON"}
    )
    pipe.analyzer_hr = FakeAnalyzer({})
    chat = "chat-t2"

    t1 = await pipe.inlet(_body(chat, [_u("Ivan\nHorvat je stigao.")]), user=user_payload(True))
    t2 = await pipe.inlet(_body(chat, [_u("Ivan Horvat je otisao.")]), user=user_payload(True))

    assert t1["metadata"]["pii_detections"][0]["placeholder"] == "[PERSON_1]"
    assert t2["metadata"]["pii_detections"][0]["placeholder"] == "[PERSON_1]"


# ---------------------------------------------------------------------------
# T3 — LEAK GUARD: masking uses the LITERAL surface, not the normalized key
# ---------------------------------------------------------------------------


async def test_t3_masking_still_uses_the_literal_surface():
    """T3. LEAK GUARD, not a feature test.

    `key` (normalized, -> vault) and `surface` (literal, -> masker) are separate
    on purpose. If someone 'tidies up' the inlet by feeding the normalized form
    to `_mask_full_values_all_occurrences`, it searches for "..., Dublin 2" in
    text that contains "...,\\nDublin 2", matches nothing, and the raw address
    reaches the LLM UNMASKED — silently, with no exception and no log.

    This test fails loudly the moment that happens.
    """
    pipe = make_gliner_pipeline(name_spans={ADDR_MULTILINE: "ADDRESS"})
    pipe.analyzer_hr = FakeAnalyzer({})

    out = await pipe.inlet(
        _body("chat-t3", [_u(f"Adresa: {ADDR_MULTILINE}.")]), user=user_payload(True)
    )
    content = out["messages"][0]["content"]

    assert content == "Adresa: [ADDRESS_1]."
    assert "Baggot" not in content, f"multi-line address leaked unmasked: {content!r}"
    assert "Dublin" not in content, f"multi-line address leaked unmasked: {content!r}"


async def test_t3_both_literal_forms_mask_in_one_message():
    """Both whitespace variants in ONE message: the vault dedupes them onto one
    placeholder, and the masker still needs both literal forms to find them."""
    pipe = make_gliner_pipeline(
        name_spans={ADDR_MULTILINE: "ADDRESS", ADDR_SINGLELINE: "ADDRESS"}
    )
    pipe.analyzer_hr = FakeAnalyzer({})

    out = await pipe.inlet(
        _body("chat-t3b", [_u(f"Prvo {ADDR_MULTILINE} pa {ADDR_SINGLELINE}.")]),
        user=user_payload(True),
    )
    content = out["messages"][0]["content"]

    assert "Baggot" not in content, content
    assert content.count("[ADDRESS_1]") == 2, content
    assert "[ADDRESS_2]" not in content, content


# ---------------------------------------------------------------------------
# Vault-less fallback must expose the SAME metadata shape as the vault path
# ---------------------------------------------------------------------------


async def test_fallback_placeholder_map_is_keyed_by_the_literal_value():
    """`metadata.pii_placeholder_map` is documented as `original_value ->
    placeholder` (pii_filter.py:1242) and the vault path fills it from
    `snapshot_for_request`, whose keys are literal stored originals.

    The vault-less fallback builds the map itself, so it must publish the same
    literal-keyed shape. An interim revision keyed it by the NORMALIZED lookup
    value, which made the two paths disagree for the same input and broke
    lookups by the literal text for any whitespace-variant value.
    """
    pipe = make_gliner_pipeline(name_spans={ADDR_MULTILINE: "ADDRESS"})
    pipe.analyzer_hr = FakeAnalyzer({})
    pipe.valves.vault_enabled = False  # force the fallback path

    out = await pipe.inlet(
        _body("chat-fallback", [_u(f"A: {ADDR_MULTILINE}")]), user=user_payload(True)
    )
    forward = out["metadata"]["pii_placeholder_map"]
    reverse = out["metadata"]["pii_reverse_map"]

    assert ADDR_MULTILINE in forward, f"map is not keyed by the literal value: {forward}"
    assert ADDR_SINGLELINE not in forward, "normalized key leaked into the metadata map"
    assert reverse["[ADDRESS_1]"] == ADDR_MULTILINE


async def test_fallback_still_dedupes_whitespace_variants():
    """The literal-keyed map must not cost us the dedup: both variants in one
    message share a placeholder, and only the first is recorded (first-write-
    wins, mirroring the vault's ON CONFLICT)."""
    pipe = make_gliner_pipeline(
        name_spans={ADDR_MULTILINE: "ADDRESS", ADDR_SINGLELINE: "ADDRESS"}
    )
    pipe.analyzer_hr = FakeAnalyzer({})
    pipe.valves.vault_enabled = False

    out = await pipe.inlet(
        _body("chat-fallback-2", [_u(f"Prvo {ADDR_MULTILINE} pa {ADDR_SINGLELINE}.")]),
        user=user_payload(True),
    )
    content = out["messages"][0]["content"]
    forward = out["metadata"]["pii_placeholder_map"]

    assert content.count("[ADDRESS_1]") == 2, content
    assert "[ADDRESS_2]" not in content, content
    assert len(forward) == 1, f"first-write-wins broken: {forward}"
    assert ADDR_MULTILINE in forward


# ---------------------------------------------------------------------------
# T4 — normalization must not merge what it should not
# ---------------------------------------------------------------------------


async def test_t4_genuinely_different_values_keep_distinct_placeholders():
    """T4. Whitespace collapse must not become a general-purpose fuzzy match."""
    pipe = make_gliner_pipeline(
        name_spans={ADDR_SINGLELINE: "ADDRESS", ADDR_CONTROL: "ADDRESS"}
    )
    pipe.analyzer_hr = FakeAnalyzer({})
    chat = "chat-t4"

    t1 = await pipe.inlet(_body(chat, [_u(f"A: {ADDR_SINGLELINE}")]), user=user_payload(True))
    t2 = await pipe.inlet(_body(chat, [_u(f"B: {ADDR_CONTROL}")]), user=user_payload(True))

    assert t1["metadata"]["pii_detections"][0]["placeholder"] == "[ADDRESS_1]"
    assert t2["metadata"]["pii_detections"][0]["placeholder"] == "[ADDRESS_2]"


@pytest.mark.parametrize(
    "a,b",
    [
        ("Dublin 2", "Dublin 3"),
        ("Ivan Horvat", "Ivana Horvat"),
        ("Dublin", "dublin"),
        ("45 Baggot Street", "46 Baggot Street"),
    ],
    ids=["digit", "extra-char", "case", "number"],
)
def test_t4_distinct_values_normalize_distinctly(a, b):
    assert _normalize(a) != _normalize(b)


# ---------------------------------------------------------------------------
# T5 — DIO 1 unchanged: Step 0.5 still merges across ",\n"
# ---------------------------------------------------------------------------


def test_t5_step_0_5_merges_across_newline_gap():
    """T5. The multi-line address must become ONE span — that is the whole point
    of keeping \\n and \\r in `_CITY_ADJACENCY_ALLOWED_GAP_CHARS`. Without this,
    the key normalization has nothing to normalize."""
    text = f"Adresa: {ADDR_MULTILINE}"
    street_start = text.index("45 Baggot Street Lower")
    city_start = text.index("Dublin 2")
    dets = [
        pii_mod.RecognizerResult(
            entity_type="ADDRESS",
            start=street_start,
            end=street_start + len("45 Baggot Street Lower"),
            score=0.9,
        ),
        pii_mod.RecognizerResult(
            entity_type=pii_mod._ADDRESS_CITY_TYPE,
            start=city_start,
            end=city_start + len("Dublin 2"),
            score=0.9,
        ),
    ]
    out = pii_mod._select_accepted_detections(text, dets, {"ADDRESS": "ADDRESS"})

    assert len(out) == 1, out
    assert text[out[0].start : out[0].end] == ADDR_MULTILINE


def test_t5_newline_chars_are_still_allowed_gap_chars():
    """Pinned explicitly: an earlier revision of TRAU-530 considered removing
    these. Removing them reopens the split-address bug."""
    assert "\n" in pii_mod._CITY_ADJACENCY_ALLOWED_GAP_CHARS
    assert "\r" in pii_mod._CITY_ADJACENCY_ALLOWED_GAP_CHARS


# ---------------------------------------------------------------------------
# T6 — E2E: the production onboarding scenario
# ---------------------------------------------------------------------------


async def test_t6_production_scenario_end_to_end():
    """T6. The reported thread, reconstructed: turn 1 pastes the address broken
    over two lines (as it arrives from a form or a signature block), turn 2
    retypes it inline, and the history carries turn 1 forward.

    Before the fix the LLM saw [ADDRESS_2] in the history and [ADDRESS_6] in the
    new turn. It must now see ONE placeholder for one place.
    """
    pipe = make_gliner_pipeline(
        name_spans={
            ADDR_MULTILINE: "ADDRESS",
            ADDR_SINGLELINE: "ADDRESS",
            ADDR_CONTROL: "ADDRESS",
        }
    )
    pipe.analyzer_hr = FakeAnalyzer({})
    chat = "chat-t6-onboarding"

    turn1 = await pipe.inlet(
        _body(chat, [_u(f"Moja adresa je:\n{ADDR_MULTILINE}\nHvala.")]),
        user=user_payload(True),
    )
    masked_turn1 = turn1["messages"][0]["content"]
    assert "Baggot" not in masked_turn1, masked_turn1

    # Turn 2 carries turn 1 in the history and repeats the address inline.
    turn2 = await pipe.inlet(
        _body(
            chat,
            [
                _u(masked_turn1),
                _u(f"Da potvrdim, {ADDR_SINGLELINE} — je li to ok?"),
            ],
        ),
        user=user_payload(True),
    )
    content = turn2["messages"][-1]["content"]

    assert "Baggot" not in content, content
    assert "[ADDRESS_1]" in content, content
    # The bug: a second placeholder for the same place.
    assert "[ADDRESS_2]" not in content, content

    # One vault row per distinct place, not one per whitespace variant.
    forward, reverse = await pipe.vault.snapshot_for_request(chat)
    assert len(reverse) == 1, reverse


async def test_t6_outlet_restores_the_first_seen_literal_form():
    """The stored value is LITERAL, and first-write-wins picks which literal.

    `ON CONFLICT DO UPDATE` only bumps `expires_at`, never `original_value`, so
    the first form to claim a key is what `outlet` returns for that key's
    lifetime. Turn 1 typed it multi-line, so the newline survives.
    """
    pipe = make_gliner_pipeline(
        name_spans={ADDR_MULTILINE: "ADDRESS", ADDR_SINGLELINE: "ADDRESS"}
    )
    pipe.analyzer_hr = FakeAnalyzer({})
    chat = "chat-t6-restore"

    await pipe.inlet(_body(chat, [_u(f"A: {ADDR_MULTILINE}")]), user=user_payload(True))
    await pipe.inlet(_body(chat, [_u(f"B: {ADDR_SINGLELINE}")]), user=user_payload(True))

    _, reverse = await pipe.vault.snapshot_for_request(chat)
    restored, _, _ = pii_mod.restore_text("Adresa je [ADDRESS_1].", reverse)

    assert restored == f"Adresa je {ADDR_MULTILINE}."
    assert "\n" in restored, "the stored value was normalized — the split is broken"


# ---------------------------------------------------------------------------
# T7 — conventional spacing survives the round trip
# ---------------------------------------------------------------------------


IBAN_SPACED = "HR12 3456 7890"
IBAN_TIGHT = "HR1234567890"
IBAN_DOUBLE = "HR12  3456  7890"


def test_t7_whitespace_variants_of_an_iban_share_a_key():
    """Collapse merges spacing variants..."""
    assert _normalize(IBAN_SPACED) == _normalize(IBAN_DOUBLE)


def test_t7_removing_spaces_entirely_is_a_different_key():
    """...but does NOT delete spaces. "HR12 3456 7890" and "HR1234567890" are
    different keys, because collapsing is not stripping. Recorded so the
    boundary is a decision, not an accident — closing this gap needs
    per-entity-type canonicalization, which this ticket does not do."""
    assert _normalize(IBAN_SPACED) != _normalize(IBAN_TIGHT)


async def test_t7_iban_spacing_survives_restore():
    """T7. RISK 2, demonstrated closed rather than asserted.

    Two turns type the same IBAN with different spacing. They share a
    placeholder, and `outlet` hands back the FIRST literal form with its
    conventional grouping intact — not a whitespace-collapsed rewrite of the
    user's own bank account number.
    """
    pipe = make_gliner_pipeline(
        name_spans={IBAN_SPACED: "HR_IBAN", IBAN_DOUBLE: "HR_IBAN"}
    )
    pipe.analyzer_hr = FakeAnalyzer({})
    chat = "chat-t7"

    t1 = await pipe.inlet(
        _body(chat, [_u(f"IBAN mi je {IBAN_SPACED}.")]), user=user_payload(True)
    )
    t2 = await pipe.inlet(
        _body(chat, [_u(f"Ponavljam: {IBAN_DOUBLE}.")]), user=user_payload(True)
    )

    # One identity across both turns.
    assert t1["metadata"]["pii_detections"][0]["placeholder"] == "[HR_IBAN_1]"
    assert t2["metadata"]["pii_detections"][0]["placeholder"] == "[HR_IBAN_1]"

    # Restore returns the literal first-seen spacing, NOT a normalized rewrite.
    _, reverse = await pipe.vault.snapshot_for_request(chat)
    restored, _, _ = pii_mod.restore_text("Racun: [HR_IBAN_1]", reverse)
    assert restored == f"Racun: {IBAN_SPACED}"


async def test_t7_double_spaced_occurrence_is_still_masked():
    """The second turn's literal form differs from the stored one, so the masker
    must still find it via its own literal — proof that `inturn_forward` stayed
    keyed on `surface`."""
    pipe = make_gliner_pipeline(name_spans={IBAN_DOUBLE: "HR_IBAN"})
    pipe.analyzer_hr = FakeAnalyzer({})

    out = await pipe.inlet(
        _body("chat-t7b", [_u(f"IBAN: {IBAN_DOUBLE}.")]), user=user_payload(True)
    )
    assert out["messages"][0]["content"] == "IBAN: [HR_IBAN_1]."


async def test_multiline_history_occurrence_is_still_remasked():
    """LEAK GUARD — the defect that forced the lookup/stored split.

    An interim revision normalized the value passed to `get_placeholder`, which
    normalized the STORED value too (one argument fed both `lookup_hash` and
    `original_value`). `snapshot_for_request` then returned normalized originals,
    so the `re.escape`d `_build_vault_remasker` pattern no longer matched the
    multi-line occurrence, and a middle-of-history message — which per the
    middleground NER scope gets re-mask ONLY, no NER — reached the LLM with the
    raw address. Silent: no exception, no log.

    Turn 3 carries the turn-1 raw message in the MIDDLE of the history. The raw
    address must not reach the LLM.
    """
    pipe = make_gliner_pipeline(name_spans={ADDR_MULTILINE: "ADDRESS"})
    pipe.analyzer_hr = FakeAnalyzer({})
    chat = "chat-remask-gap"

    await pipe.inlet(_body(chat, [_u(f"A: {ADDR_MULTILINE}")]), user=user_payload(True))

    pipe._gliner = FakeGliner(name_spans={})  # NER recall gap on history
    out = await pipe.inlet(
        _body(
            chat,
            [
                _u(f"A: {ADDR_MULTILINE}"),
                {"role": "assistant", "content": "ok"},
                _u("novo pitanje"),
            ],
        ),
        user=user_payload(True),
    )

    assert "Baggot" not in out["messages"][0]["content"]
