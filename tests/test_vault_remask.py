"""Option A — deterministic vault re-mask (TRAU-522).

NER re-derives masking every turn and its recall degrades on accumulated
multi-turn context, so a previously-masked name slips through unmasked (E2E:
repeated name on turn 2+, carried in an assistant turn). The vault already holds
every known (original -> placeholder) for the thread; re-mask them by exact
match BEFORE any analyzer, independent of NER recall.

Determinism comes from the vault, not message position: re-mask is self-limiting
to the START snapshot (prior turns). A name new to this turn is absent from the
snapshot, untouched by re-mask, and caught by the normal NER path.

Two layers:
  * unit — `_build_vault_remasker` / `_apply_vault_remask` string behavior.
  * integration — full inlet flow proving recall-gap coverage, assistant-turn
    masking, Step-0 coordination, and idempotency.
"""

from __future__ import annotations

from tests.conftest import FakeGliner, make_gliner_pipeline, pii_mod, user_payload

_build = pii_mod._build_vault_remasker
_apply = pii_mod._apply_vault_remask
_translate = pii_mod._analyzed_to_original_offset

CHAT = "chat-remask-1"


def _remask(text, forward):
    masked, _spans = _apply(text, _build(forward), forward)
    return masked


# --------------------------------------------------------------------------- #
# Unit: re-mask string behavior
# --------------------------------------------------------------------------- #

def test_longest_match_first():
    fwd = {"Robert Plant": "[PERSON_1]", "Plant": "[PERSON_2]"}
    assert _remask("Robert Plant pjeva", fwd) == "[PERSON_1] pjeva"


def test_standalone_shorter_original_still_masks():
    fwd = {"Robert Plant": "[PERSON_1]", "Plant": "[PERSON_2]"}
    assert _remask("samo Plant", fwd) == "samo [PERSON_2]"


def test_word_boundary_no_substring_match():
    fwd = {"Plant": "[PERSON_1]"}
    assert _remask("velika Plantation ovdje", fwd) == "velika Plantation ovdje"


def test_original_with_leading_non_word_char_is_remasked():
    # A vaulted phone starting with "+" has no \w<->\W boundary between a
    # preceding space and the "+", so plain \b would never re-mask it. The
    # matcher must use lookarounds so such originals are still caught.
    fwd = {"+385 91 234 5678": "[PHONE_1]"}
    assert _remask("Nazovi +385 91 234 5678 danas", fwd) == "Nazovi [PHONE_1] danas"


def test_original_with_trailing_non_word_char_is_remasked():
    fwd = {"@jdoe": "[USERNAME_1]"}
    assert _remask("Korisnik @jdoe je tu", fwd) == "Korisnik [USERNAME_1] je tu"


def test_case_sensitive():
    fwd = {"Ann": "[PERSON_1]"}
    assert _remask("ann i Ann", fwd) == "ann i [PERSON_1]"


def test_idempotent_on_already_masked_text():
    fwd = {"Robert Plant": "[PERSON_1]"}
    assert _remask("[PERSON_1] met [PERSON_2]", fwd) == "[PERSON_1] met [PERSON_2]"


def test_empty_forward_is_noop():
    assert _build({}) is None
    assert _apply("Robert Plant", None, {}) == ("Robert Plant", [])


def test_multiple_occurrences_all_masked():
    fwd = {"Plant": "[PERSON_1]"}
    assert _remask("Plant pa Plant", fwd) == "[PERSON_1] pa [PERSON_1]"


def test_coreference_collision_no_entry_lost():
    # Both originals share ONE placeholder (surname coreference). Building the
    # re-mask off the forward map (keyed by original) preserves BOTH entries.
    fwd = {"Robert Plant": "[PERSON_1]", "Plant": "[PERSON_1]"}
    assert _remask("Robert Plant", fwd) == "[PERSON_1]"
    assert _remask("samo Plant", fwd) == "samo [PERSON_1]"


def test_masks_non_person_entity_types():
    fwd = {"ivan@example.com": "[EMAIL_1]", "12345678903": "[HR_OIB_1]"}
    assert _remask("mail ivan@example.com i OIB 12345678903", fwd) == "mail [EMAIL_1] i OIB [HR_OIB_1]"


# --------------------------------------------------------------------------- #
# Unit: re-mask emits card spans (TRAU-522 — restore card on repeated PII)
# --------------------------------------------------------------------------- #
# The re-mask masks known originals BEFORE the analyzers, so a repeated vaulted
# name never reaches the NER `accepted` path that feeds the card. To keep the
# PII card appearing on every mention, `_apply_vault_remask` also returns the
# spans it masked — in ORIGINAL-text coordinates (finditer runs over the input
# text, before substitution) so the frontend can slice the user's own message.

def test_apply_returns_masked_and_spans_in_original_coords():
    fwd = {"Robert Plant": "[PERSON_1]"}
    masked, spans = _apply("Tko je Robert Plant", _build(fwd), fwd)
    assert masked == "Tko je [PERSON_1]"
    assert spans == [
        {
            "start": 7,
            "end": 19,
            "original": "Robert Plant",
            "placeholder": "[PERSON_1]",
            "entity_type": "PERSON",
        }
    ]
    # Spans index the ORIGINAL text (pre-substitution), so a slice recovers the value.
    assert "Tko je Robert Plant"[7:19] == "Robert Plant"


def test_apply_span_entity_type_derived_from_placeholder():
    # Multi-underscore standard types (HR_OIB) must resolve correctly, not to "HR".
    fwd = {"12345678903": "[HR_OIB_1]"}
    masked, spans = _apply("OIB 12345678903", _build(fwd), fwd)
    assert masked == "OIB [HR_OIB_1]"
    assert spans == [
        {
            "start": 4,
            "end": 15,
            "original": "12345678903",
            "placeholder": "[HR_OIB_1]",
            "entity_type": "HR_OIB",
        }
    ]


def test_apply_no_known_original_yields_empty_spans():
    fwd = {"Plant": "[PERSON_1]"}
    masked, spans = _apply("nothing here", _build(fwd), fwd)
    assert masked == "nothing here"
    assert spans == []


def test_apply_multiple_occurrences_emit_one_span_each():
    fwd = {"Plant": "[PERSON_1]"}
    masked, spans = _apply("Plant pa Plant", _build(fwd), fwd)
    assert masked == "[PERSON_1] pa [PERSON_1]"
    assert [(s["start"], s["end"]) for s in spans] == [(0, 5), (9, 14)]
    assert all(s["entity_type"] == "PERSON" for s in spans)


# --------------------------------------------------------------------------- #
# Unit: analyzed<->original offset translation (for NEW names in a mixed msg)
# --------------------------------------------------------------------------- #

# Marcus [9,25] (16 chars) -> "[PERSON_1]" (10) occupies analyzed [9,19).
_MARCUS = [{"start": 9, "end": 25, "placeholder": "[PERSON_1]", "entity_type": "PERSON"}]


def test_translate_no_spans_is_identity():
    assert _translate(25, []) == 25


def test_translate_pos_before_placeholder_unchanged():
    assert _translate(3, _MARCUS) == 3  # "does" sits before the placeholder


def test_translate_pos_after_placeholder_shifts_back():
    # A pos after the 6-char-shorter placeholder maps back by +6.
    assert _translate(25, _MARCUS) == 31


def test_translate_multiple_placeholders_cumulative():
    spans = [
        {"start": 0, "end": 12, "placeholder": "[PERSON_1]", "entity_type": "PERSON"},   # -2
        {"start": 20, "end": 31, "placeholder": "[HR_OIB_1]", "entity_type": "HR_OIB"},  # -1
    ]
    # p1 analyzed [0,10); p2 original 20 -> analyzed 18, [18,28). A pos past both
    # maps back by 2 + 1 = 3.
    assert _translate(30, spans) == 33


def test_translate_pos_inside_placeholder_returns_none():
    # analyzed placeholder occupies [9,19).
    assert _translate(13, _MARCUS) is None   # strictly inside
    assert _translate(9, _MARCUS) is None    # start boundary counts as inside
    assert _translate(19, _MARCUS) == 25     # end boundary is "after", not inside


# --------------------------------------------------------------------------- #
# Integration: full inlet flow
# --------------------------------------------------------------------------- #

def _body(chat_id, messages):
    return {"chat_id": chat_id, "messages": messages}


def _u(t):
    return {"role": "user", "content": t}


def _a(t):
    return {"role": "assistant", "content": t}


async def test_recall_gap_still_masks_known_name():
    """MAIN PROOF: turn 1 vaults 'Robert Plant'->[PERSON_1]. On turn 2 the NER
    recall gaps completely (GLiNER returns nothing), yet re-mask masks every
    occurrence of the known name deterministically from the vault snapshot."""
    pipe = make_gliner_pipeline(name_spans={"Robert Plant": "PERSON"})
    t1 = await pipe.inlet(_body(CHAT, [_u("Tko je Robert Plant")]), user=user_payload(True))
    assert t1["messages"][0]["content"] == "Tko je [PERSON_1]"

    # Simulate total NER recall gap on turn 2: GLiNER detects no real names.
    pipe._gliner = FakeGliner(name_spans={})
    t2 = await pipe.inlet(_body(CHAT, [_u("Robert Plant je super")]), user=user_payload(True))
    assert t2["messages"][0]["content"] == "[PERSON_1] je super"
    # No new vault row: re-mask reused the existing mapping.
    forward, _ = await pipe.vault.snapshot_for_request(CHAT)
    assert forward == {"Robert Plant": "[PERSON_1]"}


async def test_assistant_turn_history_is_masked():
    """E2E Robert-Plant repro: the leak lived in the ASSISTANT turn (real name,
    outlet-restored, never re-masked). With Option A the assistant history turn
    is re-masked before the LLM sees it."""
    pipe = make_gliner_pipeline(name_spans={"Robert Plant": "PERSON"})
    await pipe.inlet(_body(CHAT, [_u("Tko je Robert Plant")]), user=user_payload(True))  # seed [PERSON_1]

    pipe._gliner = FakeGliner(name_spans={})  # recall gap — only re-mask can catch it
    body = _body(
        CHAT,
        [
            _u("Tko je Robert Plant"),
            _a("Robert Plant je pjevac i frontmen Led Zeppelina."),
            _u("Daj mi zivotopis"),
        ],
    )
    out = await pipe.inlet(body, user=user_payload(True))
    assert out["messages"][0]["content"] == "Tko je [PERSON_1]"
    assert out["messages"][1]["content"] == "[PERSON_1] je pjevac i frontmen Led Zeppelina."
    assert "Robert Plant" not in out["messages"][1]["content"]


async def test_new_name_not_in_vault_caught_by_ner():
    """A name new to this turn is absent from the snapshot -> re-mask leaves it
    -> the normal NER path masks it."""
    pipe = make_gliner_pipeline(name_spans={"Jimmy Page": "PERSON"})
    out = await pipe.inlet(_body(CHAT, [_u("Tko je Jimmy Page")]), user=user_payload(True))
    assert out["messages"][0]["content"] == "Tko je [PERSON_1]"


async def test_step0_coordination_no_double_mask():
    """Re-mask inserts [PERSON_1]; GLiNER re-detects it as PERSON; Step-0 drops
    that overlap -> the placeholder appears exactly once, not renumbered."""
    pipe = make_gliner_pipeline(name_spans={"Robert Plant": "PERSON"})
    await pipe.inlet(_body(CHAT, [_u("Robert Plant")]), user=user_payload(True))  # [PERSON_1]

    # Default FakeGliner re-detects placeholders as PERSON (root-cause behavior).
    out = await pipe.inlet(_body(CHAT, [_u("Robert Plant opet")]), user=user_payload(True))
    content = out["messages"][0]["content"]
    assert content == "[PERSON_1] opet"
    assert content.count("[PERSON_1]") == 1
    assert "[PERSON_2]" not in content


async def test_idempotent_pure_placeholder_no_renumber():
    pipe = make_gliner_pipeline(name_spans={"Robert Plant": "PERSON"})
    await pipe.inlet(_body(CHAT, [_u("Robert Plant")]), user=user_payload(True))
    before_calls = len(pipe.vault.get_placeholder_calls)
    out = await pipe.inlet(_body(CHAT, [_u("[PERSON_1] i [PERSON_2]")]), user=user_payload(True))
    assert out["messages"][0]["content"] == "[PERSON_1] i [PERSON_2]"
    assert len(pipe.vault.get_placeholder_calls) == before_calls  # no new mints


# --------------------------------------------------------------------------- #
# Integration: PII card re-appears on repeated mentions (TRAU-522)
# --------------------------------------------------------------------------- #

async def test_card_shows_on_repeated_vaulted_name():
    """MAIN card fix: turn 1 vaults 'Robert Plant'->[PERSON_1]. On turn 2 NER
    recall gaps completely, but the deterministic re-mask now feeds the card, so
    the repeated mention still shows — span in ORIGINAL coords, type from the
    placeholder. The masking (security) path is unchanged."""
    pipe = make_gliner_pipeline(name_spans={"Robert Plant": "PERSON"})
    await pipe.inlet(_body(CHAT, [_u("Tko je Robert Plant")]), user=user_payload(True))  # seed [PERSON_1]

    pipe._gliner = FakeGliner(name_spans={})  # total NER recall gap on turn 2
    out = await pipe.inlet(_body(CHAT, [_u("Robert Plant je super")]), user=user_payload(True))

    # Security regression: the LLM still sees the placeholder on the repeat.
    assert out["messages"][0]["content"] == "[PERSON_1] je super"
    # Card regression restored: the repeat re-appears with correct original-coord span.
    public = out["metadata"]["pii_detections_public"]
    assert public == [{"type": "PERSON", "start": 0, "end": 12}]
    assert "Robert Plant je super"[0:12] == "Robert Plant"


async def test_card_first_mention_new_name_via_ner_unaffected():
    """Regression: a name new to this turn has no vault entry, so re-mask leaves
    it and the normal NER path both masks it and feeds the card as before."""
    pipe = make_gliner_pipeline(name_spans={"Jimmy Page": "PERSON"})
    out = await pipe.inlet(_body(CHAT, [_u("Tko je Jimmy Page")]), user=user_payload(True))
    assert out["messages"][0]["content"] == "Tko je [PERSON_1]"
    public = out["metadata"]["pii_detections_public"]
    assert public == [{"type": "PERSON", "start": 7, "end": 17}]
    assert "Tko je Jimmy Page"[7:17] == "Jimmy Page"


async def test_card_mixed_repeated_and_new_name_both_correct_offsets():
    """In a message mixing a repeated vaulted name with a NEW name, BOTH card
    spans must land in ORIGINAL coordinates. The repeat is re-masked (already
    original coords); the NEW name is detected by NER on the already-re-masked
    `analyzed` text, so its raw offset is shifted — the pipeline translates it
    back to the original message using the re-mask length deltas so the frontend
    slices the correct value."""
    pipe = make_gliner_pipeline(name_spans={"Robert Plant": "PERSON"})
    await pipe.inlet(_body(CHAT, [_u("Robert Plant")]), user=user_payload(True))  # seed [PERSON_1]

    pipe._gliner = FakeGliner(name_spans={"Jimmy Page": "PERSON"})
    text = "Robert Plant Jimmy Page"
    out = await pipe.inlet(_body(CHAT, [_u(text)]), user=user_payload(True))

    # Security intact: both names masked to the LLM.
    assert out["messages"][0]["content"] == "[PERSON_1] [PERSON_2]"

    public = sorted(out["metadata"]["pii_detections_public"], key=lambda d: d["start"])
    # Re-masked repeat (original coords) + new name (translated back to original).
    assert public == [
        {"type": "PERSON", "start": 0, "end": 12},
        {"type": "PERSON", "start": 13, "end": 23},
    ]
    assert text[0:12] == "Robert Plant"
    assert text[13:23] == "Jimmy Page"


async def test_card_two_turn_repeat_plus_new_name_offsets_correct():
    """Regression for the 'know Eleanor Fitz' report: turn 1 vaults
    'Marcus Thornbury'. On turn 2 Marcus is re-masked and 'Eleanor Fitzgerald'
    is new — the new name's NER offset (measured on the shortened, re-masked
    text) is translated back to the original message so the card shows the whole
    'Eleanor Fitzgerald', not the shifted 'know Eleanor Fitz'."""
    pipe = make_gliner_pipeline(name_spans={"Marcus Thornbury": "PERSON"})
    await pipe.inlet(
        _body(CHAT, [_u("Tell me about Marcus Thornbury")]), user=user_payload(True)
    )  # seed [PERSON_1]

    pipe._gliner = FakeGliner(name_spans={"Eleanor Fitzgerald": "PERSON"})
    text = "How does Marcus Thornbury know Eleanor Fitzgerald?"
    out = await pipe.inlet(
        _body(
            CHAT,
            [
                _u("Tell me about Marcus Thornbury"),
                _a("Marcus Thornbury is a fictional character."),
                _u(text),
            ],
        ),
        user=user_payload(True),
    )
    assert out["messages"][-1]["content"] == "How does [PERSON_1] know [PERSON_2]?"

    public = sorted(out["metadata"]["pii_detections_public"], key=lambda d: d["start"])
    assert public == [
        {"type": "PERSON", "start": 9, "end": 25},
        {"type": "PERSON", "start": 31, "end": 49},
    ]
    assert text[9:25] == "Marcus Thornbury"
    assert text[31:49] == "Eleanor Fitzgerald"
