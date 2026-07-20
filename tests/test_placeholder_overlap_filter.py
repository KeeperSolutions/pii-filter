"""TRAU-522 follow-up: placeholder overlap-filter + already-masked-skip removal.

Recon proved GLiNER (neural) re-detects existing placeholders as entities
([PERSON_1]->PERSON 0.862, [ADDRESS_1]->ADDRESS 0.937), while regex/checksum
recognizers never do. Idempotency of re-analysis previously rested 100% on the
per-message already-masked skip, which leaks on mixed content (placeholder + a
new name in the same message, e.g. embedded follow-up history).

The fix drops any detection whose span OVERLAPS an existing `[TYPE_N]`
placeholder (computed via `_PLACEHOLDER_RE`), then removes the skip so every
message is analyzed. Adjacent real names (non-overlapping spans) are untouched.

Two layers of test:
  * unit — `_select_accepted_detections` filtering, independent of the skip.
  * integration — full inlet flow, proving idempotency + vault parity after the
    skip is removed.
"""

from __future__ import annotations

from tests.conftest import RecognizerResult, make_gliner_pipeline, pii_mod, user_payload

_select = pii_mod._select_accepted_detections
STD = pii_mod.Pipeline.PRESIDIO_TO_STANDARD

PERSON = "Jimmy Page"
CHAT = "chat-zeppelin-1"


def _rr(start, end, etype="PERSON", score=0.9):
    return RecognizerResult(entity_type=etype, start=start, end=end, score=score)


# --------------------------------------------------------------------------- #
# Unit: _select_accepted_detections placeholder overlap filtering
# --------------------------------------------------------------------------- #

def test_select_drops_detection_exactly_on_placeholder():
    text = "[PERSON_1] met someone"
    accepted = _select(text, [_rr(0, 10)], STD)
    assert accepted == []


def test_select_drops_detection_partially_overlapping_placeholder():
    # GLiNER returns a span that engulfs the placeholder plus trailing text.
    text = "[PERSON_1] and friends"
    accepted = _select(text, [_rr(0, 14)], STD)  # "[PERSON_1] and"
    assert accepted == []


def test_select_keeps_adjacent_real_name_next_to_placeholder():
    text = "[PERSON_1] Jimmy Page"  # placeholder (0,10), name (11,21)
    dets = [_rr(0, 10), _rr(11, 21)]
    accepted = _select(text, dets, STD)
    assert len(accepted) == 1
    survivor = accepted[0]
    assert (survivor.start, survivor.end) == (11, 21)
    assert text[survivor.start : survivor.end] == "Jimmy Page"


def test_select_no_placeholder_is_unaffected():
    text = "Jimmy Page plays"
    accepted = _select(text, [_rr(0, 10)], STD)
    assert len(accepted) == 1
    assert (accepted[0].start, accepted[0].end) == (0, 10)


def test_select_multiple_placeholders_all_dropped_name_kept():
    text = "[PERSON_1] met [PERSON_2] and Jimmy Page"
    dets = [_rr(0, 10), _rr(15, 25), _rr(30, 40)]  # two placeholders + name
    accepted = _select(text, dets, STD)
    assert len(accepted) == 1
    assert text[accepted[0].start : accepted[0].end] == "Jimmy Page"


# --------------------------------------------------------------------------- #
# Integration: full inlet flow (requires skip removed + filter active)
# --------------------------------------------------------------------------- #

def _body(chat_id, text):
    return {"chat_id": chat_id, "messages": [{"role": "user", "content": text}]}


async def test_pure_placeholder_message_is_idempotent():
    """MAIN PROOF: a message that is only placeholders is returned UNCHANGED with
    ZERO new vault rows — re-analysis mints nothing, restore stays intact."""
    pipe = make_gliner_pipeline(masking_enabled=True)
    out = await pipe.inlet(_body(CHAT, "[PERSON_1] met [PERSON_2]"), user=user_payload(True))
    assert out["messages"][0]["content"] == "[PERSON_1] met [PERSON_2]"
    assert pipe.vault.get_placeholder_calls == []
    forward, _ = await pipe.vault.snapshot_for_request(CHAT)
    assert forward == {}


async def test_mixed_content_masks_new_name_keeps_placeholder():
    """[PERSON_1] (already someone else in the vault) + a NEW name -> the new name
    gets the next placeholder, [PERSON_1] is untouched, both coexist in the vault."""
    pipe = make_gliner_pipeline(masking_enabled=True, name_spans={"John Bonham": "PERSON", "Jimmy Page": "PERSON"})

    # Turn 1: John Bonham -> [PERSON_1]
    t1 = await pipe.inlet(_body(CHAT, "John Bonham"), user=user_payload(True))
    assert t1["messages"][0]["content"] == "[PERSON_1]"

    # Turn 2 (follow-up-shaped): history placeholder + a new name.
    t2 = await pipe.inlet(_body(CHAT, "[PERSON_1] and Jimmy Page"), user=user_payload(True))
    assert t2["messages"][0]["content"] == "[PERSON_1] and [PERSON_2]"

    forward, reverse = await pipe.vault.snapshot_for_request(CHAT)
    assert forward == {"John Bonham": "[PERSON_1]", "Jimmy Page": "[PERSON_2]"}
    assert reverse["[PERSON_1]"] == "John Bonham"  # untouched, not renumbered


async def test_overlap_span_engulfing_placeholder_leaves_it_literal():
    """When the neural model returns a span covering placeholder + text, the whole
    detection is dropped: the placeholder stays literal, nothing renumbered."""
    pipe = make_gliner_pipeline(masking_enabled=True)

    class EngulfingGliner:
        def detect(self, text):
            # one span covering "[PERSON_1] and" (0..14)
            return [RecognizerResult(entity_type="PERSON", start=0, end=14, score=0.9)]

    pipe._gliner = EngulfingGliner()
    out = await pipe.inlet(_body(CHAT, "[PERSON_1] and rest"), user=user_payload(True))
    assert out["messages"][0]["content"] == "[PERSON_1] and rest"
    assert pipe.vault.get_placeholder_calls == []


async def test_adjacent_new_name_still_masked():
    """A brand-new name sitting right next to a placeholder is still masked."""
    pipe = make_gliner_pipeline(
        masking_enabled=True, name_spans={"John Bonham": "PERSON", "Jimmy Page": "PERSON"}
    )
    # Seed [PERSON_1] = someone else so the new name becomes [PERSON_2].
    await pipe.inlet(_body(CHAT, "John Bonham"), user=user_payload(True))  # -> [PERSON_1]
    out = await pipe.inlet(_body(CHAT, "[PERSON_1], Jimmy Page"), user=user_payload(True))
    assert out["messages"][0]["content"] == "[PERSON_1], [PERSON_2]"


async def test_followup_shaped_history_masks_only_new_names():
    """E2E repro of the Jimmy Page follow-up: embedded history with several
    existing placeholders + one new name -> only the new name is masked, every
    existing placeholder is byte-for-byte preserved."""
    pipe = make_gliner_pipeline(
        masking_enabled=True,
        name_spans={"John Bonham": "PERSON", "Robert Plant": "PERSON", "Jimmy Page": "PERSON"},
    )
    # Pre-seed [PERSON_1] and [PERSON_2] from earlier turns.
    await pipe.inlet(_body(CHAT, "John Bonham"), user=user_payload(True))     # [PERSON_1]
    await pipe.inlet(_body(CHAT, "Robert Plant"), user=user_payload(True))    # [PERSON_2]

    history = (
        "User: tko je [PERSON_1]? "
        "Assistant: [PERSON_1] i [PERSON_2] su bili u bendu. "
        "User: a Jimmy Page?"
    )
    out = await pipe.inlet(_body(CHAT, history), user=user_payload(True))
    content = out["messages"][0]["content"]
    assert "[PERSON_1]" in content and "[PERSON_2]" in content  # preserved
    assert "Jimmy Page" not in content  # new name masked
    assert "[PERSON_3]" in content       # Jimmy Page got the next free number
    forward, _ = await pipe.vault.snapshot_for_request(CHAT)
    assert forward == {
        "John Bonham": "[PERSON_1]",
        "Robert Plant": "[PERSON_2]",
        "Jimmy Page": "[PERSON_3]",
    }


async def test_regression_plain_name_still_masks():
    """No placeholders in the message: plain name masking is unchanged."""
    pipe = make_gliner_pipeline(masking_enabled=True, name_spans={"Jimmy Page": "PERSON"})
    out = await pipe.inlet(_body(CHAT, "Jimmy Page"), user=user_payload(True))
    assert out["messages"][0]["content"] == "[PERSON_1]"
