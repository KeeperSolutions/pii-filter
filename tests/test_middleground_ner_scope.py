"""Middle-ground NER scope for MAIN CHAT turns (TRAU-522 latency/OOM follow-up).

Full NER (GLiNER + Presidio) runs ONLY on the last user and last assistant
message; every other history message gets the deterministic vault re-mask alone.
This makes main-chat inlet latency constant in history length (~2 NER jobs per
turn instead of one per history message) while keeping the full-NER backstop the
KORAK 5 task skip relies on:

  * last USER message — where new user-typed PII enters;
  * last ASSISTANT message — where an LLM-generated name first appears; NER-ing
    it vaults the name on the SAME turn, so the task re-mask can cover it.

Safety gate unchanged: the skip applies only when `remask_pattern is not None`.
An empty/disabled vault (first turn) yields no matcher, so EVERY targeted
message falls through to full NER — otherwise raw history would ship unmasked
(the Layer-2 leak).

Same spy technique as test_task_ner_skip.py: count analyzer/GLiNER calls to
prove exactly which messages were NER-ed, against the real `Pipeline.inlet`.
"""

from __future__ import annotations

from tests.conftest import FakeAnalyzer, FakeGliner, make_gliner_pipeline, user_payload

PERSON = "Ivan Horvat"          # vaulted on a prior turn in most tests
NEW_USER_NAME = "Ana Anić"      # new name typed in the LAST USER message
LLM_NAME = "Marko Marulić"      # new name produced in the LAST ASSISTANT message
CHAT_ID = "chat-middleground"


class _CountingGliner(FakeGliner):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = 0
        self.texts: list[str] = []

    def detect(self, text):
        self.calls += 1
        self.texts.append(text)
        return super().detect(text)


class _CountingAnalyzer(FakeAnalyzer):
    def __init__(self, spans):
        super().__init__(spans)
        self.calls = 0
        self.texts: list[str] = []

    def analyze(self, text, language):
        self.calls += 1
        self.texts.append(text)
        return super().analyze(text, language)


def _spy_pipeline(*, masking_enabled: bool = True, extra_names: dict[str, str] | None = None):
    """Counting GLiNER (production PERSON source) + counting no-op HR analyzer.

    Every NER-ed message part costs exactly 1 gliner.detect + 1 analyzer_hr.analyze
    (analyzer_en is None), so per-engine call counts == number of NER-ed parts.
    """
    spans = {PERSON: "PERSON", NEW_USER_NAME: "PERSON", LLM_NAME: "PERSON"}
    spans.update(extra_names or {})
    pipe = make_gliner_pipeline(masking_enabled=masking_enabled, name_spans=spans)
    pipe._gliner = _CountingGliner(name_spans=spans)
    pipe.analyzer_hr = _CountingAnalyzer({})
    return pipe


def _reset(pipe):
    pipe._gliner.calls = 0
    pipe._gliner.texts = []
    pipe.analyzer_hr.calls = 0
    pipe.analyzer_hr.texts = []


async def _vault_name(pipe, name: str = PERSON):
    """Prior main turn: vault `name` via full NER, then reset spy counters."""
    out = await pipe.inlet(
        {"chat_id": CHAT_ID, "messages": [{"role": "user", "content": f"Ja sam {name}."}]},
        user=user_payload(True),
    )
    assert "[PERSON_1]" in out["messages"][0]["content"]
    _reset(pipe)
    return out


def _main_body(messages: list[dict]) -> dict:
    return {"chat_id": CHAT_ID, "messages": messages}


# ---------------------------------------------------------------------------
# Latency spy: exactly the last user + last assistant message are NER-ed
# ---------------------------------------------------------------------------


async def test_long_history_ner_called_exactly_twice():
    """15-message history, non-empty vault: full NER runs on exactly TWO messages
    (last user + last assistant) — per-engine call count is 2, not 15. This is
    the latency win: NER cost is constant in history length."""
    pipe = _spy_pipeline()
    await _vault_name(pipe)

    history = []
    for i in range(6):  # 12 alternating history turns without new PII
        history.append({"role": "user", "content": f"Pitanje broj {i}."})
        history.append({"role": "assistant", "content": f"Odgovor broj {i}."})
    history.append({"role": "assistant", "content": f"Zadnji odgovor, {LLM_NAME} je autor."})
    history.append({"role": "user", "content": f"Tko je {NEW_USER_NAME}?"})
    assert len(history) == 14

    await pipe.inlet(_main_body(history), user=user_payload(True))

    assert pipe._gliner.calls == 2, f"gliner ran {pipe._gliner.calls}x: {pipe._gliner.texts}"
    assert pipe.analyzer_hr.calls == 2, f"presidio ran {pipe.analyzer_hr.calls}x"
    # ...and on the RIGHT two messages.
    assert any(NEW_USER_NAME in t or "[PERSON_" in t for t in pipe._gliner.texts)
    assert any(LLM_NAME in t or "[PERSON_" in t for t in pipe._gliner.texts)


async def test_new_name_in_last_user_message_is_masked():
    """New user-typed PII in the last user message is caught by NER and masked."""
    pipe = _spy_pipeline()
    await _vault_name(pipe)

    out = await pipe.inlet(
        _main_body(
            [
                {"role": "user", "content": "Prvo pitanje."},
                {"role": "assistant", "content": "Prvi odgovor."},
                {"role": "user", "content": f"Javi se {NEW_USER_NAME} sutra."},
            ]
        ),
        user=user_payload(True),
    )

    last = out["messages"][-1]["content"]
    assert NEW_USER_NAME not in last, last
    assert "[PERSON_" in last, last


async def test_new_name_in_last_assistant_message_masked_and_vaulted_same_turn():
    """The KORAK 5 backstop: an LLM-generated name in the last assistant message
    is NER-ed, masked AND vaulted on the SAME turn — so a follow-up task's
    re-mask (which skips NER) can still cover it."""
    pipe = _spy_pipeline()
    await _vault_name(pipe)

    out = await pipe.inlet(
        _main_body(
            [
                {"role": "user", "content": "Tko je autor?"},
                {"role": "assistant", "content": f"Autor je {LLM_NAME}."},
                {"role": "user", "content": "Reci mi više."},
            ]
        ),
        user=user_payload(True),
    )

    assistant = out["messages"][1]["content"]
    assert LLM_NAME not in assistant, assistant
    assert "[PERSON_" in assistant, assistant
    # Vaulted: a subsequent snapshot re-masks the same original to the same placeholder.
    forward, _ = await pipe.vault.snapshot_for_request(CHAT_ID)
    assert LLM_NAME in forward, forward


async def test_vaulted_name_in_history_remasked_without_ner():
    """A name vaulted on a prior turn recurs in OLD history messages: the
    deterministic re-mask replaces every occurrence (same placeholder), and those
    messages are NOT NER-ed (call count stays 2: last user + last assistant)."""
    pipe = _spy_pipeline()
    await _vault_name(pipe)

    out = await pipe.inlet(
        _main_body(
            [
                {"role": "user", "content": f"Ja sam {PERSON}."},
                {"role": "assistant", "content": f"Drago mi je, {PERSON}."},
                {"role": "user", "content": "Nastavimo."},
                {"role": "assistant", "content": "Može."},
                {"role": "user", "content": "Zadnje pitanje."},
            ]
        ),
        user=user_payload(True),
    )

    joined = "\n".join(m["content"] for m in out["messages"])
    assert PERSON not in joined, joined
    assert joined.count("[PERSON_1]") == 2, joined
    assert pipe._gliner.calls == 2, f"expected 2 NER-ed messages, got {pipe._gliner.calls}"
    assert pipe.analyzer_hr.calls == 2


# ---------------------------------------------------------------------------
# Layer-2 fallback: empty vault (first turn) => FULL NER on everything
# ---------------------------------------------------------------------------


async def test_empty_vault_first_turn_full_ner_on_all_messages():
    """KRITIČNO: with an EMPTY vault there is no re-mask matcher
    (`remask_pattern is None`), so skipping NER on history would ship raw
    unmasked content (the Layer-2 leak). Every targeted message must fall
    through to full NER, and names everywhere must be masked."""
    pipe = _spy_pipeline()  # vault enabled but EMPTY — no prior turn

    out = await pipe.inlet(
        _main_body(
            [
                {"role": "user", "content": f"Ja sam {PERSON}."},
                {"role": "assistant", "content": f"Bok, {PERSON}!"},
                {"role": "user", "content": "Kako si?"},
            ]
        ),
        user=user_payload(True),
    )

    assert pipe._gliner.calls == 3, (
        f"empty vault must run full NER on ALL messages, got {pipe._gliner.calls}"
    )
    joined = "\n".join(m["content"] for m in out["messages"])
    assert PERSON not in joined, joined


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


async def test_regeneration_last_message_is_assistant():
    """Regeneration: the payload ENDS with an assistant message. Both indices
    resolve correctly — the last user message (not messages[-1]) and the final
    assistant message are NER-ed; new names in both are masked."""
    pipe = _spy_pipeline()
    await _vault_name(pipe)

    out = await pipe.inlet(
        _main_body(
            [
                {"role": "user", "content": "Staro pitanje."},
                {"role": "assistant", "content": "Stari odgovor."},
                {"role": "user", "content": f"Tko je {NEW_USER_NAME}?"},
                {"role": "assistant", "content": f"{NEW_USER_NAME} radi s {LLM_NAME}."},
            ]
        ),
        user=user_payload(True),
    )

    assert pipe._gliner.calls == 2
    last_user = out["messages"][2]["content"]
    last_assistant = out["messages"][3]["content"]
    assert NEW_USER_NAME not in last_user, last_user
    assert NEW_USER_NAME not in last_assistant and LLM_NAME not in last_assistant, last_assistant


async def test_cap_one_last_assistant_outside_window():
    """multi_turn_history_max_messages=1: only the final message is targeted; the
    last-assistant index falls OUTSIDE the window. The unmatched set entry is
    harmless — no crash, NER runs once (on the in-window last user message)."""
    pipe = _spy_pipeline()
    await _vault_name(pipe)
    pipe.valves.multi_turn_history_max_messages = 1

    out = await pipe.inlet(
        _main_body(
            [
                {"role": "user", "content": "Staro pitanje."},
                {"role": "assistant", "content": f"Odgovor od {LLM_NAME}."},
                {"role": "user", "content": f"Pozdrav od {NEW_USER_NAME}."},
            ]
        ),
        user=user_payload(True),
    )

    assert pipe._gliner.calls == 1, f"cap=1 must NER exactly the window message, got {pipe._gliner.calls}"
    last = out["messages"][-1]["content"]
    assert NEW_USER_NAME not in last, last
    # Out-of-window assistant message is passed through (cap semantics, unchanged).
    assert LLM_NAME in out["messages"][1]["content"]


async def test_first_turn_single_user_message_no_assistant():
    """No assistant message at all (first turn): `last_assistant_idx == -1` must
    be inert — the single user message is NER-ed and masked, nothing crashes."""
    pipe = _spy_pipeline()

    out = await pipe.inlet(
        _main_body([{"role": "user", "content": f"Ja sam {PERSON}."}]),
        user=user_payload(True),
    )

    assert pipe._gliner.calls == 1
    assert "[PERSON_1]" in out["messages"][0]["content"]


# ---------------------------------------------------------------------------
# Regression guards (main interactions with existing seams)
# ---------------------------------------------------------------------------


async def test_task_skip_still_wins_over_ner_indices():
    """KORAK 5 unchanged: for an LLM-facing task the last user message IS in
    `ner_indices`, but `skip_ner_for_task` must still skip NER entirely."""
    pipe = _spy_pipeline()
    await _vault_name(pipe)

    out = await pipe.inlet(
        {
            "metadata": {"task": "follow_up_generation", "chat_id": CHAT_ID},
            "messages": [{"role": "user", "content": f"Povijest s {PERSON}."}],
        },
        user=user_payload(True),
    )

    assert pipe._gliner.calls == 0, "task allowlist skip must ignore ner_indices"
    content = out["messages"][0]["content"]
    assert PERSON not in content and "[PERSON_1]" in content, content


async def test_masking_off_short_circuit_unchanged():
    """masking OFF still short-circuits above everything: verbatim content, no
    NER, vault untouched."""
    pipe = _spy_pipeline(masking_enabled=False)
    original = [
        {"role": "user", "content": f"Ja sam {PERSON}."},
        {"role": "assistant", "content": f"Bok, {PERSON}!"},
        {"role": "user", "content": "Kako si?"},
    ]

    out = await pipe.inlet(_main_body([dict(m) for m in original]), user=user_payload(False))

    assert [m["content"] for m in out["messages"]] == [m["content"] for m in original]
    assert pipe._gliner.calls == 0 and pipe.analyzer_hr.calls == 0
    assert pipe.vault.get_placeholder_calls == []
