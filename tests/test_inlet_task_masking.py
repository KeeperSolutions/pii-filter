"""Layer-2 fix (TRAU-522) + KORAK 5 refinement: background-task payloads honor
the per-chat PII masking toggle, and LLM-facing tasks skip the NER detection pass.

Layer-2 removed the old unconditional `metadata.task` early-return (which shipped
raw PII to the external LLM even with masking ON). KORAK 5 then narrows the
handling: LLM-facing tasks (title / tags / follow-ups), whose output feeds back to
the SAME LLM, skip the heavy GLiNER + Presidio NER pass and rely on the
deterministic vault re-mask ALONE (OOM mitigation). Consequences under test here:

  * masking OFF still short-circuits above the seam (toggle honored);
  * a title/tags/follow-up task masks names ALREADY vaulted (re-mask), but does
    NOT detect a brand-new name (NER skipped) — accepted, since a name new to the
    payload entered via the main chat, whose full NER runs and vaults it;
  * external-facing tasks (queries / image_prompt) and normal chat turns keep full
    NER — covered in test_task_ner_skip.py.

These tests wire real Pipeline control flow against an in-memory analyzer + vault
(see conftest.py). Detection and the Postgres vault are the only fakes; the
inlet gating, detection selection, and splice logic under test are the real code.
"""

from __future__ import annotations

from tests.conftest import make_pipeline, user_payload

PERSON = "Ivan Horvat"
CHAT_ID = "chat-abc-123"


def _task_body(chat_id: str, text: str, task: str = "title_generation") -> dict:
    """A background-task payload as tasks.py builds it: metadata.task is truthy
    and chat_id lives under metadata (not top-level)."""
    return {
        "metadata": {"task": task, "chat_id": chat_id},
        "messages": [{"role": "user", "content": text}],
    }


def _main_body(chat_id: str, text: str) -> dict:
    """A primary chat-completion payload: chat_id at top level, no task."""
    return {
        "chat_id": chat_id,
        "messages": [{"role": "user", "content": text}],
    }


async def test_llm_facing_task_empty_vault_runs_ner():
    """KORAK 5 safety guard: the NER-skip is gated on an ACTIVE re-mask
    (`remask_pattern is not None`). With an empty thread snapshot the matcher is
    inert, so skipping NER would ship raw content unmasked — instead the task
    falls through to full NER and the new name IS masked. This prevents the skip
    from reopening the Layer-2 leak on a first-turn / empty-vault task."""
    pipe = make_pipeline(masking_enabled=True)  # analyzer detects PERSON
    body = _task_body(CHAT_ID, f"Poruka za {PERSON} danas.")

    out = await pipe.inlet(body, user=user_payload(True))

    content = out["messages"][0]["content"]
    # Empty vault -> no re-mask matcher -> NER runs -> new name is masked.
    assert "[PERSON_1]" in content, content
    assert PERSON not in content, content
    assert pipe.vault.get_placeholder_calls, "NER should have run and minted a row"
    forward, _reverse = await pipe.vault.snapshot_for_request(CHAT_ID)
    assert forward.get(PERSON) == "[PERSON_1]", forward


async def test_task_payload_masking_off_is_untouched():
    """masking OFF + background task -> short-circuits at the pii_masking_enabled
    gate: content is returned verbatim and the vault is never touched (toggle
    respected in the OFF direction)."""
    pipe = make_pipeline(masking_enabled=False)
    original_text = f"Poruka za {PERSON} danas."
    body = _task_body(CHAT_ID, original_text)

    out = await pipe.inlet(body, user=user_payload(False))

    content = out["messages"][0]["content"]
    assert content == original_text, content
    assert PERSON in content
    assert pipe.vault.get_placeholder_calls == []
    forward, _reverse = await pipe.vault.snapshot_for_request(CHAT_ID)
    assert forward == {}


async def test_repeated_entity_reuses_placeholder_across_main_and_task():
    """The same (chat_id, entity_type, original) minted by the main chat is
    reused by a later background task sharing metadata.chat_id — same [PERSON_1],
    no renumbering. Proves vault-thread consistency across the two payload shapes."""
    pipe = make_pipeline(masking_enabled=True)  # single shared FakeVault

    # 1) Main chat turn masks the person first.
    main = await pipe.inlet(
        _main_body(CHAT_ID, f"Pozdrav, ja sam {PERSON}."),
        user=user_payload(True),
    )
    main_content = main["messages"][0]["content"]
    assert "[PERSON_1]" in main_content and PERSON not in main_content

    # 2) A background task on the SAME chat references the same person.
    task = await pipe.inlet(
        _task_body(CHAT_ID, f"Naslov razgovora s {PERSON}."),
        user=user_payload(True),
    )
    task_content = task["messages"][0]["content"]

    # Reuse, not renumber: the task must get [PERSON_1], never [PERSON_2].
    assert "[PERSON_1]" in task_content, task_content
    assert "[PERSON_2]" not in task_content, task_content
    assert PERSON not in task_content, task_content

    forward, _reverse = await pipe.vault.snapshot_for_request(CHAT_ID)
    assert forward == {PERSON: "[PERSON_1]"}


async def test_main_chat_across_different_chat_gets_own_thread():
    """Sanity: the same name in a different chat_id mints its own [PERSON_1]
    (per-chat thread isolation), keyed on chat_id, not global. Uses main-chat
    turns on both threads — the KORAK 5 NER-skip applies only to LLM-facing
    tasks, so a task on chat-B with an unvaulted name would not mint anything;
    isolation is a vault property, exercised here via the full-NER main path."""
    pipe = make_pipeline(masking_enabled=True)

    await pipe.inlet(_main_body("chat-A", f"Ja sam {PERSON}."), user=user_payload(True))
    other = await pipe.inlet(
        _main_body("chat-B", f"Ja sam {PERSON} isto."), user=user_payload(True)
    )

    assert "[PERSON_1]" in other["messages"][0]["content"]
    fwd_b, _ = await pipe.vault.snapshot_for_request("chat-B")
    assert fwd_b == {PERSON: "[PERSON_1]"}
