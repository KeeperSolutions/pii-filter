"""KORAK 5: LLM-facing background tasks skip the NER detection pass.

title / tags / follow-ups run the deterministic vault re-mask ONLY (GLiNER +
Presidio skipped) — this collapses ~4 heavy NER jobs per chat turn to 1 and fixes
the OOM spike. Safety rests on scope: the skip is an allowlist of LLM-facing task
types (output consumed by the SAME LLM). External-facing tasks (query_generation,
image_prompt_generation) send output to a third-party service where an un-vaulted
name would leak, so they KEEP full NER. Normal chat turns keep full NER too.

These tests spy on the analyzer/GLiNER call count to prove NER is (not) invoked,
against the real `Pipeline.inlet` control flow (only detection + vault are fakes).
"""

from __future__ import annotations

import pytest

from tests.conftest import FakeAnalyzer, FakeGliner, make_gliner_pipeline, user_payload

PERSON = "Ivan Horvat"
CHAT_ID = "chat-korak5"

LLM_FACING = ["title_generation", "tags_generation", "follow_up_generation"]
EXTERNAL_FACING = ["query_generation", "image_prompt_generation"]


class _CountingGliner(FakeGliner):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = 0

    def detect(self, text):
        self.calls += 1
        return super().detect(text)


class _CountingAnalyzer(FakeAnalyzer):
    def __init__(self, spans):
        super().__init__(spans)
        self.calls = 0

    def analyze(self, text, language):
        self.calls += 1
        return super().analyze(text, language)


def _spy_pipeline(*, masking_enabled: bool = True, name: str = PERSON):
    """Pipeline whose PERSON source is a call-counting GLiNER (production shape:
    GLiNER on, spaCy NER off) plus a counting HR analyzer (returns [] but its
    call count reveals whether the Presidio pass ran)."""
    pipe = make_gliner_pipeline(masking_enabled=masking_enabled, name_spans={name: "PERSON"})
    pipe._gliner = _CountingGliner(name_spans={name: "PERSON"})
    pipe.analyzer_hr = _CountingAnalyzer({})
    return pipe


def _ner_calls(pipe) -> int:
    return pipe._gliner.calls + pipe.analyzer_hr.calls


def _main_body(chat_id: str, text: str) -> dict:
    return {"chat_id": chat_id, "messages": [{"role": "user", "content": text}]}


def _task_body(chat_id: str, text: str, task: str) -> dict:
    return {
        "metadata": {"task": task, "chat_id": chat_id},
        "messages": [{"role": "user", "content": text}],
    }


# ---------------------------------------------------------------------------
# LLM-facing tasks: NER skipped, re-mask still masks vaulted names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", LLM_FACING)
async def test_llm_facing_task_remasks_vaulted_name_without_calling_ner(task):
    """A name vaulted on a prior main turn is re-masked to the SAME [PERSON_1] in
    the task, and NER (GLiNER + Presidio) is NOT called for the task."""
    pipe = _spy_pipeline()

    # Main chat turn vaults the name via full NER.
    main = await pipe.inlet(_main_body(CHAT_ID, f"Ja sam {PERSON}."), user=user_payload(True))
    assert "[PERSON_1]" in main["messages"][0]["content"]
    assert _ner_calls(pipe) > 0, "main chat must run NER"

    # Reset counters, then fire the LLM-facing task on the same chat.
    pipe._gliner.calls = 0
    pipe.analyzer_hr.calls = 0

    out = await pipe.inlet(_task_body(CHAT_ID, f"Naslov s {PERSON}.", task), user=user_payload(True))
    content = out["messages"][0]["content"]

    # Re-mask masked the vaulted name to the identical placeholder...
    assert "[PERSON_1]" in content, content
    assert PERSON not in content, content
    # ...and NER was skipped entirely for the task.
    assert _ner_calls(pipe) == 0, f"NER was called for LLM-facing task {task!r}"


@pytest.mark.parametrize("task", LLM_FACING)
async def test_llm_facing_task_empty_vault_falls_through_to_ner(task):
    """Safety guard (`remask_pattern is not None`): an EMPTY vault snapshot yields
    no re-mask matcher, so skipping NER would ship raw content unmasked. The task
    must fall through to full NER — proven by a non-zero call count — and the new
    name is masked. This keeps the KORAK 5 skip from reopening the Layer-2 leak."""
    pipe = _spy_pipeline()  # vault enabled but EMPTY (no prior turn vaulted anything)

    out = await pipe.inlet(_task_body(CHAT_ID, f"Naslov s {PERSON}.", task), user=user_payload(True))
    content = out["messages"][0]["content"]

    assert _ner_calls(pipe) > 0, f"NER was skipped on an empty vault for {task!r}"
    assert "[PERSON_1]" in content, content
    assert PERSON not in content, content


async def test_llm_facing_task_vault_disabled_falls_through_to_ner():
    """Safety guard: with the vault disabled there is no re-mask at all, so an
    LLM-facing task must run full NER (per-request masking) rather than skip and
    leak. NER call count proves it ran."""
    pipe = _spy_pipeline()
    pipe.valves.vault_enabled = False
    pipe.vault = None

    out = await pipe.inlet(
        _task_body(CHAT_ID, f"Naslov s {PERSON}.", "title_generation"), user=user_payload(True)
    )
    content = out["messages"][0]["content"]

    assert _ner_calls(pipe) > 0, "NER was skipped with the vault disabled"
    assert "[PERSON_1]" in content, content
    assert PERSON not in content, content


@pytest.mark.parametrize("task", LLM_FACING)
async def test_llm_facing_task_masking_off_short_circuits(task):
    """masking OFF short-circuits ABOVE the KORAK 5 seam: content is verbatim, the
    vault is untouched, and NER is never called (regression on the toggle gate)."""
    pipe = _spy_pipeline(masking_enabled=False)
    original = f"Naslov s {PERSON}."

    out = await pipe.inlet(_task_body(CHAT_ID, original, task), user=user_payload(False))

    assert out["messages"][0]["content"] == original
    assert PERSON in out["messages"][0]["content"]
    assert _ner_calls(pipe) == 0
    assert pipe.vault.get_placeholder_calls == []


# ---------------------------------------------------------------------------
# External-facing tasks and main chat: full NER retained
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", EXTERNAL_FACING)
async def test_external_facing_task_keeps_full_ner(task):
    """queries / image_prompt go to a third-party service: NER MUST still run so a
    new (un-vaulted) name is detected and masked before it leaves the pipeline."""
    pipe = _spy_pipeline()

    out = await pipe.inlet(_task_body(CHAT_ID, f"Traži {PERSON} online.", task), user=user_payload(True))
    content = out["messages"][0]["content"]

    assert _ner_calls(pipe) > 0, f"NER was skipped for external-facing task {task!r}"
    assert "[PERSON_1]" in content, content
    assert PERSON not in content, content


async def test_main_chat_keeps_full_ner():
    """A normal chat turn (no metadata.task) always runs full NER."""
    pipe = _spy_pipeline()

    out = await pipe.inlet(_main_body(CHAT_ID, f"Ja sam {PERSON}."), user=user_payload(True))

    assert _ner_calls(pipe) > 0
    assert "[PERSON_1]" in out["messages"][0]["content"]


async def test_unknown_task_type_keeps_full_ner():
    """Allowlist, not `metadata.task`-generic: an unrecognized task keeps NER."""
    pipe = _spy_pipeline()

    out = await pipe.inlet(
        _task_body(CHAT_ID, f"Ja sam {PERSON}.", "emoji_generation"), user=user_payload(True)
    )

    assert _ner_calls(pipe) > 0, "unknown task type must not skip NER"
    assert "[PERSON_1]" in out["messages"][0]["content"]


# ---------------------------------------------------------------------------
# Cross-payload consistency
# ---------------------------------------------------------------------------


async def test_followup_flattened_history_blob_remasks_all_without_ner():
    """Realistic follow_up shape: OWUI flattens the whole conversation into ONE
    user message's content. The vaulted name recurs several times across that blob
    (user AND assistant turns embedded); re-mask must mask EVERY occurrence to
    [PERSON_1] with NER still skipped. This is the large/flattened payload that
    drives the OOM — the single-message happy path did not exercise it."""
    pipe = _spy_pipeline()

    # Vault the name via a real main turn.
    await pipe.inlet(_main_body(CHAT_ID, f"Ja sam {PERSON}."), user=user_payload(True))
    pipe._gliner.calls = 0
    pipe.analyzer_hr.calls = 0

    # OWUI follow_up prompt: history flattened into a single templated blob.
    blob = (
        "### Chat History:\n"
        f"Korisnik: Ja sam {PERSON}.\n"
        f"Asistent: Drago mi je, {PERSON}.\n"
        "Korisnik: Ispričaj mi vic.\n"
        f"Asistent: Naravno {PERSON}, evo jednog.\n"
        "### Task:\nPredloži 3 follow-up pitanja.\n### Output:"
    )
    assert blob.count(PERSON) == 3  # sanity: the name is in there 3x

    out = await pipe.inlet(
        _task_body(CHAT_ID, blob, "follow_up_generation"), user=user_payload(True)
    )
    content = out["messages"][0]["content"]

    # EVERY occurrence in the flattened blob is re-masked; none leaks.
    assert PERSON not in content, content
    assert content.count("[PERSON_1]") == 3, content
    # NER never ran despite the large payload.
    assert _ner_calls(pipe) == 0, "NER was called on the flattened follow_up blob"


async def test_followup_six_message_payload_remasks_every_turn_without_ner():
    """Realistic follow_up shape variant: the payload carries the actual 6-message
    history (user + assistant turns), not a single blob. Every turn containing the
    vaulted name is re-masked to [PERSON_1], across both roles, with NER skipped."""
    pipe = _spy_pipeline()

    await pipe.inlet(_main_body(CHAT_ID, f"Ja sam {PERSON}."), user=user_payload(True))
    pipe._gliner.calls = 0
    pipe.analyzer_hr.calls = 0

    history = [
        {"role": "user", "content": f"Ja sam {PERSON}."},
        {"role": "assistant", "content": f"Bok {PERSON}!"},
        {"role": "user", "content": "Kako si?"},
        {"role": "assistant", "content": "Dobro, hvala."},
        {"role": "user", "content": f"Reci nešto o {PERSON}."},
        {"role": "assistant", "content": f"{PERSON} je super."},
    ]
    body = {
        "metadata": {"task": "follow_up_generation", "chat_id": CHAT_ID},
        "messages": history,
    }

    out = await pipe.inlet(body, user=user_payload(True))
    contents = [m["content"] for m in out["messages"]]

    # The name appeared in 4 of the 6 turns (msgs 0,1,4,5) — all re-masked.
    joined = "\n".join(contents)
    assert PERSON not in joined, joined
    assert joined.count("[PERSON_1]") == 4, joined
    # Untouched turns stay verbatim (no spurious masking).
    assert contents[2] == "Kako si?" and contents[3] == "Dobro, hvala."
    # NER skipped for the whole multi-message task.
    assert _ner_calls(pipe) == 0, "NER was called on the 6-message follow_up payload"


async def test_vaulted_name_consistent_task_matches_main_chat():
    """The task re-mask yields the exact placeholder the main chat minted."""
    pipe = _spy_pipeline()

    main = await pipe.inlet(_main_body(CHAT_ID, f"Pozdrav, {PERSON}."), user=user_payload(True))
    main_ph = "[PERSON_1]"
    assert main_ph in main["messages"][0]["content"]

    task = await pipe.inlet(
        _task_body(CHAT_ID, f"Sažetak razgovora s {PERSON}.", "tags_generation"),
        user=user_payload(True),
    )
    assert main_ph in task["messages"][0]["content"], task["messages"][0]["content"]
    assert "[PERSON_2]" not in task["messages"][0]["content"]
