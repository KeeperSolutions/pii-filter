"""NER offload (KORAK 3 + 4): the synchronous, CPU-bound, NOT-thread-safe NER
calls (Presidio spaCy `.analyze()`, GLiNER torch `.detect()`) must run on a
dedicated single-worker executor, off the event loop.

Root cause recap (read-only recon): those calls were awaited inline on the sole
uvicorn event loop, so a large prompt blocked the loop and other requests timed
out. The fix offloads each call via `Pipeline._offload_ner` onto a
`ThreadPoolExecutor(max_workers=1)`. Two properties must both hold:

  * event loop stays free while an inference runs (offload works), and
  * inferences never run concurrently on the shared model instances
    (`max_workers=1` serializes them) — the models are not thread-safe.

These tests wire real `Pipeline.inlet` control flow against instrumented fakes
(see conftest for the base fakes). Only detection + vault are faked; the offload
path, serialization, and splice logic under test are the real code.
"""

from __future__ import annotations

import asyncio
import threading
import time

from tests.conftest import (
    _REAL_GLINER_LOAD,
    FakeAnalyzer,
    FakeGliner,
    make_gliner_pipeline,
    pii_mod,
    user_payload,
)

CHAT_ID = "chat-offload-1"


def _body(chat_id: str, text: str) -> dict:
    return {"chat_id": chat_id, "messages": [{"role": "user", "content": text}]}


# ---------------------------------------------------------------------------
# KORAK 3 — thread limit
# ---------------------------------------------------------------------------


def test_ner_thread_limit_default_is_four(monkeypatch):
    """Default (env unset) is 4 — the Cloud Run vCPU allocation."""
    monkeypatch.delenv("PII_FILTER_NER_TORCH_THREADS", raising=False)
    assert pii_mod._ner_thread_limit() == 4


def test_ner_thread_limit_env_override(monkeypatch):
    monkeypatch.setenv("PII_FILTER_NER_TORCH_THREADS", "1")
    assert pii_mod._ner_thread_limit() == 1
    monkeypatch.setenv("PII_FILTER_NER_TORCH_THREADS", "8")
    assert pii_mod._ner_thread_limit() == 8


def test_ner_thread_limit_malformed_falls_back_to_one(monkeypatch):
    monkeypatch.setenv("PII_FILTER_NER_TORCH_THREADS", "not-a-number")
    assert pii_mod._ner_thread_limit() == 1
    monkeypatch.setenv("PII_FILTER_NER_TORCH_THREADS", "0")
    assert pii_mod._ner_thread_limit() == 1
    monkeypatch.setenv("PII_FILTER_NER_TORCH_THREADS", "-3")
    assert pii_mod._ner_thread_limit() == 1


def test_gliner_load_caps_torch_threads(monkeypatch):
    """`GLiNER2Detector.load()` calls `torch.set_num_threads(_ner_thread_limit())`.

    torch/gliner2 are not installed in the test venv, so inject fakes into
    sys.modules and assert the cap is applied on load with the configured value.
    """
    import sys
    import types

    # conftest's session-wide `_stub_gliner_model` replaces `load()` with a
    # no-op so unrelated tests can run on_startup without torch. This test is
    # about what the genuine `load()` does, so restore it for this test only.
    monkeypatch.setattr(pii_mod.GLiNER2Detector, "load", _REAL_GLINER_LOAD)

    monkeypatch.setenv("PII_FILTER_NER_TORCH_THREADS", "3")

    recorded: dict[str, int] = {}
    fake_torch = types.ModuleType("torch")
    fake_torch.set_num_threads = lambda n: recorded.__setitem__("n", n)  # type: ignore[attr-defined]

    class _FakeModel:
        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

    fake_gliner2 = types.ModuleType("gliner2")
    fake_gliner2.GLiNER2 = _FakeModel  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "gliner2", fake_gliner2)

    det = pii_mod.GLiNER2Detector()
    det.load()

    assert recorded.get("n") == 3, "torch.set_num_threads not called with the env value"


# ---------------------------------------------------------------------------
# KORAK 4 — offload correctness
# ---------------------------------------------------------------------------


class _ThreadRecordingAnalyzer(FakeAnalyzer):
    """FakeAnalyzer that records which thread its `analyze` ran on."""

    def __init__(self, spans, seen_threads):
        super().__init__(spans)
        self._seen = seen_threads

    def analyze(self, text, language):
        self._seen.append(threading.current_thread().name)
        return super().analyze(text, language)


async def test_ner_runs_off_the_main_thread_and_result_is_identical():
    """analyze/detect execute on the pii-ner worker thread (not the loop thread),
    and masking output is identical to the synchronous behavior."""
    main_thread = threading.current_thread().name
    seen: list[str] = []

    pipe = make_gliner_pipeline(name_spans={"Jimmy Page": "PERSON"})
    pipe.analyzer_hr = _ThreadRecordingAnalyzer({}, seen)

    out = await pipe.inlet(_body(CHAT_ID, "Poruka za Jimmy Page danas."), user=user_payload(True))

    content = out["messages"][0]["content"]
    assert "[PERSON_1]" in content, content
    assert "Jimmy Page" not in content, content
    # The Presidio analyze ran on the dedicated worker thread, never the loop.
    assert seen, "analyzer.analyze was never called"
    assert all(name != main_thread for name in seen), seen
    assert all(name.startswith("pii-ner") for name in seen), seen


async def test_ner_calls_are_serialized_max_concurrency_one():
    """Two concurrent inlet requests must NOT run NER simultaneously — the shared
    models are not thread-safe, and the single-worker executor guarantees it.

    The fake detect() tracks live concurrency; with max_workers=1 the observed
    maximum is exactly 1.
    """
    state = {"live": 0, "max": 0}
    lock = threading.Lock()

    class _ConcurrencyProbeGliner(FakeGliner):
        def detect(self, text):
            with lock:
                state["live"] += 1
                state["max"] = max(state["max"], state["live"])
            time.sleep(0.05)  # simulate a CPU-bound inference
            with lock:
                state["live"] -= 1
            return super().detect(text)

    # Two independent pipelines sharing ONE executor would still serialize, but
    # to model "same shared model instance" we use one pipeline for both requests.
    pipe = make_gliner_pipeline(name_spans={"Ann": "PERSON"})
    pipe._gliner = _ConcurrencyProbeGliner(name_spans={"Ann": "PERSON"})

    await asyncio.gather(
        pipe.inlet(_body("chat-A", "hello Ann one"), user=user_payload(True)),
        pipe.inlet(_body("chat-B", "hello Ann two"), user=user_payload(True)),
    )

    assert state["max"] == 1, f"NER ran concurrently (max live={state['max']}) — not thread-safe"


async def test_event_loop_not_blocked_during_ner():
    """While a slow NER inference runs on the worker thread, another coroutine on
    the event loop must make progress. If the call were inline/sync, the loop
    would be blocked and the releasing coroutine could never run -> the gate is
    never set and detect() waits forever (caught by the outer timeout).
    """
    ner_entered = asyncio.Event()
    release = threading.Event()
    progressed = {"value": False}
    loop = asyncio.get_running_loop()

    class _BlockingGliner(FakeGliner):
        def detect(self, text):
            # Signal (thread-safe) that we're inside the offloaded call, then
            # block the WORKER thread until the loop-side coroutine releases us.
            loop.call_soon_threadsafe(ner_entered.set)
            assert release.wait(timeout=5), "release was never signaled — loop blocked"
            return super().detect(text)

    pipe = make_gliner_pipeline(name_spans={"Ann": "PERSON"})
    pipe._gliner = _BlockingGliner(name_spans={"Ann": "PERSON"})

    async def other_coroutine():
        # Only runnable if the event loop is NOT blocked by the NER call.
        await ner_entered.wait()
        progressed["value"] = True
        release.set()  # let the worker thread finish

    inlet_task = asyncio.create_task(
        pipe.inlet(_body(CHAT_ID, "hello Ann"), user=user_payload(True))
    )
    await asyncio.wait_for(asyncio.gather(inlet_task, other_coroutine()), timeout=5)

    assert progressed["value"], "event loop was blocked during NER — offload not effective"
    content = inlet_task.result()["messages"][0]["content"]
    assert "[PERSON_1]" in content, content


async def test_regression_masking_identical_order_and_numbering():
    """Offload must not alter detection/masking: multiple distinct names get
    sequential [PERSON_N] in first-appearance order, unchanged from sync."""
    pipe = make_gliner_pipeline(name_spans={"Ann Smith": "PERSON", "Bob Jones": "PERSON"})

    text = "Meeting with Ann Smith and Bob Jones and Ann Smith again."
    out = await pipe.inlet(_body(CHAT_ID, text), user=user_payload(True))
    content = out["messages"][0]["content"]

    # Both names masked; first-seen name is _1, second is _2; the repeat of the
    # first reuses _1 (vault dedupe) — order and numbering preserved.
    assert "Ann Smith" not in content and "Bob Jones" not in content, content
    assert content.count("[PERSON_1]") == 2, content
    assert content.count("[PERSON_2]") == 1, content
    assert content == "Meeting with [PERSON_1] and [PERSON_2] and [PERSON_1] again.", content


def test_executor_is_single_worker_and_torn_down():
    """The executor is a single-worker pool, reused across calls, and cleared on
    on_shutdown so a Pipelines reload does not leak worker threads."""
    pipe = pii_mod.Pipeline()
    ex1 = pipe._get_ner_executor()
    ex2 = pipe._get_ner_executor()
    assert ex1 is ex2, "executor should be reused, not recreated per call"
    assert ex1._max_workers == 1, ex1._max_workers

    asyncio.run(pipe.on_shutdown())
    assert pipe._ner_executor is None, "on_shutdown must clear the executor"
