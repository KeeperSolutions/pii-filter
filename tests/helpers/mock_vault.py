"""In-memory `MockThreadVault` for non-Postgres-specific tests.

Implements the same public async API as `ThreadVault` (Postgres backend)
using nested `dict` state, so the `started_pipeline` fixture in
`test_masking.py` and `test_recognizers.py` can exercise the full
inlet/outlet code paths without requiring `pg_ctl` on PATH.

Storage model — one entry per chat_id::

    {
        "chatA": {
            "lookup":   {"Ivan Horvat": "[PERSON_1]", ...},   # normalized key
            "forward":  {"Ivan Horvat": "[PERSON_1]", ...},   # literal original
            "reverse":  {"[PERSON_1]": "Ivan Horvat", ...},
            "counters": {"PERSON": 1, "HR_OIB": 1, ...},
        },
        ...
    }

`lookup` is keyed by the normalized `lookup_value` (dedup identity) while
`forward`/`reverse` hold the LITERAL stored text — the TRAU-530 split, mirrored
from the real vault so tests exercise the same semantics.

No TTL semantics — entries live for the lifetime of the instance.
"""

from __future__ import annotations

from typing import Any, TypedDict


class _ThreadState(TypedDict):
    lookup: dict[str, str]
    forward: dict[str, str]
    reverse: dict[str, str]
    counters: dict[str, int]


def _new_thread_state() -> _ThreadState:
    return {"lookup": {}, "forward": {}, "reverse": {}, "counters": {}}


class MockThreadVault:
    """In-memory thread vault for tests that need a vault without a real backend.

    Matches the public async API of the production `ThreadVault` (Postgres
    backend) so tests can swap implementations transparently. Set
    ``force_unhealthy = True`` on an instance to make ``healthcheck()`` return
    False — used by tests that exercise the inlet's degradation branches.

    The constructor accepts and discards arbitrary kwargs so the class can
    drop in as a substitute for ``ThreadVault`` (which takes ``dsn``,
    ``pool_min``, etc.) in module-scoped monkeypatches.
    """

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self._threads: dict[str, _ThreadState] = {}
        self.force_unhealthy: bool = False

    async def initialize(self) -> None:
        return None

    async def aclose(self) -> None:
        self._threads.clear()

    async def healthcheck(self) -> bool:
        return not self.force_unhealthy

    async def get_or_create_thread(self, chat_id: str) -> None:
        # API parity with the Postgres backend, where this method is a no-op.
        return None

    async def get_placeholder(
        self,
        chat_id: str,
        original_value: str,
        entity_type: str,
        lookup_value: str | None = None,
    ) -> str:
        """Dedup on `lookup_value` (normalized identity), store `original_value`
        (the literal text). Mirrors the real vault's TRAU-530 split, including
        first-write-wins: `ON CONFLICT DO UPDATE` never rewrites the stored
        value, so the first literal form to claim a key keeps it."""
        state = self._threads.setdefault(chat_id, _new_thread_state())
        key = original_value if lookup_value is None else lookup_value

        existing = state["lookup"].get(key)
        if existing is not None:
            return existing

        counter = state["counters"].get(entity_type, 0) + 1
        state["counters"][entity_type] = counter
        placeholder = f"[{entity_type}_{counter}]"

        state["lookup"][key] = placeholder
        state["forward"][original_value] = placeholder
        state["reverse"][placeholder] = original_value
        return placeholder

    async def restore(self, chat_id: str, placeholder: str) -> str | None:
        state = self._threads.get(chat_id)
        if state is None:
            return None
        return state["reverse"].get(placeholder)

    async def snapshot_for_request(self, chat_id: str) -> tuple[dict[str, str], dict[str, str]]:
        state = self._threads.get(chat_id)
        if state is None:
            return {}, {}
        return dict(state["forward"]), dict(state["reverse"])
