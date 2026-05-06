"""Unit tests for `ThreadVault` (Task 5).

These tests run against `fakeredis.aioredis.FakeRedis` so they don't require
a running Redis daemon. Atomic get-or-mint is implemented in Lua server-side;
fakeredis EVAL support requires `lupa` (declared in `requirements-dev.txt`).
"""

from __future__ import annotations

import asyncio

import pytest
from fakeredis import aioredis as fake_aioredis

from pii_filter import ThreadVault, make_ephemeral_thread_id

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Mint / dedupe / counters
# ---------------------------------------------------------------------------


async def test_get_placeholder_mints_first_time(thread_vault: ThreadVault) -> None:
    """Empty vault: first call mints `[PERSON_1]`, counter advances to 1."""
    placeholder = await thread_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    assert placeholder == "[PERSON_1]"

    counter = await thread_vault._client.get(  # type: ignore[union-attr]
        "pii:thread:chatA:counter:PERSON"
    )
    assert counter == "1"


async def test_get_placeholder_returns_existing(thread_vault: ThreadVault) -> None:
    """Same call twice with the same original returns the same placeholder."""
    first = await thread_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    second = await thread_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")

    assert first == second == "[PERSON_1]"

    counter = await thread_vault._client.get(  # type: ignore[union-attr]
        "pii:thread:chatA:counter:PERSON"
    )
    assert counter == "1", "counter must not advance when value is already minted"


async def test_distinct_originals_increment_counter(thread_vault: ThreadVault) -> None:
    """Two different PERSON values in the same thread get [PERSON_1] / [PERSON_2]."""
    p1 = await thread_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    p2 = await thread_vault.get_placeholder("chatA", "Ana Marić", "PERSON")

    assert p1 == "[PERSON_1]"
    assert p2 == "[PERSON_2]"


async def test_distinct_types_have_separate_counters(thread_vault: ThreadVault) -> None:
    """PERSON and HR_OIB in the same thread both start at _1 (independent counters)."""
    person = await thread_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    oib = await thread_vault.get_placeholder("chatA", "12345678903", "HR_OIB")

    assert person == "[PERSON_1]"
    assert oib == "[HR_OIB_1]"


# ---------------------------------------------------------------------------
# Cross-thread isolation (the core epic acceptance criterion)
# ---------------------------------------------------------------------------


async def test_cross_thread_isolation(thread_vault: ThreadVault) -> None:
    """Same original PERSON in chat A and chat B → both [PERSON_1] (independent counters)."""
    pa = await thread_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    pb = await thread_vault.get_placeholder("chatB", "Ivan Horvat", "PERSON")

    assert pa == "[PERSON_1]"
    assert pb == "[PERSON_1]"

    fwd_a, _ = await thread_vault.snapshot_for_request("chatA")
    fwd_b, _ = await thread_vault.snapshot_for_request("chatB")
    assert fwd_a == {"Ivan Horvat": "[PERSON_1]"}
    assert fwd_b == {"Ivan Horvat": "[PERSON_1]"}


# ---------------------------------------------------------------------------
# restore()
# ---------------------------------------------------------------------------


async def test_restore_returns_original(thread_vault: ThreadVault) -> None:
    placeholder = await thread_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    restored = await thread_vault.restore("chatA", placeholder)
    assert restored == "Ivan Horvat"


async def test_restore_returns_none_for_unknown(thread_vault: ThreadVault) -> None:
    """Unknown placeholder (e.g. LLM hallucinated `[PERSON_99]`) → None."""
    result = await thread_vault.restore("chatA", "[PERSON_99]")
    assert result is None


async def test_restore_returns_none_for_unknown_thread(thread_vault: ThreadVault) -> None:
    """A placeholder minted in chatA cannot be restored from chatB."""
    minted = await thread_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    result = await thread_vault.restore("chatB", minted)
    assert result is None


# ---------------------------------------------------------------------------
# snapshot_for_request()
# ---------------------------------------------------------------------------


async def test_snapshot_returns_full_maps(thread_vault: ThreadVault) -> None:
    """snapshot returns dicts that round-trip: forward and reverse are inverses."""
    await thread_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    await thread_vault.get_placeholder("chatA", "Ana Marić", "PERSON")
    await thread_vault.get_placeholder("chatA", "12345678903", "HR_OIB")

    forward, reverse = await thread_vault.snapshot_for_request("chatA")

    assert forward == {
        "Ivan Horvat": "[PERSON_1]",
        "Ana Marić": "[PERSON_2]",
        "12345678903": "[HR_OIB_1]",
    }
    assert reverse == {ph: orig for orig, ph in forward.items()}


async def test_snapshot_empty_thread_returns_empty_dicts(
    thread_vault: ThreadVault,
) -> None:
    forward, reverse = await thread_vault.snapshot_for_request("brand-new-chat")
    assert forward == {}
    assert reverse == {}


# ---------------------------------------------------------------------------
# TTL renewal
# ---------------------------------------------------------------------------


async def test_ttl_renewed_on_get_placeholder(thread_vault: ThreadVault) -> None:
    """Calling `get_placeholder` again pushes EXPIRE back out on the touched keys."""
    await thread_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")

    # Manually drop TTL on the forward key to a low number so we can detect renewal.
    client = thread_vault._client
    assert client is not None
    await client.expire("pii:thread:chatA:forward", 5)
    assert await client.ttl("pii:thread:chatA:forward") <= 5

    # Re-fetch the same placeholder — Lua must EXPIRE the touched keys back to 60s.
    await thread_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")

    ttl_after = await client.ttl("pii:thread:chatA:forward")
    assert ttl_after > 5
    assert ttl_after <= 60


async def test_ttl_renewed_on_snapshot(thread_vault: ThreadVault) -> None:
    """`snapshot_for_request` is a public method that must renew TTL per spec §3.5."""
    await thread_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")

    client = thread_vault._client
    assert client is not None
    await client.expire("pii:thread:chatA:forward", 5)
    await client.expire("pii:thread:chatA:reverse", 5)

    await thread_vault.snapshot_for_request("chatA")

    assert await client.ttl("pii:thread:chatA:forward") > 5
    assert await client.ttl("pii:thread:chatA:reverse") > 5


async def test_ttl_renewed_on_restore(thread_vault: ThreadVault) -> None:
    """`restore` must renew TTL per spec §3.5 even when the placeholder is unknown."""
    await thread_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")

    client = thread_vault._client
    assert client is not None
    await client.expire("pii:thread:chatA:reverse", 3)

    await thread_vault.restore("chatA", "[PERSON_1]")

    assert await client.ttl("pii:thread:chatA:reverse") > 3


# ---------------------------------------------------------------------------
# Ephemeral threads
# ---------------------------------------------------------------------------


async def test_ephemeral_thread_uses_short_ttl(thread_vault: ThreadVault) -> None:
    """A thread id starting with `ephemeral:` uses ephemeral_ttl_seconds (10s)
    rather than thread_ttl_seconds (60s)."""
    ephemeral_id = make_ephemeral_thread_id()
    assert ephemeral_id.startswith("ephemeral:")

    await thread_vault.get_placeholder(ephemeral_id, "Ivan Horvat", "PERSON")

    client = thread_vault._client
    assert client is not None
    ttl = await client.ttl(f"pii:thread:{ephemeral_id}:forward")
    assert 0 < ttl <= 10, f"ephemeral TTL must be ≤10s, got {ttl}"


async def test_real_thread_uses_long_ttl(thread_vault: ThreadVault) -> None:
    """A non-ephemeral chat_id uses the longer thread_ttl_seconds (60s)."""
    await thread_vault.get_placeholder("real-chat-id", "Ivan Horvat", "PERSON")

    client = thread_vault._client
    assert client is not None
    ttl = await client.ttl("pii:thread:real-chat-id:forward")
    assert 10 < ttl <= 60, f"non-ephemeral TTL must be in (10, 60], got {ttl}"


# ---------------------------------------------------------------------------
# Healthcheck
# ---------------------------------------------------------------------------


async def test_healthcheck_ok(thread_vault: ThreadVault) -> None:
    assert await thread_vault.healthcheck() is True


async def test_healthcheck_failure_returns_false_on_ping_exception() -> None:
    """healthcheck must return False (not raise) when the underlying client
    fails to PING, so the inlet's degradation path can branch on that bool."""

    class _BrokenClient:
        def register_script(self, _script: str) -> object:
            return object()

        async def ping(self) -> bool:
            raise ConnectionError("simulated Redis unreachable")

    vault = ThreadVault(client=_BrokenClient())  # type: ignore[arg-type]
    assert await vault.healthcheck() is False


# ---------------------------------------------------------------------------
# Concurrent callers — Lua atomicity smoke test
# ---------------------------------------------------------------------------


async def test_concurrent_get_placeholder_returns_same_placeholder(
    thread_vault: ThreadVault,
) -> None:
    """Lua atomicity: 10 concurrent get_placeholder calls for the same original
    must all return the same placeholder. With the read-then-mint race that
    Lua exists to prevent, callers would split across `[PERSON_1]` and
    `[PERSON_2]`."""
    results = await asyncio.gather(
        *[thread_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON") for _ in range(10)]
    )
    assert set(results) == {"[PERSON_1]"}

    counter = await thread_vault._client.get(  # type: ignore[union-attr]
        "pii:thread:chatA:counter:PERSON"
    )
    # Counter MAY have been INCRed transiently in fakeredis if Lua wasn't
    # truly atomic; assert the final stable value is 1.
    assert counter == "1"


# ---------------------------------------------------------------------------
# decode_responses + bytes round-trip
# ---------------------------------------------------------------------------


async def test_keys_use_documented_schema(thread_vault: ThreadVault) -> None:
    """Spec §3.2: schema is `pii:thread:{chat_id}:{forward|reverse|counter:TYPE}`."""
    await thread_vault.get_placeholder("chatX", "12345678903", "HR_OIB")

    client = thread_vault._client
    assert client is not None
    keys = sorted(await client.keys("pii:thread:chatX:*"))
    assert keys == [
        "pii:thread:chatX:counter:HR_OIB",
        "pii:thread:chatX:forward",
        "pii:thread:chatX:reverse",
    ]


async def test_fake_redis_fixture_yields_decoded_strings(
    fake_redis: fake_aioredis.FakeRedis,
) -> None:
    """Smoke test the fixture: decode_responses=True so reads are str not bytes."""
    await fake_redis.set("k", "v")
    assert await fake_redis.get("k") == "v"
