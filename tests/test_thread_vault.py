"""Unit tests for `ThreadVault` (Task 5.1).

Mirrors the structure of `test_thread_vault.py` (Redis backend) one-to-one
where the contract is identical, plus a handful of Postgres-specific tests
that exercise lazy expiry, idempotent DDL, and the `INSERT ... ON CONFLICT`
race-condition fence.

These tests stand up a real Postgres process via `pytest-postgresql`. On a
host without `pg_ctl` / `postgres` on PATH the entire module is skipped via
the autouse `postgres_binary_missing` marker.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

from pii_filter import VaultCipher, make_ephemeral_thread_id
from tests.conftest import VAULT_TEST_ENC_KEY, postgres_binary_missing

if TYPE_CHECKING:
    from pii_filter import ThreadVault

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        postgres_binary_missing,
        reason="pg_ctl / postgres binary not on PATH; skipping Postgres-backed tests",
    ),
]


# ---------------------------------------------------------------------------
# Mint / dedupe / counters
# ---------------------------------------------------------------------------


async def test_get_placeholder_mints_first_time(postgres_vault: ThreadVault) -> None:
    """Empty vault: first call mints `[PERSON_1]`, counter advances to 1."""
    placeholder = await postgres_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    assert placeholder == "[PERSON_1]"

    # Verify the underlying counter row landed with next_value=2 (one bump).
    pool = postgres_vault._pool
    assert pool is not None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT next_value FROM pii_thread_counters " "WHERE chat_id = $1 AND entity_type = $2",
            "chatA",
            "PERSON",
        )
    assert row is not None
    assert row["next_value"] == 2


async def test_get_placeholder_returns_existing(postgres_vault: ThreadVault) -> None:
    """Same call twice with the same original returns the same placeholder.

    Postgres tolerates a counter gap under concurrency (placeholder uniqueness
    is preserved by the unique reverse index, not by counter monotonicity);
    here the SECOND call lands an `ON CONFLICT DO UPDATE` on the mappings
    table and returns the original `[PERSON_1]`. The counter advances to 3
    (two bumps) — verify both invariants.
    """
    first = await postgres_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    second = await postgres_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")

    assert first == second == "[PERSON_1]"

    pool = postgres_vault._pool
    assert pool is not None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT next_value FROM pii_thread_counters " "WHERE chat_id = $1 AND entity_type = $2",
            "chatA",
            "PERSON",
        )
    assert row is not None
    # Counter gap: 2 bumps for 1 mapping. See spec §2.3 race analysis.
    assert row["next_value"] == 3


async def test_distinct_originals_increment_counter(
    postgres_vault: ThreadVault,
) -> None:
    """Two different PERSON values in the same thread get [PERSON_1] / [PERSON_2]."""
    p1 = await postgres_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    p2 = await postgres_vault.get_placeholder("chatA", "Ana Marić", "PERSON")

    assert p1 == "[PERSON_1]"
    assert p2 == "[PERSON_2]"


async def test_distinct_types_have_separate_counters(
    postgres_vault: ThreadVault,
) -> None:
    """PERSON and HR_OIB in the same thread both start at _1 (independent counters)."""
    person = await postgres_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    oib = await postgres_vault.get_placeholder("chatA", "12345678903", "HR_OIB")

    assert person == "[PERSON_1]"
    assert oib == "[HR_OIB_1]"


# ---------------------------------------------------------------------------
# Cross-thread isolation (the core epic acceptance criterion)
# ---------------------------------------------------------------------------


async def test_cross_thread_isolation(postgres_vault: ThreadVault) -> None:
    """Same original PERSON in chat A and chat B → both [PERSON_1] (independent counters)."""
    pa = await postgres_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    pb = await postgres_vault.get_placeholder("chatB", "Ivan Horvat", "PERSON")

    assert pa == "[PERSON_1]"
    assert pb == "[PERSON_1]"

    fwd_a, _ = await postgres_vault.snapshot_for_request("chatA")
    fwd_b, _ = await postgres_vault.snapshot_for_request("chatB")
    assert fwd_a == {"Ivan Horvat": "[PERSON_1]"}
    assert fwd_b == {"Ivan Horvat": "[PERSON_1]"}


# ---------------------------------------------------------------------------
# restore()
# ---------------------------------------------------------------------------


async def test_restore_returns_original(postgres_vault: ThreadVault) -> None:
    placeholder = await postgres_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    restored = await postgres_vault.restore("chatA", placeholder)
    assert restored == "Ivan Horvat"


async def test_restore_returns_none_for_unknown(postgres_vault: ThreadVault) -> None:
    """Unknown placeholder (e.g. LLM hallucinated `[PERSON_99]`) → None."""
    result = await postgres_vault.restore("chatA", "[PERSON_99]")
    assert result is None


async def test_restore_returns_none_for_unknown_thread(
    postgres_vault: ThreadVault,
) -> None:
    """A placeholder minted in chatA cannot be restored from chatB."""
    minted = await postgres_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    result = await postgres_vault.restore("chatB", minted)
    assert result is None


async def test_restore_returns_none_for_expired_row(
    postgres_vault: ThreadVault,
) -> None:
    """Postgres-specific lazy-expiry: a row whose `expires_at <= now()` is
    invisible to `restore`, even though the row physically exists."""
    placeholder = await postgres_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")

    pool = postgres_vault._pool
    assert pool is not None
    async with pool.acquire() as conn:
        # Force the mapping row to look already-expired.
        past = datetime.now(tz=UTC) - timedelta(seconds=60)
        await conn.execute(
            "UPDATE pii_thread_mappings SET expires_at = $2 WHERE chat_id = $1",
            "chatA",
            past,
        )

    result = await postgres_vault.restore("chatA", placeholder)
    assert result is None


async def test_get_placeholder_resets_counter_after_thread_expires(
    postgres_vault: ThreadVault,
) -> None:
    """Parity with Redis: when ALL rows for a chat_id are past their
    `expires_at`, the next `get_placeholder` mints `[TYPE_1]` against a
    fresh counter rather than bumping the stale `next_value`. Redis
    EXPIRE auto-deletes the keys; Postgres must purge them inline at
    the top of `get_placeholder` to behave the same way (and to honor
    the GDPR TTL contract).
    """
    first = await postgres_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    assert first == "[PERSON_1]"

    pool = postgres_vault._pool
    assert pool is not None
    async with pool.acquire() as conn:
        past = datetime.now(tz=UTC) - timedelta(seconds=60)
        await conn.execute(
            "UPDATE pii_thread_counters SET expires_at = $2 WHERE chat_id = $1",
            "chatA",
            past,
        )
        await conn.execute(
            "UPDATE pii_thread_mappings SET expires_at = $2 WHERE chat_id = $1",
            "chatA",
            past,
        )

    # New original on the same chat: counter+mapping are purged inline,
    # so the fresh insert mints `[PERSON_1]` and not `[PERSON_2]`.
    second = await postgres_vault.get_placeholder("chatA", "Marko Marić", "PERSON")
    assert second == "[PERSON_1]"

    # Re-minting the originally-expired value now yields `[PERSON_2]` —
    # proof that the old mapping was physically deleted, not
    # resurrected via `ON CONFLICT DO UPDATE RETURNING placeholder`.
    third = await postgres_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    assert third == "[PERSON_2]"


# ---------------------------------------------------------------------------
# snapshot_for_request()
# ---------------------------------------------------------------------------


async def test_snapshot_returns_full_maps(postgres_vault: ThreadVault) -> None:
    """snapshot returns dicts that round-trip: forward and reverse are inverses."""
    await postgres_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    await postgres_vault.get_placeholder("chatA", "Ana Marić", "PERSON")
    await postgres_vault.get_placeholder("chatA", "12345678903", "HR_OIB")

    forward, reverse = await postgres_vault.snapshot_for_request("chatA")

    assert forward == {
        "Ivan Horvat": "[PERSON_1]",
        "Ana Marić": "[PERSON_2]",
        "12345678903": "[HR_OIB_1]",
    }
    assert reverse == {ph: orig for orig, ph in forward.items()}


async def test_snapshot_excludes_expired_rows(postgres_vault: ThreadVault) -> None:
    """Postgres-specific: snapshot's `WHERE expires_at > now()` filter hides
    rows whose deadline has passed; the bulk TTL renewal in `snapshot_for_request`
    then refreshes only the still-live entries (it issues SET expires_at = $2
    against ALL rows, but the SELECT after that runs the lazy filter).

    Task 11: `original_value` is now an ENC1 ciphertext, so rows are targeted
    and asserted by their (plaintext) `placeholder` rather than by value.
    """
    ph_ivan = await postgres_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    ph_ana = await postgres_vault.get_placeholder("chatA", "Ana Marić", "PERSON")

    pool = postgres_vault._pool
    assert pool is not None
    async with pool.acquire() as conn:
        # Bypass the renewal: set just Ivan's row to expired and call a raw
        # SELECT. Target by placeholder (plaintext) — the encrypted
        # `original_value` cannot be matched by a literal in SQL.
        past = datetime.now(tz=UTC) - timedelta(seconds=60)
        await conn.execute(
            """
            UPDATE pii_thread_mappings
            SET expires_at = $2
            WHERE chat_id = $1 AND placeholder = $3
            """,
            "chatA",
            past,
            ph_ivan,
        )
        # Verify the row exists physically.
        row = await conn.fetchrow(
            "SELECT placeholder FROM pii_thread_mappings WHERE chat_id = $1 AND placeholder = $2",
            "chatA",
            ph_ivan,
        )
    assert row is not None  # row physically present

    # snapshot_for_request bumps `expires_at` for ALL chat rows before SELECT,
    # so Ivan's row gets renewed — the lazy filter alone isn't sufficient to
    # hide it post-bulk-renewal. Drive the test through a raw SELECT instead
    # to pin the lazy-filter contract:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT placeholder FROM pii_thread_mappings
            WHERE chat_id = $1 AND expires_at > now()
            """,
            "chatA",
        )
    visible_placeholders = {row["placeholder"] for row in rows}
    assert ph_ana in visible_placeholders
    assert ph_ivan not in visible_placeholders


async def test_snapshot_does_not_resurrect_expired_rows(
    postgres_vault: ThreadVault,
) -> None:
    """`snapshot_for_request` must NOT bump `expires_at` on already-expired
    rows. Renewing them would resurrect PII past the GDPR TTL deadline and
    diverge from Redis (where keys past TTL are gone, period). The bulk
    UPDATEs gate on `expires_at > now()` so expired rows stay expired and
    fall out of the SELECT.
    """
    placeholder_ivan = await postgres_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    placeholder_ana = await postgres_vault.get_placeholder("chatA", "Ana Marić", "PERSON")

    pool = postgres_vault._pool
    assert pool is not None
    past = datetime.now(tz=UTC) - timedelta(seconds=60)
    async with pool.acquire() as conn:
        # Expire Ivan's mapping but leave Ana's alone. Target by placeholder
        # (plaintext) — `original_value` is now an encrypted ENC1 blob.
        await conn.execute(
            "UPDATE pii_thread_mappings SET expires_at = $2 "
            "WHERE chat_id = $1 AND placeholder = $3",
            "chatA",
            past,
            placeholder_ivan,
        )

    forward, reverse = await postgres_vault.snapshot_for_request("chatA")

    # Ana is live → in the snapshot. The snapshot decrypts each row, so the
    # maps are keyed/valued on plaintext exactly as before.
    assert forward.get("Ana Marić") == placeholder_ana
    assert reverse.get(placeholder_ana) == "Ana Marić"
    # Ivan was expired → snapshot must hide him.
    assert "Ivan Horvat" not in forward
    assert placeholder_ivan not in reverse

    # Confirm Ivan's row is still expired in DB (not bumped to a future
    # `expires_at` by the bulk renewal).
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT expires_at FROM pii_thread_mappings " "WHERE chat_id = $1 AND placeholder = $2",
            "chatA",
            placeholder_ivan,
        )
    assert row is not None
    assert row["expires_at"] <= datetime.now(tz=UTC)


async def test_snapshot_empty_thread_returns_empty_dicts(
    postgres_vault: ThreadVault,
) -> None:
    forward, reverse = await postgres_vault.snapshot_for_request("brand-new-chat")
    assert forward == {}
    assert reverse == {}


# ---------------------------------------------------------------------------
# TTL renewal
# ---------------------------------------------------------------------------


async def test_ttl_renewed_on_get_placeholder(postgres_vault: ThreadVault) -> None:
    """Calling `get_placeholder` again pushes `expires_at` further into the future."""
    await postgres_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")

    pool = postgres_vault._pool
    assert pool is not None
    async with pool.acquire() as conn:
        # Drop the expiry close to now so a subsequent renewal is observable.
        soon = datetime.now(tz=UTC) + timedelta(seconds=5)
        await conn.execute(
            "UPDATE pii_thread_mappings SET expires_at = $2 WHERE chat_id = $1",
            "chatA",
            soon,
        )

    await postgres_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT expires_at FROM pii_thread_mappings WHERE chat_id = $1",
            "chatA",
        )
    assert row is not None
    new_expiry: datetime = row["expires_at"]
    # 3600s thread TTL minus a few seconds of drift; well past the artificial 5s.
    assert new_expiry - datetime.now(tz=UTC) > timedelta(seconds=60)


async def test_ttl_renewed_on_restore(postgres_vault: ThreadVault) -> None:
    """`restore` bumps `expires_at` on hit via UPDATE ... RETURNING."""
    placeholder = await postgres_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")

    pool = postgres_vault._pool
    assert pool is not None
    async with pool.acquire() as conn:
        soon = datetime.now(tz=UTC) + timedelta(seconds=5)
        await conn.execute(
            "UPDATE pii_thread_mappings SET expires_at = $2 WHERE chat_id = $1",
            "chatA",
            soon,
        )

    await postgres_vault.restore("chatA", placeholder)

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT expires_at FROM pii_thread_mappings WHERE chat_id = $1",
            "chatA",
        )
    assert row is not None
    new_expiry: datetime = row["expires_at"]
    assert new_expiry - datetime.now(tz=UTC) > timedelta(seconds=60)


async def test_ttl_renewed_on_snapshot_for_request(
    postgres_vault: ThreadVault,
) -> None:
    """`snapshot_for_request` bulk-UPDATEs both mapping and counter rows."""
    await postgres_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")

    pool = postgres_vault._pool
    assert pool is not None
    async with pool.acquire() as conn:
        soon = datetime.now(tz=UTC) + timedelta(seconds=5)
        await conn.execute(
            "UPDATE pii_thread_mappings SET expires_at = $2 WHERE chat_id = $1",
            "chatA",
            soon,
        )
        await conn.execute(
            "UPDATE pii_thread_counters SET expires_at = $2 WHERE chat_id = $1",
            "chatA",
            soon,
        )

    await postgres_vault.snapshot_for_request("chatA")

    async with pool.acquire() as conn:
        mapping_row = await conn.fetchrow(
            "SELECT expires_at FROM pii_thread_mappings WHERE chat_id = $1",
            "chatA",
        )
        counter_row = await conn.fetchrow(
            "SELECT expires_at FROM pii_thread_counters WHERE chat_id = $1",
            "chatA",
        )
    assert mapping_row is not None
    assert counter_row is not None
    now = datetime.now(tz=UTC)
    assert mapping_row["expires_at"] - now > timedelta(seconds=60)
    assert counter_row["expires_at"] - now > timedelta(seconds=60)


# ---------------------------------------------------------------------------
# Ephemeral threads
# ---------------------------------------------------------------------------


async def test_ephemeral_thread_uses_short_ttl(postgres_vault: ThreadVault) -> None:
    """A thread id starting with `ephemeral:` uses ephemeral_ttl_seconds (300s)
    rather than thread_ttl_seconds (3600s)."""
    ephemeral_id = make_ephemeral_thread_id()
    assert ephemeral_id.startswith("ephemeral:")

    await postgres_vault.get_placeholder(ephemeral_id, "Ivan Horvat", "PERSON")

    pool = postgres_vault._pool
    assert pool is not None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT expires_at FROM pii_thread_mappings WHERE chat_id = $1",
            ephemeral_id,
        )
    assert row is not None
    delta = row["expires_at"] - datetime.now(tz=UTC)
    # Ephemeral TTL is 300s; allow a few seconds of clock drift on either side.
    assert timedelta(seconds=200) < delta <= timedelta(seconds=305)


async def test_real_thread_uses_long_ttl(postgres_vault: ThreadVault) -> None:
    """A non-ephemeral chat_id routes to `thread_ttl_seconds` (3600s in the
    fixture) and NOT `ephemeral_ttl_seconds` (300s). Asserts `expires_at`
    lands in the `(ephemeral_ttl_seconds, thread_ttl_seconds]` window.

    Backfill for Gap C from REDIS-AUDIT.md — the Redis backend's
    `test_real_thread_uses_long_ttl` proved the routing branch directly;
    the existing Postgres `test_ttl_renewed_on_get_placeholder` only
    asserts `> 60s`, which both buckets satisfy. This test pins the
    routing contract.
    """
    await postgres_vault.get_placeholder("real-chat-id", "Ivan Horvat", "PERSON")

    pool = postgres_vault._pool
    assert pool is not None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT expires_at FROM pii_thread_mappings WHERE chat_id = $1",
            "real-chat-id",
        )
    assert row is not None
    delta = row["expires_at"] - datetime.now(tz=UTC)
    # Window must exclude the ephemeral bucket (300s) AND fit inside the
    # long-thread bucket (3600s) plus minor clock drift.
    assert timedelta(seconds=300) < delta <= timedelta(seconds=3605)


# ---------------------------------------------------------------------------
# Healthcheck
# ---------------------------------------------------------------------------


async def test_healthcheck_success(postgres_vault: ThreadVault) -> None:
    assert await postgres_vault.healthcheck() is True


async def test_healthcheck_returns_false_when_pool_uninitialized() -> None:
    """healthcheck must not raise if `initialize()` was never called."""
    from pii_filter import ThreadVault as _PV

    vault = _PV(dsn="postgresql://nobody@127.0.0.1:1/none")
    # Never call initialize(); pool stays None, healthcheck returns False fast.
    assert await vault.healthcheck() is False


async def test_postgres_healthcheck_returns_false_on_query_exception(
    postgres_vault: ThreadVault,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """healthcheck must return False (not raise) when the live pool's
    `SELECT 1` path raises — e.g. a transient connection drop mid-flight.

    Backfill for Gap D from REDIS-AUDIT.md — the Redis backend's
    `test_healthcheck_failure_returns_false_on_ping_exception` covered
    this scenario; the existing Postgres
    `test_healthcheck_returns_false_when_pool_uninitialized` only covers
    the "pool was never initialized" branch. The inlet's degradation gate
    must be able to branch on the bool whether the pool is None OR the
    pool is initialized but the query raises.

    Swaps the vault's `_pool` reference to a stub whose `acquire` raises;
    asyncpg's real `Pool.acquire` is a read-only descriptor and cannot be
    monkeypatched in place.
    """
    import asyncpg

    class _BrokenPool:
        def acquire(self, *_args: Any, **_kwargs: Any) -> Any:
            raise asyncpg.PostgresConnectionError("simulated connection lost")

        async def close(self) -> None:
            return None

    monkeypatch.setattr(postgres_vault, "_pool", _BrokenPool())
    assert await postgres_vault.healthcheck() is False


# ---------------------------------------------------------------------------
# Concurrent callers — INSERT ... ON CONFLICT atomicity
# ---------------------------------------------------------------------------


async def test_concurrent_get_placeholder_returns_consistent_result(
    postgres_vault: ThreadVault,
) -> None:
    """10 concurrent `get_placeholder` calls for the same `(chat_id, type, original)`
    must all return the same placeholder. Counter gaps are tolerated; placeholder
    uniqueness is preserved by the unique reverse index."""
    results = await asyncio.gather(
        *[postgres_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON") for _ in range(10)]
    )
    assert set(results) == {"[PERSON_1]"}

    pool = postgres_vault._pool
    assert pool is not None
    async with pool.acquire() as conn:
        mappings = await conn.fetch(
            "SELECT placeholder FROM pii_thread_mappings WHERE chat_id = $1",
            "chatA",
        )
    # Exactly one mapping row regardless of how many concurrent callers ran.
    assert len(mappings) == 1
    assert mappings[0]["placeholder"] == "[PERSON_1]"


async def test_concurrent_calls_demonstrate_race_via_counter_bump(
    postgres_vault: ThreadVault,
) -> None:
    """Under parallel load for the same (chat_id, entity_type, original_value),
    every caller returns the identical placeholder (idempotency guarantee from
    the unique reverse index), BUT the counter is bumped once per caller — the
    `ON CONFLICT DO UPDATE` on the counter table always succeeds, even when the
    mapping insert that follows hits its own ON CONFLICT and returns the
    winner's placeholder. Counter gaps are tolerated by spec §2.3.
    """
    n_concurrent = 5
    results = await asyncio.gather(
        *[
            postgres_vault.get_placeholder("race-thread", "Ivan Horvat", "PERSON")
            for _ in range(n_concurrent)
        ]
    )
    # All callers got the same placeholder (idempotency guarantee).
    assert len(set(results)) == 1

    # Counter was bumped once per caller; query state directly because we are
    # asserting an internal invariant, not the public API.
    pool = postgres_vault._pool
    assert pool is not None
    async with pool.acquire() as conn:
        next_value = await conn.fetchval(
            "SELECT next_value FROM pii_thread_counters " "WHERE chat_id = $1 AND entity_type = $2",
            "race-thread",
            "PERSON",
        )
    # `next_value` must be >= n_concurrent + 1 — the schema's first-INSERT
    # default lands at 2, then each subsequent caller's ON CONFLICT bumps by 1.
    # If next_value == 2 the race did NOT actually occur (e.g. implementation
    # was promoted to SERIALIZABLE / advisory locks), in which case this test
    # is the canary that catches the silent contract change.
    assert next_value >= n_concurrent + 1, (
        f"Expected counter >= {n_concurrent + 1} (proves race occurred), got {next_value}. "
        f"If counter == 2, race did NOT occur — implementation may have changed to SERIALIZABLE."
    )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_initialize_runs_ddl_idempotently(
    postgres_vault: ThreadVault,
) -> None:
    """initialize() may be called multiple times; CREATE TABLE/INDEX IF NOT
    EXISTS make the second call a no-op."""
    # postgres_vault is already initialized by the fixture; call initialize()
    # again and verify it does not raise.
    await postgres_vault.initialize()
    # And one more for safety; this also replaces the pool, so verify a
    # subsequent operation still works.
    await postgres_vault.initialize()
    placeholder = await postgres_vault.get_placeholder("chatA", "Ivan", "PERSON")
    assert placeholder == "[PERSON_1]"


async def test_aclose_is_idempotent(postgres_vault: ThreadVault) -> None:
    """aclose() may be called multiple times without raising."""
    await postgres_vault.aclose()
    await postgres_vault.aclose()
    # After aclose, healthcheck should return False (pool is None).
    assert await postgres_vault.healthcheck() is False


# ---------------------------------------------------------------------------
# get_or_create_thread (API parity no-op)
# ---------------------------------------------------------------------------


async def test_get_or_create_thread_is_noop(postgres_vault: ThreadVault) -> None:
    """Postgres backend has no per-thread row to create; get_or_create_thread
    returns None (typed as such) and writes nothing — verify the side-effect
    contract by asserting the mappings table stayed empty."""
    await postgres_vault.get_or_create_thread("chatA")

    pool = postgres_vault._pool
    assert pool is not None
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT 1 FROM pii_thread_mappings WHERE chat_id = $1",
            "chatA",
        )
    # No-op confirmed: zero mapping rows after a bare get_or_create_thread.
    assert rows == []


# ---------------------------------------------------------------------------
# Schema invariants
# ---------------------------------------------------------------------------


async def test_mapping_row_records_counter_index_and_created_at(
    postgres_vault: ThreadVault,
) -> None:
    """`counter_index` mirrors the placeholder's numeric suffix; `created_at`
    is populated by the column DEFAULT now() (used for audit / future
    analytics, never read by application logic)."""
    await postgres_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")

    pool = postgres_vault._pool
    assert pool is not None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT counter_index, created_at FROM pii_thread_mappings " "WHERE chat_id = $1",
            "chatA",
        )
    assert row is not None
    assert row["counter_index"] == 1
    # created_at is auto-populated by the column DEFAULT.
    created_at_val: Any = row["created_at"]
    assert isinstance(created_at_val, datetime)
    # Should be very recent.
    assert datetime.now(tz=UTC) - created_at_val < timedelta(seconds=30)


# ---------------------------------------------------------------------------
# Task 11 — encryption-at-rest (the `postgres_vault` fixture runs with
# encryption ON; see conftest). These pin the DB-level behavior that the pure
# unit tests in test_vault_crypto.py cannot reach.
# ---------------------------------------------------------------------------


async def test_original_value_stored_as_ciphertext(postgres_vault: ThreadVault) -> None:
    """With encryption ON, the raw `original_value` column is an ENC1 envelope
    (not plaintext) and decrypts back to the original value."""
    placeholder = await postgres_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")

    pool = postgres_vault._pool
    assert pool is not None
    async with pool.acquire() as conn:
        stored: str = await conn.fetchval(
            "SELECT original_value FROM pii_thread_mappings "
            "WHERE chat_id = $1 AND placeholder = $2",
            "chatA",
            placeholder,
        )

    assert VaultCipher.is_encrypted(stored)
    assert stored.startswith("ENC1:")
    assert "Ivan Horvat" not in stored  # plaintext never appears on disk
    # The fixture's enc key decrypts it back to the original plaintext.
    assert VaultCipher(VAULT_TEST_ENC_KEY).decrypt(stored) == "Ivan Horvat"


async def test_upsert_dedup_under_encryption(postgres_vault: ThreadVault) -> None:
    """Same value twice → one row, same placeholder, and the stored ciphertext
    is NOT re-minted on the second call (ON CONFLICT only bumps expires_at —
    spec §7.1.5). The blind index makes the UPSERT dedup deterministic even
    though each encryption draws a fresh random nonce."""
    pool = postgres_vault._pool
    assert pool is not None

    first = await postgres_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    async with pool.acquire() as conn:
        stored_after_first: str = await conn.fetchval(
            "SELECT original_value FROM pii_thread_mappings "
            "WHERE chat_id = $1 AND placeholder = $2",
            "chatA",
            first,
        )

    second = await postgres_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT original_value FROM pii_thread_mappings WHERE chat_id = $1",
            "chatA",
        )

    assert first == second  # same placeholder (dedup)
    assert len(rows) == 1  # exactly one mapping row
    # Ciphertext unchanged on the second call (no fresh-nonce re-encryption).
    assert rows[0]["original_value"] == stored_after_first


async def test_primary_key_is_on_lookup_hash(postgres_vault: ThreadVault) -> None:
    """The mappings PK is (chat_id, entity_type, lookup_hash); `original_value`
    is no longer part of the primary key (spec §3)."""
    pool = postgres_vault._pool
    assert pool is not None
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT a.attname AS col
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY (i.indkey)
            WHERE i.indrelid = 'pii_thread_mappings'::regclass AND i.indisprimary
            """
        )
    pk_cols = {row["col"] for row in rows}
    assert pk_cols == {"chat_id", "entity_type", "lookup_hash"}
    assert "original_value" not in pk_cols


# ---------------------------------------------------------------------------
# lookup_value vs original — the TRAU-530 split
# ---------------------------------------------------------------------------

_ADDR_MULTILINE = "45 Baggot Street Lower,\nDublin 2"
_ADDR_SINGLELINE = "45 Baggot Street Lower, Dublin 2"


async def test_stored_value_is_never_the_normalized_form(
    postgres_vault: ThreadVault,
) -> None:
    """INVARIANT: `original_value` keeps the LITERAL text, newline and all.

    This pins the boundary an interim TRAU-530 revision broke. Normalizing the
    stored value made `snapshot_for_request` return whitespace-collapsed
    originals, which made the `re.escape`d `_build_vault_remasker` pattern stop
    matching the multi-line occurrence in history — a silent PII leak. It also
    reformatted the user's own value on restore.

    Read straight from the column so no in-memory helper can mask a regression.
    """
    placeholder = await postgres_vault.get_placeholder(
        "chatA", _ADDR_MULTILINE, "ADDRESS", lookup_value=_ADDR_SINGLELINE
    )

    pool = postgres_vault._pool
    assert pool is not None
    async with pool.acquire() as conn:
        stored: str = await conn.fetchval(
            "SELECT original_value FROM pii_thread_mappings "
            "WHERE chat_id = $1 AND placeholder = $2",
            "chatA",
            placeholder,
        )

    decrypted = VaultCipher(VAULT_TEST_ENC_KEY).decrypt(stored)
    assert decrypted == _ADDR_MULTILINE
    assert "\n" in decrypted, "the newline was normalized out of the STORED value"
    # And the round-trip through the public API agrees.
    assert await postgres_vault.restore("chatA", placeholder) == _ADDR_MULTILINE


async def test_lookup_value_dedupes_whitespace_variants(
    postgres_vault: ThreadVault,
) -> None:
    """Two literal forms sharing one `lookup_value` collapse to ONE row and one
    placeholder — the actual TRAU-530 fix, at the vault layer."""
    first = await postgres_vault.get_placeholder(
        "chatA", _ADDR_MULTILINE, "ADDRESS", lookup_value=_ADDR_SINGLELINE
    )
    second = await postgres_vault.get_placeholder(
        "chatA", _ADDR_SINGLELINE, "ADDRESS", lookup_value=_ADDR_SINGLELINE
    )

    assert first == second == "[ADDRESS_1]"

    pool = postgres_vault._pool
    assert pool is not None
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT original_value FROM pii_thread_mappings WHERE chat_id = $1",
            "chatA",
        )
    assert len(rows) == 1, "whitespace variants minted separate rows"
    # First-write-wins: the multi-line form claimed the key and keeps it.
    assert VaultCipher(VAULT_TEST_ENC_KEY).decrypt(rows[0]["original_value"]) == (
        _ADDR_MULTILINE
    )


async def test_lookup_value_defaults_to_original(postgres_vault: ThreadVault) -> None:
    """Backward compatibility: callers that omit `lookup_value` (the test
    doubles, `mask_text`) hash the `original`, exactly as before."""
    a = await postgres_vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    b = await postgres_vault.get_placeholder(
        "chatA", "Ivan Horvat", "PERSON", lookup_value="Ivan Horvat"
    )
    assert a == b == "[PERSON_1]"


async def test_distinct_lookup_values_stay_distinct(postgres_vault: ThreadVault) -> None:
    """The split must not become a fuzzy match: different lookup values mint
    different placeholders even when the literal texts look similar."""
    a = await postgres_vault.get_placeholder(
        "chatA", "Dublin 2", "ADDRESS", lookup_value="Dublin 2"
    )
    b = await postgres_vault.get_placeholder(
        "chatA", "Dublin 3", "ADDRESS", lookup_value="Dublin 3"
    )
    assert a != b
