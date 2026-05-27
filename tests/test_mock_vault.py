"""Conformance tests for `MockThreadVault` (Task 9 §2.1).

These verify the mock honors the same contract as the production
`ThreadVault` (Postgres backend) for the slice of behaviors exercised by
the `started_pipeline` fixtures in `test_masking.py` / `test_recognizers.py`.
"""

from __future__ import annotations

import pytest

from tests.helpers.mock_vault import MockThreadVault

pytestmark = pytest.mark.asyncio


async def test_get_placeholder_mints_first_time() -> None:
    """Empty vault: first call mints `[PERSON_1]`."""
    vault = MockThreadVault()
    placeholder = await vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    assert placeholder == "[PERSON_1]"


async def test_get_placeholder_returns_existing() -> None:
    """Same `(chat_id, original)` twice returns the same placeholder, counter does not advance."""
    vault = MockThreadVault()
    first = await vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    second = await vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    assert first == second == "[PERSON_1]"

    # A different original value still mints `_2`, proving the counter did not
    # advance on the duplicate call.
    third = await vault.get_placeholder("chatA", "Ana Marić", "PERSON")
    assert third == "[PERSON_2]"


async def test_distinct_types_have_separate_counters() -> None:
    """PERSON and HR_OIB in the same thread both start at `_1`."""
    vault = MockThreadVault()
    person = await vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    oib = await vault.get_placeholder("chatA", "12345678903", "HR_OIB")
    assert person == "[PERSON_1]"
    assert oib == "[HR_OIB_1]"


async def test_cross_thread_isolation() -> None:
    """Same original PERSON in chat A and chat B → both `[PERSON_1]`."""
    vault = MockThreadVault()
    pa = await vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    pb = await vault.get_placeholder("chatB", "Ivan Horvat", "PERSON")
    assert pa == "[PERSON_1]"
    assert pb == "[PERSON_1]"

    fwd_a, _ = await vault.snapshot_for_request("chatA")
    fwd_b, _ = await vault.snapshot_for_request("chatB")
    assert fwd_a == {"Ivan Horvat": "[PERSON_1]"}
    assert fwd_b == {"Ivan Horvat": "[PERSON_1]"}


async def test_restore_returns_original() -> None:
    vault = MockThreadVault()
    placeholder = await vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    restored = await vault.restore("chatA", placeholder)
    assert restored == "Ivan Horvat"


async def test_restore_returns_none_for_unknown_placeholder() -> None:
    vault = MockThreadVault()
    assert await vault.restore("chatA", "[PERSON_99]") is None


async def test_restore_returns_none_for_unknown_thread() -> None:
    """A placeholder minted in chatA cannot be restored from chatB."""
    vault = MockThreadVault()
    minted = await vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    assert await vault.restore("chatB", minted) is None


async def test_snapshot_returns_inverse_maps() -> None:
    """snapshot returns forward + reverse dicts that invert each other."""
    vault = MockThreadVault()
    await vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    await vault.get_placeholder("chatA", "Ana Marić", "PERSON")
    await vault.get_placeholder("chatA", "12345678903", "HR_OIB")

    forward, reverse = await vault.snapshot_for_request("chatA")
    assert forward == {
        "Ivan Horvat": "[PERSON_1]",
        "Ana Marić": "[PERSON_2]",
        "12345678903": "[HR_OIB_1]",
    }
    assert reverse == {ph: orig for orig, ph in forward.items()}


async def test_snapshot_empty_thread_returns_empty_dicts() -> None:
    vault = MockThreadVault()
    forward, reverse = await vault.snapshot_for_request("brand-new-chat")
    assert forward == {}
    assert reverse == {}


async def test_healthcheck_true_by_default() -> None:
    vault = MockThreadVault()
    assert await vault.healthcheck() is True


async def test_healthcheck_returns_false_when_force_unhealthy() -> None:
    """`force_unhealthy = True` flips healthcheck — drives inlet degradation branch."""
    vault = MockThreadVault()
    vault.force_unhealthy = True
    assert await vault.healthcheck() is False


async def test_initialize_and_aclose_are_idempotent_noops() -> None:
    vault = MockThreadVault()
    await vault.initialize()
    await vault.initialize()
    await vault.get_placeholder("chatA", "Ivan Horvat", "PERSON")
    await vault.aclose()
    await vault.aclose()


async def test_get_or_create_thread_is_noop() -> None:
    """API parity with Postgres backend — method exists and writes nothing."""
    vault = MockThreadVault()
    await vault.get_or_create_thread("chatA")
    # No mappings were created.
    forward, reverse = await vault.snapshot_for_request("chatA")
    assert forward == {}
    assert reverse == {}
