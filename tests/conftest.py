"""Pytest fixtures for PII Filter tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Generator
from typing import Any

import pytest
import pytest_asyncio
from fakeredis import aioredis as fake_aioredis

from pii_filter import Pipeline, ThreadVault


@pytest.fixture(scope="session", autouse=True)
def _swap_redis_to_fakeredis() -> Generator[None, None, None]:
    """Globally redirect `redis.asyncio.Redis.from_url` to fakeredis for the
    test session so `Pipeline.on_startup` can build a working `ThreadVault`
    without a running Redis daemon. Tests that explicitly inject a client
    via `ThreadVault(client=...)` short-circuit `_get_client` and are
    unaffected.

    A single `FakeServer` backs every `from_url` call so state minted by
    one Pipeline instance is visible to a later one within the same test
    session — necessary for Task 5 cross-request consistency assertions
    against the module-scoped `started_pipeline` fixture.
    """
    from redis.asyncio import Redis

    server = fake_aioredis.FakeServer()
    original_from_url = Redis.from_url

    def _from_url(url: str, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        return fake_aioredis.FakeRedis(
            server=server,
            decode_responses=bool(kwargs.get("decode_responses", False)),
        )

    Redis.from_url = _from_url  # type: ignore[method-assign]
    try:
        yield
    finally:
        Redis.from_url = original_from_url  # type: ignore[method-assign]


@pytest.fixture
def pipeline() -> Pipeline:
    """Fresh Pipeline instance for each test."""
    return Pipeline()


@pytest_asyncio.fixture
async def fake_redis() -> AsyncIterator[fake_aioredis.FakeRedis]:
    """Per-test fakeredis client. `decode_responses=True` matches what
    `ThreadVault` configures on the real client, so HGET/HGETALL return
    `str` instead of `bytes`.
    """
    client = fake_aioredis.FakeRedis(decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


@pytest_asyncio.fixture
async def thread_vault(
    fake_redis: fake_aioredis.FakeRedis,
) -> AsyncIterator[ThreadVault]:
    """ThreadVault wired to fakeredis with short, observable test TTLs.

    `thread_ttl_seconds` is small enough that TTL-renewal tests can detect
    EXPIRE pushing the deadline back; `ephemeral_ttl_seconds` is smaller
    still so the prefix-driven TTL switch is observable as a numeric diff.
    The `fake_redis` fixture owns the client lifecycle, so this fixture
    only resets the vault's internal references on teardown.
    """
    vault = ThreadVault(
        thread_ttl_seconds=60,
        ephemeral_ttl_seconds=10,
        client=fake_redis,
    )
    try:
        yield vault
    finally:
        vault._client = None
        vault._lua = None


@pytest.fixture
def sample_user_body() -> dict[str, Any]:
    """Sample chat completion request body (OpenAI format)."""
    return {
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "Zovem se Ivan Horvat, OIB: 12345678901."},
        ],
        "metadata": {"chat_id": "test-chat-123"},
    }


@pytest.fixture
def sample_assistant_body() -> dict[str, Any]:
    """Sample chat completion response body."""
    return {
        "model": "gpt-4",
        "messages": [
            {"role": "assistant", "content": "Bok [PERSON_1], kako vam mogu pomoći?"},
        ],
    }
