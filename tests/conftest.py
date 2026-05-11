"""Pytest fixtures for PII Filter tests."""

from __future__ import annotations

import platform
import shutil
import subprocess
import tempfile
from collections.abc import AsyncIterator, Generator
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio
from fakeredis import aioredis as fake_aioredis
from pytest_postgresql import factories as _pg_factories
from pytest_postgresql.executor import PostgreSQLExecutor as _PgExecutor

# pytest-postgresql sets LC_ALL=C.UTF-8 in the initdb subprocess env, but
# the Windows `initdb` binary ignores it and falls back to the system
# locale (e.g. "Croatian_Croatia.1252"), for which PostgreSQL ships no
# text-search configuration — so initdb fails before the executor can
# start. Patch init_directory to pass `--locale=C --encoding=UTF8` to
# initdb explicitly so a known-good text-search config is always picked.
if platform.system() == "Windows":

    def _init_directory_locale_c(self: _PgExecutor) -> None:  # type: ignore[no-untyped-def]
        if self._directory_initialised:
            return
        self.clean_directory()
        cmd = [self.executable, "initdb", "--pgdata", self.datadir]
        opts = [f"--username={self.user}", "--locale=C", "--encoding=UTF8"]
        if self.password:
            with tempfile.NamedTemporaryFile() as pwfile:
                opts += ["--auth=password", f"--pwfile={pwfile.name}"]
                pw = (
                    self.password.encode("utf-8")
                    if hasattr(self.password, "encode")
                    else self.password
                )
                pwfile.write(pw)
                pwfile.flush()
                cmd += ["-o", " ".join(opts)]
                subprocess.check_output(cmd, env=self.envvars)
        else:
            opts += ["--auth=trust"]
            cmd += ["-o", " ".join(opts)]
            subprocess.check_output(cmd, env=self.envvars)
        self._directory_initialised = True

    _PgExecutor.init_directory = _init_directory_locale_c  # type: ignore[method-assign]

    # pytest-postgresql's BASE_PROC_START_COMMAND wraps `stderr` and the
    # unix-socket directory in single quotes. Windows cmd.exe doesn't strip
    # single quotes, so Postgres receives literally "'stderr'" (with quotes
    # in the value) and aborts with `invalid value for parameter
    # "log_destination": "'stderr'"`. Re-template without single quotes.
    _PgExecutor.BASE_PROC_START_COMMAND = (
        '{executable} start -D "{datadir}" '
        '-o "-F -p {port} -c log_destination=stderr '
        "-c logging_collector=off "
        '-c unix_socket_directories={unixsocketdir} {postgres_options}" '
        '-l "{logfile}" {startparams}'
    )

    # mirakuru's SimpleExecutor.stop calls os.killpg, which doesn't exist
    # on Windows. pytest-postgresql's stop() already calls `pg_ctl stop -m f`
    # first, which gracefully shuts the cluster down — the super().stop()
    # call after it is just process-group cleanup. On Windows, terminate the
    # Popen directly and skip the killpg path.
    def _stop_windows(self: _PgExecutor, sig=None, exp_sig=None):  # type: ignore[no-untyped-def]
        subprocess.check_output(
            f'"{self.executable}" stop -D "{self.datadir}" -m f',
            shell=True,
        )
        proc = getattr(self, "process", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001 — best-effort teardown
                try:  # noqa: SIM105
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
        # Mark the executor as stopped so mirakuru's __del__ doesn't retry.
        self._popen = None  # type: ignore[attr-defined]
        return self

    _PgExecutor.stop = _stop_windows  # type: ignore[method-assign]

from pii_filter import Pipeline, PostgresThreadVault, ThreadVault

if TYPE_CHECKING:
    import psycopg
    from pytest_postgresql.executor import PostgreSQLExecutor


# pytest-postgresql needs BOTH `pg_ctl` (to initdb/start/stop the cluster) and
# the `postgres` server binary at runtime. Skip if either is missing —
# `and` would only skip when both are absent, leaving tests to fail with a
# binary-not-found error on hosts that ship one but not the other.
postgres_binary_missing = shutil.which("pg_ctl") is None or shutil.which("postgres") is None


@pytest.fixture(scope="module", autouse=True)
def _swap_redis_to_fakeredis() -> Generator[None, None, None]:
    """Redirect `redis.asyncio.Redis.from_url` to fakeredis for each test
    module so `Pipeline.on_startup` can build a working `ThreadVault`
    without a running Redis daemon. Tests that explicitly inject a client
    via `ThreadVault(client=...)` short-circuit `_get_client` and are
    unaffected.

    A single `FakeServer` backs every `from_url` call within the current
    module, so state minted by one Pipeline instance is visible to a later
    one in the same module — preserving cross-request consistency for
    module-scoped fixtures such as `started_pipeline` while preventing
    Redis state from leaking across modules.
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


# ---------------------------------------------------------------------------
# Task 5.1 — PostgreSQL fixtures
# ---------------------------------------------------------------------------
#
# pytest-postgresql ships two factory helpers we use here:
#   * `postgresql_proc` — a session-scoped fixture that boots a real Postgres
#     process via `pg_ctl` and tears it down at session end.
#   * `postgresql` — a function-scoped fixture that template-clones a fresh
#     database against the running `postgresql_proc` for every test.
#
# Both require a `pg_ctl` / `postgres` binary on PATH. Tests that depend on
# them gate on `postgres_binary_missing` so a host without local Postgres
# still runs the rest of the suite.

postgresql_proc = _pg_factories.postgresql_proc()
postgresql = _pg_factories.postgresql("postgresql_proc")


# Module-scoped pipeline fixture uses its own DB name so it doesn't collide
# with the per-test `postgresql` fixture (which template-clones a "tests"
# database via DatabaseJanitor — DatabaseJanitor.init() has no IF NOT EXISTS
# guard, so a leftover DB from this fixture would break every postgres_vault
# test with DuplicateDatabase).
_PIPELINE_POSTGRES_DBNAME = "pipeline_tests"


def _proc_dsn(proc: PostgreSQLExecutor, dbname: str | None = None) -> str:
    """Build a libpq DSN pointing at the postgres-process default DB."""
    return f"postgresql://{proc.user}@{proc.host}:{proc.port}/{dbname or proc.dbname}"


@pytest_asyncio.fixture
async def postgres_vault(postgresql: psycopg.Connection[Any]) -> AsyncIterator[Any]:
    """Per-test PostgresThreadVault with a freshly initialized schema.

    Each test gets a brand-new database (template-cloned by pytest-postgresql)
    so tables start empty. We expose the vault rather than the connection so
    tests can drive the public async API.

    Short test TTLs (3600s thread / 300s ephemeral) make TTL-renewal tests
    observable as numeric diffs without slowing the suite down — pytest-
    postgresql's per-test DB cleanup is the heavy work here.
    """
    info = postgresql.info
    dsn = f"postgresql://{info.user}@{info.host}:{info.port}/{info.dbname}"
    vault = PostgresThreadVault(
        dsn=dsn,
        pool_min=1,
        pool_max=2,
        command_timeout=2.0,
        thread_ttl_seconds=3600,
        ephemeral_ttl_seconds=300,
    )
    await vault.initialize()
    try:
        yield vault
    finally:
        await vault.aclose()


@pytest_asyncio.fixture(loop_scope="module", scope="module")
async def started_pipeline_postgres(
    postgresql_proc: PostgreSQLExecutor,
) -> AsyncIterator[Pipeline]:
    """Module-scoped Pipeline started against the postgres backend.

    Heavy spaCy load happens once per module. The DSN points at the
    `postgresql_proc` default database; tests use unique chat_ids to avoid
    cross-test state collisions because we share a single DB across the
    module instead of cycling per-test DBs (the per-test fresh-DB pattern
    would re-run on_startup — and the spaCy load — on every test).
    """
    # `postgresql_proc` only initdb's the cluster; it does not create any
    # named database. We use a dedicated dbname (`_PIPELINE_POSTGRES_DBNAME`)
    # rather than `postgresql_proc.dbname` ("tests") because the latter is
    # owned by the per-test `postgresql` janitor fixture — leaving "tests"
    # behind would break every test_postgres_vault test with
    # DuplicateDatabase. CREATE DATABASE can't run inside a transaction,
    # hence asyncpg's autocommit-by-default for `execute`.
    import asyncpg

    admin_dsn = (
        f"postgresql://{postgresql_proc.user}@"
        f"{postgresql_proc.host}:{postgresql_proc.port}/postgres"
    )
    _admin = await asyncpg.connect(admin_dsn)
    try:
        existing = await _admin.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", _PIPELINE_POSTGRES_DBNAME
        )
        if not existing:
            await _admin.execute(f'CREATE DATABASE "{_PIPELINE_POSTGRES_DBNAME}"')
    finally:
        await _admin.close()

    dsn = _proc_dsn(postgresql_proc, _PIPELINE_POSTGRES_DBNAME)
    p = Pipeline()
    p.valves.vault_backend = "postgres"
    p.valves.postgres_url = dsn
    await p.on_startup()
    try:
        yield p
    finally:
        # Drop tables so a same-session re-run starts clean, then drop the
        # whole DB so we don't leak state to a later test session reusing
        # the same data directory.
        if isinstance(p.vault, PostgresThreadVault) and p.vault._pool is not None:
            async with p.vault._pool.acquire() as conn:
                await conn.execute("DROP TABLE IF EXISTS pii_thread_mappings, pii_thread_counters")
        await p.on_shutdown()
        _admin = await asyncpg.connect(admin_dsn)
        try:
            await _admin.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity " "WHERE datname = $1",
                _PIPELINE_POSTGRES_DBNAME,
            )
            await _admin.execute(f'DROP DATABASE IF EXISTS "{_PIPELINE_POSTGRES_DBNAME}"')
        finally:
            await _admin.close()
