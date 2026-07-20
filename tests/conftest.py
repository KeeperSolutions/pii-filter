"""Pytest fixtures for PII Filter tests."""

from __future__ import annotations

import base64
import platform
import shutil
import subprocess
import tempfile
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio
from presidio_analyzer import RecognizerResult
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

from pii_filter import BlindIndex, GLiNER2Detector, Pipeline, ThreadVault, VaultCipher
from tests.helpers.mock_vault import MockThreadVault

# Task 11: two fixed, independent 32-byte test keys (base64) so the
# Postgres-backed vault exercises the real AES-256-GCM encrypt/decrypt path and
# the HMAC blind index. Distinct keys assert the enc-key / blind-key separation
# (E2). Dev/test only — never used in any deployed environment.
VAULT_TEST_ENC_KEY = bytes(range(32))  # 0x00..0x1f
VAULT_TEST_BLIND_KEY = bytes(range(32, 64))  # 0x20..0x3f

if TYPE_CHECKING:
    import psycopg
    from pytest_postgresql.executor import PostgreSQLExecutor


# pytest-postgresql needs BOTH `pg_ctl` (to initdb/start/stop the cluster) and
# the `postgres` server binary at runtime. Skip if either is missing —
# `and` would only skip when both are absent, leaving tests to fail with a
# binary-not-found error on hosts that ship one but not the other.
postgres_binary_missing = shutil.which("pg_ctl") is None or shutil.which("postgres") is None


# The real, unpatched bound methods, captured before `_stub_gliner_model`
# installs the session-wide stubs.
_REAL_GLINER_LOAD = GLiNER2Detector.load
_REAL_GLINER_DETECT = GLiNER2Detector.detect


class _DummyGlinerModel:
    """Sentinel installed by the stubbed `load()` in place of the torch model."""


def _stub_gliner_load(self: GLiNER2Detector) -> None:
    self._model = _DummyGlinerModel()


def _stub_gliner_detect(self: GLiNER2Detector, text: str) -> list[RecognizerResult]:
    """Return [] only for a detector carrying the stubbed-in dummy model.

    The TRAU-522/529 suites (chunking, sub-token collapse, adjacency merge)
    construct a real `GLiNER2Detector` and inject their OWN fake model, then
    assert on what the genuine `detect()` does with it — chunk iteration, offset
    remap, dedup, containment collapse. A blanket `return []` would silently turn
    every one of those assertions into "no detections found" rather than failing
    loudly, so the short-circuit is narrowed to the dummy model that only the
    stubbed `load()` installs.
    """
    if isinstance(self._model, _DummyGlinerModel):
        return []
    return _REAL_GLINER_DETECT(self, text)


@pytest.fixture(scope="session", autouse=True)
def _stub_gliner_model() -> Iterator[None]:
    """GLiNER2's real model pulls torch + HF weights, unavailable in tests.

    With ``gliner_enabled`` defaulting to True, ``Pipeline.on_startup`` would
    call ``GLiNER2Detector.load()`` and crash on the missing ``gliner2`` import.
    Stub ``load()`` to a no-op and ``detect()`` to ``[]`` so on_startup runs;
    existing tests rely on pattern recognizers (OIB etc.), not PERSON NER, so
    they are unaffected. The GLiNER detection path has its own dedicated test
    that injects a canned detector.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(GLiNER2Detector, "load", _stub_gliner_load)
        mp.setattr(GLiNER2Detector, "detect", _stub_gliner_detect)
        yield


@pytest.fixture
def pipeline() -> Pipeline:
    """Fresh Pipeline instance for each test."""
    return Pipeline()


# ---------------------------------------------------------------------------
# TRAU-522 / TRAU-529 — in-memory fakes for the fast (no-Postgres, no-torch)
# inlet/outlet tests ported from the pipelines-v4 deployment repo.
# ---------------------------------------------------------------------------
#
# The v4 harness stubs `presidio_analyzer` in `sys.modules` and loads the
# pipeline by file path, because that repo does not install presidio. Here
# presidio IS a real dependency and the module under test is importable as
# `pii_filter`, so `pii_mod` is simply the real module and `RecognizerResult`
# the real presidio value object. Keeping the *names* identical is what lets
# the ported test files run unmodified.

import pii_filter as pii_mod  # noqa: E402


class FakeVault:
    """In-memory ThreadVault stand-in.

    `get_placeholder` is idempotent per `(chat_id, entity_type, original)` and
    mints `[TYPE_n]` with a per-(chat_id, entity_type) counter — mirroring the
    real vault's `ON CONFLICT ... RETURNING placeholder` contract without a
    Postgres round-trip. This is what lets a repeated entity (same chat_id +
    same original) reuse the same placeholder across the main chat and a
    background task.
    """

    def __init__(self) -> None:
        # (chat_id, entity_type, lookup_value) -> placeholder
        self._mappings: dict[tuple[str, str, str], str] = {}
        # same key -> the LITERAL original stored for it (first-write-wins)
        self._originals: dict[tuple[str, str, str], str] = {}
        # (chat_id, entity_type) -> next index
        self._counters: dict[tuple[str, str], int] = {}
        self.get_placeholder_calls: list[tuple[str, str, str]] = []

    async def healthcheck(self) -> bool:
        return True

    async def get_or_create_thread(self, chat_id: str) -> None:
        return None

    async def get_placeholder(
        self,
        chat_id: str,
        original: str,
        entity_type: str,
        lookup_value: str | None = None,
    ) -> str:
        """Mirrors `ThreadVault.get_placeholder`, including the TRAU-530 split:
        dedup on `lookup_value` (normalized identity), store `original` (the
        literal text). First-write-wins on the stored value, matching the real
        vault's `ON CONFLICT DO UPDATE` which never rewrites `original_value`."""
        self.get_placeholder_calls.append((chat_id, original, entity_type))
        key = (chat_id, entity_type, original if lookup_value is None else lookup_value)
        existing = self._mappings.get(key)
        if existing is not None:
            return existing
        n = self._counters.get((chat_id, entity_type), 0) + 1
        self._counters[(chat_id, entity_type)] = n
        placeholder = f"[{entity_type}_{n}]"
        self._mappings[key] = placeholder
        self._originals[key] = original
        return placeholder

    async def snapshot_for_request(self, chat_id: str) -> tuple[dict[str, str], dict[str, str]]:
        forward: dict[str, str] = {}
        reverse: dict[str, str] = {}
        for key, placeholder in self._mappings.items():
            if key[0] == chat_id:
                # Keyed/valued on the LITERAL stored text, never the lookup key —
                # `_build_vault_remasker` matches these against raw message text.
                original = self._originals[key]
                forward[original] = placeholder
                reverse[placeholder] = original
        return forward, reverse


class FakeAnalyzer:
    """Deterministic analyzer that flags configured substrings as entities.

    Replaces the spaCy-backed AnalyzerEngine. `spans` maps a literal substring
    to an entity_type; every occurrence in `text` is returned as a
    RecognizerResult, so tests control detection exactly.
    """

    def __init__(self, spans: dict[str, str], score: float = 0.95) -> None:
        self._spans = spans
        self._score = score

    def analyze(self, text: str, language: str) -> list[RecognizerResult]:
        results = []
        for needle, entity_type in self._spans.items():
            start = text.find(needle)
            while start != -1:
                results.append(
                    RecognizerResult(
                        entity_type=entity_type,
                        start=start,
                        end=start + len(needle),
                        score=self._score,
                    )
                )
                start = text.find(needle, start + 1)
        return results


class FakeGliner:
    """In-memory stand-in for GLiNER2Detector reproducing the recon's root cause.

    Real GLiNER (neural) re-detects existing placeholders as entities
    ([PERSON_1]->PERSON 0.862, [ADDRESS_1]->ADDRESS 0.937, ...). This fake does
    the same: every `[TYPE_N]` placeholder in the text is emitted as a PERSON
    detection (a whitelisted type, so absent the overlap-filter it would survive
    to the mask loop and renumber). Configured real-name substrings are emitted
    too — mirroring GLiNER being the production PERSON source (spaCy NER off).
    """

    def __init__(self, name_spans: dict[str, str] | None = None, score: float = 0.9) -> None:
        self.name_spans = name_spans or {}
        self._score = score

    def detect(self, text: str) -> list[RecognizerResult]:
        results = []
        # Re-detect placeholders (the dangerous behavior the filter must catch).
        for m in pii_mod._PLACEHOLDER_RE.finditer(text):
            results.append(
                RecognizerResult(
                    entity_type="PERSON", start=m.start(), end=m.end(), score=self._score
                )
            )
        # Real entities.
        for needle, entity_type in self.name_spans.items():
            start = text.find(needle)
            while start != -1:
                results.append(
                    RecognizerResult(
                        entity_type=entity_type,
                        start=start,
                        end=start + len(needle),
                        score=self._score,
                    )
                )
                start = text.find(needle, start + 1)
        return results


def make_gliner_pipeline(
    *, masking_enabled: bool = True, name_spans: dict[str, str] | None = None
) -> Pipeline:
    """Pipeline whose PERSON source is a FakeGliner (mirrors production: GLiNER on,
    spaCy NER off). analyzer_hr is a no-op non-None analyzer so inlet's
    'analyzer loaded' guard passes; all detection comes from the fake GLiNER."""
    pipe = Pipeline()
    pipe.valves.enabled = True
    pipe.valves.presidio_enabled = True
    pipe.valves.vault_enabled = True
    pipe.user_valves.pii_masking_enabled = masking_enabled
    pipe.analyzer_hr = FakeAnalyzer({})  # returns [] but non-None (guard)
    pipe.analyzer_en = None
    pipe._gliner = FakeGliner(name_spans=name_spans or {"Jimmy Page": "PERSON"})
    pipe.vault = FakeVault()
    return pipe


def make_pipeline(*, masking_enabled: bool = True, person_needle: str = "Ivan Horvat") -> Pipeline:
    """Build a Pipeline wired with in-memory fakes, no on_startup, no DB.

    Analyzer detections run through the HR analyzer with no EN markers in the
    fixture text, so the spillover guard classifies the window as 'hr' (the
    tie-break default) and keeps the PERSON detection.
    """
    pipe = Pipeline()
    pipe.valves.enabled = True
    pipe.valves.presidio_enabled = True
    pipe.valves.vault_enabled = True
    pipe.user_valves.pii_masking_enabled = masking_enabled
    pipe.analyzer_hr = FakeAnalyzer({person_needle: "PERSON"})
    pipe.analyzer_en = None
    pipe._gliner = None
    pipe.vault = FakeVault()
    return pipe


def user_payload(masking_enabled: bool) -> dict[str, Any]:
    return {"valves": {"pii_masking_enabled": masking_enabled}}


@pytest.fixture
def pii():
    return pii_mod


@pytest.fixture
def make_pipe():
    return make_pipeline


@pytest_asyncio.fixture
async def mock_thread_vault() -> AsyncIterator[MockThreadVault]:
    """Per-test `MockThreadVault` instance — in-memory thread vault double
    for tests that need a vault without a real backend.
    """
    vault = MockThreadVault()
    try:
        yield vault
    finally:
        await vault.aclose()


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
    """Per-test ThreadVault with a freshly initialized schema.

    Each test gets a brand-new database (template-cloned by pytest-postgresql)
    so tables start empty. We expose the vault rather than the connection so
    tests can drive the public async API.

    Short test TTLs (3600s thread / 300s ephemeral) make TTL-renewal tests
    observable as numeric diffs without slowing the suite down — pytest-
    postgresql's per-test DB cleanup is the heavy work here.
    """
    info = postgresql.info
    dsn = f"postgresql://{info.user}@{info.host}:{info.port}/{info.dbname}"
    # Task 11: encryption ON so the public-API round-trip tests exercise the
    # real encrypt-on-write / decrypt-on-read path transparently (spec §10).
    vault = ThreadVault(
        dsn=dsn,
        pool_min=1,
        pool_max=2,
        command_timeout=2.0,
        thread_ttl_seconds=3600,
        ephemeral_ttl_seconds=300,
        cipher=VaultCipher(VAULT_TEST_ENC_KEY, key_id=1),
        blind_index=BlindIndex(VAULT_TEST_BLIND_KEY),
        encryption_strict=False,
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
    p.valves.postgres_url = dsn
    p.valves.languages = ["hr"]  # HR-only for postgres fixture; avoids EN model load
    # Task 11: encryption ON so the pipeline integration tests exercise the
    # full encrypt-on-write / decrypt-on-read path through inlet/outlet.
    p.valves.vault_encryption_enabled = True
    p.valves.vault_encryption_key = base64.b64encode(VAULT_TEST_ENC_KEY).decode("ascii")
    p.valves.vault_blind_index_key = base64.b64encode(VAULT_TEST_BLIND_KEY).decode("ascii")
    await p.on_startup()
    try:
        yield p
    finally:
        # Drop tables so a same-session re-run starts clean, then drop the
        # whole DB so we don't leak state to a later test session reusing
        # the same data directory.
        if isinstance(p.vault, ThreadVault) and p.vault._pool is not None:
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
