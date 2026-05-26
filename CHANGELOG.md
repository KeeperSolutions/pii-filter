# Changelog

All notable changes to `pii-filter` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.5] — 2026-05-26

### Removed — Redis backend (Task 9, breaking)

The Redis vault backend, which was never enabled in production, has been
removed in full. Postgres is now the only supported backend.

- **Removed** the `class ThreadVault` Redis implementation (~187 lines), the
  `_LUA_GET_OR_MINT` Lua atomic-mint script, and the `redis.asyncio` imports.
- **Removed** the `vault_backend` Valves field. The `PII_FILTER_VAULT_BACKEND`
  env var is now silently ignored (pydantic-settings `extra="ignore"`).
- **Removed** the `redis_enabled`, `redis_url`, `redis_connect_timeout_ms`,
  and `redis_socket_timeout_ms` Valves fields. The corresponding
  `PII_FILTER_REDIS_*` env vars are now silently ignored.
- **Renamed** `PostgresThreadVault` → `ThreadVault`. There is no longer a
  polymorphic vault union; the inlet/outlet code paths see a single concrete
  type, and the `cast("ThreadVault | PostgresThreadVault", ...)` shims are
  gone.
- **Removed** the runtime dependency on `redis>=5.0.1` and the dev
  dependencies on `fakeredis>=2.20` and `lupa>=2.0`.

### Changed — inlet log format (breaking for log parsers)

- The inlet summary `logger.info` line no longer contains the `backend=`
  token. The remaining fields (`chat_id`, `thread_id`, `messages_processed`,
  `messages_skipped_already_masked`, `detections`, `masked`, `vault`,
  `languages_active`, `ner_spillover_dropped`, `user_masking_disabled`,
  `presidio_disabled`) are unchanged. The same change applies to the
  `presidio_disabled=True` branch of the log line.

### Changed — user-facing exception strings

- The three `RuntimeError` messages in `inlet`/`outlet` that previously read
  `"PII filter blocked the request: Redis thread vault is …"` now read
  `"PII filter blocked the request: vault is …"`. The exception type and the
  trailing `degradation_mode='block'` substring are unchanged, so callers
  matching against those parts continue to work.

### Changed — `on_startup` simplification

- The `if backend == "postgres" / elif backend == "redis" / else` dispatch
  collapses to a single Postgres init path gated by the existing
  `vault_enabled` Valves field. When `vault_enabled=False`, `self.vault`
  remains `None` and the inlet falls back to Task 4's per-request dicts on
  every call. The startup log line is reworded to include a `healthy=`
  token from the post-`initialize` healthcheck.

### Tests

- **Deleted** `tests/test_thread_vault.py` (Redis-backed, 20 tests).
- **Renamed** `tests/test_postgres_vault.py` → `tests/test_thread_vault.py`.
- **Added** `tests/helpers/mock_vault.py` exposing an in-memory
  `MockThreadVault` test double that implements the same public async API
  as the production `ThreadVault`.
- **Added** `tests/test_mock_vault.py` with 13 conformance tests.
- **Added** two regression tests on the Postgres vault that the deleted
  Redis suite previously covered:
  - `test_real_thread_uses_long_ttl` — asserts a non-ephemeral chat_id
    routes to `thread_ttl_seconds` (3600s) and not `ephemeral_ttl_seconds`
    (300s).
  - `test_postgres_healthcheck_returns_false_on_query_exception` — asserts
    `healthcheck()` returns False (does not raise) when the live pool's
    `SELECT 1` raises.
- **Migrated** the `started_pipeline` fixtures in `test_masking.py` and
  `test_recognizers.py` to use `MockThreadVault` via a module-scoped autouse
  monkeypatch on `pii_filter.ThreadVault`. This keeps the analyzer-heavy
  test paths runnable on Windows hosts without `pg_ctl` on PATH.
- **Renamed** `test_inlet_redis_down_*` → `test_inlet_vault_down_*` and
  **deleted** the duplicate `test_inlet_postgres_backend_down_*` cases.
- **Removed** the `_swap_redis_to_fakeredis`, `fake_redis`, and
  `thread_vault` fixtures from `tests/conftest.py`; added a new
  `mock_thread_vault` fixture.
- **Removed** the `test_valves_loads_vault_backend_from_env_var` and
  `test_valves_invalid_vault_backend_raises` tests from `test_skeleton.py`
  (the field they covered no longer exists). `test_lifecycle_hooks_dont_throw`
  now opts out of vault construction via `vault_enabled=False`.

### Docs

- README dependency table, valves table, deployment env-var block, and
  "Implemented tasks" log updated to reflect the single-backend layout.
- The `docs/references/README.md` reference to Redis is footnoted with the
  Task 9 / v0.9.5 consolidation note.

### Migration notes

- Cloud Run env vars `PII_FILTER_REDIS_URL` and `PII_FILTER_VAULT_BACKEND`
  are now silently ignored. Per `extra="ignore"` on Valves, no startup
  warning is logged. Operators may delete those env vars at their leisure;
  no action is required for the pipeline to come up cleanly.
- Log-parsing tooling that keyed off the `backend=` token in the inlet
  summary line must be updated.

---

Earlier versions are tracked in the "Implemented tasks" table in
`README.md` and the per-task completion documents under `tasks/`.
