# Changelog

All notable changes to `pii-filter` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.7] — 2026-06-03

### Added — vault encryption-at-rest (Task 11, Option E)

Application-layer AES-256-GCM encryption of the stored PII so the vault can
hold customer data under GDPR Art. 32 in a multi-tenant deployment. The
`original_value` column now stores a random-nonce `ENC1:` envelope; dedup is
preserved via a keyed HMAC **blind index** (`lookup_hash`).

- **Added** an inline crypto section in `pii_filter.py` (above `class
  ThreadVault`): `VaultCipher` (AES-256-GCM `ENC1:<base64(...)>` envelope —
  `[1B version][4B key_id][12B nonce][ciphertext‖16B tag]`, ported from the
  keeper-openwebui `crypto.py`), `BlindIndex` (HMAC-SHA256 with domain-tagged,
  length-prefixed framing of `chat_id‖entity_type‖plaintext`), and
  `KeyManager` (dual `local` / `gcp_kms` key backend).
- **Added** eight `Valves` fields (env prefix `PII_FILTER_`):
  `vault_encryption_enabled` (default `false`), `vault_encryption_strict`
  (default `false`), `vault_kms_backend` (`local`|`gcp_kms`, default `local`),
  `vault_encryption_key`, `vault_blind_index_key`, `vault_encryption_key_id`
  (default `1`), `vault_gcp_enc_secret`, `vault_gcp_blind_secret`.
- **Added** fail-closed key validation at `on_startup`: the blind-index key is
  always required (it backs the NOT NULL `lookup_hash`); the encryption key is
  required when encryption is enabled. An empty / non-base64 / non-32-byte key
  raises `RuntimeError` at startup. No key material is ever logged.
- **Added** `cryptography>=42.0` to the `requirements:` frontmatter and
  `requirements.txt`. `google-cloud-secret-manager` is **not** in the default
  deps — it is lazy-imported only in the `gcp_kms` backend (heavy grpc/protobuf
  chain; added to the prod image by Senka).
- **Decrypt never crashes the outlet:** a row that fails GCM tag verification
  (tamper / wrong key) or is unexpectedly plaintext in strict mode is logged
  and **skipped** — its placeholder stays masked in the user-facing text.

### Changed — schema (breaking for existing dev DBs)

- `pii_thread_mappings` gains `lookup_hash BYTEA NOT NULL`; the primary key
  moves from `(chat_id, entity_type, original_value)` to
  `(chat_id, entity_type, lookup_hash)`. `pii_thread_counters` and both
  indexes are unchanged.
- **Breaking for dev:** `CREATE TABLE IF NOT EXISTS` does **not** migrate an
  existing table, so a local `keeper-postgres` volume with the old schema must
  be dropped before first run on v0.9.7:

  ```sql
  DROP TABLE IF EXISTS pii_thread_mappings;
  DROP TABLE IF EXISTS pii_thread_counters;
  ```

  Production is greenfield (empty DB) → first `initialize()` creates the new
  schema; no backfill / data migration is required.

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
