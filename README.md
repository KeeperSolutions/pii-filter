# pii-filter

PII detection and masking pipeline for the Keeper AI Gateway (OpenWebUI Pipelines integration).

**Current version: 0.9.7** — multi-language (HR + EN), PostgreSQL vault with AES-256-GCM encryption-at-rest, multi-turn history, 19 entity types.

---

## How it works

`pii_filter.py` is a single-file OpenWebUI Pipeline that sits between the user and the LLM. Every user message passes through `inlet` before reaching the model — PII is replaced with typed placeholders. The LLM receives and responds with placeholders. `outlet` then swaps them back to the originals before the response reaches the user.

```
User types:   "My name is John Smith, SSN 123-45-6789, email john@example.com"
                                   │
                               [ inlet ]
                                   │
LLM receives: "My name is [PERSON_1], SSN [US_SSN_1], email [EMAIL_1]"
                                   │
                           LLM responds:
               "Got it. I have noted [US_SSN_1] for [PERSON_1] ([EMAIL_1])."
                                   │
                               [ outlet ]
                                   │
User sees:    "Got it. I have noted 123-45-6789 for John Smith (john@example.com)."
```

Placeholders are scoped per conversation thread — the same value always gets the same placeholder across turns (e.g. `[PERSON_1]` in message 3 maps to the same name as `[PERSON_1]` in message 1). Mappings are stored in a PostgreSQL vault with a 24-hour TTL.

---

## Detected entity types (19)

| Placeholder | Entity | Notes |
|---|---|---|
| `[PERSON_1]` | Full names | spaCy NER (HR + EN), deny-list and trailing-token strip |
| `[EMAIL_1]` | Email addresses | Presidio built-in |
| `[PHONE_1]` | Phone numbers | libphonenumber, disambiguates from OIB via context window |
| `[CREDIT_CARD_1]` | Credit card numbers | Luhn-validated |
| `[HR_OIB_1]` | Croatian personal ID (OIB) | Checksum-validated, 11 digits |
| `[HR_JMBG_1]` | Yugoslav birth number (JMBG) | 13 digits, date/region validated |
| `[HR_IBAN_1]` | Croatian IBAN | HR country code + checksum |
| `[IE_PPSN_1]` | Irish PPS number | Format + check character |
| `[IE_IBAN_1]` | Irish IBAN | IE country code + checksum |
| `[RO_CNP_1]` | Romanian personal ID | 13 digits, full checksum |
| `[RO_IBAN_1]` | Romanian IBAN | RO country code + checksum |
| `[UK_NINO_1]` | UK National Insurance number | Format + prefix/suffix rules |
| `[UK_UTR_1]` | UK Unique Tax Reference | 10 digits, keyword context required |
| `[GB_IBAN_1]` | UK IBAN | GB country code + checksum |
| `[US_SSN_1]` | US Social Security Number | Format + reserved block rejection |
| `[US_EIN_1]` | US Employer Identification Number | `XX-XXXXXXX` format |
| `[UK_NHS_1]` | UK NHS number | Mod-11 validated |
| `[IBAN_CODE_1]` | Generic IBAN (DE, FR, ES, IT…) | Presidio built-in, EN analyzer only |

---

## Examples

### Basic masking and restoration

**Inlet — user message before it reaches the LLM:**

```
Input:
  "Please verify my credit card 4111111111111111, expiry 12/27, and email john@acme.com"

After inlet:
  "Please verify my credit card [CREDIT_CARD_1], expiry 12/27, and email [EMAIL_1]"

Vault stores:
  CREDIT_CARD_1 → "4111111111111111"
  EMAIL_1       → "john@acme.com"
```

Note: `12/27` is not masked — date substrings inside card numbers are intentionally excluded to avoid false positives (v0.9.2 fix).

**Outlet — LLM response restored before reaching the user:**

```
LLM response: "I have registered [CREDIT_CARD_1] linked to [EMAIL_1]."
After outlet: "I have registered 4111111111111111 linked to john@acme.com."
```

---

### Multi-turn consistency

The same PII value always gets the same placeholder across the entire conversation thread.

**Turn 1:**
```
User:      "My name is Alice Johnson, NINO AB123456C"
→ inlet:   "My name is [PERSON_1], NINO [UK_NINO_1]"
Assistant: "Hello [PERSON_1], I have your NINO on file."
```

**Turn 2 — same name appears again:**
```
User:      "Can you confirm the NINO for Alice Johnson once more?"
→ inlet:   "Can you confirm the NINO for [PERSON_1] once more?"
           ↑ vault returns the existing PERSON_1 mapping — no new placeholder minted
```

**Turn 2 — already-masked history messages are skipped:**
```
body.messages = [
  {"role": "user",      "content": "My name is [PERSON_1], NINO [UK_NINO_1]"},  ← skipped (already masked)
  {"role": "assistant", "content": "Hello [PERSON_1], I have your NINO on file."},
  {"role": "user",      "content": "Can you confirm the NINO for Alice Johnson once more?"}  ← re-masked
]
```

---

### Cross-language NER spillover guard

When both HR and EN analyzers run on the same text, the EN spaCy model can fire on Croatian words and vice versa. The spillover guard classifies the ±30-character window around each NER detection and drops it if the window language does not match the source analyzer.

```
Input: "Send the report to ana.kovac@keeper.hr and call her on +385 91 234 5678"

HR analyzer → EMAIL_ADDRESS, PHONE_NUMBER  (correct)
EN analyzer → PERSON on "ana"              (false positive — Croatian context)

Window classifier sees Croatian context around "ana" → drops EN PERSON detection

Result: only [EMAIL_1] and [PHONE_1] are masked, no spurious [PERSON_1]
```

---

## Architecture

```
[OpenWebUI] ──HTTP──► [Pipelines Container]
                             │
                        pii_filter.py
                   ┌─────────────────────────────────┐
                   │  on_startup                      │
                   │    ├─ HR AnalyzerEngine           │
                   │    │   (hr_core_news_lg + custom) │
                   │    └─ EN AnalyzerEngine           │
                   │        (en_core_web_lg + custom)  │
                   │                                   │
                   │  inlet(body)                      │
                   │    ├─ skip OWU background tasks   │
                   │    ├─ run HR + EN analyzers        │
                   │    ├─ merge + NER spillover guard  │
                   │    ├─ _select_accepted_detections  │
                   │    │    ├─ deny-list filter        │
                   │    │    ├─ trailing-token strip    │
                   │    │    ├─ OIB/phone context check │
                   │    │    └─ overlap resolution      │
                   │    ├─ replace spans → placeholders │
                   │    └─ write to ThreadVault         │
                   │                                   │
                   │  outlet(body)                     │
                   │    ├─ read reverse map (metadata) │
                   │    │   fallback: read from vault  │
                   │    └─ restore placeholders        │
                   └─────────────────────────────────┘
                             │
                       [PostgreSQL]  ◄── PII_FILTER_POSTGRES_URL
                       ThreadVault — thread-scoped, TTL 24h
```

---

## Tech stack

| Component | Version | Role |
|---|---|---|
| Python | 3.11 | Pipelines container constraint |
| Microsoft Presidio | ≥ 2.2.0 | Detection engine + anonymizer |
| spaCy `hr_core_news_lg` | 3.7.0 | Croatian NER |
| spaCy `en_core_web_lg` | 3.7.0 | English NER |
| asyncpg | ≥ 0.29.0 | PostgreSQL vault (async connection pool) |
| pydantic-settings | ≥ 2.0 | Valve env-var loading |
| cryptography | ≥ 42.0 | AES-256-GCM vault encryption-at-rest (ENC1 envelope) |

---

## Configuration (Valves)

All valves are visible in OpenWebUI Admin → Pipelines and can be overridden via env vars prefixed with `PII_FILTER_`.

| Valve | Default | Env var | Description |
|---|---|---|---|
| `enabled` | `true` | `PII_FILTER_ENABLED` | Master on/off switch |
| `languages` | `["hr","en"]` | `PII_FILTER_LANGUAGES` | Active analyzers — `"hr"`, `"en"`, or both |
| `postgres_url` | `""` | `PII_FILTER_POSTGRES_URL` | Full DSN — required when `vault_enabled=true` |
| `vault_enabled` | `true` | `PII_FILTER_VAULT_ENABLED` | Disable vault (use per-request maps only) |
| `degradation_mode` | `"block"` | `PII_FILTER_DEGRADATION_MODE` | `"block"` (fail-closed, GDPR-safe) or `"passthrough"` |
| `multi_turn_history_scope` | `true` | `PII_FILTER_MULTI_TURN_HISTORY_SCOPE` | Mask all user messages in history, not just the latest |
| `multi_turn_history_max_messages` | `20` | `PII_FILTER_MULTI_TURN_HISTORY_MAX_MESSAGES` | Max user messages processed per request |
| `ner_deny_list` | *(built-in list)* | `PII_FILTER_NER_DENY_LIST` | Suppress false-positive PERSON entities by exact match |
| `ner_oib_phone_context_window` | `30` | `PII_FILTER_NER_OIB_PHONE_CONTEXT_WINDOW` | Look-back window (chars) to detect phone context near 11-digit numbers |
| `vault_encryption_enabled` | `false` | `PII_FILTER_VAULT_ENCRYPTION_ENABLED` | Store `original_value` as an AES-256-GCM `ENC1:` envelope (vs plaintext) |
| `vault_encryption_strict` | `false` | `PII_FILTER_VAULT_ENCRYPTION_STRICT` | Refuse to serve an unexpected plaintext row (recommended `true` in prod) |
| `vault_kms_backend` | `"local"` | `PII_FILTER_VAULT_KMS_BACKEND` | Key source — `"local"` (env base64) or `"gcp_kms"` (Secret Manager) |
| `vault_encryption_key` | `""` | `PII_FILTER_VAULT_ENCRYPTION_KEY` | base64 32-byte AES key (local backend; required when encryption enabled) |
| `vault_blind_index_key` | `""` | `PII_FILTER_VAULT_BLIND_INDEX_KEY` | base64 32-byte HMAC key (local backend; **always** required when vault runs) |
| `vault_encryption_key_id` | `1` | `PII_FILTER_VAULT_ENCRYPTION_KEY_ID` | Envelope key_id (u32), packed for future key rotation |
| `vault_gcp_enc_secret` | `""` | `PII_FILTER_VAULT_GCP_ENC_SECRET` | Secret Manager resource name for the enc key (`gcp_kms` backend) |
| `vault_gcp_blind_secret` | `""` | `PII_FILTER_VAULT_GCP_BLIND_SECRET` | Secret Manager resource name for the blind-index key (`gcp_kms` backend) |

**Per-user toggle (UserValves):** each user can disable PII masking for their own sessions via `pii_masking_enabled`.

---

## Vault encryption-at-rest (Option E)

The vault stores PII under application-layer **AES-256-GCM** encryption so the
`pii_thread_mappings.original_value` column never holds plaintext customer data
(GDPR Art. 32). Because a random-nonce cipher would break the UPSERT dedup
(same value → different ciphertext → no conflict), dedup is preserved by a
keyed **blind index**: `lookup_hash = HMAC-SHA256(blind_key, framed(chat_id,
entity_type, plaintext))`. `lookup_hash` is `NOT NULL` and part of the primary
key, so it is **always** computed when the vault runs — the blind-index key is
required whenever `vault_enabled=true`, even with encryption disabled.
`vault_encryption_enabled` only controls whether `original_value` is ciphertext
or plaintext.

Stored envelope format: `ENC1:<base64( [1B version][4B key_id][12B nonce][ciphertext‖16B GCM tag] )>`.

**Key management**

- **`local` (dev / self-hosted):** keys are base64-encoded 32-byte values in
  `PII_FILTER_VAULT_ENCRYPTION_KEY` / `PII_FILTER_VAULT_BLIND_INDEX_KEY`.
  Generate one with:
  ```bash
  python -c "import os,base64; print(base64.b64encode(os.urandom(32)).decode())"
  ```
  See `.env.example` for a full local block. The enc key and blind-index key
  must be **independent** keys.
- **`gcp_kms` (production):** set `PII_FILTER_VAULT_KMS_BACKEND=gcp_kms` and
  point `PII_FILTER_VAULT_GCP_ENC_SECRET` / `PII_FILTER_VAULT_GCP_BLIND_SECRET`
  at Google Secret Manager resource names whose payloads are the base64 keys.
  This path lazy-imports `google-cloud-secret-manager`, which is **not** in the
  default container image — **handoff to Senka:** add it to the prod image / a
  prod requirements profile before enabling `gcp_kms`.

Keys are validated fail-closed at `on_startup` (empty / non-base64 / non-32-byte
→ `RuntimeError`); no key material is ever logged. Set
`vault_encryption_strict=true` in production so an unexpected plaintext row is
refused rather than served.

**Breaking schema change (existing dev DBs):** `CREATE TABLE IF NOT EXISTS`
does not migrate an existing table, so a local `keeper-postgres` volume with
the pre-v0.9.7 schema must be dropped before first run:

```sql
DROP TABLE IF EXISTS pii_thread_mappings;
DROP TABLE IF EXISTS pii_thread_counters;
```

Production is greenfield (empty DB) → the new schema is created on first
`initialize()`; no backfill is required.

---

## Development

### Prerequisites

- Python 3.11 (NOT 3.12 — the Pipelines container is locked to 3.11)
- Docker (for local Postgres)

### Setup

```bash
# Create and activate venv
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate    # Linux / Mac / Git Bash

# Install dev dependencies
pip install -r requirements-dev.txt
```

### Local Postgres (default backend)

```bash
docker run -d --name keeper-postgres -p 5432:5432 \
  -e POSTGRES_USER=keeper -e POSTGRES_PASSWORD=keeper -e POSTGRES_DB=keeper \
  postgres:16-alpine
```

Export the DSN before running tests or the pipeline locally:

```bash
export PII_FILTER_POSTGRES_URL="postgresql://keeper:keeper@localhost:5432/keeper"
```

### Tests

```bash
pytest                     # full suite — 288 passed, 5 skipped
pytest -k "test_inlet"     # run a subset
```

The test suite uses `pytest-postgresql` (real Postgres spun up per session) for vault integration tests, and `MockThreadVault` (in-memory) for analyzer/inlet tests — no external services required.

### Lint, format, type check

```bash
ruff check .               # lint
ruff format --check .      # format check
mypy --strict pii_filter.py  # type check

ruff format .              # auto-format
```

All three gates must pass before merging. Current status: ✅ `ruff check` ✅ `ruff format` ✅ `mypy --strict` ✅ `pytest 288 passed`.

---

## Deployment

1. **First-time deploy (with new dependencies):** set the `PIPELINES_URLS` env var on the Pipelines Cloud Run service to the GitHub raw URL of `pii_filter.py`, then restart the container. The container parses the `requirements:` frontmatter and installs all dependencies.

2. **Code-only updates:** upload the updated `pii_filter.py` via OpenWebUI Admin → Pipelines → Upload Pipeline. Does not install new dependencies — use step 1 if `requirements:` changed.

3. **Required env vars on Cloud Run:**

```
PII_FILTER_POSTGRES_URL=postgresql://user:pass@/db?host=/cloudsql/<INSTANCE_CONN_NAME>
```

---

## Implemented tasks

| Task | Ticket | What was delivered |
|---|---|---|
| Task 1 | TRAU-400 | Repo skeleton, tooling (ruff, mypy, pytest), Pipeline class stub |
| Task 3 | TRAU-401 | Presidio engine + 12 custom recognizers (OIB, JMBG, IBAN ×4, IE PPSN, RO CNP, UK NINO, UK UTR, US SSN, US EIN) |
| Task 4 | TRAU-410 | Inlet masking — PII → typed placeholders, forward/reverse maps, MISC entity suppression |
| Task 5 *(historical)* | TRAU-414 | Redis ThreadVault — thread-scoped placeholder consistency, 24h TTL. **Removed in Task 9 (v0.9.5).** |
| Task 6 | TRAU-416 | Outlet restoration — placeholder → original in LLM responses, hallucination handling |
| Task 5.1 | TRAU-422 | PostgreSQL ThreadVault — asyncpg pool, idempotent DDL; default backend since v0.6.0 |
| Task 3.1 | TRAU-424 | Recognizer accuracy — deny-list, trailing-token strip, OIB/phone context window |
| Task 8.5 | TRAU-425 | Multi-turn history — masks all user messages in history, already-masked skip optimisation |
| Task 3.2 | TRAU-426 | Dual-analyzer architecture (HR + EN), result merge and deduplication |
| Task 3.3 | TRAU-426 | NER spillover guard — window language classifier eliminates cross-language false positives |
| Task 8 | TRAU-451 | UserValves `pii_masking_enabled` per-user toggle wired into inlet; Valves `presidio_enabled` admin kill switch |
| Task 9 | — | Redis backend removal — `PostgresThreadVault` renamed to `ThreadVault`; `vault_backend` / `redis_*` valves dropped; `redis` / `fakeredis` / `lupa` deps gone (v0.9.5) |
| Task 11 | — | Vault encryption-at-rest (Option E) — AES-256-GCM `ENC1` envelope for `original_value` + HMAC-SHA256 blind index (`lookup_hash`); `local` / `gcp_kms` key backends; `cryptography` dep (v0.9.7) |

---

## License

MIT
