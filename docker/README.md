# Local OpenWebUI test stack

Runs this repo's `pii_filter.py` as a filter inside an OpenWebUI Pipelines
server, backed by Postgres, so the masking behaviour can be exercised from a
real chat UI.

## Setup

Create the env file and generate a vault key:

```sh
cp docker/.env.example docker/.env
python3 -c "import os,base64; print(base64.b64encode(os.urandom(32)).decode())"
# paste the result as PII_FILTER_VAULT_BLIND_INDEX_KEY in docker/.env
```

`docker/.env` is gitignored. No key is committed to the repo — the compose file
fails immediately with a named error if the variable is missing, rather than
booting a half-configured vault.

Keep that key **stable** for the life of the `pii-postgres` volume. The blind
index is a keyed HMAC, so changing it orphans every row already written:
existing placeholders stop resolving and outlet quietly stops restoring
originals.

## Run

From the **repo root** (the build context is `..` so the Dockerfile can reach
`pii_filter.py` and `requirements.txt`):

```sh
docker compose -f docker/docker-compose.yaml up -d --build
```

Then open <http://localhost:3000>.

The first build takes ~20-30 min and produces a multi-GB image — almost all of
it CPU torch plus the three models (spaCy `hr`/`en` 3.7.0 and GLiNER2-PII,
vendored so startup needs no HuggingFace egress). It builds from public images
only, so it is reproducible on any machine, and it caches afterwards.

| Service      | Port | Purpose                                  |
|--------------|------|------------------------------------------|
| `open-webui` | 3000 | Chat UI                                  |
| `pipelines`  | 9099 | Pipelines server hosting the PII filter  |
| `postgres`   | —    | Thread vault storage                     |

OpenWebUI is preconfigured to see the Pipelines server, so the filter appears
under **Admin > Settings > Pipelines**. You still need a real LLM connection
for chat — add one in the UI, or fill in the commented `OPENAI_API_BASE_URLS` /
`OPENAI_API_KEYS` pair in the compose file.

## Iterating on the filter

Edit `pii_filter.py`, then:

```sh
docker compose -f docker/docker-compose.yaml restart pipelines
```

No rebuild. The repo root is mounted read-only at `/src`, and the container's
entrypoint copies `/src/pii_filter.py` into place on every start, so a restart
always runs the current code.

A rebuild (`up -d --build pipelines`) is only needed when the *image* changes —
a new dependency in the frontmatter `requirements:` line, or a change to the
Dockerfile.

Startup takes ~30-60s regardless: the spaCy models and the 300M-parameter
GLiNER model load into RAM. That is the floor on the iteration loop.

### Why the whole repo is mounted, not just the one file

A single-file bind mount pins one inode. Editors that save atomically (write a
temp file, then rename over the original) swap the inode, so the container
would keep seeing the *original* file indefinitely while the host shows the new
content — a genuinely confusing failure, because nothing errors. Mounting the
directory and reading a path inside it re-resolves on every open, which
survives atomic saves.

The image still `COPY`s the filter at build time. That is what keeps the image
self-contained if it is ever run without this compose file; the mount just
shadows it.

## Watching what the filter does

`PII_DEBUG_UNMASK_LOG` is on, which enables the TRAU-529 mask report — original
vs masked text plus a `value -> placeholder` list with per-value occurrence
counts. It is the fastest way to see the effect of a change:

```sh
docker compose -f docker/docker-compose.yaml logs -f pipelines | grep PII_DEBUG
```

## Configuration notes

Valves are settable as `PII_FILTER_*` env vars, or in the OpenWebUI admin UI.
Three defaults in this repo differ from the deployment copy, and the compose
file leaves all three at the **repo** default:

- `degradation_mode: "block"` — fail-closed. If the analyzer errors, the
  request is rejected rather than forwarded unfiltered. Kept so failures are
  loud during testing; the deployment copy uses `passthrough`.
- `vault_encryption_enabled: false` — no ENC1 envelope. See `.env.example` for
  how to turn it on.
- `priority: 0` — the deployment copy uses `-10` purely to order the PII filter
  ahead of the Langfuse filter, which this stack does not run.

The blind-index key **is** required even with encryption off (`lookup_hash` is
`NOT NULL` and part of the primary key, spec D1), which is why setup asks for
one regardless.

## Teardown

```sh
docker compose -f docker/docker-compose.yaml down          # keep vault data
docker compose -f docker/docker-compose.yaml down -v       # wipe volumes too
```
