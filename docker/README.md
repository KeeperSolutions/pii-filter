# Local OpenWebUI test stack

Runs this repo's `pii_filter.py` as a filter inside an OpenWebUI Pipelines
server, backed by Postgres, so the masking behaviour can be exercised from a
real chat UI.

## Prerequisite: build the base image once

This repo contains only `pii_filter.py`. It has none of the Pipelines server
code (`main.py`, `start.sh`, `utils/`) and none of the heavy runtime the filter
needs — CPU torch, the spaCy `hr`/`en` 3.7.0 models, and the vendored
GLiNER2-PII model. The sibling `keeper/pipelines-v4` repo's Dockerfile already
assembles all of that, so this image layers on top of it:

```sh
cd ../pipelines-v4
docker compose build pipelines      # ~30 min, several GB — one time only
```

That produces `pipelines-v4-pipelines:latest`, which `docker/Dockerfile` uses
as its base. Building on top of it takes seconds instead of repeating the
dependency bake here.

**Trade-off:** the test image is coupled to a locally-built tag that is not
reproducible from this repo alone. That is deliberate — it buys a fast
edit/test loop. If you ever need a standalone image, copy the builder stage
from `pipelines-v4/Dockerfile` into `docker/Dockerfile` and vendor the
Pipelines server code alongside it.

## Run

From the **repo root** (the build context is `..` so the Dockerfile can reach
`pii_filter.py`):

```sh
docker compose -f docker/docker-compose.yaml up -d --build
```

Then open <http://localhost:3000>.

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

`pii_filter.py` is baked into the image at build time, so after editing it:

```sh
docker compose -f docker/docker-compose.yaml up -d --build pipelines
```

The rebuild is a single `COPY` layer, so it is fast. Startup itself takes
~30-60s: the spaCy models and the 300M-parameter GLiNER model load into RAM.

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
- `vault_encryption_enabled: false` — no ENC1 envelope. See the compose file
  for how to turn it on.
- `priority: 0` — the deployment copy uses `-10` purely to order the PII filter
  ahead of the Langfuse filter, which this stack does not run.

The blind-index key **is** required even with encryption off (`lookup_hash` is
`NOT NULL` and part of the primary key, spec D1). The compose file supplies a
fixed throwaway key for that reason. It is a local-test value only — never use
it anywhere deployed.

## Teardown

```sh
docker compose -f docker/docker-compose.yaml down          # keep vault data
docker compose -f docker/docker-compose.yaml down -v       # wipe volumes too
```
