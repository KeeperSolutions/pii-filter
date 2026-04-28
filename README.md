# pii-filter

PII detection and masking pipeline for Keeper AI Gateway (OpenWebUI Pipelines integration).

## Status

**Phase 1 — MVP in progress.** See `Dokument-4-v4.0-Implementation-Roadmap.md` (project files) for full task breakdown.

Current task: **Task 1 — Skeleton + repo setup** ✅

## Overview

Pipeline file (`pii_filter.py`) that runs inside an OpenWebUI Pipelines container (`ghcr.io/open-webui/pipelines:main`). It intercepts chat messages going to LLMs (OpenAI, Anthropic, etc.), masks PII (names, emails, IBANs, OIBs, ...), and restores original values in responses.

**This is NOT a standalone microservice.** It's a single Python file uploaded into a shared Pipelines container.

## Architecture

```
[OpenWebUI] --HTTP--> [Pipelines Container] --loads--> pii_filter.py (this repo)
                              |
                              | (Task 5)
                              v
                          [Redis Vault]
```



## Tech stack

- **Python 3.11** (Pipelines container constraint, only officially supported version)
- **Microsoft Presidio** (Task 3 — PII detection engine)
- **spaCy `hr_core_news_lg`** (Task 3 — Croatian NER)
- **Redis** (Task 5 — thread-scoped mapping vault)

## Development

### Prerequisites

- Python 3.11 (NOT 3.12 — Pipelines container is locked to 3.11)
- pip

### Setup

```bash
# Create venv
py -3.11 -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
# Or (Linux/Mac/Git Bash)
source .venv/bin/activate

# Install dev dependencies
pip install -r requirements-dev.txt
```

### Tests

```bash
pytest
```

### Lint & type check

```bash
ruff check .
ruff format --check .
mypy pii_filter.py tests
```

### Format code

```bash
ruff format .
```

## Deployment

Pipeline file (`pii_filter.py`) is deployed by uploading it to the Keeper Pipelines container via:

1. **First-time deploy (with dependencies):** Senka/Antonio adds GitHub raw URL to `PIPELINES_URLS` env var on the Pipelines Cloud Run service, then restarts the container. The container parses the `requirements:` frontmatter and installs dependencies.

2. **Subsequent code-only updates:** Upload the updated `pii_filter.py` via OpenWebUI Admin → Pipelines → Upload Pipeline. Dependencies must already be installed (from step 1) — UI upload does NOT install new dependencies.

Pipelines container URL (dev): `https://pipelines-135620221720.europe-west1.run.app`

For full deployment guide, see `docs/deployment.md` (TBD in Task 9).


## License

MIT
