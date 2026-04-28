# Reference materials

## `presidio_filter_pipeline.py` (official Open WebUI example)

The skeleton structure of `pii_filter.py` is loosely based on the official Open WebUI Presidio filter example:

**Source:** https://github.com/open-webui/pipelines/blob/main/examples/filters/presidio_filter_pipeline.py

**Why we don't copy it directly:**

- It's Apache 2.0 licensed (different from our MIT)
- It's one-way redaction (no `outlet()` restoration) — we need bidirectional mask + restore
- It uses generic `[REDACTED]` placeholders — we need numbered, type-specific (`[PERSON_1]`, `[HR_OIB_1]`)
- It's stateless — we need thread-scoped consistency via Redis (Task 5)
- It uses default Presidio recognizers — we add 12 custom HR/IE/RO/UK/US recognizers (Task 3)

**What we borrow:**

- Frontmatter docstring format
- Pipeline class structure (`type = "filter"`, `Valves`, `__init__`, `on_startup/shutdown`, `inlet/outlet`)
- Pattern for env-var-overridable Valve defaults

**Other references worth reading:**

- **Langfuse filter:** https://github.com/open-webui/pipelines/blob/main/examples/filters/langfuse_filter_pipeline.py — closest to our use case (stateful, bidirectional, used by Keeper as `MaskingExclusion-v3`)
- **Libretranslate filter:** https://github.com/open-webui/pipelines/blob/main/examples/filters/libretranslate_filter_pipeline.py — bidirectional transformation pattern (translate IN → translate OUT)
- **Open WebUI Pipelines docs:** https://docs.openwebui.com/pipelines/
