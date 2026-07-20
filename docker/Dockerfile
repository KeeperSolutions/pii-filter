# Local test image: run THIS repo's pii_filter.py inside an OpenWebUI Pipelines
# server, so the filter can be exercised end-to-end from the OpenWebUI chat UI.
#
# WHY THIS BASE:
# This repo contains only `pii_filter.py` — it has none of the Pipelines server
# code (main.py, start.sh, utils/) and none of the heavy runtime the filter
# needs (CPU torch, spaCy hr+en 3.7.0 models, the vendored GLiNER2-PII model).
# All of that is already assembled by the sibling `keeper/pipelines-v4` repo's
# Dockerfile, so this image layers on top of that build instead of duplicating
# a ~30-minute, multi-GB dependency bake here.
#
# PREREQUISITE — build the base once, from the pipelines-v4 checkout:
#     cd ../pipelines-v4 && docker compose build pipelines
# (or `docker build -t pipelines-v4-pipelines:latest .`)
#
# TRADE-OFF: this couples the test image to a locally-built, non-reproducible
# tag. That is deliberate for a fast local edit/test loop. If you need an image
# that builds standalone from this repo alone, the dependency bake from
# pipelines-v4's Dockerfile has to be copied in here — see docker/README.md.
FROM pipelines-v4-pipelines:latest

# The base image ships the deployment copy of the filter plus the Langfuse
# filter. Drop both: this image tests THIS repo's filter in isolation, and the
# Langfuse pipeline would fail to load without LANGFUSE_* credentials.
#
# Dropping Langfuse is also why the `priority: 0` default in this repo is fine
# here — the deployment copy uses -10 solely to order the PII filter ahead of
# the Langfuse filter, which is not present in this stack.
RUN rm -f /app/pipelines/pii_filter_pipeline.py \
          /app/pipelines/langfuse_v4_filter_pipeline.py

# The Pipelines server pip-installs each comma-separated entry of a pipeline's
# frontmatter `requirements:` line on every load (main.py:128), with no opt-out.
# Every entry on this repo's line is already satisfied by the base image, so
# each resolves as "already satisfied" and load stays fast. Keep it that way:
# if you add a NEW dependency to the frontmatter line, bake it into the base
# image too, or every container start will re-download it.
COPY pii_filter.py /app/pipelines/pii_filter_pipeline.py
