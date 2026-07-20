# Local test image: run THIS repo's pii_filter.py inside an OpenWebUI Pipelines
# server, so the filter can be exercised end-to-end from the OpenWebUI chat UI.
#
# Builds from public images only, so it is reproducible on any machine. The base
# ships the Pipelines server itself (main.py, start.sh, utils/), which this repo
# does not contain; everything the filter needs on top is baked in below.
#
# First build takes ~20-30 min and produces a multi-GB image, almost entirely
# torch plus the three models. It caches after that — editing pii_filter.py
# re-triggers none of it, and does not even need a rebuild, because the compose
# file live-mounts the filter over the copy baked in at the end.
FROM ghcr.io/open-webui/pipelines:main

# CPU-only torch, installed BEFORE anything else. gliner2[local] otherwise pulls
# the default PyPI torch, which drags in ~2.5GB of CUDA/nvidia wheels (nccl,
# cufft, cusolver, triton) this stack never uses. Installing it first satisfies
# gliner2's torch dependency so the requirements install below will not
# re-resolve it.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# The filter's own runtime deps. requirements.txt already pins the ABI-sensitive
# native set (numpy<2 / thinc<8.3 / blis<0.8 against spaCy 3.7) and includes the
# Croatian spaCy model wheel.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# The English model is NOT in requirements.txt, but Valves.languages defaults to
# ["hr", "en"], so on_startup loads it and would fail closed without this.
RUN pip install --no-cache-dir \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.0/en_core_web_lg-3.7.0-py3-none-any.whl

# Vendor the GLiNER2-PII model so on_startup loads it from disk rather than
# pulling hundreds of MB from HuggingFace on every container start.
# The revision is pinned: an unpinned download tracks the repo's latest commit
# and could silently swap the model artifact (and break the tokenizer patch
# below). Bump this SHA deliberately when upgrading the model.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('fastino/gliner2-privacy-filter-PII-multi', revision='e40819b10177c9b9cdea62a3ece5cfdebd921e01', local_dir='/opt/gliner2-pii')"

# The fastino model ships a tokenizer_config.json whose `extra_special_tokens`
# is a LIST, which the transformers 4.x DeBERTa-v2 fast tokenizer rejects
# ('list' object has no attribute 'keys'). Strip that key — the special tokens
# remain in tokenizer.json's added_tokens, so detection is unaffected.
RUN python -c "import json,pathlib; p=pathlib.Path('/opt/gliner2-pii/tokenizer_config.json'); d=json.loads(p.read_text(encoding='utf-8')); d.pop('extra_special_tokens', None); p.write_text(json.dumps(d), encoding='utf-8')"

ENV HF_HUB_OFFLINE=1 \
    PII_FILTER_GLINER_MODEL_NAME=/opt/gliner2-pii \
    PII_FILTER_GLINER_LOCAL_FILES_ONLY=true

# NER intra-op thread caps. NER is serialized on a single-worker executor, so a
# single inference may use several cores but must not oversubscribe them.
# OMP/OPENBLAS/MKL cap the spaCy/thinc/blis side; PII_FILTER_NER_TORCH_THREADS
# caps the torch side (read by `_ner_thread_limit`). Keep all four in sync.
ENV OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    PII_FILTER_NER_TORCH_THREADS=4

# The Pipelines server pip-installs each comma-separated entry of a pipeline's
# frontmatter `requirements:` line on every load (main.py), with no opt-out.
# Every entry on this repo's line is satisfied by the installs above, so each
# resolves as "already satisfied" and load stays fast. If you add a NEW
# frontmatter dependency, add it here too or it re-downloads on every start.
COPY pii_filter.py /app/pipelines/pii_filter_pipeline.py
