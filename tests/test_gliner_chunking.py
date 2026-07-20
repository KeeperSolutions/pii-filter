"""GLiNER2Detector input chunking (OOM + recall fix, isolation-confirmed).

The gliner2 library performs NO internal chunking: `detect()` fed the WHOLE text
into one mdeberta forward — O(n²) attention memory (5 KB → +1 GB RSS, 10 KB →
OOM) AND collapsed recall on long input (5 KB dense: 12 detections whole vs 92
chunked). `detect()` therefore slices its input into overlapping windows, runs
the model per window, remaps offsets into ORIGINAL-text coordinates and dedupes
the overlap zone. Presidio (linear, ~2.3 MB/KB) is deliberately NOT chunked.

Safety property under test (the critical one): NO entity may be lost because a
chunk boundary cuts through it — the overlap guarantees every entity shorter
than the overlap lies fully inside at least one window.

The torch model is replaced by a deterministic fake; everything else
(GLiNER2Detector chunk iteration, offset remap, dedup, containment collapse) is
the real production code.
"""

from __future__ import annotations

import pytest

from tests.conftest import FakeAnalyzer, make_gliner_pipeline, pii_mod, user_payload

PERSON = "Marcus Thornbury"
ADDRESS = "Ulica kneza Branimira 42"


class FakeModel:
    """Stands in for the gliner2 torch model. Finds configured needles in the
    text it is GIVEN — i.e. a needle cut in half by a chunk boundary is NOT
    found in that chunk (exactly how the real model misses severed entities).
    Records every text it receives so tests can assert input bounding."""

    def __init__(self, needles: dict[str, str]):
        self.needles = needles  # literal substring -> gliner label
        self.received: list[str] = []

    def extract_entities(self, text, labels, include_confidence=True, include_spans=True):
        self.received.append(text)
        entities: dict[str, list[dict]] = {}
        for needle, label in self.needles.items():
            start = text.find(needle)
            while start != -1:
                entities.setdefault(label, []).append(
                    {
                        "text": needle,
                        "start": start,
                        "end": start + len(needle),
                        "confidence": 0.9,
                    }
                )
                start = text.find(needle, start + 1)
        return {"entities": entities}


def make_detector(*, chunk_size: int, chunk_overlap: int, needles: dict[str, str]):
    det = pii_mod.GLiNER2Detector(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    det._model = FakeModel(needles)
    return det


# ---------------------------------------------------------------------------
# Constructor validation (invalid chunk params must be rejected, not silently
# produce a non-positive stride -> infinite loop / coverage gaps -> leak)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "size,overlap",
    [
        (0, 0),        # chunk_size <= 0 -> stride <= 0
        (-100, 10),    # negative chunk_size
        (1024, -1),    # negative overlap -> gaps between windows
        (1024, 1024),  # overlap == size -> stride 0 -> infinite loop
        (1024, 2048),  # overlap > size
    ],
)
def test_invalid_chunk_params_rejected(size, overlap):
    with pytest.raises(ValueError):
        pii_mod.GLiNER2Detector(chunk_size=size, chunk_overlap=overlap)


def test_zero_overlap_rejected():
    """overlap == 0 makes stride == size: windows abut with no overlap, so an
    entity on a boundary is severed and lost (a leak). Must be rejected."""
    with pytest.raises(ValueError):
        pii_mod.GLiNER2Detector(chunk_size=1024, chunk_overlap=0)


# ---------------------------------------------------------------------------
# Input bounding (the memory fix itself)
# ---------------------------------------------------------------------------


def test_model_input_bounded_by_chunk_size():
    """The O(n²) bomb: the model must NEVER receive more than chunk_size chars
    in one forward. 5 KB in, every received piece <= 1024 (production default)."""
    det = pii_mod.GLiNER2Detector()
    det._model = FakeModel({PERSON: "person"})
    text = ("Filler riječi bez entiteta idu ovdje. " * 200)[: 5 * 1024]

    det.detect(text)

    assert det._model.received, "model was never called"
    oversized = [len(t) for t in det._model.received if len(t) > 1024]
    assert not oversized, f"model received over-sized inputs: {oversized}"
    assert len(det._model.received) > 1, "5KB text must be split into multiple chunks"


def test_production_defaults_window_and_overlap():
    """Pin the isolation-derived defaults: window 1024 (measured ~+63 MB peak per
    forward, recall intact), overlap 256 (>= 2x the longest realistic mapped
    entity — full address ~120-150 chars)."""
    det = pii_mod.GLiNER2Detector()
    assert det.chunk_size == 1024
    assert det.chunk_overlap == 256


def test_chunks_advance_by_stride_and_cover_whole_text():
    """Consecutive windows start stride=(size-overlap) apart and jointly cover
    the entire text — no gap a chunk boundary could hide an entity in."""
    det = make_detector(chunk_size=100, chunk_overlap=30, needles={})
    text = "x" * 350

    det.detect(text)

    received = det._model.received
    assert all(len(t) <= 100 for t in received)
    # Reconstruct coverage from stride: starts at 0, advances by 70.
    starts = [i * 70 for i in range(len(received))]
    assert starts[0] == 0
    covered_to = max(s + len(t) for s, t in zip(starts, received, strict=True))
    assert covered_to >= len(text), "chunks do not cover the whole text"
    # strict=False on the outer zip is deliberate: it pairs each chunk with the
    # NEXT chunk's start, so `starts[1:]` is one shorter and the last chunk
    # (which has no successor) is correctly dropped.
    for (s1, t1), s2 in zip(zip(starts, received, strict=True), starts[1:], strict=False):
        assert s1 + len(t1) - s2 >= 30, "adjacent chunks must overlap by >= overlap"


# ---------------------------------------------------------------------------
# THE critical test: entity straddling a chunk boundary is still caught
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "needle,label",
    [(PERSON, "person"), (ADDRESS, "street address")],
    ids=["person", "address"],
)
def test_entity_straddling_chunk_boundary_is_detected(needle, label):
    """LEAK GUARD: place the entity ACROSS the first chunk boundary (cut at
    chunk_size). Without overlap, chunk 1 sees a severed prefix and chunk 2 a
    severed suffix — the fake (like the real model) matches neither and the
    entity would leak unmasked. The overlap guarantees one window contains it
    whole; the result must carry ORIGINAL-text offsets."""
    det = make_detector(chunk_size=100, chunk_overlap=30, needles={needle: label})
    # Entity starts 8 chars before the boundary at 100 -> straddles it.
    start_pos = 100 - 8
    text = "y" * start_pos + needle + " " + "y" * 200

    results = det.detect(text)

    assert len(results) == 1, f"straddling entity lost or duplicated: {results}"
    r = results[0]
    assert (r.start, r.end) == (start_pos, start_pos + len(needle))
    assert text[r.start : r.end] == needle, "offsets are not original-text coordinates"


def test_entity_on_every_boundary_of_a_long_text_is_detected():
    """Sweep: an entity planted across EVERY chunk boundary of a 10-chunk text —
    all of them must be found (no positional blind spot anywhere)."""
    size, overlap = 100, 30
    stride = size - overlap
    det = make_detector(chunk_size=size, chunk_overlap=overlap, needles={PERSON: "person"})

    planted = []
    chars = list("z" * (stride * 10 + size))
    for k in range(1, 10):
        boundary = k * stride + size - 8  # 8 chars before the end of chunk k
        if boundary + len(PERSON) < len(chars):
            chars[boundary : boundary + len(PERSON)] = PERSON
            planted.append(boundary)
    text = "".join(chars)

    results = det.detect(text)

    found = sorted(r.start for r in results)
    assert found == planted, f"planted at {planted}, found at {found}"
    assert all(text[r.start : r.end] == PERSON for r in results)


# ---------------------------------------------------------------------------
# Offset remap + dedup
# ---------------------------------------------------------------------------


def test_offsets_in_later_chunks_are_original_text_coordinates():
    """An entity deep in the 3rd window must come back with offsets valid in the
    ORIGINAL text (the splice/mask path slices the original with them)."""
    det = make_detector(chunk_size=100, chunk_overlap=30, needles={PERSON: "person"})
    start_pos = 180  # inside the 3rd window (starts at 140), not near a boundary
    text = "w" * start_pos + PERSON + "w" * 100

    results = det.detect(text)

    assert len(results) == 1
    assert text[results[0].start : results[0].end] == PERSON
    assert results[0].start == start_pos


def test_entity_in_overlap_zone_deduplicated_to_single_result():
    """An entity fully inside the overlap zone is seen by BOTH adjacent windows;
    the merged output must contain it exactly once."""
    det = make_detector(chunk_size=100, chunk_overlap=30, needles={PERSON: "person"})
    # Overlap zone of windows 1 and 2 is [70, 100); 16-char entity at 75 fits fully.
    start_pos = 75
    text = "q" * start_pos + PERSON + "q" * 150

    results = det.detect(text)

    assert len(results) == 1, f"overlap-zone entity duplicated: {results}"
    assert (results[0].start, results[0].end) == (start_pos, start_pos + len(PERSON))
    # Sanity: both windows really did see it.
    seen_in = [t for t in det._model.received if PERSON in t]
    assert len(seen_in) == 2


# ---------------------------------------------------------------------------
# No-regression edges
# ---------------------------------------------------------------------------


def test_short_text_single_call_unchanged():
    """Text shorter than one window: exactly one model call with the verbatim
    text, offsets exactly as before chunking existed."""
    det = make_detector(chunk_size=1024, chunk_overlap=256, needles={PERSON: "person"})
    text = f"Pozdrav od {PERSON} iz Zagreba."

    results = det.detect(text)

    assert det._model.received == [text]
    assert len(results) == 1
    assert text[results[0].start : results[0].end] == PERSON


def test_empty_text_and_unloaded_model_return_empty():
    det = make_detector(chunk_size=100, chunk_overlap=30, needles={PERSON: "person"})
    assert det.detect("") == []
    det2 = pii_mod.GLiNER2Detector(chunk_size=100, chunk_overlap=30)
    assert det2.detect(PERSON) == []  # _model is None


def test_containment_collapse_still_applies_across_chunks():
    """person + first-name sub-span both map to PERSON; the sub-span must still
    collapse into the full span after the per-chunk merge."""
    det = make_detector(
        chunk_size=100, chunk_overlap=30, needles={PERSON: "person", "Marcus": "first name"}
    )
    text = "p" * 130 + PERSON + "p" * 60  # inside window 2

    results = det.detect(text)

    assert len(results) == 1, f"sub-span not collapsed: {results}"
    assert text[results[0].start : results[0].end] == PERSON


# ---------------------------------------------------------------------------
# Integration: full inlet flow with the real detector (fake torch model only)
# ---------------------------------------------------------------------------


async def test_inlet_masks_entity_beyond_first_chunk_with_correct_splice():
    """End-to-end: a multi-KB message whose new name sits past the first 1024-char
    window. The real inlet flow (re-mask -> NER -> Step-0 -> splice) must mask it
    with vault-consistent placeholders and untouched surrounding text — proving
    remapped offsets are what the splice needs. Chunking changes NOTHING in that
    ordering; it only slices the GLiNER input internally."""
    pipe = make_gliner_pipeline(masking_enabled=True)
    det = pii_mod.GLiNER2Detector()  # production window/overlap
    det._model = FakeModel({PERSON: "person"})
    pipe._gliner = det
    pipe.analyzer_hr = FakeAnalyzer({})

    filler = "Ovo je obična rečenica bez ikakvih osobnih podataka. "
    prefix = (filler * 40)[:1500]  # pushes the name into the second window
    text = prefix + f"Kontakt: {PERSON}." + filler * 5

    out = await pipe.inlet(
        {"chat_id": "chat-chunk-int", "messages": [{"role": "user", "content": text}]},
        user=user_payload(True),
    )

    content = out["messages"][0]["content"]
    assert PERSON not in content, "name past the first window leaked"
    assert "[PERSON_1]" in content
    # Splice used correct offsets: the text around the mask is untouched.
    assert "Kontakt: [PERSON_1]." in content
    assert content.startswith(prefix)
