"""
title: PII Filter
author: Keeper Solutions AI Lab
author_url: https://github.com/keeper-solutions/pii-filter
date: 2026-04-30
version: 0.3.0
license: MIT
description: PII detection and masking filter for Keeper AI Gateway. Task 4 — inlet masking + placeholder generation. Detected spans in the last user message are replaced with deterministic placeholders ([ENTITY_N]); forward + reverse maps are stashed in body.metadata for outlet restoration (Task 6).
requirements: presidio-analyzer>=2.2.0, presidio-anonymizer>=2.2.0, spacy>=3.7.0, https://github.com/explosion/spacy-models/releases/download/hr_core_news_lg-3.7.0/hr_core_news_lg-3.7.0-py3-none-any.whl
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterator
from typing import Any, ClassVar

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import CreditCardRecognizer
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Module-level language constant. Recognizers default to this language; tests
# can override by passing `supported_language` explicitly.
LANG = "hr"

# Cached spaCy NLP engine. The model loads ~240 MB of contiguous word
# vectors; on memory-constrained hosts a second load can fail with a
# fragmented-heap allocation error. We cache the engine per-process so
# repeated `on_startup` calls (test reruns, multiple Pipeline instances)
# reuse the same load. AnalyzerEngine itself is still rebuilt per startup.
_nlp_engine_cache: Any = None


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _iban_mod_97_check(iban: str) -> bool:
    """ISO 13616 / MOD 97-10 IBAN checksum verification.

    Strips spaces, moves the first 4 chars to the end, converts letters to
    digits (A=10, B=11, ..., Z=35), and checks `int(numeric) % 97 == 1`.
    """
    pt = iban.replace(" ", "")
    rearranged = pt[4:] + pt[:4]
    numeric = ""
    for ch in rearranged:
        numeric += str(ord(ch) - 55) if ch.isalpha() else ch
    try:
        return int(numeric) % 97 == 1
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# HR recognizers
# ---------------------------------------------------------------------------


class OIBRecognizer(PatternRecognizer):
    """HR OIB — 11 digits with ISO 7064 mod 11,10 checksum."""

    PATTERNS: ClassVar[list[Pattern]] = [Pattern("OIB", r"\b\d{11}\b", 0.4)]

    def __init__(self, supported_language: str = LANG) -> None:
        super().__init__(
            supported_entity="HR_OIB",
            patterns=self.PATTERNS,
            supported_language=supported_language,
        )

    def validate_result(self, pattern_text: str) -> bool:
        if len(pattern_text) != 11 or not pattern_text.isdigit():
            return False
        a = 10
        for d in pattern_text[:10]:
            a = (a + int(d)) % 10
            if a == 0:
                a = 10
            a = (a * 2) % 11
        control = (11 - a) % 10
        return control == int(pattern_text[10])


class JMBGRecognizer(PatternRecognizer):
    """HR JMBG — legacy 13-digit ID with weighted mod 11 checksum."""

    PATTERNS: ClassVar[list[Pattern]] = [Pattern("JMBG", r"\b\d{13}\b", 0.4)]

    def __init__(self, supported_language: str = LANG) -> None:
        super().__init__(
            supported_entity="HR_JMBG",
            patterns=self.PATTERNS,
            supported_language=supported_language,
        )

    def validate_result(self, pattern_text: str) -> bool:
        if len(pattern_text) != 13 or not pattern_text.isdigit():
            return False
        weights = [7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2]
        s = sum(int(pattern_text[i]) * weights[i] for i in range(12))
        check = 11 - (s % 11)
        if check > 9:
            check = 0
        return check == int(pattern_text[12])


class HRIBANRecognizer(PatternRecognizer):
    """HR IBAN — 21-char IBAN starting HR, MOD 97-10 checksum."""

    PATTERNS: ClassVar[list[Pattern]] = [Pattern("HR IBAN", r"\bHR\d{19}\b", 0.5)]

    def __init__(self, supported_language: str = LANG) -> None:
        super().__init__(
            supported_entity="HR_IBAN",
            patterns=self.PATTERNS,
            supported_language=supported_language,
        )

    def validate_result(self, pattern_text: str) -> bool:
        return _iban_mod_97_check(pattern_text)


# ---------------------------------------------------------------------------
# IE recognizers
# ---------------------------------------------------------------------------


class IEPPSNRecognizer(PatternRecognizer):
    """IE PPSN — 7 digits + check letter [A-W] + optional [AB], mod 23 checksum."""

    PATTERNS: ClassVar[list[Pattern]] = [Pattern("IE PPSN", r"\b\d{7}[A-W][AB]?\b", 0.4)]

    def __init__(self, supported_language: str = LANG) -> None:
        super().__init__(
            supported_entity="IE_PPSN",
            patterns=self.PATTERNS,
            supported_language=supported_language,
        )

    def validate_result(self, pattern_text: str) -> bool:
        pt = pattern_text.upper()
        if len(pt) not in (8, 9):
            return False
        if not pt[:7].isdigit() or not pt[7].isalpha():
            return False
        digits = [int(c) for c in pt[:7]]
        weights = [8, 7, 6, 5, 4, 3, 2]
        total = sum(d * w for d, w in zip(digits, weights, strict=True))
        if len(pt) == 9:
            if not pt[8].isalpha():
                return False
            total += (ord(pt[8]) - ord("A") + 1) * 9
        mod = total % 23
        expected = "W" if mod == 0 else chr(ord("A") + mod - 1)
        return pt[7] == expected


# ---------------------------------------------------------------------------
# RO recognizers
# ---------------------------------------------------------------------------


class ROCNPRecognizer(PatternRecognizer):
    """RO CNP — 13 digits with structural validation + mod 11 weighted checksum."""

    PATTERNS: ClassVar[list[Pattern]] = [Pattern("RO CNP", r"\b\d{13}\b", 0.4)]

    def __init__(self, supported_language: str = LANG) -> None:
        super().__init__(
            supported_entity="RO_CNP",
            patterns=self.PATTERNS,
            supported_language=supported_language,
        )

    def validate_result(self, pattern_text: str) -> bool:
        if len(pattern_text) != 13 or not pattern_text.isdigit():
            return False
        first = int(pattern_text[0])
        if not (1 <= first <= 8):
            return False
        month = int(pattern_text[3:5])
        if not (1 <= month <= 12):
            return False
        day = int(pattern_text[5:7])
        if not (1 <= day <= 31):
            return False
        county = int(pattern_text[7:9])
        if not (1 <= county <= 52):
            return False
        weights = [2, 7, 9, 1, 4, 6, 3, 5, 8, 2, 7, 9]
        total = sum(int(pattern_text[i]) * weights[i] for i in range(12))
        check = total % 11
        if check == 10:
            check = 1
        return check == int(pattern_text[12])


# ---------------------------------------------------------------------------
# UK recognizers
# ---------------------------------------------------------------------------


class UKNINORecognizer(PatternRecognizer):
    """UK NINO — 2 letters + 6 digits + suffix [A-D] (spaced or compact), HMRC prefix exclusions."""

    PATTERNS: ClassVar[list[Pattern]] = [
        Pattern(
            "UK NINO spaced",
            r"\b[A-Z]{2}\s\d{2}\s\d{2}\s\d{2}\s[A-D]\b",
            0.7,
        ),
        Pattern("UK NINO compact", r"\b[A-Z]{2}\d{6}[A-D]\b", 0.7),
    ]
    INVALID_PREFIXES: ClassVar[frozenset[str]] = frozenset(
        {"BG", "GB", "KN", "NK", "NT", "TN", "ZZ"}
    )
    INVALID_FIRST: ClassVar[frozenset[str]] = frozenset({"D", "F", "I", "Q", "U", "V"})
    INVALID_SECOND: ClassVar[frozenset[str]] = frozenset({"D", "F", "I", "O", "Q", "U", "V"})

    def __init__(self, supported_language: str = LANG) -> None:
        super().__init__(
            supported_entity="UK_NINO",
            patterns=self.PATTERNS,
            supported_language=supported_language,
        )

    def validate_result(self, pattern_text: str) -> bool:
        clean = pattern_text.replace(" ", "").upper()
        if len(clean) != 9:
            return False
        prefix = clean[:2]
        if prefix in self.INVALID_PREFIXES:
            return False
        if prefix[0] in self.INVALID_FIRST:
            return False
        return prefix[1] not in self.INVALID_SECOND


class UKUTRRecognizer(PatternRecognizer):
    """UK UTR — 10 digits, requires UTR/utr keyword immediately before via lookbehind.

    Per spec AC #14: the regex uses a fixed-width lookbehind so the matched
    span contains ONLY the 10 digits, not the keyword. This keeps downstream
    masking spans clean. The benchmark version captured the keyword in the
    span, which would corrupt mask placement.
    """

    PATTERNS: ClassVar[list[Pattern]] = [Pattern("UK UTR", r"(?<=\b(?:UTR|utr)\s)\d{10}\b", 0.5)]

    def __init__(self, supported_language: str = LANG) -> None:
        super().__init__(
            supported_entity="UK_UTR",
            patterns=self.PATTERNS,
            supported_language=supported_language,
        )

    def validate_result(self, pattern_text: str) -> bool:
        return len(pattern_text) == 10 and pattern_text.isdigit()


# ---------------------------------------------------------------------------
# US recognizers
# ---------------------------------------------------------------------------


class USSSNRecognizer(PatternRecognizer):
    """US SSN — XXX-XX-XXXX, rejects reserved area/group/serial blocks."""

    PATTERNS: ClassVar[list[Pattern]] = [Pattern("US SSN", r"\b\d{3}-\d{2}-\d{4}\b", 0.7)]

    def __init__(self, supported_language: str = LANG) -> None:
        super().__init__(
            supported_entity="US_SSN",
            patterns=self.PATTERNS,
            supported_language=supported_language,
        )

    def validate_result(self, pattern_text: str) -> bool:
        parts = pattern_text.split("-")
        if len(parts) != 3:
            return False
        area, group, serial = parts
        if area in ("000", "666"):
            return False
        if int(area) >= 900:
            return False
        if group == "00":
            return False
        return serial != "0000"


class USEINRecognizer(PatternRecognizer):
    """US EIN — XX-XXXXXXX (Employer Identification Number), format-only validation."""

    PATTERNS: ClassVar[list[Pattern]] = [Pattern("US EIN", r"\b\d{2}-\d{7}\b", 0.6)]
    CONTEXT: ClassVar[list[str]] = ["EIN", "Employer Identification", "Federal Tax ID", "Tax ID"]

    def __init__(self, supported_language: str = LANG) -> None:
        super().__init__(
            supported_entity="US_EIN",
            patterns=self.PATTERNS,
            supported_language=supported_language,
            context=self.CONTEXT,
        )

    def validate_result(self, pattern_text: str) -> bool:
        # Format-only validation. Benchmark returned True unconditionally; we
        # re-check the format here so unit tests can directly exercise
        # invalid-format rejection without going through the AnalyzerEngine.
        return re.fullmatch(r"\d{2}-\d{7}", pattern_text) is not None


# ---------------------------------------------------------------------------
# IBAN factory (IE / RO / GB)
# ---------------------------------------------------------------------------


def make_iban_recognizer(
    country_code: str,
    bban_length: int,
    entity_name: str,
    supported_language: str = LANG,
) -> PatternRecognizer:
    """Build a generic IBAN PatternRecognizer for a country.

    The factory delegates checksum validation to the shared
    `_iban_mod_97_check` helper, eliminating the duplication in the benchmark.
    """
    iban_pattern = Pattern(
        f"{country_code} IBAN",
        rf"\b{country_code}\d{{2}}[A-Z0-9]{{{bban_length}}}\b",
        0.5,
    )

    class _IBANRecog(PatternRecognizer):
        def __init__(self) -> None:
            super().__init__(
                supported_entity=entity_name,
                patterns=[iban_pattern],
                supported_language=supported_language,
            )

        def validate_result(self, pattern_text: str) -> bool:
            return _iban_mod_97_check(pattern_text)

    return _IBANRecog()


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


# Entity types produced by our 12 custom recognizers. Used as a tiebreaker
# in overlap resolution: when two detections have equal score, the one whose
# entity_type is in this set wins over a built-in (PERSON, EMAIL_ADDRESS,
# PHONE_NUMBER, LOCATION, DATE_TIME, CREDIT_CARD). Rationale: custom
# recognizers encode strict structural / checksum knowledge, so on a tie
# they're more trustworthy than a generic built-in match. CreditCard is
# intentionally OUT — its Luhn check already gives it score 1.0 in normal
# cases, so score DESC handles it; ties involving CC are very rare.
CUSTOM_ENTITY_TYPES: frozenset[str] = frozenset(
    {
        "HR_OIB",
        "HR_JMBG",
        "HR_IBAN",
        "IE_PPSN",
        "IE_IBAN",
        "RO_CNP",
        "RO_IBAN",
        "UK_NINO",
        "UK_UTR",
        "GB_IBAN",
        "US_SSN",
        "US_EIN",
    }
)


def mask_text(
    text: str,
    detections: list[RecognizerResult],
    presidio_to_standard: dict[str, str],
    counter_state: dict[str, int],
    forward_map: dict[str, str],
    reverse_map: dict[str, str],
) -> tuple[str, list[dict[str, Any]]]:
    """Replace detected spans in `text` with deterministic placeholders.

    Args:
        text: original input string.
        detections: raw Presidio `RecognizerResult` list (may contain overlaps
            and non-whitelisted entity types).
        presidio_to_standard: whitelist mapping from raw Presidio entity_type
            to the Keeper-standardized type used in the placeholder. Detections
            whose `entity_type` is not a key are dropped silently.
        counter_state: per-entity-type next-N counter; mutated in place across
            calls so multi-part messages share counters.
        forward_map: `original_value -> placeholder`; mutated in place.
            In-request dedupe: same original value reuses an existing
            placeholder instead of allocating a new one.
        reverse_map: `placeholder -> original_value`; mutated in place. Outlet
            (Task 6) reads this to restore originals.

    Returns:
        `(masked_text, surviving_detections)` where each surviving detection
        is `{entity_type, start, end, score, raw_entity_type, original,
        placeholder}`. `start`/`end` are offsets into the *original* text.

    Overlap resolution:
        Sort by `(score DESC, custom_recognizer_first, start ASC)`. Iterate
        and accept a detection only if its `[start, end)` span does not
        intersect any already-accepted span. The algorithm is O(n^2) in the
        number of detections, which is fine for the typical n < 50 case.
    """
    if not text or not detections:
        return text, []

    # Step 1: filter to whitelisted types.
    candidates: list[RecognizerResult] = [
        d for d in detections if d.entity_type in presidio_to_standard
    ]
    if not candidates:
        return text, []

    # Step 2: sort by score DESC, custom-first, start ASC.
    candidates.sort(
        key=lambda d: (
            -d.score,
            0 if d.entity_type in CUSTOM_ENTITY_TYPES else 1,
            d.start,
        )
    )

    # Step 3: greedily accept non-overlapping detections in priority order.
    # Skip zero-length / inverted spans defensively — a buggy custom recognizer
    # producing start >= end would otherwise allocate a placeholder for an
    # empty original string and inject it into the masked text.
    accepted: list[RecognizerResult] = []
    for det in candidates:
        if det.start >= det.end:
            continue
        if any(not (det.end <= a.start or a.end <= det.start) for a in accepted):
            continue
        accepted.append(det)

    # Step 4: re-sort survivors by start ASC for the masking pass.
    accepted.sort(key=lambda d: d.start)

    # Step 5: build masked text in a single left-to-right pass + enrich detections.
    pieces: list[str] = []
    enriched: list[dict[str, Any]] = []
    last_end = 0
    for det in accepted:
        original = text[det.start : det.end]
        standard_type = presidio_to_standard[det.entity_type]
        placeholder = forward_map.get(original)
        if placeholder is None:
            n = counter_state.get(standard_type, 0) + 1
            counter_state[standard_type] = n
            placeholder = f"[{standard_type}_{n}]"
            forward_map[original] = placeholder
            reverse_map[placeholder] = original
        pieces.append(text[last_end : det.start])
        pieces.append(placeholder)
        last_end = det.end
        enriched.append(
            {
                "entity_type": standard_type,
                "start": det.start,
                "end": det.end,
                "score": det.score,
                "raw_entity_type": det.entity_type,
                "original": original,
                "placeholder": placeholder,
            }
        )
    pieces.append(text[last_end:])

    return "".join(pieces), enriched


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


# Built-in Presidio recognizers that conflict with our custom HR set or
# generate false positives on Croatian text. Removed at startup.
_DISABLED_BUILTIN_RECOGNIZERS: tuple[str, ...] = (
    "IbanRecognizer",
    "UrlRecognizer",
    "OrganizationRecognizer",
    "MedicalLicenseRecognizer",
)


class Pipeline:
    """PII Filter pipeline — Keeper AI Gateway.

    Task 3: Presidio analyzer wired up with 12 custom recognizers + HR spaCy
    NLP. Inlet runs detection and attaches results to body metadata. Outlet
    is still pass-through (masking/restoration come in Tasks 4 + 6).
    """

    class Valves(BaseModel):
        """Admin-configurable settings (visible in OpenWebUI Admin → Pipelines)."""

        pipelines: list[str] = ["*"]
        priority: int = 0
        enabled: bool = True
        languages: list[str] = ["hr"]
        # Behavior when the analyzer fails mid-request.
        #   "block" (default) — fail-closed: raise so the request never
        #     reaches the LLM unfiltered. GDPR-safe; recommended for prod.
        #   "passthrough" — fail-open: log and let the request through
        #     without PII filtering. Use only if availability outweighs
        #     leak risk. Any unrecognized value is treated as "block".
        degradation_mode: str = "block"

    class UserValves(BaseModel):
        """Per-user toggles. Schema only — Task 8 wires the masking toggle.

        OpenWebUI Pipelines container has open issue #19179 around UserValves
        propagation; we publish the schema now so Task 8 can flip the switch
        without a separate schema migration.
        """

        pii_masking_enabled: bool = True

    # Whitelist mapping: only entities in this dict are forwarded downstream;
    # everything else is dropped. Keys are raw Presidio entity types, values
    # are the Keeper-standardized type names used in metadata + masking.
    PRESIDIO_TO_STANDARD: ClassVar[dict[str, str]] = {
        "PERSON": "PERSON",
        "EMAIL_ADDRESS": "EMAIL",
        "PHONE_NUMBER": "PHONE",
        "LOCATION": "ADDRESS",
        "DATE_TIME": "DATE",
        "CREDIT_CARD": "CREDIT_CARD",
        "HR_OIB": "HR_OIB",
        "HR_JMBG": "HR_JMBG",
        "HR_IBAN": "HR_IBAN",
        "IE_PPSN": "IE_PPSN",
        "IE_IBAN": "IE_IBAN",
        "RO_CNP": "RO_CNP",
        "RO_IBAN": "RO_IBAN",
        "UK_NINO": "UK_NINO",
        "UK_UTR": "UK_UTR",
        "GB_IBAN": "GB_IBAN",
        "US_SSN": "US_SSN",
        "US_EIN": "US_EIN",
    }

    def __init__(self) -> None:
        """Initialize the pipeline.

        Heavy work (AnalyzerEngine, spaCy load, recognizer registration) is
        deferred to `on_startup` per Pipelines lifecycle: `__init__` runs at
        import, `on_startup` runs when the pipeline is enabled.
        """
        self.type = "filter"
        self.name = "PII Filter"
        self.valves = self.Valves()
        self.user_valves = self.UserValves()
        self.analyzer: AnalyzerEngine | None = None

        logger.info("PII Filter pipeline initialized (analyzer not loaded yet)")

    async def on_startup(self) -> None:
        """Load spaCy HR model, build AnalyzerEngine, register recognizers, warm up."""
        logger.info("PII Filter on_startup: loading spaCy HR model + Presidio analyzer")

        # Hard requirement: hr_core_news_lg. No EN fallback (spec AC #4).
        # Use the per-process cache so a second startup in the same interpreter
        # (test reruns, fixture teardown + new Pipeline) doesn't re-allocate
        # the ~240 MB word vector block.
        global _nlp_engine_cache
        try:
            if _nlp_engine_cache is None:
                # `labels_to_ignore` drops MISC and the catch-all "O" tag at
                # the spaCy NER stage so they never reach Presidio's mapper.
                # This silences the noisy "MISC is not registered with the
                # Presidio entity mapping" warnings on Croatian text and
                # keeps detections to the entity types we actually mask.
                _nlp_engine_cache = NlpEngineProvider(
                    nlp_configuration={
                        "nlp_engine_name": "spacy",
                        "models": [{"lang_code": "hr", "model_name": "hr_core_news_lg"}],
                        "ner_model_configuration": {
                            "labels_to_ignore": ["MISC", "O"],
                        },
                    }
                ).create_engine()
            nlp_engine = _nlp_engine_cache
        except Exception as exc:
            logger.error(
                "Failed to load spaCy 'hr_core_news_lg' NLP engine: %s. "
                "If the model is not installed, install it via the wheel URL in "
                "requirements.txt or run: python -m spacy download hr_core_news_lg. "
                "Other causes can include insufficient memory (the model loads "
                "~240 MB of word vectors as a contiguous block), a broken install, "
                "or a spaCy major-version mismatch. "
                "This pipeline requires the Croatian language model and will not "
                "start without it.",
                exc,
            )
            raise RuntimeError("Required spaCy model 'hr_core_news_lg' is unavailable") from exc

        analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=[LANG],
        )

        # Disable built-in recognizers that conflict with our HR custom set
        # or produce noisy FPs on Croatian text. Tolerate missing names —
        # Presidio defaults vary across versions.
        for rec_name in _DISABLED_BUILTIN_RECOGNIZERS:
            try:
                analyzer.registry.remove_recognizer(rec_name)
            except Exception as exc:
                logger.warning("Could not remove built-in recognizer %r: %s", rec_name, exc)

        # Register the 12 custom recognizers + built-in CreditCardRecognizer
        # under HR. CreditCardRecognizer is built-in but defaults to "en" only.
        analyzer.registry.add_recognizer(OIBRecognizer())
        analyzer.registry.add_recognizer(JMBGRecognizer())
        analyzer.registry.add_recognizer(HRIBANRecognizer())
        analyzer.registry.add_recognizer(IEPPSNRecognizer())
        analyzer.registry.add_recognizer(ROCNPRecognizer())
        analyzer.registry.add_recognizer(UKNINORecognizer())
        analyzer.registry.add_recognizer(UKUTRRecognizer())
        analyzer.registry.add_recognizer(USSSNRecognizer())
        analyzer.registry.add_recognizer(USEINRecognizer())
        analyzer.registry.add_recognizer(make_iban_recognizer("IE", 18, "IE_IBAN"))
        analyzer.registry.add_recognizer(make_iban_recognizer("RO", 20, "RO_IBAN"))
        analyzer.registry.add_recognizer(make_iban_recognizer("GB", 18, "GB_IBAN"))
        analyzer.registry.add_recognizer(CreditCardRecognizer(supported_language=LANG))

        # Warmup: force lazy spaCy + recognizer init now, not on first request.
        analyzer.analyze(text="warmup", language=LANG, entities=None)

        self.analyzer = analyzer
        logger.info("PII Filter on_startup complete: 12 custom + CreditCard recognizers registered")

    async def on_shutdown(self) -> None:
        """Called when Pipelines container stops."""
        logger.info("PII Filter on_shutdown")
        self.analyzer = None

    def _iter_maskable_parts(
        self, message: dict[str, Any]
    ) -> Iterator[tuple[str, Callable[[str], None]]]:
        """Yield `(text, write_back)` pairs for each maskable text segment.

        Handles both content shapes accepted by the OpenAI chat-completion API:
          * `content` is a `str` — yields one pair; write_back replaces the
            whole `message["content"]` value.
          * `content` is a `list[dict]` (multi-modal) — yields one pair per
            `{"type": "text", ...}` part; write_back updates that part's
            `text` field. Non-text parts (image_url, file, etc.) are skipped.

        Empty / whitespace-only / non-string text segments are skipped so
        the analyzer is never invoked on uninteresting input.
        """
        content = message.get("content")
        if isinstance(content, str):
            if content.strip():

                def _write_back_str(new_text: str) -> None:
                    message["content"] = new_text

                yield content, _write_back_str
            return
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "text":
                    continue
                text_val = item.get("text", "")
                if not isinstance(text_val, str) or not text_val.strip():
                    continue

                # Bind `item` via default arg to avoid the Python late-binding
                # closure pitfall (every closure would otherwise reference the
                # final loop value). ruff's B023 is silenced by the binding.
                def _write_back_part(new_text: str, _item: dict[str, Any] = item) -> None:
                    _item["text"] = new_text

                yield text_val, _write_back_part

    async def inlet(
        self, body: dict[str, Any], user: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Detect PII in the last user message, mask it in place, and stash
        the placeholder maps in `body["metadata"]` for the outlet (Task 6).

        Mutates the matched message's `content` field. All other body keys
        are left untouched. On analyzer failure, behavior follows
        `valves.degradation_mode` (block → raise; passthrough → return).
        """
        if not self.valves.enabled:
            return body

        if self.analyzer is None:
            logger.warning("inlet called before on_startup completed; returning body unchanged")
            return body

        # The last entry in `messages` may be assistant/tool/system in
        # multi-turn or tool-call flows, so iterate backwards until we
        # find the most recent user-authored message.
        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            logger.warning("inlet: no messages in body, skipping analysis")
            return body
        target_msg: dict[str, Any] | None = None
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                target_msg = msg
                break
        if target_msg is None:
            logger.debug("inlet: no user message found, skipping analysis")
            return body

        parts = list(self._iter_maskable_parts(target_msg))
        if not parts:
            logger.debug("inlet: target message has no maskable text parts")
            return body

        # Per-request mapping state. Counters and maps are shared across all
        # text parts of the message so dedupe works across multi-modal parts.
        counter_state: dict[str, int] = {}
        forward_map: dict[str, str] = {}
        reverse_map: dict[str, str] = {}
        all_enriched: list[dict[str, Any]] = []

        try:
            for text, write_back in parts:
                results = self.analyzer.analyze(text=text, language=LANG)
                masked, enriched = mask_text(
                    text,
                    results,
                    self.PRESIDIO_TO_STANDARD,
                    counter_state,
                    forward_map,
                    reverse_map,
                )
                if enriched:
                    write_back(masked)
                all_enriched.extend(enriched)
        except Exception as exc:
            logger.exception("inlet: analyzer/mask pipeline failed")
            if self.valves.degradation_mode == "passthrough":
                # Partial-write semantics: if the failure happened on the Nth
                # text part of a multi-part message, parts [0..N-1] were
                # already mutated to their masked form before the exception.
                # We deliberately do NOT roll those back — partial masking is
                # closer to safe than zero masking, and rolling back would
                # require capturing a snapshot of every part's pre-mask text
                # before the loop. Acceptable trade for an opt-in availability
                # mode; revisit if Task 7 (streaming) needs strict atomicity.
                logger.warning(
                    "inlet: degradation_mode=passthrough, request will reach LLM "
                    "without PII filtering"
                )
                return body
            # Default: fail-closed. Any value other than "passthrough" blocks.
            raise RuntimeError(
                "PII filter encountered an internal error and could not analyze "
                "the request. Request blocked to prevent unfiltered PII leak. "
                "Set valves.degradation_mode='passthrough' to allow requests "
                "through on filter errors (NOT recommended in production)."
            ) from exc

        metadata = body.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            body["metadata"] = metadata
        metadata["pii_detections"] = all_enriched
        metadata["pii_placeholder_map"] = forward_map
        metadata["pii_reverse_map"] = reverse_map

        chat_id = metadata.get("chat_id") or body.get("chat_id")
        logger.info(
            "pii_filter inlet processed: chat_id=%s detections=%d masked=%d",
            chat_id,
            len(all_enriched),
            len(forward_map),
        )

        return body

    async def outlet(
        self, body: dict[str, Any], user: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Pass-through for now. Task 6 adds placeholder restoration."""
        if not self.valves.enabled:
            return body
        return body
