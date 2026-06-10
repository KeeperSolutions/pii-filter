"""
title: PII Filter
author: Keeper Solutions AI Lab
author_url: https://github.com/keeper-solutions/pii-filter
date: 2026-06-10
version: 0.9.8
license: MIT
description: PII detection and masking filter for Keeper AI Gateway. Task 3.2 — dual-analyzer architecture (HR + EN): two independent AnalyzerEngine instances (hr_core_news_lg + en_core_web_lg) run over every text fragment; results are merged and deduplicated before the existing _select_accepted_detections pipeline. Task 3.3 — cross-lingual NER spillover guard: per-detection window language classifier drops PERSON/LOCATION/NRP detections whose local context language does not match the source analyzer, eliminating EN-NER false positives on Croatian text. The thread vault is PostgreSQL-backed (asyncpg pool, idempotent DDL, lazy expiry), keyed by chat_id, gated by `vault_enabled`. v0.9.1 skips OpenWebUI background tasks (title/tags/follow-up generation) which embed chat history as user content and would produce false-positive detections. v0.9.2 removes DATE_TIME from the NER spillover guard and from PRESIDIO_TO_STANDARD — date substrings within credit card numbers (e.g. "12/27" in an expiry) were incorrectly tagged as DATE, producing false positives. v0.9.3 (Task 8) wires the `UserValves.pii_masking_enabled` per-user toggle into `inlet` (early return on opt-out, no vault touch, audit INFO log) and adds the admin-level `Valves.presidio_enabled` kill switch (runtime guard: skips analyzer/masking but still pulls vault snapshot for outlet restoration symmetry). v0.9.5 (Task 9) consolidates the vault to a single PostgreSQL backend: the legacy backend-selector valve and the legacy alternate-backend valves are removed, the alternate-backend class is dropped, the kept `ThreadVault` class is the sole vault implementation, and the alternate-backend Python dependencies are gone from `requirements.txt`. `outlet` is unchanged; `on_startup` is collapsed to a single Postgres-only init path gated by `vault_enabled`.
requirements: presidio-analyzer>=2.2.0, presidio-anonymizer>=2.2.0, spacy>=3.7.0, asyncpg>=0.29.0, pydantic-settings>=2.0, cryptography>=42.0, hr-core-news-lg==3.7.0, en-core-web-lg==3.7.0
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import re
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import CreditCardRecognizer
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)

# Per-process spaCy NLP engine cache keyed by lang_code ("hr", "en").
# Models load ~240–800 MB of word vectors; caching prevents re-allocation
# across repeated on_startup calls (test reruns, multiple Pipeline instances).
# AnalyzerEngine instances are still rebuilt per startup.
_nlp_engine_cache: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _iban_mod_97_check(iban: str) -> bool:
    """ISO 13616 / MOD 97-10 IBAN checksum verification.

    Strips all whitespace (ASCII space, tabs, newlines), moves the first 4
    chars to the end, converts letters to digits (A=10, B=11, ..., Z=35), and
    checks `int(numeric) % 97 == 1`. Whitespace stripping lets the recognizer
    patterns match the ISO 13616 4-char grouped form ("HR12 1001 ...") that
    banking apps display, then funnel the same value through this checksum.
    """
    pt = re.sub(r"\s+", "", iban)
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

    def __init__(self, supported_language: str) -> None:
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

    def __init__(self, supported_language: str) -> None:
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
    """HR IBAN — 21-char IBAN starting HR, MOD 97-10 checksum.

    The pattern accepts both concatenated form ("HR1210010051863000160")
    and the ISO 13616 4-char grouped form ("HR12 1001 0051 8630 0016 0")
    that banking apps display by default. The whitespace is stripped before
    the checksum runs, so the same input passes both gates.
    """

    PATTERNS: ClassVar[list[Pattern]] = [
        Pattern("HR IBAN", r"\bHR\d{2}(?:\s?\d{4}){4}\s?\d\b", 0.5)
    ]

    def __init__(self, supported_language: str) -> None:
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

    def __init__(self, supported_language: str) -> None:
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

    def __init__(self, supported_language: str) -> None:
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

    def __init__(self, supported_language: str) -> None:
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

    def __init__(self, supported_language: str) -> None:
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

    def __init__(self, supported_language: str) -> None:
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

    def __init__(self, supported_language: str) -> None:
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
    supported_language: str,
) -> PatternRecognizer:
    """Build a generic IBAN PatternRecognizer for a country.

    The factory delegates checksum validation to the shared
    `_iban_mod_97_check` helper, eliminating the duplication in the benchmark.

    The regex accepts both concatenated form and the ISO 13616 4-char
    grouped form ("IE29 AIBK 9311 5212 3456 78") banking apps display.
    Country code + check digits form the first 4-char group; the BBAN is
    split into `bban_length // 4` groups of 4 plus an optional trailing
    group with the remaining `bban_length % 4` chars. Whitespace between
    groups is optional; `_iban_mod_97_check` strips it before validation.
    """
    full_groups = bban_length // 4
    remainder = bban_length % 4
    pattern_str = rf"\b{country_code}\d{{2}}(?:\s?[A-Z0-9]{{4}}){{{full_groups}}}"
    if remainder:
        pattern_str += rf"\s?[A-Z0-9]{{{remainder}}}"
    pattern_str += r"\b"
    iban_pattern = Pattern(f"{country_code} IBAN", pattern_str, 0.5)

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


def _select_accepted_detections(
    text: str,
    detections: list[RecognizerResult],
    presidio_to_standard: dict[str, str],
    deny_list: frozenset[str] = frozenset(),
    trailing_strip: frozenset[str] = frozenset(),
    oib_phone_window: int = 0,
) -> list[RecognizerResult]:
    """Filter, prioritize, and de-overlap raw analyzer detections.

    Shared by `mask_text` (Task 4 fallback path) and the Task 5/5.1 inlet
    vault path. The placeholder *source* differs between them but detection
    *selection* is identical and must stay in lockstep.

    Processing order (Task 3.1 additions in steps 2-4):
      1. Whitelist filter — drop entity types not in `presidio_to_standard`.
      2. Deny-list filter — drop PERSON entities whose lowercased text either
         exactly matches an entry in `deny_list` or starts with a deny-listed
         entry followed by a space (suppresses spaCy false positives on
         English/code keywords).
      3. Trailing-token strip — for PERSON entities whose last whitespace-
         separated token is in `trailing_strip`, shorten the span to exclude
         that token (also strips trailing punctuation). Drop if span becomes
         empty after stripping.
      4. OIB phone-context check — for HR_OIB entities, examine the
         `oib_phone_window` chars immediately preceding the detection start.
         If a phone-keyword pattern matches AND no OIB context word is present,
         drop the detection (likely a phone number, not a real OIB).
      5. Overlap resolution — sort by `(score DESC, custom_first, start ASC)`,
         accept non-overlapping spans. Zero-length/inverted spans skipped.

    Returns surviving detections sorted by `start` ASC, ready to be spliced
    into the masked output. Empty list when `text` is falsy, no detections,
    or no detection survives all filters.

    The `deny_list`, `trailing_strip`, and `oib_phone_window` parameters
    default to empty/zero so `mask_text` (Task 4 path) continues working
    without modification — it calls this function without the new kwargs.
    """
    if not text or not detections:
        return []

    # Step 1: Whitelist filter
    candidates: list[RecognizerResult] = [
        d for d in detections if d.entity_type in presidio_to_standard
    ]
    if not candidates:
        return []

    # Step 2: Deny-list filter (PERSON entities only)
    if deny_list:
        kept: list[RecognizerResult] = []
        for d in candidates:
            if d.entity_type == "PERSON":
                entity_lower = text[d.start : d.end].lower().strip()
                if entity_lower in deny_list:
                    continue
                if any(entity_lower.startswith(denied + " ") for denied in deny_list):
                    continue
            kept.append(d)
        candidates = kept
        if not candidates:
            return []

    # Step 3: Trailing-token strip (PERSON entities only)
    if trailing_strip:
        processed: list[RecognizerResult] = []
        for d in candidates:
            if d.entity_type == "PERSON":
                entity_text = text[d.start : d.end]
                # Strip trailing punctuation first
                clean = entity_text.rstrip(".,;!?:()")
                if clean:
                    tokens = clean.split()
                    if tokens and tokens[-1].lower() in trailing_strip:
                        last_tok = tokens[-1]
                        last_tok_offset = clean.rfind(last_tok)
                        new_text = clean[:last_tok_offset].rstrip()
                    else:
                        new_text = clean
                else:
                    new_text = ""
                if not new_text:
                    continue  # Drop empty/whitespace-only span
                if new_text != entity_text:
                    d = RecognizerResult(
                        entity_type=d.entity_type,
                        start=d.start,
                        end=d.start + len(new_text),
                        score=d.score,
                    )
            processed.append(d)
        candidates = processed
        if not candidates:
            return []

    # Step 4: OIB phone-context check
    if oib_phone_window > 0:
        oib_kept: list[RecognizerResult] = []
        for d in candidates:
            if d.entity_type == "HR_OIB":
                window_start = max(0, d.start - oib_phone_window)
                window = text[window_start : d.start]
                if _OIB_CONTEXT_PATTERN.search(window):
                    # Positive OIB context overrides any phone-context match
                    oib_kept.append(d)
                elif any(p.search(window) for p in _PHONE_CONTEXT_PATTERNS):
                    continue  # Phone context — drop
                else:
                    oib_kept.append(d)
            else:
                oib_kept.append(d)
        candidates = oib_kept
        if not candidates:
            return []

    # Step 5: Overlap resolution (unchanged from Task 3 baseline)
    candidates.sort(
        key=lambda d: (
            -d.score,
            0 if d.entity_type in CUSTOM_ENTITY_TYPES else 1,
            d.start,
        )
    )
    accepted: list[RecognizerResult] = []
    for det in candidates:
        if det.start >= det.end:
            continue
        if any(not (det.end <= a.start or a.end <= det.start) for a in accepted):
            continue
        accepted.append(det)
    accepted.sort(key=lambda d: d.start)
    return accepted


def _build_enriched_detection(
    det: RecognizerResult,
    text: str,
    standard_type: str,
    original: str,
    placeholder: str,
) -> dict[str, Any]:
    """Assemble the per-detection metadata dict the inlet stashes in body.metadata."""
    return {
        "entity_type": standard_type,
        "start": det.start,
        "end": det.end,
        "score": det.score,
        "raw_entity_type": det.entity_type,
        "original": original,
        "placeholder": placeholder,
    }


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

    Note:
        Selection logic is shared with the inlet vault path via
        `_select_accepted_detections`. Only the placeholder *source* differs.
    """
    accepted = _select_accepted_detections(text, detections, presidio_to_standard)
    if not accepted:
        return text, []

    # Build masked text in a single left-to-right pass and enrich detections.
    # The placeholder *source* here is the local `forward_map` / `counter_state`
    # passed in by the caller; the inlet vault path uses the same
    # selection logic but sources placeholders from `ThreadVault`.
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
        enriched.append(_build_enriched_detection(det, text, standard_type, original, placeholder))
    pieces.append(text[last_end:])

    return "".join(pieces), enriched


# ---------------------------------------------------------------------------
# Restoration (Task 6 — outlet placeholder → original substitution)
# ---------------------------------------------------------------------------


# Matches every placeholder shape minted by `mask_text` and the Task 5 vault:
# square brackets, an UPPER_SNAKE entity type (e.g. `HR_OIB`, `PERSON`,
# `CREDIT_CARD`), an underscore, then a positive integer counter. Compiled
# once at module load so `restore_text` does not pay re.compile per call.
_PLACEHOLDER_RE = re.compile(r"\[[A-Z_]+_\d+\]")

# Compiled once at module load so _select_accepted_detections pays no re.compile per call.
# Each pattern uses a $ anchor so it only matches when the phone keyword is immediately
# before the 11-digit number (i.e. at the end of the look-behind window string).
_PHONE_CONTEXT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in [
        r"phone\s*[:=]?\s*$",
        r"tel(?:ephone)?\s*[:=]?\s*$",
        r"mob(?:ile)?\s*[:=]?\s*$",
        r"mobitel\s*[:=]?\s*$",
        r"telefon\s*[:=]?\s*$",
        r"\+1[-.\s]?$",
        r"\+385[-.\s]?$",
        r"\+49[-.\s]?$",
        r"\(\d{3}\)\s*$",
        r"\d{3}[-.\s]?\d{3}[-.\s]?$",
    ]
)

# Positive override: if an OIB context word is found in the same window, the detection
# is kept regardless of any phone-context match.
_OIB_CONTEXT_PATTERN: re.Pattern[str] = re.compile(
    r"\boib\b|osobni identifikacijski broj|osobni broj",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Task 3.3 — Cross-lingual NER spillover filter constants
# ---------------------------------------------------------------------------

# HR markers: diacritics (any hit = strong HR signal) + common HR stopwords/verb forms.
# Case-insensitive. ≥ 12 markers required by AC 3.3.1.
_HR_MARKERS: re.Pattern[str] = re.compile(
    r"\b(?:je|su|sam|ću|nije|ima|nema|moj|moja|moje|mojeg|mojim|"
    r"tvoj|naš|naša|ovaj|ova|ovo|taj|ta|to|ali|nego|gdje|"
    r"kako|što|kao|samo|već|još|biti|bio|bila|bilo)\b|[čšžđćČŠŽĐĆ]",
    re.IGNORECASE,
)

# EN markers: common EN stopwords and function words. ≥ 12 markers required by AC 3.3.1.
_EN_MARKERS: re.Pattern[str] = re.compile(
    r"\b(?:the|is|are|was|were|am|been|being|have|has|had|"
    r"my|your|our|his|her|its|their|this|that|these|those|"
    r"and|but|where|how|what|with|from|about|please|thank|"
    r"not|for|into|onto|upon|after|before|during)\b",
    re.IGNORECASE,
)

# NER entity types subject to the cross-lingual window filter.
# Regex-based entity types (HR_OIB, US_SSN, EMAIL_ADDRESS, …) are excluded — they are
# language-agnostic and must never be filtered by window language.
_NER_ENTITY_TYPES: frozenset[str] = frozenset({"PERSON", "LOCATION", "NRP"})


def _classify_window_language(
    text: str,
    start: int,
    end: int,
    window_chars: int = 30,
) -> Literal["hr", "en"]:
    """Classify the language of the local text window surrounding a detection span.

    Counts HR vs EN marker matches in text[start-window_chars : end+window_chars].
    On tie (equal counts or no markers at all), returns 'hr' — deployment region
    default per Task 3.3 pre-locked decision Q4.

    Window is clipped to text boundaries to avoid index underflow/overflow.
    """
    window_start = max(0, start - window_chars)
    window_end = min(len(text), end + window_chars)
    window = text[window_start:window_end]

    hr_count = len(_HR_MARKERS.findall(window))
    en_count = len(_EN_MARKERS.findall(window))

    return "en" if en_count > hr_count else "hr"


def restore_text(
    text: str,
    reverse_map: dict[str, str],
) -> tuple[str, list[str], list[str]]:
    """Replace placeholders in `text` with their originals from `reverse_map`.

    Args:
        text: assistant response containing zero or more placeholders of the
            shape minted by `mask_text` / `ThreadVault.get_placeholder`.
        reverse_map: `placeholder -> original_value`, populated by inlet and
            persisted on `body["metadata"]["pii_reverse_map"]`. Outlet treats
            this as the only source of truth (Task 6 Decision 1).

    Returns:
        `(restored_text, restored_placeholders, hallucinated_placeholders)`:
          * `restored_text` — input with each known placeholder substituted
            for its original value in a single left-to-right pass.
          * `restored_placeholders` — sorted, deduped list of placeholders
            that were actually substituted. Useful for counter aggregation.
          * `hallucinated_placeholders` — sorted, deduped list of
            placeholder-shaped substrings that the regex matched but that
            the reverse_map could not resolve. They are left **literally**
            in `restored_text`. The outlet logs these at WARN level so an
            LLM that fabricates `[PERSON_99]` is observable, not silently
            substituted (epic AC: "zero hallucinated restorations").

    Implementation note:
        Uses `re.sub` with a callable replacement, not a sequence of
        `str.replace` calls. A `str.replace` chain would do N passes over
        the text and could re-replace already-restored substrings if an
        original happened to contain a placeholder-shaped substring (rare
        but real). The single-pass `re.sub` is O(text length) and atomic.
    """
    if not text or not reverse_map:
        return text, [], []

    restored_set: set[str] = set()
    hallucinated_set: set[str] = set()

    def _sub(match: re.Match[str]) -> str:
        placeholder = match.group(0)
        original = reverse_map.get(placeholder)
        if original is None:
            hallucinated_set.add(placeholder)
            return placeholder
        restored_set.add(placeholder)
        return original

    restored_text = _PLACEHOLDER_RE.sub(_sub, text)
    return restored_text, sorted(restored_set), sorted(hallucinated_set)


_EPHEMERAL_PREFIX = "ephemeral:"


def make_ephemeral_thread_id() -> str:
    """Generate a fresh ephemeral thread id used when chat_id is missing.

    The `ephemeral:` prefix is recognized by `ThreadVault` for selecting the
    short TTL (per spec §2.1.4 — `ephemeral_ttl_seconds`). Single-request
    mask/unmask works; cross-request consistency does not — there's no
    chat_id to thread on next time anyway.
    """
    return f"{_EPHEMERAL_PREFIX}{uuid.uuid4()}"


# ---------------------------------------------------------------------------
# Thread vault — PostgreSQL-backed, thread-scoped placeholder storage
# ---------------------------------------------------------------------------


# Idempotent DDL run on every `initialize()` call. Two tables: mappings
# (PK = chat_id+type+original; unique reverse index on chat_id+placeholder)
# and counters (PK = chat_id+type). Counter is bumped before mapping insert
# so the candidate placeholder string can encode the freshly minted index.
# Counter gaps under concurrency are tolerated — placeholder uniqueness
# within a thread is preserved by the unique reverse index. See spec
# §2.3 for the race-condition analysis.
_POSTGRES_DDL = """
CREATE TABLE IF NOT EXISTS pii_thread_mappings (
  chat_id        TEXT NOT NULL,
  entity_type    TEXT NOT NULL,
  original_value TEXT NOT NULL,          -- ENC1:<base64...> ciphertext, or plaintext when encryption disabled
  lookup_hash    BYTEA NOT NULL,         -- HMAC-SHA256(blind_key, framed(chat_id, entity_type, plaintext))
  placeholder    TEXT NOT NULL,
  counter_index  INTEGER NOT NULL,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  expires_at     TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (chat_id, entity_type, lookup_hash)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_pii_mappings_reverse
  ON pii_thread_mappings (chat_id, placeholder);

CREATE INDEX IF NOT EXISTS idx_pii_mappings_expires
  ON pii_thread_mappings (expires_at);

CREATE TABLE IF NOT EXISTS pii_thread_counters (
  chat_id     TEXT NOT NULL,
  entity_type TEXT NOT NULL,
  next_value  INTEGER NOT NULL DEFAULT 1,
  expires_at  TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (chat_id, entity_type)
);

CREATE INDEX IF NOT EXISTS idx_pii_counters_expires
  ON pii_thread_counters (expires_at);
"""


# ---------------------------------------------------------------------------
# Vault encryption-at-rest (Task 11 — Option E: encrypted value + blind index)
# ---------------------------------------------------------------------------
#
# All crypto/KMS logic lives inline here (no sibling top-level `.py` files —
# the Pipelines loader would try to load them as pipelines and fail). The
# envelope format is ported from the keeper-openwebui fork's chat encryption
# (`crypto.py`): `open_webui` cannot be imported in the Pipelines container,
# so the logic is reimplemented rather than imported.


class VaultCipher:
    """AES-256-GCM envelope cipher for vault values (port of the
    keeper-openwebui ``crypto.py`` ``ENC1:`` format).

    Envelope layout — packed big-endian, then base64-encoded behind an
    ``ENC1:`` ASCII prefix::

        ENC1:<base64( [1B version=1][4B key_id BE][12B nonce][ciphertext‖16B GCM tag] )>

    A fresh 12-byte random nonce is drawn per ``encrypt`` call. The
    (key, nonce) pair must never repeat and the nonce must never be derived
    from a counter: with random 96-bit nonces and the vault's 24-48 h TTL the
    per-key encryption count stays far below the 2**32 birthday bound where
    nonce collisions become a concern. ``key_id`` is packed for a future
    read-old/write-new rotation; the v1 decrypt path parses past it but always
    uses the single configured key.
    """

    _PREFIX = "ENC1:"
    _VERSION = 1
    _NONCE_LEN = 12
    _TAG_LEN = 16
    _HEADER_LEN = 1 + 4  # version + key_id

    def __init__(self, key: bytes, key_id: int = 1) -> None:
        if len(key) != 32:
            raise ValueError("VaultCipher key must be exactly 32 bytes (AES-256).")
        if not 0 <= key_id <= 0xFFFFFFFF:
            raise ValueError("VaultCipher key_id must fit in an unsigned 32-bit integer.")
        self._aesgcm = AESGCM(key)
        self._key_id = key_id

    @staticmethod
    def is_encrypted(value: str) -> bool:
        """True if ``value`` carries the ``ENC1:`` envelope prefix."""
        return value.startswith(VaultCipher._PREFIX)

    def encrypt(self, plaintext: str) -> str:
        """Encrypt ``plaintext`` into an ``ENC1:`` envelope string."""
        nonce = os.urandom(self._NONCE_LEN)
        ct_and_tag = self._aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
        header = bytes([self._VERSION]) + self._key_id.to_bytes(4, "big")
        envelope = header + nonce + ct_and_tag
        return self._PREFIX + base64.b64encode(envelope).decode("ascii")

    def decrypt(self, value: str) -> str:
        """Decrypt an ``ENC1:`` envelope back to plaintext.

        Raises ``cryptography.exceptions.InvalidTag`` on a wrong key or a
        tampered ciphertext/tag, and ``ValueError`` on a structurally
        malformed envelope. Callers on the read path catch both and treat the
        row as a miss (spec D3 — never raise out of the outlet).
        """
        if not self.is_encrypted(value):
            raise ValueError("VaultCipher.decrypt called on a non-ENC1 value.")
        raw = base64.b64decode(value[len(self._PREFIX) :], validate=True)
        if len(raw) < self._HEADER_LEN + self._NONCE_LEN + self._TAG_LEN:
            raise ValueError("VaultCipher envelope too short to be a valid ENC1 blob.")
        version = raw[0]
        if version != self._VERSION:
            raise ValueError(f"Unsupported VaultCipher envelope version: {version}.")
        # key_id (raw[1:5]) is parsed past but ignored for v1 (single key).
        nonce = raw[self._HEADER_LEN : self._HEADER_LEN + self._NONCE_LEN]
        ct_and_tag = raw[self._HEADER_LEN + self._NONCE_LEN :]
        return self._aesgcm.decrypt(nonce, ct_and_tag, None).decode("utf-8")


class BlindIndex:
    """Keyed HMAC-SHA256 blind index for thread-scoped dedup.

    The vault PK includes ``lookup_hash`` instead of the (now-encrypted)
    ``original_value`` so the ``INSERT ... ON CONFLICT`` UPSERT can still dedup
    a repeated value within a thread. The token is a keyed HMAC, so an attacker
    with read access to the table cannot recover the plaintext from the hash
    without the key. The key is independent from the GCM encryption key — byte
    material is never shared between the two (spec E2).
    """

    _DOMAIN = b"pii-vault-blind-index-v1"

    def __init__(self, key: bytes) -> None:
        if len(key) != 32:
            raise ValueError("BlindIndex key must be exactly 32 bytes.")
        self._key = key

    def compute(self, chat_id: str, entity_type: str, plaintext: str) -> bytes:
        """Return the 32-byte blind-index token for ``(chat_id, entity_type, plaintext)``.

        Deterministic within a thread (same inputs → same token, so dedup
        works) and isolated across threads (``chat_id`` is part of the framed
        input, so the same PII in chatA vs chatB hashes differently).

        Framing is collision-resistant: a fixed domain tag followed by each
        field length-prefixed with its big-endian u32 byte length, so no two
        distinct ``(chat_id, entity_type, plaintext)`` triples can produce the
        same byte stream (e.g. ``("a", "bc", v)`` vs ``("ab", "c", v)``).
        """
        framed = self._DOMAIN
        for part in (
            chat_id.encode("utf-8"),
            entity_type.encode("utf-8"),
            plaintext.encode("utf-8"),
        ):
            framed += len(part).to_bytes(4, "big") + part
        return hmac.new(self._key, framed, hashlib.sha256).digest()


class KeyManager:
    """Loads the vault encryption + blind-index keys from the configured
    backend (mirrors the keeper-openwebui ``kms.py`` dual-backend shape).

    ``local``   — keys are base64 valve fields (dev / self-hosted).
    ``gcp_kms`` — keys are Google Secret Manager payloads; the heavy
                  ``google-cloud-secret-manager`` import is deferred to that
                  branch so the default container never pays for it (§8).

    The enc key and the blind-index key are independent 32-byte keys. Keys are
    loaded once at ``on_startup`` and held in memory by the constructed
    ``VaultCipher`` / ``BlindIndex``; there is no per-request fetch. Every
    failure mode (empty key, bad base64, wrong length, missing backend package)
    raises ``RuntimeError`` so startup fails closed (spec §6 / E6).
    """

    def __init__(
        self,
        *,
        backend: str,
        encryption_key_b64: str = "",
        blind_index_key_b64: str = "",
        gcp_enc_secret: str = "",
        gcp_blind_secret: str = "",
    ) -> None:
        self._backend = backend
        self._encryption_key_b64 = encryption_key_b64
        self._blind_index_key_b64 = blind_index_key_b64
        self._gcp_enc_secret = gcp_enc_secret
        self._gcp_blind_secret = gcp_blind_secret

    def load_blind_index_key(self) -> bytes:
        """Resolve and validate the 32-byte blind-index HMAC key."""
        return self._resolve("blind_index", self._blind_index_key_b64, self._gcp_blind_secret)

    def load_encryption_key(self) -> bytes:
        """Resolve and validate the 32-byte AES-256-GCM encryption key."""
        return self._resolve("encryption", self._encryption_key_b64, self._gcp_enc_secret)

    def _resolve(self, label: str, local_b64: str, gcp_secret: str) -> bytes:
        if self._backend == "local":
            return self._decode_32(local_b64, label, f"valve vault_{label}_key")
        if self._backend == "gcp_kms":
            return self._decode_32(
                self._fetch_gcp_secret(gcp_secret, label), label, f"gcp secret '{gcp_secret}'"
            )
        raise RuntimeError(
            f"Unknown vault_kms_backend '{self._backend}'; expected 'local' or 'gcp_kms'."
        )

    @staticmethod
    def _decode_32(raw_b64: str, label: str, source: str) -> bytes:
        if not raw_b64:
            raise RuntimeError(
                f"Vault {label} key is empty ({source}). "
                "A base64-encoded 32-byte key is required."
            )
        try:
            key = base64.b64decode(raw_b64, validate=True)
        except (ValueError, TypeError) as exc:
            raise RuntimeError(f"Vault {label} key ({source}) is not valid base64: {exc}") from exc
        if len(key) != 32:
            raise RuntimeError(
                f"Vault {label} key ({source}) decodes to {len(key)} bytes; exactly 32 required."
            )
        return key

    @staticmethod
    def _fetch_gcp_secret(secret_name: str, label: str) -> str:
        if not secret_name:
            # Map the internal label back to the real valve field name so the
            # fail-closed message points operators at the right env var.
            field = {
                "encryption": "vault_gcp_enc_secret",
                "blind_index": "vault_gcp_blind_secret",
            }.get(label, f"vault_gcp_{label}_secret")
            raise RuntimeError(f"{field} is empty but vault_kms_backend='gcp_kms'.")
        try:
            # Lazy import (§8): keeps the heavy grpc/protobuf chain out of the
            # default container, which only needs the `local` backend.
            from google.cloud import secretmanager
        except ImportError as exc:
            raise RuntimeError(
                "vault_kms_backend='gcp_kms' requires the 'google-cloud-secret-manager' "
                "package, which is not installed in this container. Add it to the prod "
                "image / a prod requirements profile (handoff to Senka)."
            ) from exc
        client = secretmanager.SecretManagerServiceClient()
        response = client.access_secret_version(name=secret_name)
        payload: bytes = response.payload.data
        return payload.decode("utf-8")


class ThreadVault:
    """Thread-scoped placeholder vault backed by PostgreSQL.

    Thread-scoped placeholder vault for cross-message PII consistency.
    Public async API:

        get_or_create_thread(chat_id) -> None
        get_placeholder(chat_id, original, entity_type) -> str
        restore(chat_id, placeholder) -> str | None
        snapshot_for_request(chat_id) -> tuple[dict[str, str], dict[str, str]]
        healthcheck() -> bool
        aclose() -> None

    NOTE on `get_placeholder` arg order — TASK-05.1 fixed an earlier mismatch
    signature `(chat_id, original, entity_type)` so the inlet's call site
    works against either backend without modification. Spec §2.1.1 / §2.3
    listed `(chat_id, entity_type, original_value)` but the earlier vault
    shipped in Task 5 with `(original, entity_type)` order; preserving that
    is mandatory for the duck-typed call to keep working.

    Atomicity: `get_placeholder` issues two `INSERT ... ON CONFLICT`
    statements inside a single transaction. Counter bump is first (so the
    candidate placeholder string can include the new index); the mapping
    insert is second, falling back to `DO UPDATE SET expires_at` on
    conflict so concurrent callers observe the same placeholder. Counter
    gaps under concurrency are tolerated; placeholder uniqueness within a
    thread is preserved by the unique reverse index. See spec §2.3.

    Lazy expiry: every read query filters `WHERE expires_at > now()` so
    expired rows are invisible to callers without a background cleanup
    job. TTL renewal is performed on every public method that touches a
    thread's data; the renewal is in-line with the same query when
    possible (UPDATE ... RETURNING pattern), or as a sibling UPDATE for
    bulk paths like `snapshot_for_request`.
    """

    def __init__(
        self,
        dsn: str,
        *,
        pool_min: int = 2,
        pool_max: int = 10,
        command_timeout: float = 5.0,
        thread_ttl_seconds: int = 86400,
        ephemeral_ttl_seconds: int = 600,
        cipher: VaultCipher | None = None,
        blind_index: BlindIndex | None = None,
        encryption_strict: bool = False,
    ) -> None:
        self._dsn = dsn
        self._pool_min = pool_min
        self._pool_max = pool_max
        self._command_timeout = command_timeout
        self._thread_ttl = thread_ttl_seconds
        self._ephemeral_ttl = ephemeral_ttl_seconds
        # Task 11 vault encryption-at-rest. `blind_index` is required for any
        # write (lookup_hash is NOT NULL and part of the PK — spec D1); the
        # production `on_startup` path always constructs one. `cipher` is None
        # when encryption is disabled, in which case `original_value` is stored
        # as plaintext. `encryption_strict` controls whether the read path
        # refuses an unexpected plaintext row (spec §6/§7.2).
        self._cipher = cipher
        self._blind_index = blind_index
        self._encryption_strict = encryption_strict
        # Pool is created lazily by `initialize()` so `__init__` never opens
        # sockets and remains safe for unit-test instantiation.
        self._pool: asyncpg.Pool[asyncpg.Record] | None = None

    def _decrypt_stored_value(self, stored: str, chat_id: str, placeholder: str) -> str | None:
        """Decrypt a stored ``original_value`` for the read path.

        Returns the plaintext, or ``None`` when the row must be treated as a
        miss/skip. Never raises (spec D3 — the outlet must never crash, so a
        skipped row simply leaves its placeholder in the user-facing text):

          * ENC1 envelope → decrypt; on ``InvalidTag`` / malformed envelope →
            log WARN and return ``None``.
          * plaintext while encryption is enabled (cipher present) and
            ``encryption_strict`` → unexpected; log ERROR and return ``None``.
          * plaintext otherwise (encryption disabled, or non-strict legacy
            row) → return as-is.
        """
        cipher = self._cipher
        if cipher is not None and cipher.is_encrypted(stored):
            try:
                return cipher.decrypt(stored)
            except (InvalidTag, ValueError) as exc:
                logger.warning(
                    "Vault decrypt failed for chat_id=%s placeholder=%s (%s: %s); "
                    "skipping row, placeholder left masked.",
                    chat_id,
                    placeholder,
                    type(exc).__name__,
                    exc,
                )
                return None
        if cipher is not None and self._encryption_strict:
            logger.error(
                "Vault strict mode: unexpected plaintext original_value for "
                "chat_id=%s placeholder=%s; refusing to serve, row skipped.",
                chat_id,
                placeholder,
            )
            return None
        return stored

    # -- helpers -------------------------------------------------------------

    def _ttl_for(self, chat_id: str) -> int:
        return self._ephemeral_ttl if chat_id.startswith(_EPHEMERAL_PREFIX) else self._thread_ttl

    def _expires_at(self, chat_id: str) -> datetime:
        return datetime.now(tz=UTC) + timedelta(seconds=self._ttl_for(chat_id))

    def _require_pool(self) -> asyncpg.Pool[asyncpg.Record]:
        if self._pool is None:
            raise RuntimeError("ThreadVault not initialized: call await vault.initialize() first.")
        return self._pool

    # -- lifecycle -----------------------------------------------------------

    async def initialize(self) -> None:
        """Create the asyncpg pool and run idempotent DDL.

        Safe to call multiple times in the same process — `CREATE TABLE IF
        NOT EXISTS` and `CREATE INDEX IF NOT EXISTS` make the schema setup
        a no-op on subsequent calls. If a prior pool exists from an earlier
        `initialize()`, it is closed before being replaced so its open
        sockets don't leak until GC.
        """
        import asyncpg as _asyncpg  # local import keeps top-level cheap

        # Close any pool from a prior `initialize()` call before
        # overwriting `self._pool`. `aclose()` is idempotent and a no-op
        # when `self._pool is None`, so this is safe on the cold path.
        await self.aclose()

        self._pool = await _asyncpg.create_pool(
            dsn=self._dsn,
            min_size=self._pool_min,
            max_size=self._pool_max,
            command_timeout=self._command_timeout,
            max_inactive_connection_lifetime=300.0,
        )
        assert self._pool is not None  # for mypy; create_pool returns Pool, not None
        async with self._pool.acquire() as conn:
            await conn.execute(_POSTGRES_DDL)

    async def aclose(self) -> None:
        """Close the connection pool. Idempotent — safe to call multiple times."""
        if self._pool is not None:
            try:
                await self._pool.close()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("ThreadVault aclose() raised: %s", exc)
            finally:
                self._pool = None

    async def healthcheck(self) -> bool:
        """Return True if `SELECT 1` succeeds within ~1s, False otherwise.

        Acquires a connection with a 1-second timeout so a saturated pool or
        an unreachable database fails fast and the inlet's degradation path
        can branch on the bool. Never raises.
        """
        if self._pool is None:
            return False
        try:
            async with self._pool.acquire(timeout=1.0) as conn:
                # asyncpg ships no `py.typed`, so `fetchval` returns `Any`.
                # Annotate locally to keep mypy --strict from flagging the
                # `value == 1` comparison as an Any-leaking return.
                value: int | None = await conn.fetchval("SELECT 1", timeout=1.0)
            return value == 1
        except Exception as exc:
            logger.warning("ThreadVault healthcheck failed: %s", exc)
            return False

    # -- public API ----------------------------------------------------------

    async def get_or_create_thread(self, chat_id: str) -> None:
        """API-parity no-op.

        The chat_id is the only thread identifier; there is no per-thread
        row to create until the first mapping is written. Returns None to
        match `ThreadVault.get_or_create_thread`'s signature so the inlet
        calls either backend without conditional branching.
        """
        return None

    async def get_placeholder(self, chat_id: str, original: str, entity_type: str) -> str:
        """Atomic get-or-mint. Idempotent under concurrency for the same
        `(chat_id, entity_type, original)`.

        Step A bumps the per-(chat_id, entity_type) counter and returns the
        new index. Step B inserts the mapping with `[entity_type_N]` as the
        placeholder; on conflict (another caller already wrote this row),
        the existing placeholder is returned via `RETURNING placeholder`.
        Both steps run inside a single transaction. TTL is bumped on both
        rows in this call.

        Expired rows for `chat_id` are deleted at the top of the
        transaction so a stale counter never bumps off an old `next_value`
        and a stale mapping never resurrects an old placeholder via
        `ON CONFLICT DO UPDATE RETURNING` so that a fresh thread always
        starts at index 1, and is the cleanup hook for GDPR TTLs (PII rows
        get physically purged on the next access against the same chat_id).
        """
        pool = self._require_pool()
        expires_at = self._expires_at(chat_id)

        async with pool.acquire() as conn, conn.transaction():
            # Purge expired rows for this chat before the UPSERTs.
            # Scoped to chat_id so unrelated threads' rows are untouched
            # (a global sweep belongs in a separate cleanup job, not on
            # the request-path hot path).
            await conn.execute(
                "DELETE FROM pii_thread_counters " "WHERE chat_id = $1 AND expires_at <= now()",
                chat_id,
            )
            await conn.execute(
                "DELETE FROM pii_thread_mappings " "WHERE chat_id = $1 AND expires_at <= now()",
                chat_id,
            )

            counter_row = await conn.fetchrow(
                """
                INSERT INTO pii_thread_counters (chat_id, entity_type, next_value, expires_at)
                VALUES ($1, $2, 2, $3)
                ON CONFLICT (chat_id, entity_type) DO UPDATE
                  SET next_value = pii_thread_counters.next_value + 1,
                      expires_at = EXCLUDED.expires_at
                RETURNING next_value - 1 AS minted_index
                """,
                chat_id,
                entity_type,
                expires_at,
            )
            # asyncpg returns Record | None; the INSERT ... RETURNING above
            # always produces exactly one row, so None is unreachable in
            # practice — assert for mypy.
            assert counter_row is not None
            minted_index = int(counter_row["minted_index"])
            candidate = f"[{entity_type}_{minted_index}]"

            # Task 11: the blind index is always computed (lookup_hash is NOT
            # NULL and part of the PK — spec D1); `original_value` is ciphertext
            # only when encryption is enabled (cipher present), else the raw
            # plaintext is stored.
            blind_index = self._blind_index
            if blind_index is None:
                raise RuntimeError(
                    "ThreadVault.get_placeholder requires a BlindIndex (lookup_hash is "
                    "NOT NULL); construct the vault with a blind_index (spec D1)."
                )
            lookup_hash = blind_index.compute(chat_id, entity_type, original)
            stored = self._cipher.encrypt(original) if self._cipher is not None else original

            mapping_row = await conn.fetchrow(
                """
                INSERT INTO pii_thread_mappings
                  (chat_id, entity_type, original_value, lookup_hash,
                   placeholder, counter_index, expires_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (chat_id, entity_type, lookup_hash) DO UPDATE
                  SET expires_at = EXCLUDED.expires_at
                RETURNING placeholder
                """,
                chat_id,
                entity_type,
                stored,
                lookup_hash,
                candidate,
                minted_index,
                expires_at,
            )
            assert mapping_row is not None
            return cast(str, mapping_row["placeholder"])

    async def restore(self, chat_id: str, placeholder: str) -> str | None:
        """Reverse-lookup a placeholder. Returns None for unknown / expired.

        Bumps `expires_at` on hit via `UPDATE ... RETURNING` so a single
        round-trip covers both the lookup and the TTL renewal. A miss
        leaves the table untouched; this implements renew-on-touch
        behavior (the bulk UPDATE renews counters/forward rows
        on every read but only for the chat_id, not on a miss against a
        specific placeholder).
        """
        pool = self._require_pool()
        expires_at = self._expires_at(chat_id)

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE pii_thread_mappings
                SET expires_at = $3
                WHERE chat_id = $1
                  AND placeholder = $2
                  AND expires_at > now()
                RETURNING original_value
                """,
                chat_id,
                placeholder,
                expires_at,
            )
        if row is None:
            return None
        # Task 11: `original_value` is an ENC1 envelope (or plaintext when
        # encryption is disabled). Decrypt with the never-raise fallback so a
        # tampered / wrong-key / unexpected-plaintext row reads as a miss.
        stored = cast(str, row["original_value"])
        return self._decrypt_stored_value(stored, chat_id, placeholder)

    async def snapshot_for_request(self, chat_id: str) -> tuple[dict[str, str], dict[str, str]]:
        """Return forward + reverse maps for this thread.

        Bulk TTL renewal: UPDATE non-expired mapping rows for the chat_id,
        UPDATE non-expired counter rows, then SELECT the snapshot. The
        renewal WHERE clauses include `expires_at > now()` so an
        already-expired row is NOT bumped back to life by a later
        snapshot call — TTL-expired rows must stay invisible (a row past
        TTL is gone) and silently extend PII retention past the GDPR
        deadline. The SELECT applies the same filter so expired rows
        are also invisible to the caller.
        """
        pool = self._require_pool()
        expires_at = self._expires_at(chat_id)

        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE pii_thread_mappings SET expires_at = $2 "
                "WHERE chat_id = $1 AND expires_at > now()",
                chat_id,
                expires_at,
            )
            await conn.execute(
                "UPDATE pii_thread_counters SET expires_at = $2 "
                "WHERE chat_id = $1 AND expires_at > now()",
                chat_id,
                expires_at,
            )
            rows = await conn.fetch(
                """
                SELECT original_value, placeholder
                FROM pii_thread_mappings
                WHERE chat_id = $1 AND expires_at > now()
                """,
                chat_id,
            )

        # Task 11: decrypt each stored `original_value` before building the
        # plaintext-keyed maps. A row that fails decryption (tampered / wrong
        # key) or is unexpected plaintext in strict mode is skipped — never
        # raising preserves the never-crash outlet contract (spec D3 / §7.3).
        # Both maps stay keyed/valued on plaintext (in-memory only), exactly as
        # before; the outlet `restore_text` is unchanged.
        forward: dict[str, str] = {}
        reverse: dict[str, str] = {}
        for row in rows:
            placeholder = cast(str, row["placeholder"])
            original = self._decrypt_stored_value(
                cast(str, row["original_value"]), chat_id, placeholder
            )
            if original is None:
                continue
            forward[original] = placeholder
            reverse[placeholder] = original
        return forward, reverse


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


# Built-in Presidio recognizers disabled per language registry.
# HR: IbanRecognizer conflicts with our HRIBANRecognizer; the others produce
#     noisy false positives on Croatian text.
# EN: IbanRecognizer is intentionally kept active — covers DE/FR/ES/IT IBANs
#     that our country-specific custom recognizers do not handle. Our custom
#     IE/RO/GB IBAN recognizers (duplicated in EN registry) have checksum
#     validation and win overlap resolution on their own spans.
_DISABLED_BUILTIN_RECOGNIZERS_HR: tuple[str, ...] = (
    "IbanRecognizer",
    "UrlRecognizer",
    "OrganizationRecognizer",
    "MedicalLicenseRecognizer",
)
_DISABLED_BUILTIN_RECOGNIZERS_EN: tuple[str, ...] = (
    "UrlRecognizer",
    "OrganizationRecognizer",
    "MedicalLicenseRecognizer",
)


def _find_last_user_index(messages: list[dict[str, Any]]) -> int:
    """Return the index of the last message with role=='user', or -1 if none.

    Preserved for the Task 4 backward-compat path used when
    multi_turn_history_scope=False or multi_turn_history_max_messages==0.
    """
    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]
        if isinstance(message, dict) and message.get("role") == "user":
            return i
    return -1


def _get_message_text_for_precheck(msg: dict[str, Any]) -> str:
    """Return all text content from a message as a single string for regex pre-check.

    Handles both str content and list[dict] multimodal content.
    Image-only parts contribute nothing; returns '' if no text exists.
    """
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts)
    return ""


def _merge_dedupe_detections(
    hr_results: list[RecognizerResult],
    en_results: list[RecognizerResult],
    text: str,
) -> tuple[list[RecognizerResult], int]:
    """Concatenate HR + EN detections; filter NER detections by window language; deduplicate.

    Filter step (Task 3.3): for each detection whose entity_type is in
    _NER_ENTITY_TYPES (PERSON, LOCATION, NRP), the ±30-char window
    around the span is classified as 'hr' or 'en'. If the window language does
    not match the detection's source analyzer, the detection is dropped and its
    count is added to the returned spillover_dropped counter. Regex-based entity
    types are never filtered — they are language-agnostic.

    Dedupe rule (Task 3.2, unchanged): when two RecognizerResults share the same
    span and entity type, keep the one with the higher score. On tie, keep the HR
    one (stable order: HR results come first in the concatenation).

    Returns:
        (merged_list, ner_spillover_dropped_count) — the count is logged by
        inlet as ner_spillover_dropped=%d (AC 3.3.13).
    """
    spillover_dropped = 0

    hr_filtered: list[RecognizerResult] = []
    for det in hr_results:
        if (
            det.entity_type in _NER_ENTITY_TYPES
            and _classify_window_language(text, det.start, det.end) != "hr"
        ):
            spillover_dropped += 1
            continue
        hr_filtered.append(det)

    en_filtered: list[RecognizerResult] = []
    for det in en_results:
        if (
            det.entity_type in _NER_ENTITY_TYPES
            and _classify_window_language(text, det.start, det.end) != "en"
        ):
            spillover_dropped += 1
            continue
        en_filtered.append(det)

    merged: dict[tuple[int, int, str], RecognizerResult] = {}
    for det in hr_filtered:
        key = (det.start, det.end, det.entity_type)
        merged[key] = det
    for det in en_filtered:
        key = (det.start, det.end, det.entity_type)
        existing = merged.get(key)
        if existing is None or det.score > existing.score:
            merged[key] = det

    return list(merged.values()), spillover_dropped


class Pipeline:
    """PII Filter pipeline — Keeper AI Gateway.

    Task 3: Presidio analyzer wired up with 12 custom recognizers + HR spaCy
    NLP. Inlet runs detection and attaches results to body metadata. Outlet
    is still pass-through (masking/restoration come in Tasks 4 + 6).
    """

    class Valves(BaseSettings):
        """Admin-configurable settings (visible in OpenWebUI Admin → Pipelines).

        Backed by `pydantic_settings.BaseSettings`: env vars prefixed with
        `PII_FILTER_` are auto-loaded, lowercased, stripped of the prefix,
        and coerced into the declared field types (e.g. `bool`, `int`,
        `Literal[...]`). Direct kwargs (used by tests) and admin-edited
        values both still work — env vars only fill in unset fields.
        """

        model_config = SettingsConfigDict(
            env_prefix="PII_FILTER_",
            case_sensitive=False,
            # Defensive: silently ignore stray env vars that share the
            # prefix but don't match a declared field, so an operator
            # typo never crashes Pipelines startup.
            extra="ignore",
            # Never auto-read .env files; only os.environ. Prevents
            # surprise loads from a developer .env that happens to be
            # in the Pipelines container working directory.
            env_file=None,
        )

        pipelines: list[str] = ["*"]
        priority: int = 0
        enabled: bool = True
        # ---- Task 8: Presidio detection kill switch ------------------------
        # Admin-level kill switch for the Presidio detection layer. When
        # False, `inlet` skips the analyzer + masking loop but still pulls
        # the vault snapshot so already-vaulted history placeholders remain
        # restorable by `outlet` (decision #4 / §2.1). Analyzers are still
        # instantiated by `on_startup` regardless of this flag (decision
        # #2 — runtime guard, not on_startup skip — so toggling does not
        # require a container restart). Use cases: incident response when
        # Presidio is crashing; "audit-only mode" where requests reach the
        # LLM unmodified while existing vault snapshots can still be
        # surfaced for `outlet` restoration symmetry. Configurable via the
        # `PII_FILTER_PRESIDIO_ENABLED` env var.
        presidio_enabled: bool = True
        languages: list[str] = Field(
            default_factory=lambda: ["hr", "en"],
            description=(
                "Active detection languages. Allowed values: 'hr', 'en'. "
                "Configurable via PII_FILTER_LANGUAGES env var "
                '(JSON array: \'["hr","en"]\' or comma-separated: "hr,en"). '
                "Validation occurs at on_startup; unsupported codes raise RuntimeError."
            ),
        )
        # Behavior when the analyzer (or the vault) fails mid-request.
        #   "block" (default) — fail-closed: raise so the request never
        #     reaches the LLM unfiltered. GDPR-safe; recommended for prod.
        #   "passthrough" — fail-open: log and let the request through
        #     without PII filtering. Use only if availability outweighs
        #     leak risk. Any unrecognized value is treated as "block".
        degradation_mode: str = "block"
        # ---- Vault configuration -------------------------------------------
        # Global vault kill switch. When False, the inlet always uses
        # Task 4's per-request dicts and never touches the configured
        # Postgres-backed vault. Default True.
        vault_enabled: bool = True
        # 24h. Renewed on every read or write touching the thread.
        thread_ttl_seconds: int = 86400
        # 10 min for chat_id-less ephemeral fallback threads.
        ephemeral_ttl_seconds: int = 600
        # Full Postgres DSN. Empty default means the operator must set
        # `PII_FILTER_POSTGRES_URL` for the Postgres backend to start —
        # `on_startup` raises if `vault_enabled=True` and this is "".
        # Cloud SQL pattern:
        #   "postgresql://user:pass@/db?host=/cloudsql/<INSTANCE_CONN_NAME>"
        postgres_url: str = ""
        # Connection pool sizing. min=2 keeps two warm connections post-
        # cold-start so the first request avoids TCP+TLS+auth handshake.
        # max=10 caps parallel DB ops; well under Cloud SQL's default
        # max_connections (~200 on db-custom-2-13312).
        postgres_pool_min: int = 2
        postgres_pool_max: int = 10
        # Per-query timeout (ms). Caps any single query at 5s — prevents
        # zombie connections from hanging the pool.
        postgres_command_timeout_ms: int = 5000
        # ---- Task 11: vault encryption-at-rest (Option E) ------------------
        # Application-layer AES-256-GCM encryption of `original_value` plus a
        # keyed HMAC blind index (`lookup_hash`) for dedup. The blind index is
        # ALWAYS computed when the vault runs (lookup_hash is NOT NULL and part
        # of the PK — spec D1), so the blind-index key is required regardless of
        # the flags below; `vault_encryption_enabled` only controls whether
        # `original_value` is stored as ciphertext (True) or plaintext (False).
        # Keys are validated fail-closed at on_startup (§6); no key material is
        # ever logged.
        #
        # When True, `original_value` is stored as an ENC1 envelope. Default
        # OFF; greenfield prod sets this (and `vault_encryption_strict`) True.
        vault_encryption_enabled: bool = False
        # When True, the read path refuses an unexpected plaintext row (one not
        # carrying the ENC1 envelope while encryption is enabled): it logs ERROR
        # and treats the row as a miss rather than serving raw PII. Recommended
        # True in prod (greenfield prod should never hold plaintext rows).
        vault_encryption_strict: bool = False
        # Key backend. "local" reads the base64 keys from the valve fields
        # below; "gcp_kms" lazy-loads google-cloud-secret-manager and fetches
        # the named secrets. Prod uses gcp_kms (handoff to Senka).
        vault_kms_backend: Literal["local", "gcp_kms"] = "local"
        # base64-encoded 32-byte AES-256 key (local backend). Required when
        # vault_encryption_enabled=True. Dev-only; prod uses Secret Manager.
        vault_encryption_key: str = ""
        # base64-encoded 32-byte HMAC key for the blind index (local backend).
        # ALWAYS required when the vault runs (spec D1). Dev-only.
        vault_blind_index_key: str = ""
        # Envelope key_id (u32), packed into the ENC1 header for a future
        # read-old/write-new key rotation. v1 ships a single key.
        vault_encryption_key_id: int = 1
        # gcp_kms backend: Secret Manager resource names whose payloads are the
        # base64 32-byte keys. Ignored for the local backend.
        vault_gcp_enc_secret: str = ""
        vault_gcp_blind_secret: str = ""
        # ---- Task 3.1: Recognizer accuracy --------------------------------
        # Case-insensitive denylist for PERSON entities. An entity is dropped
        # if its lowercased text exactly matches an entry OR starts with an
        # entry followed by a space (prefix + word-boundary rule). Suppresses
        # spaCy false positives on common English/code keywords that the
        # hr_core_news_lg model misclassifies as PERSON.
        ner_deny_list: list[str] = Field(
            default_factory=lambda: [
                "task",
                "tasks",
                "json",
                "json output",
                "json array",
                "json array of strings",
                "raw",
                "output",
                "input",
                "true",
                "false",
                "null",
                "none",
                "undefined",
                "default",
                "auto",
                "custom",
                "emoji",
                "emojis",
                "emoji summarizing",
                "emojis summarizing",
                "emoji summarizing the conversation",
                "emojis that enhance understanding",
                "summarize",
                "summarization",
                "summary",
                "assistant",
                "user",
                "system",
                "prompt",
                "response",
                "completion",
                "get",
                "post",
                "put",
                "delete",
                "patch",
                "request",
                "endpoint",
                "header",
                "body",
                "payload",
                "please",
                "thank you",
                "error",
                "warning",
                "info",
                "debug",
                "success",
                "failed",
                "pending",
                # Task 3.3: Croatian label words and pronoun phrases.
                # Window filter drops most EN-sourced spillover; deny-list
                # catches residual HR-NER noise (HR model detects these in
                # HR-dominant text, so they pass the window filter but should
                # never enter the vault as PII spans).
                "moj oib",
                "moja oib",
                "moj jmbg",
                "moja jmbg",
                "moj iban",
                "moja iban",
                "moj email",
                "moja email",
                "moja mail",
                "moj telefon",
                "moj mobitel",
                "moja adresa",
                "moj broj",
                "email",
                "mail",
                "adresa",
                "broj",
                "oib",
                "jmbg",
                "iban",
                "ime",
                "prezime",
            ],
            description=(
                "Lowercase, exact-match denylist for PERSON entities. "
                "Useful for suppressing spaCy false positives on common "
                "English/code keywords and Croatian label words. Configurable via "
                "PII_FILTER_NER_DENY_LIST env var."
            ),
        )
        # If a PERSON entity ends with one of these tokens, the trailing
        # token is stripped from the span before vault insertion.
        ner_trailing_token_strip: list[str] = Field(
            default_factory=lambda: [
                # English function words
                "has",
                "had",
                "have",
                "is",
                "was",
                "are",
                "were",
                "be",
                "been",
                "being",
                "does",
                "did",
                "do",
                "says",
                "said",
                "say",
                "goes",
                "went",
                "go",
                "comes",
                "came",
                "come",
                # Croatian function words
                "je",
                "su",
                "ima",
                "imaju",
                "bio",
                "bila",
                "bile",
                "bili",
                "kaže",
                "rekao",
                "rekla",
                "rekli",
                "ide",
                "došao",
                "došla",
            ],
            description=(
                "If a PERSON entity ends with one of these tokens, the "
                "trailing token is stripped from the entity span. "
                "Configurable via PII_FILTER_NER_TRAILING_TOKEN_STRIP."
            ),
        )
        # Character window before an 11-digit number to check for phone
        # context keywords. Set to 0 to disable the OIB context check.
        ner_oib_phone_context_window: int = Field(
            default=30,
            ge=0,
            le=200,
            description=(
                "Character window before an 11-digit number to check for "
                "phone-context keywords. If found (and no OIB context word "
                "present), the HR_OIB detection is rejected as a likely "
                "phone number. Set 0 to disable. "
                "Configurable via PII_FILTER_NER_OIB_PHONE_CONTEXT_WINDOW."
            ),
        )
        # ---- Task 8.5: Multi-turn history scope ----------------------------
        multi_turn_history_scope: bool = Field(
            default=True,
            description=(
                "When True (default), inlet masks ALL user messages in "
                "body.messages[], not just the last one. Prevents PII from "
                "prior turns leaking to the LLM vendor. When False, reverts "
                "to Task 4 behavior (only last user message). Disable only "
                "for debugging or compat with single-turn flows."
            ),
        )
        multi_turn_history_max_messages: int = Field(
            default=20,
            ge=0,
            le=100,
            description=(
                "Safety cap on how many user messages (from the tail) are "
                "processed by inlet per request. Older user messages are "
                "passed through unchanged. Set to 0 to disable history "
                "processing (equivalent to disabling multi_turn_history_scope)."
            ),
        )
        multi_turn_already_masked_pattern: str = Field(
            default=r"\[[A-Z_]+_\d+\]",
            description=(
                "Regex pattern to detect messages already masked from a prior "
                "inlet call. Messages matching this pattern are skipped without "
                "calling Presidio (performance optimization)."
            ),
        )

    class UserValves(BaseModel):
        """Per-user toggles. Schema only — Task 8 wires the masking toggle.

        OpenWebUI Pipelines container has open issue #19179 around UserValves
        propagation; we publish the schema now so Task 8 can flip the switch
        without a separate schema migration.
        """

        # Ignore unknown keys so a UI/Pipelines version that ships extra
        # fields cannot break per-request resolution at the inlet boundary.
        model_config = ConfigDict(extra="ignore")

        pii_masking_enabled: bool = True

    # Whitelist mapping: only entities in this dict are forwarded downstream;
    # everything else is dropped. Keys are raw Presidio entity types, values
    # are the Keeper-standardized type names used in metadata + masking.
    #
    # NOTE on LOCATION: spaCy NER + Presidio's built-in LocationRecognizer
    # emit LOCATION for country/city names (e.g. "Hrvatska", "Njemačka").
    # These are NOT addresses and masking them destroys LLM context. Real
    # ADDRESS detection (street + number + postal code) is Task 10 scope —
    # it will land its own canonical type then. Until then, LOCATION is
    # intentionally absent from this whitelist so any LOCATION detection is
    # silently dropped before masking, the same way unmapped types already are.
    PRESIDIO_TO_STANDARD: ClassVar[dict[str, str]] = {
        "PERSON": "PERSON",
        "EMAIL_ADDRESS": "EMAIL",
        "PHONE_NUMBER": "PHONE",
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
        "UK_NHS": "UK_NHS",
        "IBAN_CODE": "IBAN_CODE",
    }

    def __init__(self) -> None:
        """Initialize the pipeline.

        Heavy work (AnalyzerEngine, spaCy load, recognizer registration) is
        deferred to `on_startup` per Pipelines lifecycle: `__init__` runs at
        import, `on_startup` runs when the pipeline is enabled.
        """
        self.type = "filter"
        self.name = "PII Filter"
        # `Valves` is a `pydantic_settings.BaseSettings` subclass with
        # `env_prefix="PII_FILTER_"` — it reads os.environ itself and
        # coerces to the declared field types. No manual env-var plumbing
        # needed here.
        self.valves = self.Valves()
        self.user_valves = self.UserValves()
        self.analyzer_hr: AnalyzerEngine | None = None
        self.analyzer_en: AnalyzerEngine | None = None
        # The vault is built in `on_startup` from the current valves so
        # admin-edited vault settings take effect on Pipelines restart.
        self.vault: ThreadVault | None = None

        logger.info("PII Filter pipeline initialized (analyzer not loaded yet)")

    def _build_analyzer(self, lang_code: str) -> AnalyzerEngine:
        """Build a single-language AnalyzerEngine with all relevant recognizers.

        Custom recognizers are registered for both 'hr' and 'en' (duplicated
        per spec §2 Q5) so that cross-language entities (e.g. a Croatian OIB
        mentioned in an English sentence) are caught regardless of which
        analyzer runs first.

        EN-only built-ins (NhsRecognizer, UsSsnRecognizer, EmailRecognizer,
        PhoneRecognizer, IbanRecognizer) are auto-registered by Presidio for
        'en' and remain active. For 'hr', only language-neutral built-ins
        (Email, Phone, Crypto, Date, etc.) are auto-registered; IbanRecognizer
        is disabled there to prevent conflicts with our custom HRIBANRecognizer.
        """
        model_name = {"hr": "hr_core_news_lg", "en": "en_core_web_lg"}[lang_code]
        global _nlp_engine_cache
        if lang_code not in _nlp_engine_cache:
            try:
                _nlp_engine_cache[lang_code] = NlpEngineProvider(
                    nlp_configuration={
                        "nlp_engine_name": "spacy",
                        "models": [{"lang_code": lang_code, "model_name": model_name}],
                        "ner_model_configuration": {
                            # Drop MISC and catch-all "O" at the spaCy NER stage
                            # so they never reach Presidio's entity mapper.
                            "labels_to_ignore": ["MISC", "O"],
                        },
                    }
                ).create_engine()
            except Exception as exc:
                logger.error(
                    "Failed to load spaCy %r NLP engine: %s. "
                    "Install via the wheel URL in requirements or run: "
                    "python -m spacy download %s",
                    model_name,
                    exc,
                    model_name,
                )
                raise RuntimeError(f"Required spaCy model {model_name!r} is unavailable") from exc

        nlp_engine = _nlp_engine_cache[lang_code]

        analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=[lang_code],
        )

        disabled = (
            _DISABLED_BUILTIN_RECOGNIZERS_HR
            if lang_code == "hr"
            else _DISABLED_BUILTIN_RECOGNIZERS_EN
        )
        for rec_name in disabled:
            try:
                analyzer.registry.remove_recognizer(rec_name)
            except Exception as exc:
                logger.warning(
                    "Could not remove built-in recognizer %r from %s registry: %s",
                    rec_name,
                    lang_code,
                    exc,
                )

        # All 12 custom recognizers duplicated in both registries (spec §2 Q5).
        analyzer.registry.add_recognizer(OIBRecognizer(supported_language=lang_code))
        analyzer.registry.add_recognizer(JMBGRecognizer(supported_language=lang_code))
        analyzer.registry.add_recognizer(HRIBANRecognizer(supported_language=lang_code))
        analyzer.registry.add_recognizer(IEPPSNRecognizer(supported_language=lang_code))
        analyzer.registry.add_recognizer(ROCNPRecognizer(supported_language=lang_code))
        analyzer.registry.add_recognizer(UKNINORecognizer(supported_language=lang_code))
        analyzer.registry.add_recognizer(UKUTRRecognizer(supported_language=lang_code))
        analyzer.registry.add_recognizer(USSSNRecognizer(supported_language=lang_code))
        analyzer.registry.add_recognizer(USEINRecognizer(supported_language=lang_code))
        analyzer.registry.add_recognizer(
            make_iban_recognizer("IE", 18, "IE_IBAN", supported_language=lang_code)
        )
        analyzer.registry.add_recognizer(
            make_iban_recognizer("RO", 20, "RO_IBAN", supported_language=lang_code)
        )
        analyzer.registry.add_recognizer(
            make_iban_recognizer("GB", 18, "GB_IBAN", supported_language=lang_code)
        )
        # CreditCardRecognizer is auto-registered only for 'en'; add it
        # explicitly for 'hr' so credit card detection works in HR-only mode.
        if lang_code == "hr":
            analyzer.registry.add_recognizer(CreditCardRecognizer(supported_language=lang_code))

        analyzer.analyze(text="warmup", language=lang_code, entities=None)
        return analyzer

    def _build_vault_crypto(self) -> tuple[VaultCipher | None, BlindIndex]:
        """Build the vault's blind index (always) and cipher (when encryption
        is enabled), validating keys fail-closed per spec §6.

        The blind-index key is required whenever the vault runs (lookup_hash is
        NOT NULL and part of the PK — spec D1); the encryption key is required
        only when ``vault_encryption_enabled``. ``extra="ignore"`` on the Valves
        means a typo'd env var silently stays empty, so an empty / non-base64 /
        wrong-length key raises ``RuntimeError`` here — analogous to the
        existing ``postgres_url`` guard. Logs one INFO line (backend, encryption
        on/off, strict on/off, key_id) with NO key material.
        """
        valves = self.valves
        key_manager = KeyManager(
            backend=valves.vault_kms_backend,
            encryption_key_b64=valves.vault_encryption_key,
            blind_index_key_b64=valves.vault_blind_index_key,
            gcp_enc_secret=valves.vault_gcp_enc_secret,
            gcp_blind_secret=valves.vault_gcp_blind_secret,
        )
        # Blind index key is always required (spec D1).
        blind_index = BlindIndex(key_manager.load_blind_index_key())
        cipher: VaultCipher | None = None
        if valves.vault_encryption_enabled:
            cipher = VaultCipher(
                key_manager.load_encryption_key(), key_id=valves.vault_encryption_key_id
            )
        logger.info(
            "PII Filter vault crypto ready: backend=%s encryption=%s strict=%s key_id=%d",
            valves.vault_kms_backend,
            "on" if cipher is not None else "off",
            "on" if valves.vault_encryption_strict else "off",
            valves.vault_encryption_key_id,
        )
        return cipher, blind_index

    async def on_startup(self) -> None:
        """Validate language config, build per-language AnalyzerEngines, wire vault."""
        languages = self.valves.languages
        if not languages:
            raise RuntimeError(
                "valves.languages is empty; at least one of 'hr' or 'en' is required."
            )
        invalid = [lang for lang in languages if lang not in {"hr", "en"}]
        if invalid:
            raise RuntimeError(
                f"Unsupported language(s): {invalid}. Allowed: hr, en. "
                "Multi-language detection in Task 3.2 supports only Croatian and English."
            )

        logger.info("PII Filter on_startup: building analyzers for languages=%s", languages)

        self.analyzer_hr = self._build_analyzer("hr") if "hr" in languages else None
        self.analyzer_en = self._build_analyzer("en") if "en" in languages else None

        logger.info(
            "PII Filter on_startup: analyzers ready (hr=%s, en=%s)",
            self.analyzer_hr is not None,
            self.analyzer_en is not None,
        )

        # Vault initialization (Postgres-only). `initialize()` opens the
        # connection pool and runs idempotent DDL — failure here is fatal
        # and the container fails to start (explicit failure beats silent
        # fallback per spec §3.7). With `vault_enabled=False`, `self.vault`
        # stays None and the inlet falls back to Task 4's per-request dicts.
        if self.valves.vault_enabled:
            if not self.valves.postgres_url:
                raise RuntimeError(
                    "PII_FILTER_POSTGRES_URL must be set when vault_enabled=True. "
                    "Set the env var (or the valves.postgres_url admin setting) "
                    "to a valid DSN."
                )
            # Task 11: build + validate vault crypto BEFORE opening the pool so
            # a missing/short/invalid key fails closed at startup without any
            # DB I/O (spec §6).
            cipher, blind_index = self._build_vault_crypto()
            self.vault = ThreadVault(
                dsn=self.valves.postgres_url,
                pool_min=self.valves.postgres_pool_min,
                pool_max=self.valves.postgres_pool_max,
                command_timeout=self.valves.postgres_command_timeout_ms / 1000.0,
                thread_ttl_seconds=self.valves.thread_ttl_seconds,
                ephemeral_ttl_seconds=self.valves.ephemeral_ttl_seconds,
                cipher=cipher,
                blind_index=blind_index,
                encryption_strict=self.valves.vault_encryption_strict,
            )
            await self.vault.initialize()
            healthy = await self.vault.healthcheck()
            logger.info(
                "PII Filter on_startup complete: ThreadVault wired "
                "(pool_min=%d, pool_max=%d, command_timeout_ms=%d, healthy=%s)",
                self.valves.postgres_pool_min,
                self.valves.postgres_pool_max,
                self.valves.postgres_command_timeout_ms,
                healthy,
            )
            if not healthy:
                logger.warning(
                    "Vault healthcheck failed at startup. The inlet will hit "
                    "its degradation_mode path on first use."
                )
        else:
            self.vault = None
            logger.info(
                "PII Filter on_startup complete: vault disabled "
                "(vault_enabled=False; running in per-request mode)"
            )

    async def on_shutdown(self) -> None:
        """Called when Pipelines container stops."""
        logger.info("PII Filter on_shutdown")
        self.analyzer_hr = None
        self.analyzer_en = None
        if self.vault is not None:
            await self.vault.aclose()
            self.vault = None

    @staticmethod
    def _is_single_text_part(msg: dict[str, Any]) -> bool:
        """True when `msg` carries exactly one text segment.

        Used by the PII card builder: per-detection offsets are relative to a
        single text string, so they only align with the frontend's flat
        message string when the message has one text part (plain string, or a
        multimodal list with exactly one `{"type": "text"}` part). With two or
        more text parts the offsets would collide on the flattened string.
        """
        content = msg.get("content")
        if isinstance(content, str):
            return True
        if isinstance(content, list):
            return sum(
                1
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ) == 1
        return False

    def _iter_text_parts(
        self, message: dict[str, Any]
    ) -> Iterator[tuple[str, Callable[[str], None]]]:
        """Yield `(text, write_back)` pairs for each text segment in `message`.

        Used by both inlet (writeback masks PII) and outlet (writeback
        restores originals). Handles both content shapes accepted by the
        OpenAI chat-completion API:
          * `content` is a `str` — yields one pair; write_back replaces the
            whole `message["content"]` value.
          * `content` is a `list[dict]` (multi-modal) — yields one pair per
            `{"type": "text", ...}` part; write_back updates that part's
            `text` field. Non-text parts (image_url, file, etc.) are skipped.

        Empty / whitespace-only / non-string text segments are skipped so
        callers never operate on uninteresting input.
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

    @staticmethod
    def _resolve_chat_id(body: dict[str, Any]) -> tuple[str | None, str]:
        """Pull `chat_id` from body (top-level then metadata) and pick the
        thread_id used by the vault. Returns `(raw_chat_id, thread_id)`:

        * `raw_chat_id` is the original `chat_id` from the body, or `None`
          when the request didn't supply one.
        * `thread_id` is what's used for vault key building — equal to
          `raw_chat_id` when present, otherwise a fresh ephemeral id.
        """
        raw = body.get("chat_id")
        if not raw:
            metadata = body.get("metadata")
            if isinstance(metadata, dict):
                raw = metadata.get("chat_id")
        if isinstance(raw, str) and raw:
            return raw, raw
        return None, make_ephemeral_thread_id()

    def _resolve_user_valves(self, user: dict[str, Any] | None) -> Pipeline.UserValves:
        """Build the effective UserValves for this request.

        OpenWebUI injects per-user valve overrides under ``user["valves"]``.
        When that payload is present we instantiate a fresh ``UserValves``
        from it so the UI toggle actually takes effect per request. When
        it's absent (older Pipelines container, issue #19179, or tests
        that mutate ``self.user_valves`` directly) we fall back to the
        instance-level default. Malformed payloads are logged and treated
        as "no override" rather than failing the request — masking-on is
        the safe default.
        """
        raw = (user or {}).get("valves") if user else None
        if isinstance(raw, dict):
            try:
                return self.UserValves(**raw)
            except ValidationError as exc:
                logger.warning(
                    "pii_filter: malformed user.valves payload, "
                    "falling back to default (err=%s)",
                    exc,
                )
        return self.user_valves

    async def inlet(
        self, body: dict[str, Any], user: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Detect PII in user messages, mask in place via the thread-scoped
        vault (Task 5/5.1), and stash forward + reverse maps in
        `body["metadata"]` so the outlet (Task 6) keeps reading the same
        keys regardless of backend.

        Task 8.5: by default (`multi_turn_history_scope=True`), ALL user
        messages in `body["messages"]` are processed (up to
        `multi_turn_history_max_messages`), not just the last one.  Already-
        masked messages (matching `multi_turn_already_masked_pattern`) are
        skipped via a cheap regex pre-check to avoid redundant Presidio calls.
        Assistant messages pass through unchanged.

        Mutates each processed message's `content` field in place. All other
        body keys are left untouched. On analyzer / vault failure, behavior
        follows `valves.degradation_mode` (block → raise; passthrough → return).
        """
        if not self.valves.enabled:
            return body

        # ---- Task 8: UserValves per-user toggle (early return) ------------
        # If the user has opted out via `UserValves.pii_masking_enabled=False`,
        # short-circuit the entire inlet pipeline: no vault snapshot, no
        # analyzer, no metadata writes. `outlet` stays ungated (decision #3),
        # so any pre-existing vault placeholders from prior turns are still
        # restored if/when the user re-enables the toggle. Audit emitted as
        # a single INFO line per request (decision #5).
        #
        # Resolve per-request: OpenWebUI ships the toggle under
        # `user["valves"]`. Falling back to `self.user_valves` keeps
        # backwards compatibility with Pipelines container versions that
        # don't propagate the payload (issue #19179).
        effective_user_valves = self._resolve_user_valves(user)
        if not effective_user_valves.pii_masking_enabled:
            user_id = (user or {}).get("id", "unknown")
            raw_chat_id_user, _ = self._resolve_chat_id(body)
            logger.info(
                "pii_filter inlet user_disabled: user_id=%s chat_id=%s",
                user_id,
                raw_chat_id_user,
            )
            return body

        # ---- Task 8: Admin Presidio kill switch (audit-only mode) ---------
        # If the admin has disabled Presidio via `Valves.presidio_enabled=False`,
        # skip the analyzer + masking loop but still pull the vault snapshot
        # so `outlet` can restore history placeholders that were vaulted
        # before the flip (decision #4 / §2.1). Body mutation is limited to
        # `metadata.pii_placeholder_map` / `pii_reverse_map`, mirroring the
        # normal path's outlet contract. Vault snapshot failure is logged
        # and swallowed — the request still reaches the LLM (we are already
        # in a degraded mode; blocking here would defeat the purpose of an
        # admin kill switch).
        if not self.valves.presidio_enabled:
            raw_chat_id_pd, thread_id_pd = self._resolve_chat_id(body)
            logger.info(
                "pii_filter inlet presidio_disabled: chat_id=%s",
                raw_chat_id_pd,
            )
            forward_map_pd: dict[str, str] = {}
            reverse_map_pd: dict[str, str] = {}
            vault_state_pd = "off"
            if self.vault is not None and self.valves.vault_enabled:
                try:
                    forward_map_pd, reverse_map_pd = await self.vault.snapshot_for_request(
                        thread_id_pd
                    )
                    vault_state_pd = "ephemeral" if raw_chat_id_pd is None else "on"
                except Exception:
                    logger.exception(
                        "pii_filter inlet presidio_disabled: vault snapshot "
                        "failed for thread_id=%s chat_id=%s — proceeding with empty maps",
                        thread_id_pd,
                        raw_chat_id_pd,
                    )
            metadata_pd = body.get("metadata")
            if not isinstance(metadata_pd, dict):
                metadata_pd = {}
                body["metadata"] = metadata_pd
            metadata_pd["pii_placeholder_map"] = forward_map_pd
            metadata_pd["pii_reverse_map"] = reverse_map_pd
            languages_active_pd = ",".join(
                lang
                for lang, an in (("hr", self.analyzer_hr), ("en", self.analyzer_en))
                if an is not None
            )
            logger.info(
                "pii_filter inlet processed: chat_id=%s thread_id=%s "
                "messages_processed=0 messages_skipped_already_masked=0 "
                "detections=0 masked=0 vault=%s languages_active=%s "
                "ner_spillover_dropped=0 user_masking_disabled=False "
                "presidio_disabled=True",
                raw_chat_id_pd,
                thread_id_pd,
                vault_state_pd,
                languages_active_pd,
            )
            return body

        # Task 3.2/3.3 OpenWebUI integration: skip background tasks.
        # OpenWebUI sends embedded chat history as user content for
        # title/tags/follow-up generation, which would produce false-positive
        # PII detections on assistant content inside the embedded history.
        # Background tasks are identified by metadata["task"] key
        # (e.g., "title_generation", "tags_generation", "follow_up_generation").
        # The original user message has already been processed by the primary
        # inlet call; outlet will still restore placeholders in task responses.
        metadata = body.get("metadata")
        if isinstance(metadata, dict):
            task = metadata.get("task")
            if task:
                logger.info(
                    "inlet: skipping OpenWebUI background task '%s' "
                    "(chat_id=%s, body_size=%d chars)",
                    task,
                    metadata.get("chat_id"),
                    sum(
                        len(str(m.get("content", "")))
                        for m in body.get("messages", [])
                        if isinstance(m, dict)
                    )
                    if isinstance(body.get("messages"), list)
                    else 0,
                )
                return body

        if self.analyzer_hr is None and self.analyzer_en is None:
            logger.warning("inlet called before on_startup completed; returning body unchanged")
            return body

        # Task 8.5: determine which user messages to process.
        # Legacy Task 4 path (scope=False or cap=0): only the last user message.
        # Multi-turn path (default): all user messages up to cap, oldest-first.
        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            logger.warning("inlet: no messages in body, skipping analysis")
            return body

        target_indices: list[int]
        if (
            not self.valves.multi_turn_history_scope
            or self.valves.multi_turn_history_max_messages == 0
        ):
            last = _find_last_user_index(messages)
            if last == -1:
                logger.debug("inlet: no user message found, skipping analysis")
                return body
            target_indices = [last]
        else:
            all_user_indices = [
                i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "user"
            ]
            if not all_user_indices:
                logger.debug("inlet: no user messages found, skipping analysis")
                return body
            cap = self.valves.multi_turn_history_max_messages
            target_indices = all_user_indices[-cap:]

        raw_chat_id, thread_id = self._resolve_chat_id(body)
        if raw_chat_id is None:
            logger.warning("inlet: chat_id missing, using ephemeral thread_id=%s", thread_id)

        # Decide whether to use the vault or fall back to per-request dicts.
        # Vault path requires `vault_enabled=True`, a vault instance, and
        # a healthy backend (SELECT 1 for Postgres). On any
        # of those failing in `block` mode we raise so the request never
        # reaches the LLM unfiltered.
        use_vault = self.valves.vault_enabled and self.vault is not None
        if use_vault:
            assert self.vault is not None  # for mypy
            try:
                healthy = await self.vault.healthcheck()
            except Exception:
                logger.exception("inlet: vault healthcheck raised; treating as unhealthy")
                healthy = False
            if not healthy:
                if self.valves.degradation_mode != "passthrough":
                    raise RuntimeError(
                        "PII filter blocked the request: vault is "
                        "unavailable and degradation_mode='block'. Set "
                        "valves.degradation_mode='passthrough' to fall back to "
                        "per-request scope on vault outages (NOT recommended "
                        "in production)."
                    )
                logger.warning(
                    "Vault unavailable, falling back to per-request scope. "
                    "Thread consistency disabled for chat_id=%s",
                    raw_chat_id,
                )
                use_vault = False

        # Per-request mapping state. Used in the fallback path; in the vault
        # path we read the snapshot back from the vault at the end.
        counter_state: dict[str, int] = {}
        forward_map: dict[str, str] = {}
        reverse_map: dict[str, str] = {}
        all_enriched: list[dict[str, Any]] = []
        total_spillover_dropped: int = 0

        if use_vault:
            assert self.vault is not None  # for mypy
            try:
                await self.vault.get_or_create_thread(thread_id)
            except Exception:
                logger.exception("inlet: vault get_or_create_thread raised")
                if self.valves.degradation_mode != "passthrough":
                    raise RuntimeError(
                        "PII filter blocked the request: vault is "
                        "unreachable and degradation_mode='block'."
                    ) from None
                use_vault = False

        deny_set = frozenset(s.lower() for s in self.valves.ner_deny_list)
        trailing_set = frozenset(s.lower() for s in self.valves.ner_trailing_token_strip)
        # Per-call compile: respects operator changes to multi_turn_already_masked_pattern
        # via valve without requiring a restart.
        _default_masked_pattern = r"\[[A-Z_]+_\d+\]"
        try:
            already_masked_re = re.compile(self.valves.multi_turn_already_masked_pattern)
        except re.error:
            logger.error(
                "inlet: invalid multi_turn_already_masked_pattern %r — falling back to default",
                self.valves.multi_turn_already_masked_pattern,
            )
            already_masked_re = re.compile(_default_masked_pattern)

        messages_processed = 0
        messages_skipped = 0

        try:
            for msg_idx in target_indices:
                target_msg = messages[msg_idx]

                # Regex pre-check: skip messages already containing placeholders
                # (from a prior inlet call). This avoids a redundant Presidio
                # analyzer call — the dominant latency source — for history
                # messages that were masked in an earlier turn.
                full_content = _get_message_text_for_precheck(target_msg)
                if already_masked_re.search(full_content):
                    messages_skipped += 1
                    continue

                parts = list(self._iter_text_parts(target_msg))
                if not parts:
                    continue

                for text, write_back in parts:
                    hr_results: list[RecognizerResult] = (
                        self.analyzer_hr.analyze(text=text, language="hr")
                        if self.analyzer_hr is not None
                        else []
                    )
                    en_results: list[RecognizerResult] = (
                        self.analyzer_en.analyze(text=text, language="en")
                        if self.analyzer_en is not None
                        else []
                    )
                    results, spillover_count = _merge_dedupe_detections(
                        hr_results, en_results, text
                    )
                    total_spillover_dropped += spillover_count
                    accepted = _select_accepted_detections(
                        text,
                        results,
                        self.PRESIDIO_TO_STANDARD,
                        deny_set,
                        trailing_set,
                        self.valves.ner_oib_phone_context_window,
                    )
                    if not accepted:
                        continue

                    pieces: list[str] = []
                    last_end = 0
                    for det in accepted:
                        original = text[det.start : det.end]
                        standard_type = self.PRESIDIO_TO_STANDARD[det.entity_type]
                        placeholder: str
                        if use_vault:
                            assert self.vault is not None  # for mypy
                            placeholder = await self.vault.get_placeholder(
                                thread_id, original, standard_type
                            )
                        else:
                            existing = forward_map.get(original)
                            if existing is None:
                                n = counter_state.get(standard_type, 0) + 1
                                counter_state[standard_type] = n
                                placeholder = f"[{standard_type}_{n}]"
                                forward_map[original] = placeholder
                                reverse_map[placeholder] = original
                            else:
                                placeholder = existing
                        pieces.append(text[last_end : det.start])
                        pieces.append(placeholder)
                        last_end = det.end
                        enriched_det = _build_enriched_detection(
                            det, text, standard_type, original, placeholder
                        )
                        enriched_det["message_index"] = msg_idx
                        all_enriched.append(enriched_det)
                    pieces.append(text[last_end:])
                    write_back("".join(pieces))

                messages_processed += 1

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

        if messages_processed == 0 and messages_skipped == 0:
            logger.debug("inlet: no user messages with maskable content, skipping metadata update")
            return body

        # Spec §3.3 step 7 — body-metadata snapshot is the forward-compat hinge
        # for Task 6: outlet keeps reading from these keys regardless of vault
        # being the source of truth.
        if use_vault:
            assert self.vault is not None  # for mypy
            try:
                forward_map, reverse_map = await self.vault.snapshot_for_request(thread_id)
            except Exception:
                logger.exception("inlet: vault snapshot_for_request raised")
                if self.valves.degradation_mode != "passthrough":
                    raise RuntimeError(
                        "PII filter blocked the request: vault is "
                        "unreachable and degradation_mode='block'."
                    ) from None
                # Passthrough: masking already completed against vault, but we
                # cannot rebuild the request-scoped maps. Leave them empty so
                # the request still reaches the LLM; outlet restoration for
                # this turn will be a no-op.
                use_vault = False

        metadata = body.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            body["metadata"] = metadata
        metadata["pii_detections"] = all_enriched
        metadata["pii_placeholder_map"] = forward_map
        metadata["pii_reverse_map"] = reverse_map

        # --- PII card: safe-to-expose subset (UI only) ---
        # Slim, PII-free mirror of the last user message's detections for the
        # per-message "PII masked" card. ONLY {type, start, end} — never
        # `original`, `placeholder`, or the maps. The plaintext-bearing
        # `pii_detections` list never leaves the pipeline (defense-in-depth);
        # the frontend reconstructs values from the user's own message text.
        # Only the *current* user turn gets a card: the last message must itself
        # be a user message. On regeneration (trailing assistant/non-user
        # message) we emit nothing, so the card never attaches to a stale prior
        # turn whose offsets no longer match what the frontend renders.
        # `messages` is guaranteed non-empty here (guarded at the top of inlet).
        last_user_idx = (
            len(messages) - 1
            if isinstance(messages[-1], dict) and messages[-1].get("role") == "user"
            else None
        )
        emit_card = last_user_idx is not None and self._is_single_text_part(
            messages[last_user_idx]
        )
        metadata["pii_detections_public"] = (
            [
                {"type": d["entity_type"], "start": d["start"], "end": d["end"]}
                for d in all_enriched
                if d.get("message_index") == last_user_idx
            ]
            if emit_card
            else []
        )
        logger.debug(
            "pii card: %d public detections, msg idx=%s",
            len(metadata["pii_detections_public"]),
            last_user_idx,
        )

        vault_state = ("ephemeral" if raw_chat_id is None else "on") if use_vault else "off"
        languages_active = ",".join(
            lang
            for lang, an in (("hr", self.analyzer_hr), ("en", self.analyzer_en))
            if an is not None
        )
        logger.info(
            "pii_filter inlet processed: chat_id=%s thread_id=%s "
            "messages_processed=%d messages_skipped_already_masked=%d "
            "detections=%d masked=%d vault=%s languages_active=%s "
            "ner_spillover_dropped=%d user_masking_disabled=False "
            "presidio_disabled=False",
            raw_chat_id,
            thread_id,
            messages_processed,
            messages_skipped,
            len(all_enriched),
            len(forward_map),
            vault_state,
            languages_active,
            total_spillover_dropped,
        )

        return body

    def _extract_assistant_message(self, body: Any) -> dict[str, Any] | None:
        """Return the assistant message dict if `body` matches a non-streaming
        completion shape, else None (and log at DEBUG with the specific
        failure reason).

        Preference order:
          1. ``body["choices"][0]["message"]`` — OpenAI chat-completion shape
             (production path).
          2. ``body["messages"][-1]`` when ``role == "assistant"`` — legacy /
             test fixture fallback. May be removed once production traffic is
             verified.

        A streaming chunk shape (``choices[0]["delta"]``) and a tool-calling
        response (``message.content is None``) both fail the guard and skip
        restoration — outlet is non-streaming-only by Task 6 design (Task 7
        owns the streaming path).
        """
        if not isinstance(body, dict):
            logger.debug("pii_filter outlet skipped: body is not a dict")
            return None

        # Preferred shape: OpenAI chat-completion `choices[0].message`.
        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if not isinstance(first, dict):
                logger.debug("pii_filter outlet skipped: choices[0] is not a dict")
                return None
            message = first.get("message")
            if not isinstance(message, dict):
                if isinstance(first.get("delta"), dict):
                    logger.debug(
                        "pii_filter outlet skipped: choices[0] has delta (streaming chunk)"
                    )
                else:
                    logger.debug(
                        "pii_filter outlet skipped: choices[0].message is not a dict "
                        "(likely tool_calls or non-completion shape)"
                    )
                return None
            content = message.get("content")
            if isinstance(content, str):
                if not content:
                    logger.debug("pii_filter outlet skipped: message.content is empty string")
                    return None
                return message
            if isinstance(content, list):
                if not content:
                    logger.debug("pii_filter outlet skipped: message.content is empty list")
                    return None
                return message
            logger.debug(
                "pii_filter outlet skipped: message.content is not str|list "
                "(got %s, likely None / tool_calls response)",
                type(content).__name__,
            )
            return None

        # Fallback shape: legacy/test bodies that carry the assistant turn at
        # `body["messages"][-1]`. Documented as legacy support; may be removed
        # once production traffic is verified to always use the OpenAI shape.
        messages = body.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            if not isinstance(last, dict):
                logger.debug("pii_filter outlet skipped: messages[-1] is not a dict")
                return None
            if last.get("role") != "assistant":
                logger.debug("pii_filter outlet skipped: messages[-1].role is not assistant")
                return None
            content = last.get("content")
            if isinstance(content, str):
                if not content:
                    logger.debug("pii_filter outlet skipped: messages[-1].content is empty string")
                    return None
                return last
            if isinstance(content, list):
                if not content:
                    logger.debug("pii_filter outlet skipped: messages[-1].content is empty list")
                    return None
                return last
            logger.debug(
                "pii_filter outlet skipped: messages[-1].content is not str|list (got %s)",
                type(content).__name__,
            )
            return None

        logger.debug("pii_filter outlet skipped: body has neither choices nor messages")
        return None

    async def outlet(
        self, body: dict[str, Any], user: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Restore PII placeholders in the assistant response back to their
        original values before forwarding the body to the OpenWebUI client.

        Source-of-truth selection (post-merge bugfix #2, 2026-05-08): outlet
        prefers `body["metadata"]["pii_reverse_map"]` when populated by inlet
        (Task 4) — this preserves backward compatibility for unit tests and
        any caller that propagates metadata. When metadata is missing or
        empty, outlet falls back to a direct read of the configured vault
        via `snapshot_for_request(chat_id)`. The fallback is required by
        real-world OpenWebUI Pipelines integration: inlet and outlet are
        served as separate HTTP endpoints with independent request bodies,
        so the metadata populated by inlet does NOT survive the round-trip
        to outlet (revising the original Task 6 Decision 1 assumption that
        outlet never reads from the vault).

        Outlet **never raises** (Decision 5): the entire restoration path is
        wrapped in a top-level two-tier `try/except`. The first arm catches
        the *expected* fail-safe family (``KeyError``/``AttributeError``/
        ``TypeError`` from a malformed body that slipped past the defensive
        shape guard) and logs at WARN — these are operational, not bugs.
        The second arm catches anything else and logs at ERROR with
        ``exc_info=True`` so the full traceback reaches the log sink — these
        are programmer errors that need to be triaged. Either way the body
        is returned unchanged because a partially restored or placeholder-
        leaking response is strictly better UX than a 500 from the
        Pipelines container.

        Hallucinated placeholders (regex match without a reverse_map entry)
        are left literally in the response and surfaced as a single
        deduplicated WARN line per outlet call (epic AC: zero hallucinated
        restorations — we never substitute a placeholder we cannot prove
        originated from inlet).
        """
        # Task 8 (v0.9.3) deliberately leaves `outlet` ungated by
        # `UserValves.pii_masking_enabled` (decision #3, AC 8.5). Rationale:
        # outlet is a pure restoration utility — a user who flips masking
        # off mid-conversation still expects previously-vaulted PII to be
        # restored in their history view. Gating outlet would also break
        # the "audit-only mode" semantics of `Valves.presidio_enabled=False`
        # (decision #4), which relies on outlet restoring snapshots that
        # inlet pulled despite skipping detection.

        if not self.valves.enabled:
            return body

        chat_id_for_log: Any = "unknown"
        try:
            if not isinstance(body, dict):
                logger.debug("pii_filter outlet skipped: body is not a dict")
                return body

            raw_chat_id, thread_id = self._resolve_chat_id(body)
            chat_id_for_log = raw_chat_id  # may be None when no chat_id supplied

            # Prefer body metadata (unit tests + callers that propagate it),
            # then fall back to a direct vault read (real OpenWebUI Pipelines,
            # which serves outlet as a separate HTTP endpoint without metadata
            # propagation — see post-merge bugfix #2 in TASK-05.1-COMPLETION).
            reverse_map: dict[str, str] | None = None
            metadata = body.get("metadata")
            if isinstance(metadata, dict):
                candidate = metadata.get("pii_reverse_map")
                if isinstance(candidate, dict) and candidate:
                    reverse_map = candidate

            if reverse_map is None:
                if self.vault is not None and self.valves.vault_enabled and raw_chat_id:
                    try:
                        _forward, vault_reverse = await self.vault.snapshot_for_request(thread_id)
                    except Exception:
                        logger.exception(
                            "pii_filter outlet: vault snapshot_for_request raised; "
                            "returning body unchanged chat_id=%s",
                            chat_id_for_log,
                        )
                        return body
                    if vault_reverse:
                        reverse_map = vault_reverse
                    else:
                        logger.debug(
                            "pii_filter outlet skipped: vault snapshot empty " "for chat_id=%s",
                            chat_id_for_log,
                        )
                        return body
                else:
                    logger.debug(
                        "pii_filter outlet skipped: pii_reverse_map missing and "
                        "vault fallback unavailable"
                    )
                    return body

            target_message = self._extract_assistant_message(body)
            if target_message is None:
                # _extract_assistant_message has already logged the reason at DEBUG.
                return body

            restored_set: set[str] = set()
            hallucinated_set: set[str] = set()
            for text, write_back in self._iter_text_parts(target_message):
                restored_text_value, restored_keys, hallucinated_keys = restore_text(
                    text, reverse_map
                )
                if restored_text_value != text:
                    write_back(restored_text_value)
                restored_set.update(restored_keys)
                hallucinated_set.update(hallucinated_keys)

            logger.info(
                "pii_filter outlet processed: chat_id=%s placeholders_restored=%d "
                "hallucinations=%d",
                chat_id_for_log,
                len(restored_set),
                len(hallucinated_set),
            )
            if hallucinated_set:
                # One WARN per outlet call (per-call dedupe via the set), not
                # per occurrence — long responses with repeated hallucinations
                # would otherwise spam logs and drown out signal.
                logger.warning(
                    "pii_filter outlet hallucinations detected: chat_id=%s count=%d unique=%s",
                    chat_id_for_log,
                    len(hallucinated_set),
                    sorted(hallucinated_set),
                )

            return body
        except (KeyError, AttributeError, TypeError) as exc:
            # Expected fail-safe path: the defensive shape guard tries to
            # cover every malformed body up front, but a future caller could
            # still pass something the guard misses (missing key, wrong
            # nested type). These are operational issues, not bugs — log at
            # WARN and pass the body through.
            logger.warning(
                "pii_filter outlet failed (expected fail-safe): " "chat_id=%s error=%s: %s",
                chat_id_for_log,
                type(exc).__name__,
                exc,
            )
            return body
        except Exception as exc:
            # Unexpected programmer error (typo in restore_text, regex
            # compile bug, future-task regression). Log at ERROR with the
            # full traceback so production observability surfaces it
            # loudly; still swallow so the user sees raw placeholders
            # rather than a 500.
            logger.error(
                "pii_filter outlet UNEXPECTED error (returning body unchanged): "
                "chat_id=%s error=%s: %s",
                chat_id_for_log,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return body


# Resolve forward references for the nested `Valves` BaseSettings.
# `from __future__ import annotations` defers all annotations to strings;
# Pydantic v2 in some container Pydantic versions (2.7.x) fails to look up
# `Literal` while building the nested model and raises
# `PydanticUserError: Valves is not fully defined`. Rebuilding here with
# the module globals in scope makes `Literal` resolvable. Cheap at import
# time, and a no-op if Pydantic already finalized the model.
Pipeline.Valves.model_rebuild()
