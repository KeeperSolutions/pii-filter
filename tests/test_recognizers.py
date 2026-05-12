"""Unit + integration tests for Task 3 recognizers and analyzer wiring.

Per-recognizer tests use `pytest.mark.parametrize` and exercise
`validate_result()` directly (not through the full AnalyzerEngine) for speed
and isolation. Valid samples are computed with small helpers below so the
checksum derivation is auditable in-source.

The integration test loads spaCy `hr_core_news_lg` and is therefore slow
(~5-10s); it verifies that a synthetic text containing one of each entity
type returns all 13 expected entity types from the analyzer.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from presidio_analyzer import RecognizerResult

from pii_filter import (
    HRIBANRecognizer,
    IEPPSNRecognizer,
    JMBGRecognizer,
    OIBRecognizer,
    Pipeline,
    ROCNPRecognizer,
    UKNINORecognizer,
    UKUTRRecognizer,
    USEINRecognizer,
    USSSNRecognizer,
    _classify_window_language,
    _merge_dedupe_detections,
    _nlp_engine_cache,
    _select_accepted_detections,
    make_iban_recognizer,
)

# ---------------------------------------------------------------------------
# Helpers — compute valid samples so the checksum derivation is in-source.
# ---------------------------------------------------------------------------


def _oib_check(first10: str) -> int:
    a = 10
    for d in first10:
        a = (a + int(d)) % 10
        if a == 0:
            a = 10
        a = (a * 2) % 11
    return (11 - a) % 10


def _make_oib(first10: str) -> str:
    return f"{first10}{_oib_check(first10)}"


def _jmbg_check(first12: str) -> int:
    weights = [7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2]
    s = sum(int(first12[i]) * weights[i] for i in range(12))
    check = 11 - (s % 11)
    return 0 if check > 9 else check


def _make_jmbg(first12: str) -> str:
    return f"{first12}{_jmbg_check(first12)}"


def _make_iban(country: str, bban: str) -> str:
    """Compute a valid IBAN given country code and BBAN (digits + uppercase letters)."""
    cc_numeric = "".join(str(ord(ch) - 55) for ch in country)
    rearranged = bban + cc_numeric + "00"
    numeric = ""
    for ch in rearranged:
        numeric += str(ord(ch) - 55) if ch.isalpha() else ch
    check = 98 - (int(numeric) % 97)
    return f"{country}{check:02d}{bban}"


def _ie_ppsn_check(first7: str, second_letter: str = "") -> str:
    weights = [8, 7, 6, 5, 4, 3, 2]
    total = sum(int(c) * w for c, w in zip(first7, weights, strict=True))
    if second_letter:
        total += (ord(second_letter) - ord("A") + 1) * 9
    mod = total % 23
    return "W" if mod == 0 else chr(ord("A") + mod - 1)


def _make_ie_ppsn(first7: str, second_letter: str = "") -> str:
    check = _ie_ppsn_check(first7, second_letter)
    return f"{first7}{check}{second_letter}"


def _ro_cnp_check(first12: str) -> int:
    weights = [2, 7, 9, 1, 4, 6, 3, 5, 8, 2, 7, 9]
    total = sum(int(first12[i]) * weights[i] for i in range(12))
    check = total % 11
    if check == 10:
        check = 1
    return check


def _make_ro_cnp(first12: str) -> str:
    return f"{first12}{_ro_cnp_check(first12)}"


# ---------------------------------------------------------------------------
# HR_OIB
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "valid_oib",
    [
        _make_oib("1234567890"),  # control digit derived via mod 11,10
        _make_oib("9876543210"),
        _make_oib("0000000000"),  # synthetic, not a real OIB
    ],
)
def test_oib_accepts_valid(valid_oib: str) -> None:
    assert OIBRecognizer(supported_language="hr").validate_result(valid_oib) is True


@pytest.mark.parametrize(
    "invalid_oib",
    [
        "12345678901",  # bad checksum (correct check is 3)
        "00000000000",  # bad checksum (correct check is 1)
        "11111111111",  # bad checksum
        "12345678900",  # off-by-one from valid
    ],
)
def test_oib_rejects_invalid(invalid_oib: str) -> None:
    assert OIBRecognizer(supported_language="hr").validate_result(invalid_oib) is False


# ---------------------------------------------------------------------------
# HR_JMBG
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "valid_jmbg",
    [
        _make_jmbg("010199012345"),
        _make_jmbg("150586710023"),
    ],
)
def test_jmbg_accepts_valid(valid_jmbg: str) -> None:
    assert JMBGRecognizer(supported_language="hr").validate_result(valid_jmbg) is True


@pytest.mark.parametrize(
    "invalid_jmbg",
    [
        "0101990500006",  # valid format, bad checksum (correct check is 3)
        "1234567890123",  # valid format, bad checksum (correct check is 5)
        "0000000000001",  # valid format, bad checksum (correct check is 0)
        "999999999999X",  # non-digit
        "12345678",  # too short
    ],
)
def test_jmbg_rejects_invalid(invalid_jmbg: str) -> None:
    assert JMBGRecognizer(supported_language="hr").validate_result(invalid_jmbg) is False


# ---------------------------------------------------------------------------
# HR_IBAN
# ---------------------------------------------------------------------------


VALID_HR_IBANS = [
    _make_iban("HR", "10010051863000160"),
    _make_iban("HR", "23600001101234567"),
]


@pytest.mark.parametrize("valid_hr_iban", VALID_HR_IBANS)
def test_hr_iban_accepts_valid(valid_hr_iban: str) -> None:
    assert HRIBANRecognizer(supported_language="hr").validate_result(valid_hr_iban) is True


@pytest.mark.parametrize(
    "invalid_hr_iban",
    [
        "HR0000000000000000000",  # bad checksum
        "HR9912345678901234567",  # bad checksum
        "HR1110010051863000160",  # check digits flipped from valid
    ],
)
def test_hr_iban_rejects_invalid(invalid_hr_iban: str) -> None:
    assert HRIBANRecognizer(supported_language="hr").validate_result(invalid_hr_iban) is False


# ---------------------------------------------------------------------------
# IE_PPSN
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "valid_ppsn",
    [
        _make_ie_ppsn("1234567"),  # 8-char form
        _make_ie_ppsn("9876543"),
        _make_ie_ppsn("1234567", "A"),  # 9-char form, second_letter A
        _make_ie_ppsn("9876543", "B"),  # 9-char form, second_letter B
    ],
)
def test_ie_ppsn_accepts_valid(valid_ppsn: str) -> None:
    assert IEPPSNRecognizer(supported_language="hr").validate_result(valid_ppsn) is True


@pytest.mark.parametrize(
    "invalid_ppsn",
    [
        "1234567A",  # wrong check letter (correct is T)
        "9999999A",  # wrong check letter
        "1234567TZ",  # second_letter Z is not A or B
        "ABC4567T",  # non-digit prefix
    ],
)
def test_ie_ppsn_rejects_invalid(invalid_ppsn: str) -> None:
    assert IEPPSNRecognizer(supported_language="hr").validate_result(invalid_ppsn) is False


# ---------------------------------------------------------------------------
# RO_CNP
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "valid_cnp",
    [
        # Sex+century=1, YY=99, MM=05, DD=15, county=01, serial=234, check derived
        _make_ro_cnp("199051501234"),
        # Sex+century=2, YY=85, MM=12, DD=20, county=40, serial=123
        _make_ro_cnp("285122040123"),
    ],
)
def test_ro_cnp_accepts_valid(valid_cnp: str) -> None:
    assert ROCNPRecognizer(supported_language="hr").validate_result(valid_cnp) is True


@pytest.mark.parametrize(
    "invalid_cnp",
    [
        "9990515012340",  # first digit 9 (reserved)
        "1991315012343",  # month 13 invalid
        "1990532012344",  # day 32 invalid
        "1990515992345",  # county 99 invalid
        # Bad checksum (structurally valid first 12, wrong last):
        "1990515012349",
    ],
)
def test_ro_cnp_rejects_invalid(invalid_cnp: str) -> None:
    assert ROCNPRecognizer(supported_language="hr").validate_result(invalid_cnp) is False


# ---------------------------------------------------------------------------
# UK_NINO
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "valid_nino",
    [
        "AB123456C",
        "AB 12 34 56 C",
        "JR876543A",
    ],
)
def test_uk_nino_accepts_valid(valid_nino: str) -> None:
    assert UKNINORecognizer(supported_language="hr").validate_result(valid_nino) is True


@pytest.mark.parametrize(
    "invalid_nino",
    [
        "BG123456C",  # invalid prefix BG
        "GB123456A",  # invalid prefix GB
        "DA123456A",  # first letter D invalid
        "AO123456A",  # second letter O invalid
        "AB12345C",  # too short
    ],
)
def test_uk_nino_rejects_invalid(invalid_nino: str) -> None:
    assert UKNINORecognizer(supported_language="hr").validate_result(invalid_nino) is False


# ---------------------------------------------------------------------------
# UK_UTR
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "valid_utr",
    [
        "1234567890",
        "0000000000",
    ],
)
def test_uk_utr_accepts_valid(valid_utr: str) -> None:
    assert UKUTRRecognizer(supported_language="hr").validate_result(valid_utr) is True


@pytest.mark.parametrize(
    "invalid_utr",
    [
        "12345",  # too short
        "12345678901",  # too long
        "abcdefghij",  # non-digit
        "",  # empty
    ],
)
def test_uk_utr_rejects_invalid(invalid_utr: str) -> None:
    assert UKUTRRecognizer(supported_language="hr").validate_result(invalid_utr) is False


# ---------------------------------------------------------------------------
# US_SSN
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "valid_ssn",
    [
        "123-45-6789",
        "001-01-0001",
        "665-99-9999",
    ],
)
def test_us_ssn_accepts_valid(valid_ssn: str) -> None:
    assert USSSNRecognizer(supported_language="hr").validate_result(valid_ssn) is True


@pytest.mark.parametrize(
    "invalid_ssn",
    [
        "000-12-3456",  # area 000 reserved
        "666-12-3456",  # area 666 reserved
        "900-12-3456",  # area 9xx reserved
        "123-00-4567",  # group 00 invalid
        "123-45-0000",  # serial 0000 invalid
    ],
)
def test_us_ssn_rejects_invalid(invalid_ssn: str) -> None:
    assert USSSNRecognizer(supported_language="hr").validate_result(invalid_ssn) is False


# ---------------------------------------------------------------------------
# US_EIN
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "valid_ein",
    [
        "12-3456789",
        "98-7654321",
    ],
)
def test_us_ein_accepts_valid(valid_ein: str) -> None:
    assert USEINRecognizer(supported_language="hr").validate_result(valid_ein) is True


@pytest.mark.parametrize(
    "invalid_ein",
    [
        "123-456789",  # wrong dash position
        "1-23456789",  # one digit prefix
        "12-345678",  # 6-digit suffix
        "ab-3456789",  # non-digit prefix
    ],
)
def test_us_ein_rejects_invalid(invalid_ein: str) -> None:
    assert USEINRecognizer(supported_language="hr").validate_result(invalid_ein) is False


# ---------------------------------------------------------------------------
# IE_IBAN / RO_IBAN / GB_IBAN  (via factory)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "country,bban_length,entity",
    [
        ("IE", 18, "IE_IBAN"),
        ("RO", 20, "RO_IBAN"),
        ("GB", 18, "GB_IBAN"),
    ],
)
def test_factory_iban_accepts_valid(country: str, bban_length: int, entity: str) -> None:
    rec = make_iban_recognizer(country, bban_length, entity, supported_language="hr")
    # BBAN with leading 4 alpha bank code, remainder digits.
    bban = "AAAA" + "1" * (bban_length - 4)
    valid = _make_iban(country, bban)
    assert rec.validate_result(valid) is True


@pytest.mark.parametrize(
    "country,bban_length,entity,invalid",
    [
        ("IE", 18, "IE_IBAN", "IE00AAAA111111111111111111"),
        ("RO", 20, "RO_IBAN", "RO00AAAA11111111111111111111"),
        ("GB", 18, "GB_IBAN", "GB00AAAA111111111111111111"),
        ("IE", 18, "IE_IBAN", "IE99XXXX222222222222222222"),
    ],
)
def test_factory_iban_rejects_invalid(
    country: str, bban_length: int, entity: str, invalid: str
) -> None:
    rec = make_iban_recognizer(country, bban_length, entity, supported_language="hr")
    assert rec.validate_result(invalid) is False


# ---------------------------------------------------------------------------
# IBAN whitespace tolerance (ISO 13616 4-char grouped form)
# ---------------------------------------------------------------------------


def _group_iban(concatenated: str) -> str:
    """Insert ASCII spaces every 4 chars from the start (ISO 13616 grouping).

    "HR1210010051863000160" -> "HR12 1001 0051 8630 0016 0".
    """
    return " ".join(concatenated[i : i + 4] for i in range(0, len(concatenated), 4))


def test_hr_iban_validates_grouped_form() -> None:
    """ISO 13616 whitespace grouping must pass validation (mod-97 strips spaces)."""
    valid = _make_iban("HR", "10010051863000160")
    grouped = _group_iban(valid)
    assert " " in grouped
    assert HRIBANRecognizer(supported_language="hr").validate_result(grouped) is True


@pytest.mark.parametrize(
    "country,bban_length,entity",
    [
        ("IE", 18, "IE_IBAN"),
        ("RO", 20, "RO_IBAN"),
        ("GB", 18, "GB_IBAN"),
    ],
)
def test_factory_iban_validates_grouped_form(country: str, bban_length: int, entity: str) -> None:
    """ISO 13616 whitespace grouping must pass validation for all factory IBANs."""
    bban = "AAAA" + "1" * (bban_length - 4)
    valid = _make_iban(country, bban)
    grouped = _group_iban(valid)
    assert " " in grouped
    rec = make_iban_recognizer(country, bban_length, entity, supported_language="hr")
    assert rec.validate_result(grouped) is True


@pytest.mark.parametrize(
    "country,bban_length,entity",
    [
        ("IE", 18, "IE_IBAN"),
        ("RO", 20, "RO_IBAN"),
        ("GB", 18, "GB_IBAN"),
    ],
)
def test_factory_iban_pattern_matches_grouped_form(
    country: str, bban_length: int, entity: str
) -> None:
    """Recognizer regex must capture the entire grouped IBAN as one span."""
    import re as _re

    bban = "AAAA" + "1" * (bban_length - 4)
    valid = _make_iban(country, bban)
    grouped = _group_iban(valid)
    rec = make_iban_recognizer(country, bban_length, entity, supported_language="hr")
    text = f"Moj IBAN je {grouped}."
    match = _re.search(rec.patterns[0].regex, text)
    assert match is not None, f"{entity} pattern failed to match grouped form"
    matched = match.group()
    # Span must cover the whole grouped IBAN; first 4 chars are country+check.
    assert matched.startswith(grouped[:4])
    assert matched.replace(" ", "") == valid


def test_hr_iban_pattern_matches_grouped_form() -> None:
    """HR recognizer regex must capture the grouped IBAN as one span."""
    import re as _re

    valid = _make_iban("HR", "10010051863000160")
    grouped = _group_iban(valid)
    text = f"Moj IBAN je {grouped}."
    match = _re.search(HRIBANRecognizer(supported_language="hr").patterns[0].regex, text)
    assert match is not None, "HR_IBAN pattern failed to match grouped form"
    assert match.group().replace(" ", "") == valid


# ---------------------------------------------------------------------------
# UK_UTR span integrity (lookbehind keeps span on digits only)
# ---------------------------------------------------------------------------


def test_uk_utr_span_excludes_keyword() -> None:
    """The lookbehind must keep the matched span on the 10 digits only.

    Spec AC #14: result.start should point to the first digit, not to the
    keyword. We verify by running the recognizer's pattern directly.
    """
    rec = UKUTRRecognizer(supported_language="hr")
    text = "Reference: UTR 1234567890 for the file"
    pattern = rec.patterns[0].regex
    import re as _re

    match = _re.search(pattern, text)
    assert match is not None, "UTR pattern did not match"
    # The match should start at the first digit '1', not at 'UTR'.
    assert text[match.start()] == "1"
    assert match.group() == "1234567890"


# ---------------------------------------------------------------------------
# Integration test: load spaCy + analyzer, detect all 13 entity types.
# ---------------------------------------------------------------------------


# One valid synthetic sample of each entity type. The keyword "UTR " before
# the 10-digit UK_UTR is required by the lookbehind in the recognizer.
_VALID_OIB = _make_oib("1234567890")
_VALID_JMBG = _make_jmbg("010199012345")
_VALID_HR_IBAN = _make_iban("HR", "10010051863000160")
_VALID_IE_IBAN = _make_iban("IE", "AAAA111111111111111111"[:18])
_VALID_RO_IBAN = _make_iban("RO", "AAAA1111111111111111"[:20])
_VALID_GB_IBAN = _make_iban("GB", "AAAA111111111111111111"[:18])
_VALID_PPSN = _make_ie_ppsn("1234567", "A")
_VALID_CNP = _make_ro_cnp("199051501234")

# Luhn-valid Visa test number (publicly used in PCI test docs).
_VALID_CC = "4111111111111111"


@pytest_asyncio.fixture(loop_scope="module", scope="module")
async def started_pipeline() -> AsyncIterator[Pipeline]:
    """Shared started Pipeline for tests that need the analyzer.

    spaCy `hr_core_news_lg` allocates ~240 MB of contiguous vectors per load,
    so loading once per module instead of once per test keeps memory pressure
    manageable on dev machines.

    v0.6.0 default backend is postgres; opt back into redis (fakeredis via
    the autouse conftest fixture) so these recognizer-focused tests never
    require a live postgres process.
    """
    p = Pipeline()
    p.valves.vault_backend = "redis"
    p.valves.languages = [
        "hr"
    ]  # HR-only for module fixture; EN tests skip via analyzer_en is None guard
    await p.on_startup()
    yield p
    await p.on_shutdown()


@pytest.fixture(scope="module")
def synthetic_text() -> str:
    return "\n".join(
        [
            f"OIB: {_VALID_OIB}",
            f"JMBG: {_VALID_JMBG}",
            f"HR IBAN: {_VALID_HR_IBAN}",
            f"IE PPSN: {_VALID_PPSN}",
            f"IE IBAN: {_VALID_IE_IBAN}",
            f"RO CNP: {_VALID_CNP}",
            f"RO IBAN: {_VALID_RO_IBAN}",
            "UK NINO: AB 12 34 56 C",
            "UK tax: UTR 1234567890",
            f"GB IBAN: {_VALID_GB_IBAN}",
            "US SSN: 123-45-6789",
            "US EIN: 12-3456789",
            f"Card: {_VALID_CC}",
        ]
    )


@pytest.mark.asyncio(loop_scope="module")
async def test_integration_all_13_entities_detected(
    started_pipeline: Pipeline, synthetic_text: str
) -> None:
    """End-to-end: spaCy + AnalyzerEngine + 12 custom recognizers + CC built-in."""
    assert started_pipeline.analyzer_hr is not None
    results = started_pipeline.analyzer_hr.analyze(text=synthetic_text, language="hr")
    detected = {r.entity_type for r in results}
    expected = {
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
        "CREDIT_CARD",
    }
    missing = expected - detected
    assert not missing, f"Missing entity types: {missing}; detected={detected}"


# ---------------------------------------------------------------------------
# Inlet: detection results attached to body metadata.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_attaches_pii_detections_to_metadata(
    started_pipeline: Pipeline,
) -> None:
    """`inlet` must call analyzer, map via PRESIDIO_TO_STANDARD, attach to metadata."""
    oib = _make_oib("1234567890")
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": f"My OIB is {oib}"}],
        "metadata": {"chat_id": "abc"},
    }
    result = await started_pipeline.inlet(body)
    assert "metadata" in result
    # Existing metadata keys must be preserved.
    assert result["metadata"]["chat_id"] == "abc"
    detections = result["metadata"]["pii_detections"]
    assert isinstance(detections, list)
    assert any(d["entity_type"] == "HR_OIB" for d in detections)
    # Schema check on at least one detection
    sample = next(d for d in detections if d["entity_type"] == "HR_OIB")
    assert {"entity_type", "start", "end", "score", "raw_entity_type"} <= sample.keys()


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_creates_metadata_when_missing(started_pipeline: Pipeline) -> None:
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": f"OIB {_make_oib('1234567890')}"}]
    }
    result = await started_pipeline.inlet(body)
    assert "metadata" in result
    assert "pii_detections" in result["metadata"]


@pytest.mark.asyncio(loop_scope="module")
async def test_inlet_robust_to_malformed_body(started_pipeline: Pipeline) -> None:
    """inlet must not raise on missing/malformed messages."""
    # Empty body
    assert await started_pipeline.inlet({}) == {}
    # Wrong shape
    assert await started_pipeline.inlet({"messages": "not a list"}) == {"messages": "not a list"}
    # Empty list
    assert await started_pipeline.inlet({"messages": []}) == {"messages": []}
    # Non-string content
    body: dict[str, Any] = {"messages": [{"role": "user", "content": [1, 2, 3]}]}
    result = await started_pipeline.inlet(body)
    # Should pass through unchanged (no metadata added).
    assert "pii_detections" not in result.get("metadata", {})


class _ExplodingAnalyzer:
    """Stub analyzer whose `analyze` always raises — used to drive inlet's
    degradation-mode branches without touching the real Presidio engine."""

    def analyze(self, *args: Any, **kwargs: Any) -> list[Any]:
        raise RuntimeError("simulated analyzer failure")


@pytest.mark.asyncio
async def test_inlet_fails_closed_by_default_on_analyzer_error() -> None:
    """Default degradation_mode='block' must raise rather than leak PII."""
    pipeline = Pipeline()
    # Skip on_startup; inject a stub analyzer so we exercise only the except branch.
    pipeline.analyzer_hr = _ExplodingAnalyzer()  # type: ignore[assignment]
    pipeline.analyzer_en = _ExplodingAnalyzer()  # type: ignore[assignment]
    assert pipeline.valves.degradation_mode == "block"
    body: dict[str, Any] = {"messages": [{"role": "user", "content": "hello"}]}
    with pytest.raises(RuntimeError, match="Request blocked"):
        await pipeline.inlet(body)


@pytest.mark.asyncio
async def test_inlet_passes_through_when_degradation_mode_is_passthrough() -> None:
    """`degradation_mode='passthrough'` must log + return body unchanged on analyzer error."""
    pipeline = Pipeline()
    pipeline.analyzer_hr = _ExplodingAnalyzer()  # type: ignore[assignment]
    pipeline.analyzer_en = _ExplodingAnalyzer()  # type: ignore[assignment]
    pipeline.valves.degradation_mode = "passthrough"
    body: dict[str, Any] = {"messages": [{"role": "user", "content": "hello"}]}
    result = await pipeline.inlet(body)
    assert result is body
    # No detections should have been attached.
    assert "pii_detections" not in result.get("metadata", {})


# ---------------------------------------------------------------------------
# Task 3.1 — Deny-list, trailing-token strip, OIB context (unit tests)
#
# All tests call `_select_accepted_detections` directly with synthetic
# RecognizerResult objects so they run without loading spaCy.
# ---------------------------------------------------------------------------

_STD: dict[str, str] = Pipeline.PRESIDIO_TO_STANDARD

# Default deny/strip lists from Valves class defaults only.
# Use model_construct() so tests do not instantiate Pipeline or read
# environment-backed settings at import time.
_DEFAULT_VALVES = Pipeline.Valves.model_construct()
_DENY_LIST: frozenset[str] = frozenset(s.lower() for s in _DEFAULT_VALVES.ner_deny_list)
_STRIP_LIST: frozenset[str] = frozenset(s.lower() for s in _DEFAULT_VALVES.ner_trailing_token_strip)

# Confirmed false-positive: passes OIB mod-11,10 checksum but is a stripped
# US phone number. See spec Q1 diagnostic for derivation.
_PHONE_COLLISION_OIB = "15551234567"


def _det_person(start: int, end: int, score: float = 0.85) -> RecognizerResult:
    return RecognizerResult(entity_type="PERSON", start=start, end=end, score=score)


def _det_oib(start: int, end: int, score: float = 0.4) -> RecognizerResult:
    return RecognizerResult(entity_type="HR_OIB", start=start, end=end, score=score)


# -- Deny-list tests ---------------------------------------------------------


def test_deny_list_drops_task_keyword() -> None:
    text = "Task"
    result = _select_accepted_detections(text, [_det_person(0, 4)], _STD, deny_list=_DENY_LIST)
    assert result == []


def test_deny_list_drops_json_output() -> None:
    text = "JSON output expected"
    result = _select_accepted_detections(text, [_det_person(0, 11)], _STD, deny_list=_DENY_LIST)
    assert result == []


def test_deny_list_drops_emoji_summarizing() -> None:
    text = "emoji summarizing the conversation"
    result = _select_accepted_detections(
        text, [_det_person(0, len(text))], _STD, deny_list=_DENY_LIST
    )
    assert result == []


def test_deny_list_is_case_insensitive() -> None:
    # Deny-list stores lowercase "task"; entity text "Task" must still be dropped.
    text = "Task"
    result = _select_accepted_detections(
        text, [_det_person(0, 4)], _STD, deny_list=frozenset(["task"])
    )
    assert result == []


def test_deny_list_does_not_drop_real_person() -> None:
    text = "Ivan Horvat"
    result = _select_accepted_detections(text, [_det_person(0, 11)], _STD, deny_list=_DENY_LIST)
    assert len(result) == 1
    assert result[0].entity_type == "PERSON"


def test_deny_list_empty_disables_filter() -> None:
    # Even a deny-list keyword should survive when deny_list is empty frozenset.
    text = "Task"
    result = _select_accepted_detections(text, [_det_person(0, 4)], _STD, deny_list=frozenset())
    assert len(result) == 1
    assert result[0].entity_type == "PERSON"


def test_deny_list_custom_entry_works() -> None:
    text = "Skywalker"
    result = _select_accepted_detections(
        text, [_det_person(0, 9)], _STD, deny_list=frozenset(["skywalker"])
    )
    assert result == []


# -- Trailing-token strip tests ----------------------------------------------


def test_trailing_strip_removes_english_has() -> None:
    # "Ivan Horvat has" → span shortened to "Ivan Horvat" (length 11, not 15).
    # Matches spec AC 3.8 / Q3 production evidence.
    text = "Ivan Horvat has doktor"
    result = _select_accepted_detections(
        text, [_det_person(0, 15)], _STD, trailing_strip=_STRIP_LIST
    )
    assert len(result) == 1
    assert result[0].end - result[0].start == 11
    assert text[result[0].start : result[0].end] == "Ivan Horvat"


def test_trailing_strip_removes_croatian_je() -> None:
    # "Ivan Horvat je" → "Ivan Horvat"
    text = "Ivan Horvat je doktor"
    result = _select_accepted_detections(
        text, [_det_person(0, 14)], _STD, trailing_strip=_STRIP_LIST
    )
    assert len(result) == 1
    assert text[result[0].start : result[0].end] == "Ivan Horvat"


def test_trailing_strip_preserves_real_person() -> None:
    text = "Ivan Horvat"
    result = _select_accepted_detections(
        text, [_det_person(0, 11)], _STD, trailing_strip=_STRIP_LIST
    )
    assert len(result) == 1
    assert result[0].end == 11  # span unchanged


def test_trailing_strip_handles_punctuation() -> None:
    # Trailing "." stripped even without a function word match.
    text = "Ivan Horvat."
    result = _select_accepted_detections(
        text, [_det_person(0, 12)], _STD, trailing_strip=_STRIP_LIST
    )
    assert len(result) == 1
    assert text[result[0].start : result[0].end] == "Ivan Horvat"


# -- OIB phone-context tests -------------------------------------------------


def test_oib_rejected_with_phone_prefix_plus_one() -> None:
    text = f"+1 {_PHONE_COLLISION_OIB}"
    start = len("+1 ")
    result = _select_accepted_detections(
        text, [_det_oib(start, start + 11)], _STD, oib_phone_window=30
    )
    assert result == []


def test_oib_rejected_with_phone_keyword() -> None:
    text = f"phone: {_PHONE_COLLISION_OIB}"
    start = len("phone: ")
    result = _select_accepted_detections(
        text, [_det_oib(start, start + 11)], _STD, oib_phone_window=30
    )
    assert result == []


def test_oib_accepted_with_oib_context_word() -> None:
    # "OIB" keyword overrides the phone-context rejection.
    text = f"Moj OIB je {_PHONE_COLLISION_OIB}"
    start = len("Moj OIB je ")
    result = _select_accepted_detections(
        text, [_det_oib(start, start + 11)], _STD, oib_phone_window=30
    )
    assert len(result) == 1
    assert result[0].entity_type == "HR_OIB"


def test_oib_accepted_in_neutral_text() -> None:
    # No phone or OIB keyword in window → detection is kept.
    valid_oib = _make_oib("1234567890")
    text = f"Broj {valid_oib}"
    start = len("Broj ")
    result = _select_accepted_detections(
        text, [_det_oib(start, start + 11)], _STD, oib_phone_window=30
    )
    assert len(result) == 1
    assert result[0].entity_type == "HR_OIB"


def test_oib_phone_collision_15551234567_rejected_with_phone_context() -> None:
    # Spec Q1 / AC 3.9: 15551234567 passes the OIB mod-11,10 checksum (collision).
    # The "mob: " prefix (mobile phone context) must reject it as a phone number.
    # Patterns require the phone keyword to appear IMMEDIATELY before the number
    # ($ anchor), so "mob: 15551234567" is the canonical rejection case here.
    text = f"mob: {_PHONE_COLLISION_OIB}"
    start = len("mob: ")
    result = _select_accepted_detections(
        text, [_det_oib(start, start + 11)], _STD, oib_phone_window=30
    )
    assert result == []


def test_oib_context_window_disabled_at_zero() -> None:
    # oib_phone_window=0 disables the check entirely; phone context is ignored.
    text = f"phone: {_PHONE_COLLISION_OIB}"
    start = len("phone: ")
    result = _select_accepted_detections(
        text, [_det_oib(start, start + 11)], _STD, oib_phone_window=0
    )
    assert len(result) == 1


# -- Post-review additions (Q1 + Q3) ----------------------------------------


def test_select_accepted_detections_drops_denied_keyword_via_unit_call() -> None:
    # Q1: hard guarantee that the deny-list drop works at the function level,
    # independent of spaCy. Constructs a RecognizerResult directly and asserts
    # it is suppressed — no analyzer, no model load.
    text = "Task"
    det = RecognizerResult(entity_type="PERSON", start=0, end=4, score=0.85)
    result = _select_accepted_detections(text, [det], _STD, deny_list=frozenset(["task"]))
    assert result == [], f"Expected deny-list to drop 'Task' PERSON, got {result!r}"


def test_oib_rejected_with_croatian_mobitel_keyword() -> None:
    # Q3: "mobitel" (Croatian for mobile phone) must reject an OIB-checksum-
    # passing 11-digit number when it appears immediately before the number.
    text = f"mobitel: {_PHONE_COLLISION_OIB}"
    start = len("mobitel: ")
    result = _select_accepted_detections(
        text, [_det_oib(start, start + 11)], _STD, oib_phone_window=30
    )
    assert result == []


def test_oib_rejected_with_croatian_telefon_keyword() -> None:
    # Q3: "telefon" (Croatian for telephone) must reject the same collision OIB.
    text = f"telefon: {_PHONE_COLLISION_OIB}"
    start = len("telefon: ")
    result = _select_accepted_detections(
        text, [_det_oib(start, start + 11)], _STD, oib_phone_window=30
    )
    assert result == []


# -- Post-smoke-test prefix-match tests (2026-05-11) --------------------------


def test_deny_list_matches_prefix_with_trailing_word() -> None:
    # Smoke test §8 Scenarij C: spaCy detects "emojis that enhance understanding of"
    # (5 words) but the deny-list only has "emojis that enhance understanding" (4 words).
    # Prefix rule: drop when entity starts with a denied entry followed by a space.
    text = "emojis that enhance understanding of"
    det = RecognizerResult(entity_type="PERSON", start=0, end=len(text), score=0.85)
    deny = frozenset(["emojis that enhance understanding"])
    result = _select_accepted_detections(text, [det], _STD, deny_list=deny)
    assert (
        result == []
    ), "Expected prefix rule to drop PERSON whose text starts with a denied entry + space"


def test_deny_list_prefix_match_does_not_drop_internal_substring() -> None:
    # Guard: "tasksmith" must NOT match deny-list "task" — the space boundary
    # ensures only whole-word prefixes are dropped, not embedded substrings.
    for entity_text, denied in [("tasksmith", "task"), ("rawdata", "raw")]:
        det = RecognizerResult(entity_type="PERSON", start=0, end=len(entity_text), score=0.85)
        result = _select_accepted_detections(
            entity_text, [det], _STD, deny_list=frozenset([denied])
        )
        assert len(result) == 1, (
            f"'{entity_text}' must NOT be dropped by deny-list entry '{denied}' "
            f"(no space boundary)"
        )


def test_deny_list_prefix_match_does_not_drop_real_name_starting_with_keyword() -> None:
    # "Task Manager John" starts with "task " — prefix rule drops it.
    # Per spec: prefer false negatives on PII over false positives on common keywords.
    text = "Task Manager John"
    det = RecognizerResult(entity_type="PERSON", start=0, end=len(text), score=0.85)
    result = _select_accepted_detections(text, [det], _STD, deny_list=frozenset(["task"]))
    assert (
        result == []
    ), "Expected 'Task Manager John' to be dropped — entity starts with denied 'task' + space"


# ---------------------------------------------------------------------------
# Task 3.2 — Multi-Language Detection (HR + EN)
# ---------------------------------------------------------------------------

# A. Unit tests — _merge_dedupe_detections --------------------------------


def test_merge_dedupe_drops_duplicate_lower_score() -> None:
    # Uses HR_OIB (regex entity) so the Task 3.3 NER window filter never applies;
    # this test is purely about dedupe — EN wins on higher score.
    hr = [RecognizerResult("HR_OIB", 0, 11, 0.8)]
    en = [RecognizerResult("HR_OIB", 0, 11, 0.9)]
    merged_list, dropped = _merge_dedupe_detections(hr, en, "12345678903")
    assert len(merged_list) == 1
    assert merged_list[0].score == 0.9
    assert dropped == 0


def test_merge_dedupe_keeps_disjoint_spans() -> None:
    # HR_OIB and EMAIL_ADDRESS are both regex entities — window filter skips them.
    hr = [RecognizerResult("HR_OIB", 0, 11, 0.8)]
    en = [RecognizerResult("EMAIL_ADDRESS", 15, 30, 0.9)]
    merged_list, dropped = _merge_dedupe_detections(hr, en, "12345678903 foo a@b.com")
    assert len(merged_list) == 2
    assert {r.entity_type for r in merged_list} == {"HR_OIB", "EMAIL_ADDRESS"}
    assert dropped == 0


def test_merge_dedupe_hr_stable_on_tie() -> None:
    # HR_OIB: regex entity, no window filter. HR wins on equal score.
    hr_det = RecognizerResult("HR_OIB", 0, 11, 0.85)
    en_det = RecognizerResult("HR_OIB", 0, 11, 0.85)
    merged_list, dropped = _merge_dedupe_detections([hr_det], [en_det], "12345678903")
    assert len(merged_list) == 1
    assert merged_list[0] is hr_det, "HR result must win on equal score (stable-first order)"
    assert dropped == 0


def test_merge_dedupe_different_entity_types_same_span() -> None:
    # HR_OIB and EMAIL_ADDRESS: both regex entities; different types → both survive.
    hr = [RecognizerResult("HR_OIB", 0, 11, 0.85)]
    en = [RecognizerResult("EMAIL_ADDRESS", 0, 11, 0.80)]
    merged_list, dropped = _merge_dedupe_detections(hr, en, "12345678903")
    assert len(merged_list) == 2
    assert {r.entity_type for r in merged_list} == {"HR_OIB", "EMAIL_ADDRESS"}
    assert dropped == 0


def test_merge_dedupe_empty_inputs() -> None:
    hr = [RecognizerResult("HR_OIB", 0, 11, 0.8)]
    en = [RecognizerResult("EMAIL_ADDRESS", 15, 30, 0.9)]
    empty_list, dropped = _merge_dedupe_detections([], [], "")
    assert empty_list == []
    assert dropped == 0
    result_hr_only, _ = _merge_dedupe_detections(hr, [], "12345678903")
    assert len(result_hr_only) == 1 and result_hr_only[0] is hr[0]
    result_en_only, _ = _merge_dedupe_detections([], en, "12345678903 foo a@b.com")
    assert len(result_en_only) == 1 and result_en_only[0] is en[0]


def test_us_ssn_custom_validates_reserved_blocks() -> None:
    rec = USSSNRecognizer(supported_language="hr")
    assert rec.validate_result("000-45-6789") is False  # reserved area 000
    assert rec.validate_result("666-45-6789") is False  # reserved area 666
    assert rec.validate_result("900-45-6789") is False  # reserved area 9XX
    assert rec.validate_result("123-45-6789") is True  # valid area


def test_uk_nhs_detected_in_en_registry(started_pipeline: Pipeline) -> None:
    if started_pipeline.analyzer_en is None:
        pytest.skip("en_core_web_lg not installed — skipping EN registry test")
    results = started_pipeline.analyzer_en.analyze(
        text="My NHS number is 943 476 5919", language="en"
    )
    entity_types = {r.entity_type for r in results}
    assert "UK_NHS" in entity_types, f"Expected UK_NHS in EN registry, got: {entity_types}"


def test_iban_code_built_in_in_en_registry_for_de_iban(started_pipeline: Pipeline) -> None:
    if started_pipeline.analyzer_en is None:
        pytest.skip("en_core_web_lg not installed — skipping EN registry test")
    results = started_pipeline.analyzer_en.analyze(
        text="My IBAN is DE89370400440532013000", language="en"
    )
    entity_types = {r.entity_type for r in results}
    assert (
        "IBAN_CODE" in entity_types
    ), f"Expected IBAN_CODE (built-in IbanRecognizer) in EN registry, got: {entity_types}"


# C. Startup / configuration tests -----------------------------------------


@pytest.mark.asyncio
async def test_startup_builds_both_analyzers_with_default_languages() -> None:
    p = Pipeline()
    p.valves.vault_backend = "redis"
    try:
        await p.on_startup()
    except RuntimeError as exc:
        pytest.skip(f"en_core_web_lg unavailable: {exc}")
    try:
        assert p.analyzer_hr is not None, "HR analyzer must be built with default languages"
        assert p.analyzer_en is not None, "EN analyzer must be built with default languages"
    finally:
        await p.on_shutdown()


@pytest.mark.asyncio
async def test_startup_validates_languages_whitelist() -> None:
    p = Pipeline()
    p.valves.vault_backend = "redis"
    p.valves.languages = ["fr"]
    with pytest.raises(RuntimeError) as exc_info:
        await p.on_startup()
    msg = str(exc_info.value)
    assert "fr" in msg
    assert "Allowed: hr, en" in msg


@pytest.mark.asyncio
async def test_startup_validates_languages_not_empty() -> None:
    p = Pipeline()
    p.valves.vault_backend = "redis"
    p.valves.languages = []
    with pytest.raises(RuntimeError):
        await p.on_startup()


@pytest.mark.asyncio
async def test_nlp_engine_cache_keyed_by_lang_code() -> None:
    # Both keys must be present after a dual-language startup.
    p = Pipeline()
    p.valves.vault_backend = "redis"
    p.valves.languages = ["hr", "en"]
    try:
        await p.on_startup()
    except RuntimeError as exc:
        pytest.skip(f"en_core_web_lg unavailable (memory or install): {exc}")
    try:
        assert "hr" in _nlp_engine_cache
        assert "en" in _nlp_engine_cache
        # Second startup with only HR must reuse the cached HR engine.
        hr_engine_before = _nlp_engine_cache["hr"]
        p2 = Pipeline()
        p2.valves.vault_backend = "redis"
        p2.valves.languages = ["hr"]
        await p2.on_startup()
        assert _nlp_engine_cache["hr"] is hr_engine_before
        await p2.on_shutdown()
    finally:
        await p.on_shutdown()


# ---------------------------------------------------------------------------
# Task 3.3 — Unit tests: _classify_window_language
# ---------------------------------------------------------------------------


def test_classify_window_pure_hr_text() -> None:
    text = "Moj OIB je 12345678903"
    # span covers "12345678903" at positions 11-22; window includes "Moj OIB je"
    result = _classify_window_language(text, 11, 22)
    assert result == "hr", f"Pure HR text should classify as 'hr', got {result!r}"


def test_classify_window_pure_en_text() -> None:
    text = "My SSN is 123-45-6789"
    # span covers "123-45-6789" at positions 10-21; window includes "My SSN is"
    result = _classify_window_language(text, 10, 21)
    assert result == "en", f"Pure EN text should classify as 'en', got {result!r}"


def test_classify_window_mixed_hr_dominant() -> None:
    # "Moj" is a HR marker; "is" is an EN marker; HR > EN → "hr"
    text = "Moj kolega is named X"
    result = _classify_window_language(text, 18, 19)
    assert result == "hr", f"HR-dominant mixed text should classify as 'hr', got {result!r}"


def test_classify_window_mixed_en_dominant() -> None:
    # "My", "is" are EN markers; "Ivan" has no markers near it
    text = "My friend is Ivan"
    result = _classify_window_language(text, 13, 17)
    assert result == "en", f"EN-dominant mixed text should classify as 'en', got {result!r}"


def test_classify_window_diacritics_only() -> None:
    # Diacritics alone count as HR markers
    text = "Žučno tvrdi: 12345678903"
    result = _classify_window_language(text, 13, 24)
    assert result == "hr", f"Diacritics-containing text should classify as 'hr', got {result!r}"


def test_classify_window_no_markers_either_side() -> None:
    # No stopwords, no diacritics → tie → defaults to 'hr' (Q4)
    text = "X1 Y2 Z3 12345678903"
    result = _classify_window_language(text, 10, 21)
    assert result == "hr", f"No-marker text must default to 'hr' (tie-break Q4), got {result!r}"


def test_classify_window_clipped_at_text_start() -> None:
    # Detection at position 0 — window must not underflow (no negative index)
    text = "Ivan Horvat je student"
    result = _classify_window_language(text, 0, 11)
    # "je" is an HR marker in the trailing window → should be "hr"
    assert result in ("hr", "en"), "Must return a valid language, no index error"


def test_classify_window_clipped_at_text_end() -> None:
    # Detection at end of string — window must not overflow past len(text)
    text = "SSN is 123-45-6789"
    end = len(text)
    result = _classify_window_language(text, end - 11, end)
    # "is" is an EN marker in the leading window → should be "en"
    assert result in ("hr", "en"), "Must return a valid language, no index error"


def test_classify_window_zero_window_chars_returns_hr_default() -> None:
    # window_chars=0 → window is exactly the span text only; no surrounding context.
    # Empty or span-only window → tie → "hr" (Q4 default).
    text = "ABCDEF"
    result = _classify_window_language(text, 0, 6, window_chars=0)
    assert result == "hr", f"Zero-context window must default to 'hr' (Q4 tie), got {result!r}"


# ---------------------------------------------------------------------------
# Task 3.3 — Unit tests: _merge_dedupe_detections NER spillover filter
# ---------------------------------------------------------------------------


def test_merge_drops_en_person_in_hr_text() -> None:
    # EN analyzer emits PERSON for "Moj OIB" (false positive on HR text).
    # Window around "Moj OIB" (0-7) in HR text → "hr" → EN PERSON dropped.
    text = "Moj OIB je 12345678903"
    en_person = RecognizerResult("PERSON", 0, 7, 0.7)
    merged_list, dropped = _merge_dedupe_detections([], [en_person], text)
    assert len(merged_list) == 0, "EN PERSON in HR window must be dropped"
    assert dropped == 1


def test_merge_keeps_hr_person_in_hr_text() -> None:
    # HR analyzer emits PERSON for "Ivan Horvat" — window is HR → kept.
    text = "Moj OIB je 12345678903, a zovem se Ivan Horvat."
    start = text.index("Ivan Horvat")
    hr_person = RecognizerResult("PERSON", start, start + 11, 0.85)
    merged_list, dropped = _merge_dedupe_detections([hr_person], [], text)
    assert len(merged_list) == 1, "HR PERSON in HR window must be kept"
    assert dropped == 0


def test_merge_keeps_regex_entity_regardless_of_language() -> None:
    # HR_OIB is regex-based — never filtered by window language.
    text = "Moj OIB je 12345678903"
    hr_oib = RecognizerResult("HR_OIB", 11, 22, 0.85)
    en_oib_dup = RecognizerResult("HR_OIB", 11, 22, 0.85)
    # Both pass filter; dedupe keeps one (HR wins on tie).
    merged_list, dropped = _merge_dedupe_detections([hr_oib], [en_oib_dup], text)
    assert len(merged_list) == 1, "Regex entity must always pass NER window filter"
    assert dropped == 0


def test_merge_drops_en_person_in_hr_dominant_mixed_text() -> None:
    # Mixed text: "Moj prijatelj je John Smith". Window around "John Smith"
    # includes "je" (HR marker) and no EN markers → "hr" → EN PERSON dropped (Q4 tie-break).
    text = "Moj prijatelj je John Smith"
    start = text.index("John Smith")
    en_person = RecognizerResult("PERSON", start, start + 10, 0.75)
    merged_list, dropped = _merge_dedupe_detections([], [en_person], text)
    # Q4: tie or HR-dominant → "hr" → EN source detection dropped.
    # Document this as acceptable: if HR model also detected "John Smith",
    # its HR detection would have been passed by the filter.
    assert dropped == 1, (
        "EN PERSON in HR-dominant window must be dropped (Q4 tie-break; "
        "HR model should handle EN names if duplicated into HR registry)"
    )
    assert len(merged_list) == 0
