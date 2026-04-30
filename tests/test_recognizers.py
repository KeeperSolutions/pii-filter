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
    assert OIBRecognizer().validate_result(valid_oib) is True


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
    assert OIBRecognizer().validate_result(invalid_oib) is False


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
    assert JMBGRecognizer().validate_result(valid_jmbg) is True


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
    assert JMBGRecognizer().validate_result(invalid_jmbg) is False


# ---------------------------------------------------------------------------
# HR_IBAN
# ---------------------------------------------------------------------------


VALID_HR_IBANS = [
    _make_iban("HR", "10010051863000160"),
    _make_iban("HR", "23600001101234567"),
]


@pytest.mark.parametrize("valid_hr_iban", VALID_HR_IBANS)
def test_hr_iban_accepts_valid(valid_hr_iban: str) -> None:
    assert HRIBANRecognizer().validate_result(valid_hr_iban) is True


@pytest.mark.parametrize(
    "invalid_hr_iban",
    [
        "HR0000000000000000000",  # bad checksum
        "HR9912345678901234567",  # bad checksum
        "HR1110010051863000160",  # check digits flipped from valid
    ],
)
def test_hr_iban_rejects_invalid(invalid_hr_iban: str) -> None:
    assert HRIBANRecognizer().validate_result(invalid_hr_iban) is False


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
    assert IEPPSNRecognizer().validate_result(valid_ppsn) is True


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
    assert IEPPSNRecognizer().validate_result(invalid_ppsn) is False


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
    assert ROCNPRecognizer().validate_result(valid_cnp) is True


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
    assert ROCNPRecognizer().validate_result(invalid_cnp) is False


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
    assert UKNINORecognizer().validate_result(valid_nino) is True


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
    assert UKNINORecognizer().validate_result(invalid_nino) is False


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
    assert UKUTRRecognizer().validate_result(valid_utr) is True


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
    assert UKUTRRecognizer().validate_result(invalid_utr) is False


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
    assert USSSNRecognizer().validate_result(valid_ssn) is True


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
    assert USSSNRecognizer().validate_result(invalid_ssn) is False


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
    assert USEINRecognizer().validate_result(valid_ein) is True


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
    assert USEINRecognizer().validate_result(invalid_ein) is False


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
    rec = make_iban_recognizer(country, bban_length, entity)
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
    rec = make_iban_recognizer(country, bban_length, entity)
    assert rec.validate_result(invalid) is False


# ---------------------------------------------------------------------------
# UK_UTR span integrity (lookbehind keeps span on digits only)
# ---------------------------------------------------------------------------


def test_uk_utr_span_excludes_keyword() -> None:
    """The lookbehind must keep the matched span on the 10 digits only.

    Spec AC #14: result.start should point to the first digit, not to the
    keyword. We verify by running the recognizer's pattern directly.
    """
    rec = UKUTRRecognizer()
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
    """
    p = Pipeline()
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
    assert started_pipeline.analyzer is not None
    results = started_pipeline.analyzer.analyze(text=synthetic_text, language="hr")
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
    pipeline.analyzer = _ExplodingAnalyzer()  # type: ignore[assignment]
    assert pipeline.valves.degradation_mode == "block"
    body: dict[str, Any] = {"messages": [{"role": "user", "content": "hello"}]}
    with pytest.raises(RuntimeError, match="Request blocked"):
        await pipeline.inlet(body)


@pytest.mark.asyncio
async def test_inlet_passes_through_when_degradation_mode_is_passthrough() -> None:
    """`degradation_mode='passthrough'` must log + return body unchanged on analyzer error."""
    pipeline = Pipeline()
    pipeline.analyzer = _ExplodingAnalyzer()  # type: ignore[assignment]
    pipeline.valves.degradation_mode = "passthrough"
    body: dict[str, Any] = {"messages": [{"role": "user", "content": "hello"}]}
    result = await pipeline.inlet(body)
    assert result is body
    # No detections should have been attached.
    assert "pii_detections" not in result.get("metadata", {})
