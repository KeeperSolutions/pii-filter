"""Skeleton smoke tests for Task 1.

These tests verify the basic Pipeline class structure without any detection logic.
Detection-specific tests come in Tasks 3-7.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock

import pytest

from pii_filter import Pipeline


def test_pipeline_instantiates() -> None:
    """Pipeline class can be instantiated without errors."""
    pipeline = Pipeline()
    assert pipeline is not None


def test_pipeline_type_is_filter(pipeline: Pipeline) -> None:
    """Pipeline.type must be 'filter' (required by Open WebUI Pipelines protocol)."""
    assert pipeline.type == "filter"


def test_pipeline_name_is_set(pipeline: Pipeline) -> None:
    """Pipeline.name is the human-readable name shown in OpenWebUI Admin UI."""
    assert pipeline.name == "PII Filter"


def test_valves_have_defaults(pipeline: Pipeline) -> None:
    """Valves must initialize with the expected defaults from the spec."""
    valves = pipeline.valves
    assert valves.pipelines == ["*"]
    assert valves.priority == 0
    assert valves.enabled is True
    assert valves.languages == ["hr", "en"]
    assert valves.degradation_mode == "block"


def test_valves_loads_postgres_url_from_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`PII_FILTER_POSTGRES_URL` must populate `Valves.postgres_url`.

    BaseSettings strips the `PII_FILTER_` prefix and lowercases the rest
    to match a declared field. Verifies the runtime path that Pipelines
    container relies on (admin sets env var, valve picks it up).
    """
    dsn = "postgresql://user:pw@/db?host=/cloudsql/proj:region:inst"
    monkeypatch.setenv("PII_FILTER_POSTGRES_URL", dsn)
    valves = Pipeline.Valves()
    assert valves.postgres_url == dsn


def test_valves_loads_vault_backend_from_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`PII_FILTER_VAULT_BACKEND=redis` must override the postgres default."""
    monkeypatch.setenv("PII_FILTER_VAULT_BACKEND", "redis")
    valves = Pipeline.Valves()
    assert valves.vault_backend == "redis"


def test_valves_invalid_vault_backend_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bogus vault backend must fail loud at construction.

    Confirms `Literal["redis", "postgres"]` validation survives the
    BaseSettings migration so an env-var typo aborts startup instead of
    silently leaving the wrong backend selected.
    """
    monkeypatch.setenv("PII_FILTER_VAULT_BACKEND", "mongodb")
    with pytest.raises(Exception, match="vault_backend"):
        Pipeline.Valves()


async def test_inlet_returns_body_unchanged(
    pipeline: Pipeline, sample_user_body: dict[str, Any]
) -> None:
    """Skeleton inlet must echo the body unchanged (no detection logic yet)."""
    result = await pipeline.inlet(sample_user_body)
    assert result == sample_user_body


async def test_outlet_returns_body_unchanged(
    pipeline: Pipeline, sample_assistant_body: dict[str, Any]
) -> None:
    """Skeleton outlet must echo the body unchanged (no restoration logic yet)."""
    result = await pipeline.outlet(sample_assistant_body)
    assert result == sample_assistant_body


async def test_inlet_skips_when_disabled(
    pipeline: Pipeline, sample_user_body: dict[str, Any]
) -> None:
    """When valves.enabled=False, inlet must short-circuit and return body untouched."""
    pipeline.valves.enabled = False
    result = await pipeline.inlet(sample_user_body)
    assert result == sample_user_body


async def test_outlet_skips_when_disabled(
    pipeline: Pipeline, sample_assistant_body: dict[str, Any]
) -> None:
    """When valves.enabled=False, outlet must short-circuit and return body untouched."""
    pipeline.valves.enabled = False
    result = await pipeline.outlet(sample_assistant_body)
    assert result == sample_assistant_body


async def test_lifecycle_hooks_dont_throw(pipeline: Pipeline) -> None:
    """on_startup() and on_shutdown() must complete without raising."""
    # v0.6.0 default backend is postgres which requires a DSN; opt back into
    # redis (fakeredis via the autouse conftest fixture) so this lifecycle
    # smoke test stays infrastructure-free.
    pipeline.valves.vault_backend = "redis"
    pipeline.valves.languages = ["hr"]  # HR-only to avoid EN model load
    await pipeline.on_startup()
    await pipeline.on_shutdown()


# ---------------------------------------------------------------------------
# Task 8 — UserValves wiring + Valves.presidio_enabled kill switch (unit)
# ---------------------------------------------------------------------------
#
# These tests exercise the two early-return branches in `inlet` without
# spinning up the analyzer (slow). They use the function-scoped `pipeline`
# fixture (no `on_startup` call), so `pipeline.vault is None` and the
# masking loop never gets a chance to run.


def test_user_valves_default_is_true() -> None:
    """UserValves.pii_masking_enabled defaults to True (opt-out, not opt-in)."""
    user_valves = Pipeline.UserValves()
    assert user_valves.pii_masking_enabled is True


def test_valves_presidio_enabled_default_is_true() -> None:
    """Valves.presidio_enabled defaults to True so existing deployments
    keep detection on without explicit env config (AC 8.6).
    """
    valves = Pipeline.Valves()
    assert valves.presidio_enabled is True


def test_valves_loads_presidio_enabled_from_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`PII_FILTER_PRESIDIO_ENABLED=false` must flip the admin kill switch."""
    monkeypatch.setenv("PII_FILTER_PRESIDIO_ENABLED", "false")
    valves = Pipeline.Valves()
    assert valves.presidio_enabled is False


async def test_user_valves_disabled_returns_body_unchanged(
    pipeline: Pipeline, sample_user_body: dict[str, Any]
) -> None:
    """User opt-out path returns the body untouched: no message mutation,
    no metadata keys added by inlet.
    """
    pipeline.user_valves = Pipeline.UserValves(pii_masking_enabled=False)
    original_content = sample_user_body["messages"][-1]["content"]

    result = await pipeline.inlet(sample_user_body, user={"id": "user_42"})

    assert result is sample_user_body
    assert result["messages"][-1]["content"] == original_content
    metadata = result.get("metadata", {})
    assert "pii_placeholder_map" not in metadata
    assert "pii_reverse_map" not in metadata
    assert "pii_detections" not in metadata


async def test_user_valves_disabled_emits_info_log(
    pipeline: Pipeline,
    sample_user_body: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """User opt-out path emits exactly one structured INFO line with
    `user_id` and `chat_id` keys (AC 8.3).
    """
    pipeline.user_valves = Pipeline.UserValves(pii_masking_enabled=False)
    with caplog.at_level(logging.INFO, logger="pii_filter"):
        await pipeline.inlet(sample_user_body, user={"id": "user_42"})

    matches = [r for r in caplog.records if "user_disabled" in r.getMessage()]
    assert len(matches) == 1
    msg = matches[0].getMessage()
    assert "user_id=user_42" in msg
    assert "chat_id=test-chat-123" in msg


async def test_user_valves_disabled_no_vault_call(
    pipeline: Pipeline, sample_user_body: dict[str, Any]
) -> None:
    """User opt-out path must not invoke any vault method. We attach a
    spy `AsyncMock` and assert nothing was awaited (AC 8.2, decision #1).
    """
    pipeline.user_valves = Pipeline.UserValves(pii_masking_enabled=False)
    spy = AsyncMock()
    pipeline.vault = spy  # type: ignore[assignment]

    await pipeline.inlet(sample_user_body, user={"id": "user_42"})

    spy.snapshot_for_request.assert_not_awaited()
    spy.get_or_create_thread.assert_not_awaited()
    spy.get_placeholder.assert_not_awaited()
    spy.healthcheck.assert_not_awaited()


async def test_user_valves_disabled_no_metadata_written(
    pipeline: Pipeline,
) -> None:
    """User opt-out must not create `body['metadata']` when absent, and
    must not overwrite existing metadata keys.
    """
    pipeline.user_valves = Pipeline.UserValves(pii_masking_enabled=False)
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": "OIB je 12345678901"}],
        "metadata": {"chat_id": "ch_no_meta", "custom_key": "preserved"},
    }

    result = await pipeline.inlet(body, user={"id": "u1"})

    assert result["metadata"] == {"chat_id": "ch_no_meta", "custom_key": "preserved"}


async def test_user_valves_disabled_with_user_id_missing_uses_unknown(
    pipeline: Pipeline,
    sample_user_body: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When the `user` dict is None or lacks `id`, the audit log falls back
    to literal `unknown` (AC 8.3, §6.5 — never leak email/name).
    """
    pipeline.user_valves = Pipeline.UserValves(pii_masking_enabled=False)
    with caplog.at_level(logging.INFO, logger="pii_filter"):
        await pipeline.inlet(sample_user_body, user=None)

    matches = [r for r in caplog.records if "user_disabled" in r.getMessage()]
    assert len(matches) == 1
    assert "user_id=unknown" in matches[0].getMessage()


async def test_presidio_enabled_disabled_skips_analyzer_but_writes_metadata(
    pipeline: Pipeline,
) -> None:
    """`presidio_enabled=False` skips the analyzer loop (analyzer is None
    anyway in this unit-scope fixture) but still writes the metadata keys
    used by outlet for restoration symmetry (AC 8.8).
    """
    pipeline.valves.presidio_enabled = False
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": "OIB je 12345678901"}],
        "metadata": {"chat_id": "ch_presidio_off"},
    }

    result = await pipeline.inlet(body, user={"id": "user_42"})

    # Body content unchanged — analyzer never ran.
    assert result["messages"][-1]["content"] == "OIB je 12345678901"
    # Metadata keys present with empty maps (vault is None in unit scope).
    assert result["metadata"]["pii_placeholder_map"] == {}
    assert result["metadata"]["pii_reverse_map"] == {}


async def test_presidio_disabled_respects_vault_enabled_kill_switch(
    pipeline: Pipeline,
) -> None:
    """When `vault_enabled=False`, the presidio-disabled branch must NOT
    call `snapshot_for_request` — the admin vault kill switch overrides
    the outlet-symmetry snapshot pull. Mirrors the gating used by the
    normal inlet path and outlet (regression for Copilot review feedback).
    """
    pipeline.valves.presidio_enabled = False
    pipeline.valves.vault_enabled = False
    spy = AsyncMock()
    pipeline.vault = spy  # type: ignore[assignment]
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": "OIB je 12345678901"}],
        "metadata": {"chat_id": "ch_vault_disabled"},
    }

    result = await pipeline.inlet(body, user={"id": "user_42"})

    spy.snapshot_for_request.assert_not_awaited()
    # Metadata keys still written for outlet symmetry, but empty.
    assert result["metadata"]["pii_placeholder_map"] == {}
    assert result["metadata"]["pii_reverse_map"] == {}
