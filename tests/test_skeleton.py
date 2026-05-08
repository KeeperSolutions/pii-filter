"""Skeleton smoke tests for Task 1.

These tests verify the basic Pipeline class structure without any detection logic.
Detection-specific tests come in Tasks 3-7.
"""

from __future__ import annotations

from typing import Any

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
    assert valves.languages == ["hr"]
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
    await pipeline.on_startup()
    await pipeline.on_shutdown()
