"""Skeleton smoke tests for Task 1.

These tests verify the basic Pipeline class structure without any detection logic.
Detection-specific tests come in Tasks 3-7.
"""

from __future__ import annotations

from typing import Any

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
    await pipeline.on_startup()
    await pipeline.on_shutdown()
