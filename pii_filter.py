"""
title: PII Filter
author: Keeper Solutions AI Lab
author_url: https://github.com/keeper-solutions/pii-filter
date: 2026-04-28
version: 0.1.0
license: MIT
description: PII detection and masking filter for Keeper AI Gateway. Skeleton (Task 1) — detection logic comes in Tasks 3-7.
requirements:
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

# Logging setup
logger = logging.getLogger(__name__)


class Pipeline:
    """PII Filter pipeline — Keeper AI Gateway.

    Skeleton implementation. Inlet/outlet are pass-through (echo).
    Detection logic to be added in Tasks 3-7.
    """

    class Valves(BaseModel):
        """Admin-configurable settings (visible in OpenWebUI Admin → Pipelines).

        These can be edited from the UI without re-uploading the file.
        """

        # Models this filter applies to. ["*"] = all models.
        pipelines: list[str] = ["*"]

        # Execution priority (lower = earlier). 0 = normal.
        # PII filter should run BEFORE Langfuse trace pipelines so trace doesn't log raw PII.
        priority: int = 0

        # Global kill switch. If False, inlet/outlet are no-op.
        enabled: bool = True

        # Languages supported by detection (used in Tasks 3-7).
        # Default HR + EN per Dokument 1 sekcija 5.1 scope.
        languages: list[str] = ["hr", "en"]

        # Degradation mode if detection engine fails (used in Task 8).
        # "block" = return error to user (default, GDPR-safe)
        # "passthrough" = let request through unmodified (logs warning)
        degradation_mode: str = "block"

    def __init__(self) -> None:
        """Initialize the pipeline."""
        # Required by Open WebUI Pipelines protocol
        self.type = "filter"
        self.name = "PII Filter"

        # Initialize Valves with defaults
        self.valves = self.Valves()

        logger.info("PII Filter pipeline initialized (skeleton, no detection logic yet)")

    async def on_startup(self) -> None:
        """Called when Pipelines container starts."""
        logger.info("PII Filter on_startup")
        # Future: load Presidio analyzer, spaCy models, connect to Redis (Tasks 3, 5)

    async def on_shutdown(self) -> None:
        """Called when Pipelines container stops."""
        logger.info("PII Filter on_shutdown")
        # Future: close Redis connections, flush metrics

    async def inlet(
        self, body: dict[str, Any], user: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Process incoming chat request BEFORE it reaches the LLM.

        In Task 4, this will mask PII in user messages.
        For now (Task 1 skeleton), it returns the body unchanged (echo).

        Args:
            body: Chat completion request body (OpenAI format).
                  Contains "messages" list, "model", "metadata", etc.
            user: User info dict (id, email, role, etc.). May be None.

        Returns:
            Modified body. For skeleton, identical to input.
        """
        if not self.valves.enabled:
            return body

        # TODO Task 4: detect PII in body["messages"], replace with placeholders, store mapping
        return body

    async def outlet(
        self, body: dict[str, Any], user: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Process LLM response BEFORE it reaches the user.

        In Task 6, this will restore original PII values from placeholders.
        For now (Task 1 skeleton), it returns the body unchanged (echo).

        Args:
            body: Chat completion response body. Contains "messages" with assistant reply.
            user: User info dict.

        Returns:
            Modified body. For skeleton, identical to input.
        """
        if not self.valves.enabled:
            return body

        # TODO Task 6: parse placeholders in assistant messages, restore from vault
        return body
