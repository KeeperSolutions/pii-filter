"""Pytest fixtures for PII Filter tests."""

from __future__ import annotations

from typing import Any

import pytest

from pii_filter import Pipeline


@pytest.fixture
def pipeline() -> Pipeline:
    """Fresh Pipeline instance for each test."""
    return Pipeline()


@pytest.fixture
def sample_user_body() -> dict[str, Any]:
    """Sample chat completion request body (OpenAI format)."""
    return {
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "Zovem se Ivan Horvat, OIB: 12345678901."},
        ],
        "metadata": {"chat_id": "test-chat-123"},
    }


@pytest.fixture
def sample_assistant_body() -> dict[str, Any]:
    """Sample chat completion response body."""
    return {
        "model": "gpt-4",
        "messages": [
            {"role": "assistant", "content": "Bok [PERSON_1], kako vam mogu pomoći?"},
        ],
    }
