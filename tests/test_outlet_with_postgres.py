"""Task 5.1 contract preservation — outlet runs unchanged after Postgres inlet.

The single most load-bearing AC of Task 5.1 is that `Pipeline.outlet` keeps
working without modification when `inlet` used the Postgres backend instead
of Redis. The outlet only ever reads `body["metadata"]["pii_reverse_map"]`,
which both backends populate to the same `dict[str, str]` shape via
`snapshot_for_request`. This module pins that contract end-to-end.

Requires `pg_ctl` / `postgres` on PATH (skipped otherwise).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.conftest import postgres_binary_missing

if TYPE_CHECKING:
    from pii_filter import Pipeline


pytestmark = [
    pytest.mark.skipif(
        postgres_binary_missing,
        reason="pg_ctl / postgres binary not on PATH; skipping Postgres-backed tests",
    ),
]


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


@pytest.mark.asyncio(loop_scope="module")
async def test_outlet_after_postgres_inlet_round_trip(
    started_pipeline_postgres: Pipeline,
) -> None:
    """End-to-end: real OIB → Postgres-backed inlet masks → simulate the
    LLM echoing the placeholder → outlet restores the original. Outlet code
    path is byte-identical to Task 6; only the inlet's vault backend changed."""
    oib = _make_oib("3334445550")
    chat_id = "task5.1-outlet-roundtrip"

    request_body: dict[str, Any] = {
        "chat_id": chat_id,
        "messages": [{"role": "user", "content": f"Moj OIB je {oib}."}],
    }
    request_body = await started_pipeline_postgres.inlet(request_body)

    masked_user = request_body["messages"][-1]["content"]
    assert oib not in masked_user
    placeholder = request_body["metadata"]["pii_placeholder_map"][oib]
    assert placeholder in masked_user

    # Construct an OpenAI-shaped completion response containing the
    # placeholder. Carry the inlet's metadata forward — that's what
    # Pipelines does in production for outlet.
    response_body: dict[str, Any] = {
        "chat_id": chat_id,
        "metadata": request_body["metadata"],
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Vaš OIB ({placeholder}) je u redu.",
                },
                "finish_reason": "stop",
            }
        ],
    }

    response_body = await started_pipeline_postgres.outlet(response_body)
    restored_text: str = response_body["choices"][0]["message"]["content"]

    assert oib in restored_text
    assert placeholder not in restored_text


@pytest.mark.asyncio(loop_scope="module")
async def test_outlet_after_postgres_inlet_reverse_map_is_dict_str_str(
    started_pipeline_postgres: Pipeline,
) -> None:
    """Pin the `body["metadata"]["pii_reverse_map"]` contract: outlet's
    runtime expectation is `dict[str, str]`, populated by the Postgres
    `snapshot_for_request` call inside inlet. This is the assertion that
    catches a future regression where `snapshot_for_request` returns the
    wrong shape (e.g. tuples, asyncpg.Records, anything but `dict[str, str]`)."""
    oib = _make_oib("2223334440")
    chat_id = "task5.1-outlet-reverse-map-shape"

    body: dict[str, Any] = {
        "chat_id": chat_id,
        "messages": [{"role": "user", "content": f"OIB {oib}"}],
    }
    body = await started_pipeline_postgres.inlet(body)

    reverse_map = body["metadata"]["pii_reverse_map"]
    assert isinstance(reverse_map, dict)
    assert reverse_map, "expected at least one entry after a successful inlet"
    for k, v in reverse_map.items():
        assert isinstance(k, str), f"reverse_map key must be str, got {type(k).__name__}"
        assert isinstance(v, str), f"reverse_map value must be str, got {type(v).__name__}"
        assert k.startswith("[") and k.endswith("]"), f"placeholder shape broken: {k}"


@pytest.mark.asyncio(loop_scope="module")
async def test_outlet_falls_back_to_vault_when_metadata_missing(
    started_pipeline_postgres: Pipeline,
) -> None:
    """Real OpenWebUI Pipelines serves inlet and outlet as separate HTTP
    endpoints — the outlet body has no `metadata` key. Outlet must fall
    back to `vault.snapshot_for_request(chat_id)` to recover the reverse
    map and successfully restore placeholders. Without this fallback,
    placeholders leak through to the user-facing UI."""
    oib = _make_oib("4445556660")
    chat_id = "task5.1-outlet-vault-fallback"

    request_body: dict[str, Any] = {
        "chat_id": chat_id,
        "messages": [{"role": "user", "content": f"Moj OIB je {oib}."}],
    }
    request_body = await started_pipeline_postgres.inlet(request_body)
    placeholder = request_body["metadata"]["pii_placeholder_map"][oib]

    # Simulate the real Pipelines outlet body shape: NO metadata key.
    response_body: dict[str, Any] = {
        "chat_id": chat_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Vaš OIB ({placeholder}) je u redu.",
                },
                "finish_reason": "stop",
            }
        ],
    }
    assert "metadata" not in response_body

    response_body = await started_pipeline_postgres.outlet(response_body)
    restored_text: str = response_body["choices"][0]["message"]["content"]

    assert oib in restored_text, "vault fallback failed to restore placeholder"
    assert placeholder not in restored_text


@pytest.mark.asyncio(loop_scope="module")
async def test_outlet_prefers_metadata_when_both_available(
    started_pipeline_postgres: Pipeline,
) -> None:
    """When the outlet body carries `metadata.pii_reverse_map`, outlet must
    use it directly and skip the vault snapshot call. Pinned by feeding a
    `metadata.pii_reverse_map` whose entries do NOT match what the Postgres
    vault holds — restoration must use the metadata-supplied mapping
    (proves vault was not consulted)."""
    oib = _make_oib("5556667770")
    chat_id = "task5.1-outlet-metadata-priority"

    request_body: dict[str, Any] = {
        "chat_id": chat_id,
        "messages": [{"role": "user", "content": f"Moj OIB je {oib}."}],
    }
    request_body = await started_pipeline_postgres.inlet(request_body)
    placeholder = request_body["metadata"]["pii_placeholder_map"][oib]

    # Override metadata reverse_map with a sentinel value distinguishable
    # from what the vault would return for the same placeholder.
    sentinel = "FROM-METADATA-NOT-VAULT"
    overridden_metadata = {
        **request_body["metadata"],
        "pii_reverse_map": {placeholder: sentinel},
    }

    response_body: dict[str, Any] = {
        "chat_id": chat_id,
        "metadata": overridden_metadata,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Vaš OIB ({placeholder}) je u redu.",
                },
                "finish_reason": "stop",
            }
        ],
    }

    response_body = await started_pipeline_postgres.outlet(response_body)
    restored_text: str = response_body["choices"][0]["message"]["content"]

    assert sentinel in restored_text, "outlet must prefer body metadata over vault"
    assert oib not in restored_text, "vault must not have been consulted"
    assert placeholder not in restored_text
