"""Tests for LLMClient: init, availability, provider routing, and parsing."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.llm.client import LLMClient
from app.workers.schemas import GrammarOutput


def test_client_unavailable_without_api_key():
    client = LLMClient(api_key="", provider="gemini")
    assert client.available is False


def test_gemini_client_available_with_api_key():
    mock_google = MagicMock()
    mock_genai = MagicMock()
    mock_google.genai = mock_genai
    with patch.dict("sys.modules", {"google": mock_google, "google.genai": mock_genai}):
        client = LLMClient(api_key="test-key-123", provider="gemini")
        assert client.available is True
        assert client.provider == "gemini"


def test_openrouter_client_available_with_api_key():
    client = LLMClient(api_key="test-key-123", provider="openrouter")
    assert client.available is True
    assert client.provider == "openrouter"


@pytest.mark.asyncio
async def test_generate_raises_when_unavailable():
    client = LLMClient(api_key="", provider="gemini")
    with pytest.raises(RuntimeError, match="not available"):
        await client.generate(system_prompt="test", content="test")


@pytest.mark.asyncio
async def test_generate_structured_output_gemini():
    client = LLMClient(api_key="", provider="gemini")
    client._available = True
    client._client = MagicMock()

    mock_response = MagicMock()
    mock_response.text = '{"issues": [], "overall_assessment": "Good", "score": 90.0}'
    client._client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    result = await client.generate(
        system_prompt="Check grammar",
        content="This is a test.",
        response_schema=GrammarOutput,
    )

    assert isinstance(result, GrammarOutput)
    assert result.score == 90.0
    assert result.issues == []


@pytest.mark.asyncio
async def test_generate_plain_text_gemini():
    client = LLMClient(api_key="", provider="gemini")
    client._available = True
    client._client = MagicMock()

    mock_response = MagicMock()
    mock_response.text = "The document is well written."
    client._client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    result = await client.generate(
        system_prompt="Review this",
        content="Some text.",
    )

    assert isinstance(result, str)
    assert "well written" in result


@pytest.mark.asyncio
async def test_generate_retries_on_empty_response():
    client = LLMClient(api_key="", provider="gemini")
    client._available = True
    client._client = MagicMock()

    empty_response = MagicMock()
    empty_response.text = ""

    good_response = MagicMock()
    good_response.text = "Valid response."

    client._client.aio.models.generate_content = AsyncMock(
        side_effect=[empty_response, good_response]
    )

    result = await client.generate(
        system_prompt="test",
        content="test",
        max_retries=1,
    )

    assert result == "Valid response."
    assert client._client.aio.models.generate_content.call_count == 2


@pytest.mark.asyncio
async def test_openrouter_structured_output():
    client = LLMClient(api_key="test-key-123", provider="openrouter")
    client._available = True
    client._client = MagicMock()

    response_payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "issues": [],
                        "overall_assessment": "Good",
                        "score": 92.0,
                    })
                }
            }
        ]
    }
    client._client.post = AsyncMock(return_value=MagicMock(
        json=MagicMock(return_value=response_payload),
        raise_for_status=MagicMock(),
    ))

    result = await client.generate(
        system_prompt="Check grammar",
        content="This is a test.",
        response_schema=GrammarOutput,
    )

    assert isinstance(result, GrammarOutput)
    assert result.score == 92.0
    assert client._client.post.await_count == 1
