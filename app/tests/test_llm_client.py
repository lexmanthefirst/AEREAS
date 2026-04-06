"""Tests for LLMClient — validates init, availability, and structured output parsing."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.llm.client import LLMClient
from app.workers.schemas import GrammarOutput


def test_client_unavailable_without_api_key():
    with patch.dict("os.environ", {"GEMINI_API_KEY": ""}, clear=False):
        client = LLMClient(api_key="")
        assert client.available is False


def test_client_available_with_api_key():
    mock_genai = MagicMock()
    with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": mock_genai}):
        client = LLMClient(api_key="test-key-123")
        assert client.available is True


@pytest.mark.asyncio
async def test_generate_raises_when_unavailable():
    client = LLMClient(api_key="")
    with pytest.raises(RuntimeError, match="not available"):
        await client.generate(
            system_prompt="test",
            content="test",
        )


@pytest.mark.asyncio
async def test_generate_structured_output():
    """Verify structured output is parsed into a Pydantic model."""
    client = LLMClient(api_key="")
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
async def test_generate_plain_text():
    """Verify plain text output when no schema is provided."""
    client = LLMClient(api_key="")
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
    """Verify retry logic when LLM returns empty text."""
    client = LLMClient(api_key="")
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
