"""Tests for LLMClient — validates init, availability, and structured output parsing."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.llm.client import LLMClient
from app.workers.schemas import GrammarOutput


def _mock_chat_response(text: str) -> MagicMock:
    """Build a fake OpenAI ChatCompletion response with a single choice."""
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def test_client_unavailable_without_api_key():
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": ""}, clear=False):
        client = LLMClient(api_key="")
        assert client.available is False


def test_client_available_with_api_key():
    mock_openai_module = MagicMock()
    with patch.dict("sys.modules", {"openai": mock_openai_module}):
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
    client._client.chat.completions.create = AsyncMock(
        return_value=_mock_chat_response(
            '{"issues": [], "overall_assessment": "Good", "score": 90.0}'
        )
    )

    result = await client.generate(
        system_prompt="Check grammar",
        content="This is a test.",
        response_schema=GrammarOutput,
    )

    assert isinstance(result, GrammarOutput)
    assert result.score == 90.0
    assert result.issues == []

    # json_schema response_format should have been requested.
    call_kwargs = client._client.chat.completions.create.await_args.kwargs
    assert call_kwargs["response_format"]["type"] == "json_schema"
    assert call_kwargs["response_format"]["json_schema"]["name"] == "GrammarOutput"


@pytest.mark.asyncio
async def test_generate_plain_text():
    """Verify plain text output when no schema is provided."""
    client = LLMClient(api_key="")
    client._available = True
    client._client = MagicMock()
    client._client.chat.completions.create = AsyncMock(
        return_value=_mock_chat_response("The document is well written.")
    )

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
    client._client.chat.completions.create = AsyncMock(
        side_effect=[_mock_chat_response(""), _mock_chat_response("Valid response.")]
    )

    result = await client.generate(
        system_prompt="test",
        content="test",
        max_retries=1,
    )

    assert result == "Valid response."
    assert client._client.chat.completions.create.call_count == 2
