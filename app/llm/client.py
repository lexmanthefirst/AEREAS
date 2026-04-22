"""Shared async LLM client for all workers and services.

Backed by OpenRouter via the OpenAI-compatible Chat Completions API.
Structured output is requested with `response_format={"type": "json_schema", ...}`
and the raw JSON is validated against the caller's Pydantic schema.
"""

from __future__ import annotations

import asyncio
from typing import Any, Type, TypeVar

from pydantic import BaseModel

from app.core.config import settings
from app.utils.logger import logger

T = TypeVar("T", bound=BaseModel)

# Keywords Anthropic (via OpenRouter) rejects on `number`/`integer` fields in
# structured-output schemas. Pydantic still enforces these on our side when we
# validate the returned JSON, so stripping them from the wire schema is safe.
_UNSUPPORTED_SCHEMA_KEYS = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
)


def _sanitize_schema(schema: Any) -> Any:
    """Recursively remove schema keywords that some providers reject."""
    if isinstance(schema, dict):
        return {
            k: _sanitize_schema(v)
            for k, v in schema.items()
            if k not in _UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(schema, list):
        return [_sanitize_schema(item) for item in schema]
    return schema


class LLMClient:
    """Async wrapper around OpenRouter with structured output support.

    Initialised once during app lifespan and shared across all workers,
    the synthesis engine, the revision service, and the critic.
    """

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._api_key = api_key or settings.OPENROUTER_API_KEY
        self._default_model = default_model or settings.WORKER_MODEL_NAME
        self._base_url = base_url or settings.OPENROUTER_BASE_URL
        self._client: Any | None = None
        self._available = False
        self._extra_headers = self._build_extra_headers()
        self._init_client()

    def _build_extra_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if settings.OPENROUTER_SITE_URL:
            headers["HTTP-Referer"] = settings.OPENROUTER_SITE_URL
        if settings.OPENROUTER_APP_NAME:
            headers["X-Title"] = settings.OPENROUTER_APP_NAME
        return headers

    def _init_client(self) -> None:
        if not self._api_key:
            logger.warning("No OPENROUTER_API_KEY configured; LLM client unavailable.")
            return
        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
            self._available = True
            logger.info(
                "LLM client initialised (base_url=%s, model=%s).",
                self._base_url,
                self._default_model,
            )
        except Exception as exc:
            logger.warning("Could not initialise LLM client: %s", exc)

    @property
    def available(self) -> bool:
        return self._available and self._client is not None

    async def generate(
        self,
        *,
        system_prompt: str,
        content: str,
        response_schema: Type[T] | None = None,
        model_name: str | None = None,
        max_retries: int = 2,
    ) -> T | str:
        """Generate content via OpenRouter.

        Args:
            system_prompt: System-level instruction for the model.
            content: The user/document content to process.
            response_schema: If provided, requests structured JSON output and
                returns a validated Pydantic model instance.
            model_name: Override the default model for this call
                (e.g. ``"anthropic/claude-sonnet-4.5"``).
            max_retries: Retries on transient errors (429, 500, 503).

        Returns:
            A validated Pydantic model instance when *response_schema* is set,
            otherwise the raw response text.

        Raises:
            RuntimeError: If the client is not available.
            ValueError: If the model returns empty output.
        """
        if not self.available:
            raise RuntimeError("LLM client is not available (no API key or init failed).")

        model = model_name or self._default_model
        request_kwargs = self._build_request(system_prompt, content, response_schema)

        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 2):
            try:
                response = await asyncio.wait_for(
                    self._call(model, request_kwargs),
                    timeout=settings.LLM_REQUEST_TIMEOUT,
                )
                text = self._extract_text(response)
                if not text:
                    raise ValueError("LLM returned empty response.")

                if response_schema is not None:
                    return response_schema.model_validate_json(text)
                return text

            except (asyncio.TimeoutError, ValueError) as exc:
                last_exc = exc
                if attempt <= max_retries:
                    wait = 2 ** (attempt - 1)
                    logger.warning(
                        "LLM call attempt %d/%d failed (%s: %s), retrying in %ds...",
                        attempt,
                        max_retries + 1,
                        exc.__class__.__name__,
                        exc or "no message",
                        wait,
                    )
                    await asyncio.sleep(wait)
                continue

            except Exception as exc:
                err_str = str(exc).lower()
                is_transient = any(code in err_str for code in ("429", "500", "503", "rate"))
                if is_transient and attempt <= max_retries:
                    wait = 2 ** (attempt - 1)
                    logger.warning(
                        "LLM transient error attempt %d/%d (%s), retrying in %ds...",
                        attempt, max_retries + 1, exc, wait,
                    )
                    await asyncio.sleep(wait)
                    last_exc = exc
                    continue
                raise

        raise last_exc  # type: ignore[misc]

    async def _call(self, model: str, request_kwargs: dict[str, Any]) -> Any:
        """Execute the chat completion call."""
        return await self._client.chat.completions.create(
            model=model,
            extra_headers=self._extra_headers or None,
            **request_kwargs,
        )

    def _build_request(
        self,
        system_prompt: str,
        content: str,
        response_schema: Type[BaseModel] | None,
    ) -> dict[str, Any]:
        """Build the kwargs for the chat completion request."""
        kwargs: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        }
        if response_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema.__name__,
                    "strict": False,
                    "schema": _sanitize_schema(response_schema.model_json_schema()),
                },
            }
        return kwargs

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Pull the first choice's content off a chat completion response."""
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        return (content or "").strip()
