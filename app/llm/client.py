"""Shared async LLM client for all workers and services."""

from __future__ import annotations

import asyncio
from typing import Any, Type, TypeVar

from pydantic import BaseModel

from app.core.config import settings
from app.utils.logger import logger

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Async wrapper around google-genai with structured output support.

    Initialised once during app lifespan and shared across all workers,
    the synthesis engine, the revision service, and the critic.
    """

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str | None = None,
    ) -> None:
        self._api_key = api_key or settings.GEMINI_API_KEY
        self._default_model = default_model or settings.WORKER_MODEL_NAME
        self._client: Any | None = None
        self._available = False
        self._init_client()

    def _init_client(self) -> None:
        if not self._api_key:
            logger.warning("No GEMINI_API_KEY configured; LLM client unavailable.")
            return
        try:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
            self._available = True
            logger.info("LLM client initialised (model=%s).", self._default_model)
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
        """Generate content via Gemini.

        Args:
            system_prompt: System-level instruction for the model.
            content: The user/document content to process.
            response_schema: If provided, requests structured JSON output and
                returns a validated Pydantic model instance.
            model_name: Override the default model for this call.
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
        config = self._build_config(system_prompt, response_schema)

        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 2):
            try:
                response = await asyncio.wait_for(
                    self._call(model, content, config),
                    timeout=settings.LLM_REQUEST_TIMEOUT,
                )
                text = (getattr(response, "text", "") or "").strip()
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
                        "LLM call attempt %d/%d failed (%s), retrying in %ds...",
                        attempt, max_retries + 1, exc, wait,
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

    async def _call(self, model: str, content: str, config: dict[str, Any]) -> Any:
        """Execute the actual API call using the async interface."""
        return await self._client.aio.models.generate_content(
            model=model,
            contents=content,
            config=config,
        )

    def _build_config(
        self,
        system_prompt: str,
        response_schema: Type[BaseModel] | None,
    ) -> dict[str, Any]:
        """Build the generation config dict."""
        config: dict[str, Any] = {
            "system_instruction": system_prompt,
        }
        if response_schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = response_schema
        return config
