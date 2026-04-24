"""Shared async LLM client for all workers and services."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Type, TypeVar

import httpx
from pydantic import BaseModel

from app.core.config import settings
from app.utils.logger import logger

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Provider-aware async LLM client with structured output support.

    The client supports Gemini directly and OpenRouter through its OpenAI-
    compatible chat completions interface. It is initialized once during app
    lifespan and shared across workers, synthesis, revision, and critic flows.
    """

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str | None = None,
        provider: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._provider = (provider or settings.LLM_PROVIDER or "gemini").strip().lower()
        self._default_model = default_model or settings.WORKER_MODEL_NAME
        self._client: Any | None = None
        self._available = False

        self._api_key = api_key or self._resolve_api_key(self._provider)
        self._base_url = base_url or settings.OPENROUTER_BASE_URL

        self._init_client()

    def _resolve_api_key(self, provider: str) -> str | None:
        if provider == "openrouter":
            return settings.OPENROUTER_API_KEY
        if provider == "auto":
            return settings.OPENROUTER_API_KEY or settings.GEMINI_API_KEY
        return settings.GEMINI_API_KEY

    def _effective_provider(self) -> str:
        if self._provider != "auto":
            return self._provider
        if settings.OPENROUTER_API_KEY:
            return "openrouter"
        return "gemini"

    def _init_client(self) -> None:
        provider = self._effective_provider()
        if not self._api_key:
            logger.warning("No API key configured for provider '%s'; LLM client unavailable.", provider)
            return

        try:
            if provider == "openrouter":
                self._client = httpx.AsyncClient(
                    base_url=self._base_url.rstrip("/"),
                    headers=self._openrouter_headers(),
                    timeout=settings.LLM_REQUEST_TIMEOUT,
                )
                self._available = True
                logger.info("LLM client initialised (provider=openrouter, model=%s).", self._default_model)
                return

            from google import genai

            self._client = genai.Client(api_key=self._api_key)
            self._available = True
            logger.info("LLM client initialised (provider=gemini, model=%s).", self._default_model)
        except Exception as exc:
            logger.warning("Could not initialise LLM client for provider '%s': %s", provider, exc)

    def _openrouter_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if settings.OPENROUTER_HTTP_REFERER:
            headers["HTTP-Referer"] = settings.OPENROUTER_HTTP_REFERER
        if settings.OPENROUTER_X_TITLE:
            headers["X-Title"] = settings.OPENROUTER_X_TITLE
        return headers

    @property
    def available(self) -> bool:
        return self._available and self._client is not None

    @property
    def provider(self) -> str:
        return self._effective_provider()

    async def aclose(self) -> None:
        if self.provider == "openrouter" and isinstance(self._client, httpx.AsyncClient):
            await self._client.aclose()

    async def generate(
        self,
        *,
        system_prompt: str,
        content: str,
        response_schema: Type[T] | None = None,
        model_name: str | None = None,
        max_retries: int = 2,
    ) -> T | str:
        """Generate content from the configured provider.

        Args:
            system_prompt: System-level instruction for the model.
            content: The user/document content to process.
            response_schema: Optional Pydantic schema for structured output.
            model_name: Optional per-call model override.
            max_retries: Retry count for transient failures.
        """
        if not self.available:
            raise RuntimeError("LLM client is not available (no API key or init failed).")

        model = self._normalize_model_name(model_name or self._default_model)
        config = self._build_config(system_prompt, response_schema)

        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 2):
            try:
                response = await asyncio.wait_for(
                    self._call(model, content, config),
                    timeout=settings.LLM_REQUEST_TIMEOUT,
                )
                text = self._extract_text(response).strip()
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
                is_transient = any(code in err_str for code in ("429", "500", "503", "rate", "timeout"))
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
        if self.provider == "openrouter":
            return await self._call_openrouter(model, content, config)
        return await self._client.aio.models.generate_content(  # type: ignore[union-attr]
            model=model,
            contents=content,
            config=config,
        )

    async def _call_openrouter(self, model: str, content: str, config: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": config["system_instruction"]},
                {"role": "user", "content": content},
            ],
        }

        response_schema = config.get("response_schema")
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema.__name__,
                    "strict": True,
                    "schema": response_schema.model_json_schema(),
                },
            }

        response = await self._client.post("/chat/completions", json=payload)  # type: ignore[union-attr]
        response.raise_for_status()
        return response.json()

    def _extract_text(self, response: Any) -> str:
        if self.provider != "openrouter":
            return getattr(response, "text", "") or ""

        if not isinstance(response, dict):
            return ""

        choices = response.get("choices") or []
        if not choices:
            return ""

        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    text_parts.append(item)
            return "".join(text_parts)
        if isinstance(content, dict):
            return json.dumps(content)
        return str(content)

    def _normalize_model_name(self, model: str) -> str:
        if self.provider == "openrouter" and "/" not in model:
            if model.startswith("gemini"):
                return f"google/{model}"
            if model.startswith("gpt-"):
                return f"openai/{model}"
        return model

    def _build_config(
        self,
        system_prompt: str,
        response_schema: Type[BaseModel] | None,
    ) -> dict[str, Any]:
        config: dict[str, Any] = {
            "system_instruction": system_prompt,
        }
        if response_schema is not None:
            config["response_schema"] = response_schema
            if self.provider != "openrouter":
                config["response_mime_type"] = "application/json"
        return config
