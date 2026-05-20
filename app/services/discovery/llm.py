"""
LLM provider shim.

Both OpenAI and OpenRouter speak the OpenAI-compatible chat-completions API,
so the only thing that changes between providers is the base URL and the
model identifier. This module gives the discovery pipeline a single
`get_llm_client()` that hides the provider switch.

Precedence:
  1. OPENROUTER_API_KEY  → uses https://openrouter.ai/api/v1, default model
                           "openai/gpt-4o-mini" (overridable via LLM_MODEL).
  2. OPENAI_API_KEY      → uses OpenAI directly, default model "gpt-4o-mini".
  3. Neither set         → returns (None, None); callers must fall back.
"""
from __future__ import annotations

from openai import AsyncOpenAI

from app.core.config import settings


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def _is_real_key(key: str | None) -> bool:
    return bool(key) and not key.startswith("sk-your")


def get_llm_client() -> tuple[AsyncOpenAI | None, str | None]:
    """Returns (client, model) or (None, None) when no provider is configured."""
    if _is_real_key(settings.openrouter_api_key):
        client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=OPENROUTER_BASE_URL,
            default_headers={
                # Optional but recommended by OpenRouter for analytics.
                # HTTP headers must be ASCII — keep this string plain.
                "HTTP-Referer": "https://stan.app",
                "X-Title": "Stan Club Stanley Discovery",
            },
        )
        model = settings.llm_model or DEFAULT_OPENROUTER_MODEL
        return client, model

    if _is_real_key(settings.openai_api_key):
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        model = settings.llm_model or DEFAULT_OPENAI_MODEL
        return client, model

    return None, None


def has_llm() -> bool:
    return _is_real_key(settings.openrouter_api_key) or _is_real_key(settings.openai_api_key)
