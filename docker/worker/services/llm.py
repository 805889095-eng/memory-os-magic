"""
LLM client — remote OpenAI-compatible ``/chat/completions``.

Remote-only: there is no local inference backend. All generation is a remote
HTTP call to ``POST {LLM_API_BASE}/chat/completions``.
"""

import os
import logging

import httpx

logger = logging.getLogger("cognitive-worker.llm")

LLM_API_BASE = os.environ.get("LLM_API_BASE", "https://openrouter.ai/api/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek/deepseek-chat")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"


def _resolve_backend() -> tuple:
    """
    Resolve ``(base_url, api_key, model)`` for the remote chat endpoint.

    Fallback order (contract section 3):
      1. ``LLM_API_KEY`` on ``LLM_API_BASE``
      2. ``OPENROUTER_API_KEY`` when the base points at OpenRouter
      3. ``DEEPSEEK_API_KEY`` with base ``https://api.deepseek.com/v1``

    A missing key raises explicitly — never a silent unauthenticated request.
    """
    if LLM_API_KEY:
        return LLM_API_BASE, LLM_API_KEY, LLM_MODEL
    if "openrouter" in LLM_API_BASE.lower() and OPENROUTER_API_KEY:
        return LLM_API_BASE, OPENROUTER_API_KEY, LLM_MODEL
    if DEEPSEEK_API_KEY:
        return DEEPSEEK_API_BASE, DEEPSEEK_API_KEY, LLM_MODEL
    raise RuntimeError(
        "LLM_API_KEY is required. Fallbacks: OPENROUTER_API_KEY (when "
        "LLM_API_BASE points at OpenRouter) or DEEPSEEK_API_KEY. "
        "No API key is configured."
    )


async def chat_completion(
    prompt: str,
    model: str | None = None,
    timeout: int = 120,
) -> str:
    """
    Sends a prompt to a remote OpenAI-compatible ``/chat/completions`` endpoint
    and returns ``choices[0].message.content``.
    """
    base_url, api_key, default_model = _resolve_backend()
    model = model or default_model

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"LLM returned no choices: {str(data)[:200]}")

    message = choices[0].get("message") or {}
    return message.get("content") or ""


# ─── Backwards-compatible alias ─────────────────────────────────────────────
# Legacy chat-helper name retained on purpose (contract section 3): existing
# callers that import it keep working, and it points at the remote
# OpenAI-compatible implementation above. Contract-mandated, do not remove.
ollama_chat = chat_completion
