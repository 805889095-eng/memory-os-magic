"""
Embedding client — remote-only OpenAI-compatible ``/embeddings``.

Calls ``POST {EMBEDDING_API_BASE}/embeddings`` (default: OpenRouter). This client
is remote-only: there is no local inference backend.
An API key is mandatory; a missing key is a hard error, never a silent
unauthenticated request. Mandatory dimension validation with in-memory LRU cache.
"""

import os
import logging
from collections import OrderedDict

import httpx

logger = logging.getLogger("cognitive-worker.embedding")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "")
EMBEDDING_DIMS = int(os.environ.get("EMBEDDING_DIMS", "4096"))
EMBEDDING_API_BASE = os.environ.get(
    "EMBEDDING_API_BASE", "https://openrouter.ai/api/v1"
)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")
EMBEDDING_PASS_DIMENSIONS = os.environ.get("EMBEDDING_PASS_DIMENSIONS", "auto").lower()

# In-memory LRU cache — avoids recomputing embeddings for repeated text
_EMBEDDING_CACHE_MAXSIZE = 256
_embedding_cache: OrderedDict = OrderedDict()


def _resolve_api_key() -> str:
    """
    Resolve the bearer token for the remote embedding endpoint.

    Compat: when the base URL points at OpenRouter, fall back to
    OPENROUTER_API_KEY. For any other remote base, EMBEDDING_API_KEY is
    mandatory — a missing key raises explicitly instead of issuing an
    unauthenticated request.
    """
    if "openrouter" in EMBEDDING_API_BASE.lower():
        key = EMBEDDING_API_KEY or OPENROUTER_API_KEY
        if not key:
            raise RuntimeError(
                "EMBEDDING_API_KEY (or OPENROUTER_API_KEY) is required for the "
                f"OpenRouter embedding endpoint {EMBEDDING_API_BASE}"
            )
        return key
    if not EMBEDDING_API_KEY:
        raise RuntimeError(
            "EMBEDDING_API_KEY is required for the remote embedding endpoint "
            f"{EMBEDDING_API_BASE} (no fallback key is configured)"
        )
    return EMBEDDING_API_KEY


async def get_embedding(text: str) -> list:
    """
    Generates embedding via the configured remote backend.
    Validates that the returned dimensions match EMBEDDING_DIMS.
    Results are cached in-memory (LRU, max 256 entries).
    """
    # Check cache
    if text in _embedding_cache:
        _embedding_cache.move_to_end(text)
        return _embedding_cache[text]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_resolve_api_key()}",
    }
    if "openrouter" in EMBEDDING_API_BASE.lower():
        headers["HTTP-Referer"] = "https://localhost"
        headers["X-Title"] = "Cognitive-Agent-MaaS"

    payload: dict = {
        "model": EMBEDDING_MODEL,
        "input": text,
    }
    # Pass dimensions only when configured, or auto for models that support it
    should_pass_dims = EMBEDDING_PASS_DIMENSIONS == "true" or (
        EMBEDDING_PASS_DIMENSIONS == "auto" and "text-embedding-3" in EMBEDDING_MODEL.lower()
    )
    if should_pass_dims:
        payload["dimensions"] = EMBEDDING_DIMS

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{EMBEDDING_API_BASE}/embeddings",
            headers=headers,
            json=payload,
        )
        if resp.status_code == 400 and ("20015" in resp.text or "dimensions" in resp.text.lower()) and "dimensions" in payload:
            logger.warning("Embedding endpoint rejected 'dimensions' (e.g. SiliconFlow 20015), retrying without it.")
            payload.pop("dimensions", None)
            resp = await client.post(
                f"{EMBEDDING_API_BASE}/embeddings",
                headers=headers,
                json=payload,
            )
        resp.raise_for_status()
        data = resp.json()

    vec = data["data"][0]["embedding"]

    # ─── Critical dimension validation ──────────────────────────────────────
    if len(vec) != EMBEDDING_DIMS:
        raise ValueError(
            f"Embedding dimension mismatch: "
            f"expected {EMBEDDING_DIMS}, got {len(vec)}. "
            f"Check EMBEDDING_DIMS in .env and the Qdrant collection."
        )

    logger.debug(f"Embedding generated: {len(vec)} dims")

    # Store in LRU cache
    _embedding_cache[text] = vec
    if len(_embedding_cache) > _EMBEDDING_CACHE_MAXSIZE:
        _embedding_cache.popitem(last=False)

    return vec
