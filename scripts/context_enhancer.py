#!/usr/bin/env python3
"""
Context Enhancer — HYBRID search (semantic + BM25) on the configured collection
for prompt enrichment.

Runs as a synchronous function (fast, <1s) before each Hermes response.
If Qdrant is offline or embedding fails, returns "" (fail-open).

Zero local models: dense embedding is remote-only (OpenAI-compatible
`POST {EMBEDDING_API_BASE}/embeddings`) and the sparse BM25 vector is pure
Python (`scripts/bm25_tf.py`, no model, no subprocess, no network).

Dense embedding fails loudly: a missing API key raises RuntimeError instead of
silently degrading to keyword-only retrieval.

Usage:
  python3 context_enhancer.py "your query here"
  python3 context_enhancer.py --top-k 5 --threshold 0.50 "deploy docker"
  python3 context_enhancer.py --hybrid-off "your query here"   # forces dense-only
"""

import os
import sys
import json
import uuid
import hashlib
import sqlite3
import requests
import argparse
import re
import glob
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import time
from datetime import datetime, timezone

# Pure-Python BM25 tf (shared verbatim with docker/worker/services/bm25_tf.py).
# Works both as a standalone script and as `from scripts.context_enhancer import ...`.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from bm25_tf import bm25_tf  # noqa: E402

# ─── Config ────────────────────────────────────────────────────────────────
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "knowledge_base")

# Dense embedding — remote only (unified config contract)
EMBEDDING_API_BASE = os.environ.get(
    "EMBEDDING_API_BASE", "https://openrouter.ai/api/v1"
)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")
EMBEDDING_DIMS = int(os.environ.get("EMBEDDING_DIMS", "4096"))
EMBEDDING_PASS_DIMENSIONS = os.environ.get("EMBEDDING_PASS_DIMENSIONS", "auto").lower()
TOP_K_DEFAULT = 3
SCORE_THRESHOLD_DEFAULT = 0.55
MAX_TEXT_LEN = 8000
REQUEST_TIMEOUT = 10

# Sparse BM25 hyper-parameters (Qdrant applies the IDF modifier server-side).
BM25_K1 = float(os.environ.get("BM25_K1", "1.2"))
BM25_B = float(os.environ.get("BM25_B", "0.75"))
BM25_AVG_LEN = float(os.environ.get("BM25_AVG_LEN", "256.0"))

# Lineage config
LINEAGE_DB = os.environ.get(
    "STATE_DB_PATH",
    os.path.join(
        os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")), "state.db"
    ),
)

# Telemetry config
TELEMETRY_LOG = os.environ.get(
    "TELEMETRY_LOG_PATH",
    os.path.join(
        os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")),
        "logs",
        "query-telemetry.jsonl",
    ),
)
TELEMETRY_MAX_BYTES = 10 * 1024 * 1024  # 10MB rotation

# ─── Lineage Registration ───────────────────────────────────────────────────


def register_lineage(
    session_id: str,
    query: str,
    retrieved_chunk_ids: List[str],
    generation_context_hash: str,
    generation_model: str = "unknown",
) -> Optional[str]:
    """
    Register generation provenance in the lineage DB.
    Fail-open: if it fails, log error and return None. Never breaks the critical path.
    """
    lineage_id = str(uuid.uuid4())
    try:
        with sqlite3.connect(LINEAGE_DB) as conn:
            conn.execute(
                """
                INSERT INTO lineage (lineage_id, session_id, query, retrieved_chunk_ids,
                                     generation_model, generation_context_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    lineage_id,
                    session_id,
                    query,
                    json.dumps(retrieved_chunk_ids, ensure_ascii=False),
                    generation_model,
                    generation_context_hash,
                ),
            )
            conn.commit()
        return lineage_id
    except Exception as e:
        print(f"[LINEAGE-WARNING] Failed to register lineage: {e}", file=sys.stderr)
        return None


# ─── Telemetry ─────────────────────────────────────────────────────────────


def _rotate_telemetry_if_needed() -> None:
    """Rotate telemetry file if >10MB. Rename to .1 and restart."""
    try:
        if (
            os.path.exists(TELEMETRY_LOG)
            and os.path.getsize(TELEMETRY_LOG) > TELEMETRY_MAX_BYTES
        ):
            rotated = TELEMETRY_LOG + ".1"
            if os.path.exists(rotated):
                os.remove(rotated)
            os.rename(TELEMETRY_LOG, rotated)
    except Exception as e:
        print(f"[TELEMETRY-WARNING] Rotation failed: {e}", file=sys.stderr)


def emit_telemetry(record: dict) -> None:
    """
    Append-only JSONL for query telemetry.
    Fail-open: never breaks the critical path.
    """
    try:
        os.makedirs(os.path.dirname(TELEMETRY_LOG), exist_ok=True)
        _rotate_telemetry_if_needed()
        with open(TELEMETRY_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[TELEMETRY-WARNING] Failed to write telemetry: {e}", file=sys.stderr)


def estimate_tokens(text: str) -> int:
    """Simple estimate: ~1.3 tokens per word (basic heuristic)."""
    return int(len(text.split()) * 1.3)


# ─── Prompt Injection Sanitization ──────────────────────────────────────────

_INJECTION_PATTERNS_CE = [
    # "ignore all previous/prior instructions/directives"
    (
        re.compile(
            r"(?i)\bignore\s+all\s+(previous|prior)\s+(instructions|directives|commands|messages|prompts|context)"
        ),
        "[REDACTED]",
    ),
    # "you are/will now become/act/acting as (a/an) AI/assistant..."
    (
        re.compile(
            r"(?i)\byou\s+(are|will\s+now)\s+(now\s+)?(become|act|acting)\s+as\s+(a\s+|an\s+)?(AI\s+assistant|assistant|AI|agent|LLM|chatbot|model|system)"
        ),
        "[REDACTED]",
    ),
    # "new instructions/directives/commands follow/above/below"
    (
        re.compile(
            r"(?i)\bnew\s+(instructions|directives|commands)\s+(follow|above|below)"
        ),
        "[REDACTED]",
    ),
    # Template injection: {{...}}, ${...}
    (re.compile(r"\{\{.*?\}\}|\$\{.*?\}"), "[REDACTED]"),
    # Markdown/javascript data: URLs in links and images
    (re.compile(r"(?i)(javascript|data)\s*:"), "sanitized:"),
    # Known system prefixes
    (re.compile(r"(?i)\[IMPORTANT:.*?\]|\[SYSTEM:.*?\]"), "[REDACTED]"),
    # Control characters (keep newlines and tabs)
    (re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"), ""),
]


def _strip_prompt_injection(text: str) -> str:
    """Strip known prompt-injection patterns from retrieved text.
    Fail-open: never raises. Empty/null input returns empty string.
    """
    if not text:
        return ""
    try:
        result = str(text)
        for pattern, replacement in _INJECTION_PATTERNS_CE:
            result = pattern.sub(replacement, result)
        # Normalize excessive whitespace (4+ newlines → 3, 8+ spaces → 1)
        result = re.sub(r"\n{4,}", "\n\n\n", result)
        result = re.sub(r" {8,}", " ", result)
        return result.strip()
    except Exception:
        return str(text)[:400]


# ─── Core ───────────────────────────────────────────────────────────────────


def resolve_embedding_api_key() -> Optional[str]:
    """Resolve the embedding API key at call time.

    Order: EMBEDDING_API_KEY → OPENROUTER_API_KEY (only when the base points at
    OpenRouter) → ``~/.env`` (host deployments keep the key there).
    Resolved per call so runtime env mutations (e.g. the Icarus hook exporting
    OPENROUTER_API_KEY right before invoking us) are honoured.
    """
    key = os.environ.get("EMBEDDING_API_KEY") or ""
    if key:
        return key
    if "openrouter" not in EMBEDDING_API_BASE.lower():
        return None
    key = os.environ.get("OPENROUTER_API_KEY") or ""
    if key:
        return key
    env_path = Path.home() / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"')
        except OSError as e:
            print(f"[CE-ERROR] Could not read {env_path}: {e}", file=sys.stderr)
    return None


def embed_query(text: str) -> List[float]:
    """Generate a dense embedding via the remote OpenAI-compatible endpoint.

    Remote-only, no local model, no silent degradation:
      * missing key  → RuntimeError
      * wrong dims   → ValueError
      * HTTP error   → requests.HTTPError (propagated)
    """
    api_key = resolve_embedding_api_key()
    if not api_key:
        raise RuntimeError(
            "Dense embedding unavailable: EMBEDDING_API_KEY is not configured "
            "(nor OPENROUTER_API_KEY for an OpenRouter base). Set one of them in "
            "the environment / .env — local embedding models are no longer supported."
        )

    payload: dict = {
        "model": EMBEDDING_MODEL,
        "input": text[:MAX_TEXT_LEN],
    }
    should_pass_dims = EMBEDDING_PASS_DIMENSIONS == "true" or (
        EMBEDDING_PASS_DIMENSIONS == "auto" and "text-embedding-3" in EMBEDDING_MODEL.lower()
    )
    if should_pass_dims:
        payload["dimensions"] = EMBEDDING_DIMS

    resp = requests.post(
        f"{EMBEDDING_API_BASE.rstrip('/')}/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code == 400 and ("20015" in resp.text or "dimensions" in resp.text.lower()) and "dimensions" in payload:
        payload.pop("dimensions", None)
        resp = requests.post(
            f"{EMBEDDING_API_BASE.rstrip('/')}/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    resp.raise_for_status()
    vector = resp.json()["data"][0]["embedding"]
    if len(vector) != EMBEDDING_DIMS:
        raise ValueError(
            f"Embedding dimension mismatch: EMBEDDING_DIMS={EMBEDDING_DIMS}, "
            f"API returned {len(vector)}"
        )
    return vector


def embed_query_sparse(text: str) -> Dict[str, List]:
    """Generate the sparse BM25 vector as a plain Python dict.

    Returns ``{"indices": [int, ...], "values": [float, ...]}`` (Qdrant
    compatible), indices deduplicated and ascending. Pure Python — no model, no
    subprocess, no network. IDF is applied by Qdrant's Modifier.IDF, never here.
    """
    try:
        return bm25_tf(text, k1=BM25_K1, b=BM25_B, avg_len=BM25_AVG_LEN)
    except Exception as e:
        # Never silent — bm25_tf only raises on programming errors.
        print(f"[CE-ERROR] Sparse BM25 generation failed: {e}", file=sys.stderr)
        raise


def _normalize_sparse(sparse_vector) -> Optional[Tuple[List[int], List[float]]]:
    """Accept either the canonical dict or a legacy (indices, values) tuple."""
    if sparse_vector is None:
        return None
    if isinstance(sparse_vector, dict):
        indices = sparse_vector.get("indices") or []
        values = sparse_vector.get("values") or []
    else:
        try:
            indices, values = sparse_vector[0], sparse_vector[1]
        except (TypeError, IndexError, KeyError):
            print(
                "[CE-ERROR] Invalid sparse vector (expected dict or 2-tuple), ignoring.",
                file=sys.stderr,
            )
            return None
    if not indices:
        return None
    return list(indices), list(values)


def search_knowledge_base(
    dense_vector: List[float],
    sparse_vector: Optional[Dict] = None,
    top_k: int = TOP_K_DEFAULT,
    score_threshold: float = SCORE_THRESHOLD_DEFAULT,
) -> List[Dict]:
    """
    Hybrid search in Qdrant: dense (semantic) + sparse (BM25 keyword)
    via prefetch + RRF. If sparse fails, fall back to dense-only.
    ``sparse_vector`` is the pure-Python dict from ``embed_query_sparse``
    (a legacy (indices, values) tuple is still accepted).
    """
    sparse = _normalize_sparse(sparse_vector)
    try:
        if sparse is not None:
            # Hybrid: prefetch dense + prefetch sparse → RRF
            resp = requests.post(
                f"{QDRANT_URL}/collections/{COLLECTION}/points/query",
                headers={"Content-Type": "application/json"},
                json={
                    "prefetch": [
                        {"query": dense_vector, "using": "dense", "limit": top_k * 3},
                        {
                            "query": {
                                "indices": sparse[0],
                                "values": sparse[1],
                            },
                            "using": "sparse",
                            "limit": top_k * 3,
                        },
                    ],
                    "query": {"fusion": "rrf"},
                    "limit": top_k * 2,
                    "with_payload": True,
                },
                timeout=REQUEST_TIMEOUT + 5,
            )
        else:
            # Fallback: dense-only (collections with compatible named vectors)
            resp = requests.post(
                f"{QDRANT_URL}/collections/{COLLECTION}/points/query",
                headers={"Content-Type": "application/json"},
                json={
                    "query": dense_vector,
                    "using": "dense",
                    "limit": top_k,
                    "with_payload": True,
                },
                timeout=REQUEST_TIMEOUT,
            )
        resp.raise_for_status()
        data = resp.json()
        results = []
        raw_results = data.get("result", {})
        # API /points/search returns direct list; API /points/query returns {"points": [...]}
        points = (
            raw_results
            if isinstance(raw_results, list)
            else raw_results.get("points", [])
        )
        for r in points:
            score = r.get("score", 0)
            if score < score_threshold:
                continue
            payload = r.get("payload", {})
            results.append(
                {
                    "id": r.get("id", "unknown"),
                    "score": score,
                    "title": payload.get("title", "Untitled"),
                    "content_preview": _strip_prompt_injection(
                        (payload.get("text", "") or "")[:400]
                    ),
                    "source": payload.get("source", "unknown"),
                    "tags": payload.get("tags", []),
                }
            )
        return results
    except Exception as e:
        print(f"[CE-ERROR] Qdrant search failed: {e}", file=sys.stderr)
        return []


# ─── Fallback: Lexical Search in Vault ──────────────────────────────────────


# Portuguese + English stopwords for the fallback lexical search. The upstream
# set was inline in tokenize_query; hoisted so cjk_tokens can receive it.
_PT_STOPWORDS = frozenset(
    """
    o a os as um uma de da do em no na para com por que se e ou mas
    the an is are was were be been have has had do does did will would could should
    """.split()
)


def tokenize_query(text: str) -> List[str]:
    """Tokenize a query into search terms, CJK-aware.

    Both consumers of this list (``lexical_search_in_vault`` and
    ``sqlite_keyword_search``) use ``LIKE '%term%'`` substring matching, so the
    right unit for Chinese is a 2-char bigram — and the old
    ``re.findall(r"\\b[a-zA-Z0-9]+\\b")`` returned an EMPTY list for any Chinese
    query, which made both fallback levels dead ends. That mattered more than it
    looks: the lexical and SQLite levels are exactly what runs when Qdrant or the
    embedding API is down, so Chinese users silently lost knowledge-base recall
    precisely during an outage.

    The namespace is the cjk_tokens module. It is imported lazily because
    scripts/ is a sibling of icarus/, not a subpackage of it.
    """
    tokens = _cjk_tokens().tokenize(text, stopwords=_PT_STOPWORDS, min_ascii_len=3)
    # Ordered, deterministic: sorting keeps query construction reproducible.
    return sorted(t for t in tokens if len(t) >= 2)


def _cjk_tokens():
    """Import the shared CJK tokenizer from scripts/ or from icarus/.

    scripts/ and icarus/ are siblings, so neither is importable as a package
    path from the other. Try both layouts, most specific first.
    """
    import importlib.util

    for modname in ("icarus.cjk_tokens", "cjk_tokens"):
        try:
            return importlib.import_module(modname)
        except ImportError:
            continue
    # Last resort: load by path relative to this file (repo root/icarus/).
    here = Path(__file__).resolve().parent
    cand = here.parent / "icarus" / "cjk_tokens.py"
    if not cand.exists():
        cand = here / "cjk_tokens.py"
    spec = importlib.util.spec_from_file_location("icarus_cjk_tokens", cand)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def lexical_search_in_vault(
    query_terms: List[str],
    top_k: int = TOP_K_DEFAULT,
    vault_root: str = os.environ.get("WIKI_PATH", os.path.expanduser("~/vault/wiki")),
) -> List[Dict]:
    """
    Lexical search in .md files under vault/wiki/.
    Rank by term match density / document size.
    Return top-k simulated chunks (or entire file if small).
    """
    if not query_terms:
        return []

    md_files = glob.glob(f"{vault_root}/**/*.md", recursive=True)
    if not md_files:
        return []

    scored = []
    for filepath in md_files:
        try:
            text = Path(filepath).read_text(encoding="utf-8", errors="replace").lower()
            if not text:
                continue

            matches = 0
            for term in query_terms:
                matches += text.count(term)

            if matches == 0:
                continue

            # Heuristic: matches / sqrt(word_count) — favors concise documents
            word_count = max(1, len(text.split()))
            density = matches / (
                word_count**0.5
            )  # sqrt(word_count) to not over-penalize medium texts
            scored.append(
                {
                    "filepath": filepath,
                    "matches": matches,
                    "word_count": word_count,
                    "density": density,
                    "title": Path(filepath).stem,
                    "text": text[:2000],  # truncate for return
                }
            )
        except Exception:
            continue

    if not scored:
        return []

    # Sort by descending density
    scored.sort(key=lambda x: x["density"], reverse=True)

    results = []
    for i, item in enumerate(scored[:top_k]):
        # Simulate a result chunk
        results.append(
            {
                "id": f"lexical-{hashlib.md5(item['filepath'].encode()).hexdigest()[:16]}",
                "score": round(min(1.0, item["density"]), 2),
                "title": item["title"],
                "content_preview": _strip_prompt_injection(item["text"][:400]),
                "source": f"vault-{item['filepath'].replace(vault_root, '').lstrip('/')[:40]}",
                "tags": ["fallback", "lexical"],
                "fallback_level": "lexical",
            }
        )

    if results:
        print(
            f"[CE-FALLBACK] Lexical search returned {len(results)} results from vault",
            file=sys.stderr,
        )
    return results


# ─── Fallback: SQLite Keyword Search ────────────────────────────────────────


def sqlite_keyword_search(
    query_terms: List[str], top_k: int = TOP_K_DEFAULT
) -> List[Dict]:
    """
    Search terms in the lineage table (query field) and other state tables.
    Last resort — does not replace vault.
    """
    if not query_terms:
        return []

    try:
        with sqlite3.connect(LINEAGE_DB) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()

            results = []
            # Search in lineage.query
            placeholders = " OR ".join(["query LIKE ?"] * len(query_terms))
            params = [f"%{term}%" for term in query_terms]
            c.execute(
                f"""
                SELECT lineage_id, session_id, query, generation_context_hash, created_at
                FROM lineage
                WHERE {placeholders}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                params + [top_k * 2],
            )
            for row in c.fetchall():
                results.append(
                    {
                        "id": f"sqlite-{row['lineage_id'][:16]}",
                        "score": 0.5,
                        "title": f"Lineage {row['lineage_id'][:8]}...",
                        "content_preview": _strip_prompt_injection(
                            (row["query"] or "")[:400]
                        ),
                        "source": f"sqlite-history-{row['session_id']}",
                        "tags": ["fallback", "sqlite"],
                        "fallback_level": "sqlite",
                    }
                )

            if results:
                print(
                    f"[CE-FALLBACK] SQLite keyword search returned {len(results)} results",
                    file=sys.stderr,
                )
            return results[:top_k]
    except Exception as e:
        print(f"[CE-FALLBACK] SQLite search failed: {e}", file=sys.stderr)
        return []


# ─── Fallback Wrapper ──────────────────────────────────────────────────────


def search_with_fallback(
    dense_vector: Optional[List[float]] = None,
    sparse_vector: Optional[Dict] = None,
    query_text: str = "",
    top_k: int = TOP_K_DEFAULT,
    score_threshold: float = SCORE_THRESHOLD_DEFAULT,
) -> Tuple[List[Dict], str, float, float]:
    """
    4-level fallback cascade:
      1. Hybrid (dense + sparse + RRF) — normal mode
      2. Dense-only — if sparse fails or is None
      3. Lexical-only — if Qdrant goes down (ConnectionError, Timeout)
      4. SQLite keyword — if vault is inaccessible

    ``sparse_vector`` is the pure-Python dict from ``embed_query_sparse``
    (a legacy (indices, values) tuple is still accepted).

    Returns (results, fallback_level, qdrant_latency_ms, fallback_latency_ms).
    """
    fallback_level = "hybrid"
    qdrant_latency_ms = 0.0
    t0 = time.perf_counter()

    sparse = _normalize_sparse(sparse_vector)

    # Level 1 or 2: Qdrant (hybrid if sparse available, otherwise dense-only).
    # Without a dense vector Qdrant cannot be queried at all — say so explicitly
    # instead of firing a request with a null query.
    if dense_vector is None:
        print(
            "[CE-ERROR] No dense vector available (remote embedding unavailable) — "
            "skipping Qdrant, going straight to the lexical/SQLite fallbacks.",
            file=sys.stderr,
        )
    else:
        try:
            t_q0 = time.perf_counter()
            if sparse is not None:
                resp = requests.post(
                    f"{QDRANT_URL}/collections/{COLLECTION}/points/query",
                    headers={"Content-Type": "application/json"},
                    json={
                        "prefetch": [
                            {
                                "query": dense_vector,
                                "using": "dense",
                                "limit": top_k * 3,
                            },
                            {
                                "query": {
                                    "indices": sparse[0],
                                    "values": sparse[1],
                                },
                                "using": "sparse",
                                "limit": top_k * 3,
                            },
                        ],
                        "query": {"fusion": "rrf"},
                        "limit": top_k * 2,
                        "with_payload": True,
                    },
                    timeout=REQUEST_TIMEOUT + 5,
                )
                fallback_level = "hybrid"
            else:
                resp = requests.post(
                    f"{QDRANT_URL}/collections/{COLLECTION}/points/query",
                    headers={"Content-Type": "application/json"},
                    json={
                        "query": dense_vector,
                        "using": "dense",
                        "limit": top_k,
                        "with_payload": True,
                    },
                    timeout=REQUEST_TIMEOUT,
                )
                fallback_level = "dense-only"

            resp.raise_for_status()
            qdrant_latency_ms = (time.perf_counter() - t_q0) * 1000
            data = resp.json()
            results = []
            raw_results = data.get("result", {})
            points = (
                raw_results
                if isinstance(raw_results, list)
                else raw_results.get("points", [])
            )
            for r in points:
                score = r.get("score", 0)
                if score < score_threshold:
                    continue
                payload = r.get("payload", {})
                results.append(
                    {
                        "id": r.get("id", "unknown"),
                        "score": score,
                        "title": payload.get("title", "Untitled"),
                        "content_preview": _strip_prompt_injection(
                            (payload.get("text", "") or "")[:400]
                        ),
                        "source": payload.get("source", "unknown"),
                        "tags": payload.get("tags", []),
                    }
                )

            if results:
                fallback_latency_ms = (time.perf_counter() - t0) * 1000
                return results, fallback_level, qdrant_latency_ms, fallback_latency_ms
        except requests.exceptions.ConnectionError:
            print(
                "[CE-FALLBACK] Qdrant unavailable (ConnectionError), falling back to lexical search.",
                file=sys.stderr,
            )
        except requests.exceptions.Timeout:
            print(
                "[CE-FALLBACK] Qdrant timeout, falling back to lexical search.",
                file=sys.stderr,
            )
        except Exception as e:
            # Unexpected Qdrant error (e.g. SparseIndexError, 5xx, etc.)
            # If a sparse vector existed, it might be the sparse error — try dense-only
            if sparse is not None and "sparse" in str(e).lower():
                print(
                    f"[CE-FALLBACK] Sparse index failed ('{e}'), trying dense-only...",
                    file=sys.stderr,
                )
                try:
                    t_q2 = time.perf_counter()
                    resp = requests.post(
                        f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
                        headers={"Content-Type": "application/json"},
                        json={
                            "vector": dense_vector,
                            "using": "dense",
                            "limit": top_k,
                            "with_payload": True,
                        },
                        timeout=REQUEST_TIMEOUT,
                    )
                    resp.raise_for_status()
                    qdrant_latency_ms = (time.perf_counter() - t_q2) * 1000
                    data = resp.json()
                    results = []
                    raw_results = data.get("result", {})
                    points = (
                        raw_results
                        if isinstance(raw_results, list)
                        else raw_results.get("points", [])
                    )
                    for r in points:
                        score = r.get("score", 0)
                        if score < score_threshold:
                            continue
                        payload = r.get("payload", {})
                        results.append(
                            {
                                "id": r.get("id", "unknown"),
                                "score": score,
                                "title": payload.get("title", "Untitled"),
                                "content_preview": _strip_prompt_injection(
                                    (payload.get("text", "") or "")[:400]
                                ),
                                "source": payload.get("source", "unknown"),
                                "tags": payload.get("tags", []),
                            }
                        )
                    if results:
                        fallback_latency_ms = (time.perf_counter() - t0) * 1000
                        print(
                            "[CE-FALLBACK] Dense-only worked after sparse error.",
                            file=sys.stderr,
                        )
                        return (
                            results,
                            "dense-only",
                            qdrant_latency_ms,
                            fallback_latency_ms,
                        )
                except (
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                ):
                    print(
                        "[CE-FALLBACK] Qdrant also unavailable for dense-only — lexical fallback.",
                        file=sys.stderr,
                    )
                except Exception as e2:
                    print(
                        f"[CE-FALLBACK] Dense-only also failed: {e2}", file=sys.stderr
                    )
            else:
                print(
                    f"[CE-FALLBACK] Qdrant general error ({e}), falling back to lexical.",
                    file=sys.stderr,
                )

    # Level 3: Lexical search in vault
    terms = tokenize_query(query_text)
    lexical_results = lexical_search_in_vault(terms, top_k=top_k)
    if lexical_results:
        fallback_latency_ms = (time.perf_counter() - t0) * 1000
        return lexical_results, "lexical", qdrant_latency_ms, fallback_latency_ms

    # Level 4: SQLite keyword search
    sqlite_results = sqlite_keyword_search(terms, top_k=top_k)
    if sqlite_results:
        fallback_latency_ms = (time.perf_counter() - t0) * 1000
        return sqlite_results, "sqlite", qdrant_latency_ms, fallback_latency_ms

    # Nothing worked
    fallback_latency_ms = (time.perf_counter() - t0) * 1000
    print("[CE-FALLBACK] All fallback levels exhausted.", file=sys.stderr)
    return [], "none", qdrant_latency_ms, fallback_latency_ms


def update_last_accessed_at(chunk_ids: list) -> None:
    """
    Update last_accessed_at on chunks returned by search.
    Reset decay for chunks that are actually being used.
    Fail-open: never breaks the critical query path.
    """
    if not chunk_ids:
        return
    try:
        now = datetime.now().astimezone().isoformat()
        requests.post(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/payload",
            headers={"Content-Type": "application/json"},
            json={
                "points": chunk_ids,
                "payload": {"last_accessed_at": now},
            },
            timeout=5,
        )
    except Exception as e:
        print(f"[CE-WARNING] Failed to update last_accessed_at: {e}", file=sys.stderr)


def enhance_prompt(
    user_message: str,
    top_k: int = TOP_K_DEFAULT,
    score_threshold: float = SCORE_THRESHOLD_DEFAULT,
    format_mode: str = "markdown",
    hybrid: bool = True,
    session_id: Optional[str] = None,
    generation_model: str = "unknown",
) -> str:
    """
    Return a context block with relevant vault content (hybrid dense+BM25),
    or an empty string if nothing relevant found / services offline.
    Register lineage with chunk provenance and generated context hash.
    """
    # Resolve session_id
    if session_id is None:
        session_id = os.environ.get("HERMES_SESSION_ID", "standalone")

    # Skip very short or irrelevant queries
    if len(user_message.strip()) < 5:
        return ""

    # Very short or social query — skip
    social_keywords = {
        "hi",
        "hello",
        "hey",
        "yo",
        "sup",
        "ok",
        "thanks",
        "bye",
        "okay",
        "great",
    }
    if user_message.strip().lower() in social_keywords:
        return ""

    # Dense embedding is remote-only and fails loudly; here we report the reason
    # explicitly and continue with the designed fallback cascade (never silently
    # pretend the semantic level worked).
    try:
        dense_vector = embed_query(user_message)
    except Exception as e:
        print(
            f"[CE-ERROR] Dense embedding unavailable: {e} — continuing with the "
            "lexical/SQLite fallback levels.",
            file=sys.stderr,
        )
        dense_vector = None

    sparse_vector = None
    if hybrid:
        sparse_vector = embed_query_sparse(user_message)

    # Fallback cascade: hybrid → dense-only → lexical → sqlite
    hits, fallback_level, qdrant_latency_ms, fallback_latency_ms = search_with_fallback(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        query_text=user_message,
        top_k=top_k,
        score_threshold=score_threshold,
    )

    # If empty and we had no embedding, try pure lexical
    if not hits and dense_vector is None:
        terms = tokenize_query(user_message)
        lexical_results = lexical_search_in_vault(terms, top_k=top_k)
        if lexical_results:
            hits = lexical_results
            fallback_level = "lexical"

    if not hits:
        # Emit telemetry even when empty (records that search ran and returned nothing)
        emit_telemetry(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id,
                "query": user_message,
                "retrieval_mode": fallback_level,
                "retrieved_chunk_ids": [],
                "semantic_scores": [],
                "rerank_scores": [],
                "final_context_tokens": 0,
                "fallback_level": {
                    "hybrid": 0,
                    "dense-only": 1,
                    "lexical": 2,
                    "sqlite": 3,
                    "none": -1,
                }.get(fallback_level, -1),
                "llm_response_quality": None,
                "qdrant_latency_ms": round(qdrant_latency_ms, 2),
                "fallback_latency_ms": round(fallback_latency_ms, 2),
            }
        )
        return ""

    # Update last_accessed_at for returned chunks (reset decay on real usage)
    retrieved_chunk_ids = [h["id"] for h in hits]
    update_last_accessed_at(retrieved_chunk_ids)

    if format_mode == "markdown":
        lines = ["\n## Relevant Context from Vault\n"]
        for i, h in enumerate(hits, 1):
            lines.append(f"### [{i}] {h['title']} (score: {h['score']:.2f})")
            lines.append(
                f"- **Source:** `{h['source']}` | Tags: {', '.join(str(t) for t in h['tags'])}"
            )
            lines.append(f"- **Excerpt:** {h['content_preview']}...")
            lines.append("")
        context_str = "\n".join(lines)
    elif format_mode == "compact":
        parts = []
        for h in hits:
            parts.append(f"[{h['title'][:40]}] (s:{h['score']:.2f})")
        context_str = "\n".join(parts)
    else:
        context_str = json.dumps(hits, indent=2, ensure_ascii=False)

    generation_context_hash = hashlib.sha256(context_str.encode("utf-8")).hexdigest()[
        :32
    ]

    register_lineage(
        session_id=session_id,
        query=user_message,
        retrieved_chunk_ids=retrieved_chunk_ids,
        generation_context_hash=generation_context_hash,
        generation_model=generation_model,
    )

    # Extract scores for telemetry
    semantic_scores = [h.get("score", 0) for h in hits]
    # For rerank_scores, use the same scores (internal Qdrant RRF, not exposed via simple API)
    rerank_scores = semantic_scores[:]

    final_context_tokens = estimate_tokens(context_str)

    emit_telemetry(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "query": user_message,
            "retrieval_mode": fallback_level,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "semantic_scores": semantic_scores,
            "rerank_scores": rerank_scores,
            "final_context_tokens": final_context_tokens,
            "fallback_level": {
                "hybrid": 0,
                "dense-only": 1,
                "lexical": 2,
                "sqlite": 3,
                "none": -1,
            }.get(fallback_level, -1),
            "llm_response_quality": None,
            "qdrant_latency_ms": round(qdrant_latency_ms, 2),
            "fallback_latency_ms": round(fallback_latency_ms, 2),
        }
    )

    return context_str


def main():
    parser = argparse.ArgumentParser(
        description="Context Enhancer — Hybrid search in knowledge_base"
    )
    parser.add_argument("query", nargs="?", help="User message/query")
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_DEFAULT,
        help=f"Number of results (default: {TOP_K_DEFAULT})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=SCORE_THRESHOLD_DEFAULT,
        help=f"Minimum score (default: {SCORE_THRESHOLD_DEFAULT})",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "compact", "json"],
        default="markdown",
        help="Output format",
    )
    parser.add_argument(
        "--silent", action="store_true", help="Silent — return empty on error"
    )
    parser.add_argument(
        "--hybrid-off", action="store_true", help="Force dense-only (ignore BM25)"
    )
    args = parser.parse_args()

    if not args.query:
        args.query = sys.stdin.read().strip() or "how ARQ worker works"

    try:
        result = enhance_prompt(
            args.query,
            top_k=args.top_k,
            score_threshold=args.threshold,
            format_mode=args.format,
            hybrid=not args.hybrid_off,
            generation_model=os.environ.get("HERMES_MODEL", "unknown"),
        )
        print(result)
    except Exception as e:
        if not args.silent:
            print(f"[CE-ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
