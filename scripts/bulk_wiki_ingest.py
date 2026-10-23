#!/usr/bin/env python3
"""
Bulk ingest script — popula a knowledge_base Qdrant com todo conteúdo da wiki.
Fase A: one-shot dos arquivos existentes.
"""

import os
import re
import sys
import json
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

import aiohttp
import asyncio

# Sparse embedding (BM25) — pure Python, no model, no network, no subprocess.
# `scripts/bm25_tf.py` is byte-identical to `docker/worker/services/bm25_tf.py`;
# works both as a standalone script and via `from scripts.bulk_wiki_ingest ...`.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from bm25_tf import bm25_tf  # noqa: E402

# ─── Config ────────────────────────────────────────────────────────────────
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "knowledge_base")
WIKI_ROOT = Path(os.environ.get("WIKI_ROOT", str(Path.home() / "vault" / "wiki")))
# Dense embedding — remote only (unified config contract)
EMBEDDING_API_BASE = os.environ.get(
    "EMBEDDING_API_BASE", "https://openrouter.ai/api/v1"
)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY") or ""
if not EMBEDDING_API_KEY and "openrouter" in EMBEDDING_API_BASE.lower():
    EMBEDDING_API_KEY = os.environ.get("OPENROUTER_API_KEY") or ""
EMBEDDING_DIMS = int(os.environ.get("EMBEDDING_DIMS", "4096"))
EMBEDDING_PASS_DIMENSIONS = os.environ.get("EMBEDDING_PASS_DIMENSIONS", "auto").lower()
MAX_TEXT_LEN = 8000  # truncate text for embedding (model context limit)
BATCH_SIZE = 8  # parallel embedding requests
RATE_LIMIT_SLEEP = 0.5  # seconds between batches

if not EMBEDDING_API_KEY:
    print(
        "❌ EMBEDDING_API_KEY não encontrada no ambiente "
        "(nem OPENROUTER_API_KEY para base OpenRouter)"
    )
    sys.exit(1)

print(f"📁 Wiki root: {WIKI_ROOT}")
print(f"🎯 Coleção: {COLLECTION}")
print(f"🔑 Embedding key configured: {'yes' if EMBEDDING_API_KEY else 'no'}")

# ─── Encontrar todos os .md ────────────────────────────────────────────────
md_files = sorted(WIKI_ROOT.rglob("*.md"))
print(f"📄 Arquivos .md encontrados: {len(md_files)}")


# ─── Helpers ──────────────────────────────────────────────────────────────
def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extrai YAML frontmatter e retorna (metadata, body)."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                import yaml

                meta = yaml.safe_load(parts[1])
                body = parts[2].strip()
                return (meta if isinstance(meta, dict) else {}), body
            except Exception:
                pass
    return {}, text


def get_source_tag(path: Path) -> str:
    """Deriva source tag do path relativo à wiki."""
    rel = path.relative_to(WIKI_ROOT)
    parts = rel.parts
    if len(parts) > 1:
        return f"wiki-{parts[0]}"
    return "wiki-root"


def get_tags_from_frontmatter(meta: dict) -> list[str]:
    """Extrai tags do frontmatter."""
    tags = meta.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]
    return tags if isinstance(tags, list) else []


def get_sparse_vector(text: str) -> dict:
    """Gera sparse vector BM25 (puro Python, sem modelo/rede/subprocesso).

    ``bm25_tf`` só levanta exceção em erro de programação — nunca silenciado.
    """
    return bm25_tf(text)


async def get_embedding(
    session: aiohttp.ClientSession, text: str
) -> list[float] | None:
    """Gera embedding denso via API remota (OpenAI-compatível)."""
    payload: dict = {
        "model": EMBEDDING_MODEL,
        "input": text[:MAX_TEXT_LEN],
    }
    should_pass_dims = EMBEDDING_PASS_DIMENSIONS == "true" or (
        EMBEDDING_PASS_DIMENSIONS == "auto" and "text-embedding-3" in EMBEDDING_MODEL.lower()
    )
    if should_pass_dims:
        payload["dimensions"] = EMBEDDING_DIMS

    try:
        async with session.post(
            f"{EMBEDDING_API_BASE.rstrip('/')}/embeddings",
            headers={
                "Authorization": f"Bearer {EMBEDDING_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status == 400 and "dimensions" in payload:
                body = await resp.text()
                if "20015" in body or "dimensions" in body.lower():
                    payload.pop("dimensions", None)
                    async with session.post(
                        f"{EMBEDDING_API_BASE.rstrip('/')}/embeddings",
                        headers={
                            "Authorization": f"Bearer {EMBEDDING_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp2:
                        if resp2.status != 200:
                            body2 = await resp2.text()
                            print(f"⚠️ Embedding HTTP {resp2.status}: {body2[:200]}")
                            return None
                        data2 = await resp2.json()
                        return data2["data"][0]["embedding"]
            if resp.status != 200:
                body = await resp.text()
                print(f"⚠️ Embedding HTTP {resp.status}: {body[:200]}")
                return None
            data = await resp.json()
            return data["data"][0]["embedding"]
    except Exception as e:
        print(f"⚠️ Embedding error: {e}")
        return None


async def upsert_to_qdrant(session: aiohttp.ClientSession, points: list[dict]) -> bool:
    """Upsert batch de pontos no Qdrant."""
    try:
        async with session.put(
            f"{QDRANT_URL}/collections/{COLLECTION}/points",
            headers={"Content-Type": "application/json"},
            json={"points": points},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                print(f"⚠️ Qdrant HTTP {resp.status}: {body[:200]}")
                return False
            return True
    except Exception as e:
        print(f"⚠️ Qdrant error: {e}")
        return False


# ─── Processamento principal ──────────────────────────────────────────────
async def main():
    stats = Counter({"ok": 0, "fail": 0, "skip": 0, "empty": 0})
    errors = []
    processed = 0
    total = len(md_files)

    connector = aiohttp.TCPConnector(limit=20)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Verificar coleção — criar se não existir
        async with session.get(f"{QDRANT_URL}/collections/{COLLECTION}") as r:
            if r.status != 200:
                print(f"⚠️  Coleção {COLLECTION} não existe. Criando...")
                collection_config = {
                    "vectors": {
                        "dense": {
                            "size": EMBEDDING_DIMS,
                            "distance": "Cosine",
                        }
                    },
                    "sparse_vectors": {
                        "sparse": {},
                    },
                }
                async with session.put(
                    f"{QDRANT_URL}/collections/{COLLECTION}",
                    headers={"Content-Type": "application/json"},
                    json=collection_config,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as cr:
                    if cr.status not in (200, 201):
                        body = await cr.text()
                        print(f"❌ Falha ao criar coleção ({cr.status}): {body[:200]}")
                        sys.exit(1)
                print(
                    f"✅ Coleção {COLLECTION} criada (dense {EMBEDDING_DIMS}d + sparse BM25)"
                )

        print("\n🚀 Iniciando ingestão em batches...\n")

        batch = []
        for idx, path in enumerate(md_files, 1):
            text = path.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                stats["empty"] += 1
                continue

            meta, body = parse_frontmatter(text)
            source = get_source_tag(path)
            tags = get_tags_from_frontmatter(meta)
            # Tag adicional da pasta
            folder_tag = source.replace("wiki-", "")
            if folder_tag not in tags:
                tags.append(folder_tag)

            # Título do frontmatter ou filename
            title = meta.get("title", path.stem)

            # Texto para embedding: título + body (sem frontmatter)
            embed_text = f"{title}\n\n{body}"[:MAX_TEXT_LEN]

            batch.append(
                {
                    "idx": idx,
                    "path": str(path),
                    "title": title,
                    "source": source,
                    "tags": tags,
                    "embed_text": embed_text,
                    "meta": meta,
                }
            )

            if len(batch) >= BATCH_SIZE or idx == total:
                # Gerar embeddings em paralelo
                embed_tasks = [get_embedding(session, b["embed_text"]) for b in batch]
                vectors = await asyncio.gather(*embed_tasks)

                # Gerar sparse vectors (sync, puro Python — sem modelo)
                sparse_vecs = [get_sparse_vector(b["embed_text"]) for b in batch]

                # Preparar pontos Qdrant
                points = []
                for b, vec, sparse in zip(batch, vectors, sparse_vecs):
                    if vec is None:
                        stats["fail"] += 1
                        errors.append(f"Embedding failed: {b['path']}")
                        continue

                    vector_payload = {"dense": vec}
                    if sparse is not None:
                        vector_payload["sparse"] = sparse

                    # Heurística de importance_score baseada no path/nome
                    importance_score = 0.5
                    path_str_lower = b["path"].lower()
                    if any(
                        k in path_str_lower
                        for k in ["architecture", "core", "important"]
                    ):
                        importance_score = 0.7
                    if any(t.lower() in ["important", "critical"] for t in b["tags"]):
                        importance_score = 0.8
                    if any(k in path_str_lower for k in ["draft", "temp", "old"]):
                        importance_score = 0.2

                    now_iso = datetime.now(timezone.utc).isoformat()

                    point = {
                        "id": str(uuid.uuid4()),
                        "vector": vector_payload,
                        "payload": {
                            "text": b["embed_text"],
                            "source": b["source"],
                            "tags": b["tags"],
                            "created_at": now_iso,
                            "reflection_count": 0,
                            "last_reflected": None,
                            "source_file": b["path"],
                            "file_path": b["path"],  # legacy alias (backward compat)
                            "title": b["title"],
                            "word_count": len(b["embed_text"].split()),
                            # ── Lineage fields (Fase 1)
                            "lineage_id": None,
                            "generation_model": None,
                            "generation_context_hash": None,
                            "retrieved_chunk_ids": None,
                            # ── Decay fields (Fase 2)
                            "decay_score": 1.0,
                            "last_accessed_at": now_iso,
                            "importance_score": importance_score,
                            "source_type": "human",
                            "confidence_score": 1.0,
                            "archived": False,
                        },
                    }
                    points.append(point)

                # Upsert
                if points:
                    ok = await upsert_to_qdrant(session, points)
                    if ok:
                        stats["ok"] += len(points)
                    else:
                        stats["fail"] += len(points)
                        for p in points:
                            errors.append(
                                f"Qdrant upsert failed: {p['payload']['source_file']}"
                            )

                processed += len(batch)
                batch = []

                # Progresso
                pct = (processed / total) * 100
                print(
                    f"  [{processed}/{total}] {pct:.1f}% | ✅ {stats['ok']} | ⚠️ {stats['fail']} | ⏭️ {stats['skip']} | 🈳 {stats['empty']}"
                )

                # Rate limit breathing
                await asyncio.sleep(RATE_LIMIT_SLEEP)

    # ─── Relatório final ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 RELATÓRIO DE INGESTÃO")
    print("=" * 60)
    print(f"  Total arquivos:     {total}")
    print(f"  Ingestados (ok):    {stats['ok']}")
    print(f"  Falhas:             {stats['fail']}")
    print(f"  Vazios:             {stats['empty']}")
    print(
        f"  Taxa de sucesso:    {(stats['ok'] / max(total - stats['empty'], 1) * 100):.1f}%"
    )
    print(f"\n  ⏱️  Finalizado: {datetime.now(timezone.utc).isoformat()}")

    if errors:
        print(f"\n  ⚠️  Primeiros erros ({min(10, len(errors))} de {len(errors)}):")
        for e in errors[:10]:
            print(f"     - {e}")

    # Verificar count final
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{QDRANT_URL}/collections/{COLLECTION}") as r:
            data = await r.json()
            final_count = data.get("result", {}).get("points_count", "?")
            print(f"\n  📦 Pontos na coleção: {final_count}")

    print("\n✅ Bulk ingest completo.")
    return stats


if __name__ == "__main__":
    stats = asyncio.run(main())
    sys.exit(0 if stats["fail"] == 0 else 1)
