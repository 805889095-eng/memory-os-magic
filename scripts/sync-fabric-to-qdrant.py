#!/usr/bin/env python3
"""Mirror fabric entries into the Qdrant ``knowledge_base`` collection.

WHY THIS EXISTS
---------------
Layer 5 (Qdrant hybrid search) and layer 4 (fabric) were wired to *read* from
the same prompt but nothing ever wrote fabric entries into Qdrant. The upstream
feeder is ``wiki_continuous_ingest.py``, which requires a Redis + ARQ worker
stack (Docker) and a ``~/vault/wiki`` tree — neither exists in a plain Hermes
install. Result: the collection stayed at 0 points forever and every
``[qdrant]`` block in the prompt was empty, while the health check said "green".

This script closes that loop with no extra infrastructure: it reads fabric
markdown, embeds with the same remote embedding endpoint the rest of Memory OS
uses, and upserts dense + pure-Python-BM25 sparse vectors under the payload
shape ``context_enhancer.search_knowledge_base`` expects.

Point ids are a UUIDv5 of the fabric entry id, so re-running is idempotent —
an entry that was edited updates in place instead of duplicating.

Usage:
    python3 sync-fabric-to-qdrant.py [--dry-run] [--prune] [--quiet]

Runs standalone (own venv/python is fine — needs only `requests`, which Hermes
already ships).
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

HERMES_HOME = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
FABRIC_DIR = Path(os.environ.get("FABRIC_DIR", HERMES_HOME / "fabric"))
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "knowledge_base")
EMBEDDING_API_BASE = os.environ.get("EMBEDDING_API_BASE", "").rstrip("/")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "")
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "")
EMBEDDING_DIMS = int(os.environ.get("EMBEDDING_DIMS", "1024"))
EMBEDDING_PASS_DIMENSIONS = os.environ.get("EMBEDDING_PASS_DIMENSIONS", "auto").lower()

POINT_NAMESPACE = uuid.UUID("6f5f1a3e-0d4c-4d3e-9f1a-2b8c7d6e5f40")
MAX_TEXT_LEN = 8000

SCRIPTS_DIR = HERMES_HOME / "memory-os" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _env_from_dotenv():
    """Pull embedding creds from $HERMES_HOME/.env when not already in environ."""
    global EMBEDDING_API_BASE, EMBEDDING_MODEL, EMBEDDING_API_KEY
    global EMBEDDING_DIMS, EMBEDDING_PASS_DIMENSIONS
    if EMBEDDING_API_KEY and EMBEDDING_API_BASE:
        return
    env_file = HERMES_HOME / ".env"
    if not env_file.exists():
        return
    values = {}
    for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip().strip("'\"")
    EMBEDDING_API_BASE = (values.get("EMBEDDING_API_BASE") or EMBEDDING_API_BASE).rstrip("/")
    EMBEDDING_MODEL = values.get("EMBEDDING_MODEL") or EMBEDDING_MODEL
    EMBEDDING_API_KEY = values.get("EMBEDDING_API_KEY") or EMBEDDING_API_KEY
    if values.get("EMBEDDING_DIMS"):
        EMBEDDING_DIMS = int(values["EMBEDDING_DIMS"])
    EMBEDDING_PASS_DIMENSIONS = (
        values.get("EMBEDDING_PASS_DIMENSIONS") or EMBEDDING_PASS_DIMENSIONS
    ).lower()


def _imports():
    try:
        import requests
    except ImportError:
        sys.exit("sync-fabric-to-qdrant: `requests` is required (hermes runtime ships it)")
    try:
        from bm25_tf import bm25_tf
    except ImportError:
        sys.exit(f"sync-fabric-to-qdrant: cannot import bm25_tf from {SCRIPTS_DIR}")
    return requests, bm25_tf


# ── frontmatter ────────────────────────────────────────────────────────────


def _yaml_helper():
    """Reuse the icarus YAML backend (hermes_yaml -> PyYAML -> mini)."""
    icarus_dir = HERMES_HOME / "plugins" / "icarus"
    if str(icarus_dir) not in sys.path:
        sys.path.insert(0, str(icarus_dir))
    try:
        import _yaml  # type: ignore

        return _yaml
    except Exception:
        return None


def parse_entry(path):
    """Return (frontmatter dict, body) for one fabric markdown file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    helper = _yaml_helper()
    meta = {}
    if helper is not None:
        try:
            meta = helper.safe_load(parts[1]) or {}
        except Exception:
            meta = {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, parts[2].strip()


# ── embeddings ─────────────────────────────────────────────────────────────


def _embed_payload(text):
    payload = {"model": EMBEDDING_MODEL, "input": text[:MAX_TEXT_LEN]}
    # bge-m3 outputs its native width and rejects an explicit `dimensions`
    # (SiliconFlow answers 20015). Only send it when explicitly opted in.
    if EMBEDDING_PASS_DIMENSIONS in ("1", "true", "yes", "on"):
        payload["dimensions"] = EMBEDDING_DIMS
    return payload


def embed_dense(requests, texts, retries=3):
    """Embed a batch; returns list of vectors (one per input)."""
    if not EMBEDDING_API_BASE or not EMBEDDING_MODEL or not EMBEDDING_API_KEY:
        raise RuntimeError(
            "embedding endpoint not configured (EMBEDDING_API_BASE / _MODEL / _API_KEY)"
        )
    url = f"{EMBEDDING_API_BASE}/embeddings"
    headers = {
        "Authorization": f"Bearer {EMBEDDING_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = _embed_payload(texts[0]) if len(texts) == 1 else _embed_payload(texts)
    payload["input"] = [t[:MAX_TEXT_LEN] for t in texts]
    last = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                vectors = [item["embedding"] for item in data.get("data", [])]
                if len(vectors) != len(texts):
                    raise RuntimeError(
                        f"expected {len(texts)} embeddings, got {len(vectors)}"
                    )
                return vectors
            body = resp.text[:300]
            # 20015 = endpoint refuses a `dimensions` param: drop it and retry.
            if "20015" in body and "dimensions" in payload:
                payload.pop("dimensions", None)
                last = f"HTTP {resp.status_code}: {body}"
                continue
            last = f"HTTP {resp.status_code}: {body}"
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"embedding failed after {retries} attempts — {last}")


def embed_sparse(bm25_tf, text):
    vec = bm25_tf(text[:MAX_TEXT_LEN])
    if isinstance(vec, dict):
        return vec.get("indices", []), vec.get("values", [])
    if isinstance(vec, (list, tuple)) and len(vec) == 2:
        return list(vec[0]), list(vec[1])
    return [], []


# ── qdrant ─────────────────────────────────────────────────────────────────


def ensure_collection(requests):
    url = f"{QDRANT_URL}/collections/{COLLECTION}"
    resp = requests.get(url, timeout=10)
    if resp.status_code == 200:
        return False
    body = {
        "vectors": {"dense": {"size": EMBEDDING_DIMS, "distance": "Cosine"}},
        "sparse_vectors": {"sparse": {"modifier": "idf"}},
    }
    created = requests.put(url, json=body, timeout=30)
    if created.status_code not in (200, 201):
        raise RuntimeError(f"cannot create collection: HTTP {created.status_code} {created.text[:200]}")
    return True


def existing_ids(requests):
    """All point ids currently in the collection (paginated scroll)."""
    ids = set()
    offset = None
    while True:
        body = {"limit": 512, "with_payload": False, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        resp = requests.post(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll", json=body, timeout=30
        )
        if resp.status_code != 200:
            break
        result = resp.json().get("result", {})
        points = result.get("points", [])
        for p in points:
            ids.add(str(p.get("id")))
        offset = result.get("next_page_offset")
        if not offset or not points:
            break
    return ids


def point_id(entry_id, filename):
    """Deterministic UUID per fabric entry (idempotent re-ingest)."""
    key = entry_id or hashlib.sha1(filename.encode()).hexdigest()[:16]
    return str(uuid.uuid5(POINT_NAMESPACE, str(key)))


def build_point(requests, bm25_tf, path):
    meta, body = parse_entry(path)
    entry_id = str(meta.get("id") or "").strip()
    summary = str(meta.get("summary") or "").strip()
    title = summary or path.stem
    # The embedded text leads with the summary so a short query matches the
    # entry's own headline before the long body dilutes the vector.
    text = f"{title}\n\n{body}" if summary else body
    if not text.strip():
        return None
    dense = embed_dense(requests, [text])[0]
    indices, values = embed_sparse(bm25_tf, text)
    payload = {
        "text": text,
        "title": title,
        "source": "fabric",
        "tags": [t.strip() for t in str(meta.get("tags") or "").split(",") if t.strip()],
        "entry_id": entry_id,
        "entry_type": str(meta.get("type") or ""),
        "agent": str(meta.get("agent") or ""),
        "timestamp": str(meta.get("timestamp") or ""),
        "tier": str(meta.get("tier") or ""),
        "file": path.name,
        "content_hash": hashlib.sha1(text.encode()).hexdigest(),
    }
    vector = {"dense": dense}
    if indices:
        vector["sparse"] = {"indices": indices, "values": values}
    return {"id": point_id(entry_id, path.name), "vector": vector, "payload": payload}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="parse + embed, do not write")
    ap.add_argument("--prune", action="store_true", help="delete points whose fabric file is gone")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    requests, bm25_tf = _imports()
    _env_from_dotenv()

    def log(*a):
        if not args.quiet:
            print(*a)

    if not FABRIC_DIR.is_dir():
        log(f"no fabric directory at {FABRIC_DIR} — nothing to sync")
        return 0

    files = sorted(FABRIC_DIR.glob("*.md"))
    if not files:
        log(f"fabric is empty ({FABRIC_DIR}) — nothing to sync")
        return 0

    try:
        if not args.dry_run:
            created = ensure_collection(requests)
            if created:
                log(f"created collection {COLLECTION} (dense={EMBEDDING_DIMS} + sparse BM25)")
        before = set() if args.dry_run else existing_ids(requests)
    except Exception as exc:
        print(f"qdrant unavailable at {QDRANT_URL}: {exc}", file=sys.stderr)
        return 2

    points, failures = [], 0
    for path in files:
        try:
            point = build_point(requests, bm25_tf, path)
        except Exception as exc:
            failures += 1
            print(f"  skip {path.name}: {exc}", file=sys.stderr)
            continue
        if point:
            points.append(point)

    if args.dry_run:
        for p in points:
            log(f"  would upsert {p['id']}  {p['payload']['title'][:60]}")
        log(f"dry run: {len(points)} point(s) ready, {failures} failure(s)")
        return 0 if not failures else 1

    upserted = 0
    for chunk_start in range(0, len(points), 32):
        chunk = points[chunk_start : chunk_start + 32]
        resp = requests.put(
            f"{QDRANT_URL}/collections/{COLLECTION}/points?wait=true",
            json={"points": chunk},
            timeout=60,
        )
        if resp.status_code != 200:
            print(f"  upsert failed: HTTP {resp.status_code} {resp.text[:200]}", file=sys.stderr)
            failures += len(chunk)
            continue
        upserted += len(chunk)

    pruned = 0
    if args.prune:
        keep = {p["id"] for p in points}
        stale = before - keep
        if stale:
            resp = requests.post(
                f"{QDRANT_URL}/collections/{COLLECTION}/points/delete?wait=true",
                json={"points": sorted(stale)},
                timeout=30,
            )
            if resp.status_code == 200:
                pruned = len(stale)
            else:
                print(f"  prune failed: HTTP {resp.status_code}", file=sys.stderr)

    log(
        f"synced {upserted}/{len(points)} fabric entr{'y' if len(points) == 1 else 'ies'} "
        f"into {COLLECTION}"
        + (f", pruned {pruned} stale" if pruned else "")
        + (f", {failures} failure(s)" if failures else "")
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
