# Infrastructure

> Docker services, cronjobs, and environment configuration that support the 6 memory layers.

## Docker Services

The vector and pipeline layers run as Docker containers:

```yaml
# docker-compose.yml
services:
  qdrant:
    image: qdrant/qdrant:v1.17.1
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_data:/qdrant/storage
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --requirepass ${REDIS_PASSWORD}
    restart: unless-stopped

  worker:
    build: ./worker
    depends_on:
      - qdrant
      - redis
    environment:
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - EMBEDDING_API_BASE=${EMBEDDING_API_BASE:-https://openrouter.ai/api/v1}
      - EMBEDDING_MODEL=${EMBEDDING_MODEL:-qwen/qwen3-embedding-8b}
      - EMBEDDING_API_KEY=${EMBEDDING_API_KEY:-}
      - EMBEDDING_DIMS=${EMBEDDING_DIMS:-4096}
      - LLM_API_BASE=${LLM_API_BASE:-https://openrouter.ai/api/v1}
      - LLM_API_KEY=${LLM_API_KEY:-}
      - LLM_MODEL=${LLM_MODEL:-deepseek/deepseek-chat}
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-}
      - ARQ_MAX_TRIES=${ARQ_MAX_TRIES:-3}
      - HERMES_DLQ_PATH=${HERMES_DLQ_PATH:-/hermes/wiki_ingest_failures.json}
      - COLLECTION_NAME=${COLLECTION_NAME:-knowledge_base}
    restart: unless-stopped
```

**Key configuration:**
- `EMBEDDING_DIMS=4096` — must match Qdrant collection schema
- `COLLECTION_NAME=knowledge_base` — target collection for all wiki ingestion
- Sparse vectors are pure-Python BM25 (`docker/worker/services/bm25_tf.py` in the worker) — the client emits only the tf saturation term and Qdrant applies IDF server-side; no model or native extension is loaded
- Redis password required — set in `.env`, used by both Redis container and worker

## Cronjobs

`setup.sh` installs exactly **six** maintenance jobs as an idempotent crontab block (markers `# >>> memory-os scheduler >>>` … `# <<< memory-os scheduler <<<`). Each line is `cd $REPO_ROOT && HERMES_HOME=$HERMES_HOME python3 scripts/<job>.py >> $HERMES_HOME/logs/<job>.log 2>&1` (the fact-decay job runs `icarus/evolution.py` directly):

| Job (script) | Recommended schedule | What it does |
|-----|---------------------|--------------|
| **wiki_continuous_ingest.py** | Hourly (`0 * * * *`) | SHA-256 diff detection → embed new wiki files → Qdrant |
| **reflection_trigger.py** | Every 5 min (`*/5 * * * *`) | Triggers `micro_reflection` when the ARQ worker is idle |
| **decay_scanner.py** | Weekly, Sun 03:00 (`0 3 * * 0`) | Archive low-importance, aged AI content from Qdrant |
| **semantic_dedup.py** | Monthly, 1st 03:00 (`0 3 1 * *`) | Merge near-duplicate vectors (cosine > 0.92) |
| **dlq_manager.py --report** | Every 2 h (`0 */2 * * *`) | Read, classify, and report dead-letter-queue failures |
| **icarus/evolution.py --decay 30** | Daily 03:30 (`30 3 * * *`) | Decay `trust_score` of facts unused for ≥30 days — the forgetting half of the self-evolution loop |

On platforms without `crontab` (e.g. Windows / git-bash) the scheduler step is skipped **gracefully** — the stack still runs; schedule the six scripts with your OS scheduler. The wiki agent (curation cron) and Vault Curator run independently of these six.

**Interaction between jobs:**
- Newly curated or newly added wiki pages are picked up by the next `wiki_continuous_ingest.py` run and sent to Qdrant
- `decay_scanner.py` and `semantic_dedup.py` both read `knowledge_base` payloads — run `scripts/backfill_decay_metadata.py` once before enabling decay, otherwise the scanner finds zero eligible points

## Environment Variables

### Required

| Variable | Purpose | Example |
|----------|---------|---------|
| `FABRIC_DIR` | Where Icarus writes fabric entries | `/home/your-user/vault/fabric` |
| `EMBEDDING_API_BASE` | Remote embedding endpoint (OpenAI-compatible) | `https://openrouter.ai/api/v1` |
| `EMBEDDING_MODEL` | Embedding model id | `qwen/qwen3-embedding-8b` |
| `EMBEDDING_API_KEY` | Key for the embedding endpoint (**required** — no offline fallback) | `sk-or-...` |
| `LLM_API_BASE` | Remote LLM endpoint used by the worker | `https://openrouter.ai/api/v1` |
| `LLM_API_KEY` | Key for the LLM endpoint (**required**) | `sk-or-...` |
| `REDIS_PASSWORD` | Redis authentication | (generated) |
| `OPENROUTER_API_KEY` | Fallback for both endpoints when the keys above are empty | `sk-or-...` |

### Strongly recommended

| Variable | Default | Recommended | Why |
|----------|---------|-------------|-----|
| `ICARUS_EXTRACTION_MAX_TOKENS` | 1024 | **4096** | 1024 causes fabric truncation |
| `ICARUS_EXTRACTION_MODEL` | `deepseek/deepseek-chat` | same | Any OpenAI-compatible chat model works |
| `EMBEDDING_DIMS` | 4096 | **4096** | Must match Qdrant collection schema |
| `LLM_MODEL` | `deepseek/deepseek-chat` | same | Worker extraction / reflection model |
| `ARQ_MAX_TRIES` | 3 | 3 | Retries per job before it is written to the DLQ |
| `HERMES_DLQ_PATH` | `$HERMES_HOME/wiki_ingest_failures.json` | same | Where the worker writes failed jobs; read by `dlq_manager.py` |

### Optional

| Variable | Purpose |
|----------|---------|
| `ICARUS_OBSIDIAN=1` | Enable Obsidian wikilinks and daily notes |
| `OBSIDIAN_VAULT_PATH` | Vault root (if fabric is a subfolder) |
| `ICARUS_RESULT_MAX_CHARS` | Fallback truncation limit (default 500) |
| `ICARUS_TASK_MAX_CHARS` | Fallback task truncation (default 300) |
| `TOGETHER_API_KEY` | For training/eval tools |
| `OPENROUTER_FULL_API_KEY` | Alternative key for LLM extraction |
| `OPENROUTER_DS_API_KEY` | Alternative key for LLM extraction |
| `DEEPSEEK_API_KEY` | Alternative LLM key; when set, the LLM base switches to `https://api.deepseek.com/v1` |
| `CURATOR_LOG_LEVEL` | Logging level for Vault Curator |
| `VAULT_PATH` | Path to vault root for Vault Curator |

## File locations

| Component | Path |
|-----------|------|
| Workspace memory | `$HERMES_HOME/memories/` |
| Session DB | `$HERMES_HOME/state.db` |
| Fact store DB | `$HERMES_HOME/memory_store.db` |
| Icarus plugin | `$HERMES_HOME/plugins/icarus/` |
| Fabric entries | `$FABRIC_DIR` |
| Wiki files | `$VAULT_PATH/wiki/` |
| Qdrant data | `./qdrant_data/` (Docker volume) |
| Docker compose | Project root |
| Cron scripts | Project scripts directory |

## System requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 8 GB | 16 GB (Qdrant + Redis + ARQ worker) |
| Disk | 20 GB | 50 GB (Qdrant vectors + wiki files) |
| Docker | 24.0+ | Latest stable |
| Python | 3.11+ | 3.11 (tested) |
| Hermes Agent | 0.14.0+ | 0.15.2 (tested) |
| Qdrant | 1.17+ | 1.17.1 (tested) |
