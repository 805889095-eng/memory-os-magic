# Setup Guide

> **Prefer automated install?** Run `curl -sSL https://raw.githubusercontent.com/ClaudioDrews/memory-os/main/setup.sh | bash` — one command, 10 phases, fully idempotent. This manual guide is kept for reference and troubleshooting.

> Step-by-step installation of the Memory OS stack. Assumes Hermes Agent is already installed and configured.

## Prerequisites

- Hermes Agent 0.14.0+ (tested on 0.15.2)
- Python 3.11+
- Docker 24.0+
- **A remote Embedding endpoint and a remote LLM endpoint, each with its base_url + api_key** — required. There is no offline path: `EMBEDDING_API_KEY` / `LLM_API_KEY` are mandatory, and a missing key is a hard error, never a silent fallback. See [Layer 5: Qdrant](../layers/05-qdrant.md)
- 16 GB RAM recommended (8 GB minimum)

## Installation

### 1. Icarus Plugin (bundled)

```bash
# Copy the bundled Icarus fork into the Hermes plugins directory
cp -r icarus/ ~/.hermes/plugins/icarus/
```

### 2. Database Setup

Install the Python dependencies first:

```bash
pip install -r requirements.txt
```

Memory OS requires two SQLite databases with FTS5 full-text search indexes:
`state.db` (session history, lineage, reflection budget) and `memory_store.db`
(facts, entities, memory banks). The setup script creates both with idempotent
`CREATE TABLE IF NOT EXISTS` statements — safe to run multiple times.

```bash
python setup/setup_db.py
```

**What it creates:**

| Database | Tables |
|---|---|
| `state.db` | `sessions`, `messages`, `messages_fts` (FTS5), `messages_fts_trigram`, `lineage`, `reflection_budget`, `compression_locks`, `schema_version`, `state_meta` |
| `memory_store.db` | `entities`, `facts`, `facts_fts` (FTS5), `fact_entities`, `memory_banks` |

Options:

```bash
python setup/setup_db.py --dry-run          # preview without executing
python setup/setup_db.py --state-db /custom/path/state.db
python setup/setup_db.py --memory-db /custom/path/memory_store.db
```

Environment variables override defaults:

```bash
export STATE_DB_PATH=/home/your-user/.hermes/state.db
export MEMORY_STORE_PATH=/home/your-user/.hermes/memory_store.db
```

### 3. Enable Icarus in Hermes Config

Icarus must be registered as an enabled plugin. Edit `~/.hermes/config.yaml`:

```yaml
enabled:
  - hermes-achievements       # optional
  - icarus                    # required — activates fabric tools + context injection hooks
```

Then restart the gateway:

```bash
hermes gateway restart
```

Verify the plugin loaded:

```bash
hermes status
# → Should show: icarus v0.3.0 (16 tools, 4 hooks)
```

### 4. Docker Infrastructure

The compose file lives in the `docker/` directory of this repository and must be run **in-place** — the worker build context (`./worker`) is relative to the compose file location.

```bash
# Navigate to the docker directory inside your clone
cd /path/to/memory-os/docker

# Create .env with required variables
cat > .env << EOF
# Required: the remote embedding + LLM endpoints (OpenAI-compatible, base_url + api_key)
EMBEDDING_API_BASE=https://openrouter.ai/api/v1
EMBEDDING_MODEL=qwen/qwen3-embedding-8b
EMBEDDING_API_KEY=sk-or-...
LLM_API_BASE=https://openrouter.ai/api/v1
LLM_MODEL=deepseek/deepseek-chat
LLM_API_KEY=sk-or-...
REDIS_PASSWORD=$(openssl rand -hex 16)
# Optional overrides (defaults shown)
OPENROUTER_API_KEY=sk-or-...          # fallback for both endpoints when the keys above are empty
EMBEDDING_DIMS=4096
ARQ_MAX_TRIES=3
HERMES_DLQ_PATH=/hermes/wiki_ingest_failures.json
COLLECTION_NAME=knowledge_base
LOG_LEVEL=INFO
EOF

# Optional — if you want the Docker stack to use your existing production
# directories instead of local test volumes, uncomment and set these:
# MEMORY_OS_WIKI_PATH=/home/your-user/vault/wiki
# MEMORY_OS_HERMES_HOME=/home/your-user/.hermes
# MEMORY_OS_FABRIC_DIR=/home/your-user/vault/fabric
# ⚠️  Do NOT set these to production paths unless you understand the risk.
#     The worker mounts /fabric and /hermes as read-write.

# Start the stack
docker compose up -d
```

Verify all three services are running:

```bash
docker compose ps
# → Should show redis, qdrant, and worker all with Status: Up

curl -s http://localhost:6333/healthz  # → {"title":"ok","version":"1.17.1"}
redis-cli -a "$REDIS_PASSWORD" ping    # → PONG
```

### 5. Environment Variables

Add to your Hermes profile `.env` (e.g. `~/.hermes/.env`):

```bash
# Required
FABRIC_DIR=/home/your-user/vault/fabric

# Required — remote embedding endpoint (base_url + api_key). No offline path:
# a missing key is a hard error, never a silent fallback.
EMBEDDING_API_BASE=https://openrouter.ai/api/v1
EMBEDDING_MODEL=qwen/qwen3-embedding-8b
EMBEDDING_API_KEY=sk-or-...
EMBEDDING_DIMS=4096

# Required — remote LLM endpoint used by the ARQ worker for wiki extraction
# and reflection (OpenAI-compatible chat completions).
LLM_API_BASE=https://openrouter.ai/api/v1
LLM_MODEL=deepseek/deepseek-chat
LLM_API_KEY=sk-or-...

# Strongly recommended
ICARUS_EXTRACTION_MAX_TOKENS=4096
ICARUS_EXTRACTION_MODEL=deepseek/deepseek-chat

# ARQ worker resilience
ARQ_MAX_TRIES=3
HERMES_DLQ_PATH=/home/your-user/.hermes/wiki_ingest_failures.json

# OpenRouter compatibility: when EMBEDDING_API_BASE / LLM_API_BASE contains
# "openrouter" and the matching *_API_KEY is empty, OPENROUTER_API_KEY is used.
# OPENROUTER_API_KEY=sk-or-...

# Optional
ICARUS_OBSIDIAN=1
ICARUS_RESULT_MAX_CHARS=500
ICARUS_TASK_MAX_CHARS=300
```

**⚠️ Use absolute paths.** The Hermes gateway runs as a systemd service — `~` is not expanded. Always use `/home/your-user/...`.

### 6. Core File Modifications

Apply the additions documented in
[setup/rulebook.md](rulebook.md) and
[modifications/soul-rulebook.md](../modifications/soul-rulebook.md):

**`~/.hermes/rulebook.md`** — apply the three amendments from
`modifications/execution-agent-protocol.md` (see `setup/rulebook.md`
for a summary). Each amendment targets a specific section of the
Execution Agent protocol — insert it after the referenced section.

- Each amendment starts with `<!-- Memory OS amendment — do not duplicate -->`.
  Before applying, check whether this marker already exists in your
  rulebook — if it does, skip that amendment.

**`SOUL.md`** — add Ground Truth level 2 (injected memory) and context
injection convention as documented in `modifications/soul-rulebook.md`.

**`~/.hermes/.env`** — set `HERMES_AGENT_NAME=hermes` (or any unique name).
This distinguishes your agent in fabric entries and enables multi-agent
handoff. Without it, all entries use the fallback `agent: "agent"` and
cross-agent features are disabled.

These modifications ensure the agent treats injected memory as more
authoritative than training knowledge, and knows where to find
persisted information without re-discovering it.

### 7. Wiki + Vault Setup

Memory OS stores its knowledge pipeline inside an Obsidian vault. The vault
path is user-specific — set it as an environment variable first:

```bash
# Set this to your Obsidian vault path
export VAULT_PATH=/home/your-user/path/to/vault
```

Create the wiki directory structure:

```bash
mkdir -p $VAULT_PATH/wiki/{raw,concepts,entities,comparisons,_meta,_archive}
```

**What goes where:**
- `raw/` — source documents to be ingested and curated
- `concepts/`, `entities/`, `comparisons/` — auto-generated by vault-curator
- `_meta/` — pipeline metadata (SCHEMA.md, indexes)
- `_archive/` — aged-out content from decay scanner

The wiki starts empty. Add source documents to `raw/` and the wiki-continuous-ingest
cronjob (step 7) will begin extracting structured pages.

**Optional — Vault Curator:** For automatic enrichment, semantic linking, and
MOC generation, install [vault-curator](https://github.com/ClaudioDrews/vault-curator)
as a separate tool. It runs independently and is not required for Memory OS
core functionality.

### 8. Maintenance Scripts

The `scripts/` directory in this repository contains the maintenance tools
that keep the memory stack healthy. `setup.sh` installs the scheduled jobs for
you (Phase 7b); they live inside an idempotent crontab block, so re-running the
installer never duplicates an entry.

**Scheduled jobs (installed by `setup.sh`):**

| Script | Schedule | Purpose |
|---|---|---|
| `wiki_continuous_ingest.py` | Hourly | Detects new/modified .md files and enqueues them to the ARQ worker |
| `reflection_trigger.py` | Every 5 min | Triggers micro_reflection when the ARQ worker is idle |
| `decay_scanner.py` | Weekly (Sun 03:00) | Archives low-importance chunks based on age and importance_score |
| `semantic_dedup.py` | Monthly (1st 03:00) | Scans for near-duplicate vectors (cosine > 0.92) |
| `dlq_manager.py --report` | Every 2 hours | Reads, classifies, and reports dead letter queue failures |
| `icarus/evolution.py --decay 30` | Daily 03:30 | Decays `trust_score` of facts unused for ≥30 days (forgetting half of the self-evolution loop) |

**On-demand / one-shot helpers (not scheduled):**

| Script | Purpose |
|---|---|
| `backfill_decay_metadata.py` | Populates missing metadata (created_at, last_accessed_at, importance_score, confidence_score) for the decay scanner — run once before enabling decay |
| `bulk_wiki_ingest.py` | One-shot bulk ingestion of all existing wiki content into `knowledge_base` |
| `pre_validator.py` | Semantic decision linter — queries `knowledge_base` before I/O or API actions |
| `test_ingestion.py` | End-to-end ingestion test (enqueue → worker → embedding → Qdrant upsert) |
| `verify_soul_config.py` | Checks that SOUL.md has the Memory OS Ground Truth hierarchy |

`bm25_tf.py` and `context_enhancer.py` are library modules used by the pipeline, not standalone jobs.

**Manual crontab (equivalent to what `setup.sh` installs):**

```bash
# >>> memory-os scheduler >>>
0 * * * *    cd /path/to/memory-os && HERMES_HOME=$HOME/.hermes python3 scripts/wiki_continuous_ingest.py >> $HOME/.hermes/logs/wiki-ingest.log 2>&1
*/5 * * * *  cd /path/to/memory-os && HERMES_HOME=$HOME/.hermes python3 scripts/reflection_trigger.py     >> $HOME/.hermes/logs/reflection_trigger.log 2>&1
0 3 * * 0    cd /path/to/memory-os && HERMES_HOME=$HOME/.hermes python3 scripts/decay_scanner.py         >> $HOME/.hermes/logs/decay_scanner.log 2>&1
0 3 1 * *    cd /path/to/memory-os && HERMES_HOME=$HOME/.hermes python3 scripts/semantic_dedup.py        >> $HOME/.hermes/logs/semantic_dedup.log 2>&1
0 */2 * * *  cd /path/to/memory-os && HERMES_HOME=$HOME/.hermes python3 scripts/dlq_manager.py --report  >> $HOME/.hermes/logs/dlq_manager.log 2>&1
30 3 * * *   cd /path/to/memory-os && HERMES_HOME=$HOME/.hermes python3 icarus/evolution.py --decay 30   >> $HOME/.hermes/logs/fact_decay.log 2>&1
# <<< memory-os scheduler <<<
```

On platforms without `crontab` (Windows / git-bash) `setup.sh` skips this step with a
warning instead of failing — schedule the six scripts with your OS task scheduler.

**Before enabling decay scanner:** run `backfill_decay_metadata.py` once to
populate `created_at`, `last_accessed_at`, `importance_score`, and
`confidence_score` on existing Qdrant points. Without backfill, the decay
scanner will find zero eligible points.

**Exempting collections:** Set `DECAY_EXEMPT_PREFIXES` and
`DEDUP_EXEMPT_PREFIXES` env vars (comma-separated prefixes) to exclude
specific Qdrant collections from automated maintenance.

### 9. Gateway Restart

```bash
hermes gateway restart
```

Changes to `.env`, `SOUL.md`, `rulebook.md`, and Icarus plugin code only take effect after restart.

### 10. Verify

Inside Hermes chat:

```
/plugins
# → Should show: icarus v0.3.0 (16 tools, 4 hooks)

fabric_brief()
# → Should show recent fabric entries (initially empty)

qdrant_search("test query")
# → Should return results from knowledge_base (if wiki has content)

fact_store(action='probe', entity='test')
# → Should return empty (no facts stored yet)
```

## What to expect

**Day 1:** Infrastructure running. Fabric entries begin accumulating at session end. Qdrant indexing starts as wiki files are added.

**Week 1:** Context injection active. Agent references past decisions automatically. Wiki pipeline producing curated pages from raw documents.

**Month 1:** Decay scanner has aged content to evaluate. Structured facts accumulating with trust scores.

## Troubleshooting

### Qdrant collection shows 0 points
Check: `EMBEDDING_DIMS=4096` matches collection schema. Mismatch → vectors rejected silently.

### Fabric entries are truncated
Check: `ICARUS_EXTRACTION_MAX_TOKENS=4096` in `.env` AND gateway was restarted after setting it.

### Memory tool reports "Icarus write conflict"
Icarus is writing to MEMORY.md instead of CREATIVE.md. Verify Icarus fork is installed (not upstream esaradev version).

### Context injection not working
Check: OpenRouter API key is set, `context_enhancer.py` can import, gateway restarted after `hooks.py` edits.

### Decay scanner produces "0 archived" every week
Most likely: point payloads missing `last_accessed_at` or `importance_score` metadata. Run backfill before enabling decay.

### Multiple collections in Qdrant dashboard

The Memory OS uses the `knowledge_base` collection exclusively. Other
collections you may see (e.g., from other Hermes agent plugins or standalone
agents) are safe to coexist — Qdrant isolates each collection at the storage
and query level. Do NOT delete collections you did not create — they may
belong to other agents sharing the same Qdrant instance.
