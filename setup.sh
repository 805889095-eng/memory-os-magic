#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Memory OS — Setup Script
# ──────────────────────────────────────────────────────────────────────────────
# Installs the complete Memory OS stack into your Hermes Agent.
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/ClaudioDrews/memory-os/main/setup.sh | bash
#
# Or, if you already cloned the repo:
#   bash setup.sh
#
# What this script does:
#   1. Checks prerequisites (Docker, Python, Hermes)
#   2. Clones the repo (if needed)
#   3. Installs Python dependencies
#   4. Creates SQLite databases (state.db, memory_store.db)
#   5. Installs the Icarus plugin
#   6. Creates wiki/vault directory structure
#   7. Starts Redis + Qdrant + Worker (Docker Compose)
#   8. Installs the 6-job maintenance scheduler (crontab; skipped on Windows)
#   9. Configures environment variables
#  10. Applies rulebook modifications
#
# Idempotent — safe to run multiple times.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Safe defaults for optional env vars ──────────────────────────────────────
# Must come before any reference to these names.  set -u would abort with
# \"unbound variable\" if they were never exported by the caller.
QDRANT_API_KEY="${QDRANT_API_KEY:-}"
REDIS_PASSWORD="${REDIS_PASSWORD:-}"

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0

ok()  { printf "  ${GREEN}✅${NC} %s\n" "$1"; PASS=$((PASS + 1)); }
fail() { printf "  ${RED}❌${NC} %s\n" "$1"; FAIL=$((FAIL + 1)); }
warn() { printf "  ${YELLOW}⚠️${NC}  %s\n" "$1"; WARN=$((WARN + 1)); }
info() { printf "  📘 %s\n" "$1"; }

banner() {
    echo ""
    echo -e "${BOLD}── $1 ──${NC}"
    echo ""
}

# ── Detect script directory ──────────────────────────────────────────────────
# When run via curl|bash, SCRIPT_DIR is the current directory.
# When run from a cloned repo, it's the script's location.
if [ -n "${BASH_SOURCE[0]:-}" ] && [ "${BASH_SOURCE[0]}" != "bash" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    SCRIPT_DIR="$(pwd)"
fi

REPO_URL="https://github.com/ClaudioDrews/memory-os.git"
REPO_DIR="${HOME}/memory-os"
HERMES_HOME="${HOME}/.hermes"
VAULT_PATH="${VAULT_PATH:-${HOME}/vault}"
ENV_FILE="${HERMES_HOME}/.env"

# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Bootstrap — clone repo if needed
# ──────────────────────────────────────────────────────────────────────────────
banner "Phase 1: Bootstrap"

if [ -d "${REPO_DIR}/.git" ]; then
    ok "Repo already exists at ${REPO_DIR}"
    cd "${REPO_DIR}"
else
    info "Cloning Memory OS..."
    git clone "${REPO_URL}" "${REPO_DIR}" 2>&1 | tail -1
    cd "${REPO_DIR}"
    ok "Repo cloned to ${REPO_DIR}"
fi

# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Pre-flight Checks
# ──────────────────────────────────────────────────────────────────────────────
banner "Phase 2: Pre-flight Checks"

# Docker
if docker info >/dev/null 2>&1; then
    ok "Docker $(docker --version | awk '{print $3}' | tr -d ',')"
else
    fail "Docker is not running — install and start Docker first"
    exit 1
fi

# Docker Compose
if docker compose version >/dev/null 2>&1; then
    ok "Docker Compose $(docker compose version --short 2>/dev/null || echo 'ok')"
else
    warn "Docker Compose plugin not detected — required to start the stack"
fi

# Python
PYTHON_VERSION=$(python3 --version 2>/dev/null | awk '{print $2}' || echo "none")
if [ "$PYTHON_VERSION" != "none" ]; then
    ok "Python ${PYTHON_VERSION}"
else
    fail "Python 3 not found"
    exit 1
fi

# Hermes
if command -v hermes >/dev/null 2>&1 || [ -f "${HERMES_HOME}/hermes-agent/cli.py" ]; then
    ok "Hermes Agent detected at ${HERMES_HOME}"
else
    warn "Hermes Agent CLI not found — some features will be limited"
fi

# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Python Dependencies
# ──────────────────────────────────────────────────────────────────────────────
banner "Phase 3: Python Dependencies"

if [ -f "requirements.txt" ]; then
    # Install in current environment (local user or venv)
    if pip install --user -r requirements.txt --quiet 2>&1 | tail -3; then
        ok "Python dependencies installed"
    else
        warn "pip install failed — trying with --break-system-packages"
        pip install --break-system-packages -r requirements.txt --quiet 2>&1 || {
            fail "Could not install Python dependencies"
            exit 1
        }
        ok "Python dependencies installed (--break-system-packages)"
    fi
else
    fail "requirements.txt not found at $(pwd)"
    exit 1
fi

# ──────────────────────────────────────────────────────────────────────────────
# Phase 4: SQLite Databases
# ──────────────────────────────────────────────────────────────────────────────
banner "Phase 4: Database Setup"

if [ -f "setup/setup_db.py" ]; then
    python3 setup/setup_db.py 2>&1 && \
        ok "SQLite databases created (state.db, memory_store.db)" || \
        fail "setup_db.py failed"
else
    fail "setup/setup_db.py not found"
    exit 1
fi

# ──────────────────────────────────────────────────────────────────────────────
# Phase 5: Icarus Plugin
# ──────────────────────────────────────────────────────────────────────────────
banner "Phase 5: Icarus Plugin"

ICARUS_DEST="${HERMES_HOME}/plugins/icarus"

if [ -d "icarus" ]; then
    mkdir -p "${HERMES_HOME}/plugins"
    cp -r icarus/ "${ICARUS_DEST}/"
    ok "Icarus plugin installed at ${ICARUS_DEST}"
else
    fail "icarus/ directory not found"
    exit 1
fi

# ──────────────────────────────────────────────────────────────────────────────
# Phase 5b: Context Enhancer Symlink
# ──────────────────────────────────────────────────────────────────────────────
banner "Phase 5b: Context Enhancer"

CE_SRC="${REPO_DIR}/scripts/context_enhancer.py"
CE_DEST="${HERMES_HOME}/scripts/context_enhancer.py"

if [ -f "${CE_SRC}" ]; then
    mkdir -p "${HERMES_HOME}/scripts"
    ln -sf "${CE_SRC}" "${CE_DEST}"
    ok "context_enhancer.py symlinked to ${CE_DEST}"
else
    fail "${CE_SRC} not found"
    exit 1
fi

# ──────────────────────────────────────────────────────────────────────────────
# Phase 6: Wiki + Vault Structure (BEFORE Docker — prevents root ownership)
# ──────────────────────────────────────────────────────────────────────────────
banner "Phase 6: Wiki & Vault"

mkdir -p "${VAULT_PATH}/wiki/"{raw,concepts,entities,comparisons,_meta,_archive}
mkdir -p "${VAULT_PATH}/fabric"
# Ensure Docker worker can read wiki files (umask may create 600)
chmod -R 755 "${VAULT_PATH}/wiki"
chmod -R 755 "${VAULT_PATH}/fabric"
ok "Directory structure created at ${VAULT_PATH}"
info "Permissions set to 755 on wiki/ and fabric/ (ensures Docker worker read access)"

# ──────────────────────────────────────────────────────────────────────────────
# Phase 7: Docker Stack
# ──────────────────────────────────────────────────────────────────────────────
banner "Phase 7: Docker Stack"

DOCKER_DIR="${REPO_DIR}/docker"

if [ ! -d "${DOCKER_DIR}" ]; then
    fail "docker/ directory not found at ${REPO_DIR}"
    exit 1
fi

cd "${DOCKER_DIR}"

# Detect API key from Hermes .env
OPENROUTER_KEY=""
if [ -f "${ENV_FILE}" ]; then
    OPENROUTER_KEY=$(grep -oP '(?:OPENROUTER.*API_KEY|LLM_API_KEY)=\K.*' "${ENV_FILE}" 2>/dev/null | head -1 || true)
fi

if [ -z "${OPENROUTER_KEY}" ]; then
    echo ""
    echo -e "  ${YELLOW}Could not find an API key in Hermes .env.${NC}"
    echo "  The worker needs an embedding-capable API key (OpenRouter or compatible)."
    echo ""
    read -r -p "  Paste your API key (e.g. sk-or-v1-...): " OPENROUTER_KEY
    echo ""
fi

# Generate random Redis password
REDIS_PW=$(openssl rand -hex 16)

# Create Docker Compose .env
cat > .env << DOCKERENV
OPENROUTER_API_KEY=${OPENROUTER_KEY}
REDIS_PASSWORD=${REDIS_PW}
QDRANT_API_KEY=
EMBEDDING_DIMS=4096
COLLECTION_NAME=knowledge_base
LOG_LEVEL=INFO
MEMORY_OS_WIKI_PATH=${VAULT_PATH}/wiki
MEMORY_OS_HERMES_HOME=${HERMES_HOME}
MEMORY_OS_FABRIC_DIR=${VAULT_PATH}/fabric
DOCKERENV

ok "docker/.env created"

# Patch docker-compose.yml: remove Qdrant API key line when empty (v1.17+ enables auth on empty key)
if [ -z "${QDRANT_API_KEY}" ]; then
    info "Qdrant API key not set — disabling Qdrant authentication"
    sed -i '/^.*QDRANT__SERVICE__API_KEY:/s/.*/      # QDRANT__SERVICE__API_KEY: disabled (no key set)/' docker-compose.yml
fi

# Pull pre-built images first (Redis, Qdrant) — fast
info "Downloading pre-built images (Redis, Qdrant)..."
docker compose pull redis qdrant 2>&1 | tail -3
ok "Base images downloaded"

# Build worker image — SLOW on first run
info "Building worker image (may take a few minutes on first run)..."
info "  (Future builds will use Docker cache)"
if docker compose build worker 2>&1; then
    ok "Worker image built"
else
    fail "Failed to build worker image"
    exit 1
fi

# Start everything
info "Starting containers..."
if docker compose up -d 2>&1; then
    ok "Docker stack started (redis, qdrant, worker)"
else
    fail "docker compose up failed — check Docker"
    exit 1
fi

# Wait for healthy
info "Waiting for services to become healthy..."
sleep 3
if docker compose ps --format json 2>/dev/null | grep -q '"Health":"healthy"'; then
    ok "All services healthy"
else
    warn "Services may still be starting — check with: docker compose ps"
fi

# Return to repo directory
cd "${REPO_DIR}"

# ──────────────────────────────────────────────────────────────────────────────
# Phase 7b: Maintenance Scheduler (crontab)
# ──────────────────────────────────────────────────────────────────────────────
banner "Phase 7b: Maintenance Scheduler"

# All paths derive from HERMES_HOME / the repo root — never hardcode user paths.
if [ -f "${SCRIPT_DIR}/scripts/wiki_continuous_ingest.py" ]; then
    REPO_ROOT="${SCRIPT_DIR}"
else
    REPO_ROOT="${REPO_DIR}"
fi
CRON_BEGIN="# >>> memory-os scheduler >>>"
CRON_END="# <<< memory-os scheduler <<<"
CRON_LOG_DIR="${HERMES_HOME}/logs"
mkdir -p "${CRON_LOG_DIR}" 2>/dev/null || true

# The six maintenance jobs: schedule | python script | log file | purpose.
CRON_WIKI="0 * * * * cd ${REPO_ROOT} && HERMES_HOME=${HERMES_HOME} python3 scripts/wiki_continuous_ingest.py >> ${CRON_LOG_DIR}/wiki-ingest.log 2>&1"
CRON_REFLECT="*/5 * * * * cd ${REPO_ROOT} && HERMES_HOME=${HERMES_HOME} python3 scripts/reflection_trigger.py >> ${CRON_LOG_DIR}/reflection_trigger.log 2>&1"
CRON_DECAY="0 3 * * 0 cd ${REPO_ROOT} && HERMES_HOME=${HERMES_HOME} python3 scripts/decay_scanner.py >> ${CRON_LOG_DIR}/decay_scanner.log 2>&1"
CRON_DEDUP="0 3 1 * * cd ${REPO_ROOT} && HERMES_HOME=${HERMES_HOME} python3 scripts/semantic_dedup.py >> ${CRON_LOG_DIR}/semantic_dedup.log 2>&1"
CRON_DLQ="0 */2 * * * cd ${REPO_ROOT} && HERMES_HOME=${HERMES_HOME} python3 scripts/dlq_manager.py --report >> ${CRON_LOG_DIR}/dlq_manager.log 2>&1"
# Trust-score decay for the facts store (the self-evolution loop's forgetting half).
# Runs daily at 03:30 so it never collides with the 03:00 weekly/monthly scanners.
CRON_FACTDECAY="30 3 * * * cd ${REPO_ROOT} && HERMES_HOME=${HERMES_HOME} python3 icarus/evolution.py --decay 30 >> ${CRON_LOG_DIR}/fact_decay.log 2>&1"

info "Installing 6 maintenance cron jobs:"
info "  hourly             -> scripts/wiki_continuous_ingest.py   (wiki incremental ingestion)"
info "  every 5 minutes    -> scripts/reflection_trigger.py       (idle micro-reflection trigger)"
info "  weekly (Sun 03:00) -> scripts/decay_scanner.py            (Qdrant decay + archival)"
info "  monthly (1st 03:00)-> scripts/semantic_dedup.py           (near-duplicate scan)"
info "  every 2 hours      -> scripts/dlq_manager.py --report     (dead-letter queue report)"
info "  daily (03:30)      -> icarus/evolution.py --decay 30      (facts trust-score decay)"

if ! command -v crontab >/dev/null 2>&1; then
    warn "crontab not available on this platform (e.g. Windows / git-bash) — skipping scheduler installation"
    info "The stack still works; schedule the 6 scripts above with your OS scheduler if needed."
else
    EXISTING_CRON="$(crontab -l 2>/dev/null || true)"

    # Idempotency: drop any previous managed block, then append the fresh one.
    CLEANED_CRON="$(printf '%s\n' "${EXISTING_CRON}" | awk -v b="${CRON_BEGIN}" -v e="${CRON_END}" '
        $0 == b { skip = 1; next }
        $0 == e { skip = 0; next }
        !skip   { print }
    ')"

    if {
        printf '%s\n' "${CLEANED_CRON}"
        printf '%s\n' "${CRON_BEGIN}"
        printf '%s\n' "${CRON_WIKI}"
        printf '%s\n' "${CRON_REFLECT}"
        printf '%s\n' "${CRON_DECAY}"
        printf '%s\n' "${CRON_DEDUP}"
        printf '%s\n' "${CRON_DLQ}"
        printf '%s\n' "${CRON_FACTDECAY}"
        printf '%s\n' "${CRON_END}"
    } | crontab -; then
        ok "Maintenance scheduler installed (6 cron jobs, idempotent managed block)"
    else
        warn "crontab update failed — schedule the 6 scripts manually"
    fi
fi

# ──────────────────────────────────────────────────────────────────────────────
# Phase 8: Environment Variables
# ──────────────────────────────────────────────────────────────────────────────
banner "Phase 8: Hermes .env"

if [ ! -f "${ENV_FILE}" ]; then
    warn "${ENV_FILE} not found — creating a new one"
    touch "${ENV_FILE}"
fi

add_env() {
    local key="$1"
    local value="$2"
    if grep -q "^${key}=" "${ENV_FILE}" 2>/dev/null; then
        # Already exists — don't overwrite
        return 0
    fi
    echo "${key}=${value}" >> "${ENV_FILE}"
}

add_env "FABRIC_DIR" "${VAULT_PATH}/fabric"
add_env "ICARUS_EXTRACTION_MAX_TOKENS" "4096"
add_env "ICARUS_EXTRACTION_MODEL" "deepseek/deepseek-chat"
add_env "EMBEDDING_API_BASE" "https://openrouter.ai/api/v1"
add_env "EMBEDDING_MODEL" "qwen/qwen3-embedding-8b"
add_env "EMBEDDING_API_KEY" ""
add_env "EMBEDDING_DIMS" "4096"
add_env "LLM_API_BASE" "https://openrouter.ai/api/v1"
add_env "LLM_MODEL" "deepseek/deepseek-chat"
add_env "LLM_API_KEY" ""
add_env "ARQ_MAX_TRIES" "3"
add_env "HERMES_DLQ_PATH" "${HERMES_HOME}/wiki_ingest_failures.json"
add_env "HERMES_AGENT_NAME" "hermes"
add_env "REDIS_PASSWORD" "${REDIS_PW}"
add_env "OPENROUTER_API_KEY" "${OPENROUTER_KEY}"

ok "Environment variables added to Hermes .env"

# ──────────────────────────────────────────────────────────────────────────────
# Phase 9: Rulebook Modifications
# ──────────────────────────────────────────────────────────────────────────────
banner "Phase 9: Rulebook"

SOUL_FILE="${HERMES_HOME}/SOUL.md"
RULEBOOK="${HERMES_HOME}/rulebook.md"
PROTOCOL_FILE="${REPO_DIR}/modifications/execution-agent-protocol.md"
MARKER="Mandatory Pre-Action Protocol"

if [ ! -f "${PROTOCOL_FILE}" ]; then
    warn "execution-agent-protocol.md not found — skipping modifications"
else
    # Try SOUL.md first — behavioral tests show 3/6 compliance when protocol
    # is in SOUL.md vs 0/6 when it is only in rulebook.md.
    if [ -f "${SOUL_FILE}" ]; then
        if grep -q "${MARKER}" "${SOUL_FILE}" 2>/dev/null; then
            ok "Mandatory Pre-Action Protocol already in SOUL.md"
        else
            echo "" >> "${SOUL_FILE}"
            echo "<!-- Memory OS additions — do not duplicate -->" >> "${SOUL_FILE}"
            cat "${PROTOCOL_FILE}" >> "${SOUL_FILE}"
            ok "Mandatory Pre-Action Protocol appended to SOUL.md"
        fi
    elif [ -f "${RULEBOOK}" ]; then
        if grep -q "${MARKER}" "${RULEBOOK}" 2>/dev/null; then
            ok "Rulebook amendments already applied"
        else
            echo "" >> "${RULEBOOK}"
            cat "${PROTOCOL_FILE}" >> "${RULEBOOK}"
            ok "Mandatory Pre-Action Protocol appended to rulebook"
        fi
    else
        warn "Neither SOUL.md nor rulebook.md found"
        info "To install the protocol manually:"
        info "  cat ${PROTOCOL_FILE} >> ${HERMES_HOME}/SOUL.md"
    fi
fi

# ──────────────────────────────────────────────────────────────────────────────
# Phase 10: Gateway
# ──────────────────────────────────────────────────────────────────────────────
banner "Phase 10: Gateway"

if command -v hermes >/dev/null 2>&1; then
    info "Restarting Hermes gateway..."
    if hermes gateway restart 2>&1; then
        ok "Gateway restarted"
    else
        warn "Gateway restart failed — restart manually with: hermes gateway restart"
    fi
else
    warn "'hermes' command not available — restart the gateway manually"
fi

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
banner "Summary"

echo "  Passed:  ${PASS}"
echo "  Failed:  ${FAIL}"
echo "  Warnings: ${WARN}"
echo ""

if [ "${FAIL}" -eq 0 ]; then
    echo -e "  ${GREEN}${BOLD}✅ Memory OS installed successfully!${NC}"
    echo ""
    echo "  To verify:"
    echo "    • /plugins          → should show 'icarus'"
    echo "    • docker compose ps → 3 services (redis, qdrant, worker)"
    echo "    • fabric_brief()    → fabric entries (initially empty)"
    echo "    • qdrant_search()   → semantic search (requires populated wiki)"
    echo ""
    echo "  Next step: add .md files to ${VAULT_PATH}/wiki/raw/"
    echo "  and the ingestion pipeline will index them automatically."
    echo ""
else
    echo -e "  ${RED}${BOLD}❌ ${FAIL} error(s) found — review the output above.${NC}"
    exit 1
fi
