"""
Cognitive Worker ARQ — Memory-as-a-Service (MaaS)
Main entry point for the processing worker.
"""
import asyncio
import functools
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# ─── Load .env ──────────────────────────────────────────────────────────────
ENV_PATH = Path(__file__).parent.parent / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# ─── Logging configuration ──────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cognitive-worker")

# ─── Import ARQ worker ──────────────────────────────────────────────────────
from arq import create_pool, cron
from arq.connections import RedisSettings

from tasks.ingestion import ingest_memory
from tasks.reflection import reflect_on_memories, micro_reflection
from tasks.file_ingestion import ingest_file

# ─── Redis configuration ────────────────────────────────────────────────────
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")

redis_settings = RedisSettings(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD or None,
)

# ─── Retry policy & DLQ ─────────────────────────────────────────────────────
ARQ_MAX_TRIES = int(os.environ.get("ARQ_MAX_TRIES", "3"))
DLQ_PATH = os.path.expanduser(
    os.environ.get("HERMES_DLQ_PATH", "~/.hermes/wiki_ingest_failures.json")
)


def _error_hash(file_id: str, error: str) -> str:
    """Stable 8-char hash for de-duplicating similar failures.

    Identical to ``scripts/dlq_manager.compute_error_hash`` so the host-side
    manager can group entries written by the worker.
    """
    return hashlib.md5(f"{file_id}:{error[:80]}".encode("utf-8")).hexdigest()[:8]


def _write_dlq_entry(entry: dict) -> None:
    """Append a single entry to the DLQ file (atomic tmp-write + os.replace)."""
    directory = os.path.dirname(DLQ_PATH) or "."
    os.makedirs(directory, exist_ok=True)

    existing = []
    if os.path.exists(DLQ_PATH):
        try:
            with open(DLQ_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                existing = data
            elif isinstance(data, dict) and isinstance(data.get("failures"), list):
                existing = data["failures"]
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"DLQ file unreadable ({e}); starting a new one")

    existing.append(entry)

    tmp_path = f"{DLQ_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, DLQ_PATH)


async def on_job_failure(ctx, job) -> None:
    """
    Record a permanently-failed job in the DLQ file.

    NOTE: arq 0.28.0 (the pinned version) has **no** native ``on_job_failure``
    hook — its only hooks are ``on_startup``/``on_shutdown``/``on_job_start``/
    ``on_job_end``/``after_job_end`` and ``arq.worker.get_kwargs`` silently
    drops any unrecognised key. This coroutine therefore implements the
    required behaviour and is invoked by ``_dlq_guard`` (the wrapper every job
    function is registered through), which fires it once a job has exhausted
    ``ARQ_MAX_TRIES``.

    Entry fields mirror ``scripts/dlq_manager.DLQEntry`` exactly:
    ``file, error, timestamp, failure_class, reported, retry_count,
    last_retry, error_hash``.
    """
    args = job.get("args") or ()

    # Prefer a .md path argument (wiki ingest); fall back to the first argument
    # or the job identity.
    file_id = ""
    for candidate in args:
        if isinstance(candidate, str) and candidate.endswith(".md"):
            file_id = candidate
            break
    if not file_id:
        file_id = str(args[0]) if args else f"{job.get('function', 'job')}:{job.get('job_id')}"

    error = str(job.get("error") or "unknown error")

    entry = {
        "file": file_id,
        "error": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "failure_class": "unknown",
        "reported": False,
        "retry_count": 0,
        "last_retry": None,
        "error_hash": _error_hash(file_id, error),
    }

    try:
        _write_dlq_entry(entry)
        logger.error(
            "Job %s:%s exhausted retries → DLQ entry written for %s",
            job.get("function"),
            job.get("job_id"),
            file_id,
        )
    except Exception as e:
        # Never let DLQ bookkeeping crash the worker; but log loudly.
        logger.error(f"Failed to write DLQ entry for {file_id}: {e}")


def _dlq_guard(func):
    """
    Wrap a job coroutine so terminal failures are recorded in the DLQ.

    Runs the wrapped job; on exception, if this was the last permitted attempt
    (``ctx['job_try'] >= ARQ_MAX_TRIES``) it calls ``on_job_failure`` and then
    re-raises so arq still performs its normal retry / give-up bookkeeping.
    ``functools.wraps`` preserves ``__qualname__``, so arq still registers the
    job under its original name (needed for ``enqueue_job("process_...")``).
    """
    @functools.wraps(func)
    async def _wrapped(ctx, *args, **kwargs):
        try:
            return await func(ctx, *args, **kwargs)
        except Exception as exc:
            try:
                job_try = int(ctx.get("job_try", 1) or 1)
            except (TypeError, ValueError):
                job_try = 1
            if job_try >= ARQ_MAX_TRIES:
                await on_job_failure(
                    ctx,
                    {
                        "job_id": ctx.get("job_id"),
                        "function": func.__name__,
                        "args": args,
                        "kwargs": kwargs,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
            raise
    return _wrapped


# ─── Startup/shutdown functions ────────────────────────────────────────────
async def startup(ctx):
    """Connect to Qdrant and validate the collection."""
    from services.local_qdrant import get_qdrant_client, ensure_collection

    logger.info("Worker starting...")
    ctx["qdrant"] = get_qdrant_client()
    await ensure_collection(ctx["qdrant"])
    logger.info("Qdrant connection validated")


async def shutdown(ctx):
    """Clean up connections."""
    logger.info("Worker shutting down...")
    if "qdrant" in ctx:
        await ctx["qdrant"].close()


# ─── ARQ function definitions ────────────────────────────────────────────
async def process_ingestion(ctx, memory_text: str, source: str, tags: list = None):
    """ARQ job: ingests a memory into the vector store."""
    return await ingest_memory(ctx["qdrant"], memory_text, source, tags)


async def process_wiki_file(ctx, file_path: str):
    """ARQ job: ingests a .md file from the vault."""
    return await ingest_file(ctx["qdrant"], file_path)


async def process_reflection(ctx):
    """ARQ job: runs periodic reflection."""
    return await reflect_on_memories(ctx["qdrant"])


async def process_micro_reflection(ctx):
    """ARQ job: runs on-demand micro-reflection (Phase 3)."""
    return await micro_reflection(ctx["qdrant"])


# Wrap every job coroutine once so that both the functions registry and the
# cron jobs share the same DLQ-guarded callable (names are preserved by wraps).
process_ingestion = _dlq_guard(process_ingestion)
process_wiki_file = _dlq_guard(process_wiki_file)
process_reflection = _dlq_guard(process_reflection)
process_micro_reflection = _dlq_guard(process_micro_reflection)


# ─── ARQ Worker Settings ─────────────────────────────────────────────────
class WorkerSettings:
    """ARQ worker settings."""
    redis_settings = redis_settings
    functions = [process_ingestion, process_reflection, process_micro_reflection, process_wiki_file]
    on_startup = startup
    on_shutdown = shutdown
    max_jobs = int(os.environ.get("ARQ_MAX_JOBS", "10"))
    job_timeout = int(os.environ.get("ARQ_JOB_TIMEOUT", "300"))
    keep_result = int(os.environ.get("ARQ_KEEP_RESULT", "3600"))
    max_tries = ARQ_MAX_TRIES
    cron_jobs = [
        # Reflection every 2 hours (minute 0 of even hours)
        cron(process_reflection, hour={0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22}, minute=0),
        # Micro-reflection now triggered via reflection_trigger.py (idle + budget)
    ]


# ─── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--run-worker":
        from arq.worker import run_worker
        logger.info("Starting ARQ worker (programmatic)...")
        run_worker(WorkerSettings)
    else:
        print("Usage: python main.py --run-worker")
        print("")
        print("To enqueue jobs via Redis, use the enqueue functions")
        print("or the Hermes cron scripts in the scripts/ directory.")