"""Shared state: fabric I/O, retriever, training helpers, model registry."""

import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import urllib.error
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# HERMES_HOME is ALWAYS resolved. It used to be ``None`` when the env var was
# unset, which silently produced multi-profile misrouting (e.g. state.db under
# ~/.hermes while the plugin lived under <home>/plugins) and made the
# ``if not HERMES_HOME: return`` guard in write_memory_file skip writing
# entirely. Falling back to ~/.hermes matches setup_db.py's default.
HERMES_HOME = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
FABRIC_DIR = Path(os.environ.get("FABRIC_DIR", HERMES_HOME / "fabric"))
AGENT_NAME = os.environ.get("HERMES_AGENT_NAME", "")
PLUGIN_DIR = Path(__file__).parent

if not AGENT_NAME and ".hermes-" in str(HERMES_HOME):
    AGENT_NAME = str(HERMES_HOME).split(".hermes-")[-1].rstrip("/")

# ── Shared regexes (used by hooks.py and scoring) ────────
# These regexes gate two things that silently never fire for Chinese: the
# ``post_llm_call`` decision capture and ``score_session``'s decision component
# (weight 3 of 10 — the single largest term). Upstream matched only English
# verbs, so a Chinese session that fixed a bug and shipped a fix scored
# decision=0.0 and was filtered out by on_session_end's 0.2 threshold, meaning
# NOTHING was ever preserved from Chinese sessions. CJK alternatives are added
# inline rather than as a separate pass so there is one authoritative pattern.
DECISION_RE = re.compile(
    r"(?i)\b(decided|resolved|completed|fixed|deployed|shipped|reviewed|approved|rejected)\b"
    r"|(决定|确定了|已解决|解决|修复|修好|已完成|完成|部署|上线|发布|合并|通过|批准|驳回|改用|改为|采用)"
)
OUTCOME_RE = re.compile(
    r"(?i)(result:|outcome:|conclusion:|because|root cause|instead of|\d+%|\d+x)"
    r"|(结果|结论|原因是|根因|根本原因|实测|验证|报错|错误|失败|成功|全过|通过|提升|降到|从.*改)",
)
COMPLETION_RE = re.compile(
    r"(?i)\b(completed|finished|done|shipped|deployed|resolved|closed|merged|fixed)\b"
    r"|(已完成|完成|搞定|修好|修复|部署|上线|合并|关闭|解决)"
)

# ── Entry-type whitelist ─────────────────────────────────
# ``entry_type`` goes verbatim into the output filename
# (``{agent}-{entry_type}-{slug}-{suffix}.md``). An untrusted value such as
# ``../../evil`` therefore escaped FABRIC_DIR — a path-traversal write. Only a
# strict slug is accepted; invalid values are rejected loudly, never sanitised
# into something silently different from what the caller asked for.
_ENTRY_TYPE_RE = re.compile(r"^[a-z0-9_-]{1,64}$")


def is_valid_entry_type(entry_type) -> bool:
    """True when ``entry_type`` is a safe filename slug."""
    return bool(_ENTRY_TYPE_RE.match(str(entry_type or "").strip()))


def _safe_entry_type(entry_type) -> str:
    """Return the validated entry type, or raise ``ValueError``."""
    value = str(entry_type or "").strip()
    if not _ENTRY_TYPE_RE.match(value):
        raise ValueError(
            f"invalid entry_type {entry_type!r}: must match {_ENTRY_TYPE_RE.pattern}"
        )
    return value


# ── Session state (session-isolated) ─────────────────────
# These were process-wide singletons, so two sessions sharing one Hermes
# process shared a single ``exchanges`` list and ``session_id``: a concurrent
# session's turns were attributed to whichever session started last.
# ``bind_session`` rebinds the three module-level names below to the record for
# the active session, so every existing call site
# (``state.exchanges.append(...)``, ``state.session_id``, ``state._recall_log``)
# keeps working unchanged — only the backing storage moved.
_SESSION_CACHE: dict = {}
_SESSION_CACHE_MAX = 64
_DEFAULT_SESSION_KEY = "__no_session__"


class _SessionRecord:
    __slots__ = ("session_id", "exchanges", "recall_log")

    def __init__(self, session_id=""):
        self.session_id = session_id
        self.exchanges = []
        self.recall_log = []


def _session_record(session_id=""):
    """Return the record for ``session_id``, LRU-refreshed, capped."""
    key = session_id or _DEFAULT_SESSION_KEY
    rec = _SESSION_CACHE.pop(key, None)
    if rec is None:
        rec = _SessionRecord(session_id)
    _SESSION_CACHE[key] = rec
    while len(_SESSION_CACHE) > _SESSION_CACHE_MAX:
        oldest = next(iter(_SESSION_CACHE))
        if oldest == key:
            break
        _SESSION_CACHE.pop(oldest, None)
    return rec


def bind_session(sid="", reset=False):
    """Bind ``session_id``/``exchanges``/``_recall_log`` to one session.

    ``reset`` starts a fresh record (session start) instead of reusing the
    existing one. Returns the bound record.
    """
    global session_id, exchanges, _recall_log
    key = sid or _DEFAULT_SESSION_KEY
    if reset:
        _SESSION_CACHE.pop(key, None)
    rec = _session_record(sid)
    session_id = rec.session_id
    exchanges = rec.exchanges
    _recall_log = rec.recall_log
    return rec


session_id = ""
exchanges: list = []

# ── Training job tracking ────────────────────────────────
_JOB_FILE = (HERMES_HOME or Path.home()) / ".icarus-training-job.txt"


def _last_job_id():
    if _JOB_FILE.exists():
        return _JOB_FILE.read_text("utf-8").strip()
    return ""


def _save_job_id(jid):
    _JOB_FILE.write_text(jid, "utf-8")


# ── Model registry ───────────────────────────────────────
_REGISTRY_FILE = (HERMES_HOME or Path.home()) / ".icarus-models.json"


def _load_registry():
    if _REGISTRY_FILE.exists():
        try:
            return json.loads(_REGISTRY_FILE.read_text("utf-8"))
        except Exception:
            pass
    return {"models": [], "active_model": None}


def _save_registry(registry):
    try:
        _REGISTRY_FILE.write_text(json.dumps(registry, indent=2), "utf-8")
    except Exception as exc:
        logger.warning("icarus: failed to save model registry: %s", exc)


def list_models():
    return _load_registry()


# ── Retrieval telemetry ──────────────────────────────────
_TELEMETRY_FILE = (HERMES_HOME or Path.home()) / ".icarus-telemetry.jsonl"

# in-memory buffer for current session
_recall_log: list = []


def _summarize_telemetry_events(events):
    """Compute telemetry summary for a filtered event list."""
    recalls = [e for e in events if e.get("event") == "recall"]
    usages = [e for e in events if e.get("event") == "usage"]
    recalled_ids = set()
    for r in recalls:
        recalled_ids.update(r.get("result_ids", []))
    used_ids = set(u.get("entry_id", "") for u in usages)
    used_ids.discard("")
    recalled_ids.discard("")
    used_from_recall = used_ids & recalled_ids
    return {
        "total_recalls": len(recalls),
        "total_usages": sum(1 for u in usages if u.get("entry_id", "") in recalled_ids),
        "unique_entries_recalled": len(recalled_ids),
        "unique_entries_used": len(used_from_recall),
        "usage_rate": round(len(used_from_recall) / max(len(recalled_ids), 1), 2),
    }


def log_recall(query, results, source="pre_llm_call"):
    """Log what was recalled and injected."""
    entry = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "event": "recall",
        "source": source,
        "query": query[:100],
        "result_count": len(results),
        "result_ids": [r.get("id", "") for r in results[:5] if isinstance(r, dict)],
        "result_summaries": [
            r.get("summary", "")[:60] for r in results[:5] if isinstance(r, dict)
        ],
        "session_id": session_id,
        "agent": AGENT_NAME,
    }
    _recall_log.append(entry)
    try:
        with open(_TELEMETRY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def was_recalled(entry_id):
    """Return True if this entry_id was recalled in the current session."""
    if not entry_id:
        return False
    current_session = session_id
    for event in reversed(_recall_log):
        if event.get("event") != "recall":
            continue
        if current_session and event.get("session_id") != current_session:
            continue
        if entry_id in event.get("result_ids", []):
            return True
    return False


def log_usage(entry_id, action="referenced"):
    """Log when a recalled entry is actually used (referenced in a write)."""
    entry = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "event": "usage",
        "action": action,
        "entry_id": entry_id,
        "session_id": session_id,
        "agent": AGENT_NAME,
    }
    _recall_log.append(entry)
    try:
        with open(_TELEMETRY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def get_telemetry(last_n=50, session_id_filter=None, agent_filter=None):
    """Read recent telemetry entries."""
    empty_summary = {
        "total_recalls": 0,
        "total_usages": 0,
        "unique_entries_recalled": 0,
        "unique_entries_used": 0,
        "usage_rate": 0.0,
    }
    if not _TELEMETRY_FILE.exists():
        return {"events": [], "summary": empty_summary}
    path = _TELEMETRY_FILE
    with path.open(encoding="utf-8") as f:
        dq = deque(f, maxlen=last_n)
    events = []
    for line in dq:
        try:
            events.append(json.loads(line.strip()))
        except Exception:
            pass

    if session_id_filter:
        events = [e for e in events if e.get("session_id") == session_id_filter]
    if agent_filter:
        events = [e for e in events if e.get("agent") == agent_filter]

    return {
        "events": events,
        "summary": _summarize_telemetry_events(events),
    }


def build_brief():
    """Build the daily brief: pending work, recent own work, changes, suggested action."""
    agent = AGENT_NAME or "agent"
    brief = {}

    # pending
    open_tasks, reviews, open_tickets = read_pending()
    brief["pending"] = {
        "open_tasks": len(open_tasks),
        "reviews_of_my_work": len(reviews),
        "open_tickets": len(open_tickets),
        "items": [],
    }
    for t in open_tasks[:3]:
        brief["pending"]["items"].append(
            {
                "from": t.get("agent", "?"),
                "summary": t.get("summary", "?"),
                "type": t.get("type", "?"),
                "id": t.get("id", "?"),
            }
        )
    for r in reviews[:3]:
        brief["pending"]["items"].append(
            {
                "from": r.get("agent", "?"),
                "summary": r.get("summary", "?"),
                "type": "review",
                "id": r.get("id", "?"),
                "review_of": r.get("review_of", ""),
            }
        )
    for t in open_tickets[:3]:
        brief["pending"]["items"].append(
            {
                "from": t.get("agent", "?"),
                "summary": t.get("summary", "?"),
                "type": t.get("type", "ticket"),
                "id": t.get("id", "?"),
                "customer_id": t.get("customer_id", "?"),
            }
        )

    # recent own work (last 5 entries by this agent)
    own = read_recent(agent=agent, limit=5)
    brief["recent_work"] = [
        {
            "summary": e.get("summary", "?"),
            "timestamp": str(e.get("timestamp", ""))[:16],
        }
        for e in own
    ]

    # recent activity from others (changes since last session)
    others = read_cross_agent(limit=5)
    brief["from_others"] = others

    # suggested action
    if open_tasks:
        t = open_tasks[0]
        brief["suggested_action"] = (
            f"Pick up: {t.get('summary', '?')} from {t.get('agent', '?')} (id {t.get('id', '?')})"
        )
    elif reviews:
        r = reviews[0]
        brief["suggested_action"] = (
            f"Address review: {r.get('summary', '?')} from {r.get('agent', '?')} ({r.get('review_of', '')})"
        )
    elif open_tickets:
        t = open_tickets[0]
        brief["suggested_action"] = (
            f"Resolve ticket: {t.get('summary', '?')} [{t.get('customer_id', '?')}]"
        )
    else:
        brief["suggested_action"] = (
            "No pending work. Continue current task or check fabric_recall for context."
        )

    # telemetry summary (if available)
    tel = get_telemetry(last_n=20)
    if tel.get("summary", {}).get("total_recalls", 0) > 0:
        brief["recall_stats"] = tel["summary"]

    return brief


# ── Creative state ───────────────────────────────────────
_STATE_FILE = (HERMES_HOME or Path.home()) / ".icarus-state.json"


def load_creative():
    if _STATE_FILE.exists():
        try:
            return json.loads(_STATE_FILE.read_text("utf-8"))
        except Exception:
            pass
    return {"cycle": 0, "themes": [], "questions": [], "learnings": []}


def save_creative(s):
    try:
        _STATE_FILE.write_text(json.dumps(s, indent=2), "utf-8")
    except Exception as exc:
        logger.warning("icarus: save state failed: %s", exc)


# ── Fabric I/O ───────────────────────────────────────────


def _yaml_scalar(value):
    """Encode a scalar as a YAML-safe quoted string."""
    return json.dumps(str(value))


def _yaml_module():
    """Return the YAML helper, working under both package and path loading."""
    try:
        from . import _yaml as helper

        return helper
    except ImportError:
        import _yaml as helper

        return helper


def _parse_frontmatter_scalar(text, key):
    """Read a scalar frontmatter value and normalize quoted YAML strings.

    Delegates to ``_yaml.load_scalar``, which resolves a YAML backend that
    actually exists in a Hermes runtime. A bare ``import yaml`` here used to
    fail there (Hermes ships ruamel, not PyYAML) and the fallback returned the
    quoted text verbatim, so ``training_value: "high"`` compared unequal to
    ``high`` in the report, the tier filter and every id lookup.
    """
    return _yaml_module().load_scalar(text, key)


def write_entry(
    entry_type,
    content,
    summary,
    tier="hot",
    tags="",
    platform="cli",
    status="",
    outcome="",
    review_of="",
    revises="",
    customer_id="",
    assigned_to="",
    training_value="",
    verified="",
    evidence="",
    source_tool="",
    artifact_paths="",
):
    """Write a fabric entry with full schema v1 fields. Returns the filepath.

    Raises ``ValueError`` when ``entry_type`` is not a safe slug, and refuses
    to write outside ``FABRIC_DIR`` (path-traversal guard).
    """
    entry_type = _safe_entry_type(entry_type)
    FABRIC_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H%MZ")
    ts_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    agent = AGENT_NAME or "agent"
    if not AGENT_NAME:
        logger.warning(
            'icarus: HERMES_AGENT_NAME not set — fabric entries will use agent="agent". '
            "Set HERMES_AGENT_NAME=<name> in .env for multi-agent deployments."
        )
    suffix = secrets.token_hex(2)
    # derive a short slug from the summary for human-readable filenames
    slug = re.sub(r"[^a-z0-9]+", "-", summary.lower().strip())[:40].strip("-")
    if slug:
        filename = f"{agent}-{entry_type}-{slug}-{suffix}.md"
    else:
        filename = f"{agent}-{entry_type}-{ts}-{suffix}.md"

    sid = session_id or os.environ.get(
        "FABRIC_SESSION_ID", f"sess-{now.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    )
    project_id = os.environ.get(
        "FABRIC_PROJECT_ID", Path.cwd().name if Path.cwd() != Path.home() else "unknown"
    )

    lines = [
        "---",
        f"id: {_yaml_scalar(secrets.token_hex(4))}",
        f"agent: {_yaml_scalar(agent)}",
        f"platform: {_yaml_scalar(platform)}",
        f"timestamp: {_yaml_scalar(ts_iso)}",
        f"type: {_yaml_scalar(entry_type)}",
        f"tier: {_yaml_scalar(tier)}",
        f"summary: {_yaml_scalar(summary)}",
        f"project_id: {_yaml_scalar(project_id)}",
        f"session_id: {_yaml_scalar(sid)}",
    ]
    if tags:
        lines.append(
            f"tags: {json.dumps([t.strip() for t in str(tags).split(',') if t.strip()])}"
        )
    if status:
        lines.append(f"status: {_yaml_scalar(status)}")
    if outcome:
        lines.append(f"outcome: {_yaml_scalar(outcome)}")
    if review_of:
        lines.append(f"review_of: {_yaml_scalar(review_of)}")
    if revises:
        lines.append(f"revises: {_yaml_scalar(revises)}")
    if customer_id:
        lines.append(f"customer_id: {_yaml_scalar(customer_id)}")
    if assigned_to:
        lines.append(f"assigned_to: {_yaml_scalar(assigned_to)}")
    if training_value:
        lines.append(f"training_value: {_yaml_scalar(training_value)}")
    if verified:
        lines.append(f"verified: {_yaml_scalar(verified)}")
    if evidence:
        lines.append(f"evidence: {_yaml_scalar(evidence)}")
    if source_tool:
        lines.append(f"source_tool: {_yaml_scalar(source_tool)}")
    if artifact_paths:
        lines.append(
            f"artifact_paths: {json.dumps([p.strip() for p in str(artifact_paths).split(',') if p.strip()])}"
        )
    lines.extend(["---", "", content])

    path = FABRIC_DIR / filename
    # Containment guard: even with a validated entry_type, a hostile agent name
    # (env-controlled) could still contain a separator. Refuse anything that
    # would land outside FABRIC_DIR rather than writing it.
    try:
        base = FABRIC_DIR.resolve()
        resolved = path.resolve()
    except OSError:
        base, resolved = FABRIC_DIR, path
    if resolved.parent != base:
        raise ValueError(f"refusing to write outside FABRIC_DIR ({base}): {resolved}")
    content_str = "\n".join(lines)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content_str, "utf-8")
    tmp.rename(path)
    logger.info("icarus: wrote %s", filename)

    # opt-in obsidian formatting
    if os.environ.get("ICARUS_OBSIDIAN"):
        try:
            from . import obsidian

            obsidian.format_entry(
                path, FABRIC_DIR, review_of=review_of, revises=revises
            )
            obsidian.ensure_daily_note(FABRIC_DIR, filename, summary)
        except Exception as exc:
            logger.debug("icarus: obsidian formatting failed: %s", exc)

    _sync_entry_to_qdrant(path)
    return str(path)


def _sync_entry_to_qdrant(path):
    """Mirror a freshly written entry into the Qdrant knowledge base.

    Layer 5 only ever *read*; nothing wrote fabric into Qdrant, so the
    ``[qdrant]`` prompt block stayed empty on every install without the
    Redis/ARQ worker stack. This makes each write self-feeding.

    Runs in a detached subprocess so an embedding round-trip never delays the
    turn, and opts out with ``ICARUS_QDRANT_SYNC=0``. Failures are logged at
    DEBUG only — the knowledge base is a cache, never the source of truth.
    """
    if os.environ.get("ICARUS_QDRANT_SYNC", "1").lower() in ("0", "false", "no", "off"):
        return
    script = HERMES_HOME / "memory-os" / "scripts" / "sync-fabric-to-qdrant.py"
    if not script.is_file():
        return
    try:
        subprocess.Popen(
            [sys.executable, str(script), "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # detach: survive this process's lifecycle
            cwd=str(HERMES_HOME),
        )
    except Exception as exc:
        logger.debug("icarus: qdrant sync spawn failed: %s", exc)


def read_recent(agent="", limit=5):
    """Read recent hot entries."""
    if not FABRIC_DIR.exists():
        return []
    out = []
    for f in sorted(
        FABRIC_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        head = _parse_head(f)
        if head.get("tier") != "hot":
            continue
        if agent and head.get("agent") != agent:
            continue
        out.append(
            {
                "agent": head.get("agent", ""),
                "timestamp": head.get("timestamp", ""),
                "summary": head.get("summary", ""),
            }
        )
        if len(out) >= limit:
            break
    return out


def read_cross_agent(limit=3):
    """Read recent entries from OTHER agents."""
    if not FABRIC_DIR.exists():
        return []
    out = []
    for f in sorted(
        FABRIC_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        head = _parse_head(f)
        if AGENT_NAME and head.get("agent") == AGENT_NAME:
            continue
        if head.get("type") not in ("review", "dialogue", "decision"):
            continue
        agent = head.get("agent", "")
        summary = head.get("summary", "")
        if summary:
            out.append(f"{agent}: {summary}")
        if len(out) >= limit:
            break
    return out


def _parse_head(filepath, max_bytes=800):
    """Parse frontmatter fields from a fabric entry header."""
    text = filepath.read_text("utf-8")[:max_bytes]
    fields = {}
    for key in (
        "agent",
        "type",
        "tier",
        "status",
        "summary",
        "timestamp",
        "review_of",
        "revises",
        "customer_id",
        "assigned_to",
        "id",
        "project_id",
        "session_id",
        "outcome",
        "training_value",
        "verified",
        "evidence",
        "source_tool",
        "artifact_paths",
    ):
        value = _parse_frontmatter_scalar(text, key)
        if value != "":
            fields[key] = value
    fields["file"] = filepath.name
    return fields


def has_entry_ref(ref):
    """Return True when agent:id resolves to a real fabric entry."""
    if not ref or ":" not in ref or not FABRIC_DIR.exists():
        return False
    agent, entry_id = ref.split(":", 1)
    agent = agent.strip()
    entry_id = entry_id.strip()
    if not agent or not entry_id:
        return False

    for d in (FABRIC_DIR, FABRIC_DIR / "cold"):
        if not d.exists():
            continue
        for f in d.glob("*.md"):
            h = _parse_head(f)
            if (
                h.get("agent", "").strip() == agent
                and h.get("id", "").strip() == entry_id
            ):
                return True
    return False


def curate_entry(entry_id, training_value):
    """Update the training_value field on an existing fabric entry."""
    if training_value not in ("high", "normal", "low"):
        return {
            "error": f"training_value must be high/normal/low, got '{training_value}'"
        }

    for d in (FABRIC_DIR, FABRIC_DIR / "cold"):
        if not d.exists():
            continue
        for f in d.glob("*.md"):
            head = f.read_text("utf-8")[:400]
            m = re.search(r"^id: (.+)$", head, re.MULTILINE)
            if not m or m.group(1).strip().strip('"') != entry_id:
                continue

            text = f.read_text("utf-8")
            if re.search(r"^training_value: .+$", text, re.MULTILINE):
                text = re.sub(
                    r"^training_value: .+$",
                    f"training_value: {training_value}",
                    text,
                    count=1,
                    flags=re.MULTILINE,
                )
            else:
                text = text.replace(
                    "\n---\n", f"\ntraining_value: {training_value}\n---\n", 1
                )
            f.write_text(text, "utf-8")
            return {
                "status": "updated",
                "file": f.name,
                "training_value": training_value,
            }

    return {"error": f"entry {entry_id} not found"}


def read_pending(customer_id=None):
    """Find entries needing this agent's attention."""
    if not FABRIC_DIR.exists():
        return [], [], []

    agent = AGENT_NAME or "agent"
    open_tasks = []
    reviews = []
    open_tickets = []

    for f in sorted(
        FABRIC_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        h = _parse_head(f)
        entry_agent = h.get("agent", "")
        assigned_to = h.get("assigned_to", "").strip()

        if h.get("status") == "open" and entry_agent != agent:
            if not agent or assigned_to != agent:
                continue
            if customer_id and h.get("customer_id") != customer_id:
                continue
            open_tasks.append(h)

        if h.get("type") == "review" and entry_agent != agent:
            ref = h.get("review_of", "")
            if agent and ref.startswith(f"{agent}:"):
                reviews.append(h)

        if h.get("status") == "open" and h.get("customer_id"):
            if not agent or assigned_to != agent:
                continue
            if customer_id and h.get("customer_id") != customer_id:
                continue
            if h not in open_tasks:
                open_tickets.append(h)

        if len(open_tasks) + len(reviews) + len(open_tickets) >= 30:
            break

    return open_tasks, reviews, open_tickets


def search_entries(query, limit=10):
    """Keyword search across fabric."""
    if not FABRIC_DIR.exists():
        return []
    results = []
    q = query.lower()
    for d in [FABRIC_DIR, FABRIC_DIR / "cold"]:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
            text = f.read_text("utf-8")
            if q not in text.lower():
                continue
            head = _parse_head(f)
            summary = head.get("summary", "")
            agent = head.get("agent", "")
            matches = [line.strip() for line in text.split("\n") if q in line.lower()][
                :3
            ]
            results.append(
                {"file": f.name, "agent": agent, "summary": summary, "matches": matches}
            )
            if len(results) >= limit:
                return results
    return results


# ── Retriever ────────────────────────────────────────────

_retriever = None


def _load_retriever():
    """Load ``fabric-retrieve.py`` as a submodule of this package.

    ``fabric-retrieve.py`` starts with ``from .parsing import parse_entry`` and
    keeps a bare ``from parsing import ...`` fallback for script mode. Loading
    it as a *top-level* module (``fabric_retrieve``) satisfies neither: there is
    no parent package for the relative import and the plugin directory is not on
    ``sys.path`` for the bare one. Every attempt failed, ``_retriever`` stayed
    ``None`` forever, and ``recall()`` silently degraded to ``read_recent`` —
    the whole ranked-retrieval layer was dead while looking healthy.

    Registering it under ``<package>.fabric_retrieve`` with the plugin directory
    in ``submodule_search_locations`` makes ``from .parsing import ...`` resolve
    exactly as it does for ``hooks.py``/``tools.py``.
    """
    import importlib.util
    import sys

    package = __package__ or "icarus"
    paths = [
        PLUGIN_DIR / "fabric-retrieve.py",
        Path(os.environ.get("FABRIC_RETRIEVE_PATH", "")),
    ]
    if HERMES_HOME:
        paths.append(HERMES_HOME / "plugins" / "icarus" / "fabric-retrieve.py")
    for p in paths:
        if not p or not p.exists():
            continue
        module_name = f"{package}.fabric_retrieve"
        if module_name in sys.modules:
            mod = sys.modules[module_name]
            mod.FABRIC_DIR = FABRIC_DIR
            return mod
        plugin_dir = str(p.parent)
        if plugin_dir not in sys.path:
            sys.path.insert(0, plugin_dir)
        try:
            spec = importlib.util.spec_from_file_location(
                module_name, str(p), submodule_search_locations=[plugin_dir]
            )
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mod
            spec.loader.exec_module(mod)
            mod.FABRIC_DIR = FABRIC_DIR
            return mod
        except Exception as exc:
            sys.modules.pop(module_name, None)
            logger.warning("icarus: retriever load failed from %s: %s", p, exc)
    logger.warning(
        "icarus: ranked retrieval unavailable — fabric_recall falls back to "
        "recency order (%d candidate path(s) tried)",
        len(paths),
    )
    return None


def recall(query, max_results=5, agent=None, project=None):
    """Smart ranked retrieval. Falls back to read_recent."""
    global _retriever
    if _retriever is None:
        _retriever = _load_retriever()
    if _retriever is None:
        return read_recent(agent, max_results)

    _retriever.FABRIC_DIR = FABRIC_DIR
    try:
        results = _retriever.retrieve(
            query, max_results=max_results, agent=agent, project=project
        )
        return [{"score": score, **entry} for score, entry in results]
    except Exception as exc:
        logger.debug("icarus: retrieval error: %s", exc)
        return read_recent(agent, max_results)


# ── Training ─────────────────────────────────────────────


def _together_key():
    key = os.environ.get("TOGETHER_API_KEY", "")
    if key:
        return key
    if HERMES_HOME and (HERMES_HOME / ".env").exists():
        for line in (HERMES_HOME / ".env").read_text().split("\n"):
            if line.startswith("TOGETHER_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


def _together_request(method, url, data=None):
    """Make an authenticated request to Together AI."""
    key = _together_key()
    if not key:
        raise RuntimeError("TOGETHER_API_KEY not set")
    headers = {"Authorization": f"Bearer {key}"}
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    resp = urllib.request.urlopen(req, timeout=30)
    return json.loads(resp.read())


def export_training(mode="normal"):
    """Export fabric entries as training pairs. Returns stats dict."""
    export_script = PLUGIN_DIR / "export-training.py"
    if not export_script.exists():
        return {"error": "export-training.py not found"}

    with tempfile.TemporaryDirectory() as tmpdir:
        # ``--fabric-dir`` is mandatory in practice: the script defaults to
        # ``~/fabric``, while Hermes keeps entries under ``$HERMES_HOME/fabric``.
        # Without it the child process exited 1 with "does not exist" and every
        # export call returned the generic "export failed".
        # ``sys.executable`` (not "python3") so the child inherits the same
        # interpreter and site-packages the plugin is running under.
        cmd = [
            sys.executable,
            str(export_script),
            "--output",
            tmpdir,
            "--fabric-dir",
            str(FABRIC_DIR),
        ]
        if mode != "normal":
            cmd.extend(["--mode", mode])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            return {"error": f"export failed: {detail.splitlines()[-1] if detail else 'no output'}"}

        output = result.stdout
        pairs = 0
        m = re.search(r"total pairs:\s+(\d+)", output)
        if m:
            pairs = int(m.group(1))

        tokens = 0
        m = re.search(r"estimated tokens:\s+([\d,]+)", output)
        if m:
            tokens = int(m.group(1).replace(",", ""))

        together_path = Path(tmpdir) / "together.jsonl"
        training_data = (
            together_path.read_text("utf-8") if together_path.exists() else ""
        )

        pair_types = {}
        raw_pairs_path = Path(tmpdir) / "raw-pairs.json"
        if raw_pairs_path.exists():
            try:
                raw_pairs = json.loads(raw_pairs_path.read_text("utf-8"))
                for pair in raw_pairs:
                    ptype = pair.get("metadata", {}).get("type", "unknown")
                    pair_types[ptype] = pair_types.get(ptype, 0) + 1
            except Exception:
                pair_types = {}

        return {
            "pairs": pairs,
            "estimated_tokens": tokens,
            "mode": mode,
            "pair_types": pair_types,
            "output": output.strip(),
            "training_data_path": str(together_path)
            if together_path.exists()
            else None,
            "_training_data": training_data,
        }


def _select_training_export_mode(min_pairs):
    """Choose the highest-quality export mode that still has enough pairs."""
    tried = {}
    for mode in ("high-precision", "normal", "high-volume"):
        result = export_training(mode=mode)
        tried[mode] = result
        if "error" in result:
            continue
        if result.get("pairs", 0) >= min_pairs:
            return mode, result, tried
    # fallback to normal result if all are below threshold but export succeeded
    for mode in ("normal", "high-volume", "high-precision"):
        result = tried.get(mode)
        if result and "error" not in result:
            return mode, result, tried
    return "normal", {"error": "export failed"}, tried


def start_training(
    model=None,
    suffix=None,
    epochs=3,
    batch_size=None,
    learning_rate=None,
    checkpoints=None,
    mode=None,
    min_pairs=10,
):
    """Export, upload, and start a Together AI fine-tune."""
    key = _together_key()
    if not key:
        return {"error": "TOGETHER_API_KEY not set in .env"}

    if mode:
        export = export_training(mode=mode)
        export_mode = mode
        tried_modes = {mode: export}
    else:
        export_mode, export, tried_modes = _select_training_export_mode(min_pairs)
    if "error" in export:
        return export
    if export["pairs"] < min_pairs:
        return {
            "error": f"only {export['pairs']} pairs in {export_mode}, need at least {min_pairs}",
            "mode": export_mode,
            "tried_modes": {
                k: v.get("pairs", 0) for k, v in tried_modes.items() if "error" not in v
            },
        }

    training_data = export.get("_training_data", "")
    if not training_data:
        return {"error": "no training data produced"}

    boundary = secrets.token_hex(16)
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="purpose"\r\n\r\nfine-tune\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="training.jsonl"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
        f"{training_data}\r\n"
        f"--{boundary}--\r\n"
    ).encode()

    req = urllib.request.Request(
        "https://api.together.xyz/v1/files/upload",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        upload_data = json.loads(resp.read())
    except Exception as exc:
        return {"error": f"upload failed: {exc}"}

    file_id = upload_data.get("id", "")
    if not file_id:
        return {"error": "upload succeeded but no file ID returned"}

    agent = AGENT_NAME or "agent"
    ft_model = model or os.environ.get("TOGETHER_MODEL", "Qwen/Qwen2-7B-Instruct")
    ft_suffix = suffix or os.environ.get("TOGETHER_SUFFIX", f"{agent}-v1")
    ft_batch = int(
        batch_size
        if batch_size is not None
        else os.environ.get("TOGETHER_BATCH_SIZE", "8")
    )
    ft_lr = float(
        learning_rate
        if learning_rate is not None
        else os.environ.get("TOGETHER_LR", "1e-5")
    )
    ft_checkpoints = int(
        checkpoints
        if checkpoints is not None
        else os.environ.get("TOGETHER_CHECKPOINTS", "1")
    )

    if ft_batch < 8:
        return {"error": f"batch_size must be >= 8 (got {ft_batch})"}
    if ft_lr <= 0:
        return {"error": f"learning_rate must be > 0 (got {ft_lr})"}
    if ft_checkpoints < 1:
        return {"error": f"n_checkpoints must be >= 1 (got {ft_checkpoints})"}

    try:
        ft_data = _together_request(
            "POST",
            "https://api.together.xyz/v1/fine-tunes",
            {
                "training_file": file_id,
                "model": ft_model,
                "n_epochs": epochs,
                "suffix": ft_suffix,
                "batch_size": ft_batch,
                "learning_rate": ft_lr,
                "n_checkpoints": ft_checkpoints,
            },
        )
    except Exception as exc:
        return {"error": f"fine-tune start failed: {exc}"}

    job_id = ft_data.get("id", "")
    if not job_id:
        return {"error": "fine-tune accepted but no job ID"}

    _save_job_id(job_id)

    # register as pending in model registry
    registry = _load_registry()
    registry["models"].append(
        {
            "job_id": job_id,
            "base_model": ft_model,
            "output_model": None,
            "suffix": ft_suffix,
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pair_count": export["pairs"],
            "estimated_tokens": export.get("estimated_tokens", 0),
            "pair_types": export.get("pair_types", {}),
            "export_mode": export_mode,
            "status": "pending",
            "eval_scores": None,
            "active": False,
        }
    )
    _save_registry(registry)

    return {
        "job_id": job_id,
        "model": ft_model,
        "suffix": ft_suffix,
        "epochs": epochs,
        "batch_size": ft_batch,
        "learning_rate": ft_lr,
        "n_checkpoints": ft_checkpoints,
        "pairs": export["pairs"],
        "estimated_tokens": export.get("estimated_tokens", 0),
        "pair_types": export.get("pair_types", {}),
        "mode": export_mode,
        "file_id": file_id,
    }


def check_training(job_id=None):
    """Check a Together AI fine-tune job status."""
    jid = job_id or _last_job_id()
    if not jid:
        return {"error": "no job ID — run fabric_train first"}
    try:
        data = _together_request("GET", f"https://api.together.xyz/v1/fine-tunes/{jid}")
    except Exception as exc:
        return {"error": f"status check failed: {exc}"}

    result = {"job_id": jid, "status": data.get("status", "unknown")}

    # update model registry
    registry = _load_registry()
    for m in registry["models"]:
        if m["job_id"] != jid:
            continue
        if data.get("status") == "completed":
            m["status"] = "completed"
            m["output_model"] = data.get("model_output_name", "")
            result["model_id"] = m["output_model"]
            result["instruction"] = (
                f"Run fabric_eval to test, then fabric_switch_model to activate."
            )
        elif data.get("status") in ("failed", "cancelled", "error"):
            m["status"] = data["status"]
            result["error"] = data.get("error", "unknown")
        break
    _save_registry(registry)

    return result


def run_eval(candidate_model, base_model=None, sample_count=10):
    """Run replacement-model eval. Returns comparison results."""
    eval_script = PLUGIN_DIR / "scripts" / "eval-replacement.py"
    if not eval_script.exists():
        return {"error": "eval-replacement.py not found"}

    key = _together_key()
    if not key:
        return {"error": "TOGETHER_API_KEY not set"}

    base = base_model or os.environ.get("LLM_MODEL", "Qwen/Qwen2-7B-Instruct")

    cmd = [
        "python3",
        str(eval_script),
        "--candidate-model",
        candidate_model,
        "--base-model",
        base,
        "--sample-count",
        str(sample_count),
        "--fabric-dir",
        str(FABRIC_DIR),
    ]
    env = os.environ.copy()
    env["TOGETHER_API_KEY"] = key
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, env=env
        )
    except subprocess.TimeoutExpired:
        return {"error": "eval timed out (5 min limit)"}

    if result.returncode != 0:
        return {"error": result.stderr or "eval failed"}

    try:
        scores = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": "eval output not valid JSON", "raw": result.stdout[:500]}

    # update registry with eval scores
    registry = _load_registry()
    for m in registry["models"]:
        if m.get("output_model") == candidate_model:
            m["eval_scores"] = scores.get("candidate_scores")
            break
    _save_registry(registry)

    return scores


# Written into .env instead of the real secret. Persisting the API key in
# plaintext on disk is a credential leak: .env files get backed up, synced and
# pasted into tickets. The operator supplies the value via the environment.
_ENV_KEY_PLACEHOLDER = "__SET_FROM_ENV_TOGETHER_API_KEY__"


def _new_env_backup_path() -> Path:
    """Unique, timestamped ``.env`` backup path.

    The old fixed ``.env.backup`` name was overwritten by every switch, so a
    second switch destroyed the only path back to the original configuration.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return HERMES_HOME / f".env.backup.{stamp}.{secrets.token_hex(3)}"


def _latest_env_backup():
    """Most recent ``.env`` backup (also matches the legacy fixed name)."""
    try:
        candidates = [p for p in HERMES_HOME.glob(".env.backup*") if p.is_file()]
    except OSError:
        candidates = []
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def switch_model(model_id, min_eval_score=0.7):
    """Switch the agent to use a replacement model."""
    env_file = HERMES_HOME / ".env"
    if not env_file.exists():
        return {"error": f".env not found at {env_file}"}

    registry = _load_registry()
    target = None
    for m in registry["models"]:
        if m.get("output_model") == model_id:
            target = m
            break
    if not target:
        return {"error": f"model {model_id} not in registry"}

    if target.get("eval_scores") is None:
        return {"error": "no eval scores — run fabric_eval first"}

    scores = target["eval_scores"]
    if isinstance(scores, dict):
        avg = sum(scores.values()) / max(len(scores), 1)
    elif isinstance(scores, (int, float)):
        avg = scores
    else:
        return {"error": f"unexpected eval_scores format: {type(scores)}"}

    if avg < min_eval_score:
        return {
            "error": f"eval score {avg:.2f} below threshold {min_eval_score}",
            "scores": scores,
        }

    # find current model for rollback
    lines = env_file.read_text("utf-8").split("\n")
    old_model = None
    for l in lines:
        if l.startswith("LLM_MODEL="):
            old_model = l.split("=", 1)[1].strip()

    # backup .env — unique, timestamped name (never clobbers an earlier backup)
    backup = _new_env_backup_path()
    shutil.copy2(env_file, backup)

    key = _together_key()
    if not key:
        return {"error": "TOGETHER_API_KEY not set in .env"}

    # Replacement models are served from Together. Repoint the OpenAI-compatible
    # provider config deliberately and rely on the backup / rollback path to
    # preserve the previous provider state.
    filtered = [
        l
        for l in lines
        if not l.startswith(("LLM_MODEL=", "OPENAI_BASE_URL=", "OPENAI_API_KEY="))
    ]
    filtered.append(f"LLM_MODEL={model_id}")
    filtered.append("OPENAI_BASE_URL=https://api.together.xyz/v1")
    # A placeholder, not the secret: the old code wrote the live API key into
    # .env in plaintext, which leaks the credential to every backup/sync/ticket
    # that file ever touches.
    filtered.append(f"OPENAI_API_KEY={_ENV_KEY_PLACEHOLDER}")

    # Atomic write. os.replace() overwrites the destination on both POSIX and
    # Windows; Path.rename() raises FileExistsError on Windows when .env already
    # exists, which made every switch fail with the file present.
    tmp = env_file.with_suffix(".tmp")
    tmp.write_text("\n".join(filtered), "utf-8")
    os.replace(tmp, env_file)

    logger.warning(
        "icarus: OPENAI_API_KEY was written to %s as the placeholder %r — no "
        "secret was persisted. Set OPENAI_API_KEY (value = your "
        "TOGETHER_API_KEY) in the process environment before the next run, or "
        "the replacement model cannot authenticate.",
        env_file,
        _ENV_KEY_PLACEHOLDER,
    )

    # update registry
    for m in registry["models"]:
        if m.get("active"):
            m["active"] = False
    target["active"] = True
    registry["active_model"] = model_id
    _save_registry(registry)

    rollback = (
        f"fabric_switch_model(model_id='{old_model}')"
        if old_model
        else f"restore from {backup.name}"
    )
    return {
        "status": "switched",
        "old_model": old_model,
        "new_model": model_id,
        "eval_score": avg,
        "backup": str(backup),
        "rollback": rollback,
    }


def _env_keys(path):
    """Map of ``KEY -> line`` for a dotenv file (last occurrence wins)."""
    out = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key = stripped.split("=", 1)[0].strip()
            if key:
                out[key] = line
    except OSError:
        pass
    return out


def rollback_model():
    """Restore the previous model selection in .env, without dropping keys.

    This used to be ``shutil.copy2(backup, env_file)``: the whole live .env was
    replaced by a file that was merely the *newest* backup. Any key added after
    that backup was silently destroyed — a routine "roll back the model" wiped
    unrelated credentials (measured: nine Memory OS keys vanished, including the
    embedding API key), and the call still reported ``status: rolled_back``.

    Two guards now:

    * **Only keys the rollback is allowed to touch are copied** (the model and
      provider keys). Everything else in the live file is preserved verbatim.
    * **A backup is created before writing**, so a rollback is itself undoable.

    If the backup holds no model/provider key the call refuses with an error
    instead of writing an identical file and claiming success.
    """
    env_file = HERMES_HOME / ".env"

    backup = _latest_env_backup()
    if backup is None:
        return {"error": "no .env.backup found — nothing to roll back to"}

    # Keys a model rollback is responsible for. Anything outside this set must
    # never be rewritten by this function.
    MANAGED_PREFIXES = ("LLM_MODEL", "LLM_API_BASE", "MODEL_", "HERMES_CUSTOM_")
    MANAGED_EXACT = {"LLM_MODEL", "LLM_API_BASE"}

    def managed(key):
        return key in MANAGED_EXACT or key.startswith(MANAGED_PREFIXES)

    current = _env_keys(env_file) if env_file.exists() else {}
    previous = _env_keys(backup)

    restorable = {k: v for k, v in previous.items() if managed(k)}
    if not restorable:
        return {
            "error": (
                f"backup {backup.name} contains no model/provider keys "
                f"(expected one of {sorted(MANAGED_EXACT)} or a MODEL_/HERMES_CUSTOM_ key) "
                "— refusing to overwrite .env"
            )
        }

    before_model = current.get("LLM_MODEL", "")
    after_model = restorable.get("LLM_MODEL", "")
    if before_model == after_model and len(restorable) == sum(1 for k in current if managed(k)):
        return {
            "error": (
                f"nothing to roll back: live .env already matches {backup.name} "
                f"for model keys (LLM_MODEL={after_model or '<unset>'})"
            )
        }

    # Snapshot the live file first so this operation is reversible.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    safety = HERMES_HOME / f".env.prerollback.{stamp}"
    if env_file.exists():
        try:
            shutil.copy2(env_file, safety)
        except OSError as exc:
            return {"error": f"cannot snapshot .env before rollback: {exc}"}

    lines = env_file.read_text(encoding="utf-8").split("\n") if env_file.exists() else []
    replaced = set()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in restorable:
            lines[i] = restorable[key]
            replaced.add(key)
    appended = [v for k, v in restorable.items() if k not in replaced]
    while lines and lines[-1] == "":
        lines.pop()
    lines.extend(appended)
    lines.append("")

    tmp = env_file.with_suffix(".env.tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    tmp.replace(env_file)

    # update registry
    registry = _load_registry()
    for m in registry["models"]:
        if m.get("active"):
            m["active"] = False
    restored_model = restorable.get("LLM_MODEL", "").split("=", 1)[-1] if restorable.get("LLM_MODEL") else None
    restored_match = None
    for m in registry["models"]:
        if restored_model and m.get("output_model") == restored_model:
            m["active"] = True
            restored_match = restored_model
    registry["active_model"] = restored_match
    _save_registry(registry)

    return {
        "status": "rolled_back",
        "from_model": before_model.split("=", 1)[-1] if before_model else None,
        "to_model": restored_model,
        "keys_restored": sorted(replaced | set(appended)),
        "keys_preserved": sorted(k for k in current if k not in restorable),
        "safety_copy": str(safety),
    }


# ── Session scoring ───────────────────────────────────────


def _count_session_entries():
    """Count entries written during the current session."""
    if not session_id or not FABRIC_DIR.exists():
        return 0
    count = 0
    for f in FABRIC_DIR.glob("*.md"):
        h = _parse_head(f)
        if h.get("session_id") == session_id:
            count += 1
    return count


def _count_session_linked_entries():
    """Count linked workflow entries written during the current session."""
    if not session_id or not FABRIC_DIR.exists():
        return 0
    count = 0
    for f in FABRIC_DIR.glob("*.md"):
        head = _parse_head(f)
        if head.get("session_id") != session_id:
            continue
        if head.get("review_of") or head.get("revises"):
            count += 1
    return count


def list_session_entries():
    """List entries written during the current session."""
    if not session_id or not FABRIC_DIR.exists():
        return []
    results = []
    for f in sorted(FABRIC_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime):
        h = _parse_head(f)
        if h.get("session_id") != session_id:
            continue
        results.append(h)
    return results


def score_session():
    """Score the current session quality. Returns component scores and total."""
    scores = {}

    substantive = [ex for ex in exchanges if len(ex.get("assistant", "").strip()) > 100]
    scores["depth"] = min(len(substantive) / 5, 1.0)

    all_text = " ".join(ex.get("assistant", "") for ex in exchanges)
    has_decision = bool(DECISION_RE.search(all_text))
    has_outcome = bool(OUTCOME_RE.search(all_text))
    scores["decision"] = (
        1.0 if (has_decision and has_outcome) else (0.5 if has_decision else 0.0)
    )

    tel = get_telemetry(
        last_n=500, session_id_filter=session_id, agent_filter=AGENT_NAME
    )
    scores["recall_usage"] = tel.get("summary", {}).get("usage_rate", 0.0)

    scores["linked_entries"] = min(_count_session_linked_entries() / 2, 1.0)

    substantial_user = sum(
        1 for ex in exchanges if len(ex.get("user", "").strip()) > 50
    )
    scores["user_engagement"] = min(substantial_user / 3, 1.0)

    weights = {
        "depth": 2,
        "decision": 3,
        "recall_usage": 2,
        "linked_entries": 2,
        "user_engagement": 1,
    }
    total = sum(scores[k] * weights[k] for k in scores) / sum(weights.values())
    scores["total"] = round(total, 2)

    return scores


# ── Corpus reporting ─────────────────────────────────────


def get_entry_usage_stats():
    """Per-entry-type recall and usage rates from telemetry."""
    tel = get_telemetry(last_n=500)
    recalled_ids = set()
    used_ids = set()
    for event in tel.get("events", []):
        if event.get("event") == "recall":
            recalled_ids.update(event.get("result_ids", []))
        elif event.get("event") == "usage":
            eid = event.get("entry_id", "")
            if eid:
                used_ids.add(eid)
    used_ids &= recalled_ids

    type_recalled: dict = {}
    type_used: dict = {}
    if FABRIC_DIR.exists():
        for f in FABRIC_DIR.glob("*.md"):
            h = _parse_head(f)
            eid = h.get("id", "")
            etype = h.get("type", "unknown")
            if eid in recalled_ids:
                type_recalled[etype] = type_recalled.get(etype, 0) + 1
            if eid in used_ids:
                type_used[etype] = type_used.get(etype, 0) + 1

    all_types = sorted(set(list(type_recalled.keys()) + list(type_used.keys())))
    return {
        "by_type": {
            t: {
                "recalled": type_recalled.get(t, 0),
                "used": type_used.get(t, 0),
                "usage_rate": round(
                    type_used.get(t, 0) / max(type_recalled.get(t, 0), 1), 2
                ),
            }
            for t in all_types
        },
    }


def build_weekly_report():
    """Corpus health report: entry types, training values, recall stats."""
    entries = []
    if FABRIC_DIR.exists():
        for f in FABRIC_DIR.glob("*.md"):
            entries.append(_parse_head(f))

    by_type: dict = {}
    by_tv = {"high": 0, "normal": 0, "low": 0, "unset": 0}
    verified_count = 0
    trainable_ids = set()

    for e in entries:
        t = e.get("type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1
        tv = e.get("training_value", "")
        by_tv[tv if tv in by_tv else "unset"] += 1
        if str(e.get("verified", "")).lower() == "true":
            verified_count += 1
        if tv == "high" or str(e.get("verified", "")).lower() == "true":
            entry_id = e.get("id")
            if entry_id:
                trainable_ids.add(str(entry_id))

    usage_stats = get_entry_usage_stats()
    tel = get_telemetry(last_n=200)

    return {
        "total_entries": len(entries),
        "by_type": by_type,
        "by_training_value": by_tv,
        "verified_entries": verified_count,
        "recall_usage": tel.get("summary", {}),
        "usage_by_type": usage_stats.get("by_type", {}),
        "trainable_estimate": len(trainable_ids),
    }


# ── SOUL ─────────────────────────────────────────────────


def load_soul():
    if HERMES_HOME:
        soul = HERMES_HOME / "SOUL.md"
        if soul.exists():
            return soul.read_text("utf-8")
    return ""


# ── Memory file ──────────────────────────────────────────


def write_memory_file(s):
    """Write Icarus creative state to CREATIVE.md — NOT to MEMORY.md.

    MEMORY.md is owned exclusively by the memory tool (§-delimited format).
    Writing to it from here caused drift that blocked memory persistence
    (issue #26045).  CREATIVE.md is safe because nothing else writes to it.
    """
    # HERMES_HOME is always resolved, so this never silently no-ops.
    mem_dir = HERMES_HOME / "memories"
    mem_dir.mkdir(parents=True, exist_ok=True)
    agent = AGENT_NAME or "agent"
    mem_path = mem_dir / "CREATIVE.md"

    lines = [f"# {agent} creative state\n"]
    if s.get("questions"):
        lines.append("## open questions")
        for q in s["questions"][-5:]:
            lines.append(f"- {q}")
        lines.append("")
    if s.get("learnings"):
        lines.append("## learnings")
        for ln in s["learnings"][-5:]:
            lines.append(f"- {ln}")
        lines.append("")
    lines.append(f"cycles: {s.get('cycle', 0)}")

    content_str = "\n".join(lines)
    tmp_path = mem_path.with_suffix(".tmp")
    tmp_path.write_text(content_str, "utf-8")
    tmp_path.rename(mem_path)
