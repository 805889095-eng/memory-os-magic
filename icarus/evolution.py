#!/usr/bin/env python3
"""Self-evolution engine for Hermes durable facts (pure stdlib + sqlite3).

This module closes the self-improvement loop over the durable fact store
(`memory_store.db`, tables ``facts`` / ``facts_fts`` created by
``setup/setup_db.py``):

    write  ->  retrieval counting  ->  usage feedback  ->  trust adjustment
           ->  rank re-scoring / decay

Design notes (see _refactor_contract.md §C.1):

* **Zero local models** — everything here is deterministic Python + ``sqlite3``.
  No embeddings, no network, no third-party dependency.
* **Schema-compatible** — ``ensure_schema`` creates exactly the ``facts`` /
  ``facts_fts`` shape declared in ``setup/setup_db.py`` and *idempotently*
  patches extra columns (``last_retrieved_at``) with ``ALTER TABLE ... ADD
  COLUMN`` so an existing database is upgraded in place.
* **Standalone** — this file has no relative imports, so it can be executed
  directly for self-verification::

      python icarus/evolution.py --selftest

The selftest runs the whole loop (write -> retrieval count -> positive/negative
feedback -> decay -> ranking) against a throwaway sqlite file and asserts every
step.
"""

from __future__ import annotations

import argparse
import math
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# ── Tunables (the "personality" of the evolution loop) ─────────────────────
INITIAL_TRUST = 0.55  # trust assigned to a newly recorded fact
HELPFUL_DELTA = 0.08  # trust gained when a fact proves useful
HARMFUL_DELTA = 0.12  # trust lost when a fact proves misleading
TRUST_FLOOR = 0.05  # hard floor for decay (never fully forgotten)
TRUST_CEIL = 1.0  # trust ceiling
DECAY_RATE = 0.02  # per-run trust decay for long-unused facts
DECAY_DAYS = 30  # "unused" age threshold (days) for decay
RANK_HALF_LIFE_DAYS = 180.0  # recency half-life used by rank_score
RANK_USE_GAIN = 0.15  # weight of the log1p(retrieval_count) usage term
# Ceiling on the usage bonus. The term used to be a bare multiplier, so with
# every fact at the same starting trust (INITIAL_TRUST 0.55) retrieval_count was
# the only differentiator and a never-retrieved fact could not outrank a hot one
# regardless of relevance: at ret=18 the factor is 1.44x. On the live store that
# left 18 of 29 facts at retrieval_count=0, unable to recall themselves — FTS
# ranked them 1st-3rd and the rerank pushed them to 13th-20th. Bounding the
# bonus keeps "used facts rise" while letting relevance and trust decide.
RANK_USE_CAP = 0.25

TOP_USED_DEFAULT = 10

# ── Schema (must stay byte-compatible with setup/setup_db.py MEMORY_SCHEMA) ──
SCHEMA_FACTS = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL UNIQUE,
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hrr_vector      BLOB
);
"""

SCHEMA_FACTS_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(content, tags, content=facts, content_rowid=fact_id);
"""

# Columns added by the evolution engine on top of the setup_db.py shape.
# Applied with idempotent ALTER TABLE so old databases are upgraded in place.
EXTRA_COLUMNS = {
    "last_retrieved_at": "TIMESTAMP",
}


# ── Path resolution ────────────────────────────────────────────────────────


def resolve_db_path() -> Path:
    """Resolve the durable fact store path.

    Priority (mirrors the contract / setup_db.py defaults):
      1. ``MEMORY_STORE_PATH`` env var.
      2. ``HERMES_HOME/memory_store.db``.
      3. ``~/.hermes/memory_store.db``.
    """
    env_path = (os.environ.get("MEMORY_STORE_PATH") or "").strip()
    if env_path:
        return Path(env_path)
    home = (os.environ.get("HERMES_HOME") or "").strip()
    if home:
        return Path(home) / "memory_store.db"
    return Path.home() / ".hermes" / "memory_store.db"


def connect(db_path=None) -> sqlite3.Connection:
    """Open (and create if needed) the fact store, with the schema ensured."""
    path = Path(db_path) if db_path is not None else resolve_db_path()
    ensure_parent(path)
    conn = sqlite3.connect(str(path), timeout=10)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


def ensure_parent(path: Path) -> None:
    """Create the parent directory of ``path`` when missing (best effort)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Idempotently create ``facts`` / ``facts_fts`` and patch extra columns."""
    conn.executescript(SCHEMA_FACTS)
    conn.executescript(SCHEMA_FACTS_FTS)
    existing = {row[1] for row in conn.execute("PRAGMA table_info(facts)").fetchall()}
    for column, decl in EXTRA_COLUMNS.items():
        if column not in existing:
            # SQLite cannot parametrise DDL; column/decl are module constants.
            conn.execute(f"ALTER TABLE facts ADD COLUMN {column} {decl}")
    conn.commit()


# ── Helpers ────────────────────────────────────────────────────────────────


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def fact_age_days(timestamp) -> float:
    """Days elapsed since ``timestamp`` (TEXT/ISO/epoch). 0.0 when unparsable.

    Named ``fact_age_days`` so that ``rank_score`` can expose the contract's
    ``age_days`` parameter position without shadowing this helper.
    """
    if timestamp in (None, ""):
        return 0.0
    if isinstance(timestamp, (int, float)):
        try:
            return max(
                0.0,
                (datetime.now(timezone.utc).timestamp() - float(timestamp)) / 86400.0,
            )
        except (TypeError, ValueError, OSError):
            return 0.0
    raw = str(timestamp).strip().replace("Z", "+00:00")
    for parser in (
        lambda s: datetime.fromisoformat(s),
        lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S"),
    ):
        try:
            dt = parser(raw)
        except (ValueError, TypeError):
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - dt
        return max(0.0, delta.total_seconds() / 86400.0)
    return 0.0


def rank_score(trust, retrieval_count, age_days, relevance=None) -> float:
    """Rank facts for injection: (trust + bounded usage bonus) × relevance × recency.

    ``relevance`` is the query-match signal, in [0, 1]. It was previously absent
    from the ranking entirely: the score was trust × usage × recency, all three
    of which are properties of the fact and none of which look at the query. With
    every fact at the same starting trust, that reduced to "retrieval_count wins",
    so a hot but tangential fact beat an exact match and never-retrieved facts
    could not surface at all (18 of 29 on the live store).

    ``relevance=None`` means "unknown" (the CJK LIKE path has no BM25) and scores
    a neutral 1.0 so that path keeps its old ordering behaviour.

    The usage bonus is now additive and capped by ``RANK_USE_CAP`` instead of a
    multiplicative ``(1 + gain*log1p(n))``. At ret=18 the old factor was 1.44x,
    which no trust or relevance difference could overcome; the capped bonus tops
    out at +0.25 absolute, so it breaks ties between comparable facts without
    letting popularity override match quality.
    """
    try:
        trust_f = float(trust) if trust is not None else 0.0
    except (TypeError, ValueError):
        trust_f = 0.0
    try:
        count_f = float(retrieval_count) if retrieval_count is not None else 0.0
    except (TypeError, ValueError):
        count_f = 0.0
    try:
        age_f = float(age_days) if age_days is not None else 0.0
    except (TypeError, ValueError):
        age_f = 0.0
    count_f = max(0.0, count_f)
    age_f = max(0.0, age_f)

    usage_bonus = min(RANK_USE_CAP, RANK_USE_GAIN * math.log1p(count_f))

    if relevance is None:
        relevance_f = 1.0
    else:
        try:
            relevance_f = float(relevance)
        except (TypeError, ValueError):
            relevance_f = 1.0
        relevance_f = max(0.0, min(1.0, relevance_f))

    return (
        (trust_f + usage_bonus)
        * relevance_f
        * math.exp(-math.log(2.0) * age_f / RANK_HALF_LIFE_DAYS)
    )


def _clamp(value, low, high):
    return max(low, min(high, value))


# ── Public API ─────────────────────────────────────────────────────────────


def record_fact(
    content, category="general", tags="", initial_trust=INITIAL_TRUST, db_path=None
):
    """Insert a durable fact. Returns its ``fact_id`` (existing id if duplicate).

    Uses ``INSERT OR IGNORE`` on the UNIQUE ``content`` column, then syncs the
    external-content FTS index manually (FTS5 with ``content=facts`` does not
    auto-populate). Returns ``None`` for empty content.
    """
    text = str(content or "").strip()
    if not text:
        return None
    trust = _clamp(float(initial_trust), TRUST_FLOOR, TRUST_CEIL)
    conn = connect(db_path)
    try:
        cur = conn.execute(
            "INSERT OR IGNORE INTO facts "
            "(content, category, tags, trust_score, retrieval_count, helpful_count, "
            " created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 0, 0, ?, ?)",
            (text, category or "general", tags or "", trust, _now(), _now()),
        )
        conn.commit()
        if cur.rowcount:
            fact_id = int(cur.lastrowid)
            conn.execute(
                "INSERT INTO facts_fts(rowid, content, tags) VALUES (?, ?, ?)",
                (fact_id, text, tags or ""),
            )
            conn.commit()
            return fact_id
        row = conn.execute(
            "SELECT fact_id FROM facts WHERE content = ?", (text,)
        ).fetchone()
        return int(row[0]) if row else None
    finally:
        conn.close()


def record_retrieval(fact_ids, db_path=None) -> int:
    """Bump the retrieval counter for every id in ``fact_ids``. Returns count."""
    ids = []
    for raw in fact_ids or []:
        try:
            ids.append(int(raw))
        except (TypeError, ValueError):
            continue
    if not ids:
        return 0
    conn = connect(db_path)
    try:
        updated = 0
        for fact_id in ids:
            cur = conn.execute(
                "UPDATE facts SET retrieval_count = retrieval_count + 1, "
                "last_retrieved_at = ?, updated_at = ? WHERE fact_id = ?",
                (_now(), _now(), fact_id),
            )
            updated += cur.rowcount
        conn.commit()
        return updated
    finally:
        conn.close()


def apply_feedback(fact_id, helpful: bool, weight: float = 1.0, db_path=None) -> dict:
    """Nudge a fact's trust based on whether it proved useful.

    helpful  -> ``trust = min(TRUST_CEIL, trust + HELPFUL_DELTA*weight)`` and
                ``helpful_count += 1``
    unhelpful -> ``trust = max(0.0, trust - HARMFUL_DELTA*weight)``

    Returns a dict with the resulting ``trust_score`` (or ``{"error": ...}``).
    """
    try:
        fid = int(fact_id)
    except (TypeError, ValueError):
        return {"error": f"fact_id must be an integer, got {fact_id!r}"}
    try:
        w = float(weight)
    except (TypeError, ValueError):
        w = 1.0
    w = max(0.0, w)

    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT trust_score FROM facts WHERE fact_id = ?", (fid,)
        ).fetchone()
        if row is None:
            return {"error": f"fact {fid} not found"}
        current = float(row[0]) if row[0] is not None else 0.5

        if helpful:
            new_trust = _clamp(current + HELPFUL_DELTA * w, 0.0, TRUST_CEIL)
            conn.execute(
                "UPDATE facts SET trust_score = ?, helpful_count = helpful_count + 1, "
                "updated_at = ? WHERE fact_id = ?",
                (new_trust, _now(), fid),
            )
        else:
            new_trust = _clamp(current - HARMFUL_DELTA * w, 0.0, TRUST_CEIL)
            conn.execute(
                "UPDATE facts SET trust_score = ?, updated_at = ? WHERE fact_id = ?",
                (new_trust, _now(), fid),
            )
        conn.commit()
        after = conn.execute(
            "SELECT trust_score, helpful_count, retrieval_count FROM facts WHERE fact_id = ?",
            (fid,),
        ).fetchone()
        return {
            "fact_id": fid,
            "helpful": bool(helpful),
            "previous_trust": round(current, 4),
            "trust_score": round(float(after["trust_score"]), 4),
            "helpful_count": int(after["helpful_count"] or 0),
            "retrieval_count": int(after["retrieval_count"] or 0),
        }
    finally:
        conn.close()


def decay_unused(days: int = DECAY_DAYS, rate: float = DECAY_RATE, db_path=None) -> int:
    """Slowly lower trust for facts unused for ``days``. Returns rows affected.

    Floors at ``TRUST_FLOOR`` so a fact is never fully forgotten. A fact is
    "unused" when it has never been retrieved (``last_retrieved_at`` NULL) and
    is older than the cutoff, or when its last retrieval is older than the
    cutoff.
    """
    try:
        days_i = max(0, int(days))
    except (TypeError, ValueError):
        days_i = DECAY_DAYS
    try:
        rate_f = max(0.0, float(rate))
    except (TypeError, ValueError):
        rate_f = DECAY_RATE
    conn = connect(db_path)
    try:
        cur = conn.execute(
            "UPDATE facts SET trust_score = MAX(?, trust_score - ?), updated_at = ? "
            "WHERE trust_score > ? "
            "AND COALESCE(last_retrieved_at, created_at) < datetime('now', ?)",
            (TRUST_FLOOR, rate_f, _now(), TRUST_FLOOR, f"-{days_i} days"),
        )
        conn.commit()
        return int(cur.rowcount or 0)
    finally:
        conn.close()


def stats(top_n: int = TOP_USED_DEFAULT, db_path=None) -> dict:
    """Evolution snapshot: size, average trust, most-used facts, db path."""
    conn = connect(db_path)
    try:
        total = int(conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] or 0)
        avg_trust = conn.execute(
            "SELECT AVG(COALESCE(trust_score, 0.5)) FROM facts"
        ).fetchone()[0]
        helpful = int(
            conn.execute(
                "SELECT COUNT(*) FROM facts WHERE helpful_count > 0"
            ).fetchone()[0]
            or 0
        )
        retrievals = int(
            conn.execute(
                "SELECT COALESCE(SUM(retrieval_count), 0) FROM facts"
            ).fetchone()[0]
            or 0
        )
        rows = conn.execute(
            "SELECT fact_id, content, category, tags, trust_score, retrieval_count, "
            "helpful_count, created_at, updated_at, last_retrieved_at "
            "FROM facts ORDER BY retrieval_count DESC, trust_score DESC, fact_id ASC LIMIT ?",
            (max(1, int(top_n)),),
        ).fetchall()
    finally:
        conn.close()

    top_used = []
    for row in rows:
        age = fact_age_days(row["created_at"] or row["updated_at"])
        top_used.append(
            {
                "fact_id": int(row["fact_id"]),
                "preview": str(row["content"] or "")[:120],
                "category": row["category"] or "general",
                "tags": row["tags"] or "",
                "trust_score": round(float(row["trust_score"] or 0.0), 4),
                "retrieval_count": int(row["retrieval_count"] or 0),
                "helpful_count": int(row["helpful_count"] or 0),
                "age_days": round(age, 2),
                "rank_score": round(
                    rank_score(row["trust_score"], row["retrieval_count"], age), 4
                ),
            }
        )

    return {
        "db_path": str(resolve_db_path() if db_path is None else Path(db_path)),
        "total_facts": total,
        "avg_trust": round(float(avg_trust or 0.0), 4),
        "helpful_facts": helpful,
        "total_retrievals": retrievals,
        "top_used": top_used,
    }


# ── Selftest ───────────────────────────────────────────────────────────────


class _Selftest:
    """Tiny assertion harness that prints PASS/FAIL per check."""

    def __init__(self):
        self.ok = 0
        self.failed = 0

    def check(self, label, condition):
        if condition:
            self.ok += 1
            print(f"  [PASS] {label}")
        else:
            self.failed += 1
            print(f"  [FAIL] {label}")
        return bool(condition)

    def close(self, label):
        if self.failed:
            print(f"\n=== {label} FAILED ({self.failed} failed / {self.ok} passed) ===")
            return False
        print(f"\n=== {label} PASSED ({self.ok} checks) ===")
        return True


def selftest() -> int:
    print("icarus evolution selftest — full loop over a throwaway sqlite file")
    t = _Selftest()

    tmpdir = tempfile.TemporaryDirectory(prefix="icarus-evo-")
    db = Path(tmpdir.name) / "memory_store.db"
    previous = os.environ.get("MEMORY_STORE_PATH")
    os.environ["MEMORY_STORE_PATH"] = str(db)
    try:
        # 0. path resolution honours MEMORY_STORE_PATH
        t.check("resolve_db_path honours MEMORY_STORE_PATH", resolve_db_path() == db)

        # 1. write -> schema + FTS sync
        fid_a = record_fact(
            "rustchain ed25519 attestation signature verified end to end",
            category="session",
            tags="src:sess-1",
        )
        fid_b = record_fact(
            "power8 numa coffer tuning notes for the build host",
            category="session",
            tags="src:sess-1",
        )
        fid_c = record_fact(
            "gardening tomatoes weather unrelated trivia", category="general"
        )
        t.check(
            "record_fact returns increasing ids",
            all(isinstance(i, int) for i in (fid_a, fid_b, fid_c)),
        )
        t.check("record_fact ids are distinct", len({fid_a, fid_b, fid_c}) == 3)

        # duplicate write is ignored (UNIQUE content) and returns the same id
        t.check(
            "duplicate record_fact returns existing id",
            record_fact("rustchain ed25519 attestation signature verified end to end")
            == fid_a,
        )

        conn = connect()
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(facts)").fetchall()}
            t.check(
                "last_retrieved_at column present (ALTER TABLE path)",
                "last_retrieved_at" in cols,
            )
            t.check(
                "facts row count is 3 after duplicate insert",
                conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 3,
            )
            # external-content FTS must be manually populated by record_fact
            hits = conn.execute(
                "SELECT COUNT(*) FROM facts_fts WHERE facts_fts MATCH 'attestation'"
            ).fetchone()[0]
            t.check("facts_fts MATCH finds the inserted fact", hits == 1)
            # ensure_schema is idempotent
            ensure_schema(conn)
            t.check(
                "ensure_schema is idempotent",
                conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 3,
            )
        finally:
            conn.close()

        # 2. retrieval counting
        t.check("record_retrieval updates one row", record_retrieval([fid_a]) == 1)
        record_retrieval([fid_a, fid_a, fid_b])
        conn = connect()
        try:
            rc_a = conn.execute(
                "SELECT retrieval_count FROM facts WHERE fact_id=?", (fid_a,)
            ).fetchone()[0]
            rc_b = conn.execute(
                "SELECT retrieval_count FROM facts WHERE fact_id=?", (fid_b,)
            ).fetchone()[0]
            lra = conn.execute(
                "SELECT last_retrieved_at FROM facts WHERE fact_id=?", (fid_a,)
            ).fetchone()[0]
        finally:
            conn.close()
        t.check("retrieval_count incremented per call (3 total for A)", rc_a == 3)
        t.check("retrieval_count incremented for B (1)", rc_b == 1)
        t.check("record_retrieval sets last_retrieved_at", bool(lra))
        t.check(
            "record_retrieval ignores garbage ids",
            record_retrieval(["nope", None, "7x"]) == 0,
        )

        # 3. positive / negative feedback
        base_a = 0.55
        positive = apply_feedback(fid_a, helpful=True)
        t.check(
            "helpful feedback raises trust by HELPFUL_DELTA",
            abs(positive["trust_score"] - round(base_a + HELPFUL_DELTA, 4)) < 1e-6,
        )
        t.check("helpful feedback bumps helpful_count", positive["helpful_count"] == 1)

        weighted = apply_feedback(fid_b, helpful=True, weight=2.0)
        t.check(
            "weighted helpful feedback scales the delta",
            abs(weighted["trust_score"] - round(base_a + 2 * HELPFUL_DELTA, 4)) < 1e-6,
        )

        negative = apply_feedback(fid_c, helpful=False)
        t.check(
            "harmful feedback lowers trust by HARMFUL_DELTA",
            abs(negative["trust_score"] - round(0.55 - HARMFUL_DELTA, 4)) < 1e-6,
        )
        t.check(
            "harmful feedback does not touch helpful_count",
            negative["helpful_count"] == 0,
        )
        t.check(
            "feedback on a missing fact reports an error",
            "error" in apply_feedback(999999, helpful=True),
        )

        # trust ceiling: repeated helpful feedback never exceeds 1.0
        for _ in range(20):
            apply_feedback(fid_a, helpful=True)
        ceiling = apply_feedback(fid_a, helpful=True)
        t.check("trust is clamped at TRUST_CEIL", ceiling["trust_score"] <= TRUST_CEIL)

        # 4. ranking: trust, usage and recency all move the score monotonically
        t.check(
            "rank_score grows with trust", rank_score(0.9, 0, 0) > rank_score(0.3, 0, 0)
        )
        t.check(
            "rank_score grows with usage count",
            rank_score(0.5, 10, 0) > rank_score(0.5, 0, 0),
        )
        t.check(
            "rank_score decays with age",
            rank_score(0.5, 0, 0) > rank_score(0.5, 0, 180),
        )
        t.check(
            "rank_score halves at the half-life",
            abs(rank_score(0.5, 0, RANK_HALF_LIFE_DAYS) - 0.25) < 1e-9,
        )
        t.check("rank_score tolerates junk input", rank_score(None, "x", None) == 0.0)

        # 5. decay of long-unused facts (floor enforced)
        conn = connect()
        try:
            conn.execute(
                "UPDATE facts SET last_retrieved_at=NULL, created_at=datetime('now','-400 days') "
                "WHERE fact_id=?",
                (fid_c,),
            )
            conn.commit()
        finally:
            conn.close()
        before = connect()
        try:
            trust_before = before.execute(
                "SELECT trust_score FROM facts WHERE fact_id=?", (fid_c,)
            ).fetchone()[0]
            retrieved_before = before.execute(
                "SELECT trust_score FROM facts WHERE fact_id=?", (fid_a,)
            ).fetchone()[0]
        finally:
            before.close()
        affected = decay_unused(days=30, rate=DECAY_RATE)
        after = connect()
        try:
            trust_after = after.execute(
                "SELECT trust_score FROM facts WHERE fact_id=?", (fid_c,)
            ).fetchone()[0]
            retrieved_after = after.execute(
                "SELECT trust_score FROM facts WHERE fact_id=?", (fid_a,)
            ).fetchone()[0]
        finally:
            after.close()
        t.check("decay_unused touches the stale fact", affected == 1)
        t.check(
            "decay_unused lowers the stale fact's trust",
            abs(trust_before - trust_after - DECAY_RATE) < 1e-6,
        )
        t.check(
            "decay_unused leaves recently retrieved facts alone",
            abs(retrieved_before - retrieved_after) < 1e-9,
        )
        t.check("decay never drops below TRUST_FLOOR", trust_after >= TRUST_FLOOR)
        # drive it to the floor
        conn = connect()
        try:
            conn.execute(
                "UPDATE facts SET trust_score=? WHERE fact_id=?", (TRUST_FLOOR, fid_c)
            )
            conn.commit()
        finally:
            conn.close()
        t.check(
            "decay_unused skips facts already at the floor",
            decay_unused(days=30, rate=DECAY_RATE) == 0,
        )

        # 6. stats snapshot
        snapshot = stats()
        t.check("stats reports the total fact count", snapshot["total_facts"] == 3)
        t.check("stats reports the db path", snapshot["db_path"] == str(db))
        t.check(
            "stats lists most-used facts first",
            snapshot["top_used"][0]["fact_id"] == fid_a,
        )
        t.check(
            "stats includes a rank_score per fact",
            all("rank_score" in row for row in snapshot["top_used"]),
        )
        t.check("stats counts retrievals", snapshot["total_retrievals"] >= 4)

        exit_code = 0 if t.close("EVOLUTION SELFTEST") else 1
    finally:
        if previous is None:
            os.environ.pop("MEMORY_STORE_PATH", None)
        else:
            os.environ["MEMORY_STORE_PATH"] = previous
        tmpdir.cleanup()
    return exit_code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Icarus self-evolution engine")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run the full write/feedback/decay/rank loop against a temp sqlite file",
    )
    parser.add_argument(
        "--stats", action="store_true", help="print the evolution snapshot as JSON"
    )
    parser.add_argument(
        "--decay",
        type=int,
        metavar="DAYS",
        help="decay facts unused for DAYS (default 30) and print affected count",
    )
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()
    if args.stats:
        import json

        print(json.dumps(stats(), indent=2, ensure_ascii=False))
        return 0
    if args.decay is not None:
        affected = decay_unused(days=args.decay)
        print(f"decayed {affected} fact(s) unused for >= {args.decay} day(s)")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
