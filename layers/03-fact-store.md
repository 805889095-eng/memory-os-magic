# Layer 3 — Structured Facts (Holographic Memory)

> **File:** `$HERMES_HOME/memory_store.db` (SQLite + HRR + FTS5 + trust scoring)
> **Tool:** `fact_store` (CRUD: add, search, probe, reason, update, remove, contradict)
> **Feedback:** `fact_feedback` (helpful/unhelpful)

## What it stores

Durable, structured facts with entity resolution and trust scoring:

```
facts table:
  fact_id, content, category (user_pref|project|tool|general),
  entities (JSON array), tags, trust_score, retrieval_count,
  helpful_count, created_at, last_accessed_at
```

## How the agent uses it

6 actions:

| Action | What it does | Example |
|--------|-------------|---------|
| `add` | Store a new fact | `fact_store(action='add', content='...', entities=['Qdrant'])` |
| `search` | Keyword lookup (FTS5) | `fact_store(action='search', query='Qdrant named vectors')` |
| `probe` | All facts about an entity | `fact_store(action='probe', entity='Docker')` |
| `reason` | Compositional: facts connected to multiple entities | `fact_store(action='reason', entities=['Qdrant', 'Docker'])` |
| `contradict` | Find conflicting claims | `fact_store(action='contradict')` |
| `update/remove` | CRUD maintenance | `fact_store(action='update', fact_id=87, trust_delta=0.1)` |

## Trust scoring

The system tracks which facts are actually useful:

- `retrieval_count` — how many times the fact was retrieved
- `helpful_count` — how many times it was marked helpful
- `trust_score` — moves with feedback: starts at **0.55** and is adjusted by the self-evolution engine (see below)

**Critical rule:** When you retrieve a fact via `probe`, `search`, or `reason` and reference it in your response, you MUST call `fact_feedback` in the same turn. Without feedback, `trust_score` is ornamental and fact quality degrades silently.

## Self-evolution engine

`icarus/evolution.py` turns the fact store into a closed feedback loop — facts earn or lose trust by being used. Five steps:

1. **Extract & write** — the session extractor records facts with a category and tags and a starting trust of **0.55** (`evolution.record_fact`).
2. **Retrieval counting** — every fact injected into the prompt bumps its `retrieval_count` (`evolution.record_retrieval`).
3. **Automatic feedback** — when the assistant's reply reuses an injected fact (keyword overlap), the loop rewards it with no user action (`hooks.py: _auto_feedback`). Explicit votes come from the `fact_feedback` tool.
4. **Trust adjustment** — `evolution.apply_feedback` adds **+0.08** to a helpful fact (and increments `helpful_count`) or subtracts **0.12** from a misleading one; trust is clamped to **[0.05, 1.0]**.
5. **Re-rank & decay** — injection order uses `rank_score = trust × (1 + 0.15·log1p(retrieval_count)) × exp(−ln2·age_days/180)`, so trusted, frequently-used, recent facts surface first. `evolution.decay_unused` fades facts untouched for **30+ days** by **0.02** per run, down to the **0.05** floor.

Constants: `INITIAL_TRUST=0.55`, `HELPFUL_DELTA=0.08`, `HARMFUL_DELTA=0.12`, `TRUST_FLOOR=0.05`, `TRUST_CEIL=1.0`, `DECAY_RATE=0.02`, `DECAY_DAYS=30`, `RANK_HALF_LIFE_DAYS=180`, `RANK_USE_GAIN=0.15`.

**Enabling:** nothing to configure — the loop is active whenever the Icarus plugin is enabled. Facts live in `$HERMES_HOME/memory_store.db` (tables `facts` + `facts_fts`, created by `setup/setup_db.py`).

**Observing:** `memory_evolution_stats()` returns a snapshot (total facts, average trust, most-used facts with their rank scores); `fact_feedback(fact_id=..., helpful=True|False)` casts an explicit vote. `python icarus/evolution.py --selftest` exercises the whole loop end to end.

## Context injection (Icarus enhancement)

The Icarus fork adds `_search_facts()` — FTS5 search during `pre_llm_call` that injects relevant facts on the **first turn only** (to avoid per-turn cost). Labeled `[facts]` in prompt.

## Configuration

No configuration needed — the trust/feedback loop runs automatically whenever the Icarus plugin is enabled, and the DB path (`$HERMES_HOME/memory_store.db`) is managed by Hermes. The `facts` and `facts_fts` tables are created by `setup/setup_db.py`.

## Pitfalls

- **`retrieval_count` was broken for `probe()`/`reason()`** — only `search()` incremented it. Fixed in current code
- **Trust scores need feedback to move:** explicit `fact_feedback` votes and the automatic reply-overlap feedback both move the score; without either, a fact keeps its starting 0.55
- **HRR is not exposed to the agent directly:** It powers entity resolution internally — the agent sees keyword search + entity linking, not the holographic vectors
