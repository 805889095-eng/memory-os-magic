# Skills

Reusable workflows and reference guides for operating the Memory OS stack. These skills document how each component works in practice — not just what it does, but how to configure it, tune it, and debug it when something breaks.

| Skill | Scope |
|-------|-------|
| [memory-architecture](memory-architecture.md) | Qdrant collection setup, embedding pipeline, decay/dedup, fallback cascade, context injection thresholds, and all production pitfalls (named vectors, DeepSeek json_object, pure-Python BM25 sparse vectors) |
| [context-injection](context-injection.md) | The Icarus `pre_llm_call` pipeline: 4 sources (Fabric, Qdrant, Sessions, Facts), relevance gating, social closer filter, per-session dedup, threshold tuning, and troubleshooting |
| [llm-wiki](llm-wiki.md) | Building and maintaining a Karpathy-style wiki: ingest workflow, page quality standards, lint/health checks, frontmatter conventions, and cross-referencing |
