#!/usr/bin/env python3
"""CJK retrieval regression tests.

WHY THIS FILE EXISTS
--------------------
The upstream test suite (_test_collapse.py, icarus/evolution.py --selftest) is
100% ASCII. Every Chinese failure mode below passed CI while being completely
broken in production, because no test ever fed a Han character to the retrieval
path. These tests pin the behavior that was measured broken:

  * hooks._tokenize("帮我看看部署规范") -> set()          (recall never fired)
  * collapse with a Chinese query -> 1 of 3 survivors      (rank by source prior)
  * _search_sessions / _search_facts -> [] on `len(t)>=4`  (both layers dead)
  * fabric-retrieve.tokenize(chinese) -> set()             (recall -> read_recent)

Run: python3 _test_cjk.py
No network, no DB fixtures, no deps beyond stdlib.
"""

import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from icarus import cjk_tokens as C  # noqa: E402
from icarus import collapse as COL  # noqa: E402
from icarus import hooks as H  # noqa: E402

_checks = {"pass": 0, "fail": 0}


def check(label, cond):
    if cond:
        _checks["pass"] += 1
        print(f"  [PASS] {label}")
    else:
        _checks["fail"] += 1
        print(f"  [FAIL] {label}")


# ── 1. Tokenization: the root cause ────────────────────────────────────────

print("\n[1] CJK tokenization")

ZH = "帮我看看这个部署规范有没有问题"

check("Chinese query yields a NON-EMPTY token set", len(C.tokenize(ZH)) > 0)
check(
    "Chinese query yields real bigrams (部署/规范/问题)",
    {"部署", "规范", "问题"}.issubset(C.tokenize(ZH)),
)
check(
    "ASCII-only text is unaffected",
    C.tokenize("check the docker deployment spec") == {"check", "docker", "deployment", "spec"},
)
check(
    "Mixed text keeps both scripts",
    {"docker", "部署", "规范"}.issubset(C.tokenize("Docker 部署规范")),
)
check(
    "Function characters break bigrams (no '个部' noise)",
    "个部" not in C.tokenize(ZH),
)
check("Empty input -> empty set", C.tokenize("") == set())
check("Stopword-only ASCII -> empty set", C.tokenize("the a an is") == set())
check(
    "Lone non-stop Han character survives (not silently dropped)",
    C.tokenize("鑫") == {"鑫"},
)
# A lone STOP character ("啊") legitimately yields nothing — it carries no signal.
check("Lone stop character yields empty set", C.tokenize("啊") == set())
check("is_cjk detects Han", C.is_cjk("部署") and not C.is_cjk("deploy"))

# The three scoring tokenizers must agree, or a fact can be retrieved and then
# never rewarded (retrieval says relevant, feedback says 0% overlap).
check(
    "hooks._tokenize == collapse.tokenize on Chinese",
    H._tokenize(ZH) == COL.tokenize(ZH),
)
check("hooks._tokenize matched bigrams", {"部署", "规范"}.issubset(H._tokenize(ZH)))

# ── 2. FTS query building ──────────────────────────────────────────────────

print("\n[2] FTS query building")

q = C.fts_query(ZH, ascii_only=False)
check("Chinese FTS query is non-empty (diagnostic mode)", bool(q))
check("Han chars become prefix terms (部*)", "部*" in q)
check("ASCII terms are quoted whole", '"docker"' in C.fts_query("docker 部署"))
check("No usable terms -> empty string", C.fts_query("的了呢", ascii_only=False) == "")
check("Empty input -> empty string", C.fts_query("") == "")
check(
    "max_terms is respected",
    len(C.fts_query("一二三四五六七八九十甲乙丙丁", max_terms=3, ascii_only=False).split(" OR ")) == 3,
)

# ── 3. FTS actually matches Chinese rows (real SQLite) ─────────────────────

print("\n[3] FTS5 round-trip + CJK substring path")

fts_tmp = os.path.join(tempfile.mkdtemp(), "fts.db")
con = sqlite3.connect(fts_tmp)
con.execute("CREATE VIRTUAL TABLE t USING fts5(content)")
rows = [
    "帮我看看这个部署规范有没有问题",
    "Docker 部署规范：Qdrant 端口 6333",
    "完全无关的一条记录",
]
for r in rows:
    con.execute("INSERT INTO t(content) VALUES (?)", (r,))
con.commit()
con.close()

con = sqlite3.connect(fts_tmp)

# MEASURED LIMITATION, pinned as a test so nobody "simplifies" the LIKE path
# away: unicode61 indexes a whole Han run as ONE token, so a prefix term only
# matches when the word is at the START of that run. "部署规范" is a standalone
# token in row 2 (matched by 部*) but sits mid-run in row 1 (not matched).
# ascii_only=False is the diagnostic form that actually emits Han prefix terms;
# retrieval uses ascii_only=True so it never suppresses the LIKE path.
q_clause = C.fts_query("部署规范", ascii_only=False)
fts_hits = con.execute("SELECT count(*) FROM t WHERE t MATCH ?", (q_clause,)).fetchone()[0]
check(
    f"FTS prefix-OR matches only the row where the word starts the Han run (got {fts_hits}/2)",
    fts_hits == 1,
)
check(
    "Retrieval-mode fts_query emits NO Han terms (so LIKE still runs)",
    C.fts_query("部署规范") == "",
)

# The LIKE path is what actually works for CJK — this is why hooks uses it.
def like_hits(terms):
    clauses = " OR ".join(["content LIKE ?"] * len(terms))
    return con.execute(
        f"SELECT count(*) FROM t WHERE ({clauses})", [f"%{x}%" for x in terms]
    ).fetchone()[0]

terms = C.cjk_terms("部署规范")
check(f"cjk_terms yields substrings (got {terms})", "部署" in terms and "规范" in terms)
check(f"LIKE finds BOTH relevant rows (got {like_hits(terms)}/2)", like_hits(terms) == 2)
check(
    "LIKE beats FTS prefix-OR for CJK (the reason for the dual path)",
    like_hits(terms) > fts_hits,
)

unrelated = C.cjk_terms("完全无关的另外内容")
check(
    f"LIKE does not match the deployment rows for unrelated text (got {like_hits(unrelated)}/3)",
    like_hits(unrelated) <= 1,
)

# ASCII must still go through FTS and still rank.
ascii_clause = C.fts_query("docker deployment")
ascii_hits = con.execute("SELECT count(*) FROM t WHERE t MATCH ?", (ascii_clause,)).fetchone()[0]
check(f"ASCII FTS query still matches (got {ascii_hits})", ascii_hits == 1)
con.close()

# ── 4. Collapse ranking is relevance-driven, not source-prior-driven ──────

print("\n[4] Collapse ranking")

def _cands():
    return [
        {"key": ("fabric", 0), "source": "fabric", "rank": 0, "score": None,
         "text": "Docker 部署规范：Qdrant 使用 6333 端口，Redis 需要密码"},
        {"key": ("qdrant", 0), "source": "qdrant", "rank": 0, "score": 0.7,
         "text": "redis 配置 requirepass 密码"},
        {"key": ("sessions", 0), "source": "sessions", "rank": 0, "score": None,
         "text": "上次讨论过 Kubernetes 集群扩容"},
    ]

qt = COL.tokenize(ZH)
check("Collapse query tokens are non-empty", len(qt) > 0)

scored = {r["candidate"]["source"]: r["salience"] for r in COL.score_all(_cands(), qt)}
check(
    "Relevant Chinese candidate outranks an irrelevant one",
    scored["fabric"] > scored["sessions"],
)
check(
    "Relevant candidate clears the prune floor (survives)",
    "fabric" in [c["source"] for c in COL.collapse(_cands(), qt, budget=6)],
)

# Regression guard for the exact upstream failure: with an ASCII-only tokenizer
# the query had 0 tokens, every candidate scored 0.45*base*prior, and only the
# highest-prior source survived. Relevance must now beat source prior.
survivors = COL.collapse(_cands(), qt, budget=6)
check(f"More than one candidate survives a Chinese query (got {len(survivors)})", len(survivors) > 1)
check(
    "Top survivor is the topical one, not the highest-prior one",
    survivors[0]["source"] == "fabric",
)

# ── 5. Feedback tokenizer must match retrieval tokenizer ──────────────────

print("\n[5] Feedback / retrieval tokenizer alignment")

check(
    "_fact_tokens sees Chinese content",
    len(H._fact_tokens("部署规范要求 Qdrant 端口 6333")) > 0,
)
# A Chinese fact that reappears in the answer must register real overlap, else
# it can never be rewarded and its trust_score decays to the floor unused.
fact = "部署规范要求 Qdrant 端口 6333"
answer = "根据部署规范，Qdrant 端口应该是 6333。"
ftoks = H._fact_tokens(fact)
rtoks = H._fact_tokens(answer)
overlap = len(ftoks & rtoks) / len(ftoks)
check(f"Chinese fact/answer overlap is measurable (got {overlap:.2f})", overlap > 0.3)

# ── 6. fabric-retrieve tokenizer (bare-script import path) ────────────────

print("\n[6] fabric-retrieve module")

import importlib.util

spec = importlib.util.spec_from_file_location(
    "fabric_retrieve_test",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "icarus", "fabric-retrieve.py"),
)
fr = importlib.util.module_from_spec(spec)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "icarus"))
spec.loader.exec_module(fr)

check("fabric-retrieve.tokenize sees Chinese", len(fr.tokenize(ZH)) > 0)
check("fabric-retrieve.tokenize keeps bigrams", "部署" in fr.tokenize(ZH))
check(
    "fabric-retrieve._cjk_sequence yields ordered Han units",
    fr._cjk_sequence("部署规范") == ["部", "署", "规", "范"],
)

# ── 7. Session scoring / decision capture for Chinese ─────────────────────

print("\n[7] Chinese decision/outcome detection (state regexes)")

from icarus import state as ST  # noqa: E402

ZH_DECISION = "新增 cjk_tokens.py 做汉字差分词，问题已解决，最终测试全过。"
ZH_NEUTRAL = "今天天气不错，随便聊聊。"
EN_DECISION = "We fixed the bug and deployed the patch."

check("DECISION_RE matches Chinese decisions", bool(ST.DECISION_RE.search(ZH_DECISION)))
check("DECISION_RE does not match neutral Chinese", not ST.DECISION_RE.search(ZH_NEUTRAL))
check("DECISION_RE still matches English", bool(ST.DECISION_RE.search(EN_DECISION)))
check("OUTCOME_RE matches Chinese outcomes", bool(ST.OUTCOME_RE.search(ZH_DECISION)))
check("COMPLETION_RE matches Chinese completion", bool(ST.COMPLETION_RE.search(ZH_DECISION)))

# score_session is what gates on_session_end (total >= 0.2). With English-only
# regexes a pure-Chinese technical session scored 0.08 and was discarded, so no
# Chinese session ever reached the fact store.
ST.bind_session("cjk-score-test")
ST.exchanges.clear()
ST.exchanges.extend([
    {"user": "帮我深度分析这个仓库的中文支持情况",
     "assistant": "发现严重问题：_tokenize 是 ASCII-only，纯中文 query 返回空集，召回链路对中文失效。这个必须修复。"},
    {"user": "那具体怎么改的",
     "assistant": "新增 cjk_tokens 做汉字 bigram 分词，改了五处调用点。实测 unicode61 把整段汉字当单个 token，"
                  "所以中文改用 LIKE 子串匹配。最终 42 项测试全过，问题已解决。"},
])
_sc = ST.score_session()
check(f"Chinese session decision score is non-zero (got {_sc['decision']})", _sc["decision"] > 0)
check(f"Chinese session clears the 0.2 write gate (got {_sc['total']})", _sc["total"] >= 0.2)

# The other half: post_llm_call must capture a Chinese decision as a fabric entry.
check(
    "post_llm_call gate would fire on Chinese (decision+outcome+len+user)",
    bool(ST.DECISION_RE.search(ZH_DECISION)) and bool(ST.OUTCOME_RE.search(ZH_DECISION)),
)

# ── 8. scripts/context_enhancer (Qdrant-layer fallback terms) ─────────────

print("\n[8] context_enhancer loading + fallback tokenization")

try:
    import scripts.context_enhancer as CE  # noqa: E402

    check("scripts is importable as a package (was ModuleNotFoundError)", True)
    check(
        "tokenize_query sees Chinese (both fallback levels depended on it)",
        len(CE.tokenize_query(ZH)) > 0,
    )
    check(
        "tokenize_query keeps Chinese bigrams for LIKE matching",
        "部署" in CE.tokenize_query(ZH),
    )
    check(
        "tokenize_query still filters English stopwords",
        CE.tokenize_query("the docker deployment") == sorted(["docker", "deployment"]),
    )
except ImportError as exc:
    check(f"scripts.context_enhancer importable ({exc})", False)

# The hook must be able to LOAD the enhancer even when the name ``scripts`` has
# been claimed by another package (Hermes ships its own scripts/ directory, which
# is a namespace package and shadows ours). Path-loading is what makes this work.
print("\n[9] hook _load_context_enhancer survives the ``scripts`` name collision")

import types as _types

saved = sys.modules.get("scripts")
sys.modules["scripts"] = _types.ModuleType("scripts")  # no __path__: no submodules
try:
    mod = H._load_context_enhancer()
    check("enhancer loads by path despite a shadowing 'scripts' module", mod is not None)
    if mod is not None:
        check("loaded module exposes the 4-level fallback chain", hasattr(mod, "search_with_fallback"))
        check("loaded module exposes embed_query", hasattr(mod, "embed_query"))
finally:
    if saved is None:
        sys.modules.pop("scripts", None)
    else:
        sys.modules["scripts"] = saved

# [10] Fact-store LIKE path must apply the same precision gate as sessions.
# Without it, OR-ing bare bigrams accepts any fact sharing ONE common word:
# "完全不存在的话题xyz" matched a real fact through "%完全%" alone. The
# sessions path already gates on `need = len(terms)//2`; the facts path did not.
print("\n[10] _search_facts precision gate on the CJK LIKE path")

import sqlite3 as _sqlite3
import tempfile as _tf
import pathlib as _pl

_fdir = _pl.Path(_tf.mkdtemp())
_fdb_path = _fdir / "memory_store.db"
_con = _sqlite3.connect(str(_fdb_path))
_con.executescript(
    """
    CREATE TABLE facts (
        fact_id INTEGER PRIMARY KEY, content TEXT, trust_score REAL,
        retrieval_count INTEGER, helpful_count INTEGER,
        created_at TEXT, updated_at TEXT
    );
    CREATE VIRTUAL TABLE facts_fts USING fts5(content, content='facts',
                                              content_rowid='fact_id');
    """
)
_rows = [
    (1, "查了完全没关系的资料", "2026-10-20T00:00:00Z"),
    (2, "worker 反思把矛盾标出来了", "2026-10-20T00:00:00Z"),
    (3, "redis 队列没有落盘", "2026-10-20T00:00:00Z"),
]
_con.executemany(
    "INSERT INTO facts VALUES (?,?,0.55,0,0,?,?)",
    [(fid, c, ts, ts) for fid, c, ts in _rows],
)
_con.executemany(
    "INSERT INTO facts_fts(rowid, content) VALUES (?,?)",
    [(fid, c) for fid, c, _ in _rows],
)
_con.commit()
_con.close()

_orig_candidates = H._facts_db_candidates
H._facts_db_candidates = lambda: [_fdb_path]
try:
    # The gate is a >=3-term protection. "完全不存在的量子纠缠话题" tokenizes
    # to 11 bigrams (need=5); fact 3 shares only "完全" (1 hit) and must be
    # rejected. Before the gate existed, that single common bigram was enough.
    _noise = H._search_facts("完全不存在的量子纠缠话题", top_k=3)
    check(
        f"multi-term noise query rejected at need=5 (got {[x['fact_id'] for x in _noise]})",
        _noise == [],
    )

    # A short noise query is still permissive: need = max(1, 2//2) = 1, so one
    # shared common bigram passes. Documented limitation, identical to the
    # sessions layer — bigrams have no IDF, so a common word carries a short
    # query all by itself.
    _short = H._search_facts("完全不存在的话题", top_k=3)
    check(
        f"short noise query still leaks at need=1 (documented, got {[x['fact_id'] for x in _short]})",
        [x["fact_id"] for x in _short] == [1],
    )

    _real = H._search_facts("worker 反思矛盾", top_k=3)
    check(
        f"relevant fact still returns (got {[x['fact_id'] for x in _real]})",
        any(x["fact_id"] == 2 for x in _real),
    )

finally:
    H._facts_db_candidates = _orig_candidates
    import shutil as _sh
    _sh.rmtree(_fdir, ignore_errors=True)

# ── Summary ────────────────────────────────────────────────────────────────
print()
print(f"=== {_checks['pass']} passed, {_checks['fail']} failed ===")
if _checks["fail"]:
    print("=== CJK TESTS FAILED ===")
    sys.exit(1)
print("=== ALL CJK TESTS PASS ===")
