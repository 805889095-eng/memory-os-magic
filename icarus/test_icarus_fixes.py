"""Regression tests for the icarus plugin fixes.

Every case here corresponds to a bug that shipped silently: the code ran, said
nothing, and produced wrong results. Run under the real Hermes runtime
interpreter, because the yaml-shadowing bug only reproduces there.

    <hermes runtime python> -I test_icarus_fixes.py
"""

import json
import logging
import os
import shutil
import sys
import tempfile
import traceback
import urllib.error
import urllib.request
from pathlib import Path

# Plugin under test. Default: the package next to this file, so the suite runs
# on any checkout. ICARUS_PLUGIN_DIR points it at a live install instead, which
# is how a bug that only reproduces under the Hermes runtime is reproduced.
REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_DIR = Path(os.environ.get("ICARUS_PLUGIN_DIR") or (REPO_ROOT / "icarus")).resolve()

# The Hermes runtime ships ruamel.yaml, not PyYAML, which is the whole reason
# icarus/_yaml.py exists. Its bootstrap puts that runtime's site-packages on
# sys.path when this suite runs under the Hermes interpreter; everywhere else
# it is simply absent and the PyYAML backend is exercised instead.
_hermes_agent = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")) / "hermes-agent"
if (_hermes_agent / "hermes_bootstrap.py").is_file():
    sys.path.insert(0, str(_hermes_agent))
    try:
        import hermes_bootstrap  # noqa: F401
    except Exception:  # pragma: no cover - bootstrap is optional
        pass

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "icarus", str(PLUGIN_DIR / "__init__.py"), submodule_search_locations=[str(PLUGIN_DIR)]
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["icarus"] = _mod
_spec.loader.exec_module(_mod)

from icarus import _yaml, evolution, parsing, state  # noqa: E402
from icarus import hooks  # noqa: E402

SCRATCH = os.environ.get("ICARUS_TEST_SCRATCH") or None
PASS, FAIL = [], []


def check(label, cond, detail=""):
    (PASS if cond else FAIL).append((label, detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail and not cond else ""))


# ── 1. yaml backend must exist in the Hermes runtime ──────────────────────
print("\n1. YAML backend resolution (PyYAML is absent from the Hermes runtime)")
try:
    import yaml  # noqa: F401

    pyyaml_present = True
except ImportError:
    pyyaml_present = False
print(f"  (PyYAML importable here: {pyyaml_present})")
check("backend is not the mini fallback", _yaml.backend_name() != "mini", _yaml.backend_name())
check(
    "quoted scalar unquoted",
    _yaml.load_scalar('v: "high"\n', "v") == "high",
    repr(_yaml.load_scalar('v: "high"\n', "v")),
)
check(
    "single-quoted scalar unquoted",
    _yaml.load_scalar("v: 'hot'\n", "v") == "hot",
    repr(_yaml.load_scalar("v: 'hot'\n", "v")),
)
check(
    "boolean normalised to lowercase text",
    _yaml.load_scalar('v: "true"\n', "v") == "true",
    repr(_yaml.load_scalar('v: "true"\n', "v")),
)
check(
    "flow list joined",
    _yaml.load_scalar('v: ["a", "b"]\n', "v") == "a, b",
    repr(_yaml.load_scalar('v: ["a", "b"]\n', "v")),
)
check(
    "escaped CJK summary round-trips",
    _yaml.load_scalar('summary: "\\u914d\\u7f6e LLM"\n', "summary") == "配置 LLM",
    repr(_yaml.load_scalar('summary: "\\u914d\\u7f6e LLM"\n', "summary")),
)
check("missing key returns empty", _yaml.load_scalar("a: b\n", "v") == "")
check(
    "key is not matched as a prefix of another key",
    _yaml.load_scalar("type_extra: nope\ntype: decision\n", "type") == "decision",
    repr(_yaml.load_scalar("type_extra: nope\ntype: decision\n", "type")),
)

# ── 2. _parse_head unquotes every field it exposes ────────────────────────
print("\n2. _parse_head on a real file written by write_entry")
tmp = Path(tempfile.mkdtemp(prefix="icarus_t_", dir=SCRATCH))
saved = (state.FABRIC_DIR, state._retriever)
state.FABRIC_DIR = tmp
state.bind_session("test-fixes", reset=True)
try:
    path = Path(
        state.write_entry(
            "decision",
            '## Task\n修复 "引号"\n\n## Result\n已完成，实测通过。',
            '摘要含"引号"',
            tags="x,y",
            status="completed",
            training_value="high",
            verified="true",
        )
    )
    head = state._parse_head(path)
    for key, want in [
        ("type", "decision"),
        ("tier", "hot"),
        ("status", "completed"),
        ("training_value", "high"),
        ("verified", "true"),
        ("summary", '摘要含"引号"'),
    ]:
        check(f"{key} == {want!r}", head.get(key) == want, repr(head.get(key)))

    # ── 3. id-based lookups work (they compared against a quoted id) ──────
    print("\n3. has_entry_ref / curate_entry")
    ref = f"{head.get('agent')}:{head.get('id')}"
    check("has_entry_ref resolves", state.has_entry_ref(ref), ref)
    state.curate_entry(head["id"], "low")
    check(
        "curate_entry persists",
        state._parse_head(path).get("training_value") == "low",
        repr(state._parse_head(path).get("training_value")),
    )

    # ── 4. report buckets are keyed by unquoted type/value ────────────────
    print("\n4. build_weekly_report bucketing")
    rep = state.build_weekly_report()
    check("by_type has no quote noise", rep["by_type"] == {"decision": 1}, json.dumps(rep["by_type"]))
    check("by_training_value sees 'low'", rep["by_training_value"]["low"] == 1, json.dumps(rep["by_training_value"]))

    # ── 5. read_recent / recall honour tier + ranked retrieval ────────────
    print("\n5. read_recent and ranked recall")
    check("read_recent returns the entry", len(state.read_recent(limit=5)) == 1)
    hit = state.recall("引号 修复", max_results=5)
    check("recall returns a scored hit", bool(hit) and "score" in hit[0], json.dumps(hit[:1], ensure_ascii=False)[:200])

    # ── 6. fabric index fingerprint: deletion + directory switch ──────────
    print("\n6. fabric_index fingerprint (deletion / directory isolation)")
    state._retriever = None
    state.recall("引号", max_results=5)  # force index build for `tmp`
    path.unlink()
    after_delete = state.recall("引号", max_results=5)
    check("deleted entry leaves the index", len(after_delete) == 0, f"{len(after_delete)} still indexed")

    other = Path(tempfile.mkdtemp(prefix="icarus_t2_", dir=SCRATCH))
    state.FABRIC_DIR = other
    state.write_entry("decision", "## Task\n别的目录\n\n## Result\n完成", "另一目录唯一条目")
    titles = [str(e.get("title")) for e in state.recall("引号", max_results=5)]
    # `other` may legitimately return its own entry (recency/type baseline
    # score), so the assertion is "nothing from `tmp` leaked", not "empty".
    check(
        "retrieval does not leak across FABRIC_DIR",
        all("引号" not in t for t in titles),
        titles,
    )
    state.FABRIC_DIR = tmp
    check(
        "switching back does not leak either",
        all("另一目录" not in str(e.get("title")) for e in state.recall("引号", max_results=5)),
    )
    shutil.rmtree(other, ignore_errors=True)

    # ── 7. export passes --fabric-dir (defaulted to ~/fabric before) ──────
    print("\n7. fabric_export with a non-default FABRIC_DIR")
    state.FABRIC_DIR = tmp
    state.write_entry("resolution", "## Task\nA\n\n## Result\nB 完成", "导出测试条目", training_value="high")
    res = json.loads(state.export_training.__wrapped__(mode="normal") if hasattr(state.export_training, "__wrapped__") else json.dumps(state.export_training(mode="normal")))
    check("export succeeds", "error" not in res, json.dumps(res, ensure_ascii=False)[:300])
    check("export produced pairs", res.get("pairs", 0) > 0, json.dumps(res, ensure_ascii=False)[:300])
finally:
    state.FABRIC_DIR, state._retriever = saved
    shutil.rmtree(tmp, ignore_errors=True)

# ── 8. parsing.parse_entry matches the scalar path ────────────────────────
print("\n8. parsing.parse_entry")
sample = Path(tempfile.mkdtemp(prefix="icarus_p_", dir=SCRATCH)) / "e.md"
sample.write_text(
    '---\nid: "abcd1234"\ntraining_value: "high"\nverified: "true"\ntags: ["a", "b"]\n---\n\nbody\n',
    encoding="utf-8",
)
try:
    meta = parsing.parse_entry(sample)
    for key, want in [("id", "abcd1234"), ("training_value", "high"), ("verified", "true")]:
        check(f"parse_entry {key}", meta.get(key) == want, repr(meta.get(key)))
    check("parse_entry tags list", meta.get("tags") == ["a", "b"], repr(meta.get("tags")))
finally:
    shutil.rmtree(sample.parent, ignore_errors=True)

# ── 9. fallback parser still works with no YAML backend at all ────────────
print("\n9. mini fallback (backend forced to None)")
real_backend = _yaml._BACKEND
_yaml._BACKEND = None
try:
    check(
        "mini parses a quoted scalar",
        _yaml.load_scalar('v: "high"\n', "v") == "high",
        repr(_yaml.load_scalar('v: "high"\n', "v")),
    )
    check(
        "mini parses a flow list",
        _yaml.load_scalar('v: ["a", "b"]\n', "v") == "a, b",
        repr(_yaml.load_scalar('v: ["a", "b"]\n', "v")),
    )
finally:
    _yaml._BACKEND = real_backend

# ── 10. extraction must not store a reply the model ran out of tokens on ──
# `_parse_json_robust` strips trailing characters until json.loads succeeds, so
# a truncated array still yields a *prefix* of the entries. With max_tokens=1024
# the model stopped at finish_reason="length" and six half-written facts (all
# exactly 1024 chars, identical openings, dangling endings) reached the store.
print("\n10. LLM extraction truncation guard")

check(
    "default extraction budget is not the truncating 1024",
    hooks._EXTRACTION_MAX_TOKENS > 1024,
    str(hooks._EXTRACTION_MAX_TOKENS),
)

import urllib.request as _urlreq


class _FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


_truncated_choice = {
    "choices": [
        {
            # A syntactically valid prefix: the array closes, but the single
            # entry's content is cut mid-sentence.
            "message": {
                "content": '[{"type": "note", "summary": "half written entry here", '
                '"content": "## Context\\n这是一个被截断的内容，写到一半就没了", '
                '"training_value": "normal"}]'
            },
            "finish_reason": "length",
        }
    ]
}

_orig_urlopen = _urlreq.urlopen
_orig_key = hooks._resolve_llm_api_key
hooks._resolve_llm_api_key = lambda: "test-key"
_urlreq.urlopen = lambda *a, **k: _FakeResponse(_truncated_choice)
try:
    got = hooks._llm_extract_entries("[Turn 1 — User]\nhello\n\n[Turn 2 — Agent]\nworld")
    check("truncated (finish_reason=length) reply yields no entries", got == [], str(got))
finally:
    _urlreq.urlopen = _orig_urlopen
    hooks._resolve_llm_api_key = _orig_key

# The guard must not swallow well-formed replies.
_ok_choice = {
    "choices": [
        {
            "message": {
                "content": '[{"type": "note", "summary": "a complete entry summary here", '
                '"content": "## Context\\n这是一段足够长的完整内容，用来通过提炼结果的长度校验。'
                '\\n\\n## Outcome\\n已经完成并验证通过，收尾完整。", '
                '"training_value": "high"}]'
            },
            "finish_reason": "stop",
        }
    ]
}
hooks._resolve_llm_api_key = lambda: "test-key"
_urlreq.urlopen = lambda *a, **k: _FakeResponse(_ok_choice)
try:
    got = hooks._llm_extract_entries("[Turn 1 — User]\nhello\n\n[Turn 2 — Agent]\nworld")
    check("complete reply still parses", len(got) == 1, str(got))
finally:
    _urlreq.urlopen = _orig_urlopen
    hooks._resolve_llm_api_key = _orig_key

# ── 11. fact excerpts must window on the match, not the head ──────────────
# _search_facts used to return content[:200]. Facts are multi-section markdown,
# so the query terms routinely land past offset 200: one fact had "token" at
# 252 behind an unrelated preamble. The injected excerpt then had zero overlap
# with the query, and the collapse pruner — which scores candidates by token
# overlap — dropped every fact candidate as noise. [facts] silently vanished
# from the prompt even though the layer had found real hits.
print("\n11. fact excerpt windowing")

_probe = "A" * 250 + "token 泄露风险在这里" + "B" * 300
_ex = hooks._fact_excerpt(_probe, ["token"])
check("excerpt contains the match past offset 200", "token" in _ex, repr(_ex[:80]))
check("excerpt is bounded", len(_ex) <= 210, str(len(_ex)))
check("excerpt marks the elision", _ex.startswith("…"), repr(_ex[:10]))

_short = "很短的一条事实"
check("short fact returned whole", hooks._fact_excerpt(_short, ["token"]) == _short)

_no_hit = "C" * 500
check(
    "no term present falls back to the head slice",
    hooks._fact_excerpt(_no_hit, ["token"]) == "C" * 200,
)

check(
    "ascii query words are picked up for windowing",
    hooks._query_words("token 泄露") == ["token"],
    str(hooks._query_words("token 泄露")),
)

# End to end: a fact whose match sits past 200 chars must keep query overlap
# in the excerpt the collapse ranker sees.
_deep_content = "无关前言。" * 60 + "token 泄露的处理办法"
_ex2 = hooks._fact_excerpt(_deep_content, hooks._query_words("token"))
check(
    "deep match survives into the excerpt",
    "token" in _ex2,
    repr(_ex2[-60:]),
)

# ── 11b. excerpt must anchor on the DENSEST hit window, not the earliest ──
# min(positions) anchors on the first place ANY term appears. A CJK query is
# split into bigrams ("修好" "代码" "文件") and common bigrams appear all over a
# long multi-section fact, so the earliest hit is routinely an unrelated
# paragraph while the section that actually answers the query sits 400 chars
# later. Measured on the live store: 7 of 304 (query, fact) pairs had <40% of
# the query's hits inside the excerpt; the worst was 12% — the model was shown
# a PAT-token warning when asked where the fixed code files were.
print("\n11b. excerpt anchors on the densest window")

# Shaped like the real fact that exposed this (fact 29 on the live store):
# a common bigram appears once in an early unrelated section, while the
# section that actually answers the query sits ~270 chars later.
_dense = (
    "## Decision\n"
    + "这里有代码两个字但无关。"
    + "无关段落。" * 50
    + "\n## Result\n"
    + "修好了。代码在 patch.py，文件已提交，文件清单如下，文件路径 ~/x，代码行数 42。\n"
)
_dense_terms = ["修好", "代码", "文件"]
_dense_ex = hooks._fact_excerpt(_dense, _dense_terms, 200)
_dense_hits = sum(_dense_ex.count(t) for t in _dense_terms)
_dense_all = sum(_dense.count(t) for t in _dense_terms)
check(
    "densest window beats the earliest hit",
    _dense_hits >= 5,
    f"{_dense_hits}/{_dense_all} hits kept: {_dense_ex!r}",
)

# The Result section (where the answer lives) must be inside the window.
check(
    "the answering section is inside the window",
    "修好了" in _dense_ex,
    repr(_dense_ex[:100]),
)

# ── 11c. excerpt must not cut mid-word ────────────────────────────────────
# start/end were raw character offsets, so 15% of excerpts began or ended in
# the middle of a token: "Action/Decision" arrived as "tion/Decision",
# "Provider" as "Prov", "USER.md" as "SER.md". Fragments are noise in the
# prompt and poison the token-overlap feedback scorer.
print("\n11c. excerpt boundaries snap to word/section edges")

# The real symptom: "Action/Decision" arrived as "tion/Decision", "Provider" as
# "Prov", "USER.md" as "SER.md". start/end were raw character offsets with no
# alignment, so the cut landed inside tokens. Shaped like a real fact: multi-line
# markdown with paths, ASCII tokens and CJK punctuation.
_cut = (
    "## Context\n"
    "用户要求把 memory-os 完整接入 Hermes。安装副本位于 ~/.hermes/memory-os，"
    "工作副本在 ~/.hermes/repos/memory-os-magic。\n\n"
    "## Action/Decision\n"
    "逐一排查并修复：yaml 后端缺失导致 frontmatter 解析失败；"
    "export 子命令不传 --fabric-dir 导致子进程找不到目录；\n"
    "rollback 直接整体替换 .env 会覆盖用户自定义配置。\n\n"
    "## Result\n"
    "修好了。三个补丁已提交，回归测试 57 项全过。代码文件在 icarus/hooks.py，"
    "文件清单见 git log，路径是 ~/.hermes/repos/memory-os-magic。\n"
)
_cut_ex = hooks._fact_excerpt(_cut, ["修好", "代码", "文件"], 200)
_core = _cut_ex.lstrip("…").rstrip("…")
_idx = _cut.find(_core)
check("excerpt is a real slice of the fact", _idx != -1, repr(_core[:40]))

# A start/end is clean when the character just outside it is a boundary: the
# excerpt must never resume in the middle of a token. This is the actual
# invariant — "frontmatter" starting an excerpt is fine, "rontmatter" is not.
_BOUNDARY = set(" \t\n\u3002\uff1b\uff1a\uff0c\u3001\uff08\uff09`*|-/\\")
if _idx > 0:
    check(
        "head sits on a token boundary",
        _cut[_idx - 1] in _BOUNDARY,
        f"preceded by {_cut[_idx - 1]!r}",
    )
_end_idx = _idx + len(_core)
if _end_idx < len(_cut):
    check(
        "tail sits on a token boundary",
        _cut[_end_idx] in _BOUNDARY,
        f"followed by {_cut[_end_idx]!r}",
    )
check(
    "no ASCII token is sliced mid-word",
    not any(
        _core.startswith(frag)
        for frag in ("tion/", "cision", "Prov", "SER.md", "ider", "rontmatter", "eader")
    ),
    repr(_core[:40]),
)

# ── 11d. extraction failures must log the HTTP status, not just the class ──
# HTTPError subclasses URLError, so logging ``type(e).__name__`` recorded 400,
# 401, 404 and 413 identically as "HTTPError". 15 consecutive failures in the
# live log were unreadable; distinguishing them is the difference between
# "unknown model ID", "bad key" and "wrong base URL".
print("\n11d. extraction failure logs carry the status code")


class _FakeHTTPError(urllib.error.HTTPError):
    def __init__(self, code, body):
        super().__init__("http://x/v1/chat/completions", code, "err", {}, None)
        self._body = body.encode()

    def read(self, amt=None):
        return self._body


_cls = []


class _Cap(logging.Handler):
    def emit(self, record):
        _cls.append(record.getMessage())


def _extract_with_failure(code, body):
    """Drive the real _llm_extract_entries with urlopen forced to fail."""
    _cls.clear()
    logger = logging.getLogger("icarus.hooks")
    handler = _Cap()
    logger.addHandler(handler)
    prev = logger.level
    logger.setLevel(logging.WARNING)
    real_urlopen = urllib.request.urlopen
    real_key = os.environ.get("LLM_API_KEY")

    def _boom(*a, **k):
        raise _FakeHTTPError(code, body)

    try:
        os.environ["LLM_API_KEY"] = real_key or "test-key"
        # Module globals are read at import; force the resolved key so the call
        # reaches urlopen instead of the "no API key" early return.
        _saved_key = hooks._LLM_API_KEY
        hooks._LLM_API_KEY = "test-key"
        urllib.request.urlopen = _boom
        try:
            hooks._llm_extract_entries("user: hi\nassistant: hello there this is long enough")
        finally:
            hooks._LLM_API_KEY = _saved_key
    finally:
        urllib.request.urlopen = real_urlopen
        if real_key is None:
            os.environ.pop("LLM_API_KEY", None)
        logger.removeHandler(handler)
        logger.setLevel(prev)
    return list(_cls)


_msgs_400 = _extract_with_failure(400, '{"error":{"message":"model not found"}}')
_msgs_404 = _extract_with_failure(404, "no credentials for provider")
check(
    "400 is distinguishable from 404 in the log",
    any("HTTP 400" in m for m in _msgs_400) and any("HTTP 404" in m for m in _msgs_404),
    f"{_msgs_400} / {_msgs_404}",
)
check(
    "the provider's message rides along",
    any("model not found" in m for m in _msgs_400),
    str(_msgs_400),
)

# ── 12. extraction must not re-consume the whole transcript ───────────────
# Hermes fires on_session_end at every boundary it defines, including each
# context compression, always with the full growing transcript. Without a
# cursor each compression re-extracted all prior turns and wrote them back as
# new facts (a 9-row cluster of near-identical entries sharing a 734-char
# header). The cursor must advance, survive a rebind, and reset on a new
# session.
print("\n12. extraction cursor")

state.bind_session("cursor-regression", reset=True)
check("fresh session starts at zero", state.extract_cursor == 0, str(state.extract_cursor))

state.exchanges.extend(
    [{"user": "u1", "assistant": "a" * 120}, {"user": "u2", "assistant": "b" * 120}]
)
check("first pass sees everything", len(state.exchanges[state.extract_cursor:]) == 2)

hooks._commit_extract_cursor(len(state.exchanges))
check("cursor advanced past consumed turns",
      len(state.exchanges[state.extract_cursor:]) == 0, str(state.extract_cursor))

state.exchanges.append({"user": "u3", "assistant": "c" * 120})
check("only the new turn is pending",
      len(state.exchanges[state.extract_cursor:]) == 1, str(state.extract_cursor))

# Cursor holds the last COMMITTED position (2), not len() — the pending turn
# stays pending across a rebind, which is the property under test.
state.bind_session("cursor-regression")
check("cursor survives a rebind", state.extract_cursor == 2, str(state.extract_cursor))
check("pending turn still pending after rebind",
      len(state.exchanges[state.extract_cursor:]) == 1, str(state.extract_cursor))

state.bind_session("cursor-regression", reset=True)
check("cursor resets on a new session", state.extract_cursor == 0, str(state.extract_cursor))

# ── 13. harness scaffolding must be stripped from extracted content ───────
print("\n13. scaffolding strip")

_envelope = (
    "## Task\n[ASYNC DELEGATION COMPLETE — deleg_abc123]\n"
    "A background subagent you dispatched earlier has finished. "
    "You may have moved on since dispatching it; the full task source is below.\n\n"
    "## Decision\n旧的一轮内容，被模型抄进了每一条。\n\n"
    "## Result\n真正的结论在这里，长度足够通过校验。"
)
_out = hooks._strip_scaffolding(_envelope)
check("delegation envelope removed", "ASYNC DELEGATION" not in _out, repr(_out[:50]))
check("stale decision removed", "旧的一轮内容" not in _out, repr(_out[:50]))
check("real result kept", "真正的结论" in _out, repr(_out[:60]))
check("strip is idempotent", hooks._strip_scaffolding(_out) == _out)

_plain = "## Context\n普通条目。\n\n## Outcome\n完成。"
check("normal entry untouched", hooks._strip_scaffolding(_plain) == _plain)

# ── 14. the report must not publish an unmeasurable usage_rate ────────────
# usage_rate divides declared usages by recalls. Declarations come only from
# fabric_write(review_of=...), which the agent never passes, so the number was
# structurally 0.0 — and score_session() read it as a weight-2 component, so
# every session silently lost 20% of its quality score against a
# ``total < 0.2`` extraction gate. An automatic replacement was measured and
# rejected: token overlap between a recalled summary and the ensuing reply is
# indistinguishable from overlap with an unrelated reply (median 0.12 vs 0.14;
# 110 recalled / 66 control samples).
print("\n14. telemetry reports counts, not an invented usage rate")

_sum = state._summarize_telemetry_events(
    [
        {"event": "recall", "result_ids": ["a", "b"]},
        {"event": "recall", "result_ids": ["b", "c"]},
        {"event": "usage", "entry_id": "a"},
    ]
)
check("recall count kept", _sum["total_recalls"] == 2, str(_sum))
check("declared usage count kept", _sum["total_usages"] == 1, str(_sum))
check("unique recalled kept", _sum["unique_entries_recalled"] == 3, str(_sum))
check("unique used kept", _sum["unique_entries_used"] == 1, str(_sum))
check("usage_rate is not published", "usage_rate" not in _sum, str(_sum))

_tel = state.get_telemetry(last_n=1)
check(
    "empty/telemetry summary has no usage_rate either",
    "usage_rate" not in _tel["summary"],
    str(_tel["summary"]),
)


# ── 14b. session scoring must not be pinned by an unmeasurable component ──
# With usage_rate at a structural 0, recall_usage contributed 0/2 and the
# maximum reachable total was 0.8 — then a real session with modest depth sank
# under the 0.2 gate and its entries were never extracted.
print("\n14b. session score is not deflated by the usage component")

_saved_ex = state.exchanges
_saved_sid = state.session_id
_saved_tel = state.get_telemetry
try:
    # Two substantive turns: the old implementation's permanent 0.0 (weight 2)
    # capped the total here. Measured (old vs new, isolated subprocesses):
    #   exchanges  recalls=0        recalls>0
    #   2          0.15 -> 0.15     0.15 -> 0.35
    #   3          0.22 -> 0.22     0.22 -> 0.42
    #   5          0.30 -> 0.30     0.30 -> 0.50
    # So the fix only lifts sessions that actually recalled something — which
    # is the honest claim. It does NOT rescue a session that recalled nothing.
    state.exchanges = [
        {"user": "u" * 60, "assistant": "## Result\n" + "a" * 150} for _ in range(2)
    ]
    state.session_id = "score-probe"

    def _no_recalls(last_n=500, session_id_filter=None, agent_filter=None):
        return {"events": [], "summary": {
            "total_recalls": 0, "total_usages": 0,
            "unique_entries_recalled": 0, "unique_entries_used": 0,
        }}

    state.get_telemetry = _no_recalls
    _s0 = state.score_session()
    check(
        "component is named recall_activity",
        "recall_activity" in _s0 and "recall_usage" not in _s0,
        str(sorted(_s0)),
    )
    check(
        "no recalls scores the component 0 (not a hidden penalty)",
        _s0["recall_activity"] == 0.0,
        str(_s0["recall_activity"]),
    )
    check(
        "a session without recalls is not artificially lifted",
        abs(_s0["total"] - 0.15) < 0.005,
        f"total={_s0['total']} (both implementations score 0.15 here)",
    )

    def _some_recalls(last_n=500, session_id_filter=None, agent_filter=None):
        return {"events": [], "summary": {
            "total_recalls": 3, "total_usages": 0,
            "unique_entries_recalled": 2, "unique_entries_used": 0,
        }}

    state.get_telemetry = _some_recalls
    _s1 = state.score_session()
    check(
        "recalls lift the component",
        _s1["recall_activity"] == 1.0,
        str(_s1["recall_activity"]),
    )
    # The regression this fixes: a session that DID recall was still scored as
    # if the layer had never fired, dropping it from 0.35 to 0.15 and below the
    # ``total < 0.2`` extraction gate.
    check(
        "recalls lift the total by the component's weight",
        abs((_s1["total"] - _s0["total"]) - 0.20) < 0.02,
        f"{_s0['total']:.2f} -> {_s1['total']:.2f} (expected +0.20)",
    )
    check(
        "a recalling session clears the 0.2 extraction gate",
        _s1["total"] >= 0.2,
        f"total={_s1['total']} (old implementation scored 0.15 here)",
    )
finally:
    state.exchanges = _saved_ex
    state.session_id = _saved_sid
    state.get_telemetry = _saved_tel

# The weekly report must say why the number is absent rather than just drop it.
_report = state.build_weekly_report()
check("report has a recall block", "recall" in _report, str(sorted(_report)))
check(
    "report carries no usage_rate",
    "usage_rate" not in _report.get("recall", {}),
    str(_report.get("recall")),
)
check(
    "report explains the omission",
    "usage_rate" in (_report.get("notes") or {}),
    str(_report.get("notes")),
)

# ── 15. fact ranking must not be decided by retrieval_count alone ────────
# rank_score multiplied trust by (1 + 0.15*log1p(retrieval_count)). Every fact
# starts at the same trust (INITIAL_TRUST 0.55), so the usage term was the only
# differentiator, and it is unbounded: at ret=18 it multiplies the score by 1.44.
# Consequence on the live store: 18 of 29 facts sat at retrieval_count=0 and
# could not recall THEMSELVES — a query built from a fact's own opening text
# returned the same six hot facts every time. FTS had them at rank 1-3; the
# rerank pushed them to 13-20 or out of the candidate pool entirely.
print("\n15. fact rerank must not starve unretrieved facts")

_starv_root = Path(tempfile.mkdtemp(prefix="icarus_rank_", dir=SCRATCH))
_starv_db = _starv_root / "memory_store.db"
_saved_env = os.environ.get("MEMORY_STORE_PATH")
_saved_home = getattr(state, "HERMES_HOME", None)
os.environ["MEMORY_STORE_PATH"] = str(_starv_db)
state.HERMES_HOME = _starv_root
try:
    # A hot fact that happens to contain one query word, in an unrelated
    # context, and is retrieved constantly.
    _hot = "## Result 缓存策略调整：上限从 512 改为 256，重启后生效。之前讨论过 docker 一次。"
    # A cold fact that is actually about what is being asked.
    _cold = (
        "## Decision docker 部署必须先把 compose 文件里的镜像 tag 固定到具体版本，"
        "不要用 latest，否则回滚时找不到对应镜像。"
    )
    _hot_id = evolution.record_fact(_hot)
    _cold_id = evolution.record_fact(_cold)

    _con = evolution.connect()
    _con.execute("UPDATE facts SET retrieval_count=18 WHERE fact_id=?", (_hot_id,))
    _con.commit()
    _con.close()

    _res = hooks._search_facts("docker 部署", top_k=5)
    _ids = [r["fact_id"] for r in _res]
    check(
        "an exact match outranks a merely-mentioned hot fact",
        bool(_ids) and _ids[0] == _cold_id,
        f"top={_ids[0] if _ids else None}, expected {_cold_id} (order {_ids})",
    )

    # A cold fact must still be able to recall itself even with a crowded
    # candidate pool. Two facts is not enough to show starvation — add filler
    # so the top_k window is genuinely contested.
    for _i in range(8):
        evolution.record_fact(f"## Note 无关条目编号{_i}：缓存策略与部署流程的通用说明。")
    _self = hooks._search_facts(_cold, top_k=3)
    check(
        "an unretrieved fact can recall itself in a crowded pool",
        _cold_id in [r["fact_id"] for r in _self],
        str([r["fact_id"] for r in _self]),
    )

    # The usage term still has to reward genuinely-used facts — just enough to
    # break ties, not enough to override relevance. Both facts match here, so
    # the tie-break should favour the hot one.
    _tie_hot = evolution.record_fact("## Note 缓存策略：LRU 上限 512 条。")
    _tie_cold = evolution.record_fact("## Note 缓存策略：LRU 上限 256 条。")
    _con = evolution.connect()
    _con.execute("UPDATE facts SET retrieval_count=40 WHERE fact_id=?", (_tie_hot,))
    _con.commit()
    _con.close()
    check(
        "usage still breaks ties between equally relevant facts",
        evolution.rank_score(0.55, 40, 0) > evolution.rank_score(0.55, 0, 0),
        str(evolution.rank_score(0.55, 40, 0)),
    )
    # ...but the bonus is bounded: high counts must not multiply the score.
    check(
        "usage bonus is bounded, not multiplicative",
        evolution.rank_score(0.55, 10_000, 0) < 0.55 * 1.5,
        f"score at ret=10000 is {evolution.rank_score(0.55, 10_000, 0):.3f}",
    )
finally:
    if _saved_env is None:
        os.environ.pop("MEMORY_STORE_PATH", None)
    else:
        os.environ["MEMORY_STORE_PATH"] = _saved_env
    state.HERMES_HOME = _saved_home

print("\n" + "=" * 64)
print(f"pass {len(PASS)} / fail {len(FAIL)}")
for label, detail in FAIL:
    print(f"  FAILED: {label}  {detail}")
sys.exit(1 if FAIL else 0)
