"""Regression tests for the icarus plugin fixes.

Every case here corresponds to a bug that shipped silently: the code ran, said
nothing, and produced wrong results. Run under the real Hermes runtime
interpreter, because the yaml-shadowing bug only reproduces there.

    <hermes runtime python> -I test_icarus_fixes.py
"""

import json
import os
import shutil
import sys
import tempfile
import traceback
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

print("\n" + "=" * 64)
print(f"pass {len(PASS)} / fail {len(FAIL)}")
for label, detail in FAIL:
    print(f"  FAILED: {label}  {detail}")
sys.exit(1 if FAIL else 0)
