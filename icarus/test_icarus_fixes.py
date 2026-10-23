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

sys.path.insert(0, "/home/xc/.hermes/hermes-agent")
import hermes_bootstrap  # noqa: F401

import importlib.util

PLUGIN_DIR = Path("/home/xc/.hermes/plugins/icarus")
_spec = importlib.util.spec_from_file_location(
    "icarus", str(PLUGIN_DIR / "__init__.py"), submodule_search_locations=[str(PLUGIN_DIR)]
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["icarus"] = _mod
_spec.loader.exec_module(_mod)

from icarus import _yaml, evolution, parsing, state  # noqa: E402

SCRATCH = Path("/home/xc/.hermes/cache/scratch")
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

print("\n" + "=" * 64)
print(f"pass {len(PASS)} / fail {len(FAIL)}")
for label, detail in FAIL:
    print(f"  FAILED: {label}  {detail}")
sys.exit(1 if FAIL else 0)
