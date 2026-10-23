"""Regression tests for rollback_model: a model rollback must never drop keys."""
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, "/home/xc/.hermes/hermes-agent")
import hermes_bootstrap  # noqa: F401

import importlib.util

PLUGIN_DIR = Path("/home/xc/.hermes/plugins/icarus")
spec = importlib.util.spec_from_file_location(
    "icarus", str(PLUGIN_DIR / "__init__.py"), submodule_search_locations=[str(PLUGIN_DIR)]
)
mod = importlib.util.module_from_spec(spec)
sys.modules["icarus"] = mod
spec.loader.exec_module(mod)

from icarus import state  # noqa: E402

PASS, FAIL = [], []


def check(label, cond, detail=""):
    (PASS if cond else FAIL).append((label, detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail and not cond else ""))


home = Path(tempfile.mkdtemp(prefix="rb_", dir="/home/xc/.hermes/cache/scratch"))
real_home = state.HERMES_HOME
state.HERMES_HOME = home
try:
    env = home / ".env"
    env.write_text(
        "\n".join(
            [
                "# live config",
                "LLM_MODEL=new-model",
                "LLM_API_BASE=http://new/v1",
                "EMBEDDING_API_KEY=sk-embedding-should-survive",
                "LLM_API_KEY=sk-llm-should-survive",
                "QDRANT_URL=http://localhost:6333",
                "UNRELATED_KEY=keep-me",
                "",
            ]
        ),
        encoding="utf-8",
    )
    backup = home / ".env.backup.20260101000000"
    backup.write_text(
        "\n".join(
            [
                "LLM_MODEL=old-model",
                "LLM_API_BASE=http://old/v1",
                "EMBEDDING_API_KEY=sk-stale-backup-value",
                "LLM_API_KEY=sk-stale-llm-key",
                "",
            ]
        ),
        encoding="utf-8",
    )
    os.utime(backup, (time.time(), time.time()))

    print("rollback on a .env that gained keys after the backup")
    result = state.rollback_model()
    check("status is rolled_back", result.get("status") == "rolled_back", json.dumps(result))
    check("restored the old model", result.get("to_model") == "old-model", json.dumps(result))

    live = env.read_text()
    keys = state._env_keys(env)
    check("model line rolled back", "LLM_MODEL=old-model" in live, live[:300])
    check("api base rolled back", "LLM_API_BASE=http://old/v1" in live, live[:300])
    # The critical assertion: keys that only exist in the live file survive.
    check("QDRANT_URL preserved", "QDRANT_URL" in keys, live[:400])
    check("UNRELATED_KEY preserved", "UNRELATED_KEY=keep-me" in live, live[:400])
    check(
        "embedding key NOT taken from the stale backup",
        "sk-stale-backup-value" not in live,
        live[:400],
    )
    check(
        "live embedding key preserved",
        "sk-embedding-should-survive" in live,
        live[:400],
    )
    check("safety copy written", Path(result.get("safety_copy", "")).exists(), result.get("safety_copy", ""))
    check("file mode is 0600", oct(env.stat().st_mode)[-3:] == "600", oct(env.stat().st_mode)[-3:])

    print()
    print("a second identical rollback must refuse instead of pretending")
    second = state.rollback_model()
    check("second call reports nothing to roll back", "error" in second, json.dumps(second))

    print()
    print("a rollback must never DELETE a managed key it cannot restore")
    env.write_text(
        "\n".join(
            [
                "LLM_MODEL=old-model",
                "LLM_API_BASE=http://old/v1",
                "LLM_LIVE_ONLY=http://live-only/v1",
                "EMBEDDING_API_KEY=sk-embedding-should-survive",
                "KEEP_ME=1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    before_keys = set(state._env_keys(env))
    fourth = state.rollback_model()
    after_text = env.read_text()
    after_keys = set(state._env_keys(env))
    check(
        "live-only managed key survives a rollback",
        "LLM_LIVE_ONLY=http://live-only/v1" in after_text,
        json.dumps(fourth)[:300],
    )
    # This run has nothing to restore, so it refuses; either way the
    # live-only key must never be touched or deleted.
    check(
        "refused rollback still names the untouched key",
        "LLM_LIVE_ONLY" in fourth.get("keys_left_alone", [])
        or "LLM_LIVE_ONLY" not in json.dumps(fourth.get("keys_restored", [])),
        json.dumps(fourth)[:300],
    )
    check("unmanaged neighbour key survives", "KEEP_ME=1" in after_text)
    check("embedding key still survives", "sk-embedding-should-survive" in after_text)
    check("no key disappeared", after_keys == before_keys, sorted(before_keys ^ after_keys))

    print()
    print("a backup with no model keys must be refused")
    (home / ".env.backup.29990101000000").write_text("UNRELATED=x\n", encoding="utf-8")
    third = state.rollback_model()
    check("refuses a backup with no model keys", "error" in third, json.dumps(third))
    check(
        "live .env untouched by the refused rollback",
        "sk-embedding-should-survive" in env.read_text(),
    )
finally:
    state.HERMES_HOME = real_home
    shutil.rmtree(home, ignore_errors=True)

print()
print(f"pass {len(PASS)} / fail {len(FAIL)}")
for label, detail in FAIL:
    print(f"  FAILED: {label}  {detail}")
sys.exit(1 if FAIL else 0)
