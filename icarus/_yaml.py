"""YAML backend resolution for the icarus plugin.

WHY THIS MODULE EXISTS
----------------------
Upstream icarus does ``import yaml`` in ``state._parse_frontmatter_scalar`` and
``parsing.parse_entry``. Inside a real Hermes install that import **fails**:
Hermes runs the plugin under a bundled CPython whose site-packages ships
``ruamel.yaml`` but *not* PyYAML, and Hermes deliberately never installs PyYAML
(it routes every YAML read through its own ``hermes_yaml`` policy module).

Both call sites swallow the ImportError and fall back to "return the raw text".
That silently turns every quoted frontmatter value into a quoted *string*:

    training_value: "high"   ->  '"high"'   (not 'high')
    tier: "hot"              ->  '"hot"'    (not 'hot')
    verified: "true"         ->  '"true"'   (not 'true')

Downstream that is not cosmetic — it breaks four separate features:

* ``state.read_recent`` drops every entry (``tier != "hot"``),
* ``state.build_weekly_report`` reports ``by_type: {'"decision"': 1}`` and files
  every entry under ``training_value: unset``, so ``trainable_estimate`` is 0
  and ``fabric_export``/``fabric_train`` never see a single pair,
* ``state.has_entry_ref`` / ``curate_entry`` compare ``id`` against a quoted
  value and never match,
* ``fabric-retrieve`` scores against quoted type/tier fields.

So resolution order here is: Hermes's own ``hermes_yaml`` (ruamel, always
present in the runtime) -> PyYAML (present in a dev venv) -> a dependency-free
mini parser that at minimum strips quotes and reads flow lists. The mini parser
is a *last* resort; it is never reached in a normal Hermes install.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _backend():
    """Return a module exposing ``safe_load``, or None."""
    try:  # Hermes runtime: ruamel-backed, always importable inside Hermes
        import hermes_yaml

        return hermes_yaml
    except Exception:
        pass
    try:  # dev venv / plain python with PyYAML installed
        import yaml

        return yaml
    except Exception:
        pass
    try:  # last resort: ruamel directly, same policy as hermes_yaml
        from ruamel.yaml import YAML

        class _R:
            @staticmethod
            def safe_load(stream):
                y = YAML(typ="safe", pure=True)
                y.version = (1, 1)
                return y.load(stream)

        return _R
    except Exception:
        return None


_BACKEND = _backend()


def backend_name() -> str:
    if _BACKEND is None:
        return "mini"
    return getattr(_BACKEND, "__name__", type(_BACKEND).__name__)


def _mini_scalar(raw: str) -> Any:
    """Parse one scalar/collection the way YAML would, without a YAML lib.

    Handles the cases fabric frontmatter actually contains: double/single quoted
    strings (with escapes), flow lists, booleans, nulls, bare words and numbers.
    """
    text = raw.strip()
    if not text:
        return ""
    if text.startswith("[") and text.endswith("]"):
        body = text[1:-1].strip()
        if not body:
            return []
        return [_mini_scalar(part) for part in _split_flow(body)]
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        if text[0] == '"':
            try:
                return json.loads(text)
            except Exception:
                return text[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        return text[1:-1].replace("''", "'")
    low = text.lower()
    if low in ("null", "~", ""):
        return None
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _split_flow(body: str):
    """Split a flow sequence body on commas that are not inside quotes."""
    out, buf, quote = [], [], None
    for ch in body:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            continue
        if ch == ",":
            out.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out


def safe_load(document: Any) -> Any:
    """``yaml.safe_load`` with a guaranteed-working backend."""
    if isinstance(document, (bytes, bytearray)):
        document = document.decode("utf-8", "replace")
    if not isinstance(document, str):
        document = str(document)
    if not document.strip():
        return None
    if _BACKEND is not None:
        try:
            return _BACKEND.safe_load(document)
        except Exception:
            pass
    # Whole-document mini path: flat ``key: value`` frontmatter only.
    result = {}
    for line in document.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("- "):
            continue
        if ": " in stripped:
            key, value = stripped.split(": ", 1)
            result[key.strip()] = _mini_scalar(value)
        elif stripped.endswith(":"):
            result[stripped[:-1].strip()] = ""
    return result or None


_KEY_RE_CACHE: dict = {}


def load_scalar(text: str, key: str) -> str:
    """Read one frontmatter scalar, always unquoted, missing -> "".

    This is the single hot path used by ``state._parse_head`` for every entry on
    every recall, so it regexes the line out first and only hands the value to a
    YAML parser — never the whole document.
    """
    pattern = _KEY_RE_CACHE.get(key)
    if pattern is None:
        pattern = re.compile(rf"^{re.escape(key)}:[ \t]*(.*)$", re.MULTILINE)
        _KEY_RE_CACHE[key] = pattern
    match = pattern.search(text)
    if not match:
        return ""
    raw = match.group(1).strip()
    if not raw:
        return ""
    value = None
    if _BACKEND is not None:
        try:
            value = _BACKEND.safe_load(raw)
        except Exception:
            value = None
    if value is None and not raw.strip("'\"") == "":
        value = _mini_scalar(raw)
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
