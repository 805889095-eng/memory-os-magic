"""
Pure-Python BM25 sparse encoder -- no model, no network, no heavy dependencies.

Emits Qdrant-compatible sparse vectors::

    {"indices": [int, ...], "values": [float, ...]}

Semantics (unified refactor contract, section 2):

* Only the BM25 **tf saturation** term is emitted here. The ``idf`` factor is
  applied **server-side** by Qdrant via ``Modifier.IDF`` and is never computed
  on the client.
* Per-token weight::

      w = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avg_len))

  where ``tf`` is the term frequency of the token inside this text and ``dl``
  is the total number of tokens in this text.
* Tokenisation: ``text.lower()`` then
  ``re.findall(r"[a-z0-9]+|[\\u4e00-\\u9fff]", text)`` -- ASCII words/digits stay
  whole, each CJK ideograph becomes its own token. No stemming, deliberately,
  so that no model / heavyweight dependency is required.
* token -> index: ``zlib.crc32(token.encode("utf-8"))``, a stable uint32, so
  ingest and query generate identical indices across processes and languages.
* Repeated tokens collapse into a single index with their tf accumulated; the
  returned ``indices`` are de-duplicated and sorted ascending.
* Empty text (or text with no alphanumeric/CJK token) returns
  ``{"indices": [], "values": []}`` and never raises.
"""

import re
import zlib

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


def _tokenize(text: str) -> list:
    """Lower-case, then split into ASCII word/digit tokens and single CJK chars."""
    return _TOKEN_RE.findall((text or "").lower())


def _token_index(token: str) -> int:
    """Stable unsigned 32-bit index for a token (cross-process / cross-language)."""
    return zlib.crc32(token.encode("utf-8")) & 0xFFFFFFFF


def bm25_tf(
    text: str,
    *,
    k1: float = 1.2,
    b: float = 0.75,
    avg_len: float = 256.0,
) -> dict:
    """Return a Qdrant-compatible BM25 tf vector: {"indices": [...], "values": [...]}.

    Only the tf saturation term is computed; idf is applied by Qdrant.
    """
    tokens = _tokenize(text)
    if not tokens:
        return {"indices": [], "values": []}

    dl = len(tokens)

    tf_map = {}
    for token in tokens:
        idx = _token_index(token)
        tf_map[idx] = tf_map.get(idx, 0) + 1

    denom_scale = k1 * (1.0 - b + b * dl / avg_len)
    indices = sorted(tf_map)
    values = [tf_map[idx] * (k1 + 1.0) / (tf_map[idx] + denom_scale) for idx in indices]
    return {"indices": indices, "values": values}
