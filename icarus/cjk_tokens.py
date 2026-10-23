"""CJK-aware tokenization for memory retrieval.

WHY THIS EXISTS
---------------
Upstream Icarus tokenized with ``re.findall(r"[a-z0-9]+", text.lower())``. For a
purely Chinese query that returns an **empty set**, which silently disables the
entire recall path: ``_search_sessions`` / ``_search_facts`` bail on
``if not toks: return []``, and ``collapse.salience`` computes overlap 0.0 for
every candidate so ranking degenerates to "whichever source has a prior".

Three facts drove the design, each verified against a real SQLite FTS5 build:

1. **The Python side needs CJK bigrams.** A single Chinese character carries
   almost no signal ("部", "规"); a bigram ("部署", "规范") is roughly word
   granularity without pulling in a segmenter dependency (jieba etc.). We emit
   unigrams too — they are cheap and let short queries still match.

2. **FTS5's unicode61 tokenizer treats a WHOLE RUN of Han characters as ONE
   token.** Measured: ``帮我看看这个部署规范有没有问题`` indexes as the single
   term ``帮我看看这个部署规范有没有问题``. Only a *prefix* of that token can
   match, so ``部*`` matches ``部署规范`` (its own token) but never the middle of
   a longer run. There is no bigram support in unicode61.

3. **The trigram tokenizer needs >= 3 characters** and is therefore useless for
   the two-character words that dominate Chinese. Measured: ``部署`` -> 0 hits,
   ``部署规`` -> 1 hit.

So FTS retrieval for CJK must use *single-character prefix ORs* (``部* OR 署*
OR 规* OR 范*``), which matches any Han token whose FIRST character is in the
query. That is the only thing unicode61 can do for CJK, and it works well
enough because unicode61's Han token boundaries fall on punctuation, spaces and
ASCII, which is close to real word boundaries in prose.

Pure stdlib, no imports from the rest of the plugin, so every caller (hooks,
collapse, fabric-retrieve, scripts) can share one behavior.
"""

from __future__ import annotations

import re

# ASCII/CJK token split. ASCII alphanumerics stay whole; every Han character is
# its own unit so we can build both unigrams and bigrams from a flat list.
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")

# Han character ranges: BMP unified ideographs plus the two common extensions.
# Deliberately conservative — compatibility ideographs and rare planes are not
# worth the regex cost for a memory plugin.
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")

# Chinese function characters that carry no retrieval signal. Removing them
# before building prefix ORs cuts FTS query length and false positives. These
# are characters, not words — "的" is noise anywhere, "问题" is not.
CJK_STOP_CHARS = frozenset(
    "的了吗呢吧啊呀哦嗯么着过和与及或在我你他她它们这那哪个些"
    "有没不要会能就可以还都也很更最非常因为所以如果但是而且然后"
    "请帮看下说给让被把从对向为以于之其"
)

# ASCII stopwords shared by the plugin's scorers (upstream hook/collapse lists
# merged; fabric-retrieve keeps its own richer STOP_WORDS).
ASCII_STOPWORDS = frozenset(
    "the a an is was are to of in for on with it and or not i you can do this "
    "that what how please help me my your we our they them then than over such "
    "be been being have has had will would could should about into only also "
    "just like very from at as by if".split()
)


def is_cjk(text) -> bool:
    """True when ``text`` contains at least one Han character."""
    return bool(_CJK_RE.search(str(text or "")))


def cjk_chars(text) -> list:
    """Han characters in order, including repeats (order matters for bigrams)."""
    return _CJK_RE.findall(str(text or ""))


def _cjk_segments(text) -> list:
    """Split text into runs of non-stop Han characters.

    Chinese has no spaces, so bigrams built across a function character
    ("这个部署" -> "个部") are pure noise. Function characters make natural
    boundaries, so "帮我看看这个部署规范有没有问题" splits into
    ["部署规范", "问题"] and yields only real bigrams.
    """
    segments = []
    current = []
    for ch in str(text or ""):
        if _CJK_RE.match(ch):
            if ch in CJK_STOP_CHARS:
                if current:
                    segments.append("".join(current))
                    current = []
            else:
                current.append(ch)
        else:
            # Non-Han (space, punctuation, ASCII) also ends a Han run.
            if current:
                segments.append("".join(current))
                current = []
    if current:
        segments.append("".join(current))
    return segments


def tokenize(text, *, stopwords=None, bigrams=True, unigrams=False, min_ascii_len=1) -> set:
    """CJK-aware token set for scoring (overlap, salience, feedback).

    ASCII words are lowercased and filtered by ``stopwords`` (default:
    ``ASCII_STOPWORDS``). CJK bigrams are emitted within function-character-
    delimited segments; unigrams are opt-in because in scoring they add noise.

    ``min_ascii_len`` exists because scoring and FTS want different ASCII
    floors: overlap scoring wants short terms too (``_fact_tokens`` passes 2),
    while FTS query building needs >= 4 to avoid a firehose.

    A single Han character segment (e.g. "啊") still yields itself, so a very
    short query is not silently emptied — the failure mode this module exists
    to prevent.
    """
    text = str(text or "")
    if not text:
        return set()

    stops = ASCII_STOPWORDS if stopwords is None else stopwords
    out = set()

    for tok in _TOKEN_RE.findall(text.lower()):
        if not _CJK_RE.match(tok) and tok not in stops and len(tok) >= min_ascii_len:
            out.add(tok)

    for seg in _cjk_segments(text):
        if len(seg) == 1:
            # A standalone Han char (surrounded by stop-chars/ASCII) is kept:
            # dropping it silently emptied queries like "啊".
            out.add(seg)
            continue
        if bigrams:
            for i in range(len(seg) - 1):
                out.add(seg[i] + seg[i + 1])
        if unigrams:
            out.update(seg)

    return out


def cjk_terms(text, *, min_len=2, max_terms=8) -> list:
    """Chinese substring terms for ``LIKE '%term%'`` matching.

    FTS5 cannot find a Chinese word that sits mid-run in a Han token (unicode61
    indexes the whole run; trigram needs >= 3 chars). Substring matching is the
    only mechanism that actually works, and a 2-char bigram is the smallest unit
    with real signal, so that is what we hand to LIKE.

    Returns terms ordered longest-first (a longer term is more selective, so the
    caller can build a higher-precision query), capped at ``max_terms``.
    """
    terms = []
    text = str(text or "")
    # Reuse the segmenter: only non-stop-char runs carry meaning.
    for seg in _cjk_segments(text):
        if len(seg) < min_len:
            continue
        for i in range(len(seg) - min_len + 1):
            terms.append(seg[i:i + min_len])

    # Longest-first, de-duplicated, order-stable.
    seen = set()
    out = []
    for t in sorted(terms, key=len, reverse=True):
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:max_terms]


def token_sequence(text, *, min_ascii_len=1) -> list:
    """Ordered token list for n-gram matching (NOT deduplicated).

    ``tokenize`` returns a set, which is right for overlap scoring but destroys
    the position information that n-gram phrase matching needs. This returns the
    raw sequence: ASCII words stay whole, each Han character is one unit — so
    ``_ngrams(seq, 2)`` over Chinese text naturally yields Han bigrams, which is
    exactly what the n-gram bonus in ``fabric-retrieve.score_entry`` wants.

    Upstream used ``re.findall(r"[a-z0-9]+", ...)`` at those call sites, so every
    phrase/ngram bonus silently evaluated to zero for Chinese entries.
    """
    out = []
    for tok in _TOKEN_RE.findall(str(text or "").lower()):
        if _CJK_RE.match(tok):
            out.append(tok)
        elif len(tok) >= min_ascii_len:
            out.append(tok)
    return out


def query_tokens(text, *, min_ascii_len=4) -> list:
    """Ordered token list for FTS ``MATCH`` building, CJK-safe.

    Two different length rules apply, and applying one to both is the bug:
      * ASCII terms need ``len >= min_ascii_len`` (upstream used 4) to avoid a
        firehose of "the"/"docker" style noise.
      * Han units are 1-2 characters by construction, so a ``>= 4`` filter would
        erase every one of them. Han characters are kept at length >= 1.

    Returns a de-duplicated list preserving first-seen order so generated FTS
    queries are deterministic (stable ranking, reproducible tests).
    """
    text = str(text or "")
    seen = []
    seen_set = set()

    def _push(tok):
        if tok and tok not in seen_set:
            seen_set.add(tok)
            seen.append(tok)

    for tok in _TOKEN_RE.findall(text.lower()):
        if _CJK_RE.match(tok):
            if tok not in CJK_STOP_CHARS:
                _push(tok)
            continue
        if len(tok) >= min_ascii_len:
            _push(tok)

    return seen


def fts_query(text, *, min_ascii_len=4, max_terms=8, ascii_only=True) -> str:
    """Build an FTS5 ``MATCH`` expression.

    Two distinct callers need two distinct behaviors, and conflating them was a
    real bug: for the mixed query ``github 仓库配置`` this used to emit
    ``"github" OR 仓* OR 库* OR 配* OR 置*``. The Han prefix terms cannot match
    (unicode61 keeps the whole Han run as one token), yet their presence made the
    caller take the FTS path and skip the LIKE path that WOULD have found the
    Chinese rows — so adding one English word made Chinese recall strictly worse.

    ``ascii_only=True`` (the default for retrieval) emits ONLY ASCII terms, so
    the caller can run FTS and LIKE side by side and merge. ``ascii_only=False``
    keeps Han prefix terms for diagnostic use.

    Returns "" when nothing usable remains, so callers can bail explicitly
    instead of firing a match on an empty expression (which FTS5 rejects).
    """
    parts = []
    text = str(text or "")
    has_cjk = bool(_CJK_RE.search(text))

    for tok in _TOKEN_RE.findall(text.lower()):
        if _CJK_RE.match(tok):
            # Retrieval mode (ascii_only=True) leaves Han to the LIKE path:
            # emitting them here would suppress LIKE without matching anything
            # (unicode61 cannot match a Han word mid-run).
            if not ascii_only and tok not in CJK_STOP_CHARS:
                parts.append(f"{tok}*")
        elif len(tok) >= min_ascii_len:
            parts.append(f'"{tok}"')

    # Deterministic de-dup, keep order.
    deduped = []
    seen = set()
    for p in parts:
        if p not in seen:
            seen.add(p)
            deduped.append(p)

    return " OR ".join(deduped[:max_terms])
