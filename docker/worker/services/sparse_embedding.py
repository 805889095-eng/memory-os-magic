"""
Pure-Python BM25 tf 编码，无模型无网络。

No local model is loaded and no network call is made: this module simply
forwards to the dependency-free ``services.bm25_tf.bm25_tf`` implementation.
"""
import logging

from services.bm25_tf import bm25_tf

logger = logging.getLogger("cognitive-worker.sparse_embedding")


def get_sparse_embedding(text: str) -> dict:
    """
    Generates a BM25 sparse embedding in pure Python.

    Returns a Qdrant-compatible dict: {"indices": [...], "values": [...]}.
    Only the BM25 tf term is produced; ``idf`` is applied server-side by
    Qdrant (``Modifier.IDF``). No model, no network, no heavy dependency.
    """
    return bm25_tf(text)