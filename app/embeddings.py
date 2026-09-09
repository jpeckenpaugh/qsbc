"""MiniLM sentence-embedding encoder, loaded lazily and shared across calls.

Uses sentence-transformers' all-MiniLM-L6-v2: a small, CPU-friendly model
(~80MB) tuned for semantic sentence similarity. Lazy-loading keeps the API
snappy and avoids paying model-load cost on startup.
"""

import logging

log = logging.getLogger("claims.embeddings")

_model = None
_model_checked = False
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_model():
    global _model, _model_checked
    if not _model_checked:
        _model_checked = True
        try:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(MODEL_NAME)
            log.info("loaded MiniLM model %s", MODEL_NAME)
        except Exception as e:  # pragma: no cover - depends on env
            log.error("could not load MiniLM: %s", e)
            _model = None
    return _model


def encode_batch(texts):
    """Return a list of embedding vectors (lists of floats) for `texts`, or
    None if the model is unavailable."""
    model = _get_model()
    if model is None:
        return None
    vecs = model.encode(list(texts), normalize_embeddings=True, batch_size=64)
    return [v.tolist() for v in vecs]


def available():
    return _get_model() is not None