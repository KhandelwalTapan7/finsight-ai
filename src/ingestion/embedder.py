"""
Local embedding model.

Uses fastembed (ONNX runtime) rather than sentence-transformers/torch.
Same all-MiniLM-L6-v2 weights and same 384-dim output, but the runtime
footprint is ~100 MB instead of ~1.5 GB of torch — the difference between
fitting in a small Render instance and being OOM-killed on first query.

Falls back to sentence-transformers if fastembed isn't installed, so
local dev setups that already have it keep working.
"""
from functools import lru_cache

from src.config import settings

_BACKEND = None


@lru_cache(maxsize=1)
def _get_model():
    global _BACKEND
    try:
        from fastembed import TextEmbedding

        _BACKEND = "fastembed"
        return TextEmbedding(model_name=settings.EMBEDDING_MODEL)
    except ImportError:
        from sentence_transformers import SentenceTransformer

        _BACKEND = "sentence-transformers"
        return SentenceTransformer(settings.EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = _get_model()
    if _BACKEND == "fastembed":
        return [v.tolist() for v in model.embed(texts)]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


@lru_cache(maxsize=1)
def embedding_dim() -> int:
    model = _get_model()
    if _BACKEND == "fastembed":
        return len(next(iter(model.embed(["dimension probe"]))))
    return model.get_sentence_embedding_dimension()
