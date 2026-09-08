"""
Local embedding model — runs on CPU, no API key, no per-call cost.
Loaded once per process (module-level singleton) since loading the model
is the slow part, not encoding.
"""
from functools import lru_cache
from sentence_transformers import SentenceTransformer

from src.config import settings


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


def embedding_dim() -> int:
    return _get_model().get_sentence_embedding_dimension()
