"""
Central config. Every other module reads settings from here — never from
os.environ directly — so the local-vs-hosted switch stays in one place.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- Branding / domain (nothing here is finance-specific) ---
    APP_NAME: str = os.getenv("APP_NAME", "DocSight AI")
    # Optional nudge for the LLM, e.g. "legal contracts" or "research papers".
    # Leave empty for a fully general-purpose document assistant.
    DOMAIN_HINT: str = os.getenv("DOMAIN_HINT", "")

    # LLM
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_TEXT_MODEL: str = os.getenv("GROQ_TEXT_MODEL", "openai/gpt-oss-120b")
    GROQ_VISION_MODEL: str = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")

    # Storage
    STORAGE_MODE: str = os.getenv("STORAGE_MODE", "local")  # "local" | "aws"
    LOCAL_UPLOAD_DIR: Path = Path(os.getenv("LOCAL_UPLOAD_DIR", "./data/uploads"))
    LOCAL_QDRANT_PATH: Path = Path(os.getenv("LOCAL_QDRANT_PATH", "./data/qdrant"))

    # Qdrant Cloud / self-hosted. If QDRANT_URL is set it wins over the
    # embedded on-disk mode — required on Render, where the filesystem is
    # ephemeral and embedded Qdrant takes an exclusive single-process lock.
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")

    # Shared corpus: OFF by default. On a public deployment every visitor
    # shares one backend, so a "shared" pool means strangers' documents
    # leak into each other's answers. Turn on only for a curated corpus
    # you ingested yourself.
    ENABLE_SHARED_CORPUS: bool = os.getenv("ENABLE_SHARED_CORPUS", "false").lower() == "true"

    # Uploads
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "25"))
    MAX_PAGES_VISION: int = int(os.getenv("MAX_PAGES_VISION", "20"))

    # AWS (only used when STORAGE_MODE == "aws")
    AWS_REGION: str = os.getenv("AWS_REGION", "ap-south-1")
    S3_BUCKET: str = os.getenv("S3_BUCKET", "")
    DYNAMODB_TABLE: str = os.getenv("DYNAMODB_TABLE", "")

    # MLflow
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

    SESSION_TTL_HOURS: int = int(os.getenv("SESSION_TTL_HOURS", "24"))

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    TELEMETRY_DB_PATH: Path = Path(os.getenv("TELEMETRY_DB_PATH", "./data/telemetry.db"))

    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "documents")


settings = Settings()

settings.LOCAL_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.LOCAL_QDRANT_PATH.parent.mkdir(parents=True, exist_ok=True)
settings.TELEMETRY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
