"""
Central config. Every other module reads settings from here — never from
os.environ directly — so the local-vs-AWS switch stays in one place.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # LLM
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    # Groq's free-tier lineup changes fairly often — if either of these
    # starts throwing "model_decommissioned", check the current list at
    # https://console.groq.com/docs/models and update .env accordingly.
    GROQ_TEXT_MODEL: str = os.getenv("GROQ_TEXT_MODEL", "openai/gpt-oss-120b")
    GROQ_VISION_MODEL: str = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")

    # Storage mode
    STORAGE_MODE: str = os.getenv("STORAGE_MODE", "local")  # "local" | "aws"
    LOCAL_UPLOAD_DIR: Path = Path(os.getenv("LOCAL_UPLOAD_DIR", "./data/uploads"))
    LOCAL_QDRANT_PATH: Path = Path(os.getenv("LOCAL_QDRANT_PATH", "./data/qdrant"))

    # AWS (only used when STORAGE_MODE == "aws")
    AWS_REGION: str = os.getenv("AWS_REGION", "ap-south-1")
    S3_BUCKET: str = os.getenv("S3_BUCKET", "")
    DYNAMODB_TABLE: str = os.getenv("DYNAMODB_TABLE", "")

    # MLflow
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

    # Session hygiene
    SESSION_TTL_HOURS: int = int(os.getenv("SESSION_TTL_HOURS", "24"))

    # Embedding model (local, free, no API calls)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Local telemetry db
    TELEMETRY_DB_PATH: Path = Path("./data/telemetry.db")

    COLLECTION_NAME: str = "finsight_docs"


settings = Settings()

# Make sure local dirs exist on import
settings.LOCAL_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.LOCAL_QDRANT_PATH.parent.mkdir(parents=True, exist_ok=True)
settings.TELEMETRY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
