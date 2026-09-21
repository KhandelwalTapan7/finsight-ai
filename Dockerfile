FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image. Without this, the first query
# after every deploy (and every cold start on an ephemeral filesystem)
# stalls while it downloads ~90 MB from HuggingFace.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')"

COPY . .

# Render injects $PORT; shell form so it expands at runtime.
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
