FROM python:3.12-slim

WORKDIR /app

# System deps for building wheels (tokenizers/sentence-transformers need a compiler
# for some optional pieces; keep the layer small otherwise).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY ui/ ./ui/

# The embedding model (~90 MB) downloads on first run and is cached here.
ENV HF_HOME=/app/.cache/huggingface

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
