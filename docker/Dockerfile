# RAG Trustworthiness -- Gradio demo container.
#
# Build-cache-friendly layer order: dependency manifests are copied and installed
# before the source tree, so editing app/source code does not invalidate the (slow)
# pip install layer.
FROM python:3.12-slim

# Faster, quieter pip; no .pyc cache bloating the image.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps needed by torch/faiss/pdfplumber wheels at runtime (kept minimal).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- dependency layer (cached unless requirements change) ---------------------------
COPY pyproject.toml ./pyproject.toml
COPY app/requirements.txt ./app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

# --- source layer ---------------------------------------------------------------------
COPY src ./src
RUN pip install --no-cache-dir --no-deps -e .

COPY app ./app

# Non-root user: create it after installs so pip caches/build artefacts are owned
# correctly, and the runtime process has no write access outside /app.
RUN useradd --create-home --uid 1000 ragtrust \
    && chown -R ragtrust:ragtrust /app
USER ragtrust

EXPOSE 7860
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=4)" || exit 1

CMD ["python", "app/app.py"]
