# API container. Works unchanged on Railway, Render, Fly.io and plain Docker.
#
#   docker build -t quantlab-api .
#   docker run -p 8000:8000 -v quantlab-data:/data quantlab-api
#
# The volume matters: without it the SQLite file is wiped on every redeploy and
# your paper-trading history goes with it.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    QUANTLAB_DATA_DIR=/data

WORKDIR /app

# Dependencies first so edits to source don't invalidate the layer cache.
COPY pyproject.toml README.md ./
COPY quantlab ./quantlab
RUN pip install --no-cache-dir ".[api]"

# Persisted state lives here. Mount a volume at /data in production.
RUN mkdir -p /data

# Run unprivileged.
RUN useradd --create-home --uid 10001 quant && chown -R quant:quant /app /data
USER quant

EXPOSE 8000

# Single-quoted inside, double-quoted outside — no nested escaping to get wrong.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT','8000') + '/health', timeout=4)"

# $PORT is injected by most PaaS platforms; default to 8000 locally.
CMD ["sh", "-c", "uvicorn quantlab.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
