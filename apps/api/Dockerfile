FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ENV=production \
    PORT=7860 \
    HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user

WORKDIR /app

COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir .

COPY models/foshollens_leaf.onnx ./models/foshollens_leaf.onnx
COPY data ./data

RUN mkdir -p /app/storage/images /app/storage/audio /home/user/.cache/huggingface \
    && chown -R user:user /app /home/user/.cache

USER user

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, sys, urllib.request; r = urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '7860') + '/api/v1/health', timeout=4); sys.exit(0 if r.status == 200 else 1)"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
