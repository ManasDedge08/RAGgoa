# API image for Render. The corpus and index are built at image build time so
# the container starts fast and every deploy pins the same artefacts.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/data/cache \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch: the GPU wheel is several gigabytes and useless on this plan.
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY rag ./rag
COPY scripts ./scripts
COPY benchmark.py ./

# A smaller corpus slice than the local default keeps the container inside a
# 2 GB instance. Override at build time with --build-arg.
ARG N_QUERY_IDS=800
ENV RAG_N_QUERY_IDS=${N_QUERY_IDS}

RUN python scripts/prepare_data.py \
    && python -m rag.index.build \
    && python -m rag.index.build_baseline \
    && python scripts/calibrate.py 160 \
    # The parquet downloads are only needed to build the index, not to serve it.
    && rm -rf /app/data/cache/datasets*

# The encoder is baked into the image by the build above; going to the hub at
# boot would add a network round trip to every cold start.
ENV HF_HUB_OFFLINE=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "rag.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
