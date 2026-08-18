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

# Languages are pinned rather than left to the registry default, so the image
# cannot silently change shape when that default moves. Build and runtime read
# the same list: an index built for eleven languages served by a process that
# thinks there are four would answer from vectors it never loaded.
ENV RAG_LANGUAGES=eng_Latn,hin_Deva,ben_Beng,tam_Taml,tel_Telu,mar_Deva,guj_Gujr,kan_Knda,mal_Mlym,pan_Guru,ori_Orya

# The corpus slice is what decides whether this fits its instance. Eleven
# languages at 1,200 query ids measured 1.29 GB RSS locally; the free instance
# caps at 512 MB, and every index is memory-mapped, so the resident cost scales
# with this number. 250 is the largest slice expected to fit. Override at build
# time with --build-arg N_QUERY_IDS=...
ARG N_QUERY_IDS=250
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
