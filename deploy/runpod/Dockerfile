# NeuroBeat RunPod serverless image: pinned environment + datasets + handler.
# Latency-bound, tiny model -> use a CHEAP GPU (RTX A4000 / 3090). An A100/H100
# wastes money on this workload.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY experiments ./experiments
COPY configs ./configs

# Project + deps (torch cu128 from the configured index) and the RunPod SDK.
RUN uv sync --frozen --no-dev \
 && uv pip install runpod

# Bake in the fast datasets: MIT-BIH (benchmark) and svdb (SVEB source for augmentation).
# wfdb is used directly here to avoid importing torch just to download.
RUN uv run python -c "import wfdb; wfdb.dl_database('mitdb','data/mitdb'); wfdb.dl_database('svdb','data/svdb')"
# INCART (external test set) is large; enable it if you need external validation on the workers:
# RUN uv run python -c "import wfdb; wfdb.dl_database('incartdb','data/incartdb')"

COPY deploy/runpod/handler.py ./handler.py
CMD ["uv", "run", "python", "handler.py"]
