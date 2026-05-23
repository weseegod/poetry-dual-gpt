# ── Salad GPU Container: Qwen2.5-1.5B QLoRA training ──
# Build:  docker build -t poetry-trainer .
# Run locally: docker run --gpus all -e HF_TOKEN=... poetry-trainer --stage 1

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Application ──
WORKDIR /app

# Training script
COPY train.py /app/

# Corpus files (baked into image — 125MB + 77MB)
COPY corpus/ /app/corpus/

# HuggingFace cache dir (optional — set HF_HOME to persist across runs)
ENV HF_HOME=/app/.cache/huggingface

# Entrypoint script: handles GCS credentials at runtime
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--stage", "1"]
