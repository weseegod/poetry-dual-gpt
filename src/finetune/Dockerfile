# PoetryDuel-GPT v5 QLoRA Training — containerized for local/Salad GPU
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY src/finetune/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Corpus files — pre-generated from data/poems_dataset_clean.csv via src/preprocess.py
COPY src/finetune/corpus/poetry_corpus.txt src/finetune/corpus/corpus_luc_bat.txt /app/corpus/

# Training code
COPY src/finetune/train.py /app/
COPY src/finetune/entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/app/entrypoint.sh"]
