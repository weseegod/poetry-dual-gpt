#!/bin/bash
# ──────────────────────────────────────────────
# PoetryDuel-GPT v5 QLoRA — one script to rule them all
#
#   bash run.sh setup     One-time: corpus + docker build
#   bash run.sh test      Smoke test (100 steps)
#   bash run.sh train     Full training
# ──────────────────────────────────────────────
set -e
cd "$(dirname "$0")"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
CMD="${1:-help}"
IMAGE="poetry-trainer:v5"

# ── Load secrets from any available source ──
[ -f src/finetune/.env ] && export $(grep -v '^#' src/finetune/.env | xargs) 2>/dev/null
[ -f ~/.bashrc ] && source ~/.bashrc 2>/dev/null
HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN in ~/.bashrc or src/finetune/.env: export HF_TOKEN=hf_xxx}"

# ═══════════════════════════════════════════════
usage() {
    echo "Usage: bash run.sh <command>"
    echo ""
    echo "  setup    Generate corpus + build Docker image (run once)"
    echo "  test     Smoke test — 100 steps, verify everything works"
    echo "  train    Full training (Stage 1, 5000 steps, ~2.5h)"
    echo ""
    echo "Env: HF_TOKEN required. Optional: STAGE=2, AUTO_CHAIN=1"
    exit 0
}

# ═══════════════════════════════════════════════
do_setup() {
    echo "🔧  Setup..."
    command -v docker >/dev/null 2>&1 || { echo "${RED}❌ Install Docker first${NC}"; exit 1; }

    CORPUS_DIR="src/finetune/corpus"
    CORPUS="$CORPUS_DIR/poetry_corpus.txt"

    # Decompress if .gz exists but .txt missing
    if [ ! -f "$CORPUS" ] && [ -f "$CORPUS.gz" ]; then
        echo "   📦  Decompressing corpus..."
        gunzip -k "$CORPUS_DIR/poetry_corpus.txt.gz"
        gunzip -k "$CORPUS_DIR/corpus_luc_bat.txt.gz"
    fi

    # Generate corpus from CSV if completely missing
    if [ ! -f "$CORPUS" ]; then
        if [ ! -f data/poems_dataset_clean.csv ]; then
            echo "   ${YELLOW}⚠️  data/poems_dataset_clean.csv not found${NC}"
            echo "   Copy it from your Mac: scp data/poems_dataset_clean.csv user@ubuntu:poetry-dual-gpt/data/"
            exit 1
        fi
        echo "   📖  Generating corpus..."
        pip install -q pandas 2>/dev/null || true
        python3 src/preprocess.py --csv data/poems_dataset_clean.csv --output "$CORPUS"
        grep '\[LUC_BAT\]' "$CORPUS" > src/finetune/corpus/corpus_luc_bat.txt
    fi
    echo "   ✅  Corpus: $(wc -l < "$CORPUS") pairs"

    # Build
    echo "   🔨  Building Docker image..."
    docker build -t "$IMAGE" -f src/finetune/Dockerfile .
    echo "   ${GREEN}✅  Image built${NC}"
    echo ""
    echo "   Next: bash run.sh test"
}

# ═══════════════════════════════════════════════
do_test() {
    echo "🧪  Smoke test — 100 steps..."
    mkdir -p test_output
    docker run --rm --gpus all \
        -e HF_TOKEN="$HF_TOKEN" \
        -e HF_HOME="/root/.cache/huggingface" \
        -e TRAIN_MAX_STEPS=100 \
        -e TRAIN_BATCH_SIZE=2 \
        -e TRAIN_GRAD_ACCUM=4 \
        -e S3_BUCKET="${S3_BUCKET:-}" \
        -e S3_ENDPOINT="${S3_ENDPOINT:-}" \
        -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-}" \
        -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-}" \
        -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-}" \
        -v "$(pwd)/test_output:/app/checkpoints" \
        -v "${HF_CACHE_DIR:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
        "$IMAGE" 2>&1 | tee test_output/log.txt

    if grep -q "CUDA out of memory\|RuntimeError" test_output/log.txt 2>/dev/null; then
        echo "${RED}❌  CUDA errors — check test_output/log.txt${NC}"; exit 1
    fi
    echo "${GREEN}✅  Smoke test passed${NC}"
    echo "   Next: bash run.sh train"
}

# ═══════════════════════════════════════════════
do_train() {
    echo "🔥  Full training..."
    mkdir -p checkpoints
    STAGE="${STAGE:-1}"
    echo "   Stage: $STAGE | Data: $(wc -l < data/poetry_corpus.txt) pairs"

    docker run --rm --gpus all \
        -e HF_TOKEN="$HF_TOKEN" \
        -e HF_HOME="/root/.cache/huggingface" \
        -e STAGE="$STAGE" \
        -e AUTO_CHAIN_STAGES="${AUTO_CHAIN:-0}" \
        -e TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}" \
        -e TRAIN_GRAD_ACCUM="${TRAIN_GRAD_ACCUM:-4}" \
        -e TRAIN_BLOCK_SIZE="${TRAIN_BLOCK_SIZE:-256}" \
        -e S3_BUCKET="${S3_BUCKET:-}" \
        -e S3_ENDPOINT="${S3_ENDPOINT:-}" \
        -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-}" \
        -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-}" \
        -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-}" \
        -v "$(pwd)/checkpoints:/app/checkpoints" \
        -v "${HF_CACHE_DIR:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
        "$IMAGE"

    echo "${GREEN}✅  Done — checkpoints/$(ls checkpoints/ | head -1)${NC}"
}

# ═══════════════════════════════════════════════
case "$CMD" in
    setup)  do_setup ;;
    test)   do_test ;;
    train)  do_train ;;
    *)      usage ;;
esac
