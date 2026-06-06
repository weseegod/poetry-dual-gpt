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
HF_TOKEN="${HF_TOKEN:?Set: export HF_TOKEN=hf_xxx}"

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

    # Data deps for preprocess.py
    pip install -q pandas 2>/dev/null || true

    # Generate corpus if missing
    if [ ! -f data/poetry_corpus.txt ]; then
        echo "   📖  Generating corpus..."
        python3 src/preprocess.py --csv data/poems_dataset_clean.csv --output data/poetry_corpus.txt
        grep '\[LUC_BAT\]' data/poetry_corpus.txt > data/corpus_luc_bat.txt
    fi
    echo "   ✅  Corpus: $(wc -l < data/poetry_corpus.txt) pairs"

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
        -e TRAIN_MAX_STEPS=100 \
        -e TRAIN_BATCH_SIZE=2 \
        -e TRAIN_GRAD_ACCUM=4 \
        -v "$(pwd)/test_output:/app/checkpoints" \
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
        -e STAGE="$STAGE" \
        -e AUTO_CHAIN_STAGES="${AUTO_CHAIN:-0}" \
        -v "$(pwd)/checkpoints:/app/checkpoints" \
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
