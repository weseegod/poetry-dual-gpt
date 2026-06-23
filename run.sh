#!/bin/bash
# ──────────────────────────────────────────────
# PoetryDuel-GPT v5.1 — Instruction Fine-Tuning
#
#   bash run.sh setup    Pre-download model + build Docker
#   bash run.sh test     Smoke test (100 steps)
#   bash run.sh train    Full training
# ──────────────────────────────────────────────
set -e
cd "$(dirname "$0")"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
CMD="${1:-help}"; shift 2>/dev/null || true
IMAGE="poetry-instruct:v5.1"

# ── Load HF token ──
[ -f src/finetune/.env ] && export $(grep -v '^#' src/finetune/.env | xargs) 2>/dev/null
[ -f ~/.bashrc ] && source ~/.bashrc 2>/dev/null
if [ -z "$HF_TOKEN" ]; then
    echo "${YELLOW}⚠️  HF_TOKEN not set — set in src/finetune/.env or ~/.bashrc${NC}"
    echo "   export HF_TOKEN=hf_xxx"
fi

# ── Parse flags ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch-size)       TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --grad-accum)       TRAIN_GRAD_ACCUM="$2"; shift 2 ;;
        --max-steps)        TRAIN_MAX_STEPS="$2"; shift 2 ;;
        --max-seq-length)   TRAIN_MAX_SEQ_LENGTH="$2"; shift 2 ;;
        --resume)           RESUME_FROM="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

HF_CACHE="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
CORPUS_DIR="src/finetune/corpus"
CORPUS="$CORPUS_DIR/poetry_corpus.txt"

# ═══════════════════════════════════════════════
usage() {
    echo "Usage: bash run.sh <command> [flags]"
    echo ""
    echo "Commands:"
    echo "  setup    Pre-download model + generate data + build Docker"
    echo "  test     Smoke test (100 steps)"
    echo "  train    Full training (5000 steps)"
    echo ""
    echo "Flags:"
    echo "  --batch-size N        (default: 4)"
    echo "  --grad-accum N        (default: 4)"
    echo "  --max-steps N         (default: 5000, test: 100)"
    echo "  --max-seq-length N    (default: 256)"
    echo "  --resume PATH         Resume from checkpoint dir"
    exit 0
}

# ═══════════════════════════════════════════════
do_setup() {
    echo "🔧  Setup v5.1 Instruct..."
    command -v docker >/dev/null 2>&1 || { echo "${RED}❌ Install Docker first${NC}"; exit 1; }

    # ── Decompress corpus if needed ──
    if [ ! -f "$CORPUS" ] && [ -f "$CORPUS.gz" ]; then
        gunzip -k "$CORPUS_DIR/poetry_corpus.txt.gz"
        gunzip -k "$CORPUS_DIR/corpus_luc_bat.txt.gz"
    fi

    # ── Generate instruction JSONL from corpus ──
    TRAIN_JSONL="data/instruct_train.jsonl"
    if [ ! -f "$TRAIN_JSONL" ]; then
        echo "   📝  Generating instruction data..."
        python3 src/finetune/preprocess_instruct.py
    fi
    echo "   ✅  Data: $(wc -l < "$TRAIN_JSONL") train + $(wc -l < data/instruct_val.jsonl) val"

    # ── Pre-download model to host cache ──
    echo "   📥  Downloading Qwen2.5-1.5B-Instruct (~3GB, one-time)..."
    mkdir -p "$HF_CACHE"
    # Fix permissions if root-owned from previous Docker runs
    sudo chown -R "$USER:$USER" "$HF_CACHE" 2>/dev/null || true
    pip install -q huggingface_hub 2>/dev/null || true
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', cache_dir='$HF_CACHE', token='$HF_TOKEN')
print('✅ Model cached')
"

    # ── Build Docker ──
    echo "   🔨  Building Docker image..."
    docker build -t "$IMAGE" -f src/finetune/Dockerfile .
    echo "   ${GREEN}✅  Ready!${NC}"
    echo "   Next: bash run.sh test"
}

# ═══════════════════════════════════════════════
do_test() {
    echo "🧪  Smoke test (100 steps)..."
    mkdir -p test_output
    docker run --rm --gpus all \
        -e HF_TOKEN="$HF_TOKEN" \
        -e TRAIN_MAX_STEPS=100 \
        -e TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}" \
        -e TRAIN_GRAD_ACCUM="${TRAIN_GRAD_ACCUM:-4}" \
        -e TRAIN_MAX_SEQ_LENGTH=256 \
        -e DATA_DIR=/app/data \
        -e CHECKPOINT_DIR=/app/checkpoints \
        -v "$(pwd)/test_output:/app/checkpoints" \
        -v "$HF_CACHE:/root/.cache/huggingface" \
        "$IMAGE" 2>&1 | tee test_output/log.txt
    echo "${GREEN}✅  Smoke test done${NC}"
}

# ═══════════════════════════════════════════════
do_train() {
    echo "🔥  Training..."
    mkdir -p checkpoints
    BATCH="${TRAIN_BATCH_SIZE:-4}"
    ACCUM="${TRAIN_GRAD_ACCUM:-4}"
    STEPS="${TRAIN_MAX_STEPS:-5000}"
    echo "   Steps: $STEPS | Batch: ${BATCH}×${ACCUM}=$((BATCH*ACCUM))"
    echo "   Data: $(wc -l < data/instruct_train.jsonl) examples"

    RESUME_MOUNT=""
    if [ -n "${RESUME_FROM:-}" ]; then
        RESUME_DIR="$(realpath "$RESUME_FROM")"
        echo "   📂  Resume: $RESUME_DIR"
        RESUME_MOUNT="-v $RESUME_DIR:/app/checkpoints/resume -e RESUME_FROM=/app/checkpoints/resume"
    fi

    docker run --rm --gpus all \
        -e HF_TOKEN="$HF_TOKEN" \
        -e TRAIN_MAX_STEPS="$STEPS" \
        -e TRAIN_BATCH_SIZE="$BATCH" \
        -e TRAIN_GRAD_ACCUM="$ACCUM" \
        -e TRAIN_MAX_SEQ_LENGTH="${TRAIN_MAX_SEQ_LENGTH:-256}" \
        -e DATA_DIR=/app/data \
        -e CHECKPOINT_DIR=/app/checkpoints \
        -v "$(pwd)/checkpoints:/app/checkpoints" \
        -v "$HF_CACHE:/root/.cache/huggingface" \
        $RESUME_MOUNT \
        "$IMAGE"

    echo "${GREEN}✅  Done — checkpoints/$(ls checkpoints/ 2>/dev/null | head -1)${NC}"
}

# ═══════════════════════════════════════════════
case "$CMD" in
    setup)  do_setup ;;
    test)   do_test ;;
    train)  do_train ;;
    *)      usage ;;
esac
