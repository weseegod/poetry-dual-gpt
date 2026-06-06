#!/bin/bash
# ──────────────────────────────────────────────
# Quick Local Test (no GHCR pull, no Docker push)
# Builds image locally, runs 100-step GPU test.
# Use during development before pushing to CI.
# ──────────────────────────────────────────────
set -e

GREEN='\033[0;32m'
NC='\033[0m'

echo "╔══════════════════════════════════════════════╗"
echo "║  🏠  Local Dev Test — build + 100 steps     ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

IMAGE="poetry-trainer:dev"
TEST_DIR="./test_output"
STEPS="${1:-100}"

# ── Build ──
echo "🔨  Building image..."
docker build -t "$IMAGE" -f src/finetune/Dockerfile .
echo ""

# ── Run ──
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

echo "🚀  Running $STEPS steps..."
docker run --rm --gpus all \
    -e HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}" \
    -e TRAIN_MAX_STEPS="$STEPS" \
    -e STAGE="1" \
    -e TRAIN_BATCH_SIZE="2" \
    -e TRAIN_GRAD_ACCUM="4" \
    -e TRAIN_BLOCK_SIZE="128" \
    -v "$(pwd)/$TEST_DIR:/app/checkpoints" \
    "$IMAGE" 2>&1 | tee "$TEST_DIR/test_log.txt"

echo ""
echo "${GREEN}✅  Local test done${NC}"
echo "   Log: $TEST_DIR/test_log.txt"
