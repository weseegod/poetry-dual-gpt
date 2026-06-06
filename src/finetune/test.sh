#!/bin/bash
# ──────────────────────────────────────────────
# Local GPU Smoke Test (100 steps)
# Pulls the CI-built image from GHCR, runs a
# short training test to verify the pipeline
# works end-to-end on real hardware.
# ──────────────────────────────────────────────
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "╔══════════════════════════════════════════════╗"
echo "║  🔧 QLoRA Smoke Test — 100 steps on GPU     ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── Config ──
IMAGE="${POETRY_TRAINER_IMAGE:-ghcr.io/${GITHUB_USER:-REPLACE_ME}/poetry-dual-gpt/poetry-trainer-v5:${TAG:-main}}"
TEST_DIR="${TEST_DIR:-./test_output}"
STAGE="${STAGE:-1}"
STEPS="${STEPS:-100}"

echo "📋  Image:  $IMAGE"
echo "    Stage:   $STAGE"
echo "    Steps:   $STEPS"
echo "    Output:  $TEST_DIR"
echo ""

# ── Pull ──
echo "📥  Pulling image..."
docker pull "$IMAGE"
echo ""

# ── Check GPU ──
echo "🖥️  Checking GPU..."
if docker run --rm --gpus all --entrypoint nvidia-smi "$IMAGE" 2>/dev/null; then
    GPU_FLAG="--gpus all"
    echo "   ✅ NVIDIA GPU available"
else
    echo "   ${YELLOW}⚠️  No GPU detected — running on CPU (will be slow)${NC}"
    GPU_FLAG=""
fi
echo ""

# ── Run test ──
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

echo "🚀  Starting test training ($STEPS steps)..."
echo ""

docker run --rm $GPU_FLAG \
    -e HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}" \
    -e TRAIN_MAX_STEPS="$STEPS" \
    -e STAGE="$STAGE" \
    -e TRAIN_BATCH_SIZE="2" \
    -e TRAIN_GRAD_ACCUM="4" \
    -e TRAIN_BLOCK_SIZE="128" \
    -e TRAIN_NUM_WORKERS="2" \
    -v "$(pwd)/$TEST_DIR:/app/checkpoints" \
    "$IMAGE" 2>&1 | tee "$TEST_DIR/test_log.txt"

# ── Verify results ──
echo ""
echo "────────────────────────────────────────"
echo "📊  Verifying results..."
echo ""

# Check checkpoint was saved
if ls "$TEST_DIR"/qwen_stage"${STAGE}"_final/adapter_model.safetensors 2>/dev/null; then
    echo "${GREEN}✅  Checkpoint saved${NC}"
else
    echo "${YELLOW}⚠️  No final checkpoint (may be expected with very low steps)${NC}"
fi

# Check loss decreased (from log)
LOSSES=$(grep -oP 'loss=\K[0-9.]+' "$TEST_DIR/test_log.txt" 2>/dev/null || true)
if [ -n "$LOSSES" ]; then
    FIRST=$(echo "$LOSSES" | head -1)
    LAST=$(echo "$LOSSES" | tail -1)
    echo "   First loss: $FIRST"
    echo "   Last loss:  $LAST"
    if [ "$(echo "$LAST < $FIRST" | bc -l 2>/dev/null)" = "1" ] || [ "$FIRST" = "$LAST" ]; then
        echo "   ${GREEN}✅  Loss trending down or flat${NC}"
    else
        echo "   ${YELLOW}⚠️  Loss increasing — check if LR is too high${NC}"
    fi
else
    echo "   ${YELLOW}⚠️  No loss values found in log${NC}"
fi

# Check for CUDA errors
if grep -qi "cuda.*out of memory\|CUDA error\|RuntimeError" "$TEST_DIR/test_log.txt" 2>/dev/null; then
    echo "   ${RED}❌  CUDA errors detected! Check log: $TEST_DIR/test_log.txt${NC}"
    exit 1
else
    echo "   ${GREEN}✅  No CUDA errors${NC}"
fi

echo ""
echo "────────────────────────────────────────"
echo "${GREEN}✅  Smoke test complete!${NC}"
echo "   Full log: $TEST_DIR/test_log.txt"
echo "   Checkpoints: $TEST_DIR/"
echo ""
echo "   → If this passed, run:  bash src/finetune/train.sh"
