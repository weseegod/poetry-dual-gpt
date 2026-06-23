#!/bin/bash
set -e

echo "╔══════════════════════════════════════╗"
echo "║  PoetryDuel-GPT v5.1 — Instruct SFT ║"
echo "╚══════════════════════════════════════╝"

# ── GPU info ──
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "🖥️  GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
fi

# ── Build args ──
ARGS=""
if [ -n "$TRAIN_MAX_STEPS" ]; then
    ARGS="$ARGS --max-steps $TRAIN_MAX_STEPS"
fi
if [ -n "$RESUME_FROM" ]; then
    ARGS="$ARGS --resume $RESUME_FROM"
fi

echo ""
echo "📋  Steps: ${TRAIN_MAX_STEPS:-5000} | Batch: ${TRAIN_BATCH_SIZE:-4}×${TRAIN_GRAD_ACCUM:-4}"
echo "🚀  Starting training..."
echo ""

exec python /app/train.py $ARGS
