#!/bin/bash
set -e

echo "╔══════════════════════════════════════╗"
echo "║  PoetryDuel-GPT v5 — QLoRA Trainer  ║"
echo "╚══════════════════════════════════════╝"

# ── GPU info ──
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "🖥️  GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
fi

# ── GCS credentials (if provided) ──
if [ -n "$GCS_CREDENTIALS_JSON" ]; then
    echo "$GCS_CREDENTIALS_JSON" > /tmp/gcs-key.json
    export GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcs-key.json
    echo "🔑  GCS credentials loaded"
fi

# ── Build args ──
STAGE=${STAGE:-1}
ARGS="--stage $STAGE"

if [ -n "$RESUME_FROM" ]; then
    ARGS="$ARGS --resume $RESUME_FROM"
fi

echo ""
echo "📋  Config: Stage=$STAGE | Auto-chain=${AUTO_CHAIN_STAGES:-0}"
echo "🚀  Starting training..."
echo ""

exec python /app/train.py $ARGS
