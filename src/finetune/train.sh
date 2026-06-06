#!/bin/bash
# ──────────────────────────────────────────────
# Full QLoRA Training (Stage 1 + optional Stage 2)
# Pulls the CI-built image from GHCR, runs
# full training on local GPU. Uploads checkpoints
# to S3/R2/GCS if configured.
# ──────────────────────────────────────────────
set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "╔══════════════════════════════════════════════╗"
echo "║  🔥 QLoRA Full Training                     ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── Config ──
IMAGE="${POETRY_TRAINER_IMAGE:-ghcr.io/${GITHUB_USER:-REPLACE_ME}/poetry-dual-gpt/poetry-trainer-v5:${TAG:-main}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"
AUTO_CHAIN="${AUTO_CHAIN:-0}"
STAGE="${STAGE:-1}"

echo "📋  Image:       $IMAGE"
echo "    Stage:        $STAGE"
echo "    Auto-chain:   $AUTO_CHAIN"
echo "    Checkpoints:  $CHECKPOINT_DIR"
echo ""

# ── Pull latest ──
echo "📥  Pulling latest image..."
docker pull "$IMAGE"
echo ""

# ── Verify GPU ──
echo "🖥️  Verifying GPU access..."
if ! docker run --rm --gpus all --entrypoint nvidia-smi "$IMAGE" 2>/dev/null; then
    echo "${RED}❌  No GPU available!${NC}"
    echo "   Ensure nvidia-docker is installed:"
    echo "   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
    exit 1
fi
echo ""

# ── Run training ──
mkdir -p "$CHECKPOINT_DIR"

echo "🚀  Starting training..."
echo "   Press Ctrl+C to stop (checkpoints saved every 2000 steps)"
echo ""

docker run --rm --gpus all \
    -e HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}" \
    -e STAGE="$STAGE" \
    -e AUTO_CHAIN_STAGES="$AUTO_CHAIN" \
    -v "$(pwd)/$CHECKPOINT_DIR:/app/checkpoints" \
    ${S3_BUCKET:+-e S3_BUCKET="$S3_BUCKET"} \
    ${S3_ENDPOINT:+-e S3_ENDPOINT="$S3_ENDPOINT"} \
    ${AWS_ACCESS_KEY_ID:+-e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID"} \
    ${AWS_SECRET_ACCESS_KEY:+-e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY"} \
    ${AWS_DEFAULT_REGION:+-e AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION"} \
    ${GCS_BUCKET:+-e GCS_BUCKET="$GCS_BUCKET"} \
    ${GCS_CREDENTIALS_JSON:+-e GCS_CREDENTIALS_JSON="$GCS_CREDENTIALS_JSON"} \
    "$IMAGE"

echo ""
echo "────────────────────────────────────────"
echo "${GREEN}✅  Training complete!${NC}"
echo "   Checkpoints: $CHECKPOINT_DIR/"
ls -lh "$CHECKPOINT_DIR"/
