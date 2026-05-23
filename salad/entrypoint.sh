#!/bin/bash
# Salad container entrypoint
# Handles runtime-only secrets (GCS credentials JSON passed as env var)

set -e

# If GCS credentials were passed as a JSON env var, write to file
if [ -n "$GCS_CREDENTIALS_JSON" ]; then
    echo "$GCS_CREDENTIALS_JSON" > /app/gcs-key.json
    export GOOGLE_APPLICATION_CREDENTIALS=/app/gcs-key.json
    echo "✅  GCS credentials loaded from GCS_CREDENTIALS_JSON"
fi

echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"

echo "=== Starting training ==="
exec python /app/train.py "$@"
